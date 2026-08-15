"""NeuralRouter — IMM router backed by a pluggable neural model.

The router owns feature extraction and normalisation.
The model owns architecture, training, and inference.

Usage::

    from kalman.framework.routers import NeuralRouter
    from kalman.framework.routers.nn import MLP

    router = NeuralRouter(model=MLP(hidden=32), window=5)

    ikf = InteractiveKalmanFilter(models, Pi, router=router)
    ikf.fit(training_zk_list, labels_list)      # trains the model
    record = ikf.forward(test_zk)

Any callable ``model`` with signature ``model(features) → probs`` works,
including PyTorch modules, sklearn classifiers, or custom numpy networks.

For lazy parameter init and training support, the model may optionally expose:

* ``model.build(d_in, d_out, seed=None)`` — called once before fitting
* ``model.fit(X, y, **kwargs)`` — training on normalised features

If these methods are absent the model is assumed to be pre-trained.

NIS feature extraction
----------------------
Input features are the last ``window`` NIS values per model, flattened
to a vector of size M × window.

NIS_j = ν_j^T S_j^{-1} ν_j — under the correct model NIS_j ~ χ²(m).

NIS is used rather than raw innovations or mode probabilities because:
- Raw ν_j: different scales across models.
- Mode probs μ: distribution shift (Bayesian μ at train ≠ neural μ at test).
- NIS: dimensionless, scale-free, no distribution shift.

Before ``fit()`` is called the router falls back to the Bayesian update.
"""
from __future__ import annotations
import numpy as np
from .base import KalmanRouter


class NeuralRouter(KalmanRouter):
    """IMM router that delegates mode-probability estimation to a neural model.

    Parameters
    ----------
    model : callable
        Any object callable as ``probs = model(feat)`` where ``feat`` is a
        1-D normalised numpy array of length M × window.
        Optionally implements ``build(d_in, d_out, seed)`` and
        ``fit(X, y, **kw)`` for lazy init and training.
    window : int
        NIS sliding window length.  Padding value 1.0 = χ²(1) mean.
    """

    def __init__(self, model, window: int = 5):
        self.model   = model
        self.window  = int(window)
        self._x_mean: np.ndarray | None = None
        self._x_std:  np.ndarray | None = None

    # ── feature extraction ────────────────────────────────────────────────────

    def _window_feat(self, nis_arr: np.ndarray) -> np.ndarray:
        """Last ``window`` NIS rows, padded with 1.0, flattened (M×window,)."""
        block = nis_arr[-self.window:]
        if len(block) < self.window:
            pad   = np.ones((self.window - len(block), nis_arr.shape[1]))
            block = np.vstack([pad, block])
        return block.ravel()

    @property
    def _is_trained(self) -> bool:
        return self._x_mean is not None

    # ── routing ───────────────────────────────────────────────────────────────

    def route(self, nis_arr, c, likelihoods):
        if not self._is_trained:
            return self._normalise(c * likelihoods)   # Bayesian fallback
        feat   = self._window_feat(nis_arr)
        feat_n = (feat - self._x_mean) / self._x_std
        probs  = np.asarray(self.model(feat_n), dtype=float)
        return self._normalise(probs)

    # ── training ──────────────────────────────────────────────────────────────

    def fit(self, nis_arrays, labels_arrays, seed=None, **kwargs) -> "NeuralRouter":
        """Extract features, normalise, build model, train model.

        Called by ``InteractiveKalmanFilter.fit()``.

        Parameters
        ----------
        nis_arrays : list of ndarray (T_i, M)
            NIS history per training scenario (from a Bayesian-IMM run).
        labels_arrays : list of int ndarray (T_i,)
            True regime 0 … M−1 at every step.
        seed : int, optional
            Forwarded to ``model.build()`` if supported.
        """
        M    = nis_arrays[0].shape[1]
        d_in = M * self.window

        X_all, y_all = [], []
        for nis_arr, labels in zip(nis_arrays, labels_arrays):
            for k, lbl in enumerate(labels):
                X_all.append(self._window_feat(nis_arr[:k + 1]))
                y_all.append(int(lbl))

        X = np.array(X_all, dtype=float)
        y = np.array(y_all,  dtype=int)

        self._x_mean = X.mean(0)
        self._x_std  = X.std(0)
        self._x_std[self._x_std < 1e-8] = 1.0
        Xn = (X - self._x_mean) / self._x_std

        if hasattr(self.model, "build"):
            self.model.build(d_in, M, seed=seed)
        if hasattr(self.model, "fit"):
            self.model.fit(Xn, y, **kwargs)

        return self

    # ── export ────────────────────────────────────────────────────────────────

    def export_onnx(self, path, *, opset: int = 17, ir_version: int | None = 9,
                    bake_normalisation: bool = True):
        """Write the trained policy to ONNX, for execution by ``Kalman::Routers::Neural``.

        This is the seam between the two implementations of the framework. Training
        happens here, where the data loaders and the optimiser live; inference
        happens in C++, which then needs neither Python nor Torch.

        The exported graph is **self-contained by default**: the feature
        standardisation learned in :meth:`fit` and the softmax head are baked in as
        constants, so the graph maps a raw NIS window straight to probabilities.
        That is deliberate — a normalisation shipped separately from the weights it
        belongs to is a silent failure waiting to happen, because the wrong one
        still produces a valid-looking probability vector.

        A ``<path>.json`` sidecar records the window length and mode count so the
        C++ side (or a human) can check compatibility rather than discover a
        mismatch at runtime.

        Parameters
        ----------
        path : str
            Destination ``.onnx`` file.
        opset : int
            ONNX opset version.
        ir_version : int or None
            Pin the model's IR version after export. Torch emits the newest IR it
            knows, which *deployed* runtimes routinely reject — ONNX Runtime 1.16
            refuses anything above 9 with "Unsupported model IR version: 10". The
            IR version is a container format, not the operator semantics, so
            lowering it is safe as long as the opset is supported. Pass None to
            keep whatever Torch produced.
        bake_normalisation : bool
            Fold the standardisation into the graph. Turn it off only if the
            consumer applies its own via ``SetNormalisation``.

        Returns
        -------
        dict — the metadata also written to the sidecar.
        """
        import json
        import torch

        net, mean, std, d_in = self._export_prologue("export_onnx")
        policy = self._policy_module(net, mean, std, bake_normalisation)
        dummy = torch.zeros(1, d_in, dtype=torch.float32)

        with torch.no_grad():
            d_out = int(policy(dummy).shape[-1])

        torch.onnx.export(
            policy, dummy, str(path),
            input_names=["nis_window"], output_names=["mode_probabilities"],
            dynamic_axes={"nis_window": {0: "batch"},
                          "mode_probabilities": {0: "batch"}},
            opset_version=opset,
        )

        if ir_version is not None:
            import onnx
            model = onnx.load(str(path))
            if model.ir_version > ir_version:
                model.ir_version = ir_version
                onnx.save(model, str(path))

        meta = {
            "format": "onnx",
            "window": int(self.window),
            "opset": int(opset),
            "ir_version": ir_version,
            "n_models": d_out,
            "n_features": d_in,
            "normalisation_baked": bool(bake_normalisation),
            "feature_layout": "last `window` NIS rows, front-padded with 1.0, flattened row-major",
        }
        if not bake_normalisation:
            meta["mean"] = [float(v) for v in self._x_mean]
            meta["scale"] = [float(v) for v in self._x_std]

        with open(f"{path}.json", "w") as fh:
            json.dump(meta, fh, indent=2)

        return meta

    @staticmethod
    def _policy_module(net, mean, std, bake):
        """Standardise → network → softmax, as one traceable module.

        Shared by both export paths so ONNX and TorchScript cannot drift apart in
        what they consider "the policy".
        """
        import copy
        import torch
        from torch import nn

        class _Policy(nn.Module):
            def __init__(self, net, mean, std, bake):
                super().__init__()
                self.net = net
                self.bake = bake
                self.register_buffer("mean", mean)
                self.register_buffer("std", std)

            def forward(self, x):
                if self.bake:
                    x = (x - self.mean) / self.std
                return torch.softmax(self.net(x), dim=-1)

        # Deep-copy before moving to CPU: `.to()` is in-place for a Module, so
        # exporting would otherwise strand the caller's model on the wrong device
        # and every later route() call would fail.
        return _Policy(copy.deepcopy(net).to("cpu").eval(), mean, std, bake).eval()

    def _export_prologue(self, what):
        """Checks and tensors both export paths need."""
        import torch

        if not self._is_trained:
            raise RuntimeError(
                f"NeuralRouter.{what}: nothing to export — call fit() "
                "(or InteractiveKalmanFilter.fit()) first.")

        net = getattr(self.model, "_net", None)
        if net is None:
            raise RuntimeError(
                f"NeuralRouter.{what}: {type(self.model).__name__} exposes no "
                "torch module (`_net`), so it cannot be traced.")

        return (net,
                torch.tensor(self._x_mean, dtype=torch.float32),
                torch.tensor(self._x_std, dtype=torch.float32),
                int(len(self._x_mean)))

    def export_torchscript(self, path, *, bake_normalisation: bool = True):
        """Trace the trained policy to TorchScript, for ``Kalman::Routers::NN::Torch``.

        The counterpart of :meth:`export_onnx`, and the higher-fidelity one: the
        C++ side runs *this* module rather than a conversion of it, so an
        architecture the ONNX exporter mangles still deploys. The cost is that the
        consumer links libTorch instead of ONNX Runtime — roughly 2 GB against 10
        MB — and is pinned to a compatible Torch version.

        Like :meth:`export_onnx` the traced module is self-contained: the feature
        standardisation and the softmax head are inside it, so a raw NIS window
        maps straight to probabilities.

        Parameters
        ----------
        path : str
            Destination ``.pt`` file.
        bake_normalisation : bool
            Fold the standardisation into the module.

        Returns
        -------
        dict — the metadata also written to ``<path>.json``.
        """
        import json
        import torch

        net, mean, std, d_in = self._export_prologue("export_torchscript")
        policy = self._policy_module(net, mean, std, bake_normalisation)

        dummy = torch.zeros(1, d_in, dtype=torch.float32)
        with torch.no_grad():
            traced = torch.jit.trace(policy, dummy)
            d_out = int(policy(dummy).shape[-1])
        traced.save(str(path))

        meta = {
            "format": "torchscript",
            "window": int(self.window),
            "n_models": d_out,
            "n_features": d_in,
            "normalisation_baked": bool(bake_normalisation),
            "feature_layout": "last `window` NIS rows, front-padded with 1.0, flattened row-major",
        }
        if not bake_normalisation:
            meta["mean"] = [float(v) for v in self._x_mean]
            meta["scale"] = [float(v) for v in self._x_std]

        with open(f"{path}.json", "w") as fh:
            json.dump(meta, fh, indent=2)

        return meta
