"""NeuralRouter — IMM router backed by a pluggable neural model.

The router owns feature extraction and normalisation.
The model owns architecture, training, and inference.

Usage::

    from kalman.filters.routers import NeuralRouter
    from kalman.filters.routers.nn import MLP

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
