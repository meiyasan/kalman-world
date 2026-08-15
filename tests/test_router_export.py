"""The ONNX export contract: a router trained here must reproduce itself elsewhere.

``NeuralRouter.export_onnx()`` is the seam between the Python framework and
libKalmanFramework: Python owns training, C++ owns inference, and a *trained
policy* — not a language binding — is what crosses between them. That only works
if the exported graph computes exactly what ``router.route()`` computes, so this
suite pins the three things that can silently break it:

* the feature layout (window, front-padding, row-major flatten),
* the standardisation, which is baked into the graph so it cannot be shipped
  separately from the weights it belongs to,
* the IR version, which deployed runtimes reject when it is too new.

The equivalence is checked with the ``onnxruntime`` Python package, which is the
same engine ``Kalman::Routers::Neural`` uses in C++.
"""
import sys
import unittest

import numpy as np

sys.path.insert(0, ".")

torch = pytest_skip = None
try:
    import torch  # noqa: F401
    import onnx  # noqa: F401
    import onnxruntime  # noqa: F401
    HAVE_EXPORT = True
except ImportError:
    HAVE_EXPORT = False


def _training_set(n_scenarios=12, T=100, M=2, seed=0):
    """Labelled NIS histories where the correct model sits at chi2(1) and the other far above."""
    rng = np.random.default_rng(seed)
    nis_arrays, label_arrays = [], []
    for _ in range(n_scenarios):
        labels = (np.arange(T) >= T // 2).astype(int)
        nis = np.empty((T, M))
        for k, lbl in enumerate(labels):
            nis[k, lbl] = rng.chisquare(1)
            nis[k, 1 - lbl] = rng.chisquare(1) * 8 + 5
        nis_arrays.append(nis)
        label_arrays.append(labels)
    return nis_arrays, label_arrays


@unittest.skipUnless(HAVE_EXPORT, "needs torch + onnx + onnxruntime")
class TestNeuralRouterExport(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from modules.framework.routers import NeuralRouter
        from modules.framework.routers.nn import MLP

        cls.window = 5
        cls.nis_arrays, cls.labels = _training_set()
        cls.router = NeuralRouter(model=MLP(hidden=16), window=cls.window)
        cls.router.fit(cls.nis_arrays, cls.labels, seed=0, epochs=60)

    def _export(self, tmpdir, **kwargs):
        path = f"{tmpdir}/router.onnx"
        meta = self.router.export_onnx(path, **kwargs)
        return path, meta

    def test_untrained_router_refuses_to_export(self):
        from modules.framework.routers import NeuralRouter
        from modules.framework.routers.nn import MLP
        import tempfile

        fresh = NeuralRouter(model=MLP(hidden=8), window=5)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError):
                fresh.export_onnx(f"{tmp}/nope.onnx")

    def test_metadata_describes_the_graph(self):
        import tempfile, json

        with tempfile.TemporaryDirectory() as tmp:
            path, meta = self._export(tmp)
            self.assertEqual(meta["window"], self.window)
            self.assertEqual(meta["n_models"], 2)
            self.assertEqual(meta["n_features"], 2 * self.window)
            self.assertTrue(meta["normalisation_baked"])

            with open(f"{path}.json") as fh:
                sidecar = json.load(fh)
            self.assertEqual(sidecar, meta)

    def test_ir_version_is_pinned_for_deployed_runtimes(self):
        # Torch emits the newest IR it knows; ONNX Runtime 1.16 rejects anything
        # above 9 outright. Exporting a policy no deployment can load is the kind
        # of failure that only shows up on the far side of the seam.
        import tempfile, onnx

        with tempfile.TemporaryDirectory() as tmp:
            path, _ = self._export(tmp)
            self.assertLessEqual(onnx.load(path).ir_version, 9)

    def test_export_does_not_strand_the_model(self):
        # export_onnx() moves the network to CPU to trace it; doing that in place
        # would leave the caller's router unusable on an accelerator.
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            self._export(tmp)
        mu = self.router.route(self.nis_arrays[0][:20], np.array([0.5, 0.5]), np.ones(2))
        self.assertTrue(np.all(np.isfinite(mu)))
        self.assertAlmostEqual(float(mu.sum()), 1.0, places=6)

    def test_graph_reproduces_the_python_router(self):
        """The whole point: same NIS history in, same probabilities out."""
        import tempfile
        import onnxruntime as ort

        c, likelihoods = np.array([0.5, 0.5]), np.ones(2)

        with tempfile.TemporaryDirectory() as tmp:
            path, _ = self._export(tmp)
            session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
            name = session.get_inputs()[0].name

            worst = 0.0
            # Short histories exercise the front-padding, long ones the window.
            for nis in (self.nis_arrays[0][:1], self.nis_arrays[0][:3],
                        self.nis_arrays[0][:40], self.nis_arrays[3][:100]):
                expected = self.router.route(nis, c, likelihoods)

                # Reproduce the C++ feature extraction: raw window, unnormalised,
                # because the standardisation is inside the graph.
                block = nis[-self.window:]
                if len(block) < self.window:
                    block = np.vstack([np.ones((self.window - len(block), nis.shape[1])), block])
                feat = block.ravel().astype(np.float32)[None, :]

                got = session.run(None, {name: feat})[0].ravel()
                worst = max(worst, float(np.max(np.abs(got - expected))))

            self.assertLess(worst, 1e-5, f"graph and router disagree by {worst:.2e}")

    def test_unbaked_export_carries_its_normalisation(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            _, meta = self._export(tmp, bake_normalisation=False)
            self.assertFalse(meta["normalisation_baked"])
            # Without it in the graph, the consumer needs it explicitly — so it
            # must travel in the sidecar rather than being lost.
            self.assertEqual(len(meta["mean"]), 2 * self.window)
            self.assertEqual(len(meta["scale"]), 2 * self.window)


if __name__ == "__main__":
    unittest.main()
