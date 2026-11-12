"""libKalmanLearn from Python: the libTorch field solver behind ``SpectralSolver(backend="torch")``."""
import os
import re
import unittest

import numpy as np

from modules.framework.native import learn
from modules.framework.constants import KalmanIntegratorKind

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HEADER = os.path.join(REPO, "learn", "include", "Kalman", "Learn", "CAPI.h")

_PROTO = re.compile(r"^\s*(?:[A-Za-z_][\w ]*\s*\*?\s*)\b(kalman_\w+)\s*\(", re.M)


def _prototypes(text):
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"//.*", "", text)
    return set(_PROTO.findall(text))


class TestHeaderContract(unittest.TestCase):
    """``native/learn.py``'s cdef is a copy of ``Kalman/Learn/CAPI.h``."""

    def test_cdef_matches_header(self):
        with open(HEADER) as fh:
            header = _prototypes(fh.read())
        self.assertTrue(header)
        self.assertEqual(header, _prototypes(learn._CDEF))


@unittest.skipUnless(learn.has_torch(), "libKalmanLearn with libTorch is not built")
class TestTorchSolver(unittest.TestCase):

    def setUp(self):
        self._env = os.environ.pop("KALMAN_NATIVE_TORCH", None)

    def tearDown(self):
        os.environ.pop("KALMAN_NATIVE_TORCH", None)
        if self._env is not None:
            os.environ["KALMAN_NATIVE_TORCH"] = self._env

    def test_torch_backend_runs_on_libtorch(self):
        from modules.apps.weather.flow import NS2D
        s = NS2D(n=16, dt=0.01, seed=0, backend="torch", device="cpu")
        s.advance(s.random_ic(), 1)
        self.assertIsNotNone(s._torch_native())
        self.assertEqual(s.device, "cpu")

    def test_every_layout_and_integrator_matches_numpy(self):
        from modules.apps.weather.flow import NS2D
        for kind in (KalmanIntegratorKind.EULER, KalmanIntegratorKind.RK4,
                     KalmanIntegratorKind.DORMAND_PRINCE):
            ref = NS2D(n=16, nu=1/40., kf=4, chi=2.0, dt=0.01, seed=3, integrator=kind)
            s = NS2D(n=16, nu=1/40., kf=4, chi=2.0, dt=0.01, seed=3, integrator=kind,
                     backend="torch", device="cpu")
            ens = np.stack([ref.random_ic() for _ in range(3)])
            want = np.stack([ref.advance(e.copy(), 5) for e in ens])
            np.testing.assert_allclose(s.advance(ens.copy(), 5), want, atol=1e-10)             # (N, ny, nx)
            np.testing.assert_allclose(s.advance(ens.reshape(3, -1).copy(), 5),
                                       want.reshape(3, -1), atol=1e-10)                        # (N, ny*nx)
            np.testing.assert_allclose(s.advance(ens[0].copy(), 5), want[0], atol=1e-10)       # (ny, nx)
            np.testing.assert_allclose(s.advance(ens[0].ravel().copy(), 5),
                                       want[0].ravel(), atol=1e-10)                            # (ny*nx,)

    def test_beta_plane_rectangular_grid_matches_numpy(self):
        from modules.apps.weather.barotropic import BetaPlaneNS2D
        ref = BetaPlaneNS2D(n=(16, 24), seed=1)
        s = BetaPlaneNS2D(n=(16, 24), seed=1, backend="torch", device="cpu")
        w = ref.random_ic(amp=1e-5)                      # 1/s, a mid-latitude vorticity
        want = ref.advance(w.copy(), 20)
        self.assertGreater(np.abs(want - w).max(), 1e-8)   # the field did move
        np.testing.assert_allclose(s.advance(w.copy(), 20), want, rtol=0, atol=1e-16)
        self.assertIsNotNone(s._torch_native())

    def test_python_torch_path_still_available(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch not installed")
        from modules.apps.weather.flow import NS2D
        w = NS2D(n=16, dt=0.01, seed=2).random_ic()
        native = NS2D(n=16, dt=0.01, seed=2, backend="torch", device="cpu").advance(w.copy(), 4)
        os.environ["KALMAN_NATIVE_TORCH"] = "0"
        s = NS2D(n=16, dt=0.01, seed=2, backend="torch", device="cpu")
        out = s.advance(w.copy(), 4)
        self.assertIsNone(s._torch_native())
        np.testing.assert_allclose(out, native, atol=1e-12)

    def test_other_physics_keeps_its_own_rhs(self):
        """A solver that does not state vorticity terms is not sent to C++."""
        from modules.framework.solvers.spectral import SpectralSolver
        s = SpectralSolver(n=8, backend="torch", device="cpu")
        self.assertIsNone(s._torch_native())
        with self.assertRaises(NotImplementedError):
            s.advance(np.zeros((8, 8)), 1)

    def test_bad_shape_is_an_error_not_a_crash(self):
        from modules.apps.weather.flow import NS2D
        s = NS2D(n=16, dt=0.01, backend="torch", device="cpu")
        with self.assertRaises(ValueError):
            s.advance(np.zeros((2, 100)), 1)


if __name__ == "__main__":
    unittest.main()
