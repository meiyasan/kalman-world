"""Unit tests for the pure utilities in ``modules.helpers``.

These are deterministic, dependency-light functions (jacobians, shape coercion,
multi-rate interleaving, complex<->real packing, colour math, signal helpers) that
the rest of the framework relies on. Follows the conventions of ``test_module.py``.
"""
import sys
import unittest

import numpy as np

sys.path.insert(0, ".")
from modules.helpers import (
    numerical_jacobian, as_col, format_shape,
    vectorize, complexify, str2enum, apply_crop,
    darken, lighten, fade, vivid, rotate, rainbow, barycentric_interpolation,
    bandpass_filter, resample_poly_float, make_multirate_input,
)
from modules.apps import GPSState


# ===========================================================================
# Jacobian / shape coercion
# ===========================================================================
class TestJacobianAndShapes(unittest.TestCase):

    def test_numerical_jacobian_linear_map(self):
        """For a linear map f(x)=Ax the numerical Jacobian must recover A."""
        A = np.array([[2.0, 1.0], [0.0, 3.0]])
        J = numerical_jacobian(lambda x: A @ x, np.array([1.0, 2.0]))
        np.testing.assert_allclose(J, A, atol=1e-5)

    def test_numerical_jacobian_nonlinear(self):
        """d/dx [x0^2, sin(x1)] = diag(2 x0, cos x1)."""
        x = np.array([1.5, 0.3])
        J = numerical_jacobian(lambda v: np.array([[v[0, 0] ** 2], [np.sin(v[1, 0])]]), x)
        np.testing.assert_allclose(J, np.diag([2 * x[0], np.cos(x[1])]), atol=1e-5)

    def test_as_col_promotes_1d(self):
        self.assertEqual(as_col([1, 2, 3]).shape, (3, 1))
        self.assertEqual(as_col(np.array([[1.0], [2.0]])).shape, (2, 1))

    def test_as_col_rejects_matrix(self):
        with self.assertRaises(ValueError):
            as_col(np.ones((2, 2)))

    def test_format_shape(self):
        self.assertEqual(format_shape(np.zeros((3, 4))), "3x4")


# ===========================================================================
# Complex <-> real packing
# ===========================================================================
class TestComplexPacking(unittest.TestCase):

    def test_vectorize_complexify_roundtrip(self):
        z = np.array([1 + 2j, 3 - 1j, -2 + 0j])
        v = vectorize(z)
        self.assertEqual(v.shape, (3, 2))
        np.testing.assert_allclose(complexify(v), z)

    def test_vectorize_splits_real_imag(self):
        v = vectorize(np.array([4 + 5j]))
        np.testing.assert_allclose(v, [[4.0, 5.0]])


# ===========================================================================
# enum / crop helpers
# ===========================================================================
class TestEnumAndCrop(unittest.TestCase):

    def test_str2enum_resolves_name(self):
        self.assertIs(str2enum("X", GPSState), GPSState.X)
        self.assertIs(str2enum(GPSState.Y, GPSState), GPSState.Y)

    def test_str2enum_rejects_unknown(self):
        with self.assertRaises(ValueError):
            str2enum("NOT_A_STATE", GPSState)

    def test_apply_crop_tuple_and_scalar(self):
        data = np.arange(10)
        np.testing.assert_array_equal(apply_crop(data, (2, 5)), [2, 3, 4])
        np.testing.assert_array_equal(apply_crop(data, 2), [2, 3, 4, 5, 6, 7])
        np.testing.assert_array_equal(apply_crop(data, None), data)


# ===========================================================================
# Colour math
# ===========================================================================
class TestColours(unittest.TestCase):

    def _valid_rgb(self, c):
        self.assertEqual(len(c), 3)
        for ch in c:
            self.assertGreaterEqual(ch, 0.0)
            self.assertLessEqual(ch, 1.0)

    def test_shade_helpers_stay_in_gamut(self):
        for fn in (darken, lighten, fade, vivid):
            self._valid_rgb(fn("#3366cc", 0.3))

    def test_darken_is_darker_lighten_is_lighter(self):
        base = np.array([0.3, 0.4, 0.5])
        self.assertLess(sum(darken("#4d6680", 0.5)), sum(base) + 1e-9)
        self.assertGreater(sum(lighten("#4d6680", 0.5)), sum(base) - 1e-9)

    def test_rotate_returns_rgb(self):
        self._valid_rgb(rotate("#3366cc", 30))

    def test_rainbow_length(self):
        self.assertEqual(len(rainbow(5)), 5)

    def test_barycentric_midpoint_red_blue(self):
        mid = barycentric_interpolation("red", "blue")
        np.testing.assert_allclose(mid, (0.5, 0.0, 0.5), atol=1e-9)


# ===========================================================================
# Signal helpers
# ===========================================================================
class TestSignalHelpers(unittest.TestCase):

    def test_bandpass_preserves_length_and_passband(self):
        fs, N = 256, 1024
        t = np.arange(N) / fs
        x = np.sin(2 * np.pi * 40 * t) + np.sin(2 * np.pi * 5 * t)
        y = bandpass_filter(x, fs, fc=40, bandwidth=10)
        self.assertEqual(y.shape, (N,))
        # the 40 Hz component survives; the 5 Hz component is strongly attenuated
        burn = slice(N // 4, 3 * N // 4)
        self.assertGreater(np.std(y[burn]), 0.3)

    def test_resample_halves_sample_count(self):
        x = np.random.RandomState(0).randn(500)
        y = resample_poly_float(x, 256, 128)
        self.assertEqual(y.shape[0], 250)

    def test_make_multirate_input_nan_pads_slow_sensors(self):
        slow = np.arange(10, dtype=float).reshape(10, 1)   # 10 Hz
        fast = np.random.RandomState(1).randn(50, 2)        # 50 Hz
        zk = make_multirate_input([(slow, 10), (fast, 50)], base_fs=50)
        self.assertEqual(zk.shape, (50, 3))
        # slow sensor fires 10/50 of the time -> ~80% NaN in its column
        self.assertAlmostEqual(np.isnan(zk[:, 0]).mean(), 0.8, delta=0.05)
        # fast sensor fires every step -> no NaN
        self.assertFalse(np.any(np.isnan(zk[:, 1:])))

    def test_make_multirate_input_requires_data(self):
        with self.assertRaises(ValueError):
            make_multirate_input([], base_fs=10)


if __name__ == "__main__":
    unittest.main()
