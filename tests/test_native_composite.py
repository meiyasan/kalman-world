"""Composite models run as models of matrices, in C++.

A :class:`CompositeKalmanModel` stacks its parts: block-diagonal F and Q, and an
H that stacks (INDEPENDENT) or sums (FUSED) the parts' readouts. When every part
is constant matrices the composite is too, and says so (``structure`` is
MATRICES), so its filtering pass and its smoother run in C++ on those matrices
(``kf_forward_linear`` / ``kf_backward_rts``) instead of the C++ loop calling the
Python closures back at every step. These tests pin that the pass gives the
Python answer, that it really is the C++ pass that runs, and that a composite
which is not made of matrices keeps the path it had.

The reference is the Python loop (``native.python_reference()`` with the fused
linear pass switched off as well).
"""
import sys
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, ".")

import modules.framework.models.native as native
from modules.framework.filter import KalmanFilter
from modules.framework.native import HAS_C_ACCEL
from modules.framework.constants import KalmanStructureKind as S, KalmanEngineKind, KalmanObservableReadout
from modules.framework.engines.linear import LinearKalmanEngine
from modules.framework.models.composite import CompositeKalmanModel
from modules.apps.resonators.oscillators import (
    AnalyticOscillatorModel, OscillatorModel, OscillatorBankModel, InharmonicOscillatorModel,
    HarmonicOscillatorModel, PhasorOscillatorModel, partial_frequencies)
from modules.apps.gravitational.interferometer import InterferometerModel, InterferometerK1

needs_kernels = unittest.skipUnless(HAS_C_ACCEL, "libKalman is not built")

# As in test_native_twins: the state to 1e-7 and the covariance to 1e-6 of the
# largest entry. The two passes are the same arithmetic in another order (the
# C++ update is not the Python one term for term), so they agree far closer,
# to about 1e-12; the shared bound is kept.
STATE_RTOL = 1e-7
COVARIANCE_RTOL = 1e-6

FS = 4096.0


def close(test, a, b, rtol, what):
    a, b = np.asarray(a, float), np.asarray(b, float)
    scale = max(1.0, float(np.nanmax(np.abs(a))))
    test.assertLess(float(np.nanmax(np.abs(a - b))) / scale, rtol, what)


def strain(freqs, amps, Qs, noise, T=4.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(T * FS)) / FS
    x = noise * rng.standard_normal(t.size)
    for a, f, q in zip(amps, freqs, Qs):
        x += a * np.exp(-t / (q / (np.pi * f))) * np.cos(2 * np.pi * f * t)
    return x


def reference(make, z, smooth=False):
    """The Python loop: no C++ loop, no twin, no fused linear pass."""
    with native.python_reference(), \
            mock.patch.object(KalmanFilter, "_native_forward_eligible", return_value=False):
        return make().filter(z, smooth=smooth)


def close_smoothed(test, py, cx, model, what):
    """A smoothed covariance, against the scale of the forward pass it comes from.

    The windows start from a loose prior ((1000 sigma)^2) and the RTS sweep takes
    differences of the forward covariances on its way back, down to a smoothed
    covariance thousands of times smaller: its rounding is that of the forward
    covariance it was subtracted from. Two Python RTS sweeps over forward passes
    that agree to 1e-14 differ there by 2e-4 of the smoothed value; measured
    against the forward scale both stay below 1e-7."""
    scale = max(1.0, max(float(np.max(np.abs(P))) for _, P in model._filter.prior))
    a, b = np.asarray(py.Pk, float), np.asarray(cx.Pk, float)
    test.assertLess(float(np.max(np.abs(a - b))) / scale, COVARIANCE_RTOL, what)


def on_cpp(test, make, z, smooth=False):
    """The default pass, asserting it is the fused C++ one."""
    with mock.patch.object(KalmanFilter, "_forward_native", autospec=True,
                           side_effect=KalmanFilter._forward_native) as fused, \
            mock.patch.object(KalmanFilter, "_forward_callback", autospec=True,
                              side_effect=KalmanFilter._forward_callback) as callback:
        model = make()
        record = model.filter(z, smooth=smooth)
    test.assertEqual(fused.call_count, 1, "the pass ran in kf_forward_linear")
    test.assertEqual(callback.call_count, 0, "no Python callbacks")
    test.assertIsInstance(model._filter.engine, LinearKalmanEngine)
    test.last_model = model
    return record


def one_line():
    return OscillatorModel("mode", [180.1], fc=180.1, dfc=5.0, Q=1.3e4, fs=FS,
                           measurement_uncertainty=0.1, process_uncertainty=1e-4)


def cluster():
    return OscillatorModel("cluster", [178.0, 180.1, 182.5], fc=180.0, dfc=8.0, Q=[1.3e4] * 3, fs=FS,
                           measurement_uncertainty=0.1, process_uncertainty=1e-4)


PARTIALS = partial_frequencies(170.0, 0.13, 3)


def bank():
    return OscillatorBankModel(*[
        InharmonicOscillatorModel(f"ih_{k}", f0=170.0, fc=fk, dfc=8.0, n_harmonics=3, B=0.13, Q=q, fs=FS,
                                  measurement_uncertainty=0.05, process_uncertainty=1e-5)
        for k, (fk, q) in enumerate(zip(PARTIALS, [1.3e4, 1.8e4, 2.1e4]))])


@needs_kernels
class TestCompositeOfMatrices(unittest.TestCase):

    def test_structure_follows_the_parts(self):
        self.assertIs(one_line().structure, S.MATRICES)
        self.assertIs(bank().structure, S.MATRICES)
        self.assertIs(HarmonicOscillatorModel("h", f0=60.0, fc=120.0, dfc=5.0, fs=FS).structure, S.MATRICES)
        a = AnalyticOscillatorModel("a", fs=100.0, f0=10.0, Q=100)
        b = AnalyticOscillatorModel("b", fs=100.0, f0=12.0, Q=100)
        for mode in (KalmanObservableReadout.INDEPENDENT, KalmanObservableReadout.FUSED):
            self.assertIs(CompositeKalmanModel(a, b, obs_mode=mode).structure, S.MATRICES, mode)

    def test_engine_follows_the_structure(self):
        # The class keeps its declaration (what the catalogue shows); an instance
        # made of matrices runs on the linear engine.
        self.assertIs(CompositeKalmanModel.DEFAULT_ENGINE, KalmanEngineKind.EXTENDED)
        self.assertIs(OscillatorModel.DEFAULT_ENGINE, KalmanEngineKind.EXTENDED)
        self.assertIs(one_line().DEFAULT_ENGINE, KalmanEngineKind.LINEAR)
        self.assertIs(bank().DEFAULT_ENGINE, KalmanEngineKind.LINEAR)

    def test_one_line(self):
        z = strain([180.1], [1.0], [1.3e4], 0.1)
        py, cx = reference(one_line, z), on_cpp(self, one_line, z)
        close(self, py.xk, cx.xk, STATE_RTOL, "state")
        close(self, py.Pk, cx.Pk, COVARIANCE_RTOL, "covariance")

    def test_three_lines_in_one_window_smoothed(self):
        z = strain([178.0, 180.1, 182.5], [1.0, 0.6, 0.4], [1.3e4] * 3, 0.1, seed=1)
        py, cx = reference(cluster, z, smooth=True), on_cpp(self, cluster, z, smooth=True)
        close(self, py.xk, cx.xk, STATE_RTOL, "smoothed state")
        close_smoothed(self, py, cx, self.last_model, "smoothed covariance")

    def test_bank_of_windows(self):
        z = strain(PARTIALS, [1.0, 0.4, 0.15], [1.3e4, 1.8e4, 2.1e4], 0.05, seed=2)
        py, cx = reference(bank, z, smooth=True), on_cpp(self, bank, z, smooth=True)
        close(self, py.xk, cx.xk, STATE_RTOL, "smoothed state")
        close_smoothed(self, py, cx, self.last_model, "smoothed covariance")
        # The record itself: the reconstruction from the C++ pass is the Python one.
        model = bank()
        recon = model.reconstruct(model.filter(z, smooth=True))
        ref = bank()
        with native.python_reference(), \
                mock.patch.object(KalmanFilter, "_native_forward_eligible", return_value=False):
            want = ref.reconstruct(ref.filter(z, smooth=True))
        close(self, want, recon, STATE_RTOL, "reconstruction")

    def test_interferometer(self):
        peaks = [f for band in InterferometerK1.preloaded_peaks("O3GK") for f, _ in band]
        lines = peaks[::20]
        z = strain(lines, [1.0] * len(lines), [1e5] * len(lines), 0.1, T=1.0, seed=3)

        def make():
            return InterferometerK1.from_peaks("O3GK")
        self.assertIs(make().structure, S.MATRICES)
        py, cx = reference(make, z), on_cpp(self, make, z)
        # 88 modes in three windows, about thirty phasors sharing each 2-D
        # envelope: the covariance reaches a condition number of 8e9, and a
        # rounding of 1e-16 becomes 1e-6. The pass this replaces (the C++ EKF
        # calling the Python closures) was as far from the Python loop: 2.5e-6
        # on the state, 6.6e-7 on the covariance. Against it the C++ linear pass
        # agrees to 2e-12.
        close(self, py.xk, cx.xk, 1e-5, "state")
        close(self, py.Pk, cx.Pk, COVARIANCE_RTOL, "covariance")
        before = make()
        before.DEFAULT_ENGINE = KalmanEngineKind.EXTENDED      # the old default: the C++ loop with callbacks
        # A model of matrices on the extended engine runs on a C++ model of them
        # today; that path is put aside here to get the pass the composites used to take.
        with mock.patch.object(KalmanFilter, "_matrices_eligible", lambda self, uk_fn: False), \
                mock.patch.object(KalmanFilter, "_forward_callback", autospec=True,
                                  side_effect=KalmanFilter._forward_callback) as callback:
            old = before.filter(z)
        self.assertEqual(callback.call_count, 1)
        close(self, old.xk, cx.xk, 1e-10, "state, against the C++ loop it replaces")
        close(self, old.Pk, cx.Pk, 1e-10, "covariance, against the C++ loop it replaces")

    def test_missing_samples(self):
        z = strain([180.1], [1.0], [1.3e4], 0.1, seed=4)
        # A gap in the decimated envelope: NaN after the demodulation, both components.
        model = one_line()
        zk, _ = model.preprocess(z.reshape(-1, 1))
        zk = zk.copy()
        zk[20:30] = np.nan
        zk[40, 1] = np.nan

        def run(make):
            model = make()
            return model.filter(zk, preprocess=False)
        with native.python_reference(), \
                mock.patch.object(KalmanFilter, "_native_forward_eligible", return_value=False):
            py = run(one_line)
        cx = run(one_line)
        close(self, py.xk, cx.xk, STATE_RTOL, "state through the gap")
        close(self, py.Pk, cx.Pk, COVARIANCE_RTOL, "covariance through the gap")

    def test_a_change_after_construction_reaches_the_pass(self):
        z = strain([180.1], [1.0], [1.3e4], 0.1, seed=5)

        def make():
            model = one_line()
            model.models[0]._Q = model.models[0]._Q * 9.0     # a leaf retuned after the build
            return model
        py, cx = reference(make, z), on_cpp(self, make, z)
        close(self, py.xk, cx.xk, STATE_RTOL, "state")
        self.assertGreater(float(np.max(np.abs(np.asarray(cx.xk) - np.asarray(one_line().filter(z).xk)))), 1e-6)

    def test_what_is_not_matrices_keeps_its_path(self):
        # An h replaced after construction: code, the extended engine, the C++ loop
        # calling the Python model.
        model = one_line()
        inner = model.h
        model.h = lambda x, v=None, t=None: 1.0 * inner(x, v, t)
        self.assertIs(model.structure, S.MAPS)
        self.assertIs(model.DEFAULT_ENGINE, KalmanEngineKind.EXTENDED)

        # A part that is code makes the composite code.
        from modules.apps.gravitational.gravwave import PolarizationModel
        mixed = CompositeKalmanModel(PolarizationModel("gw", fs=100.0),
                                     AnalyticOscillatorModel("a", fs=100.0, f0=10.0, Q=100))
        self.assertIs(mixed.structure, S.MAPS)
        self.assertIs(mixed.DEFAULT_ENGINE, KalmanEngineKind.EXTENDED)

        # A declared STRUCTURE wins.
        class Declared(CompositeKalmanModel):
            STRUCTURE = S.MAPS
        a = AnalyticOscillatorModel("a", fs=100.0, f0=10.0, Q=100)
        self.assertIs(Declared(a).structure, S.MAPS)


@needs_kernels
class TestCompositeTimeVarying(unittest.TestCase):
    """A part whose H is a function of time (the lab-frame phasor) makes the
    composite time-varying: its stacked H is stored as a function of time, the
    way a model built from a function of time stores it, so the filter's pass
    over tables (and the exporter) tabulate it. It must never take the
    constant-matrix pass, which would read H once."""

    def make(self):
        return CompositeKalmanModel(PhasorOscillatorModel("p", [50.0, 100.0], fs=1000.0),
                                    AnalyticOscillatorModel("a", fs=1000.0, f0=10.0, Q=100))

    def test_declares_time_varying(self):
        model = self.make()
        self.assertIs(model.structure, S.TIME_VARYING)
        self.assertIs(model.DEFAULT_ENGINE, KalmanEngineKind.LINEAR)
        self.assertTrue(callable(model._H))
        self.assertFalse(callable(model._F))
        np.testing.assert_array_equal(model._H(t=0.013), model.H(t=0.013))
        self.assertFalse(np.allclose(model.H(t=0.0), model.H(t=0.013)))

    def test_not_the_constant_pass(self):
        rng = np.random.default_rng(6)
        t = np.arange(400) / 1000.0
        z = np.c_[np.cos(2 * np.pi * 50 * t), np.zeros_like(t), np.zeros_like(t)] + 0.1 * rng.standard_normal((400, 3))
        with mock.patch.object(KalmanFilter, "_forward_native", autospec=True,
                               side_effect=KalmanFilter._forward_native) as fused:
            cx = self.make().filter(z)
        self.assertEqual(fused.call_count, 0)
        with native.python_reference(), \
                mock.patch.object(KalmanFilter, "_native_forward_eligible", return_value=False):
            py = self.make().filter(z)
        close(self, py.xk, cx.xk, STATE_RTOL, "state")
        close(self, py.Pk, cx.Pk, COVARIANCE_RTOL, "covariance")


@needs_kernels
class TestSamplingMatrices(unittest.TestCase):
    """``sample()`` of a model of constant matrices on the linear engine: the
    state chain runs in C++ on those matrices (kalman_propagate), with the noise
    drawn by numpy, so a seeded run gives the Python run's numbers."""

    def check(self, make, T, uk=None):
        from modules.framework.sampler import KalmanSampler
        with native.python_reference():
            np.random.seed(3)
            py = make().sample(T, uk=uk)
        model = make()
        self.assertIsNotNone(KalmanSampler(model)._matrices_handle())
        np.random.seed(3)
        with mock.patch("modules.framework.native.core.propagate",
                        side_effect=__import__("modules.framework.native.core", fromlist=["x"]).propagate) as chain:
            cx = model.sample(T, uk=uk)
        self.assertEqual(chain.call_count, 1, "the chain ran in C++")
        close(self, py.xk, cx.xk, 1e-12, "sampled state")
        close(self, py.Pk, cx.Pk, 1e-12, "sampled covariance")
        close(self, py.zk, cx.zk, 1e-12, "sampled measurement")

    def test_analytic_oscillator(self):
        self.check(lambda: AnalyticOscillatorModel("A", fs=256.0, f0=10.0, fc=8.0, Q=1e4, measurement_uncertainty=2.0,
                                                   process_uncertainty=0.01), 8.0)

    def test_window_of_three_modes(self):
        # Two in-band modes and more share one measured envelope: its noise is added once.
        self.check(cluster, 4.0)

    def test_suspension_with_its_road(self):
        from modules.apps.resonators.suspensions.model import SuspensionModel
        road = np.sin(np.arange(1000) / 50.0).reshape(-1, 1, 1)
        self.check(lambda: SuspensionModel("s", fs=100.0), 10.0, uk=road)

    def test_what_is_not_matrices_samples_as_before(self):
        from modules.framework.sampler import KalmanSampler
        self.assertIsNone(KalmanSampler(PhasorOscillatorModel("p", [50.0], fs=1000.0))._matrices_handle())


class TestCompositeWithoutTheLibrary(unittest.TestCase):

    def test_structure_does_not_need_the_library(self):
        self.assertIs(one_line().structure, S.MATRICES)


if __name__ == "__main__":
    unittest.main()


@needs_kernels
class TestControlledInterferometer(unittest.TestCase):
    """The violin-bank controller retunes the windows' Q as the run goes; the pass
    still runs on the C++ model of the composite's matrices, with Q pushed at each
    step, and the controller sees what it sees in the Python loop."""

    def test_violin_bank_controller(self):
        peaks = [f for band in InterferometerK1.preloaded_peaks("O3GK") for f, _ in band]
        lines = peaks[::20]
        z = strain(lines, [1.0] * len(lines), [1e5] * len(lines), 0.1, T=0.25).reshape(-1, 1)

        def run(reference):
            model = InterferometerK1.from_peaks("O3GK")
            controller = model.make_controller()
            if reference:
                with native.python_reference():
                    record = KalmanFilter(model).forward(z, uk=controller)
            else:
                with mock.patch.object(KalmanFilter, "_forward_matrices", autospec=True,
                                       side_effect=KalmanFilter._forward_matrices) as spy:
                    record = KalmanFilter(model).forward(z, uk=controller)
                self.assertEqual(spy.call_count, 1)
            log = sum(len(c.scale_log) for a in controller._window_aqrs for c in a._ctrls)
            return record, log

        (py, log_py), (cx, log_cx) = run(True), run(False)
        self.assertEqual(log_py, log_cx)
        self.assertGreater(log_cx, 0)
        # As for the uncontrolled K1 bank above: a condition number of 8e9.
        close(self, py.xk, cx.xk, 1e-5, "state")
        close(self, py.Pk, cx.Pk, COVARIANCE_RTOL, "covariance")
