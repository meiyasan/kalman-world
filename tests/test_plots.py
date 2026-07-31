"""Smoke tests for the apps' signature figures (``model.plot(record, ...)``).

These don't assert on pixels — they verify that each bespoke ``_plot_static`` renders
a Matplotlib figure end to end without raising, exercising the (large) plotting paths in
the apps and ``io/record.py``. Runs head-less on the Agg backend.
"""
import sys
import unittest

import matplotlib
matplotlib.use("Agg")          # head-less; must precede pyplot import
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, ".")
from modules.apps import (
    BallisticLinearModel, SuspensionModel, PowerGridStateModel,
    PolarizationModel, jones_signal,
)
from modules.apps.astro.space import TransLunarModel
from modules.runtime.astro import R_EARTH, KM, D_MOON, GM_EARTH
from modules.filters import ExtendedKalmanFilter


class TestSignatureFigures(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)
        # The rocket gravity divides by |r| (0 at the default zero-state probe) — benign.
        self._errstate = np.errstate(invalid="ignore", divide="ignore")
        self._errstate.__enter__()

    def tearDown(self):
        self._errstate.__exit__(None, None, None)
        plt.close("all")

    def _assert_figure(self, fig):
        # apps return a Figure (or (fig, ax)); just confirm something renderable came back
        if isinstance(fig, tuple):
            fig = fig[0]
        self.assertIsInstance(fig, plt.Figure)

    def test_ballistic_signature_figure(self):
        b = BallisticLinearModel("gps", fs=20)
        truth = b.sample(1.0)
        self._assert_figure(b.plot(b.filter(truth.zk), sample=truth))

    def test_suspension_signature_figure(self):
        s = SuspensionModel("road", fs=256)
        rec = s.filter(0.001 * np.random.randn(256, 1), np.zeros((256, 1)))
        self._assert_figure(s.plot(rec))

    def test_powergrid_signature_figure(self):
        fs = 1024
        t = np.arange(fs) / fs
        v = 230 * np.sqrt(2) * np.sin(2 * np.pi * 50 * t) + 0.5 * np.random.randn(fs)
        pg = PowerGridStateModel("L1", f0=50, harmonics=3, fs=fs)
        self._assert_figure(pg.plot(pg.filter(v, smooth=True)))

    def test_gravwave_signature_figure(self):
        fs = 100.0
        t = np.arange(int(8 * fs)) / fs
        a = 1 + 0.1 * np.cos(2 * np.pi * 0.2 * t)
        psi = 2 * np.pi * 1.0 * t
        theta = (np.pi / 2) * np.ones_like(t)
        phi = np.pi * np.ones_like(t)
        z = np.array([jones_signal(a[k], psi[k], theta[k], phi[k]) for k in range(len(t))])
        zk = np.column_stack([z[:, 0].real, z[:, 0].imag, z[:, 1].real, z[:, 1].imag])
        zk = zk + 0.05 * np.random.randn(*zk.shape)
        gw = PolarizationModel("gw", fs=fs, sigma_n=0.5)
        gw.initial_state([np.log(a[0]), 0.0, psi[0], 2 * np.pi, theta[0], 0.0, phi[0], 0.0])
        self._assert_figure(gw.plot(gw.filter(zk, smooth=False)))

    def test_rocket_trajectory_figure(self):
        tl = TransLunarModel(dt_s=120.0, track_km=5.0)
        r0 = R_EARTH + 400 * KM
        a_tx = 0.5 * (r0 + D_MOON)
        v_tli = np.sqrt(GM_EARTH * (2.0 / r0 - 1.0 / a_tx)) + 0.034
        tl.initial_state(np.array([[r0], [0], [0], [0], [v_tli], [0]]),
                         P0=np.diag([(50 * KM) ** 2] * 3 + [0.01 ** 2] * 3))
        full = tl.sample(40 * 0.12, process_noise=False)
        est = ExtendedKalmanFilter(tl).forward(full.zk)
        self._assert_figure(tl.plot(est, sample=full))


if __name__ == "__main__":
    unittest.main()
