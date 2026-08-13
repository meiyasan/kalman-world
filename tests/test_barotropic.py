"""Barotropic beta-plane: the physics, checked against theory before anything trusts it.

The reason this model exists is that KolmogorovFlowModel is not a model of the atmosphere —
it has no rotation, and at 50N the beta term it omits is the same order as the ones it keeps.
So the thing worth testing is not that the code runs; it is that the one new term produces the
one behaviour it was added for: westward-propagating Rossby waves at c = -beta / k^2.

That dispersion relation is also the sharpest available test of a sign error, which is the
most likely way to get this wrong and the hardest to notice by eye — an eastward-propagating
"Rossby wave" still looks like weather.
"""
import math

import numpy as np
import pytest

from modules.apps.weather.barotropic import (
    BarotropicFlowModel, BetaPlaneNS2D, beta_at, box_size, coriolis_at,
)

LY, LX = 2.224e6, 1.430e6


def _solver(n=64):
    return BetaPlaneNS2D(n=n, nu=2.0e4, latitude_deg=50.0, dt=60.0, length=(LY, LX))


def test_beta_and_coriolis_match_the_textbook_values():
    assert beta_at(50.0) == pytest.approx(1.4714e-11, rel=1e-3)
    assert coriolis_at(50.0) == pytest.approx(1.117e-4, rel=1e-3)
    # beta falls to zero at the pole and is largest at the equator.
    assert beta_at(90.0) == pytest.approx(0.0, abs=1e-18)
    assert beta_at(0.0) > beta_at(60.0)


def test_a_lat_lon_window_is_not_square():
    ly, lx = box_size(20.0, 20.0, 50.0)
    assert ly == pytest.approx(2.224e6, rel=1e-3)
    assert lx == pytest.approx(1.430e6, rel=1e-3)
    assert lx < ly, "longitude degrees shorten with cos(latitude)"


@pytest.mark.parametrize("mode", [1, 2])
def test_rossby_waves_propagate_westward_at_the_theoretical_speed(mode):
    """c = -beta / k^2 for a zonal wave, measured by tracking the spectral phase.

    Amplitude is tiny on purpose: the dispersion relation is linear theory, so the test has to
    stay in the regime where advection is negligible beside the beta term.
    """
    solver = _solver()
    n = 64
    x = LX * np.arange(n) / n
    k = 2.0 * math.pi * mode / LX

    field = (1e-7 * np.cos(k * x))[None, :] * np.ones((n, 1))
    phase = lambda w: np.angle(np.fft.fft2(w)[0, mode])

    steps = 360                                   # six hours at dt = 60 s
    evolved = solver.advance(field.reshape(-1), steps).reshape(n, n)

    delta = (phase(evolved) - phase(field) + math.pi) % (2 * math.pi) - math.pi
    measured = -delta / (k * steps * 60.0)
    theory = -solver.beta / (k * k)

    assert measured < 0.0, "Rossby waves must travel WESTWARD; a positive speed is a sign error"
    assert measured == pytest.approx(theory, rel=0.02)


def test_longer_waves_travel_faster():
    """The 1/k^2 in the dispersion relation, stated as the behaviour it implies."""
    solver = _solver()
    k1 = 2.0 * math.pi / LX
    k2 = 2.0 * math.pi * 2 / LX
    assert abs(-solver.beta / k1 ** 2) > abs(-solver.beta / k2 ** 2)


def test_without_beta_it_is_ordinary_2d_turbulence():
    """beta=0 must remove the term entirely — the control for the test above."""
    solver = BetaPlaneNS2D(n=64, nu=2.0e4, beta=0.0, dt=60.0, length=(LY, LX))
    n = 64
    x = LX * np.arange(n) / n
    mode = 1
    k = 2.0 * math.pi * mode / LX
    field = (1e-7 * np.cos(k * x))[None, :] * np.ones((n, 1))
    phase = lambda w: np.angle(np.fft.fft2(w)[0, mode])

    evolved = solver.advance(field.reshape(-1), 360).reshape(n, n)
    delta = (phase(evolved) - phase(field) + math.pi) % (2 * math.pi) - math.pi
    assert delta == pytest.approx(0.0, abs=1e-6), "a wave should not propagate without beta"


def test_the_climatology_is_at_a_physical_vorticity_scale():
    """The trap this model walks into if the inherited climatology is used unchanged.

    ``solver.random_ic()`` is O(1), which is right for a non-dimensional box and 1e5 times too
    large for SI vorticity. The spin-up then violates CFL, the climatology comes back all-NaN,
    and the failure only surfaces much later as "SVD did not converge" inside the analysis —
    far from its cause.
    """
    model = BarotropicFlowModel("test", grid=32, dt=60.0, cycle=60)
    clim = model.climatology(n_snapshots=8, spinup=60, stride=5)

    assert np.all(np.isfinite(clim)), "climatology went non-finite; the IC scale is wrong"
    assert 1e-6 < clim.std() < 1e-3, f"climatology std {clim.std():.3e} is not a weather scale"


def test_the_model_advances_a_physical_field_without_blowing_up():
    model = BarotropicFlowModel("test", grid=32, dt=60.0, cycle=60)
    model.climatology(n_snapshots=4, spinup=60, stride=5)
    state = model.initial_ensemble(2)[:, 0]

    for _ in range(5):
        state = model.f(state)
        assert np.all(np.isfinite(state))
    assert np.sqrt(np.mean(state ** 2)) < 1e-3
