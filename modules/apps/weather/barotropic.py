"""Barotropic vorticity on a mid-latitude beta-plane — a model of the atmosphere.

Why this exists
---------------
:class:`~.flow.KolmogorovFlowModel` is a superb data-assimilation testbed and a poor model of
the atmosphere. Assimilating measured ERA5 500 hPa vorticity into it is *worse than
persistence* — worse, in fact, than the window's own time mean — and no inflation setting
repairs that, because the failure is not tuning. Kolmogorov flow is forced 2D turbulence on a
doubly-periodic torus with no rotation; mid-latitude flow is rotation-dominated. Measured at
50°N, relative vorticity is ~0.40 of the planetary value, so the term the Kolmogorov model
omits is the same size as the ones it keeps.

The barotropic vorticity equation is the smallest correction that is still *about the right
system*:

.. code-block:: text

    ∂ζ/∂t = −u·∇ζ − β v + ν ∇²ζ

The one new term, ``−β v``, is advection of planetary vorticity: move north and f increases, so
ζ must fall. That term alone is what makes the flow support **Rossby waves**, the westward-
propagating structures that organise mid-latitude weather, and it is the physics the
Kolmogorov model cannot express at any parameter setting.

This is still not a sphere. It is a beta-plane: f linearised about a reference latitude on a
doubly-periodic Cartesian box. For a 20°×20° window in the mid-latitudes — which is exactly
what the ERA5 slice is — that is the approximation the beta-plane was invented for, and it
buys the dominant missing physics without spherical harmonics. Anything global needs real
spherical geometry and this class should not be stretched to pretend otherwise.

Units
-----
Deliberately **physical** (SI), unlike the non-dimensional Kolmogorov testbed: β is a real
number about the real planet and the model is worthless if the field it acts on has had its
amplitude rescaled to something else. Feed it ``reanalysis_truth(..., rescale=False)``.
"""
from __future__ import annotations

import numpy as np

from ...framework.solvers.spectral import SpectralSolver
from .flow import KolmogorovFlowModel

#: Earth's rotation rate, rad/s.
OMEGA = 7.292e-5

#: Earth's mean radius, m.
EARTH_RADIUS = 6.371e6


def beta_at(latitude_deg: float) -> float:
    """Rossby parameter β = df/dy = 2Ω cos(φ) / a, in 1/(m·s).

    About 1.47e-11 in the mid-latitudes. It is small in absolute terms and dominant in
    practice, because it multiplies a velocity rather than a vorticity.
    """
    return 2.0 * OMEGA * np.cos(np.radians(latitude_deg)) / EARTH_RADIUS


def coriolis_at(latitude_deg: float) -> float:
    """Planetary vorticity f = 2Ω sin(φ), 1/s — the scale ζ is measured against."""
    return 2.0 * OMEGA * np.sin(np.radians(latitude_deg))


def box_size(lat_span_deg: float, lon_span_deg: float, latitude_deg: float):
    """Physical ``(Ly, Lx)`` in metres for a lat/lon window at a reference latitude.

    Longitude degrees shorten with cos(φ), which is why a 20°×20° window at 50°N is not
    square: about 2,224 km north-south against 1,430 km east-west.
    """
    ly = np.radians(lat_span_deg) * EARTH_RADIUS
    lx = np.radians(lon_span_deg) * EARTH_RADIUS * np.cos(np.radians(latitude_deg))
    return float(ly), float(lx)


class BetaPlaneNS2D(SpectralSolver):
    """Barotropic vorticity on a beta-plane, pseudo-spectral, in SI units.

    A sibling of :class:`~.flow.NS2D` rather than a subclass: that one hard-codes a
    non-dimensional ``[0, 2π]²`` box and its Kolmogorov forcing, and this needs a physical
    rectangular domain and no forcing at all. Real mid-latitude flow is not stirred at one
    wavenumber — it is driven baroclinically from below, which a barotropic model does not
    carry — so leaving it unforced is the honest omission. It decays.
    """

    NAME = "Barotropic beta-plane"
    DESCRIPTION = "2D barotropic vorticity with the beta effect — mid-latitude flow, SI units."

    def __init__(self, n=64, nu=2.0e4, beta=None, latitude_deg=50.0, dt=60.0,
                 length=(2.224e6, 1.430e6), seed=0, **kw):
        super().__init__(n=n, dt=dt, length=length, seed=seed, **kw)
        self.nu = float(nu)
        self.beta = beta_at(latitude_deg) if beta is None else float(beta)
        self.latitude_deg = float(latitude_deg)

    def rhs_hat(self, wh, t=None):
        u, v = self._vel_spectral(wh)
        wx, wy = self._grad_spectral(wh)
        # −u·∇ζ − βv, dealiased together since both are products evaluated on the grid.
        adv = np.fft.fft2(u * wx + v * wy + self.beta * v) * self.dealias
        return -adv - self.nu * self.k2 * wh

    def _vorticity_terms(self):
        return dict(nu=self.nu, beta=self.beta)

    def _rhs_hat_torch(self, wh):                       # pragma: no cover - parity path
        t, tt = self._torch, self._tt
        psih = wh * tt["k2inv"]
        u = t.fft.ifft2(1j * tt["ky"] * psih).real
        v = t.fft.ifft2(-1j * tt["kx"] * psih).real
        wx = t.fft.ifft2(1j * tt["kx"] * wh).real
        wy = t.fft.ifft2(1j * tt["ky"] * wh).real
        adv = t.fft.fft2((u * wx + v * wy + self.beta * v).to(self._cdtype)) * tt["dealias"]
        return -adv - self.nu * tt["k2"] * wh


class BarotropicFlowModel(KolmogorovFlowModel):
    """The Kalman model wrapper, with the beta-plane solver in place of Kolmogorov flow.

    Everything the testbed already provides — sparse probes, Gaspari–Cohn localization,
    batched ensemble forecasts, the surrogate plug-in point — is inherited unchanged. Only the
    dynamics differ, which is the whole point of the comparison this class exists for.
    """

    NAME = "Barotropic beta-plane flow"
    DESCRIPTION = ("2D barotropic vorticity with the beta effect and sparse point probes — "
                   "a mid-latitude model for assimilating measured reanalysis.")
    VERSION = "1.0"

    def __init__(self, sensor_id: str, *, grid: int = 64, nu: float = 2.0e4,
                 latitude_deg: float = 50.0, beta: float | None = None,
                 dt: float = 60.0, cycle: int = 60,
                 lat_span_deg: float = 20.0, lon_span_deg: float = 20.0,
                 clim_vorticity_rms: float = 4.5e-5, seed: int = 7, **kw):
        super().__init__(sensor_id, grid=grid, nu=nu, dt=dt, cycle=cycle, seed=seed, **kw)
        # Replace the solver the parent built, and drop any climatology cached from it.
        self.solver = BetaPlaneNS2D(
            n=int(grid), nu=nu, beta=beta, latitude_deg=latitude_deg, dt=dt,
            length=box_size(lat_span_deg, lon_span_deg, latitude_deg), seed=seed)
        self._clim = None
        self.latitude_deg = float(latitude_deg)
        self.clim_vorticity_rms = float(clim_vorticity_rms)

    def climatology_from(self, fields):
        """Use MEASURED fields as the climatology, instead of a synthetic spin-up.

        An unforced barotropic model has no attractor worth sampling — it decays — so its own
        spin-up produces states that are dynamically self-consistent and meteorologically
        unlike anything the atmosphere is doing this week. An ensemble drawn from those starts
        far from the truth in a way no number of assimilation cycles can repair on a short
        window. Drawing from real fields instead is standard practice (a climatological
        ensemble) and is the honest fix.

        **The caller must pass a period DISJOINT from the one being scored.** Seeding from the
        scoring window would put the answer in the initial ensemble and every score after it
        would be circular — the same failure as tuning process noise on truth. This method
        cannot check that for you; it is the caller's contract, and the OSSE script splits the
        window explicitly.

        ``fields`` is ``(T, n)`` or ``(T, grid, grid)``; columns become ensemble candidates.
        """
        arr = np.asarray(fields, dtype=float)
        arr = arr.reshape(arr.shape[0], -1)
        if arr.shape[1] != self.n:
            raise ValueError(f"climatology fields have {arr.shape[1]} states, model has {self.n}")
        if not np.all(np.isfinite(arr)):
            raise ValueError("climatology fields contain non-finite values")
        self._clim = arr.T.copy()
        return self._clim

    def climatology(self, n_snapshots=150, spinup=600, stride=10):
        """Snapshots at a **physical** vorticity amplitude.

        The inherited version spins up from ``solver.random_ic()``, whose amplitude is O(1) —
        correct for the non-dimensional Kolmogorov box and catastrophic here, where vorticity
        is O(1e-5) 1/s. An O(1) field on this grid violates CFL on the first step and the
        climatology comes back all-NaN, which then propagates silently into the initial
        ensemble and surfaces much later as ``SVD did not converge`` inside the analysis.

        So the initial condition is rescaled to ``clim_vorticity_rms`` (4.5e-5 1/s by default,
        which is what the ERA5 slice actually measures at 500 hPa) before the spin-up, and the
        spin-up is shorter because this model is unforced: it decays on the viscous timescale
        (~30 days at nu = 2e4), so a long spin-up would quietly deflate the very scale being
        set here.
        """
        if self._clim is None:
            w0 = np.asarray(self.solver.random_ic(), dtype=float).ravel()
            scale = np.sqrt(np.mean(w0 ** 2))
            if scale > 0:
                w0 = w0 * (self.clim_vorticity_rms / scale)
            w = self.solver.advance(w0, spinup)
            snaps = []
            for _ in range(n_snapshots):
                w = self.solver.advance(w, stride)
                snaps.append(np.asarray(w).ravel().copy())
            self._clim = np.array(snaps).T
        return self._clim
