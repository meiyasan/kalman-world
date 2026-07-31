"""Calorimeter energy-reconstruction models (linear / EM / hadronic)."""
from __future__ import annotations
"""
calorimeter.py — Generic Kalman calorimeter: geometry, simulation, reconstruction.

Standalone — no ROOT / uproot dependency. Define a detector by a stack of named
material layers, simulate EM or hadronic showers, reconstruct with Kalman filters,
and draw geometry views.

Quick start
-----------
    from kalman.apps.calorimeter import (
        CaloLayer, CalorimeterGeometry, CalorimeterSampler,
        CalorimeterElectromagneticModel, CalorimeterHadronicModel,
    )

    # 1 — define geometry (25 PbWO4 crystals, each 23 mm)
    geo = CalorimeterGeometry(
        [CaloLayer("pwo", 23., is_active=True, cell_size_mm=22.)] * 25
    )

    # 2 — simulate a 10 GeV photon
    hits = CalorimeterSampler(geo).sample(10., "gamma", seed=42)

    # 3 — reconstruct with Kalman filter
    model = CalorimeterElectromagneticModel.from_geometry(0, hits, geo)
    _, _, meas = build_measurements(hits)
    model.engine.forward(meas); model.engine.backward()
    print(model.extract_smoothed())
"""


import numpy as np
import scipy.special as sps
from dataclasses import dataclass
from typing import List, Optional

import os, sys
_HERE    = os.path.dirname(os.path.abspath(__file__))
_ROOT_KF = os.path.dirname(os.path.dirname(_HERE))
if _ROOT_KF not in sys.path:
    sys.path.insert(0, _ROOT_KF)

from scipy.linalg import expm

from ...framework.models.linear import LinearKalmanModel
from ...helpers             import van_loan_discretize
from ...framework.constants import KalmanState, KalmanObservable
from ...framework.filter    import KalmanFilter as _KalmanFilter


# ── State / observable enumerations ──────────────────────────────────────────


class _CaloBase(LinearKalmanModel):
    @property
    def engine(self):
        return self._engine

    @staticmethod
    def _add_floor(R_layers, sigma_cell):
        floor = (sigma_cell ** 2) * np.eye(2) if sigma_cell > 0. else None
        out = []
        for R in R_layers:
            R2 = np.asarray(R, dtype=float)
            out.append(R2 + floor if floor is not None else R2)
        return out


# ── 4-state: straight-line tracker ────────────────────────────────────────────

class CalorimeterLinearModel(_CaloBase):
    """4-state straight-line tracker: state = [x, y, tx, ty]ᵀ."""
    NAME = "Calorimeter Linear Track Model"; VERSION = "1.0"

    _H = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.]])
    # Continuous-time generator w.r.t. depth z: dx/dz=tx, dy/dz=ty, dtx/dz=0,
    # dty/dz=0 (straight-line track, constant slope). L drives the transverse
    # multiple-scattering disturbance into each slope independently.
    _Adz = np.array([[0., 0., 1., 0.], [0., 0., 0., 1.],
                   [0., 0., 0., 0.], [0., 0., 0., 0.]])
    _Ldz = np.array([[0., 0.], [0., 0.], [1., 0.], [0., 1.]])

    @property
    def states(self):      return CalorimeterState4
    @property
    def observables(self): return CalorimeterObservable2

    def __init__(self, sensor_id, z_layers, R_xy_layers, *,
                 sigma_t=1e-4, sigma_cell=5.0, x0=None, P0=None, **kw):
        self._z_k = np.asarray(z_layers, dtype=float)
        self._R_k = self._add_floor(R_xy_layers, sigma_cell)
        self._sigma_t = float(sigma_t)
        self._step = 0
        if x0 is None: x0 = np.zeros((4, 1))
        if P0 is None: P0 = np.diag([1e6, 1e6, 1e2, 1e2]).astype(float)
        kw.setdefault("fs", 1)
        super().__init__(sensor_id, F=self._F, H=self._H.copy(),
                         Q=self._Q, R=self._R, x0=x0, P0=P0, **kw)
        self._engine = _KalmanFilter(self)

    def _dz(self, k):
        k = min(k, len(self._z_k) - 2)
        return float(self._z_k[k + 1] - self._z_k[k])

    def _F(self, t=None, x=None, u=None):
        dz = self._dz(self._step)
        return expm(self._Adz * dz)

    def _Q(self, t=None):
        dz = self._dz(self._step)
        Qc = (self._sigma_t ** 2) * np.eye(2)
        _, Q = van_loan_discretize(self._Adz, self._Ldz, Qc, dz)
        return Q

    def _R(self, t=None):
        k = min(self._step, len(self._R_k) - 1)
        Rk = self._R_k[k]; self._step += 1
        return Rk

    def preprocess(self, zk, uk=None, **kw):
        self._step = 0; return zk, uk
    def inprocess(self, xk_hat, **kw):   return xk_hat
    def postprocess(self, zk_hat, **kw): return zk_hat

    @staticmethod
    def build_measurements(hits):
        z, R, m = build_measurements(hits)
        return z, R, m[:, :2]


# ── 5-state: energy-aware base ────────────────────────────────────────────────

class _CaloEnergyModel(_CaloBase):
    """5-state model: state = [x, y, tx, ty, E]ᵀ."""

    # Continuous-time generator w.r.t. depth z: same straight-line (x,y,tx,ty)
    # block as CalorimeterLinearModel, plus dE/dz=0 (energy carried, not
    # dynamically modeled — its noise is a fixed per-step term, not a
    # continuous-PSD one, so it stays outside A/L/Qc; see _Q below).
    _Adz = np.array([[0., 0., 1., 0., 0.], [0., 0., 0., 1., 0.],
                   [0., 0., 0., 0., 0.], [0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0.]])
    _Ldz = np.array([[0., 0.], [0., 0.], [1., 0.], [0., 1.], [0., 0.]])

    @property
    def states(self):      return CalorimeterState5
    @property
    def observables(self): return CalorimeterObservable3

    def __init__(self, sensor_id, z_layers, R_xy_layers, E_layers, E_seed, *,
                 sigma_t=1e-4, sigma_cell=5.0, sigma_sampling=0.02,
                 sigma_noise=1e-4, sigma_E_state=1e-3, x0=None, P0=None, **kw):

        self._z_k    = np.asarray(z_layers, dtype=float)
        N            = len(self._z_k)
        self._R_xy   = self._add_floor(R_xy_layers, sigma_cell)
        self._E_k    = np.asarray(E_layers, dtype=float)
        self._E_seed = max(float(E_seed), 1e-9)
        self._sigma_t = float(sigma_t)
        self._sigma_s = float(sigma_sampling)
        self._sigma_n = float(sigma_noise)
        self._sigma_E = float(sigma_E_state)
        self._step   = 0

        dz = np.diff(self._z_k)
        self._dz_layers = np.append(dz, dz[-1])

        self._f_k = self._profile_fractions(
            self._z_k, self._dz_layers, self._E_seed,
            **self._profile_kwargs(kw))

        meas_xy = kw.pop("meas_xy", None)
        if x0 is None:
            if meas_xy is not None and len(meas_xy) >= 2:
                x0_s = float(meas_xy[0, 0]); y0_s = float(meas_xy[0, 1])
                dz01 = float(self._z_k[1] - self._z_k[0]) if N >= 2 else 1.
                tx_s = (float(meas_xy[1, 0]) - x0_s) / dz01 if dz01 > 0 else 0.
                ty_s = (float(meas_xy[1, 1]) - y0_s) / dz01 if dz01 > 0 else 0.
            else:
                x0_s = y0_s = tx_s = ty_s = 0.
            x0 = np.array([[x0_s], [y0_s], [tx_s], [ty_s], [self._E_seed]])

        if P0 is None:
            dz01 = float(self._z_k[1] - self._z_k[0]) if N >= 2 else 1.
            P0 = np.diag([sigma_cell**2, sigma_cell**2,
                          (10 * sigma_cell / dz01)**2, (10 * sigma_cell / dz01)**2,
                          (0.5 * self._E_seed)**2]).astype(float)

        kw.setdefault("fs", 1)
        super().__init__(sensor_id, F=self._F, H=self._H_cb,
                         Q=self._Q, R=self._R_cb, x0=x0, P0=P0, **kw)
        self._engine = _KalmanFilter(self)

    def _profile_fractions(self, z_layers, dz_layers, E_seed, **kw):
        raise NotImplementedError

    def _profile_kwargs(self, kw): return {}

    def _dz(self, k):
        return float(self._z_k[min(k, len(self._z_k) - 2) + 1]
                     - self._z_k[min(k, len(self._z_k) - 2)])

    def _F(self, t=None, x=None, u=None):
        dz = self._dz(self._step)
        return expm(self._Adz * dz)

    def _Q(self, t=None):
        dz = self._dz(self._step)
        Qc = (self._sigma_t ** 2) * np.eye(2)
        _, Q = van_loan_discretize(self._Adz, self._Ldz, Qc, dz)
        Q[4, 4] = self._sigma_E ** 2
        return Q

    def _H_cb(self, t=None, x=None):
        fk = float(self._f_k[min(self._step, len(self._f_k) - 1)])
        return np.array([[1., 0., 0., 0., 0.], [0., 1., 0., 0., 0.],
                         [0., 0., 0., 0., fk]])

    def _R_cb(self, t=None):
        k   = min(self._step, len(self._R_xy) - 1)
        fk  = float(self._f_k[k])
        sig = (self._sigma_s * np.sqrt(fk * self._E_seed) + self._sigma_n) ** 2
        R3  = np.zeros((3, 3)); R3[:2, :2] = self._R_xy[k]
        R3[2, 2] = max(sig, 1e-12); self._step += 1
        return R3

    def preprocess(self, zk, uk=None, **kw):
        self._step = 0; return zk, uk
    def inprocess(self, xk_hat, **kw):   return xk_hat
    def postprocess(self, zk_hat, **kw): return zk_hat

    def extract_smoothed(self) -> dict:
        """Return {x0, y0, tx, ty, E, P, theta_rad, phi_rad, chi2_dof}."""
        eng = self._engine
        if eng.smooth is None or (hasattr(eng.smooth, '__len__') and len(eng.smooth) == 0):
            raise RuntimeError("Call engine.forward() and engine.backward() first.")
        x_sm, P_sm = eng.smooth[0]
        x0, y0, tx, ty, E = np.asarray(x_sm).ravel()[:5]
        return dict(x0=float(x0), y0=float(y0), tx=float(tx), ty=float(ty),
                    E=float(E), P=P_sm,
                    theta_rad=float(np.hypot(tx, ty)),
                    phi_rad=float(np.arctan2(ty, tx)),
                    chi2_dof=self._chi2(eng))

    def _H_at(self, k):
        fk = float(self._f_k[min(k, len(self._f_k) - 1)])
        return np.array([[1., 0., 0., 0., 0.], [0., 1., 0., 0., 0.],
                         [0., 0., 0., 0., fk]])

    def _R_at(self, k):
        fk  = float(self._f_k[min(k, len(self._f_k) - 1)])
        s   = (self._sigma_s * np.sqrt(fk * self._E_seed) + self._sigma_n) ** 2
        Rxy = self._R_xy[min(k, len(self._R_xy) - 1)]
        R3  = np.zeros((3, 3)); R3[:2, :2] = Rxy; R3[2, 2] = max(s, 1e-12)
        return R3

    def _chi2(self, engine):
        chi2 = 0.; n_obs = 0
        N = min(len(engine.prior), len(engine.measurement))
        for k in range(N):
            xp, Pp = engine.prior[k]
            xp = np.asarray(xp).reshape(-1, 1)
            z  = np.asarray(engine.measurement[k]).ravel()
            fk = float(self._f_k[min(k, len(self._f_k) - 1)])
            Hk = self._H_at(k); Rk = self._R_at(k)
            if fk < 1e-6:
                Hk = Hk[:2, :]; Rk = Rk[:2, :2]; z = z[:2]
            inn = (z - (Hk @ xp).ravel()).reshape(-1, 1)
            S   = Hk @ np.asarray(Pp) @ Hk.T + Rk
            try:
                val = float((inn.T @ np.linalg.solve(S, inn)).ravel()[0])
            except np.linalg.LinAlgError:
                val = float((inn.T @ np.linalg.lstsq(S, inn, rcond=None)[0]).ravel()[0])
            chi2 += val; n_obs += len(z)
        return chi2 / max(1, n_obs - self.n)

    @classmethod
    def from_hits(cls, sensor_id, hits, **kw):
        z, R, m = build_measurements(hits)
        return cls(sensor_id, z, R, m[:, 2], m[:, 2].sum(), meas_xy=m[:, :2], **kw)

    @classmethod
    def from_geometry(cls, sensor_id, hits, geometry: CalorimeterGeometry, **kw):
        """Build model using parameters derived from a CalorimeterGeometry."""
        raise NotImplementedError

    @staticmethod
    def build_measurements(hits):
        return build_measurements(hits)


# ── Concrete models ───────────────────────────────────────────────────────────

class CalorimeterElectromagneticModel(_CaloEnergyModel):
    """5-state EM shower model for photons/gammas (Longo-Sestili in X₀)."""
    NAME = "Calorimeter Electromagnetic Shower Model"; VERSION = "1.1"

    def __init__(self, sensor_id, z_layers, R_xy_layers, E_layers, E_seed, **kw):
        self._X0_mm  = float(kw.pop("X0_mm",  11.4))
        self._Ec_GeV = float(kw.pop("Ec_GeV", 0.010))
        self._a_EM   = float(kw.pop("a_EM",   0.50))
        self._b_EM   = float(kw.pop("b_EM",   0.50))
        kw.setdefault("sigma_sampling", 0.020)
        kw.setdefault("sigma_noise",    1e-4)
        super().__init__(sensor_id, z_layers, R_xy_layers, E_layers, E_seed, **kw)

    def _profile_fractions(self, z_layers, dz_layers, E_seed, **kw):
        return em_shower_profile(z_layers, dz_layers, E_seed,
                                 X0_mm=self._X0_mm, Ec_GeV=self._Ec_GeV,
                                 a_EM=self._a_EM, b_EM=self._b_EM)

    @classmethod
    def from_geometry(cls, sensor_id, hits, geometry: CalorimeterGeometry, **kw):
        kw.setdefault("X0_mm", geometry.X0_eff_mm)
        kw.setdefault("sigma_sampling", geometry.sigma_sampling_em)
        kw.setdefault("sigma_cell",
                      geometry._active_layers[0].cell_size_mm / (12 ** 0.5)
                      if geometry._active_layers else 5.)
        return cls.from_hits(sensor_id, hits, **kw)


class CalorimeterHadronicModel(_CaloEnergyModel):
    """5-state hadronic shower model for neutral particles (Gamma in λ_int)."""
    NAME = "Calorimeter Hadronic Shower Model"; VERSION = "1.1"

    def __init__(self, sensor_id, z_layers, R_xy_layers, E_layers, E_seed, **kw):
        self._lambda_mm = float(kw.pop("lambda_mm", 168.0))
        self._c_had     = float(kw.pop("c_had",     0.025))
        self._b_had     = float(kw.pop("b_had",     1.00))
        kw.setdefault("sigma_sampling", 0.50)
        kw.setdefault("sigma_noise",    5e-4)
        super().__init__(sensor_id, z_layers, R_xy_layers, E_layers, E_seed, **kw)

    def _profile_fractions(self, z_layers, dz_layers, E_seed, **kw):
        return hadronic_shower_profile(z_layers, dz_layers, E_seed,
                                       lambda_mm=self._lambda_mm,
                                       c_had=self._c_had, b_had=self._b_had)

    @classmethod
    def from_geometry(cls, sensor_id, hits, geometry: CalorimeterGeometry, **kw):
        kw.setdefault("lambda_mm", geometry.lambda_eff_mm)
        kw.setdefault("sigma_sampling", geometry.sigma_sampling_had)
        kw.setdefault("sigma_cell",
                      geometry._active_layers[0].cell_size_mm / (12 ** 0.5)
                      if geometry._active_layers else 10.)
        return cls.from_hits(sensor_id, hits, **kw)


# ── Convenience presets ───────────────────────────────────────────────────────

ECAL_PRESETS = {
    "lyso": dict(X0_mm=11.4, Ec_GeV=0.010, sigma_sampling=0.020),
    "pwo":  dict(X0_mm= 8.9, Ec_GeV=0.009, sigma_sampling=0.015),
    "wsi":  dict(X0_mm= 3.5, Ec_GeV=0.008, sigma_sampling=0.015),
}
HCAL_PRESETS = {
    "fe_scint": dict(lambda_mm=168.0, sigma_sampling=0.50),
    "pb_scint": dict(lambda_mm=194.0, sigma_sampling=0.55),
    "cu_scint": dict(lambda_mm=153.0, sigma_sampling=0.45),
}


def ecal_model_from_hits(hits, sensor_id=0, preset="lyso", **kw):
    """Build CalorimeterElectromagneticModel from hits + named ECAL preset."""
    p = dict(ECAL_PRESETS.get(preset, ECAL_PRESETS["lyso"])); p.update(kw)
    return CalorimeterElectromagneticModel.from_hits(sensor_id, hits, **p)


def hcal_model_from_hits(hits, sensor_id=0, preset="fe_scint", **kw):
    """Build CalorimeterHadronicModel from hits + named HCAL preset."""
    p = dict(HCAL_PRESETS.get(preset, HCAL_PRESETS["fe_scint"])); p.update(kw)
    return CalorimeterHadronicModel.from_hits(sensor_id, hits, **p)
