"""
calorimeter.py — Generic Kalman calorimeter shower models for EIC ePIC.

Detector-agnostic Kalman building blocks for sampling-calorimeter shower
reconstruction.  Particle-level interpretation (photon / neutron / decay
reconstruction) is left to the experiment-specific layer.

Hit input (from uproot / eicrecon)
-----------------------------------
  rechits_to_hits(arrays, prefix)       list of {x,y,z,E,layer} dicts

Kalman models (direct use)
--------------------------
  CalorimeterLinearModel
  CalorimeterElectromagneticModel
  CalorimeterHadronicModel

Model factories
---------------
  ecal_model_from_hits(hits, preset)    CalorimeterElectromagneticModel
  hcal_model_from_hits(hits, preset)    CalorimeterHadronicModel
"""

from __future__ import annotations

import os, sys
import numpy as np
import scipy.special as sps
from dataclasses import dataclass
from typing import List

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from ..models.linear       import KalmanLinearModel
from ..framework.constants import KalmanState, KalmanObservable
from ..framework.filter    import KalmanFilter as _KalmanFilter


# ── State / observable enumerations ──────────────────────────────────────────

class CalorimeterState4(KalmanState):
    X = "mm"; Y = "mm"; TX = "rad"; TY = "rad"

class CalorimeterState5(KalmanState):
    X = "mm"; Y = "mm"; TX = "rad"; TY = "rad"; E = "GeV"

class CalorimeterObservable2(KalmanObservable):
    X = "mm"; Y = "mm"

class CalorimeterObservable3(KalmanObservable):
    X = "mm"; Y = "mm"; E_LAYER = "GeV"


# ── Geometry config (replaces XML geometry_reader) ───────────────────────────

@dataclass
class HCALConfig:
    """
    HCAL physics parameters for the Kalman filter.

    Defaults match the EIC ePIC SiPM-on-Tile Fe/Scint HCAL.
    Override when using the compact XML values directly:

        cfg = HCALConfig(lambda_int_mm=168.0, sigma_sampling=0.55)
    """
    lambda_int_mm:  float = 168.0   # Fe nuclear interaction length [mm]
    sigma_sampling: float = 0.55    # hadronic sampling resolution (~55%/√E)


# ── Longitudinal shower profiles ─────────────────────────────────────────────

def _gamma_profile_fractions(z_layers, dz_layers, scale, a, b):
    t_lo = np.maximum(0., (z_layers - 0.5 * dz_layers) / scale)
    t_hi = (z_layers + 0.5 * dz_layers) / scale
    fracs = np.clip(sps.gammainc(a, b * t_hi) - sps.gammainc(a, b * t_lo), 0., 1.)
    total = fracs.sum()
    if total > 1e-9:
        fracs /= total
    else:
        fracs = np.ones(len(z_layers)) / len(z_layers)
    return fracs


def em_shower_profile(z_layers, dz_layers, E_GeV,
                      X0_mm=11.4, Ec_GeV=0.010, a_EM=0.50, b_EM=0.50):
    """Longo-Sestili Gamma shower fractions per layer for EM showers."""
    a = max(0.1, a_EM * np.log(max(float(E_GeV), 1e-6) / max(float(Ec_GeV), 1e-9)))
    return _gamma_profile_fractions(z_layers, dz_layers, float(X0_mm), a, float(b_EM))


def hadronic_shower_profile(z_layers, dz_layers, E_GeV,
                            lambda_mm=168.0, c_had=0.025, b_had=1.00):
    """Gamma shower fractions per layer for hadronic showers."""
    a = max(0.1, 1.0 + c_had * np.log(max(float(E_GeV), 1e-6)))
    return _gamma_profile_fractions(z_layers, dz_layers, float(lambda_mm), a, float(b_had))


# ── Measurement builder ───────────────────────────────────────────────────────

def build_measurements(hits):
    """
    Collapse hit dicts into per-layer (x_bar, y_bar, E_dep) measurements.

    Parameters
    ----------
    hits : list of dict  {x [mm], y [mm], z [mm], E [GeV]}

    Returns
    -------
    z_layers : ndarray (N,)
    R_xy     : list of ndarray (2,2)  per-layer transverse covariance
    meas     : ndarray (N, 3)         columns: [x_bar, y_bar, E_dep]
    """
    if not hits:
        raise ValueError("Empty cluster passed to build_measurements().")

    layers = {}
    for h in hits:
        layers.setdefault(float(h["z"]), []).append(h)

    z_sorted = sorted(layers)
    meas_list, R_list = [], []

    for z in z_sorted:
        hs   = layers[z]
        es   = np.array([float(h["E"]) for h in hs])
        xs   = np.array([float(h["x"]) for h in hs])
        ys   = np.array([float(h["y"]) for h in hs])
        E_tot = es.sum()
        E_w   = E_tot if E_tot > 0. else 1.

        xbar = np.dot(es, xs) / E_w
        ybar = np.dot(es, ys) / E_w
        sxx  = np.dot(es, (xs - xbar) ** 2) / E_w
        syy  = np.dot(es, (ys - ybar) ** 2) / E_w
        sxy  = np.dot(es, (xs - xbar) * (ys - ybar)) / E_w
        if len(hs) == 1:
            sxx = syy = sxy = 0.

        e2    = float(np.dot(es, es))
        n_eff = max(1.0, (E_tot ** 2 / e2) if e2 > 0. else 1.0)

        meas_list.append([xbar, ybar, E_tot])
        R_list.append(np.array([[sxx / n_eff, sxy / n_eff],
                                 [sxy / n_eff, syy / n_eff]]))

    return np.asarray(z_sorted), R_list, np.asarray(meas_list)


# ── KF model base classes ─────────────────────────────────────────────────────

class _CalorimeterBase(KalmanLinearModel):
    @property
    def engine(self):
        return self._engine

    @staticmethod
    def _add_floor(R_layers, sigma_cell):
        if sigma_cell <= 0.:
            return [np.asarray(R, dtype=float) for R in R_layers]
        floor = (sigma_cell ** 2) * np.eye(2)
        return [np.asarray(R, dtype=float) + floor for R in R_layers]


class CalorimeterLinearModel(_CalorimeterBase):
    """4-state straight-line tracker: state = [x, y, tx, ty]ᵀ."""

    NAME    = "Calorimeter Linear Track Model"
    VERSION = "1.0"

    _H_STATIC = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.]])

    @property
    def states(self): return CalorimeterState4
    @property
    def observables(self): return CalorimeterObservable2

    def __init__(self, sensor_id, z_layers, R_xy_layers, *,
                 sigma_t=1e-4, sigma_cell=5.0, x0=None, P0=None, **kwargs):
        self._z_k     = np.asarray(z_layers, dtype=float)
        self._R_k     = self._add_floor(R_xy_layers, sigma_cell)
        self._sigma_t = float(sigma_t)
        self._step    = 0
        if x0 is None: x0 = np.zeros((4, 1))
        if P0 is None: P0 = np.diag([1e6, 1e6, 1e2, 1e2]).astype(float)
        kwargs.setdefault("fs", 1)
        super().__init__(sensor_id, F=self._F_callable, H=self._H_STATIC.copy(),
                         Q=self._Q_callable, R=self._R_callable,
                         x0=x0, P0=P0, **kwargs)
        self._engine = _KalmanFilter(self)

    def _dz(self, k):
        k = min(k, len(self._z_k) - 2)
        return float(self._z_k[k + 1] - self._z_k[k])

    def _F_callable(self, t=None, x=None, u=None):
        dz = self._dz(self._step)
        return np.array([[1., 0., dz, 0.], [0., 1., 0., dz],
                         [0., 0., 1., 0.], [0., 0., 0., 1.]])

    def _Q_callable(self, t=None):
        dz = self._dz(self._step); st2 = self._sigma_t ** 2
        return st2 * np.array([
            [dz**3/3, 0,       dz**2/2, 0      ],
            [0,       dz**3/3, 0,       dz**2/2],
            [dz**2/2, 0,       dz,      0      ],
            [0,       dz**2/2, 0,       dz     ],
        ])

    def _R_callable(self, t=None):
        k = min(self._step, len(self._R_k) - 1)
        Rk = self._R_k[k]; self._step += 1
        return Rk

    def preprocess(self, zk, uk=None, **kwargs):
        self._step = 0; return zk, uk
    def inprocess(self, state_vector, **kwargs): return state_vector
    def postprocess(self, xk_hat, **kwargs): return xk_hat

    @staticmethod
    def build_measurements(hits):
        z_layers, R_xy, meas3 = build_measurements(hits)
        return z_layers, R_xy, meas3[:, :2]


class _CalorimeterEnergyModel(_CalorimeterBase):
    """5-state calorimeter model: state = [x, y, tx, ty, E]ᵀ."""

    @property
    def states(self): return CalorimeterState5
    @property
    def observables(self): return CalorimeterObservable3

    def __init__(self, sensor_id, z_layers, R_xy_layers, E_layers, E_seed, *,
                 sigma_t=1e-4, sigma_cell=5.0, sigma_sampling=0.02,
                 sigma_noise=1e-4, sigma_E_state=1e-3, x0=None, P0=None, **kwargs):

        self._z_k     = np.asarray(z_layers, dtype=float)
        N             = len(self._z_k)
        self._R_xy    = self._add_floor(R_xy_layers, sigma_cell)
        self._E_k     = np.asarray(E_layers, dtype=float)
        self._E_seed  = max(float(E_seed), 1e-9)
        self._sigma_t = float(sigma_t)
        self._sigma_s = float(sigma_sampling)
        self._sigma_n = float(sigma_noise)
        self._sigma_E = float(sigma_E_state)
        self._step    = 0

        dz = np.diff(self._z_k)
        self._dz_layers = np.append(dz, dz[-1])

        self._f_k = self._profile_fractions(
            self._z_k, self._dz_layers, self._E_seed,
            **self._profile_kwargs(kwargs))

        if x0 is None:
            meas = kwargs.pop("meas_xy", None)
            if meas is not None and len(meas) >= 2:
                x0_s = float(meas[0, 0]); y0_s = float(meas[0, 1])
                dz01 = self._z_k[1] - self._z_k[0] if N >= 2 else 1.
                tx_s = (float(meas[1, 0]) - x0_s) / dz01 if dz01 > 0 else 0.
                ty_s = (float(meas[1, 1]) - y0_s) / dz01 if dz01 > 0 else 0.
            else:
                kwargs.pop("meas_xy", None)
                x0_s = y0_s = tx_s = ty_s = 0.
            x0 = np.array([[x0_s], [y0_s], [tx_s], [ty_s], [self._E_seed]])
        else:
            kwargs.pop("meas_xy", None)

        if P0 is None:
            dz01 = float(self._z_k[1] - self._z_k[0]) if N >= 2 else 1.
            P0 = np.diag([
                sigma_cell ** 2, sigma_cell ** 2,
                (10. * sigma_cell / dz01) ** 2, (10. * sigma_cell / dz01) ** 2,
                (0.5 * self._E_seed) ** 2,
            ]).astype(float)

        kwargs.setdefault("fs", 1)
        super().__init__(sensor_id, F=self._F_callable, H=self._H_callable,
                         Q=self._Q_callable, R=self._R_callable,
                         x0=x0, P0=P0, **kwargs)
        self._engine = _KalmanFilter(self)

    def _profile_fractions(self, z_layers, dz_layers, E_seed, **kwargs):
        raise NotImplementedError

    def _profile_kwargs(self, kwargs):
        return {}

    def _dz(self, k):
        k = min(k, len(self._z_k) - 2)
        return float(self._z_k[k + 1] - self._z_k[k])

    def _F_callable(self, t=None, x=None, u=None):
        dz = self._dz(self._step)
        return np.array([[1., 0., dz, 0., 0.], [0., 1., 0., dz, 0.],
                         [0., 0., 1., 0., 0.], [0., 0., 0., 1., 0.],
                         [0., 0., 0., 0., 1.]])

    def _Q_callable(self, t=None):
        dz = self._dz(self._step); st2 = self._sigma_t ** 2
        Q4 = st2 * np.array([
            [dz**3/3, 0,       dz**2/2, 0      ],
            [0,       dz**3/3, 0,       dz**2/2],
            [dz**2/2, 0,       dz,      0      ],
            [0,       dz**2/2, 0,       dz     ],
        ])
        Q5 = np.zeros((5, 5)); Q5[:4, :4] = Q4; Q5[4, 4] = self._sigma_E ** 2
        return Q5

    def _H_callable(self, t=None, x=None):
        fk = float(self._f_k[min(self._step, len(self._f_k) - 1)])
        return np.array([[1., 0., 0., 0., 0.], [0., 1., 0., 0., 0.],
                         [0., 0., 0., 0., fk]])

    def _R_callable(self, t=None):
        k   = min(self._step, len(self._R_xy) - 1)
        fk  = float(self._f_k[k])
        sig = (self._sigma_s * np.sqrt(fk * self._E_seed) + self._sigma_n) ** 2
        R3  = np.zeros((3, 3))
        R3[:2, :2] = self._R_xy[k]
        R3[2, 2]   = max(sig, 1e-12)
        self._step += 1
        return R3

    def preprocess(self, zk, uk=None, **kwargs):
        self._step = 0; return zk, uk

    def inprocess(self, state_vector, **kwargs):
        return state_vector

    def postprocess(self, xk_hat, **kwargs):
        return xk_hat

    def extract_smoothed(self):
        """Return dict {x0, y0, tx, ty, E, P, theta_rad, phi_rad, chi2_dof}."""
        eng    = self._engine
        smooth = eng.smooth
        if smooth is None or (hasattr(smooth, '__len__') and len(smooth) == 0):
            raise RuntimeError("Call engine.forward() and engine.backward() first.")
        x_sm0, P_sm0 = smooth[0]
        x0_s, y0_s, tx_s, ty_s, E_s = np.asarray(x_sm0).ravel()[:5]
        return dict(
            x0=float(x0_s), y0=float(y0_s),
            tx=float(tx_s), ty=float(ty_s),
            E=float(E_s), P=P_sm0,
            theta_rad=float(np.hypot(tx_s, ty_s)),
            phi_rad=float(np.arctan2(ty_s, tx_s)),
            chi2_dof=self._chi2(eng),
        )

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
        N    = min(len(engine.prior), len(engine.measurement))
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
    def from_hits(cls, sensor_id, hits, **kwargs):
        z_layers, R_xy, meas = build_measurements(hits)
        E_seed = float(meas[:, 2].sum())
        return cls(sensor_id, z_layers, R_xy, meas[:, 2], E_seed,
                   meas_xy=meas[:, :2], **kwargs)

    @staticmethod
    def build_measurements(hits):
        return build_measurements(hits)


# ── Concrete Kalman models ────────────────────────────────────────────────────

class CalorimeterElectromagneticModel(_CalorimeterEnergyModel):
    """5-state EM shower model for photons (Longo-Sestili Gamma in X₀)."""
    NAME    = "Calorimeter Electromagnetic Shower Model"
    VERSION = "1.0"
    _DEFAULT_X0_MM          = 11.4
    _DEFAULT_EC_GEV         = 0.010
    _DEFAULT_A_EM           = 0.50
    _DEFAULT_B_EM           = 0.50
    _DEFAULT_SIGMA_SAMPLING = 0.02
    _DEFAULT_SIGMA_NOISE    = 1e-4

    def __init__(self, sensor_id, z_layers, R_xy_layers, E_layers, E_seed, **kwargs):
        self._X0_mm  = float(kwargs.pop("X0_mm",  self._DEFAULT_X0_MM))
        self._Ec_GeV = float(kwargs.pop("Ec_GeV", self._DEFAULT_EC_GEV))
        self._a_EM   = float(kwargs.pop("a_EM",   self._DEFAULT_A_EM))
        self._b_EM   = float(kwargs.pop("b_EM",   self._DEFAULT_B_EM))
        kwargs.setdefault("sigma_sampling", self._DEFAULT_SIGMA_SAMPLING)
        kwargs.setdefault("sigma_noise",    self._DEFAULT_SIGMA_NOISE)
        super().__init__(sensor_id, z_layers, R_xy_layers, E_layers, E_seed, **kwargs)

    def _profile_fractions(self, z_layers, dz_layers, E_seed, **kw):
        return em_shower_profile(z_layers, dz_layers, E_seed,
                                 X0_mm=self._X0_mm, Ec_GeV=self._Ec_GeV,
                                 a_EM=self._a_EM, b_EM=self._b_EM)


class CalorimeterHadronicModel(_CalorimeterEnergyModel):
    """5-state hadronic shower model for neutrons (Gamma in λ_int)."""
    NAME    = "Calorimeter Hadronic Shower Model"
    VERSION = "1.0"
    _DEFAULT_LAMBDA_MM      = 168.0
    _DEFAULT_C_HAD          = 0.025
    _DEFAULT_B_HAD          = 1.00
    _DEFAULT_SIGMA_SAMPLING = 0.50
    _DEFAULT_SIGMA_NOISE    = 5e-4

    def __init__(self, sensor_id, z_layers, R_xy_layers, E_layers, E_seed, **kwargs):
        self._lambda_mm = float(kwargs.pop("lambda_mm", self._DEFAULT_LAMBDA_MM))
        self._c_had     = float(kwargs.pop("c_had",     self._DEFAULT_C_HAD))
        self._b_had     = float(kwargs.pop("b_had",     self._DEFAULT_B_HAD))
        kwargs.setdefault("sigma_sampling", self._DEFAULT_SIGMA_SAMPLING)
        kwargs.setdefault("sigma_noise",    self._DEFAULT_SIGMA_NOISE)
        super().__init__(sensor_id, z_layers, R_xy_layers, E_layers, E_seed, **kwargs)

    def _profile_fractions(self, z_layers, dz_layers, E_seed, **kw):
        return hadronic_shower_profile(z_layers, dz_layers, E_seed,
                                       lambda_mm=self._lambda_mm,
                                       c_had=self._c_had, b_had=self._b_had)


# ── Presets ───────────────────────────────────────────────────────────────────

ECAL_PRESETS = {
    "lyso": dict(X0_mm=11.4, Ec_GeV=0.010, sigma_sampling=0.020),
    "wsi":  dict(X0_mm=3.5,  Ec_GeV=0.008, sigma_sampling=0.015),
}
HCAL_PRESETS = {
    "fe_scint": dict(lambda_mm=168.0, sigma_sampling=0.50),
    "pb_scint": dict(lambda_mm=194.0, sigma_sampling=0.55),
}


def ecal_model_from_hits(hits, sensor_id=0, preset="lyso", **kwargs):
    """Build CalorimeterElectromagneticModel from hits + ECAL preset."""
    kw = dict(ECAL_PRESETS.get(preset, ECAL_PRESETS["lyso"])); kw.update(kwargs)
    return CalorimeterElectromagneticModel.from_hits(sensor_id, hits, **kw)


def hcal_model_from_hits(hits, sensor_id=0, preset="fe_scint", **kwargs):
    """Build CalorimeterHadronicModel from hits + HCAL preset."""
    kw = dict(HCAL_PRESETS.get(preset, HCAL_PRESETS["fe_scint"])); kw.update(kwargs)
    return CalorimeterHadronicModel.from_hits(sensor_id, hits, **kw)


# ── Hit conversion (uproot → KF dicts) ───────────────────────────────────────

def rechits_to_hits(event_arrays: dict, prefix: str) -> List[dict]:
    """
    Convert a single-event uproot RecHits collection to hit dicts.

    Parameters
    ----------
    event_arrays : dict
        From ``tree.arrays([...], entry_start=i, entry_stop=i+1)``.
        Keys: ``{prefix}/{prefix}.local.x`` etc.
    prefix : str
        e.g. ``"HcalEndcapPRecHits"`` or ``"EcalEndcapNRecHits"``.

    Returns
    -------
    list of dict  — keys: x, y, z [mm], E [GeV], layer [int]
    """
    def _col(sub):
        arr = event_arrays.get(f"{prefix}/{prefix}.{sub}")
        if arr is None:
            return []
        try:
            row = arr[0]
        except (IndexError, TypeError):
            row = arr
        return [float(v) for v in row]

    xs     = _col("local.x")
    ys     = _col("local.y")
    zs     = _col("local.z")
    es     = _col("energy")
    layers = _col("layer")
    return [{"x": xs[i], "y": ys[i], "z": zs[i],
             "E": es[i], "layer": int(layers[i])}
            for i in range(len(xs))]
