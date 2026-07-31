"""Shower sampling + longitudinal profiles (EM / hadronic)."""
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


class CalorimeterSampler:
    """
    Monte Carlo shower sampler for a CalorimeterGeometry.

    Generates synthetic calorimeter hits without any external framework.
    Transverse energy is distributed over cells by integrating a 2D Gaussian
    profile using the error function — giving a physically realistic, dense
    hit cluster per layer rather than a handful of randomly placed points.
    """

    def __init__(self, geometry: CalorimeterGeometry):
        self._geo = geometry

    def sample(self, E_GeV: float, particle: str = "gamma",
               x0: float = 0., y0: float = 0.,
               tx: float = 0., ty: float = 0.,
               E_threshold_GeV: float = 1e-4,
               seed=None) -> List[dict]:
        """
        Simulate one particle shower and return hit list.

        Each active layer receives a fraction of the total energy (from the
        longitudinal shower profile). That energy is then split across cells
        by integrating a 2D Gaussian over each cell area (erf-based) — so the
        hit pattern is physically correct and dense, not randomly scattered.

        Parameters
        ----------
        E_GeV           : particle kinetic energy [GeV]
        particle        : "gamma" (EM) or "neutral" / "neutron" (hadronic)
        x0, y0          : entry transverse position [mm]
        tx, ty          : track slopes [mm/mm]
        E_threshold_GeV : minimum cell energy to record (noise floor)
        seed            : random seed (int or None)

        Returns
        -------
        list of {x, y, z, E}  — multiple hits per active layer
        """
        from scipy.special import erf as _erf
        _sqrt2 = np.sqrt(2.)

        rng = np.random.default_rng(seed)
        geo = self._geo
        if geo.n_active == 0:
            raise ValueError("Geometry has no active layers.")

        if particle == "gamma":
            fracs   = _em_fracs_from_depth(geo.t_X0_active, geo.dt_X0_active, E_GeV)
            # 2D Gaussian σ so that 90% falls within Molière radius R_M:
            #   P(r < R_M) = 1 - exp(-R_M²/2σ²) = 0.9  →  σ = R_M / sqrt(2 ln 10)
            sigma_t = geo.sigma_t_em_mm
            sigma_s = geo.sigma_sampling_em
        elif particle in ("neutral", "neutron", "hadron"):
            fracs   = _had_fracs_from_depth(geo.t_lam_active, geo.dt_lam_active, E_GeV)
            sigma_t = geo.sigma_t_had_mm   # 90% lat. containment ≈ 0.5 λ_int
            sigma_s = geo.sigma_sampling_had
        else:
            raise ValueError(f"particle must be 'gamma' or 'neutral', got {particle!r}")

        hits = []
        for i, layer in enumerate(geo._active_layers):
            E_dep = float(E_GeV * fracs[i])
            if E_dep < 1e-9:
                continue

            # Sampling fluctuation at layer level (not cell level → no bias)
            if sigma_s > 0 and E_dep > 1e-9:
                E_dep = max(0., E_dep * (1. + rng.normal(0, sigma_s / np.sqrt(E_dep))))
            if E_dep < E_threshold_GeV * 0.01:
                continue

            # Track position at this layer (slope propagated from calorimeter entrance)
            dz  = geo.z_active[i] - geo._z_start
            x_c = x0 + tx * dz
            y_c = y0 + ty * dz
            cs  = layer.cell_size_mm
            z_c = float(geo.z_active[i])

            # Grid of cell centres covering shower core (±3σ, at least ±1.5 cell)
            r_cut = max(3. * sigma_t, 1.5 * cs)
            x_lo  = np.floor((x_c - r_cut) / cs) * cs + cs / 2
            x_hi  = x_c + r_cut + cs
            y_lo  = np.floor((y_c - r_cut) / cs) * cs + cs / 2
            y_hi  = y_c + r_cut + cs

            xs_c = np.arange(x_lo, x_hi, cs)
            ys_c = np.arange(y_lo, y_hi, cs)

            for xi in xs_c:
                # fraction of 1D Gaussian in x-strip [xi-cs/2, xi+cs/2]
                fx = 0.5 * (_erf((xi + cs/2 - x_c) / (sigma_t * _sqrt2)) -
                            _erf((xi - cs/2 - x_c) / (sigma_t * _sqrt2)))
                if fx < 1e-5:
                    continue
                for yi in ys_c:
                    fy = 0.5 * (_erf((yi + cs/2 - y_c) / (sigma_t * _sqrt2)) -
                                _erf((yi - cs/2 - y_c) / (sigma_t * _sqrt2)))
                    frac_cell = float(fx * fy)
                    if frac_cell < 1e-5:
                        continue
                    E_cell = E_dep * frac_cell
                    if E_cell >= E_threshold_GeV:
                        hits.append({"x": float(xi), "y": float(yi),
                                     "z": z_c, "E": float(E_cell)})
        return hits

    # ── shower diagnostics ────────────────────────────────────────────────────

    def show_shower_diagnostics(self, hits: List[dict], E_true: float,
                                particle: str = "gamma"):
        """
        Three-panel diagnostic to verify shower description:
          1. Longitudinal profile dE/dz vs theory (Gamma distribution)
          2. Transverse profile dE/dr at shower maximum vs Gaussian theory
          3. Energy containment fraction vs depth (cumulative)

        Parameters
        ----------
        hits     : hit list from sample()
        E_true   : true particle energy [GeV]
        particle : "gamma" or "neutral"
        """
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        from scipy.special import erf as _erf

        geo = self._geo
        if not hits:
            raise ValueError("Empty hit list.")

        zs = np.array([h["z"] for h in hits])
        es = np.array([h["E"] for h in hits])
        xs = np.array([h["x"] for h in hits])
        ys = np.array([h["y"] for h in hits])

        # --- per-layer sums ---
        z_unique = np.array(sorted(set(float(z) for z in zs)))
        E_by_z   = np.array([es[zs == z].sum() for z in z_unique])
        E_total  = E_by_z.sum()

        # --- theory profile ---
        # z_fine in absolute z; depth_fine = distance from calorimeter entrance.
        # em/hadronic_shower_profile expect depth in mm, NOT absolute z.
        z_fine    = np.linspace(geo._z_start, z_unique[-1] + 10, 500)
        dz_fine_v = float(z_fine[1] - z_fine[0])
        depth_fine = z_fine - geo._z_start          # [0, total_length_mm+10]
        dz_arr     = np.full_like(depth_fine, dz_fine_v)

        if particle == "gamma":
            fracs_th = em_shower_profile(depth_fine, dz_arr, E_true,
                                          X0_mm=geo.X0_eff_mm)
            sigma_t = geo.sigma_t_em_mm
            depth_label, depth_arr, dt_arr = "X₀", geo.t_X0_active, geo.dt_X0_active
        else:
            fracs_th = hadronic_shower_profile(depth_fine, dz_arr, E_true,
                                               lambda_mm=geo.lambda_eff_mm)
            sigma_t = geo.sigma_t_had_mm
            depth_label, depth_arr, dt_arr = "λ", geo.t_lam_active, geo.dt_lam_active

        dz_scale  = (z_unique[-1] - z_unique[0]) / len(z_unique) if len(z_unique) > 1 else 1.
        # dE/dz theory [GeV/mm]: fraction per fine bin × E_true / bin_width
        E_theory  = fracs_th * E_true / dz_fine_v

        # --- shower max layer ---
        i_max    = int(np.argmax(E_by_z))
        z_max    = z_unique[i_max]
        mask_max = zs == z_max
        xs_max   = xs[mask_max]; ys_max = ys[mask_max]; es_max = es[mask_max]

        # Transverse radial profile at shower max
        x_c = float(np.average(xs_max, weights=es_max))
        y_c = float(np.average(ys_max, weights=es_max))
        rs  = np.sqrt((xs_max - x_c) ** 2 + (ys_max - y_c) ** 2)
        r_max = max(rs.max() if len(rs) > 0 else sigma_t * 3, sigma_t * 3)
        r_fine = np.linspace(0, r_max, 200)
        E_layer_max = float(es_max.sum())
        # Ring-integrated 2D Gaussian: dE/dr = E * r/σ² * exp(-r²/2σ²)
        E_ring_theory = (E_layer_max * r_fine / sigma_t**2
                         * np.exp(-r_fine**2 / (2 * sigma_t**2)))
        # Cumulative containment
        cum_theory = 1. - np.exp(-r_fine**2 / (2 * sigma_t**2))

        # --- depth cumulative from active layers ---
        depth_cum_data  = np.cumsum(E_by_z) / max(E_total, 1e-9)
        depth_cum_theory = np.cumsum(fracs_th)

        fig = plt.figure(figsize=(15, 4.5))
        gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38)

        # ── 1. Longitudinal profile ──────────────────────────────────────────
        ax1 = fig.add_subplot(gs[0])
        bar_w = dz_scale * 0.65
        ax1.bar(z_unique, E_by_z / dz_scale, width=bar_w,
                color="#1E88E5" if particle == "gamma" else "#43A047",
                edgecolor="k", lw=0.5, alpha=0.85, label="Simulated hits")
        ax1.plot(z_fine, E_theory, "r-", lw=2,
                 label="Theory (Gamma profile)")
        ax1.axvline(z_max, color="orange", lw=1.2, ls="--", label=f"Shower max z={z_max:.0f}")
        ax1.set_xlabel("z [mm]"); ax1.set_ylabel("dE/dz [GeV/mm]")
        ax1.set_title("Longitudinal Profile")
        ax1.legend(fontsize=8); ax1.grid(alpha=0.25)

        # ── 2. Transverse profile at shower max ──────────────────────────────
        ax2 = fig.add_subplot(gs[1])
        # bin hits by radius
        r_edges = np.linspace(0, r_max * 1.05, 12)
        r_mids  = 0.5 * (r_edges[:-1] + r_edges[1:])
        dr      = np.diff(r_edges)
        E_ring  = np.array([es_max[(rs >= r_edges[k]) & (rs < r_edges[k+1])].sum()
                            for k in range(len(r_edges)-1)])
        ax2.bar(r_mids, E_ring / dr, width=dr * 0.8,
                color="#1E88E5" if particle == "gamma" else "#43A047",
                edgecolor="k", lw=0.5, alpha=0.85, label="Simulated (shower max)")
        ax2.plot(r_fine, E_ring_theory, "r-", lw=2, label=f"Gaussian σ={sigma_t:.1f} mm")
        ax2.axvline(sigma_t, color="purple", lw=1, ls=":", label=f"σ_t = {sigma_t:.1f} mm")
        ax2.axvline(geo.moliere_mm if particle == "gamma" else geo.lambda_eff_mm * 0.5,
                    color="green", lw=1, ls=":", label="R₉₀%")
        ax2.set_xlabel("r [mm]"); ax2.set_ylabel("dE/dr [GeV/mm]")
        ax2.set_title(f"Transverse Profile (z={z_max:.0f} mm)")
        ax2.legend(fontsize=8); ax2.grid(alpha=0.25)

        # ── 3. Energy containment ────────────────────────────────────────────
        ax3 = fig.add_subplot(gs[2])
        rs_all  = np.sqrt((xs - x_c) ** 2 + (ys - y_c) ** 2)
        r_sort  = np.sort(rs_all)
        e_sort  = np.array([es[np.argsort(rs_all)][:k+1].sum() for k in range(len(rs_all))])
        if E_total > 0:
            ax3.plot(r_sort, e_sort / E_total, "bo-", ms=3, lw=1.5, label="Simulated")
        ax3.plot(r_fine, cum_theory, "r-", lw=2, label="2D Gaussian theory")
        ax3.axhline(0.9, color="green", lw=1, ls="--", label="90% (Molière)")
        ax3.axvline(geo.moliere_mm if particle == "gamma" else geo.lambda_eff_mm * 0.5,
                    color="green", lw=1, ls=":")
        ax3.set_xlabel("r [mm]"); ax3.set_ylabel("Fraction contained")
        ax3.set_title("Lateral Energy Containment")
        ax3.set_ylim(0, 1.05); ax3.legend(fontsize=8); ax3.grid(alpha=0.25)

        ptag = "gamma" if particle == "gamma" else "neutral"
        fig.suptitle(f"Shower Diagnostics — {E_true:.0f} GeV {ptag}  "
                     f"(E_reco = {E_total:.2f} GeV)", fontweight="bold")
        return fig


# ── Longitudinal shower profiles (depth-space) ───────────────────────────────

def _gamma_profile(t_centers, dt, a, b):
    """Gamma-distribution shower fractions in depth units."""
    t_lo   = np.maximum(0., t_centers - 0.5 * dt)
    t_hi   = t_centers + 0.5 * dt
    fracs  = np.clip(sps.gammainc(a, b * t_hi) - sps.gammainc(a, b * t_lo), 0., 1.)
    total  = fracs.sum()
    return fracs / total if total > 1e-9 else np.ones(len(t_centers)) / len(t_centers)


def _em_fracs_from_depth(t_X0_centers, dt_X0, E_GeV, Ec_GeV=0.010,
                          a_EM=0.50, b_EM=0.50):
    """EM shower fractions from cumulative X₀ depth arrays."""
    a = max(0.1, a_EM * np.log(max(float(E_GeV), 1e-6) / max(Ec_GeV, 1e-9)))
    return _gamma_profile(np.asarray(t_X0_centers), np.asarray(dt_X0), a, b_EM)


def _had_fracs_from_depth(t_lam_centers, dt_lam, E_GeV,
                           c_had=0.025, b_had=1.00):
    """Hadronic shower fractions from cumulative λ_int depth arrays."""
    a = max(0.1, 1.0 + c_had * np.log(max(float(E_GeV), 1e-6)))
    return _gamma_profile(np.asarray(t_lam_centers), np.asarray(dt_lam), a, b_had)


def em_shower_profile(z_layers, dz_layers, E_GeV,
                      X0_mm=11.4, Ec_GeV=0.010, a_EM=0.50, b_EM=0.50):
    """Longo-Sestili EM shower fractions (z in mm, homogeneous X0_mm)."""
    return _em_fracs_from_depth(
        np.asarray(z_layers) / X0_mm, np.asarray(dz_layers) / X0_mm,
        E_GeV, Ec_GeV=Ec_GeV, a_EM=a_EM, b_EM=b_EM)


def hadronic_shower_profile(z_layers, dz_layers, E_GeV,
                             lambda_mm=168.0, c_had=0.025, b_had=1.00):
    """Hadronic shower fractions (z in mm, homogeneous lambda_mm)."""
    return _had_fracs_from_depth(
        np.asarray(z_layers) / lambda_mm, np.asarray(dz_layers) / lambda_mm,
        E_GeV, c_had=c_had, b_had=b_had)


# ── Measurement builder ───────────────────────────────────────────────────────

def build_measurements(hits: List[dict]):
    """
    Collapse hit list into per-layer (x̄, ȳ, E_dep) measurements.

    Returns
    -------
    z_layers : ndarray (N,)
    R_xy     : list of (2,2) ndarrays — per-layer transverse covariance
    meas     : ndarray (N, 3)  — columns: [x̄, ȳ, E_dep]
    """
    if not hits:
        raise ValueError("Empty hit list passed to build_measurements().")

    layers = {}
    for h in hits:
        layers.setdefault(float(h["z"]), []).append(h)

    z_sorted, meas_list, R_list = sorted(layers), [], []
    for z in z_sorted:
        hs   = layers[z]
        es   = np.array([float(h["E"]) for h in hs])
        xs   = np.array([float(h["x"]) for h in hs])
        ys   = np.array([float(h["y"]) for h in hs])
        E_w  = es.sum() if es.sum() > 0. else 1.
        xbar = np.dot(es, xs) / E_w
        ybar = np.dot(es, ys) / E_w
        sxx  = np.dot(es, (xs - xbar) ** 2) / E_w
        syy  = np.dot(es, (ys - ybar) ** 2) / E_w
        sxy  = np.dot(es, (xs - xbar) * (ys - ybar)) / E_w
        if len(hs) == 1:
            sxx = syy = sxy = 0.
        e2   = float(np.dot(es, es))
        neff = max(1.0, (E_w ** 2 / e2) if e2 > 0. else 1.0)
        meas_list.append([xbar, ybar, es.sum()])
        R_list.append(np.array([[sxx / neff, sxy / neff],
                                  [sxy / neff, syy / neff]]))

    return np.asarray(z_sorted), R_list, np.asarray(meas_list)


# ── Kalman model base ─────────────────────────────────────────────────────────

