"""Calorimeter geometry — layers, materials, Moliere radius."""
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


class CalorimeterState4(KalmanState):
    X = "mm"; Y = "mm"; TX = "rad"; TY = "rad"

class CalorimeterState5(KalmanState):
    X = "mm"; Y = "mm"; TX = "rad"; TY = "rad"; E = "GeV"

class CalorimeterObservable2(KalmanObservable):
    X = "mm"; Y = "mm"

class CalorimeterObservable3(KalmanObservable):
    X = "mm"; Y = "mm"; E_LAYER = "GeV"


# ── Material database ─────────────────────────────────────────────────────────

MATERIALS = {
    # key: X0_mm (radiation length), lambda_int_mm (nuclear), label, sigma_s (sampling res.)
    "lyso":       {"X0_mm":  11.4, "lambda_int_mm": 210.0, "label": "LYSO",              "sigma_s": 0.020},
    "pwo":        {"X0_mm":   8.9, "lambda_int_mm": 200.0, "label": "PbWO₄",        "sigma_s": 0.015},
    "wsi":        {"X0_mm":   3.5, "lambda_int_mm":  96.0, "label": "W/Si",              "sigma_s": 0.015},
    "pb":         {"X0_mm":   5.6, "lambda_int_mm": 194.0, "label": "Lead",              "sigma_s": None},
    "fe":         {"X0_mm":  17.6, "lambda_int_mm": 168.0, "label": "Iron",              "sigma_s": None},
    "w":          {"X0_mm":   3.5, "lambda_int_mm":  96.0, "label": "Tungsten",          "sigma_s": None},
    "cu":         {"X0_mm":  14.4, "lambda_int_mm": 153.0, "label": "Copper",            "sigma_s": None},
    "scint":      {"X0_mm": 400.0, "lambda_int_mm": 700.0, "label": "Scintillator",      "sigma_s": 0.10},
    # SiPM-on-tile: 25 cm² hexagonal tiles (≈53 mm flat-to-flat, 3 mm thick)
    # Same bulk physics as scint, but hex cell_shape for drawing and better pos resolution
    "scint_tile": {"X0_mm": 400.0, "lambda_int_mm": 700.0, "label": "SiPM-on-Tile",     "sigma_s": 0.50},
    "si":         {"X0_mm":  93.7, "lambda_int_mm": 465.0, "label": "Silicon",           "sigma_s": 0.05},
    "air":        {"X0_mm": 3.04e5,"lambda_int_mm":7.49e5, "label": "Air",               "sigma_s": None},
}

# Active scintillator layers → RED (detector-drawing convention)
MATERIAL_COLORS = {
    "lyso":       "#4FC3F7",   # light blue (crystal ECAL)
    "pwo":        "#29B6F6",   # blue (crystal ECAL)
    "wsi":        "#039BE5",   # dark blue (W/Si ECAL)
    "pb":         "#90A4AE",   # gray (absorber)
    "fe":         "#78909C",   # blue-gray (absorber)
    "w":          "#546E7A",   # dark gray (absorber)
    "cu":         "#FF8A65",   # orange (absorber)
    "scint":      "#E53935",   # RED — active scintillator pads
    "scint_tile": "#C62828",   # DARK RED — SiPM-on-tile pads
    "si":         "#CE93D8",   # purple — silicon sensors
    "air":        "#F5F5F5",
}

_EC_GEV = {  # critical energy for Molière radius R_M ≈ 21.2 MeV·X0/Ec
    "lyso": 0.010, "pwo": 0.009, "wsi": 0.008,
    "pb":   0.007, "fe":  0.022, "w":   0.008,
    "cu":   0.019, "scint":0.080,"scint_tile":0.080, "si": 0.040, "air": 0.085,
}



def moliere_radius_mm(material: str) -> float:
    """Estimate Molière radius [mm]: R_M ≈ 21.2 MeV · X₀ / Ec."""
    mat = MATERIALS.get(material, {})
    return 0.0212 * mat.get("X0_mm", 100.) / _EC_GEV.get(material, 0.040)


# ── Layer & geometry ──────────────────────────────────────────────────────────

@dataclass
class CaloLayer:
    """One slab in a calorimeter stack."""
    material: str        # key in MATERIALS
    thickness_mm: float  # along z
    is_active: bool      # True = sensitive / readout
    cell_size_mm: float  = 30.0   # transverse cell/pad pitch [mm]
    sigma_sampling: Optional[float] = None  # override material default

    @property
    def X0_mm(self) -> float:
        return MATERIALS[self.material]["X0_mm"]

    @property
    def lambda_int_mm(self) -> float:
        return MATERIALS[self.material]["lambda_int_mm"]

    @property
    def t_X0(self) -> float:
        return self.thickness_mm / self.X0_mm

    @property
    def t_lambda(self) -> float:
        return self.thickness_mm / self.lambda_int_mm

    @property
    def sigma_s(self) -> float:
        if self.sigma_sampling is not None:
            return self.sigma_sampling
        return MATERIALS[self.material].get("sigma_s") or 0.10

    @property
    def pos_resolution_mm(self) -> float:
        """Intrinsic position resolution from cell pitch: sigma = cell/sqrt(12)."""
        return self.cell_size_mm / (12 ** 0.5)


class CalorimeterGeometry:
    """
    Calorimeter geometry defined by a layer stack.

    Parameters
    ----------
    layers : list of CaloLayer
    z_start : float   Upstream face z [mm].
    aperture_mm : float   Transverse full-width [mm].
    """

    def __init__(self, layers: List[CaloLayer],
                 z_start: float = 0., aperture_mm: float = 200.):
        self._layers   = list(layers)
        self._z_start  = float(z_start)
        self.aperture_mm = float(aperture_mm)
        self._build()

    # ── internal build ────────────────────────────────────────────────────────

    def _build(self):
        z = self._z_start
        cum_X0 = 0.; cum_lam = 0.
        z_iface = [z]

        z_act, dz_act = [], []
        t_X0_act, dt_X0_act = [], []
        t_lam_act, dt_lam_act = [], []
        act_layers = []

        for layer in self._layers:
            dX0  = layer.t_X0
            dlam = layer.t_lambda
            if layer.is_active:
                z_ctr = z + 0.5 * layer.thickness_mm
                z_act.append(z_ctr)
                dz_act.append(layer.thickness_mm)
                t_X0_act.append(cum_X0 + 0.5 * dX0)
                dt_X0_act.append(dX0)
                t_lam_act.append(cum_lam + 0.5 * dlam)
                dt_lam_act.append(dlam)
                act_layers.append(layer)
            cum_X0 += dX0
            cum_lam += dlam
            z += layer.thickness_mm
            z_iface.append(z)

        self._z_interfaces = np.asarray(z_iface)
        self.z_active      = np.asarray(z_act)
        self.dz_active     = np.asarray(dz_act)
        self.t_X0_active   = np.asarray(t_X0_act)
        self.dt_X0_active  = np.asarray(dt_X0_act)
        self.t_lam_active  = np.asarray(t_lam_act)
        self.dt_lam_active = np.asarray(dt_lam_act)
        self._active_layers = act_layers
        self.total_X0      = cum_X0
        self.total_lambda  = cum_lam
        self.total_length_mm = z - self._z_start

    # ── derived physics ───────────────────────────────────────────────────────

    @property
    def n_active(self) -> int:
        return len(self.z_active)

    @property
    def X0_eff_mm(self) -> float:
        """mm per radiation length (for shower profile scaling)."""
        return self.total_length_mm / max(self.total_X0, 1e-9)

    @property
    def lambda_eff_mm(self) -> float:
        """mm per interaction length."""
        return self.total_length_mm / max(self.total_lambda, 1e-9)

    @property
    def moliere_mm(self) -> float:
        """Mean Molière radius of active layers [mm]."""
        if not self._active_layers:
            return 20.
        return float(np.mean([moliere_radius_mm(l.material)
                               for l in self._active_layers]))

    @property
    def sigma_t_em_mm(self) -> float:
        """1-σ transverse EM spread [mm] from 2D-Gaussian 90% containment at R_Molière."""
        return self.moliere_mm / np.sqrt(2. * np.log(10.))  # = R_M / 2.145

    @property
    def sigma_t_had_mm(self) -> float:
        """1-σ transverse hadronic spread [mm]. 90% lat. containment ≈ 0.5 λ_int."""
        return self.lambda_eff_mm * 0.5 / np.sqrt(2. * np.log(10.))

    @property
    def hadronic_spread_mm(self) -> float:
        return self.sigma_t_had_mm

    @property
    def sigma_sampling_em(self) -> float:
        if not self._active_layers:
            return 0.020
        return float(np.mean([l.sigma_s for l in self._active_layers]))

    @property
    def sigma_sampling_had(self) -> float:
        # Use HCAL_PRESETS reference (~50%/√E for Fe/Scint) as floor;
        # the active-layer sigma_s is typically set for EM showers.
        return max(self.sigma_sampling_em, 0.50)

    def summary(self) -> str:
        return (
            f"CalorimeterGeometry\n"
            f"  Layers : {len(self._layers)} total, {self.n_active} active\n"
            f"  z range: {self._z_start:.0f} – {self._z_start + self.total_length_mm:.0f} mm\n"
            f"  Depth  : {self.total_X0:.1f} X₀,  {self.total_lambda:.2f} λ_int\n"
            f"  X₀ eff : {self.X0_eff_mm:.1f} mm/X₀   "
            f"λ eff: {self.lambda_eff_mm:.1f} mm/λ\n"
            f"  Molière : {self.moliere_mm:.1f} mm\n"
            f"  σ_s (EM): {self.sigma_sampling_em:.3f}  "
            f"σ_s (had): {self.sigma_sampling_had:.2f}"
        )

    # ── visualization ─────────────────────────────────────────────────────────

    def plot_z(self, ax=None, show_depth: bool = True, height_frac: float = 0.35,
               hits: Optional[List[dict]] = None, particle: str = ""):
        """
        Side view (z–y): colored layer rectangles with optional hit overlay.
        Active scintillator layers are shown in RED.
        If *hits* is provided they are overlaid as energy-weighted scatter points.
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        if ax is None:
            _, ax = plt.subplots(figsize=(12, 3))

        half_bar  = self.aperture_mm * height_frac   # visual bar height
        half_axis = self.aperture_mm * 0.5           # full aperture for ylim
        seen_mats = {}
        z = self._z_start
        for layer in self._layers:
            mat   = layer.material
            color = MATERIAL_COLORS.get(mat, "#cccccc")
            alpha = 0.90 if layer.is_active else 0.45
            lw    = 0.6  if layer.is_active else 0.3
            rect  = mpatches.FancyBboxPatch(
                (z, -half_bar), layer.thickness_mm, 2 * half_bar,
                boxstyle="square,pad=0", facecolor=color,
                edgecolor="k", linewidth=lw, alpha=alpha,
            )
            ax.add_patch(rect)
            if mat not in seen_mats:
                label = MATERIALS.get(mat, {}).get("label", mat)
                tag   = "active" if layer.is_active else "passive"
                seen_mats[mat] = (color, f"{label} ({tag})")
            z += layer.thickness_mm

        ax.set_xlim(self._z_start - 5, self._z_start + self.total_length_mm + 5)
        ax.set_ylim(-half_axis * 1.05, half_axis * 1.05)
        ax.set_xlabel("z [mm]"); ax.set_ylabel("y [mm]")
        title = "Calorimeter — Side View (z–y)"
        if particle:
            title += f" | {particle}"

        if show_depth and self.n_active > 0:
            ax2 = ax.twinx()
            ax2.plot(self.z_active, self.t_X0_active, "b--", lw=1.5,
                     label=r"Cumulative $X_0$")
            ax2.plot(self.z_active, self.t_lam_active, "r--", lw=1.5,
                     label=r"Cumulative $\lambda_{\rm int}$")
            ax2.set_ylabel(r"Shower depth [$X_0$ or $\lambda$]", color="navy")
            ax2.legend(loc="upper left", fontsize=8)

        if hits:
            zs = np.array([h["z"] for h in hits])
            ys = np.array([h["y"] for h in hits])
            es = np.array([h["E"] for h in hits])
            sc = ax.scatter(zs, ys, c=es, s=50, cmap="hot_r", zorder=6,
                            vmin=0, vmax=es.max() or 1, edgecolors="k", lw=0.4)
            plt.colorbar(sc, ax=ax if not show_depth else ax2, label="E [GeV]",
                         fraction=0.015, pad=0.01)
            title += " — hits"

        ax.set_title(title)
        patches = [mpatches.Patch(color=c, label=lbl)
                   for _, (c, lbl) in seen_mats.items()]
        ax.legend(handles=patches, loc="upper right", fontsize=8)
        return ax

    def plot_xy(self, ax=None, z_slice_mm: Optional[float] = None,
                hits: Optional[List[dict]] = None):
        """
        Transverse (x–y) view at *z_slice_mm*.
        If *hits* is provided the cells fired by those hits are filled
        proportionally to energy (hot colormap). Empty cells drawn as outlines.
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import matplotlib.colors as mcolors
        if ax is None:
            _, ax = plt.subplots(figsize=(5, 5))

        half = self.aperture_mm * 0.5

        if z_slice_mm is None and self.n_active > 0:
            z_slice_mm = self.z_active[self.n_active // 2]

        # find layer at z_slice
        active_layer = None
        if z_slice_mm is not None:
            z = self._z_start
            for layer in self._layers:
                if z <= z_slice_mm < z + layer.thickness_mm:
                    active_layer = layer
                    break
                z += layer.thickness_mm

        mat   = active_layer.material if active_layer else "air"
        color = MATERIAL_COLORS.get(mat, "#4FC3F7")
        label = MATERIALS.get(mat, {}).get("label", "")

        # aperture outline
        ax.add_patch(mpatches.Rectangle(
            (-half, -half), 2 * half, 2 * half,
            facecolor=color, alpha=0.12, edgecolor="k", linewidth=1.5))

        if active_layer:
            cs = active_layer.cell_size_mm
            xs_c = np.arange(-half, half, cs) + cs / 2
            ys_c = np.arange(-half, half, cs) + cs / 2

            # build hit energy map keyed by (snapped_x, snapped_y)
            hit_map: dict = {}
            if hits:
                z_vals = np.array([h["z"] for h in hits])
                # pick hits at this z layer (within ±half thickness)
                tol = (active_layer.thickness_mm / 2 + 1.) if active_layer else 5.
                for h in hits:
                    if abs(h["z"] - z_slice_mm) <= tol:
                        sx = float(np.round((h["x"] - xs_c[0]) / cs) * cs + xs_c[0])
                        sy = float(np.round((h["y"] - ys_c[0]) / cs) * cs + ys_c[0])
                        hit_map[(sx, sy)] = hit_map.get((sx, sy), 0.) + h["E"]

            E_max = max(hit_map.values()) if hit_map else 1.
            cmap  = plt.cm.hot_r

            for xc in xs_c:
                for yc in ys_c:
                    key = (float(xc), float(yc))
                    E   = hit_map.get(key, 0.)
                    fc  = cmap(E / E_max) if E > 0 else "none"
                    ec  = color if E > 0 else "lightgray"
                    lw  = 0.8   if E > 0 else 0.25
                    ax.add_patch(mpatches.Rectangle(
                        (xc - cs/2, yc - cs/2), cs, cs,
                        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=3 if E > 0 else 1))

            if hit_map:
                sm = plt.cm.ScalarMappable(cmap=cmap,
                                            norm=mcolors.Normalize(0, E_max))
                plt.colorbar(sm, ax=ax, label="E [GeV]", fraction=0.046, pad=0.04)

        z_tag = f"z = {z_slice_mm:.0f} mm" if z_slice_mm is not None else ""
        ax.set_xlim(-half * 1.05, half * 1.05)
        ax.set_ylim(-half * 1.05, half * 1.05)
        ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
        ax.set_title(f"x–y Pad Map ({label}, {z_tag})")
        ax.set_aspect("equal")
        return ax

    def plot_hits_projections(self, hits: List[dict], particle: str = "",
                              smoothed_states: Optional[list] = None,
                              z_slice_mm: Optional[float] = None):
        """
        Three-panel hit visualization:
          Left  : z–x projection — energy aggregated by (z, x) column
          Middle: z–y projection — energy aggregated by (z, y) row
          Right : x–y pad map   — cells colored by energy at z_slice layer

        Energy at each (z, pad-row/column) is summed across all hits in that
        row/column before plotting, so overlapping hits from the same z layer
        are correctly represented as one marker per row/column.
        Marker area is proportional to deposited energy.

        Parameters
        ----------
        hits            : list of {x, y, z, E} from CalorimeterSampler.sample()
        particle        : label string for the title
        smoothed_states : list from engine.smooth — overlays the KF track
        z_slice_mm      : z for the XY pad map (default: shower-maximum layer)
        """
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        from collections import defaultdict
        if not hits:
            raise ValueError("hits list is empty")

        zs = np.array([h["z"] for h in hits])
        xs = np.array([h["x"] for h in hits])
        ys = np.array([h["y"] for h in hits])
        es = np.array([h["E"] for h in hits])

        # default slice = layer with max total energy
        if z_slice_mm is None:
            E_by_z = defaultdict(float)
            for h in hits:
                E_by_z[h["z"]] += h["E"]
            z_slice_mm = max(E_by_z, key=E_by_z.get)

        # ── aggregate energy by (z, x-column) and (z, y-row) ─────────────────
        agg_zx: dict = defaultdict(float)   # (z, x) → E
        agg_zy: dict = defaultdict(float)   # (z, y) → E
        for h in hits:
            agg_zx[(h["z"], h["x"])] += h["E"]
            agg_zy[(h["z"], h["y"])] += h["E"]

        def _unpack(agg):
            keys = list(agg.keys())
            zz  = np.array([k[0] for k in keys])
            cc  = np.array([k[1] for k in keys])
            ee  = np.array([agg[k] for k in keys])
            return zz, cc, ee

        zz_x, xx, ee_x = _unpack(agg_zx)
        zz_y, yy, ee_y = _unpack(agg_zy)

        E_max = max(ee_x.max(), ee_y.max(), 1e-9)
        norm  = mcolors.Normalize(vmin=0, vmax=E_max)
        cmap  = plt.cm.hot_r

        # marker size proportional to energy (area ~ E)
        def _sz(ee):
            return np.clip(20 + 280 * ee / E_max, 8, 300)

        half  = self.aperture_mm * 0.5
        z_lo  = self._z_start - 5
        z_hi  = self._z_start + self.total_length_mm + 5

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

        # ── z–x ──────────────────────────────────────────────────────────────
        ax_zx = axes[0]
        sc = ax_zx.scatter(zz_x, xx, c=ee_x, cmap=cmap, norm=norm,
                           s=_sz(ee_x), edgecolors="k", lw=0.3, zorder=4)
        if smoothed_states is not None:
            z_sm = self.z_active[:len(smoothed_states)]
            x_sm = np.array([np.asarray(s[0]).ravel()[0] for s in smoothed_states])
            ax_zx.plot(z_sm, x_sm, "b-o", ms=4, lw=1.8, label="KF track", zorder=5)
            ax_zx.legend(fontsize=8)
        ax_zx.axvline(z_slice_mm, color="lime", lw=1.2, ls="--")
        ax_zx.set_xlabel("z [mm]"); ax_zx.set_ylabel("x [mm]")
        ax_zx.set_xlim(z_lo, z_hi)
        ax_zx.set_ylim(-half * 1.05, half * 1.05)
        ax_zx.set_title("z–x projection (Σ E per column)")
        plt.colorbar(sc, ax=ax_zx, label="E [GeV]", fraction=0.046, pad=0.04)

        # ── z–y ──────────────────────────────────────────────────────────────
        ax_zy = axes[1]
        sc2 = ax_zy.scatter(zz_y, yy, c=ee_y, cmap=cmap, norm=norm,
                            s=_sz(ee_y), edgecolors="k", lw=0.3, zorder=4)
        if smoothed_states is not None:
            y_sm = np.array([np.asarray(s[0]).ravel()[1] for s in smoothed_states])
            ax_zy.plot(z_sm, y_sm, "b-o", ms=4, lw=1.8, zorder=5)
        ax_zy.axvline(z_slice_mm, color="lime", lw=1.2, ls="--")
        ax_zy.set_xlabel("z [mm]"); ax_zy.set_ylabel("y [mm]")
        ax_zy.set_xlim(z_lo, z_hi)
        ax_zy.set_ylim(-half * 1.05, half * 1.05)
        ax_zy.set_title("z–y projection (Σ E per row)")
        plt.colorbar(sc2, ax=ax_zy, label="E [GeV]", fraction=0.046, pad=0.04)

        # ── x–y pad map ───────────────────────────────────────────────────────
        self.plot_xy(ax=axes[2], z_slice_mm=z_slice_mm, hits=hits)

        ptitle = f"{particle} — " if particle else ""
        fig.suptitle(f"{ptitle}Hit Projections (z–x | z–y | x–y pad map)",
                     fontsize=12, fontweight="bold")
        plt.tight_layout()
        return fig, axes


# ── Shower sampler ────────────────────────────────────────────────────────────

