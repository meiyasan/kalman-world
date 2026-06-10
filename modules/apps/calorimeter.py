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

from __future__ import annotations

import numpy as np
import scipy.special as sps
from dataclasses import dataclass
from typing import List, Optional

import os, sys
_HERE    = os.path.dirname(os.path.abspath(__file__))
_ROOT_KF = os.path.dirname(os.path.dirname(_HERE))
if _ROOT_KF not in sys.path:
    sys.path.insert(0, _ROOT_KF)

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

        half = self.aperture_mm * height_frac
        seen_mats = {}
        z = self._z_start
        for layer in self._layers:
            mat   = layer.material
            color = MATERIAL_COLORS.get(mat, "#cccccc")
            alpha = 0.90 if layer.is_active else 0.45
            lw    = 0.6  if layer.is_active else 0.3
            rect  = mpatches.FancyBboxPatch(
                (z, -half), layer.thickness_mm, 2 * half,
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
        ax.set_ylim(-half * 1.15, half * 1.15)
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

class _CaloBase(KalmanLinearModel):
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
        return np.array([[1., 0., dz, 0.], [0., 1., 0., dz],
                         [0., 0., 1., 0.], [0., 0., 0., 1.]])

    def _Q(self, t=None):
        dz = self._dz(self._step); s2 = self._sigma_t ** 2
        return s2 * np.array([
            [dz**3/3, 0,       dz**2/2, 0      ],
            [0,       dz**3/3, 0,       dz**2/2],
            [dz**2/2, 0,       dz,      0      ],
            [0,       dz**2/2, 0,       dz     ],
        ])

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
        return np.array([[1., 0., dz, 0., 0.], [0., 1., 0., dz, 0.],
                         [0., 0., 1., 0., 0.], [0., 0., 0., 1., 0.],
                         [0., 0., 0., 0., 1.]])

    def _Q(self, t=None):
        dz = self._dz(self._step); s2 = self._sigma_t ** 2
        Q4 = s2 * np.array([
            [dz**3/3, 0,       dz**2/2, 0      ],
            [0,       dz**3/3, 0,       dz**2/2],
            [dz**2/2, 0,       dz,      0      ],
            [0,       dz**2/2, 0,       dz     ],
        ])
        Q = np.zeros((5, 5)); Q[:4, :4] = Q4; Q[4, 4] = self._sigma_E ** 2
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
