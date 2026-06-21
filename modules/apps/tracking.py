"""
Charged-particle tracking detector for the Combinatorial Kalman Filter (CKF).

A pedagogical Belle II–style tracker: a charged particle bends in a uniform axial field
``Bz`` and is sampled on a stack of **concentric cylindrical layers** (the CDC analogue),
seen here in the transverse ``x–y`` plane. The trajectory is a circular arc; a "hit" is the
``(x, y)`` position where the arc crosses a layer (equivalent to the CDC r–φ measurement at a
known radius). Because each layer carries **several hits** (signal tracks + background), track
finding is combinatorial — exactly what :class:`~kalman.filters.ckf.CombinatorialKalmanFilter`
is for.

* **State** ``x = [x, y, px, py]`` — transverse position and momentum (speed ``|p|`` is
  conserved by the field, so curvature ``1/R = q·Bz/|p|`` is constant along the arc).
* **Propagation** ``f(x, t=r_target)`` advances the arc (RK4 in arc length) until it crosses
  radius ``r_target`` — the *next layer*, supplied by the filter via the station tag ``t``.
* **Measurement** ``h(x) = [x, y]`` — the crossing position (``m = 2``), noise ``R = σ²·I``.

The model also **generates synthetic events** (:meth:`generate_event`) using its own ``f``, so
truth and filter share one propagator and the example runs with no external data.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from ..framework.models.extended import ExtendedKalmanModel
from ..framework.constants import EngineKind, ALL_ENGINES, KalmanState, KalmanObservable


# ---------------------------------------------------------------------------
#  State / observable semantics
# ---------------------------------------------------------------------------
class TrackState(KalmanState):
    X = "cm"; Y = "cm"; PX = "GeV"; PY = "GeV"


class TrackObservable(KalmanObservable):
    X = "cm"; Y = "cm"


@dataclass
class TrackingEvent:
    """One generated event: ragged hits per layer plus truth bookkeeping.

    ``hits[l]``  : (K_l, 2) array of (x, y) hit positions on layer ``l``.
    ``truth[l]`` : (K_l,)  array of truth track ids per hit (``-1`` = background).
    ``tracks``   : list of (L, 4) true state arrays (one per signal track, per crossed layer).
    ``radii``    : (L,) layer radii.
    """
    hits: list
    truth: list
    tracks: list
    radii: np.ndarray

    @property
    def n_tracks(self):
        return len(self.tracks)


class DriftChamberModel(ExtendedKalmanModel):
    """
    Transverse charged-particle tracker on concentric layers (Belle II CDC style).

    The default engine is the **combinatorial** one (track finding); the extended engine is
    also allowed (single-track fit / comparison)::

        model = DriftChamberModel(n_layers=8, Bz=0.05)
        event = model.generate_event(n_tracks=3, n_background=4, seed=0)
        tracks = CombinatorialKalmanFilter(model).forward(event.hits, layer_radii=event.radii)
    """

    NAME = "Drift Chamber (transverse tracker)"; VERSION = "1.0"

    DEFAULT_ENGINE  = EngineKind.COMBINATORIAL
    ALLOWED_ENGINES = ALL_ENGINES - {EngineKind.LINEAR}

    @property
    def states(self):      return TrackState
    @property
    def observables(self): return TrackObservable

    def __init__(self, n_layers=8, r_inner=1.0, r_step=1.0, Bz=0.05, q=+1.0,
                 sigma_hit=0.05, p_seed=1.0, q_pos=1e-6, q_mom=1e-4,
                 step=0.02, **kw):
        self.dtype = np.float64
        self.Bz = float(Bz)              # axial field (sets curvature)
        self.q  = float(q)               # particle charge (fixed for the sample)
        self.sigma_hit = float(sigma_hit)
        self.p_seed = float(p_seed)      # nominal |p| guess used when seeding
        self._step = float(step)         # RK4 arc-length sub-step (cm)
        self.radii = r_inner + r_step * np.arange(int(n_layers), dtype=float)

        self.n, self.m, self.l = 4, 2, 0
        Q = np.diag([q_pos, q_pos, q_mom, q_mom]).astype(float)
        R = (sigma_hit ** 2) * np.eye(2)
        # L = I keeps the (additive) process noise alive: the base predict derives L as a
        # numerical ∂f/∂w, which would be 0 here (f ignores w) and zero out Q.
        super().__init__("CDC", Q=Q, R=R, L=np.eye(4), fs=1.0, **kw)

    # ------------------------------------------------------------------
    #  Dynamics — circular arc in a uniform field, propagated by radius
    # ------------------------------------------------------------------
    def _rhs(self, s):
        """d(state)/d(arc length): unit-speed motion whose momentum rotates at q·Bz/|p|."""
        x, y, px, py = s
        p = np.hypot(px, py)
        if p == 0.0:
            return np.zeros(4)
        c = self.q * self.Bz / p
        return np.array([px / p, py / p, c * py, -c * px])

    def _rk4(self, s, dl):
        k1 = self._rhs(s)
        k2 = self._rhs(s + 0.5 * dl * k1)
        k3 = self._rhs(s + 0.5 * dl * k2)
        k4 = self._rhs(s + dl * k3)
        return s + dl / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

    def propagate_to_radius(self, s, r_target):
        """Advance the arc until it first crosses radius ``r_target`` (outward).

        Returns the 4-state at the crossing, or ``None`` if the track curls back before
        reaching it (within a generous arc-length budget)."""
        s = np.asarray(s, dtype=float).reshape(-1).copy()
        r0 = np.hypot(s[0], s[1])
        if r_target <= r0 + 1e-12:
            return s
        budget = 4.0 * (r_target - r0) + 20.0
        travelled = 0.0
        h = self._step
        while travelled < budget:
            s_next = self._rk4(s, h)
            if np.hypot(s_next[0], s_next[1]) >= r_target:
                lo, hi = 0.0, h                      # bisect the final sub-step onto r_target
                for _ in range(40):
                    mid = 0.5 * (lo + hi)
                    sm = self._rk4(s, mid)
                    if np.hypot(sm[0], sm[1]) >= r_target:
                        hi = mid
                    else:
                        lo = mid
                return self._rk4(s, hi)
            s = s_next
            travelled += h
        return None

    def f(self, x, u=None, w=None, t=None):
        """Propagate the state to the target-layer radius ``t`` (NaN-filled if unreachable)."""
        x = np.asarray(x, dtype=float).reshape(-1)
        if t is None:
            return x.reshape(-1, 1)
        s = self.propagate_to_radius(x, float(t))
        if s is None:
            return np.full((4, 1), np.nan)
        return s.reshape(-1, 1)

    def h(self, x, v=None, t=None):
        """Observe the transverse position (x, y)."""
        z = np.asarray(x, dtype=float).reshape(-1)[:2].reshape(2, 1)
        return z if v is None else z + np.asarray(v, dtype=float).reshape(2, 1)

    # ------------------------------------------------------------------
    #  Seeding — physics guess for a track that produced an inner hit
    # ------------------------------------------------------------------
    def seed_state(self, hit, p_sigma=None):
        """Initial ``(x0, P0)`` for a track seeded from one (inner-layer) hit.

        Assumes the track comes from the beam spot ≈ origin, so the momentum points
        radially outward with nominal magnitude ``p_seed`` and large uncertainty."""
        hit = np.asarray(hit, dtype=float).reshape(-1)
        r = np.hypot(hit[0], hit[1]) or 1.0
        d = hit[:2] / r
        x0 = np.array([hit[0], hit[1], self.p_seed * d[0], self.p_seed * d[1]]).reshape(-1, 1)
        ps = self.p_seed if p_sigma is None else float(p_sigma)
        P0 = np.diag([self.sigma_hit ** 2, self.sigma_hit ** 2, ps ** 2, ps ** 2]).astype(float)
        return x0, P0

    # ------------------------------------------------------------------
    #  Synthetic event generation (truth uses this model's own f)
    # ------------------------------------------------------------------
    def generate_event(self, n_tracks=3, n_background=4, efficiency=1.0,
                       pt_range=(0.7, 1.5), phi_range=(0.2, 1.3),
                       origin_sigma=0.05, seed=None):
        """Generate one event: ``n_tracks`` signal tracks + per-layer background hits.

        Returns a :class:`TrackingEvent`. Each track is launched near the origin with a
        random direction (``phi_range``) and transverse momentum (``pt_range``), then
        propagated outward through the layers with this model's ``f``; hits get Gaussian
        resolution noise and may be dropped (``efficiency`` < 1 → detector holes).
        Background hits are random points on each layer (truth id ``-1``)."""
        rng = np.random.default_rng(seed)
        L = len(self.radii)
        hits = [[] for _ in range(L)]
        truth = [[] for _ in range(L)]
        tracks = []

        # Space the signal tracks across the angular range (with jitter) so they emanate
        # from the common vertex at distinct directions rather than overlapping.
        base = (np.linspace(phi_range[0], phi_range[1], n_tracks) if n_tracks > 1
                else np.array([0.5 * (phi_range[0] + phi_range[1])]))
        spread = (phi_range[1] - phi_range[0]) / max(1, n_tracks)
        for tid in range(n_tracks):
            phi = base[tid] + rng.uniform(-0.15, 0.15) * spread
            pt = rng.uniform(*pt_range)
            o = rng.normal(0.0, origin_sigma, size=2)
            s = np.array([o[0], o[1], pt * np.cos(phi), pt * np.sin(phi)])
            states = []
            for l, r in enumerate(self.radii):
                s_next = self.propagate_to_radius(s, r)
                if s_next is None:
                    break                                  # track left the detector
                s = s_next
                states.append(s.copy())
                if rng.uniform() <= efficiency:
                    pos = s[:2] + rng.normal(0.0, self.sigma_hit, size=2)
                    hits[l].append(pos)
                    truth[l].append(tid)
            if states:
                tracks.append(np.array(states))

        # angular span of the signal, used to make background overlap the tracks
        for l, r in enumerate(self.radii):
            for _ in range(int(n_background)):
                ang = rng.uniform(phi_range[0] - 0.3, phi_range[1] + 0.3)
                pos = r * np.array([np.cos(ang), np.sin(ang)])
                pos = pos + rng.normal(0.0, self.sigma_hit, size=2)
                hits[l].append(pos)
                truth[l].append(-1)

        hits = [np.array(h, dtype=float).reshape(-1, 2) for h in hits]
        truth = [np.array(t, dtype=int).reshape(-1) for t in truth]
        return TrackingEvent(hits=hits, truth=truth, tracks=tracks, radii=self.radii.copy())
