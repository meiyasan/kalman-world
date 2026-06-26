"""
Combinatorial Kalman Filter (CKF) — track *finding* by track following.

Where KF/EKF/UKF/PF estimate **one** trajectory from a time series of single measurements,
the CKF reconstructs **several tracks** from an event whose every layer holds **many hits**
(signal + background). For each seed it follows the track outward layer by layer; at each
layer it opens a **branch** per hit that passes a χ² gate (plus a **hole** branch for detector
inefficiency), **prunes** to a bounded beam of best candidates, and finally **arbitrates**
shared hits so the reported tracks are disjoint. This is the Belle II CKF in miniature.

The per-step math (predict / χ² gate / Joseph update / RTS smooth) lives in
:class:`~kalman.framework.engines.combinatorial.CombinatorialKalmanEngine`; this class owns
only the search **workflow**, so it overrides ``forward`` (and ``backward``) rather than using
the base single-measurement loop::

    model  = DriftChamberModel(n_layers=8, Bz=0.05)
    event  = model.generate_event(n_tracks=3, n_background=4, seed=0)
    ckf    = CombinatorialKalmanFilter(model)
    tracks = ckf.forward(event.hits, layer_radii=event.radii)   # list[KalmanRecord]
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

from ..framework.filter import KalmanFilter
from ..framework.constants import KalmanEngineKind, KalmanStateFlavor
from ..framework.io.record import KalmanRecord
from ..framework.io.collection import KalmanRecordCollection


def _default_chi2_cut(dof: int, p: float = 0.99) -> float:
    """χ² gate threshold for ``dof`` measurement channels (falls back if SciPy absent)."""
    try:
        from scipy.stats import chi2
        return float(chi2.ppf(p, df=dof))
    except Exception:
        return {1: 6.63, 2: 9.21, 3: 11.34}.get(int(dof), 9.21)


@dataclass
class _Candidate:
    """One track hypothesis under construction (a node in the search tree)."""
    x: np.ndarray                       # current posterior mean (n×1)
    P: np.ndarray                       # current posterior covariance (n×n)
    chi2: float = 0.0                   # accumulated measurement χ² (for ranking)
    score: float = 0.0                  # chi2 + hole penalties (for pruning)
    holes: int = 0
    hit_ids: list = field(default_factory=list)   # (layer, hit_index) of assigned hits
    radii: list = field(default_factory=list)     # per assigned-hit: layer radius
    xk: list = field(default_factory=list)        # per assigned-hit: posterior mean
    Pk: list = field(default_factory=list)        # per assigned-hit: posterior cov
    zk: list = field(default_factory=list)        # per assigned-hit: the hit (m,)
    priors: list = field(default_factory=list)    # per assigned-hit: (x_pred, P_pred)
    Fc: list = field(default_factory=list)        # per assigned-hit: transition from prev posterior
    Faccum: np.ndarray = None                     # transition product since the last assigned hit

    @property
    def n_hits(self):
        return len(self.zk)

    def branch(self):
        """Copy for a new tree node (lists duplicated; arrays shared, never mutated in place)."""
        return _Candidate(
            x=self.x, P=self.P, chi2=self.chi2, score=self.score, holes=self.holes,
            hit_ids=list(self.hit_ids), radii=list(self.radii),
            xk=list(self.xk), Pk=list(self.Pk), zk=list(self.zk),
            priors=list(self.priors), Fc=list(self.Fc), Faccum=self.Faccum,
        )


class CombinatorialKalmanFilter(KalmanFilter):
    """
    Combinatorial Kalman filter (track finding with χ² gating, beam pruning, arbitration).

    Parameters (hyper-parameters of the search, analogous to UKF's ``alpha/beta/kappa``)::

        chi2_cut        per-hit χ² gate (default: 99% of a χ²_m distribution)
        max_candidates  beam width kept per seed after each layer (default 6)
        max_holes       missed layers allowed on a track (default 2)
        min_hits        minimum assigned hits to accept a track (default 3)
        shared_hit_frac max fraction of a track's hits shared with an already-accepted
                        track during arbitration (default 0.3)
        hole_penalty    score added per hole (default: one gate's worth of χ²)
    """

    NAME = "Combinatorial Kalman Filter"
    DESCRIPTION = "CKF: layer-by-layer track following with branching, pruning and hit arbitration."
    KIND = KalmanEngineKind.COMBINATORIAL

    def __init__(self, model=None, *, chi2_cut=None, max_candidates=6, max_holes=2,
                 min_hits=3, shared_hit_frac=0.3, hole_penalty=None, **kwargs):
        from ..framework.engines.combinatorial import CombinatorialKalmanEngine
        super().__init__(model, engine=CombinatorialKalmanEngine(), **kwargs)
        self.chi2_cut = float(chi2_cut) if chi2_cut is not None else _default_chi2_cut(self.m)
        self.max_candidates = int(max_candidates)
        self.max_holes = int(max_holes)
        self.min_hits = int(min_hits)
        self.shared_hit_frac = float(shared_hit_frac)
        self.hole_penalty = float(hole_penalty) if hole_penalty is not None else self.chi2_cut
        self.candidates = []      # best candidate per seed (pre-arbitration)
        self.tracks_raw = []      # accepted candidates (post-arbitration)

    # ------------------------------------------------------------------
    #  Forward pass: seed → grow (branch/prune) → arbitrate
    # ------------------------------------------------------------------
    def forward(self, hits_by_layer, *, layer_radii=None, seeds=None, record=True, **kwargs):
        radii = np.asarray(layer_radii if layer_radii is not None else self.model.radii, dtype=float)
        hits = [np.asarray(h, dtype=float).reshape(-1, self.m) for h in hits_by_layer]
        L = len(hits)
        I = np.eye(self.model.n)

        seeds = self._build_seeds(hits, radii) if seeds is None else list(seeds)

        seed_tracks = []
        for seed in seeds:
            cands = [seed]
            for l in range(1, L):
                r = float(radii[l]); layer = hits[l]
                grown = []
                for c in cands:
                    x_pred, P_pred, F = self.engine._predict(self.model, c.x, c.P, t=r)
                    if not np.all(np.isfinite(x_pred)):
                        continue                                  # track left the detector
                    Faccum = F @ c.Faccum
                    for j in range(len(layer)):
                        hit = layer[j].reshape(self.m, 1)
                        chi2, x_post, P_post = self.engine.gate(self.model, x_pred, P_pred, hit, t=r)
                        if chi2 < self.chi2_cut:
                            nc = c.branch()
                            nc.x, nc.P = x_post, P_post
                            nc.chi2 += chi2; nc.score += chi2
                            nc.hit_ids.append((l, j))
                            nc.radii.append(r)
                            nc.xk.append(x_post); nc.Pk.append(P_post)
                            nc.zk.append(layer[j])
                            nc.priors.append((x_pred, P_pred)); nc.Fc.append(Faccum)
                            nc.Faccum = I
                            grown.append(nc)
                    if c.holes < self.max_holes:                  # no-hit (hole) branch
                        nc = c.branch()
                        nc.x, nc.P = x_pred, P_pred
                        nc.holes += 1; nc.score += self.hole_penalty
                        nc.Faccum = Faccum
                        grown.append(nc)
                if not grown:
                    break
                grown.sort(key=lambda c: c.score)
                cands = grown[: self.max_candidates]

            finished = [c for c in cands if c.n_hits >= self.min_hits]
            if finished:
                seed_tracks.append(min(finished, key=lambda c: (-c.n_hits, c.chi2 / c.n_hits)))

        self.candidates = seed_tracks
        self.tracks_raw = self._arbitrate(seed_tracks)
        self.tracks = KalmanRecordCollection(
            self._to_record(c, KalmanStateFlavor.POSTERIOR) for c in self.tracks_raw)
        return self.tracks if record else self

    # ------------------------------------------------------------------
    #  Backward pass: RTS-smooth each accepted track
    # ------------------------------------------------------------------
    def backward(self, **kwargs):
        assert self.tracks_raw, "No tracks to smooth; run forward() first."
        smoothed = []
        for c in self.tracks_raw:
            N = c.n_hits
            xs = [None] * N; Ps = [None] * N
            xs[-1], Ps[-1] = c.xk[-1], c.Pk[-1]
            for k in range(N - 2, -1, -1):
                xs[k], Ps[k] = self.engine._smooth(
                    self.model, (c.xk[k], c.Pk[k]), c.priors[k + 1], (xs[k + 1], Ps[k + 1]), c.Fc[k + 1]
                )
            smoothed.append(self._to_record(c, KalmanStateFlavor.SMOOTHED, xk=xs, Pk=Ps))
        self.tracks = KalmanRecordCollection(smoothed)
        return self.tracks

    def smooth(self, **kwargs):
        return self.backward(**kwargs)

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------
    def _build_seeds(self, hits, radii):
        """One seed per hit on the innermost non-empty layer."""
        seeds = []
        I = np.eye(self.model.n)
        for l in range(len(hits)):
            if len(hits[l]) == 0:
                continue
            r = float(radii[l])
            for j in range(len(hits[l])):
                hit = hits[l][j]
                x0, P0 = self.model.seed_state(hit)
                seeds.append(_Candidate(
                    x=x0, P=P0, hit_ids=[(l, j)], radii=[r],
                    xk=[x0], Pk=[P0], zk=[np.asarray(hit, float).reshape(-1)],
                    priors=[(x0, P0)], Fc=[I], Faccum=I,
                ))
            break                                   # seed only from the first populated layer
        return seeds

    def _arbitrate(self, tracks):
        """Greedy hit competition: accept best tracks, reject those overlapping too much."""
        order = sorted(tracks, key=lambda c: (-c.n_hits, c.chi2 / max(1, c.n_hits)))
        accepted, used = [], set()
        for c in order:
            ids = set(c.hit_ids)
            if len(ids & used) <= self.shared_hit_frac * len(ids):
                accepted.append(c)
                used |= ids
        return accepted

    def _to_record(self, c, flavor, xk=None, Pk=None):
        xk = c.xk if xk is None else xk
        Pk = c.Pk if Pk is None else Pk
        X = np.array([np.asarray(x, float).reshape(-1) for x in xk])
        P = np.array([np.asarray(p, float) for p in Pk])
        Z = np.array([np.asarray(z, float).reshape(-1) for z in c.zk])
        return KalmanRecord(time=np.asarray(c.radii, float), xk=X, Pk=P, zk=Z,
                            model=self.model, flavor=flavor)

    # CKF generates its own events; block the time-series sampler.
    def sample(self, *args, **kwargs):
        raise NotImplementedError("Use DriftChamberModel.generate_event() to make CKF input.")
