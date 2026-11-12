"""
Combinatorial Kalman engine (CKF) — track *finding* by track following.

Where the other engines estimate **one** trajectory from a time series of single
measurements, this one reconstructs **several tracks** from an event whose every layer
holds **many hits** (signal + background). For each seed it follows the track outward
layer by layer; at each layer it opens a **branch** per hit that passes a χ² gate (plus a
**hole** branch for detector inefficiency), **prunes** to a bounded beam of best
candidates, and finally **arbitrates** shared hits so the reported tracks are disjoint.
This is the Belle II CKF in miniature.

Unlike the Jacobian engines, a single predict/update step is not the whole algorithm here:
the *search* is. So this engine declares ``OWNS_PASS`` and implements :meth:`forward` /
:meth:`backward` itself, and :class:`~kalman.framework.filters.ckf.CombinatorialKalmanFilter`
stays the same thin leaf as ``LinearKalmanFilter`` — it names the engine and passes the
search hyper-parameters, and owns no algorithm of its own.

The per-step math still comes from :class:`ExtendedKalmanEngine`: nonlinear propagation
(``model.f`` / Jacobian ``model.F``), the measurement map (``model.h`` / ``model.H``) and
the RTS smoother are inherited unchanged. The additions are the χ² :meth:`gate` and the
search around it.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

from .extended import ExtendedKalmanEngine
from ..constants import KalmanEngineKind, KalmanStateFlavor
from ..io.record import KalmanRecord


def _default_chi2_cut(dof: int, p: float = 0.99) -> float:
    """χ² gate threshold for ``dof`` measurement channels (falls back if SciPy absent)."""
    try:
        from scipy.stats import chi2
        return float(chi2.ppf(p, df=dof))
    except Exception:
        return {1: 6.63, 2: 9.21, 3: 11.34}.get(int(dof), 9.21)


@dataclass
class TrackCandidate:
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
    priors: list = field(default_factory=list)    # per assigned-hit: predicted (x, P)
    Fc: list = field(default_factory=list)        # per assigned-hit: transition from prev posterior
    Faccum: np.ndarray = None                     # transition product since the last assigned hit

    @property
    def n_hits(self):
        return len(self.zk)

    @property
    def seed_layer(self):
        """Layer the search continues *after* — the last layer this candidate has visited."""
        return self.hit_ids[-1][0] if self.hit_ids else -1

    @property
    def quality(self):
        """Sort key for picking the best track: most hits first, then lowest χ²/hit.

        Used both to choose a seed's winning candidate and to order arbitration, so the
        two rankings cannot drift apart.
        """
        return (-self.n_hits, self.chi2 / max(1, self.n_hits))

    def branch(self):
        """Copy for a new tree node (lists duplicated; arrays shared, never mutated in place)."""
        return TrackCandidate(
            x=self.x, P=self.P, chi2=self.chi2, score=self.score, holes=self.holes,
            hit_ids=list(self.hit_ids), radii=list(self.radii),
            xk=list(self.xk), Pk=list(self.Pk), zk=list(self.zk),
            priors=list(self.priors), Fc=list(self.Fc), Faccum=self.Faccum,
        )

    def to_record(self, model, flavor=KalmanStateFlavor.POSTERIOR):
        """This track as a :class:`KalmanRecord`, indexed by layer radius rather than time.

        The whole-pass contract: a filter turns whatever an ``OWNS_PASS`` engine returns
        into records by calling this.
        """
        X = np.array([np.asarray(x, float).reshape(-1) for x in self.xk])
        P = np.array([np.asarray(p, float) for p in self.Pk])
        Z = np.array([np.asarray(z, float).reshape(-1) for z in self.zk])
        return KalmanRecord(time=np.asarray(self.radii, float), xk=X, Pk=P, zk=Z,
                            model=model, flavor=flavor)


class GateContext:
    """Everything about a gate that depends on the **prediction only**, not on the hit.

    At one layer a candidate is tested against every hit on it. The predicted
    measurement ``h(x)``, its Jacobian ``H``, the innovation covariance ``S``, the gain
    ``K`` — and, because the Joseph form ``(I-KH) P (I-KH)ᵀ + K R Kᵀ`` never mentions
    ``z``, the *posterior covariance itself* — are identical for all of them. Computing
    them once per (candidate, layer) instead of once per (candidate, hit) lifts the
    pseudo-inverse and the n×n products out of the inner loop.

    Build one with :meth:`CombinatorialKalmanEngine.gate_context`, then call
    :meth:`score` per hit.
    """

    __slots__ = ("zpred", "H", "R", "Sinv", "K", "P")

    def __init__(self, zpred, H, R, Sinv, K, P):
        self.zpred = zpred    # (m,1) predicted measurement h(x)
        self.H = H            # (m,n) observation Jacobian
        self.R = R            # (m,m) measurement noise
        self.Sinv = Sinv      # (m,m) inverse innovation covariance
        self.K = K            # (n,m) Kalman gain
        self.P = P            # (n,n) posterior covariance — shared by every hit

    def score(self, x, z):
        """Score one candidate measurement → ``(chi2, x_post)``.

        ``z`` is a single hit ``(m,)`` or ``(m,1)`` with no missing channels. The
        posterior covariance is :attr:`P` for every hit, so it is not returned.
        """
        y = np.asarray(z, dtype=float).reshape(-1, 1) - self.zpred
        chi2 = float((y.T @ self.Sinv @ y).item())
        return chi2, x + self.K @ y


class CombinatorialKalmanEngine(ExtendedKalmanEngine):
    """
    Extended-Kalman step + a measurement **gate**, driving a branch/prune/arbitrate search.

    Search hyper-parameters, the CKF's counterpart of the UKF's ``alpha/beta/kappa``::

        chi2_cut        per-hit χ² gate (default: 99% of a χ²_m distribution)
        max_candidates  beam width kept per seed after each layer (default 6)
        max_holes       missed layers allowed on a track (default 2)
        min_hits        minimum assigned hits to accept a track (default 3)
        shared_hit_frac max fraction of a track's hits shared with an already-accepted
                        track during arbitration (default 0.3)
        hole_penalty    score added per hole (default: one gate's worth of χ²)

    Two gate entry points, same math:

    - :meth:`gate` — one hit at a time, tolerates missing (NaN) channels.
    - :meth:`gate_context` — hoists the hit-independent work out of the loop when a
      whole layer is scored against one prediction. This is what :meth:`forward` uses.
    """

    NAME = "Combinatorial Kalman Engine"
    DESCRIPTION = "EKF track-following with a χ² gate, beam pruning and hit arbitration."
    VERSION = "2.0"
    KIND = KalmanEngineKind.COMBINATORIAL

    # The search *is* the algorithm here, so this engine drives the whole pass
    # rather than exposing a single step for the filter's loop to call.
    OWNS_PASS = True

    def __init__(self, *, chi2_cut=None, max_candidates=6, max_holes=2,
                 min_hits=3, shared_hit_frac=0.3, hole_penalty=None):
        self.candidates = []
        # chi2_cut/hole_penalty stay None until a model is in hand: the default
        # threshold depends on the number of measurement channels.
        self.chi2_cut = None if chi2_cut is None else float(chi2_cut)
        self.hole_penalty = None if hole_penalty is None else float(hole_penalty)
        self.max_candidates = int(max_candidates)
        self.max_holes = int(max_holes)
        self.min_hits = int(min_hits)
        self.shared_hit_frac = float(shared_hit_frac)

    # ------------------------------------------------------------------
    #  Hyper-parameter resolution
    # ------------------------------------------------------------------
    def gate_threshold(self, model) -> float:
        """The χ² cut in force for ``model`` — explicit if given, else 99% of χ²(m)."""
        if self.chi2_cut is not None:
            return self.chi2_cut
        return _default_chi2_cut(model.m)

    def hole_cost(self, model) -> float:
        """Score charged for a missed layer — explicit if given, else one gate's worth."""
        if self.hole_penalty is not None:
            return self.hole_penalty
        return self.gate_threshold(model)

    # ------------------------------------------------------------------
    #  The gate
    # ------------------------------------------------------------------
    def gate_context(self, model, xk, Pk, t=None) -> GateContext:
        """Prepare the hit-independent half of the gate for the prediction ``(xk, Pk)``.

        Assumes every measurement channel is present; for hits with NaN channels use
        :meth:`gate`, which compresses the system per hit.
        """
        zpred, H = self._measure(model, xk, t)
        return self._context(model, Pk,
                             np.asarray(zpred, dtype=float).reshape(-1, 1),
                             np.asarray(H, dtype=float),
                             np.asarray(model.R(t=t), dtype=float))

    @staticmethod
    def _context(model, Pk, zpred, H, R) -> GateContext:
        """Shared gain/covariance algebra — see :class:`GateContext`."""
        S = R + H @ Pk @ H.T
        Sinv = np.linalg.pinv(S)
        K = Pk @ H.T @ Sinv
        IKH = np.eye(model.n) - K @ H
        P = IKH @ Pk @ IKH.T + K @ R @ K.T     # Joseph form (numerically stable)
        return GateContext(zpred, H, R, Sinv, K, P)

    def gate(self, model, xk, Pk, z, t=None):
        """Score and apply one candidate measurement.

        Parameters
        ----------
        model : KalmanModel       the system (provides ``h``/``H``/``R``).
        xk, Pk                    the *predicted* state mean (n×1) and covariance (n×n).
        z                         one candidate measurement (m×1); NaN channels are skipped.
        t                         station/time tag forwarded to ``h``/``H``/``R``.

        Returns
        -------
        (chi2, xk, Pk) : the innovation χ² (float), and the Kalman-updated mean
        and covariance. ``chi2`` is ``+inf`` if every channel is missing.
        """
        z = np.asarray(z, dtype=float).reshape(-1, 1)
        keep = ~np.isnan(z).ravel()
        if not keep.any():
            return np.inf, xk, Pk

        zpred, H = self._measure(model, xk, t)
        zpred = np.asarray(zpred, dtype=float).reshape(-1, 1)
        H = np.asarray(H, dtype=float)
        R = np.asarray(model.R(t=t), dtype=float)

        ctx = self._context(model, Pk, zpred[keep], H[keep, :], R[np.ix_(keep, keep)])
        chi2, xk = ctx.score(xk, z[keep])
        return chi2, xk, ctx.P

    # ------------------------------------------------------------------
    #  Forward pass: seed → grow (branch/prune) → arbitrate
    # ------------------------------------------------------------------
    def forward(self, model, hits_by_layer, *, layer_radii=None, seeds=None, **kwargs):
        """Find tracks in one event.

        Parameters
        ----------
        hits_by_layer : sequence of (n_hits_l, m) arrays, innermost layer first.
        layer_radii   : (L,) radii of those layers; defaults to ``model.radii``.
        seeds         : explicit :class:`TrackCandidate` seeds, or None to build one
                        per hit on the innermost populated layer.

        Returns
        -------
        list[TrackCandidate] — the accepted, arbitrated tracks.
        """
        radii = np.asarray(layer_radii if layer_radii is not None else model.radii, dtype=float)
        hits = [np.asarray(h, dtype=float).reshape(-1, model.m) for h in hits_by_layer]
        L = len(hits)
        I = np.eye(model.n)

        cut = self.gate_threshold(model)
        penalty = self.hole_cost(model)
        seeds = self.seed(model, hits, radii) if seeds is None else list(seeds)

        if self._native_search(model, seeds):
            self.candidates = self._find_native(model, hits, radii, seeds)
            return self.arbitrate(self.candidates)

        seed_tracks = []
        for seed in seeds:
            cands = [seed]
            # Follow outward from the seed's own layer, whichever that is: seed() picks
            # the innermost *populated* layer (not necessarily layer 0), and callers may
            # pass seeds of their own. Starting at a fixed layer 1 would re-scan the
            # seeding layer, letting a track take a second hit at the same radius off a
            # zero-length propagation.
            for l in range(seed.seed_layer + 1, L):
                r = float(radii[l]); layer = hits[l]
                # NaN channels need the per-hit compression in gate(); without them the
                # whole layer shares one prepared gate (see GateContext).
                masked = layer.size and bool(np.isnan(layer).any())
                grown = []
                for c in cands:
                    x, P, F = self._predict(model, c.x, c.P, t=r)
                    if not np.all(np.isfinite(x)):
                        continue                                  # track left the detector
                    Faccum = F @ c.Faccum
                    ctx = None if masked else self.gate_context(model, x, P, t=r)
                    for j in range(len(layer)):
                        if ctx is None:
                            chi2, xk, Pk = self.gate(model, x, P, layer[j].reshape(model.m, 1), t=r)
                        else:
                            chi2, xk = ctx.score(x, layer[j]); Pk = ctx.P
                        if chi2 < cut:
                            nc = c.branch()
                            nc.x, nc.P = xk, Pk
                            nc.chi2 += chi2; nc.score += chi2
                            nc.hit_ids.append((l, j))
                            nc.radii.append(r)
                            nc.xk.append(xk); nc.Pk.append(Pk)
                            nc.zk.append(layer[j])
                            nc.priors.append((x, P)); nc.Fc.append(Faccum)
                            nc.Faccum = I
                            grown.append(nc)
                    # no-hit (hole) branch: the track survives a missed layer
                    if c.holes < self.max_holes:
                        nc = c.branch()
                        nc.x, nc.P = x, P
                        nc.holes += 1; nc.score += penalty
                        nc.Faccum = Faccum
                        grown.append(nc)
                if not grown:
                    break
                grown.sort(key=lambda c: c.score)
                cands = grown[: self.max_candidates]

            finished = [c for c in cands if c.n_hits >= self.min_hits]
            if finished:
                seed_tracks.append(min(finished, key=lambda c: c.quality))

        # Kept for inspection: the best candidate of every seed, before arbitration
        # decides which of them survive as tracks (CombinatorialKalmanFilter.candidates).
        self.candidates = seed_tracks
        return self.arbitrate(seed_tracks)

    # ------------------------------------------------------------------
    #  The search in C++ (Kalman::Engines::Combinatorial), on the Python
    #  model through callbacks. The Python loop above is the reference, and
    #  serves a subclass that changes the gate or the search.
    # ------------------------------------------------------------------
    def _native_search(self, model, seeds) -> bool:
        from ..models import native
        if type(self) is not CombinatorialKalmanEngine or not native.native_loop_enabled():
            return False
        # Seeds built by seed(): one assigned hit each, nothing accumulated yet.
        if any(len(s.hit_ids) != 1 or s.holes or s.chi2 or s.score for s in seeds):
            return False
        return bool(seeds) and native.callback_eligible(model)

    def _find_native(self, model, hits, radii, seeds):
        """The best candidate of every seed, in seed order, found by the C++ engine."""
        from ..models.native import callback_handle
        from ..native import core
        lib, ffi = core.lib, core.ffi
        n, m = model.n, model.m
        handle, keep = callback_handle(model)

        def ints(values):
            return ffi.new("int[]", [int(v) for v in values])

        flat = np.ascontiguousarray(np.concatenate([h.reshape(-1) for h in hits]) if hits else np.zeros(0))
        rr = np.ascontiguousarray(radii, dtype=np.float64)
        sx = np.ascontiguousarray([np.asarray(s.x, float).reshape(-1) for s in seeds])
        sP = np.ascontiguousarray([np.asarray(s.P, float).reshape(n, n) for s in seeds])
        nan = float("nan")
        # shared_hit_frac = 1 keeps every seed's best candidate: arbitration is
        # done here afterwards, so that `candidates` holds them all.
        found = lib.kalman_ckf_find(
            handle, len(hits), core.cdptr(rr), ints(len(h) for h in hits), core.cdptr(flat),
            len(seeds), ints(s.hit_ids[0][0] for s in seeds), ints(s.hit_ids[0][1] for s in seeds),
            core.cdptr(sx), core.cdptr(sP),
            nan if self.chi2_cut is None else self.chi2_cut,
            nan if self.hole_penalty is None else self.hole_penalty,
            self.max_candidates, self.max_holes, self.min_hits, 1.0)
        if found == ffi.NULL:
            if keep[-1]:
                raise keep[-1][0]                    # an exception raised by the Python model
            raise RuntimeError(f"track finding: {core.last_error()}")
        found = ffi.gc(found, lib.kalman_tracks_destroy)

        by_seed = {}
        chi2, score, holes = ffi.new("double *"), ffi.new("double *"), ffi.new("int *")
        for i in range(core.check(lib.kalman_tracks_count(found), "tracks_count")):
            H = core.check(lib.kalman_track_hits(found, i), "track_hits")
            x, P, Fa = np.empty(n), np.empty((n, n)), np.empty((n, n))
            core.check(lib.kalman_track_summary(found, i, chi2, score, holes,
                                                core.dptr(x), core.dptr(P), core.dptr(Fa)), "track_summary")
            layers, idx = ffi.new("int[]", H), ffi.new("int[]", H)
            rad = np.empty(H); xk = np.empty((H, n)); Pk = np.empty((H, n, n)); zk = np.empty((H, m))
            px = np.empty((H, n)); pP = np.empty((H, n, n)); Fc = np.empty((H, n, n))
            core.check(lib.kalman_track_copy(found, i, layers, idx, core.dptr(rad), core.dptr(xk), core.dptr(Pk),
                                             core.dptr(zk), core.dptr(px), core.dptr(pP), core.dptr(Fc)), "track_copy")
            track = TrackCandidate(
                x=x.reshape(n, 1), P=P, chi2=float(chi2[0]), score=float(score[0]), holes=int(holes[0]),
                hit_ids=[(int(layers[k]), int(idx[k])) for k in range(H)], radii=[float(r) for r in rad],
                xk=[xk[k].reshape(n, 1) for k in range(H)], Pk=[Pk[k] for k in range(H)],
                zk=[zk[k] for k in range(H)],
                priors=[(px[k].reshape(n, 1), pP[k]) for k in range(H)], Fc=[Fc[k] for k in range(H)],
                Faccum=Fa)
            by_seed[track.hit_ids[0]] = track
        return [by_seed[s.hit_ids[0]] for s in seeds if s.hit_ids[0] in by_seed]

    # ------------------------------------------------------------------
    #  Backward pass: RTS-smooth each accepted track
    # ------------------------------------------------------------------
    def backward(self, model, tracks, **kwargs):
        """RTS-smooth every track from :meth:`forward` → new, smoothed candidates."""
        smoothed = []
        for c in tracks:
            N = c.n_hits
            xs = [None] * N; Ps = [None] * N
            xs[-1], Ps[-1] = c.xk[-1], c.Pk[-1]
            for k in range(N - 2, -1, -1):
                xs[k], Ps[k] = self._smooth(
                    model, (c.xk[k], c.Pk[k]), c.priors[k + 1], (xs[k + 1], Ps[k + 1]), c.Fc[k + 1]
                )
            out = c.branch()
            out.xk, out.Pk = xs, Ps
            smoothed.append(out)
        return smoothed

    # ------------------------------------------------------------------
    #  Search helpers
    # ------------------------------------------------------------------
    def seed(self, model, hits, radii):
        """One seed per hit on the innermost non-empty layer."""
        seeds = []
        I = np.eye(model.n)
        for l in range(len(hits)):
            if len(hits[l]) == 0:
                continue
            r = float(radii[l])
            for j in range(len(hits[l])):
                hit = hits[l][j]
                x0, P0 = model.seed_state(hit)
                seeds.append(TrackCandidate(
                    x=x0, P=P0, hit_ids=[(l, j)], radii=[r],
                    xk=[x0], Pk=[P0], zk=[np.asarray(hit, float).reshape(-1)],
                    priors=[(x0, P0)], Fc=[I], Faccum=I,
                ))
            break                                   # seed only from the first populated layer
        return seeds

    def arbitrate(self, tracks):
        """Greedy hit competition: accept best tracks, reject those overlapping too much."""
        accepted, used = [], set()
        for c in sorted(tracks, key=lambda c: c.quality):
            ids = set(c.hit_ids)
            if len(ids & used) <= self.shared_hit_frac * len(ids):
                accepted.append(c)
                used |= ids
        return accepted
