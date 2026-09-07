"""Combinatorial Kalman Filter (CKF) — wraps a model with the track-finding engine."""
from __future__ import annotations

from ..filter import KalmanFilter
from ..constants import KalmanEngineKind


class CombinatorialKalmanFilter(KalmanFilter):
    """
    Combinatorial Kalman filter: track *finding* rather than track following.

    Where KF/EKF/UKF/PF estimate **one** trajectory from a time series of single
    measurements, the CKF reconstructs **several tracks** from an event whose every
    layer holds **many hits** (signal + background)::

        model  = DriftChamberModel(n_layers=8, Bz=0.05)
        event  = model.generate_event(n_tracks=3, n_background=4, seed=0)
        ckf    = CombinatorialKalmanFilter(model)
        tracks = ckf.forward(event.hits, layer_radii=event.radii)   # KalmanRecordCollection
        tracks = ckf.backward()                                     # RTS-smoothed

    The search — seeding, χ² gating, branching, beam pruning, hit arbitration and the
    smoother over the found tracks — is the algorithm, so it lives in
    :class:`~kalman.framework.engines.combinatorial.CombinatorialKalmanEngine`, which
    declares ``OWNS_PASS`` and drives the pass itself. This class only names that engine
    and hands it the search hyper-parameters, exactly as ``UnscentedKalmanFilter`` hands
    ``alpha/beta/kappa`` to its own engine.
    """

    NAME = "Combinatorial Kalman Filter"
    DESCRIPTION = "CKF: layer-by-layer track following with branching, pruning and hit arbitration."
    KIND = KalmanEngineKind.COMBINATORIAL

    def __init__(self, model=None, *, chi2_cut=None, max_candidates=6, max_holes=2,
                 min_hits=3, shared_hit_frac=0.3, hole_penalty=None, **kwargs):
        from ..engines.combinatorial import CombinatorialKalmanEngine
        super().__init__(model, engine=CombinatorialKalmanEngine(
            chi2_cut=chi2_cut, max_candidates=max_candidates, max_holes=max_holes,
            min_hits=min_hits, shared_hit_frac=shared_hit_frac,
            hole_penalty=hole_penalty), **kwargs)

    # --- the search results, under the names an event reconstruction expects ---
    @property
    def tracks(self):
        """The accepted tracks as a :class:`KalmanRecordCollection` (posterior, or
        smoothed once :meth:`backward` has run)."""
        return self.pass_records

    @property
    def candidates(self):
        """The best candidate of every seed *before* arbitration -- one per seed, so
        ``len(ckf.candidates)`` is the number of seeds that grew into a track at all.
        Arbitration then keeps the disjoint subset in :attr:`tracks_raw`."""
        return self.engine.candidates

    @property
    def tracks_raw(self):
        """The accepted tracks as raw ``TrackCandidate`` objects, which still carry the
        search bookkeeping the records drop: ``hit_ids``, ``chi2``, ``holes``."""
        return self.pass_result

    # --- the resolved search hyper-parameters (they default from the model) ---
    @property
    def chi2_cut(self):
        """The per-hit χ² gate in force — explicit if given, else 99% of a χ²(m)."""
        return self.engine.gate_threshold(self.model)

    @property
    def hole_penalty(self):
        """The score charged per missed layer — explicit if given, else one gate's worth."""
        return self.engine.hole_cost(self.model)

    @property
    def max_candidates(self):
        """Beam width kept per seed after each layer."""
        return self.engine.max_candidates

    @property
    def max_holes(self):
        """Missed layers allowed on a track."""
        return self.engine.max_holes

    @property
    def min_hits(self):
        """Minimum assigned hits for a track to be accepted."""
        return self.engine.min_hits

    @property
    def shared_hit_frac(self):
        """Max fraction of a track's hits shared with an already-accepted track."""
        return self.engine.shared_hit_frac
