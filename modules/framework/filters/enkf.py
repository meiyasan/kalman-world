"""Ensemble Kalman Filter (EnKF) — wraps a model with the perturbed-obs ensemble engine."""
from __future__ import annotations

from ..filter import KalmanFilter
from ..constants import KalmanEngineKind


class EnsembleKalmanFilter(KalmanFilter):
    """
    Perturbed-observation EnKF. Forward-only (no RTS smoother)::

        enkf = EnsembleKalmanFilter(my_model, n_members=32, inflation=1.05)
        rec  = enkf.forward(zk)

    The workhorse for high-dimensional data assimilation (fields, grids): the
    covariance is carried by the ensemble, never formed densely during the
    analysis.  ``model.localization()`` (if defined) supplies the Gaspari–Cohn
    tapers; ``model.initial_ensemble(N, rng)`` (if defined) seeds the members
    from climatology instead of a Gaussian ``x0/P0`` draw.
    """

    NAME = "Ensemble Kalman Filter"
    DESCRIPTION = "EnKF: localized, inflated perturbed-observation ensemble analysis over a model."
    KIND = KalmanEngineKind.ENSEMBLE

    def __init__(self, model=None, *, n_members=50, inflation=1.0,
                 localization=None, seed=None, **kwargs):
        from ..engines.ensemble import EnsembleKalmanEngine
        super().__init__(model, engine=EnsembleKalmanEngine(
            n_members=n_members, inflation=inflation,
            localization=localization, seed=seed), **kwargs)

    def backward(self, **kwargs):
        raise NotImplementedError("EnsembleKalmanFilter has no RTS smoother; use forward() only.")

    def sample(self, duration, **kwargs):
        # Sample the *truth* with the model's default (Gaussian) engine, not the ensemble.
        from ..sampler import KalmanSampler
        return KalmanSampler(self.model).measurement(duration, **kwargs)
