"""Particle Filter (PF) — wraps a model with the sequential-importance-resampling engine."""
from __future__ import annotations

from ..framework.filter import KalmanFilter
from ..framework.constants import KalmanEngineKind


class ParticleFilter(KalmanFilter):
    """
    Bootstrap / SIR particle filter. Forward-only (no RTS smoother)::

        pf  = ParticleFilter(my_model, n_particles=1000)
        rec = pf.forward(zk)
    """

    NAME = "Particle Filter"
    DESCRIPTION = "Particle filter (PF): sequential importance resampling over a model."
    KIND = KalmanEngineKind.PARTICLE

    def __init__(self, model=None, *, n_particles=500, **kwargs):
        from ..framework.engines.particle import ParticleEngine
        super().__init__(model, engine=ParticleEngine(n_particles=n_particles), **kwargs)

    def backward(self, **kwargs):
        raise NotImplementedError("ParticleFilter has no RTS smoother; use forward() only.")

    def sample(self, duration, **kwargs):
        # Sample the *truth* with the model's default (Gaussian) engine, not the PF cloud.
        from ..framework.sampler import KalmanSampler
        return KalmanSampler(self.model).measurement(duration, **kwargs)
