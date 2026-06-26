"""HybridRouter — geometric blend of two arbitrary routers."""
from __future__ import annotations
import numpy as np
from .base import KalmanRouter


class HybridRouter(KalmanRouter):
    """Geometric blend (log-linear opinion pool) of two routers.

        μ = router_a(...)^(1−α) · router_b(...)^α  / Z

    At α = 0 → pure router_a.  At α = 1 → pure router_b.

    Common uses::

        # Bayesian anchor + learned refinement
        HybridRouter(BayesianRouter(), NeuralRouter(MLP()), alpha=0.3)

        # Learned primary + rule-based guard
        HybridRouter(NeuralRouter(MLP()), ThresholdRouter(soft=True), alpha=0.5)

    ``fit()`` is forwarded to both children.

    Parameters
    ----------
    router_a, router_b : KalmanRouter
    alpha : float in [0, 1]
        Blend weight toward router_b.
    """

    def __init__(self,
                 router_a: KalmanRouter,
                 router_b: KalmanRouter,
                 alpha: float = 0.5):
        self.router_a = router_a
        self.router_b = router_b
        self.alpha    = float(alpha)

    def route(self, nis_arr, c, likelihoods):
        mu_a = np.maximum(np.asarray(self.router_a(nis_arr, c, likelihoods), float), 1e-12)
        mu_b = np.maximum(np.asarray(self.router_b(nis_arr, c, likelihoods), float), 1e-12)
        return self._normalise(mu_a ** (1 - self.alpha) * mu_b ** self.alpha)

    def fit(self, nis_arrays, labels_arrays, **kwargs) -> "HybridRouter":
        self.router_a.fit(nis_arrays, labels_arrays, **kwargs)
        self.router_b.fit(nis_arrays, labels_arrays, **kwargs)
        return self
