"""BayesianRouter — classical IKF Bayesian mode-probability update."""
from __future__ import annotations
import numpy as np
from .base import KalmanRouter


class BayesianRouter(KalmanRouter):
    """Classical IKF update: μ_j ∝ c_j · L_j.

    Combines the Markov transition prior ``c`` with the Gaussian innovation
    likelihood ``L_j = N(ν_j; 0, S_j)``.  Theoretically optimal when all
    models are correctly specified Gaussians.

    No training required — ``fit()`` is a no-op.

    Prefer over NeuralRouter when
    - noise is truly Gaussian and models are correctly specified;
    - training data is scarce;
    - transitions are genuinely Markov (no learned pattern adds value);
    - interpretability matters (every μ update follows Bayes' theorem).

    Decided by ``Kalman::Routers::Bayesian`` when libKalman is installed;
    :meth:`_reference_route` is the Python reference.
    """

    NATIVE = "bayesian"

    def route(self, nis_arr, c, likelihoods):
        mu = self._native_route(nis_arr, c, likelihoods)
        return mu if mu is not None else self._reference_route(nis_arr, c, likelihoods)

    def _reference_route(self, nis_arr, c, likelihoods):
        return self._normalise(c * likelihoods)

    def _native_rows(self):
        return 0                                      # Bayes reads no NIS

    def _native_create(self):
        from ..native import core
        return core.create_router("bayesian"), (), []
