"""BayesianRouter — classical IMM Bayesian mode-probability update."""
from __future__ import annotations
import numpy as np
from .base import KalmanRouter


class BayesianRouter(KalmanRouter):
    """Classical IMM update: μ_j ∝ c_j · L_j.

    Combines the Markov transition prior ``c`` with the Gaussian innovation
    likelihood ``L_j = N(ν_j; 0, S_j)``.  Theoretically optimal when all
    models are correctly specified Gaussians.

    No training required — ``fit()`` is a no-op.

    Prefer over NeuralRouter when
    - noise is truly Gaussian and models are correctly specified;
    - training data is scarce;
    - transitions are genuinely Markov (no learned pattern adds value);
    - interpretability matters (every μ update follows Bayes' theorem).
    """

    def route(self, nis_arr, c, likelihoods):
        return self._normalise(c * likelihoods)
