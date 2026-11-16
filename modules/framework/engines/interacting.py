"""Interacting engine: the predict/update step of a model with several modes.

The Python face of ``Kalman::Engines::Interacting``
(``include/Kalman/Engines/Interacting.h``), the Interacting Multiple Model
filter of the literature. It runs a model made by ``route``
(:class:`~kalman.framework.models.route.Router`): one Gaussian belief per mode
and a probability on each; at every step the beliefs are mixed, each mode
predicts and updates with its own engine, the decision layer gives the new
probabilities, and the estimate is the modes' combined.

The whole pass runs in C++ (``Kalman::Filter``); nothing is computed here. After
a pass the engine holds what the C++ engine recorded: ``mode_probabilities``
and ``nis_history``, each ``(T, M)``.

Words of this field, kept here: *mode*, *mixing*, *mode probability*.
"""
from __future__ import annotations

import numpy as np

from ..constants import KalmanEngineKind
from ..engine import KalmanEngine


class InteractingKalmanEngine(KalmanEngine):

    NAME = "Interacting Engine"
    DESCRIPTION = "Steps several modes of one system, mixing their beliefs and routing between them (IMM)."
    VERSION = "1.0"
    KIND = KalmanEngineKind.INTERACTING

    def __init__(self):
        super().__init__()
        self.mode_probabilities = np.empty((0, 0))
        self.nis_history = np.empty((0, 0))

    # The leading mode's, which is what the router model answers with.
    def _propagate(self, model, x, u, t):
        return model.f(x, u, t=t), model.F(x, u, t)

    def _measure(self, model, x, t):
        return model.h(x, t=t), model.H(x, t)

    def _predict(self, model, x, P, u=None, t=None):
        raise NotImplementedError("the interacting engine runs in C++ (Kalman::Engines::Interacting): filter a model "
                                  "made by route() with the C++ library loaded.")

    _update = _predict

    def _after_native_pass(self, handle, modes):
        """Read what the C++ engine recorded during the pass."""
        from ..native import core
        steps = core.check(core.lib.kalman_engine_interacting_steps(handle), "interacting_steps")
        mu, nis = np.empty((steps, modes)), np.empty((steps, modes))
        if steps:
            core.check(core.lib.kalman_engine_interacting_history(handle, core.dptr(mu), core.dptr(nis)),
                       "interacting_history")
        self.mode_probabilities, self.nis_history = mu, nis
