"""
The tuner: it retunes the filter's own noise model as it runs.

A controller acts on the system (it computes the command ``u``). A tuner acts
on the *filter*: once per step, with the innovation in hand, it adjusts the
model's ``Q`` or ``R``. They are different roles, and a filter takes one of
each::

    filter.forward(zk, uk=lqr, tuner=adaptive)     # a command from lqr, the noise retuned
    filter.forward(zk, tuner=adaptive)             # no command, the noise retuned

A tuner is called at the same point of the step as a controller and shares its
machinery (the belief it is handed, ``reset``, the logs), which is why
:class:`KalmanTuner` derives from :class:`~.control.KalmanController`. Its
command is fixed: zero. What a concrete tuner writes is ``_update``. The C++
role is ``Kalman::Tuner``; the concretes live in ``framework/tuners/``.
"""
from __future__ import annotations

import numpy as np

from .control import KalmanController


class KalmanTuner(KalmanController):
    """Base of the tuners: commands nothing, retunes the model in ``_update``."""

    def _policy(self, x_hat, P, t, *, nu=None):
        u = self._native_policy(x_hat, P, t, nu)
        return u if u is not None else self._reference_policy(x_hat, P, t, nu=nu)

    def _reference_policy(self, x_hat, P, t, *, nu=None):
        return np.zeros((max(self.model.l, 1), 1))


def with_tuner(controller, tuner):
    """One callable for a filter's closed loop: ``controller`` decides the command
    (``None`` for no command) and ``tuner`` retunes the model at the same point of
    the step."""
    if controller is None:
        return tuner

    def both(x_hat, P, t, nu=None):
        u = controller(x_hat, P, t, nu=nu)
        tuner(x_hat, P, t, nu=nu)
        return u
    return both
