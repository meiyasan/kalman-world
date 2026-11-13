"""The toy model written in Python: the same physics as examples/cxx/toy_plugin.cc.

A cup of coffee cools toward the temperature of the room (Newton's law of
cooling), and nobody knows how fast: the cooling rate is part of the state.

    state        x = [T, k]      temperature [degC], cooling rate [1/s]
    dynamics     T' = -k (T - T_room),   k' = 0
    measurement  z = T           a thermometer

This file and ``toy_plugin.cc`` are the two ways to write a model for the
framework. They state the same thing line for line:

    =====================  ==========================  ===========================
    what                   C++ (toy_plugin.cc)         Python (this file)
    =====================  ==========================  ===========================
    the class              Kalman::Models::Extended    ExtendedKalmanModel
    one step               Vector f(x, u, t)           def f(self, x, u, w, t)
    the measurement        Vector h(x, t)              def h(self, x, v, t)
    a setting              DeclareParameter("room")    a constructor argument
    noise, initial state   SetQ, SetR, SetInitial...   Q=, R=, x0=, P0=
    found by name          KALMAN_REGISTER_MODEL       the class itself
    =====================  ==========================  ===========================

A model written in C++ runs wholly in C++. A model written in Python is run by
the same C++ filter, which calls back into ``f`` and ``h`` below at each step:
slower, and nothing to compile. ``toy_sensor.py`` runs both on the same
readings and prints how far apart they are.
"""
from __future__ import annotations
import os
import sys

import numpy as np

try:
    import kalman
except ImportError:                                # running from a checkout that is not installed
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    import modules as kalman

_framework = kalman.__name__ + ".framework"
ExtendedKalmanModel = __import__(_framework + ".models.extended", fromlist=["x"]).ExtendedKalmanModel
_constants = __import__(_framework + ".constants", fromlist=["x"])
KalmanState, KalmanObservable = _constants.KalmanState, _constants.KalmanObservable
KalmanDifferentiatorKind = _constants.KalmanDifferentiatorKind


class CoffeeState(KalmanState):
    TEMPERATURE = "degC"
    COOLING_RATE = "1/s"


class CoffeeObservable(KalmanObservable):
    TEMPERATURE = "degC"


class CoolingCoffeeModel(ExtendedKalmanModel):
    """Newton's law of cooling with an unknown rate; a thermometer reads the temperature."""

    NAME = "Cooling Coffee (toy)"
    DESCRIPTION = "Newton's law of cooling with an unknown rate; a thermometer reads the temperature."
    # The C++ class takes its Jacobians by central differences (the default of
    # Kalman::Models::Extended); the same here, so the two linearise alike.
    DIFFERENTIATOR = KalmanDifferentiatorKind.CENTRAL

    @property
    def states(self):
        return CoffeeState

    @property
    def observables(self):
        return CoffeeObservable

    def __init__(self, sensor_id="thermometer", room=21.0, fs=1.0, **kwargs):
        self.room = float(room)                    # degC
        self.n, self.m, self.l = 2, 1, 0
        # The model is nearly exact: a little slack on T, and a very slow
        # random walk on k (the cup's cooling rate barely changes).
        kwargs.setdefault("Q", np.diag([1e-4, 1e-12]))
        kwargs.setdefault("R", np.array([[0.5 ** 2]]))          # thermometer noise: 0.5 degC
        # What the filter believes before the first reading: hot coffee, and a
        # cooling rate that is only a guess.
        kwargs.setdefault("x0", np.array([[80.0], [1e-3]]))
        kwargs.setdefault("P0", np.diag([25.0, 1e-6]))
        super().__init__(sensor_id, fs=fs, L=np.eye(2), **kwargs)

    def f(self, x, u=None, w=None, t=None):
        """The exact solution over one step, so no integrator is needed."""
        T, k = np.asarray(x, dtype=float).reshape(-1)
        out = np.array([[self.room + (T - self.room) * np.exp(-k / self.fs)], [k]])
        return out if w is None else out + np.asarray(w, dtype=float).reshape(2, 1)

    def h(self, x, v=None, t=None):
        z = np.asarray(x, dtype=float).reshape(-1)[:1].reshape(1, 1)
        return z if v is None else z + np.asarray(v, dtype=float).reshape(1, 1)
