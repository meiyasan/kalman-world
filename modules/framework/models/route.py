"""
``route``: several models of one system, with a probability on each.

The Python face of ``Kalman::Models::Router`` (``include/Kalman/Models/Router.h``).
A system that changes regime (a car on a dry road, then on ice; a storm that
runs straight, then recurves) is described by one model per regime, the
*modes*. They share the state and the measurement; a Markov matrix says how
likely the system is to go from one to another in a step, and the model
carries a probability on each.

Three ways to the same model::

    route([dry, icy], Pi)
    KalmanModel.route([dry, icy], Pi)
    Router([dry, icy], Pi, initial_probs, router=ThresholdRouter())

    record = modes.filter(zk)            # the combined estimate
    modes.mode_probabilities             # (T, M): which regime, when

It is run by the interacting engine (``Kalman::Engines::Interacting``, the
Interacting Multiple Model filter of the literature), its default engine, by
the ordinary filter: there is no filter class of its own. Who decides the
probabilities at each step is the ``router=``, a decision layer from
``kalman.framework.routers`` (Bayesian by default; a rule, a network, a
blend), or any function ``(nis, c, likelihoods) -> mu``.

Seen as one model it answers ``f``, ``h`` and its matrices with those of its
leading mode, the most probable one. Nothing is computed here.
"""
from __future__ import annotations

import numpy as np

from ..model import KalmanModel
from ..native import core
from .operation import OperationModel


class Router(OperationModel):
    """``models`` as the modes of one system.

    Parameters
    ----------
    models : the model of each regime; all with the same ``n``, ``m`` and ``fs``.
    transition : ``(M, M)`` Markov matrix; row ``i`` is where mode ``i`` goes, and sums to 1.
    initial_probs : the initial probability of each mode; uniform by default.
    router : the decision layer, a ``KalmanRouter`` or a function
        ``(nis, c, likelihoods) -> mu``; ``BayesianRouter()`` by default.
    names : what the modes are called (for ``"<mode>.<parameter>"``); their
        sensor names by default.
    """

    NAME = "Router Model"
    DESCRIPTION = "Several models of one system, with a probability on each."
    VERSION = "1.0"

    def __init__(self, models, transition, initial_probs=None, *, router=None, names=None, **kwargs):
        models = list(models)
        if len(models) < 2:
            raise ValueError("route: at least two models are required.")
        for model in models:
            if not isinstance(model, KalmanModel):
                raise TypeError(f"route: a KalmanModel is expected, got {type(model).__name__}.")
        lib, ffi = core.require(), core.ffi
        M = len(models)
        self.models = models
        self.Pi = np.ascontiguousarray(np.asarray(transition, dtype=np.float64))
        if self.Pi.shape != (M, M):
            raise ValueError(f"route: the transition matrix has shape {self.Pi.shape}, expected ({M}, {M}).")
        mu0 = None if initial_probs is None else np.ascontiguousarray(np.asarray(initial_probs, dtype=np.float64).reshape(-1))
        if mu0 is not None and mu0.size != M:
            raise ValueError(f"route: {mu0.size} initial probabilities for {M} models.")
        if names is None:
            names, seen = [], {}
            for model in models:
                base = str(getattr(model, "_sensor_name", None) or type(model).__name__)
                seen[base] = seen.get(base, 0) + 1
                names.append(base if seen[base] == 1 else f"{base}{seen[base]}")
        self.mode_names = [str(name) for name in names]
        from ..routers.bayesian import BayesianRouter
        self.router = router if router is not None else BayesianRouter()

        self._inner_cxxs = [self._inner(model) for model in models]      # borrowed by the C++ model
        encoded = [ffi.new("char[]", name.encode("utf-8", "replace")) for name in self.mode_names]
        handle = lib.kalman_model_create_router(ffi.new("kalman_model_t *[]", self._inner_cxxs), M, core.cdptr(self.Pi),
                                                ffi.NULL if mu0 is None else core.cdptr(mu0),
                                                ffi.new("const char *[]", encoded))
        if handle == ffi.NULL:
            raise self._failed("route")
        self._inner_cxx = self._inner_cxxs[0]
        self._decision = None
        self._adopt(handle, models, **kwargs)

    # ------------------------------------------------------------------
    # The decision layer goes to the C++ model before a pass
    # ------------------------------------------------------------------
    def native_handle(self):
        handle = super().native_handle()
        from ..routers.hybrid import HybridRouter
        decision = HybridRouter._child(self.router)          # its own C++ router, or a callback to it
        self._decision = decision                            # kept alive with the model
        core.check(core.lib.kalman_model_router_set_decision(handle, decision[0]), "router_set_decision")
        return handle

    def filter(self, z, u=None, **kwargs):
        """The combined estimate over ``z``; every pass starts again from the modes'
        initial states. ``mode_probabilities`` then says which regime, when."""
        kwargs.setdefault("reset", True)
        if kwargs.get("smooth"):
            raise ValueError("route: a model with several modes has no smoother (the interacting engine keeps one "
                             "belief per mode); filter without smooth=True.")
        try:
            return super().filter(z, u, **kwargs)
        except RuntimeError as shadow:
            errors = self._decision[2] if self._decision is not None else []
            if errors:                                       # the router's own exception, not C++'s report of it
                raise errors[0] from shadow
            raise

    def initial_state(self, x0=None, P0=None, *, samples=None, rng=None):
        from .native import NativeModel
        if samples is not None:
            return NativeModel.initial_state(self, samples=samples, rng=rng)
        return NativeModel.initial_state(self, x0, P0)

    # ------------------------------------------------------------------
    # What the last pass found
    # ------------------------------------------------------------------
    def _engine_of_last_pass(self):
        flt = self.__dict__.get("_filter")
        return getattr(flt, "engine", None) if flt is not None else None

    @property
    def mode_probabilities(self) -> np.ndarray:
        """The probability of each mode at each step of the last pass, ``(T, M)``."""
        engine = self._engine_of_last_pass()
        found = getattr(engine, "mode_probabilities", None)
        return found if found is not None and found.size else np.empty((0, len(self.models)))

    @property
    def nis_history(self) -> np.ndarray:
        """The NIS of each mode at each step of the last pass, ``(T, M)``: what the
        decision layers read."""
        engine = self._engine_of_last_pass()
        found = getattr(engine, "nis_history", None)
        return found if found is not None and found.size else np.empty((0, len(self.models)))

    def fit(self, training_zk_list, true_regime_list, epochs: int = 500, lr: float = 2e-3, seed=None, **kwargs):
        """Train the decision layer on labelled data: the NIS histories are collected
        with the Bayesian decision, then given to ``router.fit`` with the labels
        (the true mode at each step). Nothing to do for a rule that does not learn."""
        from ..routers.bayesian import BayesianRouter
        saved, self.router = self.router, BayesianRouter()
        try:
            nis_arrays, label_arrays = [], []
            for zk, labels in zip(training_zk_list, true_regime_list):
                self.filter(zk)
                nis_arrays.append(np.array(self.nis_history))
                label_arrays.append(np.asarray(labels, dtype=int))
        finally:
            self.router = saved
        if hasattr(self.router, "fit"):
            self.router.fit(nis_arrays, label_arrays, epochs=epochs, lr=lr, seed=seed, **kwargs)
        return self

    # ------------------------------------------------------------------
    # What the modes share beyond the contract (their figures, a storm's
    # forecast cone) is asked of the first mode: every mode means the same
    # thing by the state.
    # ------------------------------------------------------------------
    def __getattr__(self, name):
        models = self.__dict__.get("models")
        if not models or name.startswith("__"):
            raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")
        return getattr(models[0], name)

    def _plot_static(self, record, *args, **kwargs):
        return self.models[0]._plot_static(record, *args, **kwargs)

    def _plot_interactive(self, record, *args, **kwargs):
        return self.models[0]._plot_interactive(record, *args, **kwargs)

    @property
    def states(self):
        return self.models[0].states

    @property
    def observables(self):
        return self.models[0].observables

    def __repr__(self):
        return f"Router({', '.join(self.mode_names)}; n={self.n}, m={self.m}, l={self.l}, fs={self.fs})"
