"""
The base of the model operations: a model built from other models.

``joint``, ``observe``, ``lag``, ``compose`` and ``route`` each take models and
give back a model. The work is done by a C++ class (``Kalman::Models::Joint``
...) that wraps the inner models' C++ handles; the Python class is a
:class:`~.native.NativeModel` around it, so ``f``, ``h``, the Jacobians and the
noise are the C++ model's and a filtering pass runs in ``Kalman::Filter``.

An inner model reaches C++ through :func:`~.native.handle_of`: as it is when it
is C++ already, as its counterpart, as its matrices, or through callbacks to its
Python methods.
"""
from __future__ import annotations

import numpy as np

from . import native
from .native import NativeModel


class OperationModel(NativeModel):
    """A model made from ``inners`` by one operation. A subclass builds the C++
    handle from the inner handles and calls :meth:`_adopt`."""

    def _inner(self, model, parameters=()):
        """The C++ handle of an inner model; kept alive and refreshed with this one."""
        handle, keep, refresh = native.handle_of(model, parameters)
        self.__dict__.setdefault("_keeps", []).append(keep)
        self.__dict__.setdefault("_refreshers", []).append(refresh)
        return handle

    def _failed(self, what):
        """The error of a C++ constructor that returned NULL: the Python exception
        raised inside a callback when there is one, the C++ message otherwise."""
        from ..native import core
        for keep in self.__dict__.get("_keeps", []):
            errors = native.callback_errors(keep)
            if errors:
                return errors[0]
        return ValueError(f"{what}: {core.last_error()}")

    def _adopt(self, handle, inners, **kwargs):
        self.inners = list(inners)
        self.dtype = self.inners[0].dtype
        try:
            NativeModel.__init__(self, sensor=self._own_sensor_name(inners[0]), _handle=handle, **kwargs)
        except RuntimeError as shadow:
            # A Python function called back from C++ raised: give its exception, not
            # the C++ report of it.
            error = self._failed("operation")
            if isinstance(error, ValueError) and str(error).startswith("operation:"):
                raise
            raise error from shadow

    @staticmethod
    def _own_sensor_name(inner):
        """The name of the sensor the inner model was built with: this model's too."""
        return getattr(inner, "_sensor_name", None) or type(inner).__name__

    def _refresh(self):
        for refresh in self.__dict__.get("_refreshers", []):
            refresh()

    def native_handle(self):
        self._refresh()
        return super().native_handle()

    def initial_state(self, x0=None, P0=None, *, samples=None, rng=None):
        """The initial state of this model, or one its inner model takes: the latter
        is handed to the inner model, and this model derives its own from it again
        in C++ (``Model::AdoptInitialState``), by the rule it was built with."""
        if samples is not None:
            return super().initial_state(samples=samples, rng=rng)
        if getattr(self, "x0", None) is None or "inners" not in self.__dict__:
            return super().initial_state(x0, P0)            # during construction
        if x0 is None and P0 is None:
            return super().initial_state()
        from ..native import core
        inner, n = self.inners[0], int(self.n)
        size = lambda a: None if a is None else int(np.size(a))
        own = n != int(inner.n) and size(x0) in (None, n) and size(P0) in (None, n * n)
        if own:
            return super().initial_state(x0, P0)
        full_P0 = P0 if (n != int(inner.n) and size(P0) == n * n) else None
        inner.initial_state(x0, None if full_P0 is not None else P0)
        native._push_initial_state(inner, self._inner_cxx)
        core.check(core.lib.kalman_model_adopt_initial_state(self._handle), "adopt_initial_state")
        x = self._vec(core.lib.kalman_model_x0, "x0").reshape(-1, 1)
        P = self._mat(core.lib.kalman_model_P0, (n, n), "P0")
        return super().initial_state(x, full_P0 if full_P0 is not None else P)

    def _label_enum(self, state: bool, base, suffix):
        """The labels the C++ model carries, as an enum (indices as values)."""
        labels = [name for name, _ in native.labels_of(self._handle, state)]
        return base(f"{type(self.inners[0]).__name__}{suffix}", {label: i for i, label in enumerate(labels)})

    @property
    def observables(self):
        return self.inners[0].observables

    def limit_control(self, u):
        return self.inners[0].limit_control(u)
