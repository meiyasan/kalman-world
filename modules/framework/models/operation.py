"""
The base of the model operations: a model built from other models.

``joint``, ``observe``, ``lag``, ``compose`` and ``route`` each take models and
give back a model. The work is done by a C++ class (``Kalman::Models::Joint``
...) that wraps the inner models' C++ handles; the Python class is a
:class:`~.native.NativeModel` around it, so ``f``, ``h``, the Jacobians and the
noise are the C++ model's and a filtering pass runs in ``Kalman::Filter``.

An inner model reaches C++ through :func:`~.native.handle_of`: as it is when it
is C++ already, as its twin, as its matrices, or through callbacks to its
Python methods.
"""
from __future__ import annotations

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

    def _adopt(self, handle, inners, sensor_id, **kwargs):
        self.inners = list(inners)
        self.dtype = self.inners[0].dtype
        NativeModel.__init__(self, sensor_id=sensor_id, _handle=handle, **kwargs)

    def _refresh(self):
        for refresh in self.__dict__.get("_refreshers", []):
            refresh()

    def native_handle(self):
        self._refresh()
        return super().native_handle()

    @property
    def observables(self):
        return self.inners[0].observables

    def clip_u(self, u):
        return self.inners[0].clip_u(u)
