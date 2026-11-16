"""
``compose``: several models side by side, each with its own state.

The Python face of ``Kalman::Models::Composite``
(``include/Kalman/Models/Composite.h``), which does the work. The parts evolve
independently, so the state is theirs one after the other and ``F`` and ``Q``
are block-diagonal. What is measured depends on the readout:

    STACKED:  z = [ h1(x1) ; h2(x2) ; ... ]     each part brings its sensors
    SUMMED:   z = h1(x1) + h2(x2) + ...         the parts share one measurement

Three ways to the same model::

    a.compose(b)
    compose(a, b, readout=KalmanReadout.SUMMED)
    Composite(a, b)

A parameter of a part is ``"<part>.<name>"``, the part going by its sensor's
name (or by ``names=``). Models of matrices side by side are a model of
matrices, on the linear engine. Nothing is computed here.

This is not ``add_sensor``, where a second sensor reads the *same* state.
"""
from __future__ import annotations

from ..constants import KalmanObservable, KalmanReadout, KalmanState
from ..model import KalmanModel
from ..native import core
from .operation import OperationModel


class Composite(OperationModel):
    """``models`` side by side."""

    NAME = "Composite Model"
    DESCRIPTION = "Several models side by side, each with its own state."
    VERSION = "2.0"

    def __init__(self, *models: KalmanModel, readout: KalmanReadout = KalmanReadout.STACKED, names=None, **kwargs):
        if len(models) == 1 and isinstance(models[0], (list, tuple)):
            models = tuple(models[0])
        if not models:
            raise ValueError("compose: give at least one model.")
        for model in models:
            if not isinstance(model, KalmanModel):
                raise TypeError(f"compose: a KalmanModel is expected, got {type(model).__name__}.")
        if not isinstance(readout, KalmanReadout):
            raise TypeError(f"compose: readout is a KalmanReadout, got {readout!r}.")
        lib, ffi = core.require(), core.ffi
        self.models, self.readout = list(models), readout
        if names is None:
            names, seen = [], {}
            for model in models:                                   # numbered from the second that shares a name
                base = str(getattr(model, "_sensor_name", None) or type(model).__name__)
                seen[base] = seen.get(base, 0) + 1
                names.append(base if seen[base] == 1 else f"{base}{seen[base]}")
        names = [str(name) for name in names]
        if len(names) != len(models) or len(set(names)) != len(names) or not all(names):
            raise ValueError(f"compose: every part needs a name of its own, got {names}.")
        self.part_names = names

        self._inner_cxxs = [self._inner(model) for model in models]      # borrowed by the C++ model
        encoded = [ffi.new("char[]", name.encode("utf-8", "replace")) for name in names]
        handle = lib.kalman_model_create_composite(ffi.new("kalman_model_t *[]", self._inner_cxxs), len(models),
                                                   int(readout is KalmanReadout.SUMMED),
                                                   ffi.new("const char *[]", encoded))
        if handle == ffi.NULL:
            raise self._failed("compose")
        self._inner_cxx = self._inner_cxxs[0]
        self._adopt(handle, list(models), **kwargs)

        # Parts that are matrices, one of them a function of time: the composite is
        # time-varying matrices, and the filter runs it over their tables, as it does
        # any model built from a function of time.
        from ..constants import KalmanEngineKind, KalmanStructureKind as S
        kinds = [model.structure for model in models]
        self._time_varying = S.MAPS not in kinds and S.TIME_VARYING in kinds
        if self._time_varying:
            self.DEFAULT_ENGINE = KalmanEngineKind.LINEAR
            self._analytic = False                       # read at a state: the parts answer H(t) themselves
            for stored, bound in (("_F", self.F), ("_H", self.H), ("_Q", self.Q), ("_R", self.R)):
                if any(callable(getattr(model, stored, None)) for model in models):
                    setattr(self, stored, bound)

    @property
    def structure(self):
        from ..constants import KalmanStructureKind as S
        if self.__dict__.get("_time_varying"):
            return S.TIME_VARYING
        return OperationModel.structure.fget(self)

    def native_handle(self):
        return None if self.__dict__.get("_time_varying") else super().native_handle()

    # ------------------------------------------------------------------
    def part(self, which) -> slice:
        """Where a part's state sits in this model's: ``record.xk[:, model.part("gps")]``.
        ``which`` is a part's name, its index, or the model itself."""
        if isinstance(which, KalmanModel):
            index = next((i for i, model in enumerate(self.models) if model is which), None)
        elif isinstance(which, str):
            index = self.part_names.index(which) if which in self.part_names else None
        else:
            index = int(which) if -len(self.models) <= int(which) < len(self.models) else None
        if index is None:
            raise KeyError(f"compose: no part {which!r}; the parts are {self.part_names}.")
        index %= len(self.models)
        start = sum(int(model.n) for model in self.models[:index])
        return slice(start, start + int(self.models[index].n))

    def initial_state(self, x0=None, P0=None, *, samples=None, rng=None):
        """The parts' initial states one after the other, or one given for the whole."""
        from .native import NativeModel
        if samples is not None:
            return NativeModel.initial_state(self, samples=samples, rng=rng)
        return NativeModel.initial_state(self, x0, P0)

    def _shared(self, which, base, suffix):
        """The enum the parts share (``record.plot(component="position", dim=2)`` then
        picks the third part's), or its nearest common base; when the parts name
        different things, the components as the C++ model names them."""
        enums = [getattr(model, which) for model in self.models]
        if all(isinstance(e, type) for e in enums):
            if all(e is enums[0] for e in enums):
                return enums[0]
            for cls in enums[0].mro():
                if cls not in (base, object) and issubclass(cls, base) and all(issubclass(e, cls) for e in enums) \
                        and len(cls.__members__):
                    return cls
        return self._label_enum(which == "states", base, suffix)

    @property
    def states(self) -> KalmanState:
        if getattr(self, "_states_enum", None) is None:
            self._states_enum = self._shared("states", KalmanState, "CompositeState")
        return self._states_enum

    @property
    def observables(self) -> KalmanObservable:
        if getattr(self, "_observables_enum", None) is None:
            self._observables_enum = self._shared("observables", KalmanObservable, "CompositeObservable")
        return self._observables_enum

    def limit_control(self, u):
        return u if not self.l else super(OperationModel, self).limit_control(u)

    def __repr__(self):
        return (f"Composite({', '.join(self.part_names)}; {self.readout.name.lower()}, "
                f"n={self.n}, m={self.m}, l={self.l}, fs={self.fs})")
