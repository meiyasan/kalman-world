from enum import Enum, auto

class EnumBis(Enum):
    """Enhanced Enum base with class-safe introspection methods."""

    def __eq__(self, other):
        if isinstance(other, Enum): return self is other
        if isinstance(other, str) : return super().name.lower() == other.lower()
        if isinstance(other, int) : return self.value == other
        return False
    def __hash__(self): 
        return hash(self.name.lower())
    def __repr__(self):
        return f"{self.__class__.__name__}.{super().name}(index={int(self)}, value='{self.value}')"
    def __len__(self):
        self = type(self) if isinstance(self, Enum) else self
        return len(self.__members__)
    def __int__(self): 
        return list(self.__class__).index(self)
    def __str__(self): 
        return super().name.lower()
    def matches(self, *args):
        return any(self == arg for arg in args)
    
    @classmethod
    def name(cls) -> str:
        return cls.__name__
    @classmethod
    def repr(cls) -> str:
        return cls.__name__ + "<" + ", ".join([e.upper() for e in cls.keys()]) + ">"
    
    @classmethod
    def keys(cls):
        """Return member names for this specific Enum subclass."""
        return [k.lower() for k, v in cls.items()]
    @classmethod
    def values(cls):
        """Return member values for this Enum subclass."""
        return [v for k, v in cls.items()]
    @classmethod
    def items(cls):
        """Return (name, value) pairs for this Enum subclass."""
        return list(cls.__members__.items())

    @classmethod
    def index(cls, member=None):
        """Return index of a member or all indices as dict."""
        members = list(cls)
        if member is None:
            return {m: i for i, m in enumerate(members)}
        if not isinstance(member, cls):
            raise TypeError(f"{member!r} is not a member of {cls.__name__}")
        return members.index(member)

class KalmanState(EnumBis):
    pass

class KalmanStateFlavor(EnumBis):
    SAMPLED   = 0 # Sampled state
    PRIOR     = 1 # State before update step
    POSTERIOR = 2 # State after update step
    SMOOTHED  = 3 # State after smoothing step

# -----------------------------
# ----- Observation enums -----
# -----------------------------
class KalmanObservable(EnumBis):
    pass

class KalmanObservableFlavor(EnumBis):
    IDEAL     = 0 # Full ideal observation (no noise)
    TRUE      = 1 # True observation (with process noise)
    MEASURED  = 2 # Measured observation (with measurement noise)

class KalmanObservableReadout(Enum):
    INDEPENDENT = auto() # Each submodel provides an independent measurement.
    FUSED = auto() # All submodels contribute additively to the same measurement


# -----------------------------
# ----- Engine kinds ----------
# -----------------------------
class EngineKind(Enum):
    """
    Algorithm family of an estimation engine.

    Threads the shared term (Linear / Extended / Unscented / Particle) through every
    layer so cross-layer compatibility is both readable and checkable:

      - engines tag their ``KIND``;
      - models declare ``DEFAULT_ENGINE`` + ``ALLOWED_ENGINES`` (these enum values,
        so a model never imports an engine);
      - the ``KalmanFilter`` resolves the default and validates
        ``engine.KIND in model.ALLOWED_ENGINES``.
    """
    LINEAR        = auto()  # constant-matrix Kalman step (KF)
    EXTENDED      = auto()  # Jacobian linearisation (EKF)
    UNSCENTED     = auto()  # sigma-point transform (UKF)
    PARTICLE      = auto()  # sequential importance resampling (PF)
    COMBINATORIAL = auto()  # branch-on-ambiguous-hit track finding (CKF)


# All engine kinds — convenient default for ``ALLOWED_ENGINES``.
ALL_ENGINES = frozenset(EngineKind)
