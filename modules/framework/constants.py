from enum import Enum, auto

class EnumBis(Enum):
    def __int__(self): return self.value
    def __str__(self): return self.name.lower().capitalize()

    def __eq__(self, other):
        if isinstance(other, Enum):
            return self is other
        if isinstance(other, str):
            return self.name.lower() == other.lower()
        if isinstance(other, int):
            return self.value == other
        return False

    def __hash__(self):
        return hash(self.name.lower())

    def matches(self, *args):
        return any(self == arg for arg in args)

# -----------------------------
# ----- State enums -----------
# -----------------------------

class KalmanState(EnumBis):
    pass

class KalmanStateUnits(EnumBis):
    pass

class KalmanStateFlavor(EnumBis):
    SAMPLED   = 0 # Sampled state
    PRIOR     = 1 # State before update step
    POSTERIOR = 2 # State after update step
    SMOOTHED  = 3 # State after smoothing step

# -----------------------------
# ----- Observation enums -----
# -----------------------------
class KalmanObservation(EnumBis):
    pass

class KalmanObservationUnits(EnumBis):
    pass

class KalmanObservationFlavor(EnumBis):
    IDEAL     = 0 # Full ideal observation (no noise)
    TRUE      = 1 # True observation (with process noise)
    MEASURED  = 2 # Measured observation (with measurement noise)

class KalmanObservationMode(Enum):
    """
    Defines how submodel observations are combined into a global measurement.

    - INDEPENDENT: each submodel contributes its own observation (block-diagonal H)
    - FUSED:       observations from all submodels are summed / averaged (shared z)
    - SHARED:      all submodels observe the same global state (stacked H)
    """

    INDEPENDENT = auto()
    """Each submodel provides an independent measurement.
    z = [h₁(x₁); h₂(x₂); …],  H = block_diag(H₁, H₂, …)
    """

    FUSED = auto()
    """All submodels contribute additively to the same measurement.
    z = Σ hᵢ(xᵢ),  H = [H₁ H₂ H₃ …] (horizontal)
    """

    SHARED = auto()
    """All submodels observe the same global state.
    z = [h₁(x); h₂(x); …],  H = [H₁; H₂; H₃; …] (vertical)
    """