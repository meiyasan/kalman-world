"""Shared ballistics domain definitions — state/observable enums, noise presets,
and the estimate record types used by every ballistics model variant."""
from collections import namedtuple

from ...framework.constants import KalmanObservable, KalmanState, EnumBis

# Results of the trajectory estimators below; the *_std fields are 1-σ uncertainties
# (None when no state covariance is supplied). Times are relative to the state's epoch.
ApexEstimate   = namedtuple("ApexEstimate",   ["time", "height", "time_std", "height_std"])
ImpactEstimate = namedtuple("ImpactEstimate", ["time", "speed", "time_std"])

# ===== State enums/units =====
class BallisticState(KalmanState):
    POSITION     = "m"
    SPEED        = "m/s"
    ACCELERATION = "m/s²"

class BallisticObservable(KalmanObservable):
    POSITION = "m"

# Process-noise type — the member VALUE is the physical unit of
# ``process_uncertainty``, which is exactly what distinguishes the two: feeding a
# value in m/s² means "acceleration noise" (constant-acceleration model), m/s³
# means "jerk noise" (nearly-constant-acceleration model). Picking the unit picks
# the model. Compares equal to its name too (e.g. ``noise == "acceleration"``).
class BallisticNoise(EnumBis):
    ACCELERATION = "m/s²"   # white-noise-acceleration → accel stays flat (constant-accel)
    JERK         = "m/s³"   # white-noise-jerk        → accel random-walks (nearly-constant-accel)

    @classmethod
    def of(cls, noise):
        """Resolve a member from itself, its name ('acceleration'/'jerk'), or its
        UNIT ('m/s²'/'m/s³') — so the unit of process_uncertainty selects the model."""
        if isinstance(noise, cls):
            return noise
        for m in cls:
            if m == noise or m.value == noise:   # match by name (EnumBis.__eq__) or by unit
                return m
        raise ValueError(f"Unknown noise {noise!r}; use {cls.repr()} or a unit in {cls.values()}.")

