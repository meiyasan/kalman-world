"""
Kalman framework — the Model / Engine / Filter contracts and their concrete pieces.

Layers (dependency points downward only: Filter → Engine → Model):
- ``model.py``   : ``KalmanModel``  — the system (WHAT): f/h/F/H/B/L/M/Q/R.
- ``engine.py``  : ``KalmanEngine`` — the algorithm (HOW): one predict/update/smooth step.
- ``filter.py``  : ``KalmanFilter`` — the workflow (WHEN): forward/backward + run state.
- ``models/``    : typed model bases (``LinearKalmanModel`` / ``...Extended`` / ``...Unscented`` / ``...Composite``).
- ``engines/``   : concrete engines (``LinearKalmanEngine`` / ``Extended`` / ``Unscented`` / ``Particle``).
- ``constants.py``: ``EngineKind`` (LINEAR/EXTENDED/UNSCENTED/PARTICLE) used to bind models↔engines.

Domain applications live in ``kalman.apps``; the recognisable filters (KF/EKF/UKF/PF) in
``kalman.filters``.
"""
from .constants import EngineKind
from .model import KalmanModel
from .engine import KalmanEngine
from .filter import KalmanFilter
from .models import (
    LinearKalmanModel,
    ExtendedKalmanModel,
    UnscentedKalmanModel,
    CompositeKalmanModel,
)
from .engines import (
    LinearKalmanEngine,
    ExtendedKalmanEngine,
    UnscentedKalmanEngine,
    ParticleEngine,
)

__all__ = [
    "EngineKind",
    "KalmanModel",
    "KalmanEngine",
    "KalmanFilter",
    "LinearKalmanModel",
    "ExtendedKalmanModel",
    "UnscentedKalmanModel",
    "CompositeKalmanModel",
    "LinearKalmanEngine",
    "ExtendedKalmanEngine",
    "UnscentedKalmanEngine",
    "ParticleEngine",
]
