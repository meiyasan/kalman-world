"""
Kalman framework — the Model / Engine / Filter contracts and their concrete pieces.

Layers (dependency points downward only: Filter → Engine → Model):
- ``model.py``   : ``KalmanModel``  — the system (WHAT): f/h/F/H/B/L/M/Q/R.
- ``engine.py``  : ``KalmanEngine`` — the algorithm (HOW): one predict/update/smooth step.
- ``filter.py``  : ``KalmanFilter`` — the workflow (WHEN): forward/backward + run state.
- ``models/``    : typed model bases (``LinearKalmanModel`` / ``...Extended`` / ``...Unscented`` / ``...Composite``).
- ``engines/``   : concrete engines (``LinearKalmanEngine`` / ``Extended`` / ``Unscented`` / ``Particle``).
- ``constants.py``: ``KalmanEngineKind`` (LINEAR/EXTENDED/UNSCENTED/PARTICLE) used to bind models↔engines.

Domain applications live in ``kalman.apps``; the recognisable filters (KF/EKF/UKF/PF) in
``kalman.filters``.
"""
from .constants import KalmanEngineKind
from .model import KalmanModel
from .engine import KalmanEngine
from .filter import KalmanFilter
from .control import KalmanController
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
from .controllers import (
    ProportionalKalmanController,
    PIDKalmanController,
    LQRKalmanController,
    MPPIKalmanController,
    NeuralKalmanController,
    BoostKalmanController,
    AdaptiveQRKalmanController,
    NeuralAdaptiveQRKalmanController,
)
from .routers import (
    KalmanRouter,
    BayesianRouter,
    ThresholdRouter,
    NeuralRouter,
    HybridRouter,
)

__all__ = [
    "KalmanEngineKind",
    "KalmanModel",
    "KalmanEngine",
    "KalmanFilter",
    "KalmanController",
    "LinearKalmanModel",
    "ExtendedKalmanModel",
    "UnscentedKalmanModel",
    "CompositeKalmanModel",
    "LinearKalmanEngine",
    "ExtendedKalmanEngine",
    "UnscentedKalmanEngine",
    "ParticleEngine",
    "ProportionalKalmanController",
    "PIDKalmanController",
    "LQRKalmanController",
    "MPPIKalmanController",
    "NeuralKalmanController",
    "BoostKalmanController",
    "AdaptiveQRKalmanController",
    "NeuralAdaptiveQRKalmanController",
    "KalmanRouter",
    "BayesianRouter",
    "ThresholdRouter",
    "NeuralRouter",
    "HybridRouter",
]
