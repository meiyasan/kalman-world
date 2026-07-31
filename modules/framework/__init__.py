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
from .constants import (KalmanEngineKind, KalmanIntegratorKind, KalmanSolverKind,
                        KalmanProviderKind)
from .model import KalmanModel
from .engine import KalmanEngine, KalmanEstimator
from .integrator import KalmanIntegrator
from .integrators import EulerIntegrator, RK4Integrator, DormandPrinceIntegrator, resolve_integrator
from .solver import KalmanSolver
from .solvers import SpectralSolver, resolve_solver
from .provider import KalmanProvider
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
    "KalmanIntegratorKind",
    "KalmanSolverKind",
    "KalmanProviderKind",
    "KalmanModel",
    "KalmanEngine",
    "KalmanEstimator",
    "KalmanIntegrator",
    "EulerIntegrator",
    "RK4Integrator",
    "DormandPrinceIntegrator",
    "resolve_integrator",
    "KalmanSolver",
    "SpectralSolver",
    "resolve_solver",
    "KalmanProvider",
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
