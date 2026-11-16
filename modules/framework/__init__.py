"""
Kalman framework — the Model / Engine / Filter contracts and their concrete pieces.

Layers (dependency points downward only: Filter → Engine → Model):
- ``model.py``   : ``KalmanModel``  — the system (WHAT): f/h/F/H/B/L/M/Q/R.
- ``engine.py``  : ``KalmanEngine`` — the algorithm (HOW): one predict/update/smooth step.
- ``filter.py``  : ``BayesFilter`` — the workflow (WHEN): forward/backward + run state; ``KalmanFilter`` is its Gaussian family, ``ParticleFilter`` stands beside it.
- ``models/``    : typed model bases (``LinearKalmanModel`` / ``...Extended`` / ``...Unscented`` / ``...Composite`` / ``Joint`` / ``Fusion``).
- ``engines/``   : concrete engines (``LinearKalmanEngine`` / ``Extended`` / ``Unscented`` / ``Particle``).
- ``filters/``   : the recognisable filter names (``LinearKalmanFilter`` / ``Extended`` / ``Unscented`` / ``Particle`` / ``Ensemble`` / ``Combinatorial`` / ``Interactive``).
- ``constants.py``: ``KalmanEngineKind`` (LINEAR/EXTENDED/UNSCENTED/PARTICLE) used to bind models↔engines.

Domain applications live in ``kalman.apps``.
"""
from .constants import (KalmanEngineKind, KalmanIntegratorKind, KalmanSolverKind, KalmanDynamicsKind, KalmanDifferentiatorKind,
                        KalmanProviderKind)
from .model import KalmanModel
from .engine import KalmanEngine, KalmanEstimator
from .integrator import KalmanIntegrator
from .integrators import (EulerIntegrator, RK4Integrator, DormandPrinceIntegrator,
                          resolve_integrator, register_integrator, unregister_integrator,
                          available_integrators)
from .solver import KalmanSolver
from .solvers import (SpectralSolver, FiniteDifferenceSolver, resolve_solver,
                      register_solver, unregister_solver, available_solvers)
from .provider import KalmanProvider
from .filter import BayesFilter, KalmanFilter
from .control import KalmanController
from .tuner import KalmanTuner
from .tuners import BoostKalmanTuner, AdaptiveQRKalmanTuner, NeuralAdaptiveQRKalmanTuner
from .operations import joint, add_sensor, add_instrument, compose
from .sensor import KalmanSensor
from .instrument import KalmanInstrument
from .engine_models import KalmanEnsembleModel, KalmanDetectorModel
from .models import (
    LinearKalmanModel,
    ExtendedKalmanModel,
    UnscentedKalmanModel,
    CompositeKalmanModel,
    Joint,
    Fusion,
    NativeModel,
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
    MPCKalmanController,
    MPPIKalmanController,
    NeuralKalmanController,
)
from .routers import (
    KalmanRouter,
    BayesianRouter,
    ThresholdRouter,
    NeuralRouter,
    HybridRouter,
)
from .filters import (
    LinearKalmanFilter,
    ExtendedKalmanFilter,
    UnscentedKalmanFilter,
    ParticleFilter,
    CombinatorialKalmanFilter,
    InteractiveKalmanFilter,
    EnsembleKalmanFilter,
)

__all__ = [
    "KalmanEngineKind",
    "KalmanIntegratorKind",
    "KalmanSolverKind",
    "KalmanDynamicsKind",
    "KalmanDifferentiatorKind",
    "KalmanProviderKind",
    "KalmanModel",
    "KalmanEngine",
    "KalmanEstimator",
    "KalmanIntegrator",
    "EulerIntegrator",
    "RK4Integrator",
    "DormandPrinceIntegrator",
    "resolve_integrator",
    "register_integrator",
    "unregister_integrator",
    "available_integrators",
    "KalmanSolver",
    "SpectralSolver",
    "FiniteDifferenceSolver",
    "resolve_solver",
    "register_solver",
    "unregister_solver",
    "available_solvers",
    "KalmanProvider",
    "BayesFilter",
    "KalmanFilter",
    "KalmanController", "KalmanTuner",
    "LinearKalmanModel",
    "ExtendedKalmanModel",
    "UnscentedKalmanModel",
    "CompositeKalmanModel", "Joint", "joint", "Fusion", "add_sensor", "KalmanSensor", "add_instrument", "compose", "KalmanInstrument", "KalmanEnsembleModel", "KalmanDetectorModel", "NativeModel",
    "LinearKalmanEngine",
    "ExtendedKalmanEngine",
    "UnscentedKalmanEngine",
    "ParticleEngine",
    "ProportionalKalmanController",
    "PIDKalmanController",
    "LQRKalmanController",
    "MPCKalmanController", "MPPIKalmanController",
    "NeuralKalmanController",
    "BoostKalmanTuner",
    "AdaptiveQRKalmanTuner",
    "NeuralAdaptiveQRKalmanTuner",
    "KalmanRouter",
    "BayesianRouter",
    "ThresholdRouter",
    "NeuralRouter",
    "HybridRouter",
    "LinearKalmanFilter",
    "ExtendedKalmanFilter",
    "UnscentedKalmanFilter",
    "ParticleFilter",
    "CombinatorialKalmanFilter",
    "InteractiveKalmanFilter",
    "EnsembleKalmanFilter",
]
