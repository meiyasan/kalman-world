"""
Concrete Kalman engines (the *how* — one estimation step per algorithm).

Each engine tags its :class:`~kalman.framework.constants.KalmanEngineKind`. The
``ENGINE_BY_KIND`` registry lets a model/filter resolve its declared
``DEFAULT_ENGINE`` to a concrete class without the model importing any engine.
"""
from ..engine import KalmanEngine
from ..constants import KalmanEngineKind

from .linear import LinearKalmanEngine
from .extended import ExtendedKalmanEngine
from .unscented import UnscentedKalmanEngine
from .particle import ParticleEngine
from .combinatorial import CombinatorialKalmanEngine
from .ensemble import EnsembleKalmanEngine, gaspari_cohn

ENGINE_BY_KIND = {
    KalmanEngineKind.LINEAR:        LinearKalmanEngine,
    KalmanEngineKind.EXTENDED:      ExtendedKalmanEngine,
    KalmanEngineKind.UNSCENTED:     UnscentedKalmanEngine,
    KalmanEngineKind.PARTICLE:      ParticleEngine,
    KalmanEngineKind.COMBINATORIAL: CombinatorialKalmanEngine,
    KalmanEngineKind.ENSEMBLE:      EnsembleKalmanEngine,
}

def resolve_engine(kind: KalmanEngineKind) -> KalmanEngine:
    """Instantiate the concrete engine for ``kind`` (defaults if unknown -> Extended).

    Robust to ``KalmanEngineKind`` *identity* changes caused by ``%autoreload`` or
    duplicate imports: if the member is not found by identity, fall back to matching
    by its ``name`` so a reloaded ``KalmanEngineKind.LINEAR`` still resolves to the linear
    engine instead of silently degrading to Extended.
    """
    cls = ENGINE_BY_KIND.get(kind)
    if cls is None and kind is not None:
        name = getattr(kind, "name", None)
        cls = next((c for k, c in ENGINE_BY_KIND.items() if getattr(k, "name", None) == name), None)
    return (cls or ExtendedKalmanEngine)()

__all__ = [
    "KalmanEngine",
    "LinearKalmanEngine",
    "ExtendedKalmanEngine",
    "UnscentedKalmanEngine",
    "ParticleEngine",
    "CombinatorialKalmanEngine",
    "EnsembleKalmanEngine",
    "gaspari_cohn",
    "ENGINE_BY_KIND",
    "resolve_engine",
]
