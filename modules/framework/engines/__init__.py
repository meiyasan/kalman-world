"""
Concrete Kalman engines (the *how* — one estimation step per algorithm).

Each engine tags its :class:`~kalman.framework.constants.EngineKind`. The
``ENGINE_BY_KIND`` registry lets a model/filter resolve its declared
``DEFAULT_ENGINE`` to a concrete class without the model importing any engine.
"""
from ..engine import KalmanEngine
from ..constants import EngineKind

from .linear import LinearKalmanEngine
from .extended import ExtendedKalmanEngine
from .unscented import UnscentedKalmanEngine
from .particle import ParticleEngine
from .combinatorial import CombinatorialKalmanEngine

ENGINE_BY_KIND = {
    EngineKind.LINEAR:        LinearKalmanEngine,
    EngineKind.EXTENDED:      ExtendedKalmanEngine,
    EngineKind.UNSCENTED:     UnscentedKalmanEngine,
    EngineKind.PARTICLE:      ParticleEngine,
    EngineKind.COMBINATORIAL: CombinatorialKalmanEngine,
}

def resolve_engine(kind: EngineKind) -> KalmanEngine:
    """Instantiate the concrete engine for ``kind`` (defaults if unknown -> Extended).

    Robust to ``EngineKind`` *identity* changes caused by ``%autoreload`` or
    duplicate imports: if the member is not found by identity, fall back to matching
    by its ``name`` so a reloaded ``EngineKind.LINEAR`` still resolves to the linear
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
    "ENGINE_BY_KIND",
    "resolve_engine",
]
