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

def available_engines() -> list:
    """Every resolvable engine name."""
    return sorted(k.name.lower() for k in ENGINE_BY_KIND)


def resolve_engine(kind=None) -> KalmanEngine:
    """Instantiate the concrete engine for ``kind`` (``None`` -> Extended, the
    safe universal default).

    Accepts a :class:`KalmanEngineKind`, its name, an engine class, or a ready
    instance.  An unrecognised kind raises :class:`ValueError` rather than
    quietly degrading to Extended — a typo that silently runs a different
    algorithm is worse than a failure.

    Robust to ``KalmanEngineKind`` *identity* changes caused by ``%autoreload`` or
    duplicate imports: if the member is not found by identity, fall back to matching
    by its ``name`` so a reloaded ``KalmanEngineKind.LINEAR`` still resolves to the linear
    engine instead of silently degrading to Extended.
    """
    if isinstance(kind, KalmanEngine):
        return kind
    if isinstance(kind, type) and issubclass(kind, KalmanEngine):
        return kind()
    if kind is None:
        return ExtendedKalmanEngine()

    if isinstance(kind, str):
        key = kind.strip().lower()
        cls = next((c for k, c in ENGINE_BY_KIND.items() if k.name.lower() == key), None)
    else:
        cls = ENGINE_BY_KIND.get(kind)
        if cls is None:                              # %autoreload identity fallback
            name = getattr(kind, "name", None)
            cls = next((c for k, c in ENGINE_BY_KIND.items()
                        if getattr(k, "name", None) == name), None)
    if cls is None:
        raise ValueError(f"unknown engine kind {kind!r}; choose from {available_engines()}.")
    return cls()

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
    "available_engines",
    "resolve_engine",
]
