"""
``kalman.providers`` — the data/knowledge-source layer (sibling of ``kalman.apps``).

Each provider tags its :class:`~kalman.framework.constants.KalmanProviderKind`;
``PROVIDER_BY_KIND`` + :func:`resolve_provider` mirror the engine/solver
registries.  Providers *feed* the estimator: a fetched dataset flows straight
into ``kalman.api.run_filter`` / ``run_osse``.
"""
from ..framework.constants import KalmanProviderKind

from ..framework.provider import KalmanProvider
from .synthetic import SyntheticProvider
from .file import FileProvider
from .sensor import SensorProvider
from .remote import RemoteProvider
# Domain-specific providers live with their domain (weather); registered here.
from ..apps.weather.providers import ERA5Provider, TurbulenceProvider

PROVIDER_BY_KIND = {
    KalmanProviderKind.SYNTHETIC:  SyntheticProvider,
    KalmanProviderKind.ERA5:       ERA5Provider,
    KalmanProviderKind.TURBULENCE: TurbulenceProvider,
    KalmanProviderKind.FILE:       FileProvider,
    KalmanProviderKind.SENSOR:     SensorProvider,
    KalmanProviderKind.REMOTE:     RemoteProvider,
}


def resolve_provider(kind, **kwargs):
    """Instantiate the provider for ``kind`` (name-fallback for autoreload)."""
    if isinstance(kind, str):
        try:
            kind = KalmanProviderKind[kind.upper()]
        except KeyError:
            raise ValueError(f"unknown provider kind {kind!r}.")
    cls = PROVIDER_BY_KIND.get(kind)
    if cls is None and kind is not None:
        name = getattr(kind, "name", None)
        cls = next((c for k, c in PROVIDER_BY_KIND.items() if getattr(k, "name", None) == name), None)
    if cls is None:
        raise ValueError(f"unknown provider kind {kind!r}.")
    return cls(**kwargs)


__all__ = [
    "KalmanProvider", "SyntheticProvider", "FileProvider", "SensorProvider",
    "ERA5Provider", "TurbulenceProvider", "RemoteProvider",
    "PROVIDER_BY_KIND", "resolve_provider",
]
