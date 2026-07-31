"""
kalman.apps.space — astrodynamics domain.

- translunar     — :class:`TransLunarModel` (Earth-Moon cruise);
- transfers      — Lambert/Hohmann/porkchop/gravity-assist toolkit;
- interplanetary — :class:`InterplanetaryModel` + :class:`TrajectoryControlManeuver`.
"""
from .translunar import (TransLunarModel, TransLunarState, TransLunarObservable)
from .transfers import (hohmann_transfer, hohmann_sequence, lambert, porkchop, best_departure, gravity, gravity_assist, flyby_porkchop, plot_flyby_porkchop, plot_gravity_assist, plot_two_leg, chain_lambert, plot_multi_leg, lunar_porkchop)
from .interplanetary import (InterplanetaryModel, InterplanetaryState, InterplanetaryObservable, TrajectoryControlManeuver, plot_solar_system, plot_porkchop)

VARIANTS = {
    "translunar": TransLunarModel,
    "interplanetary": InterplanetaryModel,
}

__all__ = [
    "TransLunarModel",
    "TransLunarState",
    "TransLunarObservable",
    "hohmann_transfer",
    "hohmann_sequence",
    "lambert",
    "porkchop",
    "best_departure",
    "gravity",
    "gravity_assist",
    "flyby_porkchop",
    "plot_flyby_porkchop",
    "plot_gravity_assist",
    "plot_two_leg",
    "chain_lambert",
    "plot_multi_leg",
    "lunar_porkchop",
    "plot_porkchop",
    "InterplanetaryModel",
    "InterplanetaryState",
    "InterplanetaryObservable",
    "TrajectoryControlManeuver",
    "plot_solar_system",
    "VARIANTS",
]
