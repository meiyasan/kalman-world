"""
``kalman.apps.powergrids`` — mains power-grid harmonics domain.

- ``harmonics`` — :class:`PowerGridStateModel` (lab-frame joint phasor
  estimator specialised to 50/60 Hz mains) and the phasor diagram plot.
"""
from .harmonics import (PowerGridStateModel,
                   PowerGridObservable,
                   plot_phasor_diagram)

VARIANTS = {
    "phasor": PowerGridStateModel,
}

__all__ = [
    "PowerGridStateModel",
    "PowerGridObservable",
    "plot_phasor_diagram",
    "VARIANTS",
]
