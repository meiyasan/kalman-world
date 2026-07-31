"""Turbulence provider — Kraichnan/NSKT datasets as a truth trajectory.

A **weather-domain** provider.  Wraps ``apps.reanalysis.turbulence_truth`` /
``load_turbulence`` (guarded on ``h5py`` for HDF5 files); a doubly-periodic box
is geometrically ideal for the spectral flow model.
"""
from __future__ import annotations
import numpy as np

from ....framework.provider import KalmanProvider
from ....framework.constants import KalmanProviderKind


class TurbulenceProvider(KalmanProvider):

    NAME = "Turbulence Provider"
    DESCRIPTION = "Kraichnan/NSKT turbulence dataset → truth trajectory (spectral flow)."
    VERSION = "1.0"
    KIND = KalmanProviderKind.TURBULENCE

    def __init__(self, model=None, **kw):
        super().__init__(**kw)
        self.model = model

    def fetch(self, model=None, *, source=None, variable="vorticity", stride=1, **kw):
        """Return ``{"truth": (T, n)}`` from a turbulence dataset (path/array)
        block-coarsened onto ``model``'s grid."""
        model = model or self.model
        if model is None:
            raise ValueError("TurbulenceProvider needs a flow model (grid to coarsen onto).")
        if source is None:
            raise ValueError("provide source= (path/array of fields).")
        from ..geo import turbulence_truth
        truth = turbulence_truth(source, model, variable=variable, stride=stride)
        return {"truth": np.asarray(truth, dtype=float),
                "meta": {"provider": "turbulence", "variable": variable, "stride": stride}}
