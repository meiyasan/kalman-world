"""ERA5 provider — reanalysis fields as a truth trajectory for a flow model.

A **weather-domain** provider (lives under ``apps/weather``, beside the models it
feeds).  Wraps the ERA5 loaders (``apps.weather.geo``); guarded on ``cdsapi``/
``xarray``.  The outbound CDS credential is handled by ``cdsapi`` itself
(``~/.cdsapirc``); this provider only turns fetched/loaded fields into a
``truth`` trajectory the ensemble estimator consumes.
"""
from __future__ import annotations
import numpy as np

from ....framework.provider import KalmanProvider
from ....framework.constants import KalmanProviderKind


class ERA5Provider(KalmanProvider):

    NAME = "ERA5 Provider"
    DESCRIPTION = "ERA5 reanalysis fields → truth trajectory for a KolmogorovFlowModel."
    VERSION = "1.0"
    KIND = KalmanProviderKind.ERA5

    def __init__(self, model=None, **kw):
        super().__init__(**kw)
        self.model = model

    def fetch(self, model=None, *, path=None, dates=None, area=None, level=500,
              stride=1, download_to="era5.nc", **kw):
        """Return ``{"truth": (T, n)}`` regridded onto ``model``'s mesh.

        Provide a local ``path`` (NetCDF/xarray) or ``dates``+``area`` to
        download first (needs CDS credentials in ``~/.cdsapirc``)."""
        model = model or self.model
        if model is None:
            raise ValueError("ERA5Provider needs a flow model (mesh to regrid onto).")
        from ..geo import fetch_era5, reanalysis_truth
        src = path
        if src is None:
            if dates is None or area is None:
                raise ValueError("provide path=, or dates=+area= to download ERA5.")
            src = fetch_era5(download_to, dates=dates, area=area, level=level)
        truth = reanalysis_truth(src, model, level=level, stride=stride)
        return {"truth": np.asarray(truth, dtype=float),
                "meta": {"provider": "era5", "level": level, "stride": stride}}
