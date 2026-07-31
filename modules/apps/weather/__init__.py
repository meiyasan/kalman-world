"""
``kalman.apps.weather`` — the weather / geophysical-fluids domain package.

The reference **domain-package** layout: everything for one domain lives
together and the package ``__init__`` is its public interface.

- ``flow``       — the 2-D turbulence field model + OSSE harness
  (:class:`NS2D` spectral solver, :class:`KolmogorovFlowModel`,
  :class:`TwinExperiment`);
- ``storms``     — reduced storm models (:class:`TyphoonModel`,
  :class:`TornadoEnvironmentModel`) that sit on top of the field estimate;
- ``geo``        — Earth/geo utilities and dataset transforms (land–sea mask,
  global plotting, vorticity/regrid/coarsen, ERA5 + NSKT loaders);
- ``surrogate``  — the learned CNN forecast surrogate for the flow model;
- ``providers/`` — the domain's data sources (:class:`ERA5Provider`,
  :class:`TurbulenceProvider`).
"""
from .flow import NS2D, KolmogorovFlowModel, FlowObservable, TwinExperiment
from .storms import (TyphoonModel, TyphoonState, TyphoonObservable, synthetic_track,
                     synthetic_steered_track, analytic_steering, analytic_environment,
                     LearnedIntensityModel, japan_steering, field_steering,
                     japan_region, landfall_probability, JAPAN_BBOX,
                     TornadoEnvironmentModel, TornadoState, TornadoObservable,
                     synthetic_environment, brazil_season, brazil_tornado_scenario,
                     seasonal_tornado_probability)
from .geo import (earth_geometry, land_mask, plot_global_field, band_coordinates,
                  vorticity_from_uv, vorticity_cartesian, regrid, coarsen,
                  rescale_to_model, reanalysis_truth, turbulence_truth,
                  load_vorticity, load_turbulence, fetch_era5, EARTH_RADIUS)
from .providers import ERA5Provider, TurbulenceProvider

__all__ = [
    # flow / OSSE
    "NS2D", "KolmogorovFlowModel", "FlowObservable", "TwinExperiment",
    # storms
    "TyphoonModel", "TyphoonState", "TyphoonObservable", "synthetic_track",
    "synthetic_steered_track", "analytic_steering", "analytic_environment",
    "LearnedIntensityModel", "japan_steering", "field_steering", "japan_region",
    "landfall_probability", "JAPAN_BBOX", "TornadoEnvironmentModel", "TornadoState",
    "TornadoObservable", "synthetic_environment", "brazil_season",
    "brazil_tornado_scenario", "seasonal_tornado_probability",
    # geo / datasets
    "earth_geometry", "land_mask", "plot_global_field", "band_coordinates",
    "vorticity_from_uv", "vorticity_cartesian", "regrid", "coarsen",
    "rescale_to_model", "reanalysis_truth", "turbulence_truth",
    "load_vorticity", "load_turbulence", "fetch_era5", "EARTH_RADIUS",
    # providers
    "ERA5Provider", "TurbulenceProvider",
]
