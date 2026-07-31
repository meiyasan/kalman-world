"""ERA5 / reanalysis truth-replay chain for the Kolmogorov-flow DA testbed.

Turns a real atmospheric field sequence into a ``truth_trajectory`` for
:class:`~kalman.apps.fluids.TwinExperiment` — the imperfect-model experiment
where the "nature run" is the actual atmosphere and the forecast model is the
toy solver (or a learned surrogate).  The chain::

    fetch_era5(...)            # cdsapi download (optional; needs CDS credentials)
      └→ load_vorticity(...)   # xarray: read u/v (or vo) at one pressure level
           └→ reanalysis_truth(...)   # vorticity → regrid → rescale → (T, n)
                └→ TwinExperiment(model, truth_trajectory=...)

A parallel chain covers **Kraichnan-type turbulence datasets** (doubly-
periodic Cartesian boxes, e.g. SuperBench NSKT at Re = 16000 / 32000,
2048×2048, channels u/v/vorticity)::

    load_turbulence(...)          # h5py / numpy: (T, C, H, W) or (T, H, W)
      └→ turbulence_truth(...)    # ζ (spectral curl if u/v) → block-coarsen → rescale → (T, n)

Only :func:`fetch_era5` and :func:`load_vorticity` need the optional data
stack (``pip install xarray netCDF4 cdsapi``); the numerical core —
:func:`vorticity_from_uv`, :func:`regrid`, :func:`rescale_to_model`,
:func:`reanalysis_truth` on in-memory arrays — is plain numpy/scipy and works
without it.

Two physical caveats the docstrings repeat because they decide whether the
experiment is meaningful:

- **Periodicity** — the solver's domain is a torus.  Choose a full zonal band
  (0–360° longitude) so the east–west seam is physically periodic; a regional
  lat–lon box is not, and the mismatch shows up as spurious boundary
  increments.
- **Nondimensionalization** — the model knows nothing about SI vorticity.
  :func:`rescale_to_model` matches the sequence's standard deviation to the
  model's climatology so that ``sig_o``, inflation and localization tunings
  keep their meaning; report results as *ratios* (``TwinExperiment.skill``),
  which are invariant to this scaling.
"""
from __future__ import annotations
import os
import numpy as np

EARTH_RADIUS = 6.371e6   # [m]
_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
_MASK_CACHE = {}


# ======================================================================
# Earth representation (land / sea) — no cartopy needed
# ======================================================================
def earth_geometry(kind="land"):
    """Natural Earth 110m geometry (public domain), shipped in-package.

    ``kind="land"`` returns polygon rings, ``kind="coastline"`` returns line
    strings — each as a list of ``(N, 2)`` arrays of ``(lon, lat)`` degrees.
    """
    import json
    with open(os.path.join(_DATA_DIR, f"ne_110m_{kind}.geojson")) as f:
        feats = json.load(f)["features"]
    parts = []
    for ft in feats:
        geom, t = ft["geometry"], ft["geometry"]["type"]
        cs = geom["coordinates"]
        if t == "Polygon":
            rings = cs
        elif t == "MultiPolygon":
            rings = [r for poly in cs for r in poly]
        elif t == "LineString":
            rings = [cs]
        else:                                   # MultiLineString
            rings = cs
        parts += [np.asarray(r, dtype=float) for r in rings]
    return parts


def land_mask(lat, lon):
    """Boolean ``(nlat, nlon)`` land mask (True = ground, False = sea).

    Rasterizes the Natural Earth land polygons with matplotlib's point-in-
    polygon test — seam-proof at any central longitude (longitudes are wrapped
    into [−180, 180) before testing) and cached per grid.
    """
    from matplotlib.path import Path as MPath
    lat = np.asarray(lat, dtype=float); lon = np.asarray(lon, dtype=float)
    key = (lat.tobytes(), lon.tobytes())
    if key in _MASK_CACHE:
        return _MASK_CACHE[key]
    verts, codes = [], []
    for ring in earth_geometry("land"):
        verts.append(ring)
        codes += [MPath.MOVETO] + [MPath.LINETO] * (len(ring) - 1)
    path = MPath(np.vstack(verts), np.asarray(codes, dtype=np.uint8))
    LO, LA = np.meshgrid(((lon + 180.0) % 360.0) - 180.0, lat)
    mask = path.contains_points(np.column_stack([LO.ravel(), LA.ravel()]))
    mask = mask.reshape(len(lat), len(lon))
    _MASK_CACHE[key] = mask
    return mask


def _wrapped_coastlines(central_longitude=0.0):
    """Coastline segments re-centered on ``central_longitude``, with NaN breaks
    inserted at the map seam so no segment is drawn across the wrap."""
    out = []
    for seg in earth_geometry("coastline"):
        x = ((seg[:, 0] - central_longitude + 180.0) % 360.0) + central_longitude - 180.0
        y = seg[:, 1].copy()
        jump = np.abs(np.diff(x)) > 180.0
        if jump.any():
            x = np.insert(x, np.where(jump)[0] + 1, np.nan)
            y = np.insert(y, np.where(jump)[0] + 1, np.nan)
        out.append(np.column_stack([x, y]))
    return out


def plot_global_field(field=None, lat=None, lon=None, *, ax=None,
                      central_longitude=0.0, cmap="coolwarm",
                      coastlines=True, land="#c9bc8f", ocean="#a6c8e0",
                      band=None, probes=None, colorbar=True, title=None,
                      figsize=(12, 6), **imkw):
    """Equirectangular Earth plot — the SuperBench-style climate snapshot.

    - ``field=None`` → plain planet: sea (``ocean``) and ground (``land``)
      from the rasterized Natural Earth mask, coastlines on top.
    - ``field (nlat, nlon)`` with ``lat``/``lon`` degrees → the field drawn
      over the whole globe (``pcolormesh``), coastlines overlaid so the
      continents read through the data.
    - ``central_longitude`` re-centers the map (180 → Pacific-centered like
      the SuperBench snapshot); ``band=(lat_min, lat_max)`` outlines the
      assimilation band; ``probes=(lats, lons)`` marks observation sites.

    Returns ``(fig, ax)``.
    """
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    lon0 = float(central_longitude)

    if field is None:
        lat = np.linspace(-90, 90, 181) if lat is None else np.asarray(lat, dtype=float)
        lon = (np.arange(lon0 - 180.0, lon0 + 180.0, 1.0) if lon is None
               else np.asarray(lon, dtype=float))
        mask = land_mask(lat, lon)
        from matplotlib.colors import ListedColormap
        ax.pcolormesh(lon, lat, mask, cmap=ListedColormap([ocean, land]),
                      vmin=0, vmax=1, shading="auto", rasterized=True)
    else:
        field = np.asarray(field, dtype=float)
        lat = np.asarray(lat, dtype=float); lon = np.asarray(lon, dtype=float)
        x = ((lon - lon0 + 180.0) % 360.0) + lon0 - 180.0
        order = np.argsort(x)
        pm = ax.pcolormesh(x[order], lat, field[:, order], cmap=cmap,
                           shading="auto", rasterized=True, **imkw)
        if colorbar:
            fig.colorbar(pm, ax=ax, shrink=0.8, pad=0.02)

    if coastlines:
        for seg in _wrapped_coastlines(lon0):
            ax.plot(seg[:, 0], seg[:, 1], color="black", lw=0.5, alpha=0.6)

    if band is not None:
        for b in band:
            ax.axhline(b, color="crimson", lw=1.6, ls="--")
    if probes is not None:
        plats, plons = probes
        px = ((np.asarray(plons, dtype=float) - lon0 + 180.0) % 360.0) + lon0 - 180.0
        ax.plot(px, plats, "k.", ms=3)

    ax.set_xlim(lon0 - 180.0, lon0 + 180.0)
    ax.set_ylim(float(np.min(lat)), float(np.max(lat)))
    ax.set_xlabel("longitude [°]"); ax.set_ylabel("latitude [°]")
    if title:
        ax.set_title(title)
    ax.set_aspect("equal")
    return fig, ax


def band_coordinates(model, lat_range=(10.0, 50.0)):
    """Geo-reference a :class:`KolmogorovFlowModel` grid onto a zonal band.

    Maps the model's ``grid × grid`` mesh onto latitudes ``lat_range`` and the
    full 0–360° of longitude (the periodic axis), so model fields and probes
    can be drawn on the Earth with :func:`plot_global_field`.  Returns a dict
    with ``lat``/``lon`` (grid axes) and ``probe_lat``/``probe_lon``.
    """
    g = model.grid
    lat = np.linspace(lat_range[0], lat_range[1], g)
    lon = np.arange(g) * (360.0 / g)
    iy, ix = np.divmod(model.obs_idx, g)
    return {"lat": lat, "lon": lon,
            "probe_lat": lat[iy], "probe_lon": lon[ix]}


# ======================================================================
# Numerical core (no optional dependencies)
# ======================================================================
def vorticity_from_uv(u, v, lat, lon):
    """Relative vorticity ζ = ∇×(u,v)·k̂ on a lat–lon grid (spherical metric).

    .. code-block:: text

        ζ = 1/(a cosφ) · [ ∂v/∂λ − ∂(u cosφ)/∂φ ]

    ``u, v`` are ``(..., nlat, nlon)`` in m/s; ``lat``/``lon`` in degrees
    (either ordering; ERA5's descending latitude is handled by the metric).
    Returns ζ with the same shape [1/s].
    """
    u = np.asarray(u, dtype=float); v = np.asarray(v, dtype=float)
    phi = np.deg2rad(np.asarray(lat, dtype=float))
    lam = np.deg2rad(np.asarray(lon, dtype=float))
    cosphi = np.clip(np.cos(phi), 0.05, None)          # keep the metric sane near poles
    dv_dlam = np.gradient(v, lam, axis=-1)
    ducos_dphi = np.gradient(u * cosphi[:, None], phi, axis=-2)
    return (dv_dlam - ducos_dphi) / (EARTH_RADIUS * cosphi[:, None])


def regrid(fields, grid, *, order=1):
    """Resample ``(T, H, W)`` (or ``(H, W)``) fields onto the model's
    ``grid × grid`` mesh — bilinear by default, periodic wrap in longitude
    (last axis).  Returns ``(T, grid, grid)``."""
    from scipy.ndimage import zoom
    fields = np.asarray(fields, dtype=float)
    single = fields.ndim == 2
    if single:
        fields = fields[None]
    T, H, W = fields.shape
    out = np.empty((T, grid, grid))
    for k in range(T):
        out[k] = zoom(fields[k], (grid / H, grid / W), order=order,
                      mode="grid-wrap", grid_mode=True)
    return out[0] if single else out


def rescale_to_model(traj, model):
    """Nondimensionalize a field sequence to the model's climatology scale.

    Removes the sequence mean and scales so the standard deviation matches
    ``model.climatology().std()``.  Returns ``(scaled, scale)`` with
    ``scaled = (traj − mean) · scale``; keep ``scale`` to convert analysis
    fields back to physical units (divide by it and add the mean back).
    """
    traj = np.asarray(traj, dtype=float)
    mean = traj.mean()
    std = traj.std()
    if std == 0:
        raise ValueError("constant field sequence cannot be rescaled")
    scale = float(model.climatology().std() / std)
    return (traj - mean) * scale, scale


def reanalysis_truth(source, model, *, level=None, stride=1, rescale=True):
    """One call from data to a ``truth_trajectory`` ready for ``TwinExperiment``.

    ``source`` may be:

    - a path / ``xarray.Dataset`` — routed through :func:`load_vorticity`
      (needs xarray);
    - a tuple ``(u, v, lat, lon)`` — vorticity computed in memory;
    - a ``(T, H, W)`` array of vorticity fields — used directly.

    Applies ``stride`` (keep every k-th time step so one step ≈ one
    assimilation cycle of physical decorrelation), regrids to the model mesh
    and (by default) rescales to model units.  Returns ``(T, n)``.
    """
    if isinstance(source, tuple) and len(source) == 4:
        fields = vorticity_from_uv(*source)
    elif isinstance(source, np.ndarray) and source.ndim == 3:
        fields = source
    else:
        fields, lat, lon = load_vorticity(source, level=level)
    fields = np.asarray(fields, dtype=float)[::int(stride)]
    fields = regrid(fields, model.grid)
    if rescale:
        fields, _ = rescale_to_model(fields, model)
    return fields.reshape(fields.shape[0], -1)


# ======================================================================
# Kraichnan-type turbulence datasets (doubly-periodic Cartesian boxes)
# ======================================================================
def vorticity_cartesian(u, v, length=2 * np.pi):
    """Relative vorticity ζ = ∂v/∂x − ∂u/∂y on a uniform doubly-periodic grid,
    computed spectrally (exact for periodic data — matches the solver's own
    conventions).  ``u, v`` are ``(..., H, W)``; ``length`` is the physical
    box size the wavenumbers are scaled by.  Returns ζ with the same shape."""
    u = np.asarray(u, dtype=float); v = np.asarray(v, dtype=float)
    H, W = u.shape[-2:]
    ky = 2 * np.pi * np.fft.fftfreq(H, d=length / H)
    kx = 2 * np.pi * np.fft.fftfreq(W, d=length / W)
    zh = 1j * kx[None, :] * np.fft.fft2(v) - 1j * ky[:, None] * np.fft.fft2(u)
    return np.real(np.fft.ifft2(zh))


def coarsen(fields, grid):
    """Block-average ``(T, H, W)`` (or ``(H, W)``) fields down to ``grid × grid``.

    The top-hat filter is the physically sensible coarse-graining for
    turbulence truth: spectral content above the model's resolution is
    averaged out instead of aliased in (2048 → 64 is an exact 32× block).
    Falls back to :func:`regrid` when the sizes are not integer multiples.
    """
    fields = np.asarray(fields, dtype=float)
    single = fields.ndim == 2
    if single:
        fields = fields[None]
    T, H, W = fields.shape
    if H % grid or W % grid:
        out = regrid(fields, grid)
    else:
        out = fields.reshape(T, grid, H // grid, grid, W // grid).mean(axis=(2, 4))
    return out[0] if single else out


def load_turbulence(source, *, variable="vorticity", channels=("u", "v", "vorticity")):
    """Read a field sequence from a Kraichnan-type turbulence dataset.

    Built for the SuperBench NSKT layout (Navier–Stokes Kraichnan Turbulence,
    Re = 16000 / 32000, 2048×2048, channels u, v, vorticity) but format-
    tolerant: ``.h5/.hdf5`` via h5py (a dataset named ``variable`` if present,
    else the first dataset), ``.npy/.npz`` via numpy, or an in-memory array.
    A ``(T, C, H, W)`` array is unpacked with ``channels``; a ``(T, H, W)``
    array is taken as the variable directly.  Returns ``(T, H, W)`` — or the
    ``(u, v)`` pair when ``variable="uv"`` (feed to
    :func:`vorticity_cartesian`).
    """
    if isinstance(source, np.ndarray):
        arr = source
    else:
        path = str(source)
        if path.endswith((".h5", ".hdf5")):
            try:
                import h5py
            except ImportError as e:
                raise ImportError("HDF5 turbulence files need h5py (pip install h5py)") from e
            with h5py.File(path, "r") as f:
                keys = list(f.keys())
                key = variable if variable in f else keys[0]
                arr = np.asarray(f[key])
        elif path.endswith(".npz"):
            z = np.load(path)
            arr = z[variable] if variable in z.files else z[z.files[0]]
        else:
            arr = np.load(path)

    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 4:                                  # (T, C, H, W) channel layout
        if arr.shape[1] != len(channels):
            raise ValueError(f"expected {len(channels)} channels {channels}, got shape {arr.shape}")
        if variable == "uv":
            return arr[:, channels.index("u")], arr[:, channels.index("v")]
        return arr[:, channels.index(variable)]
    if arr.ndim == 3:
        if variable == "uv":
            raise ValueError("variable='uv' needs a (T, C, H, W) channel layout")
        return arr
    raise ValueError(f"cannot interpret turbulence data of shape {arr.shape}")


def turbulence_truth(source, model, *, variable="vorticity", stride=1,
                     rescale=True, length=2 * np.pi):
    """Kraichnan-type dataset → ``truth_trajectory`` for ``TwinExperiment``.

    Same role as :func:`reanalysis_truth` but for doubly-periodic Cartesian
    turbulence data (SuperBench NSKT and kin) — geometrically ideal here,
    since the box is a torus exactly like the solver's domain (no seam
    caveat).  ``variable="uv"`` computes ζ spectrally from the velocity
    channels; otherwise the vorticity channel is used directly.  The sequence
    is strided in time, block-coarsened to the model mesh and (by default)
    rescaled to model units.  Returns ``(T, n)``.

    The two Reynolds variants make a natural **cross-regime protocol**: tune
    the filter (or train a surrogate) against one Re, then replay the other
    as truth and read the robustness off ``TwinExperiment.skill`` ratios.
    """
    if isinstance(source, tuple) and len(source) == 2:
        fields = vorticity_cartesian(*source, length=length)
    else:
        out = load_turbulence(source, variable=variable)
        fields = vorticity_cartesian(*out, length=length) if isinstance(out, tuple) else out
    fields = np.asarray(fields, dtype=float)[::int(stride)]
    fields = coarsen(fields, model.grid)
    if rescale:
        fields, _ = rescale_to_model(fields, model)
    return fields.reshape(fields.shape[0], -1)


# ======================================================================
# Data access (optional stack: xarray / netCDF4 / cdsapi)
# ======================================================================
def _require_xarray():
    try:
        import xarray as xr
        return xr
    except ImportError as e:
        raise ImportError(
            "reading reanalysis files needs the optional data stack: "
            "pip install xarray netCDF4 cdsapi") from e


def load_vorticity(source, *, level=None):
    """Read a vorticity sequence from an ERA5-style dataset.

    ``source`` is a file path or an open ``xarray.Dataset``.  Uses the ``vo``
    variable when present, else computes ζ from ``u``/``v`` via
    :func:`vorticity_from_uv`.  ``level`` selects a pressure level [hPa] when
    the dataset carries a level dimension.  Returns ``(fields (T, nlat, nlon),
    lat, lon)``.
    """
    xr = _require_xarray()
    ds = source if hasattr(source, "data_vars") else xr.open_dataset(source)

    lat_name = next(n for n in ("latitude", "lat") if n in ds.coords)
    lon_name = next(n for n in ("longitude", "lon") if n in ds.coords)
    lev_name = next((n for n in ("pressure_level", "level", "isobaricInhPa") if n in ds.coords), None)
    if level is not None and lev_name is not None:
        ds = ds.sel({lev_name: level})

    lat = np.asarray(ds[lat_name].values, dtype=float)
    lon = np.asarray(ds[lon_name].values, dtype=float)
    if "vo" in ds:
        fields = np.asarray(ds["vo"].values, dtype=float)
    else:
        uname = next(n for n in ("u", "u_component_of_wind") if n in ds)
        vname = next(n for n in ("v", "v_component_of_wind") if n in ds)
        fields = vorticity_from_uv(ds[uname].values, ds[vname].values, lat, lon)
    return fields.reshape((-1,) + fields.shape[-2:]), lat, lon


def fetch_era5(target, *, dates, area, level=500, grid=0.5,
               times=("00:00", "06:00", "12:00", "18:00")):
    """Download ERA5 u/v at one pressure level to ``target`` (NetCDF).

    Needs ``cdsapi`` and CDS credentials (``~/.cdsapirc``).  ``dates`` is
    ``(start, end)`` as ``YYYY-MM-DD``; ``area`` is ``(N, W, S, E)`` degrees —
    prefer a full zonal band (``W=0, E=360``) so the domain is genuinely
    periodic like the solver's.  Returns ``target``.
    """
    try:
        import cdsapi
    except ImportError as e:
        raise ImportError("fetch_era5 needs cdsapi (pip install cdsapi) and "
                          "CDS credentials in ~/.cdsapirc") from e
    import pandas as pd
    days = pd.date_range(dates[0], dates[1], freq="D")
    cdsapi.Client().retrieve(
        "reanalysis-era5-pressure-levels",
        {
            "product_type": "reanalysis",
            "variable": ["u_component_of_wind", "v_component_of_wind"],
            "pressure_level": str(level),
            "year": sorted({f"{d.year}" for d in days}),
            "month": sorted({f"{d.month:02d}" for d in days}),
            "day": sorted({f"{d.day:02d}" for d in days}),
            "time": list(times),
            "area": list(area),
            "grid": [grid, grid],
            "format": "netcdf",
        },
        target,
    )
    return target
