"""Netatmo indoor modules: the CO2 of a room, five minutes at a time.

A Netatmo weather station has a main indoor module and up to three additional
indoor modules; each reads CO2 (with temperature, humidity and, for the main one,
noise and pressure) and reports about every five minutes. A room watched by five
modules takes two stations: each module says which station it belongs to.

What the data sheet gives, and what this adapter takes from it:

- CO2 from 0 to 5000 ppm, accuracy +/- 50 ppm or +/- 5 %. Most of that is an
  offset that moves slowly (the sensor ages, and the module recalibrates
  itself), so a module is an instrument with a calibration: its
  ``"<name>.offset"`` is estimated with ``joint``, as a slow random walk (a
  ``drift`` per step, no ``tau``: an offset does not return to where it started);
- the noise from one reading to the next is much smaller than that accuracy:
  ``sigma`` (20 ppm by default) is that noise, to be checked on a quiet night;
- one reading every five minutes: the room model steps every five minutes
  (``RoomCO2Model(..., fs=1/300)``).

Where the readings come from:

- :class:`NetatmoCloud`: the Netatmo web API (``getmeasure``), the only interface
  Netatmo documents. ``scale="max"`` gives every reading; ``"30min"`` and longer
  give averages. The credentials are outbound secrets, never in the
  repository (see the class);
- :class:`NetatmoCSV`: files on disk, the export of the Netatmo web page or a
  local logger's ``time, co2`` columns. The station has no documented local
  interface; a local reader (HomeKit, a logger) writes such a file, or is any
  function returning ``(times, ppm)``, given as ``source=``.

Either way the readings are laid on the model's steps from ``start``: a step
with no reading is NaN, which the filter skips::

    from kalman.apps import RoomCO2Model, empty_at_night
    from kalman.adapters import NetatmoModule, NetatmoCloud

    room = RoomCO2Model("unread", count_people=True, fs=1 / 300)   # every reading comes from a module
    cloud = NetatmoCloud()                                     # credentials from the environment
    modules = [NetatmoModule(room, "main", (2, 2), station=STATION_1, source=cloud),
               NetatmoModule(room, "desk", (2, 9), station=STATION_1, module=MODULE_1, source=cloud),
               ...]
    watched = room
    for module in modules:
        watched = watched.add_instrument(module)
    watched = watched.add_instrument(empty_at_night(room))    # nobody between midnight and 6 am
    zk = watched.measurements(start=t0, end=t1)                # fetched, on the 5-minute steps
    run = watched.joint({f"{m.name}.offset": (50.0, 0.5) for m in modules[1:]}).filter(zk, uk=np.ones((len(zk), 1)))

The outdoor level (``outdoor=``) is what the empty room settles to, so the
offsets are measured against it. It is not known better than about 20 ppm, so
one module is kept as the reference and the others' offsets are estimated.
"""
from __future__ import annotations

import csv
import json
import os
import time as _time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np

from ..framework.constants import KalmanProviderKind
from ..framework.instrument import KalmanInstrument
from ..framework.provider import KalmanProvider
from ..providers.credentials import get_credential

STEP = 300.0                                       # seconds between two readings of a module
API = "https://api.netatmo.com"


def on_steps(times, values, start, end, step=STEP):
    """Readings at ``times`` [unix s] laid on the steps ``start, start + step, ...``
    before ``end``: one row per step, the mean of the readings within half a step
    of it, NaN where there is none."""
    n = int(np.floor((float(end) - float(start)) / float(step)))
    out = np.full(max(n, 0), np.nan)
    if n <= 0:
        return out.reshape(0, 1)
    t = np.asarray(times, dtype=float).reshape(-1)
    v = np.asarray(values, dtype=float).reshape(-1)
    k = np.rint((t - float(start)) / float(step)).astype(np.int64)
    keep = (k >= 0) & (k < n) & np.isfinite(v)
    sums, counts = np.zeros(n), np.zeros(n)
    np.add.at(sums, k[keep], v[keep])
    np.add.at(counts, k[keep], 1.0)
    read = counts > 0
    out[read] = sums[read] / counts[read]
    return out.reshape(n, 1)


class NetatmoModule(KalmanInstrument):
    """One Netatmo indoor module, reading the CO2 of one cell of a room.

    Parameters
    ----------
    room : the room model (a :class:`~kalman.apps.RoomCO2Model`); the module reads its cell ``at``.
    name : what the module is called; its offset is the parameter ``"<name>.offset"``.
    at : ``(row, column)`` of the cell where it stands.
    station : the station's id (the main module's MAC address).
    module : the additional module's id; ``None`` for the main module.
    source : where the readings come from: :class:`NetatmoCloud`, :class:`NetatmoCSV`,
        or a function ``(station, module, start, end) -> (times, ppm)``.
    sigma : the noise from one reading to the next [ppm].
    offset : the offset the module is believed to have [ppm].
    """

    def __init__(self, room, name, at, *, station, module=None, source=None, sigma=20.0, offset=0.0):
        self.station, self.module, self.source = station, module, source
        provider = None if source is None else self._readings
        super().__init__(name, [room.sensor(name, at, sigma=sigma)], rate=1.0 / STEP, offset=offset,
                         provider=provider)

    def _readings(self, *, start, end, **query):
        fetch = self.source.readings if hasattr(self.source, "readings") else self.source
        times, ppm = fetch(self.station, self.module, start, end, **query)
        return on_steps(times, ppm, start, end)


class NetatmoCloud(KalmanProvider):
    """The CO2 of Netatmo modules, from the Netatmo web API.

    The credentials of a Netatmo app (dev.netatmo.com, scope ``read_station``)
    are outbound secrets, resolved like every provider's
    (:mod:`kalman.providers.credentials`): from the environment,
    ``NETATMO_CLIENT_ID``, ``NETATMO_CLIENT_SECRET`` and ``NETATMO_REFRESH_TOKEN``,
    or from ``~/.kalman/credentials.json`` (``netatmo_client_id``, ...). They are
    never in the repository.

    Netatmo replaces the refresh token each time it is used, and the old one stops
    working: the new one is kept in ``~/.kalman/netatmo.json`` (``token_file=`` or
    ``NETATMO_TOKEN_FILE`` for another place), readable by its owner only, and is
    the one used next time.

    ``scale``: ``"max"`` for every reading, ``"30min"``, ``"1hour"``, ... for averages.
    """

    NAME = "Netatmo Cloud"
    DESCRIPTION = "CO2 readings of Netatmo indoor modules, from the Netatmo web API (getmeasure)."
    VERSION = "1.0"
    KIND = KalmanProviderKind.REMOTE
    MEASUREMENTS = ({"name": "co2", "units": "ppm", "description": "CO2 of one indoor module"},)
    REQUIRES = ("NETATMO_CLIENT_ID", "NETATMO_CLIENT_SECRET", "NETATMO_REFRESH_TOKEN")
    PAGE = 1024                                    # readings per request, at most
    TOKENS = "~/.kalman/netatmo.json"              # where the replaced refresh token is kept

    def __init__(self, *, scale="max", token_file=None, opener=None, cache=True):
        super().__init__(cache=cache)
        self.scale = scale
        self.token_file = os.path.expanduser(token_file or os.environ.get("NETATMO_TOKEN_FILE") or self.TOKENS)
        self._open = opener or (lambda request: urllib.request.urlopen(request, timeout=30))
        self._access, self._expires = None, 0.0

    def _call(self, path, params, *, post=False):
        data = urllib.parse.urlencode(params)
        if post:
            request = urllib.request.Request(API + path, data=data.encode(), method="POST")
        else:
            request = urllib.request.Request(f"{API}{path}?{data}")
            request.add_header("Authorization", f"Bearer {self._token()}")
        with self._open(request) as response:
            return json.loads(response.read().decode())

    def _kept(self):
        """The refresh token kept from the last run, or ``None``."""
        try:
            with open(self.token_file, encoding="utf-8") as f:
                return json.load(f).get("refresh_token")
        except (OSError, ValueError):
            return None

    def _keep(self, refresh_token):
        """Write the new refresh token: owner-only from its creation, replaced in one move."""
        os.makedirs(os.path.dirname(self.token_file), mode=0o700, exist_ok=True)
        draft = self.token_file + ".new"
        with os.fdopen(os.open(draft, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w", encoding="utf-8") as f:
            json.dump({"refresh_token": refresh_token}, f)
        os.replace(draft, self.token_file)

    def _token(self):
        if self._access and _time.time() < self._expires - 60:
            return self._access
        secret = lambda key: get_credential(f"netatmo_{key}", env=f"NETATMO_{key.upper()}")   # fail-closed
        client_id, client_secret = secret("client_id"), secret("client_secret")
        kept = self._kept()
        given = get_credential("netatmo_refresh_token", env="NETATMO_REFRESH_TOKEN", required=kept is None)
        answer = None
        for refresh_token in dict.fromkeys(t for t in (kept, given) if t):      # the kept one first
            try:
                answer = self._call("/oauth2/token", {"grant_type": "refresh_token", "refresh_token": refresh_token,
                                                      "client_id": client_id, "client_secret": client_secret},
                                    post=True)
                break
            except urllib.error.HTTPError as refused:
                last = refused
        if answer is None:
            raise PermissionError("NetatmoCloud: Netatmo refused the refresh token; make a new one on dev.netatmo.com "
                                  f"and set NETATMO_REFRESH_TOKEN ({last}).") from last
        self._access, self._expires = answer["access_token"], _time.time() + float(answer.get("expires_in", 10800))
        if answer.get("refresh_token"):
            self._keep(answer["refresh_token"])
        return self._access

    def readings(self, station, module, start, end):
        """``(times, ppm)`` of one module between ``start`` and ``end`` [unix s]."""
        def produce():
            times, ppm, begin = [], [], int(start)
            while begin < int(end):
                params = {"device_id": station, "scale": self.scale, "type": "co2", "date_begin": begin,
                          "date_end": int(end), "limit": self.PAGE, "optimize": "false", "real_time": "true"}
                if module:
                    params["module_id"] = module
                body = self._call("/api/getmeasure", params).get("body") or {}
                page = sorted((int(t), v[0]) for t, v in body.items() if v and v[0] is not None)
                if not page:
                    break
                times += [t for t, _ in page]
                ppm += [v for _, v in page]
                if len(page) < self.PAGE:
                    break
                begin = page[-1][0] + 1
            return np.asarray(times, dtype=float), np.asarray(ppm, dtype=float)
        return self._cached((station, module, int(start), int(end), self.scale), produce)

    def fetch(self, station, module=None, start=None, end=None):
        times, ppm = self.readings(station, module, start, end)
        return {"time": times, "zk": ppm.reshape(-1, 1), "meta": {"units": "ppm", "scale": self.scale}}


class NetatmoCSV(KalmanProvider):
    """The CO2 of Netatmo modules, from files: one per module.

    ``files`` maps a module id (the station's id for the main module) to a file.
    A file is the export of the Netatmo web page (``;``-separated, a few lines of
    title, then a header with ``Timestamp`` and ``CO2``), or any table whose
    header names a time column (``Timestamp``, ``time``: unix seconds) and a
    ``CO2`` column.
    """

    NAME = "Netatmo CSV"
    DESCRIPTION = "CO2 readings of Netatmo indoor modules, from exported or logged files."
    VERSION = "1.0"
    KIND = KalmanProviderKind.FILE
    MEASUREMENTS = NetatmoCloud.MEASUREMENTS

    def __init__(self, files, *, cache=True):
        super().__init__(cache=cache)
        self.files = dict(files)

    @staticmethod
    def parse(path):
        """``(times, ppm)`` from one file."""
        with open(path, encoding="utf-8-sig", newline="") as f:
            lines = f.read().splitlines()
        for i, line in enumerate(lines):
            lower = line.lower()
            if "co2" in lower and ("timestamp" in lower or "time" in lower):
                break
        else:
            raise ValueError(f"NetatmoCSV: {path} has no header with a time and a CO2 column.")
        delimiter = ";" if lines[i].count(";") >= lines[i].count(",") else ","
        rows = list(csv.reader(lines[i:], delimiter=delimiter))
        header = [h.strip().strip('"').lower() for h in rows[0]]
        when = next((j for j, h in enumerate(header) if h in ("timestamp", "time", "unix", "t")), None)
        what = next((j for j, h in enumerate(header) if h.startswith("co2")), None)
        if when is None or what is None:
            raise ValueError(f"NetatmoCSV: {path} needs a time column (Timestamp or time, unix seconds) and a CO2 "
                             f"column; its header is {rows[0]}.")
        times, ppm = [], []
        for row in rows[1:]:
            try:
                moment, value = float(row[when]), float(row[what])
            except (ValueError, IndexError):             # a missing reading: the row is skipped whole
                continue
            times.append(moment)
            ppm.append(value)
        return np.asarray(times), np.asarray(ppm)

    def readings(self, station, module, start, end):
        key = module or station
        if key not in self.files:
            raise KeyError(f"NetatmoCSV: no file for module {key!r}; it has {sorted(self.files)}.")
        times, ppm = self._cached(key, lambda: self.parse(self.files[key]))
        keep = (times >= float(start)) & (times < float(end))
        return times[keep], ppm[keep]

    def fetch(self, station, module=None, start=None, end=None):
        times, ppm = self.readings(station, module, start, end)
        return {"time": times, "zk": ppm.reshape(-1, 1), "meta": {"units": "ppm"}}
