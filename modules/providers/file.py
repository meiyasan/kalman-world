"""File provider — load arrays/records from disk (npz / npy / json / csv)."""
from __future__ import annotations
import json
import numpy as np

from .base import KalmanProvider
from ..framework.constants import KalmanProviderKind


class FileProvider(KalmanProvider):
    """Load a measurement set (and optional truth) from a file.

    - ``.npz`` — keys ``zk`` (required) and optionally ``xk``/``time``;
    - ``.npy`` — a single ``(T, m)`` measurement array;
    - ``.json`` — ``{"zk": [...], "xk": [...]?, "time": [...]?}``;
    - ``.csv``  — a numeric table taken as ``zk``.
    """

    NAME = "File Provider"
    DESCRIPTION = "Load measurements/truth from npz/npy/json/csv on disk."
    VERSION = "1.0"
    KIND = KalmanProviderKind.FILE

    MEASUREMENTS = (
        {"name": "zk", "units": "as stored",
         "description": "Measurement rows, shape (T, m). Required in every "
                        "supported format; a .npy or .csv file IS this array."},
        {"name": "xk", "units": "as stored",
         "description": "Optional truth trajectory (npz/json key 'xk')."},
        {"name": "time", "units": "s",
         "description": "Optional sample times (npz/json key 'time')."},
    )

    REQUIRES = ("a readable path= (.npz / .npy / .json / .csv)",)

    def __init__(self, path=None, **kw):
        super().__init__(**kw)
        self.path = path

    @staticmethod
    def save(path, zk, xk=None, time=None):
        """Write a measurement set (and optional truth) to ``.npz``/``.json``."""
        zk = np.asarray(zk, dtype=float)
        payload = {"zk": zk}
        if xk is not None:
            payload["xk"] = np.asarray(xk, dtype=float)
        if time is not None:
            payload["time"] = np.asarray(time, dtype=float)
        if str(path).endswith(".json"):
            with open(path, "w") as fh:
                json.dump({k: v.tolist() for k, v in payload.items()}, fh)
        else:
            np.savez(path, **payload)
        return path

    def fetch(self, path=None, **kw):
        p = path or self.path
        if p is None:
            raise ValueError("FileProvider needs a path (constructor or fetch(path=...)).")

        def _load():
            if p.endswith(".npz"):
                z = np.load(p)
                out = {k: np.asarray(z[k], dtype=float) for k in z.files}
            elif p.endswith(".npy"):
                out = {"zk": np.asarray(np.load(p), dtype=float)}
            elif p.endswith(".json"):
                with open(p) as fh:
                    d = json.load(fh)
                out = {k: np.asarray(v, dtype=float) for k, v in d.items()}
            elif p.endswith(".csv"):
                out = {"zk": np.atleast_2d(np.loadtxt(p, delimiter=","))}
            else:
                raise ValueError(f"unsupported file type: {p}")
            out.setdefault("truth", out.get("xk"))
            out["meta"] = {"provider": "file", "path": p}
            return out

        return self._cached(p, _load)
