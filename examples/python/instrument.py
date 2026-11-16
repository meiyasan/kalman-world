"""Calibrating an instrument while it measures.

A room: its temperature and its CO2 level, both slowly wandering. A CO2 meter
reads the second. A thermometer of unknown offset reads the first, as an
instrument with a calibration, ``z = gain * T + offset``.

1. With that thermometer alone, its offset and the temperature cannot be told
   apart: ``observability`` says so before any data.
2. A second, calibrated thermometer settles it: ``joint`` then estimates the
   offset along with the state.

Everything is computed by the C++ library (``Kalman::Instrument``,
``Models::Fusion``, ``Models::Joint``); the same scenario is
``tests/cxx/25_instrument.cc``.

    python examples/python/instrument.py
"""
from __future__ import annotations
import os
import sys
import warnings

import numpy as np

try:
    import kalman
except ImportError:                                # running from a checkout that is not installed
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    import modules as kalman

framework = sys.modules[kalman.__name__ + ".framework"]
KalmanModel, KalmanSensor, compose = framework.KalmanModel, framework.KalmanSensor, framework.compose

OFFSET = 1.5                                       # what the thermometer really adds [K]


def room():
    return KalmanModel("co2-meter", fs=1.0, F=np.eye(2), H=np.array([[0.0, 1.0]]), Q=np.diag([1e-4, 1e-2]),
                       R=np.array([[4.0]]), x0=np.array([[21.0], [450.0]]), P0=np.diag([4.0, 100.0]))


def thermometer(offset=0.0):
    probe = KalmanSensor("probe", G=[[1.0, 0.0]], R=0.04, labels={"T_PROBE": "degC"})
    return compose(probe, name="thermo").calibrate(gain=1.0, offset=offset)


reference = KalmanSensor("reference", G=[[1.0, 0.0]], R=0.01, labels={"T_REFERENCE": "degC"})

warnings.simplefilter("ignore")

# 1. one thermometer, its offset unknown
alone = room().add_instrument(thermometer()).joint({"thermo.offset": (2.0, 0.0)})
report = alone.observability(50)
print("one thermometer:")
print(f"  directions determined: {report['rank']} of {alone.n}")
for name, share in report["weakness"].items():
    print(f"  {name:<16} weakness {share:.2f}")

# 2. with a calibrated one beside it
checked = room().add_instrument(thermometer()).add_sensor(reference)
model = checked.joint({"thermo.offset": (2.0, 0.0)})
print("with a reference thermometer:")
print(f"  instruments: {[(i.name, [s.name for s in i]) for i in checked.instruments]}")
print(f"  observable: {model.observability(50)['observable']}")

truth = room().add_instrument(thermometer(OFFSET)).add_sensor(reference)
rng = np.random.default_rng(20261117)
Lq, Lr = np.linalg.cholesky(truth.Q()), np.linalg.cholesky(truth.R())
x, zk = truth.x0.copy(), []
for _ in range(400):
    x = truth.f(x) + Lq @ rng.standard_normal((2, 1))
    zk.append((truth.h(x) + Lr @ rng.standard_normal((3, 1))).ravel())

record = model.filter(np.array(zk), smooth=False, reset=True)
estimate, sigma = record.xk[-1, 2], np.sqrt(record.Pk[-1, 2, 2])
print(f"  thermo.offset: {estimate:.3f} +/- {sigma:.3f} (truth {OFFSET})")
