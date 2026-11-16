"""The model operations, one after the other, on a cart.

Each operation takes models and gives back a model; the filter is always the
ordinary one. Everything is computed by the C++ library.

    add_sensor   one more sensor on the same state
    lag          the last states carried along: a sensor that reports late
    joint        a parameter estimated with the state
    compose      two systems side by side
    route        one system, two regimes

    python examples/python/operations.py
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
KalmanModel, KalmanSensor = framework.KalmanModel, framework.KalmanSensor
compose, route = framework.compose, framework.route

warnings.simplefilter("ignore")
FS, DT = 10.0, 0.1
rng = np.random.default_rng(1)
rmse = lambda a, b: float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def cart(name="cart", q=1.0):
    """Position and velocity; a noisy position sensor."""
    Q = q * np.array([[DT**4 / 4, DT**3 / 2], [DT**3 / 2, DT**2]])
    return KalmanModel(name, fs=FS, F=np.array([[1, DT], [0, 1]]), H=np.array([[1.0, 0.0]]), Q=Q,
                       R=np.array([[0.25]]), x0=np.array([[0.0], [1.0]]), P0=np.eye(2))


def simulate(model, steps, n=2):
    """Truth and readings from the model's own f and h (noise on the first ``n`` components)."""
    Lq = np.linalg.cholesky(model.Q()[:n, :n] + 1e-15 * np.eye(n))
    Lr = np.linalg.cholesky(model.R())
    x, xs, zs = np.asarray(model.x0, dtype=float).copy(), [], []
    for _ in range(steps):
        x = model.f(x)
        x[:n] += Lq @ rng.standard_normal((n, 1))
        xs.append(x.ravel().copy())
        zs.append((model.h(x) + Lr @ rng.standard_normal((model.m, 1))).ravel())
    return np.array(xs), np.array(zs)


def states(record, steps):
    return np.asarray(record.xk).reshape(steps, -1)


N = 1000

# --- add_sensor, then lag: a precise sensor that reports 0.3 s late -------------------------
camera = KalmanSensor("camera", G=[[1.0, 0.0]], R=0.01, delay=0.3)
late = cart().add_sensor(camera).lag(3)
truth, zk = simulate(late, N)
with_camera = states(late.filter(zk, reset=True), N)
without = states(cart().filter(zk[:, :1], reset=True), N)
print("add_sensor + lag: position error %.3f with the late camera, %.3f without"
      % (rmse(with_camera[:, 0], truth[:, 0]), rmse(without[:, 0], truth[:, 0])))
print("                  the state three steps ago is known to %.3f (fixed-lag smoothing)"
      % rmse(with_camera[:, late.block(3)][:, 0], truth[:, 6]))

# --- joint: the offset of an uncalibrated thermometer-like sensor ------------------------------
probe = KalmanSensor("probe", G=[[1.0, 0.0]], R=0.04)
gauge = framework.compose(probe, name="gauge").calibrate(gain=1.0, offset=0.0)
biased = cart().add_instrument(framework.compose(probe, name="gauge").calibrate(1.0, 0.8))
truth, zk = simulate(biased, N)
model = cart().add_instrument(gauge).joint({"gauge.offset": (2.0, 0.0)})
record = model.filter(zk, reset=True)
print("joint:            gauge.offset = %.3f +/- %.3f (truth 0.8)"
      % (record.xk[-1, 2], np.sqrt(record.Pk[-1, 2, 2])))

# --- compose: two carts in one filter ------------------------------------------------------------
pair = compose(cart("left"), cart("right"))
truth, zk = simulate(pair, N, n=4)
both = states(pair.filter(zk, reset=True), N)
print("compose:          n = %d, m = %d, parts %s; position errors %.3f and %.3f"
      % (pair.n, pair.m, pair.part_names, rmse(both[:, 0], truth[:, 0]), rmse(both[:, 2], truth[:, 2])))

# --- route: a cart that is quiet, then shaken ---------------------------------------------------------
quiet, shaken = cart("quiet", q=0.01), cart("shaken", q=100.0)
_, z_quiet = simulate(cart("quiet", q=0.01), N // 2)
restart = cart("shaken", q=100.0)
restart.initial_state(np.array([[z_quiet[-1, 0]], [1.0]]), np.eye(2))
_, z_shaken = simulate(restart, N // 2)
Pi = np.array([[0.98, 0.02], [0.02, 0.98]])
modes = route([quiet, shaken], Pi)
modes.filter(np.vstack([z_quiet, z_shaken]))
mu = modes.mode_probabilities[:, 1]
print("route:            probability of the shaken regime: %.2f in the first half, %.2f in the second"
      % (mu[: N // 2].mean(), mu[N // 2:].mean()))
