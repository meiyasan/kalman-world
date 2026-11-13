"""Before / after: each application model with a C++ twin, filtered twice.

"Before" is the Python physics (the twin switched off), "after" is the same
Python model handing its pass to libKalmanApps. Both runs go through
``model.filter()``; the table gives the time of each, the ratio, and how far the
two results are apart, relative to the largest state / covariance entry.

    python tools/benchmark_twins.py            # from the repository root
    python tools/benchmark_twins.py --steps 5000 --repeat 5

Needs ``libKalman`` and ``libKalmanApps`` (``make build`` then
``cmake --install var/build``).
"""
from __future__ import annotations
import argparse
import os
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
warnings.filterwarnings("ignore")

import modules as kalman                                          # noqa: E402
import modules.framework.models.native as native                 # noqa: E402
from modules.apps.weather.storms import StormModel, synthetic_track   # noqa: E402
from modules.apps.weather.storms import TornadoEnvironmentModel, synthetic_environment   # noqa: E402
from modules.apps.ballistics import BallisticModel                # noqa: E402
from modules.apps.vehicles import CarSpeedModel                   # noqa: E402
from modules.apps.vehicles import CarLaneModel                    # noqa: E402
from modules.apps.astro.rocket import (RocketLiftoffModel, RocketAscentModel,   # noqa: E402
                                       RocketDescentModel, RocketLandingModel)
from modules.runtime.astro import R_EARTH, KM                     # noqa: E402


def timed(make, z, repeat, uk=None):
    best = float("inf")
    for _ in range(repeat):
        model = make()
        start = time.perf_counter()
        record = model.filter(z) if uk is None else model.filter(z, uk)
        best = min(best, time.perf_counter() - start)
    return best * 1e3, np.asarray(record.xk), np.asarray(record.Pk), model


def compare(make, z, repeat, uk=None):
    """Three runs of one model: the Python reference loop, the C++ loop calling
    the Python model, and the model's C++ twin when it has one. ``uk`` is the
    control sequence of a model that takes one."""
    with native.python_reference():
        t_py, x_py, P_py, _ = timed(make, z, repeat, uk)
    native.USE_NATIVE_TWINS = False
    try:
        t_loop, x_loop, _, _ = timed(make, z, repeat, uk)
    finally:
        native.USE_NATIVE_TWINS = True
    t_cx, x_cx, P_cx, model = timed(make, z, repeat, uk)
    used = model.native_handle() is not None if hasattr(model, "native_handle") else False
    scale = max(1.0, np.nanmax(np.abs(x_py)))
    dx = max(np.nanmax(np.abs(x_py - x_cx)), np.nanmax(np.abs(x_py - x_loop))) / scale
    dP = np.nanmax(np.abs(P_py - P_cx)) / max(1.0, np.nanmax(np.abs(P_py)))
    return t_py, t_loop, t_cx, dx, dP, used


def cases(steps):
    rng = np.random.default_rng(0)
    for hemisphere, omega, label in (("N", 0.0, "straight"), ("N", -0.022, "recurving"),
                                     ("S", 0.022, "recurving")):
        _, fixes = synthetic_track(n_steps=steps, hemisphere=hemisphere, seed=1)
        yield (f"StormModel, {hemisphere}H {label}",
               lambda h=hemisphere, w=omega: StormModel("TC", hemisphere=h, omega=w), fixes)
    yield "StormModel, track only", lambda: StormModel("TC", observe_intensity=False), fixes[:, :2]

    t = np.arange(steps) / 100.0
    z = np.c_[30 * t, 10 * t, 120 * t - 0.5 * 9.81 * t**2] + rng.normal(0, 0.5, (steps, 3))
    z[::17] = np.nan                                              # dropped measurements
    yield "BallisticModel, order 3, drag", lambda: BallisticModel("b", fs=100.0, c_drag=0.01), z
    yield "BallisticModel, order 2", lambda: BallisticModel("b", fs=100.0, order=2), z

    obs, _ = synthetic_environment(n_steps=steps, onsets=((steps // 3, steps // 2),), seed=0)
    yield "TornadoEnvironmentModel", lambda: TornadoEnvironmentModel("ENV", mu=3.0), obs

    car = CarSpeedModel("CAR")
    run = car.sample(steps / car.fs, seed=0)
    yield "CarSpeedModel", lambda: CarSpeedModel("CAR"), np.asarray(run.zk)

    r0 = 6.771e6
    v0 = np.sqrt(6.674e-11 * 5.972e24 / r0)
    w = v0 / r0
    orbit = np.c_[r0 * np.cos(w * t), r0 * np.sin(w * t), 0 * t] + rng.normal(0, 50.0, (steps, 3))

    def orbital():
        model = BallisticModel("o", fs=100.0, g=None, measurement_uncertainty=50.0)
        model.initial_state([r0, 0, 0, 0, v0, 0, 0, 0, 0])
        return model
    yield "BallisticModel, orbital gravity", orbital, orbit

    # The lane-keeping car and the rocket family (the scenarios of the notebooks
    # 12_KalmanAI and 06_RocketScience). A fourth item is the control sequence.
    lt = np.arange(steps) / 50.0
    steer = (0.01 * np.sin(0.8 * lt) + 0.15 * ((lt % 20 > 3.0) & (lt % 20 < 3.5))
             - 0.15 * ((lt % 20 > 3.5) & (lt % 20 < 4.0))).reshape(-1, 1)
    lane = lambda: CarLaneModel("camera", fs=50, x0=[[0.3], [0.0], [15.0]], lateral_uncertainty=0.05)
    yield "CarLaneModel", lane, np.asarray(lane().sample(steps / 50.0, uk=steer, seed=0).zk), steer

    def liftoff():
        model = RocketLiftoffModel("LIFTOFF", m0=549_000.0, m_dry=138_000.0, ve=2900.0, mdot=2600.0,
                                   drag_area=8.0, fs=10.0, track_m=50.0, q_mass=0.0)
        model.initial_state([0.0, 0.0, model.m0], P0=np.diag([50.0**2, 10.0**2, 50.0**2]))
        return model
    yield "RocketLiftoffModel", liftoff, np.asarray(liftoff().sample(steps / 10.0, seed=0).zk)

    program = np.zeros((steps, 2))
    for i in range(steps):
        pitch = np.deg2rad(89.0 - 111.0 * (i / (steps - 1)) ** 0.57)
        program[i] = [19.4 * np.sin(pitch), 19.4 * np.cos(pitch)]

    def ascent():
        model = RocketAscentModel(dt_s=2.0 * 240 / steps, track_km=1.0)
        model.initial_state(np.array([[R_EARTH], [0], [0], [0]]),
                            P0=np.diag([(1 * KM)**2, (1 * KM)**2, 0.01**2, 0.01**2]))
        return model
    yield ("RocketAscentModel", ascent,
           np.asarray(ascent().sample(steps / ascent().fs, uk=program, seed=0).zk), program)

    booster = dict(m0=30_000.0, m_dry=22_000.0, thrust_max=845e3, drag_area=10.0)
    x0 = np.array([350.0, 2000.0, -20.0, -60.0, 0.05, 0.0, 30_000.0])

    def descent(control=False):
        model = RocketDescentModel(fs=steps / 70.0, control=control, **booster)
        model.initial_state(x0)
        return model
    yield "RocketDescentModel", descent, np.asarray(descent().sample(70.0, seed=0).zk)
    commands, _ = descent(True).autopilot(x0, steps)
    yield ("RocketDescentModel, control", lambda: descent(True),
           np.asarray(descent(True).sample(70.0, uk=commands, process_noise=False, seed=0).zk), commands)

    def landing():
        model = RocketLandingModel("LANDING", fs=steps / 40.0)
        model.initial_state([3000.0, -150.0, 1000.0], P0=np.diag([20.0**2, 5.0**2, 10.0**2]))
        return model
    yield "RocketLandingModel", landing, np.asarray(landing().sample(40.0, seed=0).zk)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--steps", type=int, default=2000, help="measurements per run (default 2000)")
    parser.add_argument("--repeat", type=int, default=3, help="runs per case, best kept (default 3)")
    args = parser.parse_args()

    kalman.set_debug(0)
    if not native.apps_library_names():
        sys.exit("libKalmanApps was not found: build and install it first (see the module docstring).")

    print(f"{args.steps} steps per run, best of {args.repeat}\n")
    print(f"{'model':32s} {'Python':>10s} {'C++ loop':>10s} {'C++ model':>10s} {'faster':>8s} {'state':>9s} {'covariance':>11s}")
    for name, make, z, *uk in cases(args.steps):
        t_py, t_loop, t_cx, dx, dP, used = compare(make, z, args.repeat, *uk)
        twin = f"{t_cx:8.1f} ms {t_py / t_cx:7.1f}x" if used else f"{'none':>11s} {t_py / t_loop:7.1f}x"
        print(f"{name:32s} {t_py:8.1f} ms {t_loop:8.1f} ms {twin} {dx:9.1e} {dP:11.1e}")
    print("\nPython: the reference loop. C++ loop: the C++ filter calling the Python model.")
    print("C++ model: the model's C++ twin. faster: Python against the fastest of the two.")
    print("state / covariance: largest difference from the Python run, relative to the largest entry.")


if __name__ == "__main__":
    main()
