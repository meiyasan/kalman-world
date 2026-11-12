"""Time the controllers and routers on their C++ path against their Python reference.

    python tools/benchmark_controllers.py

Per call, best of several repeats, on the lane-keeping car of notebook 12
(CarLaneModel, n = 3, l = 1) and on random IKF inputs. "C++" is the class as
shipped with libKalman installed; "Python" is the same class on its Python
reference path (its C++ object switched off on the instance), everything else
unchanged — the model's own f still runs as it does for any caller, so the
difference is the controller's or router's alone.
"""
from __future__ import annotations

import sys
import time

import numpy as np

sys.path.insert(0, ".")

from modules.apps.vehicles import CarLaneModel                           # noqa: E402
from modules.framework.controllers import (ProportionalKalmanController, PIDKalmanController,  # noqa: E402
                                           LQRKalmanController, MPPIKalmanController,
                                           BoostKalmanController, AdaptiveQRKalmanController)
from modules.framework.routers import BayesianRouter, ThresholdRouter, HybridRouter  # noqa: E402


def best(fn, repeats=5, number=200):
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(number):
            fn()
        times.append((time.perf_counter() - t0) / number)
    return min(times)


def both(label, make, call, repeats=5, number=200):
    out = []
    for reference in (False, True):
        obj = make()
        if reference:
            obj._native_handle = lambda: None                   # a controller's reference path
            obj._native_router = lambda: None                   # a router's
        call(obj)                                               # warm up (builds the C++ object)
        out.append(best(lambda: call(obj), repeats, number))
    cxx, py = out
    print(f"{label:38s} C++ {cxx * 1e6:10.1f} us   Python {py * 1e6:10.1f} us   x{py / cxx:6.2f}")


def main():
    lane = lambda: CarLaneModel("camera", fs=50, x0=np.array([[1.0], [0.0], [15.0]]), lateral_uncertainty=0.3)
    m = lane()
    Qc, Rc, xr = m.control_cost()
    cost = dict(Qcost=Qc, Rcost=Rc, x_ref=xr)
    x, P, nu = np.array([[0.5], [0.02], [15.1]]), 0.1 * np.eye(3), np.array([[0.3]])
    step = lambda c: c(x, P, 0.1, nu=nu)

    print("controllers, one call (policy + update + cost):")
    both("Proportional", lambda: ProportionalKalmanController(lane(), Kp=np.array([[0.1, 0.7, 0.0]]), **cost), step)
    both("PID", lambda: PIDKalmanController(lane(), Kp=np.array([[0.1, 0.7, 0.0]]), Ki=np.array([[0.02, 0, 0]]),
                                             Kd=np.array([[0, 0.05, 0]]), **cost), step)
    both("LQR (policy)", lambda: LQRKalmanController(lane(), **cost).fit(), step)
    both("LQR fit (DARE + gain; Python = scipy)", lambda: LQRKalmanController(lane(), **cost),
         lambda c: c.fit() if "_native_handle" not in c.__dict__ else c._reference_fit(), number=50)
    both("Boost (window 5)", lambda: BoostKalmanController(lane(), boost_factor=5.0, threshold=2.0, window=5), step)
    both("AdaptiveQR (window 20)", lambda: AdaptiveQRKalmanController(lane(), window=20, q_min=0.5, q_max=10.0),
         step)
    both("MPPI 30 x 20 x 2 (Python f)", lambda: MPPIKalmanController(lane(), n_rollouts=30, horizon=20, n_iters=2,
                                                                     noise_sigma=0.05, seed=7, **cost),
         step, repeats=3, number=5)
    both("MPPI 100 x 40 x 3 (notebook 12)", lambda: MPPIKalmanController(lane(), n_rollouts=100, horizon=40, n_iters=3,
                                                                         noise_sigma=0.05, seed=7, **cost),
         step, repeats=2, number=2)

    from modules.framework.native import core
    from modules.framework.models.native import NativeModel
    try:
        core.load_plugin(core.resolve_plugin("KalmanPlugin_example"))
        pendulum = NativeModel("pendulum")
        xp, Pp = np.asarray(pendulum.x0, dtype=float).reshape(-1, 1) + 0.1, np.eye(2)
        both("MPPI 100 x 40 x 3 (C++ pendulum)", lambda: MPPIKalmanController(pendulum, n_rollouts=100, horizon=40,
                                                                              n_iters=3, seed=7),
             lambda c: c._policy(xp, Pp, 0.0), repeats=3, number=3)
    except FileNotFoundError:
        print("MPPI on a C++ model: skipped (make examples builds the plugin)")

    print("routers, one route() call:")
    rng = np.random.default_rng(0)
    for rows, M in ((50, 2), (500, 4)):
        nis, c, L = rng.chisquare(1, (rows, M)), rng.dirichlet(np.ones(M)), rng.uniform(0, 1, M)
        route = lambda r: r(nis, c, L)
        both(f"Bayesian ({rows} x {M})", BayesianRouter, route, number=2000)
        both(f"Threshold soft, window 10 ({rows} x {M})", lambda: ThresholdRouter(m=1, window=10, soft=True), route,
             number=2000)
        both(f"Hybrid(Bayes, Threshold) ({rows} x {M})",
             lambda: HybridRouter(BayesianRouter(), ThresholdRouter(m=1, window=10, soft=True), alpha=0.3),
             route, number=2000)


if __name__ == "__main__":
    main()
