"""Before / after: each application model with a C++ twin, filtered twice, and the
composite models of matrices, which run the C++ linear pass.

"Before" is the Python physics (the twin switched off), "after" is the same
Python model handing its pass to libKalmanApps. Both runs go through
``model.filter()``; the table gives the time of each, the ratio, and how far the
two results are apart, relative to the largest state / covariance entry.

    python tools/benchmark_twins.py            # from the repository root
    python tools/benchmark_twins.py --steps 5000 --repeat 5
    python tools/benchmark_twins.py --fields   # the field models on the ensemble engine
    python tools/benchmark_twins.py --fields --notebook   # and notebook 13's assimilation, 64 x 64

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
    from modules.apps.weather.storms import LearnedIntensityModel, synthetic_steered_track
    tracks = [synthetic_steered_track(n_steps=40, lon0=150 + 3 * k, lat0=8 + k, seed=k)[0] for k in range(6)]
    learned = LearnedIntensityModel().fit(tracks).as_intensity_fn()
    yield "StormModel, learned intensity", lambda: StormModel("TC", intensity_fn=learned), fixes

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

    # The space models, on the scenarios of notebook 08: the free return (180 s
    # samples, 4 RK4 substeps) and the Earth-Mars cruise of the 2026 window
    # (12 h samples, Mercury, Venus, Jupiter and Saturn tabulated from the ephemeris).
    from modules.apps.astro.space import TransLunarModel, InterplanetaryModel   # noqa: E402
    from modules.apps.astro.space.transfers import lambert                      # noqa: E402
    # R_EARTH and KM come from the module's import: importing them here again made
    # them locals of cases(), unbound in the ascent case above (a NameError).
    from modules.runtime.astro import D_MOON, GM_EARTH, DAY, body_state   # noqa: E402
    from astropy.time import Time, TimeDelta                                    # noqa: E402

    rp = R_EARTH + 400 * KM
    vt = np.sqrt(GM_EARTH * (2.0 / rp - 1.0 / (0.5 * (rp + D_MOON)))) + 0.034

    def free_return():
        model = TransLunarModel(moon_phase_deg=133.0, sun_phase_deg=60.0, dt_s=180.0, track_km=5.0, n_sub=4)
        model.initial_state(np.array([[rp], [0], [0], [0], [vt], [0]]),
                            P0=np.diag([(50 * KM) ** 2] * 3 + [0.01 ** 2] * 3))
        return model
    yield "TransLunarModel, free return", free_return, np.asarray(free_return().sample(steps * 0.18, seed=0).zk)

    dep, tof = Time("2026-10-28"), 311.0
    r1, _ = body_state("earth", dep)
    r2, _ = body_state("mars", dep + TimeDelta(tof * 86400, format="sec"))
    v1, _ = lambert(r1, r2, tof * DAY)

    def cruise():
        model = InterplanetaryModel("earth", "mars", epoch=dep, span_days=0.5 * steps + 20,
                                    dt_s=43200.0, track_Mm=0.02)
        model.initial_state(np.r_[r1, v1].reshape(-1, 1), P0=np.diag([1.0] * 3 + [1e-4] * 3))
        return model
    yield "InterplanetaryModel, Earth-Mars", cruise, np.asarray(cruise().sample(steps * 43.2, seed=0).zk)

    # The gravitational-wave polarization of notebook 05: a chirp from 1 to 2 Hz
    # at 100 Hz with the amplitude and both angles moving, sigma_n = 0.3.
    from modules.apps.gravitational.gravwave import PolarizationModel, jones_signal   # noqa: E402
    pt = np.arange(steps) / 100.0
    span = steps / 100.0
    a = 1.0 + 0.1 * np.cos(2 * np.pi * 0.2 * pt)
    psi = 2 * np.pi * (pt + 0.5 * pt ** 2 / span)
    theta = (np.pi / 2) * (1.0 + 0.5 * np.sin(2 * np.pi * 0.05 * pt))
    phi = np.pi * (1.0 + 0.5 * np.sin(2 * np.pi * 0.10 * pt + 1.0))
    jones = np.array([jones_signal(a[k], psi[k], theta[k], phi[k]) for k in range(steps)])
    jones = jones + 0.3 * (rng.standard_normal(jones.shape) + 1j * rng.standard_normal(jones.shape))

    def polarization():
        model = PolarizationModel("GW", fs=100.0, delta=0.5, sigma_a=0.05, sigma_psi=0.6,
                                  sigma_theta=0.05, sigma_phi=0.05, sigma_n=0.3)
        model.initial_state([0.0, 0.0, -2 * np.pi / 100.0, 2 * np.pi, theta[0], 0.0, phi[0], 0.0])
        return model
    yield ("PolarizationModel, chirp", polarization,
           np.column_stack([jones[:, 0].real, jones[:, 0].imag, jones[:, 1].real, jones[:, 1].imag]))


def composite_cases(steps):
    """The composites of notebook 02 and the K1 catalogue: models of matrices,
    with no twin. ``steps`` is the number of filter steps (decimated samples)."""
    from modules.apps.resonators.oscillators import (OscillatorModel, OscillatorBankModel,   # noqa: E402
                                                     InharmonicOscillatorModel, partial_frequencies)
    from modules.apps.gravitational.interferometer import InterferometerK1                   # noqa: E402
    fs = 4096.0
    rng = np.random.default_rng(0)

    def lines(freqs, amps, Qs, noise, rate):
        N = int(steps / rate * fs)
        t = np.arange(N) / fs
        x = noise * rng.standard_normal(N)
        for amp, f, q in zip(amps, freqs, Qs):
            x += amp * np.exp(-t / (q / (np.pi * f))) * np.cos(2 * np.pi * f * t)
        return x

    yield ("OscillatorModel, one line",
           lambda: OscillatorModel("m", [180.1], fc=180.1, dfc=5.0, Q=1.3e4, fs=fs,
                                   measurement_uncertainty=0.1, process_uncertainty=1e-4),
           lines([180.1], [1.0], [1.3e4], 0.1, 20.0))
    yield ("OscillatorModel, three lines",
           lambda: OscillatorModel("c", [178.0, 180.1, 182.5], fc=180.0, dfc=8.0, Q=[1.3e4] * 3, fs=fs,
                                   measurement_uncertainty=0.1, process_uncertainty=1e-4),
           lines([178.0, 180.1, 182.5], [1.0, 0.6, 0.4], [1.3e4] * 3, 0.1, 32.0))
    fr = partial_frequencies(170.0, 0.13, 3)
    yield ("OscillatorBankModel, 3 windows",
           lambda: OscillatorBankModel(*[InharmonicOscillatorModel(
               f"ih{k}", f0=170.0, fc=fk, dfc=8.0, n_harmonics=3, B=0.13, Q=q, fs=fs,
               measurement_uncertainty=0.05, process_uncertainty=1e-5)
               for k, (fk, q) in enumerate(zip(fr, [1.3e4, 1.8e4, 2.1e4]))]),
           lines(fr, [1.0, 0.4, 0.15], [1.3e4, 1.8e4, 2.1e4], 0.05, 32.0))
    k1 = InterferometerK1.from_peaks("O3GK")
    peaks = [f for w in k1.windows for f in w.frequencies][::20]
    yield ("InterferometerK1, 88 modes", lambda: InterferometerK1.from_peaks("O3GK"),
           lines(peaks, [1.0] * len(peaks), [1e5] * len(peaks), 0.1, 4 * k1.windows[0].dfc))


def compare_composite(make, z, repeat):
    """The Python loop, the old default (the extended engine: the C++ loop calling
    the Python closures) and the C++ linear pass a model of matrices takes now."""
    from unittest import mock
    from modules.framework.filter import KalmanFilter
    from modules.framework.constants import KalmanEngineKind

    def before():
        model = make()
        model.DEFAULT_ENGINE = KalmanEngineKind.EXTENDED
        return model
    with native.python_reference(), mock.patch.object(KalmanFilter, "_native_forward_eligible", return_value=False):
        t_py, x_py, P_py, _ = timed(make, z, repeat)
    t_loop, x_loop, _, _ = timed(before, z, repeat)
    with mock.patch.object(KalmanFilter, "_forward_native", autospec=True,
                           side_effect=KalmanFilter._forward_native) as fused:
        t_cx, x_cx, P_cx, model = timed(make, z, repeat)
    scale = max(1.0, np.nanmax(np.abs(x_py)))
    dx = np.nanmax(np.abs(x_py - x_cx)) / scale
    dP = np.nanmax(np.abs(P_py - P_cx)) / max(1.0, np.nanmax(np.abs(P_py)))
    return t_py, t_loop, t_cx, dx, dP, fused.call_count == repeat, model.n, len(x_cx)


# --- the field models: the whole ensemble pass ----------------------------------
def field_run(make, mode, repeat, run):
    """Best time of ``run(model)`` (the model made outside the timing) in one mode:
    the Python reference, the C++ solver and kernel in the Python loop (the pass
    before the twin), or the twin."""
    def once():
        model = make()
        start = time.perf_counter()
        out = run(model)
        return time.perf_counter() - start, out
    best, out = float("inf"), None
    for _ in range(repeat):
        if mode == "reference":
            with native.python_reference():
                dt, out = once()
        elif mode == "loop":
            native.USE_NATIVE_TWINS = False
            try:
                dt, out = once()
            finally:
                native.USE_NATIVE_TWINS = True
        else:
            dt, out = once()
        best = min(best, dt)
    return best * 1e3, out


def field_cases(notebook):
    from modules.apps.weather.flow import KolmogorovFlowModel, TwinExperiment
    from modules.apps.weather.barotropic import BarotropicFlowModel
    from modules.framework.filters import EnsembleKalmanFilter

    def step5(obs_type="vorticity", grid=16):
        # notebook 13, step 5: 16 x 16, 40 probes, 25 cycles, 16 members
        model = KolmogorovFlowModel("mini", grid=grid, cycle=10, dt=0.01, nu=1 / 40., n_obs=40, sig_o=0.30,
                                    loc_radius=4.0, seed=7, obs_type=obs_type)
        model.climatology(n_snapshots=60, spinup=600, stride=5)
        return model

    def measurements(model, steps=25):
        rng = np.random.default_rng(21)
        w, zs = model.equilibrium_state(decorrelate=100), []
        for _ in range(steps):
            w = model.f(w)
            zs.append(np.asarray(model.h(w)).ravel() + rng.normal(0, model.sig_o, model.m))
        return np.asarray(zs)

    def enkf(z):
        def run(model):
            f = EnsembleKalmanFilter(model, n_members=16, inflation=1.05, seed=0)
            record = f.forward(z)
            return np.asarray(record.xk), np.asarray(record.Pk)
        return run

    z = measurements(step5())
    yield "KolmogorovFlowModel, EnKF 16x16", step5, enkf(z)
    zg = measurements(step5("gradient"))
    yield "KolmogorovFlowModel, gradient probes", lambda: step5("gradient"), enkf(zg)

    def barotropic():
        model = BarotropicFlowModel("b", grid=16, cycle=6, n_obs=30, sig_o=1e-5, seed=7)
        model.climatology(n_snapshots=40, spinup=60, stride=3)
        return model
    yield "BarotropicFlowModel, EnKF 16x16", barotropic, enkf(measurements(barotropic()))

    if notebook:
        # notebook 13, step 3: 64 x 64, 100 probes, 40 cycles, 16 members, an EnKF and a free run
        def nb13():
            model = KolmogorovFlowModel("probes", grid=64, nu=1 / 150., chi=2.0, kf=4, dt=0.005, cycle=30,
                                        n_obs=100, sig_o=0.30, loc_radius=8.0, seed=7)
            model.climatology(n_snapshots=150, spinup=2500, stride=10)
            return model

        def osse(model):
            run = TwinExperiment(model, n_cycles=40, n_members=16, seed_obs=999, seed_setup=123, obs_periods=(2, 6))
            run.add_method("enkf", inflation=1.06)
            run.add_free_run()
            hist = run.run(verbose=False)
            return (np.asarray([hist["enkf"]["rmse_a"], hist["free"]["rmse_a"]]),
                    np.asarray([hist["enkf"]["spread"], hist["free"]["spread"]]))
        yield "TwinExperiment 64x64 (notebook 13)", nb13, osse


def fields(repeat, notebook):
    print(f"\n{'field model, ensemble pass':36s} {'Python':>10s} {'C++ loop':>10s} {'C++ twin':>10s} "
          f"{'faster':>8s} {'mean':>9s} {'spread/P':>9s}")
    for name, make, run in field_cases(notebook):
        times, outs = {}, {}
        for mode in ("reference", "loop", "twin"):
            times[mode], outs[mode] = field_run(make, mode, 1 if "64x64" in name else repeat, run)
        rel = lambda a, b: float(np.nanmax(np.abs(a - b)) / np.nanmax(np.abs(b)))
        dx = max(rel(outs[m][0], outs["reference"][0]) for m in ("loop", "twin"))
        dP = max(rel(outs[m][1], outs["reference"][1]) for m in ("loop", "twin"))
        print(f"{name:36s} {times['reference']:8.1f} ms {times['loop']:8.1f} ms {times['twin']:8.1f} ms "
              f"{times['loop'] / times['twin']:7.1f}x {dx:9.1e} {dP:9.1e}")
    print("\nPython: the reference (numpy solver, numpy analysis). C++ loop: the Python loop calling the C++")
    print("solver and analysis kernel (the pass before the twin). C++ twin: the whole pass in libKalmanApps.")
    print("faster: C++ loop against the twin. mean, spread/P: largest difference from the reference, relative.")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--steps", type=int, default=2000, help="measurements per run (default 2000)")
    parser.add_argument("--repeat", type=int, default=3, help="runs per case, best kept (default 3)")
    parser.add_argument("--only", default="", help="run the cases whose name contains this text")
    parser.add_argument("--composite-steps", type=int, default=500,
                        help="filter steps per composite run (default 500; the 88-mode K1 bank has n = 176)")
    parser.add_argument("--fields", action="store_true", help="the field models on the ensemble engine only")
    parser.add_argument("--notebook", action="store_true", help="with --fields: notebook 13's 64 x 64 run (minutes)")
    args = parser.parse_args()

    kalman.set_debug(0)
    if not native.apps_library_names():
        sys.exit("libKalmanApps was not found: build and install it first (see the module docstring).")

    if args.fields:
        fields(args.repeat, args.notebook)
        return
    print(f"{args.steps} steps per run, best of {args.repeat}\n")
    print(f"{'model':32s} {'Python':>10s} {'C++ loop':>10s} {'C++ model':>10s} {'faster':>8s} {'state':>9s} {'covariance':>11s}")
    for name, make, z, *uk in cases(args.steps):
        if args.only not in name:
            continue
        t_py, t_loop, t_cx, dx, dP, used = compare(make, z, args.repeat, *uk)
        twin = f"{t_cx:8.1f} ms {t_py / t_cx:7.1f}x" if used else f"{'none':>11s} {t_py / t_loop:7.1f}x"
        print(f"{name:32s} {t_py:8.1f} ms {t_loop:8.1f} ms {twin} {dx:9.1e} {dP:11.1e}")
    print(f"\nComposites (models of matrices, no twin), {args.composite_steps} filter steps, best of {args.repeat}\n")
    print(f"{'model':32s} {'n':>4s} {'Python':>10s} {'C++ loop':>10s} {'C++ linear':>10s} {'faster':>8s} {'state':>9s} {'covariance':>11s}")
    for name, make, z in composite_cases(args.composite_steps):
        if args.only not in name:
            continue
        t_py, t_loop, t_cx, dx, dP, used, n, N = compare_composite(make, z, args.repeat)
        fused = f"{t_cx:8.1f} ms {t_py / t_cx:7.1f}x" if used else f"{'none':>11s} {t_py / t_loop:7.1f}x"
        print(f"{name:32s} {n:4d} {t_py:8.1f} ms {t_loop:8.1f} ms {fused} {dx:9.1e} {dP:11.1e}")
    print("\nPython: the reference loop. C++ loop: the C++ filter calling the Python model.")
    print("C++ model: the model's C++ twin. faster: Python against the fastest of the two.")
    print("state / covariance: largest difference from the Python run, relative to the largest entry.")


if __name__ == "__main__":
    main()
