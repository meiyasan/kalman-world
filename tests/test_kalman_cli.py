"""The ``kalman`` program: the models an installation can run, and what each declares.

Skipped when the program is not built. What it prints is compared with the
Python classes the same models sit behind.
"""
import json
import os
import subprocess

import pytest

from kalman.framework.constants import KalmanDynamicsKind

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _binary():
    for d in ("var/install/bin", "var/build/bin"):
        p = os.path.join(REPO, d, "kalman")
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    return None


KALMAN = _binary()
pytestmark = pytest.mark.skipif(KALMAN is None, reason="the kalman program is not built (make build)")


def _run(*args):
    env = {k: v for k, v in os.environ.items() if k != "KALMAN_PLUGIN_PATH"}
    return subprocess.run([KALMAN, *args], capture_output=True, text=True, env=env, timeout=60)


def _models():
    out = _run("models", "--json")
    assert out.returncode == 0, out.stderr
    return {m["name"]: m for m in json.loads(out.stdout)}


def test_lists_the_application_models():
    models = _models()
    if not models:
        pytest.skip("libKalmanApps is not beside libKalman")
    for name in ("ballistic", "car-speed", "storm"):
        assert name in models
    assert models["ballistic"]["n"] == 9 and models["ballistic"]["m"] == 3
    assert not any("error" in m for m in models.values())


def test_dynamics_agree_with_the_python_classes():
    from kalman.apps.ballistics.extended import BallisticModel
    from kalman.apps.vehicles.speed import CarSpeedModel
    from kalman.apps.weather.storms import StormModel

    models = _models()
    if not models:
        pytest.skip("libKalmanApps is not beside libKalman")
    for name, cls in (("ballistic", BallisticModel), ("car-speed", CarSpeedModel), ("storm", StormModel)):
        expected = "continuous" if cls.DYNAMICS == KalmanDynamicsKind.CONTINUOUS else "discrete"
        assert models[name]["dynamics"] == expected, name
    assert models["ballistic"]["integrator"] == "RK4"
    assert models["storm"]["integrator"] is None


def test_one_model_in_full():
    out = _run("model", "car-speed", "--json")
    if out.returncode != 0:
        pytest.skip("libKalmanApps is not beside libKalman")
    sheet = json.loads(out.stdout)
    assert sheet["fs"] == 20 and sheet["process_noise"] == "additive"
    assert sheet["Q_diagonal"] == [1e-3, 5e-2] and sheet["R_diagonal"] == [0.25]
    assert sheet["parameters"]["mass"] == 1500
    assert sheet["state"] == [{"name": "distance", "unit": "m"}, {"name": "speed", "unit": "m/s"}]
    assert sheet["measurement"] == [{"name": "distance", "unit": "m"}]
    assert sheet["dt"] == 0.05
    text = _run("model", "car-speed").stdout
    assert "continuous" in text and "20 Hz, one sample every 50 ms" in text
    assert "state vector, n = 2" in text and "measurement, m = 1" in text
    assert "[1]  speed     m/s" in text
    assert not any(line != line.rstrip() for line in text.split("\n"))


def test_components_through_the_c_interface():
    from kalman.framework.native import core
    import kalman.native as native

    if not core.HAS_CXX_CORE:
        pytest.skip("libKalman is not built")
    try:
        native.load_plugin("KalmanApps")
    except Exception:
        pytest.skip("libKalmanApps is not beside libKalman")
    ffi, lib = core.ffi, core.lib
    model = lib.kalman_model_create(b"ballistic")
    assert model != ffi.NULL
    try:
        names = [ffi.string(lib.kalman_model_state_name(model, i)).decode() for i in range(9)]
        units = [ffi.string(lib.kalman_model_state_unit(model, i)).decode() for i in range(9)]
        assert names == ["x", "vx", "ax", "y", "vy", "ay", "z", "vz", "az"]
        assert units == ["m", "m/s", "m/s^2"] * 3
        assert [ffi.string(lib.kalman_model_measurement_name(model, i)).decode() for i in range(3)] == ["x", "y", "z"]
        assert ffi.string(lib.kalman_model_measurement_unit(model, 0)) == b"m"
        assert lib.kalman_model_state_name(model, 9) == ffi.NULL
    finally:
        lib.kalman_model_destroy(model)


def test_filter_table_and_errors():
    out = _run("models", "ballistic")
    if out.returncode != 0:
        pytest.skip("libKalmanApps is not beside libKalman")
    lines = out.stdout.strip().split("\n")
    assert lines[0].startswith("MODEL") and len(lines) == 3
    assert _run("model", "no-such-model").returncode == 1
    assert _run("--no-apps", "models").returncode == 1
    assert _run("frobnicate").returncode == 2
    assert _run("version").stdout.startswith("kalman ")
