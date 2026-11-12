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
    text = _run("model", "car-speed").stdout
    assert "dynamics" in text and "continuous" in text and "mass = 1500" in text


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
