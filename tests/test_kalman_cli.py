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


def test_colours_only_when_asked_or_on_a_terminal():
    plain = _run("model", "car-speed")
    if plain.returncode != 0:
        pytest.skip("libKalmanApps is not beside libKalman")
    assert "\033[" not in plain.stdout                       # a pipe gets plain text
    coloured = _run("model", "car-speed", "--color", "always").stdout
    assert "\033[" in coloured
    import re
    assert re.sub(r"\033\[[0-9;]*m", "", coloured) == plain.stdout   # same text, same alignment
    table = _run("models", "--color=always").stdout
    assert re.sub(r"\033\[[0-9;]*m", "", table) == _run("models").stdout
    assert "\033[" not in _run("models", "--json", "--color", "always").stdout
    assert _run("models", "--color", "purple").returncode == 2


def _example_plugin():
    for d in ("var/install/lib", "var/build/lib"):
        for ext in (".dylib", ".so"):
            p = os.path.join(REPO, d, "libKalmanPlugin_example" + ext)
            if os.path.isfile(p):
                return p
    return None


def test_export_a_model_given_by_code_as_a_reference(tmp_path):
    out = _run("export", "car-speed", "--set", "mass=1200")
    if out.returncode != 0:
        pytest.skip("libKalmanApps is not beside libKalman")
    document = json.loads(out.stdout)
    assert document["format"] == "kalman-model" and document["kind"] == "reference"
    assert document["origin"]["plugin"] == "KalmanApps" and document["structure"] == "maps"
    spec = document["spec"]
    assert spec["base"] == "car-speed" and spec["dynamics"]["params"]["mass"] == 1200
    assert document["state"][1] == {"name": "speed", "unit": "m/s"}

    # one archive, readable by any zip tool, and by the program itself
    path = str(tmp_path / "car.kw")
    assert _run("export", "car-speed", "-o", path).returncode == 0
    import zipfile
    assert zipfile.ZipFile(path).namelist() == ["model.json"]
    shown = _run("model", path)
    assert shown.returncode == 0 and "reference" in shown.stdout and "continuous" in shown.stdout
    # without the plugin the reference cannot be built, and the message says which one
    missing = _run("--no-apps", "model", path)
    assert missing.returncode == 1 and "KalmanApps" in missing.stderr


def test_export_a_linearization_and_feed_it_to_the_api(tmp_path):
    path = str(tmp_path / "ballistic.json")
    if _run("export", "ballistic", "--linearize", "-o", path).returncode != 0:
        pytest.skip("libKalmanApps is not beside libKalman")
    import numpy as np
    import kalman.native as native
    from kalman import api

    read = native.read_model_file(path)
    document = read["document"]
    assert document["kind"] == "linearization" and document["structure"] == "matrices"
    assert document["linearized_at"]["x"] == [0.0] * 9 and read["tables"] == {}
    model = api.make_model(document["spec"])            # the spec is what the JSON API takes
    assert (model.n, model.m, model.fs) == (9, 3, 128.0)
    F = np.asarray(document["spec"]["dynamics"]["F"])
    assert abs(F[0, 1] - 1 / 128.0) < 1e-6              # dx = v dt
    # a file that is all matrices needs no plugin to be read back
    assert _run("--no-apps", "model", path).returncode == 0


def test_export_matrices_whole_and_round_trip(tmp_path):
    plugin = _example_plugin()
    if plugin is None:
        pytest.skip("the example plugin is not built (make examples)")
    first = _run("--no-apps", "--plugin", plugin, "export", "constant-velocity")
    assert first.returncode == 0, first.stderr
    document = json.loads(first.stdout)
    assert document["kind"] == "complete" and document["spec"]["base"] == "matrices"
    path = str(tmp_path / "cv.kw")
    assert _run("--no-apps", "--plugin", plugin, "export", "constant-velocity", "-o", path).returncode == 0
    again = _run("--no-apps", "export", path)           # read with no plugin, written again
    assert again.returncode == 0 and again.stdout == first.stdout


def test_export_from_python(tmp_path):
    import kalman.native as native
    from kalman.framework.native import core

    if not core.HAS_CXX_CORE:
        pytest.skip("libKalman is not built")
    try:
        native.load_plugin("KalmanApps")
    except Exception:
        pytest.skip("libKalmanApps is not beside libKalman")
    document = native.export_model("storm", parameters={"beta_drift": 3.0})
    assert document["kind"] == "reference" and document["spec"]["dynamics"]["params"]["beta_drift"] == 3.0
    path = str(tmp_path / "storm.kw")
    assert native.export_model("storm", path, linearize=True)["kind"] == "linearization"
    assert native.read_model_file(path)["document"]["spec"]["base"] == "matrices"
    with pytest.raises(ValueError):
        native.export_model("no-such-model")


def test_settings_file_beside_a_model_file(tmp_path):
    path = str(tmp_path / "car.kw")
    if _run("export", "car-speed", "-o", path).returncode != 0:
        pytest.skip("libKalmanApps is not beside libKalman")
    beside = path + ".json"

    # written from the model file, it changes nothing; it is the file to edit
    made = _run("settings", path)
    assert made.returncode == 0 and os.path.isfile(beside)
    settings = json.load(open(beside))
    assert settings["format"] == "kalman-settings" and settings["for"] == "car-speed"
    assert settings["parameters"]["mass"] == 1500 and settings["fs"] == 20
    sheet = json.loads(_run("model", path, "--json").stdout)
    assert sheet["settings"] == beside and sheet["settings_changed"] == []

    # moved away from the defaults: a parameter by --set, the noise by hand
    assert _run("settings", path, "--set", "mass=1200").returncode == 0
    settings = json.load(open(beside))
    settings["noise"]["R"] = [[4.0]]
    json.dump(settings, open(beside, "w"))
    sheet = json.loads(_run("model", path, "--json").stdout)
    assert sheet["parameters"]["mass"] == 1200 and sheet["R_diagonal"] == [4.0]
    assert sheet["settings_changed"] == ["mass = 1200", "R"]
    assert "changes mass = 1200, R" in _run("model", path).stdout
    # the model file itself is untouched, and can be opened without its settings
    plain = json.loads(_run("model", path, "--json", "--no-settings").stdout)
    assert plain["parameters"]["mass"] == 1500 and plain["R_diagonal"] == [0.25] and plain["settings"] is None

    import kalman.native as native
    read = native.read_model_file(path)
    assert read["settings"]["parameters"]["mass"] == 1200
    assert read["document"]["spec"]["dynamics"]["params"]["mass"] == 1500

    # settings cannot change what the model is, and another model's settings are refused
    json.dump({**settings, "dynamics": {"F": [[1]]}}, open(beside, "w"))
    refused = _run("model", path)
    assert refused.returncode == 1 and "is not a setting" in refused.stderr
    json.dump({**settings, "for": "storm"}, open(beside, "w"))
    refused = _run("model", path)
    assert refused.returncode == 1 and "written for the model 'storm'" in refused.stderr
    json.dump({**settings, "parameters": {"wings": 2}}, open(beside, "w"))
    assert _run("model", path).returncode == 1
    # a whole model is not written under the name of a settings file
    assert _run("export", "car-speed", "-o", beside).returncode == 1


def test_time_varying_model_through_the_program(tmp_path):
    plugin = _example_plugin()
    if plugin is None:
        pytest.skip("the example plugin is not built (make examples)")
    base = ("--no-apps", "--plugin", plugin)
    listed = {m["name"]: m for m in json.loads(_run(*base, "models", "--json").stdout)}
    assert listed["fading-sensor"]["structure"] == "time-varying matrices"
    refused = _run(*base, "export", "fading-sensor")
    assert refused.returncode == 1 and "how many samples" in refused.stderr
    path = str(tmp_path / "fading.kw")
    assert _run(*base, "export", "fading-sensor", "--steps", "50", "-o", path).returncode == 0
    assert _run(*base, "export", "fading-sensor", "--steps", "50").returncode == 1      # tables need a file
    import zipfile
    assert zipfile.ZipFile(path).namelist() == ["model.json", "tables/R.f64"]
    shown = _run("--no-apps", "model", path)                # read back with no plugin at all
    assert shown.returncode == 0 and "complete, with tables" in shown.stdout

    import numpy as np
    import kalman.native as native
    read = native.read_model_file(path)
    R = read["tables"]["R"]
    assert R.shape == (50, 1, 1) and abs(R[10, 0, 0] / R[0, 0, 0] - 1.05) < 1e-12     # +50 % per second at 100 Hz

    # opened in Python, the tables are used: against the registered model, sample by sample
    from kalman.framework.native import core
    if not core.HAS_CXX_CORE:
        return                                              # reading the file needed no library; opening does
    native.load_plugin(plugin)
    original, opened = native.NativeModel("fading-sensor"), native.open_model(path)
    assert opened.structure.name == "TIME_VARYING"
    z = np.cumsum(np.random.default_rng(3).normal(size=(50, 1)), axis=0)
    a, b = original.filter_native(z, engine="linear"), opened.filter(z)
    assert np.abs(np.asarray(a.xk).reshape(50, 2) - np.asarray(b.xk).reshape(50, 2)).max() < 1e-12


def test_python_models_export_and_open(tmp_path):
    import numpy as np
    import kalman.native as native
    from kalman.framework.native import core
    from kalman.framework.model import KalmanModel
    from kalman.framework.constants import KalmanStructureKind

    if not core.HAS_CXX_CORE:
        pytest.skip("libKalman is not built")
    dt = 0.1
    kw = dict(fs=10, F=np.array([[1, dt], [0, 1]]), H=np.array([[1.0, 0]]), Q=np.diag([1e-4, 1e-3]),
              x0=np.zeros((2, 1)), P0=np.eye(2))
    z = np.cumsum(np.random.default_rng(1).normal(size=(80, 1)), axis=0)

    def apart(a, b):
        return np.abs(np.asarray(a.filter(z).xk).reshape(80, 2) - np.asarray(b.filter(z).xk).reshape(80, 2)).max()

    # constant matrices: written whole
    model = KalmanModel("cv", R=np.array([[0.04]]), **kw)
    assert model.structure is KalmanStructureKind.MATRICES
    path = str(tmp_path / "cv.kw")
    document = model.export(path)
    assert document["kind"] == "complete" and document["origin"]["python"].endswith("KalmanModel")
    opened = native.open_model(path)
    assert opened.structure is KalmanStructureKind.MATRICES and apart(model, opened) == 0.0

    # a matrix that is a function of time: tables, used when opened
    varying = KalmanModel("tv", R=lambda t: np.array([[0.04 * (1 + (t or 0))]]), **kw)
    assert varying.structure is KalmanStructureKind.TIME_VARYING
    with pytest.raises(ValueError):
        varying.export()
    path = str(tmp_path / "tv.kw")
    assert list(varying.export(path, steps=80)["tables"]["entries"]) == ["R"]
    opened = native.open_model(path)
    assert apart(varying, opened) < 1e-12
    assert apart(model, opened) > 1e-3              # and it is not the constant model

    # settings beside it are applied on opening
    json.dump({"format": "kalman-settings", "for": "KalmanModel", "initial": {"x0": [5.0, 0.0]}},
              open(str(tmp_path / "cv.kw.json"), "w"))
    assert float(np.asarray(native.open_model(str(tmp_path / "cv.kw")).x0).reshape(-1)[0]) == 5.0
    assert float(np.asarray(native.open_model(str(tmp_path / "cv.kw"), settings=False).x0).reshape(-1)[0]) == 0.0

    # a model given by code: a reference to its class, or its Jacobians
    from kalman.apps.vehicles.lane import CarLaneModel
    lane = CarLaneModel("car")
    assert lane.structure is KalmanStructureKind.MAPS
    path = str(tmp_path / "lane.kw")
    document = lane.export(path)
    assert document["kind"] == "reference" and document["spec"]["base"] == "CarLaneModel"
    assert document["spec"]["dynamics"]["params"]["wheelbase"] == 2.7       # the constructor's arguments
    assert type(native.open_model(path)).__name__ == "CarLaneModel"
    refused = _run("model", path) if KALMAN else None
    if refused is not None:
        assert refused.returncode == 1 and "Python class" in refused.stderr
    tangent = lane.export(linearize=True)
    assert tangent["kind"] == "linearization" and np.asarray(tangent["spec"]["dynamics"]["F"]).shape == (3, 3)
    assert tangent["state"][0] == {"name": "lateral", "unit": "m"}


def test_every_python_catalogue_model_goes_through_a_file():
    """Each Python model that can be built with its defaults is written, read
    back and filters the same. A reference carries the constructor's arguments."""
    import inspect
    import warnings
    import numpy as np
    import kalman.native as native
    from kalman import api
    from kalman.framework.native import core

    if not core.HAS_CXX_CORE:
        pytest.skip("libKalman is not built")
    done = []
    for name, cls in sorted(api._model_catalog().items()):
        if hasattr(cls, "NATIVE_NAME"):
            continue
        first = list(inspect.signature(cls.__init__).parameters)[1:2]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = cls("s") if first in (["sensor_id"], ["name"]) else cls()
                structure = model.structure
        except Exception:
            continue                    # needs arguments: not this test's business
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if structure.name == "TIME_VARYING":
                continue                # tables need a file: covered by test_python_models_export_and_open
            document = model.export()
            assert document["kind"] == ("complete" if structure.name == "MATRICES" else "reference"), name
            back = api.make_model(document["spec"])
            if document["kind"] == "reference":
                assert type(back) is cls, name
                written = document["spec"]["dynamics"]["params"]
                left_out = set(document["origin"].get("parameters_not_written", []))
                for key, value in written.items():
                    again = back._init_arguments[key]
                    again = getattr(again, "_name_", again)
                    assert np.all(np.asarray(again) == np.asarray(value)), (name, key)
                assert not left_out, (name, left_out)
            assert np.allclose(np.asarray(back.x0, float), np.asarray(model.x0, float), equal_nan=True), name
            assert np.allclose(back.Q(), model.Q(), equal_nan=True) and np.allclose(back.R(), model.R()), name
        done.append(name)
    assert len(done) >= 15, done
    assert {"CarLaneModel", "RocketAscentModel", "InterplanetaryModel", "GPSModel"} <= set(done)


def test_reference_to_a_python_class_keeps_its_arguments(tmp_path):
    import numpy as np
    import kalman.native as native
    from kalman.framework.native import core
    from kalman.apps.vehicles.lane import CarLaneModel

    if not core.HAS_CXX_CORE:
        pytest.skip("libKalman is not built")
    lane = CarLaneModel("car", wheelbase=3.1, v0=22.0, q_lat=0.02)
    path = str(tmp_path / "lane.kw")
    document = lane.export(path)
    assert document["spec"]["dynamics"]["params"]["wheelbase"] == 3.1
    assert "parameters_not_written" not in document["origin"]
    back = native.open_model(path)
    assert back._init_arguments["v0"] == 22.0 and back._init_arguments["q_lat"] == 0.02
    np.random.seed(4)
    z = np.asarray(lane.sample(5.0).zk)
    assert np.abs(np.asarray(lane.filter(z).xk) - np.asarray(back.filter(z).xk)).max() == 0.0
