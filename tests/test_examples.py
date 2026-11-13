"""The toy demonstration runs, in C++ and in Python, and the two agree.

``examples/cxx/toy_plugin.cc`` defines one model in C++ and
``examples/python/toy_model.py`` the same model in Python;
``examples/cxx/toy_sensor.cc`` simulates a test sensor with the C++ one and
filters the readings; and ``examples/python/toy_sensor.py`` filters the same
readings with both from Python. Skipped unless the examples are built
(``make build && make examples``).
"""
import os
import re
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, ".")

from modules.framework.native import core

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROGRAM = os.path.join(REPO, "var", "build", "bin", "toy_sensor")
SCRIPT = os.path.join(REPO, "examples", "python", "toy_sensor.py")


def toy_plugin():
    if not core.HAS_CXX_CORE:
        return None
    try:
        return core.resolve_plugin("KalmanPlugin_toy")
    except (FileNotFoundError, PermissionError):
        return None


@unittest.skipUnless(os.path.exists(PROGRAM) and toy_plugin(), "the examples are not built (make examples)")
class TestToyDemonstration(unittest.TestCase):

    def test_cxx_and_python_filter_the_same_readings_alike(self):
        with tempfile.TemporaryDirectory() as out:
            cxx = subprocess.run([PROGRAM, out, toy_plugin()], capture_output=True, text=True, timeout=120)
            self.assertEqual(cxx.returncode, 0, cxx.stderr)
            self.assertTrue(os.path.exists(os.path.join(out, "toy_sensor.csv")))
            self.assertTrue(os.path.exists(os.path.join(out, "toy_estimate.csv")))

            env = dict(os.environ, KALMAN_PLUGIN_PATH=os.path.dirname(toy_plugin()))
            py = subprocess.run([sys.executable, SCRIPT, out], capture_output=True, text=True,
                                timeout=300, cwd=REPO, env=env)
            self.assertEqual(py.returncode, 0, py.stderr)

        # Both report what the filter learned; the Python lines also give the
        # distance to the C++ program's estimate.
        learned = re.search(r"learned : k = ([\d.]+) \+/- ([\d.]+) /s\s+\(truth ([\d.]+)", cxx.stdout)
        self.assertIsNotNone(learned, cxx.stdout)
        k, sigma, truth = map(float, learned.groups())
        self.assertLess(abs(k - truth), 3 * sigma, "the cooling rate is learned within 3 sigma")
        self.assertLess(abs(k - truth), 0.1 * truth, "and to better than 10%")
        self.assertIn(learned.group(0).split("(")[0].strip(), py.stdout, "Python reports the same estimate")

        differences = [float(x) for x in re.findall(r"[Tk] ([\d.]+e[-+]\d+)", py.stdout)]
        self.assertEqual(len(differences), 6, py.stdout)
        self.assertEqual(max(differences[:2]), 0.0, "the model written in C++: the C++ program's own computation")
        # The two models take their Jacobians by differences of f, whose last
        # digit differs between numpy and the C math library.
        self.assertLess(max(differences[2:4]), 1e-6, "the model written in Python, run by the C++ filter")
        self.assertLess(max(differences[4:]), 1e-6, "the model written in Python, run by the Python loop")

    def test_the_model_written_in_python_states_what_the_cxx_one_does(self):
        """One step, the measurement, the noise and the initial state of the two models."""
        import importlib.util
        import numpy as np
        from modules.framework.models.native import NativeModel
        spec = importlib.util.spec_from_file_location(
            "toy_model", os.path.join(REPO, "examples", "python", "toy_model.py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules["toy_model"] = module
        spec.loader.exec_module(module)
        core.load_plugin(toy_plugin())
        written = module.CoolingCoffeeModel("thermometer", room=18.0)
        compiled = NativeModel("toy-coffee", "thermometer")
        compiled.set_parameter("room", 18.0)
        x = np.array([[63.5], [2.5e-3]])
        np.testing.assert_allclose(written.f(x), compiled.f(x), rtol=1e-15)
        np.testing.assert_array_equal(written.h(x), compiled.h(x))
        for name in ("Q", "R"):
            np.testing.assert_array_equal(getattr(written, name)(), getattr(compiled, name)())
        np.testing.assert_array_equal(np.asarray(written.x0).ravel(), np.asarray(compiled.x0).ravel())
        np.testing.assert_array_equal(written.P0, compiled.P0)
        self.assertEqual((written.n, written.m, written.fs), (compiled.n, compiled.m, compiled.fs))


if __name__ == "__main__":
    unittest.main()
