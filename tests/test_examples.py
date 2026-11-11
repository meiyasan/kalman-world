"""The toy demonstration runs, in C++ and in Python, and the two agree.

``examples/cxx/toy_plugin.cc`` defines one model; ``examples/cxx/toy_sensor.cc``
simulates a test sensor with it and filters the readings; and
``examples/python/toy_sensor.py`` filters the same readings with the same
compiled model from Python. Skipped unless the examples are built
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
        self.assertEqual(len(differences), 4, py.stdout)
        self.assertLess(max(differences[:2]), 1e-9, "Python filter on the C++ model")
        self.assertEqual(max(differences[2:]), 0.0, "the fused pass is the C++ program's own computation")


if __name__ == "__main__":
    unittest.main()
