"""The toy model from Python: written in C++ and written in Python, on the same readings.

The cooling-coffee model exists twice, stating the same physics:
``examples/cxx/toy_plugin.cc`` in C++ and ``examples/python/toy_model.py`` in
Python. ``examples/cxx/toy_sensor.cc`` simulates a test sensor with the C++
one, filters the readings and writes two CSV files. This script filters the
*same readings* three ways and compares each with what the C++ program wrote:

1. the model written in C++, loaded by name: ``model.filter(readings)`` hands
   the whole pass to C++, so this is the C++ program's own computation;
2. the model written in Python: the same C++ filter runs the pass and calls
   back into the Python ``f`` and ``h`` at each step;
3. the model written in Python with the C++ library left aside
   (``python_reference()``): the Python loop and the Python arithmetic, the
   reference the framework's tests compare against.

It prints how far apart they are, how long each took, and what the filter
learned.

    make build && make examples                    # builds the model and the C++ program
    ./var/build/bin/toy_sensor                     # writes toy_sensor.csv, toy_estimate.csv
    python examples/python/toy_sensor.py           # reads them from the current directory
    python examples/python/toy_sensor.py --plot    # also saves toy_sensor.png
"""
from __future__ import annotations
import argparse
import os
import sys

import numpy as np

try:
    import kalman
except ImportError:                                # running from a checkout that is not installed
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    import modules as kalman

import time as clock

native = sys.modules[kalman.__name__ + ".framework.native.core"]
_models = sys.modules[kalman.__name__ + ".framework.models.native"]
NativeModel, python_reference = _models.NativeModel, _models.python_reference

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from toy_model import CoolingCoffeeModel           # noqa: E402  the same model, written in Python


def load_csv(path):
    if not os.path.exists(path):
        sys.exit(f"{path} not found: run ./var/build/bin/toy_sensor first (it writes it).")
    return np.loadtxt(path, delimiter=",", skiprows=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("directory", nargs="?", default=".", help="where toy_sensor wrote its CSV files")
    parser.add_argument("--plot", action="store_true", help="save toy_sensor.png in that directory")
    args = parser.parse_args()
    kalman.set_debug(0)

    # --- the model: the compiled C++ class, found by name --------------------------
    native.load_plugin(native.resolve_plugin("KalmanPlugin_toy"))
    coffee = NativeModel("toy-coffee", "thermometer")
    coffee.set_parameter("room", 21.0)             # degC; the same named parameter as in C++
    print(f"model   : {coffee!r}, parameters {coffee.parameters()}")

    # --- the test sensor: the readings the C++ program simulated --------------------
    sensor = load_csv(os.path.join(args.directory, "toy_sensor.csv"))
    time, true_T, true_k, readings = sensor.T
    cxx = load_csv(os.path.join(args.directory, "toy_estimate.csv"))

    readings = readings.reshape(-1, 1)

    def timed(run, repeat=3):
        best = float("inf")                        # the best of a few runs: the first one loads and warms up
        for _ in range(repeat):
            start = clock.perf_counter()
            record = run()
            best = min(best, (clock.perf_counter() - start) * 1e3)
        return record, np.asarray(record.xk).reshape(len(readings), 2), best

    # --- 1. the model written in C++: the whole pass in C++ -------------------------
    record, python_x, ms_cxx = timed(lambda: coffee.filter(readings))

    # --- 2. the model written in Python, run by the C++ filter ----------------------
    written = CoolingCoffeeModel("thermometer", room=21.0)
    _, written_x, ms_written = timed(lambda: written.filter(readings))

    # --- 3. the model written in Python, with the Python loop -----------------------
    with python_reference():
        _, reference_x, ms_reference = timed(lambda: CoolingCoffeeModel("thermometer", room=21.0).filter(readings))

    # --- compare with what the C++ program computed ---------------------------------
    cxx_x = cxx[:, 1:3]
    scale = np.max(np.abs(cxx_x), axis=0)          # compare T and k each on its own scale
    readings = readings.reshape(-1)

    def apart(x):
        return (f"T {np.max(np.abs(x[:, 0] - cxx_x[:, 0])) / scale[0]:.1e}"
                f"   k {np.max(np.abs(x[:, 1] - cxx_x[:, 1])) / scale[1]:.1e}")

    print(f"sensor  : {len(readings)} readings, thermometer error "
          f"{np.sqrt(np.mean((readings - true_T) ** 2)):.3f} degC (RMS)")
    print(f"filter  : temperature error {np.sqrt(np.mean((python_x[:, 0] - true_T) ** 2)):.3f} degC (RMS)")
    sigma_k = float(np.sqrt(np.asarray(record.Pk)[-1, 1, 1]))
    print(f"learned : k = {python_x[-1, 1]:.6f} +/- {sigma_k:.6f} /s   (truth {true_k[-1]:.6f})")
    print("agreement with the C++ program (largest difference, relative to the largest value):")
    print(f"    model in C++                      : {apart(python_x)}   ({ms_cxx:.1f} ms)")
    print(f"    model in Python, C++ filter       : {apart(written_x)}   ({ms_written:.1f} ms)")
    print(f"    model in Python, Python filter    : {apart(reference_x)}   ({ms_reference:.1f} ms)")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        sigma_T = np.sqrt(np.asarray(record.Pk)[:, 0, 0])
        sig_k = np.sqrt(np.asarray(record.Pk)[:, 1, 1])
        fig, (top, bottom) = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)
        minutes = time / 60.0
        top.plot(minutes, readings, ".", color="0.7", ms=2, label="thermometer readings")
        top.plot(minutes, true_T, "k-", lw=1.5, label="true temperature")
        top.plot(minutes, python_x[:, 0], "-", color="tab:blue", lw=1.2, label="filter estimate")
        top.fill_between(minutes, python_x[:, 0] - 2 * sigma_T, python_x[:, 0] + 2 * sigma_T,
                         color="tab:blue", alpha=0.2, label="estimate ± 2σ")
        top.set_ylabel("temperature [°C]"); top.legend(fontsize=8); top.grid(alpha=0.3)
        top.set_title("A cooling cup of coffee: the model written in C++, filtered from Python")
        bottom.plot(minutes, true_k * 1e3, "k-", lw=1.5, label="true cooling rate")
        bottom.plot(minutes, python_x[:, 1] * 1e3, "-", color="tab:red", lw=1.2, label="learned by the filter")
        bottom.fill_between(minutes, (python_x[:, 1] - 2 * sig_k) * 1e3, (python_x[:, 1] + 2 * sig_k) * 1e3,
                            color="tab:red", alpha=0.2, label="estimate ± 2σ")
        bottom.set_xlabel("time [min]"); bottom.set_ylabel("cooling rate k [10⁻³ / s]")
        bottom.legend(fontsize=8); bottom.grid(alpha=0.3)
        fig.tight_layout()
        out = os.path.join(args.directory, "toy_sensor.png")
        fig.savefig(out, dpi=110)
        print(f"wrote   : {out}")


if __name__ == "__main__":
    main()
