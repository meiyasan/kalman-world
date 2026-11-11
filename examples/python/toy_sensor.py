"""The toy model in Python: the same C++ physics, on the same sensor readings.

The cooling-coffee model is written once, in C++ (``examples/cxx/toy_plugin.cc``).
``examples/cxx/toy_sensor.cc`` simulates a test sensor with it, filters the
readings and writes two CSV files. This script loads the *same compiled model*
and filters the *same readings* from Python, three ways:

1. ``model.filter(readings)`` — the ordinary Python workflow. The filter, the
   record and the plots are Python; every evaluation of the model's physics is
   a call into the C++ class.
2. ``model.filter_native(readings)`` — the whole pass in C++, in one call.
3. the estimate the C++ program wrote, read back from its CSV.

It prints how far apart the three are, and what the filter learned.

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

native = sys.modules[kalman.__name__ + ".framework.native.core"]
NativeModel = sys.modules[kalman.__name__ + ".framework.models.native"].NativeModel


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

    # --- 1. Python workflow, C++ physics ------------------------------------------
    record = coffee.filter(readings.reshape(-1, 1))
    python_x = np.asarray(record.xk)

    # --- 2. the whole pass in C++ -------------------------------------------------
    fused_x = np.asarray(coffee.filter_native(readings.reshape(-1, 1)).xk)

    # --- 3. compare with what the C++ program computed -----------------------------
    cxx_x = cxx[:, 1:3]
    scale = np.max(np.abs(cxx_x), axis=0)          # compare T and k each on its own scale
    print(f"sensor  : {len(readings)} readings, thermometer error "
          f"{np.sqrt(np.mean((readings - true_T) ** 2)):.3f} degC (RMS)")
    print(f"filter  : temperature error {np.sqrt(np.mean((python_x[:, 0] - true_T) ** 2)):.3f} degC (RMS)")
    sigma_k = float(np.sqrt(np.asarray(record.Pk)[-1, 1, 1]))
    print(f"learned : k = {python_x[-1, 1]:.6f} +/- {sigma_k:.6f} /s   (truth {true_k[-1]:.6f})")
    print("agreement with the C++ program (largest difference, relative to the largest value):")
    print(f"    Python filter, C++ model : T {np.max(np.abs(python_x[:, 0] - cxx_x[:, 0])) / scale[0]:.1e}"
          f"   k {np.max(np.abs(python_x[:, 1] - cxx_x[:, 1])) / scale[1]:.1e}")
    print(f"    fused C++ pass           : T {np.max(np.abs(fused_x[:, 0] - cxx_x[:, 0])) / scale[0]:.1e}"
          f"   k {np.max(np.abs(fused_x[:, 1] - cxx_x[:, 1])) / scale[1]:.1e}")

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
        top.set_title("A cooling cup of coffee: one C++ model, filtered from Python")
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
