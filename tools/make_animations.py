"""Precompute the animations the documentation shows.

Each animation is a real run of the framework: a model is sampled, the
measurements are filtered, and the result is drawn frame by frame and saved as
a GIF beside the page that shows it (``docs/images`` for the guide,
``docs/apps/images`` for an application page). A GIF plays everywhere a picture does — on the
repository's front page, in the guide pages of the site, in a slide — which the
interactive players saved in the notebooks cannot (a notebook viewed on GitLab
or GitHub is shown without JavaScript).

    python tools/make_animations.py                 # all of them
    python tools/make_animations.py tracking storm  # only those
    python tools/make_animations.py --list

The runs are seeded, so the files only change when the models do. Commit the
GIFs: the documentation build does not run this script.
"""
from __future__ import annotations
import os
import sys
import warnings

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
warnings.filterwarnings("ignore")

import matplotlib                                              # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter   # noqa: E402

import modules as kalman                                       # noqa: E402

GUIDE = os.path.join(REPO, "docs", "images")              # shown by the guide pages and the README
APPS = os.path.join(REPO, "docs", "apps", "images")       # shown by an application's own page
BLUE, RED, GREY = "#1779c4", "#d1495b", "0.65"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "figure.facecolor": "white"})


def save(fig, draw, frames, name, fps=12, hold=18, where=GUIDE):
    """Write ``<where>/<name>.gif``; the last frame is held so the result can be read."""
    frames = list(frames) + [list(frames)[-1]] * hold
    os.makedirs(where, exist_ok=True)
    path = os.path.join(where, name + ".gif")
    FuncAnimation(fig, draw, frames=frames, blit=False).save(path, writer=PillowWriter(fps=fps), dpi=72)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
def tracking():
    """A constant-velocity filter on noisy position readings: the estimate and
    its 2-sigma band, as the readings arrive."""
    from modules.framework.model import KalmanModel
    np.random.seed(4)
    fs, n = 10.0, 150
    dt = 1.0 / fs
    q = 0.6                                                    # m/s^1.5: how fast the velocity wanders
    make = lambda: KalmanModel(
        "cart", F=np.array([[1.0, dt], [0.0, 1.0]]), H=np.array([[1.0, 0.0]]),
        Q=q**2 * np.array([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]]), R=np.array([[1.5**2]]),
        fs=fs, x0=np.array([[0.0], [2.0]]), P0=np.diag([4.0, 4.0]))
    sample = make().sample(n / fs)
    record = make().filter(np.asarray(sample.zk))
    t = np.arange(n) / fs
    truth, z = np.asarray(sample.xk), np.asarray(sample.zk).ravel()
    x, P = np.asarray(record.xk), np.asarray(record.Pk)
    sig_p, sig_v = np.sqrt(P[:, 0, 0]), np.sqrt(P[:, 1, 1])

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(8.4, 5.6), sharex=True,
                                      gridspec_kw={"height_ratios": [1.6, 1]})

    def draw(k):
        for ax in (top, bottom):
            ax.clear()
        s = slice(0, k + 1)
        top.plot(t[s], z[s], ".", color=GREY, ms=5, label="sensor readings (σ = 1.5 m)")
        top.plot(t[s], truth[s, 0], "k-", lw=1.4, label="true position")
        top.plot(t[s], x[s, 0], "-", color=BLUE, lw=1.8, label="filter estimate")
        top.fill_between(t[s], x[s, 0] - 2 * sig_p[s], x[s, 0] + 2 * sig_p[s], color=BLUE, alpha=0.18,
                         label="estimate ± 2σ")
        top.set_xlim(0, t[-1]); top.set_ylim(truth[:, 0].min() - 4, truth[:, 0].max() + 4)
        top.set_ylabel("position [m]"); top.legend(loc="upper left", fontsize=8, ncol=2, frameon=False)
        top.set_title("A Kalman filter turns noisy readings into a sharp estimate", fontsize=11)
        bottom.plot(t[s], truth[s, 1], "k-", lw=1.4, label="true velocity (never measured)")
        bottom.plot(t[s], x[s, 1], "-", color=RED, lw=1.8, label="filter estimate")
        bottom.fill_between(t[s], x[s, 1] - 2 * sig_v[s], x[s, 1] + 2 * sig_v[s], color=RED, alpha=0.18)
        bottom.set_xlim(0, t[-1]); bottom.set_ylim(truth[:, 1].min() - 3, truth[:, 1].max() + 3)
        bottom.set_xlabel("time [s]"); bottom.set_ylabel("velocity [m/s]")
        bottom.legend(loc="upper left", fontsize=8, ncol=2, frameon=False)
        fig.tight_layout()

    return save(fig, draw, range(2, n, 2), "tracking")


# ---------------------------------------------------------------------------
def ballistics():
    """A shell tracked by radar: the filter follows it and predicts where it lands."""
    from modules.apps.ballistics import BallisticModel
    np.random.seed(2)
    fs, flight = 16.0, 28.0                                   # a little longer than the 26 s flight
    def make():
        m = BallisticModel("shell", fs=fs, process_uncertainty=1.0, measurement_uncertainty=10.0)
        m.initial_state([0, 60, 0, 0, 0, 0, 0, 128, 0])
        return m
    sample = make().sample(flight)
    truth, z = np.asarray(sample.xk), np.asarray(sample.zk)
    record = make().filter(z)
    x, P = np.asarray(record.xk), np.asarray(record.Pk)
    n = len(x)
    below = np.flatnonzero(truth[:, 6] < 0)
    if not len(below):
        raise RuntimeError("the simulated flight ended before the shell landed; lengthen it.")
    last = int(below[0])                                      # first sample under the ground
    a, b = truth[last - 1], truth[last]                       # the true impact, between the two samples
    true_impact = a[0] + (b[0] - a[0]) * a[6] / (a[6] - b[6])
    g = 9.81

    def impact(k):                                            # where the current estimate says it lands
        px, vx, pz, vz = x[k, 0], x[k, 1], x[k, 6], x[k, 7]
        tau = (vz + np.sqrt(max(vz**2 + 2 * g * max(pz, 0.0), 0.0))) / g
        return px + vx * tau

    fig, ax = plt.subplots(figsize=(8.4, 4.4))

    def draw(k):
        ax.clear()
        s = slice(0, k + 1)
        ax.plot(z[s, 0], z[s, 2], ".", color=GREY, ms=5, label="radar fixes (σ = 10 m)")
        ax.plot(truth[:last, 0], truth[:last, 6], "k--", lw=0.8, alpha=0.5)
        ax.plot(truth[s, 0], truth[s, 6], "k-", lw=1.4, label="true path")
        ax.plot(x[s, 0], x[s, 6], "-", color=BLUE, lw=1.8, label="filter estimate")
        xi = impact(k)
        ax.plot([xi], [0], "v", color=RED, ms=11, label=f"predicted impact: {xi:.0f} m")
        ax.plot([true_impact], [0], "x", color="k", ms=9, mew=2, label=f"true impact: {true_impact:.0f} m")
        ax.set_xlim(-40, truth[:last, 0].max() + 150); ax.set_ylim(-30, truth[:last, 6].max() + 110)
        ax.set_xlabel("range [m]"); ax.set_ylabel("altitude [m]")
        ax.set_title(f"Tracking a shell and predicting its impact  (t = {k / fs:4.1f} s)", fontsize=11)
        ax.legend(loc="upper left", fontsize=8, ncol=2, frameon=False)
        fig.tight_layout()

    return save(fig, draw, list(range(4, last, 5)) + [last - 1], "ballistics", where=APPS)


# ---------------------------------------------------------------------------
def storm():
    """A recurving tropical cyclone: the IKF follows the track and learns the regime."""
    from modules.apps.weather.storms import StormModel, synthetic_track
    from modules.framework.filters import InteractiveKalmanFilter
    truth, fixes = synthetic_track(n_steps=40, recurve_at=18, fix_noise=0.3, seed=3)
    models, Pi, probs = StormModel.ikf_bank(fix_noise=0.3, stall=True)
    x0 = np.array([[fixes[0, 0]], [fixes[0, 1]], [-20.0], [6.0], [fixes[0, 2]]])
    for m in models:
        m.initial_state(x0)
    ikf = InteractiveKalmanFilter(models, Pi, initial_probs=probs)
    record = ikf.forward(fixes)
    x = np.asarray(record.xk)
    mu = np.asarray(ikf.mode_probabilities)
    n = len(x)
    hours = np.arange(n) * 6.0
    names = ["straight", "recurving", "stalled"]
    colours = [BLUE, RED, "#edae49"]

    fig, (left, right) = plt.subplots(1, 2, figsize=(9.6, 4.4), gridspec_kw={"width_ratios": [1.25, 1]})

    def draw(k):
        left.clear(); right.clear()
        s = slice(0, k + 1)
        left.plot(truth[:, 0], truth[:, 1], "k--", lw=0.8, alpha=0.45)
        left.plot(fixes[s, 0], fixes[s, 1], "x", color=GREY, ms=6, label="satellite fixes (σ = 0.3°)")
        left.plot(truth[s, 0], truth[s, 1], "k-", lw=1.4, label="true track")
        left.plot(x[s, 0], x[s, 1], "-", color=BLUE, lw=1.9, label="IKF estimate")
        left.plot(x[k, 0], x[k, 1], "o", color=BLUE, ms=8)
        left.set_xlim(truth[:, 0].min() - 3, truth[:, 0].max() + 3)
        left.set_ylim(truth[:, 1].min() - 3, truth[:, 1].max() + 3)
        left.set_xlabel("longitude [°E]"); left.set_ylabel("latitude [°N]")
        left.set_title(f"Storm track  (t = {hours[k]:.0f} h)", fontsize=11)
        left.legend(loc="lower left", fontsize=8, frameon=False)
        for j, (name, colour) in enumerate(zip(names, colours)):
            right.plot(hours[s], mu[s, j], "-", color=colour, lw=1.9, label=name)
        right.axvline(18 * 6.0, color="k", ls=":", lw=1)
        right.text(18 * 6.0 + 3, 1.02, "storm starts to recurve", fontsize=8, va="bottom")
        right.set_xlim(0, hours[-1]); right.set_ylim(0, 1.12)
        right.set_xlabel("time [h]"); right.set_ylabel("probability of each motion regime")
        right.set_title("Which regime is the storm in?", fontsize=11)
        right.legend(loc="center left", fontsize=8, frameon=False)
        fig.tight_layout()

    return save(fig, draw, range(1, n), "storm", fps=7, hold=12, where=APPS)


# ---------------------------------------------------------------------------
def coffee():
    """The toy of docs/16: learning how fast a cup of coffee cools."""
    from modules.framework.models.extended import ExtendedKalmanModel

    class Coffee(ExtendedKalmanModel):
        def __init__(self, room=21.0, **kw):
            self.room, self.n, self.m, self.l, self.dtype = room, 2, 1, 0, np.float64
            super().__init__("cup", Q=np.diag([1e-4, 1e-12]), R=np.array([[0.25]]), L=np.eye(2), fs=1.0, **kw)

        def f(self, x, u=None, w=None, t=None):
            x = np.asarray(x, float).reshape(-1)
            return np.array([[self.room + (x[0] - self.room) * np.exp(-x[1])], [x[1]]])

        def h(self, x, v=None, t=None):
            return np.asarray(x, float).reshape(-1)[:1].reshape(1, 1)

    np.random.seed(7)
    n = 1500
    truth_model = Coffee(x0=np.array([[90.0], [2e-3]]), P0=1e-12 * np.eye(2))
    sample = truth_model.sample(n)
    truth, z = np.asarray(sample.xk), np.asarray(sample.zk).ravel()
    record = Coffee(x0=np.array([[80.0], [1e-3]]), P0=np.diag([25.0, 1e-6])).filter(z.reshape(-1, 1))
    x, P = np.asarray(record.xk), np.asarray(record.Pk)
    minutes = np.arange(n) / 60.0
    sig_k = np.sqrt(P[:, 1, 1])

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(8.4, 5.6), sharex=True)

    def draw(k):
        top.clear(); bottom.clear()
        s = slice(0, k + 1)
        top.plot(minutes[s], z[s], ".", color=GREY, ms=3, label="thermometer (σ = 0.5 °C)")
        top.plot(minutes[s], x[s, 0], "-", color=BLUE, lw=1.8, label="filter estimate")
        top.set_xlim(0, minutes[-1]); top.set_ylim(20, 95); top.set_ylabel("temperature [°C]")
        top.set_title("Learning how fast a cup of coffee cools, from a thermometer alone", fontsize=11)
        top.legend(loc="upper right", fontsize=8, frameon=False)
        bottom.axhline(truth[0, 1] * 1e3, color="k", lw=1.2, label="true cooling rate")
        bottom.plot(minutes[s], x[s, 1] * 1e3, "-", color=RED, lw=1.8, label="learned by the filter")
        bottom.fill_between(minutes[s], (x[s, 1] - 2 * sig_k[s]) * 1e3, (x[s, 1] + 2 * sig_k[s]) * 1e3,
                            color=RED, alpha=0.18, label="± 2σ")
        bottom.set_xlim(0, minutes[-1]); bottom.set_ylim(0, 3.2)
        bottom.set_xlabel("time [min]"); bottom.set_ylabel("cooling rate [10⁻³ / s]")
        bottom.legend(loc="lower right", fontsize=8, ncol=3, frameon=False)
        fig.tight_layout()

    frames = list(range(2, 120, 4)) + list(range(120, n, 30))
    return save(fig, draw, frames, "coffee", fps=12)


ANIMATIONS = {"tracking": tracking, "ballistics": ballistics, "storm": storm, "coffee": coffee}


def main():
    args = sys.argv[1:]
    if "--list" in args:
        for name, fn in ANIMATIONS.items():
            print(f"{name:11s} {fn.__doc__.split(chr(10))[0]}")
        return
    unknown = [a for a in args if a not in ANIMATIONS]
    if unknown:
        sys.exit(f"unknown animation(s) {unknown}; choose from {sorted(ANIMATIONS)}.")
    kalman.set_debug(0)
    for name in (args or ANIMATIONS):
        path = ANIMATIONS[name]()
        print(f"{name:11s} {os.path.getsize(path) / 1e6:5.2f} MB  {os.path.relpath(path, REPO)}")


if __name__ == "__main__":
    main()
