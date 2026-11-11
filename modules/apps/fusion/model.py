"""
Sensor fusion — one latent target, several asynchronous, differently-noisy sensors.

The essence of Kalman sensor fusion (cf. satya15july, *"Sensor Fusion with Kalman
Filter"*) is that **stacking measurements** in a single update optimally weights each
sensor by its inverse covariance, so the fused estimate beats *any* single sensor — and
sensors running at different rates fall out for free: a sensor that does not fire on a
given step is simply a ``NaN`` row, which the engine skips (predict-only for that
channel).

A :class:`SensorFusionModel` tracks a 1-D constant-velocity target ``x = [position,
velocity]`` observed by a **configurable list of** :class:`SensorSpec` (each declares what
it observes — ``"position"`` or ``"velocity"`` — its noise, and its rate). This single,
declarative model covers both demonstrated regimes:

  * **redundant** — several position sensors (the stereo-camera + lidar analog);
  * **complementary** — a slow noisy position sensor + a fast velocity sensor (the
    GPS + wheel-odometry analog).

Multi-rate measurements are built with :func:`kalman.helpers.make_multirate_input`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...helpers import make_multirate_input, rainbow, set_random_seed
from ...framework.constants import KalmanObservable, KalmanState
from ...framework.models.linear import LinearKalmanModel


# ===== State / observable enums (units) =====
class FusionState(KalmanState):
    POSITION = "m"
    VELOCITY = "m/s"


class FusionObservable(KalmanObservable):
    POSITION = "m"
    VELOCITY = "m/s"


@dataclass
class SensorSpec:
    """One measurement channel feeding the fused estimate.

    Parameters
    ----------
    name : str
        Human-readable sensor name (used in plot legends).
    observes : {"position", "velocity"}
        Which state the sensor reports.
    uncertainty : float
        Measurement noise std for this channel (so its ``R`` entry is ``uncertainty²``).
    rate : float | None
        Sampling rate [Hz]; ``None`` means the model's base rate ``fs``.
    """
    name: str
    observes: str = "position"
    uncertainty: float = 1.0
    rate: float | None = None

    def __post_init__(self):
        if self.observes not in ("position", "velocity"):
            raise ValueError(f"SensorSpec.observes must be 'position' or 'velocity', got {self.observes!r}")


class SensorFusionModel(LinearKalmanModel):
    """Constant-velocity 1-D target fused from a list of :class:`SensorSpec`.

    Parameters
    ----------
    sensor_id : str
        Channel / group name.
    sensors : list[SensorSpec]
        The sensors to fuse (stacked as rows of ``H``; ``R = diag(σ_i²)``).
    fs : float
        Base filter sample rate [Hz].
    process_uncertainty : float
        White-noise-acceleration std [m/s²/√Hz] driving the constant-velocity ``Q`` block.
    """

    NAME = "Sensor Fusion Model"
    DESCRIPTION = ("Constant-velocity 1-D target fused from a configurable list of sensors "
                   "(position/velocity, per-sensor noise and rate).")
    VERSION = "1.0"

    @property
    def states(self):
        return FusionState

    @property
    def observables(self):
        return FusionObservable

    def __init__(self, sensor_id: str | int,
                 sensors: list[SensorSpec],
                 fs: float = 50.0,
                 process_uncertainty: float = 1.0,
                 **kwargs):

        self.dtype = kwargs.pop("dtype", np.float64)
        if not sensors:
            raise ValueError("SensorFusionModel needs at least one SensorSpec.")
        self.sensors = list(sensors)

        # Continuous-time generator: ṗ=v, v̇=0 (constant velocity). Passed as
        # A= (not F=): the base KalmanModel discretizes exactly via expm(A/fs).
        A = np.array([[0.0, 1.0],
                      [0.0, 0.0]], dtype=self.dtype)

        # Stack one observation row per sensor.
        rows, rvar = [], []
        for s in self.sensors:
            rows.append([1.0, 0.0] if s.observes == "position" else [0.0, 1.0])
            rvar.append(self.dtype(s.uncertainty) ** 2)
        H = np.array(rows, dtype=self.dtype)
        R = np.diag(rvar).astype(self.dtype)

        # Continuous white-noise-acceleration disturbance driving v̇ (Lc picks
        # out that row); passed as Qc=/Lc= so the base model Van Loan-discretizes
        # it exactly alongside A=.
        Lc = np.array([[0.0], [1.0]], dtype=self.dtype)
        Qc = np.array([[self.dtype(process_uncertainty) ** 2]], dtype=self.dtype)

        self.m = len(self.sensors)
        if kwargs.get("P0") is None:
            kwargs["P0"] = np.diag([10.0, 10.0]).astype(self.dtype)

        super().__init__(sensor_id, A=A, H=H, Qc=Qc, Lc=Lc, R=R, fs=fs, **kwargs)

    def preprocess(self, zk, uk=None, **kwargs): return zk, uk
    def inprocess(self, state_vector, **kwargs): return state_vector
    def postprocess(self, xk_hat, **kwargs):     return xk_hat

    # ------------------------------------------------------------------
    # Multi-rate synthetic measurements from a sampled truth trajectory
    # ------------------------------------------------------------------
    def make_measurements(self, sample, seed=None) -> np.ndarray:
        """Build the ``NaN``-padded ``(N, n_sensors)`` measurement matrix from a truth record.

        Each sensor samples its channel of ``sample.xk`` at its own rate, adds its own
        noise, and the streams are interleaved at the base rate via
        :func:`make_multirate_input` (missing steps → ``NaN`` → predict-only).
        """
        if seed is not None:
            set_random_seed(seed)
        truth = np.asarray(sample.xk).reshape(len(sample), -1)
        N = truth.shape[0]
        base_fs = float(self.fs)
        duration = N / base_fs

        sensor_data = []
        for s in self.sensors:
            idx = 0 if s.observes == "position" else 1
            rate = float(s.rate or base_fs)
            # Sample the truth at this sensor's own firing times (robust to rates that
            # do not evenly divide base_fs); make_multirate_input re-derives the same
            # base-rate slots, so the streams interleave consistently.
            n_i = int(round(duration * rate))
            base_idx = np.round(np.arange(n_i) * base_fs / rate).astype(int)
            base_idx = base_idx[base_idx < N]
            clean = truth[base_idx, idx]
            noisy = clean + s.uncertainty * np.random.randn(base_idx.size)
            sensor_data.append((noisy.reshape(-1, 1), rate))

        zk = make_multirate_input(sensor_data, base_fs)
        # Pin the length to the truth length (guards off-by-one from fractional rates).
        if zk.shape[0] != N:
            fixed = np.full((N, zk.shape[1]), np.nan)
            n = min(N, zk.shape[0])
            fixed[:n] = zk[:n]
            zk = fixed
        return zk

    def single_sensor_record(self, zk, index: int):
        """Filter with only sensor ``index`` active (all other channels masked to ``NaN``).

        Yields the baseline each single sensor would achieve on its own — the reference the
        fused estimate is meant to beat.
        """
        zk = np.asarray(zk, dtype=float)
        masked = np.full_like(zk, np.nan)
        masked[:, index] = zk[:, index]
        return self.filter(masked, reset=True)

    # ==================================================================
    # Signature figure — model.plot(record, sample=sample, measurements=zk)
    # ==================================================================
    def _plot_static(self, record, sample=None, *, component=None, error=True,
                     measurements=None, sigma=1.0, figsize=(15, 9),
                     meas=None, colors=None, **kwargs):
        """Top: per-sensor scatter + fused position with ±σ + truth. Bottom: the fused
        position-σ compared against every single-sensor σ (fusion beats each one).

        ``meas`` is a dict of kwargs forwarded to ``Axes.scatter`` for measurement points
        (e.g. ``meas=dict(alpha=1.0, s=20, marker='x')``); a uniform ``color`` there
        overrides every sensor's color. ``colors`` overrides the default (pale) per-sensor
        palette outright — e.g. ``colors=rainbow(len(self.sensors), l=0.6)`` for saturated
        dots while keeping each sensor distinguishable."""
        import matplotlib.pyplot as plt

        t = np.asarray(record.time)
        xk = np.asarray(record.xk).reshape(len(record), -1)
        Pk = np.asarray(record.Pk)
        pos = xk[:, 0]
        pos_sig = np.sqrt(np.maximum(Pk[:, 0, 0], 0.0)) if Pk.size else np.zeros_like(pos)
        xt = np.asarray(sample.xk).reshape(len(sample), -1)[:, 0] if sample is not None else None
        colors = colors if colors is not None else rainbow(max(3, len(self.sensors)))

        fig, (axT, axB) = plt.subplots(2, 1, figsize=figsize,
                                       gridspec_kw=dict(height_ratios=[1.5, 1], hspace=0.3))

        # --- Top: trajectory + per-sensor scatter ---
        if measurements is not None:
            zk = np.asarray(measurements, dtype=float)
            for i, s in enumerate(self.sensors):
                if s.observes != "position":
                    continue
                col = zk[:, i]
                mask = ~np.isnan(col)
                scatter_kw = dict(s=12, alpha=0.45, color=colors[i])
                scatter_kw.update(meas or {})
                axT.scatter(t[mask], col[mask],
                            label=f"{s.name} (σ={s.uncertainty:g}, {s.rate or self.fs:g} Hz)",
                            **scatter_kw)
        if xt is not None:
            axT.plot(t, xt, color="gray", lw=1.6, ls="--", alpha=0.9, label="truth")
        axT.fill_between(t, pos - sigma * pos_sig, pos + sigma * pos_sig,
                         color="navy", alpha=0.2, label=rf"fused $\pm{sigma:g}\sigma$")
        axT.plot(t, pos, color="navy", lw=2.0, label="fused estimate")
        axT.set_xlabel("Time (s)"); axT.set_ylabel(f"position [{FusionState.POSITION.value}]")
        axT.legend(fontsize=8, ncol=2, loc="best"); axT.grid(True, alpha=0.3)
        axT.set_title(f"{self.name()} — fused position estimate")

        # --- Bottom: fusion beats each single sensor ---
        if measurements is not None:
            for i, s in enumerate(self.sensors):
                rec_i = self.single_sensor_record(np.asarray(measurements, dtype=float), i)
                sig_i = np.sqrt(np.maximum(np.asarray(rec_i.Pk)[:, 0, 0], 0.0))
                axB.plot(t, sig_i, color=colors[i], lw=1.3, alpha=0.8,
                         label=f"{s.name} only")
        axB.plot(t, pos_sig, color="navy", lw=2.2, label="fused")
        axB.set_xlabel("Time (s)"); axB.set_ylabel(r"position $\sigma$ [m]")
        axB.set_yscale("log"); axB.legend(fontsize=8, ncol=2, loc="best")
        axB.grid(True, alpha=0.3, which="both")
        axB.set_title("Position uncertainty — fused vs. each single sensor")

        return fig
