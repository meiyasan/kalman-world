import numpy as np
from scipy.linalg import expm

from ..framework.models.linear import LinearKalmanModel
from ..framework.constants import KalmanState, KalmanObservable


class SuspensionState(KalmanState):
    SPRUNG_DISP   = "m"     # sprung mass displacement (car body)
    SPRUNG_VEL    = "m/s"   # sprung mass velocity
    UNSPRUNG_DISP = "m"     # unsprung mass displacement (wheel/axle)
    UNSPRUNG_VEL  = "m/s"   # unsprung mass velocity


class SuspensionObservable(KalmanObservable):
    SPRUNG_DISP = "m"       # measured by proximity / laser sensor


class SuspensionModel(LinearKalmanModel):
    """
    Quarter-car linear suspension model (2-DOF).

    State vector (axis-major order):
        x = [z_s, ż_s, z_u, ż_u]

    where:
        z_s  — sprung mass displacement   (car body)
        ż_s  — sprung mass velocity
        z_u  — unsprung mass displacement (wheel + axle)
        ż_u  — unsprung mass velocity

    Continuous-time dynamics:
        z̈_s =  -(k_s/m_s)(z_s - z_u) - (b_s/m_s)(ż_s - ż_u)
        z̈_u =   (k_s/m_u)(z_s - z_u) + (b_s/m_u)(ż_s - ż_u) - (k_t/m_u)z_u + (k_t/m_u)u

    The road input u is a displacement excitation (control input, l=1).
    Only sprung displacement z_s is observed → H = [1, 0, 0, 0].

    Parameters
    ----------
    sensor_id : str
        Sensor / channel name.
    m_s : float
        Sprung mass [kg], default 300.
    m_u : float
        Unsprung mass [kg], default 50.
    k_s : float
        Suspension spring stiffness [N/m], default 20 000.
    b_s : float
        Suspension damper coefficient [Ns/m], default 1 500.
    k_t : float
        Tyre stiffness [N/m], default 180 000.
    measurement_uncertainty : float
        Measurement noise std [m], default 0.001 (1 mm).
    process_uncertainty : float
        Process noise std scaling, default 0.001.
    fs : float
        Sample rate [Hz], default 256.
    """

    NAME = "Quarter-Car Suspension Model"
    DESCRIPTION = (
        "Linear 2-DOF quarter-car suspension model. "
        "States: [z_s, dz_s, z_u, dz_u]. Observable: sprung displacement z_s. "
        "Control input: road displacement u."
    )
    VERSION = "1.0"

    @property
    def states(self):
        return SuspensionState

    @property
    def observables(self):
        return SuspensionObservable

    def __init__(self, sensor_id: str,
                 m_s: float = 300.0,
                 m_u: float = 50.0,
                 k_s: float = 20_000.0,
                 b_s: float = 1_500.0,
                 k_t: float = 180_000.0,
                 measurement_uncertainty: float = 0.001,
                 process_uncertainty: float = 0.001,
                 **kwargs):

        self.dtype = kwargs.pop("dtype", np.float64)
        fs = kwargs.pop("fs", None) or kwargs.pop("sample_rate", None) or getattr(self, "fs", 256.0)
        dt = 1.0 / fs

        # ------------------------------------------------------------------
        # Continuous-time system matrix A  (4×4)
        # x = [z_s, ż_s, z_u, ż_u]
        # ------------------------------------------------------------------
        A = np.array([
            [0,           1,            0,          0],
            [-k_s / m_s, -b_s / m_s,   k_s / m_s,  b_s / m_s],
            [0,           0,            0,          1],
            [k_s / m_u,   b_s / m_u,  -(k_s + k_t) / m_u, -b_s / m_u],
        ], dtype=self.dtype)

        # Discrete state-transition via matrix exponential
        F = expm(A * dt)

        # ------------------------------------------------------------------
        # Continuous-time input matrix B  (4×1)  [road excitation]
        # ------------------------------------------------------------------
        B_c = np.array([[0], [0], [0], [k_t / m_u]], dtype=self.dtype)
        # Discretise: B_d = A^{-1}(F - I) B_c  (zero-order hold)
        try:
            B_d = np.linalg.solve(A, (F - np.eye(4, dtype=self.dtype)) @ B_c)
        except np.linalg.LinAlgError:
            B_d = dt * B_c  # fallback: Euler approximation

        # ------------------------------------------------------------------
        # Measurement matrix (only z_s observed)
        # ------------------------------------------------------------------
        H = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=self.dtype)

        # ------------------------------------------------------------------
        # Noise covariances
        # ------------------------------------------------------------------
        R = (measurement_uncertainty ** 2) * np.eye(1, dtype=self.dtype)

        q = process_uncertainty ** 2
        Q = q * np.eye(4, dtype=self.dtype)

        # ------------------------------------------------------------------
        # Initial covariance (unit-appropriate)
        # ------------------------------------------------------------------
        if kwargs.get("P0") is None:
            kwargs["P0"] = np.diag([
                measurement_uncertainty ** 2,          # z_s  uncertainty
                (measurement_uncertainty * 10) ** 2,   # ż_s  uncertainty
                measurement_uncertainty ** 2,          # z_u  uncertainty
                (measurement_uncertainty * 10) ** 2,   # ż_u  uncertainty
            ]).astype(self.dtype)

        super().__init__(sensor_id, F=F, H=H, Q=Q, R=R, B=B_d, fs=fs, **kwargs)

    def preprocess(self, zk, uk=None, **kwargs):
        return zk, uk

    def inprocess(self, zk_hat, **kwargs):
        return zk_hat

    def postprocess(self, zk_hat, **kwargs):
        return zk_hat

    # ==================================================================
    # Signature figure — quarter-car state estimates (override KalmanModel.plot).
    # Reached via ``model.plot(record, sample=sample[, road=u_road])``.
    # ==================================================================
    def _plot_static(self, record, sample=None, *, component=None, error=True,
                     road=None, sigma=2.0, figsize=(16, 12), **kwargs):
        """Signature figure: the four states (sprung/unsprung displacement + velocity)
        with ±σ bands and truth in the left column, and a covariance trace plus a
        road-excitation-vs-response panel in the right column. The sensor scatter is
        recovered from the record's (NaN-masked) ``zk``; pass the road excitation as
        ``road`` to overlay the input."""
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        t  = np.asarray(record.time)
        xk = np.asarray(record.xk)
        Pk = np.asarray(record.Pk)
        sig = sigma * np.sqrt(np.diagonal(Pk, axis1=1, axis2=2)) if Pk.size else np.zeros_like(xk)
        xt = np.asarray(sample.xk) if sample is not None else None
        zk = np.asarray(record.zk) if np.size(record.zk) else None

        # NB: SuspensionState aliases sprung/unsprung (shared units collapse the enum),
        # so we index the raw state vector by position and label explicitly.
        labels = ["sprung disp", "sprung vel", "unsprung disp", "unsprung vel"]
        units  = ["m", "m/s", "m", "m/s"]
        colors = ["steelblue", "darkorange", "seagreen", "purple"]
        ns     = min(4, xk.shape[1])

        fig = plt.figure(figsize=figsize)
        gs  = gridspec.GridSpec(4, 2, figure=fig, hspace=0.55, wspace=0.35, width_ratios=[1.6, 1])
        for i in range(ns):
            ax = fig.add_subplot(gs[i, 0])
            ax.fill_between(t, xk[:, i] - sig[:, i], xk[:, i] + sig[:, i],
                            alpha=0.25, color=colors[i], label=rf"$\pm{sigma:g}\sigma$")
            if xt is not None and xt.shape[1] > i:
                ax.plot(t, xt[:, i], "--", lw=1.0, color="gray", alpha=0.8, label="Truth")
            ax.plot(t, xk[:, i], lw=1.8, color=colors[i], label="Kalman")
            if i == 0 and zk is not None:
                m = ~np.isnan(zk[:, 0])
                ax.scatter(t[m], zk[m, 0], s=6, color="black", alpha=0.5, zorder=5, label="Sensor")
            ax.set_ylabel(f"{labels[i]} [{units[i]}]", fontsize=9)
            if i == ns - 1:
                ax.set_xlabel("Time (s)")
            ax.legend(fontsize=8, loc="upper right"); ax.grid(True, alpha=0.3)

        ax_cov = fig.add_subplot(gs[0:2, 1])
        if Pk.size:
            ax_cov.semilogy(t, np.trace(Pk, axis1=1, axis2=2), color="indigo", lw=1.2, alpha=0.85)
        ax_cov.set_xlabel("Time (s)"); ax_cov.set_ylabel(r"tr($P_k$)")
        ax_cov.set_title("Covariance trace\n(inflates between updates → contracts on each measurement)", fontsize=9)
        ax_cov.grid(True, alpha=0.3, which="both")

        ax_road = fig.add_subplot(gs[2:, 1])
        if road is not None:
            ax_road.plot(t, np.asarray(road).reshape(-1) * 100, color="saddlebrown",
                         lw=1.5, alpha=0.7, label="Road input [cm]")
        ax_road.plot(t, xk[:, 0] * 100, color="steelblue", lw=1.8, label=r"Sprung $z_s$ [cm]")
        ax_road.fill_between(t, (xk[:, 0] - sig[:, 0]) * 100, (xk[:, 0] + sig[:, 0]) * 100,
                             alpha=0.2, color="steelblue")
        if xk.shape[1] > 2:
            ax_road.plot(t, xk[:, 2] * 100, color="seagreen", lw=1.4, alpha=0.8, label=r"Unsprung $z_u$ [cm]")
            ax_road.fill_between(t, (xk[:, 2] - sig[:, 2]) * 100, (xk[:, 2] + sig[:, 2]) * 100,
                                 alpha=0.15, color="seagreen")
        ax_road.set_xlabel("Time (s)"); ax_road.set_ylabel("Displacement [cm]")
        ax_road.set_title("Road excitation vs. suspension response", fontsize=9)
        ax_road.legend(fontsize=8); ax_road.grid(True, alpha=0.3)

        fig.suptitle(rf"{self.name()} — Kalman state estimates with $\pm{sigma:g}\sigma$", fontsize=13, y=1.01)
        return fig
