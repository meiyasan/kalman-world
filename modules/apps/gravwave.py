import numpy as np

from ..framework.constants import KalmanObservable, KalmanState
from ..framework.models.extended import ExtendedKalmanModel


# ===== State / Observable enums =====

class PolarizationState(KalmanState):
    LOG_AMPLITUDE = ""   # log(a)  — ensures a = exp(x0) > 0 always
    LOG_AMP_VEL   = ""   # d/dt log(a)
    PHASE_POS     = ""   # u_ψ
    PHASE_VEL     = ""   # v_ψ  (angular freq, rad/s)
    THETA_POS     = ""   # u_θ
    THETA_VEL     = ""   # v_θ
    PHI_POS       = ""   # u_φ
    PHI_VEL       = ""   # v_φ


class PolarizationObservable(KalmanObservable):
    RE1 = ""   # Re(z1)
    IM1 = ""   # Im(z1)
    RE2 = ""   # Re(z2)
    IM2 = ""   # Im(z2)


# ===== Helper =====

def jones_signal(a, psi, theta, phi):
    """Jones bivariate complex signal: [a·exp(iψ)·cos(θ/2),  a·exp(iψ)·sin(θ/2)·exp(iφ)]"""
    carrier = a * np.exp(1j * psi)
    return np.array([carrier * np.cos(theta / 2),
                     carrier * np.sin(theta / 2) * np.exp(1j * phi)])


# ===== Model =====

class PolarizationModel(ExtendedKalmanModel):
    """
    EKF for online estimation of gravitational-wave polarization parameters.

    Implements the Amblard & Michel (Jones formalism) model for tracking
    a bivariate GW signal in the presence of Gaussian observation noise.

    State (8-D): x = [log(a), ȧ/a, u_ψ, v_ψ, u_θ, v_θ, u_φ, v_φ]
    Observation (4-D real): z = [Re(z1), Im(z1), Re(z2), Im(z2)]

    Why log-amplitude
    -----------------
    The observation z = a·exp(iψ)·Jones is invariant under (a,ψ)→(-a,ψ+π).
    Tracking ``a`` directly allows it to flip negative, causing the EKF to
    lose phase coherence.  Using ``log(a)`` constrains a = exp(x[0]) > 0
    always and removes the sign ambiguity.

    Why ``delta`` ≠ ``dt`` for Q
    ----------------------------
    ``J`` (state transition) uses ``dt = 1/fs`` (correct physics at sample
    rate).  ``Q`` uses ``delta`` (seconds) as a bandwidth-tuning knob::

        Q[vel, vel] = sigma_p² × delta   (larger delta → wider bandwidth)

    Required tracking condition: ``ωn = sqrt(sigma_psi² × delta / sigma_n²)``
    must satisfy ``ωn² > d²ψ/dt² / max_phase_lag``.

    Why ``L()`` returns ``None``
    ---------------------------
    The default base-class ``L()`` numerically computes ∂f/∂w.  Since
    ``f()`` ignores ``w`` (additive noise model), L = 0 → Q = L@Q@L^T = 0,
    silently zeroing all process noise.  Returning ``None`` tells the
    framework to use Q directly.

    Parameters
    ----------
    sensor_id : str
        Unique identifier for the sensor.
    fs : float
        Sampling frequency [Hz].  Default: 100.
    delta : float
        Bandwidth tuning parameter [s].  Default: 0.5.
    sigma_a : float
        Log-amplitude process noise std.  Default: 0.1.
    sigma_psi : float
        Phase/frequency process noise std.  Default: 0.5.
    sigma_theta : float
        θ-polarization process noise std.  Default: 0.1.
    sigma_phi : float
        φ-polarization process noise std.  Default: 0.1.
    sigma_n : float
        Observation noise std.  Default: 1.0.
    **kwargs
        Passed through to ``ExtendedKalmanModel.__init__``
        (e.g. ``P0``, ``x0``).
    """

    NAME        = "Polarization EKF"
    DESCRIPTION = ("EKF for real-time estimation of gravitational-wave "
                   "polarization parameters using the Jones formalism "
                   "(Amblard & Michel).  8-D log-amplitude state, "
                   "4-D real observation.")
    VERSION     = "6.1"

    @property
    def states(self):      return PolarizationState
    @property
    def observables(self): return PolarizationObservable

    def __init__(self, sensor_id: str,
                 fs:          float = 100.0,
                 delta:       float = 0.5,
                 sigma_a:     float = 0.1,
                 sigma_psi:   float = 0.5,
                 sigma_theta: float = 0.1,
                 sigma_phi:   float = 0.1,
                 sigma_n:     float = 1.0,
                 **kwargs):

        self.n = 8
        self.m = 4
        self.l = 0

        dt = 1.0 / fs
        self._dt = dt

        # ── State transition: constant-velocity at sample rate dt ─
        J = np.array([[1.0, dt], [0.0, 1.0]])
        M = np.zeros((8, 8))
        for i in range(4):
            M[2*i:2*i+2, 2*i:2*i+2] = J
        self._transition = M

        # ── Process noise: bandwidth via delta (NOT dt) ───────────
        def _Qblock(sig, d):
            return (sig ** 2) * np.array([[d**3/3, d**2/2],
                                          [d**2/2, d      ]])

        Q = np.zeros((8, 8))
        for i, sig in enumerate([sigma_a, sigma_psi, sigma_theta, sigma_phi]):
            Q[2*i:2*i+2, 2*i:2*i+2] = _Qblock(sig, delta)

        R = (sigma_n ** 2) * np.eye(4)

        super().__init__(sensor_id, Q=Q, R=R, fs=fs, **kwargs)

    # ── CRITICAL: prevent base class from zeroing Q via L=0 ──────
    def L(self, x=None, u=None, w=None, t=None):
        """Return None: process noise is additive; Q is used directly."""
        return None

    def f(self, x, u=None, w=None, t=None):
        return self._transition @ np.asarray(x).reshape(8, 1)

    def h(self, x, v=None, t=None):
        x   = np.asarray(x).ravel()
        a   = np.exp(x[0])
        psi, theta, phi = x[2], x[4], x[6]
        z_c = jones_signal(a, psi, theta, phi)
        z   = np.array([z_c[0].real, z_c[0].imag,
                        z_c[1].real, z_c[1].imag]).reshape(4, 1)
        if v is not None:
            z += np.asarray(v).reshape(4, 1)
        return z

    def H(self, x=None, t=None):
        """Analytic Jacobian ∂h/∂x (4×8)."""
        if x is None:
            x = self.x0
        x   = np.asarray(x).ravel()
        a   = np.exp(x[0])
        psi, theta, phi = x[2], x[4], x[6]
        c2  = np.cos(theta / 2)
        s2  = np.sin(theta / 2)
        eip = np.exp(1j * psi)
        eif = np.exp(1j * phi)

        dz1_dloga = a * eip * c2
        dz2_dloga = a * eip * s2 * eif

        dz1_dpsi  = 1j * a * eip * c2
        dz2_dpsi  = 1j * a * eip * s2 * eif

        dz1_dth   = -0.5 * a * eip * s2
        dz2_dth   =  0.5 * a * eip * c2 * eif

        dz1_dphi  = 0 + 0j
        dz2_dphi  = 1j * a * eip * s2 * eif

        Jac = np.zeros((4, 8))

        def fill(r_re, r_im, dz, col):
            Jac[r_re, col] = dz.real
            Jac[r_im, col] = dz.imag

        fill(0, 1, dz1_dloga, 0);  fill(2, 3, dz2_dloga, 0)
        fill(0, 1, dz1_dpsi,  2);  fill(2, 3, dz2_dpsi,  2)
        fill(0, 1, dz1_dth,   4);  fill(2, 3, dz2_dth,   4)
        fill(0, 1, dz1_dphi,  6);  fill(2, 3, dz2_dphi,  6)
        return Jac

    def preprocess(self, zk, uk=None, **kwargs):
        zk = np.asarray(zk)
        if np.iscomplexobj(zk) and zk.shape[1] == 2:
            zk = np.column_stack([zk[:, 0].real, zk[:, 0].imag,
                                   zk[:, 1].real, zk[:, 1].imag])
        return zk, uk

    def inprocess(self, state_vector, **kwargs):  return state_vector
    def postprocess(self, xk_hat, **kwargs):      return xk_hat

    # ==================================================================
    # Derived physical parameters (a, ψ̇, θ, φ) from the log-amplitude state.
    # ==================================================================
    def physical_parameters(self, record):
        """Map the 8-D filter state to physical polarization parameters:
        ``amplitude = exp(x0)``, ``freq = ψ̇/2π``, ``theta = x4``, ``phi = x6``,
        ``psi = x2``. Returns a dict of (T,) arrays."""
        xk = np.asarray(record.xk)
        return dict(amplitude=np.exp(xk[:, 0]), freq=xk[:, 3] / (2 * np.pi),
                    theta=xk[:, 4], phi=xk[:, 6], psi=xk[:, 2])

    def reconstruct_component(self, record):
        """Reconstruct the (complex) first signal component
        ``z1 = a·e^{iψ}·cos(θ/2)`` from the filtered state."""
        p = self.physical_parameters(record)
        return p["amplitude"] * np.exp(1j * p["psi"]) * np.cos(p["theta"] / 2)

    # ==================================================================
    # Signature figure — physical parameters + covariance + reconstruction.
    # Reached via ``model.plot(record[, truth=..., measured=...])``.
    # ==================================================================
    def _plot_static(self, record, sample=None, *, component=None, error=True,
                     truth=None, measured=None, sigma=2.0, figsize=(13, 11), **kwargs):
        """Signature figure: amplitude, carrier frequency, θ and φ (each with ±σ band
        and EKF estimate), plus the posterior covariance trace and the component-1
        reconstruction. Pass ``truth`` (dict with amplitude/freq/theta/phi arrays, or a
        ``sample`` record) and ``measured`` (the noisy first component) to overlay."""
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        t  = np.asarray(record.time)
        Pk = np.asarray(record.Pk)
        p  = self.physical_parameters(record)
        s2 = sigma * np.sqrt(np.diagonal(Pk, axis1=1, axis2=2)) if Pk.size else np.zeros_like(record.xk)
        z1_hat = self.reconstruct_component(record)

        tru = dict(truth) if truth is not None else {}
        if not tru and sample is not None:
            sx = np.asarray(sample.xk)
            tru = dict(amplitude=np.exp(sx[:, 0]),
                       freq=np.gradient(sx[:, 2], 1.0 / self.fs) / (2 * np.pi),
                       theta=sx[:, 4], phi=sx[:, 6])

        fig = plt.figure(figsize=figsize)
        gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.50, wspace=0.35)
        params = [
            (p["amplitude"], p["amplitude"] * s2[:, 0], tru.get("amplitude"), r"Amplitude $a_t$"),
            (p["freq"],      s2[:, 3] / (2 * np.pi),    tru.get("freq"),      r"Carrier $\dot{\psi}/2\pi$ [Hz]"),
            (p["theta"],     s2[:, 4],                  tru.get("theta"),     r"$\theta_t$ [rad]"),
            (p["phi"],       s2[:, 6],                  tru.get("phi"),       r"$\varphi_t$ [rad]"),
        ]
        for idx, (hat, band, true, ylab) in enumerate(params):
            ax = fig.add_subplot(gs[idx // 2, idx % 2])
            ax.fill_between(t, hat - band, hat + band, alpha=0.15, color="red", label=rf"$\pm{sigma:g}\sigma$")
            if true is not None:
                ax.plot(t, true, color="black", lw=1.5, label="True")
            ax.plot(t, hat, color="red", lw=1.8, label="EKF")
            ax.set_xlabel("Time (s)"); ax.set_ylabel(ylab, fontsize=10)
            ax.legend(fontsize=8); ax.grid(True, alpha=0.25)

        ax_cov = fig.add_subplot(gs[2, 0])
        if Pk.size:
            ax_cov.semilogy(t, np.trace(Pk, axis1=1, axis2=2), color="black", lw=1.2)
        ax_cov.set_xlabel("Time (s)"); ax_cov.set_ylabel(r"tr($\Gamma_{t|t}$)", fontsize=10)
        ax_cov.set_title("Posterior covariance trace", fontsize=9); ax_cov.grid(True, alpha=0.25, which="both")

        ax_sig = fig.add_subplot(gs[2, 1])
        if measured is not None:
            mv = np.asarray(measured)
            ax_sig.plot(t, mv.real if np.iscomplexobj(mv) else mv, color="black", lw=0.6, alpha=0.55, label="Measured")
        ax_sig.plot(t, z1_hat.real, color="red", lw=1.8, alpha=0.9, label="EKF")
        ax_sig.set_xlabel("Time (s)"); ax_sig.set_ylabel("First component")
        ax_sig.set_title("Component 1: measured vs EKF", fontsize=9)
        ax_sig.legend(fontsize=8); ax_sig.grid(True, alpha=0.25)

        fig.suptitle(f"{self.name()} — GW polarization estimate", fontsize=13)
        return fig

    def _plot_interactive(self, record, sample=None, *, measured=None, window=None,
                          frames=80, interval=120, figsize=(11, 4.5), **kwargs):
        """Interactive replay: slide a window across the signal, showing the measured
        first component against the EKF reconstruction. Pass ``measured`` (noisy comp 1)."""
        import matplotlib.pyplot as plt
        from ..helpers import animate
        t = np.asarray(record.time)
        sig_ekf = self.reconstruct_component(record).real
        meas = None
        if measured is not None:
            mv = np.asarray(measured)
            meas = mv.real if np.iscomplexobj(mv) else mv
        N = len(t)
        win = window or max(50, int(0.12 * N))
        step = max(1, (N - win) // max(1, frames))
        starts = list(range(0, max(1, N - win), step))

        fig, ax = plt.subplots(figsize=figsize)
        lines = {}
        if meas is not None:
            (lines["m"],) = ax.plot([], [], color="black", alpha=0.55, lw=0.7, label="Measured (comp 1)")
        (lines["e"],) = ax.plot([], [], color="red", lw=1.8, label="EKF reconstruction")
        ref = meas if meas is not None else sig_ekf
        ax.set_ylim(ref.min() * 1.1, ref.max() * 1.1)
        ax.set_xlabel("Time (s)"); ax.set_ylabel("First component")
        ax.grid(True, alpha=0.3); ax.legend(loc="upper right", fontsize=9)

        def _draw(s):
            sl = slice(s, s + win); tt = t[sl]
            if "m" in lines: lines["m"].set_data(tt, meas[sl])
            lines["e"].set_data(tt, sig_ekf[sl]); ax.set_xlim(tt[0], tt[-1])
            ax.set_title(f"{self.name()} — measured vs EKF  "
                         f"(t = {t[s]:.2f}–{t[min(s + win - 1, N - 1)]:.2f} s)")
            return tuple(lines.values())

        return animate(fig, _draw, frames=starts, interval=interval)
