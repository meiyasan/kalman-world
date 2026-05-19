import numpy as np

from ..framework.constants import KalmanObservable, KalmanState
from ..models import KalmanExtendedModel


# ===== State / Observable enums =====

class GWPolarizationState(KalmanState):
    LOG_AMPLITUDE = ""   # log(a)  — ensures a = exp(x0) > 0 always
    LOG_AMP_VEL   = ""   # d/dt log(a)
    PHASE_POS     = ""   # u_ψ
    PHASE_VEL     = ""   # v_ψ  (angular freq, rad/s)
    THETA_POS     = ""   # u_θ
    THETA_VEL     = ""   # v_θ
    PHI_POS       = ""   # u_φ
    PHI_VEL       = ""   # v_φ


class GWPolarizationObservable(KalmanObservable):
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

class PolarizationModel(KalmanExtendedModel):
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
        Passed through to ``KalmanExtendedModel.__init__``
        (e.g. ``P0``, ``x0``).
    """

    NAME        = "Polarization EKF"
    DESCRIPTION = ("EKF for real-time estimation of gravitational-wave "
                   "polarization parameters using the Jones formalism "
                   "(Amblard & Michel).  8-D log-amplitude state, "
                   "4-D real observation.")
    VERSION     = "6.1"

    @property
    def states(self):      return GWPolarizationState
    @property
    def observables(self): return GWPolarizationObservable

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
        self._M = M

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
        return self._M @ np.asarray(x).reshape(8, 1)

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
