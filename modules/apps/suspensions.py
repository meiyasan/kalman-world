import numpy as np
from scipy.linalg import expm
from scipy.signal import hilbert as _hilbert

from ..models.linear import KalmanLinearModel
from ..models.composite import KalmanCompositeModel
from ..framework.constants import KalmanState, KalmanObservable
from .oscillators import AnalyticOscillatorModel, InharmonicOscillatorModel


class SuspensionState(KalmanState):
    SPRUNG_DISP   = "m"     # sprung mass displacement (car body)
    SPRUNG_VEL    = "m/s"   # sprung mass velocity
    UNSPRUNG_DISP = "m"     # unsprung mass displacement (wheel/axle)
    UNSPRUNG_VEL  = "m/s"   # unsprung mass velocity


class SuspensionObservable(KalmanObservable):
    SPRUNG_DISP = "m"       # measured by proximity / laser sensor


class SuspensionModel(KalmanLinearModel):
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


# ===========================================================================
#  Violin mode Kalman filter  (Finn & Mukherjee 2001 / KAGRA 2025 approach)
# ===========================================================================

class ViolinModel(AnalyticOscillatorModel):
    """
    Single-mode violin suspension Kalman tracker.

    Extends ``AnalyticOscillatorModel`` with:

    * **Real-strain input**: ``preprocess`` accepts ``(N, 1)`` real strain and
      internally computes the analytic signal via the Hilbert transform before
      delegating to the parent rotating-frame demodulation.  ``(N, 2)``
      pre-computed analytic input is also accepted (parent path).
    * **Physics noise defaults**: if ``process_uncertainty`` / ``measurement_uncertainty``
      are not given, they are derived from the fluctuation-dissipation theorem
      and the instrument noise PSD respectively.
    * **subtract()**: convenience method to reconstruct the violin mode from a
      ``KalmanRecord`` and subtract it from the raw strain.

    Implements the slowly-varying-envelope (SVE) approach from:

    * Finn & Mukherjee, *Phys. Rev. D* **63**, 062004 (2001)
    * KAGRA O3GK regression, arXiv:2509.20658 (2025)

    The suspension fiber oscillation x(t) = Re[ψ(t)·exp(iωct)] is tracked
    via the complex amplitude ψ(t) in a rotating frame at carrier frequency ωc.
    All state-transition and demodulation logic is inherited from
    ``AnalyticOscillatorModel``.  scipy's ``hilbert()`` returns the full-amplitude
    analytic signal, so ψ equals the full complex envelope (no factor-of-2 scaling).

    Process noise (fluctuation-dissipation theorem):
        σ_w = A·γ·Δt   (preferred — unit-agnostic, pass via ``amplitude``)
        σ_w² = ω₀·kBT / (4Q²·ωc²·m·fs)  (SI fallback when ``amplitude`` is None)

    Measurement noise:
        σ_v² = S_n · fs / 2    (noise PSD S_n [1/Hz], Nyquist conversion)

    Parameters
    ----------
    sensor_id  : str
        Channel name (e.g. "DARM", "SRM_viol1").
    f0         : float
        Violin mode resonance frequency [Hz].
    Q          : float
        Quality factor of the violin mode (typically 10⁴–10⁶).
    fs         : float
        Sample rate of the input data [Hz].
    wc         : float | None
        Carrier / demodulation frequency [rad/s].  Default: 2π·f0 (on-resonance).
    process_uncertainty : float | None
        Process noise std [same units as state].  Highest priority: overrides
        both ``amplitude`` and the FDT formula when provided.
    measurement_uncertainty : float | None
        Measurement noise std [same units as observation].
        If None, uses σ_v² = noise_psd · fs / 2.
    amplitude  : float | None
        Expected SVE amplitude (in the same units as the measurement).
        Sets σ_w = amplitude · γ · Δt, which is the per-sample thermal
        fluctuation of the slowly-varying envelope.  **Recommended for
        GW strain or any non-SI unit system.**  Ignored when
        ``process_uncertainty`` is given explicitly.
    kBT        : float
        Thermal energy [J], default 4×10⁻²¹ J  (≈ kB × 300 K).
        Used only when both ``amplitude`` and ``process_uncertainty`` are None.
    mass       : float
        Effective oscillator mass [kg], default 1.0.
        Used only when both ``amplitude`` and ``process_uncertainty`` are None.
        NOTE: calibrated for displacement in metres; not appropriate for
        dimensionless GW strain — use ``amplitude`` instead.
    noise_psd  : float
        One-sided noise PSD floor [1/Hz], default 1×10⁻⁴⁰.
    """

    NAME = "Violin Mode Kalman Filter"
    DESCRIPTION = (
        "Complex-amplitude tracker for suspension violin modes — "
        "slowly-varying-envelope approach (Finn & Mukherjee 2001). "
        "Inherits AnalyticOscillatorModel; adds Hilbert pre-processing "
        "and physics-derived noise defaults."
    )
    VERSION = "2.0"


    def __init__(self, sensor_id: str,
                 f0: float,
                 Q: float,
                 fs: float,
                 wc: float | None = None,
                 process_uncertainty: float | None = None,
                 measurement_uncertainty: float | None = None,
                 amplitude: float | None = None,
                 kBT: float = 4e-21,
                 mass: float = 1.0,
                 noise_psd: float = 1e-40,
                 **kwargs):

        w0 = 2.0 * np.pi * f0
        if wc is None:
            wc = w0

        # Store violin-specific metadata — these are read by the inherited
        # AnalyticOscillatorModel.preprocess / postprocess via self["wc"] etc.
        self["f0"]     = f0
        self["Q_mode"] = Q
        self["w0"]     = w0
        self["wc"]     = wc

        # ------------------------------------------------------------------
        # Derive noise covariances with physics defaults
        # ------------------------------------------------------------------
        if process_uncertainty is None:
            gamma = w0 / (2.0 * Q)
            if amplitude is not None:
                # Unit-agnostic FDT: σ_w = A·γ·Δt
                # Per-sample thermal fluctuation of the SVE amplitude.
                # Works for any unit system (strain, displacement, etc.)
                # giving K << 1 and a narrowband rotating-frame filter.
                sigma_w2 = (float(amplitude) * gamma / fs) ** 2
            else:
                # FDT in SI: σ_w² = ω₀·kBT/(4Q²·ωc²·m·fs)
                # NOTE: calibrated for displacement data in metres.
                # For dimensionless GW strain or other units, pass `amplitude`
                # (the expected mode envelope magnitude) instead.
                sigma_w2 = w0 * kBT / (4.0 * Q ** 2 * wc ** 2 * mass * fs)
        else:
            sigma_w2 = float(process_uncertainty) ** 2

        if measurement_uncertainty is None:
            sigma_v2 = noise_psd * fs / 2.0
        else:
            sigma_v2 = float(measurement_uncertainty) ** 2

        dtype = kwargs.pop("dtype", np.float64)

        dt    = 1.0 / fs
        gamma = w0 / (2.0 * Q)
        dw    = w0 - wc

        A_c = np.array([[-gamma, -dw],
                        [ dw,   -gamma]], dtype=dtype)
        F = expm(A_c * dt)
        H = np.eye(2, dtype=dtype)
        Q_mat = sigma_w2 * np.eye(2, dtype=dtype)
        R     = sigma_v2 * np.eye(2, dtype=dtype)

        if kwargs.get("P0") is None:
            if amplitude is not None:
                # Initial covariance scaled to the expected mode amplitude.
                # K_0 ≈ amplitude² / (amplitude² + sigma_v²) << 1 when
                # amplitude << sigma_v, or ≈ 0.5 when amplitude ≈ sigma_v.
                # Either way, a user-supplied x0 is not overwritten by the first noisy step.
                kwargs["P0"] = float(amplitude) ** 2 * np.eye(2, dtype=dtype)
            else:
                # Fall back to one measurement noise variance — keeps K_0 reasonable.
                kwargs["P0"] = sigma_v2 * np.eye(2, dtype=dtype)

        # Bypass AnalyticOscillatorModel.__init__ (which applies a non-standard
        # sqrt(2) noise scaling) and call KalmanLinearModel directly.
        # The inherited pre/postprocess use self["wc"]/self.fs set above + via fs.
        from ..models.linear import KalmanLinearModel
        KalmanLinearModel.__init__(self, sensor_id, F=F, H=H, Q=Q_mat, R=R,
                                   fs=fs, **kwargs)

    # ------------------------------------------------------------------
    #  preprocess — extend parent to accept real (N,1) strain via Hilbert
    # ------------------------------------------------------------------
    def preprocess(self, zk, uk=None, **kwargs):
        """
        Complex demodulation z(t) → ψ(t) = z_A(t) · exp(−i·ωc·t).

        Accepts:
          (N, 1)  — real strain → analytic signal via Hilbert, then demodulate.
          (N, 2)  — [Re, Im] analytic signal already available → demodulate only
                    (delegates directly to parent ``AnalyticOscillatorModel.preprocess``).
        Returns (N, 2): [Re(ψ), Im(ψ)] in the rotating frame.
        """
        if zk.shape[1] == 1:
            x  = zk[:, 0].astype(float)
            xa = _hilbert(x)
            # Convert real strain to (N,2) analytic form, then use parent logic
            zk = np.column_stack([xa.real, xa.imag])
        return super().preprocess(zk, uk, **kwargs)

    # ------------------------------------------------------------------
    #  Extraction helper
    # ------------------------------------------------------------------
    def extract(self, record, strain):
        """
        Reconstruct the violin mode from a ``KalmanRecord``.

            x_violin(t) = 2·[Re(ψ̂)·cos(ωct) − Im(ψ̂)·sin(ωct)]

        Parameters
        ----------
        record : KalmanRecord
            Output of ``filter()``.
        strain : array_like, shape (N,)
            Original real strain.

        Returns
        -------
        violin : ndarray
            Estimated violin mode contribution extracted from ``strain``.
        residual : ndarray
            ``strain`` with the violin mode removed.
        """
        xk = record.xk          # (N, 2): [Re(ψ̂), Im(ψ̂)] in rotating frame
        N  = len(xk)
        times = self.t0 + np.arange(N) / self.fs
        arg   = self["wc"] * times

        # x(t) = Re(ψ · exp(i·ωc·t)) = Re(ψ)·cos(ωc·t) − Im(ψ)·sin(ωc·t)
        # No factor of 2: scipy hilbert returns the full-amplitude analytic signal,
        # so the tracked state ψ already equals the full complex amplitude.
        violin = xk[:, 0] * np.cos(arg) - xk[:, 1] * np.sin(arg)

        strain = np.asarray(strain, dtype=float).ravel()
        L = min(len(strain), N)
        return violin[:L], strain[:L] - violin[:L]


# ===========================================================================
#  Multi-harmonic inharmonic violin mode forest  (Moisset et al. 2025)
# ===========================================================================

class ViolinForestModel(KalmanCompositeModel):
    """
    Multi-harmonic violin mode tracker with optional inharmonicity.

    Stacks one ``ViolinModel`` per tracked harmonic order.  Partial
    frequencies follow the **Fletcher–Young** inharmonicity formula::

        f_n = n · f0 · sqrt(1 + B · n²)

    where ``B`` is the dimensionless inharmonicity coefficient.  For a
    cylindrical suspension fiber (Young's modulus ``E``, diameter ``d``,
    tension ``T``, length ``L``)::

        B = π³ E d⁴ / (64 T L²)

    For KAGRA sapphire fibers ``B ≈ 0.10``, which shifts the 2nd harmonic
    from a nominal 360 Hz to ~420 Hz — exactly the observed O3GK data.
    ``B = 0`` recovers a perfectly harmonic suspension (LIGO fused-silica).

    Parameters
    ----------
    sensor_id : str
        Channel / fiber label (e.g. "ITMX_fiber1").
    f0 : float
        Fundamental violin mode frequency [Hz].
    n_modes : int
        Number of harmonic orders to track (default 3).
    B : float
        Inharmonicity coefficient (dimensionless, default 0 = perfectly harmonic).
    Q_modes : float | list
        Quality factor(s) for each mode.  Scalar → same for all.
    fs : float
        Sampling rate [Hz] of the strain data.
    process_uncertainty : float | None
        Per-mode process noise std.  ``None`` → use fluctuation-dissipation
        estimate inside ``ViolinModel``.
    measurement_uncertainty : float | None
        Per-mode measurement noise std.  ``None`` → use PSD estimate.
    **kwargs
        Additional keyword arguments forwarded to each ``ViolinModel``.

    Attributes
    ----------
    partial_freqs : ndarray, shape (n_modes,)
        Inharmonic partial frequencies [Hz].
    B : float
        Inharmonicity coefficient used at construction.
    f0 : float
        Fundamental frequency [Hz].

    Class methods (inherited logic via static helpers)
    --------------------------------------------------
    Use ``InharmonicOscillatorModel.partial_frequencies``
    and ``InharmonicOscillatorModel.fit_inharmonicity`` for pre-analysis.

    Examples
    --------
    >>> from kalman.apps.suspensions import ViolinForestModel
    >>> forest = ViolinForestModel(
    ...     'ITMX', f0=180.1, n_modes=3, B=0.10,
    ...     Q_modes=[1.3e4, 1.8e4, 2.1e4], fs=4096)
    >>> forest.partial_freqs   # [180.1, ~420, ~752] Hz
    """

    NAME = "Violin Mode Forest"
    DESCRIPTION = (
        "Multi-harmonic violin mode tracker with Fletcher inharmonicity. "
        "One ViolinModel per tracked harmonic. "
        "f_n = n·f1·sqrt(1 + B·n²).  B=0 → harmonic, B>0 → stiff-fiber inharmonic."
    )
    VERSION = "1.0"

    def __init__(self, sensor_id: str,
                 f0: float,
                 n_modes: int = 3,
                 B: float = 0.0,
                 Q_modes: float | list = 1e4,
                 fs: float = 4096.0,
                 process_uncertainty: float | None = None,
                 measurement_uncertainty: float | None = None,
                 amplitudes: float | list | None = None,
                 **kwargs):

        self.f0 = float(f0)
        self.inharmonicity = float(B)  # 'B' collides with KalmanCompositeModel.B matrix
        self.n_modes = int(n_modes)

        partial_freqs = InharmonicOscillatorModel.partial_frequencies(f0, B, n_modes)
        self.partial_freqs = partial_freqs

        if isinstance(Q_modes, (int, float)):
            q_list = [float(Q_modes)] * n_modes
        else:
            q_list = list(Q_modes)
            if len(q_list) != n_modes:
                raise ValueError(
                    f"Q_modes list length {len(q_list)} != n_modes {n_modes}")

        if amplitudes is None:
            amp_list = [None] * n_modes
        elif isinstance(amplitudes, (int, float)):
            amp_list = [float(amplitudes)] * n_modes
        else:
            amp_list = list(amplitudes)
            if len(amp_list) != n_modes:
                raise ValueError(
                    f"amplitudes list length {len(amp_list)} != n_modes {n_modes}")

        sub_models = []
        for i, (fn, Qn, An) in enumerate(zip(partial_freqs, q_list, amp_list)):
            m = ViolinModel(
                f"{sensor_id}_mode{i+1}",
                f0=fn,
                Q=Qn,
                fs=fs,
                process_uncertainty=process_uncertainty,
                measurement_uncertainty=measurement_uncertainty,
                amplitude=An,
                **kwargs,
            )
            sub_models.append(m)

        super().__init__(*sub_models)
        # Prevent the block-diagonal L=0 from zeroing process noise
        self.L = lambda x=None, u=None, w=None, t=None: None

    # ------------------------------------------------------------------
    # Convenience: subtract all modes from a strain time series
    # ------------------------------------------------------------------
    def extract(self, records, strain):
        """
        Extract all tracked violin modes from a strain time series.

        Parameters
        ----------
        records : list of KalmanRecord
            One record per mode (ordered 1 … n_modes), output of per-mode
            ``ViolinModel.filter()`` calls.
        strain : array_like, shape (N,)
            Original real strain.

        Returns
        -------
        violins : list of ndarray
            Estimated violin contribution per mode.
        residual : ndarray
            Strain with all tracked modes removed.
        """
        residual = np.asarray(strain, dtype=float).ravel()
        violins = []
        for model, record in zip(self.models, records):
            violin, residual = model.extract(record, residual)
            violins.append(violin)
        return violins, residual

