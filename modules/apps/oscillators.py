import numpy as np
from scipy.linalg import expm
from scipy.optimize import curve_fit

from ..models.linear import KalmanLinearModel
from ..models.composite import KalmanCompositeModel
from ..framework.constants import KalmanState, KalmanObservable

class OscillatorState(KalmanState):
    POSITION = "m"
    SPEED    = "m/s"

class OscillatorObservable(KalmanObservable):
    POSITION = "m"

class OscillatorModel(KalmanLinearModel):

    NAME = "Oscillator State Model"
    DESCRIPTION = "State model for oscillatory motion, simulating the behavior of a simple harmonic oscillator."
    VERSION = "1.0"

    @property
    def states(self):
        return OscillatorState
    @property
    def observables(self):
        return OscillatorObservable

    def __init__(self, sensor_id, **kwargs):

        measurement_uncertainty = kwargs.pop("measurement_uncertainty", 0.5)
        process_uncertainty = kwargs.pop("process_uncertainty", 0.05)

        self.fs = kwargs.pop("fs", None) or kwargs.pop("sample_rate", None) or self.fs
        self.measurement_uncertainty = measurement_uncertainty
        self.process_uncertainty = process_uncertainty

        dt = 1.0 / self.fs

        w0 = self["w0"] = kwargs.pop("w0", 1)
        Q  = self["Q"] = kwargs.pop("Q", None)
        b  = self["b"] = w0/Q if Q is not None else 0
        
        # the state-transition model
        F = np.array([[        1,     dt], 
                      [-w0**2*dt, 1-b*dt]])
        
        # the observation model
        H = np.array([[1, 0]])
        
        # the covariance of the process noise
        R = np.array([measurement_uncertainty]).reshape(1, 1)
        # the covariance of the observation noise
        Q = process_uncertainty * np.array([[1.0, 0.0], 
                                            [0.0, 1.0]])

        x0, P0 = kwargs.pop("x0", None), kwargs.pop("P0", None)
        super().__init__(sensor_id, F=F, H=H, Q=Q, R=R, P0=P0, x0=x0, **kwargs)

    def preprocess(self, zk, uk=None, **kwargs):
        return zk, uk

    def inprocess(self, xk_hat, **kwargs):
        return xk_hat

    def postprocess(self, zk_hat, **kwargs):
        return zk_hat
    
    # def plot(self, *args, **kwargs):
    #     t = self.time()
    #     if t is None:
    #         return

    #     measurements = self.get_measurement_vector()
    #     state_vector = self.get_posterior_state_vector()
    #     smooth_state_vector = self.get_smooth_state_vector()

    #     args = list(args)
    #     plt.figure(figsize=(10, 5))

    #     if measurements is not None:
    #         plt.scatter(t, measurements, label='Data', color='black', marker='+')
        
    #     truth = self.truth(duration=self.duration())
    #     if truth is not None:
    #         plt.plot(t, truth[:, 0], '--', label='Truth', color='lightblue')

    #     plt.plot(t, state_vector[:, 0], label='State Estimate', color='orange')
    #     if smooth_state_vector is not None:
    #         plt.plot(t, smooth_state_vector[:, 0], label='Smoothed Estimate', color='green')


class AnalyticOscillatorState(KalmanState):
    REAL = "re"   # dimensionless complex amplitude — real part
    IMAG = "im"   # dimensionless complex amplitude — imaginary part

class AnalyticOscillatorObservable(KalmanObservable):
    PSI = ""    # complex amplitude (dimensionless)

class AnalyticOscillatorModel(KalmanLinearModel):

    NAME = "Analytic Oscillator State Model (x + iy representation)"
    DESCRIPTION = "State model for oscillatory motion, simulating the behavior of a simple harmonic oscillator."
    VERSION = "1.0"

    @property
    def states(self):
        return AnalyticOscillatorState
    @property
    def observables(self):
        return AnalyticOscillatorObservable

    def __init__(self, name, **kwargs):
        measurement_uncertainty = kwargs.pop("measurement_uncertainty", 0.5)
        process_uncertainty = kwargs.pop("process_uncertainty", 0.05)

        self.fs = kwargs.pop("fs", None) or kwargs.pop("sample_rate", None) or self.fs
        self.measurement_uncertainty = measurement_uncertainty
        self.process_uncertainty = process_uncertainty

        dt = 1.0 / self.fs
        w0 = self["w0"] = kwargs.get("w0", 1)
        wc = self["wc"] = kwargs.get("wc", 1)
        dw = np.power(w0, 1) - np.power(wc, 1)
        Q = kwargs.pop("Q", None) # quality factor (not Q matrix)

        # the state-transition model
        A = expm(np.array([[-w0 / (2 * Q) * dt, -dw * dt],
                           [dw * dt,  -w0 / (2 * Q) * dt]]))
        
        # the observeation model
        C = np.identity(2)
        # the covariance of the process noise
        R = measurement_uncertainty * np.sqrt(2) * np.identity(2)
        # the covariance of the observation noise
        Q = process_uncertainty * np.sqrt(2) * np.identity(2)

        x0, P0 = kwargs.pop("x0", None), kwargs.pop("P0", None)
        super().__init__(name, F=A, H=C, Q=Q, R=R, P0=P0, x0=x0, **kwargs)

    def preprocess(self, zk, uk=None, **kwargs):
        # zk is (N, 1) real lab-frame signal x(t), or (N, 2) analytic signal [Re(xa), Im(xa)].
        # For (N, 1): apply Hilbert transform first to form the analytic signal.
        # Then demodulate into the rotating frame → (N, 2) [Re(ψ), Im(ψ)].
        from scipy.signal import hilbert
        if zk.shape[1] == 1:
            zr = zk[:, 0]
            zi = np.imag(hilbert(zr))
        else:
            zr, zi = zk[:, 0], zk[:, 1]

        times = self.t0 + np.arange(len(zk)) / self.fs
        arg = self["wc"] * times

        zk_rot = np.column_stack([
            zr * np.cos(arg) + zi * np.sin(arg),   # Re(ψ)
            zi * np.cos(arg) - zr * np.sin(arg),   # Im(ψ)
        ])
        return zk_rot, uk

    def inprocess(self, xk_hat, **kwargs):
        return xk_hat

    def postprocess(self, zk_hat, **kwargs):
        # zk_hat is (N, 2) rotating-frame state [Re(ψ), Im(ψ)]
        # Remodulate and take the real part to recover the lab-frame signal x(t).
        zr = zk_hat[:, 0]
        zi = zk_hat[:, 1]

        times = self.t0 + np.arange(len(zk_hat)) / self.fs
        arg = self["wc"] * times

        x_real = zr * np.cos(arg) - zi * np.sin(arg)   # Re[ψ · e^{iω_c t}]
        return x_real.reshape(-1, 1)


class OscillatorCompositeModel(KalmanCompositeModel):
    """
    Composite of multiple OscillatorModel or AnalyticOscillatorModel instances,
    one per frequency or axis.

    Pre/in/post-processing delegates automatically to each sub-model via
    KalmanCompositeModel._bind_preprocess / _bind_inprocess / _bind_postprocess,
    so no override is needed here.

    State layout (INDEPENDENT obs_mode, N sub-models each with n_i=2, m_i=1):
        x = [pos_0, vel_0,  pos_1, vel_1,  …,  pos_{N-1}, vel_{N-1}]

    Plotting  (record.plot):
        component="position",  dim=k  →  oscillator k position
        component="speed",     dim=k  →  oscillator k speed
    """

    NAME = "Oscillator Composite Model"
    DESCRIPTION = "Composite of OscillatorModel instances, one per frequency or axis."
    VERSION = "1.2"

    @property
    def states(self):
        # All submodels must share the same state enum for this to be meaningful.
        return self.models[0].states if self.models else OscillatorState

    @property
    def observables(self):
        return self.models[0].observables if self.models else OscillatorObservable


# ======================================================================
# Harmonic / inharmonic oscillator
# ======================================================================

class InharmonicOscillatorModel(OscillatorCompositeModel):
    """
    Multi-partial oscillator with optional **inharmonicity**.

    For an ideal string the $n$-th partial is at $n f_1$.  Real strings
    (piano, guitar, bell) have bending stiffness that shifts the partials
    according to the **Fletcher inharmonicity** formula::

        f_n = n · f0 · sqrt(1 + B · n²)

    where $B$ is the dimensionless inharmonicity coefficient.  For a
    cylindrical string it is related to Young's modulus $E$, wire diameter
    $d$, tension $T$, and vibrating length $L$ by::

        B = π³ E d⁴ / (64 T L²)

    ``B = 0`` (default) recovers a perfectly **harmonic** model ($f_n = n f_1$),
    making this class the general case that subsumes the harmonic special case.

    Each partial is tracked by an ``AnalyticOscillatorModel`` in its own rotating
    frame at carrier omega_{c,n} = omega_n, using matrix-exponential discretization
    (unconditionally stable at any frequency).  A broadcast preprocess applies
    Hilbert demodulation independently per partial so the model accepts a single
    real (N, 1) sensor signal — the only physically meaningful input when all
    partials share one measurement channel.

    Parameters
    ----------
    sensor_id : str
        Unique sensor / instrument label.
    f0 : float
        Fundamental frequency [Hz].
    n_harmonics : int
        Number of partials to track (1 … n_harmonics).
    B : float
        Inharmonicity coefficient.  ``B=0`` → perfectly harmonic.
    fs : float
        Sampling frequency [Hz].
    Q_factor : float or list
        Quality factor shared by all partials, or a per-partial list.
    measurement_uncertainty : float
        Observation noise std [same unit as the signal].
    process_uncertainty : float
        State-model mismatch std.
    **kwargs
        Forwarded to each ``AnalyticOscillatorModel`` and ultimately to
        ``KalmanLinearModel.__init__``.

    Attributes
    ----------
    partial_freqs : ndarray, shape (n_harmonics,)
        Actual tracked frequencies $f_n$ [Hz].
    B : float
        Inharmonicity coefficient used at construction.
    f0 : float
        Fundamental frequency [Hz].

    Class methods
    -------------
    fit_inharmonicity(peak_freqs, partial_numbers=None)
        Least-squares fit of ``f0`` and ``B`` from measured peak frequencies.
    partial_frequencies(f0, B, n_harmonics)
        Compute inharmonic partial frequencies for given parameters.

    Examples
    --------
    >>> model = HarmonicOscillatorModel('piano_A4', f0=440.0, n_harmonics=8,
    ...                                 B=2e-4, fs=44100, Q_factor=500,
    ...                                 measurement_uncertainty=0.1,
    ...                                 process_uncertainty=0.001)
    >>> f0_fit, B_fit = InharmonicOscillatorModel.fit_inharmonicity(
    ...     peak_freqs=[440.1, 880.6, 1321.8, 1763.9])
    """

    NAME        = "Inharmonic Oscillator"
    DESCRIPTION = ("Multi-partial analytic oscillator with Fletcher inharmonicity. "
                   "Accepts a single real (N,1) signal; each partial is tracked in "
                   "its own rotating frame via independent Hilbert demodulation. "
                   "f_n = n·f0·sqrt(1 + B·n²).")
    VERSION     = "2.0"

    def __init__(self, sensor_id: str,
                 f0:                    float,
                 n_harmonics:           int   = 6,
                 B:                     float = 0.0,
                 fs:                    float = 44100.0,
                 Q_factor:              float | list = 500.0,
                 measurement_uncertainty: float = 0.1,
                 process_uncertainty:   float = 0.001,
                 **kwargs):

        self.f0 = float(f0)
        self.inharmonicity = float(B)  # 'B' collides with KalmanCompositeModel.B matrix
        self.n_harmonics = int(n_harmonics)

        partial_freqs = self.partial_frequencies(f0, B, n_harmonics)
        self.partial_freqs = partial_freqs

        if isinstance(Q_factor, (int, float)):
            q_list = [float(Q_factor)] * n_harmonics
        else:
            q_list = list(Q_factor)
            if len(q_list) != n_harmonics:
                raise ValueError(
                    f"Q_factor list length {len(q_list)} != n_harmonics {n_harmonics}")

        sub_models = []
        for i, (fn, Qn) in enumerate(zip(partial_freqs, q_list)):
            wn = 2.0 * np.pi * fn
            m = AnalyticOscillatorModel(
                f"{sensor_id}_p{i+1}",
                fs=fs,
                w0=wn,
                wc=wn,         # carrier = partial frequency → Δω = 0 at resonance
                Q=Qn,
                measurement_uncertainty=measurement_uncertainty,
                process_uncertainty=process_uncertainty,
                **kwargs,
            )
            sub_models.append(m)

        super().__init__(*sub_models)  # INDEPENDENT obs_mode: each partial gets its own channel

        # Broadcast preprocess: accept a single real (N,1) signal and feed it to every
        # sub-model, each of which applies Hilbert + demodulation at its own carrier wc.
        # This is the only physically correct input for a single-sensor measurement.
        _models = self.models
        def _broadcast_preprocess(zk, uk=None, **kwargs_):
            parts = [m.preprocess(zk, uk, **kwargs_)[0] for m in _models]
            return np.concatenate(parts, axis=1), uk
        self.preprocess = _broadcast_preprocess

        # KalmanCompositeModel._bind_matrix("L") returns a block-diagonal of zeros
        # (each OscillatorModel.L returns None → zeros in the composite).
        # If left as-is, filter._predict would compute Q = 0@Q@0^T = 0, killing
        # process noise.  Override with None so the filter skips the LQL^T step.
        self.L = lambda x=None, u=None, w=None, t=None: None

    # ------------------------------------------------------------------
    # Extraction helper
    # ------------------------------------------------------------------

    def extract(self, record, signal):
        """
        Reconstruct each partial's real contribution from the filtered state
        and subtract them from ``signal``.

        Parameters
        ----------
        record : KalmanRecord
            Output of ``model.filter()``.  ``record.xk`` shape: (N, 2·n_harmonics).
        signal : array_like, shape (N,)
            Original real mixed signal.

        Returns
        -------
        partials : list of ndarray, each shape (N,)
            Estimated real signal per partial: Re[ψ_i · e^{iω_i t}].
        residual : ndarray, shape (N,)
            ``signal`` with all estimated partials subtracted.
        """
        signal = np.asarray(signal, dtype=float).ravel()
        t = np.arange(len(record.xk)) / self.fs
        partials = []
        for i, m in enumerate(self.models):
            arg = m["wc"] * t
            partials.append(record.xk[:, 2*i] * np.cos(arg) - record.xk[:, 2*i+1] * np.sin(arg))
        return partials, signal - sum(partials)

    # ------------------------------------------------------------------
    # Class helpers
    # ------------------------------------------------------------------

    @staticmethod
    def partial_frequencies(f0: float, B: float, n_harmonics: int) -> np.ndarray:
        """
        Compute Fletcher inharmonic partial frequencies.

        Parameters
        ----------
        f0 : float
            Fundamental frequency [Hz].
        B : float
            Inharmonicity coefficient (dimensionless).
        n_harmonics : int
            Number of partials.

        Returns
        -------
        freqs : ndarray, shape (n_harmonics,)
            $f_n = n \\cdot f_0 \\cdot \\sqrt{1 + B \\cdot n^2}$ [Hz].
        """
        n = np.arange(1, n_harmonics + 1, dtype=float)
        return n * f0 * np.sqrt(1.0 + B * n ** 2)

    @staticmethod
    def inharmonicity_from_material(E: float, d: float, T: float, L: float) -> float:
        """
        Compute inharmonicity coefficient $B$ from string physical parameters.

        Uses the stiff-string formula::

            B = π³ E d⁴ / (64 T L²)

        Parameters
        ----------
        E : float
            Young's modulus [Pa] (e.g. steel ≈ 200e9 Pa).
        d : float
            String diameter [m].
        T : float
            String tension [N].
        L : float
            Vibrating string length [m].

        Returns
        -------
        B : float
            Dimensionless inharmonicity coefficient.
        """
        return (np.pi ** 3 * E * d ** 4) / (64.0 * T * L ** 2)

    @classmethod
    def fit_inharmonicity(cls,
                          peak_freqs,
                          partial_numbers=None) -> tuple[float, float]:
        """
        Fit fundamental frequency ``f0`` and inharmonicity ``B`` from
        measured partial (peak) frequencies via least-squares.

        The model is $f_n = n \\cdot f_1 \\cdot \\sqrt{1 + B \\cdot n^2}$.

        Parameters
        ----------
        peak_freqs : array-like, shape (K,)
            Measured peak frequencies [Hz], in ascending order.
        partial_numbers : array-like of int, optional
            Partial indices (1-based) corresponding to each measured peak.
            Defaults to ``[1, 2, …, K]``.

        Returns
        -------
        f0 : float
            Fitted fundamental frequency [Hz].
        B : float
            Fitted inharmonicity coefficient.

        Raises
        ------
        RuntimeError
            If ``scipy.optimize.curve_fit`` fails to converge.

        Examples
        --------
        >>> f0, B = HarmonicOscillatorModel.fit_inharmonicity(
        ...     [440.0, 880.6, 1321.8, 1763.9, 2207.1])
        """
        peak_freqs = np.asarray(peak_freqs, dtype=float)
        K = len(peak_freqs)
        if partial_numbers is None:
            ns = np.arange(1, K + 1, dtype=float)
        else:
            ns = np.asarray(partial_numbers, dtype=float)
            if len(ns) != K:
                raise ValueError("len(partial_numbers) must equal len(peak_freqs)")

        def model_fn(n, f0, B):
            return n * f0 * np.sqrt(1.0 + B * n ** 2)

        # Initial guess: f0 from first partial, B = 0
        p0 = [peak_freqs[0], 0.0]
        bounds = ([0.0, 0.0], [np.inf, 1.0])

        try:
            popt, _ = curve_fit(model_fn, ns, peak_freqs, p0=p0, bounds=bounds,
                                maxfev=10000)
        except RuntimeError as exc:
            raise RuntimeError(f"fit_inharmonicity did not converge: {exc}") from exc

        return float(popt[0]), float(popt[1])

class HarmonicOscillatorModel(InharmonicOscillatorModel):
    """
    Multi-partial **harmonic** oscillator — special case of
    ``InharmonicOscillatorModel`` with ``B`` permanently set to zero.

    Partial frequencies are exact integer multiples of the fundamental::

        f_n = n · f1   (no inharmonicity)

    Use ``InharmonicOscillatorModel`` directly when you need a non-zero
    inharmonicity coefficient (piano strings, GW detector fibers, etc.).

    Parameters
    ----------
    sensor_id : str
    f1 : float
        Fundamental frequency [Hz].
    n_harmonics : int
        Number of harmonics to track.
    fs : float
    Q_factor : float | list
    measurement_uncertainty : float
    process_uncertainty : float
    analytic : bool
    **kwargs
        Forwarded to ``InharmonicOscillatorModel`` (``B`` is not accepted).
    """

    NAME        = "Harmonic Oscillator (B=0)"
    DESCRIPTION = ("Perfectly harmonic multi-partial oscillator. "
                   "f_n = n·f0 (exact integer multiples). "
                   "For inharmonic partials use InharmonicOscillatorModel.")
    VERSION     = "1.0"

    def __init__(self, sensor_id: str,
                 f0: float,
                 n_harmonics: int = 6,
                 fs: float = 44100.0,
                 Q_factor: float | list = 500.0,
                 measurement_uncertainty: float = 0.1,
                 process_uncertainty: float = 0.001,
                 **kwargs):
        if "B" in kwargs:
            raise TypeError(
                "HarmonicOscillatorModel does not accept 'B' (always 0). "
                "Use InharmonicOscillatorModel for inharmonic partials.")
        super().__init__(sensor_id,
                         f0=f0, n_harmonics=n_harmonics, B=0.0,
                         fs=fs, Q_factor=Q_factor,
                         measurement_uncertainty=measurement_uncertainty,
                         process_uncertainty=process_uncertainty,
                         **kwargs)