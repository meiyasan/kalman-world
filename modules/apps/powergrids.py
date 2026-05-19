import numpy as np

from ..framework.constants import KalmanState, KalmanObservable
from .oscillators import HarmonicOscillatorModel


class PowerGridState(KalmanState):
    REAL = "re"   # rotating-frame real component (dimensionless amplitude)
    IMAG = "im"   # rotating-frame imaginary component


class PowerGridObservable(KalmanObservable):
    VOLTAGE = "V"   # measured voltage (per harmonic channel: real + imag)


class PowerGridStateModel(HarmonicOscillatorModel):
    """
    Harmonic voltage tracker for power grid signals (50/60 Hz).

    Extends ``HarmonicOscillatorModel`` with ``analytic=True, B=0``:
    power grids are perfectly harmonic ($f_k = k \\cdot f_0$) and each
    harmonic is tracked in its own rotating reference frame via
    ``AnalyticOscillatorModel`` — the correct choice for high-frequency
    narrowband signals.

    State layout (INDEPENDENT obs_mode):
        x = [Re_1, Im_1,  Re_2, Im_2,  ...,  Re_N, Im_N]
        z = [Re_1, Im_1,  Re_2, Im_2,  ...,  Re_N, Im_N]

    The model expects a 2-column analytic-signal input (real + imaginary
    parts of the measured voltage). It internally replicates the signal
    so each harmonic sub-model receives its own (T, 2) copy, which
    ``AnalyticOscillatorModel.preprocess`` then rotates into the appropriate
    rotating frame.

    Parameters
    ----------
    sensor_id : str
        Channel / sensor name (e.g. "L1").
    f0 : float
        Fundamental frequency [Hz], default 50.
    harmonics : int
        Number of harmonics to track (k = 1 … harmonics), default 3.
    fs : float
        Sample rate [Hz], default 1024.
    measurement_uncertainty : float
        Measurement noise std [V], default 1.0.
    process_uncertainty : float
        Process noise std (envelope drift), default 0.01.
    Q_factor : float
        Quality factor for each harmonic oscillator, default 1000.

    Note
    ----
    Extends ``HarmonicOscillatorModel`` which enforces ``B = 0``.
    Power-grid harmonics are ideal integer multiples of ``f0``.

    Usage
    -----
    Analytic signal input — e.g. from scipy.signal.hilbert:
        v_analytic = np.column_stack([v.real, np.imag(signal.hilbert(v))])
        record = grid_model.filter(v_analytic, smooth=True)
    """

    NAME = "Power Grid Harmonic Tracker"
    DESCRIPTION = (
        "HarmonicOscillatorModel (analytic=True, B=0) tracking N harmonics "
        "of a power grid signal (50/60 Hz) in independent rotating frames."
    )
    VERSION = "2.0"

    @property
    def states(self):
        return PowerGridState

    @property
    def observables(self):
        return PowerGridObservable

    def __init__(self, sensor_id: str,
                 f0: float = 50.0,
                 harmonics: int = 3,
                 fs: float = 1024.0,
                 measurement_uncertainty: float = 1.0,
                 process_uncertainty: float = 0.01,
                 Q_factor: float = 1000.0,
                 **kwargs):

        self._n_harmonics = harmonics

        super().__init__(
            sensor_id,
            f0=f0,
            n_harmonics=harmonics,
            analytic=True,  # AnalyticOscillatorModel sub-models (rotating frame)
            fs=fs,
            Q_factor=Q_factor,
            measurement_uncertainty=measurement_uncertainty,
            process_uncertainty=process_uncertainty,
            **kwargs,
        )

        # ------------------------------------------------------------------
        # Override preprocess: broadcast 2-column analytic signal → 2*N cols,
        # then delegate to the composite per-model preprocess chain.
        # ------------------------------------------------------------------
        _composite_preprocess = self.preprocess

        def _pg_preprocess(zk, uk=None, **kwargs_):
            if zk.ndim == 2 and zk.shape[1] == 2:
                zk = np.concatenate([zk] * self._n_harmonics, axis=1)
            return _composite_preprocess(zk, uk, **kwargs_)

        self.preprocess = _pg_preprocess

