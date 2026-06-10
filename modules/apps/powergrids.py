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

    Thin wrapper over ``HarmonicOscillatorModel`` (B=0) with domain-specific
    state/observable labels.  Accepts a single real (N, 1) voltage signal;
    each harmonic is tracked independently in its own rotating frame via
    Hilbert demodulation (handled automatically by the parent's broadcast
    preprocess).

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
    """

    NAME = "Power Grid Harmonic Tracker"
    DESCRIPTION = (
        "HarmonicOscillatorModel (B=0) tracking N harmonics of a power grid "
        "signal (50/60 Hz). Accepts a single real (N,1) voltage channel."
    )
    VERSION = "3.0"

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

        super().__init__(
            sensor_id,
            f0=f0,
            n_harmonics=harmonics,
            fs=fs,
            Q_factor=Q_factor,
            measurement_uncertainty=measurement_uncertainty,
            process_uncertainty=process_uncertainty,
            **kwargs,
        )

