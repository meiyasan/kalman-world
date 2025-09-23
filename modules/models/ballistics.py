import numpy as np
from ..core import KalmanStateModel, KalmanExtendedModel, KalmanOutputType, KalmanStateType

class BallisticStateType(KalmanStateType):
    POSITION     = 0
    SPEED        = 1
    ACCELERATION = 2

class BallisticStateModel(KalmanStateModel):
    
    NAME = "Ballistic State Model"
    DESCRIPTION = "State model for ballistic trajectories, simulating the motion of a projectile under the influence of gravity with constant acceleration."
    VERSION = "1.0"

    labels  = {
        BallisticStateType.POSITION: "Position", 
        BallisticStateType.SPEED: "Speed",
        BallisticStateType.ACCELERATION: "Acceleration"
    }

    units  = {
        BallisticStateType.POSITION: "m",        
        BallisticStateType.SPEED: "m/s",  
        BallisticStateType.ACCELERATION: "m/s²"
    }

    coords = ["X", "Y", "Z"]
    colors = {
        KalmanOutputType.ESTIMATE:    {0:"darkred",    1:"darkgreen",    2:"darkblue"},
        KalmanOutputType.MEASUREMENT: {0:"rosybrown",  1:"darkseagreen", 2:"lightsteelblue"},
        KalmanOutputType.TRUTH:       {0:"lightcoral", 1:"lightgreen",   2:"blue"},
        KalmanOutputType.IDEAL:       {0:"lightcoral", 1:"lightgreen",   2:"blue"}
    }
    
    linestyles = {
        KalmanOutputType.ESTIMATE:    {0:"-",  1:"-",  2:"-"},
        KalmanOutputType.MEASUREMENT: {0:"o",  1:"o",  2:"o"},
        KalmanOutputType.TRUTH:       {0:"--", 1:"--", 2:"--"},
        KalmanOutputType.IDEAL:       {0:":",  1:":",  2:":"}
    }

    def __init__(self, dim: int = 1, measurement_uncertainty: float = 0.5,
                       process_uncertainty: float = 0.05, **kwargs):
        assert dim in (1, 2, 3), f"Dimension must be 1, 2 or 3, got {dim}"

        self.dim = dim
        self.measurement_uncertainty = measurement_uncertainty
        self.process_uncertainty     = process_uncertainty

        self.fs = kwargs.pop("fs", None) or kwargs.pop("sample_rate", None) or getattr(self, "fs", None)
        assert isinstance(self.fs, (int, float)), "You must provide fs or sample_rate"
        dt = 1.0 / self.fs

        # 1D building blocks
        F1 = np.array([[1, dt,  0],
                       [0,  1, dt],
                       [0,  0,  1]])

        H1 = np.array([1, 0, 0]).reshape(1,3)
        Q1 = process_uncertainty * np.array([[1,1,0],
                                             [1,1,0],
                                             [0,0,0]])

        R1 = measurement_uncertainty * np.eye(1)

        # expand to full dimension
        F = np.kron(np.eye(dim), F1)
        H = np.kron(np.eye(dim), H1)
        Q = np.kron(np.eye(dim), Q1)
        R = np.kron(np.eye(dim), R1)

        x0 = kwargs.pop("x0", None)
        P0 = kwargs.pop("P0", None)
        super().__init__(F=F, H=H, Q=Q, R=R, x0=x0, P0=P0, **kwargs)
    
    def preprocess(self, zk, uk=None, **kwargs):
        return zk, uk

    def inprocess(self, zk_hat, **kwargs):
        return zk_hat

    def postprocess(self, zk_hat, **kwargs):
        return zk_hat
    
    def get_mapping(self, meas_idx: KalmanStateType, axis = 0):
        """
        Returns the list of state indices that contribute to measurement[meas_idx].
        """
        meas_idx = meas_idx.value if hasattr(meas_idx, 'value') else int(meas_idx)
        idx = self.dim * axis + meas_idx

        assert 0 <= idx < self.dim * self.m, f"Measurement index out of range. Expected 0 <= idx < {self.dim * self.m}, got {idx}."
        return idx

    def plot(self, **kwargs):

        """
        Public plot method. Dispatches to _plot3D if self.dim > 2 and axis=None,
        otherwise falls back to _plot2D.
        """
        t = self.time()
        if t is None or len(t) == 0:
            return None, None

        # Mode parsing
        component = kwargs.pop("component", BallisticStateType.POSITION)
        component = self._normalize_component(component, type=BallisticStateType)
        dim = kwargs.get("dim", None)

        if dim is None and self.dim > 2:
            fig, ax = self._plot3D(component, **kwargs)
        elif dim is None and self.dim > 1:
            fig, ax = self._plot2D(component, **kwargs)
        else:
            fig, ax = self._plot1D(component, **kwargs)
        
        ax.set_title(kwargs.pop("title", f"Ballistic Trajectory"))
        return fig, ax

class BallisticExtendedModel(KalmanExtendedModel):

    NAME = "Ballistic Extended State Model"
    DESCRIPTION = "State model for ballistic trajectories, simulating the motion of a projectile under the influence of gravity with non uniform acceleration."
    VERSION = "1.0"

    def __init__():
        pass