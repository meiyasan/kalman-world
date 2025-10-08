import numpy as np
from ..core import KalmanStateModel, KalmanExtendedModel, KalmanCompositeModel
from ..core.io import KalmanStateType

class BallisticStateType(KalmanStateType):
    POSITION     = 0
    SPEED        = 1
    ACCELERATION = 2

class BallisticStateModel(KalmanStateModel):
    
    NAME = "Ballistic State Model"
    DESCRIPTION = "State model for ballistic trajectories, simulating the motion of a projectile under the influence of gravity with constant acceleration."
    VERSION = "1.0"

    @property
    def type(self):
        return BallisticStateType

    axis  = ["X", "Y", "Z"]
    units = {
        str(BallisticStateType.POSITION): "m",        
        str(BallisticStateType.SPEED): "m/s",  
        str(BallisticStateType.ACCELERATION): "m/s²"
    }

    def __init__(self, dim: int = 1, measurement_uncertainty: float = 0.5, process_uncertainty: float = 0.05, **kwargs):

        assert dim in (1, 2, 3), f"Dimension must be 1, 2 or 3, got {dim}"

        self.dim = dim
        if dim < 2:
            self.axis = ["Z"]

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

class BallisticCompositeModel(KalmanCompositeModel):

    NAME = "Ballistic Composite Model"
    DESCRIPTION = "State model for ballistic trajectories, simulating the motion of a projectile under the influence of gravity with non uniform acceleration."
    VERSION = "1.0"
    
    @property
    def type(self):
        return BallisticStateType

    axis       = []
    units      = BallisticStateModel.units

    def __init__(self, dim):

        for i in range(dim):
            self.axis.append(f"X{i}")

        pass

    def preprocess(self, zk, uk=None, **kwargs):
        return zk, uk

    def inprocess(self, zk_hat, **kwargs):
        return zk_hat

    def postprocess(self, zk_hat, **kwargs):
        return zk_hat

class BallisticExtendedModel(KalmanExtendedModel):

    NAME = "Ballistic Extended Model"
    DESCRIPTION = "State model for ballistic trajectories, simulating the motion of a projectile under the influence of gravity with non uniform acceleration."
    VERSION = "1.0"

    @property
    def type(self):
        return BallisticStateType

    def __init__(self):
        pass

    def preprocess(self, zk, uk=None, **kwargs):
        return zk, uk

    def inprocess(self, zk_hat, **kwargs):
        return zk_hat

    def postprocess(self, zk_hat, **kwargs):
        return zk_hat