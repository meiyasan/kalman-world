import numpy as np
import matplotlib.pyplot as plt
from enum import Enum

from ..core import KalmanStateModel, KalmanCompositeModel
class OscillatorStateType(Enum):
    REAL     = 0
    IMAG     = 1

class OscillatorStateModel(KalmanStateModel):

    NAME = "Oscillator State Model"
    DESCRIPTION = "State model for oscillatory motion, simulating the behavior of a simple harmonic oscillator."
    VERSION = "1.0"

    def __init__(self, **kwargs):
        measurement_uncertainty = kwargs.pop("measurement_uncertainty", 0.5)
        process_uncertainty = kwargs.pop("process_uncertainty", 0.05)

        self.fs = kwargs.pop("fs", None) or kwargs.pop("sample_rate", None) or self.fs
        self.measurement_uncertainty = measurement_uncertainty
        self.process_uncertainty = process_uncertainty

        dt = 1.0 / self.fs
        A = np.array([[1, dt], 
                      [-dt, 1]])
        
        C = np.array([1, 0]).reshape(1, 2)
        V = np.array([measurement_uncertainty]).reshape(1, 1)
        W = process_uncertainty * np.array([[1.0, 0.0], 
                                            [0.0, 1.0]])

        x0, P0 = kwargs.pop("x0", None), kwargs.pop("P0", None)
        super().__init__(F=A, H=C, Q=W, R=V, P0=P0, x0=x0, **kwargs)

    def preprocess(self, zk, uk=None, **kwargs):
        return zk, uk

    def inprocess(self, zk_hat, **kwargs):
        return zk_hat

    def postprocess(self, zk_hat, **kwargs):
        return zk_hat
    
    def plot(self, *args, **kwargs):
        t = self.time()
        if t is None:
            return

        measurements = self.get_measurement_vector()
        state_vector = self.get_posterior_state_vector()
        smooth_state_vector = self.get_smooth_state_vector()

        args = list(args)
        plt.figure(figsize=(10, 5))

        if measurements is not None:
            plt.scatter(t, measurements, label='Data', color='black', marker='+')
        
        truth = self.truth(duration=self.duration())
        if truth is not None:
            plt.plot(t, truth[:, 0], '--', label='Truth', color='lightblue')

        plt.plot(t, state_vector[:, 0], label='State Estimate', color='orange')
        if smooth_state_vector is not None:
            plt.plot(t, smooth_state_vector[:, 0], label='Smoothed Estimate', color='green')

class OscillatorCompositeModel(KalmanCompositeModel):

    NAME = "Oscillator Composite Model"
    DESCRIPTION = "Composite model for oscillatory motion, combining multiple oscillators."
    VERSION = "1.0"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def preprocess(self, zk, uk=None, **kwargs):
        return zk, uk

    def inprocess(self, zk_hat, **kwargs):
        return zk_hat

    def postprocess(self, zk_hat, **kwargs):
        return zk_hat