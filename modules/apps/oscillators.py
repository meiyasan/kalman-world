import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

from ..models.linear import KalmanLinearModel

from ..framework.model import KalmanModel
from ..models.composite import KalmanCompositeModel
from ..framework.constants import KalmanState, KalmanObservation

class OscillatorState(KalmanState):
    POSITION = 0
    SPEED = 1

class OscillatorObservation(KalmanObservation):
    POSITION = 0

class OscillatorModel(KalmanLinearModel):

    NAME = "Oscillator State Model"
    DESCRIPTION = "State model for oscillatory motion, simulating the behavior of a simple harmonic oscillator."
    VERSION = "1.0"

    @property
    def type(self):
        return OscillatorState

    units = {str(OscillatorState.POSITION): "m",
             str(OscillatorState.SPEED): "m/s"}

    def __init__(self, sensor_id, **kwargs):

        measurement_uncertainty = kwargs.pop("measurement_uncertainty", 0.5)
        process_uncertainty = kwargs.pop("process_uncertainty", 0.05)

        self.fs = kwargs.pop("fs", None) or kwargs.pop("sample_rate", None) or self.fs
        self.measurement_uncertainty = measurement_uncertainty
        self.process_uncertainty = process_uncertainty

        dt = 1.0 / self.fs
        w0 = kwargs.get("w0", 1)
        Q = kwargs.pop("Q", None)
        b = w0/Q if Q is not None else 0
        
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

    def inprocess(self, zk_hat, **kwargs):
        return zk_hat

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







class AnalyticOscillatorState(KalmanState):
    REAL = 0
    IMAG = 1
    
class AnalyticOscillatorObservation(KalmanObservation):
    REAL = 0
    IMAG = 1

class AnalyticOscillatorModel(KalmanLinearModel):

    NAME = "Analytic Oscillator State Model (x + iy representation)"
    DESCRIPTION = "State model for oscillatory motion, simulating the behavior of a simple harmonic oscillator."
    VERSION = "1.0"

    @property
    def type(self):
        return AnalyticOscillatorState

    units = {str(AnalyticOscillatorState.REAL): "m",
             str(AnalyticOscillatorState.IMAG): "m"}

    def __init__(self, name, **kwargs):
        measurement_uncertainty = kwargs.pop("measurement_uncertainty", 0.5)
        process_uncertainty = kwargs.pop("process_uncertainty", 0.05)

        self.fs = kwargs.pop("fs", None) or kwargs.pop("sample_rate", None) or self.fs
        self.measurement_uncertainty = measurement_uncertainty
        self.process_uncertainty = process_uncertainty

        dt = 1.0 / self.fs
        w0 = kwargs.get("w0", 1)
        self.wc = kwargs.get("wc", 1)
        # wc = self.wc
        delta_omega = np.power(w0, 1)# - np.power(wc, 1)
        Q = kwargs.pop("Q", None) # quality factor (not Q matrix)

        # the state-transition model
        A = expm(np.array([[-w0 / (2 * Q) * dt, -delta_omega * dt],
                           [delta_omega * dt,  -w0 / (2 * Q) * dt]]))
        
        # the observeation model
        C = np.identity(2)
        # the covariance of the process noise
        R = measurement_uncertainty * np.sqrt(2) * np.identity(2)
        # the covariance of the observation noise
        Q = process_uncertainty * np.sqrt(2) * np.identity(2)

        x0, P0 = kwargs.pop("x0", None), kwargs.pop("P0", None)
        super().__init__(name, F=A, H=C, Q=Q, R=R, P0=P0, x0=x0, **kwargs)

    def preprocess(self, zk, uk=None, **kwargs):
        zr = zk[:, 0, :].reshape((zk.shape[0], 1, zk.shape[2]))
        zi = zk[:, 1, :].reshape((zk.shape[0], 1, zk.shape[2]))

        t_start = self.time()[0] if self.time().size > 0 else self.t0
        times = t_start + np.arange(0, zk.shape[0], 1)/self.fs
        arg = self.wc * times
        arg = arg.reshape((len(times), 1, zk.shape[2]))

        zk = (np.array([zr * np.cos(arg) + zi * np.sin(arg),
                        zi * np.cos(arg) - zr * np.sin(arg)])
              .transpose((1, 0, 2, 3))
              .squeeze(axis=-1))

        return zk, uk

    def inprocess(self, zk_hat, **kwargs):
        return zk_hat

    def postprocess(self, zk_hat, **kwargs):
        zr = zk_hat[:, 0, :].reshape((zk_hat.shape[0], 1, zk_hat.shape[2]))
        zi = zk_hat[:, 1, :].reshape((zk_hat.shape[0], 1, zk_hat.shape[2]))

        t_start = self.time()[0] if self.time().size > 0 else self.t0
        times = t_start + np.arange(0, zk_hat.shape[0], 1)/self.fs
        arg = self.wc * times
        arg = arg.reshape((len(times), 1, zk_hat.shape[2]))
        
        zk_hat = (np.array([zr * np.cos(arg) - zi * np.sin(arg),
                            zi * np.cos(arg) + zr * np.sin(arg)])
              .transpose((1, 0, 2, 3))
              .squeeze(axis=-1))

        # zk_hat = np.zeros(zk_hat.shape)
        return zk_hat
    
    def test_invariance(self, zk, **kwargs):
        
        zk_hat, uk = self.preprocess(zk, **kwargs)
        zk_trans = self.postprocess(zk_hat, **kwargs)
        
        return np.allclose(zk, zk_trans, rtol=1e-8, atol=1e-10)
        

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

class AnalyticOscillatorCompositeModel(KalmanCompositeModel):

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