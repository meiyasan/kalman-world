# kalman/core/engine_sampler.py
from __future__ import annotations
from .abstract.metadata import KalmanModelMetadata
import numpy as np

from .io.record import KalmanRecord
from .abstract.mixins import KalmanModelMixin, KalmanEngineMixin
from ..helpers import format_shape
from .constants import KalmanStateFlavor

class KalmanSampler:
    """
    Generates synthetic trajectories and measurements from the model:
      - uses model.f(x,u[,w]) and model.h(x[,v])
      - draws noise from model.Q / model.R
      - respects model.fs and x0,P0

    If user requests control with l>0 but model lacks B (or dimension mismatch),
    it raises with a concrete message.
    """

    engine : KalmanEngineMixin = None

    def __init__(self, engine: KalmanEngineMixin):
        self.engine = engine

    @property
    def model(self) -> 'KalmanModelMixin | KalmanModelMetadata':
        return self.engine.model
    @property
    def fs(self) -> float:
        return self.engine.fs

    @property
    def n(self):
        return self.model._infer_n()
    @property
    def m(self):
        return self.model._infer_m()
    @property
    def l(self):
        return self.model._infer_l()

    def measurement(self, duration: float, uk = None, **kwargs):
        '''
        Generate synthetic measurements for a given duration.
        Returns the observation vector with measurement noise added.
        '''

        inprocess = kwargs.pop("inprocess", True)
        postprocess = kwargs.pop("postprocess", True)
        indexing = kwargs.pop("indexing", "xy")

        measurement_noise = kwargs.pop("measurement_noise", True)
        process_noise = kwargs.pop("process_noise", True)
        
        n_samples = int(duration * self.fs)
        zk = np.zeros((n_samples, self.m, 1))
        xk = np.zeros((n_samples, self.n, 1))
        Pk = np.zeros((n_samples, self.n, self.n))

        x, P = kwargs.pop("x0", self.engine.get_initial_state_vector()), kwargs.pop("P0", self.engine.get_initial_state_covariance())
        if x is None or x.shape != (self.n, 1):
            raise ValueError(f"Initial state vector has incorrect dimensions. Expected {self.n}x1, got {format_shape(x)}.")
        if P is None or P.shape != (self.n, self.n):
            raise ValueError(f"Initial state covariance has incorrect dimensions. Expected {self.n}x{self.n}, got {format_shape(P)}.")

        uk = uk if uk is not None else [self.model.u0] * n_samples
        vk = self.measurement_noise(duration, **kwargs) if process_noise else [self.model.v0] * n_samples
        wk = self.process_noise(duration, **kwargs) if measurement_noise else [self.model.w0] * n_samples
        for i in range(n_samples):
            xk[i], Pk[i] = x, P
            x, P, _ = self.engine._predict(x, P, u=uk[i], w=wk[i], **kwargs)

        if inprocess:
            xk = self.model.inprocess(xk, **kwargs)

        zk = KalmanRecord._normalize_out([self.model.h(x, v, self.model.t0 + self.model.duration(k)) for x, v, k in zip(xk, vk, range(n_samples))], indexing=indexing, dtype=self.model.dtype)
        if postprocess:
            zk = self.model.postprocess(zk, **kwargs)

        return KalmanRecord(self.engine.time(duration), zk=zk, xk=xk, Pk=Pk, flavor=KalmanStateFlavor.SAMPLED, engine=self.engine)

    def process_noise(self, duration: float):
        """
        Generate process noise for the given duration.
        If Q is a block diagonal covariance matrix, ensure proper sampling.
        """
            
        n_samples = int(duration * self.fs)
        if self.model.Q() is None: process_noise = np.zeros((n_samples, self.n))
        else: # Check if Q is block diagonal
            if np.allclose(self.model.Q(), np.diag(np.diag(self.model.Q()))):  # Diagonal matrix
                process_noise = np.random.normal(0, np.sqrt(np.diag(self.model.Q())), size=(n_samples, self.n))
            else:
                process_noise = np.random.multivariate_normal(np.zeros(self.n), self.model.Q(), size=n_samples)
        
        process_noise = process_noise.reshape((n_samples, self.n, 1))
        return process_noise

    def measurement_noise(self, duration: float):
        """
        Generate measurement noise for the given duration.
        If R is a block diagonal covariance matrix, ensure proper sampling.
        """

        n_samples = int(duration * self.fs)
        if self.model.R() is None: measurement_noise = np.zeros((n_samples, self.m))
        else: # Check if R is block diagonal
            if np.allclose(self.model.R(), np.diag(np.diag(self.model.R()))):  # Diagonal matrix
                measurement_noise = np.random.normal(0, np.sqrt(np.diag(self.model.R())), size=(n_samples, self.m))
            else:
                measurement_noise = np.random.multivariate_normal(np.zeros(self.m), self.model.R(), size=n_samples)
        
        measurement_noise = measurement_noise.reshape((n_samples, self.m, 1))
        return measurement_noise