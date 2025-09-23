import numpy as np
from enum import Enum
from scipy.linalg import block_diag

from . import KalmanStateModel

class KalmanObservationMode(Enum):
    IDENTITY = "identity" # Full identity (each state observed directly)
    LINEAR   = "linear"   # Horizontal blocks for H and list for R (combined observations)
    CUSTOM   = "custom"   # User provides explicit H and R

class KalmanCompositeModel(KalmanStateModel):
    """
    Composite Kalman Model that aggregates multiple Kalman models into one system.
    """

    def __init__(self, *models, mode=KalmanObservationMode.CUSTOM, H=None, R=None):
        if not models:
            raise ValueError("At least one model must be provided.")
        
        self.models = models
        self.mode = mode

        self.fs = models[0].fs
        for m in models:
            if m.fs != self.fs:
                raise ValueError("All models must have the same sampling rate.")
        
        # Build block-diagonal dynamics
        F = block_diag(*[m.F for m in models])
        Q = block_diag(*[m.Q for m in models])
        x0 = np.concatenate([m.x0.flatten() for m in models])
        P0 = block_diag(*[m.P0 for m in models])

        # Define H and R based on observation mode
        if mode == KalmanObservationMode.IDENTITY:
            H = block_diag(*[m.H for m in models])
            R = block_diag(*[m.R for m in models])
        elif mode == KalmanObservationMode.LINEAR:
            H = np.hstack([m.H for m in models])       # Stack H matrices horizontally
            R = sum(m.R for m in models) / len(models) # Average of all R matrices
        elif mode == KalmanObservationMode.CUSTOM:
            if H is None or R is None:
                raise ValueError("For CUSTOM mode, H and R must be provided by the user.")
        else:
            raise ValueError(f"Unknown observation mode: {mode}")

        super().__init__(F=F, H=H, Q=Q, R=R, x0=x0, P0=P0)

    def get_observation_matrix(self, dim=None):
        """
        Return the observation matrix (H) or a masked version for submodel `dim`.

        Parameters
        ----------
        dim : int or None
            If None: return full observation matrix H.
            If int: return a masked matrix where only columns for submodel `dim`
                    are kept; others are zeroed.

        Returns
        -------
        np.ndarray
            Observation matrix or masked matrix.
        """
        if dim is None:
            return self.H
        if dim < 0 or dim >= len(self.sub_dims):
            raise ValueError(f"Invalid submodel index: {dim}")

        H_masked = np.zeros_like(self.H)
        start = sum(self.sub_dims[:dim])
        end = start + self.sub_dims[dim]

        H_masked[:, start:end] = self.H[:, start:end]
        return H_masked

    def get_mapping(self, meas_idx, axis=None):
        """
        Returns the list of state indices that contribute to measurement[meas_idx].
        If axis is given, meas_idx is local to that submodel.
        """
        if axis is not None:
            start = sum(self.sub_dims[:axis])
            meas_idx = start + meas_idx
        meas_idx = meas_idx.value if hasattr(meas_idx, 'value') else int(meas_idx)
        assert 0 <= meas_idx < self.m, f"Measurement index out of range. Expected 0 <= idx < {self.m}, got {meas_idx}."
        return np.nonzero(self.H[meas_idx, :])[0]

    def exclusive_mapping(self, state_idx, measurement_idx, axis=None):
        """
        Check if measurement[measurement_idx] exclusively depends on the specified state.
        If axis is given, state_idx is local to that submodel.
        """
        if axis is not None:
            start = sum(self.sub_dims[:axis])
            state_idx = start + state_idx
        coeff = self.H[measurement_idx, state_idx]
        if coeff == 0:
            return False
        return np.count_nonzero(self.H[measurement_idx, :]) == 1

    def inclusive_mapping(self, state_idx, measurement_idx, axis=None):
        """
        Check if state influences the measurement in any way.
        """
        if axis is not None:
            start = sum(self.sub_dims[:axis])
            state_idx = start + state_idx
        return self.H[measurement_idx, state_idx] != 0
