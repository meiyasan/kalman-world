from abc import ABC, abstractmethod

from ..helpers import *
from ..debug import get_debug

import matplotlib.pyplot as plt
import numpy as np
import warnings
import json
import yaml

from enum import Enum
class KalmanStateType(Enum):
    def __int__(self):
        return self.value
    
class KalmanOutputType(Enum):
    IDEAL   = 0
    TRUTH = 1
    MEASUREMENT  = 2
    ESTIMATE     = 3

    def __int__(self):
        return self.value

class KalmanStateModel(ABC):

    NAME = "Kalman State Model"
    DESCRIPTION = "Base class for Kalman state models, providing a framework for state estimation and filtering."
    VERSION = "1.0"

    @abstractmethod # Preprocess the zk data before passing it to the Kalman filter
    def preprocess(self, zk, uk = None, **kwargs):
        pass
    @abstractmethod # Inprocess state vector to a different representation or computing additional operations (like filtering windowing or smoothing, etc.)
    def inprocess(self, zk_hat, **kwargs):
        pass
    @abstractmethod # Postprocess the zk_hat data after Kalman filter processing
    def postprocess(self, zk_hat, **kwargs):
        pass

    def __init__(self, **kwargs):

        self.random_seed = kwargs.pop("random_seed", None)
        if self.random_seed: set_random_seed(self.random_seed)

        self.t0 = kwargs.pop("t0", 0.0)
        self.fs = kwargs.pop("fs", None) or kwargs.pop("sample_rate", None) or getattr(self, "fs", None)
        assert self.fs is not None or not isinstance(self.fs, (int, float)), "Sampling frequency must be a number."
        
        F = kwargs.pop("F", None)
        assert F is not None, "The system dynamics matrix is required. Please provide a valid F matrix."
        self.F = F = np.asarray(F)

        H = kwargs.pop("H", None)
        assert H is not None, "The observation matrix is required. Please provide a valid H matrix."
        self.H = H = np.asarray(H)

        B = kwargs.pop("B", None)
        assert B is None or B.shape[0] == F.shape[0], f"B matrix has incorrect shape. Expected {F.shape[0]} rows, got {format_shape(B)} rows."
        Q = kwargs.pop("Q", None)
        assert Q is None or Q.shape == (F.shape[0], F.shape[0]), f"Q matrix has incorrect shape. Expected {format_shape(F)} shape, got {format_shape(Q)}."
        R = kwargs.pop("R", None)
        assert R is None or R.shape == (H.shape[0], H.shape[0]), f"R matrix has incorrect shape. Expected {format_shape(H)} shape, got {format_shape(R)}."

        P0 = kwargs.pop("P0", None)
        assert P0 is None or P0.shape == (F.shape[0], F.shape[0]), f"P0 matrix has incorrect shape. Expected {format_shape(F)} shape, got {format_shape(P0)}."
        x0 = kwargs.pop("x0", None)
        assert x0 is None or x0.shape == (F.shape[0], 1), f"x0 vector has incorrect shape. Expected {F.shape[0]} elements, got {format_shape(x0)} elements."

        self.n = F.shape[0]  # state dimension
        self.m = H.shape[0]  # output dimension
        self.l = B.shape[1] if B is not None else 1 # control input dimension

        self.Q = np.eye(self.n) if Q is None else np.asarray(Q)
        self.R = np.eye(self.m) if R is None else np.asarray(R)
        self.B = np.zeros((self.n, self.l)) if B is None else np.asarray(B)

        self.initial_state(
            np.zeros((self.n)) if x0 is None else np.asarray(x0),
            self.Q if P0 is None else np.asarray(P0)
        )

        self.reset()

    def __len__(self):
        """
        Returns the number of posterior estimates.
        """
        return len(self.posterior) if self.posterior else 0

    def print(self):
        """
        Print a summary of the Kalman state model.
        """
        print(self.__str__())

    def reset_with_latest(self):
        """
        Reset the model state to the latest estimates erasing all previous estimates.
        """
        self.reset(with_latest=True)

    def reset(self, with_latest=False):

        if with_latest:
            self.t0 = self.t0 + len(self.posterior) / self.fs
            self.P0 = self.get_latest_state_covariance()
            self.x0 = self.get_latest_state_vector()

        self.P = self.P0
        self.x = self.x0

        self.prior = []
        self.posterior = []
        self.smooth_posterior = []

        self.measurement = []
        self.control = []

        self.psik_hat = []
        self.gk_hat = []

    def time(self, duration: float = None, k = None):
        """
        Returns the time vector based on the current sample rate.
        If k is provided, returns the time of the k-th sample considering t0.
        """
        if k is not None:
            return self.t0 + k / self.fs

        duration = duration or len(self.posterior) / self.fs
        n_samples = int(duration * self.fs)
        return self.t0 + np.arange(n_samples) / self.fs if n_samples > 0 else np.array([])

    def ideal(self, duration: float = None):
        _, psik = self.sample(duration, process_noise=False, measurement_noise=False)
        return psik

    def truth(self, duration: float = None):
        _, psik = self.sample(duration, process_noise=True, measurement_noise=False)
        return psik

    def duration(self):
        """
        Returns the duration of the current state estimation.
        """
        return len(self.posterior) / self.fs if self.fs else 0
    
    def _preprocess(self, zk, uk = None, **kwargs):
        """
        Preprocess the zk data before passing it to the Kalman filter.
        Override this method in subclasses to implement specific preprocessing steps.
        """
        # Matrix dimension checks
        if zk.shape[1] != self.m:
            raise ValueError(f"zk data has incorrect number of columns. Expected {self.m}, got {format_shape(zk)}.")

        zk, uk = self.preprocess(zk, uk, **kwargs)

        # Ensure zk has valid dimensions
        if zk.shape[1] != self.m:
            raise ValueError(f"State vector data has incorrect number of columns. Expected {self.m}, got {format_shape(zk)}.")

        if uk is not None and uk.shape[1] != self.l:
            raise ValueError(f"Control data data has incorrect number of columns. Expected {self.l}, got {format_shape(uk)}.")

        return zk, uk

    def _inprocess(self, state_vector, **kwargs):
        """
        Transform the state vector to a different representation.
        Override this method in subclasses to implement specific transformations.
        """

        if state_vector.shape[1] != self.n:
            raise ValueError(f"State vector has incorrect dimensions. Expected {self.n}, got {format_shape(state_vector)}.")

        state_vector = self.inprocess(state_vector, **kwargs)
        if state_vector.shape[1] != self.n:
            raise ValueError(f"Transformed state vector has incorrect dimensions. Expected {self.n} size, got {format_shape(state_vector)}.")

        return state_vector
    
    def _postprocess(self, psik_hat, **kwargs):

        zk_hat = np.array([self.H @ psi_hat for psi_hat in psik_hat])
        gk_hat = self.postprocess(zk_hat, **kwargs)

        # Matrix dimension checks
        if gk_hat.shape[0] != len(gk_hat):
            raise ValueError(f"Postprocessed result has incorrect number of rows. Expected {len(gk_hat)}, got {gk_hat.shape[0]}.")
        if gk_hat.shape[1] != self.m:
            raise ValueError(f"Postprocessed result has incorrect number of columns. Expected {self.m}, got {gk_hat.shape[1]}.")

        return gk_hat

    def __str__(self):

        import textwrap
        opt = np.get_printoptions()
        np.set_printoptions(linewidth=np.inf)

        def _print_dimensions(name, matrix):
            if matrix is None:
                return f"{name}: None"
            if isinstance(matrix, np.ndarray):
                shape = "x".join(map(str, matrix.shape))
                dims = "dim" if len(matrix.shape) == 1 and matrix.shape[0] == 1 else "dims"
                return f"{name}: {shape} {dims}"
            if isinstance(matrix, (list, tuple)):
                def get_shape(data):
                    if isinstance(data, (list, tuple)):
                        return [len(data)] + get_shape(data[0]) if len(data) > 0 else [0]
                    return []
                shape = "x".join(map(str, get_shape(matrix)))
                return f"{name}: {shape}"
            return f"{name}: Unsupported type ({type(matrix).__name__})"

        def _print(matrix, prefix=""):
            return (str(matrix).replace("\n", f"\n{prefix}"))

        spacer = " " * 4
        description = textwrap.fill(self.DESCRIPTION, width=50, subsequent_indent=2*spacer)

        result  = ""
        result += f"KalmanStateModel(fs = {self.fs}Hz):\n\n"
        result += f"{1*spacer}{self.NAME} (v{self.VERSION})\n"
        result += f"{2*spacer}{description}\n\n"

        result += f"{1*spacer}Dimensions Summary:\n"
        result += f"{2*spacer}State Vector....................................: {self.n} dim{'s' if self.n != 1 else ''}\n"
        result += f"{2*spacer}Control Input...................................: {self.l} dim{'s' if self.l != 1 else ''}\n"
        result += f"{2*spacer}Observation.....................................: {self.m} dim{'s' if self.m != 1 else ''}\n"
        result += f"\n"

        result += f"{1*spacer}Initial State:\n"
        result += f"{2*spacer}{_print_dimensions('State Vector (x0)...............................', self.x0)}\n"
        result += f"{2*spacer}" + _print(self.x0, prefix=2*spacer) + "\n\n" if get_debug() else ""
        result += f"{2*spacer}{_print_dimensions('State Covariance (P0)...........................', self.P0)}\n"
        result += f"{2*spacer}" + _print(self.P0, prefix=2*spacer) + "\n" if get_debug() else ""
        result += f"\n"

        result += f"{1*spacer}Matrix Information:\n"
        result += f"{2*spacer}{_print_dimensions('System Dynamics F (also known as A).............', self.F)}\n"
        result += f"{2*spacer}" + _print(self.F, prefix=2*spacer) + "\n\n" if get_debug() else ""
        result += f"{2*spacer}{_print_dimensions('Observation Matrix H (also known as C)..........', self.H)}\n"
        result += f"{2*spacer}" + _print(self.H, prefix=2*spacer) + "\n" if get_debug() else ""
        result += f"\n"

        result += f"{2*spacer}{_print_dimensions('Control Input B (also known as U)...............', self.B)}\n"
        result += f"{2*spacer}" + _print(self.B, prefix=2*spacer) + "\n\n" if get_debug() else ""
        result += f"{2*spacer}{_print_dimensions('Process Noise Covariance Q (also known as W)....', self.Q)}\n"
        result += f"{2*spacer}" + _print(self.Q, prefix=2*spacer) + "\n\n" if get_debug() else ""
        result += f"{2*spacer}{_print_dimensions('Measurement Noise Covariance R (also known as V)', self.R)}\n"
        result += f"{2*spacer}" + _print(self.R, prefix=2*spacer) + "\n" if get_debug() else ""

        np.set_printoptions(**opt)
        return result

    def __repr__(self):
        return self.__str__()
    def __call__(self, gk, uk=None, **kwargs):
        return self.filter(gk, uk, **kwargs)
    def __len__(self):
        return len(self.posterior)
    def __getitem__(self, i):
        if i < 0 or i >= len(self.posterior):
            raise IndexError("Index out of range.")
        return self.get_posterior_state_vector(i), self.get_posterior_state_covariance(i)
    
    @staticmethod
    def read(filename):
        if filename.endswith('.json'):
            with open(filename, 'r') as file:
                data_dict = json.load(file)
        elif filename.endswith('.npz'):
            data = np.load(filename, allow_pickle=True)
            data_dict = {key: data[key].tolist() if isinstance(data[key], np.ndarray) else data[key] for key in data}
        elif filename.endswith('.yaml') or filename.endswith('.yml'):
            with open(filename, 'r') as file:
                data_dict = yaml.safe_load(file)
        elif filename.endswith('.xml'):
            import xml.etree.ElementTree as ET
            tree = ET.parse(filename)
            root = tree.getroot()
            data_dict = {child.tag: filter(child.text) for child in root}
        else:
            raise ValueError("Unsupported file format. Use '.json', '.npz', '.yaml', or '.xml'.")

        model = KalmanStateModel(
            fs=data_dict.get("fs"),
            F=np.array(data_dict.get("F")),
            H=np.array(data_dict.get("H"))
        )

        for key, value in data_dict.items():
            if key not in ["fs", "F", "H"]:
                setattr(model, key, np.array(value) if isinstance(value, list) else value)

        return model

    def write(self, filename):
        data_dict = {key: (value.tolist() if isinstance(value, np.ndarray) else value)
                     for key, value in self.__dict__.items()}

        if filename.endswith('.json'):
            with open(filename, 'w') as file:
                json.dump(data_dict, file, indent=4)
        elif filename.endswith('.npz'):
            np.savez(filename, **data_dict)
        elif filename.endswith('.yaml') or filename.endswith('.yml'):
            with open(filename, 'w') as file:
                yaml.safe_dump(data_dict, file, default_flow_style=False)
        elif filename.endswith('.xml'):
            import xml.etree.ElementTree as ET
            root = ET.Element("KalmanStateModel")
            for key, value in data_dict.items():
                child = ET.SubElement(root, key)
                child.text = repr(value)
            tree = ET.ElementTree(root)
            tree.write(filename)
        else:
            raise ValueError("Unsupported file format. Use '.json', '.npz', '.yaml', or '.xml'.")

    def sample_state(self, duration: float, **kwargs):
        _, psik, psik_cov = self.sample_measurement(duration, **kwargs)
        return psik, psik_cov
    def sample_state_vector(self, duration: float, **kwargs):
        return self._get_tuple(self.sample_state(duration, **kwargs), ith=0)
    def sample_state_covariance(self, duration: float, **kwargs):
        return self._get_tuple(self.sample_state(duration, **kwargs), ith=1)
        
    def sample_control(self, duration: float, **kwargs):

        """
        Generate synthetic control input for a given duration.
        Returns a control input vector of shape (n_samples, l).
        """
            
        n_samples = int(duration * self.fs)
        control_input = np.random.normal(
            0, 1, (n_samples, self.l)
        ) if self.l > 0 else np.zeros((n_samples, 1))
        
        return control_input
    
    def sample_process_noise(self, duration: float, **kwargs):
        """
        Generate process noise for the given duration.
        If Q is a block diagonal covariance matrix, ensure proper sampling.
        """
            
        n_samples = int(duration * self.fs)
        if self.Q is None: process_noise = np.zeros((n_samples, self.n))
        else: # Check if Q is block diagonal
            if np.allclose(self.Q, np.diag(np.diag(self.Q))):  # Diagonal matrix
                process_noise = np.random.normal(0, np.sqrt(np.diag(self.Q)), size=(n_samples, self.n))
            else:
                process_noise = np.random.multivariate_normal(np.zeros(self.n), self.Q, size=n_samples)
        
        process_noise = process_noise.reshape((n_samples, self.n, 1))
        return process_noise

    def sample_measurement_noise(self, duration: float, **kwargs):
        """
        Generate measurement noise for the given duration.
        If R is a block diagonal covariance matrix, ensure proper sampling.
        """

        n_samples = int(duration * self.fs)
        if self.R is None: measurement_noise = np.zeros((n_samples, self.m))
        else: # Check if R is block diagonal
            if np.allclose(self.R, np.diag(np.diag(self.R))):  # Diagonal matrix
                measurement_noise = np.random.normal(0, np.sqrt(np.diag(self.R)), size=(n_samples, self.m))
            else:
                measurement_noise = np.random.multivariate_normal(np.zeros(self.m), self.R, size=n_samples)
        
        measurement_noise = measurement_noise.reshape((n_samples, self.m, 1))
        return measurement_noise

    def sample_measurement(self, duration: float, **kwargs):
        """
        Generate synthetic measurements for a given duration.
        Returns the state vector, the observation combined using self.H, and the state covariance.
        """        

        n_samples = int(duration * self.fs)
        zk = np.zeros((n_samples, self.m, 1))
        psik = np.zeros((n_samples, self.n, 1))
        psik_cov = np.zeros((n_samples, self.n, self.n))

        x, P = kwargs.pop("x0", self.get_initial_state_vector()), kwargs.pop("P0", self.get_initial_state_covariance())
        if x.shape != (self.n, 1):
            raise ValueError(f"Initial state vector has incorrect dimensions. Expected {self.n}x1, got {format_shape(x)}.")
        if P.shape != (self.n, self.n):
            raise ValueError(f"Initial state covariance has incorrect dimensions. Expected {self.n}x{self.n}, got {format_shape(P)}.")

        for i in range(n_samples):
            psik[i] = x
            psik_cov[i] = P
            zk[i] = self.H @ x
            x, P = self.__predict(x, P, **kwargs)

        return zk, psik, psik_cov  # observation vector, state vector, and state covariance

    def sample(self, duration: float, **kwargs):
        '''
        Generate synthetic measurements for a given duration.
        Returns the observation vector with measurement noise added.
        '''

        measurement_noise = kwargs.pop("measurement_noise", True)
        process_noise = kwargs.pop("process_noise", True)
        zk, psik, _ = self.sample_measurement(duration, process_noise=process_noise, **kwargs)
        if measurement_noise:
            zk += self.sample_measurement_noise(duration, **kwargs)

        return zk, psik

    def _normalize_in(self, x, indexing=None):
        """
        Normalize the input zk to ensure it has the correct shape.
        Converts vectors or lists into a column-shaped numpy array.
        Supports indexing modes "ij" (default) or "xy" for convenience.
        """
        if x is None:
            return None

        indexing = indexing or "xy"
        if indexing not in ["ij", "xy"]:
            raise ValueError(f"Invalid indexing mode '{indexing}'. Use 'ij' or 'xy'.")

        x = np.asarray(x)
        if x.ndim == 1:
            x = np.expand_dims(x, axis=-1)

        if indexing == "ij":
            x = np.transpose(x, axes=(1, 0))

        return x

    def _normalize_out(self, x, indexing = None):
        """
        Normalize the output zk_hat to ensure it has the correct shape.
        Converts vectors or lists into a column-shaped numpy array.
        Supports indexing modes "ij" (default) or "xy" for convenience.
        """        
        if x is None:
            return None

        indexing = indexing or "xy"
        if indexing not in ["ij", "xy"]:
            raise ValueError(f"Invalid indexing mode '{indexing}'. Use 'ij' or 'xy'.")
        
        x = np.asarray(x)
        if x.ndim == 2 and x.shape[-1] == 1:
            x = np.squeeze(x, axis=-1)

        if indexing == "ij":
            x = np.transpose(x, axes=(1, 0))

        return x
        
    def __predict(self, x, P, **kwargs):

        assert x.shape == (self.n,1), f"State vector has incorrect dimensions. Expected {self.x.shape}, got {format_shape(x)}."
        assert P.shape == (self.n, self.n), f"State covariance has incorrect dimensions. Expected {self.P.shape}, got {format_shape(P)}."
        
        u = self._normalize_in(kwargs.pop("u", None) or np.zeros(self.l))
        assert u.shape == (self.l,1), f"Control input has incorrect dimensions. Expected {self.l}, got {format_shape(u)}."

        w = self._normalize_in(self.sample_process_noise(1)[0] if kwargs.pop("process_noise", False) else np.zeros((self.n, 1)))
        assert w.shape == (self.n, 1), f"Process noise has incorrect dimensions. Expected {self.n}, got {format_shape(w)}."

        x = self.F @ x + self.B @ u + w
        P = self.F @ P @ self.F.T + self.Q
        return x, P
     
    def __update(self, z):
        self.y = z - self.H @ self.x    
        self.S = self.R + self.H @ self.P @ self.H.T

        self.K = self.P @ self.H.T @ np.linalg.inv(self.S) if np.linalg.det(self.S) else np.zeros((self.n, self.m))
        self.x = self.x + self.K @ self.y

        I = np.eye(self.n)
        self.P = (I - self.K @ self.H) @ self.P @ (I - self.K @ self.H).T + self.K @ self.R @ self.K.T

        return self.x, self.P

    def __smooth(self, prior, posterior):
        """
        Rauch-Tung-Striebel (RTS) smoother.
        Requires that filter() has been run and stored prior and posterior estimates.
        """
        N = len(posterior)
        assert N > 0, "No posterior estimates available for smoothing."
        assert len(prior) == N, "Prior estimates must match the number of posterior estimates."
        
        # Initialize last smoothed state as last posterior
        smooth_posterior = [None] * N
        smooth_posterior[-1] = posterior[-1]
        
        for k in reversed(range(N - 1)):
            x, P = posterior[k]
            xp, Pp = prior[k+1]
            xs, Ps = smooth_posterior[k+1]
            
            # Compute smoothing gain (stable)
            if np.linalg.cond(Pp) < 1e12:
                C = P @ self.F.T @ np.linalg.inv(Pp)
            else:
                C = P @ self.F.T @ np.linalg.pinv(Pp)

            # Update smoothed estimates
            _xs = x + C @ (xs - xp)
            _Ps = P + C @ (Ps - Pp) @ C.T
            
            smooth_posterior[k] = (_xs, _Ps)
        
        return smooth_posterior
    
    def filter(self, gk, uk = None, **kwargs):

        indexing = kwargs.pop("indexing", "xy")

        reset    = kwargs.pop("reset", True)
        buffer   = kwargs.pop("buffer", 60) # in seconds
        if len(self.posterior) > buffer * self.fs:
            self.reset_with_latest()
        elif reset:
            self.reset()
       
        # normalize input data
        if uk is None or len(uk) < 1:
            uk = np.zeros((len(gk), self.l)) if self.l > 0 else None

        gk = self._normalize_in(gk)
        assert gk.shape[1] == self.m, f"gk data has incorrect number of columns. Expected {self.m}, got {format_shape(gk)}."
        uk = self._normalize_in(uk)
        assert uk.shape[1] == self.l, f"uk data has incorrect number of columns. Expected {self.l}, got {format_shape(uk)}."

        zk, uk = self.preprocess(gk, uk)
        assert self.m == 1 or zk.shape[1] == self.m, f"zk data has incorrect number of columns. Expected ({self.m},1), got {format_shape(zk)}."
        assert self.l == 1 or uk is None or uk.shape[1] == self.l, f"uk data has incorrect number of columns. Expected ({self.l},1), got {format_shape(uk)}."

        # kalman filter (forward pass)
        for (z, u) in zip(zk, uk):

            self.measurement.append(z)
            self.control.append(u)

            self.x, self.P = self.__predict(self.x, self.P, u=u)
            self.prior.append((self.x, self.P))
   
            self.x, self.P = self.__update(z)
            self.posterior.append((self.x, self.P)) 

        # kalman posterior estimates
        self.posterior_psik_hat = self._inprocess(self.get_posterior_state_vector(), **kwargs)
        self.posterior_gk_hat = self._postprocess(self.posterior_psik_hat, **kwargs) 

        # kalman prior estimates
        self.prior_psik_hat = self._inprocess(self.get_prior_state_vector(), **kwargs)
        self.prior_gk_hat = self._postprocess(self.prior_psik_hat, **kwargs) 

        # kalman smoothing (backward pass)
        smooth = kwargs.pop("smooth", False)
        if smooth:
            self.smooth_posterior = self.__smooth(self.prior, self.posterior)
            self.smooth_psik_hat = self._inprocess(self.get_smooth_state_vector(), **kwargs)
            self.smooth_gk_hat = self._postprocess(self.smooth_psik_hat, **kwargs)

        return self.get_output(indexing=indexing)

    def _normalize_index(self, idx, size=None):
        """Convert int, Enum, or tuple/list/array to np.array of ints."""
        if isinstance(idx, (tuple, list, np.ndarray)):
            return np.array([i.value if hasattr(i, "value") else int(i) for i in idx])
        else:
            i = idx.value if hasattr(idx, "value") else int(idx)
            if size is not None:
                return np.array([i] * size)
            return np.array([i])

    def _normalize_component(self, component, type: KalmanStateType = None, dim = 1):
        """
        Normalize component to a tuple of length `dim`.

        Rules:
        - If a single value is passed, repeat it `dim` times.
        - If a list/tuple/array of length < dim, repeat the last element to reach dim.
        - If length > dim, truncate to dim.
        - Always returns a tuple of length dim.
        """

        if not isinstance(component, (list, tuple, np.ndarray)):
            components = [component]
        else:
            components = list(component)

        if not components:
            components = [0]

        # Extend or truncate to match dim
        if len(components) < dim:
            components += [components[-1]] * (dim - len(components))
        elif len(components) > dim:
            components = components[:dim]

        # Convert all to enum
        if type is not None:
            components = [str2enum(c, type) for c in components]

        return tuple(components)

    def get_mapping(self, meas_idx: KalmanStateType):
        
        meas_idx = meas_idx.value if hasattr(meas_idx, 'value') else int(meas_idx)
        idx = meas_idx

        assert 0 <= idx < self.m, f"Measurement index out of range. Expected 0 <= idx < {self.m}, got {idx}."
        return idx
    
    def exclusive_mapping(self, state_idx: int|tuple, meas_idx: int|tuple):
        """Check if measurement(s) exclusively observe state(s)."""
        # Convert to arrays
        if isinstance(state_idx, (tuple, list, np.ndarray)) or isinstance(meas_idx, (tuple, list, np.ndarray)):
            # Determine broadcasting length
            n = max(len(state_idx) if hasattr(state_idx, "__len__") else 1,
                    len(meas_idx) if hasattr(meas_idx, "__len__") else 1)
            state_idx = self._normalize_index(state_idx, n)
            meas_idx  = self._normalize_index(meas_idx, n)

            for i in range(n):
                if self.exclusive_mapping(state_idx[i], meas_idx[i]) is False:
                    return False

            return True if n else False

        else:
            s = state_idx.value if hasattr(state_idx, 'value') else int(state_idx)
            m = meas_idx.value  if hasattr(meas_idx,  'value') else int(meas_idx)
            assert 0 <= s < self.n
            assert 0 <= m < self.m
            coeff = self.H[m, s]
            return coeff != 0

    def inclusive_mapping(self, state_idx: int|tuple, meas_idx: int|tuple):
        """Check if measurement(s) depend on state(s) in any way."""
        if isinstance(state_idx, (tuple, list, np.ndarray)) or isinstance(meas_idx, (tuple, list, np.ndarray)):
            # Determine broadcasting length
            n = max(len(state_idx) if hasattr(state_idx, "__len__") else 1,
                    len(meas_idx) if hasattr(meas_idx, "__len__") else 1)
            state_idx = self._normalize_index(state_idx, n)
            meas_idx  = self._normalize_index(meas_idx, n)

            for i in range(n):
                if self.inclusive_mapping(state_idx[i], meas_idx[i]) is False:
                    return False

            return True if n else False
            
        else:
            s = state_idx.value if hasattr(state_idx, 'value') else int(state_idx)
            m = meas_idx.value  if hasattr(meas_idx,  'value') else int(meas_idx)
            assert 0 <= s < self.n
            assert 0 <= m < self.m
            return self.H[m, s] != 0

    def get_transition_matrix(self):
        return self.F
    def get_observation_matrix(self):
        return self.H
    def get_process_noise_covariance(self):
        return self.Q
    def get_measurement_noise_covariance(self):
        return self.R

    def get_measurement(self, k = None, indexing=None):
        return self._normalize_out(self.measurement, indexing=indexing) if k is None else self.measurement[k]

    def get_output(self, k = None, indexing=None):
        psik_hat = self._normalize_out(self.psik_hat, indexing=indexing) if k is None else self.psik_hat[k]
        gk_hat = self._normalize_out(self.gk_hat, indexing=indexing) if k is None else self.gk_hat[k]
        return gk_hat, psik_hat
    
    def get_observation(self, H = None, k=None, indexing=None):
        H = self.H if H is None else np.asarray(H)
        assert H.shape == (self.m, self.n), f"Observation matrix H has incorrect dimensions. Expected {self.m}x{self.n}, got {format_shape(H)}."
        return self._normalize_out([H @ psi for psi in self.psik_hat] if k is None else H @ self.psik_hat[k], indexing=indexing)

    def get_control(self, k = None, indexing=None):
        return self.get_control_vector(k, indexing=indexing), self.B
    def get_control_vector(self, k = None, indexing=None):
        if self.control is None or len(self.control) == 0:
            return self._normalize_out(np.zeros((0, self.l))) if k is None else np.zeros((1, self.l))
        return self._normalize_out(self.control, indexing=indexing) if k is None else self.control[k]
    def get_control_matrix(self):
        return self.B

    def get_initial_state(self):
        return self.initial_state_vector(), self.initial_state_covariance()
    def get_initial_state_vector(self):
        return self.x0
    def get_initial_state_covariance(self):
        return self.P0

    def initial_state(self, x0 = None, P0 = None):

        if x0 is not None:
            x0 = self._normalize_in(x0)
            if x0.shape != (self.n,1):
                warnings.warn(
                    f"Initial state vector dimension expected {self.n}x1, got {format_shape(x0)}. Automatic resizing applied."
                )
                x0 = x0.copy()
                x0.resize((self.n, 1))

            self.x0 = x0

        if P0 is not None:
            P0 = self._normalize_in(P0)
            if P0.shape != (self.n, self.n):
                warnings.warn(
                    f"Initial state covariance dimension expected {self.n}x{self.n}, got {format_shape(P0)}. Automatic resizing applied."
                )
                P0 = P0.copy()
                P0.resize((self.n, self.n))

            self.P0 = P0

    def initial_state_vector(self, x0 = None):
        x0, _ = self.initial_state(x0=x0)
        return x0
    def initial_state_covariance(self, P0 = None):
        _, P0 = self.initial_state(P0=P0)
        return P0

    def get_latest_state(self):
        return self.x, self.P
    def get_latest_state_vector(self):
        return self.get_latest_state()[0]
    def get_latest_state_covariance(self):
        return self.get_latest_state()[1]

    def get_prior_state(self, k = None): # predicted state based on the kth previous knowledge
        return self.prior if k is None else self.prior[k]
    def get_prior_state_vector(self, k = None):
        return self._get_tuple(self.get_prior_state(k), ith=0)
    def get_prior_state_covariance(self, k = None):
        return self._get_tuple(self.get_prior_state(k), ith=1)
    def get_prior_observation(self, **kwargs):
        H = kwargs.pop("H", None) or self.H
        zk_prior_hat = np.array([H @ self._inprocess(sv) for sv in self.get_prior_state_vector()])
        return self._postprocess(zk_prior_hat, **kwargs)

    def get_posterior_state(self, k = None): # estimated state incorporating the kth measurement
        return self.posterior if k is None else self.posterior[k]
    def get_posterior_state_vector(self, k = None):
        return self._get_tuple(self.get_posterior_state(k), ith=0)
    def get_posterior_state_covariance(self, k = None):
        return self._get_tuple(self.get_posterior_state(k), ith=1)
    def get_posterior_observation(self, **kwargs):
        H = kwargs.pop("H", None) or self.H
        zk_hat = np.array([H @ self._inprocess(sv) for sv in self.get_posterior_state_vector()])
        return self._postprocess(zk_hat, **kwargs)

    def get_smooth_state(self, i = None): # estimated state incorporating the ith measurement
        return self.smooth_posterior if i is None else self.smooth_posterior[i]
    def get_smooth_state_vector(self, i = None):
        return self._get_tuple(self.get_smooth_state(i), ith=0)
    def get_smooth_state_covariance(self, i = None):
        return self._get_tuple(self.get_smooth_state(i), ith=1)
    def get_smooth_observation(self, **kwargs):
        H = kwargs.pop("H", None) or self.H
        zk_hat = np.array([H @ self._inprocess(sv) for sv in self.get_smooth_state_vector()])
        return self._postprocess(zk_hat, **kwargs)

    def get_innovation(self, k = None):
        return self.get_prefit_residual(k)
    def get_innovation_vector(self, k = None):
        return self._get_tuple(self.get_innovation(k), ith=0)
    def get_innovation_covariance(self, k = None):
        return self._get_tuple(self.get_innovation(k), ith=1)

    def get_prefit_residual(self, k = None):
        y = self.y if k is None else self.measurement[k] - self.H @ self.get_prior_state_vector(k)
        S = self.S if k is None else self.R + self.H @ self.get_prior_state_covariance(k) @ self.H.T
        return y, S
    def get_prefit_residual_vector(self, k = None):
        return self._get_tuple(self.get_prefit_residual(k), ith=0)
    def get_prefit_residual_covariance(self, k = None):
        return self._get_tuple(self.get_prefit_residual(k), ith=1)

    def get_gain(self, k = None):
        return self.K if k is None else self.get_posterior_state_covariance(k) @ self.H.T @ np.linalg.inv(self.get_prefit_residual_covariance(k))
    def get_postfit_residual(self, k = None):
        y = self.y if k is None else self.measurement[k] - self.H @ self.get_posterior_state_vector(k)
        S = self.S if k is None else self.R + self.H @ self.get_posterior_state_covariance(k) @ self.H.T
        return y, S
    def get_postfit_residual_vector(self, k = None):
        return self._get_tuple(self.get_postfit_residual(k), ith=0)
    def get_postfit_residual_covariance(self, k = None):
        return self._get_tuple(self.get_postfit_residual(k), ith=1)

    def get_latest_innovation(self):
        return self.get_latest_prefit_residual()
    def get_latest_innovation_vector(self):
        return self._get_tuple(self.get_latest_innovation(), ith=0)
    def get_latest_innovation_covariance(self):
        return self._get_tuple(self.get_latest_innovation(), ith=1)

    def get_latest_prefit_residual(self):
        return self.y, self.S
    def get_latest_prefit_residual_vector(self):
        return self._get_tuple(self.get_latest_prefit_residual(), ith=0)
    def get_latest_prefit_residual_covariance(self):
        return self._get_tuple(self.get_latest_prefit_residual(), ith=1)

    def get_latest_gain(self):
        return self.K
    def get_latest_postfit_residual(self, z):
        return z - self.H @ self.get_posterior_state_vector(), self.R + self.H @ self.get_latest_state_covariance() @ self.H.T
    def get_latest_postfit_residual_vector(self, z):
        return self._get_tuple(self.get_latest_postfit_residual(z), ith=0)
    def get_latest_postfit_residual_covariance(self, z):
        return self._get_tuple(self.get_latest_postfit_residual(z), ith=1)

    def _get_tuple(self, state, ith):
        if ith not in [0, 1]:
            raise ValueError("Component must be 0 (state vector) or 1 (covariance matrix).")
        if isinstance(state, tuple):
            return state[ith]
        return np.array([s[ith] for s in state])

    @property
    @abstractmethod
    def labels(self):
        """Dictionary mapping state types to labels."""
        pass

    @property
    @abstractmethod
    def units(self):
        """Dictionary mapping state types to units."""
        pass

    @property
    @abstractmethod
    def coords(self):
        """List of coordinate names."""
        pass

    @property
    @abstractmethod
    def colors(self):
        """Nested dictionary mapping KalmanOutputType -> axis -> color."""
        pass

    @property
    @abstractmethod
    def linestyles(self):
        """Nested dictionary mapping KalmanOutputType -> axis -> linestyle."""
        pass

    def plot(self, **kwargs):

        """
        Public plot method. Dispatches to _plot3D if self.dim > 2 and axis=None,
        otherwise falls back to _plot2D.
        """
        t = self.time()
        if t is None or len(t) == 0:
            return None, None

        # Mode parsing
        component = kwargs.pop("component", 0)
        component = self._normalize_component(component)
        dim = kwargs.get("dim", None)

        # Dispatch
        if dim is None and len(component) > 2:
            return self._plot3D(component, **kwargs)
        elif dim is None and len(component) > 1:
            return self._plot2D(component, **kwargs)
        else:
            return self._plot1D(component, **kwargs)

    def _plot1D(self, component, **kwargs):

        t = self.time()
        if t is None or len(t) == 0:
            return None, None

        # Determine axes parameter
        dim = kwargs.pop("dim", None)
        state_vec = self.get_posterior_state_vector()
        ideal = self.ideal(duration=self.duration())
        truth = kwargs.pop("truth", None)
        measurement = self._normalize_out(self.get_measurement())

        # Determine limits with padding, comparing with current plot limits
        padding = kwargs.pop("padding", 0.05)
        legend = kwargs.pop("legend", True)
        
        # --- 2D branch ---
        c = kwargs.pop("canvas", None) or get_canvas(**kwargs)

        if dim is None:
            axes = list(range(self.dim))
        elif isinstance(dim, (list, tuple)):
            axes = dim
        else:
            axes = [dim]

        for a in axes:
            if not (0 <= a < self.dim):
                raise ValueError(f"Invalid axis {a} considering {self.dim}D model")

        component = component[0]  # Use only the first component for 1D plot
        
        # Initialize y_min and y_max
        current_ylim = c.get_ylim() # Check all ylim from all canvas 
        y_min, y_max = current_ylim if current_ylim else (np.inf, -np.inf)
        for a in axes:

            idx = self.dim * a + int(component.value)
            if measurement is not None and self.exclusive_mapping(idx, a):
                y_min = min(y_min, measurement[:, a].min())
                y_max = max(y_max, measurement[:, a].max())
            if ideal is not None:
                y_min = min(y_min, ideal[:, idx].min())
                y_max = max(y_max, ideal[:, idx].max())
            if truth is not None and truth.shape[1] > idx:
                y_min = min(y_min, truth[:, idx].min())
                y_max = max(y_max, truth[:, idx].max())

            y_min = min(y_min, state_vec[:, idx].min())
            y_max = max(y_max, state_vec[:, idx].max())
            
            y_range = y_max - y_min
            y_min -= padding * y_range
            y_max += padding * y_range

            if "fig" not in kwargs and kwargs.pop("units", True):
                X0 = self._normalize_out(self.get_initial_state_vector())[idx]
                c.axhline(X0, color='black', alpha=1, linestyle=':', linewidth=0.75)  # x-axis reference
                c.axvline(0, color='black', alpha=1, linestyle=':', linewidth=0.75)  # y-axis reference

            if measurement is not None and self.exclusive_mapping(idx, a):
                c.scatter(
                    t, measurement[:, a], s=1, 
                    label=f"{self.coords[a]} (Measured)", 
                    color=self.colors[KalmanOutputType.MEASUREMENT].get(a % len(self.colors[KalmanOutputType.MEASUREMENT]), "gray")
                )

            if ideal is not None:
                c.plot(t, ideal[:, idx], self.linestyles[KalmanOutputType.IDEAL].get(a % len(self.linestyles[KalmanOutputType.IDEAL]), "-"),
                       label=f"{self.coords[a]} (Ideal)", 
                       color=self.colors[KalmanOutputType.IDEAL].get(a % len(self.colors[KalmanOutputType.IDEAL]), "gray"))

            if truth is not None and truth.shape[1] > idx:
                c.plot(t, truth[:, idx], self.linestyles[KalmanOutputType.TRUTH].get(a % len(self.linestyles[KalmanOutputType.TRUTH]), "--"),
                       label=f"{self.coords[a]} (Truth)", 
                       color=self.colors[KalmanOutputType.TRUTH].get(a % len(self.colors[KalmanOutputType.TRUTH]), "gray"))

            c.plot(t, state_vec[:, idx], 
                   label=f"{self.coords[a]} (" + ("Smoothed " if len(self.smooth_posterior) else "") + "Estimate)", 
                   color=self.colors[KalmanOutputType.ESTIMATE].get(a % len(self.colors[KalmanOutputType.ESTIMATE]), "gray"))

        c.set_xlim(kwargs.pop("xlim", None))
        c.set_ylim(kwargs.pop("ylim", (y_min, y_max)))

        c.set_xlabel(kwargs.pop("xlabel", "Time (s)"))
        if legend:
            c.set_ylabel(kwargs.pop("label", f"{self.labels[component]} ({self.units[component]})"))
        else:
            c.set_ylabel(kwargs.pop("label", f"{self.labels[component]} ({self.units[component]}) along {self.coords[dim or 0]}"))

        c.grid(kwargs.pop("grid", True))
        if legend:
            c.legend(fontsize=10, handlelength=4, handleheight=0.01)

        return get_figure(c), get_axis(c)

    def _plot2D(self, component, **kwargs):
        """
        Plot the ballistic trajectory in 2D (x vs y). Supports multiple components.
        `component` can be a single value, tuple/list, or array of length 2.
        """
        t = self.time()
        if t is None or len(t) == 0:
            return None, None

        # Normalize component to a tuple of length 2
        component = self._normalize_component(component, dim=2)

        # Determine state vectors
        state_vec = self.get_posterior_state_vector()
        ideal = self.ideal(duration=self.duration())
        truth = kwargs.pop("truth", None)
        measurement = self._normalize_out(self.get_measurement())

        # Common kwargs
        padding = kwargs.pop("padding", 0.05)
        projectors = kwargs.pop("projectors", True)
        connectors = kwargs.pop("connectors", True)

        # Canvas / figure
        fig = kwargs.get("fig") or plt.figure(figsize=kwargs.pop("figsize",(8,8)))
        ax = fig.add_subplot(1,1,1)

        # Map components to state indices
        i0, i1 = self.get_mapping(component[0], axis=0), self.get_mapping(component[1], axis=1)
        X, Y = state_vec[:, i0], state_vec[:, i1]
        X0, Y0 = self._normalize_out(self.get_initial_state_vector())[i0], self._normalize_out(self.get_initial_state_vector())[i1]
        TX, TY = (truth[:, i0], truth[:, i1]) if truth is not None else (None, None)
        IX, IY = (ideal[:, i0], ideal[:, i1]) if ideal is not None else (None, None)

        # Measurements
        if self.exclusive_mapping((i0,i1), component):
            if measurement is not None and measurement.shape[1] >= 2:
                ax.scatter(measurement[:,0], measurement[:,1], s=1, color='black', alpha=0.75, label="Measurements")

        # Determine plot limits
        xmin = min(X.min(), TX.min() if TX is not None else X.min(), IX.min() if IX is not None else X.min())
        xmax = max(X.max(), TX.max() if TX is not None else X.max(), IX.max() if IX is not None else X.max())
        ymin = min(Y.min(), TY.min() if TY is not None else Y.min(), IY.min() if IY is not None else Y.min())
        ymax = max(Y.max(), TY.max() if TY is not None else Y.max(), IY.max() if IY is not None else Y.max())

        # Apply padding
        xpad = padding * (xmax - xmin)
        ypad = padding * (ymax - ymin)
        ax.set_xlim(xmin - xpad, xmax + xpad)
        ax.set_ylim(ymin - ypad, ymax + ypad)

        # Plot ideal, truth, estimate
        if IX is not None:
            ax.plot(IX, IY, label='Ideal', color=self.colors[KalmanOutputType.IDEAL].get(0, "gray"),
                    linestyle=self.linestyles[KalmanOutputType.IDEAL].get(0, "-"))
        if TX is not None:
            ax.plot(TX, TY, label='Truth', color=self.colors[KalmanOutputType.TRUTH].get(0, "gray"),
                    linestyle=self.linestyles[KalmanOutputType.TRUTH].get(0, "--"))

        ax.plot(X, Y, label=("Smoothed " if len(self.smooth_posterior) else "") + "Estimate",
                color=self.colors[KalmanOutputType.ESTIMATE].get(0, "black"), linewidth=2)

        # Projectors / connectors (optional)
        if projectors:
            ax.plot(X, [Y0]*len(X), color=self.colors[KalmanOutputType.ESTIMATE].get(1, "gray"),
                    linestyle=self.linestyles[KalmanOutputType.ESTIMATE].get(1, "-"), alpha=0.5)
            ax.plot([X0]*len(Y), Y, color=self.colors[KalmanOutputType.ESTIMATE].get(0, "gray"),
                    linestyle=self.linestyles[KalmanOutputType.ESTIMATE].get(0, "-"), alpha=0.5)
            if connectors:
                for j in (0, len(X)-1):
                    ax.plot([X0, X[j]], [Y0, Y[j]], color='gray', linestyle='-.', alpha=0.5, linewidth=0.5)

        # Labels and grid
        ax.set_xlabel(f"{self.coords[0]} {self.labels[component[0]]} ({self.units[component[0]]})")
        ax.set_ylabel(f"{self.coords[1]} {self.labels[component[1]]} ({self.units[component[1]]})")
        ax.grid(kwargs.pop("grid", True))
        if kwargs.pop("legend", True):
            ax.legend(fontsize=10)

        return fig, ax

    def _plot3D(self, component, **kwargs):
        """
        Plot the ballistic trajectory:
          - 3D view (with projections & connectors) if dim>2 and axis=None.
          - Otherwise: 2D plot of Position / Speed / Acceleration vs time.
        """
        t = self.time()
        if t is None or len(t) == 0:
            return None, None
        
        # Normalize component to a list of length 3
        component = self._normalize_component(component, dim=self.dim)

        # Determine axes parameter
        state_vec = self.get_posterior_state_vector()
        ideal = self.ideal(duration=self.duration())
        truth = kwargs.pop("truth", None)
                
        # Common kwargs for plot styling
        plot_kwargs = kwargs.pop("plot", {"color": "black", "linestyle": "-.", "linewidth": 0.5, "alpha": 0.5})

        # Padding factor
        padding = kwargs.pop("padding", 0)

        fig = kwargs.get("fig") or plt.figure(figsize=kwargs.pop("figsize",(10,10)))
        ax = fig.add_subplot(projection='3d')

        i0, i1, i2 = self.get_mapping(component[0], axis=0), self.get_mapping(component[1], axis=1), self.get_mapping(component[2], axis=2)
        X,  Y,  Z  = state_vec[:, i0], state_vec[:, i1], state_vec[:, i2]
        X0, Y0, Z0 = self._normalize_out(self.get_initial_state_vector())[i0], self._normalize_out(self.get_initial_state_vector())[i1], self._normalize_out(self.get_initial_state_vector())[i2]
        TX, TY, TZ = None, None, None
        if truth is not None:
            TX, TY, TZ = truth[:, i0], truth[:, i1], truth[:, i2]
        IX, IY, IZ = None, None, None
        if ideal is not None:
            IX, IY, IZ = ideal[:, i0], ideal[:, i1], ideal[:, i2]

        projectors = kwargs.pop("projectors", True)
        connectors = kwargs.pop("connectors", True)
        
        if self.exclusive_mapping([0,1,2], component):
            meas = self._normalize_out(self.get_measurement())
            if meas is not None and meas.shape[1] >= 3:
                ax.scatter(meas[:,0], meas[:,1], meas[:,2], label='Measurements', color='black', s=1, alpha=0.75, depthshade=False)

        # Ensure X0, Y0, Z0 are within the limit range
        x0, y0, z0 = np.clip(X0, X.min(), X.max()), np.clip(Y0, Y.min(), Y.max()), np.clip(Z0, Z.min(), Z.max())
        
        # Apply padding
        if isinstance(padding, (tuple, list)) and len(padding) == 3:
            xpad, ypad, zpad = padding
        else:
            xpad = ypad = zpad = padding

        xlim = (min(X.min(), TX.min() if TX is not None else X.min(), IX.min() if IX is not None else X.min()) - xpad*(X.max()-X.min()),
                max(X.max(), TX.max() if TX is not None else X.max(), IX.max() if IX is not None else X.max()) + xpad*(X.max()-X.min()))
        ylim = (min(Y.min(), TY.min() if TY is not None else Y.min(), IY.min() if IY is not None else Y.min()) - ypad*(Y.max()-Y.min()),
                max(Y.max(), TY.max() if TY is not None else Y.max(), IY.max() if IY is not None else Y.max()) + ypad*(Y.max()-Y.min()))
        zlim = (min(Z.min(), TZ.min() if TZ is not None else Z.min(), IZ.min() if IZ is not None else Z.min()) - zpad*(Z.max()-Z.min()),
                max(Z.max(), TZ.max() if TZ is not None else Z.max(), IZ.max() if IZ is not None else Z.max()) + zpad*(Z.max()-Z.min()))

        xlim = kwargs.pop("xlim", xlim)
        if xlim[0] != xlim[1]: ax.set_xlim(xlim)
        ylim = kwargs.pop("ylim", ylim)
        if ylim[0] != ylim[1]: ax.set_ylim(ylim)
        zlim = kwargs.pop("zlim", zlim)
        if zlim[0] != zlim[1]: ax.set_zlim(zlim)

        # Lines along XYZ-axis
        if x0 != xlim[0] and x0 != xlim[1]:
            ax.plot([xlim[0], xlim[1]], [y0, y0], [z0, z0], **plot_kwargs)
        if y0 != ylim[0] and y0 != ylim[1]:
            ax.plot([x0, x0], [ylim[0], ylim[1]], [z0, z0], **plot_kwargs)
        if z0 != zlim[0] and z0 != zlim[1]:
            if x0 != xlim[0]:
                ax.plot([xlim[0], xlim[1]], [y0, y0], [zlim[0], zlim[0]], **plot_kwargs)
            if y0 != ylim[0]:
                ax.plot([x0, x0], [ylim[0], ylim[1]], [zlim[0], zlim[0]], **plot_kwargs)

        # Outline XY plane
        if z0 != zlim[0] and z0 != zlim[1]:
            ax.plot([xlim[0], xlim[1]], [ylim[0], ylim[0]], [z0, z0], **plot_kwargs)
            ax.plot([xlim[1], xlim[1]], [ylim[0], ylim[1]], [z0, z0], **plot_kwargs)

        # Projecting lines from Z0 to Zmin
        if z0 != zlim[0]:        
            ax.plot([xlim[0], xlim[0]], [ylim[0], ylim[0]], [zlim[0], z0], **plot_kwargs)
            ax.plot([xlim[0], xlim[0]], [ylim[1], ylim[1]], [zlim[0], z0], **plot_kwargs)
            ax.plot([xlim[1], xlim[1]], [ylim[0], ylim[0]], [zlim[0], z0], **plot_kwargs)
            ax.plot([xlim[1], xlim[1]], [ylim[1], ylim[1]], [zlim[0], z0], **plot_kwargs)
        
            if x0 != xlim[0] and x0 != xlim[1]:
                ax.plot([x0, x0], [ylim[0], ylim[0]], [zlim[0], z0], **plot_kwargs)
                ax.plot([x0, x0], [ylim[1], ylim[1]], [zlim[0], z0], **plot_kwargs)
            if x0 != xlim[0] and x0 != xlim[1] and y0 != ylim[0] and y0 != ylim[1]:
                ax.plot([x0, x0], [y0, y0], [zlim[0], z0], **plot_kwargs)
            if y0 != ylim[0] and y0 != ylim[1]:
                ax.plot([xlim[1], xlim[1]], [y0, y0], [zlim[0], z0], **plot_kwargs)
                ax.plot([xlim[0], xlim[0]], [y0, y0], [zlim[0], z0], **plot_kwargs)

        if ideal is not None:
            ax.plot(IX, IY, IZ, label='Ideal', color=self.colors[KalmanOutputType.IDEAL].get(0, "black"), linestyle=self.linestyles[KalmanOutputType.IDEAL].get(0))
        if truth is not None:
            ax.plot(TX, TY, TZ, label='Truth', color=self.colors[KalmanOutputType.TRUTH].get(0, "black"), linestyle=self.linestyles[KalmanOutputType.TRUTH].get(0))

        if projectors:
            z0, y0, x0 = float(Z0), float(ylim[1]), float(xlim[0])
            ax.plot(X,           Y,          [Z0]*len(Z), color=self.colors[KalmanOutputType.ESTIMATE].get(0, "gray"), alpha=0.75, linestyle=self.linestyles[KalmanOutputType.ESTIMATE].get(0, "-"))
            ax.plot(X,          [y0]*len(X), Z,           color=self.colors[KalmanOutputType.ESTIMATE].get(1, "gray"), alpha=0.75, linestyle=self.linestyles[KalmanOutputType.ESTIMATE].get(1, "-"))
            ax.plot([x0]*len(Y), Y,          Z,           color=self.colors[KalmanOutputType.ESTIMATE].get(2, "gray"), alpha=0.75, linestyle=self.linestyles[KalmanOutputType.ESTIMATE].get(2, "-"))
            
            ax.plot(IX,           IY,          [Z0]*len(IZ), color=self.colors[KalmanOutputType.IDEAL].get(0, "gray"), alpha=0.75, linestyle=self.linestyles[KalmanOutputType.IDEAL].get(0, "-"))
            ax.plot(IX,          [y0]*len(IX), IZ,           color=self.colors[KalmanOutputType.IDEAL].get(1, "gray"), alpha=0.75, linestyle=self.linestyles[KalmanOutputType.IDEAL].get(1, "-"))
            ax.plot([x0]*len(IY), IY,          IZ,           color=self.colors[KalmanOutputType.IDEAL].get(2, "gray"), alpha=0.75, linestyle=self.linestyles[KalmanOutputType.IDEAL].get(2, "-"))
            
            ax.plot(TX,           TY,          [Z0]*len(TZ), color=self.colors[KalmanOutputType.TRUTH].get(0, "gray"), alpha=0.75, linestyle=self.linestyles[KalmanOutputType.TRUTH].get(0, "-"))
            ax.plot(TX,          [y0]*len(TX), TZ,           color=self.colors[KalmanOutputType.TRUTH].get(1, "gray"), alpha=0.75, linestyle=self.linestyles[KalmanOutputType.TRUTH].get(1, "-"))
            ax.plot([x0]*len(TY), TY,          TZ,           color=self.colors[KalmanOutputType.TRUTH].get(2, "gray"), alpha=0.75, linestyle=self.linestyles[KalmanOutputType.TRUTH].get(2, "-"))
            if connectors:
                for j in (0, len(X)-1):
                    xi, yi, zi = float(X[j]), float(Y[j]), float(Z[j])
                    ax.plot([xi,xi],[yi,yi],[z0,zi], color=self.colors[KalmanOutputType.ESTIMATE].get(0, "gray"), linestyle='-.', linewidth=0.5, alpha=0.75)
                    ax.plot([xi,xi],[y0,yi],[zi,zi], color=self.colors[KalmanOutputType.ESTIMATE].get(1, "gray"), linestyle='-.', linewidth=0.5, alpha=0.75)
                    ax.plot([x0,xi],[yi,yi],[zi,zi], color=self.colors[KalmanOutputType.ESTIMATE].get(2, "gray"), linestyle='-.', linewidth=0.5, alpha=0.75)

        ax.plot(X, Y, Z, label=("Smoothed " if len(self.smooth_posterior) else "") + 'Estimate', color=self.colors[KalmanOutputType.ESTIMATE].get(0, "black"), linewidth=2)
        ax.set_xlabel(kwargs.pop("xlabel", self.coords[0] + " " + self.labels[component[0]] + " (" + self.units[component[0]] + ")"))
        ax.set_ylabel(kwargs.pop("ylabel", self.coords[1] + " " + self.labels[component[1]] + " (" + self.units[component[1]] + ")"))
        ax.set_zlabel(kwargs.pop("zlabel", self.coords[2] + " " + self.labels[component[2]] + " (" + self.units[component[2]] + ")"))
        
        ax.grid(kwargs.pop("grid", True))
        ax.set_box_aspect([1,1,1])
        if kwargs.pop("legend", True):
            ax.legend(fontsize=10, handlelength=4, handleheight=0.01)

        return fig, ax
    
    def plot_error(self, **kwargs):
        """
        Plot the estimation error of the state vector.
        """
        t = self.time()
        if t is None or len(t) == 0:
            return None, None
        
        state_vec = self.get_posterior_state_vector()
        ideal = self.ideal(duration=self.duration())
        truth = kwargs.pop("truth", None)
        
        if ideal is None and truth is None:
            raise ValueError("Either ideal or truth data must be provided for error plotting.")
        
        if ideal is not None:
            error_ideal = state_vec - ideal
        if truth is not None:
            error_truth = state_vec - truth
        
        fig, ax = get_canvas(figsize=kwargs.pop("figsize", (10, 6)))
        
        for i in range(state_vec.shape[1]):
            if ideal is not None:
                ax.plot(t, error_ideal[:, i], label=f'Error Ideal State {i+1}', linestyle='--')
            if truth is not None:
                ax.plot(t, error_truth[:, i], label=f'Error Truth State {i+1}', linestyle=':')
        
        ax.set_title("Estimation Error")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Error")
        ax.axhline(0, color='black', linestyle=':', linewidth=0.75)
        ax.legend()
        plt.tight_layout()
        
        return fig, ax