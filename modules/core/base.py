from abc import ABC, abstractmethod

# Keep your original imports; helpers are assumed to exist in your codebase
from .io import KalmanStateFlavor, KalmanStateRecord, KalmanStateType
from ..helpers import *
from ..debug import get_debug

import numpy as np
import warnings
import json
import yaml

# =========================
# Base Model
# =========================

class KalmanStateModel(ABC):

    NAME = "Kalman State Model"
    DESCRIPTION = "Base class for Kalman state models, providing a framework for state estimation and filtering."
    VERSION = "1.0"

    def name(self):
        return type(self).NAME if hasattr(type(self), "NAME") else "Generic Kalman Model"
    def description(self):
        return type(self).DESCRIPTION if hasattr(type(self), "DESCRIPTION") else "No description available."
    def version(self):
        return type(self).VERSION if hasattr(type(self), "VERSION") else "0.0"
    
    @property
    def type(self):
        return KalmanStateType
    
    # Abstract methods for signal processing steps
    @abstractmethod # Preprocess the zk data before passing it to the Kalman filter
    def preprocess(self, zk, uk = None, **kwargs):
        pass
    @abstractmethod # Inprocess state vector to a different representation or computing additional operations (like filtering windowing or smoothing, etc.)
    def inprocess(self, zk_hat, **kwargs):
        pass
    @abstractmethod # Postprocess the zk_hat data after Kalman filter processing
    def postprocess(self, zk_hat, **kwargs):
        pass

    # Metadata properties to configure plotting and representation
    @property
    @abstractmethod
    def axis(self):
        """List of coordinate names."""
        pass
    @property
    @abstractmethod
    def units(self):
        """Dictionary mapping state types to units."""
        pass

    def __init__(self, **kwargs):

        self.records = []
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

    def __destroy__(self):
        for record in self.records:
            record.model = None

    def reset_to_latest(self):
        self.reset(to_latest=True)
    def reset(self, to_latest=False):

        if to_latest:
            self.t0 = self.t0 + len(self.posterior) / self.fs
            self.P0 = self.get_latest_state_covariance()
            self.x0 = self.get_latest_state_vector()

        # Microstate variables
        self.P = self.P0
        self.x = self.x0

        # Macrostate variables
        self.prior = []
        self.posterior = []
        self.smooth = []

        self.measurement = []
        self.control = []

    def __len__(self):
        return len(self.posterior) if self.posterior else 0

    def print(self):
        print(self.__str__())

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

    def duration(self):
        """
        Returns the duration of the current state estimation.
        """
        return len(self.posterior) / self.fs if self.fs else 0
    
    def ideal(self, duration: float = None):
        record = self.sample(duration, process_noise=False, measurement_noise=False)
        return record.xk

    def truth(self, duration: float = None):
        record = self.sample(duration, process_noise=True, measurement_noise=False)
        return record.xk, record.Pk

    # State record management
    def get_latest_record(self):
        return self.records[-1] if self.records else None
    def get_record(self, i):
        return self.records[i] if self.records else None
    def get_records(self):
        return self.records
    def clear_records(self):
        self.records = []

    def get_posterior_record(self) -> KalmanStateRecord:
        if not self.posterior: return KalmanStateRecord()
        return KalmanStateRecord(self, self.time(), self.get_posterior_state_vector(), self.get_posterior_state_covariance(), zk=self.get_measurement(), flavor=KalmanStateFlavor.POSTERIOR, model=self)
    def get_prior_record(self) -> KalmanStateRecord:
        if not self.prior: return KalmanStateRecord()
        return KalmanStateRecord(self, self.time(), self.get_prior_state_vector(), self.get_prior_state_covariance(), zk=self.get_measurement(), flavor=KalmanStateFlavor.PRIOR, model=self)
    def get_smooth_record(self) -> KalmanStateRecord:
        if not self.smooth: return KalmanStateRecord()
        return KalmanStateRecord(self, self.time(), self.get_smooth_state_vector(), self.get_smooth_state_covariance(), zk=self.get_measurement(), flavor=KalmanStateFlavor.SMOOTHED, model=self)

    # Signal manipulation methods
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
    
    def _postprocess(self, xk_hat, **kwargs):

        zk_hat = np.array([self.H @ x_hat for x_hat in xk_hat])
        gk_hat = self.postprocess(zk_hat, **kwargs)

        # Matrix dimension checks
        if gk_hat.shape[0] != len(gk_hat):
            raise ValueError(f"Postprocessed result has incorrect number of rows. Expected {len(gk_hat)}, got {gk_hat.shape[0]}.")
        if gk_hat.shape[1] != self.m:
            raise ValueError(f"Postprocessed result has incorrect number of columns. Expected {self.m}, got {gk_hat.shape[1]}.")

        return gk_hat

    def sample_state(self, duration: float, **kwargs):
        _, xk, Pk = self.sample_measurement(duration, **kwargs)
        return xk, Pk
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
        return np.random.normal(0, 1, (n_samples, self.l)) if self.l > 0 else np.zeros((n_samples, 1))
    
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
        xk = np.zeros((n_samples, self.n, 1))
        Pk = np.zeros((n_samples, self.n, self.n))

        x, P = kwargs.pop("x0", self.get_initial_state_vector()), kwargs.pop("P0", self.get_initial_state_covariance())
        if x.shape != (self.n, 1):
            raise ValueError(f"Initial state vector has incorrect dimensions. Expected {self.n}x1, got {format_shape(x)}.")
        if P.shape != (self.n, self.n):
            raise ValueError(f"Initial state covariance has incorrect dimensions. Expected {self.n}x{self.n}, got {format_shape(P)}.")

        for i in range(n_samples):
            xk[i] = x
            Pk[i] = P
            zk[i] = self.H @ x
            x, P = self.__predict(x, P, **kwargs) # process noise sampling (in here)

        return zk, xk, Pk  # observation vector, state vector, and state covariance

    def sample(self, duration: float, **kwargs):
        '''
        Generate synthetic measurements for a given duration.
        Returns the observation vector with measurement noise added.
        '''

        measurement_noise = kwargs.pop("measurement_noise", True)
        process_noise = kwargs.pop("process_noise", True)
        zk, xk, Pk = self.sample_measurement(duration, process_noise=process_noise, **kwargs)
        if measurement_noise:
            zk += self.sample_measurement_noise(duration, **kwargs)

        return KalmanStateRecord(self.time(duration), zk=zk, xk=xk, Pk=Pk, flavor=KalmanStateFlavor.SAMPLED, model=self)

    def __predict(self, x, P, **kwargs):

        assert x.shape == (self.n,1), f"State vector has incorrect dimensions. Expected {self.x.shape}, got {format_shape(x)}."
        assert P.shape == (self.n, self.n), f"State covariance has incorrect dimensions. Expected {self.P.shape}, got {format_shape(P)}."
        
        u = KalmanStateRecord._normalize_in(kwargs.pop("u", None) or np.zeros(self.l))
        assert u.shape == (self.l,1), f"Control input has incorrect dimensions. Expected {self.l}, got {format_shape(u)}."

        w = KalmanStateRecord._normalize_in(self.sample_process_noise(1)[0] if kwargs.pop("process_noise", False) else np.zeros((self.n, 1)))
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

    def __smooth(self, prior = None, posterior = None):
        """
        Rauch-Tung-Striebel (RTS) smoother.
        Requires that filter() has been run and stored prior and posterior estimates.
        """
        prior = prior if prior is not None else self.prior
        posterior = posterior if posterior is not None else self.posterior
        
        N = len(posterior)
        assert N > 0, "No posterior estimates available for smoothing."
        assert len(prior) == N, "Prior estimates must match the number of posterior estimates."
        
        # Initialize last smoothed state as last posterior
        smooth = np.zeros((N, 2), dtype=object)
        smooth[-1] = posterior[-1]
        
        for k in reversed(range(N - 1)):
            x, P = posterior[k]
            xp, Pp = prior[k+1]
            xs, Ps = smooth[k+1]
            
            # Compute smoothing gain (stable)
            if np.linalg.cond(Pp) < 1e12:
                C = P @ self.F.T @ np.linalg.inv(Pp)
            else:
                C = P @ self.F.T @ np.linalg.pinv(Pp)

            # Update smoothed estimates
            _xs = x + C @ (xs - xp)
            _Ps = P + C @ (Ps - Pp) @ C.T
            
            smooth[k] = (_xs, _Ps)
        
        return smooth
    
    def filter(self, gk, uk = None, **kwargs):

        indexing = kwargs.pop("indexing", "xy")
        preprocess = kwargs.pop("preprocess", True)

        reset    = kwargs.pop("reset", True)
        buffer   = kwargs.pop("buffer", 60) # in seconds
        if len(self.posterior) > buffer * self.fs:
            self.reset_to_latest()
        elif reset:
            self.reset()
       
        # normalize input data
        gk = gk.zk if isinstance(gk, KalmanStateRecord) else gk
        gk = KalmanStateRecord._normalize_in(gk, indexing=indexing)
        assert gk.shape[1] == self.m, f"gk data has incorrect number of columns. Expected {self.m}, got {format_shape(gk)}."
        
        # normalize control data
        uk = uk.zk if isinstance(uk, KalmanStateRecord) else uk
        if uk is None or len(uk) < 1:
            uk = np.zeros((len(gk), self.l)) if self.l > 0 else None
        uk = KalmanStateRecord._normalize_in(uk, indexing=indexing)
        assert uk.shape[1] == self.l, f"uk data has incorrect number of columns. Expected {self.l}, got {format_shape(uk)}."

        zk, uk = self.preprocess(gk, uk) if preprocess else (gk, uk)
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

        # kalman smoothing (backward pass)
        smooth = kwargs.get("smooth", False)
        if smooth: self.smooth = self.__smooth()

        self.records.append(self.get_output(indexing=indexing, **kwargs))
        return self.records[-1]

    def get_output(self, k = None, **kwargs):

        postprocess = kwargs.pop("postprocess", True)
        smooth = kwargs.pop("smooth", False)

        if self.prior is None or self.posterior is None:
            raise ValueError("No prior or posterior estimates available. Please run filter() first.")

        xk_hat, Pk_hat = self.get_posterior_state_vector(), self.get_posterior_state_covariance()
        if smooth: xk_hat = self.get_smooth_state_vector()
            
        zk_hat = self._inprocess(xk_hat, **kwargs)
        if postprocess: zk_hat = self._postprocess(zk_hat, **kwargs)

        return KalmanStateRecord(self.time(k), zk=zk_hat, xk=xk_hat, Pk=Pk_hat, flavor=KalmanStateFlavor.SMOOTHED if smooth else KalmanStateFlavor.POSTERIOR, model=self)

    def exclusive_mapping(self, state_idx: int|tuple, meas_idx: int|tuple) -> bool:

        """Check if measurement(s) exclusively observe state(s)."""
        # Convert to arrays
        if isinstance(state_idx, (tuple, list, np.ndarray)) or isinstance(meas_idx, (tuple, list, np.ndarray)):
            
            # Determine broadcasting length
            n = max(len(state_idx) if hasattr(state_idx, "__len__") else 1,
                    len(meas_idx) if hasattr(meas_idx, "__len__") else 1)
            
            state_idx = KalmanStateRecord._normalize_index(state_idx, n)
            meas_idx  = KalmanStateRecord._normalize_index(meas_idx, n)
            
            for i in range(n):
                if not self.exclusive_mapping(state_idx[i], meas_idx[i]):
                    return False

            return True if n else False

        else:

            s = state_idx.value if hasattr(state_idx, 'value') else int(state_idx)
            assert 0 <= s < self.n
            m = meas_idx.value  if hasattr(meas_idx,  'value') else int(meas_idx)
            assert 0 <= m < self.m

            return self.H[m, s] != 0

    def inclusive_mapping(self, state_idx: int|tuple, meas_idx: int|tuple) -> bool:
        """Check if measurement(s) depend on state(s) in any way."""
        if isinstance(state_idx, (tuple, list, np.ndarray)) or isinstance(meas_idx, (tuple, list, np.ndarray)):
            # Determine broadcasting length
            n = max(len(state_idx) if hasattr(state_idx, "__len__") else 1,
                    len(meas_idx) if hasattr(meas_idx, "__len__") else 1)
            
            state_idx = KalmanStateRecord._normalize_index(state_idx, n)
            meas_idx  = KalmanStateRecord._normalize_index(meas_idx, n)

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

    def get_measurement(self, k = None, indexing = None):
        return KalmanStateRecord._normalize_out(self.measurement, indexing=indexing) if k is None else self.measurement[k]

    def get_observation(self, H = None, k=None, indexing = None):
        H = self.H if H is None else np.asarray(H)
        assert H.shape == (self.m, self.n), f"Observation matrix H has incorrect dimensions. Expected {self.m}x{self.n}, got {format_shape(H)}."
        return KalmanStateRecord._normalize_out([H @ x for x in self.xk_hat] if k is None else H @ self.xk_hat[k], indexing=indexing)

    def get_control(self, k = None, indexing = None):
        return self.get_control_vector(k, indexing=indexing), self.B
    def get_control_vector(self, k = None, indexing = None):
        if self.control is None or len(self.control) == 0:
            return KalmanStateRecord._normalize_out(np.zeros((0, self.l))) if k is None else np.zeros((1, self.l))
        return KalmanStateRecord._normalize_out(self.control, indexing=indexing) if k is None else self.control[k]
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
            x0 = KalmanStateRecord._normalize_in(x0)
            if x0.shape != (self.n,1):
                warnings.warn(
                    f"Initial state vector dimension expected {self.n}x1, got {format_shape(x0)}. Automatic resizing applied."
                )
                x0 = x0.copy()
                x0.resize((self.n, 1))

            self.x0 = x0

        if P0 is not None:
            P0 = KalmanStateRecord._normalize_in(P0)
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
        self.smooth = self.smooth if self.smooth is not None else self.__smooth()
        return self.smooth if i is None else self.smooth[i]
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