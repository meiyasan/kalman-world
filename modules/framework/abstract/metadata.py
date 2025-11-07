from __future__ import annotations

from abc import ABC

import numpy as np
import sys

import json, yaml
from ...runtime.debug import get_debug
from .contracts import KalmanModelABC, KalmanEngineABC

class KalmanMetadata(ABC):

    def name(self):
        return type(self).NAME if hasattr(type(self), "NAME") else "Generic Kalman Model"
    def description(self):
        return type(self).DESCRIPTION if hasattr(type(self), "DESCRIPTION") else "No description available."
    def version(self):
        return type(self).VERSION if hasattr(type(self), "VERSION") else "0.0"

    # ------------------------------------------------------
    # Serialization and Deserialization
    # ------------------------------------------------------
    @classmethod
    def read(cls, filename):
        if filename.endswith('.json'):
            with open(filename, 'r') as file:
                data_dict = json.load(file)
        elif filename.endswith('.npz'):
            data = np.load(filename, allow_pickle=True)
            data_dict = {key: data[key].tolist() if isinstance(data[key], np.ndarray) else data[key] for key in data}
        elif filename.endswith(('.yaml', '.yml')):
            with open(filename, 'r') as file:
                data_dict = yaml.safe_load(file)
        elif filename.endswith('.xml'):
            import xml.etree.ElementTree as ET
            tree = ET.parse(filename)
            root = tree.getroot()
            data_dict = {child.tag: child.text for child in root}
        else:
            raise ValueError("Unsupported file format. Use '.json', '.npz', '.yaml', or '.xml'.")
        # Let concrete classes decide how to instantiate:
        return data_dict

    def write(self, filename):
        data_dict = {key: (value.tolist() if isinstance(value, np.ndarray) else value)
                     for key, value in self.__dict__.items()}
        if filename.endswith('.json'):
            with open(filename, 'w') as file:
                json.dump(data_dict, file, indent=4)
        elif filename.endswith('.npz'):
            np.savez(filename, **data_dict)
        elif filename.endswith(('.yaml', '.yml')):
            with open(filename, 'w') as file:
                yaml.safe_dump(data_dict, file, default_flow_style=False)
        elif filename.endswith('.xml'):
            import xml.etree.ElementTree as ET
            root = ET.Element("KalmanModel")
            for key, value in data_dict.items():
                child = ET.SubElement(root, key)
                child.text = repr(value)
            ET.ElementTree(root).write(filename)
        else:
            raise ValueError("Unsupported file format. Use '.json', '.npz', '.yaml', or '.xml'.")

    def print(self):
        print(self.__str__())
    
class KalmanModelMetadata(KalmanModelABC, KalmanMetadata):

    NAME = "Kalman Model"
    DESCRIPTION = "Base class for Kalman state models, providing a framework for state estimation and filtering."
    VERSION = "1.0"

    dtype: type = np.float64 # Numerical data type

    t0 = None # Initial time
    fs = None # Sampling frequency

    x0 = None # Initial state vector
    P0 = None # Initial covariance matrix

    n = None  # State dimension
    m = None  # Measurement dimension
    l = None  # Control dimension

    _Q = None  # Process noise covariance
    _R = None  # Measurement noise covariance

    # ============================================================
    # Initial/default vectors
    # ============================================================
    @property
    def u0(self) -> np.ndarray:
        """Default zero control input vector."""
        return np.zeros((self.l, 1))
    @property
    def w0(self) -> np.ndarray:
        """Default zero process noise vector."""
        return np.zeros((self.n, 1))
    @property
    def v0(self) -> np.ndarray:
        """Default zero measurement noise vector."""
        return np.zeros((self.m, 1))
    
    # --------------------------------------------------------------
    # Sensor axis management
    # --------------------------------------------------------------
    _sensor: list[str] = None # List of sensor names

    def sensor(self, id) -> list[str]:
        if self._sensor is None:
            return None
        if isinstance(id, int):
            if id < 0 or id >= len(self._sensor):
                raise IndexError(f"Sensor ID #{id} is out of range: 0 to {len(self._sensor)-1}.")
            return self._sensor[id]
        elif isinstance(id, str):
            if id not in self._sensor:
                raise ValueError(f"Sensor ID {id!r} not found in sensor list.")
            return id
        else:
            raise TypeError("Sensor ID must be an integer index or a string name.")

    def add_sensor(self, sensor_id: str | int):
        if self._sensor is None:
            self._sensor = []
        if sensor_id in self._sensor:
            raise ValueError(f"Sensor ID {sensor_id!r} already exists in sensor list.")
        self._sensor.append(sensor_id)
        
    def remove_sensor(self, sensor_id: str | int):
        if self._sensor is not None and sensor_id in self._sensor:
            self._sensor.remove(sensor_id)

    def clear_sensors(self):
        self._sensor = []

    # ------------------------------------------------------
    # Optional dictionary for extra metadata
    # ------------------------------------------------------
    metadata : dict = None
    def __getitem__(self, key): 
        return self.get(key)
    def __setitem__(self, key, val): 
        self.set(key, val)
    def __contains__(self, key): 
        return key in self.items()

    def get(self, key, default=None): 
        if self.metadata is None:
            self.metadata = {}
        return self.metadata.get(key, default)
    def set(self, key, val): 
        if self.metadata is None:
            self.metadata = {}
        self.metadata[key] = val
    def keys(self): 
        if self.metadata is None:
            self.metadata = {}
        return self.metadata.keys()
    def items(self): 
        if self.metadata is None:
            self.metadata = {}
        return self.metadata.items()

    # ------------------------------------------------------
    # String Representation
    # ------------------------------------------------------
    def __str__(self):
        import textwrap
        opt = np.get_printoptions()
        np.set_printoptions(linewidth=np.inf)

        def _print_dimensions(name, matrix):
            if matrix is None:
                return f"{name}: None"
            if isinstance(matrix, np.ndarray):
                shape = "x".join(map(str, matrix.shape))
                dims = "dim" if (matrix.ndim == 1 and matrix.shape[0] == 1) else "dims"
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

        def _find_kalman_base(cls):

            name = type(self).__name__
            if not name.startswith("Kalman"):
                for base in cls.__mro__:
                    if base.__name__.startswith("Kalman"):
                        name += "("+base.__name__+")"
                        break

            return name

        kalman_base = _find_kalman_base(type(self))
        result  = f"{kalman_base} @ fs={getattr(self, 'fs', None)}Hz\n"
        result += f"{1*spacer}{self.NAME} (v{self.VERSION})\n"
        result += f"{2*spacer}{description}\n"
        result += f"{1*spacer}Dimensions Summary:\n"
        result += f"{2*spacer}State Vector....................................: {self.n} dim{'s' if self.n != 1 else ' '} ({self.states.repr()})\n"
        result += f"{2*spacer}Observation.....................................: {self.m} dim{'s' if self.m != 1 else ' '} ({self.observables.repr()})\n"
        result += f"{2*spacer}Control Input...................................: {self.l} dim{'s' if self.l != 1 else ' '}\n"

        result += f"{1*spacer}Initial State:\n"
        result += f"{2*spacer}{_print_dimensions('State Vector (x0)...............................', self.x0)}\n"
        result += (f"{2*spacer}" + _print(self.x0, prefix=2*spacer) + "\n\n") if get_debug() else ""
        result += f"{2*spacer}{_print_dimensions('State Covariance (P0)...........................', self.P0)}\n"
        result += (f"{2*spacer}" + _print(self.P0, prefix=2*spacer) + "\n") if get_debug() else ""

        np.set_printoptions(**opt)

        metadata_keys = sorted(self.keys())
        if metadata_keys:
            result += f"{1*spacer}Metadata Entries:\n"
            max_length = 0
            for key in metadata_keys:
                max_length = max(max_length, len(str(key)))
            for key in metadata_keys:
                value = self.get(key)
                spacer2 = " " * (max_length - len(str(key)))
                result += f"{2*spacer}{key}{spacer2} = {value} ({type(value).__name__})\n"

        return result

class KalmanEngineMetadata(KalmanEngineABC, KalmanMetadata):

    NAME = "Kalman Engine"
    DESCRIPTION = "Base class for Kalman engines."
    VERSION = "1.0"

    _model : KalmanModelMetadata = None
    epsilon = sys.float_info.epsilon

    # Microstate variables (single observation at a time, updated iteratively)
    x = None
    P = None

    # Macrostate variables (multiple observation combined over time)
    prior       = None
    posterior   = None
    smooth      = None
    measurement = None
    control     = None
    transition  = None

    def __str__(self):

        import textwrap
        import numpy as np
        from ...runtime.debug import get_debug

        spacer = " " * 4
        desc = textwrap.fill(self.DESCRIPTION, width=60, subsequent_indent=2*spacer)
        debug = get_debug()

        def fmt_arr(a):
            if a is None:
                return "None"
            a = np.asarray(a)
            shape = "x".join(map(str, a.shape))
            if not debug:
                return f"{shape}"
            return np.array2string(a, precision=3, max_line_width=np.inf)

        res  = f"{self.NAME} (v{self.VERSION})\n"
        res += f"{spacer}{desc}\n\n"
        if self._model:
            res += f"{spacer}Model.....................: {type(self._model).__name__}\n"
            res += f"{spacer}State Space Hz............: {self._model.state} Hz\n"
            res += f"{spacer}Observable Space Hz.......: {self._model.observables} Hz\n"
            res += f"{spacer}Sampling frequency (fs)...: {self._model.fs} Hz\n"
        res += f"{spacer}Precision.................: {self.epsilon:.3e}\n\n"

        res += f"{spacer}Microstate variable sizes:\n"
        res += f"{2*spacer}x.....................: {fmt_arr(self.x)} element(s)\n"
        res += f"{2*spacer}P.....................: {fmt_arr(self.P)} element(s)\n\n"

        res += f"{spacer}Macrostate variable sizes:\n"
        res += f"{2*spacer}Prior.................: {len(self.prior)} element(s)\n"
        res += f"{2*spacer}Posterior.............: {len(self.posterior)} element(s)\n"
        res += f"{2*spacer}Measurement...........: {len(self.measurement)} element(s)\n"
        res += f"{2*spacer}Control...............: {len(self.control)} element(s)\n"

        if debug and self.records:
            res += f"\n{spacer}Record preview (first entry):\n"
            first = self.records[0]
            for k, v in first.items():
                res += f"{2*spacer}{k:<12}: {v}\n"

        return res