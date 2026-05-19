import sys
sys.path.insert(0, 'share/notebooks')
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from kalman.apps.ballistics import BallisticModel
from kalman.framework.filter import KalmanFilter
from kalman.framework.io.record import KalmanRecord

INIT = [[100, 100, 0], [200, 100, 0], [300, 100, -9.81]]
C_DRAG_TRUE = 0.005

class BallisticAugmentedModel(BallisticModel):
    def __init__(self, name, *, drag_process_noise=1e-5, drag_initial_uncertainty=5e-3, **kwargs):
        kwargs['c_drag'] = 0.0
        super().__init__(name, **kwargs)
        q_c = float(drag_process_noise)
        sigma_c = float(drag_initial_uncertainty)
        self.n = 10
        Q10 = np.zeros((10, 10), dtype=self.dtype)
        Q10[:9, :9] = self._Q.copy()
        Q10[9, 9] = q_c ** 2
        self._Q = Q10
        x0 = np.zeros((10, 1), dtype=self.dtype)
        x0[:9] = self.x0
        self.x0 = x0
        P0 = np.zeros((10, 10), dtype=self.dtype)
        P0[:9, :9] = self.P0
        P0[9, 9] = sigma_c ** 2
        self.P0 = P0
        _parent_f = BallisticModel.f.__get__(self, type(self))
        _parent_h = BallisticModel.h.__get__(self, type(self))
        _model = self
        dtype = self.dtype

        def _aug_f(x, u=None, w=None, t=None):
            x = np.asarray(x, dtype=dtype).reshape(-1, 1)
            old = _model["c_drag"]
            _model["c_drag"] = float(x[9, 0])
            x9 = _parent_f(x[:9], u=u, w=None, t=t)
            _model["c_drag"] = old
            xn = np.zeros((10, 1), dtype=dtype)
            xn[:9] = np.asarray(x9, dtype=dtype).reshape(9, 1)
            xn[9, 0] = float(x[9, 0])
            if w is not None:
                xn += np.asarray(w, dtype=dtype).reshape(10, 1)
            return xn

        def _aug_h(x, v=None, t=None):
            x = np.asarray(x, dtype=dtype).reshape(-1, 1)
            return _parent_h(x[:9], v=v, t=t)

        self.f = _aug_f
        self.h = _aug_h
        self.attach(KalmanFilter(self))

    def initial_state(self, x0=None, P0=None):
        print(f'  [override] initial_state called, n={self.n}')
        if x0 is not None:
            x_arr = KalmanRecord._normalize_in(x0)
            print(f'  [override] normalized shape={x_arr.shape}')
            if x_arr.shape[0] == 9:
                x10 = np.zeros((10, 1), dtype=getattr(self, 'dtype', np.float64))
                x10[:9] = x_arr
                x0 = x10
                print(f'  [override] padded x0[9]={x10[9, 0]}')
        super().initial_state(x0=x0, P0=P0)
        print(f'  [override] done: self.x0.shape={self.x0.shape}, x0[9]={self.x0[9, 0]:.4f}')


# Build truth model
truth_m = BallisticModel('XYZ', order=3, fs=32, c_drag=C_DRAG_TRUE,
                          measurement_uncertainty=200, process_uncertainty=0.3, random_seed=42)
truth_m.initial_state(INIT)
truth_s = truth_m.sample(8)

# Build augmented model
aug_m = BallisticAugmentedModel('XYZ', order=3, fs=32,
                                 measurement_uncertainty=200,
                                 process_uncertainty=0.3,
                                 drag_process_noise=1e-5,
                                 drag_initial_uncertainty=5e-3)
print(f'After __init__: x0={aug_m.x0.flatten()}')
aug_m.initial_state(INIT)
print(f'After initial_state: x0={aug_m.x0.flatten()}')

r_aug = aug_m.filter(truth_s.zk, smooth=False, reset=True)
print(f'xk shape={r_aug.xk.shape}')
print(f'c_drag trajectory: first5={r_aug.xk[:5, 9]}  last5={r_aug.xk[-5:, 9]}')
print(f'c_drag_est={r_aug.xk[-1, 9]:.6f}  truth={C_DRAG_TRUE}')
