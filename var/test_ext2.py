import sys
sys.path.insert(0, 'share/notebooks')
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from kalman.apps.ballistics import BallisticModel
from kalman.framework.filter import KalmanFilter
from kalman.filters.ukf import UnscentedKalmanFilter
from kalman.helpers import make_multirate_input

INIT = [[100, 100, 0], [200, 100, 0], [300, 100, -9.81]]

# ---- Extension 1: Doppler (instance-attribute h patch) ----
class BallisticDopplerModel(BallisticModel):
    def __init__(self, *args, vel_noise=5.0, **kwargs):
        super().__init__(*args, **kwargs)
        old_R = self._R.copy()
        new_R = np.zeros((4, 4), dtype=self.dtype)
        new_R[:3, :3] = old_R
        new_R[3, 3] = float(vel_noise) ** 2
        self._R = new_R
        self.m = 4
        dtype = self.dtype
        def _h(x, v=None, t=None):
            x = np.asarray(x, dtype=dtype).reshape(-1)
            p = x[[0, 3, 6]]
            vel = x[[1, 4, 7]]
            r = float(np.linalg.norm(p))
            r_hat = p / r if r > 1 else np.array([0., 0., 1.], dtype=dtype)
            z = np.zeros((4, 1), dtype=dtype)
            z[:3, 0] = p
            z[3, 0] = float(vel @ r_hat)
            if v is not None:
                z += np.asarray(v, dtype=dtype).reshape(4, 1)
            return z
        self.h = _h
        self.attach(KalmanFilter(self))

m_dop = BallisticDopplerModel('XYZ', order=3, fs=64,
                               measurement_uncertainty=300,
                               process_uncertainty=0.3,
                               random_seed=7)
m_dop.initial_state(INIT)
s = m_dop.sample(4)
print(f'[1] Doppler sample  zk={s.zk.shape}')
r = m_dop.filter(s.zk, smooth=False, reset=True)
print(f'[1] Doppler filter  xk={r.xk.shape}  finite={np.isfinite(r.xk).all()}')

# ---- Extension 2: Augmented drag ----
class BallisticAugmentedModel(BallisticModel):
    def __init__(self, name, *, drag_process_noise=1e-5, drag_initial_uncertainty=5e-3, **kwargs):
        kwargs['c_drag'] = 0.0
        super().__init__(name, **kwargs)
        q_c = float(drag_process_noise)
        sigma_c = float(drag_initial_uncertainty)
        self.n = 10
        Q10 = np.zeros((10, 10), dtype=self.dtype)
        Q10[:9, :9] = self._Q.copy(); Q10[9, 9] = q_c ** 2; self._Q = Q10
        x0 = np.zeros((10, 1), dtype=self.dtype); x0[:9] = self.x0; self.x0 = x0
        P0 = np.zeros((10, 10), dtype=self.dtype); P0[:9, :9] = self.P0; P0[9, 9] = sigma_c ** 2; self.P0 = P0
        _parent_f = BallisticModel.f.__get__(self, type(self))
        _parent_h = BallisticModel.h.__get__(self, type(self))
        _model = self; dtype = self.dtype
        def _aug_f(x, u=None, w=None, t=None):
            x = np.asarray(x, dtype=dtype).reshape(-1, 1)
            old = _model["c_drag"]; _model["c_drag"] = float(x[9, 0])
            x9 = _parent_f(x[:9], u=u, w=None, t=t); _model["c_drag"] = old
            xn = np.zeros((10, 1), dtype=dtype)
            xn[:9] = np.asarray(x9, dtype=dtype).reshape(9, 1); xn[9, 0] = float(x[9, 0])
            if w is not None: xn += np.asarray(w, dtype=dtype).reshape(10, 1)
            return xn
        def _aug_h(x, v=None, t=None):
            x = np.asarray(x, dtype=dtype).reshape(-1, 1)
            return _parent_h(x[:9], v=v, t=t)
        self.f = _aug_f; self.h = _aug_h
        self.attach(KalmanFilter(self))

    def initial_state(self, x0=None, P0=None):
        """Override to pad a 9D initial state to 10D, setting c_drag=0."""
        if x0 is not None and getattr(self, 'n', 9) == 10:
            flat = np.asarray(x0, dtype=getattr(self, 'dtype', np.float64)).ravel()
            if flat.size == 9:
                x10 = np.zeros((10, 1), dtype=flat.dtype)
                x10[:9, 0] = flat
                x0 = x10
        super().initial_state(x0=x0, P0=P0)

truth_m = BallisticModel('XYZ', order=3, fs=32, c_drag=0.005,
                          measurement_uncertainty=200,
                          process_uncertainty=0.3,
                          random_seed=42)
truth_m.initial_state(INIT)
truth_s = truth_m.sample(8)
print(f'[2] Truth sample  zk={truth_s.zk.shape}')

aug_m = BallisticAugmentedModel('XYZ', order=3, fs=32,
                                 measurement_uncertainty=200,
                                 process_uncertainty=0.3,
                                 drag_process_noise=1e-5,
                                 drag_initial_uncertainty=5e-3)
aug_m.initial_state(INIT)
r_aug = aug_m.filter(truth_s.zk, smooth=False, reset=True)
print(f'[2] Aug filter  xk={r_aug.xk.shape}  c_drag_est={r_aug.xk[-1,9]:.5f}  (truth=0.005)')

# ---- Extension 3: UKF ----
m_ukf = BallisticModel('XYZ', order=3, fs=64, c_drag=0.002,
                        measurement_uncertainty=300,
                        process_uncertainty=0.4,
                        random_seed=99)
m_ukf.initial_state([[100, 80, 0], [200, 80, 0], [300, 80, -9.81]])
s_ukf = m_ukf.sample(4)
ukf = UnscentedKalmanFilter(model=m_ukf, alpha=1e-3)
xk_u, Pk_u = ukf.filter(s_ukf.zk, reset=True)
print(f'[3] UKF filter  xk={xk_u.shape}  finite={np.isfinite(xk_u).all()}')

# ---- Extension 4: Multi-rate ----
m_mr = BallisticModel('XYZ', order=3, fs=128,
                       measurement_uncertainty=300,
                       process_uncertainty=0.5,
                       random_seed=17)
m_mr.initial_state([[100, 100, 0], [200, 100, 0], [300, 100, -9.81]])
s_mr = m_mr.sample(4)
step = int(round(128.0 / 10.0))
z_gps = s_mr.zk[::step]
z_mri = make_multirate_input([(z_gps, 10.0)], base_fs=128)
r_mr = m_mr.filter(z_mri, smooth=False, reset=True)
print(f'[4] Multi-rate  xk={r_mr.xk.shape}  NaN frac={np.isnan(z_mri[:,0]).mean():.3f}')
print('ALL PASSED')
