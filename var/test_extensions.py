import sys
sys.path.insert(0, 'share/notebooks')
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from kalman.apps.ballistics import BallisticModel
from kalman.filters.ukf import UnscentedKalmanFilter
from kalman.helpers import make_multirate_input

INIT = [[100,100,0],[200,100,0],[300,100,-9.81]]

# ---- Doppler ----
class BallisticDopplerModel(BallisticModel):
    def __init__(self, *args, vel_noise=5.0, **kwargs):
        super().__init__(*args, **kwargs)
        old_R = self._R.copy()
        new_R = np.zeros((4, 4), dtype=self.dtype)
        new_R[:3, :3] = old_R
        new_R[3, 3] = float(vel_noise) ** 2
        self._R = new_R
        self.m = 4
    def h(self, x, v=None, t=None):
        x = np.asarray(x).reshape(-1)
        p = x[[0,3,6]]; vel = x[[1,4,7]]
        r = float(np.linalg.norm(p))
        r_hat = p/r if r>1 else np.array([0.,0.,1.], dtype=self.dtype)
        vrad = float(vel @ r_hat)
        z = np.zeros((4,1), dtype=self.dtype)
        z[:3,0] = p; z[3,0] = vrad
        if v is not None:
            z += np.asarray(v).reshape(4,1)
        return z

m_dop = BallisticDopplerModel('XYZ', order=3, fs=64, measurement_uncertainty=300,
                               process_uncertainty=0.3, random_seed=7)
m_dop.initial_state(INIT)
s = m_dop.sample(4)
print(f'[1] Doppler sample OK  zk={s.zk.shape}')
r = m_dop.filter(s.zk, smooth=False, reset=True)
print(f'[1] Doppler filter OK  xk={r.xk.shape}  finite={np.isfinite(r.xk).all()}')

# ---- Augmented drag ----
class BallisticAugmentedModel(BallisticModel):
    def __init__(self, name, *, drag_process_noise=1e-4, **kwargs):
        kwargs['c_drag'] = 0.0
        super().__init__(name, **kwargs)
        q_c = float(drag_process_noise)
        self.n = 10
        Q9 = self._Q.copy()
        Q10 = np.zeros((10,10), dtype=self.dtype)
        Q10[:9,:9] = Q9; Q10[9,9] = q_c**2
        self._Q = Q10
        x0 = np.zeros((10,1), dtype=self.dtype); x0[:9] = self.x0; self.x0 = x0
        P0 = np.zeros((10,10), dtype=self.dtype); P0[:9,:9] = self.P0; P0[9,9] = 1.0; self.P0 = P0
    def f(self, x, u=None, w=None, t=None):
        x = np.asarray(x, dtype=self.dtype).reshape(-1,1)
        old = self["c_drag"]; self["c_drag"] = float(x[9,0])
        x9 = super().f(x[:9], u=u, w=None, t=t)
        self["c_drag"] = old
        xn = np.zeros((10,1), dtype=self.dtype)
        xn[:9] = np.asarray(x9).reshape(9,1); xn[9,0] = float(x[9,0])
        if w is not None:
            xn += np.asarray(w, dtype=self.dtype).reshape(10,1)
        return xn
    def h(self, x, v=None, t=None):
        x = np.asarray(x, dtype=self.dtype).reshape(-1,1)
        return super().h(x[:9], v=v, t=t)

truth_m = BallisticModel('XYZ', order=3, fs=32, c_drag=0.005,
                          measurement_uncertainty=200, process_uncertainty=0.3, random_seed=42)
truth_m.initial_state(INIT)
truth_s = truth_m.sample(8)
print(f'[2] Truth sample OK  zk={truth_s.zk.shape}')

aug_m = BallisticAugmentedModel('XYZ', order=3, fs=32, measurement_uncertainty=200,
                                 process_uncertainty=0.3, drag_process_noise=5e-4)
aug_m.initial_state(INIT)
r_aug = aug_m.filter(truth_s.zk, smooth=False, reset=True)
print(f'[2] Augmented filter OK  xk={r_aug.xk.shape}  c_drag_est={r_aug.xk[-1,9]:.5f}  (truth=0.005)')

# ---- UKF ----
m_ukf = BallisticModel('XYZ', order=3, fs=64, c_drag=0.002,
                        measurement_uncertainty=300, process_uncertainty=0.4, random_seed=99)
m_ukf.initial_state([[100,80,0],[200,80,0],[300,80,-9.81]])
s_ukf = m_ukf.sample(4)
ukf = UnscentedKalmanFilter(model=m_ukf, alpha=1e-3)
r_ukf = ukf.forward(s_ukf.zk, reset=True)
print(f'[3] UKF filter OK  xk={r_ukf.xk.shape}  finite={np.isfinite(r_ukf.xk).all()}')

# ---- Multi-rate ----
m_mr = BallisticModel('XYZ', order=3, fs=128, measurement_uncertainty=300,
                       process_uncertainty=0.5, random_seed=17)
m_mr.initial_state([[100,100,0],[200,100,0],[300,100,-9.81]])
s_mr = m_mr.sample(4)
step = int(round(128/10))
z_gps = s_mr.zk[::step]
z_mri = make_multirate_input([(z_gps, 10.0)], base_fs=128)
r_mr = m_mr.filter(z_mri, smooth=False, reset=True)
nan_frac = np.isnan(z_mri[:,0]).mean()
print(f'[4] Multi-rate filter OK  xk={r_mr.xk.shape}  NaN fraction={nan_frac:.3f}')

print('\nAll extension tests PASSED')
