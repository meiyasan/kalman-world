"""Linear ballistic model — constant-acceleration kinematics (KF).

The ``LINEAR`` variant of the ballistics domain: exact ``expm`` discretization
of the constant-acceleration generator, no drag.  See ``extended.py`` for the
nonlinear (drag + gravity) EKF variant.
"""
import numpy as np
from scipy.linalg import expm

from ...framework.models.linear import LinearKalmanModel
from .base import BallisticState, BallisticObservable, BallisticNoise, ApexEstimate, ImpactEstimate

# ===== Linear =====
class BallisticLinearModel(LinearKalmanModel):

    NAME = "Ballistic State Model"
    DESCRIPTION = "Linear constant-acceleration toy model; position measured."
    VERSION = "1.3"

    @property
    def states(self):
        return BallisticState
    @property
    def observables(self):
        return BallisticObservable

    def __init__(self, sensor_id: str | int,
                 measurement_uncertainty: float = 0.5,  # position-sensor noise std σ [m]; R = σ²
                 process_uncertainty: float = 0.5,  # acceleration-noise std [m/s²]; see Q note for magnitude
                 **kwargs):

        self.dtype = kwargs.pop("dtype", np.float64)

        # -------------------------------------------
        # State transition and measurement matrices
        # -------------------------------------------
        fs = kwargs.pop("fs", None) or kwargs.pop("sample_rate", None) or getattr(self, "fs", None)
        assert isinstance(fs, (int, float)), "You must provide fs or sample_rate"

        # Continuous-time generator: ż=v, v̇=a, ȧ=0 (constant acceleration) —
        # see the class docstring's Motion Model / A matrix. Passed as A= (not
        # F=): the base KalmanModel discretizes exactly via F=expm(A/fs).
        A = np.array([[0.0, 1.0, 0.0],
                      [0.0, 0.0, 1.0],
                      [0.0, 0.0, 0.0]], dtype=self.dtype)

        H = np.array([[1.0, 0.0, 0.0]], dtype=self.dtype)

        # -------------------------------------------
        # Process noise — continuous white-noise-ACCELERATION disturbance,
        # designed at the v̇ equation (Lc picks out that row); passed as Qc=/Lc=
        # so the base model discretizes it exactly (Van Loan) alongside A=.
        # This is a CONSTANT-acceleration model, so there is deliberately NO
        # jerk option: the acceleration STATE itself gets no direct disturbance
        # (Lc's 3rd entry is 0), so it stays flat (e.g. -9.81) and the resulting
        # Q's acceleration row/col comes out identically zero — the disturbance
        # only smears velocity (∝dt) and position (∝dt³); Qc=0 ⇒ truth == ideal.
        #
        # Order of magnitude (~20 s flight):
        #     0.05 m/s² → ~1 m drift     0.5 m/s² → ~10 m (default)     2 m/s² → ~35 m
        # -------------------------------------------
        Lc = np.array([[0.0], [1.0], [0.0]], dtype=self.dtype)
        Qc = np.array([[self.dtype(process_uncertainty) ** 2]], dtype=self.dtype)

        # measurement_uncertainty is the position-sensor noise std σ [m]; the
        # filter needs the variance, so R = σ²·I (consistent with the EKF model
        # and with the P0 position prior below, both of which square it).
        r = self.dtype(measurement_uncertainty)
        R = (r ** 2) * np.eye(1, dtype=self.dtype)

        # Initial covariance P0 — a positive-definite diagonal built from the two
        # Kalman noises only (measurement R and process Q); there is no separate
        # acceleration knob:
        #   position     : R                       (position is what we measure)
        #   velocity     : (measurement_uncertainty/10)²   (unobserved → looser)
        #   acceleration : q = process_uncertainty²        (from the process noise)
        # Tying the acceleration prior to the PROCESS noise (not the sensor noise)
        # keeps it sensible for quiet and noisy sensors alike, and means a smaller
        # process noise automatically pins the acceleration closer to its initial value.
        if "P0" not in kwargs:
            p_var = self.dtype(measurement_uncertainty) ** 2
            v_var = (self.dtype(measurement_uncertainty) / 10.0) ** 2
            a_var = self.dtype(process_uncertainty) ** 2
            kwargs["P0"] = np.diag([p_var, v_var, a_var]).astype(self.dtype)

        super().__init__(sensor_id, A=A, H=H, Qc=Qc, Lc=Lc, R=R, fs=fs, **kwargs)

    def preprocess(self, zk, uk=None, **kwargs): return zk, uk
    def inprocess(self, state_vector, **kwargs): return state_vector
    def postprocess(self, xk_hat, **kwargs):     return xk_hat

    # ------------------------------------------------------------------
    # Trajectory estimators (closed-form for constant acceleration)
    #
    # Both take a [position, speed, acceleration] state — e.g. one row of a
    # filtered/smoothed track, ``record.xk[k]`` — and assume the linear model's
    # own constant-acceleration dynamics z(t) = z + ż·t + ½·z̈·t² hold from there
    # on. Pass the matching covariance ``record.Pk[k]`` to also get 1-σ error bars
    # by linear (first-order) propagation σ² = Jᵀ·P·J. Times are relative to the
    # epoch of ``state``; add ``record.time[k]`` for an absolute time.
    # ------------------------------------------------------------------
    def _zva(self, state, cov):
        """Unpack a state into scalars (z, ż, z̈) and a 3×3 covariance (or None)."""
        x = self.x0 if state is None else state
        x = np.asarray(x, dtype=self.dtype).reshape(-1)
        if x.shape[0] != 3:
            raise ValueError(f"apex/impact_time need a [position, speed, acceleration] "
                             f"state (n=3); got length {x.shape[0]}.")
        P = None if cov is None else np.asarray(cov, dtype=self.dtype).reshape(3, 3)
        return float(x[0]), float(x[1]), float(x[2]), P

    def apex(self, state=None, cov=None):
        """Maximum height of the constant-acceleration path and the time to reach it.

        The height z(t) = z + ż·t + ½·z̈·t² peaks where the velocity vanishes, at
        ``t* = −ż / z̈`` (needs z̈ ≠ 0; t* is negative if the body is already past its
        peak), giving ``z* = z − ż² / (2·z̈)``. With ``cov`` given, returns 1-σ bars too.

        Returns an :class:`ApexEstimate` (time, height, time_std, height_std);
        ``state`` defaults to the model's initial state (the nominal launch).
        """
        z, v, a, P = self._zva(state, cov)
        if a == 0.0:
            raise ValueError("Acceleration is zero — a constant-velocity path has no apex.")
        t_star = -v / a
        height = z - v * v / (2.0 * a)
        t_std = h_std = None
        if P is not None:
            J_t = np.array([0.0, -1.0 / a, v / (a * a)], dtype=self.dtype)
            J_h = np.array([1.0, -v / a, v * v / (2.0 * a * a)], dtype=self.dtype)
            t_std = float(np.sqrt(max(J_t @ P @ J_t, 0.0)))
            h_std = float(np.sqrt(max(J_h @ P @ J_h, 0.0)))
        return ApexEstimate(t_star, height, t_std, h_std)

    def impact_time(self, state=None, cov=None, level=0.0):
        """Time at which the trajectory crosses ``level`` (default 0) coming down.

        Solves ½·z̈·t² + ż·t + (z − level) = 0 and returns the future **descending**
        root (the "hitting time"). With ``cov`` given, the 1-σ time uncertainty is
        propagated via the implicit derivative  dt/dx = −[1, t, ½t²] / v_impact.

        Returns an :class:`ImpactEstimate` (time, speed-at-impact, time_std). Raises if
        the path never reaches ``level`` (negative discriminant) or only in the past.
        """
        z, v, a, P = self._zva(state, cov)
        c = z - level
        tol = 1e-9
        if a == 0.0:                                   # constant-velocity fallback
            if abs(v) < tol:
                raise ValueError("Static state never crosses the level.")
            roots = [-c / v]
        else:
            disc = v * v - 2.0 * a * c
            if disc < 0.0:
                raise ValueError(f"Trajectory never reaches level={level} (no real crossing).")
            sq = float(np.sqrt(disc))
            roots = [(-v + sq) / a, (-v - sq) / a]
        # prefer a future, descending (velocity < 0) crossing; else any future root
        descending = [t for t in roots if t > tol and (v + a * t) < 0.0]
        future     = descending or [t for t in roots if t > tol]
        if not future:
            raise ValueError("No positive-time crossing of the level.")
        t = min(future)
        v_imp = v + a * t
        t_std = None
        if P is not None and abs(v_imp) > tol:
            J = -np.array([1.0, t, 0.5 * t * t], dtype=self.dtype) / v_imp
            t_std = float(np.sqrt(max(J @ P @ J, 0.0)))
        return ImpactEstimate(t, v_imp, t_std)

