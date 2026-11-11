"""Nonlinear ballistic model — gravity + aerodynamic drag + thrust (EKF).

The ``EXTENDED`` variant of the ballistics domain.  See ``linear.py`` for the
constant-acceleration KF variant.
"""
import numpy as np

from ...framework.models.extended import ExtendedKalmanModel
from ...framework.models.native import NativeTwin
from .base import BallisticState, BallisticObservable, BallisticNoise, ApexEstimate, ImpactEstimate

# ===== Nonlinear ballistic EKF (use proper Q here) =====
class BallisticModel(NativeTwin, ExtendedKalmanModel):
    """
    Nonlinear 3D ballistic model with explicit gravity (G*M),
    aerodynamic drag, thrust input, and internally computed Q/R matrices.
    """

    NAME = "Ballistic Extended Model"
    DESCRIPTION = "Nonlinear 3D ballistic model with gravity, drag, and thrust."
    VERSION = "1.0"

    @property
    def states(self):
        return BallisticState
    @property
    def observables(self):
        return BallisticObservable

    def __init__(self, name, *,
                 G=6.674e-11,                 # gravitational constant [m^3/kg/s^2]
                 M=5.972e24,                  # central body mass [kg] (Earth)
                 m=1.0,                       # object mass [kg]
                 c_drag=0,                    # drag coefficient [kg/m]
                 g=9.81,                      # surface gravity [m/s²]; None → use G*M/r³ orbital formula
                 fs=128.0,                    # sampling frequency [Hz]
                 order=3,                     # model order [p,v,a]
                 process_uncertainty=0.5,     # strength of the disturbance (√ of its spectral density); UNIT set by `process_noise_model`
                 process_noise_model=BallisticNoise.ACCELERATION,  # m/s^1.5 (accel; recommended) or m/s^2.5 (jerk; see Qd note)
                 drag_uncertainty=None,       # ε: if set, the FILTER's Q grows with speed as σ_a(v)=ε·(c_drag/m)|v|² (air-friction uncertainty); None → constant Q
                 measurement_uncertainty=0.5, # measurement noise std [m],
                 substeps=1,                  # sub-steps per sample: 1 = one Taylor step (as before); N > 1 or "auto" = velocity-Verlet sub-steps
                 f = None,                    # Transition function (optional, use default one if None)
                 h = None,                    # Measurement function (optional, use default one if None)
                 **kwargs):

        self.dtype = kwargs.pop("dtype", np.float64)
        self.fs = fs
        self.order = order
        # Sub-steps per sample. One step of p += v·dt + ½a·dt², v += a·dt is first order in
        # velocity: in orbital mode at a long sampling interval it is wrong by far more than
        # any usable process noise covers (a 2.2 h orbit sampled every 86 s: ~10 m/s of
        # velocity error per step, and the filter diverged by 1,900 km). N > 1 integrates the
        # same forces in N velocity-Verlet sub-steps; "auto" takes enough that each turns the
        # orbit by at most AUTO_SUBSTEP_RADIANS. 1 keeps the original single step exactly.
        if substeps != "auto" and (int(substeps) < 1):
            raise ValueError(f"substeps must be >= 1 or 'auto', got {substeps!r}.")
        self.substeps = substeps if substeps == "auto" else int(substeps)

        self["c_drag"] = c_drag
        self["G"] = G
        self["M"] = M
        self["m"] = m
        self["g"] = g

        if f is not None:
            self.f = f
        if h is not None:
            self.h = h

        # ------------------------------------------------------------------
        # Noise covariances
        # ------------------------------------------------------------------
        Qd = self.Qd(process_uncertainty, order, process_noise_model=process_noise_model)
        # Air-friction (drag) UNCERTAINTY mode: the constant Qd stays as the base
        # floor; when drag_uncertainty (ε) is set, Q(x) rides the local speed (see
        # Q() below). _Q_unit is the σ²=1 block structure used to rescale it.
        self._sigma_a_base     = float(process_uncertainty)
        self._drag_uncertainty = None if drag_uncertainty is None else float(drag_uncertainty)
        self._Q_unit           = self.Qd(1.0, order, process_noise_model=process_noise_model)
        R = (measurement_uncertainty**2) * np.eye(3, dtype=self.dtype)

        # ------------------------------------------------------------------
        # Initial covariance P0 — PD diagonal per axis-major layout
        # [px,vx,ax, py,vy,ay, pz,vz,az], built from the two Kalman noises only
        # (measurement R and process Q); no separate acceleration knob:
        #   position     : R                              (what we measure)
        #   velocity     : (measurement_uncertainty/10)² (unobserved → looser)
        #   acceleration : process_uncertainty²          (from the process noise)
        # The acceleration prior is sensor-independent (tied to Q, not R). Harmless
        # for this EKF anyway since f() recomputes acceleration from the dynamics each
        # step, but kept consistent with BallisticLinearModel.
        if kwargs.get("P0") is None:
            p_var = measurement_uncertainty ** 2
            v_var = (measurement_uncertainty / 10.0) ** 2
            a_var = float(process_uncertainty) ** 2
            block = [p_var, v_var, a_var] if order == 3 else [p_var, v_var]
            kwargs["P0"] = np.diag(block * 3).astype(self.dtype)

        # Derived sizes
        self.n = 3 * order
        self.m = 3

        super().__init__(name, Q=Qd, R=R, **kwargs)
        self.clear_sensors()
        self.add_sensor("X")
        self.add_sensor("Y")
        self.add_sensor("Z")

    # ----------------------------------------------------------------------
    # C++ twin (libKalmanApps: Kalman::Apps::Ballistics::Ballistic). A
    # speed-dependent Q, a thrust control, a user-supplied f/h and sub-stepping
    # (the twin takes one step per sample) stay in Python.
    # ----------------------------------------------------------------------
    def native_name(self):
        if self._drag_uncertainty is not None or self.substeps != 1:
            return None
        return {3: "ballistic", 2: "ballistic-order2"}.get(self.order)

    def native_parameters(self):
        g = self["g"]
        return {"c_drag": self["c_drag"], "m": self["m"], "G": self["G"], "M": self["M"],
                "g": 0.0 if g is None else g, "surface_gravity": 0.0 if g is None else 1.0}

    # ----------------------------------------------------------------------
    # Process noise covariance generator (internal)
    # ----------------------------------------------------------------------
    def Q(self, t=None, x=None):
        # Constant base covariance, unless air-friction (drag) UNCERTAINTY is enabled
        # AND a state is supplied (the engine passes the predicted x). Then the
        # process-noise acceleration std rides the local speed,
        #     σ_a(v) = ε · (c_drag/m) · |v|² ,
        # added in quadrature to the constant base floor → Q is large at the muzzle
        # (fast, drag poorly known) and ≈ base near apex (slow). Q scales as σ_a²,
        # i.e. ∝ v⁴ in the drag term. The truth sampler still uses the constant base.
        if self._drag_uncertainty is None or x is None:
            return self._Q
        xv = np.asarray(x, dtype=self.dtype).reshape(-1)
        if self.order < 2:
            return self._Q
        vi   = [self.order * i + 1 for i in range(3)]      # velocity slot of each axis
        vmag = float(np.linalg.norm(xv[vi]))
        sigma_drag = self._drag_uncertainty * (self["c_drag"] / self["m"]) * vmag**2
        sigma2     = self._sigma_a_base**2 + sigma_drag**2
        return sigma2 * self._Q_unit
    
    def Qd(self, sigma: float, order: int, process_noise_model=BallisticNoise.ACCELERATION) -> np.ndarray:
        """
        Build the block-diagonal (per-axis) discrete process covariance Q for 3D
        ballistic motion. Uses the same per-axis kinematic blocks as
        BallisticLinearModel (kept consistent by convention, not shared code).

        Parameters
        ----------
        sigma : float
            Strength of the white disturbance (square root of its spectral
            density); its UNIT is set by ``process_noise_model``
            (BallisticNoise.ACCELERATION → m/s^1.5: the velocity wanders by
            sigma·√t; BallisticNoise.JERK → m/s^2.5).
        order : int
            Model order (2 or 3).
        process_noise_model : BallisticNoise
            ACCELERATION (white-noise-acceleration) or JERK (white-noise-jerk) —
            the same per-axis blocks BallisticLinearModel uses.

        Returns
        -------
        Q : ndarray, shape (3*order, 3*order)
            Block-diagonal process covariance matrix.

        Note
        ----
        Default ``ACCELERATION`` drives the disturbance through velocity &
        position, which is the right choice here: this EKF's ``f()`` recomputes the
        acceleration from the dynamics every step (gravity + drag), so it is not a
        free integrator state — a JERK disturbance placed on it would be dropped
        from the propagated truth while still inflating the predicted covariance.
        ``JERK`` is supported for completeness but is only physically meaningful for
        a pure-kinematic constant-acceleration model (i.e. BallisticLinearModel).
        """
        process_noise_model = BallisticNoise.of(process_noise_model)
        q = self.dtype(sigma)**2
        dt = self.dtype(1.0 / self.fs)

        if process_noise_model == BallisticNoise.ACCELERATION:
            if order == 3:
                Q1D = np.array([[dt**3/3.0, dt**2/2.0, 0.0],
                                [dt**2/2.0, dt,        0.0],
                                [0.0,       0.0,       0.0]], dtype=self.dtype)
            elif order == 2:
                Q1D = np.array([[dt**3/3.0, dt**2/2.0],
                                [dt**2/2.0, dt]], dtype=self.dtype)
            else:
                raise ValueError(f"order must be 2 or 3, got {order}.")
        else:  # BallisticNoise.JERK
            if order != 3:
                raise ValueError("JERK noise requires order==3 (acceleration must be a state).")
            Q1D = np.array([[dt**5/20.0, dt**4/8.0, dt**3/6.0],
                            [dt**4/8.0,  dt**3/3.0, dt**2/2.0],
                            [dt**3/6.0,  dt**2/2.0, dt]], dtype=self.dtype)

        Q = np.zeros((3*order, 3*order), dtype=self.dtype)
        for i in range(3):  # fixed XYZ
            s = slice(order*i, order*(i+1))
            Q[s, s] = q * Q1D

        return Q

    # ----------------------------------------------------------------------
    # Nonlinear dynamics
    # ----------------------------------------------------------------------
    def f(self, x, u=None, w=None, t=None):
        """
        Full nonlinear motion model:
            ṗ = v
            v̇ = a_grav(p) + a_drag(v) + a_ctrl(u)
        """
        x = np.asarray(x, dtype=self.dtype).reshape(-1, 1)
        dt = 1.0 / self.fs

        # Extract from axis-major layout: [px,vx,ax, py,vy,ay, pz,vz,az] at order 3, or
        # [px,vx, py,vy, pz,vz] at order 2. The stride is `order`, not a literal 3 -- see h().
        pi = [self.order * i for i in range(3)]
        vi = [self.order * i + 1 for i in range(3)]
        p = x[pi, 0]
        v = x[vi, 0]
        a = (x[[self.order * i + 2 for i in range(3)], 0] if self.order >= 3
             else np.zeros(3, dtype=self.dtype))

        steps = self._substeps_for(p, dt)
        if steps == 1:
            # The original single Taylor step, unchanged.
            a_total = self._acceleration(p, v, u)
            if self.order >= 2:
                p = p + dt * v
            p += 0.5 * a_total * dt**2
            v += a_total * dt
            a = a_total if self.order >= 3 else np.zeros_like(a_total)
        else:
            # Velocity Verlet over `steps` sub-steps: second order in position AND velocity,
            # the acceleration re-evaluated where each sub-step lands.
            h = dt / steps
            a_total = self._acceleration(p, v, u)
            for _ in range(steps):
                p = p + h * v + 0.5 * a_total * h**2
                a_next = self._acceleration(p, v + a_total * h, u)
                v = v + 0.5 * (a_total + a_next) * h
                a_total = a_next
            a = a_total if self.order >= 3 else np.zeros_like(a_total)

        # Compose next state in axis-major layout: [px,vx,ax, py,vy,ay, pz,vz,az]
        x_next = np.zeros_like(x)
        x_next[pi, 0] = p
        x_next[vi, 0] = v
        if self.order >= 3:
            x_next[[self.order * i + 2 for i in range(3)], 0] = a

        # Add process noise
        if w is not None:
            L = self.L(t)
            x_next += w if L is None else (L @ w)

        return x_next

    #: "auto" sub-steps: each turns an orbit by at most this angle (ω = sqrt(GM/r³)).
    AUTO_SUBSTEP_RADIANS = 0.01

    def _substeps_for(self, p, dt):
        """Sub-steps for one sample: the fixed count, or in orbital mode with "auto" enough
        that each covers at most AUTO_SUBSTEP_RADIANS of the local orbital motion."""
        if self.substeps != "auto":
            return self.substeps
        if self["g"] is not None:
            return 1
        r = float(np.linalg.norm(p))
        if r <= 0.0:
            return 1
        omega = np.sqrt(self["G"] * self["M"] / r**3)
        return max(1, min(10_000, int(np.ceil(omega * dt / self.AUTO_SUBSTEP_RADIANS))))

    def _acceleration(self, p, v, u=None):
        """Gravity + drag + control at position p and velocity v (the forces f() integrates)."""
        # Near-surface mode (g provided): constant downward acceleration in Z.
        # Orbital mode (g=None): full geocentric G*M/r³ formula — requires
        # positions in geocentric metres (origin = Earth's centre).
        if self["g"] is not None:
            a_grav = np.array([0.0, 0.0, -self["g"]], dtype=self.dtype)
        else:
            r = np.linalg.norm(p)
            a_grav = -self["G"] * self["M"] * p / r**3 if r > 0 else np.zeros(3, dtype=self.dtype)

        # Drag (depends on velocity, without control)
        vmag = np.linalg.norm(v)
        a_drag = -(self["c_drag"] / self["m"]) * vmag * v if vmag > 0 else np.zeros(3, dtype=self.dtype)

        # Control (optional controlled thrust force)
        a_ctrl = np.zeros(3, dtype=self.dtype)
        if u is not None:
            u_arr = np.asarray(u, dtype=self.dtype).ravel()
            if u_arr.size > 0:
                a_ctrl = u_arr.reshape(3) / self["m"]

        return a_grav + a_drag + a_ctrl

    # ----------------------------------------------------------------------
    # Measurement
    # ----------------------------------------------------------------------
    def h(self, x, v=None, t=None):
        """
        Measurement model: observes position only.
        Extracts [px, py, pz] from axis-major state [px,vx,ax, py,vy,ay, pz,vz,az].
        z = [px, py, pz] + v  (additive noise, by default)
        """
        x = np.asarray(x, dtype=self.dtype).reshape(-1, 1)
        # Positions are the first slot of each axis in the axis-major layout, so they are
        # `order` apart -- not always 0, 3, 6. Hardcoding that stride made the documented
        # order=2 (position+velocity, no acceleration state) raise IndexError on a 6-state
        # vector. Derived the same way self.velocity_indices already derives its own.
        pi = [self.order * i for i in range(3)]
        z = x[pi, :].copy()

        if v is not None:
            v = np.asarray(v, dtype=self.dtype).reshape(3, 1)
            z += v

        return z
    
    def preprocess(self, zk, uk=None, **kwargs):
        return zk, uk
    def inprocess(self, state_vector, **kwargs):
        return state_vector
    def postprocess(self, xk_hat, **kwargs):
        return xk_hat

    # ==================================================================
    # Signature figure — per-axis state grid (override KalmanModel.plot).
    # Reached via ``model.plot(record, sample=sample[, interactive=True])``.
    # ==================================================================
    def _plot_static(self, record, sample=None, *, component=None, error=True,
                     figsize=(14, 9), **kwargs):
        """Signature figure: a (component × axis) grid of state estimates — position,
        speed and acceleration (rows) for each spatial axis X/Y/Z (columns), with the
        truth and noisy measurements overlaid from ``sample``. Extra kwargs (e.g.
        ``sigma=3`` for a ±3σ covariance band) are forwarded to ``record.plot``."""
        import matplotlib.pyplot as plt
        comps = ["position", "speed", "acceleration"][:self.order]
        axis_names = ["X", "Y", "Z"]
        fig, axes = plt.subplots(len(comps), 3, figsize=figsize, squeeze=False)
        fig.suptitle(f"{self.name()} — per-axis state estimate", fontweight="bold")
        for r, comp in enumerate(comps):
            for c in range(3):
                record.plot(component=comp, dim=c, sample=sample, canvas=axes[r, c], **kwargs)
                if r == 0:
                    axes[r, c].set_title(axis_names[c])
        fig.tight_layout()
        return fig