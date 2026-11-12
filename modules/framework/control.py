"""KalmanController — base contract for closed-loop control over a Kalman filter."""
from __future__ import annotations
import numpy as np


class KalmanController:
    """Callable controller: ``u = ctrl(x̂, P, t, nu=ν)``.

    The KalmanFilter receives it via ``uk=`` and treats it like any other
    control sequence.  The controller may update ``model._B`` as a side
    effect before returning ``u``.

    Subclasses override ``_policy`` (required) and ``_update`` (optional).

    The shipped controllers compute their policy in C++ (``Kalman::Controllers``,
    the kind named by ``NATIVE``) when libKalman is installed: ``_policy`` and
    ``_update`` forward to a C++ controller built on the model
    (:meth:`_native_handle`). The Python arithmetic that remains
    (``_reference_policy``, ``_reference_update``) is the reference the tests
    compare against, and what runs inside ``native.python_reference()``, for a
    model that is not real double precision, and without the library. The logs
    (``cost_log``, ``u_log``, ``scale_log``) are kept here either way.
    """

    #: The kind of C++ controller this class forwards to (core.create_controller),
    #: or ``None`` for a policy written in Python.
    NATIVE: str | None = None

    def __init__(self, model, x_ref=None, Qcost=None, Rcost=None):
        self.model = model
        self.x_ref = (np.zeros((model.n, 1)) if x_ref is None
                      else np.asarray(x_ref, dtype=float).reshape(-1, 1))
        n, l = model.n, max(model.l, 1)
        self.Qcost = np.eye(n) if Qcost is None else np.asarray(Qcost, dtype=float)
        self.Rcost = np.eye(l) if Rcost is None else np.asarray(Rcost, dtype=float)
        self.reset()

    def __call__(self, x_hat, P, t, *, nu=None):
        x_hat = np.asarray(x_hat, dtype=float).reshape(-1, 1)
        u = np.asarray(self._policy(x_hat, P, t, nu=nu), dtype=float).reshape(-1, 1)

        B_new = self._update(x_hat, P, t, nu=nu)
        if B_new is not None:
            self.model._B = np.asarray(B_new, dtype=float)
            self.B_log.append((t, B_new.copy()))

        # Cost against the command the actuator can actually deliver (model.clip_u), not
        # the controller's raw ask -- a fixed-gain controller with no notion of an actuator
        # limit (e.g. LQR) can ask for far more than the plant ever receives, and charging
        # Rcost against that fictitious, never-applied effort would inflate its cost against
        # a controller (e.g. MPPI) whose raw output already respects the limit. u_log still
        # keeps the raw command for diagnostics -- only the logged cost uses the clipped one.
        u_applied = np.asarray(self.model.clip_u(u), dtype=float).reshape(-1, 1)
        e = x_hat - self.x_ref
        cost = float((e.T @ self.Qcost @ e + u_applied.T @ self.Rcost @ u_applied).item())
        self.cost_log.append((t, cost))
        self.u_log.append((t, u.copy()))
        return u

    def _policy(self, x_hat, P, t, *, nu=None):
        raise NotImplementedError

    def _update(self, x_hat, P, t, *, nu=None):
        return None

    def reset(self):
        self.cost_log = []
        self.u_log = []
        self.B_log = []
        state = self.__dict__.get("_native_state")
        if state is not None:
            from .native import core
            core.controller_reset(state[1])

    # ------------------------------------------------------------------
    # The C++ controller behind the class
    # ------------------------------------------------------------------
    def __getstate__(self):
        # The C++ handle does not pickle or copy; a copy builds its own on first use.
        state = self.__dict__.copy()
        state.pop("_native_state", None)
        return state

    def _native_key(self):
        """What the C++ controller was built from: a change rebuilds it (and starts
        its internal state afresh). ``None`` when this configuration has no C++
        counterpart (shapes the C++ class does not take)."""
        return None

    def _native_create(self, model_handle):
        """Build the C++ controller on ``model_handle`` (core.create_controller)."""
        raise NotImplementedError

    def _native_reads_noise(self) -> bool:
        """Whether the C++ controller reads the model's Q or R (the adapters)."""
        return False

    def _native_handle(self):
        """The C++ controller for this call, or ``None`` when the Python reference
        runs: no ``NATIVE`` kind, the reference path, no library, a model that is
        not real double precision, or a configuration the C++ class does not take.

        The model reaches C++ as itself when it is a :class:`NativeModel` (a model
        written in C++), and otherwise as a callback model whose ``f``, ``h``,
        Jacobians, noise and actuator limit (``clip_u``) call the Python model.
        A controller that reads the noise (Boost, AdaptiveQR) always gets the
        callback model, which asks for ``Q`` and ``R`` at every call: the retuning
        is made on the Python model, by this class, and the C++ side must see it.
        """
        if self.NATIVE is None:
            return None
        from .models import native
        if not native.native_loop_enabled() or not native.callback_eligible(self.model):
            return None
        try:
            key = self._native_key()
        except (TypeError, ValueError, AttributeError):
            return None
        if key is None:
            return None
        key = (id(self.model),) + tuple(key)
        state = self.__dict__.get("_native_state")
        if state is not None and state[0] == key:
            return state[1]
        model_handle, keep = self._native_model()
        handle = self._native_create(model_handle)
        self._native_state = (key, handle, model_handle, keep)
        return handle

    def _native_model(self):
        """``(model handle, keepalive)``; the keepalive's last item is the list in
        which the model's Python callbacks leave their exceptions."""
        from .models import native
        from .models.native import NativeModel
        from .model import KalmanModel
        if isinstance(self.model, NativeModel) and not self._native_reads_noise():
            return self.model._handle, (self.model, [])
        handle, keep = native.callback_handle(self.model, "extended", live_noise=True)
        errors = keep[-1]
        if type(self.model).clip_u is not KalmanModel.clip_u:
            keep = (*keep[:-1], self._clip_callback(handle, errors), errors)
        return handle, keep

    def _clip_callback(self, handle, errors):
        from .native import core
        ffi, model, l = core.ffi, self.model, int(self.model.l)

        @ffi.callback("int(void *, const double *, double *)")
        def clip(_user, u, out):
            try:
                command = np.frombuffer(ffi.buffer(u, l * 8), dtype=np.float64).reshape(l, 1).copy()
                applied = np.ascontiguousarray(np.asarray(model.clip_u(command), dtype=np.float64).reshape(-1))
                if applied.size != l:
                    raise ValueError(f"clip_u returned {applied.size} values for a command of {l}.")
                ffi.memmove(out, applied, l * 8)
                return 0
            except BaseException as exc:            # must not unwind through C++
                errors.append(exc)
                return 1

        core.check(core.lib.kalman_model_callback_set_clip(handle, clip), "set_clip")
        return clip

    def _native_call(self, fn, *args):
        """``fn(*args)`` on the C++ side; an exception raised in one of the
        model's Python callbacks comes back as itself."""
        errors = self._native_state[3][-1]
        errors.clear()
        try:
            return fn(*args)
        except RuntimeError:
            if errors:
                raise errors[0]
            raise

    def _native_belief(self, x_hat, P, nu):
        """``(x, P, nu)`` as the C API takes them, or ``None`` when they are not
        (a complex state, a shape that does not match the model)."""
        n, m = int(self.model.n), int(self.model.m)
        x = np.asarray(x_hat)
        if x.dtype.kind not in "fiu" or x.size != n:
            return None
        if P is not None:
            P = np.asarray(P)
            if P.dtype.kind not in "fiu" or P.size != n * n:
                return None
        if nu is not None:
            nu = np.asarray(nu)
            if nu.dtype.kind not in "fiu" or nu.size != m:
                return None
        return x, P, nu

    def _native_policy(self, x_hat, P, t, nu):
        """The C++ policy's command, or ``None`` when the reference must run."""
        handle = self._native_handle()
        if handle is None:
            return None
        belief = self._native_belief(x_hat, P, nu)
        if belief is None:
            return None
        from .native import core
        n, m, L = int(self.model.n), int(self.model.m), max(int(self.model.l), 1)
        return self._native_call(core.controller_policy, handle, belief[0], belief[1],
                                 0.0 if t is None else float(t), belief[2], n, m, L)

    def _native_update(self, x_hat, P, t, nu):
        """``(True, factor or None)`` from the C++ update, ``(False, None)`` when the
        reference must run."""
        handle = self._native_handle()
        if handle is None:
            return False, None
        belief = self._native_belief(x_hat, P, nu)
        if belief is None:
            return False, None
        from .native import core
        n, m = int(self.model.n), int(self.model.m)
        return True, self._native_call(core.controller_update, handle, belief[0], belief[1],
                                       0.0 if t is None else float(t), belief[2], n, m)

    def _gains_key(self, *gains):
        """The gains' bytes, or ``None`` when one is not ``(max(l, 1), n)``."""
        shape = (max(int(self.model.l), 1), int(self.model.n))
        if any(g is not None and np.shape(g) != shape for g in gains):
            return None
        return tuple(None if g is None else np.asarray(g, dtype=np.float64).tobytes() for g in gains)

    def _cost_key(self):
        """The set point and cost weights, or ``None`` when their shapes are not
        the ones the C++ class takes."""
        n, L = int(self.model.n), max(int(self.model.l), 1)
        if self.x_ref.size != n or np.shape(self.Qcost) != (n, n) or np.shape(self.Rcost) != (L, L):
            return None
        return (self.x_ref.tobytes(), np.asarray(self.Qcost, dtype=np.float64).tobytes(),
                np.asarray(self.Rcost, dtype=np.float64).tobytes())

    def _cost_args(self):
        return dict(x_ref=self.x_ref, Qcost=self.Qcost, Rcost=self.Rcost)
