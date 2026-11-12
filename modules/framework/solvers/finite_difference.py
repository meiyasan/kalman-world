"""Finite-difference 2D field solver — the first non-spectral discretization.

Where :class:`~kalman.framework.solvers.SpectralSolver` needs a *regular
periodic* mesh, this solver works on any regular rectangular grid with a choice
of **boundary condition** — so bounded domains (walls, fixed values) are
representable.  Second-order central stencils for ∂/∂x, ∂/∂y and ∇²; the time
march is delegated to the composed
:class:`~kalman.framework.integrator.KalmanIntegrator` exactly as the spectral
solver does (same Butcher-tableau path), so integrators remain swappable.

Subclasses supply the physics via ``rhs(field, t) → ∂field/∂t`` in *real* space
(contrast the spectral solver's spectral-space ``rhs_hat``).  The built-in
default is the **diffusion** equation ``∂f/∂t = ν∇²f``, which makes the class
usable (and testable) on its own.

Boundaries (``bc``): ``"periodic"`` (matches spectral), ``"dirichlet"`` (field
pinned to ``bc_value`` outside), or ``"neumann"`` (zero-gradient / reflective).
"""
from __future__ import annotations
import numpy as np

from ..solver import KalmanSolver
from ..constants import KalmanSolverKind


class FiniteDifferenceSolver(KalmanSolver):

    NAME = "Finite-Difference Solver"
    DESCRIPTION = ("Second-order finite-difference 2D field solver (periodic / dirichlet / "
                   "neumann boundaries); time march via the composed integrator.")
    VERSION = "1.0"
    KIND = KalmanSolverKind.FINITE_DIFFERENCE

    def __init__(self, n=64, dt=0.01, *, length=2*np.pi, bc="periodic", bc_value=0.0,
                 nu=0.01, integrator=None, backend=None, device=None, seed=0):
        super().__init__(integrator=integrator, backend=backend, device=device)
        ny, nx = (int(n), int(n)) if np.isscalar(n) else (int(n[0]), int(n[1]))
        Ly, Lx = ((float(length), float(length)) if np.isscalar(length)
                  else (float(length[0]), float(length[1])))
        self.ny, self.nx = ny, nx
        self.shape = (ny, nx)
        self.n = nx if nx == ny else None
        self.Ly, self.Lx = Ly, Lx
        self.dy, self.dx = Ly / ny, Lx / nx
        self.dt = float(dt)
        self.bc = str(bc).lower()
        self.bc_value = float(bc_value)
        self.nu = float(nu)
        self.rng = np.random.default_rng(seed)
        if self.bc not in ("periodic", "dirichlet", "neumann"):
            raise ValueError("bc must be 'periodic', 'dirichlet' or 'neumann'.")

    # ------------------------------------------------------------------
    def _as_grid(self, w):
        arr = np.asarray(w, dtype=float)
        ny, nx = self.shape
        if arr.ndim == 1:
            if arr.size != ny * nx:
                raise ValueError(f"field of size {arr.size} does not fit this solver's "
                                 f"{ny}x{nx} grid ({ny*nx} cells).")
            return arr.reshape(ny, nx)
        if arr.shape[-2:] != (ny, nx):
            raise ValueError(f"field grid {arr.shape[-2:]} != solver grid {(ny, nx)}.")
        return arr

    def _pad(self, f):
        """One ghost cell on each side, per the boundary condition."""
        if self.bc == "periodic":
            return np.pad(f, 1, mode="wrap")
        if self.bc == "neumann":
            return np.pad(f, 1, mode="edge")               # zero-gradient
        return np.pad(f, 1, mode="constant", constant_values=self.bc_value)

    # ------------------------------------------------------------------
    # Differential operators (second-order central differences)
    # ------------------------------------------------------------------
    def discretize(self, field):
        """Spatial discretization: pad with ghost cells per the BC (this is the
        solver's own job; the integrator then marches it in time)."""
        return self._pad(self._as_grid(field))

    def gradient(self, w):
        """∇f = (∂f/∂x, ∂f/∂y) by central differences (BC-aware)."""
        p = self.discretize(w)
        fx = (p[1:-1, 2:] - p[1:-1, :-2]) / (2 * self.dx)
        fy = (p[2:, 1:-1] - p[:-2, 1:-1]) / (2 * self.dy)
        return fx, fy

    def laplacian(self, w):
        """∇²f by the 5-point stencil (BC-aware)."""
        p = self.discretize(w)
        c = p[1:-1, 1:-1]
        return ((p[1:-1, 2:] - 2 * c + p[1:-1, :-2]) / self.dx ** 2 +
                (p[2:, 1:-1] - 2 * c + p[:-2, 1:-1]) / self.dy ** 2)

    # ------------------------------------------------------------------
    # Physics hook (real space) — default: diffusion
    # ------------------------------------------------------------------
    def rhs(self, field, t=None):
        """``∂f/∂t``.  Default is diffusion ``ν∇²f``; subclasses override."""
        return self.nu * self.laplacian(field)

    # ------------------------------------------------------------------
    # Time march — the composed integrator (same tableau path as spectral)
    # ------------------------------------------------------------------
    def _tableau(self):
        tab = self.integrator.tableau
        if tab is None:
            raise TypeError(f"{self.integrator.name()} has no explicit tableau; this solver "
                            f"needs an explicit RK integrator (Euler/RK4/Dormand-Prince).")
        return tab

    def _native_scheme(self):
        """The C++ integrator's name when this solver can run in C++: the default
        physics (diffusion), a built-in integrator, ``step`` not overridden."""
        from ..integrator import _BUILTIN_STEPS
        from ..models import native
        cls = type(self)
        if (not native.native_loop_enabled() or cls.rhs is not FiniteDifferenceSolver.rhs
                or cls.step is not FiniteDifferenceSolver.step or "rhs" in self.__dict__):
            return None
        integrator = self.integrator
        name = getattr(integrator, "NATIVE", None)
        if name is None or type(integrator)._step is not _BUILTIN_STEPS.get(name):
            return None
        return name

    def step(self, field):
        from .spectral import SpectralSolver
        return SpectralSolver._rk_step(lambda f: self.rhs(f), field, self.dt, self._tableau())

    def advance(self, w, nsteps):
        """Advance a real field by ``nsteps``; accepts ``(ny,nx)`` or flat and
        returns the same layout."""
        arr = np.asarray(w, dtype=float)
        flat = arr.ndim == 1
        f = self._as_grid(arr)
        scheme = self._native_scheme()
        if scheme is not None and f.ndim == 2:  # C++: Kalman::Solvers::FiniteDifference
            from ..native import core
            f = core.solve_diffusion(self.shape, self.dt, (self.Ly, self.Lx), self.bc, self.bc_value, self.nu,
                                     scheme, f, int(nsteps))
            return f.ravel() if flat else f
        for _ in range(int(nsteps)):
            f = self.step(f)
        return f.ravel() if flat else f

    def solve(self, field, steps=1, u=None, t=None):
        return self.advance(field, int(steps))

    def random_ic(self, amp=1.0):
        """Smooth random initial field, unit std."""
        f = self.rng.standard_normal(self.shape)
        for _ in range(8):                                  # smooth by diffusion sweeps
            f = f + 0.2 * self.laplacian(f) * min(self.dx, self.dy) ** 2
        return amp * f / (f.std() + 1e-12)
