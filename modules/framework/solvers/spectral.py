"""Pseudo-spectral 2D field solver (FFT discretization + composed time integrator).

The `SPECTRAL` :class:`~kalman.framework.solver.KalmanSolver`: it owns the spatial
part (FFT, 2/3-rule dealiasing, spectral ``ik`` operators) and delegates the time
march to a composed :class:`~kalman.framework.integrator.KalmanIntegrator` (RK4)
applied to the **spectral** field.  Subclasses supply the physics via
``rhs_hat(wh) → dω̂/dt``; this class is domain-agnostic (the Kolmogorov forcing
lives in ``apps.weather.flow.NS2D``, which subclasses this).

**Backend.** ``backend="numpy"`` (default) or ``"torch"`` with ``device`` in
``cpu/cuda/mps``.  The torch path accepts a batched ``(N, g, g)`` / ``(N, g²)``
ensemble as one advance — the EnKF forecast's dominant cost — and returns numpy,
so callers are unchanged.  It runs in C++ on libTorch
(``Kalman::Learn::Solvers::Vorticity`` in ``libKalmanLearn``) for a solver that
states its physics in ``_vorticity_terms``; Python's torch is the fallback.
"""
from __future__ import annotations
import os

import numpy as np

from ..solver import KalmanSolver
from ..constants import KalmanSolverKind


class SpectralSolver(KalmanSolver):

    NAME = "Spectral Solver"
    DESCRIPTION = "Pseudo-spectral (FFT) 2D field solver; RK4 time march in spectral space."
    VERSION = "1.0"
    KIND = KalmanSolverKind.SPECTRAL

    def __init__(self, n=64, dt=0.01, *, length=2*np.pi, integrator=None,
                 backend=None, device=None, seed=0):
        """``n`` is the grid: an ``int`` for a square ``n×n`` mesh, or
        ``(ny, nx)`` for a **rectangular** one.  ``length`` is the physical
        domain size — a scalar or ``(Ly, Lx)`` — so wavenumbers are scaled
        ``2π/L`` (default ``2π`` reproduces integer wavenumbers exactly).

        Spectral discretization requires a **regular periodic** mesh; bounded /
        irregular grids are the job of the finite-difference/volume/element
        solver kinds."""
        super().__init__(integrator=integrator, backend=backend, device=device)
        ny, nx = (int(n), int(n)) if np.isscalar(n) else (int(n[0]), int(n[1]))
        Ly, Lx = (float(length), float(length)) if np.isscalar(length) else (float(length[0]), float(length[1]))
        self.ny, self.nx = ny, nx
        self.shape = (ny, nx)
        self.Ly, self.Lx = Ly, Lx
        self.n = nx if nx == ny else None      # square-grid back-compat (None if rectangular)
        self.dt = float(dt)
        # Integer mode indices (used for the 2/3 dealias rule) and the physical
        # wavenumbers k = 2π/L · index (identical to the index for L = 2π).
        ix = np.fft.fftfreq(nx, 1.0 / nx)
        iy = np.fft.fftfreq(ny, 1.0 / ny)
        self._ix = ix[None, :] * np.ones((ny, 1))
        self._iy = iy[:, None] * np.ones((1, nx))
        self.kx = (2 * np.pi / Lx) * self._ix
        self.ky = (2 * np.pi / Ly) * self._iy
        self.k2 = self.kx**2 + self.ky**2
        self.k2inv = np.where(self.k2 > 0, 1.0 / np.maximum(self.k2, 1e-12), 0.0)
        self.dealias = (np.abs(self._ix) <= nx // 3) & (np.abs(self._iy) <= ny // 3)
        self.rng = np.random.default_rng(seed)
        self._tt = None
        self._native = None                    # the libTorch solver, made on first advance

    # ------------------------------------------------------------------
    def _as_grid(self, w):
        """Validate/reshape an input field to ``(ny, nx)``; raise a clear error
        when the given 2-D gridding does not match this solver's mesh."""
        arr = np.asarray(w, dtype=float)
        ny, nx = self.shape
        if arr.ndim == 1:
            if arr.size != ny * nx:
                raise ValueError(f"field of size {arr.size} does not fit this solver's "
                                 f"{ny}x{nx} grid ({ny*nx} cells).")
            return arr.reshape(ny, nx)
        if arr.shape[-2:] != (ny, nx):
            raise ValueError(f"field grid {arr.shape[-2:]} != solver grid {(ny, nx)}; "
                             f"construct the solver with n=({arr.shape[-2]}, {arr.shape[-1]}).")
        return arr

    # ------------------------------------------------------------------
    # Physics hook (subclass supplies) — spectral RHS dω̂/dt
    # ------------------------------------------------------------------
    def rhs_hat(self, wh, t=None):
        raise NotImplementedError("SpectralSolver subclasses must define rhs_hat(wh).")

    # backward-compat alias (NS2D's historical name)
    def rhs(self, wh):
        return self.rhs_hat(wh)

    # ------------------------------------------------------------------
    # Spectral differential operators (numpy; single source of truth)
    # ------------------------------------------------------------------
    def _grad_spectral(self, wh):
        wx = np.real(np.fft.ifft2(1j * self.kx * wh))
        wy = np.real(np.fft.ifft2(1j * self.ky * wh))
        return wx, wy

    def _vel_spectral(self, wh):
        psih = wh * self.k2inv
        u = np.real(np.fft.ifft2(1j * self.ky * psih))
        v = np.real(np.fft.ifft2(-1j * self.kx * psih))
        return u, v

    def gradient(self, w):
        """∇ω = (∂ω/∂x, ∂ω/∂y) of a real field ``(ny,nx)``/flat."""
        return self._grad_spectral(np.fft.fft2(self._as_grid(w)))

    def velocity(self, w):
        """Incompressible velocity (u, v) of a real field ``(ny,nx)``/flat."""
        return self._vel_spectral(np.fft.fft2(self._as_grid(w)))

    # ------------------------------------------------------------------
    # Time stepping — RK4 (the composed integrator) on the spectral field
    # ------------------------------------------------------------------
    @staticmethod
    def _rk_step(rhs, x, dt, tableau):
        """One explicit Runge–Kutta step from a Butcher ``tableau`` (c, A, b),
        backend-agnostic (works on numpy or torch arrays via +/*), so *any*
        explicit integrator drives the spectral field uniformly."""
        c, A, b = tableau
        ks = []
        for i in range(len(c)):
            if i == 0:
                xi = x
            else:
                acc = A[i][0] * ks[0]
                for j in range(1, i):
                    acc = acc + A[i][j] * ks[j]
                xi = x + dt * acc
            ks.append(rhs(xi))
        out = b[0] * ks[0]
        for i in range(1, len(b)):
            out = out + b[i] * ks[i]
        return x + dt * out

    def _tableau(self):
        tab = self.integrator.tableau
        if tab is None:
            raise TypeError(f"{self.integrator.name()} has no explicit tableau; the spectral "
                            f"solver needs an explicit RK integrator (Euler/RK4/Dormand-Prince).")
        return tab

    def step(self, wh):
        """One step of the spectral field with the composed integrator (numpy)."""
        return self._rk_step(self.rhs_hat, wh, self.dt, self._tableau())

    def advance(self, w, nsteps):
        """Advance a real field ``w`` by ``nsteps``.  Accepts ``(ny,nx)``, flat
        ``(ny·nx,)``, or a batched ensemble ``(N, ny, nx)`` / ``(N, ny·nx)``;
        returns the same layout.  Uses the torch backend when configured."""
        if self.backend == "torch":
            return self._advance_batched(w, int(nsteps))
        arr = np.asarray(w, dtype=float)
        flat = arr.ndim == 1
        wh = np.fft.fft2(self._as_grid(arr))
        for _ in range(int(nsteps)):
            wh = self.step(wh)
        out = np.real(np.fft.ifft2(wh))
        return out.ravel() if flat else out

    def solve(self, field, steps=1, u=None, t=None):
        return self.advance(field, int(steps))

    def _advance_torch(self, x, nsteps, tab):
        """``(N, ny, nx)`` advanced with Python's torch and ``_rhs_hat_torch``."""
        if self._tt is None:
            self._init_torch()
        t = self._torch
        wt = t.fft.fft2(t.as_tensor(x, dtype=self._rdtype, device=self._dev))
        for _ in range(int(nsteps)):
            wt = self._rk_step(self._rhs_hat_torch, wt, self.dt, tab)
        return t.fft.ifft2(wt).real.cpu().numpy()

    def random_ic(self, amp=1.0):
        """Random large-scale initial field, unit std (numpy RNG)."""
        wh = (self.rng.standard_normal(self.shape) +
              1j * self.rng.standard_normal(self.shape))
        kmag = np.sqrt(self._ix**2 + self._iy**2)          # in mode-index units
        wh *= np.exp(-0.5 * (kmag / 6.0) ** 2) * self.dealias
        w = np.real(np.fft.ifft2(wh))
        return amp * w / (w.std() + 1e-12)

    # ------------------------------------------------------------------
    # torch backend (CPU/CUDA/MPS) — batched spectral steps
    #
    # The steps run in C++ on libTorch (Kalman::Learn::Solvers::Vorticity in
    # libKalmanLearn) for a solver that states its physics in
    # ``_vorticity_terms``. Python's own torch is used only for a subclass with
    # other physics (its ``_rhs_hat_torch``), when the library is missing, or
    # with KALMAN_NATIVE_TORCH=0 in the environment (to compare the two).
    # ------------------------------------------------------------------
    def _vorticity_terms(self):
        """The physics as the terms of ``dw/dt = -u.grad(w) - beta v + nu lap(w) + F``:
        a dict with ``nu`` and optionally ``beta`` and ``forcing_hat`` (the transform
        of ``F``). ``None`` (the default) when the equation is not of that form."""
        return None

    def _torch_native(self):
        """The libTorch solver for this grid and physics, or ``None``."""
        if self._native is None:
            terms = self._vorticity_terms()
            self._native = False
            if terms is not None and os.environ.get("KALMAN_NATIVE_TORCH", "1") != "0":
                from ..native import learn
                if learn.has_torch():
                    self._native = learn.Vorticity(self.kx, self.ky, self.k2, self.k2inv, self.dealias,
                                                   device=self.device or "cpu", **terms)
                    self.device = self._native.device
        return self._native or None

    def _init_torch(self):
        import torch
        dev = self.device
        if dev == "cuda" and not torch.cuda.is_available():
            dev = "cpu"
        if dev == "mps" and not getattr(torch.backends, "mps", None):
            dev = "cpu"
        self._torch = torch
        self._dev = torch.device(dev)
        # MPS (Apple Metal) has no float64 — use single precision there.
        self._rdtype = torch.float32 if dev == "mps" else torch.float64
        self._cdtype = torch.complex64 if dev == "mps" else torch.complex128
        z = lambda a: torch.as_tensor(np.asarray(a), dtype=self._rdtype, device=self._dev)
        self._tt = dict(kx=z(self.kx), ky=z(self.ky), k2=z(self.k2),
                        k2inv=z(self.k2inv), dealias=z(self.dealias.astype(float)))

    def _rhs_hat_torch(self, wh):
        """Torch spectral RHS — subclasses override if physics differs; default
        raises so a torch-backed solver must provide it."""
        raise NotImplementedError("torch backend needs _rhs_hat_torch on the subclass.")

    def _advance_batched(self, w, nsteps):
        arr = np.asarray(w, dtype=float)
        ny, nx = self.shape
        squeeze = False
        if arr.ndim == 1:                                   # (ny·nx,) single
            x = self._as_grid(arr)[None]; flatten = "flat"
        elif arr.ndim == 2 and arr.shape == (ny, nx):       # (ny,nx) single
            x = arr.reshape(1, ny, nx); flatten = "grid"; squeeze = True
        elif arr.ndim == 2:                                 # (N, ny·nx) batch
            if arr.shape[1] != ny * nx:
                raise ValueError(f"batched rows of size {arr.shape[1]} do not fit the "
                                 f"{ny}x{nx} grid.")
            x = arr.reshape(-1, ny, nx); flatten = "flat"
        else:                                               # (N, ny, nx) batch
            x = self._as_grid(arr); flatten = "grid"
        tab = self._tableau()                       # same integrator as numpy path
        native = self._torch_native()
        if native is not None:
            out = native.advance(x, nsteps, self.dt, tab)
        else:
            out = self._advance_torch(x, nsteps, tab)
        if squeeze:
            return out[0]
        if arr.ndim == 1:
            return out[0].ravel()
        if flatten == "flat":
            return out.reshape(out.shape[0], ny * nx)
        return out
