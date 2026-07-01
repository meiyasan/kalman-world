"""Manual timing benchmark: C acceleration vs pure Python for the Kalman engines.

Run with:
    python tests/bench_native.py

Prints wall-time and speedup for the key accelerated code paths.
"""
import sys
import time
import numpy as np

sys.path.insert(0, ".")

from modules.framework.native import HAS_C_ACCEL

if not HAS_C_ACCEL:
    print("C extension not built — build with  make accel  first.")
    sys.exit(1)


def _time(fn, n_reps=5):
    fn()  # warm-up
    t0 = time.perf_counter()
    for _ in range(n_reps):
        fn()
    return (time.perf_counter() - t0) / n_reps


def bench_linear_forward(n=4, m=2, N=10_000, n_reps=10):
    from modules.apps import BallisticLinearModel
    import modules.framework.native as nat_mod

    model = BallisticLinearModel("bench", fs=100)
    np.random.seed(0)
    sample = model.sample(N / 100)
    zk = np.asarray(sample.zk)

    def run_c():
        from modules.apps import BallisticLinearModel as M
        m2 = M("bench", fs=100)
        m2.filter(zk)

    orig = nat_mod.HAS_C_ACCEL
    def run_py():
        nat_mod.HAS_C_ACCEL = False
        try:
            from modules.apps import BallisticLinearModel as M
            m2 = M("bench", fs=100)
            m2.filter(zk)
        finally:
            nat_mod.HAS_C_ACCEL = orig

    t_c  = _time(run_c,  n_reps)
    t_py = _time(run_py, n_reps)
    speedup = t_py / t_c
    print(f"Linear forward (n={model.n}, m={model.m}, N={len(zk)}): "
          f"C={t_c*1e3:.2f} ms  Py={t_py*1e3:.2f} ms  "
          f"→ {speedup:.1f}×")
    return speedup


def bench_linear_smooth(N=10_000, n_reps=10):
    from modules.apps import BallisticLinearModel
    import modules.framework.native as nat_mod

    np.random.seed(1)
    sample = BallisticLinearModel("bench", fs=100).sample(N / 100)
    zk = np.asarray(sample.zk)

    def run_c():
        from modules.apps import BallisticLinearModel as M
        M("bench", fs=100).filter(zk, smooth=True)

    orig = nat_mod.HAS_C_ACCEL
    def run_py():
        nat_mod.HAS_C_ACCEL = False
        try:
            from modules.apps import BallisticLinearModel as M
            M("bench", fs=100).filter(zk, smooth=True)
        finally:
            nat_mod.HAS_C_ACCEL = orig

    t_c  = _time(run_c,  n_reps)
    t_py = _time(run_py, n_reps)
    speedup = t_py / t_c
    print(f"Linear forward+smooth (N={len(zk)}):           "
          f"C={t_c*1e3:.2f} ms  Py={t_py*1e3:.2f} ms  "
          f"→ {speedup:.1f}×")
    return speedup


def bench_particle_update(n_particles=500, mv=2, n_reps=20):
    from modules.framework.native import lib, as_contiguous, dptr

    np.random.seed(2)
    z    = np.random.randn(mv)
    zp   = np.random.randn(n_particles, mv)
    Rinv = np.eye(mv) * 2.0

    def run_c():
        logw = np.zeros(n_particles)
        lib.pf_weight_update(n_particles, mv,
                             dptr(as_contiguous(z)), dptr(as_contiguous(zp)),
                             dptr(as_contiguous(Rinv)), dptr(logw))

    def run_py():
        logw = np.zeros(n_particles)
        for i in range(n_particles):
            dz = z - zp[i]
            logw[i] += -0.5 * dz @ Rinv @ dz

    t_c  = _time(run_c,  n_reps)
    t_py = _time(run_py, n_reps)
    speedup = t_py / t_c
    print(f"PF weight update ({n_particles} particles, mv={mv}):    "
          f"C={t_c*1e6:.1f} µs  Py={t_py*1e6:.1f} µs  "
          f"→ {speedup:.1f}×")
    return speedup


def bench_ukf_outer_sum(n_sigma=9, n=4, n_reps=50):
    from modules.framework.native import lib, as_contiguous, dptr

    np.random.seed(3)
    Wc   = np.random.dirichlet(np.ones(n_sigma))
    Y    = np.random.randn(n_sigma, n)
    mean = np.einsum("i,ij->j", Wc, Y)

    def run_c():
        P_out = np.empty((n, n))
        lib.ukf_weighted_outer_sum(n_sigma, n,
                                   dptr(as_contiguous(Wc)),
                                   dptr(as_contiguous(Y)),
                                   dptr(as_contiguous(mean)),
                                   dptr(P_out))

    def run_py():
        sum(Wc[i] * np.outer(Y[i] - mean, Y[i] - mean) for i in range(n_sigma))

    t_c  = _time(run_c,  n_reps)
    t_py = _time(run_py, n_reps)
    speedup = t_py / t_c
    print(f"UKF weighted outer sum (2n+1={n_sigma}, n={n}):       "
          f"C={t_c*1e6:.1f} µs  Py={t_py*1e6:.1f} µs  "
          f"→ {speedup:.1f}×")
    return speedup


def bench_diag_nis(N=1000, m=2, n_reps=20):
    from modules.framework.native import lib, as_contiguous, dptr

    np.random.seed(4)
    nu = np.random.randn(N, m)
    S  = np.array([np.eye(m) + 0.1 * k for k in range(N)])

    def run_c():
        nis = np.empty(N)
        lib.diag_nis_batch(N, m, dptr(as_contiguous(nu)), dptr(as_contiguous(S)), dptr(nis))

    def run_py():
        np.array([float(nu[k] @ np.linalg.solve(S[k], nu[k])) for k in range(N)])

    t_c  = _time(run_c,  n_reps)
    t_py = _time(run_py, n_reps)
    speedup = t_py / t_c
    print(f"NIS batch (N={N}, m={m}):                     "
          f"C={t_c*1e3:.2f} ms  Py={t_py*1e3:.2f} ms  "
          f"→ {speedup:.1f}×")
    return speedup


if __name__ == "__main__":
    print(f"{'='*65}")
    print(f"  kalman-framework  C acceleration benchmark")
    print(f"  HAS_C_ACCEL = {HAS_C_ACCEL}")
    print(f"{'='*65}")

    speedups = []
    speedups.append(bench_linear_forward())
    speedups.append(bench_linear_smooth())
    speedups.append(bench_particle_update())
    speedups.append(bench_ukf_outer_sum())
    speedups.append(bench_diag_nis())

    print(f"{'='*65}")
    print(f"  Geometric mean speedup: {np.exp(np.mean(np.log(speedups))):.1f}×")
    print(f"{'='*65}")
