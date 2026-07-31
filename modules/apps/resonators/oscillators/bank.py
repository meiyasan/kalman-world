"""Multi-oscillator bank — the BANK variant (independent partials)."""
import numpy as np
from scipy.linalg import expm
from scipy.optimize import curve_fit

from ....framework.models.linear import LinearKalmanModel
from ....framework.models.composite import CompositeKalmanModel
from ....framework.constants import KalmanState, KalmanObservable, KalmanObservableReadout
from .base import (OscillatorState, OscillatorObservable, wire_inharmonicity,
                   wire_fundamental, partial_frequencies)

# ======================================================================
# A bank of windows over one real signal — INDEPENDENT across distinct fc
# ======================================================================
class OscillatorBankModel(CompositeKalmanModel):
    """
    A bank of :class:`OscillatorModel` windows analysing **one** real signal across
    several distinct centers ``fc_i``.

    Internally this is a :class:`CompositeKalmanModel` with ``obs_mode=INDEPENDENT``:
    each window tracks its own demodulated band and the observations are stacked.
    The standard :class:`CompositeKalmanModel` preprocess would try to split the input
    by channel count, which cannot broadcast one real signal to N demodulators.  This
    class overrides preprocess with :meth:`_broadcast_preprocess` to send the single
    real strain to every window's own Hilbert demodulator.

    Composition pattern::

        OscillatorBankModel(w1, w2, w3)
          = CompositeKalmanModel(w1, w2, w3, obs_mode=INDEPENDENT)
          + preprocess: broadcast real strain → each window's own _demod_preprocess

    Each window's postprocess remodulates to a real band, so ``record.zk`` has one
    real column per window.  ``reconstruct(record)`` sums them to give the total
    extracted signal; ``strain − reconstruct(record)`` is the cleaned residual.
    """

    NAME = "Oscillator Bank (window bank)"
    DESCRIPTION = "INDEPENDENT bank of windowed OscillatorModels over one real signal."
    VERSION = "1.1"

    def __init__(self, *windows, **kwargs):
        if not windows:
            raise ValueError("OscillatorBankModel needs at least one window.")
        self.windows = list(windows)
        super().__init__(*windows, obs_mode=KalmanObservableReadout.INDEPENDENT, **kwargs)
        self.fs = windows[0].fs
        self._fs_original = windows[0]._fs_original
        self.preprocess = self._broadcast_preprocess   # expand: 1 real → N basebands
        self.postprocess = self._identity_postprocess  # keep decimated phasors; reconstruct() upsamples

    @staticmethod
    def _identity_postprocess(zk_hat, **kwargs):
        return np.asarray(zk_hat, dtype=float).reshape(len(zk_hat), -1)

    def _broadcast_preprocess(self, zk, uk=None, **kwargs):
        """Preprocess: broadcast one real strain to every window, each demodulates + downsamples.

        Returns ``(N_dec, 2·N_win)`` — columns ``[Re_ψ1, Im_ψ1, Re_ψ2, Im_ψ2, …]``.
        Symmetric counterpart: ``_reconstruct_postprocess``.
        """
        x = np.asarray(zk, dtype=float).reshape(np.shape(zk)[0], -1)[:, :1]   # (T, 1) real strain
        outs = [w._demod_preprocess(x)[0] for w in self.windows]               # each (N_dec, 2)
        return np.concatenate(outs, axis=1), uk                                 # (N_dec, 2·N_win)

    def _reconstruct_postprocess(self, zk_hat, **kwargs):
        """Postprocess: N window phasors (decimated) → one real signal at original rate.

        ``zk_hat`` = ``h(xk)`` from the INDEPENDENT composite, shape ``(N_dec, 2·N_win)``.
        Columns ``2i, 2i+1`` are ``[Re(Σψ̂_win_i), Im(Σψ̂_win_i)]`` — already the FUSED
        phasor sum within window ``i`` (computed by each window's FUSED ``h``).
        Each window's phasor is upsampled to ``fs_original``, remodulated at ``fc_i``,
        and summed across windows.  Returns ``(N, 1)`` real.
        Symmetric counterpart: ``_broadcast_preprocess``.
        """
        z = np.asarray(zk_hat, dtype=float).reshape(len(zk_hat), -1)
        w0 = self.windows[0]
        N = w0._preprocess_N
        t = w0.t0 + np.arange(N) / self._fs_original
        total = np.zeros(N)
        for i, w in enumerate(self.windows):
            psi = z[:, 2 * i] + 1j * z[:, 2 * i + 1]
            psi_up = w._resample(psi, w._fs_dec, self._fs_original)
            if len(psi_up) > N:
                psi_up = psi_up[:N]
            elif len(psi_up) < N:
                psi_up = np.concatenate([psi_up, np.zeros(N - len(psi_up))])
            total += psi_up.real * np.cos(w._wc * t) - psi_up.imag * np.sin(w._wc * t)
        return total.reshape(-1, 1)

    def reconstruct(self, record, *, per_window=False, band_limit=False):
        """Upsample per-window decimated phasors from ``record.zk`` and sum to a
        single real signal at the original sample rate — shape ``(N_original,)``.

        Parameters
        ----------
        per_window : bool, default False
            If ``True``: return a list of per-window ``(N_original,)`` arrays without
            re-filtering (delegates to :meth:`reconstruct_window`).
        band_limit : bool, default False
            If ``True``: clip each window's output to its ``[fc ± 2·dfc]`` band,
            removing the residual ``fc ± fs_dec`` upsampling images.
        """
        if per_window:
            return [self.reconstruct_window(record, i, band_limit=band_limit)
                    for i in range(len(self.windows))]
        z = np.asarray(record.zk, dtype=float).reshape(len(record.zk), -1)
        w0 = self.windows[0]
        N = w0._preprocess_N
        t = w0.t0 + np.arange(N) / self._fs_original
        total = np.zeros(N)
        for i, w in enumerate(self.windows):
            psi = z[:, 2 * i] + 1j * z[:, 2 * i + 1]
            psi_up = w._resample(psi, w._fs_dec, self._fs_original)
            if len(psi_up) > N:
                psi_up = psi_up[:N]
            elif len(psi_up) < N:
                psi_up = np.concatenate([psi_up, np.zeros(N - len(psi_up))])
            if band_limit:
                psi_up = w._band_limit_phasor(psi_up, 2.0 * w.dfc)
            total += psi_up.real * np.cos(w._wc * t) - psi_up.imag * np.sin(w._wc * t)
        return total

    def reconstruct_window(self, record, i, *, band_limit=False):
        """Return the real-band reconstruction for window ``i`` without re-filtering.

        Slices window ``i``'s state from ``record.xk``, computes the FUSED phasor sum,
        upsamples, and remodulates.  Does not touch the other windows.
        """
        offset = sum(w.n for w in self.windows[:i])
        n = self.windows[i].n
        xk_w = np.asarray(record.xk, dtype=float)[:, offset:offset + n]
        zk_hat_w = np.column_stack([xk_w[:, 0::2].sum(axis=1),
                                    xk_w[:, 1::2].sum(axis=1)])
        return self.windows[i]._reconstruct_postprocess(zk_hat_w, band_limit=band_limit).ravel()


