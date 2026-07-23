"""NeuralAdaptiveQRKalmanController — learned, per-state Q adaptation."""
from __future__ import annotations
import numpy as np

from ..control import KalmanController


def _window_feats(nu_window: np.ndarray) -> np.ndarray:
    """[std, lag-1 autocorr, lag-3 autocorr] of a window of scalar innovations.

    Std alone is what ``AdaptiveQRKalmanController``'s trace-ratio test uses -- it
    reacts strongly to noise entering *directly* at the measured state, but is nearly
    blind to noise on an *unmeasured* state that only reaches the measurement
    indirectly through the dynamics coupling (e.g. heading noise reaching a
    lateral-position measurement), because that shows up as added smoothness /
    autocorrelation in the innovation, not added instantaneous variance.
    """
    nu = np.asarray(nu_window, dtype=float).ravel()
    a = nu - nu.mean()
    std = float(np.std(nu))

    def _autocorr(lag):
        if len(a) <= lag:
            return 0.0
        x, y = a[:-lag], a[lag:]
        denom = np.std(x) * np.std(y)
        return float(np.mean(x * y) / denom) if denom > 1e-12 else 0.0

    return np.array([std, _autocorr(1), _autocorr(3)])


class NeuralAdaptiveQRKalmanController(KalmanController):
    """Classify *which* state's process noise is structurally elevated, from a window
    of innovations, and apply a **per-state** Q correction — something the single
    scalar in :class:`AdaptiveQRKalmanController` cannot do (it must scale every state's
    Q by the same factor, whether that state's noise actually changed or not).

    Usage::

        nqr = NeuralAdaptiveQRKalmanController(model, window=20)
        nqr.fit(regime_scales={
            "nominal": np.array([1.0, 1.0, 1.0]),
            "lat":     np.array([25.0, 1.0, 1.0]),
            "head":    np.array([1.0, 25.0, 1.0]),
        })
        record = filter.forward(zk, uk=nqr)

    ``fit()`` simulates labelled synthetic episodes for each named regime (using the
    model's own ``f()``/``h()``, exactly like ``NeuralKalmanController.fit_mppi()``
    simulates its own training data), trains a small classifier on windowed innovation
    features, and stores ``regime_scales`` so a predicted label can be looked up as a
    concrete per-state multiplier at inference time.

    This is a *classifier*, not a continuous regressor: it identifies which of a small
    number of known noise "shapes" best matches the recent innovation window and
    applies that shape's multiplier vector. It does not invent a shape it wasn't
    trained on.
    """

    def __init__(self, model, window=20, hidden=16, q_min=0.2, q_max=30.0, **kw):
        self.window = int(window)
        self.hidden = int(hidden)
        self.q_min = float(q_min)
        self.q_max = float(q_max)
        self._Q_nominal = None
        self._nu_history: list = []
        self.scale_log: list = []
        self._model_nn = None
        self._x_mean = None
        self._x_std = None
        self.regime_scales: dict | None = None
        self._labels: list | None = None
        super().__init__(model, **kw)

    # ------------------------------------------------------------------
    def _policy(self, x_hat, P, t, *, nu=None):
        return np.zeros((max(self.model.l, 1), 1))

    def _update(self, x_hat, P, t, *, nu=None):
        if nu is None or self._model_nn is None:
            return None
        if self._Q_nominal is None:
            self._Q_nominal = self.model.Q(t=t).copy()

        self._nu_history.append(float(np.asarray(nu).ravel()[0]))
        if len(self._nu_history) < self.window:
            return None

        window = np.array(self._nu_history[-self.window:])
        feat = (_window_feats(window) - self._x_mean) / self._x_std
        probs = np.asarray(self._model_nn(feat), dtype=float)
        label = self._labels[int(np.argmax(probs))]
        scale = np.clip(self.regime_scales[label], self.q_min, self.q_max)

        self.model._Q = np.diag(scale) @ self._Q_nominal
        self.scale_log.append((t, scale.copy(), label))
        return None

    def reset(self):
        super().reset()
        if self._Q_nominal is not None:
            self.model._Q = self._Q_nominal.copy()
        self._nu_history.clear()
        self.scale_log.clear()

    # ------------------------------------------------------------------
    def fit(self, regime_scales: dict, n_episodes=300, episode_len=60,
            epochs=150, lr=2e-3, batch_size=32, seed=None, model_arch=None):
        """Simulate labelled episodes for each named regime and train a classifier.

        Parameters
        ----------
        regime_scales : dict[str, array-like of shape (n,)]
            Named noise "shapes" — e.g. ``{"nominal": [1,1,1], "lat": [25,1,1]}``.
            Training simulates ``model.f()`` with process noise scaled by each vector
            in turn; at inference the predicted regime's vector is applied to Q.
        model_arch : object, optional
            Any callable ``model_arch(feat) -> probs`` with the ``build``/``fit``
            convention used by ``NeuralRouter`` (e.g. ``routers.nn.MLP``). Defaults to
            a small ``MLP``.
        """
        from ..routers.nn import MLP
        from ..models.extended import ExtendedKalmanModel
        from ..engines.extended import ExtendedKalmanEngine

        rng = np.random.default_rng(seed)
        model = self.model
        n = model.n
        dt = 1.0 / model.fs if model.fs else 1.0
        Q0 = model.Q().copy()
        q_std0 = np.sqrt(np.diag(Q0))
        r_std = np.sqrt(np.diag(model.R()))

        self.regime_scales = {k: np.asarray(v, dtype=float) for k, v in regime_scales.items()}
        self._labels = list(self.regime_scales.keys())
        label_to_idx = {lbl: i for i, lbl in enumerate(self._labels)}

        engine = ExtendedKalmanEngine()

        def simulate(scale_vec, seed_i):
            rng_i = np.random.default_rng(seed_i)
            x_true = model.x0.copy()
            x_est, P_est = model.x0.copy(), (model.P0.copy() if hasattr(model, "P0") else np.eye(n))
            nu_hist = []
            t_ = 0.0
            for _ in range(episode_len):
                x_true = np.asarray(model.f(x_true, t=t_), dtype=float).reshape(-1, 1)
                # scale_vec multiplies Q (variance), like AdaptiveQRKalmanController's
                # lambda and the diag(scale) @ Q_nominal applied at inference below --
                # so the std used to draw noise is sqrt(scale_vec), not scale_vec itself.
                w = rng_i.normal(0, q_std0 * np.sqrt(scale_vec)).reshape(-1, 1)
                x_true = x_true + w
                z = np.asarray(model.h(x_true, t=t_), dtype=float).reshape(-1, 1)
                z = z + rng_i.normal(0, r_std).reshape(-1, 1)

                x_est, P_est, _ = engine._predict(model, x_est, P_est, t=t_)
                h_pred = model.h(x_est, t=t_)
                nu = float((z - np.asarray(h_pred, dtype=float).reshape(-1, 1)).ravel()[0])
                nu_hist.append(nu)
                x_est, P_est = engine._update(model, x_est, P_est, z, t=t_)
                t_ += dt
            return np.array(nu_hist)

        X_all, y_all = [], []
        for ep in range(n_episodes):
            label = self._labels[ep % len(self._labels)]
            nu_hist = simulate(self.regime_scales[label], seed_i=(seed or 0) * 100000 + ep)
            X_all.append(_window_feats(nu_hist[-self.window:]))
            y_all.append(label_to_idx[label])

        X = np.array(X_all, dtype=np.float32)
        y = np.array(y_all)
        self._x_mean = X.mean(0)
        self._x_std = X.std(0)
        self._x_std[self._x_std < 1e-8] = 1.0
        Xn = (X - self._x_mean) / self._x_std

        self._model_nn = model_arch if model_arch is not None else MLP(hidden=self.hidden)
        self._model_nn.build(X.shape[1], len(self._labels), seed=seed)
        self._model_nn.fit(Xn, y, epochs=epochs, lr=lr, batch_size=batch_size)
        return self
