"""
What only one engine needs from a model.

The model contract (``f``, ``h``, ``Q``, ``R``, the initial state) is all most
engines ask. Two engines ask one thing more, and a model that can answer
inherits the matching class here, beside :class:`~.model.KalmanModel`. The
engine checks for the class, not for a method name. They mirror the C++
interfaces ``Kalman::Engines::EnsembleModel`` and ``Kalman::Tracking::Detector``.

    class Flow(KalmanEnsembleModel, KalmanModel): ...      # the ensemble filter
    class Chamber(KalmanDetectorModel, KalmanModel): ...   # track finding
"""
from __future__ import annotations


class KalmanEnsembleModel:
    """A model that tells the ensemble filter how far a measurement may reach.

    The ensemble filter estimates its covariances from a few members, which leaves
    small false correlations between quantities that are far apart. A model that
    knows its geometry gives a **covariance mask**: weights between 0 and 1, 1 for
    a state next to a sensor and falling to 0 beyond some distance, by which the
    engine multiplies those covariances entry by entry.

    The ensemble literature calls this *covariance localization*, and the engine
    keeps that word for its own setting (``localization=``). It has nothing to do
    with locating something in space.
    """

    def covariance_mask(self):
        """``(rho_xy, rho_yy)``: the mask on the state-measurement covariance,
        ``(n, m)``, and on the measurement-measurement covariance, ``(m, m)``; or
        ``None`` for no mask."""
        return None


class KalmanDetectorModel:
    """A model track finding can run on: measurements arrive layer by layer, several
    per layer, and a track is started from a single one.

    ``radii`` gives the layers, innermost first; the "time" handed to ``f`` is one
    of them. ``seed_state(hit)`` gives the ``(x0, P0)`` of a track that left
    ``hit`` on an inner layer. Everything else track finding needs (``f`` to the
    next layer, ``h``, ``R``) is the model contract.
    """

    #: The layer radii, innermost first.
    radii = ()

    def seed_state(self, hit):
        """``(x0, P0)`` for a track that produced ``hit`` on an inner layer."""
        raise NotImplementedError(f"{type(self).__name__} must give seed_state(hit) to be tracked.")
