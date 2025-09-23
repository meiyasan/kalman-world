from . import KalmanStateModel

# @TODO: To be tested and documented
class KalmanExtendedModel(KalmanStateModel):
    """
    Extended Kalman Model (EKF or Composite)
    - Supports linear and nonlinear system models.
    - Optionally handles composite models.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _predict(self, x, P, **kwargs):

        # Nonlinear dynamics: x_next = f(x)
        f = kwargs.get("f")
        if f is None:
            raise ValueError("Function f(x) required for EKF mode.")
        J = kwargs.get("J")
        if J is None:
            raise ValueError("Jacobian required for EKF mode.")
        
        x = f(x)
        P = J(x) @ P @ J(x).T + self.Q
        return x, P