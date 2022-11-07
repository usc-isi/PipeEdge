"""Controller utilities."""
import math
from typing import List, Optional, Tuple

class KalmanFilter:
    """
    A Kalman filter for estimating a scalar value.

    Attributes
    ----------
    Q : float
        Process noise covariance (constant).
    R : float
        Measurement noise covariance (constant).

    Parameters
    ----------
    x_hat_0 : float, optional
        The `x_hat` value (a posteriori state estimate) for time ``k = 0``.
    p_0 : float, optional
        The `P` value (a posteriori estimate covariance) for time ``k = 0``.
        Set ``p_0 > 0`` unless `x_hat` is certain (default: ``1``).

    References
    ----------
    [1] Greg Welch and Gary Bishop. An Introduction to the Kalman Filter. Tech. rep. TR 95-041.
    UNC Chapel Hill, Department of Computer Science.
    https://www.cs.unc.edu/~welch/media/pdf/kalman_intro.pdf
    """
    # pylint: disable=invalid-name

    def __init__(self, x_hat_0: float=0, p_0: float=1):
        self._x_hat = x_hat_0
        self._p = p_0
        self.Q = 0.00001
        self.R = 0.01

    @property
    def x_hat(self):
        """Get the current estimate."""
        return self._x_hat

    def __call__(self, z: float, h: float=1) -> float:
        """
        Discrete time step - recompute estimate.

        Parameters
        ----------
        z : float
            The measured value ``z(k)``.
        h : float, optional
            The measurement prediction ``h(k)``.

        Returns
        -------
        float
            The new estimate ``x_hat(k)``.
        """
        # a priori state estimate
        x_hat_minus = self._x_hat
        # a priori estimate covariance
        p_minus = self._p + self.Q
        # gain / blending factor
        k = (p_minus * h) / ((h * p_minus * h) + self.R)
        self._x_hat = x_hat_minus + (k * (z - (h * x_hat_minus)))
        self._p = (1.0 - (k * h)) * p_minus
        return self._x_hat


class AdaptiveIntegralXupController:
    """
    A control-theoretic adaptive integral X-up (e.g., speedup) controller.

    Uses a Kalman filter to estimate a base workload.
    The reciprocal of this estimate replaces ``K_I`` (the ratio of control change).

    Parameter ``u_max`` should be set based on actuator saturation values.
    This prevents controller windup by defining the range in which the controller is continuous.

    Parameters
    ----------
    reference : float
        The reference control value (set point).
    u_0 : float
        The control signal value expected based on the initial setting(s) for time ``k = 0``.
    u_max : float, optional
        The maximum control signal value where ``u_max > 1``.
    pole : float, optional
        The pole value (``0 <= pole < 1``) determines how fast the controller responds.
        Small values make the controller highly reactive but more susceptible to noise.
        Larger values make the controller slower to respond but more robust to transient dynamics.
    kf_kwargs : dict, optional
        Keyword arguments for the `KalmanFilter` initialization.

    References
    ----------
    [1] Joseph L. Hellerstein, Yixin Diao, Sujay Parekh, and Dawn M. Tilbury.
    Feedback Control of Computing Systems. 2004.
    [2] William S. Levine. The Control Handbook: Control System Fundamentals, Second Edition. 2011.
    """
    # pylint: disable=invalid-name

    def __init__(self, reference: float, u_0: float, u_max: float=float('inf'), pole: float=0,
                 kf_kwargs: Optional[dict]=None):
        # pylint: disable=too-many-arguments
        self.reference = reference
        self._u = u_0
        self._u_max = u_max
        self.pole = pole
        if kf_kwargs is None:
            kf_kwargs = {}
        self._kalman_filt = KalmanFilter(**kf_kwargs)

    @property
    def pole(self):
        """Get the pole value."""
        return self._pole

    @pole.setter
    def pole(self, pole):
        """Set the pole value in range ``[0, 1)``."""
        if pole < 0 or pole >= 1:
            raise ValueError("pole must be in range [0, 1)")
        self._pole = pole

    def __call__(self, y: float) -> float:
        """
        Compute a new control signal.

        Parameters
        ----------
        y : float
            The measured value ``y(k)``.

        Returns
        -------
        float
            The control signal ``u(k+1)``.
        """
        base_workload = self._kalman_filt(y, h=self._u)
        e = self.reference - y
        u = self._u + (1 - self._pole) * (e / base_workload)
        # Clamp the control signal if it's saturated
        self._u = max(min(u, self._u_max), 1)
        return self._u


class AdaptiveBitwidthPerformanceController(AdaptiveIntegralXupController):
    """
    An adaptive controller that computes bitwidths to meet a data movement performance constraint.

    Models speedup as inversely proportional to bitwidth, normalized to the max bitwidth.
    This model assumes perfect data packing and no other data transfer overhead.
    In reality, packing is imperfect and compression metadata may also need to be sent.

    Parameters
    ----------
    perf_constraint : float
        The performance constraint to satisfy.
    bitwidths : List[int]
        The available bitwidth values.
    bitwidth_start : int
        The bitwidth used prior to the first iteration.

    References
    ----------
    [1] H. Hoffmann, M. Maggio, M. D. Santambrogio, A. Leva and A. Agarwal.
    A generalized software framework for accurate and efficient management of performance goals.
    2013 Proceedings of the International Conference on Embedded Software (EMSOFT). 2013.
    """

    def __init__(self, perf_constraint: float, bitwidths: List[int], bitwidth_start: int):
        self._bitwidths = list(bitwidths) # copy, then sort in reverse
        self._bitwidths.sort(reverse=True)
        self._speedups = [self._bitwidths[0] / b for b in self._bitwidths]
        # Use the parent controller class to compute speedup over max bitwidth baseline.
        u_0 = self._bitwidths[0] / bitwidth_start
        # We could use a performance measurement to estimate `x_hat_0` for the underlying Kalman
        # filter, but there's no real benefit - the filter converges on the first iteration anyway.
        super().__init__(perf_constraint, u_0, u_max=self._speedups[-1])

    def __call__(self, perf_measured: float, window_len: int) -> Tuple[int, int, int]:
        """
        Split a window period between two bitwidths to achieve ``perf_constraint``.

        The number of iterations to spend in a bitwidth may be ``0`` or ``window_len``.

        Parameters
        ----------
        perf_measured : float
            The measured performance.
        window_len : int
            The window length.

        Returns
        -------
        tuple
            Tuple with 3 values: bitwidth #1, bitwidth #2, and the number of iterations to spend
            in bitwidth #1 during the next window period.
        """
        xup_targ = super().__call__(perf_measured)
        idx_slow = max(0, len([s for s in self._speedups if s <= xup_targ]) - 1)
        idx_fast = min(idx_slow + 1, len(self._speedups) - 1)
        xup_slow = self._speedups[idx_slow]
        xup_fast = self._speedups[idx_fast]
        # The time period of the combined rates must equal the time period of the target rate.
        # 1 / target_rate = x / slower_rate + (1 - x) / faster_rate
        # Solve for x:
        if math.isclose(xup_slow, xup_fast):
            _x = 0 # could also be 1.0
        else:
            _x = (xup_slow * (xup_fast - xup_targ)) / (xup_targ * (xup_fast - xup_slow))
        # Num of iterations = x * window_size
        num_iter = round(window_len * _x)
        return (self._bitwidths[idx_slow], self._bitwidths[idx_fast], num_iter)
