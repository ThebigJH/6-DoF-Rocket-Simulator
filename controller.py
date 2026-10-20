"""Discrete PID flight computer and finite-rate thrust-vector actuators."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ControllerOutput:
    """Attitude errors, requested commands, and actual TVC angles.

    Attributes:
        pitch_error_rad: Body-axis pitch attitude error in radians.
        yaw_error_rad: Body-axis yaw attitude error in radians.
        pitch_command_rad: Requested pitch-channel nozzle angle in radians.
        yaw_command_rad: Requested yaw-channel nozzle angle in radians.
        pitch_gimbal_rad: Actual pitch-channel servo angle in radians.
        yaw_gimbal_rad: Actual yaw-channel servo angle in radians.
    """

    pitch_error_rad: float
    yaw_error_rad: float
    pitch_command_rad: float
    yaw_command_rad: float
    pitch_gimbal_rad: float
    yaw_gimbal_rad: float


class FlightController:
    """Two-axis discrete PID controller with anti-windup and slew limits.

    Attitude error is the rocket's body-axis rotation vector relative to the
    upright launch attitude. The command sign mapping accounts for the
    opposite pitch moment produced by an engine mounted behind the centre of
    mass.
    """

    def __init__(
        self,
        sample_rate_hz: float = 100.0,
        max_gimbal_deg: float = 5.0,
        max_servo_rate_degps: float = 30.0,
        proportional_gain_per_s2: float = 3.0,
        integral_gain_per_s3: float = 0.12,
        derivative_gain_per_s: float = 1.7,
    ) -> None:
        """Configure sampling, PID gains, and physical actuator constraints.

        Args:
            sample_rate_hz: Flight-computer sampling frequency in hertz.
            max_gimbal_deg: Symmetric hard stop for each nozzle axis.
            max_servo_rate_degps: Maximum actuator angular rate.
            proportional_gain_per_s2: Proportional gain.
            integral_gain_per_s3: Integral gain.
            derivative_gain_per_s: Derivative gain.

        Raises:
            ValueError: If sample rate or either actuator limit is non-positive.
        """
        if sample_rate_hz <= 0.0 or max_gimbal_deg <= 0.0:
            raise ValueError("Sample rate and gimbal limit must be positive.")
        if max_servo_rate_degps <= 0.0:
            raise ValueError("Servo rate limit must be positive.")
        self.sample_period_s = 1.0 / sample_rate_hz
        self.max_gimbal_rad = math.radians(max_gimbal_deg)
        self.max_servo_rate_radps = math.radians(max_servo_rate_degps)
        self.kp = proportional_gain_per_s2
        self.ki = integral_gain_per_s3
        self.kd = derivative_gain_per_s
        self._integral_rad_s = np.zeros(2, dtype=float)
        self._previous_error_rad: FloatArray | None = None
        self._actual_gimbal_rad = np.zeros(2, dtype=float)

    @staticmethod
    def _rotation_vector_rad(quaternion_wxyz: FloatArray) -> FloatArray:
        """Convert a unit quaternion to its shortest rotation vector.

        Args:
            quaternion_wxyz: Body-to-world attitude quaternion ``[w, x, y, z]``.

        Returns:
            Rotation vector in radians, expressed about world axes.
        """
        quaternion = np.asarray(quaternion_wxyz, dtype=float)
        quaternion_norm = float(np.linalg.norm(quaternion))
        if quaternion_norm < 1e-12:
            raise ValueError("Attitude quaternion must have non-zero norm.")
        quaternion = quaternion / quaternion_norm
        if quaternion[0] < 0.0:
            quaternion = -quaternion
        vector_norm = float(np.linalg.norm(quaternion[1:]))
        if vector_norm < 1e-12:
            return np.zeros(3, dtype=float)
        angle_rad = 2.0 * math.atan2(vector_norm, max(0.0, quaternion[0]))
        return quaternion[1:] * (angle_rad / vector_norm)

    def update(
        self, quaternion_wxyz: FloatArray, delta_time_s: float | None = None
    ) -> ControllerOutput:
        """Run one discrete PID update and advance both servo positions.

        Conditional integration freezes the integral state when the output is
        saturated and the current error would drive it farther into saturation.
        Servo position is separately rate-limited and hard-clamped to the
        +/-5 degree hardware stops.

        Args:
            quaternion_wxyz: Current body-to-world attitude quaternion.
            delta_time_s: Elapsed time since the last update; defaults to the
                configured nominal controller period.

        Returns:
            Current errors, PID requests, and achieved nozzle angles.

        Raises:
            ValueError: If elapsed time is not positive.
        """
        dt_s = self.sample_period_s if delta_time_s is None else delta_time_s
        if dt_s <= 0.0:
            raise ValueError("Controller update interval must be positive.")
        rotation_vector_rad = self._rotation_vector_rad(quaternion_wxyz)
        error_rad = np.array(
            [rotation_vector_rad[1], -rotation_vector_rad[0]], dtype=float
        )
        derivative_radps = (
            np.zeros(2, dtype=float)
            if self._previous_error_rad is None
            else (error_rad - self._previous_error_rad) / dt_s
        )
        integral_candidate_rad_s = self._integral_rad_s + error_rad * dt_s
        raw_command_rad = (
            self.kp * error_rad
            + self.ki * integral_candidate_rad_s
            + self.kd * derivative_radps
        )
        bounded_command_rad = np.clip(
            raw_command_rad, -self.max_gimbal_rad, self.max_gimbal_rad
        )
        drives_further_into_limit = (
            (raw_command_rad > self.max_gimbal_rad) & (error_rad > 0.0)
        ) | ((raw_command_rad < -self.max_gimbal_rad) & (error_rad < 0.0))
        self._integral_rad_s = np.where(
            drives_further_into_limit,
            self._integral_rad_s,
            integral_candidate_rad_s,
        )
        raw_command_rad = (
            self.kp * error_rad
            + self.ki * self._integral_rad_s
            + self.kd * derivative_radps
        )
        bounded_command_rad = np.clip(
            raw_command_rad, -self.max_gimbal_rad, self.max_gimbal_rad
        )

        max_servo_step_rad = self.max_servo_rate_radps * dt_s
        servo_delta_rad = np.clip(
            bounded_command_rad - self._actual_gimbal_rad,
            -max_servo_step_rad,
            max_servo_step_rad,
        )
        self._actual_gimbal_rad = np.clip(
            self._actual_gimbal_rad + servo_delta_rad,
            -self.max_gimbal_rad,
            self.max_gimbal_rad,
        )
        self._previous_error_rad = error_rad
        return ControllerOutput(
            pitch_error_rad=float(error_rad[0]),
            yaw_error_rad=float(rotation_vector_rad[0]),
            pitch_command_rad=float(bounded_command_rad[0]),
            yaw_command_rad=float(bounded_command_rad[1]),
            pitch_gimbal_rad=float(self._actual_gimbal_rad[0]),
            yaw_gimbal_rad=float(self._actual_gimbal_rad[1]),
        )
