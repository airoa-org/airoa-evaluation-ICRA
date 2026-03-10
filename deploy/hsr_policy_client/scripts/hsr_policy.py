#!/home/policy/.venv/bin/python3
from collections import deque
import json

import os
import re
import time
from typing import Any, Optional

from actionlib import SimpleActionClient
import cv2
from geometry_msgs.msg import Twist
from hsr_policy_client.srv import StringTrigger
from hsr_policy_client.srv import StringTriggerResponse
import numpy as np

# ROS-related imports
import rospy
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from tmc_control_msgs.msg import GripperApplyEffortAction
from tmc_control_msgs.msg import GripperApplyEffortActionGoal
from trajectory_msgs.msg import JointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint

from policy_client.websocket_client_policy import WebsocketClientPolicy


MODE_CONTINUOUS = "continuous"
MODE_DISCRETE = "discrete"
MODE_HYBRID = "hybrid"
MODES = [MODE_CONTINUOUS, MODE_DISCRETE, MODE_HYBRID]

UPSAMPLE_METHOD_SPLINE = "spline"
UPSAMPLE_METHOD_LINEAR = "linear"
UPSAMPLE_METHODS = [UPSAMPLE_METHOD_SPLINE, UPSAMPLE_METHOD_LINEAR]

SERVICE_UPDATE_INSTRUCTION = "/hsr_policy_client/update_instruction"
SERVICE_SET_MOTION_ENABLED = "/hsr_policy_client/set_motion_enabled"
SERVICE_HAS_ACTION_OUTPUT = "/hsr_policy_client/has_action_output"
SERVICE_RESET_ACTION_OUTPUT_FLAG = "/hsr_policy_client/reset_action_output_flag"
TOPIC_ORIGINAL_ACTION_CHUNK = "/hsr_policy_client/original_action_chunk"

ACTION_SMOOTHING_NONE = "none"
ACTION_SMOOTHING_EMA = "ema"
ACTION_SMOOTHING_MA = "moving_average"
ACTION_SMOOTHING_METHODS = [ACTION_SMOOTHING_NONE, ACTION_SMOOTHING_EMA, ACTION_SMOOTHING_MA]

SYNTH_TEST_IMAGE_HEIGHT = 480
SYNTH_TEST_IMAGE_WIDTH = 640
SYNTH_TEST_RANDOM_SEED = 0


def _float_tag(x: float, *, ndigits: int = 3) -> str:
    """Filesystem-friendly float tag, e.g. 0.2 -> 0p200, -1.5 -> m1p500."""
    try:
        x = float(x)
    except Exception:
        return "nan"
    sign = "m" if x < 0 else ""
    x = abs(x)
    s = f"{x:.{ndigits}f}".replace(".", "p")
    return f"{sign}{s}"


def _build_trace_group_name(
    *,
    config_name: str,
    adopted_action_chunks: int,
    update_freq: int,
    upsample: bool,
    upsample_hz: int,
    upsample_method: str,
    action_smoothing: str,
    ema_alpha: float,
    ma_window: int,
    smooth_gripper: bool,
    smooth_base: bool,
) -> str:
    parts: list[str] = []
    parts.append(f"ac{int(adopted_action_chunks)}")
    parts.append(f"uf{int(update_freq)}")

    if upsample:
        parts.append(f"up{int(upsample_hz)}")
        parts.append(f"um{str(upsample_method)}")
    else:
        parts.append("original")

    if action_smoothing and action_smoothing != ACTION_SMOOTHING_NONE:
        parts.append(f"sm{str(action_smoothing)}")
        if action_smoothing == ACTION_SMOOTHING_EMA:
            parts.append(f"a{_float_tag(ema_alpha)}")
        elif action_smoothing == ACTION_SMOOTHING_MA:
            parts.append(f"w{int(ma_window)}")
        if smooth_gripper:
            parts.append("sg")
        if smooth_base:
            parts.append("sb")

    return f"{config_name}_" + "_".join(parts)


def _natural_cubic_spline_interpolate(x: np.ndarray, y: np.ndarray, xq: np.ndarray) -> np.ndarray:
    """Natural cubic spline interpolation (2nd derivative = 0 at both ends).

    Parameters
    ----------
    x : np.ndarray, shape (N,)
        Strictly increasing knot positions.
    y : np.ndarray, shape (N, D)
        Values at knots.
    xq : np.ndarray, shape (M,)
        Query positions in [x[0], x[-1]].

    Returns
    -------
    np.ndarray, shape (M, D)
        Interpolated values.
    """
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    xq = np.asarray(xq, dtype=np.float32).reshape(-1)
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 1:
        y = y[:, None]
    if x.ndim != 1:
        raise ValueError("x must be 1D.")
    if y.shape[0] != x.shape[0]:
        raise ValueError("y must have the same length as x.")
    if x.shape[0] < 2:
        return np.repeat(y[:1], xq.shape[0], axis=0)

    n = x.shape[0]
    dim = y.shape[1]
    h = np.diff(x)
    if np.any(h <= 0):
        raise ValueError("x must be strictly increasing.")

    if n == 2:
        # Linear interpolation fallback.
        t = (xq - x[0]) / (x[1] - x[0])
        t = t[:, None]
        return y[0:1] * (1.0 - t) + y[1:2] * t

    # Solve for second-derivative coefficients c (natural boundary: c0=cn-1=0).
    c = np.zeros((n, dim), dtype=np.float32)
    m = n - 2  # number of interior points
    if m > 0:
        lower = h[:-1]  # (m-1,)
        diag = 2.0 * (h[:-1] + h[1:])  # (m,)
        upper = h[1:]  # (m-1,)
        rhs = 3.0 * (
            (y[2:] - y[1:-1]) / h[1:, None] - (y[1:-1] - y[:-2]) / h[:-1, None]
        )  # (m, dim)

        if m == 1:
            c[1:-1] = rhs / diag[:, None]
        else:
            # Thomas algorithm for tridiagonal systems.
            cp = np.empty((m - 1,), dtype=np.float32)
            dp = np.empty((m, dim), dtype=np.float32)

            cp[0] = upper[0] / diag[0]
            dp[0] = rhs[0] / diag[0]
            for i in range(1, m - 1):
                denom = diag[i] - lower[i - 1] * cp[i - 1]
                cp[i] = upper[i] / denom
                dp[i] = (rhs[i] - lower[i - 1] * dp[i - 1]) / denom
            denom = diag[m - 1] - lower[m - 2] * cp[m - 2]
            dp[m - 1] = (rhs[m - 1] - lower[m - 2] * dp[m - 2]) / denom

            c_inner = np.empty((m, dim), dtype=np.float32)
            c_inner[m - 1] = dp[m - 1]
            for i in range(m - 2, -1, -1):
                c_inner[i] = dp[i] - cp[i] * c_inner[i + 1]
            c[1:-1] = c_inner

    # Coefficients for each segment [x_i, x_{i+1})
    b = (y[1:] - y[:-1]) / h[:, None] - (h[:, None] * (2.0 * c[:-1] + c[1:]) / 3.0)
    d = (c[1:] - c[:-1]) / (3.0 * h[:, None])
    a = y[:-1]
    c_seg = c[:-1]

    # Evaluate.
    idx = np.searchsorted(x[1:], xq, side="right")
    idx = np.clip(idx, 0, n - 2)
    dx = (xq - x[idx])[:, None]
    return a[idx] + b[idx] * dx + c_seg[idx] * (dx**2) + d[idx] * (dx**3)


def _linear_interpolate(x: np.ndarray, y: np.ndarray, xq: np.ndarray) -> np.ndarray:
    """Piecewise linear interpolation.

    Parameters
    ----------
    x : np.ndarray, shape (N,)
        Strictly increasing knot positions.
    y : np.ndarray, shape (N, D)
        Values at knots.
    xq : np.ndarray, shape (M,)
        Query positions in [x[0], x[-1]].

    Returns
    -------
    np.ndarray, shape (M, D)
        Interpolated values.
    """
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    xq = np.asarray(xq, dtype=np.float32).reshape(-1)
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 1:
        y = y[:, None]
    if x.ndim != 1:
        raise ValueError("x must be 1D.")
    if y.shape[0] != x.shape[0]:
        raise ValueError("y must have the same length as x.")
    if x.shape[0] < 2:
        return np.repeat(y[:1], xq.shape[0], axis=0)

    h = np.diff(x)
    if np.any(h <= 0):
        raise ValueError("x must be strictly increasing.")

    # Find segment indices so that x[idx] <= xq < x[idx+1].
    idx = np.searchsorted(x[1:], xq, side="right")
    idx = np.clip(idx, 0, x.shape[0] - 2)

    x0 = x[idx]
    x1 = x[idx + 1]
    y0 = y[idx]
    y1 = y[idx + 1]
    denom = (x1 - x0)[:, None]
    # Avoid division by zero in pathological cases.
    denom = np.where(denom == 0, 1.0, denom)
    w = ((xq - x0)[:, None]) / denom
    return (y0 * (1.0 - w) + y1 * w).astype(np.float32, copy=False)


def _cubic_spline_upsample_actions(
    actions: np.ndarray,
    *,
    in_hz: float,
    out_hz: float,
    out_steps: Optional[int] = None,
) -> np.ndarray:
    """Upsample action sequence with natural cubic spline interpolation."""
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2:
        raise ValueError("actions must be 2D (T, D).")
    in_steps = int(actions.shape[0])
    if out_steps is None:
        out_steps = int(round(in_steps * float(out_hz) / float(in_hz)))
    out_steps = max(int(out_steps), 1)

    if in_steps == 0:
        return actions
    if in_steps == 1:
        return np.repeat(actions, out_steps, axis=0)
    if out_steps == in_steps:
        return actions

    duration_s = in_steps / float(in_hz)
    x = np.linspace(0.0, duration_s, in_steps + 1, dtype=np.float32)
    y = np.concatenate([actions, actions[-1:, :]], axis=0)
    xq = np.arange(out_steps, dtype=np.float32) / float(out_hz)
    return _natural_cubic_spline_interpolate(x, y, xq).astype(np.float32, copy=False)


def _linear_upsample_actions(
    actions: np.ndarray,
    *,
    in_hz: float,
    out_hz: float,
    out_steps: Optional[int] = None,
) -> np.ndarray:
    """Upsample action sequence with linear interpolation."""
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2:
        raise ValueError("actions must be 2D (T, D).")
    in_steps = int(actions.shape[0])
    if out_steps is None:
        out_steps = int(round(in_steps * float(out_hz) / float(in_hz)))
    out_steps = max(int(out_steps), 1)

    if in_steps == 0:
        return actions
    if in_steps == 1:
        return np.repeat(actions, out_steps, axis=0)
    if out_steps == in_steps:
        return actions

    duration_s = in_steps / float(in_hz)
    x = np.linspace(0.0, duration_s, in_steps + 1, dtype=np.float32)
    y = np.concatenate([actions, actions[-1:, :]], axis=0)
    xq = np.arange(out_steps, dtype=np.float32) / float(out_hz)
    return _linear_interpolate(x, y, xq).astype(np.float32, copy=False)


def _compute_original_step_exec_indices(
    *,
    in_steps: int,
    in_hz: float,
    out_steps: int,
    out_hz: float,
) -> np.ndarray:
    """Map each original step to the nearest upsampled execution index."""
    in_steps = max(int(in_steps), 0)
    out_steps = max(int(out_steps), 0)
    if in_steps == 0 or out_steps == 0:
        return np.zeros((0,), dtype=np.int32)
    if in_hz <= 0 or out_hz <= 0:
        limit = min(in_steps, out_steps)
        return np.arange(limit, dtype=np.int32)

    original_times = np.arange(in_steps, dtype=np.float32) / float(in_hz)
    exec_indices = np.rint(original_times * float(out_hz)).astype(np.int32)
    return np.clip(exec_indices, 0, out_steps - 1)


class ActionSmoother:
    def __init__(
        self,
        *,
        method: str,
        ema_alpha: float,
        ma_window: int,
        dims_mask: np.ndarray,
    ):
        self.method = str(method)
        if self.method not in ACTION_SMOOTHING_METHODS:
            rospy.logwarn(
                "Unknown action_smoothing '%s'. Falling back to '%s'. Available: %s",
                self.method,
                ACTION_SMOOTHING_NONE,
                ", ".join(ACTION_SMOOTHING_METHODS),
            )
            self.method = ACTION_SMOOTHING_NONE

        self.ema_alpha = float(ema_alpha)
        self.ema_alpha = float(np.clip(self.ema_alpha, 0.0, 1.0))
        self.ma_window = max(int(ma_window), 1)
        self.dims_mask = np.asarray(dims_mask, dtype=bool).reshape(-1)

        self._ema_state: Optional[np.ndarray] = None
        self._ma_buf: deque[np.ndarray] = deque(maxlen=self.ma_window)

    def update(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if self.method == ACTION_SMOOTHING_NONE:
            return action

        if self.dims_mask.shape[0] != action.shape[0]:
            # If shape mismatch, fall back to smoothing all dims.
            mask = np.ones_like(action, dtype=bool)
        else:
            mask = self.dims_mask

        out = np.array(action, copy=True)
        if self.method == ACTION_SMOOTHING_EMA:
            if self._ema_state is None or self._ema_state.shape != action.shape:
                self._ema_state = np.array(action, copy=True)
                return out
            a = self.ema_alpha
            self._ema_state[mask] = a * action[mask] + (1.0 - a) * self._ema_state[mask]
            out[mask] = self._ema_state[mask]
            return out

        # Moving average
        self._ma_buf.append(action)
        if len(self._ma_buf) == 0:
            return out
        stacked = np.stack(list(self._ma_buf), axis=0)
        out[mask] = stacked[:, mask].mean(axis=0).astype(np.float32, copy=False)
        return out


def _param_to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value) != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "y"}
    return bool(value)


def _parse_motion_toggle_message(message: Any) -> Optional[bool]:
    text = str(message).strip().lower()
    if text in {"1", "true", "on", "enable", "enabled", "auto", "resume", "run", "start"}:
        return True
    if text in {"0", "false", "off", "disable", "disabled", "manual", "stop", "pause", "halt"}:
        return False
    return None


class SyntheticReplayEnv:
    """Environment for synthetic test-mode replay without requiring a robot."""

    def __init__(
        self,
        *,
        update_freq: int = 10,
    ):
        self.update_freq = int(update_freq)
        self.rate = rospy.Rate(self.update_freq)
        self.image_height = SYNTH_TEST_IMAGE_HEIGHT
        self.image_width = SYNTH_TEST_IMAGE_WIDTH
        self.random_seed = SYNTH_TEST_RANDOM_SEED
        self._rng = np.random.default_rng(self.random_seed)

        self.instruction = str(rospy.get_param("~instruction", "Grasp the apple."))
        self.gripper_state = 0
        self.control_mode = "auto"
        self.motion_enabled = True
        self.last_execute_block_reason: Optional[str] = None
        self.joint_state: Optional[np.ndarray] = None
        self._last_action: Optional[np.ndarray] = None
        self._has_action_output = False

        self.joint_state_names: list[str] = [
            "arm_lift_joint",
            "arm_flex_joint",
            "arm_roll_joint",
            "wrist_flex_joint",
            "wrist_roll_joint",
            "hand_motor_joint",
            "head_pan_joint",
            "head_tilt_joint",
        ]
        self.base_action_names: list[str] = ["base_x", "base_y", "base_theta"]

        rospy.Service(SERVICE_UPDATE_INSTRUCTION, StringTrigger, self.update_instruction_srv)
        rospy.Service(SERVICE_SET_MOTION_ENABLED, StringTrigger, self.set_motion_enabled_srv)
        rospy.Service(SERVICE_HAS_ACTION_OUTPUT, StringTrigger, self.has_action_output_srv)
        rospy.Service(SERVICE_RESET_ACTION_OUTPUT_FLAG, StringTrigger, self.reset_action_output_flag_srv)
        rospy.loginfo(
            "Test mode enabled. Using synthetic random data image=%dx%d seed=%d (infinite loop)",
            self.image_height,
            self.image_width,
            self.random_seed,
        )

    def update_instruction_srv(self, req: StringTrigger):
        self.instruction = req.message
        rospy.loginfo("Instruction updated: %s", self.instruction)
        return StringTriggerResponse(success=True)

    def set_motion_enabled_srv(self, req: StringTrigger):
        parsed = _parse_motion_toggle_message(req.message)
        if parsed is None:
            rospy.logwarn("Invalid set_motion_enabled request: %r", req.message)
            return StringTriggerResponse(success=False)
        self.motion_enabled = bool(parsed)
        rospy.loginfo(
            "Motion forwarding changed to: %s",
            "enabled" if self.motion_enabled else "disabled",
        )
        return StringTriggerResponse(success=True)

    def has_action_output_srv(self, req: StringTrigger):
        _ = req
        return StringTriggerResponse(success=bool(self._has_action_output))

    def reset_action_output_flag_srv(self, req: StringTrigger):
        _ = req
        self._has_action_output = False
        return StringTriggerResponse(success=True)

    def is_finished(self) -> bool:
        return False

    def reset_observation(self, *, reset_joint_state: bool = True):
        _ = reset_joint_state
        return

    def _make_sample(self) -> dict[str, Any]:
        head_rgb = self._rng.integers(
            low=0, high=256, size=(self.image_height, self.image_width, 3), dtype=np.uint8
        )
        hand_rgb = self._rng.integers(
            low=0, high=256, size=(self.image_height, self.image_width, 3), dtype=np.uint8
        )
        joint_state = self._rng.normal(loc=0.0, scale=0.5, size=(8,)).astype(np.float32)

        return {
            "head_rgb": head_rgb,
            "hand_rgb": hand_rgb,
            "joint_state": joint_state,
            "instruction": self.instruction,
            "gripper_state": 0,
            "control_mode": "auto",
        }

    def get_observations(self):
        sample = self._make_sample()
        self.joint_state = np.asarray(sample["joint_state"], dtype=np.float32)
        return sample

    def execute_actions(self, action: np.ndarray) -> bool:
        self._last_action = np.asarray(action, dtype=np.float32).reshape(-1)
        self._has_action_output = True
        if not self.motion_enabled:
            self.last_execute_block_reason = "motion_disabled"
            return False
        self.last_execute_block_reason = None
        return True

    def sleep(self):
        self.rate.sleep()


class HSREnv:
    """
    Runtime environment class that receives HSR sensor observations via ROS
    and applies actions to the robot.
    """

    GRIPPER_OPEN = 1
    GRIPPER_CLOSE = 0
    GRIPPER_CLOSE_THRESHOLD = 0.5  # Threshold to trigger gripper close behavior.

    def __init__(self, update_freq=10):
        self.update_freq = update_freq
        self.rate = rospy.Rate(self.update_freq)

        # Initialize sensor and runtime state.
        self.head_rgb = None
        self.hand_rgb = None
        self.joint_state = None
        self.gripper_state = 0
        self.control_mode = str(rospy.get_param("~default_control_mode", "auto")).strip().lower()
        self.motion_enabled = True
        self.last_execute_block_reason: Optional[str] = None
        self.gripper_mode = rospy.get_param("~gripper_mode", "continuous")
        self.instruction = rospy.get_param("~instruction", "Grasp the apple.")
        self._has_action_output = False
        rospy.loginfo("Initial control_mode: %s", self.control_mode)

        self.joint_state_names: list[str] = [
            "arm_lift_joint",
            "arm_flex_joint",
            "arm_roll_joint",
            "wrist_flex_joint",
            "wrist_roll_joint",
            "hand_motor_joint",
            "head_pan_joint",
            "head_tilt_joint",
        ]

        self.arm_action_names: list[str] = [
            "arm_lift_joint",
            "arm_flex_joint",
            "arm_roll_joint",
            "wrist_flex_joint",
            "wrist_roll_joint",
        ]
        self.head_action_names: list[str] = ["head_pan_joint", "head_tilt_joint"]
        self.base_action_names: list[str] = ["base_x", "base_y", "base_theta"]

        # Initialize publishers.
        self.arm_pub = rospy.Publisher("/hsrb/arm_trajectory_controller/command", JointTrajectory, queue_size=1)
        self.head_pub = rospy.Publisher("/hsrb/head_trajectory_controller/command", JointTrajectory, queue_size=1)
        self.gripper_pub = rospy.Publisher("/hsrb/gripper_controller/command", JointTrajectory, queue_size=1)
        self.base_pub = rospy.Publisher("/hsrb/command_velocity", Twist, queue_size=1)
        self.gripper_close_client = SimpleActionClient("/hsrb/gripper_controller/grasp", GripperApplyEffortAction)

        # Register service (language instruction update).
        rospy.Service(SERVICE_UPDATE_INSTRUCTION, StringTrigger, self.update_instruction_srv)
        rospy.Service(SERVICE_SET_MOTION_ENABLED, StringTrigger, self.set_motion_enabled_srv)
        rospy.Service(SERVICE_HAS_ACTION_OUTPUT, StringTrigger, self.has_action_output_srv)
        rospy.Service(SERVICE_RESET_ACTION_OUTPUT_FLAG, StringTrigger, self.reset_action_output_flag_srv)

        # Initialize subscribers.
        rospy.Subscriber(
            "/hsrb/head_rgbd_sensor/rgb/image_rect_color/compressed",
            CompressedImage,
            self.head_image_callback,
            queue_size=1,
        )
        rospy.Subscriber(
            "/hsrb/hand_camera/image_raw/compressed", CompressedImage, self.hand_image_callback, queue_size=1
        )
        rospy.Subscriber("/hsrb/joint_states", JointState, self.joint_state_callback, queue_size=1)
        rospy.Subscriber("/hsrb/gripper_controller/command", JointTrajectory, self.gripper_open_callback, queue_size=1)
        rospy.Subscriber(
            "/hsrb/gripper_controller/grasp/goal",
            GripperApplyEffortActionGoal,
            self.gripper_close_callback,
            queue_size=1,
        )
        rospy.Subscriber("/control_mode", String, self.control_mode_callback, queue_size=1)

    def head_image_callback(self, msg: CompressedImage):
        np_arr = np.frombuffer(msg.data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)[:, :, :]  # bgr -> rgb
        self.head_rgb = np.array(image)

    def hand_image_callback(self, msg: CompressedImage):
        np_arr = np.frombuffer(msg.data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)[:, :, :]  # bgr -> rgb
        self.hand_rgb = np.array(image)

    def joint_state_callback(self, msg: JointState):
        joints = [msg.position[msg.name.index(name)] for name in self.joint_state_names]
        self.joint_state = np.asarray(joints, dtype=np.float32)

    def gripper_open_callback(self, msg: JointTrajectory):
        self.gripper_state = self.GRIPPER_OPEN

    def gripper_close_callback(self, msg: GripperApplyEffortActionGoal):
        self.gripper_state = self.GRIPPER_CLOSE

    def control_mode_callback(self, msg: String):
        self.control_mode = str(msg.data).strip().lower()
        rospy.loginfo("control_mode updated: %s", self.control_mode)

    def update_instruction_srv(self, req: StringTrigger):
        self.instruction = req.message
        rospy.loginfo("Instruction updated: %s", self.instruction)
        return StringTriggerResponse(success=True)

    def set_motion_enabled_srv(self, req: StringTrigger):
        parsed = _parse_motion_toggle_message(req.message)
        if parsed is None:
            rospy.logwarn("Invalid set_motion_enabled request: %r", req.message)
            return StringTriggerResponse(success=False)
        self.motion_enabled = bool(parsed)
        if not self.motion_enabled:
            self.base_pub.publish(Twist())
        rospy.loginfo(
            "Motion forwarding changed to: %s",
            "enabled" if self.motion_enabled else "disabled",
        )
        return StringTriggerResponse(success=True)

    def has_action_output_srv(self, req: StringTrigger):
        _ = req
        return StringTriggerResponse(success=bool(self._has_action_output))

    def reset_action_output_flag_srv(self, req: StringTrigger):
        _ = req
        self._has_action_output = False
        return StringTriggerResponse(success=True)

    def reset_observation(self, *, reset_joint_state: bool = True):
        """
        Reset cached sensor observations.
        """
        self.head_rgb = None
        self.hand_rgb = None
        if reset_joint_state:
            self.joint_state = None

    def get_observations(self):
        """
        Return a dictionary of current robot observations.
        Returns None until all required observation fields are available.
        """
        if self.head_rgb is None or self.hand_rgb is None or self.joint_state is None:
            return None
        return {
            "head_rgb": self.head_rgb,
            "hand_rgb": self.hand_rgb,
            "joint_state": self.joint_state,
            "instruction": self.instruction,
            "gripper_state": self.gripper_state,
            "control_mode": self.control_mode,
        }

    def execute_actions(self, action: np.ndarray) -> bool:
        """
        Apply an action vector to the robot.

        Parameters
        ----------
        action : np.ndarray
            Action to apply to the robot:
            [
                "arm_lift_joint",
                "arm_flex_joint",
                "arm_roll_joint",
                "wrist_flex_joint",
                "wrist_roll_joint",
                "hand_motor_joint",
                "head_pan_joint",
                "head_tilt_joint",
                "base_x",
                "base_y",
                "base_t",
            ]

        Returns
        -------
        bool
            True if action execution is allowed and sent, otherwise False.
        """
        self._has_action_output = True
        if not self.motion_enabled:
            self.last_execute_block_reason = "motion_disabled"
            return False
        # Execute only when control_mode is set to "auto".
        if self.control_mode != "auto":
            self.last_execute_block_reason = f"control_mode={self.control_mode!r}"
            return False  # Return False when execution is not allowed.
        self.last_execute_block_reason = None

        # Arm control.
        arm_traj = JointTrajectory()
        arm_traj.joint_names = self.arm_action_names
        arm_point = JointTrajectoryPoint()
        arm_point.positions = action[:5]
        arm_point.velocities = []
        arm_point.time_from_start = rospy.Duration(1 / self.update_freq / 2)
        arm_traj.points = [arm_point]

        # Head control.
        head_traj = JointTrajectory()
        head_traj.joint_names = self.head_action_names
        arm_point = JointTrajectoryPoint()
        arm_point.positions = action[6:8]
        arm_point.velocities = []
        arm_point.time_from_start = rospy.Duration(1 / self.update_freq / 2)

        head_traj.points = [arm_point]

        # Base control.
        twist = Twist()
        twist.linear.x = action[8]
        twist.linear.y = action[9]
        twist.angular.z = action[10]

        # Gripper control.
        if self.gripper_mode == "continuous":
            gripper_traj = JointTrajectory()
            gripper_traj.joint_names = ["hand_motor_joint"]
            arm_point = JointTrajectoryPoint()
            gripper_value = np.clip(action[5], -0.1, 1.23)
            arm_point.positions = [gripper_value]
            arm_point.velocities = []
            arm_point.time_from_start = rospy.Duration(1)
            gripper_traj.points = [arm_point]
            self.gripper_pub.publish(gripper_traj)
        elif self.gripper_mode == "discrete":
            # Decide whether to close the gripper: 1=close, 0=open.
            gripper_action = self.GRIPPER_CLOSE if action[5] < self.GRIPPER_CLOSE_THRESHOLD else self.GRIPPER_OPEN
            if self.gripper_state != gripper_action:
                if gripper_action == self.GRIPPER_CLOSE:  # Close gripper.
                    goal = GripperApplyEffortActionGoal()
                    goal.goal.effort = -0.018
                    self.gripper_close_client.send_goal(goal.goal)
                else:  # Open gripper.
                    arm_traj = JointTrajectory()
                    arm_traj.joint_names = ["hand_motor_joint"]
                    arm_point = JointTrajectoryPoint()
                    arm_point.positions = [1.239183768915874]
                    arm_point.velocities = []
                    arm_point.time_from_start = rospy.Duration(1)
                    arm_traj.points = [arm_point]
                    self.gripper_pub.publish(arm_traj)
                self.gripper_state = gripper_action
        elif self.gripper_mode == "hybrid":
            # Hybrid mode:
            # - Closing: continuous above threshold, discrete below threshold.
            # - Opening: same behavior as continuous mode.
            gripper_value = action[5]
            if gripper_value < self.GRIPPER_CLOSE_THRESHOLD:
                # Below threshold: same behavior as discrete mode (apply grasp effort).
                if self.gripper_state != self.GRIPPER_CLOSE:
                    goal = GripperApplyEffortActionGoal()
                    goal.goal.effort = -0.018
                    self.gripper_close_client.send_goal(goal.goal)
                    self.gripper_state = self.GRIPPER_CLOSE
            else:
                # Above threshold: same behavior as continuous mode.
                gripper_traj = JointTrajectory()
                gripper_traj.joint_names = ["hand_motor_joint"]
                arm_point = JointTrajectoryPoint()
                gripper_value = np.clip(gripper_value, -0.1, 1.23)
                arm_point.positions = [gripper_value]
                arm_point.velocities = []
                arm_point.time_from_start = rospy.Duration(1)
                gripper_traj.points = [arm_point]
                self.gripper_pub.publish(gripper_traj)
                self.gripper_state = self.GRIPPER_OPEN

        self.arm_pub.publish(arm_traj)
        self.head_pub.publish(head_traj)
        self.base_pub.publish(twist)

        return True

    def sleep(self):
        self.rate.sleep()


class ActionDebugPublisher:
    def __init__(self, *, action_names: list[str]):
        self.action_names = list(action_names)
        self.original_chunk_pub = rospy.Publisher(TOPIC_ORIGINAL_ACTION_CHUNK, String, queue_size=10)

    def publish_original_action_chunk(self, payload: dict[str, Any], *, publish_stamp: rospy.Time) -> None:
        event = dict(payload)
        event["action_names"] = list(self.action_names)
        event["published_at_ros_time_ns"] = int(publish_stamp.to_nsec())
        self.original_chunk_pub.publish(
            String(data=json.dumps(event, ensure_ascii=False, separators=(",", ":")))
        )


class OpenpiPolicy:
    """
    Policy runner that performs inference and returns actions.
    It consumes observations from HSREnv and applies model outputs back to the environment.
    """

    def __init__(
        self,
        policy_server_host: str = "127.0.0.1",
        policy_server_port: int | None = 8000,
        policy_server_api_key: Optional[str] = None,
        adopted_action_chunks: int = 15,  # Number of actions consumed from each inferred action chunk.
        action_hz: int = 10,  # Action chunk time resolution (Hz). Usually matches update_freq.
        upsample: bool = False,
        upsample_hz: int = 50,
        upsample_method: str = UPSAMPLE_METHOD_SPLINE,
    ):
        self.policy = WebsocketClientPolicy(
            host=policy_server_host,
            port=policy_server_port,
            api_key=policy_server_api_key,
        )
        try:
            metadata = self.policy.get_server_metadata()
            rospy.loginfo("Connected to policy server. metadata=%s", metadata)
        except Exception as e:
            rospy.logwarn("Failed to read policy server metadata: %s", e)

        self.adopted_action_chunks: int = adopted_action_chunks
        self.action_hz: int = action_hz
        self.upsample: bool = upsample
        self.upsample_hz: int = upsample_hz
        self.upsample_method: str = str(upsample_method)
        if self.upsample_method not in UPSAMPLE_METHODS:
            rospy.logwarn(
                "Unknown upsample_method '%s'. Falling back to '%s'. Available: %s",
                self.upsample_method,
                UPSAMPLE_METHOD_SPLINE,
                ", ".join(UPSAMPLE_METHODS),
            )
            self.upsample_method = UPSAMPLE_METHOD_SPLINE

        self.execution_action_chunks: int = adopted_action_chunks
        if self.upsample:
            self.execution_action_chunks = max(
                int(round(adopted_action_chunks * float(self.upsample_hz) / float(self.action_hz))),
                1,
            )

        self.action_queue: deque = deque(maxlen=self.execution_action_chunks)
        self._last_original_action_chunk: Optional[np.ndarray] = None
        self._current_exec_index = 0
        self._pending_original_action_payloads_by_exec_index: dict[int, list[dict[str, Any]]] = {}
        self._last_original_action_publish_payloads: list[dict[str, Any]] = []
        self._infer_latencies_s: list[float] = []
        self._chunk_seq = 0

    def _record_infer_timing(self, *, start_s: float, end_s: float) -> None:
        latency = float(end_s - start_s)
        if latency >= 0:
            self._infer_latencies_s.append(latency)

    def log_inference_stats(self) -> None:
        def _summarize(values: list[float]) -> tuple[int, float, float] | None:
            if not values:
                return None
            arr = np.asarray(values, dtype=np.float64)
            return int(arr.size), float(arr.mean()), float(arr.var())

        lat = _summarize(self._infer_latencies_s)

        if lat is None:
            rospy.loginfo("Inference stats: no inference calls recorded.")
            return

        n_lat, mean_lat, var_lat = lat
        rospy.loginfo(
            "Inference latency (infer() only): n=%d mean=%.1fms var=%.3f(ms^2)",
            n_lat,
            mean_lat * 1e3,
            var_lat * 1e6,
        )

    def _delta_to_command(self, *, action: np.ndarray, joint_state: np.ndarray) -> np.ndarray:
        return np.asarray(action, dtype=np.float32).reshape(-1) + np.concatenate(
            [joint_state[:5], np.array([0], dtype=np.float32), joint_state[6:8], np.array([0, 0, 0], dtype=np.float32)]
        )

    def act(self, obs: dict[str, Any]) -> np.ndarray:
        """
        Receive observation data and return a single action.
        obs: Dict[str, Any]
            Observation dictionary
            {
                "head_rgb": <np.ndarray shape (H, W, 3)>,
                "hand_rgb": <np.ndarray shape (H, W, 3)>,
                "joint_state": <np.ndarray shape (8,)>, # ["arm_lift_joint", "arm_flex_joint", "arm_roll_joint", "wrist_flex_joint", "wrist_roll_joint","hand_motor_joint(gripper)", "head_pan_joint", "head_tilt_joint"]
                "instruction": <str>,
            }
        return: np.ndarray : shape (11,)
            Action vector
            [
                "arm_lift_joint",
                "arm_flex_joint",
                "arm_roll_joint",
                "wrist_flex_joint",
                "wrist_roll_joint",
                "gripper",
                "head_pan_joint",
                "head_tilt_joint",
                "base_x",
                "base_y",
                "base_t",
            ]
        """

        if len(self.action_queue) > 0:
            action = self.action_queue.popleft()
            self._last_original_action_publish_payloads = self._pending_original_action_payloads_by_exec_index.pop(
                int(self._current_exec_index),
                [],
            )
            self._current_exec_index += 1
            return self._delta_to_command(action=action, joint_state=obs["joint_state"])
        # Build input dictionary for policy inference.
        policy_input = {
            "head_rgb": obs["head_rgb"],
            "hand_rgb": obs["hand_rgb"],
            "state": obs["joint_state"],
            "prompt": obs["instruction"],
        }
        infer_start_s = time.perf_counter()
        raw_action_chunk = np.asarray(self.policy.infer(policy_input)["actions"], dtype=np.float32)
        infer_end_s = time.perf_counter()
        self._record_infer_timing(start_s=infer_start_s, end_s=infer_end_s)
        self._chunk_seq += 1
        self._last_original_action_chunk = raw_action_chunk[: self.adopted_action_chunks]

        if self.upsample:
            action_chunk = self._last_original_action_chunk
            if self.upsample_method == UPSAMPLE_METHOD_LINEAR:
                action_chunk = _linear_upsample_actions(
                    action_chunk,
                    in_hz=self.action_hz,
                    out_hz=self.upsample_hz,
                    out_steps=self.execution_action_chunks,
                )
            else:
                action_chunk = _cubic_spline_upsample_actions(
                    action_chunk,
                    in_hz=self.action_hz,
                    out_hz=self.upsample_hz,
                    out_steps=self.execution_action_chunks,
                )
        else:
            action_chunk = self._last_original_action_chunk

        matched_exec_indices = _compute_original_step_exec_indices(
            in_steps=int(self._last_original_action_chunk.shape[0]),
            in_hz=float(self.action_hz),
            out_steps=int(action_chunk.shape[0]),
            out_hz=float(self.upsample_hz if self.upsample else self.action_hz),
        )
        execution_hz = float(self.upsample_hz if self.upsample else self.action_hz)
        self.action_queue.extend(np.asarray(action_chunk[1:], dtype=np.float32))
        self._pending_original_action_payloads_by_exec_index = {}
        for original_index, exec_index in enumerate(matched_exec_indices.astype(int).tolist()):
            payload = {
                "chunk_id": int(self._chunk_seq),
                "original_index": int(original_index),
                "matched_exec_index": int(exec_index),
                "action_hz": float(self.action_hz),
                "execution_hz": execution_hz,
                "upsample_enabled": bool(self.upsample),
                "upsample_method": str(self.upsample_method if self.upsample else "original"),
                "original_steps": int(self._last_original_action_chunk.shape[0]),
                "upsampled_steps": int(action_chunk.shape[0]),
                "original_action": self._last_original_action_chunk[original_index].astype(float).tolist(),
            }
            self._pending_original_action_payloads_by_exec_index.setdefault(int(exec_index), []).append(payload)
        self._last_original_action_publish_payloads = self._pending_original_action_payloads_by_exec_index.pop(0, [])
        self._current_exec_index = 1
        return self._delta_to_command(
            action=action_chunk[0],
            joint_state=obs["joint_state"],
        )

    def get_last_original_action_chunk(self) -> Optional[np.ndarray]:
        return self._last_original_action_chunk

    def consume_original_action_publish_payloads(
        self,
        *,
        command_sent: bool,
    ) -> list[dict[str, Any]]:
        payloads = self._last_original_action_publish_payloads
        self._last_original_action_publish_payloads = []
        for payload in payloads:
            payload["command_sent"] = bool(command_sent)
        return payloads


class ExecTraceRecorder:
    def __init__(
        self,
        *,
        enabled: bool,
        config_name: str,
        joint_dim_names: Optional[list[str]] = None,
        base_action_names: Optional[list[str]] = None,
        base_dir: str = "/root/eval_results/traces",
    ):
        self.enabled = bool(enabled)
        self.config_name = str(config_name)
        self.joint_dim_names = list(joint_dim_names or [])
        self.base_action_names = list(base_action_names or [])
        self.base_dir = str(base_dir)

        self._t: list[float] = []
        self._joint_state: list[np.ndarray] = []
        self._action: list[np.ndarray] = []
        self._t_action_original: list[float] = []
        self._action_original_delta: list[np.ndarray] = []
        self._t_chunk_start: list[float] = []

    def add(self, *, stamp_s: float, joint_state: np.ndarray, action: np.ndarray) -> None:
        if not self.enabled:
            return
        self._t.append(float(stamp_s))
        self._joint_state.append(np.asarray(joint_state, dtype=np.float32).reshape(-1))
        self._action.append(np.asarray(action, dtype=np.float32).reshape(-1))

    def add_chunk_start(self, *, stamp_s: float) -> None:
        if not self.enabled:
            return
        self._t_chunk_start.append(float(stamp_s))

    def add_original_action_chunk(self, *, base_stamp_s: float, action_chunk: np.ndarray, action_hz: float) -> None:
        if not self.enabled:
            return
        action_chunk = np.asarray(action_chunk, dtype=np.float32)
        if action_chunk.ndim != 2:
            return
        hz = float(action_hz)
        if hz <= 0:
            return
        dt = 1.0 / hz
        base = float(base_stamp_s)
        for k in range(int(action_chunk.shape[0])):
            self._t_action_original.append(base + k * dt)
            self._action_original_delta.append(action_chunk[k].reshape(-1))

    def _output_dir(self) -> str:
        safe_name = self.config_name.replace("/", "_").replace(os.sep, "_").strip()
        if safe_name == "":
            safe_name = "unknown_config"
        return os.path.join(self.base_dir, safe_name)

    def _next_run_index(self, out_dir: str) -> int:
        try:
            names = os.listdir(out_dir)
        except FileNotFoundError:
            return 1

        max_idx = 0
        for name in names:
            m = re.match(r"^(\d+)\.(npz|png)$", name)
            if m is None:
                continue
            try:
                idx = int(m.group(1))
            except ValueError:
                continue
            max_idx = max(max_idx, idx)
        return max_idx + 1

    def _dim_name(self, dim_idx: int) -> str:
        if 0 <= dim_idx < len(self.joint_dim_names):
            return self.joint_dim_names[dim_idx]
        base_i = dim_idx - len(self.joint_dim_names)
        if 0 <= base_i < len(self.base_action_names):
            return self.base_action_names[base_i]
        return f"dim[{dim_idx}]"

    def save_and_plot(self) -> None:
        if not self.enabled:
            return
        if len(self._t) == 0:
            rospy.logwarn("ExecTraceRecorder: no samples to save.")
            return

        out_dir = self._output_dir()
        os.makedirs(out_dir, exist_ok=True)
        run_idx = self._next_run_index(out_dir)
        stem = f"{run_idx:04d}"

        npz_path = os.path.join(out_dir, f"{stem}.npz")
        plot_path = os.path.join(out_dir, f"{stem}.png")

        payload: dict[str, np.ndarray] = {
            "t": np.asarray(self._t, dtype=np.float64),
            "joint_state": np.stack(self._joint_state, axis=0),
            "action": np.stack(self._action, axis=0),
            "joint_dim_names": np.asarray(self.joint_dim_names, dtype=str),
            "base_action_names": np.asarray(self.base_action_names, dtype=str),
        }
        if len(self._t_chunk_start) > 0:
            payload["t_chunk_start"] = np.asarray(self._t_chunk_start, dtype=np.float64)
        if len(self._t_action_original) > 0 and len(self._action_original_delta) == len(self._t_action_original):
            payload["t_action_original"] = np.asarray(self._t_action_original, dtype=np.float64)
            payload["action_original_delta"] = np.stack(self._action_original_delta, axis=0)

        np.savez_compressed(npz_path, **payload)
        rospy.loginfo("Saved exec trace: %s", npz_path)

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as e:
            rospy.logwarn("ExecTraceRecorder: matplotlib unavailable, skipping plot (%s).", e)
            return

        t = np.asarray(self._t, dtype=np.float64)
        joint_state = np.stack(self._joint_state, axis=0)
        action = np.stack(self._action, axis=0)
        t_chunk_start = np.asarray(self._t_chunk_start, dtype=np.float64) if len(self._t_chunk_start) > 0 else None
        t_action_original = (
            np.asarray(self._t_action_original, dtype=np.float64) if len(self._t_action_original) > 0 else None
        )
        action_original_delta = (
            np.stack(self._action_original_delta, axis=0) if len(self._action_original_delta) > 0 else None
        )

        n_joint = int(joint_state.shape[1]) if joint_state.ndim == 2 else 1
        n_action = int(action.shape[1]) if action.ndim == 2 else 1
        n_action_original = (
            int(action_original_delta.shape[1])
            if action_original_delta is not None and action_original_delta.ndim == 2
            else 0
        )
        nrows = max(max(n_joint, n_action), 1)

        fig_h = max(6.0, 1.1 * float(nrows))
        fig, axes = plt.subplots(nrows, 1, figsize=(14, fig_h), sharex=True)
        if nrows == 1:
            axes = [axes]

        for i in range(nrows):
            ax = axes[i]
            has_joint = i < n_joint
            has_action = i < n_action
            has_action_original = action_original_delta is not None and i < n_action_original
            dim_name = self._dim_name(i)

            if has_joint:
                ax.plot(t, joint_state[:, i], label=f"{dim_name} (joint)")
            if has_action:
                ax.plot(t, action[:, i], label=f"{dim_name} (action)")
            if has_action_original and t_action_original is not None:
                # Convert delta->command using joint_state sampled at the closest previous time.
                idx = np.searchsorted(t, t_action_original, side="right") - 1
                idx = np.clip(idx, 0, max(len(t) - 1, 0))
                js = joint_state[idx]
                original_cmd = np.array(action_original_delta[:, i], copy=True)
                if i < 5:
                    original_cmd = original_cmd + js[:, i]
                elif 6 <= i < 8:
                    original_cmd = original_cmd + js[:, i]
                ax.plot(
                    t_action_original,
                    original_cmd,
                    linestyle="None",
                    marker="+",
                    markersize=4.5,
                    alpha=0.9,
                    label="original action",
                )
            ax.set_ylabel(dim_name)
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.set_title("joint/action (same index overlaid when available)")
            if has_joint or has_action:
                ax.legend(loc="upper right", fontsize=8)

        axes[-1].set_xlabel("time [s]")

        fig.tight_layout()
        fig.savefig(plot_path, dpi=150, format="png")
        plt.close(fig)
        rospy.loginfo("Saved exec trace plot: %s", plot_path)


def main():
    print("Start hsr_policy_client")

    rospy.init_node("hsr_policy_client")

    config_name: str = rospy.get_param("~config_name", "remote_policy")
    policy_server_host: str = rospy.get_param("~policy_server_host", "127.0.0.1")
    policy_server_port: int = int(rospy.get_param("~policy_server_port", 8000))
    policy_server_api_key: Optional[str] = str(rospy.get_param("~policy_server_api_key", ""))
    if policy_server_api_key.strip() == "":
        policy_server_api_key = None
    adopted_action_chunks = rospy.get_param("~adopted_action_chunks", 1)
    update_freq: int = rospy.get_param("~update_freq", 5)
    upsample: bool = rospy.get_param("~upsample", False)
    upsample_hz: int = rospy.get_param("~upsample_hz", 50)
    upsample_method: str = rospy.get_param("~upsample_method", UPSAMPLE_METHOD_SPLINE)
    execution_freq: int = upsample_hz if upsample else update_freq

    action_smoothing: str = rospy.get_param("~action_smoothing", ACTION_SMOOTHING_NONE)
    ema_alpha: float = rospy.get_param("~ema_alpha", 0.2)
    ma_window: int = rospy.get_param("~ma_window", 5)
    smooth_gripper: bool = rospy.get_param("~smooth_gripper", False)
    smooth_base: bool = rospy.get_param("~smooth_base", False)
    test_mode: bool = _param_to_bool(rospy.get_param("~test_mode", True))

    save_exec_trace: bool = rospy.get_param("~save_exec_trace", False)
    trace_base_dir: str = str(rospy.get_param("~trace_base_dir", "/root/eval_results/traces"))
    trace_group_name = _build_trace_group_name(
        config_name=config_name,
        adopted_action_chunks=int(adopted_action_chunks),
        update_freq=int(update_freq),
        upsample=bool(upsample),
        upsample_hz=int(upsample_hz),
        upsample_method=str(upsample_method),
        action_smoothing=str(action_smoothing),
        ema_alpha=float(ema_alpha),
        ma_window=int(ma_window),
        smooth_gripper=bool(smooth_gripper),
        smooth_base=bool(smooth_base),
    )

    rospy.loginfo("config_name: %s", config_name)
    rospy.loginfo("policy_server_host: %s", policy_server_host)
    rospy.loginfo("policy_server_port: %s", policy_server_port)
    rospy.loginfo("policy_server_api_key set: %s", policy_server_api_key is not None)
    rospy.loginfo("adopted_action_chunks: %s", adopted_action_chunks)
    rospy.loginfo("update_freq: %s", update_freq)
    rospy.loginfo("upsample: %s", upsample)
    rospy.loginfo("upsample_hz: %s", upsample_hz)
    rospy.loginfo("upsample_method: %s", upsample_method)
    rospy.loginfo("action_smoothing: %s", action_smoothing)
    rospy.loginfo("ema_alpha: %s", ema_alpha)
    rospy.loginfo("ma_window: %s", ma_window)
    rospy.loginfo("smooth_gripper: %s", smooth_gripper)
    rospy.loginfo("smooth_base: %s", smooth_base)
    rospy.loginfo("test_mode: %s", test_mode)
    rospy.loginfo("execution_freq: %s", execution_freq)
    rospy.loginfo("gripper_mode: %s", rospy.get_param("~gripper_mode", "continuous"))
    rospy.loginfo("save_exec_trace: %s", save_exec_trace)
    rospy.loginfo("trace_base_dir: %s", trace_base_dir)
    rospy.loginfo("exec_trace_group_name: %s", trace_group_name)

    if test_mode:
        env = SyntheticReplayEnv(
            update_freq=execution_freq,
        )
    else:
        env = HSREnv(update_freq=execution_freq)
    policy = OpenpiPolicy(
        policy_server_host=policy_server_host,
        policy_server_port=policy_server_port,
        policy_server_api_key=policy_server_api_key,
        adopted_action_chunks=adopted_action_chunks,
        action_hz=update_freq,
        upsample=upsample,
        upsample_hz=upsample_hz,
        upsample_method=upsample_method,
    )
    debug_publisher = ActionDebugPublisher(
        action_names=list(env.joint_state_names) + list(env.base_action_names),
    )
    # Default smoothing dims: arm(5) + head(2). Optionally add gripper/base.
    base_mask = np.array([True, True, True, True, True, False, True, True, False, False, False], dtype=bool)
    if smooth_gripper:
        base_mask[5] = True
    if smooth_base:
        base_mask[8:11] = True
    action_smoother = ActionSmoother(
        method=action_smoothing,
        ema_alpha=ema_alpha,
        ma_window=ma_window,
        dims_mask=base_mask,
    )
    recorder = ExecTraceRecorder(
        enabled=save_exec_trace,
        config_name=trace_group_name,
        joint_dim_names=env.joint_state_names,
        base_action_names=env.base_action_names,
        base_dir=trace_base_dir,
    )
    rospy.on_shutdown(recorder.save_and_plot)
    rospy.on_shutdown(policy.log_inference_stats)

    log_interval = 1
    if upsample and update_freq > 0:
        log_interval = max(int(round(execution_freq / update_freq)), 1)
    tick = 0
    perf0 = time.perf_counter()
    chunk_gaps_s: list[float] = []
    last_chunk_end_t_s: Optional[float] = None

    def log_chunk_gap_stats() -> None:
        if not chunk_gaps_s:
            rospy.loginfo("Chunk gap stats: no chunk gaps recorded.")
            return
        arr = np.asarray(chunk_gaps_s, dtype=np.float64)
        rospy.loginfo(
            "Chunk gap (prev chunk last action -> next chunk first action): n=%d mean=%.1fms var=%.3f(ms^2)",
            int(arr.size),
            float(arr.mean()) * 1e3,
            float(arr.var()) * 1e6,
        )

    try:
        while not rospy.is_shutdown():
            will_infer = len(policy.action_queue) == 0
            obs = env.get_observations()
            if obs is None:
                can_continue_chunk = (not will_infer) and (env.joint_state is not None) and (upsample or test_mode)
                if can_continue_chunk:
                    obs = {"joint_state": env.joint_state, "instruction": env.instruction}
                else:
                    rospy.loginfo("Observations are not ready.")
                    tick += 1
                    env.sleep()
                    continue

            action = policy.act(obs)
            action_t_s = time.perf_counter() - perf0  # immediately after act()
            action_to_send = action_smoother.update(action)
            is_executed = env.execute_actions(action_to_send)
            command_stamp = rospy.Time.now()
            payloads = policy.consume_original_action_publish_payloads(command_sent=is_executed)
            if is_executed:
                for payload in payloads:
                    debug_publisher.publish_original_action_chunk(
                        payload,
                        publish_stamp=command_stamp,
                    )
            sent_t_s = time.perf_counter() - perf0

            if is_executed:
                if will_infer:
                    recorder.add_chunk_start(stamp_s=sent_t_s)
                    if last_chunk_end_t_s is not None:
                        gap = float(sent_t_s - last_chunk_end_t_s)
                        if gap >= 0:
                            chunk_gaps_s.append(gap)

                # If we just consumed the last action of the current chunk, record chunk end time.
                if (not will_infer) and len(policy.action_queue) == 0:
                    last_chunk_end_t_s = sent_t_s

            if is_executed and "joint_state" in obs:
                recorder.add(
                    stamp_s=sent_t_s,
                    joint_state=obs["joint_state"],
                    action=action_to_send,
                )
                if upsample and will_infer:
                    original_chunk = policy.get_last_original_action_chunk()
                    if original_chunk is not None and original_chunk.ndim == 2:
                        recorder.add_original_action_chunk(
                            base_stamp_s=action_t_s,
                            action_chunk=original_chunk,
                            action_hz=update_freq,
                        )

            # # Debug: dump test images.
            # cv2.imwrite("/root/catkin_ws/head_rgb.png", obs["head_rgb"])
            # cv2.imwrite("/root/catkin_ws/hand_rgb.png", obs["hand_rgb"])
            # # cv2.imshow("hand_rgb", obs["hand_rgb"])
            # break

            if tick % log_interval == 0:
                if is_executed:
                    rospy.loginfo("Action executed.")
                else:
                    reason = getattr(env, "last_execute_block_reason", None)
                    if reason:
                        rospy.loginfo("Action not executed. reason=%s", reason)
                    else:
                        rospy.loginfo("Action not executed.")
                rospy.loginfo("Language instruction: %s", obs.get("instruction", ""))
                rospy.loginfo("Action: %s", action)
            if not test_mode:
                if not upsample:
                    env.reset_observation()
                elif will_infer:
                    env.reset_observation(reset_joint_state=False)
            tick += 1
            env.sleep()
    except KeyboardInterrupt:
        # Ensure recorder flush even if ROS shutdown hook doesn't run in time.
        rospy.loginfo("KeyboardInterrupt received. Saving exec trace and shutting down.")
        recorder.save_and_plot()
        policy.log_inference_stats()
        log_chunk_gap_stats()
        try:
            rospy.signal_shutdown("KeyboardInterrupt")
        except Exception:
            pass
        return


if __name__ == "__main__":
    main()
