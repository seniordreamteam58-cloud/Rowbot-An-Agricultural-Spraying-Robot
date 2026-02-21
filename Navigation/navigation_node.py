import os
import sys
import math
import yaml
import threading
from typing import Optional, List, Callable, Tuple

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.duration import Duration

from geometry_msgs.msg import PointStamped, PoseStamped, Quaternion
from std_srvs.srv import Trigger
from std_msgs.msg import Int32, Float32
from tf2_ros import Buffer, TransformListener


def yaw_to_quat_msg(yaw: float) -> Quaternion:
    q = Quaternion()
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(yaw * 0.5)
    q.w = math.cos(yaw * 0.5)
    return q


def yaw_from_to(x0: float, y0: float, x1: float, y1: float) -> float:
    return math.atan2(y1 - y0, x1 - x0)


def dist_xy(ax: float, ay: float, bx: float, by: float) -> float:
    return math.hypot(bx - ax, by - ay)


def straight_line_dist(a: PoseStamped, b: PoseStamped) -> float:
    return dist_xy(a.pose.position.x, a.pose.position.y, b.pose.position.x, b.pose.position.y)


def interp_pose(frame_id: str, a: PoseStamped, b: PoseStamped, t: float) -> PoseStamped:
    p = PoseStamped()
    p.header.frame_id = frame_id
    p.pose.position.x = a.pose.position.x + (b.pose.position.x - a.pose.position.x) * t
    p.pose.position.y = a.pose.position.y + (b.pose.position.y - a.pose.position.y) * t
    p.pose.position.z = a.pose.position.z + (b.pose.position.z - a.pose.position.z) * t
    p.pose.orientation = yaw_to_quat_msg(0.0)
    return p


def set_orientations_to_approach_heading(start_pose: Optional[PoseStamped], poses: List[PoseStamped]) -> None:
    if not poses:
        return

    if start_pose is not None:
        x0 = start_pose.pose.position.x
        y0 = start_pose.pose.position.y
        x1 = poses[0].pose.position.x
        y1 = poses[0].pose.position.y
        poses[0].pose.orientation = yaw_to_quat_msg(yaw_from_to(x0, y0, x1, y1))

    for i in range(1, len(poses)):
        a = poses[i - 1].pose.position
        b = poses[i].pose.position
        poses[i].pose.orientation = yaw_to_quat_msg(yaw_from_to(a.x, a.y, b.x, b.y))


def build_pause_segments(
    frame_id: str,
    start_pose: PoseStamped,
    waypoints: List[PoseStamped],
    step_m: float
) -> List[Tuple[List[PoseStamped], bool]]:
    """
    Returns list of (segment_poses, pause_after).
    Pause only after inserted step points, not at original waypoints.
    """
    if step_m <= 0.0 or not waypoints:
        return [(waypoints[:], False)]

    poly: List[PoseStamped] = [start_pose] + [w for w in waypoints]
    segments: List[Tuple[List[PoseStamped], bool]] = []
    cur_segment: List[PoseStamped] = []

    traveled_since_pause = 0.0
    remaining_to_next_pause = step_m

    for i in range(len(poly) - 1):
        A = poly[i]
        B = poly[i + 1]
        ax, ay = A.pose.position.x, A.pose.position.y
        bx, by = B.pose.position.x, B.pose.position.y
        seg_len = dist_xy(ax, ay, bx, by)
        if seg_len < 1e-6:
            continue

        seg_progress = 0.0

        while seg_progress + 1e-9 < seg_len:
            left_in_seg = seg_len - seg_progress

            if left_in_seg >= remaining_to_next_pause - 1e-9:
                t = (seg_progress + remaining_to_next_pause) / seg_len
                pause_pose = interp_pose(frame_id, A, B, t)

                cur_segment.append(pause_pose)
                segments.append((cur_segment, True))

                cur_segment = []
                traveled_since_pause = 0.0
                used = remaining_to_next_pause
                remaining_to_next_pause = step_m

                seg_progress = seg_progress + used
                if seg_progress > seg_len:
                    seg_progress = seg_len

                A = pause_pose
                ax, ay = A.pose.position.x, A.pose.position.y
                seg_len = dist_xy(ax, ay, bx, by)
                seg_progress = 0.0
                if seg_len < 1e-6:
                    break

                continue

            break

        cur_segment.append(B)

        traveled = seg_len - seg_progress if seg_progress < seg_len else seg_len
        traveled_since_pause += traveled
        remaining_to_next_pause = max(1e-6, step_m - traveled_since_pause)

        if remaining_to_next_pause > step_m:
            remaining_to_next_pause = step_m
            traveled_since_pause = 0.0

    if cur_segment:
        segments.append((cur_segment, False))

    if segments and len(segments) == 1 and segments[0][1] is True:
        segments[0] = (segments[0][0], False)

    return segments


class ClickWaypointsToNavUnified(Node):
    def __init__(self):
        super().__init__("click_waypoints_to_nav_unified")

        # Mode
        self.declare_parameter("mode", "weeds")  # "weeds" or "crops"
        self.mode = self.get_parameter("mode").get_parameter_value().string_value.lower().strip()
        if self.mode not in ("weeds", "crops"):
            self.get_logger().warn("Invalid mode, defaulting to weeds.")
            self.mode = "weeds"

        # Common params
        self.declare_parameter("frame_id", "map")
        self.declare_parameter("save_file", "/tmp/clicked_waypoints.yaml")
        self.declare_parameter("skip_radius", 0.0)
        self.declare_parameter("tf_timeout_sec", 1.0)

        # Crops params
        self.declare_parameter("action_name_crops", "follow_waypoints")
        self.declare_parameter("cruise_speed", 0.10)  # ETA only

        # Weeds params
        self.declare_parameter("action_name_weeds", "follow_waypoints")
        self.declare_parameter("step_mode", True)
        self.declare_parameter("step_distance_m", 0.60)

        # Weeds pub/sub
        self.declare_parameter("spray_status_topic", "/spray_status")
        self.declare_parameter("robot_motion_topic", "/robot_motion_state")
        self.declare_parameter("snapshot_window_sec", 0.40)
        self.declare_parameter("max_spray_wait_sec", 20.0)
        self.declare_parameter("motion_pub_hz", 10.0)

        # Battery monitoring
        self.declare_parameter("voltage_topic", "/voltage")
        self.declare_parameter("low_voltage_threshold", 10.5)
        self.declare_parameter("rearm_voltage_threshold", 10.8)  # must be >= low threshold
        self.declare_parameter("low_voltage_hold_sec", 0.5)      # debounce

        self.frame_id = self.get_parameter("frame_id").get_parameter_value().string_value
        self.save_file = self.get_parameter("save_file").get_parameter_value().string_value
        self.skip_radius = float(self.get_parameter("skip_radius").value)
        self.tf_timeout = float(self.get_parameter("tf_timeout_sec").value)

        self.action_name_crops = self.get_parameter("action_name_crops").get_parameter_value().string_value
        self.action_name_weeds = self.get_parameter("action_name_weeds").get_parameter_value().string_value
        self.cruise_speed = float(self.get_parameter("cruise_speed").value)

        self.step_mode = bool(self.get_parameter("step_mode").value)
        self.step_distance_m = float(self.get_parameter("step_distance_m").value)

        self.spray_status_topic = self.get_parameter("spray_status_topic").get_parameter_value().string_value
        self.robot_motion_topic = self.get_parameter("robot_motion_topic").get_parameter_value().string_value
        self.snapshot_window_sec = float(self.get_parameter("snapshot_window_sec").value)
        self.max_spray_wait_sec = float(self.get_parameter("max_spray_wait_sec").value)
        self.motion_pub_hz = float(self.get_parameter("motion_pub_hz").value)

        self.voltage_topic = self.get_parameter("voltage_topic").get_parameter_value().string_value
        self.low_voltage_threshold = float(self.get_parameter("low_voltage_threshold").value)
        self.rearm_voltage_threshold = float(self.get_parameter("rearm_voltage_threshold").value)
        self.low_voltage_hold_sec = float(self.get_parameter("low_voltage_hold_sec").value)

        if self.rearm_voltage_threshold < self.low_voltage_threshold:
            self.rearm_voltage_threshold = self.low_voltage_threshold

        # Clicked waypoints
        self.poses: List[PoseStamped] = []

        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # RViz click subscriber
        self.sub = self.create_subscription(PointStamped, "/clicked_point", self.on_click, 10)

        # Services (still available)
        self.commit_srv = self.create_service(Trigger, "commit_waypoints", self.on_commit)
        self.clear_srv = self.create_service(Trigger, "clear_waypoints", self.on_clear)
        self.undo_srv = self.create_service(Trigger, "delete_last_waypoint", self.on_delete_last)
        self.save_srv = self.create_service(Trigger, "save_waypoints", self.on_save)
        self.load_srv = self.create_service(Trigger, "load_waypoints", self.on_load)

        # Terminal UI
        self._cmd_thread = threading.Thread(target=self._terminal_loop, daemon=True)
        self._cmd_thread.start()

        # Shared lock
        self._lock = threading.Lock()

        # Mission state (shared)
        self._start_pose: Optional[PoseStamped] = None
        self._committed_base_poses: List[PoseStamped] = []
        self._abort_pose: Optional[PoseStamped] = None
        self._remaining_after_abort: List[PoseStamped] = []
        self._low_in_progress: bool = False

        # Crops state
        self._crops_client = None
        self._crops_goal_handle = None
        self._last_feedback_idx: int = 0
        self._oneshot_timers = []

        # Weeds state
        self._weeds_client = None
        self._weeds_goal_handle = None
        self._run_active = False
        self._segments: List[Tuple[List[PoseStamped], bool]] = []
        self._seg_idx = 0
        self._pause_after_current = False
        self._wait_timer = None
        self._wait_start_time = None
        self._saw_spraying = False
        self._battery_override_active = False

        # Weeds I/O
        self._spray_status = 0
        self._spray_sub = self.create_subscription(Int32, self.spray_status_topic, self._on_spray_status, 10)
        self._motion_pub = self.create_publisher(Int32, self.robot_motion_topic, 10)
        self._motion_state = 1
        pub_period = 1.0 / max(self.motion_pub_hz, 1e-3)
        self._motion_timer = self.create_timer(pub_period, self._publish_motion_state)

        # Battery I/O (assumes std_msgs/Float32 on /voltage)
        self._current_voltage: Optional[float] = None
        self._battery_low_active = False
        self._low_detect_start: Optional[float] = None
        self._voltage_sub = self.create_subscription(Float32, self.voltage_topic, self._on_voltage, 10)

        # Create action client(s)
        from nav2_msgs.action import FollowWaypoints
        self._FollowWaypoints = FollowWaypoints
        self._crops_client = ActionClient(self, FollowWaypoints, self.action_name_crops)
      
        self._set_motion_state(stopped=True)

        self.get_logger().info("RViz: use Publish Point (/clicked_point).")
        self.get_logger().info(f"Mode: {self.mode}")
        self.get_logger().info("Terminal: help | commit | save | load | undo | clear | continue | quit")
        self.get_logger().info(f"Battery: sub {self.voltage_topic}, low < {self.low_voltage_threshold:.2f} V")
        self.get_logger().info(f"Battery: rearm >= {self.rearm_voltage_threshold:.2f} V, hold {self.low_voltage_hold_sec:.2f} s")

        if self.mode == "crops":
            self.get_logger().info(f"Action: {self.action_name_crops}")
        else:
            self.get_logger().info(f"Action: {self.action_name_weeds}")
            self.get_logger().info(f"Sub: {self.spray_status_topic}  Pub: {self.robot_motion_topic}")
            if self.step_mode:
                self.get_logger().info(f"Step mode: pause every {self.step_distance_m:.2f} m.")

    # Terminal UI
    def _terminal_loop(self):
        while rclpy.ok():
            try:
                line = sys.stdin.readline()
                if not line:
                    continue
                cmd = line.strip().lower()
                if not cmd:
                    continue
                self._handle_cmd(cmd)
            except Exception as ex:
                self.get_logger().error(f"Terminal input error: {ex}")

    def _handle_cmd(self, cmd: str) -> None:
        if cmd in ("help", "?"):
            self.get_logger().info("Commands: commit | save | load | undo | clear | continue | quit")
            return

        if cmd in ("quit", "exit"):
            self.get_logger().info("Exiting node...")
            rclpy.shutdown()
            return

        if cmd == "commit":
            self._call_service_handler(self.on_commit)
            return

        if cmd == "save":
            self._call_service_handler(self.on_save)
            return

        if cmd == "load":
            self._call_service_handler(self.on_load)
            return

        if cmd in ("undo", "delete", "back"):
            self._call_service_handler(self.on_delete_last)
            return

        if cmd in ("reset", "clear"):
            self._call_service_handler(self.on_clear)
            return

        if cmd in ("continue", "resume"):
            self._do_continue()
            return

        self.get_logger().warn(f"Unknown command '{cmd}'. Type 'help'.")

    def _call_service_handler(self, handler_fn):
        req = Trigger.Request()
        res = Trigger.Response()
        handler_fn(req, res)
        if res.message:
            self.get_logger().info(res.message)

    # Battery monitoring
    def _on_voltage(self, msg: Float32):
        v = float(msg.data)
        self._current_voltage = v

        now = self.get_clock().now().nanoseconds * 1e-9

        if self._battery_low_active:
            if v >= self.rearm_voltage_threshold:
                self._battery_low_active = False
                self._low_detect_start = None
                self.get_logger().info(f"Battery rearmed: {v:.2f} V")
            return

        if v < self.low_voltage_threshold:
            if self._low_detect_start is None:
                self._low_detect_start = now
                return

            if (now - self._low_detect_start) >= self.low_voltage_hold_sec:
                self._battery_low_active = True
                self._low_detect_start = None
                self.get_logger().warn(f"Battery LOW: {v:.2f} V < {self.low_voltage_threshold:.2f} V")
                self._do_low()
            return

        self._low_detect_start = None

    # ROS I/O (weeds)
    def _on_spray_status(self, msg: Int32):
        self._spray_status = int(msg.data)

    def _publish_motion_state(self):
        m = Int32()
        m.data = int(self._motion_state)
        self._motion_pub.publish(m)

    def _set_motion_state(self, stopped: bool):
        self._motion_state = 1 if stopped else 0
        self._publish_motion_state()

    # TF
    def _get_robot_pose(self) -> Optional[PoseStamped]:
        try:
            tf = self.tf_buffer.lookup_transform(
                self.frame_id,
                "base_link",
                rclpy.time.Time(),
                timeout=Duration(seconds=self.tf_timeout),
            )
            ps = PoseStamped()
            ps.header.frame_id = self.frame_id
            ps.pose.position.x = tf.transform.translation.x
            ps.pose.position.y = tf.transform.translation.y
            ps.pose.position.z = tf.transform.translation.z
            ps.pose.orientation = tf.transform.rotation
            return ps
        except Exception as ex:
            self.get_logger().warn(f"TF lookup failed ({self.frame_id} <- base_link): {ex}")
            return None

    # Waypoint processing
    def _apply_skip_radius(self, poses: List[PoseStamped]) -> List[PoseStamped]:
        if self.skip_radius <= 0.0:
            return poses
        rp = self._get_robot_pose()
        if rp is None:
            return poses
        kept: List[PoseStamped] = []
        for p in poses:
            d = dist_xy(rp.pose.position.x, rp.pose.position.y, p.pose.position.x, p.pose.position.y)
            if d > self.skip_radius:
                kept.append(p)
        skipped = len(poses) - len(kept)
        if skipped > 0:
            self.get_logger().info(f"Skipped {skipped} waypoint(s) within {self.skip_radius:.2f} m of robot.")
        return kept

    def _straight_line_eta(self, poses: List[PoseStamped]) -> float:
        if not poses:
            return 0.0
        speed = self.cruise_speed if self.cruise_speed > 0.01 else 0.30
        start_pose = self._get_robot_pose()
        if start_pose is None:
            start_pose = poses[0]
        total = 0.0
        cur = start_pose
        for p in poses:
            total += straight_line_dist(cur, p)
            cur = p
        return total / speed

    def _compute_remaining_from_abort(self, abort_pose: PoseStamped, base_poses: List[PoseStamped]) -> List[PoseStamped]:
        if not base_poses:
            return []
        ax, ay = abort_pose.pose.position.x, abort_pose.pose.position.y
        best_i = 0
        best_d = 1e18
        for i, p in enumerate(base_poses):
            d = dist_xy(ax, ay, p.pose.position.x, p.pose.position.y)
            if d < best_d:
                best_d = d
                best_i = i
        return base_poses[best_i:]

    # RViz click
    def on_click(self, msg: PointStamped):
        p = PoseStamped()
        p.header.frame_id = self.frame_id
        p.pose.position.x = float(msg.point.x)
        p.pose.position.y = float(msg.point.y)
        p.pose.position.z = float(msg.point.z)
        p.pose.orientation = yaw_to_quat_msg(0.0)
        self.poses.append(p)
        self.get_logger().info(f"Added waypoint #{len(self.poses)}: ({p.pose.position.x:.2f}, {p.pose.position.y:.2f})")

    # Services
    def on_commit(self, req, res):
        if not self.poses:
            res.success = False
            res.message = "No waypoints."
            return res

        start_pose = self._get_robot_pose()
        if start_pose is None:
            res.success = False
            res.message = "TF not available for robot pose."
            return res

        final_poses = self._apply_skip_radius(self.poses[:])
        if not final_poses:
            res.success = False
            res.message = "All waypoints were skipped."
            return res

        with self._lock:
            self._start_pose = start_pose
            self._committed_base_poses = final_poses[:]
            self._abort_pose = None
            self._remaining_after_abort = []
            self._low_in_progress = False

        if self.mode == "crops":
            return self._commit_crops(start_pose, final_poses, res)

        return self._commit_weeds(start_pose, final_poses, res)

    def _commit_crops(self, start_pose: PoseStamped, final_poses: List[PoseStamped], res: Trigger.Response):
        if self._crops_client is None:
            res.success = False
            res.message = "Crops client not initialized."
            return res

        if not self._crops_client.wait_for_server(timeout_sec=3.0):
            res.success = False
            res.message = f"Action server '{self.action_name_crops}' not available."
            return res

        set_orientations_to_approach_heading(start_pose, final_poses)

        with self._lock:
            self._last_feedback_idx = 0

        eta_sec = self._straight_line_eta(final_poses)
        self.get_logger().info(f"Straight-line ETA: {eta_sec:.1f} s ({eta_sec/60.0:.2f} min)")

        self._send_crops_goal(final_poses, label="Navigate", feedback_cb=self._crops_feedback_cb)

        res.success = True
        res.message = f"Sent {len(final_poses)} waypoint(s). Start pose recorded."
        return res

    def _commit_weeds(self, start_pose: PoseStamped, final_poses: List[PoseStamped], res: Trigger.Response):
        if self._weeds_client is None:
            res.success = False
            res.message = "Weeds client not initialized."
            return res

        if not self._weeds_client.wait_for_server(timeout_sec=3.0):
            res.success = False
            res.message = f"Action server '{self.action_name_weeds}' not available."
            return res

        self._build_weeds_segments(start_pose, final_poses)

        self._run_active = True
        self._seg_idx = 0
        self._battery_override_active = False

        self._set_motion_state(stopped=False)
        self._send_next_weeds_segment()

        res.success = True
        res.message = f"Started weeds navigation with {len(self._segments)} segment(s)."
        return res

    def _build_weeds_segments(self, start_pose: PoseStamped, base_poses: List[PoseStamped]) -> None:
        if self.step_mode:
            self._segments = build_pause_segments(self.frame_id, start_pose, base_poses, self.step_distance_m)
        else:
            self._segments = [(base_poses, False)]

        seg_start = start_pose
        for seg_poses, _ in self._segments:
            set_orientations_to_approach_heading(seg_start, seg_poses)
            if seg_poses:
                seg_start = seg_poses[-1]

    def on_clear(self, req, res):
        self.poses.clear()
        res.success = True
        res.message = "Cleared all waypoints."
        return res

    def on_delete_last(self, req, res):
        if not self.poses:
            res.success = False
            res.message = "No waypoints to delete."
            return res
        self.poses.pop()
        res.success = True
        res.message = f"Deleted last waypoint. Remaining: {len(self.poses)}"
        return res

    def on_save(self, req, res):
        try:
            d = os.path.dirname(self.save_file)
            if d:
                os.makedirs(d, exist_ok=True)
            data = {
                "frame_id": self.frame_id,
                "poses": [
                    {
                        "x": float(p.pose.position.x),
                        "y": float(p.pose.position.y),
                        "z": float(p.pose.position.z),
                        "qx": float(p.pose.orientation.x),
                        "qy": float(p.pose.orientation.y),
                        "qz": float(p.pose.orientation.z),
                        "qw": float(p.pose.orientation.w),
                    }
                    for p in self.poses
                ],
            }
            with open(self.save_file, "w") as f:
                yaml.safe_dump(data, f, sort_keys=False)
            res.success = True
            res.message = f"Saved {len(self.poses)} waypoint(s) to {self.save_file}"
            return res
        except Exception as ex:
            res.success = False
            res.message = f"Save failed: {ex}"
            return res

    def on_load(self, req, res):
        try:
            if not os.path.exists(self.save_file):
                res.success = False
                res.message = f"File not found: {self.save_file}"
                return res
            with open(self.save_file, "r") as f:
                data = yaml.safe_load(f) or {}
            raw = data.get("poses", [])
            poses: List[PoseStamped] = []
            for e in raw:
                p = PoseStamped()
                p.header.frame_id = self.frame_id
                p.pose.position.x = float(e["x"])
                p.pose.position.y = float(e["y"])
                p.pose.position.z = float(e.get("z", 0.0))
                p.pose.orientation.x = float(e.get("qx", 0.0))
                p.pose.orientation.y = float(e.get("qy", 0.0))
                p.pose.orientation.z = float(e.get("qz", 0.0))
                p.pose.orientation.w = float(e.get("qw", 1.0))
                poses.append(p)
            self.poses = poses
            res.success = True
            res.message = f"Loaded {len(self.poses)} waypoint(s) from {self.save_file}"
            return res
        except Exception as ex:
            res.success = False
            res.message = f"Load failed: {ex}"
            return res


    # Low battery + resume (both modes)
    def _do_low(self) -> None:
        with self._lock:
            if self._low_in_progress:
                return
            if not self._committed_base_poses or self._start_pose is None:
                self.get_logger().warn("Low: no committed mission. Type 'commit' first.")
                return
            self._low_in_progress = True

        abort = self._get_robot_pose()
        if abort is None:
            self.get_logger().warn("Low: cannot read robot pose.")
            with self._lock:
                self._low_in_progress = False
            return

        with self._lock:
            base = self._committed_base_poses[:]
            self._abort_pose = abort
            self._remaining_after_abort = self._compute_remaining_from_abort(abort, base)

        self.get_logger().warn(
            f"Battery low: saved abort pose. Remaining waypoints: {len(self._remaining_after_abort)}. Returning to start."
        )

        if self.mode == "crops":
            self._low_crops_flow()
        else:
            self._low_weeds_flow()

    def _do_continue(self) -> None:
        with self._lock:
            if self._low_in_progress:
                self.get_logger().warn("Continue: wait until return-to-start completes.")
                return
            ap = self._abort_pose
            rem = self._remaining_after_abort[:]

        if ap is None:
            self.get_logger().warn("Continue: no abort pose saved.")
            return

        if not rem:
            self.get_logger().info("Continue: nothing to resume.")
            return

        if self._current_voltage is not None and self._current_voltage < self.rearm_voltage_threshold:
            self.get_logger().warn(
                f"Continue blocked: voltage {self._current_voltage:.2f} V < {self.rearm_voltage_threshold:.2f} V"
            )
            return

        self.get_logger().info(f"Continue: go to abort pose then resume {len(rem)} waypoint(s).")

        if self.mode == "crops":
            self._continue_crops_flow(ap, rem)
        else:
            self._continue_weeds_flow(ap, rem)

    # Crops low/continue
    def _low_crops_flow(self) -> None:
        def _after_cancel():
            self._return_to_start_crops_retry(attempt=1)

        self._cancel_crops_goal_async(_after_cancel)

    def _continue_crops_flow(self, abort_pose: PoseStamped, remaining: List[PoseStamped]) -> None:
        def _after_cancel():
            def _after_abort(ok: bool):
                if not ok:
                    self.get_logger().error("GoToAbortPose failed.")
                    return
                self._send_crops_goal(remaining, "ResumeRemaining", feedback_cb=self._crops_feedback_cb)

            self._send_crops_goal([abort_pose], "GoToAbortPose", on_done=_after_abort)

        self._cancel_crops_goal_async(_after_cancel)

    # Weeds low/continue
    def _low_weeds_flow(self) -> None:
        self._battery_override_active = True
        self._stop_wait_timer()
        self._run_active = False

        def _after_cancel():
            self._return_to_start_weeds_retry(attempt=1)

        self._cancel_weeds_goal_async(_after_cancel)

    def _continue_weeds_flow(self, abort_pose: PoseStamped, remaining: List[PoseStamped]) -> None:
        self._battery_override_active = True
        self._stop_wait_timer()
        self._run_active = False

        def _after_cancel():
            def _after_abort(ok: bool):
                if not ok:
                    self.get_logger().error("GoToAbortPose failed.")
                    self._battery_override_active = False
                    return

                start_pose = abort_pose
                self._build_weeds_segments(start_pose, remaining)
                self._run_active = True
                self._seg_idx = 0
                self._battery_override_active = False
                self._set_motion_state(stopped=False)
                self._send_next_weeds_segment()

            self._send_weeds_single_pose(abort_pose, label="GoToAbortPose", on_done=_after_abort)

        self._cancel_weeds_goal_async(_after_cancel)

    # Crops action helpers
    def _crops_feedback_cb(self, msg):
        try:
            idx = int(msg.feedback.current_waypoint)
        except Exception:
            return
        with self._lock:
            self._last_feedback_idx = idx

    def _send_crops_goal(
        self,
        poses: List[PoseStamped],
        label: str,
        on_done: Optional[Callable[[bool], None]] = None,
        feedback_cb=None,
    ) -> None:
        if self._crops_client is None:
            self.get_logger().error(f"{label}: crops client missing.")
            if on_done:
                on_done(False)
            return

        if not poses:
            self.get_logger().warn(f"{label}: empty poses.")
            if on_done:
                on_done(False)
            return

        if not self._crops_client.wait_for_server(timeout_sec=3.0):
            self.get_logger().error(f"{label}: action server '{self.action_name_crops}' not available.")
            if on_done:
                on_done(False)
            return

        goal = self._FollowWaypoints.Goal()
        goal.poses = poses

        self.get_logger().info(f"{label}: sending {len(poses)} pose(s)")
        send_fut = self._crops_client.send_goal_async(goal, feedback_callback=feedback_cb)

        def _on_sent(f):
            gh = f.result()
            if not gh or not gh.accepted:
                self.get_logger().error(f"{label}: rejected.")
                if on_done:
                    on_done(False)
                return

            with self._lock:
                self._crops_goal_handle = gh

            self.get_logger().info(f"{label}: accepted.")
            res_fut = gh.get_result_async()

            def _on_res(_):
                self.get_logger().info(f"{label}: finished.")
                with self._lock:
                    self._crops_goal_handle = None
                if on_done:
                    on_done(True)

            res_fut.add_done_callback(_on_res)

        send_fut.add_done_callback(_on_sent)

    def _cancel_crops_goal_async(self, after_cancel: Callable[[], None]) -> None:
        with self._lock:
            gh = self._crops_goal_handle

        if gh is None:
            after_cancel()
            return

        try:
            self.get_logger().info("Cancel: requesting crops goal cancel.")
            cancel_fut = gh.cancel_goal_async()

            def _on_cancel(_):
                self.get_logger().info("Cancel: crops processed.")
                with self._lock:
                    self._crops_goal_handle = None
                after_cancel()

            cancel_fut.add_done_callback(_on_cancel)
        except Exception as ex:
            self.get_logger().warn(f"Cancel crops failed: {ex}")
            with self._lock:
                self._crops_goal_handle = None
            after_cancel()

    def _oneshot(self, delay_sec: float, fn: Callable[[], None]) -> None:
        holder = {"t": None}

        def _cb():
            t = holder["t"]
            if t is not None:
                t.cancel()
            fn()

        holder["t"] = self.create_timer(delay_sec, _cb)
        self._oneshot_timers.append(holder["t"])

    def _return_to_start_crops_retry(self, attempt: int) -> None:
        with self._lock:
            sp = self._start_pose

        if sp is None:
            self.get_logger().error("ReturnToStart: missing start pose.")
            with self._lock:
                self._low_in_progress = False
            return

        max_attempts = 6
        self.get_logger().info(f"ReturnToStart (crops): attempt {attempt}/{max_attempts}")

        def _done(ok: bool):
            if ok:
                self.get_logger().info("ReturnToStart (crops): completed.")
                with self._lock:
                    self._low_in_progress = False
                return

            if attempt >= max_attempts:
                self.get_logger().error("ReturnToStart (crops): failed after retries.")
                with self._lock:
                    self._low_in_progress = False
                return

            self.get_logger().warn("ReturnToStart (crops): retrying...")
            self._oneshot(0.35, lambda: self._return_to_start_crops_retry(attempt + 1))

        self._send_crops_goal([sp], "ReturnToStart", on_done=_done)

    # Weeds action helpers
    def _send_weeds_single_pose(self, pose: PoseStamped, label: str, on_done: Callable[[bool], None]) -> None:
        if self._weeds_client is None:
            self.get_logger().error(f"{label}: weeds client missing.")
            on_done(False)
            return

        if not self._weeds_client.wait_for_server(timeout_sec=3.0):
            self.get_logger().error(f"{label}: action server '{self.action_name_weeds}' not available.")
            on_done(False)
            return

        goal = self._FollowWaypoints.Goal()
        goal.poses = [pose]

        self.get_logger().info(f"{label}: sending 1 pose")
        self._set_motion_state(stopped=False)

        fut = self._weeds_client.send_goal_async(goal)

        def _on_sent(f):
            gh = f.result()
            if not gh or not gh.accepted:
                self.get_logger().error(f"{label}: rejected.")
                on_done(False)
                return

            self._weeds_goal_handle = gh
            rf = gh.get_result_async()

            def _on_res(_):
                self.get_logger().info(f"{label}: finished.")
                self._weeds_goal_handle = None
                self._set_motion_state(stopped=True)
                on_done(True)

            rf.add_done_callback(_on_res)

        fut.add_done_callback(_on_sent)

    def _cancel_weeds_goal_async(self, after_cancel: Callable[[], None]) -> None:
        gh = self._weeds_goal_handle
        if gh is None:
            after_cancel()
            return

        try:
            self.get_logger().info("Cancel: requesting weeds goal cancel.")
            cancel_fut = gh.cancel_goal_async()

            def _on_cancel(_):
                self.get_logger().info("Cancel: weeds processed.")
                self._weeds_goal_handle = None
                after_cancel()

            cancel_fut.add_done_callback(_on_cancel)
        except Exception as ex:
            self.get_logger().warn(f"Cancel weeds failed: {ex}")
            self._weeds_goal_handle = None
            after_cancel()

    def _return_to_start_weeds_retry(self, attempt: int) -> None:
        with self._lock:
            sp = self._start_pose

        if sp is None:
            self.get_logger().error("ReturnToStart (weeds): missing start pose.")
            with self._lock:
                self._low_in_progress = False
            self._battery_override_active = False
            return

        max_attempts = 6
        self.get_logger().info(f"ReturnToStart (weeds): attempt {attempt}/{max_attempts}")

        def _done(ok: bool):
            if ok:
                self.get_logger().info("ReturnToStart (weeds): completed.")
                with self._lock:
                    self._low_in_progress = False
                self._battery_override_active = False
                return

            if attempt >= max_attempts:
                self.get_logger().error("ReturnToStart (weeds): failed after retries.")
                with self._lock:
                    self._low_in_progress = False
                self._battery_override_active = False
                return

            self.get_logger().warn("ReturnToStart (weeds): retrying...")
            self._oneshot(0.35, lambda: self._return_to_start_weeds_retry(attempt + 1))

        self._send_weeds_single_pose(sp, label="ReturnToStart", on_done=_done)

    # Weeds segment execution
    def _send_next_weeds_segment(self):
        if not self._run_active:
            return
        if self._seg_idx >= len(self._segments):
            self.get_logger().info("Weeds navigation finished.")
            self._run_active = False
            self._set_motion_state(stopped=True)
            return

        seg_poses, pause_after = self._segments[self._seg_idx]
        self._seg_idx += 1
        self._pause_after_current = pause_after

        goal = self._FollowWaypoints.Goal()
        goal.poses = seg_poses

        self.get_logger().info(
            f"Sending segment {self._seg_idx}/{len(self._segments)} with {len(seg_poses)} pose(s), pause_after={pause_after}"
        )

        self._set_motion_state(stopped=False)

        fut = self._weeds_client.send_goal_async(goal)
        fut.add_done_callback(self._on_weeds_segment_sent)

    def _on_weeds_segment_sent(self, fut):
        gh = fut.result()
        if not gh or not gh.accepted:
            self.get_logger().error("Segment goal rejected.")
            self._run_active = False
            self._set_motion_state(stopped=True)
            return

        self._weeds_goal_handle = gh
        rf = gh.get_result_async()
        rf.add_done_callback(self._on_weeds_segment_done)

    def _on_weeds_segment_done(self, _):
        if self._battery_override_active:
            return

        if not self._run_active:
            return

        self._weeds_goal_handle = None
        self.get_logger().info("Segment reached.")

        if self._pause_after_current:
            self._set_motion_state(stopped=True)
            self._start_wait_for_spraying_cycle()
            return

        self._send_next_weeds_segment()

    def _start_wait_for_spraying_cycle(self):
        if self._battery_override_active:
            return

        now = self.get_clock().now().nanoseconds * 1e-9
        self._wait_start_time = now
        self._saw_spraying = False

        self._stop_wait_timer()
        self._wait_timer = self.create_timer(0.1, self._wait_tick)
        self.get_logger().info("Stopped: waiting for sprayer cycle (spray_status 1 then 0).")

    def _wait_tick(self):
        if self._battery_override_active:
            self._stop_wait_timer()
            return

        if not self._run_active:
            self._stop_wait_timer()
            return

        now = self.get_clock().now().nanoseconds * 1e-9
        elapsed = now - (self._wait_start_time or now)

        if self._spray_status == 1:
            self._saw_spraying = True

        if self._saw_spraying:
            if self._spray_status == 0:
                self.get_logger().info("Sprayer finished (status 0). Continuing.")
                self._stop_wait_timer()
                self._set_motion_state(stopped=False)
                self._send_next_weeds_segment()
                return

            if elapsed > self.max_spray_wait_sec:
                self.get_logger().warn("Timed out waiting for sprayer. Continuing anyway.")
                self._stop_wait_timer()
                self._set_motion_state(stopped=False)
                self._send_next_weeds_segment()
                return

            return

        if elapsed >= self.snapshot_window_sec and self._spray_status == 0:
            self.get_logger().info("No spraying detected (status stayed 0). Continuing.")
            self._stop_wait_timer()
            self._set_motion_state(stopped=False)
            self._send_next_weeds_segment()
            return

    def _stop_wait_timer(self):
        if self._wait_timer is not None:
            try:
                self._wait_timer.cancel()
            except Exception:
                pass
            self._wait_timer = None


def main():
    rclpy.init()
    rclpy.spin(ClickWaypointsToNavUnified())
    rclpy.shutdown()


if __name__ == "__main__":
    main()
