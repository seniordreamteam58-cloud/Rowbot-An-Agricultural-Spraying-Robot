import time
import threading
import cv2
from ultralytics import YOLO
import RPi.GPIO as GPIO
import sys
import select
from flask import Flask, Response

# ROS 2 (Jazzy) pub/sub
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32

ROS_TOPIC_SPRAY_STATUS = "/spray_status"          # publish Int32: 0 idle, 1 busy
ROS_TOPIC_ROBOT_MOTION = "/robot_motion_state"    # subscribe Int32: 1 stopped, 0 moving
ROS_PUB_HZ = 10.0

# -----------------------------
# Streaming (browser)
# -----------------------------
STREAM_HOST = "0.0.0.0"
STREAM_PORT = 8000

STREAM_JPEG_QUALITY = 70
STREAM_FPS = 20
STREAM_MAX_W = 640
STREAM_MAX_H = 480

SHOW_ON_PI = False

# -----------------------------
# YOLO settings
# -----------------------------
WEIGHTS_PATH = "bestsenior.pt"
TARGET_LABEL = "weed"

CONF_THRESHOLD = 0.3
IMGSZ = 160
SHOW_FPS = True

SNAPSHOT_COOLDOWN_SEC = 2.0
TARGET_SELECT_MODE = "conf"  # "conf" or "area"

SNAPSHOT_SETTLE_SEC = 1.0
SNAPSHOT_REQUEST_TIMEOUT_SEC = 12.0

# Run YOLO less often to reduce CPU spikes
YOLO_EVERY_N_FRAMES = 2

# -----------------------------
# Camera settings (Arducam IMX219 via V4L2)
# -----------------------------
CAM_INDEX = 0
WARMUP_SEC = 1.0

CAM_SET_WIDTH = 640
CAM_SET_HEIGHT = 480
CAM_SET_FPS = 30

# Retry behavior (do not exit on camera failures)
CAM_READ_RETRY_SLEEP_SEC = 0.05
CAM_READ_MAX_CONSEC_FAILS = 20
CAM_REOPEN_SLEEP_SEC = 1.0

# -----------------------------
# 2-DOF Servo settings
# -----------------------------
PWM_FREQ = 50

PAN_SERVO_PIN = 19
TILT_SERVO_PIN = 12
PUMP_PIN = 23

PAN_MIN = 100
PAN_MAX = 140

TILT_MIN = 80
TILT_MAX = 120

PAN_ANGLE_CENTER = 120
TILT_ANGLE_REST = 85          # snapshot and spraying posture
TILT_ANGLE_NAV = 85          # navigation posture

# -----------------------------
# Servo stability (Option B)
# -----------------------------
SERVO_CMD_MIN_INTERVAL = 0.12
SERVO_MIN_CHANGE_DEG = 2.0
SERVO_SETTLE_DELAY = 0.08
SERVO_HOLD_SEC = 0.35

# Pixel aiming (used for snapshot targeting math)
PAN_DEG_PER_PX = 0.0971875
TILT_DEG_PER_PX = 0.1016667

PAN_INVERT = True
TILT_INVERT = True

MAX_PAN_TOTAL_DELTA_DEG = 6.0
MAX_TILT_TOTAL_DELTA_DEG = 6.0

CENTER_TOL_PX = 40
AIM_OFFSET_X_PX = 0
AIM_OFFSET_Y_PX = 0

# -----------------------------
# Global state
# -----------------------------
threads_running = True

latest_jpeg = None
jpeg_lock = threading.Lock()

snapshot_jpeg = None
snapshot_lock = threading.Lock()

snapshot_info = None
snapshot_info_lock = threading.Lock()

move_in_progress = False
move_lock = threading.Lock()

last_snapshot_time = 0.0

current_pan_angle = PAN_ANGLE_CENTER
current_tilt_angle = TILT_ANGLE_REST
last_pan_cmd_time = 0.0
last_tilt_cmd_time = 0.0
servo_lock = threading.Lock()

app = Flask(__name__)

# Snapshot request flag consumed by main loop
_manual_snapshot_flag = False
_manual_snapshot_lock = threading.Lock()

# Snapshot pending state
_snapshot_pending = False
_snapshot_pending_lock = threading.Lock()
_snapshot_requested_time = 0.0

# -----------------------------
# ROS shared state
# -----------------------------
spray_status_value = 0
spray_status_lock = threading.Lock()
_last_motion_state = 0


def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def normalize_min_max(a, b):
    return (a, b) if a <= b else (b, a)


PAN_MIN_N, PAN_MAX_N = normalize_min_max(PAN_MIN, PAN_MAX)
TILT_MIN_N, TILT_MAX_N = normalize_min_max(TILT_MIN, TILT_MAX)


def set_move_in_progress(v: bool) -> None:
    global move_in_progress
    with move_lock:
        move_in_progress = bool(v)


def is_move_in_progress() -> bool:
    with move_lock:
        return bool(move_in_progress)


def set_spray_status(v: int) -> None:
    global spray_status_value
    with spray_status_lock:
        spray_status_value = 1 if int(v) != 0 else 0


def _set_snapshot_pending(v: bool) -> None:
    global _snapshot_pending, _snapshot_requested_time
    with _snapshot_pending_lock:
        _snapshot_pending = bool(v)
        _snapshot_requested_time = time.time() if v else 0.0


def _is_snapshot_pending() -> bool:
    with _snapshot_pending_lock:
        return bool(_snapshot_pending)


def _snapshot_pending_age() -> float:
    with _snapshot_pending_lock:
        if not _snapshot_pending:
            return 0.0
        return time.time() - float(_snapshot_requested_time)


def _set_manual_snapshot_flag_true():
    global _manual_snapshot_flag
    with _manual_snapshot_lock:
        _manual_snapshot_flag = True


def try_consume_manual_snapshot_flag() -> bool:
    global _manual_snapshot_flag
    with _manual_snapshot_lock:
        if _manual_snapshot_flag:
            _manual_snapshot_flag = False
            return True
    return False


# -----------------------------
# ROS node
# -----------------------------
class RosBridgeNode(Node):
    def __init__(self):
        super().__init__("weed_sprayer_ros_bridge")
        self.pub = self.create_publisher(Int32, ROS_TOPIC_SPRAY_STATUS, 10)
        self.sub = self.create_subscription(Int32, ROS_TOPIC_ROBOT_MOTION, self._on_motion, 10)

        period = 1.0 / max(ROS_PUB_HZ, 1e-3)
        self.timer = self.create_timer(period, self._tick)

    def _tick(self):
        msg = Int32()
        with spray_status_lock:
            msg.data = int(spray_status_value)
        self.pub.publish(msg)

    def _on_motion(self, msg: Int32):
        global _last_motion_state

        new_state = 1 if int(msg.data) != 0 else 0

        if _last_motion_state == 0 and new_state == 1:
            request_snapshot_pipeline()

        if _last_motion_state == 1 and new_state == 0:
            if (not is_move_in_progress()) and (not _is_snapshot_pending()):
                threading.Thread(target=lambda: move_tilt_servo(TILT_ANGLE_NAV), daemon=True).start()

        _last_motion_state = new_state


def ros_spin_thread():
    rclpy.init(args=None)
    node = RosBridgeNode()
    try:
        rclpy.spin(node)
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass


# -----------------------------
# Web streaming
# -----------------------------
def mjpeg_generator_live():
    frame_interval = 1.0 / max(STREAM_FPS, 1)
    while True:
        t0 = time.time()
        with jpeg_lock:
            frame = latest_jpeg

        if frame is not None:
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        else:
            time.sleep(0.01)

        dt = time.time() - t0
        sleep_t = frame_interval - dt
        if sleep_t > 0:
            time.sleep(sleep_t)


def mjpeg_generator_snapshot():
    while True:
        with snapshot_lock:
            frame = snapshot_jpeg
        if frame is not None:
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        else:
            time.sleep(0.2)
        time.sleep(0.25)


@app.get("/")
def index():
    html = f"""
    <html>
      <head>
        <title>Live + Snapshot</title>
        <meta name="viewport" content="width=device-width, initial-scale=1" />
      </head>
      <body style="font-family: Arial; padding: 12px;">
        <h3>Live + Snapshot</h3>

        <div style="display:flex; gap:10px; flex-wrap:wrap; margin-bottom:10px;">
          <button onclick="fetch('/snapshot').then(r=>r.text()).then(t=>alert(t))"
                  style="padding:10px 14px; font-size:16px;">
            Take Snapshot
          </button>

          <button onclick="fetch('/clear_snapshot').then(r=>r.text()).then(t=>alert(t))"
                  style="padding:10px 14px; font-size:16px;">
            Clear Snapshot
          </button>

          <button onclick="fetch('/center').then(r=>r.text()).then(t=>alert(t))"
                  style="padding:10px 14px; font-size:16px;">
            Center Arm
          </button>
        </div>

        <div style="display:flex; gap:12px; flex-wrap:wrap;">
          <div>
            <div style="margin-bottom:6px;"><b>Live</b></div>
            <img src="/stream" style="max-width: 100%; height: auto; border:1px solid #ccc;" />
          </div>

          <div>
            <div style="margin-bottom:6px;"><b>Snapshot</b></div>
            <img src="/snapshot_stream" style="max-width: 100%; height: auto; border:1px solid #ccc;" />
          </div>
        </div>

        <p>
          Limits: Pan(X) {PAN_MIN_N}..{PAN_MAX_N}, Tilt(Y) {TILT_MIN_N}..{TILT_MAX_N}
        </p>
        <p>
          Tilt: NAV={TILT_ANGLE_NAV}, SNAPSHOT/SPRAY={TILT_ANGLE_REST}
        </p>
      </body>
    </html>
    """
    return html


@app.get("/stream")
def video_feed():
    return Response(mjpeg_generator_live(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.get("/snapshot_stream")
def snapshot_feed():
    return Response(mjpeg_generator_snapshot(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.get("/snapshot")
def web_snapshot():
    request_snapshot_pipeline()
    return "OK: snapshot requested"


@app.get("/clear_snapshot")
def web_clear_snapshot():
    clear_snapshot()
    _set_snapshot_pending(False)
    if not is_move_in_progress():
        set_spray_status(0)
        threading.Thread(target=lambda: move_tilt_servo(TILT_ANGLE_NAV), daemon=True).start()
    return "OK: snapshot cleared"


@app.get("/center")
def web_center():
    center_arm()
    return "OK: centered"


def start_stream_server():
    app.run(host=STREAM_HOST, port=STREAM_PORT, debug=False, use_reloader=False, threaded=True)


def push_jpeg_live(frame):
    global latest_jpeg
    h, w = frame.shape[:2]
    scale = min(STREAM_MAX_W / w, STREAM_MAX_H / h, 1.0)
    if scale < 1.0:
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(STREAM_JPEG_QUALITY)])
    if ok:
        with jpeg_lock:
            latest_jpeg = jpg.tobytes()


def set_snapshot_image(frame):
    global snapshot_jpeg
    h, w = frame.shape[:2]
    scale = min(STREAM_MAX_W / w, STREAM_MAX_H / h, 1.0)
    if scale < 1.0:
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(STREAM_JPEG_QUALITY)])
    if ok:
        with snapshot_lock:
            snapshot_jpeg = jpg.tobytes()


def clear_snapshot():
    global snapshot_jpeg, snapshot_info
    with snapshot_lock:
        snapshot_jpeg = None
    with snapshot_info_lock:
        snapshot_info = None


# -----------------------------
# Drawing helpers
# -----------------------------
def draw_center_crosshair(frame):
    h, w = frame.shape[:2]
    cx0 = int((w - 1) / 2)
    cy0 = int((h - 1) / 2)
    cv2.drawMarker(frame, (cx0, cy0), (255, 255, 255),
                   markerType=cv2.MARKER_CROSS, markerSize=18, thickness=2)


def draw_detection(frame, x1, y1, x2, y2, label, conf):
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
    cv2.putText(frame, f"{label} {conf:.2f}", (x1, max(y1 - 6, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)


# -----------------------------
# GPIO + Servo control (RPi.GPIO PWM)
# -----------------------------
GPIO.setmode(GPIO.BCM)
GPIO.setup(PAN_SERVO_PIN, GPIO.OUT)
GPIO.setup(TILT_SERVO_PIN, GPIO.OUT)
GPIO.setup(PUMP_PIN, GPIO.OUT)
GPIO.output(PUMP_PIN, GPIO.LOW)

pan_pwm = GPIO.PWM(PAN_SERVO_PIN, PWM_FREQ)
tilt_pwm = GPIO.PWM(TILT_SERVO_PIN, PWM_FREQ)
pan_pwm.start(0)
tilt_pwm.start(0)

def pump_on():
    GPIO.output(PUMP_PIN, GPIO.HIGH)


def pump_off():
    GPIO.output(PUMP_PIN, GPIO.LOW)


def angle_to_duty(angle: float) -> float:
    return 2.5 + (angle / 180.0) * 10.0


def _rate_limit(last_time: float) -> float:
    now = time.time()
    dt = now - last_time
    if dt < SERVO_CMD_MIN_INTERVAL:
        time.sleep(SERVO_CMD_MIN_INTERVAL - dt)
    return time.time()


def _move_servo_direct(pwm, target_angle: float, hold_sec: float = SERVO_HOLD_SEC) -> None:
    pwm.ChangeDutyCycle(angle_to_duty(target_angle))
    time.sleep(max(0.0, float(hold_sec)))
    pwm.ChangeDutyCycle(0)


def move_pan_servo(angle: float) -> None:
    global current_pan_angle, last_pan_cmd_time
    angle = clamp(float(angle), PAN_MIN_N, PAN_MAX_N)

    if abs(angle - current_pan_angle) < SERVO_MIN_CHANGE_DEG:
        return

    with servo_lock:
        last_pan_cmd_time = _rate_limit(last_pan_cmd_time)
        _move_servo_direct(pan_pwm, angle, hold_sec=SERVO_HOLD_SEC)
        current_pan_angle = angle
        last_pan_cmd_time = time.time()

    if SERVO_SETTLE_DELAY > 0:
        time.sleep(SERVO_SETTLE_DELAY)


def move_tilt_servo(angle: float) -> None:
    global current_tilt_angle, last_tilt_cmd_time
    angle = clamp(float(angle), TILT_MIN_N, TILT_MAX_N)

    if abs(angle - current_tilt_angle) < SERVO_MIN_CHANGE_DEG:
        return

    with servo_lock:
        last_tilt_cmd_time = _rate_limit(last_tilt_cmd_time)
        _move_servo_direct(tilt_pwm, angle, hold_sec=SERVO_HOLD_SEC)
        current_tilt_angle = angle
        last_tilt_cmd_time = time.time()

    if SERVO_SETTLE_DELAY > 0:
        time.sleep(SERVO_SETTLE_DELAY)


def center_arm():
    global current_pan_angle, current_tilt_angle
    move_tilt_servo(TILT_ANGLE_REST)
    time.sleep(0.10)
    move_pan_servo(PAN_ANGLE_CENTER)
    time.sleep(0.10)
    current_pan_angle = clamp(PAN_ANGLE_CENTER, PAN_MIN_N, PAN_MAX_N)
    current_tilt_angle = clamp(TILT_ANGLE_REST, TILT_MIN_N, TILT_MAX_N)


def force_set_servo_direct(pwm, angle: float, hold_sec: float = 0.35) -> None:
    with servo_lock:
        _move_servo_direct(pwm, angle, hold_sec=hold_sec)


def init_servo_pose():
    global current_pan_angle, current_tilt_angle
    global last_pan_cmd_time, last_tilt_cmd_time

    force_set_servo_direct(tilt_pwm, TILT_ANGLE_NAV, hold_sec=0.35)
    time.sleep(0.1)
    force_set_servo_direct(pan_pwm, PAN_ANGLE_CENTER, hold_sec=0.35)

    current_pan_angle = clamp(PAN_ANGLE_CENTER, PAN_MIN_N, PAN_MAX_N)
    current_tilt_angle = clamp(TILT_ANGLE_NAV, TILT_MIN_N, TILT_MAX_N)

    last_pan_cmd_time = time.time()
    last_tilt_cmd_time = time.time()

    move_tilt_servo(TILT_ANGLE_NAV)
    pump_off()
    set_spray_status(0)
    _set_snapshot_pending(False)


# -----------------------------
# Overlap filtering (extra NMS)
# -----------------------------
def iou_xyxy(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter_area

    return (inter_area / union) if union > 0 else 0.0


def nms_dets(dets, iou_thresh=0.55, score_key="conf"):
    dets_sorted = sorted(dets, key=lambda d: float(d.get(score_key, 0.0)), reverse=True)
    kept = []
    for d in dets_sorted:
        keep = True
        for k in kept:
            if iou_xyxy(d["box"], k["box"]) >= iou_thresh:
                keep = False
                break
        if keep:
            kept.append(d)
    return kept


# -----------------------------
# Snapshot pipeline
# -----------------------------
def request_snapshot_pipeline():
    """
    Called on any snapshot trigger (auto stop or manual):
    - mark busy and pending
    - move tilt to REST with exclusive lock
    - wait settle time
    - arm snapshot flag for main loop
    """
    def worker():
        global current_tilt_angle
        if is_move_in_progress():
            return

        set_spray_status(1)
        _set_snapshot_pending(True)

        with servo_lock:
            _move_servo_direct(
                tilt_pwm,
                clamp(TILT_ANGLE_REST, TILT_MIN_N, TILT_MAX_N),
                hold_sec=SERVO_HOLD_SEC
            )
            current_tilt_angle = clamp(TILT_ANGLE_REST, TILT_MIN_N, TILT_MAX_N)

        time.sleep(max(0.0, float(SNAPSHOT_SETTLE_SEC)))
        _set_manual_snapshot_flag_true()

    threading.Thread(target=worker, daemon=True).start()


# -----------------------------
# Targeting math
# -----------------------------
def compute_one_shot_targets_from_base(cx, cy, frame_w, frame_h, base_pan, base_tilt):
    center_x = (frame_w - 1) / 2.0
    center_y = (frame_h - 1) / 2.0

    err_x = (cx + AIM_OFFSET_X_PX) - center_x
    err_y = (cy + AIM_OFFSET_Y_PX) - center_y

    if abs(err_x) < CENTER_TOL_PX:
        err_x = 0.0
    if abs(err_y) < CENTER_TOL_PX:
        err_y = 0.0

    pan_delta = err_x * PAN_DEG_PER_PX
    tilt_delta = err_y * TILT_DEG_PER_PX

    if PAN_INVERT:
        pan_delta = -pan_delta
    if TILT_INVERT:
        tilt_delta = -tilt_delta

    pan_delta = clamp(pan_delta, -MAX_PAN_TOTAL_DELTA_DEG, MAX_PAN_TOTAL_DELTA_DEG)
    tilt_delta = clamp(tilt_delta, -MAX_TILT_TOTAL_DELTA_DEG, MAX_TILT_TOTAL_DELTA_DEG)

    pan_target = clamp(base_pan + pan_delta, PAN_MIN_N, PAN_MAX_N)
    tilt_target = clamp(base_tilt + tilt_delta, TILT_MIN_N, TILT_MAX_N)

    return pan_target, tilt_target, err_x, err_y


def move_sequence_from_snapshot():
    global last_snapshot_time

    if is_move_in_progress():
        return

    with snapshot_info_lock:
        info = snapshot_info

    if not info or "targets" not in info:
        return

    targets = info["targets"]
    if not targets:
        return

    set_move_in_progress(True)
    set_spray_status(1)

    try:
        for i, t in enumerate(targets, start=1):
            pan_target = t["pan_target"]
            tilt_target = t["tilt_target"]

            move_pan_servo(pan_target)
            move_tilt_servo(tilt_target)

            print(
                f"Target {i}/{len(targets)} | "
                f"{t['label']} conf={t['conf']:.2f} "
                f"pan={pan_target:.1f} tilt={tilt_target:.1f} "
                f"err_px=({t['err_x']:.1f},{t['err_y']:.1f})"
            )
            pump_on()
            time.sleep(3.0)
            pump_off()

            move_tilt_servo(TILT_ANGLE_REST)
            time.sleep(0.1)
            move_pan_servo(PAN_ANGLE_CENTER)
            time.sleep(1.0)

        clear_snapshot()
        last_snapshot_time = time.time()
        print("All snapshot targets visited. Ready for next stop.")

    finally:
        pump_off()
        clear_snapshot()
        _set_snapshot_pending(False)
        set_spray_status(0)
        move_tilt_servo(TILT_ANGLE_NAV)
        set_move_in_progress(False)


def take_snapshot_from_frame(frame, dets):
    global last_snapshot_time, snapshot_info

    set_spray_status(1)

    snap = frame.copy()
    draw_center_crosshair(snap)

    if dets:
        if TARGET_SELECT_MODE == "area":
            ordered = sorted(dets, key=lambda d: d["area"], reverse=True)
        else:
            ordered = sorted(dets, key=lambda d: d["conf"], reverse=True)
    else:
        ordered = []

    with servo_lock:
        base_pan = float(current_pan_angle)
        base_tilt = float(current_tilt_angle)

    targets = []
    for d in ordered:
        x1, y1, x2, y2 = d["box"]
        draw_detection(snap, x1, y1, x2, y2, d["label"], d["conf"])
        cv2.circle(snap, (d["cx"], d["cy"]), 5, (255, 255, 255), -1)

        pan_t, tilt_t, err_x, err_y = compute_one_shot_targets_from_base(
            d["cx"], d["cy"], d["w"], d["h"], base_pan, base_tilt
        )

        targets.append({
            "label": str(d["label"]),
            "conf": float(d["conf"]),
            "cx": int(d["cx"]),
            "cy": int(d["cy"]),
            "w": int(d["w"]),
            "h": int(d["h"]),
            "box": tuple(d["box"]),
            "pan_target": float(pan_t),
            "tilt_target": float(tilt_t),
            "err_x": float(err_x),
            "err_y": float(err_y),
        })

    with snapshot_info_lock:
        snapshot_info = {"base_pan": base_pan, "base_tilt": base_tilt, "targets": targets}

    cv2.putText(snap, "SNAPSHOT", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    set_snapshot_image(snap)
    last_snapshot_time = time.time()

    if targets and not is_move_in_progress():
        threading.Thread(target=move_sequence_from_snapshot, daemon=True).start()
        return

    print("Snapshot taken, but no targets detected.")
    clear_snapshot()
    _set_snapshot_pending(False)
    set_spray_status(0)

    if not is_move_in_progress():
        threading.Thread(target=lambda: move_tilt_servo(TILT_ANGLE_NAV), daemon=True).start()


# -----------------------------
# Camera open and retry helpers
# -----------------------------
def open_camera(index: int):
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError("Camera not available (check driver and /dev/video*)")

    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(CAM_SET_WIDTH))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(CAM_SET_HEIGHT))
        cap.set(cv2.CAP_PROP_FPS, float(CAM_SET_FPS))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    return cap


def reopen_camera(cap, index: int):
    try:
        if cap is not None:
            cap.release()
    except Exception:
        pass

    time.sleep(CAM_REOPEN_SLEEP_SEC)

    try:
        new_cap = open_camera(index)
        t0 = time.time()
        while time.time() - t0 < 0.5:
            new_cap.read()
        print("Camera reopened.")
        return new_cap
    except Exception as e:
        print(f"Camera reopen failed: {e}")
        return None


# -----------------------------
# Terminal input
# -----------------------------
def terminal_command_thread():
    global threads_running
    print("Terminal commands: snap | clear | center | quit")

    while threads_running:
        try:
            rlist, _, _ = select.select([sys.stdin], [], [], 0.2)
            if not rlist:
                continue
            line = sys.stdin.readline()
            if not line:
                continue

            cmd = line.strip().lower()
            if cmd == "snap":
                request_snapshot_pipeline()
                print("OK: snapshot requested")
            elif cmd == "clear":
                clear_snapshot()
                _set_snapshot_pending(False)
                set_spray_status(0)
                if not is_move_in_progress():
                    threading.Thread(target=lambda: move_tilt_servo(TILT_ANGLE_NAV), daemon=True).start()
                print("OK: snapshot cleared")
            elif cmd == "center":
                center_arm()
                print("OK: centered")
            elif cmd == "quit":
                threads_running = False
                break
        except Exception:
            time.sleep(0.2)


# -----------------------------
# Main
# -----------------------------
def main():
    global threads_running, last_snapshot_time

    threading.Thread(target=start_stream_server, daemon=True).start()
    threading.Thread(target=terminal_command_thread, daemon=True).start()
    threading.Thread(target=ros_spin_thread, daemon=True).start()

    init_servo_pose()

    print(f"Open on laptop: http://<PI_IP>:{STREAM_PORT}")
    print(f"Limits: Pan {PAN_MIN_N}..{PAN_MAX_N}, Tilt {TILT_MIN_N}..{TILT_MAX_N}")
    print(f"Tilt NAV={TILT_ANGLE_NAV}, SNAPSHOT/SPRAY={TILT_ANGLE_REST}")
    print("Auto snapshot triggers when /robot_motion_state rises to 1.")

    print("Loading model:", WEIGHTS_PATH)
    model = YOLO(WEIGHTS_PATH)
    names = model.names

    cap = None
    try:
        cap = open_camera(CAM_INDEX)
    except Exception as e:
        print(f"Camera open failed: {e}")
        cap = None

    if cap is not None:
        t0 = time.time()
        while time.time() - t0 < WARMUP_SEC:
            cap.read()

    prev_t = time.time()
    frame_i = 0
    last_boxes = None
    cam_fail_count = 0

    try:
        while threads_running:
            if cap is None:
                cap = reopen_camera(cap, CAM_INDEX)
                time.sleep(CAM_READ_RETRY_SLEEP_SEC)
                continue

            ret, frame = cap.read()
            if (not ret) or (frame is None):
                cam_fail_count += 1
                if cam_fail_count >= CAM_READ_MAX_CONSEC_FAILS:
                    print("Camera read failing repeatedly, reopening camera...")
                    cap = reopen_camera(cap, CAM_INDEX)
                    cam_fail_count = 0
                time.sleep(CAM_READ_RETRY_SLEEP_SEC)
                continue

            cam_fail_count = 0

            h, w = frame.shape[:2]
            draw_center_crosshair(frame)

            if _is_snapshot_pending() and _snapshot_pending_age() > SNAPSHOT_REQUEST_TIMEOUT_SEC:
                print("Snapshot pending timed out, cancelling.")
                _set_snapshot_pending(False)
                set_spray_status(0)
                if not is_move_in_progress():
                    threading.Thread(target=lambda: move_tilt_servo(TILT_ANGLE_NAV), daemon=True).start()

            frame_i += 1
            if frame_i % YOLO_EVERY_N_FRAMES == 0:
                results = model.predict(frame, imgsz=IMGSZ, conf=CONF_THRESHOLD, verbose=False)
                res = results[0]
                last_boxes = getattr(res, "boxes", None)

            dets = []
            boxes = last_boxes
            if boxes is not None:
                for b in boxes:
                    cid = int(b.cls[0])
                    label = names.get(cid, "")
                    conf = float(b.conf[0])

                    if label != TARGET_LABEL or conf < CONF_THRESHOLD:
                        continue

                    x1, y1, x2, y2 = map(int, b.xyxy[0])
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    area = max(0, (x2 - x1)) * max(0, (y2 - y1))

                    dets.append({
                        "label": label,
                        "conf": conf,
                        "box": (x1, y1, x2, y2),
                        "cx": cx,
                        "cy": cy,
                        "area": area,
                        "w": w,
                        "h": h,
                    })

            dets = nms_dets(dets, iou_thresh=0.55)

            for d in dets:
                x1, y1, x2, y2 = d["box"]
                draw_detection(frame, x1, y1, x2, y2, d["label"], d["conf"])
                cv2.circle(frame, (d["cx"], d["cy"]), 4, (255, 255, 255), -1)

            if is_move_in_progress():
                cv2.putText(frame, "BUSY (SPRAY SEQUENCE)", (20, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            elif _is_snapshot_pending():
                cv2.putText(frame, "BUSY (PENDING SNAPSHOT)", (20, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

            now = time.time()
            can_snapshot_now = (now - last_snapshot_time) >= SNAPSHOT_COOLDOWN_SEC and not is_move_in_progress()

            if _is_snapshot_pending() and can_snapshot_now:
                if try_consume_manual_snapshot_flag():
                    take_snapshot_from_frame(frame, dets)

            if SHOW_FPS:
                now2 = time.time()
                fps = 1.0 / max(now2 - prev_t, 1e-6)
                prev_t = now2
                cv2.putText(frame, f"LoopFPS {fps:.1f}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            push_jpeg_live(frame)

            if SHOW_ON_PI:
                cv2.imshow("Live", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        threads_running = False
        _set_snapshot_pending(False)
        set_spray_status(0)
        time.sleep(0.2)
        try:
            pump_off()
        except Exception:
            pass

        try:
            if cap is not None:
                cap.release()
        except Exception:
            pass

        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


if __name__ == "__main__":
    try:
        main()
    finally:
        try:
            _set_snapshot_pending(False)
        except Exception:
            pass
        try:
            set_spray_status(0)
        except Exception:
            pass
        try:
            pump_off()
        except Exception:
            pass

        try:
            pan_pwm.stop()
        except Exception:
            pass
        try:
            tilt_pwm.stop()
        except Exception:
            pass
        GPIO.cleanup()

