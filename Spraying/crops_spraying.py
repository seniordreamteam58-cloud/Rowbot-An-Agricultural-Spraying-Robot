import time
import threading
import cv2
from ultralytics import YOLO
import RPi.GPIO as GPIO

from flask import Flask, Response

# -----------------------------
# YOLO settings
# -----------------------------
WEIGHTS_PATH = "bestsenior.pt"
TARGET_LABELS = ["crop"]

DETECT_CONF = 0.35
SPRAY_CONF = 0.80  # spray only if confidence is at least 0.80

IMGSZ = 160
SHOW_FPS = True

# -----------------------------
# Camera settings
# -----------------------------
CAM_INDEX = 0
WARMUP_SEC = 1.0

# camera retry settings
CAM_RETRY_DELAY_SEC = 1.0

# -----------------------------
# Timing settings
# -----------------------------
SPRAY_DURATION = 1.5

# -----------------------------
# CENTER GATE SETTINGS
# -----------------------------
CENTER_TOL_X = 40
CENTER_TOL_Y = 40


def is_centered(cx: int, cy: int, frame_w: int, frame_h: int) -> bool:
    fx = frame_w // 2
    fy = frame_h // 2
    return (abs(cx - fx) <= CENTER_TOL_X) and (abs(cy - fy) <= CENTER_TOL_Y)


# -----------------------------
# Servo settings
# -----------------------------
PWM_FREQ = 50

TILT_SERVO_PIN = 12
ANGLE_TILT_CENTER = 120
ANGLE_TILT_DOWN = 160

PAN_SERVO_PIN = 19
PAN_ANGLE_CENTER = 60

SERVO_STEP_DEG = 1.0
SERVO_STEP_DELAY = 0.01

# -----------------------------
# Pump / Relay control settings
# -----------------------------
PUMP_RELAY_PIN = 23
RELAY_ON_IS_HIGH = True  # set False if your relay is active-low

# -----------------------------
# Streaming settings
# -----------------------------
STREAM_HOST = "0.0.0.0"
STREAM_PORT = 8000
STREAM_JPEG_QUALITY = 70

SHOW_ON_PI = False

# -----------------------------
# Global flags
# -----------------------------
spray_active = False
spray_end_time = 0.0
threads_running = True

current_tilt_angle = 130
current_pan_angle = PAN_ANGLE_CENTER

last_spray_conf = 0.0

# -----------------------------
# Frame for streaming (latest only)
# -----------------------------
latest_jpeg = None
jpeg_lock = threading.Lock()

app = Flask(__name__)


def mjpeg_generator():
    global latest_jpeg
    while True:
        with jpeg_lock:
            frame = latest_jpeg
        if frame is None:
            time.sleep(0.01)
            continue

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        time.sleep(0.01)


@app.get("/")
def video_feed():
    return Response(mjpeg_generator(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


def start_stream_server():
    app.run(host=STREAM_HOST, port=STREAM_PORT, debug=False, use_reloader=False, threaded=True)


# -----------------------------
# GPIO setup
# -----------------------------
GPIO.setmode(GPIO.BCM)

GPIO.setup(TILT_SERVO_PIN, GPIO.OUT)
GPIO.setup(PAN_SERVO_PIN, GPIO.OUT)
GPIO.setup(PUMP_RELAY_PIN, GPIO.OUT)

tilt_pwm = GPIO.PWM(TILT_SERVO_PIN, PWM_FREQ)
pan_pwm = GPIO.PWM(PAN_SERVO_PIN, PWM_FREQ)

tilt_pwm.start(0)
pan_pwm.start(0)


def pump_on():
    GPIO.output(PUMP_RELAY_PIN, GPIO.HIGH if RELAY_ON_IS_HIGH else GPIO.LOW)


def pump_off():
    GPIO.output(PUMP_RELAY_PIN, GPIO.LOW if RELAY_ON_IS_HIGH else GPIO.HIGH)


# Ensure pump is OFF at start
pump_off()


def angle_to_duty(angle: float) -> float:
    return 2.5 + (angle / 180.0) * 10.0


def _move_servo_slow(pwm, start_angle: float, target_angle: float) -> None:
    start_angle = max(0.0, min(180.0, float(start_angle)))
    target_angle = max(0.0, min(180.0, float(target_angle)))

    if abs(target_angle - start_angle) < 1e-6:
        return

    step = SERVO_STEP_DEG if target_angle > start_angle else -SERVO_STEP_DEG
    angle = start_angle

    while (target_angle - angle) * step > 0:
        pwm.ChangeDutyCycle(angle_to_duty(angle))
        time.sleep(SERVO_STEP_DELAY)
        angle += step
        if (target_angle - angle) * step <= 0:
            break

    pwm.ChangeDutyCycle(angle_to_duty(target_angle))
    time.sleep(SERVO_STEP_DELAY)
    pwm.ChangeDutyCycle(0)


def move_tilt_servo(angle: float) -> None:
    global current_tilt_angle
    angle = max(0.0, min(180.0, float(angle)))
    _move_servo_slow(tilt_pwm, current_tilt_angle, angle)
    current_tilt_angle = angle


def move_pan_servo(angle: float) -> None:
    global current_pan_angle
    angle = max(0.0, min(180.0, float(angle)))
    _move_servo_slow(pan_pwm, current_pan_angle, angle)
    current_pan_angle = angle


# -----------------------------
# Servo thread for spraying + pump control
# -----------------------------
def servo_thread():
    global spray_active, spray_end_time, threads_running

    move_tilt_servo(140)
    move_pan_servo(PAN_ANGLE_CENTER)

    pump_off()

    while threads_running:
        if spray_active:
            now = time.time()
            if now >= spray_end_time:
                spray_active = False
                pump_off()
                move_tilt_servo(130)
                time.sleep(0.1)
                continue

            # Pump ON during active spray window
            pump_on()

            move_tilt_servo(ANGLE_TILT_DOWN)
            time.sleep(0.15)
            move_tilt_servo(ANGLE_TILT_CENTER)
            time.sleep(0.15)
            move_tilt_servo(130)
            time.sleep(0.15)
            move_tilt_servo(130)
            time.sleep(0.3)
            move_tilt_servo(130)

        else:
            pump_off()
            move_tilt_servo(130)
            time.sleep(0.15)
            move_tilt_servo(130)
            time.sleep(0.15)
            time.sleep(0.02)


# open_camera retries forever (no exception)
def open_camera(index: int):
    while True:
        cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        if cap.isOpened():
            return cap

        try:
            cap.release()
        except Exception:
            pass

        print("Camera not available. Retrying...")
        time.sleep(CAM_RETRY_DELAY_SEC)


def draw_detection(frame, x1, y1, x2, y2, label, conf, color):
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        frame,
        f"{label} {conf:.2f}",
        (x1, max(y1 - 5, 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
    )


def push_frame_to_stream(frame):
    global latest_jpeg
    ok, jpg = cv2.imencode(
        ".jpg",
        frame,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(STREAM_JPEG_QUALITY)]
    )
    if ok:
        with jpeg_lock:
            latest_jpeg = jpg.tobytes()


def main():
    global spray_active, spray_end_time, threads_running, last_spray_conf

    server_thread = threading.Thread(target=start_stream_server, daemon=True)
    server_thread.start()
    print(f"Streaming on: http://<PI_IP>:{STREAM_PORT}")

    print("Loading model:", WEIGHTS_PATH)
    model = YOLO(WEIGHTS_PATH)
    names = model.names

    print("Model classes:", names)
    print("Spraying labels:", TARGET_LABELS)
    print("DETECT_CONF:", DETECT_CONF, "SPRAY_CONF:", SPRAY_CONF)

    cap = open_camera(CAM_INDEX)
    move_tilt_servo(130)

    t0 = time.time()
    while time.time() - t0 < WARMUP_SEC:
        cap.read()

    thread = threading.Thread(target=servo_thread, daemon=True)
    thread.start()

    prev_t = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Camera read failed. Reopening camera...")
                try:
                    cap.release()
                except Exception:
                    pass
                time.sleep(CAM_RETRY_DELAY_SEC)
                cap = open_camera(CAM_INDEX)
                continue

            frame_h, frame_w = frame.shape[:2]

            cx0, cy0 = frame_w // 2, frame_h // 2
            cv2.rectangle(
                frame,
                (cx0 - CENTER_TOL_X, cy0 - CENTER_TOL_Y),
                (cx0 + CENTER_TOL_X, cy0 + CENTER_TOL_Y),
                (0, 255, 0),
                2,
            )

            move_pan_servo(PAN_ANGLE_CENTER)

            if spray_active:
                cv2.putText(
                    frame,
                    "SPRAYING",
                    (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    2,
                )
            else:
                results = model.predict(frame, imgsz=IMGSZ, conf=DETECT_CONF, verbose=False)
                res = results[0]
                boxes = getattr(res, "boxes", None)

                centered_trigger = False
                trigger_conf = None

                if boxes is not None:
                    for b in boxes:
                        cid = int(b.cls[0])
                        label = names.get(cid, "")
                        conf = float(b.conf[0])

                        if label in TARGET_LABELS and conf >= DETECT_CONF:
                            x1, y1, x2, y2 = map(int, b.xyxy[0])
                            det_cx = int((x1 + x2) / 2)
                            det_cy = int((y1 + y2) / 2)

                            centered = is_centered(det_cx, det_cy, frame_w, frame_h)

                            if conf >= SPRAY_CONF and centered:
                                draw_detection(frame, x1, y1, x2, y2, label, conf, (0, 255, 255))
                                centered_trigger = True
                                if trigger_conf is None or conf > trigger_conf:
                                    trigger_conf = conf
                            elif conf >= SPRAY_CONF and not centered:
                                draw_detection(frame, x1, y1, x2, y2, label, conf, (0, 0, 255))
                            else:
                                draw_detection(frame, x1, y1, x2, y2, label, conf, (255, 255, 255))

                if centered_trigger:
                    last_spray_conf = float(trigger_conf) if trigger_conf is not None else 0.0
                    spray_active = True
                    spray_end_time = time.time() + SPRAY_DURATION
                    print(f"SPRAY TRIGGERED | confidence={last_spray_conf:.3f}")

            if SHOW_FPS:
                now = time.time()
                fps = 1.0 / max(now - prev_t, 1e-6)
                prev_t = now
                cv2.putText(
                    frame,
                    f"FPS {fps:.1f}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

            push_frame_to_stream(frame)

            if SHOW_ON_PI:
                cv2.imshow("Crop sprayer", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        threads_running = False
        pump_off()
        time.sleep(0.2)
        try:
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
            pump_off()
        except Exception:
            pass
        try:
            tilt_pwm.stop()
        except Exception:
            pass
        try:
            pan_pwm.stop()
        except Exception:
            pass
        GPIO.cleanup()

