"""
NeuroVision PD — Flask Backend
Streams processed video frames with:
- Bounding box overlay
- 68 landmark visualization
- Pose axes
- Real-time PD probability
"""

from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
import time
import threading
import json
import os
from collections import deque
from scipy.signal import welch
from sklearn.linear_model import LogisticRegression

# ── Import your existing modules ──────────────────────────────────────────────
from face_detection import FaceDetector
from mark_detection import MarkDetector
from pose_estimation import PoseEstimator
from utils import refine

app = Flask(__name__)

# ── PARAMETERS (mirror main.py) ───────────────────────────────────────────────
WINDOW_SECONDS = 5
TREMBAND       = (3, 6)
MAX_FPS        = 30
THRESHOLD      = 0.70

# ── Dummy Classifier (same as main.py — swap with your real model) ────────────
clf = LogisticRegression()
clf.coef_      = np.array([[1.5, 2.0, -1.0]])
clf.intercept_ = np.array([-5])
clf.classes_   = np.array([0, 1])

# ── Global state shared between the processing thread & Flask routes ──────────
state = {
    "prob":            0.0,
    "diagnosis":       "Initializing...",
    "yaw":             0.0,
    "pitch":           0.0,
    "roll":            0.0,
    "dominant_freq":   0.0,
    "tremor_amplitude":0.0,
    "mean_velocity":   0.0,
    "face_detected":   False,
    "frame_count":     0,
    "fps":             0.0,
    "buffer_size":     0,
    "band_power_ratio":0.0,
    "eye_distance":    0.0,
    "jaw_width":       0.0,
    "nose_tip_y":      0.0,
    "face_confidence": 0.0,
    "waveform":        [],        # last N yaw values for JS chart
}
state_lock = threading.Lock()

# ── Video source (set by /api/set_source) ─────────────────────────────────────
video_source = {"path": None, "type": "file"}  # type: "file" | "cam"

# ── Frame generator ───────────────────────────────────────────────────────────
OUTPUT_FRAME  = None
frame_lock    = threading.Lock()
stop_event    = threading.Event()
proc_thread   = None


def draw_bbox_overlay(frame, x1, y1, x2, y2, conf):
    """Draw bounding box with glowing corner accents."""
    # Main box (thin + glow via alpha blend)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 180, 0), 1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Corner accents
    cs = 18
    color = (0, 255, 210)
    lw = 3
    for (ox, oy, dx1, dy1, dx2, dy2) in [
        (x1, y1,  cs,  0,   0,  cs),
        (x2, y1, -cs,  0,   0,  cs),
        (x1, y2,  cs,  0,   0, -cs),
        (x2, y2, -cs,  0,   0, -cs),
    ]:
        cv2.line(frame, (ox, oy), (ox+dx1, oy+dy1), color, lw)
        cv2.line(frame, (ox, oy), (ox+dx2, oy+dy2), color, lw)

    # Label
    label = f"FACE_00  CONF:{conf*100:.0f}%"
    cv2.rectangle(frame, (x1, y1-22), (x1+170, y1), (0, 0, 0), -1)
    cv2.putText(frame, label, (x1+4, y1-6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 210), 1)


def draw_landmarks_overlay(frame, marks):
    """Draw 68 landmarks with group-colored polylines."""
    groups = [
        (range(0,  17), (255, 180,   0)),   # jawline     – orange
        (range(17, 22), (0,  255, 180)),    # left brow   – cyan
        (range(22, 27), (0,  255, 180)),    # right brow
        (range(27, 36), (0,  200, 255)),    # nose        – blue
        (range(36, 42), (100, 100, 255)),   # left eye    – purple
        (range(42, 48), (100, 100, 255)),   # right eye
        (range(48, 68), (200,  80, 255)),   # mouth       – pink
    ]
    for rng, color in groups:
        pts = np.array([[int(marks[i][0]), int(marks[i][1])] for i in rng],
                       dtype=np.int32)
        cv2.polylines(frame, [pts], False, color, 1, cv2.LINE_AA)

    for i, (x, y) in enumerate(marks):
        r     = 4 if i == 30 else 2          # nose tip bigger
        color = (0, 255, 0) if i == 30 else (0, 200, 255)
        cv2.circle(frame, (int(x), int(y)), r, color, -1, cv2.LINE_AA)


def draw_pose_axes(frame, marks, rvec, tvec, frame_w, frame_h):
    """Draw 3-axis pose arrows from the nose tip."""
    axis_len = 60
    axes_3d  = np.float32([[axis_len, 0, 0],
                            [0, axis_len, 0],
                            [0, 0, axis_len]])

    # Build camera matrix
    focal   = frame_w
    cx, cy  = frame_w / 2, frame_h / 2
    cam_mat = np.array([[focal, 0, cx],
                        [0, focal, cy],
                        [0,     0,  1]], dtype=np.float64)
    dist    = np.zeros((4, 1))

    nose_2d, _   = cv2.projectPoints(np.float32([[0, 0, 0]]),
                                     rvec, tvec, cam_mat, dist)
    axes_2d, _   = cv2.projectPoints(axes_3d, rvec, tvec, cam_mat, dist)

    origin = tuple(nose_2d[0][0].astype(int))
    colors = [(0, 180, 255), (0, 255, 180), (0, 140, 255)]   # BGR
    labels = ["Y", "P", "R"]

    for pt, color, label in zip(axes_2d, colors, labels):
        end = tuple(pt[0].astype(int))
        cv2.arrowedLine(frame, origin, end, color, 2, cv2.LINE_AA, tipLength=0.2)
        cv2.putText(frame, label, (end[0]+3, end[1]+3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)


def draw_hud(frame, st):
    """Top-left data HUD."""
    lines = [
        f"FRAME : {st['frame_count']:05d}",
        f"FPS   : {st['fps']:.1f}",
        f"BUF   : {st['buffer_size']:03d}/150",
        f"YAW   : {st['yaw']:+.1f} deg",
        f"PITCH : {st['pitch']:+.1f} deg",
        f"ROLL  : {st['roll']:+.1f} deg",
        f"FREQ  : {st['dominant_freq']:.2f} Hz",
        f"AMP   : {st['tremor_amplitude']:.3f}",
        f"PROB  : {st['prob']:.3f}",
    ]
    pad = 6
    lh  = 16
    bh  = len(lines) * lh + pad * 2
    bw  = 180
    overlay = frame.copy()
    cv2.rectangle(overlay, (4, 4), (4+bw, 4+bh), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    for i, line in enumerate(lines):
        cv2.putText(frame, line,
                    (8, 4 + pad + (i+1)*lh - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (0, 255, 200), 1, cv2.LINE_AA)


def draw_diagnosis_bar(frame, prob, w):
    """Bottom diagnosis bar."""
    bh = 34
    h  = frame.shape[0]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h-bh), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Confidence fill
    fill_w = int(w * prob)
    color  = (0, 0, 255) if prob > THRESHOLD else (0, 200, 100)
    cv2.rectangle(frame, (0, h-bh), (fill_w, h), color, -1)

    label = "Parkinson's Disease Detected" if prob > THRESHOLD else "Healthy"
    cv2.putText(frame, f"{label}  [{prob*100:.1f}%]",
                (10, h-10), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 2, cv2.LINE_AA)


def scan_line(frame, t):
    """Animated horizontal scan line."""
    h, w = frame.shape[:2]
    y = int((t * 60) % h)
    overlay = frame.copy()
    cv2.line(overlay, (0, y), (w, y), (0, 180, 255), 2)
    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)


# ── Main processing loop ──────────────────────────────────────────────────────
def process_video():
    global OUTPUT_FRAME, proc_thread

    src = video_source.get("path")
    if src is None:
        src = 0  # webcam fallback

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {src}")
        return

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    face_detector  = FaceDetector("assets/face_detector.onnx")
    mark_detector  = MarkDetector("assets/face_landmarks.onnx")
    pose_estimator = PoseEstimator(frame_w, frame_h)

    yaw_buffer  = deque(maxlen=WINDOW_SECONDS * MAX_FPS)
    time_buffer = deque(maxlen=WINDOW_SECONDS * MAX_FPS)

    frame_count = 0
    t_last      = time.time()
    fps_buf     = deque(maxlen=30)

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            # Loop video file
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame_count += 1
        now = time.time()
        fps_buf.append(1.0 / max(now - t_last, 1e-6))
        t_last = now
        avg_fps = float(np.mean(fps_buf))

        # Mirror for webcam
        if video_source.get("type") == "cam":
            frame = cv2.flip(frame, 1)

        # ── Detection ──────────────────────────────────────────────────────
        faces, _ = face_detector.detect(frame, 0.7)
        prob      = 0.0
        face_conf = 0.0

        local_state = {
            "frame_count":      frame_count,
            "fps":              avg_fps,
            "face_detected":    False,
            "prob":             0.0,
            "diagnosis":        "No Face Detected",
            "yaw":              0.0,
            "pitch":            0.0,
            "roll":             0.0,
            "dominant_freq":    0.0,
            "tremor_amplitude": 0.0,
            "mean_velocity":    0.0,
            "buffer_size":      len(yaw_buffer),
            "band_power_ratio": 0.0,
            "eye_distance":     0.0,
            "jaw_width":        0.0,
            "nose_tip_y":       0.0,
            "face_confidence":  0.0,
            "waveform":         list(yaw_buffer)[-60:],
        }

        if len(faces) > 0:
            face      = refine(faces, frame_w, frame_h, 0.15)[0]
            x1, y1, x2, y2 = face[:4].astype(int)
            face_conf = float(face[4]) if len(face) > 4 else 0.9

            # Landmarks
            patch  = frame[y1:y2, x1:x2]
            marks  = mark_detector.detect([patch])[0].reshape([68, 2])
            marks *= (x2 - x1)
            marks[:, 0] += x1
            marks[:, 1] += y1

            # Pose
            pose       = pose_estimator.solve(marks)
            rvec, tvec = pose
            rot_mat, _ = cv2.Rodrigues(rvec)
            proj_mat   = np.hstack((rot_mat, tvec))
            _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(proj_mat)
            pitch, yaw, roll = euler.flatten()

            yaw_buffer.append(float(yaw))
            time_buffer.append(now)

            # ── Landmark extras ────────────────────────────────────────────
            nose_tip_y   = float(marks[30, 1]) if len(marks) > 30 else 0
            left_eye_pt  = marks[36]
            right_eye_pt = marks[45]
            eye_dist     = float(np.linalg.norm(right_eye_pt - left_eye_pt))
            jaw_width    = float(abs(marks[16, 0] - marks[0, 0]))

            # ── Signal analysis ────────────────────────────────────────────
            dominant_freq    = 0.0
            tremor_amplitude = 0.0
            mean_velocity    = 0.0
            band_ratio       = 0.0

            if len(yaw_buffer) > MAX_FPS * 2:
                yaw_arr  = np.array(yaw_buffer)
                duration = time_buffer[-1] - time_buffer[0]
                fs       = len(yaw_arr) / duration if duration > 0 else 30

                freqs, power = welch(yaw_arr, fs=fs)
                band_mask    = (freqs >= TREMBAND[0]) & (freqs <= TREMBAND[1])

                if np.any(band_mask):
                    dominant_freq = float(freqs[band_mask][np.argmax(power[band_mask])])
                    band_ratio    = float(power[band_mask].sum() / (power.sum() + 1e-9))

                tremor_amplitude = float(np.std(yaw_arr))
                mean_velocity    = float(np.mean(np.abs(np.diff(yaw_arr))))

                features = np.array([[dominant_freq, tremor_amplitude, mean_velocity]])
                prob     = float(clf.predict_proba(features)[0][1])

            # ── Draw overlays ──────────────────────────────────────────────
            draw_bbox_overlay(frame, x1, y1, x2, y2, face_conf)
            draw_landmarks_overlay(frame, marks)
            pose_estimator.visualize(frame, pose, color=(0, 255, 0))
            draw_pose_axes(frame, marks, rvec, tvec, frame_w, frame_h)

            local_state.update({
                "face_detected":    True,
                "prob":             prob,
                "diagnosis":        "Parkinson's Disease Detected" if prob > THRESHOLD else "Healthy",
                "yaw":              float(yaw),
                "pitch":            float(pitch),
                "roll":             float(roll),
                "dominant_freq":    dominant_freq,
                "tremor_amplitude": tremor_amplitude,
                "mean_velocity":    mean_velocity,
                "buffer_size":      len(yaw_buffer),
                "band_power_ratio": band_ratio,
                "eye_distance":     eye_dist,
                "jaw_width":        jaw_width,
                "nose_tip_y":       nose_tip_y,
                "face_confidence":  face_conf,
                "waveform":         list(yaw_buffer)[-80:],
            })

        # ── Draw HUD + scan + diagnosis bar ───────────────────────────────
        scan_line(frame, now)
        draw_hud(frame, local_state)
        draw_diagnosis_bar(frame, prob, frame_w)

        # ── Update shared state ───────────────────────────────────────────
        with state_lock:
            state.update(local_state)

        # ── Encode & publish frame ─────────────────────────────────────────
        ret2, buf = cv2.imencode('.jpg', frame,
                                 [cv2.IMWRITE_JPEG_QUALITY, 82])
        if ret2:
            with frame_lock:
                OUTPUT_FRAME = buf.tobytes()

    cap.release()


def generate_frames():
    """MJPEG stream generator."""
    global OUTPUT_FRAME
    while True:
        with frame_lock:
            if OUTPUT_FRAME is None:
                time.sleep(0.03)
                continue
            frame = OUTPUT_FRAME
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(1 / MAX_FPS)


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/stats')
def api_stats():
    with state_lock:
        return jsonify(dict(state))


@app.route('/api/set_source', methods=['POST'])
def set_source():
    """Set video source: {'type': 'file', 'path': '1.mp4'} or {'type': 'cam'}"""
    global proc_thread, stop_event
    data = request.json

    # Stop existing thread
    stop_event.set()
    if proc_thread and proc_thread.is_alive():
        proc_thread.join(timeout=3)

    # Update source
    if data.get('type') == 'cam':
        video_source['type'] = 'cam'
        video_source['path'] = int(data.get('cam_index', 0))
    else:
        video_source['type'] = 'file'
        video_source['path'] = data.get('path', '1.mp4')

    # Restart
    stop_event = threading.Event()
    proc_thread = threading.Thread(target=process_video, daemon=True)
    proc_thread.start()

    return jsonify({"status": "ok", "source": video_source['path']})


@app.route('/api/videos')
def api_videos():
    """List available mp4 files in project root."""
    videos = [f for f in os.listdir('.') if f.endswith(('.mp4', '.avi', '.mov', '.webm'))]
    return jsonify(videos)


if __name__ == '__main__':
    # Auto-start with first available video or webcam
    stop_event = threading.Event()
    videos = [f for f in os.listdir('.') if f.endswith('.mp4')]
    if videos:
        video_source['path'] = videos[0]
        video_source['type'] = 'file'
    else:
        video_source['path'] = 0
        video_source['type'] = 'cam'

    proc_thread = threading.Thread(target=process_video, daemon=True)
    proc_thread.start()

    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)