"""
Advanced Parkinson Detection System
- Bounding box
- Pose visualization
- Real-time tremor waveform
- PD vs Healthy classifier
- Confidence bar
- Circular probability gauge
"""

from argparse import ArgumentParser
import cv2
import numpy as np
import time
from collections import deque
from scipy.signal import welch
from sklearn.linear_model import LogisticRegression

from face_detection import FaceDetector
from mark_detection import MarkDetector
from pose_estimation import PoseEstimator
from utils import refine


# =============================
# PARAMETERS
# =============================
WINDOW_SECONDS = 5
TREMBAND = (3, 6)
MAX_FPS = 30
GRAPH_WIDTH = 640
THRESHOLD = 0.70


# =============================
# Dummy Classifier (Replace with real trained model)
# =============================
clf = LogisticRegression()
clf.coef_ = np.array([[1.5, 2.0, -1.0]])
clf.intercept_ = np.array([-5])
clf.classes_ = np.array([0, 1])  # 0 = Healthy, 1 = PD


# =============================
# ARGUMENT PARSER
# =============================
parser = ArgumentParser()
parser.add_argument("--video", type=str, default=None)
parser.add_argument("--cam", type=int, default=0)
args = parser.parse_args()


# =============================
# DRAW WAVEFORM
# =============================
def draw_waveform(signal, width, height):
    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    if len(signal) < 2:
        return canvas

    signal = np.array(signal)
    signal = signal - np.mean(signal)
    max_val = max(np.max(np.abs(signal)), 1)

    norm_signal = signal / max_val
    x_scale = width / len(signal)

    for i in range(len(signal) - 1):
        x1 = int(i * x_scale)
        y1 = int(height / 2 - norm_signal[i] * (height / 3))
        x2 = int((i + 1) * x_scale)
        y2 = int(height / 2 - norm_signal[i + 1] * (height / 3))
        cv2.line(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.line(canvas, (0, height // 2), (width, height // 2), (100, 100, 100), 1)
    return canvas


# =============================
# DRAW CONFIDENCE BAR
# =============================
def draw_confidence_bar(frame, prob):
    bar_x = 20
    bar_y = 180
    bar_width = 250
    bar_height = 20

    cv2.rectangle(frame, (bar_x, bar_y),
                  (bar_x + bar_width, bar_y + bar_height),
                  (255, 255, 255), 2)

    fill_width = int(bar_width * prob)
    color = (0, 0, 255) if prob > THRESHOLD else (0, 255, 0)

    cv2.rectangle(frame, (bar_x, bar_y),
                  (bar_x + fill_width, bar_y + bar_height),
                  color, -1)

    cv2.putText(frame, "Confidence",
                (bar_x, bar_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1)


# =============================
# DRAW PROBABILITY GAUGE
# =============================
def draw_gauge(frame, prob):
    center = (450, 200)
    radius = 60

    cv2.circle(frame, center, radius, (255, 255, 255), 2)

    angle = int(270 * prob)
    cv2.ellipse(frame, center, (radius, radius),
                -90, 0, angle,
                (0, 0, 255) if prob > THRESHOLD else (0, 255, 0), -1)

    cv2.putText(frame, f"{int(prob*100)}%",
                (center[0] - 30, center[1] + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2)


# =============================
# MAIN
# =============================
def run():

    video_src = args.cam if args.video is None else args.video
    cap = cv2.VideoCapture(video_src)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    face_detector = FaceDetector("assets/face_detector.onnx")
    mark_detector = MarkDetector("assets/face_landmarks.onnx")
    pose_estimator = PoseEstimator(frame_width, frame_height)

    yaw_buffer = deque(maxlen=WINDOW_SECONDS * MAX_FPS)
    time_buffer = deque(maxlen=WINDOW_SECONDS * MAX_FPS)

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        if video_src == 0:
            frame = cv2.flip(frame, 2)

        faces, _ = face_detector.detect(frame, 0.7)

        prob = 0.0

        if len(faces) > 0:

            face = refine(faces, frame_width, frame_height, 0.15)[0]
            x1, y1, x2, y2 = face[:4].astype(int)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            patch = frame[y1:y2, x1:x2]
            marks = mark_detector.detect([patch])[0].reshape([68, 2])

            marks *= (x2 - x1)
            marks[:, 0] += x1
            marks[:, 1] += y1

            pose = pose_estimator.solve(marks)
            pose_estimator.visualize(frame, pose, color=(0, 255, 0))

            rvec, tvec = pose
            rot_mat, _ = cv2.Rodrigues(rvec)
            proj_mat = np.hstack((rot_mat, tvec))
            _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(proj_mat)
            pitch, yaw, roll = euler.flatten()

            yaw_buffer.append(yaw)
            time_buffer.append(time.time())

            if len(yaw_buffer) > MAX_FPS * 2:
                yaw_array = np.array(yaw_buffer)
                duration = time_buffer[-1] - time_buffer[0]
                fs = len(yaw_array) / duration if duration > 0 else 30

                freqs, power = welch(yaw_array, fs=fs)
                band_mask = (freqs >= TREMBAND[0]) & (freqs <= TREMBAND[1])

                dominant_freq = freqs[band_mask][np.argmax(power[band_mask])] if np.any(band_mask) else 0
                tremor_amplitude = np.std(yaw_array)
                mean_velocity = np.mean(np.abs(np.diff(yaw_array)))

                features = np.array([[dominant_freq,
                                      tremor_amplitude,
                                      mean_velocity]])

                prob = clf.predict_proba(features)[0][1]

                if prob > THRESHOLD:
                    diagnosis = "Parkinson's Disease Detected"
                    color = (0, 0, 255)
                else:
                    diagnosis = "Healthy"
                    color = (0, 255, 0)

                cv2.putText(frame, f"PD Probability: {prob:.2f}",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 255), 2)

                cv2.putText(frame, diagnosis,
                            (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, color, 3)

                draw_confidence_bar(frame, prob)
                draw_gauge(frame, prob)

        graph = draw_waveform(yaw_buffer, GRAPH_WIDTH, frame_height)
        frame_resized = cv2.resize(frame, (GRAPH_WIDTH, frame_height))
        combined = np.hstack((frame_resized, graph))

        cv2.imshow("Advanced Parkinson Monitor", combined)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    run()