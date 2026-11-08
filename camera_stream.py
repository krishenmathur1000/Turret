
import os
import time
import cv2
import numpy as np
from flask import Flask, Response
from picamera2 import Picamera2

# ====================================
# SETTINGS
# ====================================

WIDTH = 640
HEIGHT = 480

DETECT_WIDTH = 320
DETECT_HEIGHT = 240

RECOGNITION_INTERVAL = 8

SIMILARITY_THRESHOLD = 0.60
MARGIN_THRESHOLD = 0.08

DEAD_ZONE_X = 40
DEAD_ZONE_Y = 30
SMOOTHING = 0.35

cv2.setNumThreads(2)

app = Flask(__name__)

# ====================================
# LOAD MODELS
# ====================================

detector = cv2.FaceDetectorYN.create(
    "models/face_detection_yunet_2023mar.onnx",
    "",
    (DETECT_WIDTH, DETECT_HEIGHT),
    0.75,
    0.3,
    5000
)

recognizer = cv2.FaceRecognizerSF.create(
    "models/face_recognition_sface_2021dec.onnx",
    ""
)

known_faces = {}

for filename in os.listdir("known_faces"):
    if filename.endswith(".npy"):
        name = filename[:-4]
        known_faces[name] = np.load(
            os.path.join("known_faces", filename)
        )
        print(f"Loaded: {name}")

print(f"Known identities: {len(known_faces)}")

# ====================================
# CAMERA
# ====================================

camera = Picamera2()
camera.configure(
    camera.create_video_configuration(
        main={
            "size": (WIDTH, HEIGHT),
            "format": "RGB888"
        },
        buffer_count=4
    )
)

camera.start()
time.sleep(2)

# ====================================
# RECOGNITION
# ====================================

def recognize_face(frame, face):
    if not known_faces:
        return "UNKNOWN", 0.0

    aligned = recognizer.alignCrop(frame, face)
    embedding = recognizer.feature(aligned)

    norm = np.linalg.norm(embedding)

    if norm == 0:
        return "UNKNOWN", 0.0

    embedding = embedding / norm

    scores = []

    for name, known_embedding in known_faces.items():
        score = recognizer.match(
            known_embedding,
            embedding,
            cv2.FaceRecognizerSF_FR_COSINE
        )
        scores.append((name, score))

    scores.sort(key=lambda item: item[1], reverse=True)

    best_name, best_score = scores[0]

    second_score = (
        scores[1][1] if len(scores) > 1 else 0.0
    )

    if (
        best_score >= SIMILARITY_THRESHOLD
        and best_score - second_score >= MARGIN_THRESHOLD
    ):
        return best_name.upper(), best_score

    return "UNKNOWN", best_score


def direction(error, dead_zone, negative, positive):
    if error < -dead_zone:
        return negative
    if error > dead_zone:
        return positive
    return "CENTERED"


def overlap_ratio(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b

    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - intersection

    return intersection / union if union > 0 else 0


# ====================================
# LIVE VIDEO AND TRACKING
# ====================================

def generate_frames():
    frame_count = 0
    previous_time = time.monotonic()
    smoothed_fps = 0.0

    cached_name = "UNKNOWN"
    cached_score = 0.0
    previous_box = None
    last_recognition = -RECOGNITION_INTERVAL

    smooth_x = None
    smooth_y = None

    scale_x = WIDTH / DETECT_WIDTH
    scale_y = HEIGHT / DETECT_HEIGHT

    while True:
        frame = camera.capture_array()
        frame_count += 1

        # Detect on smaller image
        small = cv2.resize(
            frame,
            (DETECT_WIDTH, DETECT_HEIGHT),
            interpolation=cv2.INTER_AREA
        )

        _, faces = detector.detect(small)

        target = None

        if faces is not None and len(faces) > 0:
            # Select largest detected face
            face = max(
                faces,
                key=lambda f: f[2] * f[3]
            ).copy()

            # Scale box and landmarks to full resolution
            face[0:4] *= np.array(
                [scale_x, scale_y, scale_x, scale_y]
            )

            face[4:14:2] *= scale_x
            face[5:14:2] *= scale_y

            x, y, w, h = face[:4].astype(int)
            current_box = (x, y, w, h)

            same_face = (
                previous_box is not None
                and overlap_ratio(
                    current_box, previous_box
                ) >= 0.30
            )

            # Recognize immediately for a new face,
            # otherwise once every 8 frames
            if (
                not same_face
                or frame_count - last_recognition
                >= RECOGNITION_INTERVAL
            ):
                cached_name, cached_score = recognize_face(
                    frame, face
                )
                last_recognition = frame_count

                if not same_face:
                    smooth_x = None
                    smooth_y = None

            previous_box = current_box

            color = (
                (0, 255, 0)
                if cached_name != "UNKNOWN"
                else (0, 0, 255)
            )

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                color, 2
            )

            cv2.putText(
                frame,
                f"{cached_name}: {cached_score:.2f}",
                (x, max(20, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, color, 2
            )

            if cached_name != "UNKNOWN":
                target = (
                    x + w // 2,
                    y + h // 2,
                    cached_name
                )

        else:
            previous_box = None
            cached_name = "UNKNOWN"
            cached_score = 0.0

        # ====================================
        # TRACKING
        # ====================================

        cx = WIDTH // 2
        cy = HEIGHT // 2

        cv2.drawMarker(
            frame, (cx, cy),
            (255, 255, 0),
            cv2.MARKER_CROSS, 24, 2
        )

        cv2.rectangle(
            frame,
            (cx - DEAD_ZONE_X, cy - DEAD_ZONE_Y),
            (cx + DEAD_ZONE_X, cy + DEAD_ZONE_Y),
            (255, 255, 0), 1
        )

        status = "NO TARGET"

        if target is not None:
            tx, ty, name = target

            if smooth_x is None:
                smooth_x = float(tx)
                smooth_y = float(ty)
            else:
                smooth_x += SMOOTHING * (
                    tx - smooth_x
                )
                smooth_y += SMOOTHING * (
                    ty - smooth_y
                )

            px = int(smooth_x)
            py = int(smooth_y)

            error_x = px - cx
            error_y = py - cy

            horizontal = direction(
                error_x, DEAD_ZONE_X,
                "LEFT", "RIGHT"
            )

            vertical = direction(
                error_y, DEAD_ZONE_Y,
                "UP", "DOWN"
            )

            status = f"{name}: {horizontal}, {vertical}"

            cv2.circle(
                frame, (px, py),
                6, (0, 255, 255), -1
            )

            cv2.line(
                frame, (cx, cy), (px, py),
                (0, 255, 255), 2
            )

            cv2.putText(
                frame,
                f"Error X:{error_x:+d} Y:{error_y:+d}",
                (10, HEIGHT - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 2
            )
        else:
            smooth_x = None
            smooth_y = None

        # ====================================
        # FPS
        # ====================================

        now = time.monotonic()
        dt = now - previous_time
        previous_time = now

        instant_fps = 1 / dt if dt > 0 else 0

        smoothed_fps = (
            instant_fps if smoothed_fps == 0
            else 0.2 * instant_fps
            + 0.8 * smoothed_fps
        )

        cv2.rectangle(
            frame,
            (0, 0),
            (WIDTH, 70),
            (20, 20, 20),
            -1
        )

        cv2.putText(
            frame,
            f"FPS: {smoothed_fps:.1f}",
            (10, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 255, 0), 2
        )

        cv2.putText(
            frame,
            status,
            (10, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55, (255, 255, 255), 2
        )

        success, buffer = cv2.imencode(
            ".jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 75]
        )

        if success:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + buffer.tobytes()
                + b"\r\n"
            )


@app.route("/")
def video():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":
    print("Starting optimized face tracking...")
    app.run(
        host="0.0.0.0",
        port=5000,
        threaded=False
    )
