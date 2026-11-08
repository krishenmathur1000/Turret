import cv2
import numpy as np
import os

person_name = "anaia"

# Load YuNet face detector
detector = cv2.FaceDetectorYN.create(
    "models/face_detection_yunet_2023mar.onnx",
    "",
    (320, 320)
)

# Load SFace recognizer
recognizer = cv2.FaceRecognizerSF.create(
    "models/face_recognition_sface_2021dec.onnx",
    ""
)

# Open webcam
camera = cv2.VideoCapture(0)

embeddings = []

print("Press SPACE to capture a face sample")
print("Capture about 10-15 samples")
print("Press Q when finished")

while True:

    ret, frame = camera.read()

    if not ret:
        print("Camera error")
        break

    height, width = frame.shape[:2]

    detector.setInputSize((width, height))

    result, faces = detector.detect(frame)

    if faces is not None:

        face = faces[0]

        x, y, w, h = face[:4].astype(int)

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

    cv2.putText(
        frame,
        f"Samples: {len(embeddings)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Face Enrollment", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == 32 and faces is not None:

        face = faces[0]

        aligned_face = recognizer.alignCrop(
            frame,
            face
        )

        embedding = recognizer.feature(
            aligned_face
        )

        embeddings.append(embedding)

        print(f"Captured sample {len(embeddings)}")

    if key == ord("q"):
        break


camera.release()
cv2.destroyAllWindows()


if len(embeddings) > 0:

    embeddings = np.vstack(embeddings)

    average_embedding = np.mean(
        embeddings,
        axis=0,
        keepdims=True
    )

    average_embedding = average_embedding / np.linalg.norm(
        average_embedding
    )

    os.makedirs(
        "known_faces",
        exist_ok=True
    )

    np.save(
        f"known_faces/{person_name}.npy",
        average_embedding
    )

    print("anaia face successfully enrolled.")

else:

    print("No samples captured.")