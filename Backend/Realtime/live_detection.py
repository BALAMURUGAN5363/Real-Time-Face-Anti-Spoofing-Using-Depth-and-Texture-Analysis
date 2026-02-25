# realtime/live_detection.py
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# ✅ Load trained TEXTURE model ONLY
texture_model = load_model("model/texture_cnn.h5")

# Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def preprocess_face(face):
    face = cv2.resize(face, (224, 224))
    rgb = face.astype("float32") / 255.0
    rgb = np.expand_dims(rgb, axis=0)  # (1, 224, 224, 3)
    return rgb

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Error: Camera not accessible")
    exit()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray_frame,
            scaleFactor=1.3,
            minNeighbors=5
        )

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]

            rgb = preprocess_face(face)

            # ============================
            # 🔥 MULTI-FRAME AVERAGING
            # ============================
            scores = []
            for _ in range(5):
                s = texture_model.predict(rgb, verbose=0)[0][0]
                scores.append(s)

            score = sum(scores) / len(scores)

            # ✅ DEBUG PRINT
            print("Spoof score:", score)

            # ============================
            # THRESHOLD (WEBCAM CALIBRATED)
            # ============================
            if score > 0.50:
                label = "REAL"
                color = (0, 255, 0)
            else:
                label = "SPOOF"
                color = (0, 0, 255)

            # Draw bounding box & label
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(
                frame,
                f"{label} ({score:.2f})",
                (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                color,
                2
            )

        cv2.putText(
            frame,
            "Press Q to Exit",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.imshow("Face Anti-Spoofing (Texture Model)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
