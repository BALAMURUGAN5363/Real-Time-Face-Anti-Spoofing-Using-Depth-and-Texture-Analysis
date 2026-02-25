# realtime/anti_spoof.py
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "texture_cnn.h5")

model = load_model(MODEL_PATH)

def predict_spoof(face):
    if face is None or face.size == 0:
        return 0.0

    # MATCH TRAINING PREPROCESSING
    face = cv2.resize(face, (224, 224))
    img = face.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    score = model.predict(img, verbose=0)[0][0]
    return float(score)
