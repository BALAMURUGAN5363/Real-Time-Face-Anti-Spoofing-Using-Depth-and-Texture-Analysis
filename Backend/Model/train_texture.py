import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D,
    Dense, Flatten, Dropout, Input
)
from tensorflow.keras.optimizers import Adam

import os
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

from preprocessing.load_data import load_images
from preprocessing.labels import create_labels_from_filenames


# ========================================
# PATH CONFIGURATION
# ========================================
BASE_DIR = os.getcwd()

TRAIN_COLOR_PATH = os.path.join(BASE_DIR, "dataset", "train", "color")
TRAIN_DEPTH_PATH = os.path.join(BASE_DIR, "dataset", "train", "depth")

TEST_COLOR_PATH = os.path.join(BASE_DIR, "dataset", "test", "color")
TEST_DEPTH_PATH = os.path.join(BASE_DIR, "dataset", "test", "depth")


# ========================================
# LOAD DATA
# ========================================
train_color, train_depth = load_images(
    TRAIN_COLOR_PATH,
    TRAIN_DEPTH_PATH
)

test_color, test_depth = load_images(
    TEST_COLOR_PATH,
    TEST_DEPTH_PATH
)

print("Train shape:", train_color.shape)
print("Test shape:", test_color.shape)


# ========================================
# LABELS
# ========================================
train_labels = create_labels_from_filenames(TRAIN_COLOR_PATH)
test_labels = create_labels_from_filenames(TEST_COLOR_PATH)

print("Train REAL:", np.sum(train_labels == 1))
print("Train SPOOF:", np.sum(train_labels == 0))
print("Test REAL:", np.sum(test_labels == 1))
print("Test SPOOF:", np.sum(test_labels == 0))


# ========================================
# CLASS WEIGHT (HANDLE IMBALANCE)
# ========================================
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_labels),
    y=train_labels
)

class_weights = dict(enumerate(class_weights))
print("Class Weights:", class_weights)


# ========================================
# NORMALIZATION
# ========================================
train_color = train_color.astype("float32") / 255.0
test_color = test_color.astype("float32") / 255.0


# ========================================
# BUILD TEXTURE CNN MODEL
# ========================================
def build_texture_cnn(input_shape=(224, 224, 3)):

    inputs = Input(shape=input_shape)

    x = Conv2D(32, (3,3), activation='relu')(inputs)
    x = MaxPooling2D((2,2))(x)

    x = Conv2D(64, (3,3), activation='relu')(x)
    x = MaxPooling2D((2,2))(x)

    x = Conv2D(128, (3,3), activation='relu')(x)
    x = MaxPooling2D((2,2))(x)

    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)

    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs, outputs)
    return model


texture_model = build_texture_cnn()

texture_model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

texture_model.summary()


# ========================================
# TRAIN MODEL
# ========================================
history = texture_model.fit(
    train_color,
    train_labels,
    validation_data=(test_color, test_labels),
    epochs=10,
    batch_size=16,
    class_weight=class_weights
)


# ========================================
# FINAL TEST EVALUATION
# ========================================
test_loss, test_accuracy = texture_model.evaluate(
    test_color,
    test_labels,
    verbose=1
)

print("\n==============================")
print("FINAL TEST ACCURACY:", round(test_accuracy * 100, 2), "%")
print("FINAL TEST LOSS:", test_loss)
print("==============================\n")


# ========================================
# CONFUSION MATRIX + CLASSIFICATION REPORT
# ========================================
predictions = texture_model.predict(test_color)
predictions = (predictions > 0.5).astype(int)

print("Confusion Matrix:")
print(confusion_matrix(test_labels, predictions))

print("\nClassification Report:")
print(classification_report(test_labels, predictions))


# ========================================
# SAVE MODEL
# ========================================
os.makedirs("model", exist_ok=True)
texture_model.save("model/texture_cnn.h5")

print("\n✅ Model Saved Successfully!")