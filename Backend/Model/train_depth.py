# 1️⃣ Imports
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D,
    Flatten, Dense, Dropout
)
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight


import os
import numpy as np
from preprocessing.load_data import load_images
from preprocessing.labels import create_labels_from_filenames

# 2️⃣ Paths
BASE_DIR = os.getcwd()

TRAIN_COLOR_PATH = os.path.join(BASE_DIR, "dataset", "train", "color")
TRAIN_DEPTH_PATH = os.path.join(BASE_DIR, "dataset", "train", "depth")

TEST_COLOR_PATH = os.path.join(BASE_DIR, "dataset", "test", "color")
TEST_DEPTH_PATH = os.path.join(BASE_DIR, "dataset", "test", "depth")

# 3️⃣ Load depth data
_, train_depth = load_images(TRAIN_COLOR_PATH, TRAIN_DEPTH_PATH)
_, test_depth  = load_images(TEST_COLOR_PATH, TEST_DEPTH_PATH)

# 4️⃣ 🔴 NORMALIZE & EXPAND DEPTH (YOUR CODE GOES HERE)
train_depth = train_depth.astype("float32") / 255.0
test_depth  = test_depth.astype("float32") / 255.0

train_depth = train_depth[..., np.newaxis]
test_depth  = test_depth[..., np.newaxis]

# 5️⃣ Load labels
train_labels = create_labels_from_filenames(TRAIN_COLOR_PATH)
test_labels  = create_labels_from_filenames(TEST_COLOR_PATH)


def build_depth_cnn(input_shape=(224, 224, 1)):
    inputs = Input(shape=input_shape)

    x = Conv2D(16, (3,3), activation='relu')(inputs)
    x = MaxPooling2D((2,2))(x)

    x = Conv2D(32, (3,3), activation='relu')(x)
    x = MaxPooling2D((2,2))(x)

    x = Conv2D(64, (3,3), activation='relu')(x)
    x = MaxPooling2D((2,2))(x)

    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)

    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs, outputs)
    return model
# 6️⃣ Build depth CNN
depth_model = build_depth_cnn(input_shape=(224,224,1))
depth_model = build_depth_cnn(input_shape=(224,224,1))

depth_model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

depth_model.summary()
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_labels),
    y=train_labels
)

class_weights = dict(enumerate(class_weights))
print("Class Weights:", class_weights)
history = depth_model.fit(
    train_depth,
    train_labels,
    validation_data=(test_depth, test_labels),
    epochs=10,
    batch_size=16,
    class_weight=class_weights
)
depth_model.save("model/depth_cnn.keras")
 # or "model/depth_cnn.keras"




