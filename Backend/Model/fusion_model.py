import os
import numpy as np
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense, Concatenate, Input
from tensorflow.keras.optimizers import Adam

from preprocessing.load_data import load_images
from preprocessing.labels import create_labels_from_filenames
# Load pretrained models
texture_model = load_model("model/texture_cnn.h5")
depth_model   = load_model("model/depth_cnn.keras")
texture_model.trainable = False
depth_model.trainable = False
# Remove last Dense layer
texture_feature_model = Model(
    inputs=texture_model.input,
    outputs=texture_model.layers[-2].output
)

depth_feature_model = Model(
    inputs=depth_model.input,
    outputs=depth_model.layers[-2].output
)
# Fusion inputs
rgb_input   = Input(shape=(224,224,3))
depth_input = Input(shape=(224,224,1))

# Feature extraction
rgb_features   = texture_feature_model(rgb_input)
depth_features = depth_feature_model(depth_input)

# Concatenate
fusion = Concatenate()([rgb_features, depth_features])

# Fusion classifier
x = Dense(128, activation="relu")(fusion)
x = Dense(64, activation="relu")(x)
output = Dense(1, activation="sigmoid")(x)

fusion_model = Model(
    inputs=[rgb_input, depth_input],
    outputs=output
)
fusion_model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

fusion_model.summary()
BASE_DIR = os.getcwd()

TRAIN_COLOR_PATH = os.path.join(BASE_DIR, "dataset", "train", "color")
TRAIN_DEPTH_PATH = os.path.join(BASE_DIR, "dataset", "train", "depth")

TEST_COLOR_PATH = os.path.join(BASE_DIR, "dataset", "test", "color")
TEST_DEPTH_PATH = os.path.join(BASE_DIR, "dataset", "test", "depth")

# Load data
train_color, train_depth = load_images(TRAIN_COLOR_PATH, TRAIN_DEPTH_PATH)
test_color, test_depth   = load_images(TEST_COLOR_PATH, TEST_DEPTH_PATH)

# Normalize
train_color = train_color.astype("float32") / 255.0
test_color  = test_color.astype("float32") / 255.0

train_depth = train_depth.astype("float32") / 255.0
test_depth  = test_depth.astype("float32") / 255.0

# Expand depth
train_depth = train_depth[..., np.newaxis]
test_depth  = test_depth[..., np.newaxis]

# Labels
train_labels = create_labels_from_filenames(TRAIN_COLOR_PATH)
test_labels  = create_labels_from_filenames(TEST_COLOR_PATH)
history = fusion_model.fit(
    [train_color, train_depth],
    train_labels,
    validation_data=([test_color, test_depth], test_labels),
    epochs=10,
    batch_size=16
)
loss, acc = fusion_model.evaluate(
    [test_color, test_depth],
    test_labels
)

print("Fusion Model Accuracy:", acc)
fusion_model.save("model/fusion_cnn.keras")

