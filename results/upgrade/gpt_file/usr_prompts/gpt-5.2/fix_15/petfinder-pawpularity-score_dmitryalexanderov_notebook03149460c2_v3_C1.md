# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


# 1. Kaggle task description

## Task
Predict engagement with a pet's profile based on the photograph for that profile.

## Metric
Root mean squared error.

## Submission Format
For each `Id` in the test set, you must predict a probability for the target variable, `Pawpularity`. The file should contain a header and have the following format:

```
Id, Pawpularity
0008dbfb52aa1dc6ee51ee02adf13537, 99.24
0014a7b528f1682f0cf3b73a991c17a0, 61.71
0019c1388dfcd30ac8b112fb4250c251, 6.23
00307b779c82716b240a24f028b0031b, 9.43
00320c6dd5b4223c62a9670110d47911, 70.89
etc.
```

## Dataset
- **train/** - Folder containing training set photos of the form **{id}.jpg**, where **{id}** is a unique Pet Profile ID.
- **train.csv** - Metadata (described below) for each photo in the training set as well as the target, the photo's Pawpularity score. The Id column gives the photo's unique Pet Profile ID corresponding the photo's file name.

The train.csv and test.csv files contain metadata for photos in the training set and test set, respectively. Each pet photo is labeled with the value of 1 (Yes) or 0 (No) for each of the following features:

- **Focus** - Pet stands out against uncluttered background, not too close / far.
- **Eyes** - Both eyes are facing front or near-front, with at least 1 eye / pupil decently clear.
- **Face** - Decently clear face, facing front or near-front.
- **Near** - Single pet taking up significant portion of photo (roughly over 50% of photo width or height).
- **Action** - Pet in the middle of an action (e.g., jumping).
- **Accessory** - Accompanying physical or digital accessory / prop (i.e. toy, digital sticker), excluding collar and leash.
- **Group** - More than 1 pet in the photo.
- **Collage** - Digitally-retouched photo (i.e. with digital photo frame, combination of multiple photos).
- **Human** - Human in the photo.
- **Occlusion** - Specific undesirable objects blocking part of the pet (i.e. human, cage or fence). Note that not all blocking objects are considered occlusion.
- **Info** - Custom-added text or labels (i.e. pet name, description).
- **Blur** - Noticeably out of focus or noisy, especially for the pet's eyes and face. For Blur entries, "Eyes" column is always set to 0.

# 2. Python version

3.10

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (132 lines)
            sample_submission.csv (993 lines)
            sample_submission.csv.zip (22.7 kB)
            test.csv (993 lines)
            test.csv.zip (22.5 kB)
            test.zip (102.2 MB)
            train.csv (8921 lines)
            train.csv.zip (213.0 kB)
            train.zip (926.9 MB)
            petfinder-pawpularity-score/
                description.md (132 lines)
                sample_submission.csv (993 lines)
                ... and 7 other files
                petfinder-pawpularity-score/
                test/
                    a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                    2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                    ... and 990 other files
                    test/
                train/
                    e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                    cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                    ... and 8918 other files
                    train/
            test/
                a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                ... and 990 other files
                test/
            train/
                e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                ... and 8918 other files
                train/
        input/
            description.md (132 lines)
            sample_submission.csv (993 lines)
            sample_submission.csv.zip (22.7 kB)
            test.csv (993 lines)
            test.csv.zip (22.5 kB)
            test.zip (102.2 MB)
            train.csv (8921 lines)
            train.csv.zip (213.0 kB)
            train.zip (926.9 MB)
            petfinder-pawpularity-score/
                description.md (132 lines)
                sample_submission.csv (993 lines)
                ... and 7 other files
                petfinder-pawpularity-score/
                test/
                    a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                    2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                    ... and 990 other files
                    test/
                train/
                    e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                    cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                    ... and 8918 other files
                    train/
            test/
                a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                ... and 990 other files
                test/
                    a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                    2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                    ... and 990 other files
                    test/
            train/
                e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                ... and 8918 other files
                train/
                    e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                    cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                    ... and 8918 other files
                    train/
        working/
            petfinder-pawpularity-score/
                description.md (132 lines)
                sample_submission.csv (993 lines)
                ... and 7 other files
                petfinder-pawpularity-score/
                test/
                    a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                    2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                    ... and 990 other files
                    test/
                train/
                    e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                    cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                    ... and 8918 other files
                    train/
```

-> data/petfinder-pawpularity-score/sample_submission.csv has 992 rows and 2 columns.
The columns are: Id, Pawpularity

-> data/petfinder-pawpularity-score/test.csv has 992 rows and 13 columns.
The columns are: Id, Subject Focus, Eyes, Face, Near, Action, Accessory, Group, Collage, Human, Occlusion, Info, Blur

-> data/petfinder-pawpularity-score/train.csv has 8920 rows and 14 columns.
The columns are: Id, Subject Focus, Eyes, Face, Near, Action, Accessory, Group, Collage, Human, Occlusion, Info, Blur, Pawpularity

-> data/sample_submission.csv has 992 rows and 2 columns.
The columns are: Id, Pawpularity

-> data/test.csv has 992 rows and 13 columns.
The columns are: Id, Subject Focus, Eyes, Face, Near, Action, Accessory, Group, Collage, Human, Occlusion, Info, Blur

-> data/train.csv has 8920 rows and 14 columns.
The columns are: Id, Subject Focus, Eyes, Face, Near, Action, Accessory, Group, Collage, Human, Occlusion, Info, Blur, Pawpularity

-> (stopped after 10 files for performance)

# 5. Target score

18.870197063329275

# 6. Current score

42.24644

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 42.24644) has done: 'I fix the TensorFlow/protobuf crash by removing the environment override that forces the pure-Python protobuf implementation (it’s a known cause of `MessageFactory.GetPrototype` errors in some Kaggle images). Then I fix the dataset pipeline so the model receives both required inputs by returning `((image, tabular),)` from the `tf.data` map instead of `(image, tabular)` (which Keras interprets as `(x, y)`). Finally, I make the submission-writing cell robust so it always writes a valid `submission.csv` (even if weights are missing) and ensure predictions are shaped/clipped correctly.'
- What this solution (achieved 42.24644) has done: 'The crash happens before anything else because the code forces protobuf’s C++ implementation (`cpp`), but this Kaggle image’s `google.protobuf.pyext._message` extension is missing; removing that override (or explicitly selecting the pure-Python implementation) fixes TensorFlow import. After TensorFlow imports, the downstream `NameError`s for `tf`, `strategy`, and `res_0` go away because the earlier cell executes successfully. I also make the dataset feed Keras a proper multi-input tuple `((image, tabular),)` (not `(image, tabular)` which Keras can misinterpret), and I ensure we always write a valid `submission.csv` with correct columns and row alignment. These changes are execution/stability fixes and should allow producing a real submission; score depend on whether the weights file exists in your environment.'
- What this solution (achieved 42.24644) has done: 'I fix the TensorFlow/protobuf import crash by removing the forced protobuf implementation override (it’s triggering the `MessageFactory.GetPrototype` error in this environment). Then I keep the model/dataset/prediction logic the same, only making execution-stability tweaks: ensure TF imports cleanly, ensure the multi-input dataset yields the correct `(inputs,)` structure for Keras predict, and keep submission writing robust. These changes are primarily correctness/stability; they should also improve your score versus 42.25 because your current run likely never actually executed the intended TF graph/weights reliably due to the import crash.'
- What this solution (achieved 42.24644) has done: 'I fix the TensorFlow/protobuf import crash by forcing protobuf to use the pure-Python implementation before importing TensorFlow (this is the most common fix for the `MessageFactory.GetPrototype` error in Kaggle images). Then I keep your model/dataset logic unchanged, but make the image decode path use `tf.io.read_file` directly (faster and more reliable than `py_function`, and score-neutral) while preserving the same resize/normalization semantics. Finally, I ensure `model.predict` returns a flat `(N,)` array regardless of whether Keras returns a list or array, and I always write a valid `submission.csv` with the exact required columns.'
- What this solution (achieved 42.24644) has done: 'I fix the immediate runtime crash by removing the protobuf environment override that triggers TensorFlow’s `MessageFactory.GetPrototype` error in this Kaggle image, allowing TF to import cleanly. Then I keep your model and inference logic unchanged, but make the dataset map return `((image, tabular),)` consistently (multi-input) and ensure the decoded images are valid even if a file read fails. Finally, I keep submission writing robust and aligned to `test.csv`, guaranteeing a valid `submission.csv` with correct columns and shape. These changes are primarily execution/correctness fixes; any score change come from actually running the intended model end-to-end.'
- What this solution (achieved 42.24644) has done: 'I fix the TensorFlow/protobuf crash by forcing protobuf’s pure-Python implementation before importing TensorFlow, which is the most reliable way to avoid the `MessageFactory.GetPrototype` AttributeError in Kaggle images. I keep your model and inference logic unchanged, but make the image decode function robust by falling back to a zero image if JPEG decoding fails (so prediction can’t crash on a single bad file). I also ensure the multi-input `tf.data` pipeline continues to yield a proper `(inputs,)` tuple and that predictions are reshaped/clipped and written to an on-disk `submission.csv` with the exact required columns. These are execution-stability fixes; with your pretrained weights present they should move RMSE down substantially from the current 42.25 toward the target.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

print("TF version:", tf.__version__)
print("Eager:", tf.executing_eagerly())



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
image_size = 224
batch_size = 128

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()
    print("Device:", tpu.master())
    strategy = tf.distribute.TPUStrategy(tpu)
except Exception:
    print("Not connected to a TPU runtime. Using CPU/GPU strategy")
    strategy = tf.distribute.MirroredStrategy()

DATA_DIR = "../input/petfinder-pawpularity-score"
if not os.path.exists(DATA_DIR):
    alt = "/kaggle/input/petfinder-pawpularity-score"
    if os.path.exists(alt):
        DATA_DIR = alt

TEST_CSV = os.path.join(DATA_DIR, "test.csv")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test")

test = pd.read_csv(TEST_CSV)
test["file_path"] = test["Id"].apply(
    lambda identifier: os.path.join(TEST_IMG_DIR, f"{identifier}.jpg")
)

print("DATA_DIR:", DATA_DIR)
print("Test shape:", test.shape)
print(test.head())



## === cell 2
pass



## === cell 3
from tensorflow.keras.models import Sequential

img_augmentation = Sequential(
    [
        layers.RandomRotation(factor=0.15),
        layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
        layers.RandomFlip(),
        layers.RandomContrast(factor=0.1),
    ],
    name="img_augmentation",
)



## === cell 4
tabular_columns = [
    "Subject Focus",
    "Eyes",
    "Face",
    "Near",
    "Action",
    "Accessory",
    "Group",
    "Collage",
    "Human",
    "Occlusion",
    "Info",
    "Blur",
]


def rmse(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    return tf.sqrt(tf.reduce_mean((y_true - y_pred) ** 2))


def get_tabular_prediciton_model(inputs):
    width = 32
    depth = 3
    activation = "relu"
    kernel_regularizer = keras.regularizers.l2()
    x = keras.layers.Dense(
        width, activation=activation, kernel_regularizer=kernel_regularizer
    )(inputs)
    for i in range(depth):
        if i == 0:
            x = inputs
        x = keras.layers.Dense(
            width, activation=activation, kernel_regularizer=kernel_regularizer
        )(x)
        if (i + 1) % 3 == 0:
            x = keras.layers.Concatenate()([x, inputs])
    return x


def get_model():
    image_inputs = layers.Input(shape=(image_size, image_size, 3))
    image_x = image_inputs

    backbone = EfficientNetB0(
        include_top=False,
        input_tensor=image_x,
        weights="imagenet",
        input_shape=(image_size, image_size, 3),
    )
    backbone.trainable = False

    image_x = layers.GlobalAveragePooling2D(name="avg_pool")(backbone.output)
    top_dropout_rate = 0.2
    image_x = layers.Dropout(top_dropout_rate, name="top_dropout")(image_x)

    tabular_inputs = tf.keras.Input(
        shape=(len(tabular_columns),), name="tabular_inputs"
    )
    tabular_x = get_tabular_prediciton_model(tabular_inputs)

    x = tf.keras.layers.Concatenate(axis=1)([image_x, tabular_x])
    outputs = layers.Dense(1)(x)

    optimizer = tf.keras.optimizers.Adam(1e-3)
    model = tf.keras.Model(
        inputs=[image_inputs, tabular_inputs], outputs=[outputs], name="EfficientNet"
    )
    model.compile(optimizer=optimizer, loss=rmse, metrics=["mae", "mape"])
    return model


def _decode_and_resize(image_path):
    image_bytes = tf.io.read_file(image_path)

    def _decode():
        img = tf.image.decode_jpeg(image_bytes, channels=3)
        img = tf.image.central_crop(img, 1.0)
        img = tf.image.resize(img, (image_size, image_size))
        img = tf.cast(img, tf.float32) / 255.0
        return img

    def _fallback():
        return tf.zeros((image_size, image_size, 3), dtype=tf.float32)

    def _safe_decode():
        try:
            return _decode()
        except Exception:
            return _fallback()

    img = tf.py_function(_safe_decode, inp=[], Tout=tf.float32)
    img.set_shape((image_size, image_size, 3))
    return img


def preprocess_test_data(image_path, tabular):
    image = _decode_and_resize(image_path)
    tabular = tf.cast(tabular, tf.float32)
    return ((image, tabular),)


WEIGHTS_PATH = "../input/b0-weights/pet_trained_weights_v6.h5"
if not os.path.exists(WEIGHTS_PATH):
    alt_w = "/kaggle/input/b0-weights/pet_trained_weights_v6.h5"
    if os.path.exists(alt_w):
        WEIGHTS_PATH = alt_w

with strategy.scope():
    model_up = get_model()
    if os.path.exists(WEIGHTS_PATH):
        model_up.load_weights(WEIGHTS_PATH)
        print("Loaded weights:", WEIGHTS_PATH)
    else:
        print(
            "WARNING: weights file not found. Proceeding with imagenet backbone + random head:",
            WEIGHTS_PATH,
        )

tabular_np = test[tabular_columns].astype(np.float32).values
paths_np = test["file_path"].values.astype(str)

ds_try = (
    tf.data.Dataset.from_tensor_slices((paths_np, tabular_np))
    .map(preprocess_test_data, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(batch_size)
    .prefetch(tf.data.AUTOTUNE)
)

res_0 = model_up.predict(ds_try, verbose=1)

if isinstance(res_0, (list, tuple)):
    res_0 = res_0[0]
res_0 = np.asarray(res_0).reshape(-1)

print("Pred shape:", res_0.shape)
print("Pred sample:", res_0[:5])



## === cell 5
predictions = np.asarray(res_0).reshape(-1)

predictions = np.clip(predictions, 1.0, 100.0)
if len(predictions) != len(test):
    raise ValueError(f"Prediction length {len(predictions)} != test length {len(test)}")

submission = pd.DataFrame({"Id": test["Id"].values, "Pawpularity": predictions})
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print("submission.csv exists:", os.path.exists("submission.csv"))
