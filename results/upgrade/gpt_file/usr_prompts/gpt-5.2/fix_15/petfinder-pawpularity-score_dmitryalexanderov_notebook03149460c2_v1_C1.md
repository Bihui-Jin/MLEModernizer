# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

Not yielded

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 42.24638) has done: 'I fix the immediate runtime crash caused by an incompatible `protobuf`/TF import by forcing Python protobuf parsing before importing TensorFlow (a common Kaggle workaround for the `MessageFactory.GetPrototype` error). Then I keep the model/prediction flow the same but correct the submission validity issue by clipping predictions to the required `[1, 100]` range (instead of `[0, 100]`). I also make the input image preprocessing match EfficientNet’s expected scaling (`tf.keras.applications.efficientnet.preprocess_input`) to avoid systematically miscalibrated outputs, which should improve RMSE versus the current “not yielded” state while preserving the same architecture and inference approach. Finally, I keep paths unchanged and ensure a `submission.csv` is written with the exact required columns.'
- What this solution (achieved 42.24638) has done: 'I fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf implementation before any TensorFlow-related import, and by also setting the recommended `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION=3` plus doing the environment setup as the very first lines. I keep your model architecture, augmentation, and inference pipeline unchanged, but ensure the code runs end-to-end in Kaggle by making the import order and environment settings robust. I also keep the existing EfficientNet `preprocess_input` and `[1, 100]` clipping, as these are score-improving and metric-aligned while preserving the intended semantics. Finally, the script always write a valid `submission.csv` with the exact required columns.'
- What this solution (achieved 42.24638) has done: 'The crash happens before any of your code runs because TensorFlow is importing an incompatible protobuf C++ backend, so setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION` inside the notebook is too late. I fix this by forcing pure-Python protobuf via `sitecustomize.py` created at runtime (which Python loads before importing TensorFlow) and by also setting the env vars in the earliest possible place. This is a correctness/stability fix (score-neutral) that unblocks execution so your existing EfficientNet+tabular inference can run and write `submission.csv`. I keep the model, preprocessing, and clipping exactly as-is to preserve evaluation semantics and avoid unintended score shifts beyond getting a valid submission.'

# 9. Code solution

## === cell 0
import os, sys, textwrap, pathlib, glob, math, random, time

sitecustomize_path = pathlib.Path("sitecustomize.py")
sitecustomize_path.write_text(
    textwrap.dedent(
        """
        import os
        os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
        os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "3")
        """
    ).strip()
    + "\n"
)

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "3")

if "" not in sys.path and os.getcwd() not in sys.path:
    sys.path.insert(0, "")

if os.environ.get("__KAGGLE_TF_PROTOBUF_REEXEC_DONE__", "0") != "1":
    os.environ["__KAGGLE_TF_PROTOBUF_REEXEC_DONE__"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

SEED = 42
keras.utils.set_random_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

print("TF version:", tf.__version__)
print(
    "Using protobuf implementation:",
    os.environ.get("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"),
)

CANDIDATE_ROOTS = [
    "../input/petfinder-pawpularity-score",
    "/kaggle/input/petfinder-pawpularity-score",
]
DATA_DIR = None
for p in CANDIDATE_ROOTS:
    if os.path.exists(os.path.join(p, "train.csv")) and os.path.isdir(
        os.path.join(p, "train")
    ):
        DATA_DIR = p
        break
if DATA_DIR is None:
    raise FileNotFoundError(f"Could not locate dataset root. Tried: {CANDIDATE_ROOTS}")

TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test")

print("DATA_DIR:", DATA_DIR)
print("Train CSV exists:", os.path.exists(TRAIN_CSV))
print("Test CSV exists:", os.path.exists(TEST_CSV))
print("Train img dir exists:", os.path.isdir(TRAIN_IMG_DIR))
print("Test img dir exists:", os.path.isdir(TEST_IMG_DIR))




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

test = pd.read_csv(TEST_CSV)
test["file_path"] = test["Id"].apply(
    lambda identifier: os.path.join(TEST_IMG_DIR, f"{identifier}.jpg")
)

train = pd.read_csv(TRAIN_CSV)
train["file_path"] = train["Id"].apply(
    lambda identifier: os.path.join(TRAIN_IMG_DIR, f"{identifier}.jpg")
)

print("train/test shapes:", train.shape, test.shape)
print("train head Id:", train["Id"].iloc[0], "test head Id:", test["Id"].iloc[0])




## === cell 2
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
img_augmentation.trainable = False

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

test[tabular_columns] = test[tabular_columns].astype(np.float32)
train[tabular_columns] = train[tabular_columns].astype(np.float32)
train["Pawpularity"] = train["Pawpularity"].astype(np.float32)




## === cell 3
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input


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
        x = keras.layers.Dense(
            width, activation=activation, kernel_regularizer=kernel_regularizer
        )(x)
        if (i + 1) % 3 == 0:
            x = keras.layers.Concatenate()([x, inputs])
    return x


def get_model():
    image_inputs = layers.Input(shape=(image_size, image_size, 3))
    image_x = img_augmentation(image_inputs, training=False)

    model_backbone = EfficientNetB0(
        include_top=False,
        input_tensor=image_x,
        weights="imagenet",
        input_shape=(image_size, image_size, 3),
    )
    model_backbone.trainable = False

    image_x = layers.GlobalAveragePooling2D(name="avg_pool")(model_backbone.output)
    top_dropout_rate = 0.2
    image_x = layers.Dropout(top_dropout_rate, name="top_dropout")(image_x)

    tabular_inputs = tf.keras.Input(shape=(len(tabular_columns),), dtype=tf.float32)
    tabular_x = get_tabular_prediciton_model(tabular_inputs)

    x = tf.keras.layers.Concatenate(axis=1)([image_x, tabular_x])
    outputs = layers.Dense(1)(x)

    optimizer = tf.keras.optimizers.Adam(1e-3)
    model = tf.keras.Model(
        inputs=[image_inputs, tabular_inputs], outputs=[outputs], name="EfficientNet"
    )

    model.compile(optimizer=optimizer, loss="mse", metrics=[rmse, "mae", "mape"])
    return model


def preprocess_image_and_tabular(image_path, tabular, y=None):
    image_string = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.central_crop(image, 1.0)
    image = tf.image.resize(image, (image_size, image_size))
    image = tf.cast(image, tf.float32)
    image = preprocess_input(image)
    tabular = tf.cast(tabular, tf.float32)
    if y is None:
        return (image, tabular), 0.0
    y = tf.cast(y, tf.float32)
    return (image, tabular), y




## === cell 4
with strategy.scope():
    model_up = get_model()

WEIGHTS_CANDIDATES = [
    "../input/b0-weights/pet_trained_weights_v6.h5",
    "/kaggle/input/b0-weights/pet_trained_weights_v6.h5",
]
WEIGHTS_CANDIDATES += glob.glob(
    "/kaggle/input/**/pet_trained_weights_v6.h5", recursive=True
)
WEIGHTS_CANDIDATES += glob.glob(
    "/kaggle/input/**/pet_trained_weights*.h5", recursive=True
)
WEIGHTS_CANDIDATES += glob.glob(
    "/kaggle/input/**/pet_trained_weights*.keras", recursive=True
)

WEIGHTS_PATH = None
for p in WEIGHTS_CANDIDATES:
    if os.path.exists(p) and os.path.isfile(p):
        WEIGHTS_PATH = p
        break

loaded = False
if WEIGHTS_PATH is not None:
    try:
        model_up.load_weights(WEIGHTS_PATH)
        loaded = True
    except Exception as e1:
        print(
            "Direct load_weights failed, retrying with by_name=True, skip_mismatch=True"
        )
        try:
            model_up.load_weights(WEIGHTS_PATH, by_name=True, skip_mismatch=True)
            loaded = True
        except Exception as e2:
            print(
                f"Failed to load weights from {WEIGHTS_PATH}. Will fall back to training.\n"
                f"Direct error: {e1}\nFallback error: {e2}"
            )
            loaded = False

print("Custom weights path:", WEIGHTS_PATH)
print("Loaded custom weights ok:", loaded)




## === cell 5
options = tf.data.Options()
options.experimental_deterministic = True

if not loaded:
    VAL_SAMPLES = 2048
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(train))

    n_train = max(1, len(train) - VAL_SAMPLES)
    train_idx = perm[:n_train]
    val_idx = perm[n_train : n_train + min(VAL_SAMPLES, len(train) - n_train)]

    train_fit = train.iloc[train_idx].reset_index(drop=True)
    val_fit = train.iloc[val_idx].reset_index(drop=True)

    ds_train = (
        tf.data.Dataset.from_tensor_slices(
            (
                train_fit["file_path"].values,
                train_fit[tabular_columns].values,
                train_fit["Pawpularity"].values,
            )
        )
        .with_options(options)
        .shuffle(min(8192, len(train_fit)), seed=SEED, reshuffle_each_iteration=True)
        .map(
            lambda p, t, y: preprocess_image_and_tabular(p, t, y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    ds_val = (
        tf.data.Dataset.from_tensor_slices(
            (
                val_fit["file_path"].values,
                val_fit[tabular_columns].values,
                val_fit["Pawpularity"].values,
            )
        )
        .with_options(options)
        .map(
            lambda p, t, y: preprocess_image_and_tabular(p, t, y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    epochs = 20
    print(
        f"Training fallback head for {epochs} epochs on {len(train_fit)} samples; validating on {len(val_fit)} samples."
    )
    history = model_up.fit(ds_train, validation_data=ds_val, epochs=epochs, verbose=2)
    print(
        "Finished fallback training. Last val RMSE:",
        float(history.history["val_rmse"][-1]),
    )




## === cell 6
def _make_ds(df):
    return (
        tf.data.Dataset.from_tensor_slices(
            (
                df["file_path"].values,
                df[tabular_columns].values,
                df["Pawpularity"].values,
            )
        )
        .with_options(options)
        .map(
            lambda p, t, y: preprocess_image_and_tabular(p, t, y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )


def _predict_ds(ds):
    yt_list, yp_list = [], []
    for x_batch, y_batch in ds:
        pred = model_up(x_batch, training=False)
        yp_list.append(tf.reshape(pred, [-1]).numpy())
        yt_list.append(tf.reshape(y_batch, [-1]).numpy())
    return (
        np.concatenate(yt_list, axis=0).astype(np.float64),
        np.concatenate(yp_list, axis=0).astype(np.float64),
    )


CAL_VAL_SAMPLES = 4096  # larger holdout -> more stable shrinkage estimate
rng = np.random.default_rng(SEED)
perm = rng.permutation(len(train))
n_cal_train = max(1, len(train) - CAL_VAL_SAMPLES)
cal_tr_idx = perm[:n_cal_train]
cal_va_idx = perm[
    n_cal_train : n_cal_train + min(CAL_VAL_SAMPLES, len(train) - n_cal_train)
]

cal_val_df = train.iloc[cal_va_idx].reset_index(drop=True)

ds_cal_val = _make_ds(cal_val_df)
y_true_val, y_pred_val = _predict_ds(ds_cal_val)

mu = float(np.mean(y_true_val))

x = y_pred_val - mu
y = y_true_val - mu
den = float(np.dot(x, x))
alpha = float(np.dot(x, y) / den) if den > 0 else 0.0

alpha = float(np.clip(alpha, 0.0, 1.0))

rmse_val_before = float(np.sqrt(np.mean((y_true_val - y_pred_val) ** 2)))
y_pred_shrunk = alpha * y_pred_val + (1.0 - alpha) * mu
rmse_val_after = float(np.sqrt(np.mean((y_true_val - y_pred_shrunk) ** 2)))

print(
    f"Shrinkage fitted on held-out val ({len(y_true_val)}): alpha={alpha:.6f}, mu={mu:.6f}"
)
print(f"Val RMSE before/after shrinkage: {rmse_val_before:.4f} -> {rmse_val_after:.4f}")




## === cell 7
ds_test = (
    tf.data.Dataset.from_tensor_slices(
        (test["file_path"].values, test[tabular_columns].values)
    )
    .with_options(options)
    .map(
        lambda p, t: preprocess_image_and_tabular(p, t, None),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    .batch(batch_size)
    .prefetch(tf.data.AUTOTUNE)
)

pred_batches = []
for x_batch, _y_dummy in ds_test:
    pred_batches.append(model_up(x_batch, training=False))

res_0 = tf.concat(pred_batches, axis=0).numpy().reshape(-1)

if res_0.shape[0] != test.shape[0]:
    raise ValueError(
        f"Prediction count mismatch: got {res_0.shape[0]} preds but test has {test.shape[0]} rows."
    )

print("Raw preds:", res_0[:5], res_0.shape)

res_cal = (alpha * res_0 + (1.0 - alpha) * mu).astype(np.float32)
print("Shrunk preds:", res_cal[:5])




## === cell 8
predictions = np.clip(res_cal, 1.0, 100.0)

submission = pd.DataFrame(
    {"Id": test["Id"].values, "Pawpularity": predictions.astype(np.float32)}
)
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print("submission.csv exists:", os.path.exists("submission.csv"))
print(
    "Pawpularity min/max:",
    float(submission["Pawpularity"].min()),
    float(submission["Pawpularity"].max()),
)
