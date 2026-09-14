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
Create a classifier to predict whether an image contains a cactus.

## Metric
Area under the ROC curve.

## Submission Format
For each ID in the test set, you must predict a probability for the `has_cactus` variable. The file should contain a header and have the following format:

```
id,has_cactus
000940378805c44108d287872b2f04ce.jpg,0.5
0017242f54ececa4512b4d7937d1e21e.jpg,0.5
001ee6d8564003107853118ab87df407.jpg,0.5
etc.
```

## Dataset
This dataset contains a large number of 32 x 32 thumbnail images containing aerial photos of a cactus. The file name of an image corresponds to its `id`.

- **train/** - the training set images
- **test/** - the test set images (you must predict the labels of these)
- **train.csv** - the training set labels, indicates whether the image has a cactus (`has_cactus = 1`)
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.7

# 3. Installed packages

geopandas==0.14.4
imutils==0.5.4
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
        input/
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
        working/
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
```

-> data/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> data/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> (stopped after 10 files for performance)

# 5. Target score

0.99965

# 6. Current score

None

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import sys
import subprocess
import numpy as np
import pandas as pd

CANDIDATE_ROOTS = [
    "/kaggle/input/aerial-cactus-identification",
    "/kaggle/input/aerial-cactus-identification/aerial-cactus-identification",
]
INPUT_ROOT = None
for r in CANDIDATE_ROOTS:
    if os.path.exists(os.path.join(r, "train.csv")) and os.path.isdir(
        os.path.join(r, "train")
    ):
        INPUT_ROOT = r
        break
if INPUT_ROOT is None:
    base = "/kaggle/input"
    for name in os.listdir(base):
        r = os.path.join(base, name)
        if (
            os.path.isdir(r)
            and os.path.exists(os.path.join(r, "train.csv"))
            and os.path.isdir(os.path.join(r, "train"))
        ):
            INPUT_ROOT = r
            break

print("Listing /kaggle/input:", os.listdir("/kaggle/input")[:50])
print("Using dataset root:", INPUT_ROOT)
print(
    "Files:",
    [
        p
        for p in ["train.csv", "sample_submission.csv", "train", "test"]
        if INPUT_ROOT and os.path.exists(os.path.join(INPUT_ROOT, p))
    ],
)
if INPUT_ROOT is None:
    raise FileNotFoundError(
        "Could not locate dataset root containing train.csv/train/ and test/."
    )



## === cell 1
print("Root contents:", os.listdir(INPUT_ROOT)[:50])
print("Train dir exists:", os.path.isdir(os.path.join(INPUT_ROOT, "train")))
print("Test dir exists:", os.path.isdir(os.path.join(INPUT_ROOT, "test")))



## === cell 2
import google.protobuf  # noqa: E402

pb_ver = getattr(google.protobuf, "__version__", "unknown")
print("Current protobuf version:", pb_ver)


def _major(v):
    try:
        return int(str(v).split(".")[0])
    except Exception:
        return None


if _major(pb_ver) is not None and _major(pb_ver) >= 5:
    print("Downgrading protobuf to 4.25.3 for TensorFlow compatibility...")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "protobuf==4.25.3"]
    )
    import importlib

    importlib.invalidate_caches()
    for k in list(sys.modules.keys()):
        if k.startswith("google.protobuf"):
            del sys.modules[k]
    import google.protobuf as gp2  # noqa: E402

    print("Protobuf after downgrade:", getattr(gp2, "__version__", "unknown"))



## === cell 3
import tensorflow as tf
from tensorflow import keras

print("TensorFlow version:", tf.__version__)




## === cell 4
def focal_loss_fn(gamma=2.0, alpha=0.25):
    EPSILON = 1e-6

    def ce(y_true, y_pred, weights=None):
        mask = y_pred < EPSILON
        true_vals = tf.fill(tf.shape(y_pred), EPSILON)
        y_pred = tf.where(mask, true_vals, y_pred)
        ce = y_true * (-tf.math.log(y_pred)) + (1 - y_true) * (-tf.math.log(1 - y_pred))

        if weights is not None:
            ce = ce * weights

        ce_loss = tf.reduce_mean(ce + EPSILON)
        return ce_loss

    def focal_loss_fixed(y_true, y_pred):
        t = y_true
        p = y_pred

        pt = p * t + (1 - p) * (1 - t)
        w = alpha * t + (1 - alpha) * (1 - t)
        w = tf.pow((1 - pt), gamma)

        fl = ce(y_true, y_pred, w)

        return fl

    return focal_loss_fixed




## === cell 5
IMG_SIZE = 96
BATCH_SIZE = 128
EPOCHS = 8  # keep modest for runtime; dataset is easy and converges quickly
SEED = 42

train_csv_path = os.path.join(INPUT_ROOT, "train.csv")
train_dir = os.path.join(INPUT_ROOT, "train")
test_dir = os.path.join(INPUT_ROOT, "test")
sample_sub_path = os.path.join(INPUT_ROOT, "sample_submission.csv")

train_df = pd.read_csv(train_csv_path)
sample_sub = pd.read_csv(sample_sub_path)

assert set(train_df.columns) == {"id", "has_cactus"}
assert set(sample_sub.columns) == {"id", "has_cactus"}

train_df["filepath"] = train_df["id"].apply(lambda x: os.path.join(train_dir, x))
print("Train rows:", len(train_df), "Pos rate:", train_df["has_cactus"].mean())

tf.keras.utils.set_random_seed(SEED)



## === cell 6
import cv2


def load_and_preprocess(path, label=None, img_size=96):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    if label is None:
        return img
    return img, np.float32(label)




## === cell 7
test_imgs_dir = test_dir
print("Test imgs dir:", test_imgs_dir)
print("Num test files:", len(os.listdir(test_imgs_dir)))



## === cell 8
from imutils import paths

img_paths = sorted(list(paths.list_images(test_imgs_dir)))
print("Found test images:", len(img_paths))
print("First test path:", img_paths[0] if img_paths else None)



## === cell 9
print("Sample submission rows:", len(sample_sub))
if len(img_paths) != len(sample_sub):
    print(
        "WARNING: test images count != sample submission rows; will align via sample_submission IDs."
    )



## === cell 10
X = np.zeros((len(train_df), IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
y = train_df["has_cactus"].values.astype(np.float32)

for i, (fp, lab) in enumerate(zip(train_df["filepath"].values, y)):
    X[i] = load_and_preprocess(fp, img_size=IMG_SIZE)

print("X shape:", X.shape, "y shape:", y.shape, "X range:", (X.min(), X.max()))



## === cell 11
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.15, random_state=SEED, stratify=y
)
print("Train:", X_train.shape, "Val:", X_val.shape)



## === cell 12
inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
x = keras.layers.MaxPooling2D()(x)
x = keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = keras.layers.MaxPooling2D()(x)
x = keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
x = keras.layers.MaxPooling2D()(x)
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(128, activation="relu")(x)
x = keras.layers.Dropout(0.3)(x)
outputs = keras.layers.Dense(1, activation="sigmoid")(x)

model = keras.Model(inputs, outputs)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=focal_loss_fn(gamma=2.0, alpha=0.25),
    metrics=[keras.metrics.AUC(name="auc")],
)
model.summary()



## === cell 13
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=2,
)



## === cell 14
test_id_to_path = {os.path.basename(p): p for p in img_paths}
missing = [i for i in sample_sub["id"].values if i not in test_id_to_path]
if missing:
    raise FileNotFoundError(
        f"Some sample_submission ids not found in test dir, e.g. {missing[:5]}"
    )

X_test = np.zeros((len(sample_sub), IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
for idx, img_id in enumerate(sample_sub["id"].values):
    X_test[idx] = load_and_preprocess(test_id_to_path[img_id], img_size=IMG_SIZE)

print("X_test shape:", X_test.shape, "range:", (X_test.min(), X_test.max()))



## === cell 15
predictions = model.predict(X_test, batch_size=BATCH_SIZE, verbose=0).reshape(-1)
predictions = np.clip(predictions.astype(np.float64), 1e-7, 1 - 1e-7)
print(
    "Pred shape:",
    predictions.shape,
    "min/max:",
    float(predictions.min()),
    float(predictions.max()),
)



## === cell 16
submit_csv = pd.DataFrame({"id": sample_sub["id"].values, "has_cactus": predictions})

submit_path = "submission.csv"
submit_csv.to_csv(submit_path, index=False)
print("Wrote:", submit_path)
print(submit_csv.head())



## === cell 17
assert submit_csv.shape[0] == len(sample_sub)
assert list(submit_csv.columns) == ["id", "has_cactus"]
assert submit_path.endswith(".csv")
