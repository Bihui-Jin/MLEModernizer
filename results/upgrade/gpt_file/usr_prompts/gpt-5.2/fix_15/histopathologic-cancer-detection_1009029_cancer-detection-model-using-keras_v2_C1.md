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
Given a dataset of images from digital pathology scans, predict if the center 32x32px region of a patch contains at least one pixel of tumor tissue. Tumor tissue in the outer region of the patch does not influence the label. 

## Metric
Area under the ROC curve.

## Submission Format
For each `id` in the test set, you must predict a probability that center 32x32px region of a patch contains at least one pixel of tumor tissue. The file should contain a header and have the following format:

```
id,label
0b2ea2a822ad23fdb1b5dd26653da899fbd2c0d5,0
95596b92e5066c5c52466c90b69ff089b39f2737,0
248e6738860e2ebcf6258cdc1f32f299e0c76914,0
etc.
```

## Dataset
Files are named with an image `id`. The `train_labels.csv` file provides the ground truth for the images in the `train` folder. You are predicting the labels for the images in the `test` folder.

# 2. Python version

3.7

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (63 lines)
            sample_submission.csv (45562 lines)
            sample_submission.csv.zip (1.1 MB)
            test.zip (1.1 GB)
            train.zip (4.2 GB)
            train_labels.csv (174465 lines)
            train_labels.csv.zip (4.2 MB)
            histopathologic-cancer-detection/
                description.md (63 lines)
                sample_submission.csv (45562 lines)
                ... and 5 other files
                histopathologic-cancer-detection/
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
            test/
                7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                ... and 45559 other files
                test/
            train/
                bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                ... and 174462 other files
                train/
        input/
            description.md (63 lines)
            sample_submission.csv (45562 lines)
            sample_submission.csv.zip (1.1 MB)
            test.zip (1.1 GB)
            train.zip (4.2 GB)
            train_labels.csv (174465 lines)
            train_labels.csv.zip (4.2 MB)
            histopathologic-cancer-detection/
                description.md (63 lines)
                sample_submission.csv (45562 lines)
                ... and 5 other files
                histopathologic-cancer-detection/
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
            test/
                7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                ... and 45559 other files
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
            train/
                bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                ... and 174462 other files
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
        working/
            histopathologic-cancer-detection/
                description.md (63 lines)
                sample_submission.csv (45562 lines)
                ... and 5 other files
                histopathologic-cancer-detection/
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
```

-> data/histopathologic-cancer-detection/sample_submission.csv has 45561 rows and 2 columns.
The columns are: id, label

-> data/histopathologic-cancer-detection/train_labels.csv has 174464 rows and 2 columns.
The columns are: id, label

-> data/sample_submission.csv has 45561 rows and 2 columns.
The columns are: id, label

-> data/train_labels.csv has 174464 rows and 2 columns.
The columns are: id, label

-> input/histopathologic-cancer-detection/sample_submission.csv has 45561 rows and 2 columns.
The columns are: id, label

-> input/histopathologic-cancer-detection/train_labels.csv has 174464 rows and 2 columns.
The columns are: id, label

-> (stopped after 10 files for performance)

# 5. Target score

0.7793147687630658

# 6. Current score

0.89993

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.55502) has done: 'The main timeout is caused by enumerating ~174k training files with `glob()` and then doing a full `merge`, which is unnecessary since you only ever sample 10k labeled images; this dominates runtime before training even starts. I replace that with an O(N) path construction directly from the sampled IDs (no filesystem scan), keeping the same sampling logic (5000 per class with the same seeds) and the same model/training code. I also avoid expensive plotting work (which doesn’t affect outputs) and keep the existing tf.data test-time pipeline intact. These changes are provably equivalent for the selected training subset and preserve the model, training loop, and evaluation semantics.'
- What this solution (achieved 0.90208) has done: 'I fix the two root-cause runtime errors that prevent the notebook from running end-to-end: the protobuf/TensorFlow incompatibility triggered by forcing pure-Python protobuf, and the Keras `ModelCheckpoint` filename requirement when `save_weights_only=True`. Once the model trains, the downstream `NameError`/`KeyError` issues disappear because `model` be created and `test_df["label"]` be populated as intended. I also keep the existing core model/training logic intact and ensure a valid `submission.csv` with columns `id,label` is written in the working directory. These changes are execution/stability fixes and should move the score up substantially from “no submission produced” toward the target.'
- What this solution (achieved 0.5) has done: 'I fix the TensorFlow/protobuf runtime error by forcing the pure-Python protobuf implementation (the Kaggle TF build for this competition environment expects it), which is what’s currently causing the notebook to crash before training. I keep the model/training/inference logic the same and only adjust environment setup ordering so the import-time protobuf/TensorFlow initialization is stable. Since your current score (0.90208) is already well above the target (0.7793) and within the allowed tolerance band, I not make any score-oriented changes—only stability fixes to ensure it runs end-to-end and reliably writes `submission.csv`.'
- What this solution (achieved 0.87165) has done: 'I fix the TensorFlow/protobuf crash by removing the forced pure-Python protobuf setting and instead forcing the C++ implementation (which is what TF expects in this environment), keeping the rest of the pipeline unchanged. I also make the TIFF decoding fallback robust by using `tf.io.decode_image` if `skimage` fails inside `tf.py_function`, so inference doesn’t silently break on any odd files. Finally, I keep the same model/training logic but add a safety guard to always produce a valid `submission.csv` with the required `id,label` columns even if some test paths are missing. These changes should restore training/inference (moving score well above 0.5 toward your target) while preserving the core approach.'
- What this solution (achieved 0.5) has done: 'I fix the crash in the first cell caused by an incompatible protobuf/TensorFlow combination by forcing TensorFlow to use the pure-Python protobuf implementation *before* importing TensorFlow. I keep the model, training loop, sampling, and inference logic unchanged so the evaluation semantics stay the same (and thus score should remain close to your current 0.87165). I also add a small safety fallback so that if `skimage` import/reading fails in this environment, the pipeline still decodes TIFFs via TensorFlow, ensuring end-to-end execution and a valid `submission.csv` is always written.'
- What this solution (achieved 0.89298) has done: 'I fix the TensorFlow/protobuf crash causing the notebook to fail before training by removing the forced pure-Python protobuf setting (this environment’s TF expects the default/C++ protobuf). I keep the model, sampling, training loop, and inference logic the same, only adding a small defensive fallback so that if the TF import still fails, it retries with the pure-Python protobuf setting to keep execution robust. This should restore end-to-end execution and produce a real (non-0.5-constant) submission, moving the score up toward your target ROC-AUC. I also keep the required `submission.csv` output with `id,label` exactly as specified.'
- What this solution (achieved 0.89993) has done: 'I fix the TensorFlow/protobuf import crash (`MessageFactory` missing `GetPrototype`) by forcing the pure-Python protobuf implementation *before* importing TensorFlow, which is the known-safe setting for this specific Kaggle environment. I keep the rest of your data sampling, model architecture, training loop, and inference pipeline unchanged so the evaluation semantics stay the same (and since your current score is already above target, we avoid score-oriented changes). I also make the TF import path deterministic (no retry branch that can leave partially-initialized protobuf state) and keep the submission writing exactly as required (`submission.csv` with `id,label`). The output run end-to-end and always produce a valid `.csv` submission.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

from glob import glob
import gc
import multiprocessing
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

try:
    from skimage.io import imread as sk_imread
except Exception as e:
    sk_imread = None
    print(
        "skimage import unavailable; will use TF decode fallback only. Reason:", repr(e)
    )

import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dropout, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Sequential

print("TF version:", tf.__version__)
print(
    "Listing ../input:",
    os.listdir("../input")[:20] if os.path.exists("../input") else "no ../input",
)


def pick_existing(*paths):
    for p in paths:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("None of these paths exist: " + str(paths))


DATA_ROOT = pick_existing(
    "../input/histopathologic-cancer-detection",
    "../input",
    "/kaggle/input/histopathologic-cancer-detection",
    "/kaggle/input",
)

TRAIN_LABELS_CSV = pick_existing(
    os.path.join(DATA_ROOT, "train_labels.csv"),
    "../input/train_labels.csv",
    "/kaggle/input/train_labels.csv",
)
SAMPLE_SUB_CSV = pick_existing(
    os.path.join(DATA_ROOT, "sample_submission.csv"),
    "../input/sample_submission.csv",
    "/kaggle/input/sample_submission.csv",
)

TRAIN_DIR = pick_existing(
    os.path.join(DATA_ROOT, "train"),
    "../input/train",
    "/kaggle/input/train",
)
TEST_DIR = pick_existing(
    os.path.join(DATA_ROOT, "test"),
    "../input/test",
    "/kaggle/input/test",
)

print("Using DATA_ROOT:", DATA_ROOT)
print("TRAIN_LABELS_CSV:", TRAIN_LABELS_CSV)
print("SAMPLE_SUB_CSV:", SAMPLE_SUB_CSV)
print("TRAIN_DIR:", TRAIN_DIR)
print("TEST_DIR:", TEST_DIR)

np.random.seed(42)
tf.random.set_seed(42)

try:
    tf.config.threading.set_intra_op_parallelism_threads(
        max(1, multiprocessing.cpu_count() // 2)
    )
    tf.config.threading.set_inter_op_parallelism_threads(2)
except Exception as e:
    print("Threading config skipped:", repr(e))

try:
    tf.config.experimental.enable_op_determinism()
except Exception as e:
    print("Determinism config skipped:", repr(e))


def _decode_tiff_py(path_bytes):
    """Decode via skimage in a py_function; outputs uint8 tensor [96,96,3].
    Fallback: if skimage fails/unavailable, use TF decode_image from bytes.
    """

    def _read(p):
        if hasattr(p, "numpy"):
            p = p.numpy()
        if isinstance(p, (bytes, bytearray)):
            p = p.decode("utf-8")
        else:
            p = str(p)

        if sk_imread is not None:
            try:
                img = sk_imread(p)
                if img.ndim == 2:
                    img = np.stack([img, img, img], axis=-1)
                if img.shape[-1] > 3:
                    img = img[:, :, :3]
                return img.astype(np.uint8)
            except Exception:
                pass

        raw = tf.io.read_file(p)
        img_tf = tf.io.decode_image(raw, channels=3, expand_animations=False)
        img_tf = tf.image.resize(img_tf, [96, 96], method="nearest")
        img_np = img_tf.numpy().astype(np.uint8)
        return img_np

    img = tf.py_function(_read, [path_bytes], Tout=tf.uint8)
    img = tf.ensure_shape(img, [96, 96, 3])
    return img


def make_image_ds(paths, batch_size, shuffle=False, seed=42):
    """Deterministic tf.data pipeline => float32 images normalized to [0,1]."""
    paths = np.asarray(paths, dtype=str)
    ds = tf.data.Dataset.from_tensor_slices(
        tf.convert_to_tensor(paths, dtype=tf.string)
    )
    if shuffle:
        ds = ds.shuffle(
            buffer_size=len(paths), seed=seed, reshuffle_each_iteration=False
        )

    options = tf.data.Options()
    options.experimental_deterministic = True
    ds = ds.with_options(options)

    ds = ds.map(_decode_tiff_py, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(
        lambda x: tf.cast(x, tf.float32) / 255.0, num_parallel_calls=tf.data.AUTOTUNE
    )
    ds = ds.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
    return ds


def read_image_np(path):
    """Consistent image read/shape handling with the tf.data pipeline."""
    if sk_imread is not None:
        img = sk_imread(path)
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        if img.shape[-1] > 3:
            img = img[:, :, :3]
        return img.astype(np.uint8)

    raw = tf.io.read_file(path)
    img_tf = tf.io.decode_image(raw, channels=3, expand_animations=False)
    img_tf = tf.image.resize(img_tf, [96, 96], method="nearest")
    return img_tf.numpy().astype(np.uint8)


def load_images_parallel(paths, max_workers=None):
    paths = list(map(str, paths))
    n = len(paths)
    out = np.empty((n, 96, 96, 3), dtype=np.float32)

    if max_workers is None:
        max_workers = min(16, max(4, multiprocessing.cpu_count()))

    def _load_one(i_p):
        i, p = i_p
        img = read_image_np(p).astype(np.float32) / 255.0
        return i, img

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for i, img in ex.map(_load_one, enumerate(paths), chunksize=64):
            out[i] = img

    return out




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_df = pd.read_csv(TRAIN_LABELS_CSV)
train_df.head()



## === cell 2
train_df.label.unique()



## === cell 3
distribution = train_df.label.value_counts()
print(distribution)
p = distribution[1] / distribution.sum()
print("Percentage of cancer affected cells are {}".format(p))



## === cell 4
try:
    label_counts = train_df["label"].value_counts()
    fig, ax1 = plt.subplots(1, 1, figsize=(12, 8))
    ax1.bar(np.arange(len(label_counts)) + 0.5, label_counts)
    ax1.set_xticks(np.arange(len(label_counts)) + 0.5)
    _ = ax1.set_xticklabels(label_counts.index, rotation=90)
    plt.close(fig)
except Exception as e:
    print("Plotting skipped:", repr(e))



## === cell 5
labels = pd.read_csv(TRAIN_LABELS_CSV)
df0 = labels[labels.label == 0].sample(5000, random_state=42)
df1 = labels[labels.label == 1].sample(5000, random_state=42)
df = pd.concat([df0, df1], ignore_index=True).reset_index(drop=True)
df["path"] = df["id"].map(lambda _id: os.path.join(TRAIN_DIR, f"{_id}.tif"))
df = df[["path", "id", "label"]]
df.head(10)



## === cell 6
missing_train = (~df["path"].map(os.path.exists)).sum()
print("Sampled train images:", len(df), "Missing files:", int(missing_train))



## === cell 7
train_paths = df["path"].values.astype(str)

input_images = load_images_parallel(train_paths)

plot_idx = np.array([0, 1, 2, 5000, 5001, 5002], dtype=int)
plot_images = (input_images[plot_idx] * 255.0).astype(np.uint8)
df.sample(3)



## === cell 8
try:
    images = [
        (plot_images[0], int(df["label"].iloc[0])),
        (plot_images[1], int(df["label"].iloc[1])),
        (plot_images[2], int(df["label"].iloc[2])),
        (plot_images[3], int(df["label"].iloc[5000])),
        (plot_images[4], int(df["label"].iloc[5001])),
        (plot_images[5], int(df["label"].iloc[5002])),
    ]

    fig, m_axs = plt.subplots(1, len(images), figsize=(20, 2))
    for ii, c_ax in enumerate(m_axs):
        c_ax.imshow(images[ii][0])
        c_ax.set_title(images[ii][1])
        c_ax.axis("off")
    plt.close(fig)
except Exception as e:
    print("Plotting skipped:", repr(e))



## === cell 9
input_images.shape



## === cell 10
x = input_images
y = df["label"].astype(np.int32).values

train_x, test_x, train_y, test_y = train_test_split(
    x, y, test_size=0.10, random_state=101, stratify=y
)

train_y.shape



## === cell 11
np.random.seed(42)
tf.random.set_seed(42)

early_stopping = EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=False
)

WEIGHTS_PATH = "weights.weights.h5"
checkpointer = ModelCheckpoint(
    filepath=WEIGHTS_PATH, verbose=1, save_best_only=True, save_weights_only=True
)

model = Sequential()
model.add(
    Conv2D(
        filters=16,
        kernel_size=3,
        padding="same",
        activation="relu",
        input_shape=(96, 96, 3),
    )
)
model.add(Conv2D(filters=16, kernel_size=3, padding="same", activation="relu"))
model.add(Conv2D(filters=16, kernel_size=3, padding="same", activation="relu"))
model.add(Dropout(0.3))
model.add(MaxPooling2D(pool_size=3))

model.add(Conv2D(filters=32, kernel_size=3, padding="same", activation="relu"))
model.add(Conv2D(filters=32, kernel_size=3, padding="same", activation="relu"))
model.add(Conv2D(filters=32, kernel_size=3, padding="same", activation="relu"))
model.add(Dropout(0.3))
model.add(MaxPooling2D(pool_size=3))

model.add(Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"))
model.add(Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"))
model.add(Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"))
model.add(Dropout(0.3))
model.add(MaxPooling2D(pool_size=3))

model.add(Conv2D(filters=128, kernel_size=3, padding="same", activation="elu"))
model.add(Conv2D(filters=128, kernel_size=3, padding="same", activation="elu"))
model.add(Conv2D(filters=256, kernel_size=3, padding="same", activation="elu"))

model.add(Flatten())
model.add(Dense(1, activation="sigmoid"))

model.summary()



## === cell 12
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

epochs = 15
history = model.fit(
    train_x,
    train_y,
    validation_data=(test_x, test_y),
    epochs=epochs,
    batch_size=80,
    verbose=1,
    callbacks=[early_stopping, checkpointer],
)

print("Checkpoint exists after training:", os.path.exists(WEIGHTS_PATH))



## === cell 13
if os.path.exists(WEIGHTS_PATH):
    model.load_weights(WEIGHTS_PATH)

val_pred = model.predict(test_x, batch_size=256, verbose=0).ravel()
val_auc = roc_auc_score(test_y, val_pred)
print("Validation ROC-AUC:", val_auc)



## === cell 14
sample_sub = pd.read_csv(SAMPLE_SUB_CSV)
test_df = sample_sub[["id"]].copy()
test_df["path"] = test_df["id"].map(lambda _id: os.path.join(TEST_DIR, f"{_id}.tif"))

missing = (~test_df["path"].map(os.path.exists)).sum()
print("Test rows:", len(test_df), "Missing paths:", int(missing))
test_df.head()



## === cell 15
n = len(test_df)
preds = np.full(n, 0.5, dtype=np.float32)

valid_mask = test_df["path"].map(os.path.exists).values
valid_paths = test_df.loc[valid_mask, "path"].values.astype(str)

if len(valid_paths) > 0:
    infer_bs = 512
    test_ds = make_image_ds(valid_paths, batch_size=infer_bs, shuffle=False, seed=42)
    valid_preds = model.predict(test_ds, verbose=0).ravel().astype(np.float32)
    preds[valid_mask] = valid_preds
    del test_ds, valid_preds
    gc.collect()

test_df["label"] = preds
test_df[["id", "label"]].head()



## === cell 16
submission = test_df[["id", "label"]].copy()
submission.to_csv("submission.csv", index=False, header=True)

print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
print("submission.csv exists:", os.path.exists("submission.csv"))
print(
    "submission.csv size (bytes):",
    os.path.getsize("submission.csv") if os.path.exists("submission.csv") else None,
)
