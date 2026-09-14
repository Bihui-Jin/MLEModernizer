# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

3.14

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

# 5. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)


def find_hcd_root(base="/kaggle/input"):
    base = Path(base)
    candidates = [
        base / "histopathologic-cancer-detection",
        base / "competitions" / "histopathologic-cancer-detection",
    ]
    for c in candidates:
        if (c / "train_labels.csv").exists() and (c / "sample_submission.csv").exists():
            return c

    for p in base.rglob("train_labels.csv"):
        if p.name == "train_labels.csv":
            root = p.parent
            if (root / "sample_submission.csv").exists():
                return root

    raise FileNotFoundError(
        "Could not locate histopathologic-cancer-detection dataset under /kaggle/input. "
        "Expected train_labels.csv and sample_submission.csv."
    )


DATA_ROOT = find_hcd_root("/kaggle/input")
TRAIN_CSV = DATA_ROOT / "train_labels.csv"
SAMPLE_SUB_CSV = DATA_ROOT / "sample_submission.csv"

if (DATA_ROOT / "train").is_dir() and (DATA_ROOT / "test").is_dir():
    TRAIN_DIR = DATA_ROOT / "train"
    TEST_DIR = DATA_ROOT / "test"
elif (DATA_ROOT / "histopathologic-cancer-detection" / "train").is_dir():
    TRAIN_DIR = DATA_ROOT / "histopathologic-cancer-detection" / "train"
    TEST_DIR = DATA_ROOT / "histopathologic-cancer-detection" / "test"
else:
    train_dirs = [p for p in DATA_ROOT.rglob("train") if p.is_dir()]
    test_dirs = [p for p in DATA_ROOT.rglob("test") if p.is_dir()]
    TRAIN_DIR = train_dirs[0] if train_dirs else None
    TEST_DIR = test_dirs[0] if test_dirs else None
    if TRAIN_DIR is None or TEST_DIR is None:
        raise FileNotFoundError(
            "Could not locate train/ and test/ directories for images."
        )

print("DATA_ROOT:", str(DATA_ROOT))
print("TRAIN_CSV:", str(TRAIN_CSV))
print("SAMPLE_SUB_CSV:", str(SAMPLE_SUB_CSV))
print("TRAIN_DIR:", str(TRAIN_DIR))
print("TEST_DIR:", str(TEST_DIR))



## === cell 1
import os


def _count_tifs(folder: Path):
    try:
        return sum(1 for p in folder.iterdir() if p.suffix == ".tif")
    except Exception:
        return len(list(folder.glob("*.tif")))


for d in [TRAIN_DIR, TEST_DIR]:
    n = _count_tifs(Path(d))
    print(f"Total number of .tif files in {d}: {n}")
    for i, p in enumerate(Path(d).glob("*.tif")):
        if i >= 3:
            break
        print(str(p))



## === cell 2
train_labels = pd.read_csv(TRAIN_CSV)
print("shape:", train_labels.shape)
print(train_labels.head())
print("Missing values:\n", train_labels.isnull().sum())
print(f"Sum of duplicated labels: {train_labels.duplicated().sum()}.")

train_labels["train_filepath"] = train_labels["id"].apply(
    lambda x: str(TRAIN_DIR / f"{x}.tif")
)

missing = 0
for p in train_labels["train_filepath"].head(20):
    if not Path(p).exists():
        missing += 1
print("Missing among first 20 train images:", missing)



## === cell 3
try:
    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.countplot(x="label", data=train_labels)
    plt.xticks([0, 1], ["Non-Cancer (0)", "Cancer (1)"])
    plt.title("Distribution of Cancer vs Non-Cancer Images")
    plt.ylabel("Image Count")
    plt.xlabel("Label")
    plt.show()
except Exception as e:
    print("Plot skipped:", repr(e))



## === cell 4
try:
    import cv2
    import matplotlib.pyplot as plt

    def show_samples(label, df=train_labels, num_images=5):
        subset = df[df["label"] == label].sample(num_images, random_state=SEED)
        plt.figure(figsize=(12, 4))
        for i, row in enumerate(subset.itertuples()):
            img = cv2.imread(row.train_filepath)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            center = (32, 32, 64, 64)
            cv2.rectangle(
                img, (center[0], center[1]), (center[2], center[3]), (255, 0, 0), 1
            )
            plt.subplot(1, num_images, i + 1)
            plt.imshow(img)
            plt.title(f"Label: {row.label}")
            plt.axis("off")
        plt.show()

    show_samples(0)
    show_samples(1)
except Exception as e:
    print("Sample viz skipped:", repr(e))



## === cell 5
import warnings

warnings.filterwarnings("ignore")

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import tensorflow as tf

print("TensorFlow version:", tf.__version__)
print("GPU available:", tf.config.list_physical_devices("GPU"))

try:
    tf.keras.utils.set_random_seed(SEED)
except Exception:
    pass


def _decode_tif_to_float01(path_bytes):
    import numpy as _np
    from PIL import Image as _Image

    if isinstance(path_bytes, (bytes, bytearray)):
        p = path_bytes.decode("utf-8")
    else:
        p = str(path_bytes)

    img = _Image.open(p).convert("RGB")
    arr = _np.asarray(img, dtype=_np.float32) / 255.0
    return arr


_HAS_TFIO = False
tfio = None


def decode_tif(path):
    image = tf.numpy_function(_decode_tif_to_float01, [path], Tout=tf.float32)
    image = tf.ensure_shape(image, [96, 96, 3])
    return image




## === cell 6
from sklearn.model_selection import train_test_split

train_df, val_df = train_test_split(
    train_labels,
    test_size=0.2,
    stratify=train_labels["label"],
    random_state=SEED,
)


def load_image(path, label):
    image = decode_tif(path)
    label = tf.cast(label, tf.float32)
    return image, label


BATCH_SIZE = 64

options = tf.data.Options()
options.deterministic = True

train_dataset = tf.data.Dataset.from_tensor_slices(
    (train_df["train_filepath"].values, train_df["label"].values)
).with_options(options)
train_dataset = train_dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = (
    train_dataset.cache()
    .shuffle(1000, seed=SEED, reshuffle_each_iteration=True)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(tf.data.AUTOTUNE)
)

validation_dataset = tf.data.Dataset.from_tensor_slices(
    (val_df["train_filepath"].values, val_df["label"].values)
).with_options(options)
validation_dataset = validation_dataset.map(
    load_image, num_parallel_calls=tf.data.AUTOTUNE
)
validation_dataset = (
    validation_dataset.cache()
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(tf.data.AUTOTUNE)
)

print("Data Pre-processing complete..")
print("Train batches:", tf.data.experimental.cardinality(train_dataset).numpy())
print("Val batches:", tf.data.experimental.cardinality(validation_dataset).numpy())



## === cell 7
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    BatchNormalization,
    Activation,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

model_best = Sequential(
    [
        Conv2D(32, kernel_size=5, strides=1, padding="same", input_shape=(96, 96, 3)),
        BatchNormalization(),
        Activation("relu"),
        MaxPooling2D(pool_size=2, strides=2),
        Conv2D(64, kernel_size=3, strides=1, padding="same"),
        BatchNormalization(),
        Activation("relu"),
        MaxPooling2D(pool_size=2, strides=2),
        Conv2D(128, kernel_size=3, strides=1, padding="same"),
        BatchNormalization(),
        Activation("relu"),
        MaxPooling2D(pool_size=2, strides=2),
        Conv2D(256, kernel_size=3, strides=1, padding="same"),
        BatchNormalization(),
        Activation("relu"),
        MaxPooling2D(pool_size=2, strides=2),
        Conv2D(512, kernel_size=3, strides=1, padding="same"),
        BatchNormalization(),
        Activation("relu"),
        MaxPooling2D(pool_size=2, strides=2),
        Flatten(),
        Dropout(0.5),
        Dense(256, activation="relu"),
        Dropout(0.3),
        Dense(1, activation="sigmoid"),
    ]
)

model_best.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy", AUC(name="auroc")],
)

checkpoint = ModelCheckpoint(
    "model_best.h5", monitor="val_auroc", save_best_only=True, mode="max", verbose=0
)
early_stopping = EarlyStopping(
    monitor="val_auroc", patience=3, mode="max", restore_best_weights=True, verbose=1
)

num_epochs = 10

history_best_model = model_best.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=num_epochs,
    callbacks=[early_stopping, checkpoint],
    verbose=1,
)



## === cell 8
try:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot(history_best_model.history["loss"], label="train")
    plt.plot(history_best_model.history["val_loss"], label="val")
    plt.title("Loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(history_best_model.history["accuracy"], label="train")
    plt.plot(history_best_model.history["val_accuracy"], label="val")
    plt.title("Accuracy")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(history_best_model.history["auroc"], label="train")
    plt.plot(history_best_model.history["val_auroc"], label="val")
    plt.title("AUROC")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
except Exception as e:
    print("Diagnostics plot skipped:", repr(e))



## === cell 9
sample_sub = pd.read_csv(SAMPLE_SUB_CSV)
test_df = sample_sub.copy()
test_df["test_filepath"] = test_df["id"].apply(lambda x: str(TEST_DIR / f"{x}.tif"))


def load_image_test(path):
    image = decode_tif(path)
    return image


options = tf.data.Options()
options.deterministic = True

test_dataset = tf.data.Dataset.from_tensor_slices(
    test_df["test_filepath"].values
).with_options(options)
test_dataset = test_dataset.map(load_image_test, num_parallel_calls=tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(BATCH_SIZE, drop_remainder=False).prefetch(
    tf.data.AUTOTUNE
)

preds = model_best.predict(test_dataset, verbose=1).reshape(-1)
test_df["label"] = preds.astype(np.float32)

submission_path = "/kaggle/working/submission.csv"
test_df[["id", "label"]].to_csv(submission_path, index=False)

print("Saved submission:", submission_path)
print(test_df.head())
print("Submission shape:", test_df[["id", "label"]].shape)
assert submission_path.endswith(".csv")
assert list(test_df[["id", "label"]].columns) == ["id", "label"]
assert len(test_df) == len(sample_sub)
