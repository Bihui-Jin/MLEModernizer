# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

3.14

# 3. Installed packages

No external packages required in the script and installed.

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

# 5. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=2")

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "4")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")

import numpy as np
import pandas as pd
import zipfile

from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import warnings

warnings.filterwarnings("ignore")

try:
    tf.config.threading.set_intra_op_parallelism_threads(
        int(os.environ.get("TF_NUM_INTRAOP_THREADS", "4"))
    )
    tf.config.threading.set_inter_op_parallelism_threads(
        int(os.environ.get("TF_NUM_INTEROP_THREADS", "2"))
    )
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

print("Python:", os.sys.version)
print("TF:", tf.__version__)



## === cell 1
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.keras.utils.set_random_seed(SEED)
except Exception:
    pass
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass



## === cell 2
work_base = "/kaggle/working/aerial-cactus-identification"
train_out = os.path.join(work_base, "train")
test_out = os.path.join(work_base, "test")


def _dir_has_jpgs(p: str) -> bool:
    try:
        return os.path.isdir(p) and any(
            fn.lower().endswith(".jpg") for fn in os.listdir(p)
        )
    except Exception:
        return False


need_extract = not (_dir_has_jpgs(train_out) and _dir_has_jpgs(test_out))
if need_extract:
    os.makedirs(work_base, exist_ok=True)
    with zipfile.ZipFile(
        "/kaggle/input/aerial-cactus-identification/train.zip", "r"
    ) as z:
        z.extractall(work_base)
    with zipfile.ZipFile(
        "/kaggle/input/aerial-cactus-identification/test.zip", "r"
    ) as z:
        z.extractall(work_base)



## === cell 3
base_work_dir = "/kaggle/working/aerial-cactus-identification"
train_dir = os.path.join(base_work_dir, "train")
test_dir = os.path.join(base_work_dir, "test")

assert os.path.isdir(train_dir), f"Train directory not found: {train_dir}"
assert os.path.isdir(test_dir), f"Test directory not found: {test_dir}"

train_labels = pd.read_csv("/kaggle/input/aerial-cactus-identification/train.csv")



## === cell 4
df = train_labels.rename(columns={"id": "image", "has_cactus": "label"}).copy()
df["image"] = train_dir.rstrip("/") + "/" + df["image"].astype(str)



## === cell 5
missing = [p for p in df["image"].iloc[:50].tolist() if not os.path.exists(p)]
print("Missing among first 50:", len(missing))



## === cell 6
test_filenames = sorted(
    [
        e.name
        for e in os.scandir(test_dir)
        if e.is_file() and e.name.lower().endswith(".jpg")
    ]
)
test_df = pd.DataFrame({"id": test_filenames})
print("Train images:", len(df), "Test images:", len(test_df))



## === cell 7
assert len(test_df) > 0, "No test images found."



## === cell 8
print(df.head(2))



## === cell 9
print("Train df shape:", df.shape)



## === cell 10
pass



## === cell 11
pass



## === cell 12
df["label"] = df["label"].astype(str)



## === cell 13
pass



## === cell 14
print(df["label"].value_counts())



## === cell 15
RUN_EDA_PLOTS = False



## === cell 16
if RUN_EDA_PLOTS:
    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.countplot(x=df["label"], palette=["salmon", "skyblue"])



## === cell 17
if RUN_EDA_PLOTS:
    import matplotlib.pyplot as plt

    img_list1 = df["image"].tolist()
    label_list1 = train_labels["has_cactus"].tolist()
    fig, ax = plt.subplots(2, 5)
    fig.set_size_inches(12, 6)
    k = 0
    for i in range(2):
        for j in range(5):
            img_path = img_list1[k]
            label = label_list1[k]
            img = plt.imread(img_path)
            ax[i, j].imshow(img)
            status = "Has Cactus (1)" if label == 1 else "No Cactus(0)"
            ax[i, j].set_title(status, fontsize=10)
            ax[i, j].axis("off")
            k += 1
    plt.tight_layout()
    plt.show()



## === cell 18
print(test_df.head(2))



## === cell 19
print("Test df shape:", test_df.shape)



## === cell 20
if RUN_EDA_PLOTS:
    import matplotlib.pyplot as plt

    sample_test = test_df.sample(10, random_state=SEED).reset_index(drop=True)
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(12, 6))
    for i, ax in enumerate(axes.flat):
        img_name = sample_test.loc[i, "id"]
        full_path = os.path.join(test_dir, img_name)
        img = plt.imread(full_path)
        ax.imshow(img)
        ax.axis("off")
    plt.tight_layout()
    plt.show()



## === cell 21
IMG_SIZE = (64, 64)
BATCH_SIZE = 64



## === cell 22
y_int = df["label"].astype(int).values
cw = class_weight.compute_class_weight(
    class_weight="balanced", classes=np.array([0, 1]), y=y_int
)
cw = {0: float(cw[0]), 1: float(cw[1])}

train_df, val_df = train_test_split(
    df, test_size=0.2, random_state=SEED, stratify=df["label"]
)

AUTOTUNE = tf.data.AUTOTUNE

train_paths = train_df["image"].values
train_labels_bin = train_df["label"].astype(np.float32).values
val_paths = val_df["image"].values
val_labels_bin = val_df["label"].astype(np.float32).values

weights_dict = {0: float(cw[0]), 1: float(cw[1])}
print("class_weight used:", weights_dict)

aug_model = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip(mode="horizontal_and_vertical", seed=SEED),
        tf.keras.layers.RandomRotation(
            factor=30.0 / 180.0, fill_mode="reflect", seed=SEED
        ),
        tf.keras.layers.RandomZoom(
            height_factor=(-0.2, 0.2),
            width_factor=(-0.2, 0.2),
            fill_mode="reflect",
            seed=SEED,
        ),
    ],
    name="augmentation",
)


@tf.function
def _decode_and_resize(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(
        img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR, antialias=False
    )
    img = tf.cast(img, tf.float32) / 255.0  # rescale=1/255
    return img


@tf.function
def _augment_batch(images, labels):
    return aug_model(images, training=True), labels


options = tf.data.Options()
options.experimental_deterministic = True

cache_dir = os.path.join("/kaggle/working", "tf_cache_cactus")
os.makedirs(cache_dir, exist_ok=True)
train_cache_path = os.path.join(cache_dir, f"train_img64_seed{SEED}.cache")
val_cache_path = os.path.join(cache_dir, f"val_img64_seed{SEED}.cache")

shuffle_buf = min(len(train_paths), 4096)

train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels_bin))
train_ds = train_ds.with_options(options)
train_ds = train_ds.shuffle(shuffle_buf, seed=SEED, reshuffle_each_iteration=True)
train_ds = train_ds.map(
    lambda p, y: (_decode_and_resize(p), y), num_parallel_calls=AUTOTUNE
)
train_ds = train_ds.cache(train_cache_path)
train_ds = train_ds.batch(BATCH_SIZE, drop_remainder=False)
train_ds = train_ds.map(_augment_batch, num_parallel_calls=AUTOTUNE)
train_ds = train_ds.repeat().prefetch(AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels_bin))
val_ds = val_ds.with_options(options)
val_ds = val_ds.map(
    lambda p, y: (_decode_and_resize(p), y), num_parallel_calls=AUTOTUNE
)
val_ds = val_ds.cache(val_cache_path)
val_ds = val_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)



## === cell 23
train_steps = int(np.ceil(len(train_paths) / BATCH_SIZE))
val_steps = int(np.ceil(len(val_paths) / BATCH_SIZE))
print("Train steps:", train_steps, "Val steps:", val_steps)
assert train_steps > 0 and val_steps > 0, "Dataset has length 0 (check paths/df)."



## === cell 24
base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(64, 64, 3))
base_model.trainable = False

model = Sequential()
model.add(base_model)
model.add(GlobalAveragePooling2D())
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(1, activation="sigmoid"))

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"],
    steps_per_execution=16,
)
model.summary()



## === cell 25
callbacks = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1
    ),
]

history = model.fit(
    train_ds,
    steps_per_epoch=train_steps,
    epochs=20,
    validation_data=val_ds,
    validation_steps=val_steps,
    callbacks=callbacks,
    class_weight=weights_dict,
)



## === cell 26
print(history.history["accuracy"][-1])



## === cell 27
model.save("cactus.h5")



## === cell 28
pass



## === cell 29
if RUN_EDA_PLOTS:
    import matplotlib.pyplot as plt

    plt.plot(history.history["accuracy"], label="Accuracy")
    plt.plot(history.history["val_accuracy"], label="Val_Accuracy")
    plt.plot(history.history["loss"], label="Loss")
    plt.plot(history.history["val_loss"], label="Val_Loss")
    plt.legend()



## === cell 30
test_paths = (test_dir.rstrip("/") + "/" + test_df["id"].astype(str)).values


@tf.function
def _test_map_fn(path):
    return _decode_and_resize(path)


test_cache_path = os.path.join(cache_dir, f"test_img64_seed{SEED}.cache")

test_ds = tf.data.Dataset.from_tensor_slices(test_paths)
test_ds = test_ds.with_options(options)
test_ds = test_ds.map(_test_map_fn, num_parallel_calls=AUTOTUNE).cache(test_cache_path)
test_ds = test_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)



## === cell 31
test_steps = int(np.ceil(len(test_paths) / BATCH_SIZE))
predictions = model.predict(
    test_ds,
    steps=test_steps,
    verbose=1,
)
predictions = predictions.reshape(-1)[: len(test_df)]

print("Predictions shape:", predictions.shape)
assert len(predictions) == len(test_df), "Prediction count does not match test rows."

submission_df = pd.DataFrame({"id": test_df["id"].values, "has_cactus": predictions})
submission_df.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", submission_df.shape)
print(submission_df.head())
