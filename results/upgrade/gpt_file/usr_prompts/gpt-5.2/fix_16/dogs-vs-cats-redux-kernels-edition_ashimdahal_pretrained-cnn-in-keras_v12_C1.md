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
Given a dataset of images of dogs and cats, predict if an image is a dog or a cat.

## Metric
Log loss.

## Submission Format
For each image in the test set, you must submit a probability that image is a dog. The file should have a header and be in the following format:

```
id,label
1,0.5
2,0.5
3,0.5
...
```

## Dataset
The train folder contains 25,000 images of dogs and cats. Each image in this folder has the label as part of the filename. The test folder contains 12,500 images, named according to a numeric id.

# 2. Python version

3.8

# 3. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
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
            cat.1714.jpg (7.8 kB)
            cat.10025.jpg (18.4 kB)
            ... and 24998 other files
            description.md (50 lines)
            sample_submission.csv (2501 lines)
            sample_submission.csv.zip (6.0 kB)
            test.zip (56.6 MB)
            train.zip (513.0 MB)
            dogs-vs-cats-redux-kernels-edition/
                cat.1714.jpg (7.8 kB)
                cat.10025.jpg (18.4 kB)
                ... and 24998 other files
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                sample_submission.csv.zip (6.0 kB)
                test.zip (56.6 MB)
                train.zip (513.0 MB)
                dogs-vs-cats-redux-kernels-edition/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
            test/
                test/
                unknown/
                    900.jpg (42.3 kB)
                    572.jpg (30.6 kB)
                    ... and 2498 other files
            train/
                cat/
                    cat.4838.jpg (20.2 kB)
                    cat.1314.jpg (21.7 kB)
                    ... and 11240 other files
                dog/
                    dog.6712.jpg (35.3 kB)
                    dog.7152.jpg (36.1 kB)
                    ... and 11256 other files
                train/
        input/
            cat.1714.jpg (7.8 kB)
            cat.10025.jpg (18.4 kB)
            ... and 24998 other files
            description.md (50 lines)
            sample_submission.csv (2501 lines)
            sample_submission.csv.zip (6.0 kB)
            test.zip (56.6 MB)
            train.zip (513.0 MB)
            dogs-vs-cats-redux-kernels-edition/
                cat.1714.jpg (7.8 kB)
                cat.10025.jpg (18.4 kB)
                ... and 24998 other files
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                sample_submission.csv.zip (6.0 kB)
                test.zip (56.6 MB)
                train.zip (513.0 MB)
                dogs-vs-cats-redux-kernels-edition/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
            test/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                unknown/
                    900.jpg (42.3 kB)
                    572.jpg (30.6 kB)
                    ... and 2498 other files
            train/
                cat/
                    cat.4838.jpg (20.2 kB)
                    cat.1314.jpg (21.7 kB)
                    ... and 11240 other files
                dog/
                    dog.6712.jpg (35.3 kB)
                    dog.7152.jpg (36.1 kB)
                    ... and 11256 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
        working/
            dogs-vs-cats-redux-kernels-edition/
                cat.1714.jpg (7.8 kB)
                cat.10025.jpg (18.4 kB)
                ... and 24998 other files
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                sample_submission.csv.zip (6.0 kB)
                test.zip (56.6 MB)
                train.zip (513.0 MB)
                dogs-vs-cats-redux-kernels-edition/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
```

-> data/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> data/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> input/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> input/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> working/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

# 5. Target score

0.16863

# 6. Current score

0.03858

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

- What this solution (achieved 0.03858) has done: 'I remove the `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` override that’s triggering the protobuf `MessageFactory.GetPrototype` crash in this environment. Then I fix test image discovery by extracting zips into dedicated `./train_extracted` and `./test_extracted` folders and searching those robustly, so `test_images` is never empty. Finally, I avoid the Keras `predict()` UnboundLocalError by predicting via a simple deterministic `model(x, training=False)` loop over the dataset, and then write a `submission.csv` matching `sample_submission.csv` exactly.'

# 9. Code solution

## === cell 0
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
import zipfile
import matplotlib.pyplot as plt
import glob

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

print("TF:", tf.__version__)
print("OpenCV:", cv2.__version__)
print(
    "protobuf implementation override:",
    os.environ.get("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"),
)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
BASE_INPUT = "/kaggle/input/dogs-vs-cats-redux-kernels-edition"

TEST_ZIP = os.path.join(BASE_INPUT, "test.zip")
TRAIN_ZIP = os.path.join(BASE_INPUT, "train.zip")

assert os.path.exists(TEST_ZIP), f"Missing {TEST_ZIP}"
assert os.path.exists(TRAIN_ZIP), f"Missing {TRAIN_ZIP}"

print("Found zips:", TRAIN_ZIP, TEST_ZIP)




## === cell 2
def _has_images(root):
    if not os.path.isdir(root):
        return False
    for dp, dn, fn in os.walk(root):
        for f in fn:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                return True
    return False


TRAIN_DIR = "train_extracted"
TEST_DIR = "test_extracted"

need_extract = (not _has_images(TRAIN_DIR)) or (not _has_images(TEST_DIR))
if need_extract:
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)
    with zipfile.ZipFile(TRAIN_ZIP, "r") as zf:
        zf.extractall(TRAIN_DIR)
    with zipfile.ZipFile(TEST_ZIP, "r") as zf:
        zf.extractall(TEST_DIR)

print("Has train images:", _has_images(TRAIN_DIR))
print("Has test images:", _has_images(TEST_DIR))
print("Top-level dirs:", sorted([d for d in os.listdir(".") if os.path.isdir(d)]))




## === cell 3
def list_images_glob(folder):
    if not os.path.isdir(folder):
        return []
    exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(folder, "**", e), recursive=True))
    return files


train_search_roots = [
    TRAIN_DIR,
    os.path.join(TRAIN_DIR, "train"),
    os.path.join(TRAIN_DIR, "train", "train"),
    os.path.join(TRAIN_DIR, "train", "cat"),
    os.path.join(TRAIN_DIR, "train", "dog"),
    os.path.join(TRAIN_DIR, "train", "train", "cat"),
    os.path.join(TRAIN_DIR, "train", "train", "dog"),
]

seen = set()
train_images_all = []
for r in train_search_roots:
    for p in list_images_glob(r):
        base = os.path.basename(p).lower()
        if base.startswith(("cat.", "dog.")) and base.endswith(
            (".jpg", ".jpeg", ".png")
        ):
            if p not in seen:
                train_images_all.append(p)
                seen.add(p)

test_search_roots = [
    TEST_DIR,
    os.path.join(TEST_DIR, "test"),
    os.path.join(TEST_DIR, "test", "unknown"),
    os.path.join(TEST_DIR, "unknown"),
    os.path.join(TEST_DIR, "test", "test"),
    os.path.join(TEST_DIR, "test", "test", "unknown"),
]
seen = set()
test_images = []
for r in test_search_roots:
    for p in list_images_glob(r):
        base = os.path.basename(p)
        stem = os.path.splitext(base)[0]
        if stem.isdigit():
            if p not in seen:
                test_images.append(p)
                seen.add(p)

train_images_all = sorted(train_images_all)
test_images = sorted(
    test_images, key=lambda p: int(os.path.splitext(os.path.basename(p))[0])
)

print("Num train images:", len(train_images_all))
print("Num test images:", len(test_images))
print("Example train:", train_images_all[0] if train_images_all else None)
print("Example test:", test_images[0] if test_images else None)

assert len(train_images_all) > 0, "No training images found after extraction."
assert len(test_images) > 0, "No test images found after extraction."



## === cell 4
rng = np.random.RandomState(SEED)
perm = rng.permutation(len(train_images_all))
train_images_all = [train_images_all[i] for i in perm]

limit = int(0.8 * len(train_images_all))
train_images = train_images_all[:limit]
validation_images = train_images_all[limit:]

print(
    "Num train:",
    len(train_images),
    "Num val:",
    len(validation_images),
    "Num test:",
    len(test_images),
)

if train_images:
    img = cv2.imread(train_images[0])
    if img is None:
        raise ValueError(f"Failed to read image: {train_images[0]}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(3, 3))
    plt.imshow(img_rgb)
    plt.axis("off")
    plt.show()



## === cell 5
rows, columns = 160, 160
image_shape = (rows, columns, 3)


def is_dog(path):
    p = path.replace("\\", "/").lower()
    return ("/dog" in p and "dog." in os.path.basename(p).lower()) or os.path.basename(
        p
    ).lower().startswith("dog.")


label = np.array([1.0 if is_dog(p) else 0.0 for p in train_images], dtype=np.float32)
validation_label = np.array(
    [1.0 if is_dog(p) else 0.0 for p in validation_images], dtype=np.float32
)

print(
    "Label mean (train):",
    float(label.mean()),
    "Label mean (val):",
    float(validation_label.mean()),
)

AUTOTUNE = tf.data.AUTOTUNE


def _decode_resize_preprocess(path):
    path = tf.cast(path, tf.string)
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img_bytes, channels=3)  # dataset is jpg
    img = tf.image.resize(img, [rows, columns], method=tf.image.ResizeMethod.BICUBIC)
    img = tf.cast(img, tf.float32)
    img = tf.keras.applications.resnet.preprocess_input(img)
    return img


def make_supervised_ds(paths, y, batch_size, training):
    paths_t = tf.constant([str(p) for p in paths], dtype=tf.string)
    y_t = tf.constant(y, dtype=tf.float32)
    ds = tf.data.Dataset.from_tensor_slices((paths_t, y_t))
    if training:
        ds = ds.shuffle(
            buffer_size=len(paths), seed=SEED, reshuffle_each_iteration=True
        )
    ds = ds.map(
        lambda p, t: (_decode_resize_preprocess(p), t),
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def make_test_ds(paths, batch_size):
    paths_t = tf.constant([str(p) for p in paths], dtype=tf.string)
    ds = tf.data.Dataset.from_tensor_slices(paths_t)
    ds = ds.map(
        _decode_resize_preprocess, num_parallel_calls=AUTOTUNE, deterministic=True
    )
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


BATCH_SIZE = 128
train_ds = make_supervised_ds(train_images, label, batch_size=BATCH_SIZE, training=True)
val_ds = make_supervised_ds(
    validation_images, validation_label, batch_size=BATCH_SIZE, training=False
)
test_ds = make_test_ds(test_images, batch_size=BATCH_SIZE)



## === cell 6
base_model = tf.keras.applications.ResNet101(
    weights="imagenet", include_top=False, input_shape=image_shape
)
base_model.trainable = False

model = tf.keras.Sequential(
    [
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ]
)

model.compile(
    optimizer="adam",
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
    metrics=["accuracy"],
)

model.summary()



## === cell 7
epochs = 10

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    shuffle=True,  # preserved (ds already shuffles; this keeps API semantics consistent)
    verbose=0,
)

last_val_loss = history.history.get("val_loss", [None])[-1]
print(
    "Training done. Last val_loss:",
    None if last_val_loss is None else float(last_val_loss),
)



## === cell 8
pred_batches = []
for xb in test_ds:
    yb = model(xb, training=False)
    pred_batches.append(tf.reshape(yb, [-1]))
prediction = tf.concat(pred_batches, axis=0).numpy().astype(np.float64)

prediction = np.clip(prediction, 1e-7, 1 - 1e-7)

assert prediction.shape[0] > 0, "prediction has 0 rows; no test images were loaded."
assert prediction.shape[0] == len(
    test_images
), "Prediction length != number of test images."

idx = 4 if len(test_images) > 4 else 0
img0 = cv2.imread(test_images[idx])
if img0 is not None:
    img0 = cv2.resize(img0, (columns, rows), interpolation=cv2.INTER_CUBIC)
    plt.figure(figsize=(3, 3))
    plt.title(f"pred(dog)={prediction[idx]:.4f}")
    plt.imshow(cv2.cvtColor(img0, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

print(
    "Pred shape:",
    prediction.shape,
    "min/max:",
    float(prediction.min()),
    float(prediction.max()),
)



## === cell 9
sample_path = os.path.join(BASE_INPUT, "sample_submission.csv")
sample = pd.read_csv(sample_path)
assert list(sample.columns) == [
    "id",
    "label",
], f"Unexpected sample_submission columns: {sample.columns.tolist()}"

test_ids_from_files = [
    int(os.path.splitext(os.path.basename(p))[0]) for p in test_images
]
pred_map = dict(zip(test_ids_from_files, prediction))

out = sample.copy()
out["label"] = out["id"].map(pred_map).astype(float)
out["label"] = out["label"].fillna(0.5).clip(1e-7, 1 - 1e-7)

out.to_csv("submission.csv", index=False, header=True)

print(out.head())
print("Wrote submission.csv with shape:", out.shape)
print("submission.csv exists:", os.path.exists("submission.csv"))
print("id min/max:", int(out["id"].min()), int(out["id"].max()))
print("label min/max:", float(out["label"].min()), float(out["label"].max()))

assert (
    out.shape[0] == sample.shape[0]
), "Submission row count does not match sample_submission."
assert out["label"].notna().all(), "Submission contains NaN labels."
