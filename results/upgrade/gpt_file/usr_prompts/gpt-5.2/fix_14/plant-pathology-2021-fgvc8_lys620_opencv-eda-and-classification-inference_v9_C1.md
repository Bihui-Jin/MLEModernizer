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
Detect apple diseases from images.

## Metric
Mean F1-Score

## Submission Format
labels should be a space-delimited list.

The file should contain a header and have the following format:

```
image, labels
85f8cb619c66b863.jpg,healthy
ad8770db05586b59.jpg,healthy
c7b03e718489f3ca.jpg,healthy
```

## Dataset
**train.csv** - the training set metadata.

- `image` - the image ID.
- `labels` - the target classes, a space delimited list of all diseases found in the image. Unhealthy leaves with too many diseases to classify visually will have the `complex` class, and may also have a subset of the diseases identified.

**sample_submission.csv** - A sample submission file in the correct format.

- `image`
- `labels`

**train_images** - The training set images.

**test_images** - The test set images. This competition has a hidden test set: only three images are provided here as samples while the remaining 5,000 images will be available to your notebook once it is submitted.

# 2. Python version

3.9

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
        input/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
        working/
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
```

-> data/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> data/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> (stopped after 10 files for performance)

# 5. Target score

0.1821791320406278

# 6. Current score

0.43061

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.43061) has done: 'The timeout is dominated by input pipeline overhead (JPEG decode/resize on-the-fly every epoch) and slower deterministic execution; the model itself is small. I keep the exact same model, loss, epochs, and splits, but speed up the tf.data pipelines by enabling aggressive caching (train/val and test), turning on map/batch fusion optimizations, and making the dataset repeat with an explicit `steps_per_epoch` so the cached dataset is reused efficiently across epochs without re-tracing/overhead. I also set TensorFlow thread settings and remove an unused PIL loader path to reduce Python-side overhead while preserving determinism and correctness. These changes are provably equivalent in outputs (same decoded pixels and labels) but avoid redundant preprocessing work across epochs.'
- What this solution (achieved 0.42469) has done: 'I fix the immediate runtime crash by removing the protobuf environment override that forces the pure-Python implementation (it conflicts with TensorFlow’s bundled protobuf and triggers the `MessageFactory.GetPrototype` error). I keep the same model, data split, training loop, and prediction logic so the core behavior and score characteristics remain essentially unchanged (and should stay close to your current 0.43061, which is already above the target). I also add a small, safe path auto-detection fallback so the notebook runs reliably whether the dataset is mounted at `../input/...` or `/kaggle/input/...`, without changing any training semantics. The output still be a valid `submission.csv` with the required `image,labels` columns.'
- What this solution (achieved 0.4368) has done: 'I fix the immediate TensorFlow import crash (`MessageFactory.GetPrototype`) by forcing TensorFlow to use the pure-Python protobuf implementation before importing TensorFlow, which avoids the incompatible C++ protobuf symbols in this environment. I keep the exact same model, split, training loop, and prediction logic so behavior/score stays essentially the same (still above your target band, but the primary goal here is to run end-to-end reliably). I also make the dataset-path auto-detection a bit more robust by checking both the direct directory and a nested `plant-pathology-2021-fgvc8/` folder, without changing any I/O locations when the original path works. The script still write a valid `submission.csv` with the required `image,labels` columns.'
- What this solution (achieved 0.41361) has done: 'I fix the immediate TensorFlow/protobuf crash by removing the forced pure-Python protobuf environment variables, which are the root cause of the `MessageFactory.GetPrototype` error in this Kaggle runtime. I keep the exact same model, training loop, split, and tf.data pipeline logic, so the score behavior should remain essentially unchanged (and already above your target). I also keep the existing robust dataset path detection and ensure the notebook still writes a valid `submission.csv` with the required `image,labels` columns.'
- What this solution (achieved 0.41742) has done: 'I fix the TensorFlow/protobuf import crash by setting the protobuf implementation to the pure-Python backend *before* importing TensorFlow (this is the direct cause of the `MessageFactory.GetPrototype` error in some Kaggle TF builds). I keep the exact same model, split, tf.data pipeline, epochs, and prediction logic so the score behavior stays essentially the same (and should remain above your target). I also make the protobuf setting robust by forcing it only if not already set, and keep all I/O paths and the submission format unchanged so a valid `submission.csv` is always produced.'
- What this solution (achieved 0.42546) has done: 'I fix the TensorFlow import crash by removing the protobuf pure-Python override that triggers the `MessageFactory.GetPrototype` error in this Kaggle runtime; this is an environment compatibility issue, not a modeling one. I keep the same model, loss, split strategy, epochs, and tf.data logic so behavior remains consistent and score-impact is minimal. Since your current score (0.41742) is already far above the target (0.182...), I not make any changes intended to improve score; the goal here is stability and producing a valid `submission.csv`. I also keep the existing robust dataset path auto-detection and ensure the script runs end-to-end.'
- What this solution (achieved 0.43396) has done: 'I fix the TensorFlow import crash caused by an incompatible protobuf backend by forcing the pure-Python protobuf implementation *before* importing TensorFlow (this directly addresses the `MessageFactory.GetPrototype` error). I keep your model, training loop, data split, preprocessing, and submission formatting exactly the same so the core logic and score characteristics remain essentially unchanged (and already within the target tolerance band). I also keep the existing path auto-detection unchanged and ensure `submission.csv` is always written with the correct `image,labels` columns.'
- What this solution (achieved 0.43396) has done: 'I fix the runtime crash happening before TensorFlow imports by removing the protobuf pure-Python override, which is the direct cause of the `MessageFactory.GetPrototype` AttributeError in this Kaggle runtime. I keep the model, training loop, split, tf.data pipeline, epochs, and prediction/post-processing identical so behavior and score remain essentially unchanged (your current score is already far above the target, so no score-improvement changes are needed). I also keep the existing dataset path auto-detection and ensure the script always writes a valid `submission.csv` with the required `image,labels` columns. These changes are minimal and only aimed at making the notebook run end-to-end reliably.'
- What this solution (achieved 0.43113) has done: 'You’re crashing on `import tensorflow` due to an environment-level protobuf/TensorFlow incompatibility (`MessageFactory.GetPrototype`). I fix this by forcing TensorFlow to use the pure-Python protobuf backend *before* importing TensorFlow (and disabling the C++ implementation), which is the minimal change that unblocks execution. I not change the model, split, training loop, preprocessing, or submission formatting, since your current score (0.43396) is already well above the target band and we should avoid score-shifting edits. I also keep the same paths and ensure `submission.csv` is always produced with the required `image,labels` columns.'
- What this solution (achieved 0.43437) has done: 'I fix the TensorFlow import crash by removing the protobuf pure-Python override, which is the direct trigger of the `MessageFactory.GetPrototype` error in this Kaggle runtime. This change is environment-only and keeps your model, data pipeline, split, training loop, and prediction logic identical, so the score behavior should remain essentially unchanged (and since your current score is already well above the target, we avoid score-changing edits). I also keep all paths and submission formatting the same to ensure a valid `submission.csv` is written end-to-end.'
- What this solution (achieved 0.43061) has done: 'I fix the TensorFlow/protobuf import crash (`MessageFactory.GetPrototype`) by setting the protobuf implementation to the pure-Python backend *before* importing TensorFlow, which is the minimal environment-level change that unblocks execution. I keep your model, training loop, split, preprocessing, and submission formatting identical so the core logic and score behavior remain essentially unchanged (your current score is already above the target, so we avoid score-changing edits). I also add a small defensive fallback: if forcing pure-Python protobuf fails, we retry without it, to make the notebook robust across Kaggle images. The script still run end-to-end and write a valid `submission.csv` with `image,labels`.'

# 9. Code solution

## === cell 0
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "3")

import pandas as pd
import numpy as np

try:
    import tensorflow as tf
except AttributeError as e:
    if "GetPrototype" in str(e):
        os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
        os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)
        import tensorflow as tf
    else:
        raise

from tensorflow.keras import layers, models, optimizers

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.threading.set_intra_op_parallelism_threads(max(1, os.cpu_count() // 2))
except Exception:
    pass

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

print("TF version:", tf.__version__)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
BASE_CANDIDATES = [
    "../input/plant-pathology-2021-fgvc8",
    "/kaggle/input/plant-pathology-2021-fgvc8",
    "../input",
    "/kaggle/input",
]

base_path = None
for cand in BASE_CANDIDATES:
    if not os.path.isdir(cand):
        continue

    direct = cand
    nested = os.path.join(cand, "plant-pathology-2021-fgvc8")

    if os.path.exists(os.path.join(direct, "train.csv")):
        base_path = direct
        break
    if os.path.exists(os.path.join(nested, "train.csv")):
        base_path = nested
        break

if base_path is None:
    raise FileNotFoundError(
        "Could not locate dataset folder containing train.csv. Tried: "
        + ", ".join(BASE_CANDIDATES)
    )

train_csv_path = os.path.join(base_path, "train.csv")
sample_sub_path = os.path.join(base_path, "sample_submission.csv")
train_dir = os.path.join(base_path, "train_images")
test_dir = os.path.join(base_path, "test_images")

print("Using base_path:", base_path)
print("train_csv_path:", train_csv_path)
print("sample_sub_path:", sample_sub_path)

df = pd.read_csv(train_csv_path)
df.head()



## === cell 2
df["labels"] = df["labels"].astype("category")
df["label_num"] = df["labels"].cat.codes

label_num_to_str = dict(zip(df["label_num"].values, df["labels"].astype(str).values))
label_str_to_num = dict(zip(df["labels"].astype(str).values, df["label_num"].values))

num_classes = df["label_num"].nunique()
print("Num classes (unique label strings):", num_classes)



## === cell 3
submission = pd.read_csv(sample_sub_path)
submission.head()



## === cell 4
IMG_SIZE = (150, 150)
BATCH_SIZE = 32
EPOCHS = 3  # unchanged

from sklearn.model_selection import train_test_split

train_df, val_df = train_test_split(
    df[["image", "label_num"]].copy(),
    test_size=0.15,
    random_state=SEED,
    stratify=df["label_num"],
)
print("Train/Val:", train_df.shape, val_df.shape)


def make_dataset(frame, images_root, training=False, cache=True):
    paths = (images_root + "/" + frame["image"].astype(str)).values
    labels = frame["label_num"].values.astype(np.int32)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    options = tf.data.Options()
    options.deterministic = True
    options.experimental_optimization.map_fusion = True
    options.experimental_optimization.map_parallelization = True
    options.experimental_optimization.parallel_batch = True
    ds = ds.with_options(options)

    def _load(p, y):
        img_bytes = tf.io.read_file(p)
        img = tf.image.decode_jpeg(img_bytes, channels=3)
        img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)
        img = tf.clip_by_value(img / 255.0, 0.0, 1.0)
        return img, y

    if training:
        ds = ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)

    if cache:
        ds = ds.cache()

    ds = ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
    return ds


train_ds = make_dataset(train_df, train_dir, training=True, cache=True)
val_ds = make_dataset(val_df, train_dir, training=False, cache=True)

steps_per_epoch = int(np.ceil(len(train_df) / BATCH_SIZE))
val_steps = int(np.ceil(len(val_df) / BATCH_SIZE))

train_ds_rep = train_ds.repeat()
val_ds_rep = val_ds.repeat()



## === cell 5
model = models.Sequential(
    [
        layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()



## === cell 6
history = model.fit(
    train_ds_rep,
    validation_data=val_ds_rep,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_steps=val_steps,
    verbose=1,
)



## === cell 7
test_images = submission["image"].values
test_paths = np.array([os.path.join(test_dir, f) for f in test_images], dtype=str)

test_ds = tf.data.Dataset.from_tensor_slices(test_paths)

options = tf.data.Options()
options.deterministic = True
options.experimental_optimization.map_fusion = True
options.experimental_optimization.map_parallelization = True
options.experimental_optimization.parallel_batch = True
test_ds = test_ds.with_options(options)


def _load_test(p):
    img_bytes = tf.io.read_file(p)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.clip_by_value(img / 255.0, 0.0, 1.0)
    return img


test_ds = test_ds.map(_load_test, num_parallel_calls=tf.data.AUTOTUNE).cache()
test_ds = test_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(tf.data.AUTOTUNE)

probs = model.predict(test_ds, verbose=0)
pred_nums = np.argmax(probs, axis=1).astype(np.int32)



## === cell 8
pred_labels = pd.Series(pred_nums).map(label_num_to_str).astype(str).values

submission_result = pd.DataFrame({"image": test_images, "labels": pred_labels})

assert (
    submission_result.shape[0] == submission.shape[0]
), "Submission row count mismatch."
assert list(submission_result.columns) == [
    "image",
    "labels",
], "Submission columns mismatch."

submission_result.to_csv("submission.csv", index=False)
print(submission_result.head())
print("Wrote submission.csv with", len(submission_result), "rows")



## === cell 9
print("Competetion Complete!!")
