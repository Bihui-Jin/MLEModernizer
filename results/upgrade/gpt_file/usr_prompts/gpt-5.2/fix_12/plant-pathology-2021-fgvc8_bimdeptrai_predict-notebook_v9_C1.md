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

0.1578947368421052

# 6. Current score

0.24507

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.24507) has done: 'The timeout is dominated by the fallback training path: running ResNet50 over ~13k 256×256 images for 2 epochs is far too slow in a 600s CPU-only environment. The fastest safe fix is to ensure we never hit fallback training by resolving and loading a pretrained `.h5` deterministically with minimal filesystem scanning, and to keep the input pipeline efficient (avoid expensive caching-in-RAM, ensure parallel decode/resize, and prefetch). I replace the costly recursive `os.walk` search with a bounded, direct glob-based search under `../input` and fail fast if the model isn’t found (since training cannot meet the time budget). Prediction remains identical in semantics, using the loaded model and the same thresholding/postprocessing.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow.keras as keras

from sklearn.preprocessing import MultiLabelBinarizer

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

AUTOTUNE = tf.data.AUTOTUNE

print("TF version:", tf.__version__)
print("Keras version:", keras.__version__)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
TRAIN_CSV = "../input/plant-pathology-2021-fgvc8/train.csv"
SAMPLE_SUB = "../input/plant-pathology-2021-fgvc8/sample_submission.csv"
TRAIN_IMG_DIR = "../input/plant-pathology-2021-fgvc8/train_images"
TEST_IMG_DIR = "../input/plant-pathology-2021-fgvc8/test_images"

train = pd.read_csv(TRAIN_CSV)
submissions = pd.read_csv(SAMPLE_SUB)

print(train.shape, submissions.shape)
train.head()




## === cell 2
h_target = 256
w_target = 256
batch_size = 32

label_split = train["labels"].str.split()
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(label_split)
label_names = list(mlb.classes_)

print("Num classes:", len(label_names))
print("Classes:", label_names)




## === cell 3
@tf.function
def _decode_resize_rescale(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3, dct_method="INTEGER_FAST")
    img = tf.image.resize(
        img, [h_target, w_target], method=tf.image.ResizeMethod.BILINEAR
    )
    img = tf.cast(img, tf.float32) / 255.0
    img.set_shape((h_target, w_target, 3))
    return img


def make_dataset(
    file_paths,
    labels=None,
    training=False,
    cache=True,
    shuffle_buffer=2048,
    cache_path=None,
):
    file_paths = tf.convert_to_tensor(file_paths, dtype=tf.string)

    if labels is None:
        ds = tf.data.Dataset.from_tensor_slices(file_paths)
        ds = ds.map(
            _decode_resize_rescale,
            num_parallel_calls=AUTOTUNE,
            deterministic=False,  # faster pipeline scheduling
        )
    else:
        labels = tf.convert_to_tensor(labels, dtype=tf.float32)
        ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))

        ds = ds.map(
            lambda p, y_: (_decode_resize_rescale(p), y_),
            num_parallel_calls=AUTOTUNE,
            deterministic=False,  # faster pipeline scheduling
        )

    if cache:
        if cache_path is None:
            ds = ds.cache()
        else:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            ds = ds.cache(cache_path)

    ds = ds.apply(tf.data.experimental.ignore_errors())

    if training:
        ds = ds.shuffle(shuffle_buffer, seed=SEED, reshuffle_each_iteration=True)

    ds = ds.batch(batch_size, drop_remainder=bool(training))
    ds = ds.prefetch(AUTOTUNE)

    options = tf.data.Options()
    options.experimental_deterministic = False
    options.experimental_optimization.apply_default_optimizations = True
    ds = ds.with_options(options)
    return ds


n = len(train)
rng = np.random.RandomState(SEED)
perm = rng.permutation(n)
val_size = int(round(0.1 * n))
val_idx = perm[:val_size]
train_idx = perm[val_size:]

train_paths = (TRAIN_IMG_DIR + "/" + train.loc[train_idx, "image"].values).astype(str)
val_paths = (TRAIN_IMG_DIR + "/" + train.loc[val_idx, "image"].values).astype(str)
test_paths = (TEST_IMG_DIR + "/" + submissions["image"].values).astype(str)

y_train = y[train_idx].astype(np.float32, copy=False)
y_val = y[val_idx].astype(np.float32, copy=False)

steps_per_epoch = int(np.ceil(len(train_idx) / batch_size))
validation_steps = int(np.ceil(len(val_idx) / batch_size))
test_steps = int(np.ceil(len(submissions) / batch_size))




## === cell 4
MODEL_PATH = "../input/resnet50-256/resnet50.h5"


def build_model(num_classes: int):
    base = tf.keras.applications.ResNet50(
        include_top=False, weights="imagenet", input_shape=(h_target, w_target, 3)
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=(h_target, w_target, 3))
    x = base(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="sigmoid")(x)
    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
    )
    return model


def _find_model_fallback(expected_path: str):
    if os.path.exists(expected_path):
        return expected_path

    base_name = os.path.basename(expected_path)

    candidates = [
        expected_path,
        os.path.join("../input", base_name),
    ]

    try:
        for d in os.listdir("../input"):
            p1 = os.path.join("../input", d, base_name)
            p2 = os.path.join("../input", d, d, base_name)
            candidates.append(p1)
            candidates.append(p2)
    except Exception:
        pass

    for p in candidates:
        if os.path.exists(p):
            return p

    import glob

    try:
        patterns = [
            os.path.join("../input", "*", base_name),
            os.path.join("../input", "*", "*", base_name),
            os.path.join("../input", "*", "*", "*", base_name),
        ]
        for pat in patterns:
            hits = glob.glob(pat)
            if hits:
                hits.sort()
                return hits[0]
    except Exception:
        pass

    return expected_path


model_path_resolved = _find_model_fallback(MODEL_PATH)

if not os.path.exists(model_path_resolved):
    raise FileNotFoundError(
        f"Required pretrained model not found. Looked for: {MODEL_PATH} "
        f"(resolved to {model_path_resolved}). "
        "Fallback training is disabled to meet the 600s timeout constraint."
    )

print("Loading model from:", model_path_resolved)
model = keras.models.load_model(model_path_resolved, compile=False)

test_dataset = make_dataset(
    test_paths,
    labels=None,
    training=False,
    cache=False,
    cache_path=None,
)




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/2724137850.py in <cell line: 0>()
     78     # Timeout fix: fallback training on full train set cannot complete in 600s.
     79     # Failing fast preserves "no accuracy harm" requirement (training would be incomplete).
---> 80     raise FileNotFoundError(
     81         f"Required pretrained model not found. Looked for: {MODEL_PATH} "
     82         f"(resolved to {model_path_resolved}). "

FileNotFoundError: Required pretrained model not found. Looked for: ../input/resnet50-256/resnet50.h5 (resolved to ../input/resnet50-256/resnet50.h5). Fallback training is disabled to meet the 600s timeout constraint.

## === cell 5
preds = model.predict(test_dataset, verbose=1)
preds = preds[: len(submissions)]  # safety if last batch padding/overrun ever occurs
print("preds shape:", preds.shape)




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1432949169.py in <cell line: 0>()
----> 1 preds = model.predict(test_dataset, verbose=1)
      2 preds = preds[: len(submissions)]  # safety if last batch padding/overrun ever occurs
      3 print("preds shape:", preds.shape)
      4 
      5 

NameError: name 'model' is not defined

## === cell 6
thresh = 0.2

pred_bool = preds >= thresh
argmax_idx = np.argmax(preds, axis=1).astype(np.int32)

healthy_idx = label_names.index("healthy") if "healthy" in label_names else None

empty = ~pred_bool.any(axis=1)
if empty.any():
    pred_bool[empty, :] = False
    pred_bool[empty, argmax_idx[empty]] = True

if healthy_idx is not None:
    multi = pred_bool.sum(axis=1) > 1
    has_healthy = pred_bool[:, healthy_idx]
    fix = multi & has_healthy
    if fix.any():
        pred_bool[fix, :] = False
        pred_bool[fix, argmax_idx[fix]] = True

label_arr = np.asarray(label_names, dtype=object)
pred_indices = [np.flatnonzero(row) for row in pred_bool]
pred_labels = [" ".join(label_arr[idxs]) for idxs in pred_indices]

submissions["labels"] = pred_labels
submissions.head()




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1748200231.py in <cell line: 0>()
      1 thresh = 0.2
      2 
----> 3 pred_bool = preds >= thresh
      4 argmax_idx = np.argmax(preds, axis=1).astype(np.int32)
      5 

NameError: name 'preds' is not defined

## === cell 7
out_path = "submission.csv"
submissions.to_csv(out_path, index=False)
print("Wrote", out_path, "with shape", submissions.shape)
print(submissions.tail())
