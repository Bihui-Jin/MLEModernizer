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

0.1503231763619575

# 6. Current score

0.27161

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.31542) has done: 'I fix the runtime import crash in TensorFlow by forcing the pure‑Python protobuf implementation before importing TensorFlow, which avoids the `MessageFactory.GetPrototype` error in many Kaggle images. Then I remove the dependency on the missing external model file by training a small, standard TF/Keras CNN on the provided `train_images` using `train.csv` labels (multi-label), preserving the same end-to-end prediction→threshold→space-delimited labels submission semantics. I also make the data pipeline robust: correct paths, deterministic split, correct label binarization, and consistent ordering to match `sample_submission.csv`. Finally, I ensure a valid `submission.csv` is always written to `/kaggle/working/submission.csv` with the exact required columns.'
- What this solution (achieved 0.31925) has done: 'Main runtime is dominated by input pipeline overhead (Python `Path.exists()` checks per row, non-cached decoding/resizing every epoch, and suboptimal `tf.data` options) plus expensive directory glob counting at startup. I remove the costly image counting/glob and replace per-row `Path.exists()` with a single `tf.io.gfile.exists` check inside the dataset graph where needed, while keeping paths and semantics identical. I also add deterministic, cached, fused `tf.data` pipelines (cache after decode/resize, keep shuffle behavior for training, and set dataset/options to reduce overhead) so epoch 2 doesn’t re-decode all images. Model, training loop, loss, feature extraction, and thresholding remain unchanged.'
- What this solution (achieved 0.31925) has done: 'I fix the TensorFlow import crash by forcing the pure-Python protobuf implementation *and* purging any preloaded protobuf/TensorFlow modules before importing TF, which prevents the `MessageFactory.GetPrototype` error from persisting. Then I fix the `tf.data` pipeline crash by only setting dataset option fields that exist in this TensorFlow build (removing the unsupported `autotune_buffers` assignment) so dataset creation succeeds. These changes unblock training/inference so `preds`, `sub`, and finally `/kaggle/working/submission.csv` are created without altering the model architecture, loss, or prediction/thresholding semantics. The result is an end-to-end run that produces a valid submission CSV.'
- What this solution (achieved 0.27161) has done: 'The timeout is dominated by image decoding/resizing done twice (train/val and then test) plus caching the full 15k+ images in RAM, which can thrash memory and slow the pipeline. I keep the exact same model, epochs, batch size, and thresholding, but make the input pipeline faster by enabling `tf.data` non-deterministic map ordering (without changing shuffle/seed), removing in-memory caching, and precomputing multi-hot labels with vectorized pandas operations. I also avoid unnecessary `.repeat()` overhead by letting `fit` handle epoch iteration directly (same steps/epochs semantics) and keep prefetching/parallelism tuned. These changes preserve the algorithm and outputs up to negligible floating-point differences while cutting wall-clock time substantially.'
- What this solution (achieved 0.27161) has done: 'I fix the TensorFlow/protobuf import crash by forcing the pure-Python protobuf implementation earlier and purging any already-imported protobuf modules before TensorFlow is imported, then retrying the import in a clean state. This is a runtime-stability fix only and does not change your model, training loop, or prediction semantics. Because your current score (0.27161) is already above the target (0.1503) and higher-is-better, I not make any score-improving changes; the goal is simply to get an end-to-end run that reliably produces `submission.csv`. The rest of the pipeline (data paths, tf.data, CNN, thresholding, submission formatting) is preserved.'
- What this solution (achieved 0.27161) has done: 'I fix the TensorFlow/protobuf import crash by forcing the pure-Python protobuf implementation earlier and more aggressively purging any already-imported `google.protobuf` modules (including subpackages) before importing TensorFlow. This is a runtime-stability fix only and does not change your model, training loop, data split, or prediction/thresholding semantics. Since your current score (0.27161) is already above the target (0.1503) and higher-is-better, I not make any score-improving changes; the goal is simply to get an end-to-end run that reliably produces `/kaggle/working/submission.csv`. The rest of the pipeline (paths, tf.data, CNN, and submission formatting) is preserved.'
- What this solution (achieved 0.27161) has done: 'I fix the TensorFlow import crash by forcing the pure-Python protobuf implementation *before* any protobuf/TensorFlow modules can load, and by purging only the relevant `google.protobuf`/`tensorflow` modules (not the whole `google` namespace, which can destabilize other deps). This is a runtime-stability change only; it does not alter your model, training loop, data split, or thresholding, so the score should remain essentially unchanged (and it’s already above the target band). I also keep the same paths/IO and ensure the script always reaches the CSV write step with the required `image,labels` columns.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def import_tensorflow_safely():
    """
    Import TensorFlow with a robust fallback against protobuf descriptor issues.

    Common crash in some Kaggle images:
      AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

    This can persist if modules were partially imported, so we purge them and retry.
    """
    import sys
    import importlib

    def _purge_prefixes(prefixes):
        for k in list(sys.modules.keys()):
            for p in prefixes:
                if k == p or k.startswith(p + "."):
                    sys.modules.pop(k, None)
                    break

    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

    _purge_prefixes(("tensorflow", "google.protobuf"))

    try:
        import tensorflow as tf  # noqa: F401

        return tf
    except Exception as e1:
        _purge_prefixes(("tensorflow", "google.protobuf"))
        importlib.invalidate_caches()
        os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
        os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
        try:
            import tensorflow as tf  # noqa: F401

            return tf
        except Exception as e2:
            raise RuntimeError(
                "TensorFlow import failed (including protobuf pure-Python fallback).\n"
                f"First error: {repr(e1)}\nSecond error: {repr(e2)}"
            )


import random
import numpy as np
import pandas as pd

tf = import_tensorflow_safely()

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

try:
    tf.config.optimizer.set_jit(
        False
    )  # keep numerics stable; also avoids compile overhead
except Exception:
    pass

CANDIDATE_INPUT_ROOTS = [
    "/kaggle/input/plant-pathology-2021-fgvc8",
    "/kaggle/data/plant-pathology-2021-fgvc8",
    "/kaggle/input",
    "/kaggle/data",
]
INPUT_ROOT = None
for root in CANDIDATE_INPUT_ROOTS:
    if os.path.exists(os.path.join(root, "train.csv")) and os.path.exists(
        os.path.join(root, "sample_submission.csv")
    ):
        INPUT_ROOT = root
        break
    nested = os.path.join(root, "plant-pathology-2021-fgvc8")
    if os.path.exists(os.path.join(nested, "train.csv")) and os.path.exists(
        os.path.join(nested, "sample_submission.csv")
    ):
        INPUT_ROOT = nested
        break

if INPUT_ROOT is None:
    raise FileNotFoundError(
        "Could not find train.csv and sample_submission.csv under expected Kaggle input paths. "
        f"Tried: {CANDIDATE_INPUT_ROOTS}"
    )

WORK_ROOT = "/kaggle/working"
TMP_ROOT = "/kaggle/tmp"

train_csv_path = f"{INPUT_ROOT}/train.csv"
sample_sub_path = f"{INPUT_ROOT}/sample_submission.csv"
train_img_dir = f"{INPUT_ROOT}/train_images"
test_img_dir = f"{INPUT_ROOT}/test_images"

assert os.path.exists(train_csv_path), f"Missing train.csv at {train_csv_path}"
assert os.path.exists(
    sample_sub_path
), f"Missing sample_submission.csv at {sample_sub_path}"
assert os.path.isdir(train_img_dir), f"Missing train_images dir at {train_img_dir}"
assert os.path.isdir(test_img_dir), f"Missing test_images dir at {test_img_dir}"

print("Using INPUT_ROOT:", INPUT_ROOT)
print("TensorFlow:", tf.__version__)
print("Train images dir:", train_img_dir)
print("Test images dir:", test_img_dir)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_df = pd.read_csv(train_csv_path)
sample_sub = pd.read_csv(sample_sub_path)

LABELS = ["complex", "frog_eye_leaf_spot", "powdery_mildew", "rust", "scab"]

train_df["filepath"] = (train_img_dir + "/" + train_df["image"].astype(str)).astype(str)

sample_check = train_df["filepath"].head(32).tolist()
missing_sample = [p for p in sample_check if not tf.io.gfile.exists(p)]
assert (
    len(missing_sample) == 0
), f"Some sampled train image files are missing (sample): {missing_sample[:3]}"

labels_series = train_df["labels"].astype(str)
y = np.stack(
    [
        labels_series.str.contains(rf"(?:^|\s){lab}(?:\s|$)", regex=True).to_numpy(
            dtype=np.float32
        )
        for lab in LABELS
    ],
    axis=1,
)

idx = np.arange(len(train_df))
rng = np.random.RandomState(SEED)
rng.shuffle(idx)
val_frac = 0.1
val_size = int(len(idx) * val_frac)
val_idx = idx[:val_size]
tr_idx = idx[val_size:]

train_paths = train_df.iloc[tr_idx]["filepath"].to_numpy(dtype=str)
train_y = y[tr_idx].astype(np.float32, copy=False)
val_paths = train_df.iloc[val_idx]["filepath"].to_numpy(dtype=str)
val_y = y[val_idx].astype(np.float32, copy=False)

print("Train/Val sizes:", len(train_paths), len(val_paths))
print("Label prevalence (train):", dict(zip(LABELS, train_y.mean(axis=0).round(4))))



## === cell 2
AUTOTUNE = tf.data.AUTOTUNE
TARGET_SIZE = (380, 380)
BATCH_SIZE = 16  # unchanged


@tf.function(reduce_retracing=True)
def _decode_resize_normalize(img_bytes):
    img = tf.image.decode_jpeg(
        img_bytes,
        channels=3,
        dct_method="INTEGER_FAST",
        ratio=1,
        fancy_upscaling=False,
        try_recover_truncated=True,
        acceptable_fraction=1.0,
    )
    img = tf.image.resize(img, TARGET_SIZE, method="bilinear")
    img = tf.cast(img, tf.float32) / 255.0
    return img


def _load_preprocess(path, label=None):
    img_bytes = tf.io.read_file(path)
    img = _decode_resize_normalize(img_bytes)
    if label is None:
        return img
    return img, label


def _safe_setattr(obj, name, value):
    try:
        setattr(obj, name, value)
        return True
    except Exception:
        return False


def make_ds(paths, labels=None, training=False, cache_in_memory=False):
    options = tf.data.Options()
    _safe_setattr(options, "experimental_deterministic", False)

    opt = options.experimental_optimization
    _safe_setattr(opt, "map_parallelization", True)
    _safe_setattr(opt, "parallel_batch", True)
    _safe_setattr(opt, "map_fusion", True)
    _safe_setattr(opt, "apply_default_optimizations", True)

    tp = options.threading
    _safe_setattr(tp, "private_threadpool_size", 16)
    _safe_setattr(tp, "max_intra_op_parallelism", 1)

    if labels is None:
        ds = tf.data.Dataset.from_tensor_slices(paths).with_options(options)
        ds = ds.map(
            lambda p: _load_preprocess(p, None),
            num_parallel_calls=AUTOTUNE,
            deterministic=False,
        )
        if cache_in_memory:
            ds = ds.cache()
        ds = ds.apply(tf.data.experimental.ignore_errors())
        ds = ds.batch(BATCH_SIZE, drop_remainder=False)
        ds = ds.prefetch(AUTOTUNE)
        return ds

    ds = tf.data.Dataset.from_tensor_slices((paths, labels)).with_options(options)
    if training:
        ds = ds.shuffle(4096, seed=SEED, reshuffle_each_iteration=True)

    ds = ds.map(_load_preprocess, num_parallel_calls=AUTOTUNE, deterministic=False)

    if cache_in_memory:
        ds = ds.cache()

    ds = ds.apply(tf.data.experimental.ignore_errors())
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_ds = make_ds(train_paths, train_y, training=True, cache_in_memory=False)
val_ds = make_ds(val_paths, val_y, training=False, cache_in_memory=False)

steps_per_epoch = int(np.ceil(len(train_paths) / BATCH_SIZE))
validation_steps = int(np.ceil(len(val_paths) / BATCH_SIZE))



## === cell 3
inputs = tf.keras.Input(shape=(TARGET_SIZE[0], TARGET_SIZE[1], 3))
x = tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu")(inputs)
x = tf.keras.layers.MaxPooling2D()(x)
x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
x = tf.keras.layers.MaxPooling2D()(x)
x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = tf.keras.layers.MaxPooling2D()(x)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2, seed=SEED)(x)
outputs = tf.keras.layers.Dense(len(LABELS), activation="sigmoid")(x)

model = tf.keras.Model(inputs, outputs)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    run_eagerly=False,
)

print(model.summary())

EPOCHS = 2

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    verbose=1,
)



## === cell 4
test_paths = (test_img_dir + "/" + sample_sub["image"].astype(str)).to_numpy(dtype=str)
test_ds = make_ds(test_paths, labels=None, training=False, cache_in_memory=False)

preds = model.predict(test_ds, verbose=1).astype(np.float32)

print("Preds shape:", preds.shape)
print("Preds range:", float(np.min(preds)), float(np.max(preds)))



## === cell 5
threshold = 0.4

z = (preds > threshold).astype(np.int32)
predictions = [[LABELS[i] for i, flag in enumerate(row) if flag == 1] for row in z]
predictions_str = [" ".join(p) if len(p) > 0 else "healthy" for p in predictions]

print("Example prediction strings:", predictions_str[:5])

sub = pd.DataFrame({"image": sample_sub["image"].values, "labels": predictions_str})
assert sub.shape[0] == sample_sub.shape[0]
assert list(sub.columns) == ["image", "labels"]

sub.head()



## === cell 6
out_path = f"{WORK_ROOT}/submission.csv"
sub.to_csv(out_path, index=False)
print("Wrote:", out_path)
print("Submission preview:")
print(sub.head(10).to_string(index=False))
print("Submission rows:", len(sub))
