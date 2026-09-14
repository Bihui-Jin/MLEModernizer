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

geopandas==0.14.4
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

0.6601477377654648

# 6. Current score

0.38935

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.09848) has done: 'The timeout is dominated by per-image Python overhead and calling the model one image at a time, which prevents TensorFlow from efficiently pipelining CPU decode/resize and batching inference. I keep the exact same model loading/inference semantics and thresholding logic, but switch to a `tf.data` input pipeline that decodes/resizes in parallel, batches images, and prefetches to overlap I/O and compute. I also wrap inference in a `tf.function` (same computations, less eager overhead) and avoid repeated conversions by moving to batched NumPy post-processing. These changes are equivalent in results (same preprocessing ops, same thresholding/argmax fallback), but drastically reduce runtime.'
- What this solution (achieved 0.28474) has done: 'You’re hitting a TensorFlow/Protobuf incompatibility caused by forcing `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`, which triggers the `MessageFactory.GetPrototype` error before the model can even load; removing those env overrides fixes runtime. Next, the current low score is consistent with label-index mismatch: you build `dataset_labels` from `train.csv`, but the external SavedModel likely uses a fixed class order (commonly the competition’s standard order), so predictions are being mapped to the wrong label names; I pin the label order to the known Plant Pathology 2021 class list to align outputs correctly. Finally, I keep your exact inference/thresholding semantics, but also ensure the submission rows are aligned to `sample_submission.csv` ordering (safer for Kaggle ingestion) and keep output as a valid `submission.csv`.'
- What this solution (achieved 0.23993) has done: 'We fix the immediate crash by ensuring no protobuf-breaking environment variables are set *before* importing TensorFlow, and by forcing the safe pure-Python protobuf implementation consistently. Then we keep your exact model/inference/threshold logic, but make the external SavedModel output selection more robust (prefer the common “predictions/probabilities” keys) to avoid silently grabbing the wrong tensor. Finally, we keep the submission aligned to `sample_submission.csv` order and always write `submission.csv` with the required columns.'
- What this solution (achieved 0.22303) has done: 'I fix the immediate TensorFlow/Protobuf crash by removing the environment override that forces the pure-Python protobuf implementation (it’s incompatible with the protobuf version in this Kaggle image and triggers `MessageFactory.GetPrototype` errors). Then I keep your exact inference + threshold/argmax fallback logic, but make the SavedModel output tensor selection slightly safer by preferring 2D `(batch, classes)` outputs when multiple tensors exist (this is score-positive without changing the model). Finally, I ensure test image paths are correct, predictions align to `sample_submission.csv` order, and `submission.csv` is always written with the required columns.'
- What this solution (achieved 0.13878) has done: 'I fix the TensorFlow import crash by removing the incompatible protobuf environment overrides (they prevent TF from importing in this Kaggle image) and add a safe fallback to use the default protobuf implementation. Then I keep your exact inference + threshold/argmax fallback logic, but ensure TensorFlow is imported successfully so `tf` is defined for later cells. Finally, I keep the submission generation identical but make the test/sample paths robust to both `../input/...` and `/kaggle/input/...` so the script always writes a valid `submission.csv`.'
- What this solution (achieved 0.2275) has done: 'I fix the TensorFlow/Protobuf import crash by forcing a protobuf version compatible with TensorFlow 2.18 in this Kaggle image (the current environment’s protobuf 6.x triggers the `MessageFactory.GetPrototype` error). Then I keep your exact model/inference and threshold/argmax fallback logic, but ensure imports happen only after the protobuf pin so the pipeline runs end-to-end. Finally, I keep submission formatting identical while making sure the script always writes `submission.csv` with the required `image,labels` columns.'
- What this solution (achieved 0.33163) has done: 'Your current score (0.2275) is far below the target (0.6601), so we should improve performance with minimal, metric-aligned changes that don’t alter the model itself. The biggest likely issue is post-processing for mean F1 in a multilabel setting: using a single global threshold (0.6) is usually suboptimal and often suppresses true positives for rarer classes. I keep the same model/inference pipeline, but replace the fixed threshold with lightweight per-class threshold calibration on a small validation split from `train.csv` (using the same model outputs), optimizing mean sample-wise F1—this directly matches the competition metric and typically yields a large jump. Then we apply those calibrated thresholds to test predictions, keeping the same “argmax fallback if nothing selected” behavior and the same submission format.'
- What this solution (achieved 0.38935) has done: 'Your current score (0.33163) is far below the target (0.66015), so we should improve post-processing in a way that directly matches the mean F1 metric without changing the model itself. The largest win with minimal risk is to calibrate thresholds more effectively: (1) use a deterministic train/val split with stratification by number of labels so the calibration set better represents multilabel difficulty, and (2) optimize thresholds jointly by searching a slightly denser per-class grid around the best value rather than a coarse fixed grid. I keep your exact inference pipeline, preprocessing, and the same “argmax fallback if nothing selected” behavior, and still write `submission.csv` in the required format. These changes only affect how probabilities are turned into labels, which is exactly where mean F1 is won/lost.'

# 9. Code solution

## === cell 0
import os
import sys
import subprocess


def _ensure_protobuf_compat():
    try:
        import google.protobuf  # noqa: F401
        from google.protobuf import __version__ as pb_ver
    except Exception:
        pb_ver = None

    def _major(v):
        try:
            return int(str(v).split(".", 1)[0])
        except Exception:
            return None

    if pb_ver is None or _major(pb_ver) >= 5:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "protobuf==4.25.3"]
        )
        for k in list(sys.modules.keys()):
            if k.startswith("google.protobuf"):
                sys.modules.pop(k, None)


_ensure_protobuf_compat()

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import pandas as pd
import tensorflow as tf
import numpy as np



## === cell 1
output_dir = "./"
test_dir = "../input/plant-pathology-2021-fgvc8/test_images/"

model_dir = "../input/model-aug-epoch20/model_complete_with_augEpoch:20"

image_dims = (300, 300, 3)

dataset_labels = [
    "complex",
    "frog_eye_leaf_spot",
    "healthy",
    "powdery_mildew",
    "rust",
    "scab",
]


def _find_savedmodel_dir(preferred_path: str, fallback_root: str) -> str:
    """Return a directory that contains a TensorFlow SavedModel (saved_model.pb)."""
    preferred_path = os.path.abspath(preferred_path)
    fallback_root = os.path.abspath(fallback_root)

    def is_savedmodel_dir(p: str) -> bool:
        return os.path.isdir(p) and os.path.exists(os.path.join(p, "saved_model.pb"))

    if is_savedmodel_dir(preferred_path):
        return preferred_path

    candidates = []
    candidates.append(preferred_path.replace(":", "_"))
    candidates.append(preferred_path.replace(":", ""))
    candidates.append(preferred_path.split(":")[0])
    for c in candidates:
        if is_savedmodel_dir(c):
            return c

    if os.path.isdir(fallback_root):
        for name in sorted(os.listdir(fallback_root)):
            p = os.path.join(fallback_root, name)
            if is_savedmodel_dir(p):
                return p

        for root, dirs, files in os.walk(fallback_root):
            if "saved_model.pb" in files:
                return root

    raise FileNotFoundError(
        "Could not locate a SavedModel directory. Tried preferred path and searched under: "
        f"{fallback_root}\nPreferred: {preferred_path}"
    )


try:
    model_dir = _find_savedmodel_dir(model_dir, "../input/model-aug-epoch20")
    has_external_model = True
except FileNotFoundError:
    has_external_model = False
    model_dir = None



## === cell 2
if __name__ == "__main__":
    if not os.path.isdir(test_dir):
        alt_test_dir = "/kaggle/input/plant-pathology-2021-fgvc8/test_images"
        if os.path.isdir(alt_test_dir):
            test_dir = alt_test_dir
        else:
            alt_test_dir2 = "/kaggle/input/plant-pathology-2021-fgvc8/plant-pathology-2021-fgvc8/test_images"
            if os.path.isdir(alt_test_dir2):
                test_dir = alt_test_dir2
            else:
                raise FileNotFoundError(f"test_dir not found: {test_dir}")

    sample_sub_path = "../input/plant-pathology-2021-fgvc8/sample_submission.csv"
    if not os.path.exists(sample_sub_path):
        sample_sub_path = (
            "/kaggle/input/plant-pathology-2021-fgvc8/sample_submission.csv"
        )
    if not os.path.exists(sample_sub_path):
        sample_sub_path = "/kaggle/input/plant-pathology-2021-fgvc8/plant-pathology-2021-fgvc8/sample_submission.csv"
    sample_df = pd.read_csv(sample_sub_path)
    sample_images = sample_df["image"].tolist()

    train_csv_path = "../input/plant-pathology-2021-fgvc8/train.csv"
    if not os.path.exists(train_csv_path):
        train_csv_path = "/kaggle/input/plant-pathology-2021-fgvc8/train.csv"
    if not os.path.exists(train_csv_path):
        train_csv_path = "/kaggle/input/plant-pathology-2021-fgvc8/plant-pathology-2021-fgvcvc8/train.csv"
    if not os.path.exists(train_csv_path):
        train_csv_path = "/kaggle/input/plant-pathology-2021-fgvc8/plant-pathology-2021-fgvc8/train.csv"
    train_df = pd.read_csv(train_csv_path)

    train_img_dir = "../input/plant-pathology-2021-fgvc8/train_images/"
    if not os.path.isdir(train_img_dir):
        train_img_dir = "/kaggle/input/plant-pathology-2021-fgvc8/train_images"
    if not os.path.isdir(train_img_dir):
        train_img_dir = "/kaggle/input/plant-pathology-2021-fgvc8/plant-pathology-2021-fgvc8/train_images"
    if not os.path.isdir(train_img_dir):
        raise FileNotFoundError(
            "Could not locate train_images directory for threshold calibration."
        )

    if has_external_model:
        loaded = tf.saved_model.load(model_dir)
        if (
            hasattr(loaded, "signatures")
            and isinstance(loaded.signatures, dict)
            and len(loaded.signatures) > 0
        ):
            if "serving_default" in loaded.signatures:
                serving_fn = loaded.signatures["serving_default"]
            else:
                serving_fn = list(loaded.signatures.values())[0]
        else:
            raise RuntimeError(f"SavedModel has no signatures dict at: {model_dir}")

        structured_inputs = serving_fn.structured_input_signature
        _, kw = structured_inputs
        input_key = None
        if isinstance(kw, dict) and len(kw) > 0:
            input_key = list(kw.keys())[0]

        preferred_out_keys = [
            "predictions",
            "prediction",
            "probabilities",
            "probs",
            "outputs",
            "output",
            "dense",
            "sigmoid",
        ]

        def _select_output_tensor(out_dict):
            if not isinstance(out_dict, dict):
                return out_dict
            for k in preferred_out_keys:
                if k in out_dict:
                    return out_dict[k]
            rank2 = []
            for k, v in out_dict.items():
                try:
                    if (
                        hasattr(v, "shape")
                        and v.shape is not None
                        and len(v.shape) == 2
                    ):
                        rank2.append(k)
                except Exception:
                    pass
            if len(rank2) > 0:
                return out_dict[sorted(rank2)[0]]
            return out_dict[sorted(out_dict.keys())[0]]

        @tf.function(reduce_retracing=True)
        def _infer_batch(batch_images):
            if input_key is None:
                out = serving_fn(batch_images)
            else:
                out = serving_fn(**{input_key: batch_images})
            return _select_output_tensor(out)

    else:
        base = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights="imagenet",
            input_shape=image_dims,
            pooling="avg",
        )
        inp = tf.keras.Input(shape=image_dims, dtype=tf.float32, name="input_image")
        x = tf.keras.applications.efficientnet.preprocess_input(inp * 255.0)
        x = base(x, training=False)
        out = tf.keras.layers.Dense(
            len(dataset_labels), activation="sigmoid", name="pred"
        )(x)
        infer_model = tf.keras.Model(inputs=inp, outputs=out)

        @tf.function(reduce_retracing=True)
        def _infer_batch(batch_images):
            return infer_model(batch_images, training=False)

    def _load_and_preprocess(path):
        img_bytes = tf.io.read_file(path)
        img = tf.io.decode_jpeg(img_bytes, channels=3)
        img = tf.image.convert_image_dtype(img, dtype=tf.float32)  # -> [0,1]
        img.set_shape([None, None, 3])
        img = tf.image.resize(
            img, [image_dims[0], image_dims[1]], method="bilinear", antialias=True
        )
        return img

    def _labels_to_multihot(
        label_str: str, class_to_idx: dict, num_classes: int
    ) -> np.ndarray:
        y = np.zeros((num_classes,), dtype=np.int32)
        if isinstance(label_str, str) and label_str.strip():
            for t in label_str.split():
                if t in class_to_idx:
                    y[class_to_idx[t]] = 1
        return y

    def _mean_f1_samples(y_true_bin: np.ndarray, y_pred_bin: np.ndarray) -> float:
        tp = (y_true_bin & y_pred_bin).sum(axis=1).astype(np.float32)
        fp = ((1 - y_true_bin) & y_pred_bin).sum(axis=1).astype(np.float32)
        fn = (y_true_bin & (1 - y_pred_bin)).sum(axis=1).astype(np.float32)
        denom = 2.0 * tp + fp + fn
        f1 = np.where(denom > 0, (2.0 * tp) / denom, 1.0)  # if both empty -> perfect
        return float(f1.mean())

    rng = np.random.RandomState(1337)
    n_total = len(train_df)
    n_val = int(
        min(4096, max(1024, 0.15 * n_total))
    )  # slightly larger, still fast with batched inference

    label_counts = (
        train_df["labels"]
        .astype(str)
        .apply(lambda s: 0 if s.strip() == "" else len(s.split()))
    )
    bins = pd.cut(
        label_counts.clip(upper=4), bins=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], labels=False
    )
    bins = bins.astype(int).values

    val_idx = []
    for b in np.unique(bins):
        idx_b = np.where(bins == b)[0]
        if idx_b.size == 0:
            continue
        take = max(1, int(round(n_val * (idx_b.size / n_total))))
        take = min(take, idx_b.size)
        val_idx.append(rng.choice(idx_b, size=take, replace=False))
    val_idx = np.unique(np.concatenate(val_idx))
    if val_idx.size > n_val:
        val_idx = rng.choice(val_idx, size=n_val, replace=False)
    elif val_idx.size < n_val:
        remaining = np.setdiff1d(np.arange(n_total), val_idx, assume_unique=False)
        extra = rng.choice(remaining, size=(n_val - val_idx.size), replace=False)
        val_idx = np.concatenate([val_idx, extra])

    val_df = train_df.iloc[val_idx].reset_index(drop=True)

    class_to_idx = {c: i for i, c in enumerate(dataset_labels)}
    y_true = np.stack(
        [
            _labels_to_multihot(s, class_to_idx, len(dataset_labels))
            for s in val_df["labels"].astype(str).tolist()
        ],
        axis=0,
    )

    batch_size = 64
    val_paths = [os.path.join(train_img_dir, n) for n in val_df["image"].tolist()]
    val_ds = tf.data.Dataset.from_tensor_slices(val_paths)
    val_ds = val_ds.map(_load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)

    val_probs_list = []
    for batch_imgs in val_ds:
        batch_probs = _infer_batch(batch_imgs)
        batch_probs = tf.convert_to_tensor(batch_probs)
        if batch_probs.shape.rank is not None and batch_probs.shape.rank > 2:
            batch_probs = tf.reshape(batch_probs, [tf.shape(batch_probs)[0], -1])
        val_probs_list.append(batch_probs.numpy().astype("float32"))
    val_probs = np.concatenate(val_probs_list, axis=0)  # (N, C)

    base_grid = np.array(
        [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70],
        dtype=np.float32,
    )

    best_thrs = np.full((len(dataset_labels),), 0.50, dtype=np.float32)

    def _apply_thresholds(probs: np.ndarray, thrs: np.ndarray) -> np.ndarray:
        y_pred = (probs > thrs[None, :]).astype(np.int32)
        empty = y_pred.sum(axis=1) == 0
        if np.any(empty):
            argm = probs[empty].argmax(axis=1)
            y_pred[empty, :] = 0
            y_pred[empty, argm] = 1
        return y_pred

    for outer in range(3):
        for c in range(len(dataset_labels)):
            cur = float(best_thrs[c])
            if outer == 0:
                grid = base_grid
            else:
                lo = max(0.05, cur - 0.10)
                hi = min(0.95, cur + 0.10)
                grid = np.linspace(lo, hi, 11, dtype=np.float32)

            cur_best_thr = cur
            cur_best_sc = -1.0
            for t in grid:
                thrs_try = best_thrs.copy()
                thrs_try[c] = float(t)
                sc = _mean_f1_samples(y_true, _apply_thresholds(val_probs, thrs_try))
                if sc > cur_best_sc:
                    cur_best_sc = sc
                    cur_best_thr = float(t)
            best_thrs[c] = cur_best_thr

    best_score = _mean_f1_samples(y_true, _apply_thresholds(val_probs, best_thrs))

    print(
        "Calibrated per-class thresholds:",
        dict(zip(dataset_labels, [float(x) for x in best_thrs.tolist()])),
    )
    print("Validation mean F1 (calibration split):", best_score)

    images_path_list = sample_images
    file_paths = [os.path.join(test_dir, n) for n in images_path_list]
    ds = tf.data.Dataset.from_tensor_slices(file_paths)
    ds = ds.map(_load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    values = []
    offset = 0
    for batch_imgs in ds:
        batch_probs = _infer_batch(batch_imgs)
        batch_probs = tf.convert_to_tensor(batch_probs)
        if batch_probs.shape.rank is not None and batch_probs.shape.rank > 2:
            batch_probs = tf.reshape(batch_probs, [tf.shape(batch_probs)[0], -1])
        batch_probs_np = batch_probs.numpy().astype("float32")  # (B, C)

        bsz = batch_probs_np.shape[0]
        for i in range(bsz):
            probs_np = batch_probs_np[i]
            picked = [j for j, v in enumerate(probs_np) if v > float(best_thrs[j])]
            if len(picked) == 0:
                picked = [int(probs_np.argmax())]
            classes_img = " ".join([dataset_labels[j] for j in picked]).strip()
            values.append([images_path_list[offset + i], classes_img])
        offset += bsz

    pred_df = pd.DataFrame(values, columns=["image", "labels"])
    sub_df = sample_df[["image"]].merge(pred_df, on="image", how="left")
    sub_df["labels"] = sub_df["labels"].fillna("healthy")

    out_path = os.path.join(output_dir, "submission.csv")
    sub_df.to_csv(out_path, index=False)
    print(f"Wrote submission to: {out_path} with shape={sub_df.shape}")
    print(sub_df.head())
