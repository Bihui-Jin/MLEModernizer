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

0.7871283471837496

# 6. Current score

0.28656

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.24507) has done: 'I fix the import crash in the first cell by avoiding the `kaggle_datasets` import (it isn’t used and triggers a protobuf incompatibility in this environment). Then I fix the missing model file issue by loading the provided `sample_submission.csv` and generating a valid, correctly ordered submission even when the external `.h5` model input folder is unavailable (so the notebook always yields `submission.csv`). Finally, I fix the submission length mismatch by ensuring the `image` list used in the DataFrame matches exactly the prediction list, and I preserve your existing thresholding-to-label-string logic unchanged.'
- What this solution (achieved 0.24507) has done: 'You’re hitting a TensorFlow import crash caused by an incompatible protobuf runtime in this Kaggle environment, so the first fix is to force TensorFlow to use the pure-Python protobuf implementation before importing `tensorflow`. Next, your label mapping is missing class index `5` (which corresponds to `multiple_diseases` in this competition), which makes your post-processing ignore an entire class and severely hurts mean F1; adding this missing class preserves your existing thresholding logic while making predictions semantically correct. Finally, I keep your model-loading/fallback behavior intact but ensure paths are robust and the submission is always aligned to `sample_submission.csv` ordering and written as `submission.csv`.'
- What this solution (achieved 0.24507) has done: 'I fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf implementation *and* disabling the C++ implementation before any TensorFlow import, plus clearing any pre-imported protobuf modules that can keep the bad state. Then I remove the redundant second TensorFlow import to avoid re-triggering the same issue. Finally, I keep your model loading and label-thresholding logic intact, but make the fallback probabilities predict “healthy” (instead of all-zeros), which legitimately improves mean F1 when the external model file is missing and moves your score toward the target.'
- What this solution (achieved 0.24507) has done: 'We need to fix the TensorFlow/protobuf import crash (`MessageFactory.GetPrototype`) so the notebook can actually run end-to-end and generate predictions instead of always falling back to “healthy”. The minimal, robust fix in Kaggle is to remove the protobuf env overrides and explicitly pin the `protobuf` Python package to a TensorFlow-compatible version at runtime (no internet needed), then import TensorFlow once. I keep your dataset/model/prediction/threshold-to-label-string logic unchanged, but ensure the test image path discovery is robust and the submission is written in the sample submission order with the required `.csv` suffix. This should move the score up substantially toward the target because the model load and be used.'
- What this solution (achieved 0.24507) has done: 'Your score is far below the target, so the most likely reason is that you are not actually using the intended pretrained model (the code falls back to all-healthy when the `.h5` isn’t found), which tanks mean F1. I keep your exact model, dataset pipeline, and threshold-to-label-string logic, but make the model discovery robust by searching all standard Kaggle input locations for `5SMResNet50.h5` (including nested dataset folders) and failing only if it truly doesn’t exist. I also force deterministic ordering/alignment by building the submission strictly from `sample_submission.csv` image order (then loading images by that same order), which avoids any filename ordering mismatch harming F1. These are minimal changes focused on ensuring the real model is used and predictions align with the evaluation rows.'
- What this solution (achieved 0.24507) has done: 'Your score is far below the target, so the most likely issue is that predictions and `sample_submission.csv` rows are misaligned when the code falls back to `os.listdir()` ordering (or when some expected files aren’t found), causing many labels to be assigned to the wrong images and crushing mean F1. I make image path construction strictly follow `sample_submission.csv` order and, if any files are missing, still keep that order by filtering to existing files while preserving indices, then merging back to the full sample order (so alignment is always correct). I also fix a small but critical path typo (`fgvcvc8`) and ensure the dataset is built from exactly the same ordered filenames used to build the final submission. These are minimal changes that preserve your model, preprocessing, thresholds, and label-string logic, but should materially improve the score toward the target.'
- What this solution (achieved 0.24507) has done: 'Your score gap to the target is large, so the most likely blocker is that you’re still not using the real pretrained model at inference time (falling back to all-healthy), which caps mean F1 around the class prior. I keep your model, preprocessing, and threshold-to-label-string logic unchanged, but make model discovery more robust by also searching common writable locations (including `/kaggle/data` and `/kaggle/working`) and by printing the exact resolved path used. I also make the test dataset deterministic and slightly safer by adding `.prefetch(AUTO)` and ensuring `IMAGE_PATHS` stays aligned to the sample submission order (including when some files are missing). These are minimal, execution-safe changes aimed specifically at getting real model predictions aligned to the correct rows, which should move the score substantially toward your target.'
- What this solution (achieved 0.28656) has done: 'Your score is far below the target, and the biggest likely reason (given your current thresholds/post-processing are already reasonable) is a mismatch between the model’s output class order and your `name` mapping, which would make otherwise good probabilities turn into wrong label strings and crush mean F1. I keep your model, preprocessing, prediction call, and threshold-to-label-string logic intact, but I add a minimal, safe auto-detection step that tries a few common class-order permutations and selects the one that best matches the training-label distribution (then uses that mapping for test). This doesn’t change the architecture or inference—only corrects the interpretation of the 7 logits/probabilities into the right class names. I also make sure the model output dimension is checked (7) and keep the submission aligned to `sample_submission.csv` as you already do.'
- What this solution (achieved 0.28656) has done: 'Your current gap to the target is large, so we should make a minimal change that improves mean F1 without changing the model or thresholds: fix the post-processing to respect the selected `name` mapping when checking thresholds (right now it thresholds `line[i]` directly, which is wrong whenever the mapping isn’t the identity). I add a tiny “reorder” step that converts the model outputs into a canonical class order (`scab, frog_eye_leaf_spot, rust, powdery_mildew, multiple_diseases, complex, healthy`) and then run your exact same threshold/complex logic on that canonical order. This preserves your core logic (same model, same thresholds, same string-building rules) but corrects a semantic bug that can severely depress F1. The rest of the pipeline (paths, dataset, submission merge in sample order) stays unchanged and still writes `submission.csv`.'
- What this solution (achieved 0.28656) has done: 'Your current score (0.28656) is far below the target (0.78713), so we should make a minimal change that legitimately improves mean F1 without changing your model or thresholding rules: calibrate the per-class thresholds using the training set, then apply the same calibrated thresholds at test-time. This preserves your exact post-processing logic (two threshold dictionaries + “add complex if ≥2 diseases”) but sets the threshold values to something closer to F1-optimal for your model’s probability scale. To keep it fast and stable, the calibration runs on a small fixed subset of training images and uses the same `decode_image` pipeline and the already-selected/canonicalized class order. The rest of your pipeline (protobuf pin, model discovery/loading, canonical class reorder, submission alignment/merge, and writing `submission.csv`) stays intact.'
- What this solution (achieved 0.28656) has done: 'We keep your model/inference and label-string logic intact, but make one targeted change that usually lifts mean F1 a lot: calibrate thresholds using a *proper mean F1 proxy* (sample-wise F1 averaged across images), instead of the current micro-F1 which doesn’t match the competition metric. This is a minimal semantic fix (same predictions, same string-building rules, same “add complex if ≥2 diseases” logic), but it chooses thresholds that better align with evaluation. To stay within time, we calibrate on a modest fixed subset (same as you already do) and keep the same small threshold grids. Everything else (protobuf pin, model discovery, canonical class reorder, sample_submission alignment, writing `submission.csv`) remains unchanged.'
- What this solution (achieved 0.28656) has done: 'The current score is far below the target, so we should make a small change that directly improves mean F1 without altering your model or inference: expand the threshold calibration from a coarse 2D grid (single shared thr1/thr2) to a lightweight per-class calibration (still using your exact label-string rules, including the “add complex if ≥2 diseases” logic). This better matches each class’s probability scale and usually gives a large, legitimate lift in mean sample-wise F1 while keeping the same architecture, preprocessing, and prediction pipeline. To keep runtime under control, calibration still runs on a fixed subset (512) and uses a coordinate-ascent style sweep with a small grid per class. Everything else (protobuf pin, model discovery, canonical reorder, sample_submission alignment, and writing `submission.csv`) remains intact.'

# 9. Code solution

## === cell 0
import os, re, math, random, sys, importlib, pathlib, subprocess


def _ensure_compatible_protobuf():
    try:
        import google.protobuf  # noqa: F401
        from google.protobuf import __version__ as pb_ver
    except Exception:
        pb_ver = None

    target = "protobuf==3.20.3"
    try:
        if pb_ver is None or tuple(int(x) for x in pb_ver.split(".")[:2]) >= (4, 0):
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", target]
            )
            importlib.invalidate_caches()
            for m in list(sys.modules.keys()):
                if m.startswith("google.protobuf"):
                    sys.modules.pop(m, None)
    except Exception as e:
        print("WARNING: protobuf pin attempt failed:", repr(e))


_ensure_compatible_protobuf()

import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow.keras.backend as K

print("tf:", tf.__version__)
print("tf.keras:", tf.keras.__version__)




## === cell 1
def decode_image(filename, label=None, image_size=(512, 512)):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, image_size)
    if label is None:
        return image
    else:
        return image, label




## === cell 2
BATCH_SIZE = 32




## === cell 3
sample_path = "../input/plant-pathology-2021-fgvc8/sample_submission.csv"
if not os.path.exists(sample_path):
    sample_path_abs = "/kaggle/input/plant-pathology-2021-fgvc8/sample_submission.csv"
    if os.path.exists(sample_path_abs):
        sample_path = sample_path_abs
if not os.path.exists(sample_path):
    sample_path_alt = "/kaggle/data/plant-pathology-2021-fgvc8/sample_submission.csv"
    if os.path.exists(sample_path_alt):
        sample_path = sample_path_alt
if not os.path.exists(sample_path):
    sample_path_alt2 = "/kaggle/data/sample_submission.csv"
    if os.path.exists(sample_path_alt2):
        sample_path = sample_path_alt2

if not os.path.exists(sample_path):
    raise FileNotFoundError(
        "Could not locate sample_submission.csv in expected Kaggle paths."
    )

sample = pd.read_csv(sample_path)
test_images_order = sample["image"].tolist()

source = "../input/plant-pathology-2021-fgvc8/test_images"
if not os.path.isdir(source):
    alt = "/kaggle/input/plant-pathology-2021-fgvc8/test_images"
    if os.path.isdir(alt):
        source = alt
if not os.path.isdir(source):
    alt2 = "/kaggle/data/plant-pathology-2021-fgvc8/test_images"
    if os.path.isdir(alt2):
        source = alt2
if not os.path.isdir(source):
    alt3 = "/kaggle/data/test_images"
    if os.path.isdir(alt3):
        source = alt3

if not os.path.isdir(source):
    raise FileNotFoundError(
        "Could not locate test_images directory in expected Kaggle paths."
    )

IMAGE_PATHS_ALL = [os.path.join(source, fn) for fn in test_images_order]
exists_mask = [os.path.exists(p) for p in IMAGE_PATHS_ALL]
if not all(exists_mask):
    print(
        "WARNING: Some test images listed in sample_submission.csv were not found on disk. "
        "Will predict only for existing files and merge back to full sample order as 'healthy' for missing."
    )

IMAGE_PATHS = [p for p, ok in zip(IMAGE_PATHS_ALL, exists_mask) if ok]




## === cell 4
IMAGE_PATHS[:5], len(IMAGE_PATHS)




## === cell 5
AUTO = tf.data.experimental.AUTOTUNE




## === cell 6
test_dataset = (
    tf.data.Dataset.from_tensor_slices(IMAGE_PATHS)
    .map(decode_image, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)




## === cell 7
from tensorflow import keras




## === cell 8
class FixedDropout(tf.keras.layers.Dropout):
    def _get_noise_shape(self, inputs):
        if self.noise_shape is None:
            return self.noise_shape
        symbolic_shape = K.shape(inputs)
        noise_shape = [
            symbolic_shape[axis] if shape is None else shape
            for axis, shape in enumerate(self.noise_shape)
        ]
        return tuple(noise_shape)




## === cell 9
def _find_model_file(filename="5SMResNet50.h5"):
    candidates = [
        "../input/finalpp2021/5SMResNet50.h5",
        "/kaggle/input/finalpp2021/5SMResNet50.h5",
        "/kaggle/data/finalpp2021/5SMResNet50.h5",
        "/kaggle/working/finalpp2021/5SMResNet50.h5",
        "/kaggle/data/5SMResNet50.h5",
        "/kaggle/working/5SMResNet50.h5",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c

    roots = ["/kaggle/input", "../input", "/kaggle/data", "/kaggle/working"]
    seen = set()
    for root in roots:
        if os.path.isdir(root) and root not in seen:
            seen.add(root)
            for dirpath, _, files in os.walk(root):
                if filename in files:
                    return os.path.join(dirpath, filename)
    return None


model_path = _find_model_file("5SMResNet50.h5")

model = None
if model_path is not None and os.path.exists(model_path):
    model = tf.keras.models.load_model(
        model_path, compile=False, custom_objects={"FixedDropout": FixedDropout}
    )
    print("Loaded model:", model_path)
else:
    print("WARNING: Model file 5SMResNet50.h5 not found in common Kaggle paths.")
    print("Will write a valid fallback submission using 'healthy' for all test images.")




## === cell 10
if model is not None:
    probs = model.predict(test_dataset, verbose=1)
    temp_probs = probs
else:
    n = len(IMAGE_PATHS)
    temp_probs = np.zeros((n, 7), dtype=np.float32)
    temp_probs[:, 6] = 1.0




## === cell 11
temp_probs.shape




## === cell 12
def _locate_train_csv():
    candidates = [
        "../input/plant-pathology-2021-fgvc8/train.csv",
        "/kaggle/input/plant-pathology-2021-fgvc8/train.csv",
        "/kaggle/data/plant-pathology-2021-fgvc8/train.csv",
        "/kaggle/data/train.csv",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def _train_label_prevalence(train_csv_path, class_names):
    df = pd.read_csv(train_csv_path)
    labels = df["labels"].fillna("").astype(str).values
    counts = dict((c, 0) for c in class_names)
    for s in labels:
        toks = s.strip().split()
        for t in toks:
            if t in counts:
                counts[t] += 1
    prev = np.array([counts[c] for c in class_names], dtype=np.float64)
    prev = prev / max(prev.sum(), 1.0)
    return prev


def _pick_best_mapping(temp_probs_7, train_prev, mapping_candidates):
    mean_probs = temp_probs_7.mean(axis=0).astype(np.float64)
    mean_probs = mean_probs / max(mean_probs.sum(), 1e-12)

    best = None
    for name_map in mapping_candidates:
        class_order = [name_map[i] for i in range(7)]
        canonical = [
            "scab",
            "frog_eye_leaf_spot",
            "rust",
            "powdery_mildew",
            "multiple_diseases",
            "complex",
            "healthy",
        ]
        pred_vec = np.array(
            [
                mean_probs[class_order.index(c)] if c in class_order else 0.0
                for c in canonical
            ],
            dtype=np.float64,
        )
        pred_vec = pred_vec / max(pred_vec.sum(), 1e-12)
        dist = np.abs(pred_vec - train_prev).sum()
        if best is None or dist < best[0]:
            best = (dist, name_map)
    return best[1]


if temp_probs.ndim != 2 or temp_probs.shape[1] != 7:
    raise ValueError(
        f"Expected model probabilities with shape (N,7). Got: {temp_probs.shape}"
    )

mapping_candidates = []

mapping_candidates.append(
    {
        0: "scab",
        1: "frog_eye_leaf_spot",
        2: "complex",
        3: "rust",
        4: "powdery_mildew",
        5: "multiple_diseases",
        6: "healthy",
    }
)

mapping_candidates.append(
    {
        0: "healthy",
        1: "scab",
        2: "frog_eye_leaf_spot",
        3: "rust",
        4: "powdery_mildew",
        5: "complex",
        6: "multiple_diseases",
    }
)

mapping_candidates.append(
    {
        0: "complex",
        1: "frog_eye_leaf_spot",
        2: "healthy",
        3: "multiple_diseases",
        4: "powdery_mildew",
        5: "rust",
        6: "scab",
    }
)

mapping_candidates.append(
    {
        0: "scab",
        1: "frog_eye_leaf_spot",
        2: "rust",
        3: "powdery_mildew",
        4: "complex",
        5: "multiple_diseases",
        6: "healthy",
    }
)

train_csv_path = _locate_train_csv()
if train_csv_path is None:
    print(
        "WARNING: train.csv not found; using the original mapping without auto-selection."
    )
    name = mapping_candidates[0]
else:
    canonical_prev = _train_label_prevalence(
        train_csv_path,
        class_names=[
            "scab",
            "frog_eye_leaf_spot",
            "rust",
            "powdery_mildew",
            "multiple_diseases",
            "complex",
            "healthy",
        ],
    )
    name = _pick_best_mapping(temp_probs, canonical_prev, mapping_candidates)
    print("Selected output-index->class mapping:", name)




## === cell 13
def _micro_f1(y_true, y_pred_bin):
    tp = (y_true * y_pred_bin).sum()
    fp = ((1 - y_true) * y_pred_bin).sum()
    fn = (y_true * (1 - y_pred_bin)).sum()
    denom = 2 * tp + fp + fn
    return float((2 * tp) / denom) if denom > 0 else 0.0


def _mean_sample_f1(y_true, y_pred_bin):
    y_true = y_true.astype(np.int32)
    y_pred_bin = y_pred_bin.astype(np.int32)
    tp = (y_true & y_pred_bin).sum(axis=1).astype(np.float64)
    fp = ((1 - y_true) & y_pred_bin).sum(axis=1).astype(np.float64)
    fn = (y_true & (1 - y_pred_bin)).sum(axis=1).astype(np.float64)
    denom = 2 * tp + fp + fn
    f1 = np.where(denom > 0, (2 * tp) / denom, 0.0)
    return float(f1.mean()) if f1.size else 0.0


def _build_y_multihot(label_strs, class_to_idx, n_classes):
    y = np.zeros((len(label_strs), n_classes), dtype=np.int32)
    for i, s in enumerate(label_strs):
        toks = str(s).strip().split()
        for t in toks:
            if t in class_to_idx:
                y[i, class_to_idx[t]] = 1
    return y


def _predict_strings_from_probs(temp_probs_canon7, thr1, thr2):
    nm = {
        0: "scab",
        1: "frog_eye_leaf_spot",
        2: "rust",
        3: "powdery_mildew",
        4: "multiple_diseases",
        5: "complex",
        6: "healthy",
    }
    out = []
    for line in temp_probs_canon7:
        s = ""
        count = 0
        for i in range(6):
            if line[i] > thr1[i]:
                s = s + nm[i] + " "
        for i in range(6):
            if line[i] > thr2[i]:
                count += 1
        if count >= 2:
            notComplex = True
            for i in range(6):
                if line[i] > thr1[i] and nm[i] == "complex":
                    notComplex = False
                    break
            if notComplex is True:
                s = s + "complex" + " "
        if s == "":
            s = nm[6]
        else:
            s = s.strip()
        out.append(s)
    return out


def _calibrate_thresholds_on_train(
    model, name_map, train_csv_path, image_size=(512, 512), max_samples=512, seed=123
):
    if model is None or train_csv_path is None or not os.path.exists(train_csv_path):
        return None

    train_dir = "../input/plant-pathology-2021-fgvc8/train_images"
    if not os.path.isdir(train_dir):
        alt = "/kaggle/input/plant-pathology-2021-fgvc8/train_images"
        if os.path.isdir(alt):
            train_dir = alt
    if not os.path.isdir(train_dir):
        alt2 = "/kaggle/data/plant-pathology-2021-fgvc8/train_images"
        if os.path.isdir(alt2):
            train_dir = alt2
    if not os.path.isdir(train_dir):
        alt3 = "/kaggle/data/train_images"
        if os.path.isdir(alt3):
            train_dir = alt3
    if not os.path.isdir(train_dir):
        print("WARNING: Could not locate train_images; skipping threshold calibration.")
        return None

    df = pd.read_csv(train_csv_path)
    df = df[["image", "labels"]].copy()

    rng = np.random.default_rng(seed)
    if len(df) > max_samples:
        idx = rng.choice(len(df), size=max_samples, replace=False)
        df = df.iloc[idx].reset_index(drop=True)

    image_paths = [os.path.join(train_dir, fn) for fn in df["image"].tolist()]
    ok = [os.path.exists(p) for p in image_paths]
    if not all(ok):
        df = df.loc[ok].reset_index(drop=True)
        image_paths = [p for p, o in zip(image_paths, ok) if o]

    if len(image_paths) == 0:
        print("WARNING: No train images found for calibration; skipping.")
        return None

    ds = (
        tf.data.Dataset.from_tensor_slices(image_paths)
        .map(
            lambda x: decode_image(x, label=None, image_size=image_size),
            num_parallel_calls=AUTO,
        )
        .batch(BATCH_SIZE)
        .prefetch(AUTO)
    )
    p = model.predict(ds, verbose=0).astype(np.float32)

    idx_of_class = {v: k for k, v in name_map.items()}
    canonical_6 = [
        "scab",
        "frog_eye_leaf_spot",
        "rust",
        "powdery_mildew",
        "multiple_diseases",
        "complex",
    ]
    p_canon = np.zeros((p.shape[0], 7), dtype=np.float32)
    for j, c in enumerate(canonical_6):
        p_canon[:, j] = p[:, idx_of_class[c]]
    p_canon[:, 6] = p[:, idx_of_class["healthy"]]

    class_to_idx7 = {
        "scab": 0,
        "frog_eye_leaf_spot": 1,
        "rust": 2,
        "powdery_mildew": 3,
        "multiple_diseases": 4,
        "complex": 5,
        "healthy": 6,
    }
    y_true = _build_y_multihot(df["labels"].values, class_to_idx7, 7)

    grid1 = np.array([0.20, 0.25, 0.30, 0.35, 0.40, 0.45], dtype=np.float32)
    grid2 = np.array([0.10, 0.15, 0.20, 0.25, 0.30, 0.35], dtype=np.float32)

    thr1 = {i: 0.35 for i in range(6)}
    thr2 = {i: 0.25 for i in range(6)}

    def score(th1, th2):
        pred_str = _predict_strings_from_probs(p_canon, th1, th2)
        y_pred = _build_y_multihot(pred_str, class_to_idx7, 7)
        return _mean_sample_f1(y_true, y_pred)

    best_global = score(thr1, thr2)

    for _sweep in range(2):
        improved = False
        for k in range(6):
            best_local = (best_global, thr1[k], thr2[k])
            for t1 in grid1:
                for t2 in grid2:
                    if t2 > t1:
                        continue
                    th1_try = dict(thr1)
                    th2_try = dict(thr2)
                    th1_try[k] = float(t1)
                    th2_try[k] = float(t2)
                    f1 = score(th1_try, th2_try)
                    if f1 > best_local[0]:
                        best_local = (f1, float(t1), float(t2))
            if best_local[0] > best_global:
                best_global = best_local[0]
                thr1[k] = best_local[1]
                thr2[k] = best_local[2]
                improved = True
        if not improved:
            break

    print(
        f"Calibrated per-class thresholds on train subset (N={len(image_paths)}): meanSampleF1={best_global:.4f}\n"
        f"thr1={thr1}\n"
        f"thr2={thr2}"
    )
    return thr1, thr2


_calib = _calibrate_thresholds_on_train(
    model, name, train_csv_path, image_size=(512, 512), max_samples=512, seed=123
)




## === cell 14
threshold = {0: 0.35, 1: 0.35, 2: 0.35, 3: 0.35, 4: 0.35, 5: 0.35}
threshold2 = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25, 4: 0.25, 5: 0.25}

if _calib is not None:
    threshold, threshold2 = _calib


def get_key(val):
    for key, value in name.items():
        if val == value:
            return key
    return "key doesn't exist"


canonical_6 = [
    "scab",
    "frog_eye_leaf_spot",
    "rust",
    "powdery_mildew",
    "multiple_diseases",
    "complex",
]
canonical_idx = {c: i for i, c in enumerate(canonical_6)}
idx_of_class = {v: k for k, v in name.items()}  # class_name -> model_output_index

temp_probs_canon = np.zeros((temp_probs.shape[0], 7), dtype=np.float32)
for c, j in canonical_idx.items():
    temp_probs_canon[:, j] = temp_probs[:, idx_of_class[c]]
temp_probs_canon[:, 6] = temp_probs[:, idx_of_class["healthy"]]

name = {
    0: "scab",
    1: "frog_eye_leaf_spot",
    2: "rust",
    3: "powdery_mildew",
    4: "multiple_diseases",
    5: "complex",
    6: "healthy",
}
temp_probs = temp_probs_canon

pred_string = []
for line in temp_probs:
    s = ""
    count = 0

    for i in range(6):
        if line[i] > threshold[i]:
            s = s + name[i] + " "

    for i in range(6):
        if line[i] > threshold2[i]:
            count += 1

    if count >= 2:
        notComplex = True
        for i in range(6):
            if line[i] > threshold[i] and name[i] == "complex":
                notComplex = False
                break
        if notComplex == True:
            s = s + "complex" + " "

    if s == "":
        s = name[6]
    else:
        s = s.strip()
    pred_string.append(s)




## === cell 15
pred_string[:5], len(pred_string)




## === cell 16
image_names = [os.path.basename(p) for p in IMAGE_PATHS]

if len(image_names) != len(pred_string):
    m = min(len(image_names), len(pred_string))
    image_names = image_names[:m]
    pred_string = pred_string[:m]

df_pred = pd.DataFrame({"image": image_names, "labels": pred_string})

df = sample[["image"]].merge(df_pred, on="image", how="left")
df["labels"] = df["labels"].fillna("healthy")

df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", df.shape)
print(df.head())
