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

0.6406278855032318

# 6. Current score

0.24507

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.24507) has done: 'I fix the root cause preventing the model from loading (the protobuf/TensorFlow incompatibility that triggers the `MessageFactory.GetPrototype` error) by switching to `tf.keras.models.load_model(..., compile=False)` and, if needed, rebuilding the model from its config and loading weights. Then I ensure the pipeline always defines `model`, runs prediction, and produces `submission.csv` with the exact required columns and correct image-to-prediction alignment. I also make the test image copy step robust (only copy when missing) and add a small fallback so the generator and submission stay consistent even if directory structure differs. These changes are execution/format fixes and should be score-neutral except for restoring the intended predictions.'
- What this solution (achieved 0.24507) has done: 'I fix the root cause of the crash by avoiding TensorFlow’s protobuf-based model deserialization for this `.h5` file: instead, I load the saved predictions directly from the H5 (these are typically stored under the `model_weights` group for this project) or, if unavailable, fall back to a safe “all healthy” submission so a valid CSV is always produced. This unblocks the pipeline so `model`/`x` are always defined, preventing the downstream `NameError`s. I also keep the generator/submission alignment robust by mapping predictions to the `sample_submission.csv` image names exactly as before. Finally, I keep the original thresholding/label formatting semantics unchanged when real predictions are available, which should substantially improve score versus the current “no model” failure path.'
- What this solution (achieved 0.24507) has done: 'I remove the failing TF/protobuf model-loading path entirely (it triggers the `MessageFactory.GetPrototype` crash) and instead load the competition’s precomputed test predictions from the provided `model-best.h5` by scanning for a stored `(N,5)` dataset. Then I add robust alignment logic so the predictions are matched to `sample_submission.csv` by filename when possible (and safely fall back only for unmatched rows), which should substantially improve score versus the current mostly-“healthy” output. Finally, I keep your original thresholding and label formatting semantics unchanged and ensure a valid `submission.csv` is always written.'
- What this solution (achieved 0.24507) has done: 'The crash comes from importing TensorFlow/Keras in this environment (protobuf incompatibility), but your current approach doesn’t actually need TensorFlow because you’re trying to use precomputed predictions from the provided `.h5`. I remove the TensorFlow generator path entirely and instead align those `(N,5)` predictions to `sample_submission.csv` by inspecting the `.h5` for a matching list of filenames (if present), otherwise fall back to safe ordering when shapes match. This should eliminate the runtime error and also improve score versus the current mostly-“healthy” fallback by correctly mapping predictions to the right images. I keep your thresholding and label formatting semantics unchanged and always write a valid `submission.csv`.'
- What this solution (achieved 0.24507) has done: 'Your current score (0.24507) is far below the target (0.6406), so we should improve performance without changing the model/training logic. The biggest likely issue is that the extracted `(N,5)` arrays from the `.h5` are not guaranteed to be probabilities in `[0,1]`, so thresholding at `0.4` can be meaningless and severely hurt F1; we add a minimal, metric-aligned calibration step that auto-detects logits and applies a sigmoid only when needed. We also improve the `.h5` dataset selection to prefer float prediction tensors (and avoid accidentally picking unrelated `(N,5)` arrays), while keeping your existing label set and thresholding semantics intact for already-probabilistic outputs. These changes are narrowly targeted to make the binarization step meaningful and should move the score upward toward the target.'
- What this solution (achieved 0.24507) has done: 'Your current score is far below the target, so the most likely issue is that the extracted `(N,5)` array from the `.h5` is not the right prediction tensor and/or the class order does not match the competition label order—both can crush mean F1 without changing any “core logic.” I keep your pipeline (load `(N,5)` → optional sigmoid calibration → threshold 0.4 → space-delimited labels) but make two minimal, metric-relevant upgrades: (1) select the best `(N,5)` dataset by directly scoring candidates on a small validation split from `train.csv` (mean F1), and (2) automatically find the best permutation of the 5 columns (class order) using the same validation split. This preserves the model and post-processing semantics while making sure you’re thresholding the right numbers for the right classes, which should move the score substantially upward toward the target.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from pathlib import Path

BASE_INPUT = Path("../input/plant-pathology-2021-fgvc8")
MODEL_PATH = Path("../input/dlcv-projekt/model-best.h5")

print("Base input exists:", BASE_INPUT.exists())
print("Model path exists:", MODEL_PATH.exists())

sub_path = BASE_INPUT / "sample_submission.csv"
sub = pd.read_csv(sub_path)
print("sample_submission rows:", len(sub), "cols:", list(sub.columns))




## === cell 1
def safe_load_predictions_and_names_from_h5(model_path: Path):
    import h5py

    with h5py.File(model_path, "r") as f:
        pred_candidates = []
        name_candidates = []

        def visit(name, obj):
            if not isinstance(obj, h5py.Dataset):
                return

            try:
                shape = obj.shape
                if (
                    shape is not None
                    and len(shape) == 2
                    and shape[1] == 5
                    and shape[0] >= 1000
                ):
                    dt = obj.dtype
                    dt_str = str(dt)
                    is_float = dt.kind == "f"
                    lname = name.lower()
                    name_score = 0
                    if any(
                        k in lname
                        for k in [
                            "pred",
                            "predict",
                            "proba",
                            "prob",
                            "output",
                            "logit",
                            "yhat",
                        ]
                    ):
                        name_score += 2
                    if any(
                        k in lname
                        for k in ["val", "train", "label", "target", "y_true", "ytrue"]
                    ):
                        name_score -= 3
                    if is_float:
                        name_score += 1
                    pred_candidates.append(
                        (name, tuple(shape), dt_str, name_score, is_float)
                    )
            except Exception:
                pass

            try:
                shape = obj.shape
                if shape is not None and len(shape) == 1 and shape[0] >= 1000:
                    dt = str(obj.dtype)
                    if ("S" in dt) or ("U" in dt) or ("object" in dt):
                        name_candidates.append((name, int(shape[0]), dt))
            except Exception:
                pass

        f.visititems(visit)

        if not pred_candidates:
            raise RuntimeError(
                "No (N,5) dataset found inside the provided .h5. Cannot extract predictions."
            )

        pred_candidates_sorted = sorted(
            pred_candidates,
            key=lambda t: (t[3], 1 if t[4] else 0, t[1][0], t[0]),
            reverse=True,
        )

        print("Found (N,5) candidates (top 10 by heuristic):")
        for c in pred_candidates_sorted[:10]:
            print(" -", c)

        return pred_candidates_sorted, name_candidates


pred_candidates_sorted = None
name_candidates = None
try:
    if MODEL_PATH.exists():
        pred_candidates_sorted, name_candidates = (
            safe_load_predictions_and_names_from_h5(MODEL_PATH)
        )
except Exception as e:
    print("Precomputed scan failed:", repr(e))
    pred_candidates_sorted, name_candidates = None, None




## === cell 2
def _sigmoid(a: np.ndarray) -> np.ndarray:
    a = a.astype(np.float32, copy=False)
    a = np.clip(a, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-a))


def calibrate_if_needed(x_arr: np.ndarray) -> np.ndarray:
    x_arr = x_arr.astype(np.float32, copy=False)

    finite = x_arr[np.isfinite(x_arr)]
    if finite.size == 0:
        return x_arr

    mn = float(np.min(finite))
    mx = float(np.max(finite))

    looks_prob = (mn >= -1e-3) and (mx <= 1.0 + 1e-3)
    if looks_prob:
        print(
            f"Calibration: looks like probabilities in [{mn:.3f}, {mx:.3f}] -> no sigmoid applied."
        )
        return np.clip(x_arr, 0.0, 1.0)

    print(
        f"Calibration: values in [{mn:.3f}, {mx:.3f}] -> applying sigmoid (treat as logits)."
    )
    return _sigmoid(x_arr)


def multilabel_mean_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.astype(np.int32, copy=False)
    y_pred = y_pred.astype(np.int32, copy=False)

    tp = np.sum((y_true == 1) & (y_pred == 1), axis=1).astype(np.float32)
    fp = np.sum((y_true == 0) & (y_pred == 1), axis=1).astype(np.float32)
    fn = np.sum((y_true == 1) & (y_pred == 0), axis=1).astype(np.float32)

    denom = 2.0 * tp + fp + fn
    f1 = np.where(denom > 0, (2.0 * tp) / denom, 1.0)  # if both empty sets -> F1=1
    return float(np.mean(f1))


def parse_labels_to_y(df: pd.DataFrame, classes: list) -> np.ndarray:
    cls_to_i = {c: i for i, c in enumerate(classes)}
    y = np.zeros((len(df), len(classes)), dtype=np.int32)
    for r, s in enumerate(df["labels"].astype(str).tolist()):
        if s.strip() == "" or s.strip().lower() == "healthy":
            continue
        parts = s.split()
        for p in parts:
            if p in cls_to_i:
                y[r, cls_to_i[p]] = 1
    return y


labels = ["complex", "frog_eye_leaf_spot", "powdery_mildew", "rust", "scab"]
threshold = 0.4  # keep original semantics

train_csv_path = BASE_INPUT / "train.csv"
train_df = pd.read_csv(train_csv_path)
train_df["image"] = train_df["image"].astype(str).apply(lambda x: Path(x).name)

rng = np.random.RandomState(123)
val_n = min(2000, len(train_df))  # keep runtime low while being informative
val_idx = rng.choice(len(train_df), size=val_n, replace=False)
val_df = train_df.iloc[val_idx].reset_index(drop=True)
y_val = parse_labels_to_y(val_df, labels)
val_images = val_df["image"].tolist()

print("Validation subset:", len(val_df), "rows")



## === cell 3
import itertools


def try_load_names_for_length(name_candidates, needed_len, h5file):
    if not name_candidates:
        return None
    matches = [t for t in name_candidates if t[1] == needed_len]
    if not matches:
        return None
    matches_sorted = sorted(
        matches,
        key=lambda t: (
            (
                0
                if any(
                    k in t[0].lower()
                    for k in ["file", "filename", "image", "path", "name", "id"]
                )
                else 1
            ),
            t[0],
        ),
    )
    preferred = matches_sorted[0][0]
    raw = np.array(h5file[preferred])
    if raw.dtype.kind == "S":
        names = [r.decode("utf-8", errors="ignore") for r in raw.tolist()]
    else:
        names = [str(r) for r in raw.tolist()]
    names = [Path(n).name for n in names]
    return names


def score_candidate_on_val(
    x_full: np.ndarray, names_full, val_images, y_val, threshold: float
):
    if x_full is None or x_full.ndim != 2 or x_full.shape[1] != 5:
        return None

    if names_full is not None and len(names_full) == x_full.shape[0]:
        idx_map = {fn: i for i, fn in enumerate(names_full)}
        rows = []
        missing = 0
        for img in val_images:
            i = idx_map.get(Path(img).name)
            if i is None:
                missing += 1
                rows.append(None)
            else:
                rows.append(i)
        if missing > int(0.1 * len(val_images)):
            return None
        x_val = np.zeros((len(val_images), 5), dtype=np.float32)
        for j, i in enumerate(rows):
            if i is not None:
                x_val[j] = x_full[i]
    else:
        return None

    x_val = calibrate_if_needed(x_val)
    best = None
    for perm in itertools.permutations(range(5)):
        xp = x_val[:, perm]
        z = (xp > threshold).astype(np.int32)
        f1 = multilabel_mean_f1(y_val, z)
        if (best is None) or (f1 > best[0]):
            best = (f1, perm)
    return best  # (f1, perm)


best_overall = None  # (f1, pred_dataset_name, perm, names_dataset_used_flag)
best_loaded = None  # (x_full, names_full, perm)

x = None

if pred_candidates_sorted is not None:
    import h5py

    with h5py.File(MODEL_PATH, "r") as f:
        top_k = min(8, len(pred_candidates_sorted))
        for cand in pred_candidates_sorted[:top_k]:
            pred_name = cand[0]
            try:
                x_full = np.array(f[pred_name]).astype(np.float32, copy=False)
                names_full = try_load_names_for_length(
                    name_candidates, x_full.shape[0], f
                )
                if names_full is None:
                    continue
                scored = score_candidate_on_val(
                    x_full, names_full, val_images, y_val, threshold
                )
                if scored is None:
                    continue
                f1, perm = scored
                print(
                    f"Candidate '{pred_name}': val mean F1={f1:.5f}, best_perm={perm}"
                )
                if (best_overall is None) or (f1 > best_overall[0]):
                    best_overall = (f1, pred_name, perm)
                    best_loaded = (x_full, names_full, perm)
            except Exception as e:
                print("Candidate eval failed for", pred_name, "->", repr(e))
                continue

if best_loaded is not None:
    x_full, names_full, perm = best_loaded
    idx_map = {fn: i for i, fn in enumerate(names_full)}
    missing = 0
    x = np.zeros((len(sub), 5), dtype=np.float32)
    for j, img in enumerate(sub["image"].astype(str).tolist()):
        i = idx_map.get(Path(img).name)
        if i is None:
            missing += 1
            continue
        x[j] = x_full[i]
    x = x[:, perm]  # apply discovered class order
    print(
        "Selected best H5 dataset:",
        best_overall[1],
        "val F1:",
        best_overall[0],
        "perm:",
        perm,
    )
    print(
        "Aligned by filenames for submission. Missing images:",
        missing,
        "out of",
        len(sub),
    )
else:
    print(
        "WARNING: Could not validate-select a candidate+permutation; falling back to heuristic alignment."
    )
    x_precomputed = None
    names_precomputed = None
    try:
        if pred_candidates_sorted is not None:
            import h5py

            with h5py.File(MODEL_PATH, "r") as f:
                preferred_pred = pred_candidates_sorted[0][0]
                x_precomputed = np.array(f[preferred_pred]).astype(
                    np.float32, copy=False
                )
                names_precomputed = try_load_names_for_length(
                    name_candidates, x_precomputed.shape[0], f
                )
                print(
                    "Loaded heuristic predictions from:",
                    preferred_pred,
                    "shape:",
                    x_precomputed.shape,
                )
    except Exception as e:
        print("Heuristic load failed:", repr(e))
        x_precomputed, names_precomputed = None, None

    if (
        x_precomputed is not None
        and x_precomputed.ndim == 2
        and x_precomputed.shape[1] == 5
    ):
        if (
            names_precomputed is not None
            and len(names_precomputed) == x_precomputed.shape[0]
        ):
            idx_map = {fn: i for i, fn in enumerate(names_precomputed)}
            missing = 0
            x = np.zeros((len(sub), 5), dtype=np.float32)
            for j, img in enumerate(sub["image"].astype(str).tolist()):
                i = idx_map.get(Path(img).name)
                if i is None:
                    missing += 1
                    continue
                x[j] = x_precomputed[i]
            print(
                "Aligned by filenames (heuristic). Missing images:",
                missing,
                "out of",
                len(sub),
            )
        else:
            if x_precomputed.shape[0] == len(sub):
                x = x_precomputed
                print("Aligned by row order (heuristic, shapes match).")
            else:
                print("WARNING: cannot align predictions; falling back to all-healthy.")
                x = np.zeros((len(sub), 5), dtype=np.float32)
    else:
        print("No usable precomputed predictions found; falling back to all-healthy.")
        x = np.zeros((len(sub), 5), dtype=np.float32)

x = calibrate_if_needed(x)
print("Pred shape:", x.shape)
print("Pred min/max:", float(np.min(x)), float(np.max(x)))



## === cell 4
z = (x > threshold).astype(np.int32)
print("Binarized shape:", z.shape)



## === cell 5
predictions = [[labels[i] for i, v in enumerate(row) if v != 0] for row in z]
predictions_str = [" ".join(p) if len(p) else "healthy" for p in predictions]

sub["labels"] = predictions_str

print(sub.head())
print("Submission rows:", len(sub), "Pred rows:", len(predictions_str))

sub = sub[["image", "labels"]]
sub["image"] = sub["image"].astype(str)
sub["labels"] = sub["labels"].astype(str)



## === cell 6
out_file = "submission.csv"
sub.to_csv(out_file, index=False)
print("Wrote:", out_file, "Size(bytes):", os.path.getsize(out_file))
print("Saved columns:", list(sub.columns))
print("Unique label strings (sample):", sub["labels"].value_counts().head())
