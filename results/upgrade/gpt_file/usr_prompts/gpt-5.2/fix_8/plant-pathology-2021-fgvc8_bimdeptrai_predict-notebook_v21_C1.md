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

0.807700831024933

# 6. Current score

0.26677

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.20126) has done: 'I fix the TensorFlow import crash by setting the protobuf implementation to the pure-Python backend before importing TF, which avoids the `MessageFactory.GetPrototype` error in this environment. Then, because your notebook expects a pre-trained `.h5` model that is not available, I keep the same overall “predict → threshold → space-delimited labels → write submission.csv” flow but replace model loading/inference with a simple, deterministic image-based heuristic so the pipeline runs end-to-end and produces a valid `submission.csv`. I also make the test image directory resolution robust to the two possible dataset mount points shown in your paths, and ensure `n_test`, `preds`, and `class_names` are always defined before use. This is score-neutral in intent (it won’t reach the target), but it unblocks submission generation reliably without changing the output format semantics.'
- What this solution (achieved 0.20384) has done: 'I fix the TensorFlow/protobuf crash by avoiding TensorFlow entirely (it isn’t needed for this heuristic pipeline) while keeping the same overall flow: read CSVs → build class list → iterate over test images → produce per-class scores → threshold into space-delimited labels → write `submission.csv`. I replace the TF `ImageDataGenerator` with a small PIL-based batch loader so execution is stable in this Kaggle environment without changing the prediction semantics beyond negligible preprocessing differences. I also make dataset path resolution include `/kaggle/data/...` based on your provided filesystem, ensuring the script finds the images/CSVs reliably. This should run end-to-end and generate a valid submission file; score should improve versus the current broken/low-signal setup simply by correctly processing all test images deterministically (but it won’t change the core heuristic logic).'
- What this solution (achieved 0.26677) has done: 'The timeout is dominated by slow single-process PIL image decoding/resizing plus a very expensive nested threshold grid search implemented with Python loops. I keep the exact heuristic scoring logic and calibration semantics, but speed up I/O by switching the batch loader to OpenCV (much faster JPEG decode/resize) with safe fallback to PIL, and use parallel prefetch via a thread pool to keep CPU busy. Then I replace the per-class/per-threshold Python loop with a vectorized computation that produces identical best thresholds by evaluating all thresholds at once using NumPy broadcasting. Finally, I vectorize the final label formatting loop to reduce Python overhead while preserving the exact decision rules.'
- What this solution (achieved 0.26677) has done: 'Your current score is far below the target, so the smallest legitimate move toward the target is to improve the probability-to-label post-processing (the part that most directly affects mean F1) without changing your heuristic “model” itself. I keep the same heuristic score computation, batching, and calibration approach, but replace the per-class independent thresholding with a tiny F1-aligned decision rule: pick the best number of labels \(k\) per image based on validation (top‑k over calibrated probabilities), which is often much better for mean F1 in this competition. This preserves evaluation semantics (still outputs space-delimited labels) and stays within your core logic constraints (no new model/training loop). The output CSV format/paths remain unchanged.'
- What this solution (achieved 0.26677) has done: 'Your current score (0.26677) is far below the target (0.80770), so we should improve the F1-aligned post-processing while keeping your exact heuristic “model” unchanged. The smallest high-impact change is to tune multi-label decoding on validation with a per-image rule: choose the number of labels \(k\) by maximizing sample-wise F1 (closer to the competition’s mean F1), then optionally apply a lightweight, validation-tuned “healthy override” to avoid predicting healthy alongside diseases when it hurts F1. This preserves your existing pipeline (same features, same heuristic scores, same calibration scale search) and only changes how probabilities become space-delimited labels. The code still runs end-to-end within time and writes a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

from sklearn.preprocessing import MultiLabelBinarizer
from PIL import Image

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

print("Setup OK (TensorFlow intentionally not imported).")




## === cell 1
def first_existing(*paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


TRAIN_CSV = first_existing(
    "../input/plant-pathology-2021-fgvc8/train.csv",
    "/kaggle/input/plant-pathology-2021-fgvc8/train.csv",
    "/kaggle/data/plant-pathology-2021-fgvc8/train.csv",
    "/kaggle/data/train.csv",
)
SAMPLE_SUB = first_existing(
    "../input/plant-pathology-2021-fgvc8/sample_submission.csv",
    "/kaggle/input/plant-pathology-2021-fgvc8/sample_submission.csv",
    "/kaggle/data/plant-pathology-2021-fgvc8/sample_submission.csv",
    "/kaggle/data/sample_submission.csv",
)
TEST_DIR = first_existing(
    "../input/plant-pathology-2021-fgvc8/test_images",
    "/kaggle/input/plant-pathology-2021-fgvc8/test_images",
    "/kaggle/data/plant-pathology-2021-fgvc8/test_images",
    "/kaggle/data/test_images",
)
TRAIN_DIR = first_existing(
    "../input/plant-pathology-2021-fgvc8/train_images",
    "/kaggle/input/plant-pathology-2021-fgvc8/train_images",
    "/kaggle/data/plant-pathology-2021-fgvc8/train_images",
    "/kaggle/data/train_images",
)

if TRAIN_CSV is None or SAMPLE_SUB is None or TEST_DIR is None or TRAIN_DIR is None:
    raise FileNotFoundError(
        f"Could not resolve dataset paths. TRAIN_CSV={TRAIN_CSV}, SAMPLE_SUB={SAMPLE_SUB}, "
        f"TEST_DIR={TEST_DIR}, TRAIN_DIR={TRAIN_DIR}"
    )

train = pd.read_csv(TRAIN_CSV)
submissions = pd.read_csv(SAMPLE_SUB)

print("Resolved paths:")
print("TRAIN_CSV:", TRAIN_CSV)
print("SAMPLE_SUB:", SAMPLE_SUB)
print("TRAIN_DIR:", TRAIN_DIR)
print("TEST_DIR:", TEST_DIR)
print(train.shape, submissions.shape)
train.head()



## === cell 2
label_split = train.labels.apply(lambda x: x.split())
mlb = MultiLabelBinarizer()
mlb.fit(label_split)
class_names = list(mlb.classes_)

labels_df = pd.DataFrame(mlb.transform(label_split), columns=class_names)
print("Num classes:", len(class_names))
print("Classes:", class_names)



## === cell 3
h_target = 384
w_target = 384
batch_size = 32

try:
    import cv2  # noqa: F401

    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False

from concurrent.futures import ThreadPoolExecutor

_IO_WORKERS = max(2, min(8, (os.cpu_count() or 2)))


def _read_one_image_rgb_float(path, H, W):
    if _HAS_CV2:
        import cv2

        try:
            im = cv2.imread(path, cv2.IMREAD_COLOR)  # BGR uint8
            if im is None:
                raise ValueError("cv2.imread returned None")
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = cv2.resize(im, (W, H), interpolation=cv2.INTER_LINEAR)
            arr = im.astype(np.float32) / 255.0
            return arr
        except Exception:
            pass  # fallback to PIL below

    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            im = im.resize((W, H), resample=Image.BILINEAR)
            arr = np.asarray(im, dtype=np.float32) / 255.0
            return arr
    except Exception:
        return np.zeros((H, W, 3), dtype=np.float32)


def iter_image_batches(df, directory, target_size=(384, 384), batch_size=32):
    """
    Batch loader; yields (batch_images, batch_indices).
    Images are float32 in [0,1], shape (bs, H, W, 3).
    """
    H, W = target_size
    n = len(df)
    images = df["image"].to_numpy()
    with ThreadPoolExecutor(max_workers=_IO_WORKERS) as ex:
        for start in range(0, n, batch_size):
            end = min(n, start + batch_size)
            idxs = np.arange(start, end, dtype=np.int64)

            fnames = images[start:end]
            paths = [os.path.join(directory, f) for f in fnames]

            batch_list = list(
                ex.map(lambda p: _read_one_image_rgb_float(p, H, W), paths)
            )
            batch = np.stack(batch_list, axis=0).astype(np.float32, copy=False)
            yield batch, idxs


print("Fast generator ready. cv2:", _HAS_CV2, "workers:", _IO_WORKERS)




## === cell 4
def squash(x):
    return 1.0 / (1.0 + np.exp(-x))


_class_kind = []
for cname in class_names:
    lc = cname.lower()
    if lc == "healthy":
        _class_kind.append("healthy")
    elif "rust" in lc:
        _class_kind.append("rust")
    elif "scab" in lc:
        _class_kind.append("scab")
    elif "frog" in lc:
        _class_kind.append("frog")
    elif "complex" in lc:
        _class_kind.append("complex")
    elif "powdery" in lc:
        _class_kind.append("powdery")
    else:
        _class_kind.append("other")


def heuristic_scores_from_batch(batch, class_names):
    """
    Core logic preserved: same simple image statistics -> class score -> sigmoid mapping.
    Returns probs of shape (bs, n_classes).
    """
    bs = batch.shape[0]
    n_classes = len(class_names)
    out = np.zeros((bs, n_classes), dtype=np.float32)

    mean = batch.mean(axis=(1, 2, 3))  # (bs,)
    std = batch.std(axis=(1, 2, 3))  # (bs,)
    g = batch[:, :, :, 1].mean(axis=(1, 2))
    rb = 0.5 * (
        batch[:, :, :, 0].mean(axis=(1, 2)) + batch[:, :, :, 2].mean(axis=(1, 2))
    )
    green = g - rb

    for j, kind in enumerate(_class_kind):
        score = -1.0 + 0.0 * mean

        if kind == "healthy":
            score = 3.0 * green - 2.0 * std + 0.5 * (0.5 - np.abs(mean - 0.5))
        elif kind == "rust":
            score = 2.0 * (mean - 0.45) + 1.0 * std - 1.0 * green
        elif kind == "scab":
            score = 1.5 * std + 0.5 * (mean - 0.5) - 0.5 * green
        elif kind == "frog":
            score = 1.0 * std - 0.2 * mean
        elif kind == "complex":
            score = 1.2 * std + 0.8 * np.abs(mean - 0.5)
        elif kind == "powdery":
            score = 1.0 * (mean - 0.5) + 0.8 * std
        else:
            score = 0.5 * std - 0.5 * np.abs(mean - 0.5)

        out[:, j] = squash(score).astype(np.float32)

    return out


def macro_f1(y_true, y_pred_bin, eps=1e-9):
    yt = y_true.astype(np.int32, copy=False)
    yp = y_pred_bin.astype(np.int32, copy=False)
    tp = np.sum((yt == 1) & (yp == 1), axis=0).astype(np.float64)
    fp = np.sum((yt == 0) & (yp == 1), axis=0).astype(np.float64)
    fn = np.sum((yt == 1) & (yp == 0), axis=0).astype(np.float64)
    f1 = (2.0 * tp) / (2.0 * tp + fp + fn + eps)
    return float(np.mean(f1))


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def mean_sample_f1(y_true, y_pred_bin, eps=1e-9):
    yt = y_true.astype(np.int32, copy=False)
    yp = y_pred_bin.astype(np.int32, copy=False)
    tp = np.sum((yt == 1) & (yp == 1), axis=1).astype(np.float64)
    fp = np.sum((yt == 0) & (yp == 1), axis=1).astype(np.float64)
    fn = np.sum((yt == 1) & (yp == 0), axis=1).astype(np.float64)
    f1 = (2.0 * tp) / (2.0 * tp + fp + fn + eps)
    return float(np.mean(f1))


n_train = len(train)
idxs = np.arange(n_train)
rng = np.random.RandomState(SEED)
rng.shuffle(idxs)
split = int(0.85 * n_train)
tr_idx, va_idx = idxs[:split], idxs[split:]

train_va = train.iloc[va_idx].reset_index(drop=True)
y_va = labels_df.iloc[va_idx].to_numpy(dtype=np.int8)

va_probs_raw = np.zeros((len(train_va), len(class_names)), dtype=np.float32)
for batch, batch_idxs in iter_image_batches(
    train_va, TRAIN_DIR, target_size=(h_target, w_target), batch_size=batch_size
):
    probs = heuristic_scores_from_batch(batch, class_names)
    va_probs_raw[batch_idxs] = probs

print("Computed validation probs:", va_probs_raw.shape)

logits = logit(va_probs_raw)
scales = np.array([0.7, 0.85, 1.0, 1.15, 1.3], dtype=np.float32)

th_grid = np.linspace(0.10, 0.90, 81, dtype=np.float32)
Y = y_va.astype(np.int8, copy=False)
pos_cnt = Y.sum(axis=0)

healthy_idx = class_names.index("healthy") if "healthy" in class_names else None

best = {
    "score": -1.0,
    "scale": 1.0,
    "th": None,
    "k": 1,
    "mode": "thresholds",
    "healthy_drop": False,
}

for s in scales:
    p_s = squash(logits * s).astype(np.float32, copy=False)

    preds_bin_all = p_s[:, :, None] >= th_grid[None, None, :]
    yt1 = Y[:, :, None] == 1
    yt0 = ~yt1
    yp1 = preds_bin_all
    yp0 = ~yp1
    tp = np.sum(yt1 & yp1, axis=0, dtype=np.int32)
    fp = np.sum(yt0 & yp1, axis=0, dtype=np.int32)
    fn = np.sum(yt1 & yp0, axis=0, dtype=np.int32)
    f1 = (2.0 * tp) / (2.0 * tp + fp + fn + 1e-9)  # (K, T)

    ths = np.full((len(class_names),), 0.3, dtype=np.float32)
    valid = pos_cnt >= 5
    if np.any(valid):
        best_idx = np.argmax(f1[valid], axis=1)
        ths[valid] = th_grid[best_idx].astype(np.float32)

    y_pred_bin_th = (p_s >= ths[None, :]).astype(np.int8)
    score_th = mean_sample_f1(Y, y_pred_bin_th)
    if score_th > best["score"]:
        best = {
            "score": float(score_th),
            "scale": float(s),
            "th": ths,
            "k": 1,
            "mode": "thresholds",
            "healthy_drop": False,
        }

    max_k = min(8, p_s.shape[1])
    order = np.argsort(-p_s, axis=1)  # descending
    rows = np.arange(Y.shape[0])[:, None]

    for k in range(1, max_k + 1):
        y_pred_bin_k = np.zeros_like(Y, dtype=np.int8)
        topk = order[:, :k]
        y_pred_bin_k[rows, topk] = 1

        score_k = mean_sample_f1(Y, y_pred_bin_k)
        if score_k > best["score"]:
            best = {
                "score": float(score_k),
                "scale": float(s),
                "th": ths,
                "k": int(k),
                "mode": "topk",
                "healthy_drop": False,
            }

        if healthy_idx is not None:
            y_pred_bin_k2 = y_pred_bin_k.copy()
            has_disease = (
                np.sum(y_pred_bin_k2, axis=1) - y_pred_bin_k2[:, healthy_idx]
            ) > 0
            y_pred_bin_k2[has_disease, healthy_idx] = 0
            score_k2 = mean_sample_f1(Y, y_pred_bin_k2)
            if score_k2 > best["score"]:
                best = {
                    "score": float(score_k2),
                    "scale": float(s),
                    "th": ths,
                    "k": int(k),
                    "mode": "topk",
                    "healthy_drop": True,
                }

print("Validation mean sample-wise F1 (calibration + decoding):", best["score"])
print("Best scale:", best["scale"])
print(
    "Best mode:", best["mode"], "k:", best["k"], "healthy_drop:", best["healthy_drop"]
)
print("Example thresholds (first 10):", best["th"][:10].round(3))



## === cell 5
n_test = len(submissions)
n_classes = len(class_names)

preds = np.zeros((n_test, n_classes), dtype=np.float32)

for batch, batch_idxs in iter_image_batches(
    submissions, TEST_DIR, target_size=(h_target, w_target), batch_size=batch_size
):
    probs = heuristic_scores_from_batch(batch, class_names)
    probs = squash(logit(probs) * best["scale"]).astype(np.float32)
    preds[batch_idxs] = probs

print("Preds shape:", preds.shape)



## === cell 6
ths = best["th"].astype(np.float32)
healthy_idx = class_names.index("healthy") if "healthy" in class_names else None

if best["mode"] == "topk":
    order_test = np.argsort(-preds, axis=1)
    k = int(best["k"])
    chosen_mask = np.zeros_like(preds, dtype=bool)
    rows = np.arange(n_test)[:, None]
    chosen_mask[rows, order_test[:, :k]] = True
else:
    chosen_mask = preds >= ths[None, :]

if best.get("healthy_drop", False) and healthy_idx is not None:
    has_disease = (
        np.sum(chosen_mask, axis=1) - chosen_mask[:, healthy_idx].astype(np.int32)
    ) > 0
    chosen_mask[has_disease, healthy_idx] = False

out_labels = []
for i in range(n_test):
    p = preds[i]
    chosen_idx = np.flatnonzero(chosen_mask[i])
    chosen = [class_names[j] for j in chosen_idx]

    if len(chosen) == 0:
        lab = class_names[int(np.argmax(p))]
    else:
        if healthy_idx is not None and "healthy" in chosen:
            if p[healthy_idx] >= (np.max(p) - 1e-6) and len(chosen) == 1:
                lab = "healthy"
            elif p[healthy_idx] >= (np.max(p) - 1e-6) and len(chosen) > 1:
                lab = "healthy"
            else:
                chosen = [c for c in chosen if c != "healthy"]
                if len(chosen) == 0:
                    lab = "healthy"
                else:
                    lab = " ".join(chosen)
        else:
            lab = " ".join(chosen)

    out_labels.append(lab)

submissions_out = submissions.copy()
submissions_out["labels"] = out_labels
submissions_out = submissions_out[["image", "labels"]]
submissions_out.to_csv("submission.csv", index=False)

print(submissions_out.head())
print("Wrote submission.csv with shape:", submissions_out.shape)



## === cell 7
assert os.path.exists("submission.csv")
chk = pd.read_csv("submission.csv")
assert list(chk.columns) == ["image", "labels"]
assert len(chk) == len(submissions_out)
chk.head()
