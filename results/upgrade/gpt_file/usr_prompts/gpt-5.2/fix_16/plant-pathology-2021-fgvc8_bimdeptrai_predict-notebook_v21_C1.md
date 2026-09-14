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

0.38175

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.20126) has done: 'I fix the TensorFlow import crash by setting the protobuf implementation to the pure-Python backend before importing TF, which avoids the `MessageFactory.GetPrototype` error in this environment. Then, because your notebook expects a pre-trained `.h5` model that is not available, I keep the same overall “predict → threshold → space-delimited labels → write submission.csv” flow but replace model loading/inference with a simple, deterministic image-based heuristic so the pipeline runs end-to-end and produces a valid `submission.csv`. I also make the test image directory resolution robust to the two possible dataset mount points shown in your paths, and ensure `n_test`, `preds`, and `class_names` are always defined before use. This is score-neutral in intent (it won’t reach the target), but it unblocks submission generation reliably without changing the output format semantics.'
- What this solution (achieved 0.20384) has done: 'I fix the TensorFlow/protobuf crash by avoiding TensorFlow entirely (it isn’t needed for this heuristic pipeline) while keeping the same overall flow: read CSVs → build class list → iterate over test images → produce per-class scores → threshold into space-delimited labels → write `submission.csv`. I replace the TF `ImageDataGenerator` with a small PIL-based batch loader so execution is stable in this Kaggle environment without changing the prediction semantics beyond negligible preprocessing differences. I also make dataset path resolution include `/kaggle/data/...` based on your provided filesystem, ensuring the script finds the images/CSVs reliably. This should run end-to-end and generate a valid submission file; score should improve versus the current broken/low-signal setup simply by correctly processing all test images deterministically (but it won’t change the core heuristic logic).'
- What this solution (achieved 0.26677) has done: 'The timeout is dominated by slow single-process PIL image decoding/resizing plus a very expensive nested threshold grid search implemented with Python loops. I keep the exact heuristic scoring logic and calibration semantics, but speed up I/O by switching the batch loader to OpenCV (much faster JPEG decode/resize) with safe fallback to PIL, and use parallel prefetch via a thread pool to keep CPU busy. Then I replace the per-class/per-threshold Python loop with a vectorized computation that produces identical best thresholds by evaluating all thresholds at once using NumPy broadcasting. Finally, I vectorize the final label formatting loop to reduce Python overhead while preserving the exact decision rules.'
- What this solution (achieved 0.26677) has done: 'Your current score is far below the target, so the smallest legitimate move toward the target is to improve the probability-to-label post-processing (the part that most directly affects mean F1) without changing your heuristic “model” itself. I keep the same heuristic score computation, batching, and calibration approach, but replace the per-class independent thresholding with a tiny F1-aligned decision rule: pick the best number of labels \(k\) per image based on validation (top‑k over calibrated probabilities), which is often much better for mean F1 in this competition. This preserves evaluation semantics (still outputs space-delimited labels) and stays within your core logic constraints (no new model/training loop). The output CSV format/paths remain unchanged.'
- What this solution (achieved 0.26677) has done: 'Your current score (0.26677) is far below the target (0.80770), so we should improve the F1-aligned post-processing while keeping your exact heuristic “model” unchanged. The smallest high-impact change is to tune multi-label decoding on validation with a per-image rule: choose the number of labels \(k\) by maximizing sample-wise F1 (closer to the competition’s mean F1), then optionally apply a lightweight, validation-tuned “healthy override” to avoid predicting healthy alongside diseases when it hurts F1. This preserves your existing pipeline (same features, same heuristic scores, same calibration scale search) and only changes how probabilities become space-delimited labels. The code still runs end-to-end within time and writes a valid `submission.csv`.'
- What this solution (achieved 0.30521) has done: 'Your score is far below the target, so the smallest high-impact change is to align the validation objective and decoding with the competition’s mean F1 (sample-wise) more directly, without changing your heuristic “model” at all. I keep the exact feature→score mapping and the same calibration-by-scale idea, but replace the per-class threshold selection with a validation-tuned global threshold (and optionally top‑k) that optimizes mean sample-wise F1, which is typically much better for this metric than per-class thresholds under weak probabilities. I also tune a simple “empty prediction fallback” using the validation set (either argmax or “healthy”) so you don’t lose F1 on images where your decoding produces no labels. The pipeline remains deterministic, runs end-to-end, and writes a valid `submission.csv`.'
- What this solution (achieved 0.38147) has done: 'Your current score (0.30521) is far below the target (0.80770), so we should improve mean sample-wise F1 using the smallest change that directly affects the metric: decoding probabilities into space-delimited labels. I keep your exact heuristic feature→score mapping and calibration-by-scale, but I tune a per-class threshold vector on the validation set (coordinate ascent) because a single global threshold/top‑k is too coarse for multi-label F1 here. I also tune (on validation) the “empty prediction” fallback choice and keep your existing optional `healthy_drop`, then apply the learned per-class thresholds to test predictions. This preserves your overall pipeline (no new model/training loop) while making post-processing much closer to what the competition metric rewards.'
- What this solution (achieved 0.38185) has done: 'Your current score (0.38147) is far below the target (0.80770), so the most direct minimal improvement is to make the validation tuning better match the hidden test distribution without changing your heuristic “model.” I keep the exact heuristic score computation and calibration-by-scale, but replace the single 85/15 split with a small deterministic 3-fold out-of-fold (OOF) procedure so thresholds/fallback are tuned on predictions that are less overfit, which typically lifts mean sample-wise F1 for multi-label decoding. I also make the final label string deterministic by sorting chosen labels by descending probability (same chosen set, just stable ordering), which can reduce submission jitter risk. Everything still runs end-to-end, stays within the same pipeline, and writes `submission.csv` with the required columns.'
- What this solution (achieved 0.38185) has done: 'Your current gap to the target is large, so the most direct minimal improvement is to fix a key validation mistake: you’re tuning thresholds/decoding on in-sample OOF probabilities but comparing against the full `Y_all` in the wrong order, which misaligns predictions and labels and leads to badly tuned decoding. I compute OOF predictions aligned to the original row indices, then tune decoding on those aligned OOF predictions (still no model/training changes). I also prevent leakage by tuning only on OOF (not the same fold’s fit), and keep the exact heuristic scoring, calibration-by-scale, and decoding options unchanged. Finally, I keep submission formatting identical and deterministic.'
- What this solution (achieved 0.38185) has done: 'Your score is far below the target, so the smallest meaningful move toward it (without changing the heuristic “model”) is to tune decoding in a way that better matches the competition’s mean sample-wise F1. I keep your exact feature→score mapping, calibration-by-scale, and batching, but I add a tiny validation-tuned “complex gating” rule (only allow `complex` when at least 2 other diseases are predicted) which often reduces false positives that hurt mean F1. I also add a validation-tuned `max_labels` cap (applied after thresholding/top‑k) to prevent over-predicting many labels per image, another common mean-F1 failure mode. Both are tuned purely on OOF predictions (no leakage) and then applied to test predictions, preserving your pipeline and output semantics while typically improving F1.'
- What this solution (achieved 0.38185) has done: 'Your current score is far below the target, so we should make the smallest changes that directly improve mean sample-wise F1 without changing the heuristic “model” itself. The main fix is to tune decoding (thresholds/top‑k/healthy_drop/fallback/complex gating/max_labels) on true out-of-fold (OOF) predictions and labels, but only for the corresponding rows that were actually predicted in each fold, preventing subtle misalignment/over-optimism. Then we apply the best decoding config unchanged to test predictions. This keeps your architecture/feature extraction and inference semantics intact, but makes the post-processing tuning more reliable and typically improves Kaggle F1.'
- What this solution (achieved 0.38178) has done: 'Your current score (0.38185) is far below the target (0.8077), so we should make the smallest change that directly improves mean F1 without changing the heuristic “model” itself. The highest-leverage low-risk fix is to tune the decoding step against a metric that better matches Kaggle’s mean sample-wise F1: instead of optimizing on raw OOF F1 only, we tune on a small “CV within OOF” split (still no leakage) and add a validation-tuned class-wise prior adjustment (logit bias) to correct systematic over/under-prediction per label. This keeps the same image→heuristic-probabilities logic, same calibration-by-scale, and same decoding machinery, but improves probability calibration enough that your existing threshold/top‑k logic works better. The submission format/path stays identical and the script still runs end-to-end and writes `submission.csv`.'
- What this solution (achieved 0.38175) has done: 'Your current score is far below the target, so we should cautiously improve mean sample-wise F1 by fixing the biggest remaining metric-alignment gap without changing your heuristic “model”: you’re tuning decoding on OOF predictions, but you still tune and apply a global per-class bias using prevalence matching, which is not F1-optimal and can push probabilities into a regime where threshold search is unstable. I keep the exact heuristic scoring and the same decode search space, but replace the prevalence-matching bias with a minimal coordinate-ascent bias tuned directly to maximize mean sample-wise F1 on OOF (using your existing decoding step as the objective), which is directly aligned with the Kaggle metric. To avoid overfitting and keep runtime <600s, the bias search be coarse (small grid) and only a single pass across classes, then we re-run the existing scale/threshold decoding search unchanged. This is a small, contained post-processing change that should move the score upward toward your target while preserving the rest of the pipeline and producing a valid `submission.csv`.'

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


def decode_probs_global_threshold(p, th, healthy_idx=None, healthy_drop=False):
    """
    Global-threshold decoding (kept for compatibility + as a baseline).
    Returns boolean mask (n, K).
    """
    chosen = p >= th
    if healthy_drop and healthy_idx is not None:
        has_disease = (
            np.sum(chosen, axis=1) - chosen[:, healthy_idx].astype(np.int32)
        ) > 0
        chosen[has_disease, healthy_idx] = False
    return chosen


def decode_probs_per_class_threshold(p, th_vec, healthy_idx=None, healthy_drop=False):
    """
    Per-class threshold decoding: chosen[i,j] = p[i,j] >= th_vec[j]
    Optional healthy_drop: remove 'healthy' if any other label is chosen.
    """
    thv = np.asarray(th_vec, dtype=np.float32).reshape(1, -1)
    chosen = p >= thv
    if healthy_drop and healthy_idx is not None:
        has_disease = (
            np.sum(chosen, axis=1) - chosen[:, healthy_idx].astype(np.int32)
        ) > 0
        chosen[has_disease, healthy_idx] = False
    return chosen


def apply_fallback(chosen_mask, p, fallback_mode="argmax", healthy_idx=None):
    """
    Avoid empty predictions hurting mean F1.
    """
    chosen = chosen_mask.copy()
    empty = np.sum(chosen, axis=1) == 0
    if np.any(empty):
        if fallback_mode == "healthy" and healthy_idx is not None:
            chosen[empty, healthy_idx] = True
        else:
            mx = np.argmax(p[empty], axis=1)
            rows = np.flatnonzero(empty)
            chosen[rows, mx] = True
    return chosen


def tune_thresholds_coordinate_ascent(
    p,
    y_true,
    th_init,
    th_candidates,
    healthy_idx=None,
    healthy_drop=False,
    fallback_mode="argmax",
    n_passes=2,
):
    """
    Coordinate ascent over classes: for each class j, choose threshold from th_candidates
    that maximizes mean sample-wise F1 while holding other thresholds fixed.
    """
    th = th_init.astype(np.float32, copy=True)

    chosen = decode_probs_per_class_threshold(
        p, th, healthy_idx=healthy_idx, healthy_drop=healthy_drop
    )
    chosen = apply_fallback(
        chosen, p, fallback_mode=fallback_mode, healthy_idx=healthy_idx
    )
    best_score = mean_sample_f1(y_true, chosen.astype(np.int8))

    for _ in range(int(n_passes)):
        improved_any = False
        for j in range(p.shape[1]):
            base_th = th[j]
            best_j_th = base_th
            best_j_score = best_score

            for cand in th_candidates:
                th[j] = float(cand)
                chosen = decode_probs_per_class_threshold(
                    p, th, healthy_idx=healthy_idx, healthy_drop=healthy_drop
                )
                chosen = apply_fallback(
                    chosen, p, fallback_mode=fallback_mode, healthy_idx=healthy_idx
                )
                sc = mean_sample_f1(y_true, chosen.astype(np.int8))
                if sc > best_j_score:
                    best_j_score = sc
                    best_j_th = float(cand)

            th[j] = best_j_th
            if best_j_score > best_score + 1e-12:
                best_score = best_j_score
                improved_any = True

        if not improved_any:
            break

    return th, float(best_score)


def apply_complex_gating(chosen_mask, complex_idx, min_other=2):
    if complex_idx is None:
        return chosen_mask
    chosen = chosen_mask.copy()
    other_cnt = np.sum(chosen, axis=1) - chosen[:, complex_idx].astype(np.int32)
    drop = chosen[:, complex_idx] & (other_cnt < int(min_other))
    if np.any(drop):
        chosen[drop, complex_idx] = False
    return chosen


def apply_max_labels_cap(chosen_mask, p, max_labels=None):
    if max_labels is None:
        return chosen_mask
    m = int(max_labels)
    if m <= 0:
        return chosen_mask
    chosen = chosen_mask.copy()
    cnt = np.sum(chosen, axis=1)
    over = np.flatnonzero(cnt > m)
    if over.size == 0:
        return chosen
    for i in over:
        idx = np.flatnonzero(chosen[i])
        if idx.size <= m:
            continue
        keep = idx[np.argsort(-p[i, idx])[:m]]
        chosen[i, :] = False
        chosen[i, keep] = True
    return chosen


def decode_with_best(p, best_cfg, healthy_idx=None, complex_idx=None):
    mode = best_cfg["mode"]
    if mode == "topk":
        order = np.argsort(-p, axis=1)
        k = int(best_cfg["k"])
        chosen = np.zeros_like(p, dtype=bool)
        rows = np.arange(p.shape[0])[:, None]
        chosen[rows, order[:, :k]] = True
        if best_cfg.get("healthy_drop", False) and healthy_idx is not None:
            has_disease = (
                np.sum(chosen, axis=1) - chosen[:, healthy_idx].astype(np.int32)
            ) > 0
            chosen[has_disease, healthy_idx] = False
    elif mode == "globalth":
        chosen = decode_probs_global_threshold(
            p,
            float(best_cfg["th_global"]),
            healthy_idx=healthy_idx,
            healthy_drop=best_cfg.get("healthy_drop", False),
        )
    else:
        chosen = decode_probs_per_class_threshold(
            p,
            best_cfg["th"],
            healthy_idx=healthy_idx,
            healthy_drop=best_cfg.get("healthy_drop", False),
        )

    chosen = apply_complex_gating(
        chosen, complex_idx=complex_idx, min_other=best_cfg.get("complex_min_other", 0)
    )
    chosen = apply_max_labels_cap(
        chosen, p, max_labels=best_cfg.get("max_labels", None)
    )
    chosen = apply_fallback(
        chosen,
        p,
        fallback_mode=best_cfg.get("fallback", "argmax"),
        healthy_idx=healthy_idx,
    )
    return chosen


def apply_logit_bias(p, bias):
    b = np.asarray(bias, dtype=np.float32).reshape(1, -1)
    return squash(logit(p) + b).astype(np.float32, copy=False)


def tune_logit_bias_f1_coordinate(
    p_raw,
    y_true,
    bias_init=None,
    bias_grid=np.array([-0.8, -0.4, -0.2, 0.0, 0.2, 0.4, 0.8], dtype=np.float32),
):
    """
    CHANGE (score improvement toward target): replace prevalence-matching bias with an F1-aligned,
    lightweight coordinate-ascent bias on OOF. This keeps the same probability model (logit shift)
    but tunes directly for mean sample-wise F1 after a fixed, simple decode (global threshold + fallback),
    which better matches the Kaggle metric and stabilizes downstream threshold search.
    """
    n_classes = p_raw.shape[1]
    if bias_init is None:
        bias = np.zeros((n_classes,), dtype=np.float32)
    else:
        bias = np.asarray(bias_init, dtype=np.float32).copy()

    healthy_idx_local = (
        class_names.index("healthy") if "healthy" in class_names else None
    )

    def _score_for_bias(bvec):
        p_adj = apply_logit_bias(p_raw, bvec)
        chosen = decode_probs_global_threshold(
            p_adj, 0.5, healthy_idx=healthy_idx_local, healthy_drop=False
        )
        chosen = apply_fallback(
            chosen, p_adj, fallback_mode="argmax", healthy_idx=healthy_idx_local
        )
        return mean_sample_f1(y_true, chosen.astype(np.int8))

    best_score = _score_for_bias(bias)

    for j in range(n_classes):
        best_bj = float(bias[j])
        best_sc_j = best_score
        base = float(bias[j])

        for b in bias_grid:
            bias[j] = float(b)
            sc = _score_for_bias(bias)
            if sc > best_sc_j + 1e-12:
                best_sc_j = sc
                best_bj = float(b)

        bias[j] = best_bj
        best_score = best_sc_j
        if best_bj == base:
            bias[j] = base

    return bias.astype(np.float32), float(best_score)


n_train = len(train)
idxs = np.arange(n_train, dtype=np.int64)
rng = np.random.RandomState(SEED)
rng.shuffle(idxs)

n_folds = 3
fold_id = np.zeros(n_train, dtype=np.int32)
for i, idx in enumerate(idxs):
    fold_id[idx] = i % n_folds

Y_all = labels_df.to_numpy(dtype=np.int8, copy=False)
oof_probs_raw = np.full((n_train, len(class_names)), np.nan, dtype=np.float32)

for f in range(n_folds):
    va_idx = np.flatnonzero(fold_id == f)
    train_va = train.iloc[va_idx].reset_index(drop=True)

    probs_va = np.zeros((len(train_va), len(class_names)), dtype=np.float32)
    for batch, batch_idxs in iter_image_batches(
        train_va, TRAIN_DIR, target_size=(h_target, w_target), batch_size=batch_size
    ):
        probs = heuristic_scores_from_batch(batch, class_names)
        probs_va[batch_idxs] = probs

    oof_probs_raw[va_idx] = probs_va
    print(f"Fold {f+1}/{n_folds} done. va_size={len(va_idx)}")

oof_mask = np.isfinite(oof_probs_raw).all(axis=1)
if not np.all(oof_mask):
    raise RuntimeError(
        f"OOF predictions missing for {int(np.sum(~oof_mask))} rows; cannot tune safely."
    )

oof_probs_raw_tune = oof_probs_raw[oof_mask]
Y_tune = Y_all[oof_mask]

print(
    "Computed OOF probs:", oof_probs_raw.shape, "tune subset:", oof_probs_raw_tune.shape
)

bias_vec, bias_f1 = tune_logit_bias_f1_coordinate(oof_probs_raw_tune, Y_tune)
print("Bias tuning (F1-aligned) score under fixed decode:", bias_f1)

oof_probs_raw_tune = apply_logit_bias(oof_probs_raw_tune, bias_vec)

logits = logit(oof_probs_raw_tune)
scales = np.array([0.5, 0.65, 0.8, 0.95, 1.1, 1.25, 1.4, 1.6], dtype=np.float32)

th_candidates = np.linspace(0.05, 0.95, 61, dtype=np.float32)

healthy_idx = class_names.index("healthy") if "healthy" in class_names else None
complex_idx = class_names.index("complex") if "complex" in class_names else None

best = {
    "score": -1.0,
    "scale": 1.0,
    "mode": "perclass",
    "th_global": 0.5,
    "k": 1,
    "healthy_drop": False,
    "fallback": "argmax",
    "th": np.full((len(class_names),), 0.5, dtype=np.float32),
    "complex_min_other": 0,
    "max_labels": None,
    "bias": bias_vec.astype(np.float32, copy=True),
}

for s in scales:
    p_s = squash(logits * s).astype(np.float32, copy=False)

    for healthy_drop in (False, True):
        for fb in ("argmax", "healthy"):
            th0 = np.quantile(p_s, 0.85, axis=0).astype(np.float32)
            th0 = np.clip(th0, 0.05, 0.95)

            th_tuned, sc_tuned = tune_thresholds_coordinate_ascent(
                p_s,
                Y_tune,
                th0,
                th_candidates,
                healthy_idx=healthy_idx,
                healthy_drop=healthy_drop,
                fallback_mode=fb,
                n_passes=2,
            )

            cfg = {
                "score": float(sc_tuned),
                "scale": float(s),
                "mode": "perclass",
                "th_global": 0.5,
                "k": 1,
                "healthy_drop": bool(healthy_drop),
                "fallback": fb,
                "th": th_tuned.astype(np.float32, copy=True),
                "complex_min_other": 0,
                "max_labels": None,
                "bias": bias_vec.astype(np.float32, copy=True),
            }

            candidates_complex = [0, 1, 2] if complex_idx is not None else [0]
            candidates_maxlbl = [None, 2, 3, 4, 5, 6]

            base_chosen = decode_with_best(
                p_s, cfg, healthy_idx=healthy_idx, complex_idx=complex_idx
            )
            base_sc = mean_sample_f1(Y_tune, base_chosen.astype(np.int8))
            best_local_sc = base_sc
            best_local = (0, None)

            for cmin in candidates_complex:
                for mcap in candidates_maxlbl:
                    cfg2 = dict(cfg)
                    cfg2["complex_min_other"] = int(cmin)
                    cfg2["max_labels"] = mcap
                    chosen2 = decode_with_best(
                        p_s, cfg2, healthy_idx=healthy_idx, complex_idx=complex_idx
                    )
                    sc2 = mean_sample_f1(Y_tune, chosen2.astype(np.int8))
                    if sc2 > best_local_sc:
                        best_local_sc = sc2
                        best_local = (int(cmin), mcap)

            cfg["complex_min_other"] = best_local[0]
            cfg["max_labels"] = best_local[1]
            cfg["score"] = float(best_local_sc)

            if cfg["score"] > best["score"]:
                best = cfg

    th_grid = np.linspace(0.05, 0.95, 181, dtype=np.float32)
    for healthy_drop in (False, True):
        for th in th_grid:
            for fb in ("argmax", "healthy"):
                cfg = {
                    "score": -1.0,
                    "scale": float(s),
                    "mode": "globalth",
                    "th_global": float(th),
                    "k": 1,
                    "healthy_drop": bool(healthy_drop),
                    "fallback": fb,
                    "th": np.full((len(class_names),), float(th), dtype=np.float32),
                    "complex_min_other": 0,
                    "max_labels": None,
                    "bias": bias_vec.astype(np.float32, copy=True),
                }
                chosen_fb = decode_with_best(
                    p_s, cfg, healthy_idx=healthy_idx, complex_idx=complex_idx
                )
                score = mean_sample_f1(Y_tune, chosen_fb.astype(np.int8))
                if score > best["score"]:
                    cfg["score"] = float(score)
                    best = cfg

    max_k = min(8, p_s.shape[1])
    order = np.argsort(-p_s, axis=1)  # descending
    rows = np.arange(Y_tune.shape[0])[:, None]

    for k in range(1, max_k + 1):
        y_pred_bin_k = np.zeros_like(Y_tune, dtype=np.int8)
        topk = order[:, :k]
        y_pred_bin_k[rows, topk] = 1

        for healthy_drop in (False, True):
            yk = y_pred_bin_k.copy()
            if healthy_drop and healthy_idx is not None:
                has_disease = (np.sum(yk, axis=1) - yk[:, healthy_idx]) > 0
                yk[has_disease, healthy_idx] = 0

            cfg = {
                "score": -1.0,
                "scale": float(s),
                "mode": "topk",
                "th_global": 0.5,
                "k": int(k),
                "healthy_drop": bool(healthy_drop),
                "fallback": "argmax",
                "th": np.full((len(class_names),), 0.3, dtype=np.float32),
                "complex_min_other": 0,
                "max_labels": None,
                "bias": bias_vec.astype(np.float32, copy=True),
            }

            chosen_bool = yk.astype(bool)
            candidates_complex = [0, 1, 2] if complex_idx is not None else [0]
            candidates_maxlbl = [None, 2, 3, 4, 5, 6]

            best_local_sc = -1.0
            best_local = (0, None)
            for cmin in candidates_complex:
                for mcap in candidates_maxlbl:
                    chosen2 = chosen_bool
                    chosen2 = apply_complex_gating(
                        chosen2, complex_idx, min_other=int(cmin)
                    )
                    chosen2 = apply_max_labels_cap(chosen2, p_s, max_labels=mcap)
                    chosen2 = apply_fallback(
                        chosen2, p_s, fallback_mode="argmax", healthy_idx=healthy_idx
                    )
                    sc2 = mean_sample_f1(Y_tune, chosen2.astype(np.int8))
                    if sc2 > best_local_sc:
                        best_local_sc = sc2
                        best_local = (int(cmin), mcap)

            cfg["complex_min_other"] = best_local[0]
            cfg["max_labels"] = best_local[1]
            cfg["score"] = float(best_local_sc)

            if cfg["score"] > best["score"]:
                best = cfg

print("OOF mean sample-wise F1 (calibration + decoding):", best["score"])
print("Best scale:", best["scale"])
print("Best mode:", best["mode"], "k:", best["k"])
print("Best global th:", best.get("th_global", None))
print("Best healthy_drop:", best["healthy_drop"], "fallback:", best["fallback"])
print("Best complex_min_other:", best.get("complex_min_other", 0))
print("Best max_labels cap:", best.get("max_labels", None))
print("Per-class thresholds used:", (best["mode"] == "perclass"))
print(
    "Using logit bias calibration:",
    "bias" in best,
    "bias_shape:",
    np.asarray(best["bias"]).shape,
)



## === cell 5
n_test = len(submissions)
n_classes = len(class_names)

preds = np.zeros((n_test, n_classes), dtype=np.float32)

for batch, batch_idxs in iter_image_batches(
    submissions, TEST_DIR, target_size=(h_target, w_target), batch_size=batch_size
):
    probs = heuristic_scores_from_batch(batch, class_names)

    probs = apply_logit_bias(probs, best["bias"])
    probs = squash(logit(probs) * best["scale"]).astype(np.float32)

    preds[batch_idxs] = probs

print("Preds shape:", preds.shape)



## === cell 6
healthy_idx = class_names.index("healthy") if "healthy" in class_names else None
complex_idx = class_names.index("complex") if "complex" in class_names else None

chosen_mask = decode_with_best(
    preds, best, healthy_idx=healthy_idx, complex_idx=complex_idx
)

out_labels = []
for i in range(n_test):
    chosen_idx = np.flatnonzero(chosen_mask[i])
    if chosen_idx.size == 0:
        out_labels.append("")
        continue
    chosen_idx = chosen_idx[np.argsort(-preds[i, chosen_idx])]
    chosen = [class_names[j] for j in chosen_idx]
    out_labels.append(" ".join(chosen))

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
