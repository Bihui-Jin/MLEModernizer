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
Mean column-wise ROC AUC.

## Submission Format
For each image_id in the test set, you must predict a probability for each target variable. The file should contain a header and have the following format:

```
image_id,
test_0,0.25,0.25,0.25,0.25
test_1,0.25,0.25,0.25,0.25
test_2,0.25,0.25,0.25,0.25
etc.
```

## Dataset
Given a photo of an apple leaf, can you accurately assess its health? This competition will challenge you to distinguish between leaves which are healthy, those which are infected with apple rust, those that have apple scab, and those with more than one disease.

**train.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

**images**

A folder containing the train and test images, in jpg format.

**test.csv**

- `image_id`: the foreign key

**sample_submission.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

# 2. Python version

3.8

# 3. Installed packages

geopandas==0.14.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
sklearn-pandas==2.2.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        input/
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        working/
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
```

-> data/plant-pathology-2020-fgvc7/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/plant-pathology-2020-fgvc7/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/plant-pathology-2020-fgvc7/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> (stopped after 10 files for performance)

# 5. Target score

0.9646650712510072

# 6. Current score

0.77894

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'You’re erroring because `/kaggle/input/submissions/` doesn’t exist in this environment, so `submissions_all` is empty and indexing `[0,2,5]` crashes. I make the path detection robust by falling back to the competition’s `sample_submission.csv` and, when no external submissions are available, generate a valid baseline submission by predicting class priors from `train.csv` (score-improving vs. uniform 0.25 while keeping the “no model training” core approach). I also harden the ensembling code to skip non-CSV files, validate indices/weights, align rows by `image_id`, and always write a correct `submission.csv` with the required columns. This ensures the notebook runs end-to-end and produces a valid `.csv` submission deterministically.'
- What this solution (achieved 0.70048) has done: 'Your current 0.5 score is consistent with predicting near-uninformative constants (class priors), which can’t rank images and therefore yields ~0.5 AUC. To move toward the 0.9647 target with minimal core-logic change, I keep your “no training loop / no deep model” approach but replace constant priors with a lightweight, deterministic image-feature baseline: compute simple per-image color statistics from the provided JPGs and fit one-vs-rest logistic regression for each label. This produces per-image varying probabilities (so ROC AUC can increase substantially) while staying fast and within the installed package set (pandas + scikit-learn). I also keep your ensembling path intact; if external submissions exist, it behaves as before, otherwise it trains this baseline and writes a valid `submission.csv` in the required column order.'
- What this solution (achieved 0.71872) has done: 'I keep your exact “simple image statistics + OvR LogisticRegression” approach, but strengthen it slightly in ways that typically improve ROC AUC without changing the core logic: (1) add a few low-cost, deterministic texture/contrast features (still just summary stats, no deep model), (2) use `class_weight="balanced"` to reduce bias for rare classes (notably `multiple_diseases`), and (3) tune `C` mildly and increase `max_iter` to ensure stable convergence. I also make image loading more robust by trying `.JPG/.jpeg/.png` fallbacks (some datasets vary casing) while keeping paths unchanged. The submission writing and optional external-submission ensembling behavior stays the same.'
- What this solution (achieved 0.74282) has done: 'To move your ROC AUC closer to the 0.9647 target while preserving the same “simple image stats + OvR LogisticRegression” core, I keep the pipeline identical but make two minimal, high-impact adjustments: (1) add a few more deterministic, low-cost summary features that capture disease-relevant color/texture differences, and (2) use a slightly stronger regularization setting and a more stable solver configuration for better probability ranking (especially for the rare `multiple_diseases` class). I also ensure the output probabilities are properly normalized per row (sum to 1) to better match the mutual-exclusivity structure of labels in this dataset, which often improves column-wise AUC without changing the model family. The ensembling path is left intact; changes apply only to the fallback baseline used when no external submissions are present.'
- What this solution (achieved 0.77894) has done: 'Your current gap to the target is large (0.74282 vs 0.96467), so we should increase score while keeping the same core “simple image stats + OvR LogisticRegression” approach. I make one minimal, high-impact enhancement: add very lightweight, deterministic color/texture features (channel quantiles, central crop stats, and a couple of simple gradient quantiles) that often improve ranking signal for ROC AUC without changing the model family or training approach. I also make the probability post-processing safer by applying row-normalization only when it’s not overly distorting the OvR outputs (guarded blend), which can help AUC stability. Everything still runs end-to-end within time and writes a valid `submission.csv` with the required columns.'

# 9. Code solution

## === cell 0
import os
import pandas as pd



## === cell 1
CANDIDATE_DATA_DIRS = [
    "/kaggle/input/plant-pathology-2020-fgvc7",
    "/kaggle/data/plant-pathology-2020-fgvc7",
    "/kaggle/input",
    "/kaggle/data",
]

DATA_DIR = None
for d in CANDIDATE_DATA_DIRS:
    if os.path.exists(os.path.join(d, "sample_submission.csv")) and os.path.exists(
        os.path.join(d, "test.csv")
    ):
        DATA_DIR = d
        break

if DATA_DIR is None:
    raise FileNotFoundError(
        "Could not find competition files (sample_submission.csv, test.csv). "
        f"Tried: {CANDIDATE_DATA_DIRS}"
    )

SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")
TEST_CSV_PATH = os.path.join(DATA_DIR, "test.csv")
TRAIN_CSV_PATH = os.path.join(DATA_DIR, "train.csv")

CANDIDATE_IMAGE_DIRS = [
    os.path.join(DATA_DIR, "images"),
    os.path.join(DATA_DIR, "plant-pathology-2020-fgvc7", "images"),
]
IMAGE_DIR = None
for d in CANDIDATE_IMAGE_DIRS:
    if os.path.isdir(d):
        IMAGE_DIR = d
        break

print("Using DATA_DIR:", DATA_DIR)
print("SAMPLE_SUB_PATH:", SAMPLE_SUB_PATH)
print("IMAGE_DIR:", IMAGE_DIR)



## === cell 2
SUBMISSIONS_PATH = "/kaggle/input/submissions/"

submissions_all = []
if os.path.isdir(SUBMISSIONS_PATH):
    for dirname, _, filenames in os.walk(SUBMISSIONS_PATH):
        for filename in filenames:
            if filename.lower().endswith(".csv"):
                submissions_all.append(os.path.join(dirname, filename))
submissions_all.sort()

print("Found submissions:", submissions_all)




## === cell 3
def ensemble(submissions_all, sub_idx, weights=None, required_cols=None):
    """
    Weighted average ensemble of submission files.
    Fixes:
      - Validates indices/weights lengths to avoid IndexError.
      - Aligns by image_id to avoid row-order mismatches.
      - Filters to required_cols and returns a DataFrame with image_id + required_cols.
    """
    if required_cols is None:
        required_cols = ["healthy", "multiple_diseases", "rust", "scab"]
    if weights is None:
        weights = [1.0] * len(sub_idx)

    if len(sub_idx) != len(weights):
        raise ValueError(
            f"sub_idx and weights must have same length, got {len(sub_idx)} and {len(weights)}"
        )

    if len(submissions_all) == 0:
        raise ValueError("No submission files provided to ensemble().")

    ref = pd.read_csv(submissions_all[sub_idx[0]])
    if "image_id" not in ref.columns:
        raise ValueError(f"Missing image_id in {submissions_all[sub_idx[0]]}")
    ref = ref[["image_id"]].copy()

    acc = pd.DataFrame({"image_id": ref["image_id"]})
    for c in required_cols:
        acc[c] = 0.0

    for i, idx in enumerate(sub_idx):
        if idx < 0 or idx >= len(submissions_all):
            raise IndexError(
                f"Requested sub_idx={idx} but only {len(submissions_all)} files exist."
            )
        path = submissions_all[idx]
        w = float(weights[i])
        print(f"I'm taking submission {path} with weight {w}")

        sub = pd.read_csv(path)
        missing = [c for c in (["image_id"] + required_cols) if c not in sub.columns]
        if missing:
            raise ValueError(f"{path} is missing columns: {missing}")

        sub = sub[["image_id"] + required_cols].copy()
        sub = ref.merge(sub, on="image_id", how="left", validate="one_to_one")
        if sub[required_cols].isna().any().any():
            raise ValueError(
                f"{path} has missing predictions for some image_id after alignment."
            )

        for c in required_cols:
            acc[c] += sub[c].astype(float) * w

    return acc




## === cell 4
def make_submission_file(submission_df, out_path="submission.csv"):
    """
    Writes a valid submission CSV with correct columns/order.
    Fix: ensure required header and exact columns.
    """
    required_cols = ["image_id", "healthy", "multiple_diseases", "rust", "scab"]
    missing = [c for c in required_cols if c not in submission_df.columns]
    if missing:
        raise ValueError(f"submission_df missing columns: {missing}")

    submission_df = submission_df[required_cols].copy()
    submission_df.to_csv(out_path, index=False)
    print("Wrote:", out_path, "shape=", submission_df.shape)




## === cell 5
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def _img_path_from_id(image_id, image_dir):
    exts = [".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG"]
    for ext in exts:
        p = os.path.join(image_dir, f"{image_id}{ext}")
        if os.path.exists(p):
            return p
    return os.path.join(image_dir, f"{image_id}.jpg")


def extract_basic_rgb_stats(image_ids, image_dir, size=(128, 128)):
    """
    Deterministic, fast features. Core logic preserved (simple summary stats + OvR logreg).

    Change (score-improving, minimal): extend summary-stat feature set with a few
    robust, low-cost signals that typically improve ROC AUC ranking:
      - per-channel quantiles (10/50/90) on full image
      - per-channel mean/std on central crop (captures lesion localization)
      - a couple of simple gradient quantiles on grayscale (texture)

    Still no deep model, no heavy feature extraction; remains fast and deterministic.

    Total features: 32 (previous) + 9 (RGB quantiles) + 6 (central mean/std) + 2 (grad quantiles) = 49
    """
    feats = np.zeros((len(image_ids), 49), dtype=np.float32)
    eps = 1e-6

    for i, image_id in enumerate(image_ids):
        p = _img_path_from_id(image_id, image_dir)
        if not os.path.exists(p):
            raise FileNotFoundError(f"Image not found: {p}")

        im = Image.open(p).convert("RGB")
        if size is not None:
            im = im.resize(size)
        arr = np.asarray(im, dtype=np.float32) / 255.0  # H,W,3

        r = arr[..., 0]
        g = arr[..., 1]
        b = arr[..., 2]

        stats = []
        for ch in (r, g, b):
            stats.extend([ch.mean(), ch.std(), ch.min(), ch.max()])

        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        stats.extend([gray.mean(), gray.std()])

        idx = (g - r) / (g + r + eps)
        stats.extend([idx.mean(), idx.std()])

        stats.extend([gray.min(), gray.max()])

        dx = np.abs(gray[:, 1:] - gray[:, :-1]).mean()
        dy = np.abs(gray[1:, :] - gray[:-1, :]).mean()
        stats.extend([dx, dy])

        exg = 2.0 * g - r - b
        stats.extend([exg.mean(), exg.std()])

        rg_ratio = r / (g + eps)
        stats.extend([rg_ratio.mean(), rg_ratio.std()])

        mx = np.maximum(np.maximum(r, g), b)
        mn = np.minimum(np.minimum(r, g), b)
        diff = mx - mn

        h = np.zeros_like(mx)
        mask = diff > eps
        r_eq = (mx == r) & mask
        g_eq = (mx == g) & mask
        b_eq = (mx == b) & mask
        h[r_eq] = ((g[r_eq] - b[r_eq]) / (diff[r_eq] + eps)) % 6.0
        h[g_eq] = ((b[g_eq] - r[g_eq]) / (diff[g_eq] + eps)) + 2.0
        h[b_eq] = ((r[b_eq] - g[b_eq]) / (diff[b_eq] + eps)) + 4.0
        h = (h / 6.0).astype(np.float32)

        s = (diff / (mx + eps)).astype(np.float32)
        v = mx.astype(np.float32)

        stats.extend([h.mean(), h.std(), s.mean(), s.std(), v.mean(), v.std()])

        stats.extend([(gray < 0.2).mean(), (gray > 0.8).mean()])

        for ch in (r, g, b):
            q10, q50, q90 = np.quantile(ch, [0.1, 0.5, 0.9])
            stats.extend([float(q10), float(q50), float(q90)])

        H, W = gray.shape
        y0, y1 = int(0.25 * H), int(0.75 * H)
        x0, x1 = int(0.25 * W), int(0.75 * W)
        rc = r[y0:y1, x0:x1]
        gc = g[y0:y1, x0:x1]
        bc = b[y0:y1, x0:x1]
        stats.extend([rc.mean(), rc.std(), gc.mean(), gc.std(), bc.mean(), bc.std()])

        gx = np.abs(gray[:, 1:] - gray[:, :-1])
        gy = np.abs(gray[1:, :] - gray[:-1, :])
        gx_p = np.pad(gx, ((0, 0), (0, 1)), mode="edge")
        gy_p = np.pad(gy, ((0, 1), (0, 0)), mode="edge")
        grad = gx_p + gy_p
        gq50, gq90 = np.quantile(grad, [0.5, 0.9])
        stats.extend([float(gq50), float(gq90)])

        feats[i, :] = np.array(stats, dtype=np.float32)

    return feats


def fit_ovr_logreg_predict_proba(X_train, y_train_df, X_test, class_names):
    """
    Fits separate binary LogisticRegression models per class (OvR),
    returning test probabilities in the required column order.
    """
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)

    preds = {}
    for c in class_names:
        y = y_train_df[c].astype(int).values

        clf = LogisticRegression(
            solver="saga",
            penalty="l2",
            max_iter=2500,
            C=3.0,
            class_weight="balanced",
            n_jobs=None,
            random_state=0,
        )
        clf.fit(Xtr, y)
        p = clf.predict_proba(Xte)[:, 1]
        preds[c] = np.clip(p.astype(np.float64), 1e-8, 1 - 1e-8)

    return preds


def row_normalize_predictions(df, class_cols, eps=1e-12):
    """
    Produces per-row sum-to-1 probabilities.
    """
    arr = df[class_cols].to_numpy(dtype=np.float64)
    arr = np.clip(arr, eps, 1.0)
    s = arr.sum(axis=1, keepdims=True)
    arr = arr / np.maximum(s, eps)
    out = df.copy()
    for j, c in enumerate(class_cols):
        out[c] = arr[:, j]
    return out


def safe_blend_ovr_and_row_norm(df, class_cols, alpha=0.35):
    """
    Change (score-improving, minimal/stability): OvR probabilities are not mutually exclusive,
    while dataset labels are (nearly) exclusive. Full row-normalization can help but sometimes
    over-corrects. We blend a small amount of row-normalized probs back into raw OvR probs,
    which often improves mean AUC ranking without destabilizing columns.

    alpha in [0,1]: 0 keeps pure OvR; 1 uses pure row-normalized.
    """
    raw = df[class_cols].to_numpy(dtype=np.float64)
    rn = row_normalize_predictions(df, class_cols)[class_cols].to_numpy(
        dtype=np.float64
    )
    blended = (1.0 - alpha) * raw + alpha * rn
    blended = np.clip(blended, 1e-12, 1.0 - 1e-12)
    out = df.copy()
    for j, c in enumerate(class_cols):
        out[c] = blended[:, j]
    return out




## === cell 6
required_cols = ["healthy", "multiple_diseases", "rust", "scab"]

if len(submissions_all) >= 1:
    desired_idx = [0, 2, 5]
    if len(submissions_all) > max(desired_idx, default=0):
        sub_idx = desired_idx
        weights = [0.1, 0.8, 0.1]
    else:
        sub_idx = list(range(len(submissions_all)))
        weights = [1.0 / len(submissions_all)] * len(submissions_all)

    submission_df = ensemble(
        submissions_all, sub_idx=sub_idx, weights=weights, required_cols=required_cols
    )
    for c in required_cols:
        submission_df[c] = submission_df[c].astype(float).clip(1e-12, 1.0 - 1e-12)
    make_submission_file(submission_df, out_path="submission.csv")
else:
    if IMAGE_DIR is None:
        raise FileNotFoundError(
            "No external submissions found and could not locate images/ directory for feature extraction baseline."
        )

    train = pd.read_csv(TRAIN_CSV_PATH)
    test = pd.read_csv(TEST_CSV_PATH)

    train_ids = train["image_id"].astype(str).tolist()
    test_ids = test["image_id"].astype(str).tolist()

    X_train = extract_basic_rgb_stats(train_ids, IMAGE_DIR, size=(128, 128))
    X_test = extract_basic_rgb_stats(test_ids, IMAGE_DIR, size=(128, 128))

    preds = fit_ovr_logreg_predict_proba(
        X_train, train[required_cols], X_test, required_cols
    )

    submission_df = pd.DataFrame({"image_id": test_ids})
    for c in required_cols:
        submission_df[c] = preds[c]

    submission_df = safe_blend_ovr_and_row_norm(
        submission_df, required_cols, alpha=0.35
    )

    make_submission_file(submission_df, out_path="submission.csv")
