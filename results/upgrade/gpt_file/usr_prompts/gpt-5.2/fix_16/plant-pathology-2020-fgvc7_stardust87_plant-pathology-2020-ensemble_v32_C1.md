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

0.78391

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'You’re erroring because `/kaggle/input/submissions/` doesn’t exist in this environment, so `submissions_all` is empty and indexing `[0,2,5]` crashes. I make the path detection robust by falling back to the competition’s `sample_submission.csv` and, when no external submissions are available, generate a valid baseline submission by predicting class priors from `train.csv` (score-improving vs. uniform 0.25 while keeping the “no model training” core approach). I also harden the ensembling code to skip non-CSV files, validate indices/weights, align rows by `image_id`, and always write a correct `submission.csv` with the required columns. This ensures the notebook runs end-to-end and produces a valid `.csv` submission deterministically.'
- What this solution (achieved 0.70048) has done: 'Your current 0.5 score is consistent with predicting near-uninformative constants (class priors), which can’t rank images and therefore yields ~0.5 AUC. To move toward the 0.9647 target with minimal core-logic change, I keep your “no training loop / no deep model” approach but replace constant priors with a lightweight, deterministic image-feature baseline: compute simple per-image color statistics from the provided JPGs and fit one-vs-rest logistic regression for each label. This produces per-image varying probabilities (so ROC AUC can increase substantially) while staying fast and within the installed package set (pandas + scikit-learn). I also keep your ensembling path intact; if external submissions exist, it behaves as before, otherwise it trains this baseline and writes a valid `submission.csv` in the required column order.'
- What this solution (achieved 0.71872) has done: 'I keep your exact “simple image statistics + OvR LogisticRegression” approach, but strengthen it slightly in ways that typically improve ROC AUC without changing the core logic: (1) add a few low-cost, deterministic texture/contrast features (still just summary stats, no deep model), (2) use `class_weight="balanced"` to reduce bias for rare classes (notably `multiple_diseases`), and (3) tune `C` mildly and increase `max_iter` to ensure stable convergence. I also make image loading more robust by trying `.JPG/.jpeg/.png` fallbacks (some datasets vary casing) while keeping paths unchanged. The submission writing and optional external-submission ensembling behavior stays the same.'
- What this solution (achieved 0.74282) has done: 'To move your ROC AUC closer to the 0.9647 target while preserving the same “simple image stats + OvR LogisticRegression” core, I keep the pipeline identical but make two minimal, high-impact adjustments: (1) add a few more deterministic, low-cost summary features that capture disease-relevant color/texture differences, and (2) use a slightly stronger regularization setting and a more stable solver configuration for better probability ranking (especially for the rare `multiple_diseases` class). I also ensure the output probabilities are properly normalized per row (sum to 1) to better match the mutual-exclusivity structure of labels in this dataset, which often improves column-wise AUC without changing the model family. The ensembling path is left intact; changes apply only to the fallback baseline used when no external submissions are present.'
- What this solution (achieved 0.77894) has done: 'Your current gap to the target is large (0.74282 vs 0.96467), so we should increase score while keeping the same core “simple image stats + OvR LogisticRegression” approach. I make one minimal, high-impact enhancement: add very lightweight, deterministic color/texture features (channel quantiles, central crop stats, and a couple of simple gradient quantiles) that often improve ranking signal for ROC AUC without changing the model family or training approach. I also make the probability post-processing safer by applying row-normalization only when it’s not overly distorting the OvR outputs (guarded blend), which can help AUC stability. Everything still runs end-to-end within time and writes a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.77125) has done: 'I keep your exact “simple deterministic image summary stats + OvR LogisticRegression” approach, but fix two small issues that commonly depress ROC AUC for this competition: (1) ensure `multiple_diseases` isn’t forced to “balanced” weighting (it often over-amplifies noise for this rare class and hurts ranking), and (2) add one tiny, cheap feature group that tends to help scab/rust separation: per-channel LAB means/stds (still just summary stats, no deep model). I also make the external-submission ensembling path safer by aligning to the competition `test.csv` order (not the first submission’s order) to avoid accidental row-order AUC loss. These are minimal changes, keep semantics the same, and should move your score upward toward the 0.9647 target without changing the overall pipeline.'
- What this solution (achieved 0.77111) has done: 'Your current score (0.77125) is well below the target (0.96467), so we should cautiously increase ROC AUC while keeping your same core pipeline: deterministic image-summary features + per-class LogisticRegression (OvR) and the same submission/ensembling behavior. The smallest likely gain is to (1) add a couple of very cheap, disease-relevant “structure” features (downsampled grayscale DCT energy ratios) and (2) slightly improve how the rare `multiple_diseases` class is handled by using a less aggressive but non-None class weighting (custom weights instead of fully unweighted). These changes keep the same model family, same training approach, same loss/semantics, and remain fast (<600s) while adding ranking signal that usually helps mean column-wise AUC. The submission writing remains identical and aligned to `test.csv` order.'
- What this solution (achieved 0.77932) has done: 'We should increase the score (0.77111) toward the higher target (0.96467) while keeping your exact core approach: deterministic hand-crafted image summary features + per-class LogisticRegression (OvR). The most leverage with minimal risk is to (1) add a few very cheap but informative color/structure features (LAB quantiles + simple “spotty disease” top-hat contrast summaries) without changing the model family or training loop, and (2) slightly adjust regularization for the rare `multiple_diseases` column only to improve ranking stability (AUC) while leaving other classes unchanged. I also keep the same submission alignment and clipping, and still write `submission.csv` with the required columns. These changes are small, deterministic, and should move AUC upward without altering the pipeline semantics.'
- What this solution (achieved 0.78399) has done: 'We keep your exact “hand-crafted image summary features + per-class LogisticRegression (OvR)” pipeline, but fix one high-impact bug: your DCT features are currently extremely expensive (NxN matrix DCT) and likely forcing you to under-run or behave inconsistently; we replace them with an equivalent DCT computed via `np.fft` on small vectors (same semantics: frequency-energy ratios, just computed efficiently and deterministically). Then we add one minimal, low-cost feature group that often boosts ROC AUC here without changing the modeling approach: per-channel 16-bin normalized histograms in LAB space (captures color shifts more robustly than a few quantiles). Finally, we keep your probability post-processing but slightly reduce the row-normalization blend (alpha) so we don’t distort OvR rankings as much, which typically improves mean column-wise AUC.'
- What this solution (achieved 0.78469) has done: 'Your current score (0.78399) is far below the target (0.96467), so we should cautiously increase mean column-wise ROC AUC while keeping the exact same core pipeline: handcrafted image-summary features + per-class OvR LogisticRegression + the same submission writing/optional ensembling. The smallest likely gain is to make the LogisticRegression training a bit more metric-aligned and stable by (1) adding a deterministic stratified train/OOF calibration step (still logistic regression; no new model family) to reduce probability miscalibration, especially for the rare `multiple_diseases` class, and (2) switching to a solver/regularization setup that is more stable for small tabular feature sets (`liblinear`) while keeping L2 logistic regression. Finally, we keep your existing post-processing but slightly reduce row-normalization blending because ROC AUC is ranking-based and aggressive row-normalization can distort per-column rankings. All changes are designed to be minimal, deterministic, and to run within the time limit while producing a valid `submission.csv`.'
- What this solution (achieved 0.78595) has done: 'Your score (0.78469) is far below the target (0.96467), so we should increase mean column-wise ROC AUC with the smallest safe changes while preserving your core “handcrafted image summary stats + per-class OvR LogisticRegression (+ optional calibration) + same submission writing/ensembling” pipeline. The most likely low-risk gain is to make the per-class calibration step more metric-appropriate by using proper sigmoid calibration on OOF logits (via `CalibratedClassifierCV(method="sigmoid", cv=StratifiedKFold)`), which preserves the same base LogisticRegression but typically improves probability ranking stability. I also make a minimal, score-relevant fix in the external-submission ensembling branch: normalize by the sum of weights so predictions stay on the correct probability scale (otherwise overweighting can push values >1 then clipping harms ranking). Finally, I keep your feature extraction and model family intact and ensure the submission is aligned to `test.csv` order and written as `submission.csv`.'
- What this solution (achieved 0.78737) has done: 'We should increase score (0.78595 → 0.96467), and the smallest likely gain without changing your core approach is to make the per-class calibration more stable for ROC AUC by switching from isotonic/sigmoid CV calibration to an explicit out-of-fold Platt scaling (still logistic calibration, still based on LogisticRegression) trained on OOF decision scores, then refit the base model on full data for test inference. This avoids CalibratedClassifierCV’s fold-wise refits on probabilities and instead calibrates on the more appropriate continuous decision function, which often improves ranking stability (AUC) on small datasets. I also make the rare `multiple_diseases` handling slightly more conservative by using a milder positive-class weight (reduces overfitting noise that can hurt AUC) while keeping the same model family and no new training paradigm. All other logic (feature extraction, OvR logistic regression, submission alignment/writing, optional external ensembling) is preserved.'
- What this solution (achieved 0.78379) has done: 'We keep your exact “handcrafted image summary stats + per-class OvR LogisticRegression + OOF Platt scaling + conservative row-normalization blend” core pipeline, but make two minimal, score-relevant upgrades to push mean ROC AUC upward. First, we add a tiny set of disease-relevant *excess-red* and *excess-blue* summary features (analogous to your ExG) plus a couple of simple “spot coverage” fractions from the top-hat contrast map; these are deterministic and cheap but often improve ranking signal for rust/scab vs healthy. Second, we slightly strengthen the rare `multiple_diseases` calibration stability by using one more CV split for its Platt step (6 instead of 5) while leaving other classes untouched. Everything else (paths, ensembling behavior, model family, training semantics, and submission writing) remains the same and still produces `submission.csv`.'
- What this solution (achieved 0.78368) has done: 'Your current score (0.78379) is far below the target (0.96467), so we should cautiously increase mean column-wise ROC AUC while keeping the exact same core pipeline: handcrafted image summary features + per-class OvR LogisticRegression + OOF Platt scaling + the same submission writing/optional ensembling logic. The smallest, most reliable gain here is to fix a feature bug: your LAB histogram bin edges are defined in [0,1] but `np.histogram` defaults to the data range unless `range=` is provided, so those hist features are currently inconsistent/low-signal; forcing the intended [0,1] range makes them meaningful and should improve ranking. Second, we add one tiny, deterministic feature group that is very cheap and often helps this dataset: per-image “green mask” coverage and masked color statistics (disease spots reduce green area), without changing the modeling approach. Everything remains deterministic, runs fast, and still writes a valid `submission.csv` with the required columns aligned to `test.csv`.'
- What this solution (achieved 0.78391) has done: 'Your current score (0.78368) is well below the target (0.96467), so we should increase mean column-wise ROC AUC with the smallest low-risk changes while preserving your exact core pipeline (handcrafted image-summary features + per-class OvR LogisticRegression + OOF Platt scaling + same submission writing/ensembling). The most likely gain without changing the modeling approach is to fix a subtle but impactful issue: your Platt calibrator is trained on OOF decision scores from base models trained on *fold-specific* scaled features, but then applied to decision scores from a model trained on *globally* scaled features—this distribution shift can hurt ranking/calibration. I keep the same Platt-scaling logic but compute OOF decision scores using the same global scaling (already computed in `fit_ovr_logreg_predict_proba`) to remove that mismatch. Additionally, I replace the fixed alpha in the row-normalization blend with a tiny class-aware alpha (lower for rare `multiple_diseases`) to avoid distorting its per-column ranking, which is directly what ROC AUC measures.'

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
def ensemble(
    submissions_all, sub_idx, weights=None, required_cols=None, test_csv_path=None
):
    """
    Weighted average ensemble of submission files.

    Correctness (score-relevant): align output row order to competition test.csv
    rather than trusting the first submission's order.
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

    if test_csv_path is not None and os.path.exists(test_csv_path):
        ref = pd.read_csv(test_csv_path)
        if "image_id" not in ref.columns:
            raise ValueError(f"Missing image_id in {test_csv_path}")
        ref = ref[["image_id"]].copy()
    else:
        ref0 = pd.read_csv(submissions_all[sub_idx[0]])
        if "image_id" not in ref0.columns:
            raise ValueError(f"Missing image_id in {submissions_all[sub_idx[0]]}")
        ref = ref0[["image_id"]].copy()

    acc = pd.DataFrame({"image_id": ref["image_id"]})
    for c in required_cols:
        acc[c] = 0.0

    wsum = float(sum(weights))
    if not (wsum > 0):
        raise ValueError("Sum of weights must be positive.")

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

    for c in required_cols:
        acc[c] = acc[c] / wsum

    return acc




## === cell 4
def make_submission_file(submission_df, out_path="submission.csv"):
    """
    Writes a valid submission CSV with correct columns/order.
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
from sklearn.model_selection import StratifiedKFold


def _img_path_from_id(image_id, image_dir):
    exts = [".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG"]
    for ext in exts:
        p = os.path.join(image_dir, f"{image_id}{ext}")
        if os.path.exists(p):
            return p
    return os.path.join(image_dir, f"{image_id}.jpg")


def _dct_1d(x):
    x = np.asarray(x, dtype=np.float32)
    N = x.shape[0]
    v = np.concatenate([x, x[::-1]]).astype(np.float32)  # length 2N
    V = np.fft.fft(v)
    k = np.arange(N, dtype=np.float32)
    X = np.real(V[:N] * np.exp(-1j * np.pi * k / (2.0 * N))).astype(np.float32)
    return X


def _dct2(block):
    block = np.asarray(block, dtype=np.float32)
    tmp = np.stack([_dct_1d(row) for row in block], axis=0)
    out = np.stack([_dct_1d(col) for col in tmp.T], axis=1)
    return out


def _mean_std_quantiles(x, qs=(0.1, 0.5, 0.9)):
    x = np.asarray(x, dtype=np.float32)
    q = np.quantile(x, qs)
    return [float(x.mean()), float(x.std()), *[float(v) for v in q]]


def extract_basic_rgb_stats(image_ids, image_dir, size=(128, 128)):
    """
    Deterministic, fast features. Core logic preserved (simple summary stats + OvR logreg).
    """
    feats = np.zeros((len(image_ids), 138), dtype=np.float32)
    eps = 1e-6
    bin_edges = np.linspace(0.0, 1.0, 17, dtype=np.float32)

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

        exr = 2.0 * r - g - b
        exb = 2.0 * b - r - g
        stats.extend([exr.mean(), exr.std(), exb.mean(), exb.std()])

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

        lab = im.convert("LAB")
        lab_arr = np.asarray(lab, dtype=np.float32) / 255.0
        L = lab_arr[..., 0]
        A = lab_arr[..., 1]
        Bc = lab_arr[..., 2]
        stats.extend([L.mean(), L.std(), A.mean(), A.std(), Bc.mean(), Bc.std()])

        g16 = np.asarray(im.resize((16, 16)).convert("L"), dtype=np.float32) / 255.0
        g16 = g16 - g16.mean()
        dct = _dct2(g16)
        E = (dct * dct).astype(np.float32)
        total = float(E.sum()) + 1e-12
        low = float(E[:4, :4].sum())  # includes DC and low-freq
        high = float(E[4:, 4:].sum())
        stats.extend([low / total, high / total])

        stats.extend(_mean_std_quantiles(L))
        stats.extend(_mean_std_quantiles(A))
        stats.extend(_mean_std_quantiles(Bc))

        g64 = np.asarray(im.resize((64, 64)).convert("L"), dtype=np.float32) / 255.0
        pad = 2
        gpad = np.pad(g64, ((pad, pad), (pad, pad)), mode="reflect")
        win = np.lib.stride_tricks.sliding_window_view(gpad, (2 * pad + 1, 2 * pad + 1))
        local_min = win.min(axis=(-1, -2))
        top_hat = g64 - local_min
        th_mean = float(top_hat.mean())
        th_std = float(top_hat.std())
        th_q90 = float(np.quantile(top_hat, 0.9))
        stats.extend([th_mean, th_std, th_q90])

        stats.extend(
            [
                float((top_hat > 0.05).mean()),
                float((top_hat > 0.10).mean()),
                float((top_hat > 0.20).mean()),
            ]
        )

        green_mask = (g > r) & (g > b) & (g > 0.2)
        gm = green_mask.astype(np.float32)
        gm_frac = float(gm.mean())
        stats.append(gm_frac)
        if gm_frac > 1e-4:
            stats.extend(
                [
                    float(r[green_mask].mean()),
                    float(g[green_mask].mean()),
                    float(b[green_mask].mean()),
                    float(gray[green_mask].mean()),
                    float(exg[green_mask].mean()),
                    float(exr[green_mask].mean()),
                    float(exb[green_mask].mean()),
                ]
            )
        else:
            stats.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        for ch in (L, A, Bc):
            hist, _ = np.histogram(ch.reshape(-1), bins=bin_edges, range=(0.0, 1.0))
            hist = hist.astype(np.float32)
            hist = hist / (hist.sum() + 1e-12)
            stats.extend(hist.tolist())

        if len(stats) != 138:
            raise RuntimeError(
                f"Feature length mismatch: got {len(stats)} expected 138"
            )

        feats[i, :] = np.array(stats, dtype=np.float32)

    return feats


def _oof_platt_and_full_fit_predict(
    Xtr_scaled, y, Xte_scaled, base_lr, n_splits=5, seed=0
):
    """
    Score-relevant fix (minimal, preserves core logic):
    Previously OOF decision scores were generated from models trained on fold-specific
    scaled features, while the final model used globally scaled features. That mismatch
    can shift the score distribution and degrade AUC/calibration.

    We keep the exact same "base LogisticRegression + OOF Platt scaling + full refit"
    approach, but compute OOF decision scores using the *same globally scaled Xtr* that
    is used for the final fit/predict.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    oof_score = np.zeros(Xtr_scaled.shape[0], dtype=np.float64)
    for tr_idx, va_idx in cv.split(Xtr_scaled, y):
        m = LogisticRegression(**base_lr.get_params())
        m.fit(Xtr_scaled[tr_idx], y[tr_idx])
        oof_score[va_idx] = m.decision_function(Xtr_scaled[va_idx]).astype(np.float64)

    platt = LogisticRegression(
        solver="lbfgs",
        penalty="l2",
        C=1.0,
        max_iter=2000,
        random_state=seed,
    )
    platt.fit(oof_score.reshape(-1, 1), y)

    base_full = LogisticRegression(**base_lr.get_params())
    base_full.fit(Xtr_scaled, y)
    te_score = base_full.decision_function(Xte_scaled).astype(np.float64)
    p = platt.predict_proba(te_score.reshape(-1, 1))[:, 1].astype(np.float64)
    return p


def fit_ovr_logreg_predict_proba(X_train, y_train_df, X_test, class_names):
    """
    Fits separate binary LogisticRegression models per class (OvR).
    """
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train).astype(np.float64, copy=False)
    Xte = scaler.transform(X_test).astype(np.float64, copy=False)

    preds = {}
    for c in class_names:
        y = y_train_df[c].astype(int).values

        if c == "multiple_diseases":
            cw = {0: 1.0, 1: 2.0}
            C = 2.0
            n_splits = 6
        else:
            cw = "balanced"
            C = 3.0
            n_splits = 5

        base = LogisticRegression(
            solver="liblinear",
            penalty="l2",
            max_iter=4000,
            C=C,
            class_weight=cw,
            random_state=0,
        )

        p = _oof_platt_and_full_fit_predict(
            Xtr, y, Xte, base_lr=base, n_splits=n_splits, seed=0
        )
        preds[c] = np.clip(p, 1e-8, 1 - 1e-8)

    return preds


def row_normalize_predictions(df, class_cols, eps=1e-12):
    arr = df[class_cols].to_numpy(dtype=np.float64)
    arr = np.clip(arr, eps, 1.0)
    s = arr.sum(axis=1, keepdims=True)
    arr = arr / np.maximum(s, eps)
    out = df.copy()
    for j, c in enumerate(class_cols):
        out[c] = arr[:, j]
    return out


def safe_blend_ovr_and_row_norm(df, class_cols, alpha=0.15, per_class_alpha=None):
    """
    Score-relevant minimal tweak: ROC AUC is column-wise ranking; row-normalization can
    slightly distort a column's ranking. Use a smaller alpha for rare multiple_diseases.
    """
    raw = df[class_cols].to_numpy(dtype=np.float64)
    rn = row_normalize_predictions(df, class_cols)[class_cols].to_numpy(
        dtype=np.float64
    )

    if per_class_alpha is None:
        a = np.full((1, raw.shape[1]), float(alpha), dtype=np.float64)
    else:
        a = np.array(
            [per_class_alpha.get(c, alpha) for c in class_cols], dtype=np.float64
        ).reshape(1, -1)

    blended = (1.0 - a) * raw + a * rn
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
        submissions_all,
        sub_idx=sub_idx,
        weights=weights,
        required_cols=required_cols,
        test_csv_path=TEST_CSV_PATH,
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
        submission_df,
        required_cols,
        alpha=0.15,
        per_class_alpha={"multiple_diseases": 0.05},
    )

    make_submission_file(submission_df, out_path="submission.csv")
