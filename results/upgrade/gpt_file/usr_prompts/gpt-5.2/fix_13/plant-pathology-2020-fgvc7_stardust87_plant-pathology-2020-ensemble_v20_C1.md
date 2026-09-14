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

0.9694615556143084

# 6. Current score

0.71691

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'Your code crashes because it assumes an `/kaggle/input/submissions/` directory containing at least 3 submission files, but that folder doesn’t exist in this environment, so `submissions_all` is empty. I fix this by (1) switching to a safe, existing input path, (2) making the ensembling robust to “no external submissions found” by falling back to a valid baseline submission built from `sample_submission.csv`, and (3) ensuring the output is a correctly formatted `submission.csv` with the required columns and aligned `image_id`s from `test.csv`. This run end-to-end and produce a valid submission file; score be a simple baseline (since no model code is provided here).'
- What this solution (achieved 0.51861) has done: 'Your current 0.5 score comes from writing the sample submission probabilities (uniform 0.25), which has essentially no ranking signal for ROC AUC. To move toward the 0.969 target without changing the “no training” core approach, I replace the uniform fallback with a minimal heuristic that uses the provided train label priors (class prevalences) and a tiny per-image deterministic jitter derived from `image_id` to create consistent, non-constant rankings per class (ROC AUC needs ranking, not perfect calibration). The external-submission ensembling path remains intact; only the fallback is improved. This should increase the score above 0.5 while staying lightweight and within the same end-to-end submission-writing logic.'
- What this solution (achieved 0.41667) has done: 'Your current score is low because the fallback predictions are almost constant per class (priors + tiny jitter), which gives weak ranking signal for ROC AUC. Without changing the overall “no training / heuristic fallback” core approach, I increase the deterministic per-image variation and make it class-specific using multiple independent hash-based features, then combine them with fixed weights so each class gets a stronger, more diverse ranking. I also normalize each class’s raw scores via rank-to-uniform mapping on the test set to guarantee non-constant, well-spread predictions (good for AUC, which only cares about ranking), while keeping outputs clipped to [0,1] and submission formatting identical. The external-submission ensembling path stays intact; only the fallback is strengthened to move the score upward toward your target.'
- What this solution (achieved 0.47103) has done: 'Your current score is far below the target, so we should increase it with minimal, safe changes that keep your “no training / heuristic fallback” core approach intact. The biggest issue is that the fallback produces essentially random rankings, which won’t correlate with true labels and therefore can’t yield high ROC AUC. I change the fallback to use **image-derived features** (mean/std of RGB channels) from the provided JPGs, then apply the same **rank-to-uniform** step per class to maximize AUC-relevant ranking spread; this stays lightweight and deterministic while adding real signal. The external-submission ensembling path and submission formatting/alignment remain unchanged.'
- What this solution (achieved 0.46377) has done: 'Your current gap to the target is large (0.471 → 0.969), so we need a real signal increase while keeping your “no training / heuristic fallback” approach intact. The biggest low-risk gain is to make the fallback image features more discriminative without changing the overall pipeline: use additional simple color-index features (excess red/green, chroma ratios), add lightweight edge/texture proxies from grayscale gradients, and keep the same per-class rank-to-uniform mapping (since ROC AUC is ranking-based). I also make image loading more reliable by searching for both `.jpg` and `.JPG` and slightly increase resize resolution to stabilize these features. The ensembling path and submission formatting remain unchanged; only the fallback feature extraction and scoring are strengthened to move the score upward toward the target.'
- What this solution (achieved 0.4701) has done: 'Your current score (0.46377) is far below the target (0.96946), so we should increase it with minimal changes while keeping the same “no training / heuristic from images + rank-to-uniform” core approach. The main issue is that the current features/linear rules are still too weakly correlated with the labels, so the ROC AUC ranking signal remains near-random. I strengthen the fallback by adding a few additional lightweight, deterministic image features that better capture disease cues (HSV stats for rust color, saturation/value, and a simple high-frequency texture measure via Laplacian), then fold them into the same per-class raw scoring and keep the same rank-to-uniform mapping. This preserves the pipeline structure, doesn’t add any training, and should move the score upward toward the target while still producing a valid `submission.csv`.'
- What this solution (achieved 0.49671) has done: 'We need to increase score toward 0.969 from 0.470, so we should improve the ranking signal while keeping your “no training + image heuristics + rank-to-uniform” core approach intact. The smallest high-impact fix is to correct a likely feature-scale issue: your Laplacian uses `np.roll`, which introduces wrap-around edges that add noise; replacing it with a non-wrapping Laplacian stabilizes texture signal. I also add a couple of very lightweight, disease-relevant features (yellow/brown pixel fraction from HSV, and green-deficiency fraction) and fold them into the same per-class linear raw scores, then keep the same rank-to-uniform mapping. Finally, I make the external-submission ensembling safer by aligning by `image_id` (prevents accidental row-order mismatch from harming AUC) while preserving the existing ensembling logic.'
- What this solution (achieved 0.4997) has done: 'Your current score (0.49671) is far below the target (0.96946), so we should increase AUC by improving ranking signal while preserving your existing “no training, image-heuristics + rank-to-uniform” core approach. The smallest high-impact fix is that `h_mean` (HSV hue) is circular, but your scoring uses it linearly; replacing it with hue-bin fractions (yellow/orange/brown/green) gives more stable disease-color cues without changing the pipeline structure. I also add a simple “dark spot” fraction feature to better capture scab-like lesions, and I keep the same rank-to-uniform post-processing and submission alignment/formatting. These changes only touch the fallback feature extraction and the linear raw-score formulas, keeping your ensembling path intact.'
- What this solution (achieved 0.7298) has done: 'Your score is far below the target (0.4997 vs 0.9694), so we need more real signal while keeping your “no training + image heuristics + rank-to-uniform” approach intact. The biggest low-risk improvement is to (1) fit the existing linear raw-score formulas on the training set using the same extracted features (just solving a regularized least-squares per class), then (2) apply those fitted weights to test features and keep your rank-to-uniform post-processing (AUC cares about ranking). This preserves the same feature extraction, same overall pipeline structure, and same output semantics, but replaces hand-tuned coefficients with data-driven ones using only train labels (no leakage). I also add a tiny ridge penalty and feature standardization to stabilize the fit and keep runtime within limits.'
- What this solution (achieved 0.73006) has done: 'Your current score (0.7298) is still far below the target (0.96946), so we should cautiously increase AUC by improving the supervised fit while keeping your exact pipeline (same features → ridge least squares per class → rank-to-uniform → mix with priors). The biggest low-risk issue is that the current fit is not cross-validated and can be poorly calibrated for ranking on the test set; we can improve generalization/ranking by fitting out-of-fold (OOF) raw scores on train, then using those OOF scores to learn a tiny monotone-friendly calibration (per class) that maps raw scores to probabilities without changing the model family. We keep inference identical (same linear model) but apply the learned per-class affine calibration before rank-to-uniform, which tends to stabilize ranking across folds and usually improves ROC AUC. Changes are minimal, deterministic, and stay within time limits; submission formatting and external-ensemble behavior remain unchanged.'
- What this solution (achieved 0.73044) has done: 'We need to increase score toward 0.969 from 0.730 (higher is better), so we should strengthen generalization/ranking while keeping your same pipeline: same handcrafted image features → ridge least-squares per class → rank-to-uniform → mix with priors. The smallest high-impact fix is to make the OOF calibration and model fitting less sensitive to the arbitrary “no shuffle” fold assignment by switching to a deterministic stratified fold split based on the dominant class label; this keeps the same training approach but yields more reliable OOF calibration. Additionally, we select the ridge strength from a tiny fixed grid using OOF ROC AUC (still ridge least squares; just choosing λ), then refit on all data with that λ—this is a minimal, metric-aligned improvement that should move AUC upward. Submission formatting, image_id alignment, external ensembling behavior, and the feature set remain unchanged.'
- What this solution (achieved 0.71691) has done: 'Your current score (0.73044) is far below the target (0.96946), so we should increase it with the smallest changes that keep your exact pipeline: same handcrafted image features → ridge least squares per class → OOF calibration → rank-to-uniform → prior mixing. The most likely bottleneck is that the linear ridge fit is underpowered due to the very small lambda grid and the unweighted regression despite strong class imbalance (especially `multiple_diseases`). I (1) expand the fixed lambda grid slightly (still ridge least squares; just selecting λ more reliably via OOF AUC) and (2) add per-sample weights inversely proportional to the dominant-class frequency inside the *same* normal equations, which usually improves ranking for minority classes without changing model family. Everything else (paths, feature extraction, folds, post-processing, and submission formatting) remains unchanged, and it still write a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import hashlib
import pandas as pd



## === cell 1
SUBMISSIONS_PATH = "/kaggle/input"

DATA_DIR_CANDIDATES = [
    "/kaggle/input/plant-pathology-2020-fgvc7",
    "/kaggle/data/plant-pathology-2020-fgvc7",
    "/kaggle/input",
    "/kaggle/data",
]


def _find_file(filename, roots):
    for r in roots:
        p = os.path.join(r, filename)
        if os.path.exists(p):
            return p
    for r in roots:
        if os.path.exists(r):
            for dirpath, _, filenames in os.walk(r):
                if filename in filenames:
                    return os.path.join(dirpath, filename)
    return None


sample_path = _find_file("sample_submission.csv", DATA_DIR_CANDIDATES)
test_path = _find_file("test.csv", DATA_DIR_CANDIDATES)
train_path = _find_file("train.csv", DATA_DIR_CANDIDATES)

if sample_path is None or test_path is None:
    raise FileNotFoundError(
        f"Could not locate required files. sample_path={sample_path}, test_path={test_path}"
    )

print("Using sample_submission:", sample_path)
print("Using test.csv:", test_path)
print("Using train.csv:", train_path)



## === cell 2
submissions_all = []
for dirname, _, filenames in os.walk(SUBMISSIONS_PATH):
    for filename in filenames:
        if not filename.lower().endswith(".csv"):
            continue
        full = os.path.join(dirname, filename)
        base = os.path.basename(full).lower()
        if base in {"sample_submission.csv", "train.csv", "test.csv"}:
            continue
        submissions_all.append(full)

submissions_all.sort()
print("Found candidate external submissions:", len(submissions_all))
print(submissions_all[:20])




## === cell 3
def ensemble(submissions_all, sub_idx, weights=None, test_csv_path=None):
    """
    Weighted sum ensemble of provided submission files.

    Change (score improvement, minimal & safe): align each external submission by image_id
    to the test.csv order. This avoids accidental row-order mismatches that can destroy AUC.
    Core ensembling logic (weighted sum) is unchanged.
    """
    if weights is None:
        weights = [1.0] * len(sub_idx)
    if len(sub_idx) != len(weights):
        raise ValueError(
            f"sub_idx and weights must have same length. Got {len(sub_idx)} vs {len(weights)}"
        )

    needed = ["healthy", "multiple_diseases", "rust", "scab"]

    test_ids = None
    if test_csv_path is not None:
        test_df = pd.read_csv(test_csv_path)
        test_ids = test_df["image_id"].astype(str).values

    submission_with_weight = []
    for i in range(len(sub_idx)):
        idx = sub_idx[i]
        if idx < 0 or idx >= len(submissions_all):
            raise IndexError(
                f"Requested submission index {idx} but only {len(submissions_all)} files available."
            )
        path = submissions_all[idx]
        w = float(weights[i])
        print(f"I'm taking submission {path} with weight {w}")

        df = pd.read_csv(path)
        missing = [c for c in (["image_id"] + needed) if c not in df.columns]
        if missing:
            raise ValueError(f"Submission file {path} is missing columns: {missing}")

        if test_ids is not None:
            df = df.copy()
            df["image_id"] = df["image_id"].astype(str)
            df = df.set_index("image_id").reindex(test_ids).reset_index()
            if df[needed].isna().any().any():
                raise ValueError(
                    f"Submission file {path} has missing predictions after reindexing to test.csv order."
                )

        arr = df.loc[:, needed].to_numpy(dtype="float64")
        submission_with_weight.append(arr * w)

    submission_sum = submission_with_weight[0]
    for arr in submission_with_weight[1:]:
        submission_sum = submission_sum + arr
    return submission_sum




## === cell 4
def make_submission_file(
    submission_avg, sample_submission_path, test_csv_path, out_path="submission.csv"
):
    """
    Fix: ensure correct columns/order and that image_id matches test.csv.
    Also clip predictions to [0,1] to be safe.
    """
    sample_df = pd.read_csv(sample_submission_path)
    test_df = pd.read_csv(test_csv_path)

    out = sample_df.copy()
    out = out.set_index("image_id")
    out = out.reindex(test_df["image_id"].values)
    out = out.reset_index()

    needed = ["healthy", "multiple_diseases", "rust", "scab"]
    if submission_avg is None:
        raise ValueError("submission_avg cannot be None")

    pred = pd.DataFrame(submission_avg, columns=needed)
    pred = pred.clip(0.0, 1.0)

    if len(pred) != len(out):
        raise ValueError(
            f"Prediction rows ({len(pred)}) do not match test rows ({len(out)})"
        )

    out[needed] = pred[needed].values
    out.to_csv(out_path, index=False)
    print(f"Wrote {out_path} with shape {out.shape} and columns {list(out.columns)}")




## === cell 5
def _id_to_u01_salt(image_id: str, salt: str) -> float:
    """
    Deterministic pseudo-random in [0,1) from image_id + salt (stable across runs).
    """
    h = hashlib.md5((salt + "::" + image_id).encode("utf-8")).hexdigest()
    return int(h[:8], 16) / float(16**8)


def _rank_to_uniform(x):
    """
    Map scores to (0,1) via ranks so each class has a full spread; AUC depends on ordering.
    """
    s = pd.Series(x)
    r = s.rank(method="average").to_numpy(dtype="float64")
    n = float(len(r))
    return r / (n + 1.0)


def _find_images_dir(roots):
    candidates = []
    for r in roots:
        p = os.path.join(r, "images")
        if os.path.isdir(p):
            candidates.append(p)
    if candidates:
        return candidates[0]

    for r in roots:
        if os.path.exists(r):
            for dirpath, dirnames, _ in os.walk(r):
                if "images" in dirnames:
                    return os.path.join(dirpath, "images")
    return None


def _resolve_image_path(images_dir, image_id):
    p1 = os.path.join(images_dir, f"{image_id}.jpg")
    if os.path.exists(p1):
        return p1
    p2 = os.path.join(images_dir, f"{image_id}.JPG")
    if os.path.exists(p2):
        return p2
    return None


def _image_features(image_path):
    """
    Uses lightweight deterministic image features (same as before).
    """
    try:
        from PIL import Image
    except Exception:
        return None

    try:
        import numpy as np

        with Image.open(image_path) as im:
            im = im.convert("RGB")
            im = im.resize((192, 192))
            arr = np.asarray(im, dtype="float32") / 255.0  # (H,W,3)
    except Exception:
        return None

    try:
        r = arr[:, :, 0]
        g = arr[:, :, 1]
        b = arr[:, :, 2]

        r_mean = float(r.mean())
        g_mean = float(g.mean())
        b_mean = float(b.mean())
        r_std = float(r.std())
        g_std = float(g.std())
        b_std = float(b.std())

        eps = 1e-6
        sum_rgb = r + g + b + eps
        r_frac = float((r / sum_rgb).mean())
        g_frac = float((g / sum_rgb).mean())
        b_frac = float((b / sum_rgb).mean())

        exg = float((2.0 * g - r - b).mean())
        exr = float((1.4 * r - g).mean())
        rg = float(((r + eps) / (g + eps)).mean())
        gb = float(((g + eps) / (b + eps)).mean())
        rb = float(((r + eps) / (b + eps)).mean())

        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        dx = gray[:, 1:] - gray[:, :-1]
        dy = gray[1:, :] - gray[:-1, :]
        edge = float(
            (__import__("numpy").abs(dx).mean() + __import__("numpy").abs(dy).mean())
            / 2.0
        )

        brightness = float(gray.mean())
        contrast = float(gray.std())

        arr_u8 = (arr * 255.0).clip(0, 255).astype("uint8")
        hsv = (
            __import__("numpy").asarray(
                Image.fromarray(arr_u8, mode="RGB").convert("HSV"), dtype="float32"
            )
            / 255.0
        )
        h = hsv[:, :, 0]  # [0,1) circular
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]
        s_mean = float(s.mean())
        v_mean = float(v.mean())
        s_std = float(s.std())
        v_std = float(v.std())

        center = gray[1:-1, 1:-1]
        lap_valid = (
            -4.0 * center
            + gray[:-2, 1:-1]
            + gray[2:, 1:-1]
            + gray[1:-1, :-2]
            + gray[1:-1, 2:]
        )
        lap_energy = float(__import__("numpy").mean(__import__("numpy").abs(lap_valid)))

        yellow_frac = float(((h > 0.08) & (h < 0.22) & (s > 0.25) & (v > 0.25)).mean())
        green_def_frac = float(((g < 0.35) & (r > 0.35) & (s > 0.20)).mean())

        orange_frac = float(((h >= 0.03) & (h < 0.10) & (s > 0.25) & (v > 0.20)).mean())
        brown_frac = float(((h >= 0.03) & (h < 0.14) & (s > 0.20) & (v < 0.55)).mean())
        green_frac_hsv = float(
            ((h >= 0.22) & (h < 0.45) & (s > 0.15) & (v > 0.15)).mean()
        )

        dark_spot_frac = float(((v < 0.35) & (s > 0.15)).mean())

        return (
            r_mean,
            g_mean,
            b_mean,
            r_std,
            g_std,
            b_std,
            r_frac,
            g_frac,
            b_frac,
            exg,
            exr,
            rg,
            gb,
            rb,
            edge,
            brightness,
            contrast,
            s_mean,
            v_mean,
            s_std,
            v_std,
            lap_energy,
            yellow_frac,
            green_def_frac,
            orange_frac,
            brown_frac,
            green_frac_hsv,
            dark_spot_frac,
        )
    except Exception:
        return None


def fallback_predictions_from_priors_and_images(
    train_csv_path, test_csv_path, data_roots
):
    """
    Change (score improvement, minimal & safe):
    - Keep the exact same feature set and the same ridge least-squares model family per class.
    - Keep deterministic stratified folds and OOF affine calibration.
    - Improve generalization for mean column-wise ROC AUC by:
        (1) slightly expanding the fixed ridge lambda grid (still selecting λ via OOF AUC),
        (2) adding per-sample weights inverse to dominant-class frequency in the same normal equations.
      This helps the minority classes (esp. multiple_diseases) influence the fit, often improving AUC.
    - Keep the same rank-to-uniform + prior mixing and identical submission formatting.
    """
    import numpy as np

    needed = ["healthy", "multiple_diseases", "rust", "scab"]

    test_df = pd.read_csv(test_csv_path)
    image_ids_test = test_df["image_id"].astype(str).tolist()

    if train_csv_path is not None and os.path.exists(train_csv_path):
        train_df = pd.read_csv(train_csv_path)
        priors = train_df[needed].mean().to_dict()
        image_ids_train = train_df["image_id"].astype(str).tolist()
        y_train = train_df[needed].to_numpy(dtype="float64")
    else:
        priors = {c: 0.25 for c in needed}
        image_ids_train = []
        y_train = None

    images_dir = _find_images_dir(data_roots)
    print("Using images_dir:", images_dir)

    def _featurize_ids(image_ids):
        X = np.zeros((len(image_ids), 28 + 3), dtype="float64")
        for i, image_id in enumerate(image_ids):
            u0 = _id_to_u01_salt(image_id, "pp2020_s0")
            u1 = _id_to_u01_salt(image_id, "pp2020_s1")
            u2 = _id_to_u01_salt(image_id, "pp2020_s2")
            z0, z1, z2 = (u0 - 0.5, u1 - 0.5, u2 - 0.5)

            feats28 = (0.0,) * 28
            if images_dir is not None:
                img_path = _resolve_image_path(images_dir, image_id)
                if img_path is not None:
                    f = _image_features(img_path)
                    if f is not None:
                        feats28 = f

            X[i, :28] = np.asarray(feats28, dtype="float64")
            X[i, 28:] = np.asarray([z0, z1, z2], dtype="float64")
        return X

    X_test = _featurize_ids(image_ids_test)

    if y_train is None or len(image_ids_train) == 0:
        raw = {}
        for j, c in enumerate(needed):
            raw[c] = X_test[:, j % X_test.shape[1]]
        preds = []
        for c in needed:
            u = _rank_to_uniform(raw[c])
            p = 0.92 * u + 0.08 * float(priors[c])
            preds.append(p)
        pred = pd.DataFrame({c: preds[i] for i, c in enumerate(needed)}).clip(0.0, 1.0)
        return pred.to_numpy(dtype="float64")

    X_train = _featurize_ids(image_ids_train)

    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0)
    sigma[sigma < 1e-6] = 1.0
    Xtr = (X_train - mu) / sigma
    Xte = (X_test - mu) / sigma

    Xtr_i = np.concatenate([np.ones((Xtr.shape[0], 1), dtype="float64"), Xtr], axis=1)
    Xte_i = np.concatenate([np.ones((Xte.shape[0], 1), dtype="float64"), Xte], axis=1)

    n = Xtr_i.shape[0]
    dom = np.argmax(y_train, axis=1).astype(int)

    counts = np.bincount(dom, minlength=y_train.shape[1]).astype("float64")
    counts[counts < 1.0] = 1.0
    inv = 1.0 / counts
    w = inv[dom]
    w = w / float(w.mean())  # normalize so average weight is 1
    sqrt_w = np.sqrt(w).reshape(-1, 1)

    class_bins = {k: [] for k in range(y_train.shape[1])}
    for i, (cid, imgid) in enumerate(zip(dom.tolist(), image_ids_train)):
        class_bins[cid].append((i, _id_to_u01_salt(str(imgid), "fold_seed_v1")))
    for k in class_bins:
        class_bins[k].sort(key=lambda t: t[1])

    fold = np.zeros((n,), dtype=int)
    K = 5
    for k in class_bins:
        idxs = [t[0] for t in class_bins[k]]
        for j, ii in enumerate(idxs):
            fold[ii] = j % K

    try:
        from sklearn.metrics import roc_auc_score
    except Exception:
        roc_auc_score = None

    lam_grid = [0.03, 0.1, 0.3, 1.0, 3.0]
    best_lam = lam_grid[0]
    best_score = -1e18

    p_dim = Xtr_i.shape[1]
    I = np.eye(p_dim, dtype="float64")
    I[0, 0] = 0.0  # do not regularize intercept

    def _fit_ridge_weighted(X, y, reg, sqrt_w_local):
        Xw = X * sqrt_w_local
        yw = y * sqrt_w_local[:, 0]
        A = Xw.T @ Xw + reg
        bvec = Xw.T @ yw
        return np.linalg.solve(A, bvec)

    for lam in lam_grid:
        reg = lam * I
        raw_oof = np.zeros((n, y_train.shape[1]), dtype="float64")

        for f in range(K):
            tr_mask = fold != f
            va_mask = ~tr_mask

            Xtr_f = Xtr_i[tr_mask]
            ytr_f = y_train[tr_mask]
            Xva_f = Xtr_i[va_mask]

            sqrt_w_f = sqrt_w[tr_mask]
            W_f = np.zeros((p_dim, y_train.shape[1]), dtype="float64")
            for kk in range(y_train.shape[1]):
                W_f[:, kk] = _fit_ridge_weighted(Xtr_f, ytr_f[:, kk], reg, sqrt_w_f)

            raw_oof[va_mask] = Xva_f @ W_f

        if roc_auc_score is not None:
            aucs = []
            for kk in range(y_train.shape[1]):
                try:
                    aucs.append(float(roc_auc_score(y_train[:, kk], raw_oof[:, kk])))
                except Exception:
                    pass
            score = float(np.mean(aucs)) if len(aucs) else -1e18
        else:
            score = float(np.mean(np.var(raw_oof, axis=0)))

        print(f"OOF mean AUC proxy for lam={lam}: {score:.6f}")
        if score > best_score:
            best_score = score
            best_lam = lam

    print("Selected ridge lam:", best_lam)

    reg = best_lam * I

    W = np.zeros((p_dim, y_train.shape[1]), dtype="float64")
    for k in range(y_train.shape[1]):
        W[:, k] = _fit_ridge_weighted(Xtr_i, y_train[:, k], reg, sqrt_w)

    raw_test = Xte_i @ W  # (n_test, 4)

    raw_oof = np.zeros((n, y_train.shape[1]), dtype="float64")
    for f in range(K):
        tr_mask = fold != f
        va_mask = ~tr_mask
        Xtr_f = Xtr_i[tr_mask]
        ytr_f = y_train[tr_mask]
        Xva_f = Xtr_i[va_mask]
        sqrt_w_f = sqrt_w[tr_mask]

        W_f = np.zeros((p_dim, y_train.shape[1]), dtype="float64")
        for kk in range(y_train.shape[1]):
            W_f[:, kk] = _fit_ridge_weighted(Xtr_f, ytr_f[:, kk], reg, sqrt_w_f)
        raw_oof[va_mask] = Xva_f @ W_f

    cal_a = np.ones((y_train.shape[1],), dtype="float64")
    cal_b = np.zeros((y_train.shape[1],), dtype="float64")
    for k in range(y_train.shape[1]):
        x = raw_oof[:, k]
        y = y_train[:, k]
        x_mean = float(x.mean())
        y_mean = float(y.mean())
        x_var = float(((x - x_mean) ** 2).mean())
        if x_var < 1e-12:
            a = 1.0
        else:
            a = float(((x - x_mean) * (y - y_mean)).mean() / x_var)
        b0 = y_mean - a * x_mean
        cal_a[k] = a
        cal_b[k] = b0

    raw_test_cal = raw_test * cal_a.reshape(1, -1) + cal_b.reshape(1, -1)

    preds = []
    for k, c in enumerate(needed):
        u = _rank_to_uniform(raw_test_cal[:, k])
        p = 0.92 * u + 0.08 * float(priors[c])
        preds.append(p)

    pred = pd.DataFrame({c: preds[i] for i, c in enumerate(needed)}).clip(0.0, 1.0)
    return pred.to_numpy(dtype="float64")




## === cell 6
if len(submissions_all) >= 3:
    submission_avg = ensemble(
        submissions_all, [0, 1, 2], [0.04, 0.2, 0.04], test_csv_path=test_path
    )
else:
    print(
        "No (or insufficient) external submission files found to ensemble; using image-feature + rank-based fallback to improve ROC AUC signal."
    )
    submission_avg = fallback_predictions_from_priors_and_images(
        train_path, test_path, DATA_DIR_CANDIDATES
    )

make_submission_file(submission_avg, sample_path, test_path, out_path="submission.csv")
