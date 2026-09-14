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

0.9700584708763096

# 6. Current score

0.68664

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'I fix the runtime error by making the script robust to the fact that `/kaggle/input/submissions/` does not exist (or is empty) in your environment, which currently makes `submissions_all` empty and triggers an IndexError. To preserve your core “weighted ensemble of CSV submissions” logic, I (1) auto-discover candidate submission CSVs from the provided dataset folders, and (2) if fewer than 3 are found, fall back to a valid baseline submission (uniform probabilities) using `sample_submission.csv` so a `.csv` submission is always produced. I also add minimal safety checks (matching lengths, column presence, and weight normalization) to prevent silent misalignment and ensure valid probabilities. This is primarily a correctness/stability fix; without existing external submissions, score tuning isn’t possible here, but you get a valid `submission.csv` to submit.'
- What this solution (achieved 0.66726) has done: 'Your current 0.5 score is coming from the uniform-probability fallback, which has no information from the images. To move toward the 0.970 target without changing your core “produce a submission.csv” semantics, I keep your ensemble logic intact but add a minimal, deterministic image-based baseline that runs only when no candidate submission CSVs are found. This baseline uses simple color/statistics features from each JPG and a one-vs-rest Logistic Regression (scikit-learn) trained on the provided train labels, then predicts probabilities for test; it’s lightweight, stays within the installed packages, and should materially improve AUC versus 0.25-uniform. The output is still a valid `submission.csv` with the required columns and correct `image_id` alignment.'
- What this solution (achieved 0.59342) has done: 'Your current score (0.66726) is far below the target (0.97006), so we should improve the baseline without changing the overall approach (simple image features + OneVsRest Logistic Regression). The smallest likely win is to make the features more informative while keeping the same model and training flow: resize images to a fixed small size (stabilizes moments/histograms), add a few simple texture/shape cues (edges + low-res pooled grayscale), and use `class_weight="balanced"` to reduce the impact of label imbalance on one-vs-rest classifiers. I also ensure the submission rows exactly match `test.csv` order (avoid any accidental reordering from merges), which can silently hurt the score even if predictions are good. These changes keep the core logic intact (still deterministic handcrafted features + LogisticRegression) but should move AUC meaningfully upward toward your target.'
- What this solution (achieved 0.66767) has done: 'Your current score (0.593) is far below the 0.970 target, so we need a real lift while preserving your core approach (handcrafted image features + OneVsRest LogisticRegression). The smallest high-impact change is to add a few more informative but still “simple stats” features (HSV stats/histograms + a slightly richer low-res grayscale pooling) and to mildly improve robustness by using `C=2.0` and `n_jobs=-1` for the OvR wrapper without changing the model family or training flow. I also fix a subtle pitfall: `density=True` histograms can be unstable across images; switching to normalized counts keeps features comparable and often improves AUC without changing semantics. All paths/output stay the same and it still always writes a valid `submission.csv`.'
- What this solution (achieved 0.67345) has done: 'We need to move your score up toward 0.970 (current 0.66767 is far below target), but keep the same core approach (handcrafted image features + OneVsRest LogisticRegression). The smallest likely gain without changing the model family/training loop is to (1) make the grayscale pooled features consistent (your 12×12 pool is correct, but the “pooled_8” reshape currently produces 8×12=96 dims, not 8×8, which is likely unintended noise), and (2) add a tiny set of rotation/flip-invariant aggregations by extracting the same features on a few deterministic transforms and averaging them at feature level (still the same feature extractor, just more robust). I also ensure the training labels are read as int consistently and keep the exact submission alignment to `test.csv` order. These are minimal, deterministic changes aimed at improving AUC while preserving your core logic and producing `submission.csv` end-to-end.'
- What this solution (achieved 0.67552) has done: 'Your current score (0.67345) is far below the target (0.97006), so we should increase performance while keeping the same core approach (handcrafted image features + OneVsRest LogisticRegression, same training flow). The smallest high-impact fix is a bug in the 8×8 pooling: `gray.reshape(8, 12, 8, 12)` does not yield an 8×8 grid (it yields 8×8 but averages over the wrong pixel blocks); we correct it to `gray.reshape(8, 12, 8, 12).mean(axis=(1,3))` *with the correct axis meaning* by instead reshaping as `(8, 12, 8, 12)` but swapping to produce true 8×8 block means via `(8,12,8,12)->(8,8)` is ambiguous—so we implement a clear, correct block pooling using a deterministic `block_reduce`-style reshape for both 12×12 and 8×8. Then, without changing model family, we add tiny ridge-stability by setting `C=1.0` (less overfit) and increasing `max_iter` slightly to ensure convergence consistency, which typically improves mean ROC AUC on this dataset. All I/O and submission formatting remain unchanged and the script still always writes `submission.csv`.'
- What this solution (achieved 0.68664) has done: 'I remove the expensive recursive filesystem scan for “candidate submissions” (it can traverse the entire input tree and read many CSV headers), because that is the most likely source of the 10‑minute timeout. Instead, I only look in the intended `/kaggle/input/submissions/` folder (fast) and, if none are found, immediately run the baseline image model as before. I also speed up feature building by preallocating the feature matrix, avoiding Python list filtering/stacking overhead, and caching deterministic per-image features so each image is decoded once. These changes preserve the exact model, features, and training semantics, only removing unnecessary work and redundant computation.'
- What this solution (achieved 0.68664) has done: 'Your current score (0.68664) is far below the target (0.97006), so we should improve the baseline while keeping the same core logic (handcrafted image features + OneVsRest LogisticRegression, same fit/predict flow). The most direct minimal gain is to align training with the metric by calibrating the decision thresholding via using out-of-fold predicted probabilities for a simple per-class Platt-style re-calibration (logistic on the model’s own logits), which usually improves ROC AUC without changing the base model family. I also add a deterministic stratified split for calibration (no early stopping, no sampling) and keep test predictions aligned to `test.csv` order. If external submission CSVs exist, your original ensemble path is unchanged.'
- What this solution (achieved 0.68664) has done: 'I fix the runtime error coming from `np.char.add` by replacing the proxy-label construction with a simple, stable string join that works across NumPy/Pandas versions. I also add a small safety fallback in case the stratification proxy degenerates (too many rare groups) by switching to a plain `KFold` only in that edge case, keeping your training approach and model unchanged. Finally, I keep the submission formatting/alignment exactly tied to `test.csv` order so the produced `submission.csv` is always valid and upload-ready.'
- What this solution (achieved 0.68664) has done: 'I keep your current “handcrafted image features + OneVsRest LogisticRegression + OOF Platt calibration” core logic unchanged, but fix one high-impact alignment issue: in the ensemble path you never reindex submissions to the `test.csv` order, which can silently scramble rows and hurt ROC AUC. I make `ensemble()` build a weighted average after aligning every candidate submission to the exact `test.csv` `image_id` order, and I also make `make_submission_file()` template from `sample_submission.csv` reindexed to `test.csv` to guarantee correct ordering. This is a minimal change that directly targets evaluation correctness (no metric/learning changes) and should move the score upward toward your target whenever you do use submissions, while keeping the baseline model path identical. The script still always write a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import glob
import pandas as pd
import numpy as np



## === cell 1
SUBMISSIONS_PATH = "/kaggle/input/submissions/"

DATA_ROOTS = [
    "/kaggle/input/plant-pathology-2020-fgvc7",
    "/kaggle/data/plant-pathology-2020-fgvc7",
    "/kaggle/input",
    "/kaggle/data",
]

TARGET_COLS = ["healthy", "multiple_diseases", "rust", "scab"]




## === cell 2
def _find_candidate_submission_csvs():
    candidates = []
    if os.path.isdir(SUBMISSIONS_PATH):
        for dirname, _, filenames in os.walk(SUBMISSIONS_PATH):
            for filename in filenames:
                if filename.lower().endswith(".csv"):
                    candidates.append(os.path.join(dirname, filename))

    candidates = sorted(set(candidates))

    filtered = []
    for p in candidates:
        base = os.path.basename(p).lower()
        if base in ("train.csv", "test.csv", "sample_submission.csv"):
            continue
        try:
            df_head = pd.read_csv(p, nrows=5)
        except Exception:
            continue
        cols = set(df_head.columns)
        if "image_id" in cols and all(c in cols for c in TARGET_COLS):
            filtered.append(p)

    filtered.sort()
    return filtered


submissions_all = _find_candidate_submission_csvs()
print("Discovered candidate submission CSVs:")
print(submissions_all if submissions_all else "(none found)")




## === cell 3
def _resolve_competition_root():
    roots = [
        "/kaggle/input/plant-pathology-2020-fgvc7",
        "/kaggle/data/plant-pathology-2020-fgvc7",
        "/kaggle/input",
        "/kaggle/data",
    ]
    for r in roots:
        if os.path.exists(os.path.join(r, "train.csv")) and os.path.exists(
            os.path.join(r, "test.csv")
        ):
            return r
    raise FileNotFoundError(
        "Could not find a root containing train.csv and test.csv in known locations."
    )


def _get_test_order_image_ids():
    root = _resolve_competition_root()
    test_path = os.path.join(root, "test.csv")
    test_df = pd.read_csv(test_path, usecols=["image_id"])
    return test_df["image_id"].astype(str).tolist()


def _align_submission_df_to_test(
    submission_df, test_image_ids, path_for_error="(submission)"
):
    if "image_id" not in submission_df.columns:
        raise KeyError(f"{path_for_error} missing required column: image_id")
    for c in TARGET_COLS:
        if c not in submission_df.columns:
            raise KeyError(f"{path_for_error} missing required column: {c}")

    s = submission_df.copy()
    s["image_id"] = s["image_id"].astype(str)

    aligned = s.set_index("image_id").reindex(test_image_ids)

    if aligned.isna().any().any():
        missing = aligned.index[aligned[TARGET_COLS].isna().any(axis=1)]
        if len(missing) > 0:
            raise ValueError(
                f"{path_for_error} does not cover all test image_ids or has NaNs after alignment. "
                f"Example missing ids: {missing[:5].tolist()}"
            )

    return aligned[TARGET_COLS].to_numpy(dtype=float)


def ensemble(submissions_all, sub_idx, weights=[]):
    if len(sub_idx) == 0:
        raise ValueError("sub_idx is empty; cannot ensemble 0 submissions.")
    if len(weights) != len(sub_idx):
        raise ValueError(
            f"weights length ({len(weights)}) must equal sub_idx length ({len(sub_idx)})."
        )
    if len(submissions_all) == 0:
        raise ValueError("No submission files found to ensemble.")

    max_idx = max(sub_idx)
    if max_idx >= len(submissions_all):
        raise IndexError(
            f"Requested submission index {max_idx}, but only {len(submissions_all)} files found."
        )

    weights = np.array(weights, dtype=float)
    if not np.isfinite(weights).all():
        raise ValueError("weights contain non-finite values.")
    if weights.sum() == 0:
        raise ValueError("Sum of weights is 0; cannot normalize.")
    weights = weights / weights.sum()

    test_image_ids = _get_test_order_image_ids()

    submission_with_weight = []
    for i in range(len(sub_idx)):
        path = submissions_all[sub_idx[i]]
        print(f"I'm taking submission {path} with weight {weights[i]:.6f}")
        submission = pd.read_csv(path)

        arr = _align_submission_df_to_test(
            submission, test_image_ids, path_for_error=path
        )
        submission_with_weight.append(arr * weights[i])

    submission_avg = np.sum(submission_with_weight, axis=0)
    submission_avg = np.clip(submission_avg, 0.0, 1.0)
    return submission_avg




## === cell 4
def make_submission_file(submission_avg, submissions_all, out_path="submission.csv"):
    sample_paths = [
        "/kaggle/input/plant-pathology-2020-fgvc7/sample_submission.csv",
        "/kaggle/data/plant-pathology-2020-fgvc7/sample_submission.csv",
        "/kaggle/input/sample_submission.csv",
        "/kaggle/data/sample_submission.csv",
    ]
    sample_path = None
    for p in sample_paths:
        if os.path.exists(p):
            sample_path = p
            break

    if sample_path is None:
        if len(submissions_all) == 0:
            raise FileNotFoundError(
                "Could not locate sample_submission.csv and no submissions_all available."
            )
        sample_path = submissions_all[0]

    submission_df = pd.read_csv(sample_path)

    test_image_ids = _get_test_order_image_ids()
    if "image_id" in submission_df.columns:
        submission_df["image_id"] = submission_df["image_id"].astype(str)
        submission_df = (
            submission_df.set_index("image_id").reindex(test_image_ids).reset_index()
        )
    else:
        submission_df.insert(0, "image_id", test_image_ids)

    if submission_avg.shape[0] != len(submission_df):
        raise ValueError(
            f"Row count mismatch: submission_avg has {submission_avg.shape[0]} rows "
            f"but template has {len(submission_df)} rows."
        )
    if submission_avg.shape[1] != len(TARGET_COLS):
        raise ValueError(
            f"Column count mismatch: submission_avg has {submission_avg.shape[1]} cols but expected {len(TARGET_COLS)}."
        )

    for c in TARGET_COLS:
        if c not in submission_df.columns:
            submission_df[c] = 0.0
    submission_df[TARGET_COLS] = submission_avg

    submission_df.to_csv(out_path, index=False)
    print(f"Wrote submission to: {out_path}")
    print(submission_df.head())




## === cell 5
def _image_path(root, image_id):
    candidates = [
        os.path.join(root, "images", f"{image_id}.jpg"),
        os.path.join(root, "plant-pathology-2020-fgvc7", "images", f"{image_id}.jpg"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def _resize_rgb(img_rgb, size=(96, 96)):
    try:
        import cv2  # type: ignore

        return cv2.resize(img_rgb, size, interpolation=cv2.INTER_AREA)
    except Exception:
        from PIL import Image

        return np.array(Image.fromarray(img_rgb).resize(size, resample=Image.BILINEAR))


def _rgb_to_hsv01(x_rgb01):
    r = x_rgb01[..., 0]
    g = x_rgb01[..., 1]
    b = x_rgb01[..., 2]
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin

    h = np.zeros_like(cmax, dtype=np.float32)
    s = np.zeros_like(cmax, dtype=np.float32)
    v = cmax.astype(np.float32)

    mask = delta > 1e-8
    s[mask] = (delta[mask] / (cmax[mask] + 1e-12)).astype(np.float32)

    mask_r = mask & (cmax == r)
    mask_g = mask & (cmax == g)
    mask_b = mask & (cmax == b)

    h[mask_r] = ((g[mask_r] - b[mask_r]) / (delta[mask_r] + 1e-12)) % 6.0
    h[mask_g] = ((b[mask_g] - r[mask_g]) / (delta[mask_g] + 1e-12)) + 2.0
    h[mask_b] = ((r[mask_b] - g[mask_b]) / (delta[mask_b] + 1e-12)) + 4.0
    h = (h / 6.0).astype(np.float32)
    return np.stack([h, s, v], axis=-1)


def _block_pool_2d(x2d, out_h, out_w):
    h, w = x2d.shape
    if h % out_h != 0 or w % out_w != 0:
        raise ValueError(f"Cannot block-pool {h}x{w} to {out_h}x{out_w}.")
    bh = h // out_h
    bw = w // out_w
    return x2d.reshape(out_h, bh, out_w, bw).mean(axis=(1, 3))


def _extract_simple_features_single(img):
    img = _resize_rgb(img, size=(96, 96))

    x = img.astype(np.float32) / 255.0
    feats = []

    feats.extend(x.mean(axis=(0, 1)).tolist())
    feats.extend(x.std(axis=(0, 1)).tolist())

    gray = (0.2989 * x[..., 0] + 0.5870 * x[..., 1] + 0.1140 * x[..., 2]).astype(
        np.float32
    )
    gmean = float(gray.mean())
    gstd = float(gray.std())
    feats.append(gmean)
    feats.append(gstd)

    z = (gray - gmean) / (gstd + 1e-6)
    feats.append(float(np.mean(z**3)))
    feats.append(float(np.mean(z**4)))

    bins = 16
    for c in range(3):
        hist, _ = np.histogram(x[..., c], bins=bins, range=(0.0, 1.0), density=False)
        hist = hist.astype(np.float32)
        hist = hist / (hist.sum() + 1e-12)
        hist = np.log1p(hist)
        feats.extend(hist.tolist())

    hsv = _rgb_to_hsv01(x)
    feats.extend(hsv.mean(axis=(0, 1)).tolist())
    feats.extend(hsv.std(axis=(0, 1)).tolist())
    for c in range(3):
        hist, _ = np.histogram(hsv[..., c], bins=bins, range=(0.0, 1.0), density=False)
        hist = hist.astype(np.float32)
        hist = hist / (hist.sum() + 1e-12)
        hist = np.log1p(hist)
        feats.extend(hist.tolist())

    feats.append(float((x[..., 1] - x[..., 0]).mean()))
    feats.append(float((x[..., 0] - x[..., 1]).mean()))
    feats.append(float((x[..., 1] - x[..., 2]).mean()))
    feats.append(float((x[..., 2] - x[..., 1]).mean()))

    gx = np.abs(gray[:, 1:] - gray[:, :-1])
    gy = np.abs(gray[1:, :] - gray[:-1, :])
    edge = (gx.mean() + gy.mean()) / 2.0
    feats.append(float(edge))
    feats.append(float(gx.std()))
    feats.append(float(gy.std()))

    pooled_12 = _block_pool_2d(gray, 12, 12)
    feats.extend(np.log1p(pooled_12.astype(np.float32)).ravel().tolist())

    pooled_8 = _block_pool_2d(gray, 8, 8)
    feats.extend(np.log1p(pooled_8.astype(np.float32)).ravel().tolist())

    return np.array(feats, dtype=np.float32)


def _extract_simple_features(img):
    img0 = img
    img1 = img[:, ::-1, :]
    img2 = img[::-1, :, :]
    img3 = np.rot90(img, k=1)

    f0 = _extract_simple_features_single(img0)
    f1 = _extract_simple_features_single(img1)
    f2 = _extract_simple_features_single(img2)
    f3 = _extract_simple_features_single(img3)

    return ((f0 + f1 + f2 + f3) / 4.0).astype(np.float32)


def _build_feature_matrix(image_ids, root):
    try:
        import cv2  # type: ignore

        use_cv2 = True
    except Exception:
        use_cv2 = False
        from PIL import Image  # type: ignore

    n = len(image_ids)
    X2 = None
    missing_mask = np.zeros(n, dtype=bool)

    cache = {}

    def _read_rgb(p):
        if use_cv2:
            img = cv2.imread(p)  # BGR uint8
            if img is None:
                return None
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            try:
                return np.array(Image.open(p).convert("RGB"))
            except Exception:
                return None

    for i, image_id in enumerate(image_ids):
        if image_id in cache:
            feat = cache[image_id]
        else:
            p = _image_path(root, image_id)
            if p is None:
                feat = None
            else:
                img = _read_rgb(p)
                feat = None if img is None else _extract_simple_features(img)
            cache[image_id] = feat

        if feat is None:
            missing_mask[i] = True
            continue

        if X2 is None:
            feat_dim = int(feat.shape[0])
            X2 = np.empty((n, feat_dim), dtype=np.float32)
        X2[i] = feat

    if X2 is None:
        raise RuntimeError("Could not read any images to build features.")

    if missing_mask.any():
        mean_feat = X2[~missing_mask].mean(axis=0)
        X2[missing_mask] = mean_feat
        print(
            f"Warning: {int(missing_mask.sum())} images missing/unreadable; imputed features with mean."
        )

    return X2


def _safe_sigmoid(z):
    z = np.clip(z, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-z))


def train_and_predict_baseline(out_path="submission.csv"):
    root = _resolve_competition_root()
    train_path = os.path.join(root, "train.csv")
    test_path = os.path.join(root, "test.csv")
    sample_path = os.path.join(root, "sample_submission.csv")
    if not os.path.exists(sample_path):
        sample_paths = [
            "/kaggle/input/plant-pathology-2020-fgvc7/sample_submission.csv",
            "/kaggle/data/plant-pathology-2020-fgvc7/sample_submission.csv",
            "/kaggle/input/sample_submission.csv",
            "/kaggle/data/sample_submission.csv",
        ]
        sample_path = next((p for p in sample_paths if os.path.exists(p)), None)
    if sample_path is None:
        raise FileNotFoundError(
            "sample_submission.csv not found for templating output."
        )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    for c in ["image_id"] + TARGET_COLS:
        if c not in train_df.columns:
            raise KeyError(f"train.csv missing required column: {c}")
    if "image_id" not in test_df.columns:
        raise KeyError("test.csv missing required column: image_id")

    X_train = _build_feature_matrix(train_df["image_id"].tolist(), root)
    X_test = _build_feature_matrix(test_df["image_id"].tolist(), root)

    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import StratifiedKFold, KFold

    y_train = train_df[TARGET_COLS].astype(np.int32).to_numpy(copy=False)

    base_lr = LogisticRegression(
        max_iter=6000,
        solver="lbfgs",
        class_weight="balanced",
        C=1.0,
        random_state=0,
    )

    clf = OneVsRestClassifier(
        Pipeline(
            steps=[
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ("lr", base_lr),
            ]
        ),
        n_jobs=-1,
    )

    proxy = y_train.astype(np.int32).astype(str)
    proxy = np.array(["".join(row) for row in proxy], dtype=str)

    unique, counts = np.unique(proxy, return_counts=True)
    min_count = int(counts.min()) if counts.size else 0
    if min_count < 2:
        splitter = KFold(n_splits=5, shuffle=True, random_state=0)
        split_iter = splitter.split(X_train)
        print(
            "Warning: stratification proxy has rare groups; using KFold for OOF calibration."
        )
    else:
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        split_iter = splitter.split(X_train, proxy)

    oof_logits = np.zeros((X_train.shape[0], len(TARGET_COLS)), dtype=np.float64)

    for tr_idx, va_idx in split_iter:
        clf_fold = OneVsRestClassifier(
            Pipeline(
                steps=[
                    ("scaler", StandardScaler(with_mean=True, with_std=True)),
                    (
                        "lr",
                        LogisticRegression(
                            max_iter=6000,
                            solver="lbfgs",
                            class_weight="balanced",
                            C=1.0,
                            random_state=0,
                        ),
                    ),
                ]
            ),
            n_jobs=-1,
        )
        clf_fold.fit(X_train[tr_idx], y_train[tr_idx])

        try:
            logits_va = clf_fold.decision_function(X_train[va_idx])
        except Exception:
            p_va = clf_fold.predict_proba(X_train[va_idx]).astype(np.float64)
            p_va = np.clip(p_va, 1e-6, 1 - 1e-6)
            logits_va = np.log(p_va / (1 - p_va))
        oof_logits[va_idx] = logits_va

    clf.fit(X_train, y_train)

    try:
        logits_test = clf.decision_function(X_test)
    except Exception:
        proba_test = clf.predict_proba(X_test).astype(np.float64)
        proba_test = np.clip(proba_test, 1e-6, 1 - 1e-6)
        logits_test = np.log(proba_test / (1 - proba_test))

    from sklearn.linear_model import LogisticRegression as LRCal

    calib_a = np.ones(len(TARGET_COLS), dtype=np.float64)
    calib_b = np.zeros(len(TARGET_COLS), dtype=np.float64)
    for j in range(len(TARGET_COLS)):
        xj = oof_logits[:, j].reshape(-1, 1)
        yj = y_train[:, j]
        cal = LRCal(
            solver="lbfgs",
            max_iter=2000,
            C=10.0,
            random_state=0,
        )
        cal.fit(xj, yj)
        calib_a[j] = float(cal.coef_.ravel()[0])
        calib_b[j] = float(cal.intercept_.ravel()[0])

    proba_test_cal = _safe_sigmoid(
        logits_test * calib_a.reshape(1, -1) + calib_b.reshape(1, -1)
    )
    proba_test_cal = np.clip(proba_test_cal, 0.0, 1.0)

    sub = pd.read_csv(sample_path)
    sub["image_id"] = sub["image_id"].astype(str)
    sub = (
        sub.set_index("image_id")
        .reindex(test_df["image_id"].astype(str).values)
        .reset_index()
    )

    for i, c in enumerate(TARGET_COLS):
        sub[c] = proba_test_cal[:, i].astype(np.float64)

    sub.to_csv(out_path, index=False)
    print(f"Wrote baseline image-model submission to: {out_path}")
    print(sub.head())




## === cell 6
if len(submissions_all) >= 3:
    submission_avg = ensemble(submissions_all, [0, 1, 2], [0.04, 0.92, 0.04])
    make_submission_file(submission_avg, submissions_all, out_path="submission.csv")
elif len(submissions_all) >= 1:
    submission_avg = ensemble(submissions_all, [0], [1.0])
    make_submission_file(submission_avg, submissions_all, out_path="submission.csv")
else:
    train_and_predict_baseline(out_path="submission.csv")
