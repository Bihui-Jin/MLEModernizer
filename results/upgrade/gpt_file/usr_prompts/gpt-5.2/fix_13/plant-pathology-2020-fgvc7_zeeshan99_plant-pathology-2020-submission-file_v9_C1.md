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

3.9

# 3. Installed packages

No external packages required in the script and installed.

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

0.9039601561465894

# 6. Current score

0.71275

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'I fix the environment-breaking TensorFlow import issue by avoiding TF entirely (it’s not needed if we don’t train/infer), and I also remove the dependency on a missing external pretrained `.h5` file that causes the `FileNotFoundError`. To ensure the notebook always runs end-to-end and produces a valid `submission.csv`, I generate a safe, metric-neutral baseline submission directly from `sample_submission.csv` (uniform probabilities per class). This preserves the overall evaluation semantics (probability outputs for four labels) and guarantees correct column names/order and row alignment with `test.csv`. If you later provide an available model file inside this Kaggle dataset, we can re-enable model inference with minimal changes.'
- What this solution (achieved 0.50031) has done: 'Your current 0.5 score comes from constant 0.25 predictions, which yields near-random ranking and low mean ROC AUC. To move toward the 0.9039 target while keeping core logic minimal, I replace the uniform probabilities with a simple, legitimate image-based heuristic: compute per-image color statistics (RGB means/stds) and use a fixed linear mapping + softmax to produce varied class probabilities. This keeps the same end-to-end structure (read test.csv, output probabilities for the four labels, write submission.csv) but introduces meaningful variation correlated with leaf appearance, typically improving ROC AUC over uniform guesses. I also keep strict column/order alignment with `sample_submission.csv` and ensure the submission file is always written.'
- What this solution (achieved 0.51224) has done: 'Your current score (~0.50) suggests the heuristic is producing weak rankings; to move toward the 0.9039 target with minimal change, I keep the same image-statistics → fixed-logits → softmax pipeline but make the features more informative and the mapping less arbitrary. Specifically, I (1) add a few extra cheap, deterministic features (HSV mean/std plus simple “yellow/brown” ratios) and (2) standardize features within the test set before applying a fixed linear map, which usually improves ranking stability without changing the overall approach. I also keep the exact submission schema/order from `sample_submission.csv`, and keep a safe fallback for missing images/PIL so a valid `submission.csv` is always written.'
- What this solution (achieved 0.70122) has done: 'Your current approach is a fixed, deterministic image-statistics heuristic, so the safest way to move the ROC-AUC upward (toward 0.9039) without changing the overall logic is to (1) use the training set to learn the linear mapping from your existing features to the 4 labels (instead of hand-tuned weights), and (2) standardize features using train statistics before applying the mapping to test. This keeps the same pipeline shape: extract cheap per-image features → standardize → linear logits → softmax → write `submission.csv`, but makes the logits aligned with the actual labels, which should substantially improve ranking and thus mean column-wise ROC AUC. I also keep robust fallbacks: if PIL/images are unavailable, it still writes a valid submission. No sampling/early stopping is introduced, and runtime stays low (only 1638+183 images at 128×128).'
- What this solution (achieved 0.70255) has done: 'Your current score (0.70122) is well below the target (0.90396), so we should legitimately increase ROC-AUC with very small, safe changes while keeping your same “image statistics → standardize → linear mapping → sigmoid” pipeline. The biggest issue is that you’re fitting one joint linear model to logit-transformed one-hot targets; switching to per-class ridge regression directly on the 0/1 targets (still linear, still closed-form, no training loop) better matches ROC-AUC ranking and avoids instability from extreme logits. I also add one minimal deterministic improvement: compute standardization on the extended feature set (so the added brightness/variance/rg/gb features are on comparable scale) and tune ridge strength slightly for better generalization. All I/O paths and submission schema remain unchanged, and it still always write a valid `submission.csv`.'
- What this solution (achieved 0.70118) has done: 'Your current approach is a linear ridge model on hand-crafted image statistics; the biggest score limiter is that per-class independent sigmoids can produce poorly calibrated rankings for a mutually-exclusive 4-class problem, which can hurt mean column-wise ROC AUC. I keep the same feature extraction, standardization, and closed-form ridge solve, but switch the final probability mapping from sigmoid to softmax so scores compete across classes while staying in [0,1] and summing to 1 per image. I also add a tiny, deterministic feature enrichment (two more simple color-difference channels) without changing the overall pipeline shape, and keep the same submission schema/paths. These are minimal changes that typically improve ROC-AUC ranking toward your target without introducing new training loops or external dependencies.'
- What this solution (achieved 0.70101) has done: 'Your current score (0.70118) is far below the target (0.90396), so we should legitimately improve ranking signal while keeping the same overall pipeline: extract cheap image statistics → standardize → closed-form ridge → probability mapping → write `submission.csv`. The largest low-risk gain here is to make the linear ridge mapping better match the evaluation (mean per-column ROC AUC) by training **one-vs-rest ridge per class** on a **binary logit (or centered) target** and then using **per-class sigmoid** outputs (AUC is ranking-based and benefits from independent monotonic scores per class). This is a minimal change: same features, same standardization, same closed-form solve (no training loop), only the last mapping and target transform are adjusted to better align with column-wise AUC. I also add a tiny deterministic safeguard to handle any missing/blank images by imputing train feature means for those rows (instead of all-zeros), which reduces noisy outliers that can hurt AUC.'
- What this solution (achieved 0.70223) has done: 'Your current score (0.70101) is far below the target (0.90396), so we should increase mean column-wise ROC AUC while keeping your exact pipeline shape (cheap image stats → standardize → closed-form linear ridge → sigmoid outputs). The minimal, high-impact fix is to align training with the metric by fitting **one-vs-rest ridge separately per class** (still the same closed-form solve, no loops/epochs), which typically improves per-column ranking versus a single multi-target solve. I also tune only the ridge strength slightly (still deterministic) and add a tiny numerical stabilization (use float64 for the linear solve, then cast back) to reduce noisy coefficient estimates that can hurt AUC. All paths, feature extraction, and submission schema remain unchanged, and the script still always writes `./submission.csv`.'
- What this solution (achieved 0.7067) has done: 'Your current score (0.70223) is far below the target (0.90396), so we should legitimately increase mean column-wise ROC AUC with minimal risk while keeping the same overall pipeline (image stats → standardize → closed-form ridge → sigmoid → submission). The biggest low-change improvement is to keep one-vs-rest ridge but fit it on the standard ridge objective **without centering the binary targets**, and then optionally add a single, deterministic **monotonic calibration** step (Platt scaling per class) fitted on train predictions; this typically improves AUC ranking stability for each column without changing model family or adding training loops. I also add a tiny feature enrichment that’s still “cheap image statistics” (excess green/vegetation index + hue histogram bins) to give the linear model more separable signal, while keeping runtime under the 600s constraint. Submission column order and alignment remain exactly matched to `sample_submission.csv`, and the script always writes `./submission.csv`.'
- What this solution (achieved 0.61963) has done: 'Your current gap to the target is large (0.7067 vs 0.90396), so we should legitimately increase mean column-wise ROC AUC while keeping your exact “cheap image stats → standardize → closed-form ridge → per-class sigmoid (with Platt) → submission” pipeline. The smallest high-impact change is to replace the fragile least-squares Platt step with a deterministic, closed-form, AUC-friendly calibration: convert each class’s train scores into probabilities via a **rank-based CDF mapping** (monotonic, so it preserves AUC ordering) and then map test scores through the same learned CDF. This keeps the same linear model and features, but typically improves stability and avoids miscalibration that can hurt AUC across classes. I also make one tiny ridge-strength retune to better generalize with the CDF mapping (still same model family, closed-form solve), and keep submission schema/paths identical.'
- What this solution (achieved 0.70666) has done: 'Your current score (0.61963) is far below the target (0.90396), so we need a legitimate lift in mean column-wise ROC AUC with minimal changes to your existing pipeline (hand-crafted image stats → standardize → closed-form ridge → monotonic calibration → write submission). The biggest issue is that your CDF calibration is label-informed and can overfit/noise-amplify; switching to a label-free, strictly monotonic **rank-to-uniform** mapping preserves AUC while improving generalization stability. I keep the same ridge solve and features, but (1) replace the current CDF calibrator with an empirical rank transform, and (2) do a tiny, deterministic ridge-strength retune to reduce coefficient variance. Submission schema, paths, and end-to-end behavior remain unchanged, and it still always write `./submission.csv`.'
- What this solution (achieved 0.71275) has done: 'Your current pipeline is already a linear ridge model on handcrafted image statistics with a monotonic rank mapping; the safest way to lift mean column-wise ROC AUC toward 0.904 without changing core logic is to (1) fix a subtle calibration issue where your rank-mapping ignores labels (so it can’t improve AUC) and (2) slightly improve generalization by choosing the ridge strength via a tiny, deterministic train-only split. Concretely, we replace the label-free rank-to-uniform mapping with a label-aware monotonic mapping built from the empirical CDF of scores for positives vs negatives (keeps monotonicity and targets per-column AUC), and we pick `lam` from a small grid using mean AUC on a fixed validation fold (no training loops/epochs). All feature extraction, standardization, linear closed-form ridge solve, I/O paths, and submission schema remain the same, and the script still always writes `./submission.csv`.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np




## === cell 1
def processAndWriteDf(df, out_path="./submission.csv"):
    required_cols = ["image_id", "healthy", "multiple_diseases", "rust", "scab"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Submission df missing required columns: {missing}. Found: {list(df.columns)}"
        )

    df_out = df[required_cols].copy()
    df_out.to_csv(out_path, index=False)
    print(f"Wrote submission to: {out_path} (shape={df_out.shape})")
    return df_out




## === cell 2
def _softmax(logits: np.ndarray, axis: int = 1) -> np.ndarray:
    z = logits - np.max(logits, axis=axis, keepdims=True)
    ez = np.exp(z)
    return ez / np.sum(ez, axis=axis, keepdims=True)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x))


def _resolve_base_paths():
    base = "../input/plant-pathology-2020-fgvc7"
    sample_path = os.path.join(base, "sample_submission.csv")
    test_path = os.path.join(base, "test.csv")
    train_path = os.path.join(base, "train.csv")
    images_dir = os.path.join(base, "images")

    if not os.path.exists(sample_path):
        sample_path = "../input/sample_submission.csv"
    if not os.path.exists(test_path):
        test_path = "../input/test.csv"
    if not os.path.exists(train_path):
        train_path = "../input/train.csv"
    if not os.path.exists(images_dir):
        images_dir = "../input/images"

    return sample_path, test_path, train_path, images_dir


def _standardize_fit(X: np.ndarray, eps: float = 1e-6):
    mu = np.mean(X, axis=0, keepdims=True)
    sd = np.std(X, axis=0, keepdims=True)
    return mu, sd + eps


def _standardize_apply(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (X - mu) / sd


def _extract_features_from_ids(
    image_ids, images_dir, Image, size=(128, 128)
) -> np.ndarray:
    feats = np.zeros((len(image_ids), 22), dtype=np.float32)

    for i, img_id in enumerate(image_ids):
        img_path = os.path.join(images_dir, f"{img_id}.jpg")
        if not os.path.exists(img_path):
            continue
        try:
            img = Image.open(img_path).convert("RGB")
            img = img.resize(size)
            arr = np.asarray(img, dtype=np.float32) / 255.0  # (H,W,3)

            mean_rgb = arr.mean(axis=(0, 1))
            std_rgb = arr.std(axis=(0, 1))
            feats[i, 0:3] = mean_rgb
            feats[i, 3:6] = std_rgb

            r = arr[:, :, 0]
            g = arr[:, :, 1]
            b = arr[:, :, 2]

            cmax = np.maximum(np.maximum(r, g), b)
            cmin = np.minimum(np.minimum(r, g), b)
            delta = cmax - cmin

            v = cmax
            s = np.where(cmax > 1e-8, delta / (cmax + 1e-8), 0.0)

            h = np.zeros_like(cmax)
            mask = delta > 1e-8

            mr = mask & (cmax == r)
            mg = mask & (cmax == g)
            mb = mask & (cmax == b)

            h[mr] = ((g[mr] - b[mr]) / (delta[mr] + 1e-8)) % 6.0
            h[mg] = ((b[mg] - r[mg]) / (delta[mg] + 1e-8)) + 2.0
            h[mb] = ((r[mb] - g[mb]) / (delta[mb] + 1e-8)) + 4.0
            h = h / 6.0  # [0,1)

            hsv = np.stack([h, s, v], axis=2)
            mean_hsv = hsv.mean(axis=(0, 1))
            std_hsv = hsv.std(axis=(0, 1))
            feats[i, 6:9] = mean_hsv
            feats[i, 9:12] = std_hsv

            yellow_mask = (r > 0.45) & (g > 0.45) & (b < 0.35)
            yellow_ratio = float(yellow_mask.mean())

            brown_mask = (r > 0.35) & (g > 0.20) & (g < 0.60) & (b < 0.30)
            brown_ratio = float(brown_mask.mean())

            feats[i, 12] = yellow_ratio
            feats[i, 13] = brown_ratio

            exg = float((2.0 * mean_rgb[1]) - mean_rgb[0] - mean_rgb[2])
            feats[i, 14] = exg

            denom = float(mean_rgb[1] + mean_rgb[0] + 1e-6)
            gmr = float((mean_rgb[1] - mean_rgb[0]) / denom)
            feats[i, 15] = gmr

            sat_mask = s > 0.10
            if np.any(sat_mask):
                hh = h[sat_mask].ravel()
                bins = np.minimum((hh * 6.0).astype(np.int32), 5)
                hist = np.bincount(bins, minlength=6).astype(np.float32)
                hist = hist / (hist.sum() + 1e-6)
                feats[i, 16:22] = hist
            else:
                feats[i, 16:22] = 0.0

        except Exception:
            continue

    return feats.astype(np.float32)


def _fit_label_aware_monotone_calibrator(scores_1d: np.ndarray, y_1d: np.ndarray):
    s = scores_1d.astype(np.float64, copy=False)
    y = y_1d.astype(np.int32, copy=False)

    pos = s[y == 1]
    neg = s[y == 0]

    if len(pos) < 2 or len(neg) < 2:
        order = np.argsort(s, kind="mergesort")
        s_sorted = s[order]
        n = len(s_sorted)
        p_sorted = (np.arange(n, dtype=np.float64) + 0.5) / float(n)
        s_knots = np.concatenate(([s_sorted[0] - 1.0], s_sorted, [s_sorted[-1] + 1.0]))
        p_knots = np.concatenate(([p_sorted[0]], p_sorted, [p_sorted[-1]]))
        return s_knots, p_knots

    pos_sorted = np.sort(pos, kind="mergesort")
    neg_sorted = np.sort(neg, kind="mergesort")

    all_sorted = np.sort(s, kind="mergesort")
    n = len(all_sorted)
    idx = np.unique(np.linspace(0, n - 1, 256).round().astype(np.int64))
    knots = all_sorted[idx]

    Fp = np.searchsorted(pos_sorted, knots, side="right") / float(len(pos_sorted))
    Fn = np.searchsorted(neg_sorted, knots, side="right") / float(len(neg_sorted))

    p = Fp / (Fp + Fn + 1e-12)

    s_knots = np.concatenate(([knots[0] - 1.0], knots, [knots[-1] + 1.0]))
    p_knots = np.concatenate(([p[0]], p, [p[-1]]))
    return s_knots, p_knots


def _apply_monotone_calibrator(
    scores_1d: np.ndarray, s_knots: np.ndarray, p_knots: np.ndarray
):
    s = scores_1d.astype(np.float64, copy=False)
    return np.interp(s, s_knots, p_knots)


def _roc_auc_binary(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y = y_true.astype(np.int32, copy=False)
    s = y_score.astype(np.float64, copy=False)

    n1 = int(y.sum())
    n0 = int(len(y) - n1)
    if n1 == 0 or n0 == 0:
        return 0.5

    order = np.argsort(s, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1, dtype=np.float64)

    s_sorted = s[order]
    i = 0
    while i < len(s_sorted):
        j = i + 1
        while j < len(s_sorted) and s_sorted[j] == s_sorted[i]:
            j += 1
        if j - i > 1:
            avg = (i + 1 + j) / 2.0
            ranks[order[i:j]] = avg
        i = j

    sum_ranks_pos = ranks[y == 1].sum()
    auc = (sum_ranks_pos - n1 * (n1 + 1) / 2.0) / (n0 * n1)
    return float(auc)




## === cell 3
def getPredictionFromTPUModel():
    sample_path, test_path, train_path, images_dir = _resolve_base_paths()

    sample_sub = pd.read_csv(sample_path)
    test_df = pd.read_csv(test_path)
    train_df = pd.read_csv(train_path)

    target_cols = ["healthy", "multiple_diseases", "rust", "scab"]
    for c in ["image_id"] + target_cols:
        if c not in sample_sub.columns:
            raise ValueError(
                f"sample_submission.csv missing column '{c}'. Found: {list(sample_sub.columns)}"
            )

    try:
        from PIL import Image
    except Exception as e:
        print(f"PIL import failed ({e}); falling back to uniform predictions.")
        preds = np.full((len(test_df), 4), 0.25, dtype=np.float32)
        pred_df = pd.DataFrame(preds, columns=target_cols)
        result = pd.concat([test_df[["image_id"]].copy(), pred_df], axis=1)
        result = result[sample_sub.columns]
        return result

    X_train_raw = _extract_features_from_ids(
        train_df["image_id"].values, images_dir, Image
    )
    X_test_raw = _extract_features_from_ids(
        test_df["image_id"].values, images_dir, Image
    )

    def _extend(X):
        Rm, Gm, Bm, Rs, Gs, Bs = [X[:, j] for j in range(6)]
        brightness = (Rm + Gm + Bm) / 3.0
        variance = (Rs + Gs + Bs) / 3.0
        rg = Rm - Gm
        gb = Gm - Bm
        rb = Rm - Bm
        gr = Gm - Rm
        return np.column_stack([X, brightness, variance, rg, gb, rb, gr]).astype(
            np.float32
        )

    X_train_ext_raw = _extend(X_train_raw)
    X_test_ext_raw = _extend(X_test_raw)

    train_row_sum = np.sum(np.abs(X_train_ext_raw), axis=1)
    test_row_sum = np.sum(np.abs(X_test_ext_raw), axis=1)
    train_mean_feat = (
        X_train_ext_raw[train_row_sum > 0].mean(axis=0, keepdims=True)
        if np.any(train_row_sum > 0)
        else np.zeros((1, X_train_ext_raw.shape[1]), dtype=np.float32)
    )
    X_train_ext_raw = X_train_ext_raw.copy()
    X_test_ext_raw = X_test_ext_raw.copy()
    X_train_ext_raw[train_row_sum == 0] = train_mean_feat
    X_test_ext_raw[test_row_sum == 0] = train_mean_feat

    mu, sd = _standardize_fit(X_train_ext_raw)
    X_train_ext = _standardize_apply(X_train_ext_raw, mu, sd).astype(np.float32)
    X_test_ext = _standardize_apply(X_test_ext_raw, mu, sd).astype(np.float32)

    Y_train = train_df[target_cols].values.astype(np.float32)

    Xtr_full = np.concatenate(
        [np.ones((X_train_ext.shape[0], 1), dtype=np.float32), X_train_ext], axis=1
    )
    Xte = np.concatenate(
        [np.ones((X_test_ext.shape[0], 1), dtype=np.float32), X_test_ext], axis=1
    )

    n = Xtr_full.shape[0]
    rng = np.random.RandomState(1337)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = max(200, int(0.2 * n))
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]

    Xtr = Xtr_full[tr_idx].astype(np.float64, copy=False)
    Ytr = Y_train[tr_idx].astype(np.float64, copy=False)
    Xva = Xtr_full[val_idx].astype(np.float64, copy=False)
    Yva = Y_train[val_idx].astype(np.float64, copy=False)

    Xt = Xtr.T
    XtX = Xt @ Xtr
    I = np.eye(XtX.shape[0], dtype=np.float64)
    I[0, 0] = 0.0  # do not regularize bias

    lam_grid = [0.3, 1.0, 3.0, 10.0]
    best_lam = lam_grid[0]
    best_auc = -1.0

    for lam in lam_grid:
        A = XtX + lam * I
        W = np.zeros((Xtr.shape[1], Ytr.shape[1]), dtype=np.float64)
        for k in range(Ytr.shape[1]):
            W[:, k] = np.linalg.solve(A, Xt @ Ytr[:, k])

        va_scores = Xva @ W  # raw linear scores
        aucs = []
        for k in range(Ytr.shape[1]):
            aucs.append(_roc_auc_binary(Yva[:, k], va_scores[:, k]))
        mean_auc = float(np.mean(aucs))
        if mean_auc > best_auc:
            best_auc = mean_auc
            best_lam = lam

    Xtr64 = Xtr_full.astype(np.float64, copy=False)
    Y64 = Y_train.astype(np.float64, copy=False)
    Xt_full = Xtr64.T
    XtX_full = Xt_full @ Xtr64
    I_full = np.eye(XtX_full.shape[0], dtype=np.float64)
    I_full[0, 0] = 0.0
    A_full = XtX_full + float(best_lam) * I_full

    W_full = np.zeros((Xtr64.shape[1], Y64.shape[1]), dtype=np.float64)
    for k in range(Y64.shape[1]):
        W_full[:, k] = np.linalg.solve(A_full, Xt_full @ Y64[:, k])

    train_scores = (Xtr64 @ W_full).astype(np.float64)
    test_scores = (Xte.astype(np.float64, copy=False) @ W_full).astype(np.float64)

    probs = np.zeros_like(test_scores, dtype=np.float64)
    for k in range(Y_train.shape[1]):
        s_knots, p_knots = _fit_label_aware_monotone_calibrator(
            train_scores[:, k], Y64[:, k]
        )
        probs[:, k] = _apply_monotone_calibrator(test_scores[:, k], s_knots, p_knots)

    probs = probs.astype(np.float32)
    probs = np.clip(probs, 1e-6, 1.0 - 1e-6)

    pred_df = pd.DataFrame(probs, columns=target_cols)
    result = pd.concat([test_df[["image_id"]].copy(), pred_df], axis=1)
    result = result[sample_sub.columns]
    return result




## === cell 4
isTPU = True

if isTPU:
    df = getPredictionFromTPUModel()
    print(df.head(5))
    processAndWriteDf(df, out_path="./submission.csv")
else:
    df = pd.read_csv("../input/notebook45bc751087/submission.csv")
    processAndWriteDf(df, out_path="./submission.csv")
