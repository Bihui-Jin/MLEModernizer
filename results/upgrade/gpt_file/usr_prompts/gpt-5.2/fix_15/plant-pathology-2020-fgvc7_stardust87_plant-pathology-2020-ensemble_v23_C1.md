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

0.9683507164505156

# 6. Current score

0.71426

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'I fix the runtime error by removing the hard dependency on a non-existent `/kaggle/input/submissions/` folder and instead reading the competition’s provided `sample_submission.csv` to build a valid submission file. Since no model predictions are available in this notebook, I generate a safe, valid baseline prediction (uniform probabilities) that matches the required column names and row order from `test.csv`, ensuring the pipeline runs end-to-end and writes `submission.csv`. I also make the ensemble code robust: it validate indices/weights, handle empty submission lists, and always align predictions by `image_id` to avoid silent misalignment bugs. These changes are minimal and directly address the failure to produce a valid submission file.'
- What this solution (achieved 0.6451) has done: 'I fix the image path bug that’s causing `FileNotFoundError` by ensuring we append the `.jpg` suffix when constructing paths from `image_id` values, since the CSV IDs don’t include the extension but the files do. I also make feature extraction robust to occasional missing/corrupt files by substituting a safe zero-feature vector instead of crashing, which keeps the pipeline end-to-end. Finally, I keep the existing baseline model and submission-writing logic intact so it produces a valid `submission.csv` with the required columns and test-row order.'
- What this solution (achieved 0.66433) has done: 'Your current score (0.6451) is well below the target (0.96835), so we should cautiously improve model quality without changing the overall approach (handcrafted features + sklearn MultiOutput LogisticRegression). The biggest low-risk gain is to add a few more discriminative but still simple image features (colorfulness and coarse HSV statistics plus green dominance), which often correlate with rust/scab coloration patterns and should lift ROC AUC while keeping the same training/prediction pipeline. I also ensure features are always finite (avoid NaNs/inf from degenerate images) to prevent silent model degradation. Everything else (paths, model class, training loop, submission format) remains intact.'
- What this solution (achieved 0.66365) has done: 'I fix the feature-vector length mismatch that causes the broadcasting ValueError by making the allocated feature matrix width match the actual number of handcrafted features being produced. This is a minimal, score-neutral bug fix that unblocks training and inference end-to-end. I also add a small internal assert to catch any future accidental feature-length drift early and clearly. The rest of the pipeline (paths, feature logic, model, and submission writing) remains unchanged so it produces a valid `submission.csv`.'
- What this solution (achieved 0.66869) has done: 'Your current score (0.66365) is far below the target (0.96835), so we should improve the model while keeping the same core approach (handcrafted image features + sklearn MultiOutput LogisticRegression). The biggest low-risk lift is to fix the label handling: this competition’s labels are mutually exclusive, but the current code treats them as independent multi-label targets, which hurts ROC AUC; we switch to a single multinomial LogisticRegression and then output the 4 class probabilities in the required columns. To avoid overfitting and improve generalization without changing the overall training approach, we add a modest regularization tuning (slightly higher C and proper class balancing via sample weights) and use the model’s native predict_proba for all classes. Submission writing and file paths remain identical, and the script still produces `submission.csv` end-to-end.'
- What this solution (achieved 0.70258) has done: 'I fix the runtime crash by correcting the handcrafted feature vector length: the extractor currently produces 54 values but preallocates 55, which triggers a ValueError during both validation and training. This is a minimal, score-neutral bug fix that preserves the exact modeling approach (same features, same LogisticRegression pipeline), just making the declared feature dimension consistent with what is actually computed. I also keep the end-to-end flow unchanged so it still trains (when no external submissions exist) and writes a valid `submission.csv` with the required columns and test-set order. No changes are made to the model architecture/training logic beyond unblocking execution.'
- What this solution (achieved 0.70907) has done: 'Your current score (0.70258) is far below the target (0.96835), so we should improve generalization while keeping the exact same core pipeline (handcrafted features → StandardScaler → multinomial LogisticRegression → predict_proba). The smallest high-impact change is to tune only LogisticRegression’s regularization strength `C` using a lightweight stratified holdout search and then retrain on full data with the best `C`, which often yields a meaningful AUC lift without changing the model class or features. I also set `n_jobs=-1` (where supported) to keep runtime within limits, and leave submission formatting and alignment untouched. This keeps evaluation semantics identical (still 4 class probabilities per image_id), but moves the score upward toward the target.'
- What this solution (achieved 0.70907) has done: 'Your current score (0.70907) is far below the target (0.96835), so the smallest likely lift without changing the core pipeline is to fix a hidden inconsistency: training uses class-balanced sample weights, but the internal C-search does not use the same weights for validation scoring, which can select a suboptimal C. I keep the exact same feature extraction and multinomial LogisticRegression, but make the C-selection use the same weighting scheme end-to-end and make the split/C-search more stable by using a small fixed set of StratifiedShuffleSplit folds (still fast). I also ensure the training-time class weights are computed once from the full training distribution and used consistently in both tuning and final fit (no change to model family or loss). Submission formatting/ordering stays identical and the script still writes `submission.csv`.'
- What this solution (achieved 0.71426) has done: 'I fix the feature-length mismatch that currently crashes training by making `N_FEATS` match the actual number of features produced (66), which is a score-neutral correctness fix. I also make the pipeline robust by disabling the internal AUC check from hard-failing the run (so it prints diagnostics but won’t stop submission creation). Finally, I keep the existing model/feature logic intact so the notebook runs end-to-end and always writes a valid `submission.csv` with the required columns and test order.'
- What this solution (achieved 0.71426) has done: 'Your current score (0.71426) is far below the target (0.96835), so we need a real uplift without changing the core pipeline (handcrafted features → StandardScaler → multinomial LogisticRegression → predict_proba). The most likely low-risk gain is to (1) fix a subtle but important weighting inconsistency by computing class-balanced sample weights from the *training fold only* during CV (and still using full-data weights for the final refit), and (2) tune `C` against the competition’s metric more faithfully by evaluating “mean column-wise ROC AUC” with a proper OvR AUC computed from the multiclass probabilities (instead of per-class one-vs-rest on onehot rows that can be slightly inconsistent when a class is absent). These are minimal, local changes that keep the same model family, loss, and prediction semantics, but should select a better-regularized model and improve generalization toward the target. Submission writing, paths, and feature extraction remain unchanged, and the script still runs end-to-end to produce `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import pandas as pd

DATA_DIR_CANDIDATES = [
    "/kaggle/input/plant-pathology-2020-fgvc7",
    "/kaggle/data/plant-pathology-2020-fgvc7",
    "/kaggle/input",
    "/kaggle/data",
]


def _find_file(filename: str):
    for base in DATA_DIR_CANDIDATES:
        path = os.path.join(base, filename)
        if os.path.exists(path):
            return path
        nested = os.path.join(base, "plant-pathology-2020-fgvc7", filename)
        if os.path.exists(nested):
            return nested
    raise FileNotFoundError(f"Could not find {filename} in {DATA_DIR_CANDIDATES}")


TRAIN_CSV = _find_file("train.csv")
TEST_CSV = _find_file("test.csv")
SAMPLE_SUB_CSV = _find_file("sample_submission.csv")

print("Using:")
print(" TRAIN_CSV =", TRAIN_CSV)
print(" TEST_CSV  =", TEST_CSV)
print(" SAMPLE    =", SAMPLE_SUB_CSV)

TARGET_COLS = ["healthy", "multiple_diseases", "rust", "scab"]




## === cell 1
def list_submission_files(submissions_path: str):
    submissions_all = []
    if submissions_path and os.path.exists(submissions_path):
        for dirname, _, filenames in os.walk(submissions_path):
            for filename in filenames:
                if filename.lower().endswith(".csv"):
                    submissions_all.append(os.path.join(dirname, filename))
    submissions_all.sort()
    return submissions_all


def ensemble(submissions_all, sub_idx, weights=None):
    if weights is None:
        weights = [1.0] * len(sub_idx)

    if len(sub_idx) != len(weights):
        raise ValueError(
            f"sub_idx and weights must have same length. Got {len(sub_idx)} vs {len(weights)}"
        )

    if len(submissions_all) == 0:
        raise ValueError("No submission files found to ensemble.")

    for j in sub_idx:
        if j < 0 or j >= len(submissions_all):
            raise IndexError(
                f"sub_idx contains {j} but submissions_all has length {len(submissions_all)}"
            )

    base = pd.read_csv(submissions_all[sub_idx[0]])
    if "image_id" not in base.columns:
        raise ValueError(
            f"Submission file missing image_id: {submissions_all[sub_idx[0]]}"
        )

    base_ids = base[["image_id"]].copy()
    pred_sum = None
    wsum = 0.0

    for i, j in enumerate(sub_idx):
        path = submissions_all[j]
        w = float(weights[i])
        print(f"I'm taking submission {path} with weight {w}")

        df = pd.read_csv(path)
        missing = [c for c in (["image_id"] + TARGET_COLS) if c not in df.columns]
        if missing:
            raise ValueError(f"Submission {path} missing columns: {missing}")

        df = df[["image_id"] + TARGET_COLS].copy()
        df = base_ids.merge(df, on="image_id", how="left", validate="1:1")

        if df[TARGET_COLS].isna().any().any():
            raise ValueError(
                f"Submission {path} has missing predictions after aligning by image_id."
            )

        arr = df[TARGET_COLS].values
        if pred_sum is None:
            pred_sum = arr * w
        else:
            pred_sum += arr * w
        wsum += w

    pred_avg = pred_sum / wsum if wsum != 0 else pred_sum
    return base_ids, pred_avg




## === cell 2
def make_submission_file(image_ids_df, preds, out_path="submission.csv"):
    sub = pd.read_csv(SAMPLE_SUB_CSV)

    test_ids = pd.read_csv(TEST_CSV)[["image_id"]]
    sub = test_ids.merge(sub[["image_id"]], on="image_id", how="left")
    sub = sub.drop(columns=[], errors="ignore")

    if image_ids_df is not None and preds is not None:
        dfp = image_ids_df.copy()
        for k, c in enumerate(TARGET_COLS):
            dfp[c] = preds[:, k]
        sub = test_ids.merge(dfp, on="image_id", how="left", validate="1:1")
    else:
        for c in TARGET_COLS:
            sub[c] = 0.25

    sub = sub[["image_id"] + TARGET_COLS].copy()
    for c in TARGET_COLS:
        sub[c] = sub[c].astype(float).clip(0.0, 1.0)

    sub.to_csv(out_path, index=False)
    print(f"Wrote {out_path} with shape {sub.shape}")
    print(sub.head())




## === cell 3
import numpy as np
from PIL import Image

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


def _find_images_dir():
    cand = [
        os.path.join(os.path.dirname(TRAIN_CSV), "images"),
        os.path.join(os.path.dirname(TEST_CSV), "images"),
    ]
    for base in DATA_DIR_CANDIDATES:
        cand.append(os.path.join(base, "images"))
        cand.append(os.path.join(base, "plant-pathology-2020-fgvc7", "images"))
    for p in cand:
        if p and os.path.isdir(p):
            return p
    raise FileNotFoundError("Could not locate images/ directory.")


IMAGES_DIR = _find_images_dir()
print("IMAGES_DIR =", IMAGES_DIR)


def _image_path(image_id: str) -> str:
    fname = str(image_id)
    if not fname.lower().endswith(".jpg"):
        fname = fname + ".jpg"
    return os.path.join(IMAGES_DIR, fname)


def _rgb_to_hsv_np(arr_rgb_0_1: np.ndarray) -> np.ndarray:
    r = arr_rgb_0_1[..., 0]
    g = arr_rgb_0_1[..., 1]
    b = arr_rgb_0_1[..., 2]

    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin

    h = np.zeros_like(cmax)
    eps = 1e-6
    mask = delta > eps

    denom = delta + eps
    r_eq = (cmax == r) & mask
    g_eq = (cmax == g) & mask
    b_eq = (cmax == b) & mask

    h[r_eq] = ((g[r_eq] - b[r_eq]) / denom[r_eq]) % 6.0
    h[g_eq] = ((b[g_eq] - r[g_eq]) / denom[g_eq]) + 2.0
    h[b_eq] = ((r[b_eq] - g[b_eq]) / denom[b_eq]) + 4.0
    h = (h / 6.0) % 1.0

    s = np.zeros_like(cmax)
    s[cmax > eps] = delta[cmax > eps] / (cmax[cmax > eps] + eps)

    v = cmax
    hsv = np.stack([h, s, v], axis=-1)
    return hsv


def _edge_strength(gray01: np.ndarray) -> float:
    gx = np.diff(gray01, axis=1)
    gy = np.diff(gray01, axis=0)
    gxm = float(np.mean(np.abs(gx))) if gx.size else 0.0
    gym = float(np.mean(np.abs(gy))) if gy.size else 0.0
    return gxm + gym


def _block_stats(x2d: np.ndarray, blocks=(4, 4)):
    h, w = x2d.shape
    bh, bw = blocks
    ys = np.linspace(0, h, bh + 1, dtype=int)
    xs = np.linspace(0, w, bw + 1, dtype=int)
    out = []
    for i in range(bh):
        for j in range(bw):
            patch = x2d[ys[i] : ys[i + 1], xs[j] : xs[j + 1]]
            out.append(float(patch.mean()) if patch.size else 0.0)
    return np.asarray(out, dtype=np.float32)


def extract_features(image_ids, size=(192, 192)):
    N_FEATS = 66

    n = len(image_ids)
    feats = np.zeros((n, N_FEATS), dtype=np.float32)

    for i, img_id in enumerate(image_ids):
        p = _image_path(img_id)
        try:
            with Image.open(p) as im:
                im = im.convert("RGB").resize(size)
                arr = np.asarray(im, dtype=np.float32) / 255.0
        except (FileNotFoundError, OSError):
            continue

        ch_mean = arr.mean(axis=(0, 1))
        ch_std = arr.std(axis=(0, 1))

        r, g, b = ch_mean
        eps = 1e-6
        rg = (r - g) / (r + g + eps)
        gb = (g - b) / (g + b + eps)
        rb = (r - b) / (r + b + eps)

        gray = 0.2989 * arr[:, :, 0] + 0.5870 * arr[:, :, 1] + 0.1140 * arr[:, :, 2]
        gray_mean = float(gray.mean())
        gray_std = float(gray.std())
        gray_p10, gray_p50, gray_p90 = np.percentile(gray, [10, 50, 90]).astype(
            np.float32
        )

        hsv = _rgb_to_hsv_np(arr)
        hsv_mean = hsv.mean(axis=(0, 1))
        hsv_std = hsv.std(axis=(0, 1))
        s = hsv[:, :, 1]
        vch = hsv[:, :, 2]
        s_p10, s_p50, s_p90 = np.percentile(s, [10, 50, 90]).astype(np.float32)
        v_p10, v_p50, v_p90 = np.percentile(vch, [10, 50, 90]).astype(np.float32)

        rg_im = arr[:, :, 0] - arr[:, :, 1]
        yb_im = 0.5 * (arr[:, :, 0] + arr[:, :, 1]) - arr[:, :, 2]
        sigma_rg = float(rg_im.std())
        sigma_yb = float(yb_im.std())
        mu_rg = float(rg_im.mean())
        mu_yb = float(yb_im.mean())
        colorfulness = np.sqrt(sigma_rg**2 + sigma_yb**2) + 0.3 * np.sqrt(
            mu_rg**2 + mu_yb**2
        )

        green_dom = float(g / (r + b + eps))
        edge = float(_edge_strength(gray))

        R = arr[:, :, 0]
        G = arr[:, :, 1]
        B = arr[:, :, 2]
        exg = 2.0 * G - R - B
        exr = 1.4 * R - G
        exg_mean = float(exg.mean())
        exg_std = float(exg.std())
        exr_mean = float(exr.mean())
        exr_std = float(exr.std())

        gx = np.diff(gray, axis=1)
        gy = np.diff(gray, axis=0)
        if gx.size and gy.size:
            gxm = gx[:, :-1] if gx.shape[1] > 1 else gx
            gym = gy[:-1, :] if gy.shape[0] > 1 else gy
            gh = min(gxm.shape[0], gym.shape[0])
            gw = min(gxm.shape[1], gym.shape[1])
            grad = np.sqrt(gxm[:gh, :gw] ** 2 + gym[:gh, :gw] ** 2)
            grad_mean = float(grad.mean()) if grad.size else 0.0
            grad_std = float(grad.std()) if grad.size else 0.0
            grad_p90 = float(np.percentile(grad, 90)) if grad.size else 0.0
        else:
            grad_mean, grad_std, grad_p90 = 0.0, 0.0, 0.0

        s_blocks = _block_stats(s, blocks=(3, 3))  # 9 feats
        exg_blocks = _block_stats(exg, blocks=(3, 3))  # 9 feats
        gray_blocks = _block_stats(gray, blocks=(3, 3))  # 9 feats

        d2x = np.diff(gray, n=2, axis=1)
        d2y = np.diff(gray, n=2, axis=0)
        lap_energy = float(np.mean(np.abs(d2x))) + float(np.mean(np.abs(d2y)))
        lap_std = float(np.std(d2x)) + float(np.std(d2y))

        if gx.size and gy.size:
            gx2 = gxm[:gh, :gw]
            gy2 = gym[:gh, :gw]
            grad_energy = float(np.mean(gx2 * gx2 + gy2 * gy2))
        else:
            grad_energy = 0.0

        v = np.array(
            [
                ch_mean[0],
                ch_mean[1],
                ch_mean[2],
                ch_std[0],
                ch_std[1],
                ch_std[2],
                rg,
                gb,
                rb,
                gray_mean,
                gray_std,
                float(gray_p10),
                float(gray_p50),
                float(gray_p90),
                hsv_mean[0],
                hsv_mean[1],
                hsv_mean[2],
                hsv_std[0],
                hsv_std[1],
                hsv_std[2],
                float(s_p10),
                float(s_p50),
                float(s_p90),
                float(v_p10),
                float(v_p50),
                float(v_p90),
                float(colorfulness),
                float(green_dom),
                edge,
                exg_mean,
                exg_std,
                exr_mean,
                exr_std,
                grad_mean,
                grad_std,
                grad_p90,
                *s_blocks.tolist(),
                *exg_blocks.tolist(),
                *gray_blocks.tolist(),
                float(lap_energy),
                float(lap_std),
                float(grad_energy),
            ],
            dtype=np.float32,
        )

        if v.shape[0] != N_FEATS:
            raise ValueError(
                f"Feature length mismatch: got {v.shape[0]} expected {N_FEATS}"
            )

        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        feats[i, :] = v

    return feats


def _onehot_to_class_index(y_onehot: np.ndarray) -> np.ndarray:
    y = y_onehot.astype(int)
    row_sums = y.sum(axis=1)
    bad = np.where(row_sums != 1)[0]
    if bad.size:
        print(
            f"Warning: found {bad.size} rows with sum(y)!=1; using argmax for those rows."
        )
        y_idx = np.argmax(y, axis=1)
    else:
        y_idx = np.argmax(y, axis=1)
    return y_idx.astype(int)


def _fit_multinomial_lr(X_tr, y_tr, C_val: float, sample_weight=None):
    base_lr = LogisticRegression(
        solver="lbfgs",
        max_iter=2000,
        C=float(C_val),
        multi_class="multinomial",
        random_state=42,
        n_jobs=-1,
    )
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", base_lr),
        ]
    )
    model.fit(X_tr, y_tr, clf__sample_weight=sample_weight)
    return model


def _class_balanced_sample_weight(y_idx: np.ndarray, n_classes: int) -> np.ndarray:
    counts = np.bincount(y_idx, minlength=n_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    class_w = counts.sum() / (n_classes * counts)
    return class_w[y_idx]


def _mean_columnwise_ovr_auc(
    y_true_onehot: np.ndarray, proba_4col: np.ndarray
) -> float:
    from sklearn.metrics import roc_auc_score

    aucs = []
    for k in range(y_true_onehot.shape[1]):
        yk = y_true_onehot[:, k]
        pk = proba_4col[:, k]
        try:
            aucs.append(float(roc_auc_score(yk, pk)))
        except ValueError:
            aucs.append(np.nan)
    return float(np.nanmean(aucs))


def train_and_predict_baseline():
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)

    X_train = extract_features(train_df["image_id"].tolist())
    y_train_onehot = train_df[TARGET_COLS].values
    y_train = _onehot_to_class_index(y_train_onehot)

    X_test = extract_features(test_df["image_id"].tolist())

    sample_weight_full = _class_balanced_sample_weight(
        y_train, n_classes=len(TARGET_COLS)
    )

    from sklearn.model_selection import StratifiedShuffleSplit

    splitter = StratifiedShuffleSplit(n_splits=3, test_size=0.2, random_state=42)

    C_CANDIDATES = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
    best_C = 2.0
    best_mean_auc = -np.inf

    for C_val in C_CANDIDATES:
        fold_aucs = []
        for fold, (tr_idx, va_idx) in enumerate(
            splitter.split(X_train, y_train), start=1
        ):
            X_tr, X_va = X_train[tr_idx], X_train[va_idx]
            y_tr, y_va = y_train[tr_idx], y_train[va_idx]
            y_va_onehot = y_train_onehot[va_idx].astype(int)

            sw_tr = _class_balanced_sample_weight(y_tr, n_classes=len(TARGET_COLS))

            m = _fit_multinomial_lr(X_tr, y_tr, C_val=C_val, sample_weight=sw_tr)

            proba = m.predict_proba(X_va)
            out_va = np.zeros((len(va_idx), len(TARGET_COLS)), dtype=np.float32)
            for j, cls in enumerate(m.named_steps["clf"].classes_):
                out_va[:, int(cls)] = proba[:, j].astype(np.float32)

            mean_auc = _mean_columnwise_ovr_auc(y_va_onehot, out_va)
            fold_aucs.append(mean_auc)
            print(f"C={C_val} fold={fold} internal mean AUC={mean_auc:.6f}")

        cv_mean = float(np.nanmean(fold_aucs))
        print(f"C={C_val} CV mean AUC over {len(fold_aucs)} folds = {cv_mean:.6f}")

        if np.isfinite(cv_mean) and cv_mean > best_mean_auc:
            best_mean_auc = cv_mean
            best_C = float(C_val)

    print(f"Selected C={best_C} based on internal CV mean AUC={best_mean_auc:.6f}")

    model = _fit_multinomial_lr(
        X_train, y_train, C_val=best_C, sample_weight=sample_weight_full
    )

    preds = model.predict_proba(X_test).astype(np.float32)
    out = np.zeros((len(test_df), len(TARGET_COLS)), dtype=np.float32)
    for j, cls in enumerate(model.named_steps["clf"].classes_):
        out[:, int(cls)] = preds[:, j]

    image_ids_df = test_df[["image_id"]].copy()
    return image_ids_df, out




## === cell 4
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score


def quick_internal_auc_check():
    train_df = pd.read_csv(TRAIN_CSV)
    X = extract_features(train_df["image_id"].tolist())
    y_onehot = train_df[TARGET_COLS].values.astype(int)
    y = _onehot_to_class_index(y_onehot)

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    (tr_idx, va_idx) = next(splitter.split(X, y))

    X_tr, X_va = X[tr_idx], X[va_idx]
    y_tr, y_va_onehot = y[tr_idx], y_onehot[va_idx]

    sample_weight = _class_balanced_sample_weight(y_tr, n_classes=len(TARGET_COLS))

    base_lr = LogisticRegression(
        solver="lbfgs",
        max_iter=2000,
        C=2.0,
        multi_class="multinomial",
        random_state=42,
        n_jobs=-1,
    )
    model = Pipeline([("scaler", StandardScaler()), ("clf", base_lr)])
    model.fit(X_tr, y_tr, clf__sample_weight=sample_weight)

    proba = model.predict_proba(X_va)
    out = np.zeros((len(va_idx), len(TARGET_COLS)), dtype=np.float32)
    for j, cls in enumerate(model.named_steps["clf"].classes_):
        out[:, int(cls)] = proba[:, j].astype(np.float32)

    aucs = []
    for k, c in enumerate(TARGET_COLS):
        try:
            aucs.append(roc_auc_score(y_va_onehot[:, k], out[:, k]))
        except ValueError:
            aucs.append(np.nan)
    print("Internal AUCs:", dict(zip(TARGET_COLS, aucs)))
    print("Internal mean AUC (nanmean):", float(np.nanmean(aucs)))


try:
    quick_internal_auc_check()
except Exception as e:
    print("quick_internal_auc_check failed (continuing):", repr(e))



## === cell 5
SUBMISSIONS_PATH = (
    "/kaggle/input/submissions/"  # keep original path, but do not assume it exists
)
submissions_all = list_submission_files(SUBMISSIONS_PATH)
print("Found submission files:", submissions_all)

if len(submissions_all) >= 3:
    image_ids_df, submission_avg = ensemble(submissions_all, [0, 1, 2], [0.4, 0.5, 0.1])
    make_submission_file(image_ids_df, submission_avg, out_path="submission.csv")
elif len(submissions_all) > 0:
    idx = list(range(len(submissions_all)))
    w = [1.0] * len(idx)
    image_ids_df, submission_avg = ensemble(submissions_all, idx, w)
    make_submission_file(image_ids_df, submission_avg, out_path="submission.csv")
else:
    image_ids_df, preds = train_and_predict_baseline()
    make_submission_file(
        image_ids_df=image_ids_df, preds=preds, out_path="submission.csv"
    )
