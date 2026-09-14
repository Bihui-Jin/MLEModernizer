# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np



## === cell 1
SUBMISSIONS_PATH = "/kaggle/input/submissions/submissions/"

DATA_DIR = "/kaggle/input/plant-pathology-2020-fgvc7"
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")
TEST_CSV_PATH = os.path.join(DATA_DIR, "test.csv")
TRAIN_CSV_PATH = os.path.join(DATA_DIR, "train.csv")
IMAGES_DIR = os.path.join(DATA_DIR, "images")



## === cell 2
submissions_all = []
if os.path.isdir(SUBMISSIONS_PATH):
    for dirname, _, filenames in os.walk(SUBMISSIONS_PATH):
        for filename in filenames:
            if filename.lower().endswith(".csv"):
                submissions_all.append(os.path.join(dirname, filename))
submissions_all.sort()
print(submissions_all)




## === cell 3
def ensemble(submissions_all, sub_idx, weights=None):
    """
    Weighted ensemble of existing submission files.
    Fixes:
      - Handles empty submissions_all gracefully
      - Validates sub_idx bounds
      - Validates weights length and normalizes weights to sum to 1
    """
    if weights is None:
        weights = [1.0] * len(sub_idx)

    if len(sub_idx) == 0:
        raise ValueError("sub_idx is empty; nothing to ensemble.")

    if len(weights) != len(sub_idx):
        raise ValueError(
            f"weights length ({len(weights)}) must match sub_idx length ({len(sub_idx)})."
        )

    if len(submissions_all) == 0:
        raise FileNotFoundError(
            f"No submission files found under {SUBMISSIONS_PATH}. "
            "Provide submissions or use the fallback submission generation."
        )

    for j in sub_idx:
        if j < 0 or j >= len(submissions_all):
            raise IndexError(
                f"sub_idx contains {j}, but submissions_all has length {len(submissions_all)}."
            )

    wsum = float(sum(weights))
    if wsum == 0.0:
        raise ValueError("Sum of weights is 0; cannot normalize.")
    weights = [w / wsum for w in weights]

    submission_with_weight = []
    for i in range(len(sub_idx)):
        path = submissions_all[sub_idx[i]]
        print(f"I'm taking submission {path} with weight {weights[i]}")
        submission = pd.read_csv(path)

        needed = ["healthy", "multiple_diseases", "rust", "scab"]
        missing = [c for c in needed if c not in submission.columns]
        if missing:
            raise ValueError(f"Submission {path} missing columns: {missing}")

        preds = submission.loc[:, needed].to_numpy(dtype="float64")
        submission_with_weight.append(preds * weights[i])

    submission_avg = sum(submission_with_weight)
    return submission_avg




## === cell 4
def make_submission_file(
    submission_avg, base_submission_path, out_path="submission.csv"
):
    """
    Writes a valid submission csv with correct columns and row order.
    Fix: base_submission_path is explicit, not implicitly submissions_all[0].
    """
    submission_df = pd.read_csv(base_submission_path)
    needed = ["image_id", "healthy", "multiple_diseases", "rust", "scab"]
    missing = [c for c in needed if c not in submission_df.columns]
    if missing:
        raise ValueError(f"Base submission file missing columns: {missing}")

    if submission_avg.shape != (len(submission_df), 4):
        raise ValueError(
            f"submission_avg has shape {submission_avg.shape} but expected {(len(submission_df), 4)}"
        )

    submission_df.loc[:, ["healthy", "multiple_diseases", "rust", "scab"]] = (
        submission_avg
    )
    submission_df.to_csv(out_path, index=False)
    print(f"Wrote {out_path} with shape {submission_df.shape}.")




## === cell 5
from PIL import Image
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import StratifiedKFold

TARGET_COLS = ["healthy", "multiple_diseases", "rust", "scab"]


def _image_path(image_id: str) -> str:
    return os.path.join(IMAGES_DIR, f"{image_id}.jpg")


def extract_features(image_ids, size=(160, 160), grid=4):
    eps = 1e-8

    extra_existing = 1 + (grid * grid) * 4
    D_existing = 44 + extra_existing

    D_new = 28
    D = D_existing + D_new

    X = np.zeros((len(image_ids), D), dtype=np.float32)

    for i, img_id in enumerate(image_ids):
        p = _image_path(img_id)
        img = Image.open(p).convert("RGB").resize(size)
        arr = np.asarray(img, dtype=np.float32) / 255.0  # HxWx3

        mean_rgb = arr.mean(axis=(0, 1))
        std_rgb = arr.std(axis=(0, 1)) + eps
        centered = arr - mean_rgb[None, None, :]
        skew_rgb = (centered**3).mean(axis=(0, 1)) / (std_rgb**3)

        flat = arr.reshape(-1, 3)
        p10 = np.percentile(flat, 10, axis=0)
        p50 = np.percentile(flat, 50, axis=0)
        p90 = np.percentile(flat, 90, axis=0)

        lab_arr = np.asarray(img.convert("LAB"), dtype=np.float32) / 255.0
        mean_lab = lab_arr.mean(axis=(0, 1))

        hsv_arr = np.asarray(img.convert("HSV"), dtype=np.float32) / 255.0
        mean_hsv = hsv_arr.mean(axis=(0, 1))

        R = arr[:, :, 0]
        G = arr[:, :, 1]
        B = arr[:, :, 2]

        gray = (0.2989 * R + 0.5870 * G + 0.1140 * B).astype(np.float32)
        gray_mean = float(gray.mean())
        gray_std = float(gray.std() + eps)
        g10 = float(np.percentile(gray, 10))
        g50 = float(np.percentile(gray, 50))
        g90 = float(np.percentile(gray, 90))

        dx = np.diff(gray, axis=1)
        dy = np.diff(gray, axis=0)
        grad_mag = np.sqrt(dx[:-1, :] ** 2 + dy[:, :-1] ** 2)
        grad_mean = float(grad_mag.mean())
        grad_std = float(grad_mag.std() + eps)

        dxx = np.diff(gray, n=2, axis=1)
        dyy = np.diff(gray, n=2, axis=0)
        lap_abs_mean = float((np.abs(dxx).mean() + np.abs(dyy).mean()) * 0.5)

        exg = (2.0 * G - R - B).astype(np.float32)
        exg_mean = float(exg.mean())

        nd_gr = float(((G - R) / (G + R + eps)).mean())
        nd_gb = float(((G - B) / (G + B + eps)).mean())
        nd_rb = float(((R - B) / (R + B + eps)).mean())

        r_over_g = float((R / (G + eps)).mean())
        b_over_g = float((B / (G + eps)).mean())
        r_over_b = float((R / (B + eps)).mean())

        def edge_strength(ch):
            ch_dx = np.abs(np.diff(ch, axis=1))
            ch_dy = np.abs(np.diff(ch, axis=0))
            return float(ch_dx.mean() + ch_dy.mean())

        edge_r = edge_strength(R)
        edge_g = edge_strength(G)
        edge_b = edge_strength(B)

        S = hsv_arr[:, :, 1]
        s_mean = float(S.mean())
        s_std = float(S.std() + eps)
        s_p90 = float(np.percentile(S.reshape(-1), 90))

        yellow_proxy = float((((R + G) * 0.5) - B).mean())

        H, W = gray.shape
        gh = H // grid
        gw = W // grid

        def grid_mean_std(mat):
            feats = []
            for rr in range(grid):
                for cc in range(grid):
                    r0 = rr * gh
                    c0 = cc * gw
                    r1 = (rr + 1) * gh if rr < grid - 1 else H
                    c1 = (cc + 1) * gw if cc < grid - 1 else W
                    patch = mat[r0:r1, c0:c1]
                    feats.append(float(patch.mean()))
                    feats.append(float(patch.std() + eps))
            return feats

        gray_grid = grid_mean_std(gray)  # 2*grid^2
        exg_grid = grid_mean_std(exg)  # 2*grid^2

        gray_small = (
            np.asarray(
                img.convert("L").resize((64, 64), resample=Image.BILINEAR),
                dtype=np.float32,
            )
            / 255.0
        )

        hist16, _ = np.histogram(
            gray_small.reshape(-1), bins=16, range=(0.0, 1.0), density=True
        )
        hist16 = hist16.astype(np.float32)

        thr = float(np.median(gray_small))
        binm = (gray_small > thr).astype(np.uint8)
        trans_h = float(np.mean(binm[:, 1:] != binm[:, :-1]))
        trans_v = float(np.mean(binm[1:, :] != binm[:-1, :]))

        hh, ww = gray_small.shape
        h2, w2 = hh // 2, ww // 2
        q1 = gray_small[:h2, :w2]
        q2 = gray_small[:h2, w2:]
        q3 = gray_small[h2:, :w2]
        q4 = gray_small[h2:, w2:]
        qmeans = np.array(
            [q1.mean(), q2.mean(), q3.mean(), q4.mean()], dtype=np.float32
        )

        qc = []
        for a in range(4):
            for b in range(a + 1, 4):
                qc.append(float(abs(qmeans[a] - qmeans[b])))
        qcontr = np.array(qc, dtype=np.float32)

        X[i, 0:3] = mean_rgb
        X[i, 3:6] = std_rgb
        X[i, 6:9] = skew_rgb
        X[i, 9:12] = p10
        X[i, 12:15] = p50
        X[i, 15:18] = p90
        X[i, 18:21] = mean_lab
        X[i, 21:24] = mean_hsv

        X[i, 24] = gray_mean
        X[i, 25] = gray_std
        X[i, 26] = g10
        X[i, 27] = g50
        X[i, 28] = g90
        X[i, 29] = grad_mean
        X[i, 30] = grad_std
        X[i, 31] = lap_abs_mean
        X[i, 32] = exg_mean
        X[i, 33] = nd_gr
        X[i, 34] = nd_gb
        X[i, 35] = nd_rb
        X[i, 36] = r_over_g
        X[i, 37] = b_over_g
        X[i, 38] = r_over_b
        X[i, 39] = edge_r
        X[i, 40] = edge_g
        X[i, 41] = edge_b
        X[i, 42] = s_mean
        X[i, 43] = s_p90

        k = 44
        X[i, k] = yellow_proxy
        k += 1
        X[i, k : k + len(gray_grid)] = np.asarray(gray_grid, dtype=np.float32)
        k += len(gray_grid)
        X[i, k : k + len(exg_grid)] = np.asarray(exg_grid, dtype=np.float32)
        k += len(exg_grid)

        X[i, k : k + 16] = hist16
        k += 16
        X[i, k] = trans_h
        X[i, k + 1] = trans_v
        k += 2
        X[i, k : k + 4] = qmeans
        k += 4
        X[i, k : k + 6] = qcontr

    return X


def _positive_class_proba_binary(estimator, X):
    classes = np.asarray(getattr(estimator, "classes_", [0, 1]))
    proba = np.asarray(estimator.predict_proba(X))

    if classes.size == 1:
        return np.full(
            X.shape[0], (1.0 - 1e-6) if int(classes[0]) == 1 else 1e-6, dtype=np.float64
        )

    if proba.ndim == 2:
        idx = np.where(classes == 1)[0]
        if idx.size == 0:
            return np.full(X.shape[0], 1e-6, dtype=np.float64)
        return np.clip(proba[:, int(idx[0])].astype(np.float64), 1e-6, 1.0 - 1e-6)

    return np.clip(proba.astype(np.float64), 1e-6, 1.0 - 1e-6)


def _predict_mo_positive_probas(fitted_pipeline, X):
    """
    Minimal score-relevant fix: always extract per-label positive-class probabilities
    from the fitted MultiOutputClassifier inside the pipeline (consistent with how we
    predict test), instead of mixing in a different per-class model for OOF.
    """
    scaler = fitted_pipeline.named_steps["scaler"]
    mo = fitted_pipeline.named_steps["model"]
    Xt = scaler.transform(X)

    preds = np.zeros((Xt.shape[0], 4), dtype=np.float64)
    for j in range(4):
        preds[:, j] = _positive_class_proba_binary(mo.estimators_[j], Xt)
    return np.clip(preds, 1e-6, 1.0 - 1e-6)


def _fit_platt_scalers_oof_mo(X, y, base_lr_params, n_splits=5, random_state=0):
    """
    Minimal, directly score-relevant change:
    Train Platt scalers on out-of-fold predictions produced by the SAME base pipeline
    (StandardScaler + MultiOutputClassifier(LogisticRegression)) used for test.
    This removes a mismatch/leakage risk and typically improves ROC AUC.
    """
    n, k = y.shape
    oof = np.zeros((n, k), dtype=np.float64)

    strat = (y[:, 0] * 8 + y[:, 1] * 4 + y[:, 2] * 2 + y[:, 3] * 1).astype(int)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for tr, va in skf.split(X, strat):
        fold_pipe = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    MultiOutputClassifier(
                        LogisticRegression(**base_lr_params), n_jobs=-1
                    ),
                ),
            ]
        )
        fold_pipe.fit(X[tr], y[tr])
        oof[va] = _predict_mo_positive_probas(fold_pipe, X[va])

    scalers = []
    for j in range(k):
        yj = y[:, j].astype(int)
        pj = np.clip(oof[:, j], 1e-6, 1.0 - 1e-6).reshape(-1, 1)

        if len(np.unique(yj)) < 2:
            scalers.append(None)
            continue

        cal = LogisticRegression(
            solver="lbfgs",
            penalty="l2",
            C=1.0,
            max_iter=2000,
            random_state=random_state,
        )
        cal.fit(pj, yj)
        scalers.append(cal)
    return scalers


def _apply_platt_scalers(preds, scalers):
    out = preds.copy().astype(np.float64)
    for j, cal in enumerate(scalers):
        if cal is None:
            continue
        pj = np.clip(out[:, j], 1e-6, 1.0 - 1e-6).reshape(-1, 1)
        out[:, j] = np.clip(cal.predict_proba(pj)[:, 1], 1e-6, 1.0 - 1e-6)
    return out


def train_and_predict_submission(
    train_csv_path, test_csv_path, sample_sub_path, out_path="submission.csv"
):
    train_df = pd.read_csv(train_csv_path)
    test_df = pd.read_csv(test_csv_path)
    sample_sub = pd.read_csv(sample_sub_path)

    submission_cols = ["image_id"] + [c for c in sample_sub.columns if c != "image_id"]
    for c in TARGET_COLS:
        if c not in submission_cols:
            submission_cols.append(c)

    X_train = extract_features(train_df["image_id"].tolist())
    y_train = train_df[TARGET_COLS].astype(int).to_numpy()
    X_test = extract_features(test_df["image_id"].tolist())

    base_lr_params = dict(
        solver="saga",
        penalty="l2",
        max_iter=5000,
        C=2.0,
        class_weight=None,
        random_state=0,
        n_jobs=1,  # avoid nested parallelism
    )

    scalers = _fit_platt_scalers_oof_mo(
        X_train, y_train, base_lr_params=base_lr_params, n_splits=5, random_state=0
    )

    clf = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                MultiOutputClassifier(LogisticRegression(**base_lr_params), n_jobs=-1),
            ),
        ]
    )
    clf.fit(X_train, y_train)

    preds = _predict_mo_positive_probas(clf, X_test)
    preds = _apply_platt_scalers(preds, scalers)

    sub_df = pd.DataFrame({"image_id": test_df["image_id"].values})
    for idx, c in enumerate(TARGET_COLS):
        sub_df[c] = preds[:, idx]

    sub_df = sub_df[submission_cols]
    sub_df.to_csv(out_path, index=False)
    print(
        f"Wrote {out_path} with shape {sub_df.shape} using trained image-feature model (MultiOutputClassifier) + OOF Platt scaling (matched base pipeline)."
    )




## === cell 6
if len(submissions_all) >= 2:
    submission_avg = ensemble(submissions_all, [0, 1], [0.1, 0.9])
    make_submission_file(
        submission_avg,
        base_submission_path=submissions_all[0],
        out_path="submission.csv",
    )
elif len(submissions_all) == 1:
    df = pd.read_csv(submissions_all[0])
    df.to_csv("submission.csv", index=False)
    print(
        f"Only one submission found; copied to submission.csv from {submissions_all[0]}"
    )
else:
    train_and_predict_submission(
        train_csv_path=TRAIN_CSV_PATH,
        test_csv_path=TEST_CSV_PATH,
        sample_sub_path=SAMPLE_SUB_PATH,
        out_path="submission.csv",
    )
