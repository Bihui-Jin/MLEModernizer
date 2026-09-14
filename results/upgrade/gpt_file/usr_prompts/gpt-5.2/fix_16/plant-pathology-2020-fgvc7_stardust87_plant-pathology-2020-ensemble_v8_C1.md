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



## === cell 1
SUBMISSIONS_PATH = "/kaggle/input/submissions/submissions/"

DATA_DIR = "/kaggle/input/plant-pathology-2020-fgvc7"
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")
TEST_CSV_PATH = os.path.join(DATA_DIR, "test.csv")
TRAIN_CSV_PATH = os.path.join(DATA_DIR, "train.csv")
IMAGES_DIR = os.path.join(DATA_DIR, "images")



## === cell 2
submissions_all = []
if os.path.exists(SUBMISSIONS_PATH):
    for dirname, _, filenames in os.walk(SUBMISSIONS_PATH):
        for filename in filenames:
            if filename.lower().endswith(".csv"):
                submissions_all.append(os.path.join(dirname, filename))
submissions_all.sort()
print(submissions_all)




## === cell 3
def ensemble(submissions_all, sub_idx, weights=None):
    """
    Weighted average ensemble of submission files.
    Bugfixes:
      - Validate indices are in range.
      - Validate weights length.
      - Normalize weights to sum to 1 to keep probabilities calibrated.
    """
    if weights is None:
        weights = [1.0] * len(sub_idx)

    if len(sub_idx) != len(weights):
        raise ValueError(
            f"len(sub_idx)={len(sub_idx)} must equal len(weights)={len(weights)}"
        )

    if len(submissions_all) == 0:
        raise ValueError("No submissions found to ensemble.")

    for j in sub_idx:
        if j < 0 or j >= len(submissions_all):
            raise IndexError(
                f"Submission index {j} out of range for {len(submissions_all)} files."
            )

    wsum = float(sum(weights))
    if wsum <= 0:
        raise ValueError("Sum of weights must be > 0.")
    weights = [w / wsum for w in weights]

    submission_with_weight = []
    for i in range(len(sub_idx)):
        path = submissions_all[sub_idx[i]]
        print(f"I'm taking submission {path} with weight {weights[i]}")
        submission = pd.read_csv(path)

        required = ["healthy", "multiple_diseases", "rust", "scab"]
        missing = [c for c in required if c not in submission.columns]
        if missing:
            raise ValueError(f"Missing columns {missing} in {path}")

        arr = submission.loc[:, required].values
        submission_with_weight.append(arr * weights[i])

    submission_avg = sum(submission_with_weight)
    return submission_avg




## === cell 4
def make_submission_file(submission_avg, template_path, out_path="submission.csv"):
    """
    Create final submission CSV.
    Bugfix: use a known-good template (sample_submission.csv) if external submissions are absent.
    """
    submission_df = pd.read_csv(template_path)

    required = ["image_id", "healthy", "multiple_diseases", "rust", "scab"]
    missing = [c for c in required if c not in submission_df.columns]
    if missing:
        raise ValueError(f"Template {template_path} missing columns {missing}")

    submission_df = submission_df[required].copy()
    submission_df[["healthy", "multiple_diseases", "rust", "scab"]] = submission_avg
    submission_df.to_csv(out_path, index=False)
    print(f"Wrote {out_path} with shape {submission_df.shape}")




## === cell 5
def _extract_image_features(image_path, do_hflip=False, do_vflip=False):
    """
    Lightweight feature extractor: color/texture statistics.

    Score-relevant fix (minimal, same feature families):
      - Compute HSV features from the same normalized RGB array used for other features,
        avoiding an inconsistent feature space (previously HSV came from the unnormalized PIL image).
    """
    from PIL import Image
    import numpy as np

    img = Image.open(image_path).convert("RGB")
    img = img.resize((128, 128))
    if do_hflip:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if do_vflip:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)

    arr = np.asarray(img, dtype=np.float32) / 255.0  # (H, W, 3)
    eps = 1e-6

    ch_mean = arr.reshape(-1, 3).mean(axis=0)
    ch_mean = np.maximum(ch_mean, eps)
    arr = arr / ch_mean[None, None, :]
    arr = np.clip(arr, 0.0, 3.0)  # prevent extreme spikes; still deterministic

    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]

    feats = []

    for ch in range(3):
        x = arr[:, :, ch]
        feats.append(float(x.mean()))
        feats.append(float(x.std()))

    feats.append(float((r.mean() + eps) / (g.mean() + eps)))
    feats.append(float((r.mean() + eps) / (b.mean() + eps)))
    feats.append(float((g.mean() + eps) / (b.mean() + eps)))

    denom = r + g + b + eps
    rn = r / denom
    gn = g / denom
    bn = b / denom
    for x in (rn, gn, bn):
        feats.append(float(x.mean()))
        feats.append(float(x.std()))

    gray = 0.299 * r + 0.587 * g + 0.114 * b
    feats.append(float(gray.mean()))
    feats.append(float(gray.std()))
    h, w = gray.shape
    c0, c1 = h // 4, 3 * h // 4
    d0, d1 = w // 4, 3 * w // 4
    center = gray[c0:c1, d0:d1]
    feats.append(float(center.mean()))
    feats.append(float(center.std()))

    gy, gx = np.gradient(gray)
    grad = (gx * gx + gy * gy) ** 0.5
    feats.append(float(grad.mean()))
    feats.append(float(grad.std()))

    rgb = np.clip(arr / 3.0, 0.0, 1.0).astype(np.float32)
    r2, g2, b2 = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    cmax = np.maximum(np.maximum(r2, g2), b2)
    cmin = np.minimum(np.minimum(r2, g2), b2)
    delta = cmax - cmin

    hue = np.zeros_like(cmax, dtype=np.float32)
    mask = delta > 1e-12
    idx = mask & (cmax == r2)
    hue[idx] = ((g2[idx] - b2[idx]) / (delta[idx] + 1e-12)) % 6.0
    idx = mask & (cmax == g2)
    hue[idx] = ((b2[idx] - r2[idx]) / (delta[idx] + 1e-12)) + 2.0
    idx = mask & (cmax == b2)
    hue[idx] = ((r2[idx] - g2[idx]) / (delta[idx] + 1e-12)) + 4.0
    hue = (hue / 6.0) % 1.0

    sat = np.zeros_like(cmax, dtype=np.float32)
    nonzero = cmax > 1e-12
    sat[nonzero] = delta[nonzero] / (cmax[nonzero] + 1e-12)

    val = cmax.astype(np.float32)

    hsv = np.stack([hue, sat, val], axis=-1)
    for ch in range(3):
        x = hsv[:, :, ch]
        feats.append(float(x.mean()))
        feats.append(float(x.std()))

    h2, w2 = h // 2, w // 2
    q1 = gray[:h2, :w2].mean()
    q2 = gray[:h2, w2:].mean()
    q3 = gray[h2:, :w2].mean()
    q4 = gray[h2:, w2:].mean()
    feats.extend([float(q1), float(q2), float(q3), float(q4)])

    bins = 8
    for x in (r, g, b):
        hist, _ = np.histogram(x, bins=bins, range=(0.0, 1.0), density=False)
        hist = hist.astype(np.float32)
        hist = hist / (hist.sum() + 1e-12)
        feats.extend([float(v) for v in hist])

    feats.append(float(hsv[:, :, 1].mean()))
    feats.append(float(hsv[:, :, 2].mean()))

    thumb = Image.fromarray(
        (np.clip(gray, 0.0, 1.0) * 255.0).astype("uint8"), mode="L"
    ).resize((16, 16))
    thumb_arr = (np.asarray(thumb, dtype=np.float32) / 255.0).reshape(-1)
    feats.extend([float(v) for v in thumb_arr])

    ang = (np.arctan2(gy, gx) + np.pi) / (2.0 * np.pi)  # [0,1]
    m = grad.reshape(-1)
    a = ang.reshape(-1)
    if float(m.sum()) <= 0.0:
        oh = np.zeros(8, dtype=np.float32)
    else:
        oh, _ = np.histogram(a, bins=8, range=(0.0, 1.0), weights=m, density=False)
        oh = oh.astype(np.float32)
        oh = oh / (oh.sum() + 1e-12)
    feats.extend([float(v) for v in oh])

    gsmall = Image.fromarray(
        (np.clip(gray, 0.0, 1.0) * 255.0).astype("uint8"), mode="L"
    ).resize((64, 64))
    gs = np.asarray(gsmall, dtype=np.uint8)

    c = gs[1:-1, 1:-1]
    lbp = np.zeros_like(c, dtype=np.uint8)
    lbp |= ((gs[0:-2, 0:-2] >= c) << 7).astype(np.uint8)
    lbp |= ((gs[0:-2, 1:-1] >= c) << 6).astype(np.uint8)
    lbp |= ((gs[0:-2, 2:] >= c) << 5).astype(np.uint8)
    lbp |= ((gs[1:-1, 2:] >= c) << 4).astype(np.uint8)
    lbp |= ((gs[2:, 2:] >= c) << 3).astype(np.uint8)
    lbp |= ((gs[2:, 1:-1] >= c) << 2).astype(np.uint8)
    lbp |= ((gs[2:, 0:-2] >= c) << 1).astype(np.uint8)
    lbp |= ((gs[1:-1, 0:-2] >= c) << 0).astype(np.uint8)

    hist16, _ = np.histogram(lbp.reshape(-1), bins=16, range=(0, 256), density=False)
    hist16 = hist16.astype(np.float32)
    hist16 = hist16 / (hist16.sum() + 1e-12)
    feats.extend([float(v) for v in hist16])

    hist32, _ = np.histogram(lbp.reshape(-1), bins=32, range=(0, 256), density=False)
    hist32 = hist32.astype(np.float32)
    hist32 = hist32 / (hist32.sum() + 1e-12)
    feats.extend([float(v) for v in hist32])

    g64 = gs.astype(np.float32) / 255.0
    gy2, gx2 = np.gradient(g64)
    mag2 = (gx2 * gx2 + gy2 * gy2) ** 0.5
    ang2 = (np.arctan2(gy2, gx2) + np.pi) / (2.0 * np.pi)  # [0,1]
    cells = 4
    bins_hog = 8
    ch = 64 // cells
    cw = 64 // cells
    for iy in range(cells):
        for ix in range(cells):
            y0, y1 = iy * ch, (iy + 1) * ch
            x0, x1 = ix * cw, (ix + 1) * cw
            mcell = mag2[y0:y1, x0:x1].reshape(-1)
            acell = ang2[y0:y1, x0:x1].reshape(-1)
            if float(mcell.sum()) <= 0.0:
                hcell = np.zeros(bins_hog, dtype=np.float32)
            else:
                hcell, _ = np.histogram(
                    acell, bins=bins_hog, range=(0.0, 1.0), weights=mcell, density=False
                )
                hcell = hcell.astype(np.float32)
                hcell = hcell / (hcell.sum() + 1e-12)
            feats.extend([float(v) for v in hcell])

    return np.asarray(feats, dtype=np.float32)


def _infer_feature_dim(train_csv_path, images_dir):
    """
    Stability: infer feature dimensionality from a real image so that fallback
    zero-vectors always match the true feature length.
    """
    train_df = pd.read_csv(train_csv_path)
    for img_id in train_df["image_id"].values:
        p = os.path.join(images_dir, f"{img_id}.jpg")
        if os.path.exists(p):
            try:
                return int(_extract_image_features(p).shape[0])
            except Exception:
                continue
    return 0


def train_and_predict_baseline(train_csv_path, test_csv_path, images_dir):
    import numpy as np
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler, FunctionTransformer
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score

    train_df = pd.read_csv(train_csv_path)
    test_df = pd.read_csv(test_csv_path)

    targets = ["healthy", "multiple_diseases", "rust", "scab"]

    feat_dim = _infer_feature_dim(train_csv_path, images_dir)
    if feat_dim <= 0:
        raise RuntimeError("Could not infer feature dimension from training images.")

    def safe_extract(path, do_hflip=False, do_vflip=False):
        try:
            return _extract_image_features(path, do_hflip=do_hflip, do_vflip=do_vflip)
        except Exception:
            return np.zeros(feat_dim, dtype=np.float32)

    X_train_0 = np.vstack(
        [
            safe_extract(
                os.path.join(images_dir, f"{img_id}.jpg"),
                do_hflip=False,
                do_vflip=False,
            )
            for img_id in train_df["image_id"].values
        ]
    )
    X_train_1 = np.vstack(
        [
            safe_extract(
                os.path.join(images_dir, f"{img_id}.jpg"), do_hflip=True, do_vflip=False
            )
            for img_id in train_df["image_id"].values
        ]
    )
    X_train_2 = np.vstack(
        [
            safe_extract(
                os.path.join(images_dir, f"{img_id}.jpg"), do_hflip=False, do_vflip=True
            )
            for img_id in train_df["image_id"].values
        ]
    )
    X_train_3 = np.vstack(
        [
            safe_extract(
                os.path.join(images_dir, f"{img_id}.jpg"), do_hflip=True, do_vflip=True
            )
            for img_id in train_df["image_id"].values
        ]
    )
    X_train = np.vstack([X_train_0, X_train_1, X_train_2, X_train_3])

    y_train_ovr = train_df[targets].values.astype(int)
    y_train_class = np.argmax(y_train_ovr, axis=1).astype(int)
    y_train = np.concatenate(
        [y_train_class, y_train_class, y_train_class, y_train_class], axis=0
    )

    X_test_0 = np.vstack(
        [
            safe_extract(
                os.path.join(images_dir, f"{img_id}.jpg"),
                do_hflip=False,
                do_vflip=False,
            )
            for img_id in test_df["image_id"].values
        ]
    )
    X_test_1 = np.vstack(
        [
            safe_extract(
                os.path.join(images_dir, f"{img_id}.jpg"),
                do_hflip=True,
                do_vflip=False,
            )
            for img_id in test_df["image_id"].values
        ]
    )
    X_test_2 = np.vstack(
        [
            safe_extract(
                os.path.join(images_dir, f"{img_id}.jpg"),
                do_hflip=False,
                do_vflip=True,
            )
            for img_id in test_df["image_id"].values
        ]
    )
    X_test_3 = np.vstack(
        [
            safe_extract(
                os.path.join(images_dir, f"{img_id}.jpg"),
                do_hflip=True,
                do_vflip=True,
            )
            for img_id in test_df["image_id"].values
        ]
    )

    def _data_driven_signed_log1p(X):
        """
        Keep your existing heavy-tail stabilization.
        """
        X = np.asarray(X, dtype=np.float32)
        Y = X.copy()

        absX = np.abs(X)
        p50 = np.percentile(absX, 50, axis=0)
        p95 = np.percentile(absX, 95, axis=0)
        ratio = p95 / (p50 + 1e-6)

        mask = ratio > 12.0
        if np.any(mask):
            block = Y[:, mask]
            Y[:, mask] = np.sign(block) * np.log1p(np.abs(block))
        return Y

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    base_lr = LogisticRegressionCV(
        Cs=[1.0, 2.0, 4.0, 8.0, 16.0],
        cv=cv,
        solver="lbfgs",
        multi_class="multinomial",
        penalty="l2",
        max_iter=8000,
        class_weight=None,
        random_state=42,
        n_jobs=-1,
        scoring="neg_log_loss",  # stable probability fit; we calibrate for AUC below
        refit=True,
    )

    clf = Pipeline(
        steps=[
            (
                "sel_log1p",
                FunctionTransformer(_data_driven_signed_log1p, validate=False),
            ),
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("lr", base_lr),
        ]
    )

    clf.fit(X_train, y_train)

    proba0 = clf.predict_proba(X_test_0)
    proba1 = clf.predict_proba(X_test_1)
    proba2 = clf.predict_proba(X_test_2)
    proba3 = clf.predict_proba(X_test_3)
    proba = (proba0 + proba1 + proba2 + proba3) / 4.0

    classes_ = clf.named_steps["lr"].classes_

    proba_full = np.full((X_test_0.shape[0], 4), 1e-6, dtype=np.float32)
    for j, cls in enumerate(classes_):
        cls = int(cls)
        if 0 <= cls < 4:
            proba_full[:, cls] = proba[:, j].astype(np.float32)

    proba_full = proba_full / (proba_full.sum(axis=1, keepdims=True) + 1e-12)
    proba_full = np.clip(proba_full, 1e-6, 1 - 1e-6).astype(np.float32)

    def mean_colwise_auc(y_true_onehot, y_pred_proba):
        aucs = []
        for k in range(4):
            yt = y_true_onehot[:, k]
            yp = y_pred_proba[:, k]
            if len(np.unique(yt)) < 2:
                continue
            aucs.append(roc_auc_score(yt, yp))
        return float(np.mean(aucs)) if aucs else 0.0

    y_onehot = np.eye(4, dtype=np.int32)[y_train_class]
    X0 = X_train_0
    y0 = y_train_class
    y0_onehot = np.eye(4, dtype=np.int32)[y0]

    alphas = [0.7, 0.85, 1.0, 1.15, 1.3]
    best_alpha = 1.0
    best_auc = -1.0

    oof = np.zeros((X0.shape[0], 4), dtype=np.float32)
    for tr_idx, va_idx in cv.split(X0, y0):
        fold_clf = Pipeline(
            steps=[
                (
                    "sel_log1p",
                    FunctionTransformer(_data_driven_signed_log1p, validate=False),
                ),
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                (
                    "lr",
                    LogisticRegressionCV(
                        Cs=[1.0, 2.0, 4.0, 8.0, 16.0],
                        cv=3,
                        solver="lbfgs",
                        multi_class="multinomial",
                        penalty="l2",
                        max_iter=8000,
                        class_weight=None,
                        random_state=42,
                        n_jobs=-1,
                        scoring="neg_log_loss",
                        refit=True,
                    ),
                ),
            ]
        )
        fold_clf.fit(X0[tr_idx], y0[tr_idx])
        p = fold_clf.predict_proba(X0[va_idx]).astype(np.float32)

        cls_fold = fold_clf.named_steps["lr"].classes_
        pf = np.full((p.shape[0], 4), 1e-6, dtype=np.float32)
        for j, c in enumerate(cls_fold):
            c = int(c)
            if 0 <= c < 4:
                pf[:, c] = p[:, j]
        pf = pf / (pf.sum(axis=1, keepdims=True) + 1e-12)
        oof[va_idx] = pf

    for a in alphas:
        pp = np.power(np.clip(oof, 1e-6, 1 - 1e-6), a)
        pp = pp / (pp.sum(axis=1, keepdims=True) + 1e-12)
        auc = mean_colwise_auc(y0_onehot, pp)
        if auc > best_auc:
            best_auc = auc
            best_alpha = float(a)

    proba_full = np.power(np.clip(proba_full, 1e-6, 1 - 1e-6), best_alpha)
    proba_full = proba_full / (proba_full.sum(axis=1, keepdims=True) + 1e-12)
    proba_full = np.clip(proba_full, 1e-6, 1 - 1e-6).astype(np.float32)

    print(f"Chosen alpha={best_alpha} (OOF mean AUC={best_auc:.6f})")
    return test_df["image_id"].values, proba_full




## === cell 6
if len(submissions_all) >= 2:
    submission_avg = ensemble(submissions_all, [0, 1], [0.3, 0.7])
    template_path = submissions_all[0]
    make_submission_file(submission_avg, template_path, out_path="submission.csv")
else:
    image_ids, proba = train_and_predict_baseline(
        TRAIN_CSV_PATH, TEST_CSV_PATH, IMAGES_DIR
    )

    sub = pd.read_csv(SAMPLE_SUB_PATH)
    required = ["image_id", "healthy", "multiple_diseases", "rust", "scab"]
    sub = sub[required].copy()

    pred_df = pd.DataFrame(
        {
            "image_id": image_ids,
            "healthy": proba[:, 0],
            "multiple_diseases": proba[:, 1],
            "rust": proba[:, 2],
            "scab": proba[:, 3],
        }
    )

    test = pd.read_csv(TEST_CSV_PATH)
    sub = test.merge(pred_df, on="image_id", how="left")
    for c in ["healthy", "multiple_diseases", "rust", "scab"]:
        sub[c] = sub[c].astype("float32").fillna(0.25)

    import numpy as np

    P = sub[["healthy", "multiple_diseases", "rust", "scab"]].values.astype(np.float32)
    P = np.clip(P, 1e-6, 1 - 1e-6)
    P = P / (P.sum(axis=1, keepdims=True) + 1e-12)
    sub[["healthy", "multiple_diseases", "rust", "scab"]] = P

    sub.to_csv("submission.csv", index=False)
    print(f"No external submissions found under {SUBMISSIONS_PATH}.")
    print(f"Wrote learned baseline submission.csv with shape {sub.shape}")
