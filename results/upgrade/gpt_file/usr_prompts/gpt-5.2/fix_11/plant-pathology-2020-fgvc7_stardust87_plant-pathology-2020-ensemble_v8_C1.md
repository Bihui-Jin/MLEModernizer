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

0.9697621918404656

# 6. Current score

0.63805

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'The crash happens because the notebook assumes `/kaggle/input/submissions/submissions/` exists and contains at least two CSVs to ensemble; in this environment it doesn’t, so `submissions_all` is empty and indexing fails. I keep the ensembling logic, but add a safe fallback: if no external submissions are found, build a simple valid baseline submission from `sample_submission.csv` (uniform class probabilities), ensuring the pipeline always writes `submission.csv`. I also make the ensembling function validate indices/weights and normalize weights so the averaged probabilities stay in a sensible range. This run end-to-end and produce a correctly formatted `.csv` submission file.'
- What this solution (achieved 0.68088) has done: 'Your current 0.5 score comes from writing a uniform-probability fallback submission because no external submissions exist; to move toward the 0.9697 target, we need a real model prediction while keeping the “ensemble if available, else fallback” core flow intact. I add a minimal classical ML baseline that trains on the provided `train.csv` labels using simple image color statistics (no deep learning, no new dependencies) and predicts probabilities for `test.csv`. This should substantially lift ROC AUC above 0.5 while remaining fast (<600s) and producing a correctly formatted `submission.csv`. The existing ensembling logic stays; we just replace the uniform fallback with this learned baseline.'
- What this solution (achieved 0.70454) has done: 'Your current baseline is limited by very low-capacity features; to move the score upward toward 0.9697 without changing the overall approach (extract simple per-image features → OneVsRest LogisticRegression → predict_proba → submission), I minimally strengthen the feature extractor while keeping it lightweight and deterministic. Specifically, I (1) increase resize resolution modestly, (2) add a small set of additional robust features (HSV moments, coarse spatial grid means, and simple color histograms) that significantly improve separability for these leaf disease classes, and (3) switch the scaler to `RobustScaler` to reduce sensitivity to outliers, while keeping the same classifier family and training flow. The ensembling path remains unchanged; only the learned-fallback baseline is improved, and it still writes a valid `submission.csv` with the required columns and test ordering.'
- What this solution (achieved 0.60288) has done: 'To move your score upward toward the 0.9698 target without changing the core “hand-crafted image features → OneVsRest LogisticRegression → predict_proba → submission” approach, I make two minimal, high-leverage adjustments: add a small set of shape/texture features (downsampled grayscale pixels + simple edge orientation histogram) and slightly strengthen the LogisticRegression regularization/iterations for better fit. These keep the same model family and training flow, but increase separability versus the current low-capacity statistics-only feature vector. I also make the feature length robust by dynamically inferring the feature dimension from a real image (instead of returning a hard-coded zero-vector length on failure), preventing silent dimension mismatch and improving stability. The submission writing logic and column/order alignment remain identical.'
- What this solution (achieved 0.62213) has done: 'Your current score (0.60288) is far below the target (0.96976), so we should legitimately increase model signal while keeping the same “hand-crafted image features → OneVsRest LogisticRegression → predict_proba → submission” core. The smallest high-leverage change is to add a compact, classic texture descriptor (LBP histogram) to the existing feature vector, which tends to help leaf disease separability with minimal extra compute and no new dependencies. I also slightly strengthen the classifier (same family) by using `class_weight="balanced"` to counter label imbalance, which often improves mean ROC AUC without changing evaluation semantics. Submission writing/order remains identical.'
- What this solution (achieved 0.62542) has done: 'Your current score (0.62213) is far below the target (0.96976), so we should increase it with minimal, safe changes while preserving the same overall pipeline (hand-crafted image features → OneVsRest LogisticRegression → predict_proba → submission). The biggest issue is that the model is being trained as multilabel (4 independent binaries), but this competition’s labels are mutually exclusive (exactly one class per image), which makes OneVsRest poorly aligned and caps AUC. Without changing the “LogisticRegression on extracted features” core, we switch to a single multinomial LogisticRegression trained on the argmax class label, then use its class probabilities as the 4 submission columns. We also add a tiny amount of deterministic feature standardization stability (keep RobustScaler) and ensure class order matches submission columns exactly.'
- What this solution (achieved 0.66746) has done: 'Your current score (0.62542) is far below the target (0.96976), so we should increase it with minimal, low-risk changes while keeping the same pipeline (hand-crafted image features → multinomial LogisticRegression → predict_proba → submission). The biggest gain per line of change is to improve feature scaling/model fit stability and slightly enrich color/texture information without changing the overall approach: we add a tiny, deterministic set of extra features (normalized rg chromaticity stats + a few more LBP bins) and switch from `RobustScaler` to `StandardScaler` which is typically better matched to LogisticRegression on dense continuous features. We also increase `C` modestly (less regularization) and set `n_jobs=-1` to ensure convergence within time, keeping the same solver and training flow. Submission writing, column order, and test-row alignment remain unchanged.'
- What this solution (achieved 0.65237) has done: 'Your current gap to the target (0.66746 → 0.96976) is large, so we should legitimately increase model signal while keeping the same overall pipeline: hand-crafted image features → multinomial LogisticRegression → submission. The smallest high-leverage change is to add a compact set of oriented-gradient (HOG-like) features from the same grayscale image, which improves texture/lesion separability without introducing new dependencies or changing the training approach. I also make one stability fix: ensure the predicted class-probability columns are always aligned to the competition’s fixed class order (healthy, multiple_diseases, rust, scab), avoiding silent misalignment when some classes are absent in training folds/environments. Everything else (paths, solver family, training flow, submission writing) is preserved and it still writes a valid `submission.csv` end-to-end.'
- What this solution (achieved 0.66289) has done: 'Your current score (0.652) is far below the target (0.9698), so we should increase it while keeping your existing pipeline (hand-crafted image features → multinomial LogisticRegression → submission) intact. The highest-leverage minimal change is to fix feature scaling/feature dominance by (1) standardizing histogram features consistently (use `density=False` + L1 normalization) and (2) adding a small `log1p` transform to heavy-tailed gradient/hist-like features (still the same features, just stabilized), which typically improves LogisticRegression calibration and separability. I also add a deterministic train-time augmentation that doesn’t change the training loop (simply duplicate each training sample with a horizontal flip) to improve generalization with minimal code and no new dependencies. Finally, I keep the strict class-order alignment and ensure the produced `submission.csv` matches `test.csv` order and required columns.'
- What this solution (achieved 0.63805) has done: 'Your current score is far below the target, so we should legitimately increase signal while keeping the same overall pipeline (hand-crafted features → multinomial LogisticRegression → submission) intact. The smallest high-leverage change is to fix the feature transform: `log1p(max(X,0))` is currently distorting many informative features that are naturally in [0,1] (thumb pixels, normalized histograms), so we instead apply a safer signed `log1p` only to the heavy-tailed gradient/texture blocks. We also add one very small, deterministic augmentation (vertical flip) in the same “duplicate samples” approach you already use (no new training loop), which typically improves generalization for leaf images. Finally, we keep strict class order alignment and ensure the submission remains properly normalized and clipped.'

# 9. Code solution

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

    Minimal score-improving adjustments:
      - Keep the same feature families, but allow an additional deterministic vertical flip
        (used only for train-time duplication) to improve generalization toward target.
      - Do not change feature definitions/values otherwise (preserve core logic).
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

    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]
    eps = 1e-6

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

    hsv = np.asarray(img.convert("HSV"), dtype=np.float32) / 255.0
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

    thumb = Image.fromarray((gray * 255.0).astype("uint8"), mode="L").resize((16, 16))
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

    gsmall = Image.fromarray((gray * 255.0).astype("uint8"), mode="L").resize((64, 64))
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
    from sklearn.linear_model import LogisticRegression

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
    X_train = np.vstack([X_train_0, X_train_1, X_train_2])

    y_train_ovr = train_df[targets].values.astype(int)
    y_train_class = np.argmax(y_train_ovr, axis=1).astype(int)
    y_train = np.concatenate([y_train_class, y_train_class, y_train_class], axis=0)

    X_test = np.vstack(
        [
            safe_extract(
                os.path.join(images_dir, f"{img_id}.jpg"),
                do_hflip=False,
                do_vflip=False,
            )
            for img_id in test_df["image_id"].values
        ]
    )

    def _selective_signed_log1p(X):
        X = np.asarray(X, dtype=np.float32)
        Y = X.copy()

        idx_slices = [
            slice(313, 321),
            slice(321, 337),
            slice(337, 369),
            slice(369, 497),
        ]
        for s in idx_slices:
            block = Y[:, s]
            Y[:, s] = np.sign(block) * np.log1p(np.abs(block))
        return Y

    clf = Pipeline(
        steps=[
            ("sel_log1p", FunctionTransformer(_selective_signed_log1p, validate=False)),
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            (
                "lr",
                LogisticRegression(
                    solver="lbfgs",
                    multi_class="multinomial",
                    C=6.0,
                    max_iter=3000,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    clf.fit(X_train, y_train)

    proba = clf.predict_proba(X_test)
    classes_ = clf.named_steps["lr"].classes_

    proba_full = np.full((X_test.shape[0], 4), 1e-6, dtype=np.float32)
    for j, cls in enumerate(classes_):
        cls = int(cls)
        if 0 <= cls < 4:
            proba_full[:, cls] = proba[:, j].astype(np.float32)

    proba_full = proba_full / (proba_full.sum(axis=1, keepdims=True) + 1e-12)
    proba_full = np.clip(proba_full, 1e-6, 1 - 1e-6).astype(np.float32)

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
