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

DATA_ROOT = "/kaggle/input/plant-pathology-2020-fgvc7"
if not os.path.exists(DATA_ROOT):
    alt_root = "/kaggle/data/plant-pathology-2020-fgvc7"
    if os.path.exists(alt_root):
        DATA_ROOT = alt_root
if not os.path.exists(DATA_ROOT):
    alt_root2 = "/kaggle/data/plant-pathology-2020-fgvc7"
    if os.path.exists(alt_root2):
        DATA_ROOT = alt_root2

TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
TEST_CSV = os.path.join(DATA_ROOT, "test.csv")
SAMPLE_SUB_CSV = os.path.join(DATA_ROOT, "sample_submission.csv")

IMAGES_DIR = os.path.join(DATA_ROOT, "images")

TARGET_COLS = ["healthy", "multiple_diseases", "rust", "scab"]



## === cell 2
submissions_all = []
if os.path.exists(SUBMISSIONS_PATH):
    for dirname, _, filenames in os.walk(SUBMISSIONS_PATH):
        for filename in filenames:
            if filename.lower().endswith(".csv"):
                submissions_all.append(os.path.join(dirname, filename))

submissions_all = submissions_all[::-1]
print(f"Found {len(submissions_all)} submission files under {SUBMISSIONS_PATH}")
if len(submissions_all) > 0:
    print("First few:", submissions_all[:5])




## === cell 3
def ensemble(submissions_all, sub_idx, weights=None, target_cols=None):
    """
    Weighted ensembling of existing submission files.
    Guards against empty list / bad indices and enforces required columns.
    """
    if target_cols is None:
        target_cols = ["healthy", "multiple_diseases", "rust", "scab"]
    if weights is None:
        weights = [1.0] * len(sub_idx)
    if len(sub_idx) == 0:
        raise ValueError("sub_idx is empty; cannot ensemble.")
    if len(weights) != len(sub_idx):
        raise ValueError(
            f"weights length ({len(weights)}) must match sub_idx length ({len(sub_idx)})."
        )
    if len(submissions_all) == 0:
        raise ValueError("No submission files found to ensemble.")

    acc = None
    for i in range(len(sub_idx)):
        idx = sub_idx[i]
        if idx < 0 or idx >= len(submissions_all):
            raise IndexError(
                f"sub_idx contains {idx}, but submissions_all has length {len(submissions_all)}"
            )
        path = submissions_all[idx]
        w = float(weights[i])
        print(f"I'm taking submission {path} with weight {w}")
        df = pd.read_csv(path)

        missing = [c for c in target_cols if c not in df.columns]
        if missing:
            raise KeyError(
                f"Submission {path} missing columns: {missing}. Has columns: {list(df.columns)}"
            )

        vals = df.loc[:, target_cols].to_numpy(dtype=np.float64)
        if acc is None:
            acc = vals * w
        else:
            acc += vals * w

    wsum = float(np.sum(weights))
    if wsum > 0:
        acc = acc / wsum

    acc = np.clip(acc, 0.0, 1.0)
    return acc




## === cell 4
def make_submission_file(
    submission_avg, base_submission_path, out_path="submission.csv", target_cols=None
):
    """
    Writes a valid submission CSV using the sample submission as the template.
    Enforces column order and length match.
    """
    if target_cols is None:
        target_cols = ["healthy", "multiple_diseases", "rust", "scab"]

    sub_df = pd.read_csv(base_submission_path)
    if "image_id" not in sub_df.columns:
        raise KeyError(
            f"Base submission at {base_submission_path} must contain image_id column."
        )
    for c in target_cols:
        if c not in sub_df.columns:
            raise KeyError(
                f"Base submission at {base_submission_path} missing required column: {c}"
            )

    submission_avg = np.asarray(submission_avg, dtype=np.float64)
    if submission_avg.shape[0] != len(sub_df):
        raise ValueError(
            f"Prediction row count {submission_avg.shape[0]} != sample_submission rows {len(sub_df)}"
        )
    if submission_avg.shape[1] != len(target_cols):
        raise ValueError(
            f"Prediction col count {submission_avg.shape[1]} != {len(target_cols)} targets"
        )

    sub_df.loc[:, target_cols] = submission_avg
    sub_df.to_csv(out_path, index=False)
    print(
        f"Wrote {out_path} with shape {sub_df.shape} and columns {list(sub_df.columns)}"
    )




## === cell 5
def _safe_imports_for_image_model():
    """
    Keeps imports local so the script still runs even if optional deps are absent.
    """
    from PIL import Image  # pillow is typically available in Kaggle
    from PIL import ImageStat, ImageFilter
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score

    return (
        Image,
        ImageStat,
        ImageFilter,
        StandardScaler,
        LogisticRegression,
        StratifiedKFold,
        roc_auc_score,
    )


def _image_path(image_id, images_dir):
    image_id = str(image_id)
    if not image_id.lower().endswith(".jpg"):
        image_id = image_id + ".jpg"
    return os.path.join(images_dir, image_id)


def _extract_color_stats(image_path, Image, ImageStat, ImageFilter):
    """
    Fixed image-stat features used by the fallback image model.
    """
    with Image.open(image_path) as im:
        im = im.convert("RGB")

        st = ImageStat.Stat(im)
        mean_rgb = np.asarray(st.mean, dtype=np.float32) / 255.0
        std_rgb = np.asarray(st.stddev, dtype=np.float32) / 255.0

        hsv = im.convert("HSV")
        st_h = ImageStat.Stat(hsv)
        mean_hsv = np.asarray(st_h.mean, dtype=np.float32) / 255.0
        std_hsv = np.asarray(st_h.stddev, dtype=np.float32) / 255.0

        w, h = im.size
        x0, y0 = int(0.25 * w), int(0.25 * h)
        x1, y1 = int(0.75 * w), int(0.75 * h)
        im_c = im.crop((x0, y0, x1, y1))
        st_c = ImageStat.Stat(im_c)
        mean_rgb_c = np.asarray(st_c.mean, dtype=np.float32) / 255.0
        std_rgb_c = np.asarray(st_c.stddev, dtype=np.float32) / 255.0

        lab = im.convert("LAB")
        st_l = ImageStat.Stat(lab)
        mean_lab = np.asarray(st_l.mean, dtype=np.float32) / 255.0
        std_lab = np.asarray(st_l.stddev, dtype=np.float32) / 255.0

        gray = im.convert("L")
        edges = gray.filter(ImageFilter.FIND_EDGES)
        st_e = ImageStat.Stat(edges)
        mean_edge = float(st_e.mean[0]) / 255.0
        std_edge = float(st_e.stddev[0]) / 255.0

        hist = np.asarray(gray.histogram(), dtype=np.float32)  # 256 bins
        hist8 = hist.reshape(8, 32).sum(axis=1)
        hist8 = hist8 / (hist8.sum() + 1e-6)

        midx, midy = w // 2, h // 2
        quads = [
            im.crop((0, 0, midx, midy)),
            im.crop((midx, 0, w, midy)),
            im.crop((0, midy, midx, h)),
            im.crop((midx, midy, w, h)),
        ]
        q_mean = []
        q_std = []
        for q in quads:
            stq = ImageStat.Stat(q)
            q_mean.append(np.asarray(stq.mean, dtype=np.float32) / 255.0)
            q_std.append(np.asarray(stq.stddev, dtype=np.float32) / 255.0)
        q_mean = np.stack(q_mean, axis=0).astype(np.float32)  # (4,3)
        q_std = np.stack(q_std, axis=0).astype(np.float32)  # (4,3)
        q_delta = (q_mean - mean_rgb[None, :]).astype(np.float32)

        g_small = gray.resize((32, 32), resample=Image.BILINEAR)
        g = np.asarray(g_small, dtype=np.float32) / 255.0
        spot_std = float(g.std())
        gx = np.abs(g[:, 1:] - g[:, :-1]).mean()
        gy = np.abs(g[1:, :] - g[:-1, :]).mean()
        spot_grad = float(0.5 * (gx + gy))

    eps = 1e-6
    r, gch, b = mean_rgb
    rg = r / (gch + eps)
    rb = r / (b + eps)
    gb = gch / (b + eps)

    delta_center = mean_rgb_c - mean_rgb

    green_minus_red = float(gch - r)
    green_ratio = float(gch / (r + eps))
    yellowing = float((r + gch) - 2.0 * b)
    sat_times_val = float(mean_hsv[1] * mean_hsv[2])
    hue_spread = float(std_hsv[0])

    feat = np.concatenate(
        [
            mean_rgb,
            std_rgb,
            mean_hsv,
            std_hsv,
            mean_rgb_c,
            std_rgb_c,
            delta_center,
            np.array([rg, rb, gb], dtype=np.float32),
            mean_lab,
            std_lab,
            np.array([mean_edge, std_edge], dtype=np.float32),
            hist8.astype(np.float32),
            q_mean.reshape(-1),
            q_std.reshape(-1),
            q_delta.reshape(-1),
            np.array([spot_std, spot_grad], dtype=np.float32),
            np.array(
                [green_minus_red, green_ratio, yellowing, sat_times_val, hue_spread],
                dtype=np.float32,
            ),
        ],
        axis=0,
    )
    return feat.astype(np.float32)


def _build_feature_cache_signature(paths):
    sig = np.empty((len(paths), 2), dtype=np.int64)
    for i, p in enumerate(paths):
        st = os.stat(p)
        sig[i, 0] = int(st.st_size)
        sig[i, 1] = int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9)))
    return sig


def _load_or_compute_features(
    image_ids, images_dir, cache_prefix, Image, ImageStat, ImageFilter, n_features
):
    paths = [_image_path(iid, images_dir) for iid in image_ids]
    for p in paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing image: {p}")

    cache_dir = "/kaggle/working"
    feat_path = os.path.join(cache_dir, f"{cache_prefix}_X.npy")
    sig_path = os.path.join(cache_dir, f"{cache_prefix}_sig.npy")

    sig = _build_feature_cache_signature(paths)

    if os.path.exists(feat_path) and os.path.exists(sig_path):
        try:
            old_sig = np.load(sig_path, allow_pickle=False)
            if old_sig.shape == sig.shape and np.array_equal(old_sig, sig):
                X = np.load(feat_path, allow_pickle=False)
                if X.shape == (len(image_ids), n_features) and X.dtype == np.float32:
                    return X
        except Exception:
            pass  # fall back to recompute safely

    X = np.zeros((len(image_ids), n_features), dtype=np.float32)
    for i, p in enumerate(paths):
        X[i] = _extract_color_stats(p, Image, ImageStat, ImageFilter)

    np.save(feat_path, X, allow_pickle=False)
    np.save(sig_path, sig, allow_pickle=False)
    return X


def _make_stratify_labels_for_cv(X, y, n_bins=10, min_count=5):
    y = np.asarray(y, dtype=np.int32)
    edge_mean = X[:, 30].astype(np.float32)
    qs = np.quantile(edge_mean, np.linspace(0, 1, n_bins + 1))
    qs = np.unique(qs)
    if qs.shape[0] < 3:
        return y
    bins = np.digitize(edge_mean, qs[1:-1], right=False).astype(np.int32)
    joint = (y.astype(np.int32) * 100 + bins).astype(np.int32)

    _, counts = np.unique(joint, return_counts=True)
    if counts.min() < int(min_count):
        return y
    return joint


def _select_C_via_cv(
    X, y, LogisticRegression, StratifiedKFold, roc_auc_score, random_state=42
):
    C_grid = [0.1, 0.3, 1.0, 3.0, 10.0]

    strat_y = _make_stratify_labels_for_cv(X, y, n_bins=10, min_count=5)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    best_C = 1.0
    best_score = -1.0

    for C in C_grid:
        fold_scores = []
        for tr_idx, va_idx in skf.split(X, strat_y):
            Xtr, Xva = X[tr_idx], X[va_idx]
            ytr, yva = y[tr_idx], y[va_idx]

            clf = LogisticRegression(
                solver="lbfgs",
                max_iter=800,
                C=float(C),
                class_weight="balanced",
                random_state=random_state,
                n_jobs=-1,
                multi_class="ovr",
            )
            clf.fit(Xtr, ytr)
            p = clf.predict_proba(Xva)[:, 1]
            fold_scores.append(roc_auc_score(yva, p))
        m = float(np.mean(fold_scores))
        if m > best_score:
            best_score = m
            best_C = float(C)

    return best_C, best_score


def _logit(p, eps=1e-6):
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def _sigmoid(z):
    z = np.asarray(z, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-z))


def _fit_logit_scale_bias_to_auc(z, y, roc_auc_score):
    """
    Change (score-improving, minimal): fit monotonic logit scaling (a*z + b) to
    maximize ROC AUC on OOF predictions (ranking metric), rather than minimizing log-loss.
    This preserves the same model/probability semantics while directly aligning with the metric.
    """
    y = np.asarray(y, dtype=np.int32)
    z = np.asarray(z, dtype=np.float64)

    a_grid = np.array([0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0], dtype=np.float64)
    b_grid = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float64)

    best = (-np.inf, 1.0, 0.0)
    for a in a_grid:
        for b in b_grid:
            p = _sigmoid(a * z + b)
            s = roc_auc_score(y, p)
            if s > best[0]:
                best = (float(s), float(a), float(b))

    _, a_best, b_best = best
    return a_best, b_best


def baseline_from_simple_image_model(
    train_csv_path,
    test_csv_path,
    sample_sub_path,
    images_dir,
    target_cols=None,
    random_state=42,
):
    """
    Fallback submission from a simple image-stat model.
    Keeps the same per-label logistic regression; uses OOF-based monotonic logit scaling
    fitted to ROC AUC (ranking-aligned) to reduce saturation and improve ordering.
    """
    if target_cols is None:
        target_cols = ["healthy", "multiple_diseases", "rust", "scab"]

    (
        Image,
        ImageStat,
        ImageFilter,
        StandardScaler,
        LogisticRegression,
        StratifiedKFold,
        roc_auc_score,
    ) = _safe_imports_for_image_model()

    train_df = pd.read_csv(train_csv_path)
    test_df = pd.read_csv(test_csv_path)
    sample_df = pd.read_csv(sample_sub_path)

    test_ids_order = test_df["image_id"].astype(str).tolist()
    train_ids = train_df["image_id"].astype(str).tolist()

    n_features = 83

    X_train = _load_or_compute_features(
        train_ids,
        images_dir,
        "pp2020_train_v8",  # cache bump so new calibration uses consistent features
        Image,
        ImageStat,
        ImageFilter,
        n_features,
    )
    X_test = _load_or_compute_features(
        test_ids_order,
        images_dir,
        "pp2020_test_v8",
        Image,
        ImageStat,
        ImageFilter,
        n_features,
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    preds = np.zeros((len(test_ids_order), len(target_cols)), dtype=np.float64)

    for j, c in enumerate(target_cols):
        if c not in train_df.columns:
            raise KeyError(
                f"Train CSV missing target column {c}. Has columns: {list(train_df.columns)}"
            )
        y = train_df[c].to_numpy(dtype=np.int32)

        if y.min() == y.max():
            prior = float(np.clip(y.mean(), 1e-6, 1 - 1e-6))
            preds[:, j] = prior
            continue

        best_C, best_cv_auc = _select_C_via_cv(
            X_train_s,
            y,
            LogisticRegression,
            StratifiedKFold,
            roc_auc_score,
            random_state=random_state,
        )
        print(f"[{c}] selected C={best_C} (CV AUC ~ {best_cv_auc:.5f})")

        strat_y = _make_stratify_labels_for_cv(X_train_s, y, n_bins=10, min_count=5)
        skf_oof = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

        oof_p = np.zeros_like(y, dtype=np.float64)
        for tr_idx, va_idx in skf_oof.split(X_train_s, strat_y):
            clf_fold = LogisticRegression(
                solver="lbfgs",
                max_iter=800,
                C=best_C,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=-1,
                multi_class="ovr",
            )
            clf_fold.fit(X_train_s[tr_idx], y[tr_idx])
            oof_p[va_idx] = clf_fold.predict_proba(X_train_s[va_idx])[:, 1]

        a, b = _fit_logit_scale_bias_to_auc(_logit(oof_p), y, roc_auc_score)
        print(f"[{c}] AUC-calibration grid: scale a={a:.4f}, bias b={b:.4f}")

        clf = LogisticRegression(
            solver="lbfgs",
            max_iter=800,
            C=best_C,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
            multi_class="ovr",
        )
        clf.fit(X_train_s, y)
        p_test = clf.predict_proba(X_test_s)[:, 1]

        p_test_cal = _sigmoid(a * _logit(p_test) + b)
        preds[:, j] = p_test_cal

    preds = np.clip(preds, 1e-6, 1.0 - 1e-6)

    sample_ids = sample_df["image_id"].astype(str).tolist()
    if sample_ids != test_ids_order:
        idx_map = {img_id: i for i, img_id in enumerate(test_ids_order)}
        reorder_idx = [idx_map[iid] for iid in sample_ids]
        preds = preds[reorder_idx]

    return preds




## === cell 6
if len(submissions_all) >= 2:
    submission_avg = ensemble(
        submissions_all, [0, 1], weights=[0.7, 0.3], target_cols=TARGET_COLS
    )
    make_submission_file(
        submission_avg,
        base_submission_path=SAMPLE_SUB_CSV,
        out_path="submission.csv",
        target_cols=TARGET_COLS,
    )
elif len(submissions_all) == 1:
    submission_avg = ensemble(
        submissions_all, [0], weights=[1.0], target_cols=TARGET_COLS
    )
    make_submission_file(
        submission_avg,
        base_submission_path=SAMPLE_SUB_CSV,
        out_path="submission.csv",
        target_cols=TARGET_COLS,
    )
else:
    print(
        "No external submissions found; creating fallback submission from a simple image-stat model."
    )
    submission_avg = baseline_from_simple_image_model(
        TRAIN_CSV,
        TEST_CSV,
        SAMPLE_SUB_CSV,
        images_dir=IMAGES_DIR,
        target_cols=TARGET_COLS,
        random_state=42,
    )
    make_submission_file(
        submission_avg,
        base_submission_path=SAMPLE_SUB_CSV,
        out_path="submission.csv",
        target_cols=TARGET_COLS,
    )
