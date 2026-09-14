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
SUBMISSIONS_PATH = "/kaggle/input/submissions/"

DATA_DIR_CANDIDATES = [
    "/kaggle/input/plant-pathology-2020-fgvc7",
    "/kaggle/data/plant-pathology-2020-fgvc7",
    "/kaggle/input",
    "/kaggle/data",
]


def _first_existing_file(rel_path: str):
    for base in DATA_DIR_CANDIDATES:
        p = os.path.join(base, rel_path)
        if os.path.exists(p):
            return p
    return None


SAMPLE_SUB_PATH = _first_existing_file("sample_submission.csv") or _first_existing_file(
    "plant-pathology-2020-fgvc7/sample_submission.csv"
)
TEST_CSV_PATH = _first_existing_file("test.csv") or _first_existing_file(
    "plant-pathology-2020-fgvc7/test.csv"
)
TRAIN_CSV_PATH = _first_existing_file("train.csv") or _first_existing_file(
    "plant-pathology-2020-fgvc7/train.csv"
)

IMAGES_DIR = _first_existing_file("images") or _first_existing_file(
    "plant-pathology-2020-fgvc7/images"
)

if SAMPLE_SUB_PATH is None or TEST_CSV_PATH is None or TRAIN_CSV_PATH is None:
    raise FileNotFoundError(
        "Could not locate sample_submission.csv, test.csv, and/or train.csv in expected Kaggle input paths."
    )
if IMAGES_DIR is None:
    raise FileNotFoundError(
        "Could not locate images/ directory in expected Kaggle input paths."
    )



## === cell 2
submissions_all = []
if os.path.exists(SUBMISSIONS_PATH):
    for dirname, _, filenames in os.walk(SUBMISSIONS_PATH):
        for filename in filenames:
            if filename.lower().endswith(".csv"):
                submissions_all.append(os.path.join(dirname, filename))
submissions_all.sort()
print("Found external submissions:", submissions_all)




## === cell 3
def ensemble(submissions_all, sub_idx, weights=[]):
    if len(sub_idx) == 0:
        raise ValueError("sub_idx is empty.")
    if len(weights) != len(sub_idx):
        raise ValueError(
            f"weights length ({len(weights)}) must equal sub_idx length ({len(sub_idx)})."
        )
    if len(submissions_all) == 0:
        raise ValueError("submissions_all is empty; no submissions to ensemble.")
    if max(sub_idx) >= len(submissions_all) or min(sub_idx) < 0:
        raise IndexError(
            f"sub_idx {sub_idx} out of range for submissions_all of length {len(submissions_all)}"
        )

    base = pd.read_csv(submissions_all[sub_idx[0]])
    if "image_id" not in base.columns:
        raise ValueError(
            f"'image_id' column not found in {submissions_all[sub_idx[0]]}"
        )
    base_ids = base["image_id"].astype(str).values

    submission_with_weight = []
    for i in range(len(sub_idx)):
        path = submissions_all[sub_idx[i]]
        w = weights[i]
        print(f"I'm taking submission {path} with weight {w}")
        df = pd.read_csv(path)

        missing = set(
            ["image_id", "healthy", "multiple_diseases", "rust", "scab"]
        ) - set(df.columns)
        if missing:
            raise ValueError(f"Submission {path} missing columns: {sorted(missing)}")

        df["image_id"] = df["image_id"].astype(str)
        df = df.set_index("image_id").reindex(base_ids)
        if df.isna().any().any():
            raise ValueError(
                f"Submission {path} could not be aligned to base image_id order (missing ids)."
            )

        arr = df.loc[:, ["healthy", "multiple_diseases", "rust", "scab"]].values
        submission_with_weight.append(arr * w)

    submission_avg = sum(submission_with_weight)
    return submission_avg, base_ids




## === cell 4
def make_submission_file(submission_avg, image_ids):
    submission_df = pd.read_csv(SAMPLE_SUB_PATH)
    submission_df["image_id"] = submission_df["image_id"].astype(str)

    test_df = pd.read_csv(TEST_CSV_PATH)
    test_ids = test_df["image_id"].astype(str).values

    if image_ids is not None:
        if len(image_ids) != len(test_ids) or (image_ids != test_ids).any():
            tmp = pd.DataFrame({"image_id": image_ids})
            tmp[["healthy", "multiple_diseases", "rust", "scab"]] = submission_avg
            tmp = tmp.set_index("image_id").reindex(test_ids).reset_index()
            if tmp.isna().any().any():
                raise ValueError(
                    "Ensembled predictions could not be aligned to test image_id list."
                )
            submission_df = tmp
        else:
            submission_df = pd.DataFrame(
                {
                    "image_id": test_ids,
                    "healthy": submission_avg[:, 0],
                    "multiple_diseases": submission_avg[:, 1],
                    "rust": submission_avg[:, 2],
                    "scab": submission_avg[:, 3],
                }
            )
    else:
        submission_df = (
            submission_df.set_index("image_id").reindex(test_ids).reset_index()
        )

    submission_df.to_csv("submission.csv", index=False)
    print("Wrote submission.csv with shape:", submission_df.shape)
    print("Columns:", submission_df.columns.tolist())
    print(submission_df.head())




## === cell 5
def _image_path(image_id: str) -> str:
    p = os.path.join(IMAGES_DIR, f"{image_id}.jpg")
    if not os.path.exists(p):
        raise FileNotFoundError(f"Image not found: {p}")
    return p


def _extract_features(image_ids):
    from PIL import Image, ImageEnhance
    import numpy as np

    def _resize(img, size):
        return img.resize(size, resample=Image.BILINEAR)

    def _avg_pool2(x, k=2):
        H, W = x.shape
        H2, W2 = (H // k) * k, (W // k) * k
        x = x[:H2, :W2]
        x = x.reshape(H2 // k, k, W2 // k, k).mean(axis=(1, 3))
        return x

    def _laplacian_variance(gray):
        g = gray
        gp = np.pad(g, ((1, 1), (1, 1)), mode="reflect")
        lap = (
            -4.0 * gp[1:-1, 1:-1]
            + gp[:-2, 1:-1]
            + gp[2:, 1:-1]
            + gp[1:-1, :-2]
            + gp[1:-1, 2:]
        )
        return float(lap.var())

    def _lbp_hist(gray_uint8, n_bins=16):
        g = gray_uint8
        gp = np.pad(g, ((1, 1), (1, 1)), mode="edge")
        c = gp[1:-1, 1:-1]
        code = np.zeros_like(c, dtype=np.uint16)
        code |= (gp[:-2, :-2] >= c).astype(np.uint16) << 0
        code |= (gp[:-2, 1:-1] >= c).astype(np.uint16) << 1
        code |= (gp[:-2, 2:] >= c).astype(np.uint16) << 2
        code |= (gp[1:-1, 2:] >= c).astype(np.uint16) << 3
        code |= (gp[2:, 2:] >= c).astype(np.uint16) << 4
        code |= (gp[2:, 1:-1] >= c).astype(np.uint16) << 5
        code |= (gp[2:, :-2] >= c).astype(np.uint16) << 6
        code |= (gp[1:-1, :-2] >= c).astype(np.uint16) << 7
        folded = (code % n_bins).astype(np.int32).ravel()
        hist = np.bincount(folded, minlength=n_bins).astype(np.float32)
        hist /= hist.sum() + 1e-8
        return hist

    def _radial_fft_features(gray_64, n_bins=12):
        g = gray_64.astype(np.float32)
        g = g - g.mean()
        F = np.fft.fft2(g)
        F = np.fft.fftshift(F)
        P = (F.real * F.real + F.imag * F.imag).astype(np.float32)

        H, W = P.shape
        yy = np.arange(H, dtype=np.float32) - (H / 2.0)
        xx = np.arange(W, dtype=np.float32) - (W / 2.0)
        Y, X = np.meshgrid(yy, xx, indexing="ij")
        R = np.sqrt(Y * Y + X * X)

        maxR = float(R.max() + 1e-6)
        edges = np.linspace(0.0, maxR, n_bins + 1, dtype=np.float32)

        feats = np.zeros((n_bins,), dtype=np.float32)
        total = float(P.sum() + 1e-8)
        for i in range(n_bins):
            m = (R >= edges[i]) & (R < edges[i + 1])
            feats[i] = float(P[m].sum()) / total
        return feats

    def _oriented_grad_energy(gray_64, angles=(0, 45, 90, 135)):
        g = gray_64.astype(np.float32)
        gx = np.diff(g, axis=1, prepend=g[:, :1])
        gy = np.diff(g, axis=0, prepend=g[:1, :])
        mag = np.sqrt(gx * gx + gy * gy) + 1e-8
        ang = np.arctan2(gy, gx)  # [-pi, pi]

        out = []
        for a in angles:
            aa = np.deg2rad(a)
            w = np.cos(ang - aa)
            w = np.maximum(w, 0.0)
            out.append(float((mag * w).mean()))
            out.append(float((mag * w).std()))
        return np.array(out, dtype=np.float32)

    def _features_from_pil_rgb(img_rgb):
        img = _resize(img_rgb, (128, 128))
        x = (np.asarray(img, dtype=np.float32) / 255.0).astype(np.float32)  # (H,W,3)

        ch_mean = x.mean(axis=(0, 1))
        ch_std = x.std(axis=(0, 1))
        sat = (x.max(axis=2) - x.min(axis=2)).mean()
        bright = x.mean()

        hsv = np.asarray(img.convert("HSV"), dtype=np.float32)
        h = (hsv[:, :, 0] / 255.0).astype(np.float32)
        s = (hsv[:, :, 1] / 255.0).astype(np.float32)
        v = (hsv[:, :, 2] / 255.0).astype(np.float32)

        h_bins, s_bins, v_bins = 12, 4, 4  # 192 dims
        hist, _ = np.histogramdd(
            np.stack([h.ravel(), s.ravel(), v.ravel()], axis=1),
            bins=(h_bins, s_bins, v_bins),
            range=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
        )
        hist = hist.astype(np.float32).ravel()
        hist /= hist.sum() + 1e-8

        gray = x.mean(axis=2).astype(np.float32)

        gx = np.diff(gray, axis=1, prepend=gray[:, :1])
        gy = np.diff(gray, axis=0, prepend=gray[:1, :])
        gmag = np.sqrt(gx * gx + gy * gy)
        g_mean = gmag.mean()
        g_std = gmag.std()
        g_p90 = np.quantile(gmag, 0.90)

        gray2 = _avg_pool2(gray, k=2)
        gx2 = np.diff(gray2, axis=1, prepend=gray2[:, :1])
        gy2 = np.diff(gray2, axis=0, prepend=gray2[:1, :])
        gmag2 = np.sqrt(gx2 * gx2 + gy2 * gy2)
        g2_mean = gmag2.mean()
        g2_std = gmag2.std()
        g2_p90 = np.quantile(gmag2, 0.90)

        lap_var = _laplacian_variance(gray)

        small = Image.fromarray((gray * 255.0).astype("uint8")).resize(
            (16, 16), resample=Image.BILINEAR
        )
        small = (np.asarray(small, dtype=np.float32) / 255.0).ravel()

        gray_lbp = Image.fromarray((gray * 255.0).astype("uint8")).resize(
            (64, 64), resample=Image.BILINEAR
        )
        gray_lbp = np.asarray(gray_lbp, dtype=np.uint8)
        lbp_hist = _lbp_hist(gray_lbp, n_bins=16)

        gray_64 = (gray_lbp.astype(np.float32) / 255.0).astype(np.float32)
        fft_rad = _radial_fft_features(gray_64, n_bins=12)
        og = _oriented_grad_energy(gray_64, angles=(0, 45, 90, 135))

        return np.concatenate(
            [
                ch_mean,
                ch_std,
                np.array([sat, bright], dtype=np.float32),
                hist,
                np.array(
                    [g_mean, g_std, g_p90, g2_mean, g2_std, g2_p90, lap_var],
                    dtype=np.float32,
                ),
                lbp_hist,
                fft_rad,
                og,
                small,
            ],
            axis=0,
        ).astype(np.float32)

    feats = []
    tta_specs = [
        ("orig", 1.0, 1.0, False),  # (name, brightness, contrast, hflip)
        ("hflip", 1.0, 1.0, True),
        ("b_up", 1.08, 1.0, False),
        ("c_up", 1.0, 1.08, False),
    ]

    pil_cache = {}

    for iid in image_ids:
        if iid not in pil_cache:
            pil_cache[iid] = Image.open(_image_path(iid)).convert("RGB")
        img0 = pil_cache[iid]

        f_acc = None
        for _, b, c, do_flip in tta_specs:
            img_t = img0
            if do_flip:
                img_t = img_t.transpose(Image.FLIP_LEFT_RIGHT)
            if b != 1.0:
                img_t = ImageEnhance.Brightness(img_t).enhance(b)
            if c != 1.0:
                img_t = ImageEnhance.Contrast(img_t).enhance(c)

            f = _features_from_pil_rgb(img_t)
            if f_acc is None:
                f_acc = f
            else:
                f_acc = f_acc + f
        f_acc = (f_acc / float(len(tta_specs))).astype(np.float32)
        feats.append(f_acc)

    return np.vstack(feats).astype(np.float32)


def train_and_predict_image_baseline():
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score

    train_df = pd.read_csv(TRAIN_CSV_PATH)
    test_df = pd.read_csv(TEST_CSV_PATH)

    targets = ["healthy", "multiple_diseases", "rust", "scab"]
    X_train = _extract_features(train_df["image_id"].astype(str).tolist())
    y_train = train_df[targets].values.astype(int)
    X_test = _extract_features(test_df["image_id"].astype(str).tolist())

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    C_LIST = [1.0, 2.0, 4.0]

    n_classes = len(targets)
    n_splits = 5

    strat_labels = train_df[targets].astype(str).agg("_".join, axis=1).values
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof_raw = np.zeros((X_train_s.shape[0], n_classes), dtype=np.float32)
    oof_cal = np.zeros((X_train_s.shape[0], n_classes), dtype=np.float32)
    test_cal_accum = np.zeros((X_test_s.shape[0], n_classes), dtype=np.float32)
    test_raw_accum = np.zeros((X_test_s.shape[0], n_classes), dtype=np.float32)

    def _sigmoid_fit(x, y):
        x = np.clip(x.astype(np.float64), -20.0, 20.0).reshape(-1, 1)
        y = y.astype(int)
        lr1d = LogisticRegression(
            solver="lbfgs",
            max_iter=500,
            C=1.0,
            class_weight=None,
            random_state=42,
        )
        lr1d.fit(x, y)
        a = float(lr1d.coef_[0, 0])
        b = float(lr1d.intercept_[0])
        return a, b

    def _sigmoid_apply(x, a, b):
        z = a * x + b
        z = np.clip(z, -50.0, 50.0)
        return 1.0 / (1.0 + np.exp(-z))

    for fold, (tr_idx, va_idx) in enumerate(
        skf.split(X_train_s, strat_labels), start=1
    ):
        tr_score_acc = np.zeros((len(tr_idx), n_classes), dtype=np.float32)
        va_score_acc = np.zeros((len(va_idx), n_classes), dtype=np.float32)
        te_score_acc = np.zeros((X_test_s.shape[0], n_classes), dtype=np.float32)

        tr_proba_acc = np.zeros((len(tr_idx), n_classes), dtype=np.float32)
        va_proba_acc = np.zeros((len(va_idx), n_classes), dtype=np.float32)
        te_proba_acc = np.zeros((X_test_s.shape[0], n_classes), dtype=np.float32)

        for C in C_LIST:
            base_lr = LogisticRegression(
                solver="saga",
                max_iter=2000,
                C=C,
                class_weight=None,
                random_state=42,
                n_jobs=1,
            )
            clf = OneVsRestClassifier(base_lr)
            clf.fit(X_train_s[tr_idx], y_train[tr_idx])

            tr_score_acc += clf.decision_function(X_train_s[tr_idx]).astype(np.float32)
            va_score_acc += clf.decision_function(X_train_s[va_idx]).astype(np.float32)
            te_score_acc += clf.decision_function(X_test_s).astype(np.float32)

            tr_proba_acc += clf.predict_proba(X_train_s[tr_idx]).astype(np.float32)
            va_proba_acc += clf.predict_proba(X_train_s[va_idx]).astype(np.float32)
            te_proba_acc += clf.predict_proba(X_test_s).astype(np.float32)

        tr_score = tr_score_acc / float(len(C_LIST))
        va_score = va_score_acc / float(len(C_LIST))
        te_score = te_score_acc / float(len(C_LIST))

        tr_raw = tr_proba_acc / float(len(C_LIST))
        va_raw = va_proba_acc / float(len(C_LIST))
        te_raw = te_proba_acc / float(len(C_LIST))

        test_raw_accum += te_raw / n_splits
        oof_raw[va_idx] = va_raw

        va_cal = np.zeros_like(va_score, dtype=np.float32)
        te_cal_fold = np.zeros_like(te_score, dtype=np.float32)
        for j in range(n_classes):
            if len(np.unique(y_train[tr_idx, j])) < 2:
                va_cal[:, j] = va_raw[:, j]
                te_cal_fold[:, j] = te_raw[:, j]
                continue
            a, b = _sigmoid_fit(tr_score[:, j], y_train[tr_idx, j])
            va_cal[:, j] = _sigmoid_apply(va_score[:, j], a, b).astype(np.float32)
            te_cal_fold[:, j] = _sigmoid_apply(te_score[:, j], a, b).astype(np.float32)

        oof_cal[va_idx] = va_cal
        test_cal_accum += te_cal_fold / n_splits

        try:
            per_col_raw = []
            per_col_cal = []
            for j in range(n_classes):
                if len(np.unique(y_train[va_idx, j])) < 2:
                    continue
                per_col_raw.append(roc_auc_score(y_train[va_idx, j], va_raw[:, j]))
                per_col_cal.append(roc_auc_score(y_train[va_idx, j], va_cal[:, j]))
            if len(per_col_raw) > 0:
                print(
                    f"Fold {fold}/{n_splits} OOF mean column-wise AUC raw/cal: "
                    f"{float(np.mean(per_col_raw)):.5f} / {float(np.mean(per_col_cal)):.5f}"
                )
        except Exception as e:
            print(f"Fold {fold}/{n_splits} AUC computation skipped: {e}")

    blend = 0.75  # keep close to calibrated to preserve existing semantics
    proba = blend * test_cal_accum + (1.0 - blend) * test_raw_accum
    proba = np.clip(proba, 1e-6, 1 - 1e-6)
    return proba, test_df["image_id"].astype(str).values




## === cell 6
if len(submissions_all) >= 3:
    submission_avg, image_ids = ensemble(submissions_all, [0, 1, 2], [0.05, 0.9, 0.05])
    make_submission_file(submission_avg, image_ids)
elif len(submissions_all) > 0:
    k = len(submissions_all)
    sub_idx = list(range(k))
    weights = [1.0 / k] * k
    submission_avg, image_ids = ensemble(submissions_all, sub_idx, weights)
    make_submission_file(submission_avg, image_ids)
else:
    submission_avg, image_ids = train_and_predict_image_baseline()
    make_submission_file(submission_avg, image_ids)
