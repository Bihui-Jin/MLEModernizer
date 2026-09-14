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
Predict whether a lesion is malignant (0 denotes **benign**, and 1 indicates **malignant**).

## Metric
Area under the ROC curve.

## Submission Format
For each `image_name` in the test set, you must predict the probability (`target`) that the sample is **malignant**. The file should contain a header and have the following format:

```
image_name,target
ISIC_0052060,0.7
ISIC_0052349,0.9
ISIC_0058510,0.8
ISIC_0073313,0.5
ISIC_0073502,0.5
etc.
```

## Dataset 
The images are provided in DICOM format.

Images are also provided in JPEG and TFRecord format (in the `jpeg` and `tfrecords` directories, respectively). Images in TFRecord format have been resized to a uniform 1024x1024.

Metadata is also provided outside of the DICOM format, in CSV files. See the `Columns` section for a description.

### Files
- **train.csv** - the training set
- **test.csv** - the test set
- **sample_submission.csv** - a sample submission file in the correct format

### Columns
- `image_name` - unique identifier, points to filename of related DICOM image
- `patient_id` - unique patient identifier
- `sex` - the sex of the patient (when unknown, will be blank)
- `age_approx` - approximate patient age at time of imaging
- `anatom_site_general_challenge` - location of imaged site
- `diagnosis` - detailed diagnosis information (train only)
- `benign_malignant` - indicator of malignancy of imaged lesion
- `target` - binarized version of the target variable

# 2. Python version

3.8

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (176 lines)
            jpeg.zip (24.7 GB)
            sample_submission.csv (4143 lines)
            sample_submission.csv.zip (16.4 kB)
            test.csv (4143 lines)
            test.csv.zip (42.5 kB)
            test.zip (6.4 GB)
            tfrecords.zip (9.3 GB)
            train.csv (28985 lines)
            train.csv.zip (299.7 kB)
            train.zip (46.0 GB)
            jpeg/
                test/
                    ISIC_1440063.jpg (1.1 MB)
                    ISIC_0815802.jpg (853.1 kB)
                    ... and 4140 other files
                train/
                    ISIC_1845271.jpg (1.0 MB)
                    ISIC_1970027.jpg (138.4 kB)
                    ... and 28982 other files
            siim-isic-melanoma-classification/
                description.md (176 lines)
                jpeg.zip (24.7 GB)
                ... and 9 other files
                jpeg/
                    test/
                        ISIC_1440063.jpg (1.1 MB)
                        ISIC_0815802.jpg (853.1 kB)
                        ... and 4140 other files
                    train/
                        ISIC_1845271.jpg (1.0 MB)
                        ISIC_1970027.jpg (138.4 kB)
                        ... and 28982 other files
                siim-isic-melanoma-classification/
                test/
                    ISIC_0052212.dcm (1.5 MB)
                    ISIC_0076545.dcm (4.0 MB)
                    ... and 4140 other files
                    test/
                tfrecords/
                    test00-2071.tfrec (579.6 MB)
                    test01-2071.tfrec (583.5 MB)
                    ... and 14 other files
                train/
                    ISIC_0015719.dcm (2.4 MB)
                    ISIC_0068279.dcm (1.3 MB)
                    ... and 28982 other files
                    train/
            test/
                ISIC_0052212.dcm (1.5 MB)
                ISIC_0076545.dcm (4.0 MB)
                ... and 4140 other files
                test/
            tfrecords/
                test00-2071.tfrec (579.6 MB)
                test01-2071.tfrec (583.5 MB)
                ... and 14 other files
            train/
                ISIC_0015719.dcm (2.4 MB)
                ISIC_0068279.dcm (1.3 MB)
                ... and 28982 other files
                train/
        input/
            description.md (176 lines)
            jpeg.zip (24.7 GB)
            sample_submission.csv (4143 lines)
            sample_submission.csv.zip (16.4 kB)
            test.csv (4143 lines)
            test.csv.zip (42.5 kB)
            test.zip (6.4 GB)
            tfrecords.zip (9.3 GB)
            train.csv (28985 lines)
            train.csv.zip (299.7 kB)
            train.zip (46.0 GB)
            jpeg/
                test/
                    ISIC_1440063.jpg (1.1 MB)
                    ISIC_0815802.jpg (853.1 kB)
                    ... and 4140 other files
                train/
                    ISIC_1845271.jpg (1.0 MB)
                    ISIC_1970027.jpg (138.4 kB)
                    ... and 28982 other files
            siim-isic-melanoma-classification/
                description.md (176 lines)
                jpeg.zip (24.7 GB)
                ... and 9 other files
                jpeg/
                    test/
                        ISIC_1440063.jpg (1.1 MB)
                        ISIC_0815802.jpg (853.1 kB)
                        ... and 4140 other files
                    train/
                        ISIC_1845271.jpg (1.0 MB)
                        ISIC_1970027.jpg (138.4 kB)
                        ... and 28982 other files
                siim-isic-melanoma-classification/
                test/
                    ISIC_0052212.dcm (1.5 MB)
                    ISIC_0076545.dcm (4.0 MB)
                    ... and 4140 other files
                    test/
                tfrecords/
                    test00-2071.tfrec (579.6 MB)
                    test01-2071.tfrec (583.5 MB)
                    ... and 14 other files
                train/
                    ISIC_0015719.dcm (2.4 MB)
                    ISIC_0068279.dcm (1.3 MB)
                    ... and 28982 other files
                    train/
            test/
                ISIC_0052212.dcm (1.5 MB)
                ISIC_0076545.dcm (4.0 MB)
                ... and 4140 other files
                test/
                    ISIC_0052212.dcm (1.5 MB)
                    ISIC_0076545.dcm (4.0 MB)
                    ... and 4140 other files
                    test/
            tfrecords/
                test00-2071.tfrec (579.6 MB)
                test01-2071.tfrec (583.5 MB)
                ... and 14 other files
            train/
                ISIC_0015719.dcm (2.4 MB)
                ISIC_0068279.dcm (1.3 MB)
                ... and 28982 other files
                train/
                    ISIC_0015719.dcm (2.4 MB)
                    ISIC_0068279.dcm (1.3 MB)
                    ... and 28982 other files
                    train/
        working/
            siim-isic-melanoma-classification/
                description.md (176 lines)
                jpeg.zip (24.7 GB)
                ... and 9 other files
                jpeg/
                    test/
                        ISIC_1440063.jpg (1.1 MB)
                        ISIC_0815802.jpg (853.1 kB)
                        ... and 4140 other files
                    train/
                        ISIC_1845271.jpg (1.0 MB)
                        ISIC_1970027.jpg (138.4 kB)
                        ... and 28982 other files
                siim-isic-melanoma-classification/
                test/
                    ISIC_0052212.dcm (1.5 MB)
                    ISIC_0076545.dcm (4.0 MB)
                    ... and 4140 other files
                    test/
                tfrecords/
                    test00-2071.tfrec (579.6 MB)
                    test01-2071.tfrec (583.5 MB)
                    ... and 14 other files
                train/
                    ISIC_0015719.dcm (2.4 MB)
                    ISIC_0068279.dcm (1.3 MB)
                    ... and 28982 other files
                    train/
```

-> data/sample_submission.csv has 4142 rows and 2 columns.
The columns are: image_name, target

-> data/siim-isic-melanoma-classification/sample_submission.csv has 4142 rows and 2 columns.
The columns are: image_name, target

-> data/siim-isic-melanoma-classification/test.csv has 4142 rows and 5 columns.
The columns are: image_name, patient_id, sex, age_approx, anatom_site_general_challenge

-> data/siim-isic-melanoma-classification/train.csv has 28984 rows and 8 columns.
The columns are: image_name, patient_id, sex, age_approx, anatom_site_general_challenge, diagnosis, benign_malignant, target

-> data/test.csv has 4142 rows and 5 columns.
The columns are: image_name, patient_id, sex, age_approx, anatom_site_general_challenge

-> data/train.csv has 28984 rows and 8 columns.
The columns are: image_name, patient_id, sex, age_approx, anatom_site_general_challenge, diagnosis, benign_malignant, target

-> input/sample_submission.csv has 4142 rows and 2 columns.
The columns are: image_name, target

-> (stopped after 10 files for performance)

# 5. Target score

0.924664431362698

# 6. Current score

0.76148

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.52941) has done: 'I remove the internet `wget` dependency (Kaggle notebooks run without outbound internet), which is why your downloaded CSVs are empty and `pd.read_csv` fails. To keep the core idea (blend several submissions) but make it runnable end-to-end, I replace the missing external submissions with three deterministic, locally-computed “base submissions” from `test.csv` metadata (patient/sex/age/site), then blend them the same way and write a valid `submission.csv`. I also fix the incorrect `/6` scaling so predictions remain valid probabilities (average of 3 models uses `/3`). Finally, I validate columns/order against `sample_submission.csv` to ensure Kaggle accepts the file.'
- What this solution (achieved 0.69768) has done: 'Your current 0.529 AUC is near-random because the “model” only uses test-set metadata priors, which can’t meaningfully rank malignant vs benign; to move toward the 0.924 target, we need real signal from the images while keeping the approach simple and within Kaggle/no-extra-packages constraints. I keep the existing submission-writing and blending structure, but replace the three metadata-based base submissions with three deterministic image-based predictors computed from the provided JPEGs (intensity/color/texture statistics), which usually yields a large AUC jump for this competition without changing the overall “blend several submissions then average” core idea. I also add robust JPEG path resolution (both `/kaggle/input/.../jpeg/test` and `/kaggle/data/.../jpeg/test`) and ensure predictions stay valid probabilities and align exactly to `sample_submission.csv`. This is still fast (single pass over ~4k test images) and writes a valid `submission.csv`.'
- What this solution (achieved 0.7074) has done: 'Your current approach is a deterministic “blend of 3 base submissions” derived from simple JPEG statistics; to move AUC upward toward 0.924 with minimal logic change, I keep the same feature extraction + z-scoring + 3-logit blending structure but make two small, high-impact tweaks. First, I compute z-scores robustly using median/MAD (less sensitive to outlier images) which typically improves ranking stability (AUC is ranking-based). Second, I calibrate the intercept automatically from the test prediction distribution to match a realistic melanoma prevalence (~1.76% from this competition), which improves probability scaling without changing ranking much but can help the blend behave more consistently. Everything still runs end-to-end, reads the same files/paths, and writes a valid `submission.csv`.'
- What this solution (achieved 0.76148) has done: 'The timeout is dominated by single-threaded JPEG decoding + per-image Python overhead in `image_features` for ~12k images (8k train + 4.1k test). I keep the exact same features and ridge/blending logic, but speed up extraction by (1) eliminating repeated PIL object conversions, (2) vectorizing grayscale/gradient computations more efficiently, (3) parallelizing feature extraction across CPU cores with deterministic ordering, and (4) avoiding unnecessary dtype conversions/copies. The model fitting and calibration remain identical; only the way features are computed (same math) is optimized. File paths and outputs remain unchanged.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

DATA_DIR_CANDIDATES = [
    "/kaggle/input/siim-isic-melanoma-classification",
    "/kaggle/data/siim-isic-melanoma-classification",
    "/kaggle/input",
    "/kaggle/data",
]


def find_file(filename):
    for base in DATA_DIR_CANDIDATES:
        path = os.path.join(base, filename)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"Could not find {filename} in any of: {DATA_DIR_CANDIDATES}"
    )


def find_dir(relpath):
    for base in DATA_DIR_CANDIDATES:
        path = os.path.join(base, relpath)
        if os.path.isdir(path):
            return path
    raise FileNotFoundError(
        f"Could not find directory {relpath} in any of: {DATA_DIR_CANDIDATES}"
    )


test_csv_path = find_file("test.csv")
train_csv_path = find_file("train.csv")
sample_sub_path = find_file("sample_submission.csv")

test = pd.read_csv(test_csv_path)
train = pd.read_csv(train_csv_path)
sample_sub = pd.read_csv(sample_sub_path)

assert "image_name" in test.columns
assert "image_name" in train.columns and "target" in train.columns
assert "image_name" in sample_sub.columns and "target" in sample_sub.columns

test = test.merge(sample_sub[["image_name"]], on="image_name", how="right")

JPEG_TEST_DIR = None
for candidate in [
    "jpeg/test",
    "siim-isic-melanoma-classification/jpeg/test",
]:
    try:
        JPEG_TEST_DIR = find_dir(candidate)
        break
    except FileNotFoundError:
        pass
if JPEG_TEST_DIR is None:
    raise FileNotFoundError(
        "Could not locate jpeg/test directory under known data roots."
    )

JPEG_TRAIN_DIR = None
for candidate in [
    "jpeg/train",
    "siim-isic-melanoma-classification/jpeg/train",
]:
    try:
        JPEG_TRAIN_DIR = find_dir(candidate)
        break
    except FileNotFoundError:
        pass
if JPEG_TRAIN_DIR is None:
    raise FileNotFoundError(
        "Could not locate jpeg/train directory under known data roots."
    )

print("Using JPEG_TEST_DIR:", JPEG_TEST_DIR)
print("Using JPEG_TRAIN_DIR:", JPEG_TRAIN_DIR)
print(
    "test rows:",
    test.shape[0],
    "train rows:",
    train.shape[0],
    "sample_sub rows:",
    sample_sub.shape[0],
)



## === cell 1
from PIL import Image
from concurrent.futures import ThreadPoolExecutor


def sigmoid(x):
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))


def image_features(image_name, jpg_dir, size=96):
    path = os.path.join(jpg_dir, f"{image_name}.jpg")
    with Image.open(path) as im:
        im = im.convert("RGB")
        if size is not None:
            im = im.resize((size, size), resample=Image.BILINEAR)
        arr = np.asarray(im, dtype=np.uint8)

    arr_f = arr.astype(np.float32) * (1.0 / 255.0)  # (H,W,3)
    r = arr_f[..., 0]
    g = arr_f[..., 1]
    b = arr_f[..., 2]

    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    mean_gray = float(gray.mean())
    std_gray = float(gray.std())

    mean_r = float(r.mean())
    mean_rgb = float(arr_f.mean()) + 1e-6
    redness = float(mean_r / mean_rgb)

    gx = np.abs(np.diff(gray, axis=1)).mean()
    gy = np.abs(np.diff(gray, axis=0)).mean()
    grad = float(gx + gy)

    return mean_gray, std_gray, redness, grad


def robust_location_scale(x):
    x = x.astype(np.float32, copy=False)
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)) + 1e-6)
    scale = float(1.4826 * mad + 1e-6)
    return med, scale


def apply_robust_zscore(x, med, scale):
    x = x.astype(np.float32, copy=False)
    return (x - med) / scale


def calibrate_intercept(base_logit, target_prevalence):
    lo, hi = -10.0, 10.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        m = float(sigmoid(base_logit + mid).mean())
        if m > target_prevalence:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


train_names_all = train["image_name"].values
train_y_all = train["target"].astype(np.float32).values

MAX_TRAIN_FIT = 8000  # keeps runtime bounded; deterministic slice
fit_idx = np.arange(min(len(train_names_all), MAX_TRAIN_FIT), dtype=np.int32)
train_names = train_names_all[fit_idx].tolist()
train_y = train_y_all[fit_idx]

print(
    "Fitting feature scaling/weights on train subset:",
    len(train_names),
    "of",
    len(train_names_all),
)

test_names = test["image_name"].values.tolist()


def _extract_batch(names, jpg_dir, size, max_workers):
    n = len(names)
    out = np.empty((n, 4), dtype=np.float32)

    def _one(i_nm):
        i, nm = i_nm
        mg, sg, rd, gr = image_features(nm, jpg_dir, size=size)
        return i, mg, sg, rd, gr

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for k, (i, mg, sg, rd, gr) in enumerate(
            ex.map(_one, enumerate(names), chunksize=32), start=1
        ):
            out[i, 0] = mg
            out[i, 1] = sg
            out[i, 2] = rd
            out[i, 3] = gr
            if k % 2000 == 0 or k == n:
                print(f"Processed {k}/{n} images from {os.path.basename(jpg_dir)}")
    return out


CPU_CNT = os.cpu_count() or 2
MAX_WORKERS = min(8, max(2, CPU_CNT))

tr_feats = _extract_batch(train_names, JPEG_TRAIN_DIR, size=96, max_workers=MAX_WORKERS)
te_feats = _extract_batch(test_names, JPEG_TEST_DIR, size=96, max_workers=MAX_WORKERS)

tr_mg, tr_sg, tr_rd, tr_gr = (
    tr_feats[:, 0],
    tr_feats[:, 1],
    tr_feats[:, 2],
    tr_feats[:, 3],
)
te_mg, te_sg, te_rd, te_gr = (
    te_feats[:, 0],
    te_feats[:, 1],
    te_feats[:, 2],
    te_feats[:, 3],
)

mg_med, mg_scale = robust_location_scale(tr_mg)
sg_med, sg_scale = robust_location_scale(tr_sg)
rd_med, rd_scale = robust_location_scale(tr_rd)
gr_med, gr_scale = robust_location_scale(tr_gr)

tr_z_mg = apply_robust_zscore(tr_mg, mg_med, mg_scale)
tr_z_sg = apply_robust_zscore(tr_sg, sg_med, sg_scale)
tr_z_rd = apply_robust_zscore(tr_rd, rd_med, rd_scale)
tr_z_gr = apply_robust_zscore(tr_gr, gr_med, gr_scale)

te_z_mg = apply_robust_zscore(te_mg, mg_med, mg_scale)
te_z_sg = apply_robust_zscore(te_sg, sg_med, sg_scale)
te_z_rd = apply_robust_zscore(te_rd, rd_med, rd_scale)
te_z_gr = apply_robust_zscore(te_gr, gr_med, gr_scale)


def fit_ridge_weights(X, y, l2=5.0):
    y = y.astype(np.float32, copy=False)
    y_soft = np.clip(0.02 + 0.96 * y, 1e-4, 1 - 1e-4).astype(np.float32, copy=False)
    t = np.log(y_soft / (1.0 - y_soft)).astype(np.float32, copy=False)

    XtX = (X.T @ X).astype(np.float32, copy=False)
    Xtt = (X.T @ t).astype(np.float32, copy=False)
    A = XtX + (l2 * np.eye(X.shape[1], dtype=np.float32))
    w = np.linalg.solve(A, Xtt).astype(np.float32, copy=False)
    return w


X_a_tr = np.stack(
    [tr_z_sg, tr_z_gr, tr_z_mg, tr_z_rd, np.ones_like(tr_z_sg)], axis=1
).astype(np.float32, copy=False)
X_b_tr = np.stack(
    [tr_z_gr, tr_z_rd, tr_z_mg, tr_z_sg, np.ones_like(tr_z_gr)], axis=1
).astype(np.float32, copy=False)
X_c_tr = np.stack(
    [tr_z_sg, tr_z_rd, tr_z_mg, tr_z_gr, np.ones_like(tr_z_sg)], axis=1
).astype(np.float32, copy=False)

w_a = fit_ridge_weights(X_a_tr, train_y, l2=8.0)
w_b = fit_ridge_weights(X_b_tr, train_y, l2=8.0)
w_c = fit_ridge_weights(X_c_tr, train_y, l2=8.0)

X_a_te = np.stack(
    [te_z_sg, te_z_gr, te_z_mg, te_z_rd, np.ones_like(te_z_sg)], axis=1
).astype(np.float32, copy=False)
X_b_te = np.stack(
    [te_z_gr, te_z_rd, te_z_mg, te_z_sg, np.ones_like(te_z_gr)], axis=1
).astype(np.float32, copy=False)
X_c_te = np.stack(
    [te_z_sg, te_z_rd, te_z_mg, te_z_gr, np.ones_like(te_z_sg)], axis=1
).astype(np.float32, copy=False)

raw_a = (X_a_te @ w_a).astype(np.float32, copy=False)
raw_b = (X_b_te @ w_b).astype(np.float32, copy=False)
raw_c = (X_c_te @ w_c).astype(np.float32, copy=False)

train_prev = float(train["target"].mean())
delta_a = calibrate_intercept(raw_a, target_prevalence=train_prev)
delta_b = calibrate_intercept(raw_b, target_prevalence=train_prev)
delta_c = calibrate_intercept(raw_c, target_prevalence=train_prev)

pred_a = sigmoid(raw_a + delta_a).astype(np.float32, copy=False)
pred_b = sigmoid(raw_b + delta_b).astype(np.float32, copy=False)
pred_c = sigmoid(raw_c + delta_c).astype(np.float32, copy=False)

a = pd.DataFrame({"image_name": test_names, "target": pred_a})
b = pd.DataFrame({"image_name": test_names, "target": pred_b})
c = pd.DataFrame({"image_name": test_names, "target": pred_c})

a.to_csv("b0-b3-b4.csv", index=False)
b.to_csv("b5.csv", index=False)
c.to_csv("b7.csv", index=False)

print(a.head())
print("Train prevalence:", train_prev)
print("Mean preds:", float(pred_a.mean()), float(pred_b.mean()), float(pred_c.mean()))



## === cell 2
a = pd.read_csv("b0-b3-b4.csv")
b = pd.read_csv("b5.csv")
c = pd.read_csv("b7.csv")

m = (
    sample_sub[["image_name"]]
    .merge(a, on="image_name", how="left", suffixes=("", "_a"))
    .merge(b, on="image_name", how="left", suffixes=("", "_b"))
    .merge(c, on="image_name", how="left", suffixes=("", "_c"))
)

for col in ["target", "target_b", "target_c"]:
    if col not in m.columns:
        raise RuntimeError(f"Missing blended column: {col}")
    m[col] = m[col].fillna(0.02)

pred = (m["target"].values + m["target_b"].values + m["target_c"].values) / 3.0
pred = np.clip(pred, 0.0, 1.0)

submission = pd.DataFrame(
    {"image_name": m["image_name"].values, "target": pred.astype(np.float32)}
)
submission = submission[sample_sub.columns]  # enforce correct column order

assert submission.shape[0] == sample_sub.shape[0]
assert list(submission.columns) == ["image_name", "target"]
assert submission["target"].between(0, 1).all()

submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print("Submission mean target:", float(submission["target"].mean()))
