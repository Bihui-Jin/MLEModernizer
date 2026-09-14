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

geopandas==0.14.4
numpy==1.26.4
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

0.9332

# 6. Current score

0.81346

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.66225) has done: 'Your notebook fails because it tries to load external “public submission” CSVs from a dataset path that doesn’t exist in this environment, so no `sub` ever gets created. To make it run end-to-end and generate a valid `submission.csv`, I remove the dependency on missing files and instead build a simple, deterministic metadata-only baseline from the provided `train.csv`/`test.csv` (using only the allowed columns). This preserves the overall intent (tabular prediction) while ensuring the correct submission format (`image_name,target`) and no index misalignment. The output be a proper probability between 0 and 1 for every test `image_name`.'
- What this solution (achieved 0.64588) has done: 'Your current metadata-only blend is leaving AUC on the table mainly because the hand-set weights (and strong smoothing) aren’t tuned to the metric; we can move toward 0.9332 by fitting those weights on out-of-fold predictions to directly maximize ROC-AUC while keeping the exact same feature logic (smoothed target encodings). I add a deterministic GroupKFold by `patient_id` to avoid leakage, generate OOF predictions for `sex/site/age_bin/patient_id`, and then fit a simple logistic regression blender on those four OOF features. Finally, I refit the encoders on full train, score the test the same way, and apply the trained blender to produce probabilities in the required `image_name,target` submission format.'
- What this solution (achieved 0.77703) has done: 'The crash comes from mapping categorical columns to numeric target-encoding rates and then calling `fillna(prior)` while the intermediate result is still a `Categorical`, which pandas disallows when the fill value isn’t an existing category. I fix this in the narrowest way by forcing the mapped Series to a float dtype before `fillna`, keeping all feature logic and the blender unchanged. I also make `pd.cut` bins explicitly include an `"unknown"` bucket for missing ages to avoid string `"nan"` categories and ensure consistent mapping. The rest of the pipeline (GroupKFold by patient_id, OOF blender training, and writing `submission.csv`) stays the same and now run end-to-end.'
- What this solution (achieved 0.75966) has done: 'We’re still far below the target AUC (0.777 vs 0.9332), so we should make a small, legitimate improvement without changing the overall approach (target-encoding + simple image stats + logistic-regression blender). The biggest low-risk gain here is to standardize the 5 image-stat features before the logistic regression: their scales differ (e.g., std vs means), and unscaled features can hurt a linear model’s fit and calibration, reducing AUC. I add a `StandardScaler` fitted on the OOF training design matrix and apply it consistently to test features, leaving the same features, same folds, and same model family/training logic intact. This is a minimal change that commonly yields a noticeable AUC lift for linear blenders while preserving evaluation semantics and producing the same valid `submission.csv`.'
- What this solution (achieved 0.76072) has done: 'Your current pipeline is valid and already includes the biggest “small change” (standardizing before the logistic regression), but it is still underfitting because the logistic regression is effectively under-optimized for this feature mix. To move the AUC upward toward the 0.9332 target while preserving the exact same features, folds, and model family, I only (1) increase `max_iter` so LBFGS can actually converge, and (2) set a slightly stronger regularization (smaller `C`) which typically improves AUC for noisy target-encoding + weak image-stat features. I also explicitly set `n_jobs=1` (LBFGS ignores it, but it keeps behavior stable) and print convergence info via a warning catch; nothing else about the core logic changes. The output remains a proper `submission.csv` with `image_name,target` aligned to `sample_submission.csv`.'
- What this solution (achieved 0.81003) has done: 'Your current approach is solid but the test-time target encoding for `patient_id` can behave inconsistently because many test patients are unseen (so it collapses to the global mean), and it can also inject noise when a patient has very few lesions. To move AUC upward toward the 0.9332 target without changing the overall modeling approach, I only tune the smoothing strengths (`m_*`) and (critically) disable `patient_id` target encoding at inference by forcing it to always use the prior (while keeping the same 4 TE features so the blender architecture and feature count stay identical). This preserves your core logic (GroupKFold OOF TE + image stats + standardized logistic regression) but reduces a known generalization failure mode. The submission writing and schema remain unchanged and still produce a valid `submission.csv`.'
- What this solution (achieved 0.80926) has done: 'We’re currently well below the target AUC (0.81003 vs 0.9332), so we should make small, legitimate improvements that keep your exact core pipeline (TE features + 5 image stats + standardized logistic regression) unchanged. The highest-impact minimal fix is to remove a silent train/test mismatch: you disable `patient_id` target encoding at test time but still train the blender using `patient_id` TE at validation time, so the model learns to rely on a feature that becomes constant at inference. I align semantics by also disabling `patient_id` TE during OOF construction (and keep the 4th TE column as the prior so feature count/architecture stays identical). Additionally, because AUC is rank-based, I mildly reduce over-regularization by setting `C=1.0` (still LBFGS logistic regression, same training approach) to move performance upward without changing the feature logic.'
- What this solution (achieved 0.81143) has done: 'We’re far below the target AUC (0.80926 vs 0.9332), so we should make the smallest legitimate changes that improve ranking without changing your core pipeline (same TE features + same 5 image stats + same standardized logistic-regression blender). The biggest low-risk uplift is to fix an information loss bug: your “gray” stats currently average R/G/B before computing mean/std, which discards color contrast that’s known to matter for melanoma; switching to PIL’s standard luminance grayscale (still deterministic, still 2 gray features) usually improves AUC while keeping the same feature count and modeling approach. Second, we reduce excessive smoothing on `sex/site/age_bin` target encodings (patient_id still disabled), which can help separation/ranking without introducing leakage. Everything else (GroupKFold by patient_id, scaler, logistic regression, submission format/path) stays unchanged.'
- What this solution (achieved 0.81346) has done: 'We’re currently below the target AUC, so the smallest legitimate improvement is to better align the logistic-regression blender’s objective with ROC-AUC under heavy class imbalance by enabling `class_weight="balanced"` (same model family/solver/training loop). To avoid changing your feature set or folds, everything else stays identical: same TE features (with patient_id disabled), same 5 image stats, same standardization, and same inference/post-processing. This typically improves ranking for minority-class positives without any leakage or semantic change. The script still run end-to-end and write a valid `submission.csv` in the required format.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

DATA_DIR = "/kaggle/data"  # as listed in the provided paths

train_path = f"{DATA_DIR}/train.csv"
test_path = f"{DATA_DIR}/test.csv"
sample_sub_path = f"{DATA_DIR}/sample_submission.csv"

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
sub = pd.read_csv(sample_sub_path)

assert "target" in train.columns
assert "image_name" in test.columns and "image_name" in sub.columns
assert len(sub) == len(test)

JPEG_DIR_CANDIDATES = [
    f"{DATA_DIR}/jpeg",
    f"{DATA_DIR}/siim-isic-melanoma-classification/jpeg",
    "/kaggle/input/siim-isic-melanoma-classification/jpeg",
    "/kaggle/input/jpeg",
]
JPEG_DIR = next((p for p in JPEG_DIR_CANDIDATES if os.path.isdir(p)), None)
if JPEG_DIR is None:
    raise FileNotFoundError(
        "Could not find a valid JPEG directory in expected locations."
    )

TRAIN_JPEG_DIR = os.path.join(JPEG_DIR, "train")
TEST_JPEG_DIR = os.path.join(JPEG_DIR, "test")




## === cell 1
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

tr = train.copy(deep=False)
te = test.copy(deep=False)

for c in ["sex", "anatom_site_general_challenge"]:
    tr[c] = tr[c].fillna("unknown").astype(str).str.strip().str.lower()
    te[c] = te[c].fillna("unknown").astype(str).str.strip().str.lower()

tr["age_approx"] = pd.to_numeric(tr["age_approx"], errors="coerce")
te["age_approx"] = pd.to_numeric(te["age_approx"], errors="coerce")

age_bins = [-np.inf, 20, 30, 40, 50, 60, 70, 80, np.inf]
age_labels = ["<20", "20s", "30s", "40s", "50s", "60s", "70s", "80+"]

tr["age_bin"] = pd.cut(tr["age_approx"], bins=age_bins, labels=age_labels).astype(
    object
)
te["age_bin"] = pd.cut(te["age_approx"], bins=age_bins, labels=age_labels).astype(
    object
)
tr["age_bin"] = tr["age_bin"].where(tr["age_bin"].notna(), "unknown")
te["age_bin"] = te["age_bin"].where(te["age_bin"].notna(), "unknown")

global_mean = float(tr["target"].mean())

tr["sex"] = tr["sex"].astype("category")
te["sex"] = te["sex"].astype("category")
tr["anatom_site_general_challenge"] = tr["anatom_site_general_challenge"].astype(
    "category"
)
te["anatom_site_general_challenge"] = te["anatom_site_general_challenge"].astype(
    "category"
)
tr["age_bin"] = tr["age_bin"].astype("category")
te["age_bin"] = te["age_bin"].astype("category")


def smoothed_rate(df, key, target_col="target", prior=None, m=200):
    if prior is None:
        prior = float(df[target_col].mean())
    stats = df.groupby(key, observed=True)[target_col].agg(["mean", "count"])
    stats["smooth"] = (stats["count"] * stats["mean"] + m * prior) / (
        stats["count"] + m
    )
    return stats["smooth"]


def build_te_maps(df_fit, prior, m_sex=50, m_site=120, m_age=120, m_pid=5000):
    sex_rate = smoothed_rate(df_fit, "sex", prior=prior, m=m_sex)
    site_rate = smoothed_rate(
        df_fit, "anatom_site_general_challenge", prior=prior, m=m_site
    )
    age_rate = smoothed_rate(df_fit, "age_bin", prior=prior, m=m_age)
    pid_rate = smoothed_rate(df_fit, "patient_id", prior=prior, m=m_pid)
    return sex_rate, site_rate, age_rate, pid_rate


def apply_te(df, sex_rate, site_rate, age_rate, pid_rate, fallback, disable_pid=False):
    x_sex = (
        pd.to_numeric(df["sex"].map(sex_rate), errors="coerce")
        .fillna(fallback)
        .astype(float)
    )
    x_site = (
        pd.to_numeric(
            df["anatom_site_general_challenge"].map(site_rate), errors="coerce"
        )
        .fillna(fallback)
        .astype(float)
    )
    x_age = (
        pd.to_numeric(df["age_bin"].map(age_rate), errors="coerce")
        .fillna(fallback)
        .astype(float)
    )

    if disable_pid:
        x_pid = np.full(len(df), float(fallback), dtype=np.float64)
    else:
        x_pid = (
            pd.to_numeric(df["patient_id"].map(pid_rate), errors="coerce")
            .fillna(fallback)
            .astype(float)
        )

    X = np.vstack([x_sex.values, x_site.values, x_age.values, x_pid]).T
    X = np.nan_to_num(X, nan=fallback, posinf=fallback, neginf=fallback)
    return X




## === cell 2
from PIL import Image, ImageStat
from concurrent.futures import ThreadPoolExecutor

Image.MAX_IMAGE_PIXELS = None


def _img_stats_from_path_fast(path):
    """
    Returns 5 deterministic features:
    - mean_gray, std_gray from standard luminance grayscale (PIL "L") in [0,1]
    - mean_r, mean_g, mean_b in [0,1]
    """
    try:
        with Image.open(path) as im:
            try:
                im.draft("RGB", (128, 128))
            except Exception:
                pass
            im = im.convert("RGB")
            im = im.resize((128, 128), resample=Image.BILINEAR)

            st = ImageStat.Stat(im)
            mean_r, mean_g, mean_b = (
                st.mean[0] / 255.0,
                st.mean[1] / 255.0,
                st.mean[2] / 255.0,
            )

            gray_im = im.convert("L")
            gray = np.asarray(gray_im, dtype=np.float32) / 255.0
            mean_gray = float(gray.mean())
            std_gray = float(gray.std())

        return np.array([mean_gray, std_gray, mean_r, mean_g, mean_b], dtype=np.float32)
    except Exception:
        return np.array([np.nan, np.nan, np.nan, np.nan, np.nan], dtype=np.float32)


def build_image_features(
    image_names, folder, cache_path=None, max_workers=None, chunksize=2048
):
    if cache_path is not None and os.path.exists(cache_path):
        feats = np.load(cache_path)
        if feats.shape == (len(image_names), 5):
            return feats.astype(np.float32, copy=False)

    n = len(image_names)
    feats = np.zeros((n, 5), dtype=np.float32)

    if max_workers is None:
        max_workers = max(4, min(16, (os.cpu_count() or 8)))

    join = os.path.join
    paths = [join(folder, f"{name}.jpg") for name in image_names]

    def _process_range(start_end):
        s, e = start_end
        out = np.empty((e - s, 5), dtype=np.float32)
        fn = _img_stats_from_path_fast
        for j, i in enumerate(range(s, e)):
            out[j] = fn(paths[i])
        return s, out

    bounds = list(range(0, n, chunksize))
    ranges = [(s, min(s + chunksize, n)) for s in bounds]

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for s, out in ex.map(_process_range, ranges):
            feats[s : s + out.shape[0]] = out

    if cache_path is not None:
        np.save(cache_path, feats)
    return feats


tr_names = tr["image_name"].values
te_names = te["image_name"].values

tr_cache = os.path.join("/kaggle/working", "tr_img_feats_128_lumgray.npy")
te_cache = os.path.join("/kaggle/working", "te_img_feats_128_lumgray.npy")

tr_img = build_image_features(
    tr_names, TRAIN_JPEG_DIR, cache_path=tr_cache, chunksize=2048
)
te_img = build_image_features(
    te_names, TEST_JPEG_DIR, cache_path=te_cache, chunksize=2048
)

col_means = np.nanmean(tr_img, axis=0)
col_means = np.where(np.isfinite(col_means), col_means, 0.0).astype(np.float32)
tr_img = np.where(np.isfinite(tr_img), tr_img, col_means)
te_img = np.where(np.isfinite(te_img), te_img, col_means)




## === cell 3
from sklearn.preprocessing import StandardScaler
import warnings
from sklearn.exceptions import ConvergenceWarning

gkf = GroupKFold(n_splits=5)
groups = tr["patient_id"].astype(str).values
y = tr["target"].values.astype(int)

oof_X = np.zeros((len(tr), 9), dtype=np.float64)

for fold, (tr_idx, va_idx) in enumerate(gkf.split(tr, y, groups=groups), 1):
    df_fit = tr.iloc[tr_idx]
    df_val = tr.iloc[va_idx]
    prior = float(df_fit["target"].mean())

    sex_rate, site_rate, age_rate, pid_rate = build_te_maps(df_fit, prior=prior)

    X_te_val = apply_te(
        df_val,
        sex_rate,
        site_rate,
        age_rate,
        pid_rate,
        fallback=prior,
        disable_pid=True,
    )

    X_img_val = tr_img[va_idx].astype(np.float64)
    oof_X[va_idx] = np.hstack([X_te_val.astype(np.float64), X_img_val])

scaler = StandardScaler()
oof_Xs = scaler.fit_transform(oof_X)

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always", ConvergenceWarning)

    blender = LogisticRegression(
        solver="lbfgs",
        max_iter=2000,
        C=1.0,
        class_weight="balanced",
        random_state=0,
        n_jobs=1,
    )
    blender.fit(oof_Xs, y)

    conv_warnings = [x for x in w if issubclass(x.category, ConvergenceWarning)]
    if len(conv_warnings) > 0:
        print(
            f"ConvergenceWarning count: {len(conv_warnings)} (consider increasing max_iter further)"
        )

oof_pred = blender.predict_proba(oof_Xs)[:, 1]
oof_auc = roc_auc_score(y, oof_pred)
print("OOF ROC-AUC (blender on TE+image features, standardized):", oof_auc)

sex_rate, site_rate, age_rate, pid_rate = build_te_maps(tr, prior=global_mean)

X_te_test = apply_te(
    te, sex_rate, site_rate, age_rate, pid_rate, fallback=global_mean, disable_pid=True
)

X_test = np.hstack([X_te_test.astype(np.float64), te_img.astype(np.float64)])
X_test_s = scaler.transform(X_test)

pred = blender.predict_proba(X_test_s)[:, 1].astype(float)
pred = np.clip(pred, 0.0, 1.0)

pred_df = pd.DataFrame({"image_name": te["image_name"].values, "target": pred})
sub_out = sub[["image_name"]].merge(pred_df, on="image_name", how="left")
sub_out["target"] = sub_out["target"].fillna(global_mean).astype(float)

sub_out.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub_out.shape)
print(sub_out.head())
