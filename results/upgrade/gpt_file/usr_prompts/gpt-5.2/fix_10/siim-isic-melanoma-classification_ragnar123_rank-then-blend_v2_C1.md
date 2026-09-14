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

0.929009075916602

# 6. Current score

0.60471

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.58641) has done: 'Your notebook fails because it tries to read five external submission CSVs from `../input/melanoma-dif-sub/`, which are not present in this environment. To make it run end-to-end and still produce a valid submission, I keep the same “rank + weighted blend” core logic but replace the missing inputs with a simple, deterministic baseline probability built from `train.csv` and the available test metadata (age/sex/site) using out-of-fold target encoding (no leakage). This yields a reasonable AUC-oriented submission without changing the submission schema, and it always write a `.csv` file (`blend_sub.csv`) with the required columns.'
- What this solution (achieved 0.59329) has done: 'Your current score is far below the target, so we should cautiously improve discrimination without changing the overall “target-encoding → rank → weighted blend” pipeline. The biggest low-risk gain here is to make the out-of-fold encoding respect `patient_id` groups (prevent same-patient leakage across folds), which improves how well the learned metadata priors generalize to the test distribution. I also fix the smoothing to use fold-specific priors inside each fold (instead of the global mean from the full dataset), and add one additional interaction feature (`sex_site_age`) blended with a small weight to add signal while preserving the same core approach. The submission format/path remains identical and still writes `blend_sub.csv`.'
- What this solution (achieved 0.592) has done: 'Your current score (0.593) is far below the target (0.929), but with the constraint of “no new model/architecture/training”, the only safe gains come from improving the metadata encoding’s generalization. I keep the exact same pipeline (OOF target encoding → rank → weighted blend) and make two minimal, AUC-relevant fixes: (1) build *balanced* group folds by `patient_id` using a deterministic greedy assignment to equalize sample counts across folds, and (2) apply the same fold-aware prior consistently while also adding a tiny amount of deterministic noise in the encodings to reduce rank ties (often helps AUC when predictions are too discrete). The submission schema, file name, and blending logic remain unchanged, and it still write a valid `blend_sub.csv`.'
- What this solution (achieved 0.6517) has done: 'Your current score (0.592) is far below the target (0.929), so we should aim to increase discrimination while keeping the same “OOF target encoding → rank → weighted blend” core pipeline. The most direct low-risk gain is to actually use the OOF-encoded train predictions to learn the blend weights (via a tiny ridge-regularized linear meta-model on ranked OOF predictions), instead of hand-picked constants that may be suboptimal. This preserves the exact base features/encodings and the rank-based semantics, but makes the final blend better aligned to AUC. I also keep a deterministic setup and ensure the submission is aligned to `sample_submission.csv` and saved as `blend_sub.csv`.'
- What this solution (achieved 0.65486) has done: 'I keep your exact OOF target-encoding → rank → ridge-blend pipeline, but make two minimal AUC-oriented fixes that usually improve discrimination without changing the approach. First, I fit the ridge meta-model on a *ranked* target (and also rank the meta-model predictions) so the stacker optimizes ordering, which is what ROC-AUC cares about. Second, I replace the plain ridge regression with a tiny ridge-regularized IRLS logistic fit (same linear blend, just a better-calibrated objective for binary labels), then still rank the final outputs to preserve the rank-based semantics. These are small, contained changes and should move your score upward toward the target while still producing `blend_sub.csv` in the required format.'
- What this solution (achieved 0.61567) has done: 'Your current score (0.65486) is far below the target AUC (0.9290), so we should improve discrimination while keeping the same OOF target-encoding → rank → logistic ridge blend core pipeline. The biggest low-risk gain is to add a few more metadata interaction encodings (still the same target encoding logic) that commonly carry extra signal: patient-level prior, sex+age, and site+sex+age. To keep changes minimal and stable, I only add these additional encoders and let the same IRLS ridge stacker learn their blend weights, while keeping ranking semantics unchanged. I also make the fold construction deterministic and keep submission alignment to `sample_submission.csv` exactly as before.'
- What this solution (achieved 0.61563) has done: 'We keep your exact OOF target-encoding → rank → IRLS ridge-logistic blend pipeline, but fix two issues that are likely holding AUC down. First, the test encodings are currently computed from the full train (including each training row when used as “reference”), which is inconsistent with the OOF construction and can distort the learned blend; we compute *test* encodings as the average of fold-specific encodings (each fold built from its corresponding training split) to match the OOF semantics. Second, we standardize the ranked OOF features before IRLS (and apply the same transform to test) to stabilize the linear solver and make the ridge penalty behave consistently across features; this is a small numerical change that often improves ordering without changing the approach. Everything else (features, folds, ranking, IRLS, output file/columns) remains the same and it still write `blend_sub.csv`.'
- What this solution (achieved 0.61875) has done: 'Your current AUC (0.61563) is far below the target (0.92901), so we should aim for a cautious improvement while keeping the exact same “OOF target encoding → rank → IRLS ridge-logistic blend” pipeline. The biggest low-risk issue here is that `patient_id` target-encoding can silently dominate the stacker even though `patient_id` in test is (effectively) unseen, adding noise and hurting ordering; we keep the feature but constrain its influence by increasing its smoothing and slightly increasing ridge regularization. Additionally, we make the ranking step less sensitive to tiny noise by using `pct=True` ranking, keeping the exact semantics (monotonic transform) but improving numerical stability. These changes are small, preserve core logic, and should move the score upward without introducing new modeling approaches.'
- What this solution (achieved 0.60471) has done: 'Your current score (0.61875) is far below the target AUC (0.9290), so we should make a small, low-risk change that improves ranking quality without altering your core pipeline (OOF target encoding → rank → IRLS ridge-logistic blend → rank). The biggest likely drag here is the deterministic per-fold encoding noise (1e-4), which can randomly scramble close pairs and hurt ROC-AUC; removing it preserves semantics while making ordering more faithful to the learned encodings. I keep the same features, folds, smoothing, and IRLS stacker, but set `TIE_NOISE=0.0` and (only for the final `pred`) use a stable tie-breaker based on `image_name` to avoid massive rank ties without injecting random noise. This should move AUC upward while staying minimal and deterministic, and it still writes a valid `blend_sub.csv`.'

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


def _find_file(fname: str) -> str:
    for d in DATA_DIR_CANDIDATES:
        p = os.path.join(d, fname)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Could not find {fname} in any of: {DATA_DIR_CANDIDATES}")


train_path = _find_file("train.csv")
test_path = _find_file("test.csv")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)


def _prep_meta(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["age_approx"] = pd.to_numeric(df["age_approx"], errors="coerce")
    for c in ["sex", "anatom_site_general_challenge"]:
        if c in df.columns:
            df[c] = (
                df[c].fillna("unknown").astype(str).str.strip().replace({"": "unknown"})
            )
        else:
            df[c] = "unknown"
    if "patient_id" not in df.columns:
        df["patient_id"] = "unknown"
    else:
        df["patient_id"] = df["patient_id"].fillna("unknown").astype(str)
    return df


train = _prep_meta(train)
test = _prep_meta(test)

y = train["target"].astype(float).values
global_mean = float(np.mean(y))


def _make_group_folds_balanced(
    df: pd.DataFrame, group_col: str, k: int = 5, seed: int = 42
):
    rng = np.random.RandomState(seed)
    groups = df[group_col].astype(str).values

    g_sizes = pd.Series(groups).value_counts().sort_values(ascending=False)
    uniq_groups = g_sizes.index.to_numpy()

    sizes = g_sizes.to_numpy()
    jitter = rng.uniform(0, 1e-6, size=len(uniq_groups))
    order = np.lexsort((jitter, -sizes))
    uniq_groups = uniq_groups[order]
    sizes = sizes[order]

    fold_groups = [[] for _ in range(k)]
    fold_counts = np.zeros(k, dtype=np.int64)

    for g, sz in zip(uniq_groups, sizes):
        j = int(np.argmin(fold_counts))
        fold_groups[j].append(g)
        fold_counts[j] += int(sz)

    folds = []
    for fg in fold_groups:
        folds.append(np.where(np.isin(groups, fg))[0])
    return folds


folds = _make_group_folds_balanced(train, "patient_id", k=5, seed=42)


def oof_target_encode(
    train_df,
    test_df,
    col,
    folds,
    min_samples_leaf=50,  # kept for API compatibility; not used in this simple smoother
    smoothing=10.0,
    noise=0.0,
    seed=0,
):
    """
    Out-of-fold smoothed mean encoding for binary target.
    Returns: oof_train_encoded, test_encoded

    Change (score-relevant, minimal):
    - Keep fold-averaged test encoding consistent with OOF semantics (already present).
    """
    rng = np.random.RandomState(seed)
    oof = np.zeros(len(train_df), dtype=np.float64)

    all_idx = np.arange(len(train_df))
    te_accum = np.zeros(len(test_df), dtype=np.float64)

    for val_idx in folds:
        tr_idx = np.setdiff1d(all_idx, val_idx, assume_unique=False)
        tr = train_df.iloc[tr_idx]
        prior = float(tr["target"].mean())

        stats = tr.groupby(col)["target"].agg(["mean", "count"])
        smooth = (stats["count"] * stats["mean"] + smoothing * prior) / (
            stats["count"] + smoothing
        )

        oof[val_idx] = (
            train_df.iloc[val_idx][col]
            .map(smooth)
            .fillna(prior)
            .astype(np.float64)
            .values
        )

        te_fold = test_df[col].map(smooth).fillna(prior).astype(np.float64).values
        te_accum += te_fold

    test_enc = te_accum / float(len(folds))

    if noise > 0:
        oof = np.clip(oof + rng.normal(0, noise, size=oof.shape), 0, 1)
        test_enc = np.clip(test_enc + rng.normal(0, noise, size=test_enc.shape), 0, 1)

    return oof, test_enc


train_tmp = train[
    [
        "image_name",
        "patient_id",
        "target",
        "sex",
        "age_approx",
        "anatom_site_general_challenge",
    ]
].copy()
test_tmp = test[
    ["image_name", "patient_id", "sex", "age_approx", "anatom_site_general_challenge"]
].copy()

age_bins = [-np.inf, 20, 30, 40, 50, 60, 70, 80, np.inf]
train_tmp["age_bin"] = pd.cut(train_tmp["age_approx"], bins=age_bins).astype(str)
test_tmp["age_bin"] = pd.cut(test_tmp["age_approx"], bins=age_bins).astype(str)

train_tmp["sex_site"] = (
    train_tmp["sex"].astype(str)
    + "_"
    + train_tmp["anatom_site_general_challenge"].astype(str)
)
test_tmp["sex_site"] = (
    test_tmp["sex"].astype(str)
    + "_"
    + test_tmp["anatom_site_general_challenge"].astype(str)
)

train_tmp["site_age"] = (
    train_tmp["anatom_site_general_challenge"].astype(str)
    + "_"
    + train_tmp["age_bin"].astype(str)
)
test_tmp["site_age"] = (
    test_tmp["anatom_site_general_challenge"].astype(str)
    + "_"
    + test_tmp["age_bin"].astype(str)
)

train_tmp["sex_site_age"] = (
    train_tmp["sex_site"].astype(str) + "_" + train_tmp["age_bin"].astype(str)
)
test_tmp["sex_site_age"] = (
    test_tmp["sex_site"].astype(str) + "_" + test_tmp["age_bin"].astype(str)
)

train_tmp["sex_age"] = (
    train_tmp["sex"].astype(str) + "_" + train_tmp["age_bin"].astype(str)
)
test_tmp["sex_age"] = (
    test_tmp["sex"].astype(str) + "_" + test_tmp["age_bin"].astype(str)
)

train_tmp["patient_id_enc"] = train_tmp["patient_id"].astype(str)
test_tmp["patient_id_enc"] = test_tmp["patient_id"].astype(str)

train_tmp["site_sex_age"] = (
    train_tmp["anatom_site_general_challenge"].astype(str)
    + "_"
    + train_tmp["sex"].astype(str)
    + "_"
    + train_tmp["age_bin"].astype(str)
)
test_tmp["site_sex_age"] = (
    test_tmp["anatom_site_general_challenge"].astype(str)
    + "_"
    + test_tmp["sex"].astype(str)
    + "_"
    + test_tmp["age_bin"].astype(str)
)

TIE_NOISE = 0.0

oof1, te1 = oof_target_encode(
    train_tmp,
    test_tmp,
    "anatom_site_general_challenge",
    folds=folds,
    smoothing=5.0,
    noise=TIE_NOISE,
    seed=1,
)
oof2, te2 = oof_target_encode(
    train_tmp, test_tmp, "sex", folds=folds, smoothing=20.0, noise=TIE_NOISE, seed=2
)
oof3, te3 = oof_target_encode(
    train_tmp, test_tmp, "age_bin", folds=folds, smoothing=10.0, noise=TIE_NOISE, seed=3
)
oof4, te4 = oof_target_encode(
    train_tmp,
    test_tmp,
    "sex_site",
    folds=folds,
    smoothing=30.0,
    noise=TIE_NOISE,
    seed=4,
)
oof5, te5 = oof_target_encode(
    train_tmp,
    test_tmp,
    "site_age",
    folds=folds,
    smoothing=30.0,
    noise=TIE_NOISE,
    seed=5,
)
oof6, te6 = oof_target_encode(
    train_tmp,
    test_tmp,
    "sex_site_age",
    folds=folds,
    smoothing=40.0,
    noise=TIE_NOISE,
    seed=6,
)

oof7, te7 = oof_target_encode(
    train_tmp,
    test_tmp,
    "sex_age",
    folds=folds,
    smoothing=25.0,
    noise=TIE_NOISE,
    seed=7,
)

oof8, te8 = oof_target_encode(
    train_tmp,
    test_tmp,
    "patient_id_enc",
    folds=folds,
    smoothing=2000.0,
    noise=TIE_NOISE,
    seed=8,
)

oof9, te9 = oof_target_encode(
    train_tmp,
    test_tmp,
    "site_sex_age",
    folds=folds,
    smoothing=50.0,
    noise=TIE_NOISE,
    seed=9,
)

sub1 = pd.DataFrame({"image_name": test_tmp["image_name"].values, "target": te1})
sub2 = pd.DataFrame({"image_name": test_tmp["image_name"].values, "target": te2})
sub3 = pd.DataFrame({"image_name": test_tmp["image_name"].values, "target": te3})
sub4 = pd.DataFrame({"image_name": test_tmp["image_name"].values, "target": te4})
sub5 = pd.DataFrame({"image_name": test_tmp["image_name"].values, "target": te5})
sub6 = pd.DataFrame({"image_name": test_tmp["image_name"].values, "target": te6})
sub7 = pd.DataFrame({"image_name": test_tmp["image_name"].values, "target": te7})
sub8 = pd.DataFrame({"image_name": test_tmp["image_name"].values, "target": te8})
sub9 = pd.DataFrame({"image_name": test_tmp["image_name"].values, "target": te9})


def rank_data(sub: pd.DataFrame) -> pd.DataFrame:
    sub = sub.copy()
    sub["target"] = sub["target"].rank(method="average", pct=True).astype(np.float64)
    return sub


def rank_array(a: np.ndarray) -> np.ndarray:
    s = pd.Series(a)
    r = s.rank(method="average", pct=True).astype(np.float64).to_numpy()
    return r


oof_list = [oof1, oof2, oof3, oof4, oof5, oof6, oof7, oof8, oof9]
oof_ranked = [rank_array(x) for x in oof_list]

sub_list = [sub1, sub2, sub3, sub4, sub5, sub6, sub7, sub8, sub9]
sub_list = [rank_data(s) for s in sub_list]

for i, s in enumerate(sub_list, start=1):
    s.columns = ["image_name", f"target{i}"]

f_sub = sub_list[0]
for s in sub_list[1:]:
    f_sub = f_sub.merge(s, on="image_name")




## === cell 1
def _sigmoid(z):
    z = np.clip(z, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-z))


def _ridge_logistic_irls(X, y, lam=1e-2, iters=25):
    """
    Ridge-regularized logistic regression via IRLS.
    X: (n, p) includes intercept column if desired.
    y: (n,) in {0,1}
    Returns beta (p,)
    """
    n, p = X.shape
    beta = np.zeros(p, dtype=np.float64)

    reg = np.ones(p, dtype=np.float64)
    reg[0] = 0.0

    I = np.eye(p, dtype=np.float64)

    for _ in range(iters):
        eta = X @ beta
        p_hat = _sigmoid(eta)

        w = p_hat * (1.0 - p_hat)
        w = np.clip(w, 1e-6, None)
        z = eta + (y - p_hat) / w

        Xw = X * w[:, None]
        A = X.T @ Xw + lam * (I * reg)
        b = X.T @ (w * z)
        beta_new = np.linalg.solve(A, b)

        beta = 0.5 * beta + 0.5 * beta_new

    return beta


X = np.vstack(oof_ranked).T  # (n, p)
yy = y.astype(np.float64)

mu = X.mean(axis=0)
sd = X.std(axis=0)
sd = np.where(sd < 1e-12, 1.0, sd)
Xz = (X - mu) / sd

X1 = np.concatenate([np.ones((Xz.shape[0], 1), dtype=np.float64), Xz], axis=1)

lam = 5e-2
beta = _ridge_logistic_irls(X1, yy, lam=lam, iters=25)

Xt = f_sub[[c for c in f_sub.columns if c.startswith("target")]].to_numpy(
    dtype=np.float64
)
Xtz = (Xt - mu) / sd
Xt1 = np.concatenate([np.ones((Xtz.shape[0], 1), dtype=np.float64), Xtz], axis=1)

pred = _sigmoid(Xt1 @ beta)

stable_tie = (
    pd.Series(f_sub["image_name"])
    .astype(str)
    .apply(lambda s: int(s.split("_")[-1]))  # stable for ISIC_########
    .astype(np.float64)
    .to_numpy()
)
stable_tie = (stable_tie - stable_tie.min()) / (
    stable_tie.max() - stable_tie.min() + 1e-12
)
pred = pred + 1e-12 * stable_tie

pred = rank_array(pred)

f_sub["target"] = np.clip(pred, 0.0, 1.0).astype(np.float64)
f_sub = f_sub[["image_name", "target"]]

sample_path = _find_file("sample_submission.csv")
sample = pd.read_csv(sample_path)[["image_name"]]
f_sub = sample.merge(f_sub, on="image_name", how="left")
f_sub["target"] = f_sub["target"].fillna(global_mean).clip(0, 1)

f_sub.to_csv("blend_sub.csv", index=False)

w0 = float(beta[0])
w = beta[1:].astype(np.float64)

print(
    f"Saved submission: blend_sub.csv with shape {f_sub.shape} and columns {list(f_sub.columns)}"
)
print("Learned intercept and weights (target1..target9):")
print(w0, w.tolist())
print(f_sub.head())
