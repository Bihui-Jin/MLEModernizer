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

0.9254427831354688

# 6. Current score

0.68613

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.62556) has done: 'Your notebook fails because it tries to read three external submission CSVs from `../input/output-of-best-public-submission/`, which do not exist in your environment; this prevents `first/second/third` from being defined and therefore also breaks the blending cell. I replace that unavailable dependency with a minimal, fully self-contained baseline that only uses the provided `train.csv`/`test.csv` metadata and produces a valid `submission.csv` with the required columns. To keep core intent similar (a simple probabilistic blend), I compute smoothed target rates per categorical group and a global prior, then combine them into final probabilities and clip to a safe range. This run end-to-end under the listed packages and write `submission.csv` in the working directory.'
- What this solution (achieved 0.64868) has done: 'Your current score (0.62556) is far below the target (0.92544), so we should legitimately improve discrimination while keeping the same “smoothed target-rate blend from metadata” core idea. The biggest gain with minimal semantic change is to use higher-signal groupings (interactions like site×age_bin and site×sex) and to blend them with the existing single-column rates, rather than relying mostly on site alone. We keep the same smoothed-mean target encoding (same logic), just add a few additional grouped rates and slightly rebalance weights toward the higher-signal interactions. We also ensure the submission stays aligned to `sample_submission.csv` ordering and remains a valid probability in [0,1].'
- What this solution (achieved 0.67536) has done: 'Your current score (0.64868) is far below the target (0.92544), so we should improve discrimination while keeping the same “smoothed target-rate blend from metadata” core idea. The biggest low-risk gain is to add two more high-signal groupings available in metadata: `patient_id` (strong within-patient correlation) and `site×age_bin×sex` (a 3-way interaction), then blend them into the existing weighted average. To reduce noise/overfitting from rare groups without changing the method, we (a) use stronger smoothing for `patient_id` and the 3-way interaction, and (b) slightly rebalance weights toward these higher-signal encodings while keeping the same submission alignment and probability clipping. This remains a purely smoothed target-rate encoding blend (same semantics), just with a couple additional grouped rates.'
- What this solution (achieved 0.67728) has done: 'Your current score (0.67536) is far below the target (0.92544), so we should increase discrimination while keeping the same “smoothed target-rate encoding blend from metadata” core logic. The most impactful minimal change is to reduce noise/overfitting in the highest-signal but potentially leaky/unstable feature (`patient_id`) by computing it out-of-fold (OOF) on the training set, then using that more reliable estimate for the test mapping. In parallel, we can safely add a similarly high-signal interaction available in metadata (`patient_id×anatom_site_general_challenge`) with strong smoothing, and rebalance weights slightly toward these higher-signal encodings without changing the method. These changes keep the same target-encoding+weighted-average semantics and should move AUC upward toward the target.'
- What this solution (achieved 0.70042) has done: 'We keep your current “smoothed target-rate encoding + weighted blend” approach intact and only adjust it in ways that typically increase AUC for this specific competition without changing modeling semantics. The main change is to compute out-of-fold (OOF) encodings not just for `patient_id` but also for the high-signal `patient_id||site` interaction, which reduces leakage/overfitting noise and usually improves generalization to the test distribution. Then we use those OOF encodings to set the blend weights in a minimally invasive, data-driven way (optimize AUC on OOF predictions), replacing the hand-tuned weights while keeping the same blended-probability structure. Finally, we still generate the same `submission.csv` with the required columns and ordering aligned to `sample_submission.csv`.'
- What this solution (achieved 0.69847) has done: 'We keep your exact “smoothed target-rate encodings + linear nonnegative simplex blend” core logic, but fix the main remaining generalization issue: most of the OOF features in `X_oof` are currently *not* truly out-of-fold (they’re mapped from full-data dictionaries), which inflates the OOF AUC used to pick weights and tends to pick weights that don’t transfer to test. I compute OOF encodings (using the same folds you already build) for all group features used in the weight search, so the weight tuning is based on leakage-free OOF predictions. Then we keep the same coordinate-ascent simplex optimization, but it choose weights that more reliably improve leaderboard AUC (moving you toward the 0.925 target). Submission writing, columns, and ordering remain unchanged.'
- What this solution (achieved 0.6828) has done: 'Your current AUC (0.69847) is far below the target (0.92544), so we should improve discrimination while keeping the same “smoothed target-rate encodings + nonnegative simplex blend” core logic. The smallest high-impact change is to use a slightly stronger, more reliable fold scheme for OOF encodings by making folds grouped by `patient_id` (to reduce leakage across the same patient between train/valid within OOF), while keeping the exact same encoding and blend optimization approach. Then we recompute all OOF encodings with these grouped folds and re-run the same coordinate-ascent simplex weight tuning, which should select weights that generalize better to test. Submission format, ordering (via merge with `sample_submission.csv`), and probability clipping remain unchanged.'
- What this solution (achieved 0.68254) has done: 'Your current AUC (0.6828) is still far below the target (0.92544), so we should increase discrimination while keeping your exact “OOF smoothed target-rate encodings + nonnegative simplex blend” approach. The smallest, highest-impact change is to fix the fold construction: your current grouped fold assignment ignores group sizes, so some folds can become heavily imbalanced and harm OOF weight selection and generalization. I replace it with a deterministic, group-aware stratified assignment that balances folds by (a) malignant counts and (b) sample counts per patient bucket, without changing any encoding/blending logic. Everything else (features, smoothing, weight search, submission format/order) remains the same.'
- What this solution (achieved 0.68392) has done: 'Your current AUC (0.6825) is far below the target (0.9254), so we should improve discrimination while keeping the same smoothed target-rate encodings + nonnegative simplex blend core logic. The biggest minimal win here is to add one more high-signal, still-metadata-only interaction (`patient_id||age_bin`) and compute it OOF with the same patient-grouped folds, then include it as an additional blended feature (everything else—smoothing idea, OOF procedure, coordinate-ascent simplex weight search—stays the same). This adds complementary signal to `patient_id` and `patient_id||site` without changing the modeling approach, and the weight optimizer automatically use it only if it improves leakage-free OOF AUC. Submission writing/ordering is kept identical and still produces a valid `submission.csv`.'
- What this solution (achieved 0.68411) has done: 'We keep your exact “OOF smoothed target-rate encodings + nonnegative simplex blend” approach, but make two minimal changes that usually improve AUC for this competition without changing semantics: (1) optimize blend weights against a *patient-grouped OOF AUC* (so weight search isn’t misled by within-patient correlations), and (2) allow a tiny amount of monotonic calibration by fitting an out-of-fold isotonic regressor on the blended OOF predictions and applying it to test predictions (AUC is rank-based, but isotonic often fixes small inversions from noisy encodings). This doesn’t change any features/encodings, training loops, or loss—just how the final blend is tuned and optionally monotonic-calibrated in an OOF-safe way. The submission format and ordering remain aligned to `sample_submission.csv`, and we still write `submission.csv` end-to-end.'
- What this solution (achieved 0.68764) has done: 'Your current AUC (0.68411) is far below the target (0.92544), so we should make a small, legitimate change that increases discrimination without changing your core “OOF smoothed target-rate encodings + nonnegative simplex blend” logic. The main issue is that weight tuning is currently optimizing a *mean of per-fold AUCs*, which is not the same as the true OOF AUC and can select suboptimal weights; we switch the objective to the standard overall OOF AUC computed on all rows (still using your patient-grouped folds for leakage control). To keep semantics identical, we won’t change features/encodings or the coordinate-ascent optimizer—only the AUC function used for weight selection (and isotonic remains OOF-safe). This minimal fix is expected to move AUC upward toward the target while preserving your pipeline and submission format.'
- What this solution (achieved 0.68309) has done: 'We keep your exact “OOF smoothed target-rate encodings + nonnegative simplex blend (+ optional isotonic)” pipeline, but make a minimal change that typically improves AUC for this competition: build folds using `StratifiedGroupKFold` (patient-grouped *and* label-stratified) instead of the custom heuristic fold builder, because better-balanced folds make both OOF encodings and weight selection more reliable. To preserve semantics, the same encodings, alphas, weight search, and isotonic OOF calibration remain unchanged; only the fold assignment mechanism changes. This should move your score upward toward the target while staying within the same modeling approach and producing the same `submission.csv` format.'
- What this solution (achieved 0.68613) has done: 'Your current AUC (0.683) is still far below the target (0.925), so we should increase discrimination while keeping the same “OOF smoothed target-rate encodings + nonnegative simplex blend (+ optional isotonic)” approach. The most score-relevant minimal fix is that folds/OOF encoding should be computed on a *train split only* (not the same rows used to compute global prior), otherwise the OOF signal for weight tuning can be biased; we compute `global_prior` inside each fold for OOF encodings (still smoothed means, same semantics) while keeping the same alphas and features. Next, we tune blend weights against a *patient-grouped OOF AUC* objective (using the same `folds`) rather than plain overall AUC, which better matches the competition’s patient correlation structure and usually improves leaderboard AUC without changing the blending logic. Finally, we keep isotonic calibration but fit it using the same grouped folds and apply it consistently, preserving the end-to-end pipeline and producing a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os

BASE_DIR = "/kaggle/input/siim-isic-melanoma-classification"
train_path = os.path.join(BASE_DIR, "train.csv")
test_path = os.path.join(BASE_DIR, "test.csv")
sample_path = os.path.join(BASE_DIR, "sample_submission.csv")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
sample_sub = pd.read_csv(sample_path)

for df in (train, test):
    df["sex"] = df["sex"].fillna("unknown").replace("", "unknown")
    df["anatom_site_general_challenge"] = (
        df["anatom_site_general_challenge"].fillna("unknown").replace("", "unknown")
    )

train["age_approx"] = pd.to_numeric(train["age_approx"], errors="coerce")
test["age_approx"] = pd.to_numeric(test["age_approx"], errors="coerce")

age_bins = [-np.inf, 20, 30, 40, 50, 60, 70, 80, np.inf]
age_labels = ["<=20", "21-30", "31-40", "41-50", "51-60", "61-70", "71-80", "81+"]
train["age_bin"] = pd.cut(train["age_approx"], bins=age_bins, labels=age_labels)
test["age_bin"] = pd.cut(test["age_approx"], bins=age_bins, labels=age_labels)
train["age_bin"] = train["age_bin"].astype("object").fillna("unknown")
test["age_bin"] = test["age_bin"].astype("object").fillna("unknown")

y = train["target"].astype(float).values
global_prior = (y.sum() + 1.0) / (len(y) + 2.0)


def smoothed_target_rate(train_df, group_col, alpha=50.0, prior=None):
    """
    Minimal score-relevant change: allow passing a fold-specific prior for OOF encodings.
    This keeps identical "smoothed mean target encoding" semantics but makes OOF truly fold-pure.
    """
    if prior is None:
        prior = global_prior
    grp = train_df.groupby(group_col)["target"].agg(["sum", "count"])
    rate = (grp["sum"] + alpha * prior) / (grp["count"] + alpha)
    return rate.to_dict()


def map_rate(series, rate_dict, default):
    return series.map(rate_dict).fillna(default).astype(float)


def make_grouped_stratified_folds_binary(y_arr, groups, seed=42, n_splits=5):
    """
    Patient-grouped, label-stratified folds (kept as-is).
    """
    y_arr = np.asarray(y_arr).astype(int)
    groups = np.asarray(groups)

    try:
        from sklearn.model_selection import StratifiedGroupKFold

        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        folds = np.empty(len(y_arr), dtype=np.int32)
        for f, (_, val_idx) in enumerate(
            sgkf.split(np.zeros(len(y_arr)), y_arr, groups=groups)
        ):
            folds[val_idx] = f
        return folds
    except Exception:
        rng = np.random.RandomState(seed)
        unique_groups = pd.unique(groups)
        rng.shuffle(unique_groups)
        fold_of_group = {g: i % n_splits for i, g in enumerate(unique_groups)}
        return np.array([fold_of_group[g] for g in groups], dtype=np.int32)


def oof_smoothed_target_rate_with_folds(train_df, group_col, folds, alpha=500.0):
    """
    Minimal score-relevant change: compute fold-specific prior using only training folds.
    This reduces subtle bias in OOF encodings used for weight tuning, often improving AUC.
    """
    df = train_df[[group_col, "target"]].copy()
    oof = np.full(len(df), global_prior, dtype=float)

    for f in range(int(folds.max()) + 1):
        trn_mask = folds != f
        val_mask = folds == f

        y_trn = df.loc[trn_mask, "target"].astype(float).values
        prior_f = (y_trn.sum() + 1.0) / (len(y_trn) + 2.0)

        rate_dict = smoothed_target_rate(
            df.loc[trn_mask], group_col, alpha=alpha, prior=prior_f
        )
        oof[val_mask] = map_rate(df.loc[val_mask, group_col], rate_dict, prior_f).values

    return pd.Series(oof, index=train_df.index)


train = train.copy()
test = test.copy()

train["site_age"] = (
    train["anatom_site_general_challenge"].astype(str)
    + "||"
    + train["age_bin"].astype(str)
)
test["site_age"] = (
    test["anatom_site_general_challenge"].astype(str)
    + "||"
    + test["age_bin"].astype(str)
)
train["site_sex"] = (
    train["anatom_site_general_challenge"].astype(str) + "||" + train["sex"].astype(str)
)
test["site_sex"] = (
    test["anatom_site_general_challenge"].astype(str) + "||" + test["sex"].astype(str)
)
train["sex_age"] = train["sex"].astype(str) + "||" + train["age_bin"].astype(str)
test["sex_age"] = test["sex"].astype(str) + "||" + test["age_bin"].astype(str)

train["site_age_sex"] = (
    train["anatom_site_general_challenge"].astype(str)
    + "||"
    + train["age_bin"].astype(str)
    + "||"
    + train["sex"].astype(str)
)
test["site_age_sex"] = (
    test["anatom_site_general_challenge"].astype(str)
    + "||"
    + test["age_bin"].astype(str)
    + "||"
    + test["sex"].astype(str)
)

train["patient_site"] = (
    train["patient_id"].astype(str)
    + "||"
    + train["anatom_site_general_challenge"].astype(str)
)
test["patient_site"] = (
    test["patient_id"].astype(str)
    + "||"
    + test["anatom_site_general_challenge"].astype(str)
)

train["patient_age"] = (
    train["patient_id"].astype(str) + "||" + train["age_bin"].astype(str)
)
test["patient_age"] = (
    test["patient_id"].astype(str) + "||" + test["age_bin"].astype(str)
)

rate_sex = smoothed_target_rate(train, "sex", alpha=50.0)
rate_site = smoothed_target_rate(train, "anatom_site_general_challenge", alpha=100.0)
rate_age = smoothed_target_rate(train, "age_bin", alpha=50.0)

rate_site_age = smoothed_target_rate(train, "site_age", alpha=200.0)
rate_site_sex = smoothed_target_rate(train, "site_sex", alpha=200.0)
rate_sex_age = smoothed_target_rate(train, "sex_age", alpha=150.0)
rate_site_age_sex = smoothed_target_rate(train, "site_age_sex", alpha=400.0)

folds = make_grouped_stratified_folds_binary(
    train["target"].values, groups=train["patient_id"].values, seed=42, n_splits=5
)

train["pred_patient_oof"] = oof_smoothed_target_rate_with_folds(
    train, "patient_id", folds=folds, alpha=700.0
)
train["pred_patient_site_oof"] = oof_smoothed_target_rate_with_folds(
    train, "patient_site", folds=folds, alpha=900.0
)
train["pred_patient_age_oof"] = oof_smoothed_target_rate_with_folds(
    train, "patient_age", folds=folds, alpha=900.0
)

rate_patient_full = smoothed_target_rate(train, "patient_id", alpha=700.0)
rate_patient_site = smoothed_target_rate(train, "patient_site", alpha=900.0)
rate_patient_age = smoothed_target_rate(train, "patient_age", alpha=900.0)

pred_sex = map_rate(test["sex"], rate_sex, global_prior)
pred_site = map_rate(test["anatom_site_general_challenge"], rate_site, global_prior)
pred_age = map_rate(test["age_bin"], rate_age, global_prior)

pred_site_age = map_rate(test["site_age"], rate_site_age, global_prior)
pred_site_sex = map_rate(test["site_sex"], rate_site_sex, global_prior)
pred_sex_age = map_rate(test["sex_age"], rate_sex_age, global_prior)

pred_patient = map_rate(test["patient_id"], rate_patient_full, global_prior)
pred_site_age_sex = map_rate(test["site_age_sex"], rate_site_age_sex, global_prior)
pred_patient_site = map_rate(test["patient_site"], rate_patient_site, global_prior)
pred_patient_age = map_rate(test["patient_age"], rate_patient_age, global_prior)

test["pred_sex"] = pred_sex.values
test["pred_site"] = pred_site.values
test["pred_age"] = pred_age.values
test["pred_site_age"] = pred_site_age.values
test["pred_site_sex"] = pred_site_sex.values
test["pred_sex_age"] = pred_sex_age.values
test["pred_patient"] = pred_patient.values
test["pred_site_age_sex"] = pred_site_age_sex.values
test["pred_patient_site"] = pred_patient_site.values
test["pred_patient_age"] = pred_patient_age.values

submission = sample_sub.copy()
submission = submission.merge(
    test[
        [
            "image_name",
            "pred_site",
            "pred_sex",
            "pred_age",
            "pred_site_age",
            "pred_site_sex",
            "pred_sex_age",
            "pred_patient",
            "pred_site_age_sex",
            "pred_patient_site",
            "pred_patient_age",
        ]
    ],
    on="image_name",
    how="left",
)

for c in [
    "pred_site",
    "pred_sex",
    "pred_age",
    "pred_site_age",
    "pred_site_sex",
    "pred_sex_age",
    "pred_patient",
    "pred_site_age_sex",
    "pred_patient_site",
    "pred_patient_age",
]:
    submission[c] = submission[c].fillna(global_prior).astype(float)




## === cell 1
def fast_auc(y_true, y_score):
    y_true = np.asarray(y_true, dtype=np.int32)
    y_score = np.asarray(y_score, dtype=np.float64)

    order = np.argsort(y_score)
    y_sorted = y_true[order]
    n1 = y_sorted.sum()
    n0 = len(y_sorted) - n1
    if n0 == 0 or n1 == 0:
        return 0.5

    ranks = np.arange(1, len(y_sorted) + 1, dtype=np.float64)

    scores_sorted = y_score[order]
    i = 0
    while i < len(scores_sorted):
        j = i + 1
        while j < len(scores_sorted) and scores_sorted[j] == scores_sorted[i]:
            j += 1
        if j - i > 1:
            avg_rank = ranks[i:j].mean()
            ranks[i:j] = avg_rank
        i = j

    sum_ranks_pos = ranks[y_sorted == 1].sum()
    auc = (sum_ranks_pos - n1 * (n1 + 1) / 2.0) / (n0 * n1)
    return float(auc)


def grouped_auc_by_folds(y_true, y_pred, folds):
    y_true = np.asarray(y_true, dtype=np.int32)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    folds = np.asarray(folds, dtype=np.int32)
    aucs = []
    for f in range(int(folds.max()) + 1):
        m = folds == f
        if m.sum() < 2:
            continue
        aucs.append(fast_auc(y_true[m], y_pred[m]))
    if len(aucs) == 0:
        return 0.5
    return float(np.mean(aucs))


oof_pred_site_age_sex = oof_smoothed_target_rate_with_folds(
    train, "site_age_sex", folds=folds, alpha=400.0
)
oof_pred_site_age = oof_smoothed_target_rate_with_folds(
    train, "site_age", folds=folds, alpha=200.0
)
oof_pred_site_sex = oof_smoothed_target_rate_with_folds(
    train, "site_sex", folds=folds, alpha=200.0
)
oof_pred_sex_age = oof_smoothed_target_rate_with_folds(
    train, "sex_age", folds=folds, alpha=150.0
)
oof_pred_site = oof_smoothed_target_rate_with_folds(
    train, "anatom_site_general_challenge", folds=folds, alpha=100.0
)
oof_pred_age = oof_smoothed_target_rate_with_folds(
    train, "age_bin", folds=folds, alpha=50.0
)
oof_pred_sex = oof_smoothed_target_rate_with_folds(
    train, "sex", folds=folds, alpha=50.0
)

X_oof = pd.DataFrame(
    {
        "pred_site_age_sex": oof_pred_site_age_sex.values,
        "pred_patient": train["pred_patient_oof"].values,
        "pred_patient_site": train["pred_patient_site_oof"].values,
        "pred_patient_age": train["pred_patient_age_oof"].values,
        "pred_site_age": oof_pred_site_age.values,
        "pred_site_sex": oof_pred_site_sex.values,
        "pred_sex_age": oof_pred_sex_age.values,
        "pred_site": oof_pred_site.values,
        "pred_age": oof_pred_age.values,
        "pred_sex": oof_pred_sex.values,
    },
    index=train.index,
)
y_true = train["target"].values.astype(int)

cols = list(X_oof.columns)

w0 = np.array(
    [0.18, 0.24, 0.14, 0.04, 0.18, 0.10, 0.04, 0.03, 0.01, 0.00],
    dtype=np.float64,
)


def project_simplex_nonneg(w):
    w = np.maximum(w, 0.0)
    s = w.sum()
    if s <= 0:
        w[:] = 0.0
        w[0] = 1.0
        return w
    return w / s


def weight_objective(y_true_arr, pred_arr, folds_arr):
    return grouped_auc_by_folds(y_true_arr, pred_arr, folds_arr)


w_best = project_simplex_nonneg(w0.copy())
pred_best = (X_oof.values @ w_best).clip(0.0, 1.0)
auc_best = weight_objective(y_true, pred_best, folds)

steps = [0.02, 0.01, 0.005]
max_passes = 2

for step in steps:
    for _ in range(max_passes):
        improved = False
        for i in range(len(w_best)):
            for delta in (-step, step):
                w_try = w_best.copy()
                w_try[i] += delta
                w_try = project_simplex_nonneg(w_try)
                pred_try = (X_oof.values @ w_try).clip(0.0, 1.0)
                auc_try = weight_objective(y_true, pred_try, folds)
                if auc_try > auc_best + 1e-6:
                    w_best, auc_best = w_try, auc_try
                    improved = True
        if not improved:
            break

w = dict(zip(cols, w_best.tolist()))

test_blend = (
    (
        w["pred_site_age_sex"] * submission["pred_site_age_sex"]
        + w["pred_patient"] * submission["pred_patient"]
        + w["pred_patient_site"] * submission["pred_patient_site"]
        + w["pred_patient_age"] * submission["pred_patient_age"]
        + w["pred_site_age"] * submission["pred_site_age"]
        + w["pred_site_sex"] * submission["pred_site_sex"]
        + w["pred_sex_age"] * submission["pred_sex_age"]
        + w["pred_site"] * submission["pred_site"]
        + w["pred_age"] * submission["pred_age"]
        + w["pred_sex"] * submission["pred_sex"]
    )
    .astype(float)
    .clip(0.0, 1.0)
)

oof_blend = (X_oof.values @ w_best).astype(float).clip(0.0, 1.0)

try:
    from sklearn.isotonic import IsotonicRegression

    oof_cal = np.zeros_like(oof_blend, dtype=float)
    test_cal_folds = []

    for f in range(int(folds.max()) + 1):
        trn_mask = folds != f
        val_mask = folds == f

        ir = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        ir.fit(oof_blend[trn_mask], y_true[trn_mask].astype(float))

        oof_cal[val_mask] = ir.predict(oof_blend[val_mask])
        test_cal_folds.append(ir.predict(test_blend.values))

    test_blend = np.mean(np.vstack(test_cal_folds), axis=0).clip(0.0, 1.0)

except Exception:
    pass

submission["target"] = np.asarray(test_blend, dtype=float).clip(0.0, 1.0)
submission = submission[["image_name", "target"]]
submission.to_csv("submission.csv", index=False)

print("W (simplex) =", w)
print("OOF AUC (grouped-fold objective used for weights) =", float(auc_best))
print(submission.head())
