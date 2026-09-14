# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.9402766097647106

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'I remove the dependency on the missing `../input/ensemble-melanoma` directory (which causes the crash) and instead build a simple, valid baseline submission from the provided competition files. Since no model packages are available here, the safest score-improving (vs. random) minimal approach is to use the training-set malignancy rate as a constant probability for all test images, which is a standard AUC baseline. I also ensure the submission has exactly the required columns (`image_name`, `target`), matches the test row order, and is written to a `.csv` file. The rest of the original “ensemble” cells be kept but guarded so they don’t error when the external folder is absent.'
- What this solution (achieved 0.66789) has done: 'Your current 0.5 AUC is consistent with producing an almost-constant prediction; to move toward the 0.94 target without changing the overall “simple tabular baseline” approach, I replace the constant-probability submission with a minimal logistic-regression model trained on the provided metadata (sex, age, anatomic site) and evaluated with ROC-AUC. This keeps the core pipeline (read CSVs → train on train.csv metadata → predict probabilities for test.csv → write submission.csv) while adding only lightweight ML using scikit-learn. I also add a patient-wise train/validation split to reduce leakage and to sanity-check that we’re learning signal before writing the final submission. The ensemble-folder logic is left intact and guarded, but the main submission now be the metadata-model output (which should improve meaningfully over 0.5).'
- What this solution (achieved 0.34234) has done: 'Your current 0.66789 AUC is far below the 0.9403 target, so we should improve the *same metadata-only pipeline* with minimal, legitimate modeling tweaks rather than changing approach. I keep the same inputs/features and LogisticRegression, but (1) switch to a more appropriate solver for sparse one-hot features, (2) add mild class balancing (this competition is highly imbalanced), and (3) use a small group-aware cross-validated search over `C` to pick a better regularization strength without changing the model family. I also ensure the output submission keeps the exact `test.csv` order and schema, and still writes a valid `submission.csv`.'
- What this solution (achieved 0.67823) has done: 'Your current score (0.34234) is far below the target (0.9403), and the biggest likely cause is that this environment doesn’t have scikit-learn installed (only sklearn-pandas is listed), so the intended logistic-regression pipeline either didn’t actually run in Kaggle or produced a degenerate submission. I keep the exact same core “metadata-only tabular model” approach, but implement the logistic regression training/prediction in pure NumPy/Pandas (no new dependencies) with the same features, L2 regularization, class balancing, and group-aware CV selection of C. I also preserve the guarded ensemble-folder logic and ensure the final `submission.csv` matches `test.csv` order and required columns. This should legitimately improve AUC versus a constant baseline and move you materially closer to the target without changing the modeling intent.'
- What this solution (achieved 0.67823) has done: 'Your current 0.67823 AUC is far below the 0.9403 target, so we should improve the same metadata-only logistic-regression pipeline with the smallest safe, legitimate changes. The biggest win without changing the core approach is to add the strongest missing tabular signal: patient-level historical malignancy rate computed out-of-fold (so it avoids leakage) and then applied to test using the full train aggregation. I also add a tiny amount of feature scaling stabilization and keep the same L2-logreg training loop, CV selection of `C`, and submission writing logic. This should materially increase AUC while preserving evaluation semantics and producing the same `submission.csv` format.'
- What this solution (achieved 0.67823) has done: 'Your current 0.67823 AUC is far below the 0.9403 target, so we should improve within the same “metadata-only L2 logistic regression” core logic by fixing the two biggest low-risk issues: (1) your one-hot encoding drops unseen test categories (they become all-zeros), and (2) your patient prior is used as a centered linear feature, but it’s often more effective (and still the same model family) to use a stabilized logit transform as an input. I minimally change the design-matrix builder to create a union-of-categories encoder (train+test) and to logit-transform the patient prior (and keep it standardized), while keeping the same training loop, same CV selection over C, same class balancing, and same submission writing. These changes should legitimately increase AUC by improving generalization on test metadata without altering the overall approach. The script still run end-to-end and write `submission.csv` with the required `image_name,target` columns in test order.'
- What this solution (achieved 0.67776) has done: 'I fix the crash in `build_design_matrices` by constructing the `sex||site` interaction using Pandas string operations (which avoids NumPy’s `ufunc add` type error on mixed string dtypes) while keeping the exact same feature idea. I also add a small safety alignment step to ensure the produced `submission.csv` follows the exact `test.csv`/`sample_submission.csv` row order (score-neutral but prevents subtle mismatches). No changes are made to the model family, training loop, loss/regularization, or the patient-prior logic—this is purely to unblock execution and produce a valid submission. The script still optionally write ensemble/blended files only if the external folder exists.'
- What this solution (achieved 0.67904) has done: 'Your current AUC (0.67776) is far below the target (0.94028), so we should improve score while staying in the same “metadata-only + patient prior + L2 logistic regression” core pipeline. The smallest high-impact change is to make the optimizer actually converge more reliably by switching from fixed-step SGD to a stable Newton/IRLS solver (still the same logistic regression objective, just a better optimizer), and to avoid overfitting from overly-aggressive sample-weighting by using gentler class weights based on the effective sample-size ratio. I also keep your exact features (age, log-age, sex/site one-hot, interaction, patient prior logit, lesion count) and the same group-aware CV selection over C. These changes are directly aimed at improving ranking quality (AUC) without changing what the model is.'
- What this solution (achieved 0.68065) has done: 'I keep your exact metadata-only + patient-prior + L2-logistic core pipeline, but make two minimal, high-signal fixes aimed at improving AUC ranking toward your 0.94 target: (1) add the well-known “site × age” interaction (age effect differs strongly by body site) using the same one-hot framework, and (2) tune the patient-prior smoothing strength `m_smoothing` via group-aware CV on the training split (still the same prior feature, just selecting its stabilization level). I also make the IRLS solver a bit more numerically stable (tiny ridge added to the Hessian diagonal) without changing the objective or convergence criteria. The script still runs end-to-end and writes a valid `submission.csv` with `image_name,target` in the exact test order.'
- What this solution (achieved 0.67464) has done: 'Your current AUC (0.68065) is far below the 0.94028 target, so we should improve ranking while keeping the same “metadata + patient prior + L2 logistic regression” pipeline intact. The smallest high-impact, low-risk fix is to add **out-of-fold target encodings** (with smoothing) for the two strongest categorical signals you already use (`anatom_site_general_challenge` and `sex`), computed patient-group-wise to avoid leakage, and then use their **logit** as additional standardized numeric features (same model, just better inputs). I also tune the smoothing strength for these encodings together with your existing `m_smoothing` and `C` using the same group-aware CV loop (same training approach/selection logic). Everything else (IRLS solver, class weighting mode, design-matrix core, submission alignment/writing, optional ensemble guard) stays the same.'
- What this solution (achieved 0.67336) has done: 'I keep your exact metadata-only + patient-prior + target-encoding + L2 logistic regression pipeline, but fix two high-impact issues that can hold AUC down without changing the “core logic”: (1) your current CV selection uses target encodings computed once using *all* of `train_df_split` (so each fold’s validation rows leak into the encoding features), which can select weak hyperparameters and generalize poorly; I recompute encodings in a fold-safe (OOF-within-CV) way for model selection. (2) The final model currently fits on the full training set, but the encoded features are computed with fixed smoothing only; I keep the same feature set but compute the encoding features in a fully consistent way (OOF for train features, full aggregation for test) using the selected hyperparameters. These are minimal changes (no new model family/loss/architecture), but they should legitimately improve generalization and move your score upward toward the 0.94 target. The script still run end-to-end and write a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)



## === cell 1
sub_path = "../input/ensemble-melanoma"
all_files = []
if os.path.isdir(sub_path):
    all_files = os.listdir(sub_path)
    all_files = [
        f
        for f in all_files
        if "submission_meta.csv" not in f
        and "seresnext50 mean tta 0.9252.csv" not in f
        and "b6 2019 mean 0.8666.csv" not in f
        and "cpu densenet121 0.8845.csv" not in f
    ]
    for extra in [
        "B3-B6 80 82 size 512.csv",
        "triple-stratified-kfold-with-tfrecords 0.9426.csv",
    ]:
        if os.path.exists(os.path.join(sub_path, extra)):
            all_files.append(extra)

all_files



## === cell 2
concat_sub = None
ncol = None

if len(all_files) > 0:
    outs = [pd.read_csv(os.path.join(sub_path, f), index_col=0) for f in all_files]
    concat_sub = pd.concat(outs, axis=1)
    cols = list(map(lambda x: "target" + str(x), range(len(concat_sub.columns))))
    concat_sub.columns = cols
    concat_sub.reset_index(inplace=True)
    ncol = concat_sub.shape[1]
    concat_sub.head()



## === cell 3
if concat_sub is not None and ncol is not None:
    concat_sub["target"] = concat_sub.iloc[:, 1:ncol].mean(axis=1)
    concat_sub[["image_name", "target"]].to_csv(
        "submission_mean.csv", index=False, float_format="%.6f"
    )



## === cell 4
DATA_ROOT_CANDIDATES = [
    "/kaggle/input/siim-isic-melanoma-classification",
    "/kaggle/data",
    "/kaggle/input",
    "../input/siim-isic-melanoma-classification",
    "../data",
]


def first_existing_file(rel_path):
    for root in DATA_ROOT_CANDIDATES:
        p = os.path.join(root, rel_path)
        if os.path.exists(p):
            return p
    return None


train_csv = first_existing_file("train.csv")
test_csv = first_existing_file("test.csv")
sample_sub_csv = first_existing_file("sample_submission.csv")

if train_csv is None or test_csv is None or sample_sub_csv is None:
    raise FileNotFoundError(
        f"Could not locate required CSVs. Found: train={train_csv}, test={test_csv}, sample={sample_sub_csv}"
    )

train_df = pd.read_csv(train_csv)
test_df = pd.read_csv(test_csv)
sample_sub = pd.read_csv(sample_sub_csv)

required_train_cols = {
    "target",
    "patient_id",
    "sex",
    "age_approx",
    "anatom_site_general_challenge",
}
required_test_cols = {
    "image_name",
    "patient_id",
    "sex",
    "age_approx",
    "anatom_site_general_challenge",
}

missing_train = required_train_cols - set(train_df.columns)
missing_test = required_test_cols - set(test_df.columns)
if missing_train:
    raise KeyError(f"train.csv missing required columns: {missing_train}")
if missing_test:
    raise KeyError(f"test.csv missing required columns: {missing_test}")

if list(sample_sub.columns) != ["image_name", "target"]:
    if "image_name" in sample_sub.columns and "target" in sample_sub.columns:
        sample_sub = sample_sub[["image_name", "target"]]
    else:
        raise KeyError("sample_submission.csv must contain columns: image_name, target")


def roc_auc_score_np(y_true, y_score):
    y_true = np.asarray(y_true).astype(np.int64)
    y_score = np.asarray(y_score).astype(np.float64)

    n_pos = int(y_true.sum())
    n = y_true.shape[0]
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.nan

    order = np.argsort(y_score, kind="mergesort")
    y_sorted = y_true[order]
    s_sorted = y_score[order]

    ranks = np.empty(n, dtype=np.float64)
    i = 0
    r = 1
    while i < n:
        j = i + 1
        while j < n and s_sorted[j] == s_sorted[i]:
            j += 1
        avg_rank = 0.5 * (r + (r + (j - i) - 1))
        ranks[i:j] = avg_rank
        r += j - i
        i = j

    sum_ranks_pos = float(ranks[y_sorted == 1].sum())
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def sigmoid(z):
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def logit(p):
    p = np.asarray(p, dtype=np.float64)
    eps = 1e-6
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def group_shuffle_split(groups, test_size=0.2, random_state=42):
    groups = np.asarray(groups)
    uniq = pd.unique(groups)
    rng_local = np.random.default_rng(int(random_state))
    rng_local.shuffle(uniq)
    n_test_g = int(np.ceil(len(uniq) * float(test_size)))
    test_g = set(uniq[:n_test_g].tolist())
    is_test = np.array([g in test_g for g in groups], dtype=bool)
    idx = np.arange(len(groups))
    return idx[~is_test], idx[is_test]


def group_kfold_indices(groups, n_splits=5, random_state=42):
    groups = np.asarray(groups)
    uniq = pd.unique(groups)
    rng_local = np.random.default_rng(int(random_state))
    rng_local.shuffle(uniq)
    folds = [[] for _ in range(n_splits)]
    for i, g in enumerate(uniq):
        folds[i % n_splits].append(g)
    group_to_fold = {}
    for k in range(n_splits):
        for g in folds[k]:
            group_to_fold[g] = k
    fold_id = np.array([group_to_fold[g] for g in groups], dtype=np.int64)

    idx = np.arange(len(groups))
    for k in range(n_splits):
        va = idx[fold_id == k]
        tr = idx[fold_id != k]
        yield tr, va


def add_patient_priors(
    train_df, test_df, n_splits=5, random_state=42, m_smoothing=20.0
):
    tr = train_df.copy()
    te = test_df.copy()

    y = tr["target"].astype(float).values
    pid = tr["patient_id"].astype(str).values
    global_mean = float(np.mean(y))

    oof_prior = np.full(len(tr), global_mean, dtype=np.float64)
    oof_cnt = np.zeros(len(tr), dtype=np.float64)

    for tr_idx, va_idx in group_kfold_indices(
        pid, n_splits=n_splits, random_state=random_state
    ):
        tr_fold = tr.iloc[tr_idx]
        agg = tr_fold.groupby("patient_id")["target"].agg(["sum", "count"])
        sm = (
            (agg["sum"] + m_smoothing * global_mean) / (agg["count"] + m_smoothing)
        ).astype(float)
        map_sm = sm.to_dict()
        map_cnt = agg["count"].astype(float).to_dict()

        va_pids = tr.iloc[va_idx]["patient_id"].astype(str).values
        oof_prior[va_idx] = np.array(
            [map_sm.get(p, global_mean) for p in va_pids], dtype=np.float64
        )
        oof_cnt[va_idx] = np.array(
            [map_cnt.get(p, 0.0) for p in va_pids], dtype=np.float64
        )

    agg_full = tr.groupby("patient_id")["target"].agg(["sum", "count"])
    sm_full = (
        (agg_full["sum"] + m_smoothing * global_mean)
        / (agg_full["count"] + m_smoothing)
    ).astype(float)
    map_full = sm_full.to_dict()
    map_full_cnt = agg_full["count"].astype(float).to_dict()

    te_pid = te["patient_id"].astype(str).values
    test_prior = np.array(
        [map_full.get(p, global_mean) for p in te_pid], dtype=np.float64
    )
    test_cnt = np.array([map_full_cnt.get(p, 0.0) for p in te_pid], dtype=np.float64)

    tr["patient_prior_oof"] = oof_prior
    te["patient_prior_full"] = test_prior

    tr["patient_lesion_count_oof"] = oof_cnt
    te["patient_lesion_count_full"] = test_cnt
    return tr, te


def add_categorical_target_encodings(
    train_df,
    test_df,
    cat_cols,
    n_splits=5,
    random_state=42,
    m_smoothing=50.0,
):
    tr = train_df.copy()
    te = test_df.copy()

    y = tr["target"].astype(float).values
    global_mean = float(np.mean(y))
    groups = tr["patient_id"].astype(str).values

    for col in cat_cols:
        te_col = te[col].astype(str).values

        oof = np.full(len(tr), global_mean, dtype=np.float64)

        for tr_idx, va_idx in group_kfold_indices(
            groups, n_splits=n_splits, random_state=random_state
        ):
            tr_fold = tr.iloc[tr_idx]
            agg = tr_fold.groupby(col)["target"].agg(["sum", "count"])
            sm = (
                (agg["sum"] + m_smoothing * global_mean) / (agg["count"] + m_smoothing)
            ).astype(float)
            map_sm = sm.to_dict()

            va_vals = tr.iloc[va_idx][col].astype(str).values
            oof[va_idx] = np.array(
                [map_sm.get(v, global_mean) for v in va_vals], dtype=np.float64
            )

        agg_full = tr.groupby(col)["target"].agg(["sum", "count"])
        sm_full = (
            (agg_full["sum"] + m_smoothing * global_mean)
            / (agg_full["count"] + m_smoothing)
        ).astype(float)
        map_full = sm_full.to_dict()
        te_enc = np.array(
            [map_full.get(v, global_mean) for v in te_col], dtype=np.float64
        )

        tr[f"te_{col}_oof"] = oof
        te[f"te_{col}_full"] = te_enc

    return tr, te


def build_design_matrices(train_df, test_df):
    tr = train_df.copy()
    te = test_df.copy()

    age_median = pd.to_numeric(tr["age_approx"], errors="coerce").median()
    tr_age = (
        pd.to_numeric(tr["age_approx"], errors="coerce")
        .fillna(age_median)
        .astype(np.float64)
    )
    te_age = (
        pd.to_numeric(te["age_approx"], errors="coerce")
        .fillna(age_median)
        .astype(np.float64)
    )

    def mode_or_unknown(s):
        s2 = s.fillna("")
        if (s2 != "").any():
            return s2[s2 != ""].mode().iloc[0]
        return "unknown"

    sex_mode = mode_or_unknown(tr["sex"])
    site_mode = mode_or_unknown(tr["anatom_site_general_challenge"])

    tr_sex = tr["sex"].fillna(sex_mode).replace("", sex_mode).astype(str)
    te_sex = te["sex"].fillna(sex_mode).replace("", sex_mode).astype(str)

    tr_site = (
        tr["anatom_site_general_challenge"]
        .fillna(site_mode)
        .replace("", site_mode)
        .astype(str)
    )
    te_site = (
        te["anatom_site_general_challenge"]
        .fillna(site_mode)
        .replace("", site_mode)
        .astype(str)
    )

    tr_sex_site = (tr_sex.astype("string") + "||" + tr_site.astype("string")).astype(
        str
    )
    te_sex_site = (te_sex.astype("string") + "||" + te_site.astype("string")).astype(
        str
    )

    tr_site_age = (tr_site.astype("string") + "||AGE").astype(str)
    te_site_age = (te_site.astype("string") + "||AGE").astype(str)

    sex_cats = sorted(pd.Index(tr_sex).append(pd.Index(te_sex)).unique().tolist())
    site_cats = sorted(pd.Index(tr_site).append(pd.Index(te_site)).unique().tolist())
    sex_site_cats = sorted(
        pd.Index(tr_sex_site).append(pd.Index(te_sex_site)).unique().tolist()
    )
    site_age_cats = sorted(
        pd.Index(tr_site_age).append(pd.Index(te_site_age)).unique().tolist()
    )

    sex_map = {c: i for i, c in enumerate(sex_cats)}
    site_map = {c: i for i, c in enumerate(site_cats)}
    sex_site_map = {c: i for i, c in enumerate(sex_site_cats)}
    site_age_map = {c: i for i, c in enumerate(site_age_cats)}

    n_tr = len(tr)
    n_te = len(te)
    n_sex = len(sex_cats)
    n_site = len(site_cats)
    n_sex_site = len(sex_site_cats)
    n_site_age = len(site_age_cats)

    Xtr_sex = np.zeros((n_tr, n_sex), dtype=np.float64)
    Xte_sex = np.zeros((n_te, n_sex), dtype=np.float64)
    for idx, v in enumerate(tr_sex.values):
        j = sex_map.get(v, None)
        if j is not None:
            Xtr_sex[idx, j] = 1.0
    for idx, v in enumerate(te_sex.values):
        j = sex_map.get(v, None)
        if j is not None:
            Xte_sex[idx, j] = 1.0

    Xtr_site = np.zeros((n_tr, n_site), dtype=np.float64)
    Xte_site = np.zeros((n_te, n_site), dtype=np.float64)
    for idx, v in enumerate(tr_site.values):
        j = site_map.get(v, None)
        if j is not None:
            Xtr_site[idx, j] = 1.0
    for idx, v in enumerate(te_site.values):
        j = site_map.get(v, None)
        if j is not None:
            Xte_site[idx, j] = 1.0

    Xtr_sex_site = np.zeros((n_tr, n_sex_site), dtype=np.float64)
    Xte_sex_site = np.zeros((n_te, n_sex_site), dtype=np.float64)
    for idx, v in enumerate(tr_sex_site.values):
        j = sex_site_map.get(v, None)
        if j is not None:
            Xtr_sex_site[idx, j] = 1.0
    for idx, v in enumerate(te_sex_site.values):
        j = sex_site_map.get(v, None)
        if j is not None:
            Xte_sex_site[idx, j] = 1.0

    age_mean = float(tr_age.mean())
    age_std = float(tr_age.std(ddof=0))
    if age_std == 0.0:
        age_std = 1.0
    tr_age_z = ((tr_age.values - age_mean) / age_std).reshape(-1, 1)
    te_age_z = ((te_age.values - age_mean) / age_std).reshape(-1, 1)

    Xtr_site_age = np.zeros((n_tr, n_site_age), dtype=np.float64)
    Xte_site_age = np.zeros((n_te, n_site_age), dtype=np.float64)
    for idx, v in enumerate(tr_site_age.values):
        j = site_age_map.get(v, None)
        if j is not None:
            Xtr_site_age[idx, j] = float(tr_age_z[idx, 0])
    for idx, v in enumerate(te_site_age.values):
        j = site_age_map.get(v, None)
        if j is not None:
            Xte_site_age[idx, j] = float(te_age_z[idx, 0])

    tr_log_age = np.log1p(tr_age.values)
    te_log_age = np.log1p(te_age.values)
    la_mean = float(np.mean(tr_log_age))
    la_std = float(np.std(tr_log_age, ddof=0))
    if la_std == 0.0:
        la_std = 1.0
    tr_log_age_z = ((tr_log_age - la_mean) / la_std).reshape(-1, 1)
    te_log_age_z = ((te_log_age - la_mean) / la_std).reshape(-1, 1)

    if "patient_prior_oof" in tr.columns and "patient_prior_full" in te.columns:
        pr_tr_raw = tr["patient_prior_oof"].astype(np.float64).values
        pr_te_raw = te["patient_prior_full"].astype(np.float64).values

        pr_tr = logit(pr_tr_raw)
        pr_te = logit(pr_te_raw)

        pr_mean = float(np.mean(pr_tr))
        pr_std = float(np.std(pr_tr, ddof=0))
        if pr_std == 0.0:
            pr_std = 1.0
        pr_tr = ((pr_tr - pr_mean) / pr_std).reshape(-1, 1)
        pr_te = ((pr_te - pr_mean) / pr_std).reshape(-1, 1)
    else:
        pr_tr = np.zeros((n_tr, 1), dtype=np.float64)
        pr_te = np.zeros((n_te, 1), dtype=np.float64)

    if (
        "patient_lesion_count_oof" in tr.columns
        and "patient_lesion_count_full" in te.columns
    ):
        cnt_tr = np.log1p(tr["patient_lesion_count_oof"].astype(np.float64).values)
        cnt_te = np.log1p(te["patient_lesion_count_full"].astype(np.float64).values)
        cnt_mean = float(np.mean(cnt_tr))
        cnt_std = float(np.std(cnt_tr, ddof=0))
        if cnt_std == 0.0:
            cnt_std = 1.0
        cnt_tr_z = ((cnt_tr - cnt_mean) / cnt_std).reshape(-1, 1)
        cnt_te_z = ((cnt_te - cnt_mean) / cnt_std).reshape(-1, 1)
    else:
        cnt_tr_z = np.zeros((n_tr, 1), dtype=np.float64)
        cnt_te_z = np.zeros((n_te, 1), dtype=np.float64)

    extra_feats_tr = []
    extra_feats_te = []
    for col in ["anatom_site_general_challenge", "sex"]:
        tr_key = f"te_{col}_oof"
        te_key = f"te_{col}_full"
        if tr_key in tr.columns and te_key in te.columns:
            tr_raw = tr[tr_key].astype(np.float64).values
            te_raw = te[te_key].astype(np.float64).values
            tr_logit = logit(tr_raw)
            te_logit = logit(te_raw)
            mu = float(np.mean(tr_logit))
            sd = float(np.std(tr_logit, ddof=0))
            if sd == 0.0:
                sd = 1.0
            extra_feats_tr.append(((tr_logit - mu) / sd).reshape(-1, 1))
            extra_feats_te.append(((te_logit - mu) / sd).reshape(-1, 1))

    if len(extra_feats_tr) == 0:
        extra_tr = np.zeros((n_tr, 0), dtype=np.float64)
        extra_te = np.zeros((n_te, 0), dtype=np.float64)
    else:
        extra_tr = np.concatenate(extra_feats_tr, axis=1)
        extra_te = np.concatenate(extra_feats_te, axis=1)

    X_tr = np.concatenate(
        [
            np.ones((n_tr, 1)),
            tr_age_z,
            tr_log_age_z,
            pr_tr,
            cnt_tr_z,
            extra_tr,
            Xtr_sex,
            Xtr_site,
            Xtr_sex_site,
            Xtr_site_age,
        ],
        axis=1,
    )
    X_te = np.concatenate(
        [
            np.ones((n_te, 1)),
            te_age_z,
            te_log_age_z,
            pr_te,
            cnt_te_z,
            extra_te,
            Xte_sex,
            Xte_site,
            Xte_sex_site,
            Xte_site_age,
        ],
        axis=1,
    )
    return X_tr, X_te


def _make_sample_weights(y, balanced_mode="sqrt"):
    y = np.asarray(y, dtype=np.float64)
    n = float(len(y))
    n_pos = max(1.0, float(y.sum()))
    n_neg = max(1.0, n - n_pos)
    if balanced_mode == "full":
        w_pos = n / (2.0 * n_pos)
        w_neg = n / (2.0 * n_neg)
    elif balanced_mode == "sqrt":
        w_pos_full = n / (2.0 * n_pos)
        w_neg_full = n / (2.0 * n_neg)
        w_pos = np.sqrt(w_pos_full)
        w_neg = np.sqrt(w_neg_full)
    else:
        w_pos = 1.0
        w_neg = 1.0
    return np.where(y > 0.5, w_pos, w_neg).astype(np.float64)


def fit_logreg_l2(
    X,
    y,
    C=1.0,
    class_weight_balanced=True,
    max_iter=50,
    tol=1e-6,
    reg_intercept=False,
):
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n, d = X.shape

    if class_weight_balanced:
        sample_w = _make_sample_weights(y, balanced_mode="sqrt")
    else:
        sample_w = np.ones(n, dtype=np.float64)

    lam = 1.0 / float(C)

    w = np.zeros(d, dtype=np.float64)
    I = np.eye(d, dtype=np.float64)
    if not reg_intercept:
        I[0, 0] = 0.0

    jitter = 1e-10

    for _ in range(int(max_iter)):
        z = X @ w
        p = sigmoid(z)

        r = p * (1.0 - p)
        wr = sample_w * r

        Xw = X * wr.reshape(-1, 1)
        H = (X.T @ Xw) / n
        H += lam * I
        H.flat[:: d + 1] += jitter

        g = (X.T @ ((p - y) * sample_w)) / n
        g += lam * (w * np.diag(I))

        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(H) @ g

        w_new = w - step

        if np.max(np.abs(w_new - w)) < tol:
            w = w_new
            break
        w = w_new

    return w


def predict_proba(X, w):
    return sigmoid(np.asarray(X, dtype=np.float64) @ np.asarray(w, dtype=np.float64))


def _add_patient_priors_fit_apply(
    df_fit, df_apply, global_mean, m_smoothing=20.0, label_col="target"
):
    agg = df_fit.groupby("patient_id")[label_col].agg(["sum", "count"])
    sm = (
        (agg["sum"] + m_smoothing * global_mean) / (agg["count"] + m_smoothing)
    ).astype(float)
    map_sm = sm.to_dict()
    map_cnt = agg["count"].astype(float).to_dict()

    apply_pid = df_apply["patient_id"].astype(str).values
    prior = np.array([map_sm.get(p, global_mean) for p in apply_pid], dtype=np.float64)
    cnt = np.array([map_cnt.get(p, 0.0) for p in apply_pid], dtype=np.float64)
    return prior, cnt


def _add_cat_te_fit_apply(
    df_fit, df_apply, col, global_mean, m_smoothing=50.0, label_col="target"
):
    agg = df_fit.groupby(col)[label_col].agg(["sum", "count"])
    sm = (
        (agg["sum"] + m_smoothing * global_mean) / (agg["count"] + m_smoothing)
    ).astype(float)
    map_sm = sm.to_dict()
    vals = df_apply[col].astype(str).values
    enc = np.array([map_sm.get(v, global_mean) for v in vals], dtype=np.float64)
    return enc


def _build_fold_features(
    df_train_all,
    df_test_all,
    tr_idx,
    va_idx,
    m_smoothing,
    te_m,
):
    df_tr = df_train_all.iloc[tr_idx].reset_index(drop=True)
    df_va = df_train_all.iloc[va_idx].reset_index(drop=True)
    df_te = df_test_all.copy()

    global_mean = float(df_tr["target"].astype(float).mean())

    pr_va, cnt_va = _add_patient_priors_fit_apply(
        df_tr,
        df_va,
        global_mean=global_mean,
        m_smoothing=float(m_smoothing),
        label_col="target",
    )
    pr_te, cnt_te = _add_patient_priors_fit_apply(
        df_tr,
        df_te,
        global_mean=global_mean,
        m_smoothing=float(m_smoothing),
        label_col="target",
    )

    df_tr2 = df_tr.copy()
    df_va2 = df_va.copy()
    df_te2 = df_te.copy()

    df_tr2["patient_prior_oof"] = (
        df_tr2["target"].astype(float).mean()
    )  # placeholder, overwritten by OOF later in full fit
    df_tr2["patient_lesion_count_oof"] = 0.0

    df_va2["patient_prior_oof"] = pr_va
    df_va2["patient_lesion_count_oof"] = cnt_va

    df_te2["patient_prior_full"] = pr_te
    df_te2["patient_lesion_count_full"] = cnt_te

    for col in ["anatom_site_general_challenge", "sex"]:
        df_va2[f"te_{col}_oof"] = _add_cat_te_fit_apply(
            df_tr2,
            df_va2,
            col=col,
            global_mean=global_mean,
            m_smoothing=float(te_m),
            label_col="target",
        )
        df_te2[f"te_{col}_full"] = _add_cat_te_fit_apply(
            df_tr2,
            df_te2,
            col=col,
            global_mean=global_mean,
            m_smoothing=float(te_m),
            label_col="target",
        )

    X_tr, _ = build_design_matrices(df_tr2, df_te2)
    X_va, _ = build_design_matrices(df_va2, df_te2)
    y_tr = df_tr2["target"].astype(int).values
    y_va = df_va2["target"].astype(int).values
    return X_tr, y_tr, X_va, y_va


y0 = train_df["target"].astype(int).values
groups0 = train_df["patient_id"].astype(str).values
train_idx0, _ = group_shuffle_split(groups0, test_size=0.2, random_state=RANDOM_STATE)
train_df_split = train_df.iloc[train_idx0].reset_index(drop=True)
g_split = train_df_split["patient_id"].astype(str).values

m_candidates = [5.0, 10.0, 20.0, 40.0]
te_m_candidates = [10.0, 30.0, 60.0, 120.0]
Cs = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]

best_m = None
best_te_m = None
best_C = None
best_auc = -np.inf

for m_smoothing in m_candidates:
    for te_m in te_m_candidates:
        for C in Cs:
            fold_aucs = []
            for tr_f, va_f in group_kfold_indices(
                g_split, n_splits=5, random_state=RANDOM_STATE
            ):
                X_tr_f, y_tr_f, X_va_f, y_va_f = _build_fold_features(
                    train_df_split,
                    test_df,
                    tr_f,
                    va_f,
                    m_smoothing=m_smoothing,
                    te_m=te_m,
                )
                w = fit_logreg_l2(
                    X_tr_f,
                    y_tr_f,
                    C=C,
                    class_weight_balanced=True,
                    max_iter=60,
                    tol=1e-6,
                    reg_intercept=False,
                )
                p = predict_proba(X_va_f, w)
                auc = roc_auc_score_np(y_va_f, p)
                if not np.isnan(auc):
                    fold_aucs.append(auc)
            mean_auc = float(np.mean(fold_aucs)) if len(fold_aucs) else -np.inf
            if mean_auc > best_auc:
                best_auc = mean_auc
                best_C = C
                best_m = m_smoothing
                best_te_m = te_m


train_df2, test_df2 = add_patient_priors(
    train_df, test_df, n_splits=5, random_state=RANDOM_STATE, m_smoothing=float(best_m)
)
train_df2, test_df2 = add_categorical_target_encodings(
    train_df2,
    test_df2,
    cat_cols=["anatom_site_general_challenge", "sex"],
    n_splits=5,
    random_state=RANDOM_STATE,
    m_smoothing=float(best_te_m),
)

y = train_df2["target"].astype(int).values
groups = train_df2["patient_id"].astype(str).values
X_all, X_test = build_design_matrices(train_df2, test_df2)

train_idx, val_idx = group_shuffle_split(
    groups, test_size=0.2, random_state=RANDOM_STATE
)
X_tr, y_tr = X_all[train_idx], y[train_idx]
X_va, y_va = X_all[val_idx], y[val_idx]

w_split = fit_logreg_l2(
    X_tr,
    y_tr,
    C=float(best_C),
    class_weight_balanced=True,
    max_iter=80,
    tol=1e-6,
    reg_intercept=False,
)
val_pred = predict_proba(X_va, w_split)
val_auc = roc_auc_score_np(y_va, val_pred)

w_final = fit_logreg_l2(
    X_all,
    y,
    C=float(best_C),
    class_weight_balanced=True,
    max_iter=100,
    tol=1e-6,
    reg_intercept=False,
)
test_pred = predict_proba(X_test, w_final).astype(np.float32)

eps = 1e-6
test_pred = np.clip(test_pred, eps, 1 - eps)

submission = pd.DataFrame(
    {"image_name": test_df2["image_name"].astype(str).values, "target": test_pred}
)

if len(submission) != len(sample_sub):
    raise ValueError(
        f"Row count mismatch: submission has {len(submission)} rows, sample has {len(sample_sub)} rows"
    )
if not submission["image_name"].equals(sample_sub["image_name"].astype(str)):
    submission = sample_sub[["image_name"]].merge(
        submission, on="image_name", how="left"
    )
    if submission["target"].isna().any():
        raise ValueError(
            "Submission alignment failed: some test image_name values missing predictions."
        )

submission.to_csv("submission.csv", index=False, float_format="%.6f")

meta_path = os.path.join(sub_path, "submission_meta.csv")
if concat_sub is not None and os.path.exists(meta_path) and ncol is not None:
    meta = pd.read_csv(meta_path)
    if "image_name" in meta.columns and "target" in meta.columns:
        meta = meta[["image_name", "target"]].copy()
        ens = concat_sub[["image_name"]].copy()
        ens["target_ens"] = concat_sub.iloc[:, 1:ncol].mean(axis=1).astype(float).values
        blended = submission.merge(ens, on="image_name", how="left")
        blended = blended.merge(
            meta.rename(columns={"target": "target_meta"}), on="image_name", how="left"
        )
        if (
            not blended["target_ens"].isna().any()
            and not blended["target_meta"].isna().any()
        ):
            blended["target"] = (
                0.5 * blended["target"].astype(float)
                + 0.25 * blended["target_ens"]
                + 0.25 * blended["target_meta"]
            )
            blended["target"] = np.clip(blended["target"].values, eps, 1 - eps)
            blended[["image_name", "target"]].to_csv(
                "submission_meta.csv", index=False, float_format="%.6f"
            )

print(
    f"Selected via fold-safe group-CV on train split: m_smoothing={best_m}, te_m={best_te_m}, C={best_C} (mean CV AUC={best_auc:.6f})"
)
print(f"Validation AUC (patient-wise holdout): {val_auc:.6f}")
print("Wrote: submission.csv")
if os.path.exists("submission_meta.csv"):
    print("Wrote: submission_meta.csv")

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/307589430.py in <cell line: 0>()
    690                     reg_intercept=False,
    691                 )
--> 692                 p = predict_proba(X_va_f, w)
    693                 auc = roc_auc_score_np(y_va_f, p)
    694                 if not np.isnan(auc):

/tmp/ipykernel_11/307589430.py in predict_proba(X, w)
    543 
    544 def predict_proba(X, w):
--> 545     return sigmoid(np.asarray(X, dtype=np.float64) @ np.asarray(w, dtype=np.float64))
    546 
    547 

ValueError: matmul: Input operand 1 has a mismatch in its core dimension 0, with gufunc signature (n?,k),(k,m?)->(n?,m?) (size 31 is different from 33)
