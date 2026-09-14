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

0.9402766097647106

# 6. Current score

0.67823

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'I remove the dependency on the missing `../input/ensemble-melanoma` directory (which causes the crash) and instead build a simple, valid baseline submission from the provided competition files. Since no model packages are available here, the safest score-improving (vs. random) minimal approach is to use the training-set malignancy rate as a constant probability for all test images, which is a standard AUC baseline. I also ensure the submission has exactly the required columns (`image_name`, `target`), matches the test row order, and is written to a `.csv` file. The rest of the original “ensemble” cells be kept but guarded so they don’t error when the external folder is absent.'
- What this solution (achieved 0.66789) has done: 'Your current 0.5 AUC is consistent with producing an almost-constant prediction; to move toward the 0.94 target without changing the overall “simple tabular baseline” approach, I replace the constant-probability submission with a minimal logistic-regression model trained on the provided metadata (sex, age, anatomic site) and evaluated with ROC-AUC. This keeps the core pipeline (read CSVs → train on train.csv metadata → predict probabilities for test.csv → write submission.csv) while adding only lightweight ML using scikit-learn. I also add a patient-wise train/validation split to reduce leakage and to sanity-check that we’re learning signal before writing the final submission. The ensemble-folder logic is left intact and guarded, but the main submission now be the metadata-model output (which should improve meaningfully over 0.5).'
- What this solution (achieved 0.34234) has done: 'Your current 0.66789 AUC is far below the 0.9403 target, so we should improve the *same metadata-only pipeline* with minimal, legitimate modeling tweaks rather than changing approach. I keep the same inputs/features and LogisticRegression, but (1) switch to a more appropriate solver for sparse one-hot features, (2) add mild class balancing (this competition is highly imbalanced), and (3) use a small group-aware cross-validated search over `C` to pick a better regularization strength without changing the model family. I also ensure the output submission keeps the exact `test.csv` order and schema, and still writes a valid `submission.csv`.'
- What this solution (achieved 0.67823) has done: 'Your current score (0.34234) is far below the target (0.9403), and the biggest likely cause is that this environment doesn’t have scikit-learn installed (only sklearn-pandas is listed), so the intended logistic-regression pipeline either didn’t actually run in Kaggle or produced a degenerate submission. I keep the exact same core “metadata-only tabular model” approach, but implement the logistic regression training/prediction in pure NumPy/Pandas (no new dependencies) with the same features, L2 regularization, class balancing, and group-aware CV selection of C. I also preserve the guarded ensemble-folder logic and ensure the final `submission.csv` matches `test.csv` order and required columns. This should legitimately improve AUC versus a constant baseline and move you materially closer to the target without changing the modeling intent.'

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

    sex_cats = sorted(tr_sex.unique().tolist())
    site_cats = sorted(tr_site.unique().tolist())

    sex_map = {c: i for i, c in enumerate(sex_cats)}
    site_map = {c: i for i, c in enumerate(site_cats)}

    n_tr = len(tr)
    n_te = len(te)
    n_sex = len(sex_cats)
    n_site = len(site_cats)

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

    age_mean = float(tr_age.mean())
    age_std = float(tr_age.std(ddof=0))
    if age_std == 0.0:
        age_std = 1.0
    tr_age_z = ((tr_age.values - age_mean) / age_std).reshape(-1, 1)
    te_age_z = ((te_age.values - age_mean) / age_std).reshape(-1, 1)

    X_tr = np.concatenate([np.ones((n_tr, 1)), tr_age_z, Xtr_sex, Xtr_site], axis=1)
    X_te = np.concatenate([np.ones((n_te, 1)), te_age_z, Xte_sex, Xte_site], axis=1)
    return X_tr, X_te


def fit_logreg_l2(X, y, C=1.0, class_weight_balanced=True, max_iter=400, lr=0.1):
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    n, d = X.shape
    w = np.zeros(d, dtype=np.float64)

    if class_weight_balanced:
        n_pos = max(1.0, float(y.sum()))
        n_neg = max(1.0, float(n - y.sum()))
        w_pos = n / (2.0 * n_pos)
        w_neg = n / (2.0 * n_neg)
        sample_w = np.where(y > 0.5, w_pos, w_neg).astype(np.float64)
    else:
        sample_w = np.ones(n, dtype=np.float64)

    lam = 1.0 / float(C)

    for _ in range(int(max_iter)):
        z = X @ w
        p = sigmoid(z)
        err = (p - y) * sample_w
        grad = (X.T @ err) / n
        grad[1:] += lam * w[1:]
        w -= lr * grad

    return w


def predict_proba(X, w):
    return sigmoid(np.asarray(X, dtype=np.float64) @ np.asarray(w, dtype=np.float64))


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


y = train_df["target"].astype(int).values
groups = train_df["patient_id"].astype(str).values

X_all, X_test = build_design_matrices(train_df, test_df)

train_idx, val_idx = group_shuffle_split(
    groups, test_size=0.2, random_state=RANDOM_STATE
)
X_tr, y_tr, g_tr = X_all[train_idx], y[train_idx], groups[train_idx]
X_va, y_va = X_all[val_idx], y[val_idx]

Cs = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
best_C = None
best_auc = -np.inf

for C in Cs:
    fold_aucs = []
    for tr_f, va_f in group_kfold_indices(g_tr, n_splits=5, random_state=RANDOM_STATE):
        w = fit_logreg_l2(
            X_tr[tr_f],
            y_tr[tr_f],
            C=C,
            class_weight_balanced=True,
            max_iter=400,
            lr=0.1,
        )
        p = predict_proba(X_tr[va_f], w)
        auc = roc_auc_score_np(y_tr[va_f], p)
        if not np.isnan(auc):
            fold_aucs.append(auc)
    mean_auc = float(np.mean(fold_aucs)) if len(fold_aucs) else -np.inf
    if mean_auc > best_auc:
        best_auc = mean_auc
        best_C = C

w_split = fit_logreg_l2(
    X_tr, y_tr, C=best_C, class_weight_balanced=True, max_iter=600, lr=0.1
)
val_pred = predict_proba(X_va, w_split)
val_auc = roc_auc_score_np(y_va, val_pred)

w_final = fit_logreg_l2(
    X_all, y, C=best_C, class_weight_balanced=True, max_iter=800, lr=0.1
)
test_pred = predict_proba(X_test, w_final).astype(np.float32)

eps = 1e-6
test_pred = np.clip(test_pred, eps, 1 - eps)

submission = pd.DataFrame(
    {"image_name": test_df["image_name"].astype(str).values, "target": test_pred}
)

if len(submission) != len(sample_sub):
    raise ValueError(
        f"Row count mismatch: submission has {len(submission)} rows, sample has {len(sample_sub)} rows"
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

print(f"Selected C via group-CV on train split: {best_C} (mean CV AUC={best_auc:.6f})")
print(f"Validation AUC (patient-wise holdout): {val_auc:.6f}")
print("Wrote: submission.csv")
if os.path.exists("submission_meta.csv"):
    print("Wrote: submission_meta.csv")
