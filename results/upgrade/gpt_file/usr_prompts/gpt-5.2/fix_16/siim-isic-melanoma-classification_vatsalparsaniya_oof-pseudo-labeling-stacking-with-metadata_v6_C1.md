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

0.9402423884987422

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.66722) has done: 'I make the notebook run end-to-end in this Kaggle environment by removing dependencies on missing external datasets (`stacking-data`, `95-cv-oof-submission`) and by adding a lightweight replacement `tfrecord` grouping derived deterministically from `image_name` so the existing fold logic stays intact. I also fix scikit-learn API breakages (`min_impurity_split`, deprecated `criterion='mse'`, `max_features='auto'`, `presort`) so the models instantiate correctly. The `crossValidate` function be minimally corrected to use the right train/val targets, to avoid the pseudo-label filter bug, and to fit a final model on all training data before generating test predictions (previously it used the last fold model). Finally, the script always write a valid `submission.csv` with columns `image_name,target` to `/kaggle/working/`.'
- What this solution (achieved 0.65794) has done: 'I make the notebook reliably produce a submission (your current run likely times out before reaching the write step) by running only a single model end-to-end instead of training 9+ models plus pseudo-labeling. To move the score upward toward your target AUC, I select the strongest model already defined (the stacking classifier) and turn off pseudo-labeling (your current pseudo-labeling uses validation predictions as labels, which typically hurts generalization). I also fix a subtle but important bug in `crossValidate`: `CLF_pesudo = CLF` aliases the same estimator, so it gets overwritten each fold; using `clone()` keeps fold models independent and makes results stable. These changes keep your core feature engineering and model definitions intact while making execution feasible under the 600s constraint and improving expected leaderboard AUC.'
- What this solution (achieved 0.65764) has done: 'Your current gap to the target is large (0.65794 vs 0.94024; ~30%+ relative), so we should make a small but high-impact fix that preserves your core approach. The biggest issue is that your one-hot encoding is done separately for train and test, which creates inconsistent feature scaling and category baselines; this can severely hurt generalization and AUC. I minimally change the encoding to fit categories on the concatenation of train+test metadata (still no label leakage because we don’t use `target`), then split back, keeping the same downstream models and cross-validation logic. I also make the age normalization use train statistics for both train/test (again minimal, but improves calibration consistency).'
- What this solution (achieved 0.65764) has done: 'Your current AUC is far below the target, so the most direct way to move upward (without changing your core modeling approach) is to fix a likely leakage/bug in your fold split: you’re using `KFold.split(np.arange(15))`, which splits “group IDs” but not the rows, and it can create unstable/incorrect folds when mapped back to rows. I minimally switch this to a proper group-based split (`GroupKFold`) using your existing `tfrecord` column as the group label, keeping the same estimator, features, and training loop semantics. I also make the number of groups (`n_groups`) consistent with the number of TFRecord shards in the dataset (16, not 15) to reduce fold noise and improve generalization. These are small, targeted changes that typically improve leaderboard AUC for this dataset while preserving your overall pipeline and producing the same submission format.'
- What this solution (achieved 0.65764) has done: 'Your current AUC (0.65764) is far below the target (0.94024), so we should make a small change that can materially improve generalization without changing your feature set or model definitions. The biggest remaining issue is that `StackingClassifier` uses an internal CV that is *not group-aware*, so it leaks correlated samples across folds (same `tfrecord` group) when training the meta-model, which can badly hurt test-time AUC. I keep your stacking core logic, but swap the stacker’s internal `cv=3` for the same `GroupKFold` logic you already use externally, and pass the `tfrecord` groups into `.fit(...)` via `final_estimator__sample_weight` workaround is not needed here—`StackingClassifier.fit` supports a `groups` argument when `cv` is a splitter. I also ensure the final training call uses the same groups so the meta-model is trained consistently.'
- What this solution (achieved 0.66103) has done: 'To move the AUC up while keeping your pipeline and models intact, I make two minimal, high-impact correctness fixes: (1) ensure the final `submission.csv` preserves the exact row order of `sample_submission.csv` (Kaggle aligns by row; a merge can silently reorder), and (2) make the stacked model’s internal CV explicitly group-aware by passing the same `tfrecord` groups into `StackingClassifier.fit(...)` so the meta-model training matches your outer `GroupKFold` logic. These changes don’t alter your feature set or model definitions, but they prevent subtle misalignment/leakage issues that can dramatically hurt leaderboard AUC. The script still run end-to-end and always write `/kaggle/working/submission.csv` with `image_name,target`.'
- What this solution (achieved 0.66103) has done: 'Your current AUC is far below the target, so the smallest high-impact move is to fix the biggest remaining generalization bottleneck without changing your feature set or model family. The stacking model’s internal CV is now group-aware, but its **default `shuffle=False`**, which makes meta-model training sensitive to group ordering and can hurt AUC; I enable shuffling with your existing `SEED` for stability and typically better generalization. I also make a minimal, metric-aligned improvement by ensuring the stacker uses probability outputs consistently via `stack_method="predict_proba"` and keeping the submission strictly aligned to `sample_submission.csv` order (already mostly done, we just make it deterministic). No architecture/loss/feature changes are introduced, and the script still writes `/kaggle/working/submission.csv`.'
- What this solution (achieved 0.73126) has done: 'Your current AUC (0.661) is far below the target (0.940), so we need a small but high-impact fix that improves generalization without changing your model family or training loop. The biggest remaining issue is that your metadata features include only sex/anatom/age; adding a few already-available, non-image, non-leaky numeric features from `train.csv/test.csv` (patient-wise count encoding + anatomical site malignancy prior computed on train only) materially helps while keeping the same stacking model and CV logic. I also ensure those new features are computed in a fold-safe way for CV (OOF target-encoding) to avoid leakage that could hurt LB AUC. All paths and the final `submission.csv` format/order remain unchanged.'

# 9. Code solution

## === cell 0
import os, re, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 200)


def seed_everything(SEED: int):
    np.random.seed(SEED)
    os.environ["PYTHONHASHSEED"] = str(SEED)




## === cell 1
import matplotlib.pyplot as plt

try:
    from colorama import Fore, Back, Style
except Exception:

    class _Dummy:
        def __getattr__(self, name):
            return ""

    Fore = Back = Style = _Dummy()



## === cell 2
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")

try:
    import lightgbm as lgb  # CLF1

    HAS_LGB = True
except Exception:
    lgb = None
    HAS_LGB = False

from sklearn.linear_model import LogisticRegression  # CLF2

try:
    from xgboost import XGBClassifier  # optional

    HAS_XGB = True
except Exception:
    XGBClassifier = None
    HAS_XGB = False

from sklearn.naive_bayes import GaussianNB  # CLF4
from sklearn.ensemble import RandomForestClassifier  # CLF5
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier  # fallback booster

try:
    from mlxtend.classifier import StackingClassifier as MlxtendStackingClassifier
except Exception:
    MlxtendStackingClassifier = None

from sklearn.ensemble import StackingClassifier as SklearnStackingClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from sklearn.base import clone



## === cell 3
FOLDS = 3
SEED = 123
Setup_Parameters = False
seed_everything(SEED)

file_add_list = [1, 2, 3, 4, 5]

pesudo_label = False

test_pipeline = False

N_TFRECORD_GROUPS = 16



## === cell 4
CANDIDATE_BASE_PATHS = [
    "../input/siim-isic-melanoma-classification",
    "/kaggle/input/siim-isic-melanoma-classification",
    "../kaggle/input/siim-isic-melanoma-classification",
    "../input",
    "/kaggle/input",
]
BASE_PATH = None
for p in CANDIDATE_BASE_PATHS:
    if os.path.exists(os.path.join(p, "train.csv")) and os.path.exists(
        os.path.join(p, "test.csv")
    ):
        BASE_PATH = p
        break
if BASE_PATH is None:
    raise FileNotFoundError(
        "Could not locate train.csv/test.csv under expected Kaggle input paths."
    )

print("Using BASE_PATH:", BASE_PATH)

train_metadata = pd.read_csv(os.path.join(BASE_PATH, "train.csv"))
test_metadata = pd.read_csv(os.path.join(BASE_PATH, "test.csv"))
sample_submission = pd.read_csv(os.path.join(BASE_PATH, "sample_submission.csv"))



## === cell 5
print("Train data shape : ", train_metadata.shape)
print("Test data shape  : ", test_metadata.shape)
print("Sample submission shape:", sample_submission.shape)




## === cell 6
def make_tfrecord_group(image_names: pd.Series, n_groups: int = 16) -> pd.Series:
    import hashlib

    def _grp(s):
        h = hashlib.md5(s.encode("utf-8")).hexdigest()
        return int(h[:8], 16) % n_groups

    return image_names.astype(str).map(_grp).astype(int)




## === cell 7
print("Unique values in train columns with frequency :")
print("\nsex :", dict(train_metadata.sex.value_counts(dropna=False).head(10)))
print(
    "\nage_approx :",
    dict(train_metadata.age_approx.value_counts(dropna=False).head(10)),
)
print(
    "\nanatom_site_general_challenge :",
    dict(
        train_metadata.anatom_site_general_challenge.value_counts(dropna=False).head(10)
    ),
)
print(
    "\ndiagnosis :", dict(train_metadata.diagnosis.value_counts(dropna=False).head(10))
)
print(
    "\nbenign_malignant :",
    dict(train_metadata.benign_malignant.value_counts(dropna=False).head(10)),
)
print("\ntarget :", dict(train_metadata.target.value_counts(dropna=False)))



## === cell 8
print("Unique values in test columns with frequency :")
print("\nsex :", dict(test_metadata.sex.value_counts(dropna=False).head(10)))
print(
    "\nage_approx :", dict(test_metadata.age_approx.value_counts(dropna=False).head(10))
)
print(
    "\nanatom_site_general_challenge :",
    dict(
        test_metadata.anatom_site_general_challenge.value_counts(dropna=False).head(10)
    ),
)




## === cell 9
def _patient_count_feature(train_df: pd.DataFrame, test_df: pd.DataFrame):
    combined = pd.concat(
        [
            train_df[["patient_id"]].assign(_is_train=1),
            test_df[["patient_id"]].assign(_is_train=0),
        ],
        ignore_index=True,
    )
    cnt = combined["patient_id"].value_counts(dropna=False)
    train_cnt = train_df["patient_id"].map(cnt).fillna(1).astype(float)
    test_cnt = test_df["patient_id"].map(cnt).fillna(1).astype(float)
    return train_cnt.rename("patient_count"), test_cnt.rename("patient_count")


def _oof_target_mean_by_category(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    col: str,
    target_col: str = "target",
    n_splits: int = 3,
    seed: int = 123,
    smoothing: float = 20.0,
    groups: pd.Series = None,
):
    tr = train_df[[col, target_col]].copy()
    te = test_df[[col]].copy()
    tr[col] = tr[col].fillna("NA").astype(str)
    te[col] = te[col].fillna("NA").astype(str)

    global_mean = tr[target_col].mean()
    oof = np.zeros(len(tr), dtype=np.float64)

    if groups is None:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_iter = splitter.split(tr)
    else:
        splitter = GroupKFold(n_splits=n_splits)
        split_iter = splitter.split(tr, y=tr[target_col].values, groups=groups.values)

    for tr_idx, va_idx in split_iter:
        fold_tr = tr.iloc[tr_idx]
        stats = fold_tr.groupby(col)[target_col].agg(["mean", "count"])
        smooth = (stats["count"] * stats["mean"] + smoothing * global_mean) / (
            stats["count"] + smoothing
        )
        oof[va_idx] = tr.iloc[va_idx][col].map(smooth).fillna(global_mean).values

    stats_full = tr.groupby(col)[target_col].agg(["mean", "count"])
    smooth_full = (
        stats_full["count"] * stats_full["mean"] + smoothing * global_mean
    ) / (stats_full["count"] + smoothing)
    test_enc = te[col].map(smooth_full).fillna(global_mean).values

    return (
        pd.Series(oof, index=train_df.index, name=f"{col}_target_mean"),
        pd.Series(test_enc, index=test_df.index, name=f"{col}_target_mean"),
    )


train = train_metadata.copy()
test = test_metadata.copy()

train["age_approx"] = train["age_approx"].fillna(train["age_approx"].mean())
test["age_approx"] = test["age_approx"].fillna(train["age_approx"].mean())

age_mean = train["age_approx"].mean()
age_std = train["age_approx"].std()
if not np.isfinite(age_std) or age_std == 0:
    age_std = 1.0

train_age_norm = ((train["age_approx"] - age_mean) / age_std).rename("age_approx_norm")
test_age_norm = ((test["age_approx"] - age_mean) / age_std).rename("age_approx_norm")

train_patient_cnt, test_patient_cnt = _patient_count_feature(train, test)

train_groups_for_te = make_tfrecord_group(
    train["image_name"], n_groups=N_TFRECORD_GROUPS
)

train_anat_te, test_anat_te = _oof_target_mean_by_category(
    train,
    test,
    col="anatom_site_general_challenge",
    n_splits=FOLDS,
    seed=SEED,
    smoothing=20.0,
    groups=train_groups_for_te,
)

combined = pd.concat(
    [
        train[["image_name", "sex", "anatom_site_general_challenge"]].assign(
            _is_train=1
        ),
        test[["image_name", "sex", "anatom_site_general_challenge"]].assign(
            _is_train=0
        ),
    ],
    axis=0,
    ignore_index=True,
)

sex_code_all = pd.get_dummies(combined["sex"], prefix="sex")
anatom_code_all = pd.get_dummies(
    combined["anatom_site_general_challenge"], prefix="anatom_site"
)

combined_coded = pd.concat(
    [combined[["image_name", "_is_train"]], sex_code_all, anatom_code_all], axis=1
)

combined_train = (
    combined_coded[combined_coded["_is_train"] == 1]
    .drop(columns=["_is_train"])
    .reset_index(drop=True)
)
combined_test = (
    combined_coded[combined_coded["_is_train"] == 0]
    .drop(columns=["_is_train"])
    .reset_index(drop=True)
)

train_coded = pd.concat(
    [
        combined_train[["image_name"]],
        combined_train.drop(columns=["image_name"]),
        train_age_norm.reset_index(drop=True),
        train_patient_cnt.reset_index(drop=True),
        train_anat_te.reset_index(drop=True),
        train["target"].reset_index(drop=True),
    ],
    axis=1,
)

train_coded["tfrecord"] = make_tfrecord_group(
    train_coded["image_name"], n_groups=N_TFRECORD_GROUPS
)

print("Shape : ", train_coded.shape)
train_coded.head()




## === cell 10
def add_OOF_pred(train_coded_in: pd.DataFrame, num: int):
    missing_paths = []
    for n in file_add_list:
        path = f"../input/95-cv-oof-submission/oof_{n}.csv"
        if not os.path.exists(path):
            missing_paths.append(path)
    if missing_paths:
        print("OOF files not found; skipping external OOF feature merge.")
        return train_coded_in
    train_coded_out = train_coded_in.copy()
    for n in file_add_list:
        df_ = pd.read_csv(f"../input/95-cv-oof-submission/oof_{n}.csv")
        train_coded_out = pd.merge(
            train_coded_out, df_[["image_name", "pred"]], on="image_name", how="left"
        )
        train_coded_out.rename({"pred": f"pred_{n}"}, axis=1, inplace=True)
    return train_coded_out


train_coded = add_OOF_pred(train_coded, 5)
train_coded.to_csv("train_coded.csv", index=False)
train_coded.tail()



## === cell 11
test_coded = pd.concat(
    [
        combined_test[["image_name"]],
        combined_test.drop(columns=["image_name"]),
        test_age_norm.reset_index(drop=True),
        test_patient_cnt.reset_index(drop=True),
        test_anat_te.reset_index(drop=True),
    ],
    axis=1,
)

train_feature_cols = [c for c in train_coded.columns if c not in ["target"]]
for c in train_feature_cols:
    if c not in test_coded.columns:
        test_coded[c] = 0
for c in test_coded.columns:
    if c not in train_coded.columns and c not in ["image_name"]:
        train_coded[c] = 0

if "tfrecord" not in test_coded.columns:
    test_coded["tfrecord"] = make_tfrecord_group(
        test_coded["image_name"], n_groups=N_TFRECORD_GROUPS
    )

ordered_test_cols = [c for c in train_coded.columns if c != "target"]
test_coded = test_coded[ordered_test_cols]

print("Test coded shape:", test_coded.shape)
test_coded.head()




## === cell 12
def add_submission_pred(test_coded_in: pd.DataFrame, num: int):
    missing_paths = []
    for n in file_add_list:
        path = f"../input/95-cv-oof-submission/submission_{n}.csv"
        if not os.path.exists(path):
            missing_paths.append(path)
    if missing_paths:
        print(
            "External submission prediction files not found; skipping external prediction feature merge."
        )
        return test_coded_in
    test_coded_out = test_coded_in.copy()
    for n in file_add_list:
        df_ = pd.read_csv(f"../input/95-cv-oof-submission/submission_{n}.csv")
        test_coded_out = pd.merge(
            test_coded_out, df_[["image_name", "target"]], on="image_name", how="left"
        )
        test_coded_out.rename({"target": f"pred_{n}"}, axis=1, inplace=True)
    return test_coded_out


test_coded = add_submission_pred(test_coded, 5)
test_coded.to_csv("test_coded.csv", index=False)
test_coded.tail()




## === cell 13
def crossValidate(
    CLF,
    X=train_coded,
    X_test=test_coded,
    FOLDS=5,
    SEED=123,
    show_roc_curve=False,
    pesudo_label=False,
):
    print(Fore.YELLOW)
    print("#" * 60)
    model_name = type(CLF).__name__
    print("#### ", model_name)
    print("#" * 60, Style.RESET_ALL)

    CV_Score = []
    Val_preds = []
    Val_imagenames = []
    val_targets = []

    CV_Score_pesudo = []
    Val_preds_pesudo = []
    Val_imagenames_pesudo = []
    val_targets_pesudo = []

    if "tfrecord" not in X.columns:
        raise ValueError(
            "Expected column 'tfrecord' in training data for fold splitting."
        )

    gkf = GroupKFold(n_splits=FOLDS)
    groups = X["tfrecord"].values

    def _fit_with_optional_groups(estimator, X_fit, y_fit, groups_fit):
        if isinstance(estimator, SklearnStackingClassifier):
            return estimator.fit(X_fit, y_fit, groups=groups_fit)
        return estimator.fit(X_fit, y_fit)

    for fold, (idxT, idxV) in enumerate(
        gkf.split(X, y=X["target"].values, groups=groups)
    ):
        X_train_main = X.iloc[idxT].reset_index(drop=True)
        X_val_main = X.iloc[idxV].reset_index(drop=True)

        print(Fore.MAGENTA)
        print("#" * 60, Style.RESET_ALL)
        print(Fore.BLUE)
        print("FOLD : ", fold)
        print("Train TFrecords : ", sorted(X_train_main.tfrecord.unique().tolist()))
        print("Validation TFrecords : ", sorted(X_val_main.tfrecord.unique().tolist()))
        image_names = list(X_val_main["image_name"])

        X_train = X_train_main.drop(["target", "tfrecord", "image_name"], axis=1)
        y_train = X_train_main["target"].astype(int)

        X_val = X_val_main.drop(["target", "tfrecord", "image_name"], axis=1)
        y_val = X_val_main["target"].astype(int)

        fold_clf = clone(CLF)
        _fit_with_optional_groups(
            fold_clf, X_train, y_train, groups_fit=X_train_main["tfrecord"].values
        )

        try:
            y_train_pred = fold_clf.predict_proba(X_train)[:, 1]
        except Exception:
            y_train_pred = fold_clf.predict(X_train)

        print("Train AUC : ", roc_auc_score(y_train, y_train_pred))

        try:
            Val_pred = fold_clf.predict_proba(X_val)[:, 1]
        except Exception:
            Val_pred = fold_clf.predict(X_val)

        Val_auc = roc_auc_score(y_val, Val_pred)
        print("Val AUC : ", Val_auc)

        CV_Score.append(Val_auc)
        Val_preds.append(Val_pred)
        Val_imagenames.append(image_names)
        val_targets.append(list(y_val))

        if pesudo_label:
            CLF_pesudo = clone(CLF)

            train2_Pesudo = X_train_main.copy()
            train2_Pesudo["target_label"] = y_train

            test2_Pesudo = X_val_main.copy()
            test2_Pesudo["target_label"] = Val_pred

            test2_Pesudo = test2_Pesudo[
                (test2_Pesudo["target_label"] >= 0.99)
                | (test2_Pesudo["target_label"] <= 0.01)
            ]

            test2_Pesudo.loc[test2_Pesudo["target_label"] >= 0.5, "target_label"] = 1
            test2_Pesudo.loc[test2_Pesudo["target_label"] < 0.5, "target_label"] = 0
            test2_Pesudo["target_label"] = test2_Pesudo["target_label"].astype(int)

            print(Fore.CYAN)
            print("Number of Pesudo Labeled Data added : ", len(test2_Pesudo))
            print(
                "target_label = 1 : ",
                len(test2_Pesudo[test2_Pesudo["target_label"] == 1]),
            )
            print(
                "target_label = 0 : ",
                len(test2_Pesudo[test2_Pesudo["target_label"] == 0]),
            )

            train_pesudo = pd.concat([train2_Pesudo, test2_Pesudo], axis=0).reset_index(
                drop=True
            )

            X_train_pesudo = train_pesudo.drop(
                ["target_label", "target", "tfrecord", "image_name"], axis=1
            )
            y_train_pesudo = train_pesudo["target_label"]

            _fit_with_optional_groups(
                CLF_pesudo,
                X_train_pesudo,
                y_train_pesudo,
                groups_fit=train_pesudo["tfrecord"].values,
            )

            try:
                y_train_pred_pesudo = CLF_pesudo.predict_proba(X_train_pesudo)[:, 1]
            except Exception:
                y_train_pred_pesudo = CLF_pesudo.predict(X_train_pesudo)

            print(
                "Pesudo Train AUC : ",
                roc_auc_score(y_train_pesudo, y_train_pred_pesudo),
            )

            try:
                Val_pred_pesudo = CLF_pesudo.predict_proba(X_val)[:, 1]
            except Exception:
                Val_pred_pesudo = CLF_pesudo.predict(X_val)

            Val_auc_pesudo = roc_auc_score(y_val, Val_pred_pesudo)
            print("Pesudo Val AUC : ", Val_auc_pesudo)
            print(Style.RESET_ALL)

            CV_Score_pesudo.append(Val_auc_pesudo)
            Val_preds_pesudo.append(Val_pred_pesudo)
            Val_imagenames_pesudo.append(image_names)
            val_targets_pesudo.append(list(y_val))

    valtargets = np.concatenate(val_targets)
    valpreds = np.concatenate(Val_preds)
    valimagenames = np.concatenate(Val_imagenames)

    auc_score = roc_auc_score(valtargets, valpreds)

    print(Fore.YELLOW)
    print("#" * 60)
    print("\nCV(auc_score) : ", auc_score)
    print(f"Mean CV : {np.mean(CV_Score)} +/- {np.std(CV_Score)}\n")

    oof = pd.DataFrame(
        {"image_name": valimagenames, "pred": valpreds, "target": valtargets}
    )

    Test_imagenames = X_test["image_name"].values

    X_train_full = X.drop(["target", "tfrecord", "image_name"], axis=1)
    y_train_full = X["target"].astype(int)

    X_test_full = X_test.drop(["tfrecord", "image_name"], axis=1)

    final_clf = clone(CLF)
    _fit_with_optional_groups(
        final_clf, X_train_full, y_train_full, groups_fit=X["tfrecord"].values
    )

    try:
        test_pred = final_clf.predict_proba(X_test_full)[:, 1]
    except Exception:
        test_pred = final_clf.predict(X_test_full)

    submission = pd.DataFrame({"image_name": Test_imagenames, "target": test_pred})

    if pesudo_label:
        valtargets_pesudo = np.concatenate(val_targets_pesudo)
        valpreds_pesudo = np.concatenate(Val_preds_pesudo)
        valimagenames_pesudo = np.concatenate(Val_imagenames_pesudo)

        auc_score_pesudo = roc_auc_score(valtargets_pesudo, valpreds_pesudo)

        print("Pesudo CV(auc_score) : ", auc_score_pesudo)
        print(
            f"Pesudo Mean CV : {np.mean(CV_Score_pesudo)} +/- {np.std(CV_Score_pesudo)}\n"
        )

        oof_pesudo = pd.DataFrame(
            {
                "image_name": valimagenames_pesudo,
                "pred": valpreds_pesudo,
                "target": valtargets_pesudo,
            }
        )

        try:
            test_pred_pesudo = CLF_pesudo.predict_proba(X_test_full)[:, 1]
        except Exception:
            test_pred_pesudo = CLF_pesudo.predict(X_test_full)

        submission_pesudo = pd.DataFrame(
            {"image_name": Test_imagenames, "target": test_pred_pesudo}
        )

    print("#" * 60, Style.RESET_ALL)

    if show_roc_curve:
        fpr, tpr, _ = roc_curve(valtargets, valpreds)
        plt.figure()
        lw = 2
        plt.plot(
            fpr,
            tpr,
            color="darkorange",
            lw=lw,
            label=f"ROC curve (area = {auc_score:0.4f})",
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve by : {model_name}")
        plt.legend(loc="lower right")
        plt.show()

    if pesudo_label:
        return (
            oof,
            submission,
            auc_score,
            model_name,
            oof_pesudo,
            submission_pesudo,
            auc_score_pesudo,
            str(model_name) + "_Pesudo",
        )
    return oof, submission, auc_score, model_name




## === cell 14
if HAS_LGB:
    clf1 = lgb.LGBMClassifier(
        max_depth=5,
        metric="auc",
        n_estimators=100,
        num_leaves=5,
        boosting_type="gbdt",
        learning_rate=0.1,
        feature_fraction=0.05,
        colsample_bytree=0.1,
        bagging_fraction=0.8,
        bagging_freq=2,
        reg_lambda=0.2,
        n_jobs=4,
        random_state=SEED,
    )
else:
    clf1 = HistGradientBoostingClassifier(
        max_depth=5,
        learning_rate=0.1,
        max_iter=200,
        random_state=SEED,
    )



## === cell 15
clf2 = Pipeline(
    steps=[
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        (
            "lr",
            LogisticRegression(
                C=1.0,
                class_weight=None,
                dual=False,
                fit_intercept=True,
                intercept_scaling=1,
                l1_ratio=None,
                max_iter=200,
                multi_class="auto",
                n_jobs=4,
                penalty="l2",
                random_state=SEED,
                solver="lbfgs",
                tol=0.0001,
                verbose=0,
                warm_start=False,
            ),
        ),
    ]
)



## === cell 16
if HAS_XGB:
    clf3 = XGBClassifier(
        base_score=0.5,
        colsample_bytree=0.8,
        gamma=1,
        learning_rate=0.002,
        max_depth=10,
        min_child_weight=1,
        n_estimators=700,
        n_jobs=4,
        objective="binary:logistic",
        random_state=SEED,
        reg_alpha=0,
        reg_lambda=1,
        scale_pos_weight=1,
        subsample=0.8,
        tree_method="hist",
        verbosity=0,
        eval_metric="logloss",
        use_label_encoder=False,
    )
else:
    clf3 = HistGradientBoostingClassifier(
        max_depth=10,
        learning_rate=0.05,
        max_iter=300,
        random_state=SEED,
    )



## === cell 17
clf4 = GaussianNB(priors=None, var_smoothing=1e-9)



## === cell 18
clf5 = RandomForestClassifier(
    bootstrap=True,
    ccp_alpha=0.0,
    class_weight=None,
    criterion="gini",
    max_depth=5,
    max_features="sqrt",
    max_leaf_nodes=30,
    min_impurity_decrease=0.0,
    min_samples_leaf=2,
    min_samples_split=100,
    min_weight_fraction_leaf=0.0,
    n_estimators=300,
    n_jobs=4,
    oob_score=False,
    random_state=SEED,
    verbose=0,
    warm_start=False,
)



## === cell 19
clf9 = KNeighborsClassifier(
    algorithm="auto",
    leaf_size=30,
    metric="minkowski",
    metric_params=None,
    n_jobs=4,
    n_neighbors=10,
    p=5,
    weights="uniform",
)



## === cell 20
clf10 = DecisionTreeClassifier(
    ccp_alpha=0.0,
    criterion="gini",
    max_depth=5,
    max_features=None,
    max_leaf_nodes=30,
    min_impurity_decrease=0.0,
    min_samples_leaf=2,
    min_samples_split=100,
    min_weight_fraction_leaf=0.0,
    random_state=SEED,
    splitter="best",
)



## === cell 21
clf11 = GradientBoostingClassifier(random_state=SEED)



## === cell 22
stack_cv = GroupKFold(n_splits=FOLDS)

if MlxtendStackingClassifier is not None:
    SCF = MlxtendStackingClassifier(
        classifiers=[clf1, clf5],
        meta_classifier=Pipeline(
            steps=[
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ("lr", LogisticRegression(random_state=SEED, max_iter=200, n_jobs=4)),
            ]
        ),
        use_probas=True,
        average_probas=True,
    )
else:

    class _ShuffledGroupKFold:
        def __init__(self, n_splits=3, random_state=123):
            self.n_splits = n_splits
            self.random_state = random_state
            self._gkf = GroupKFold(n_splits=n_splits)

        def split(self, X, y=None, groups=None):
            rng = np.random.RandomState(self.random_state)
            idx = np.arange(len(X))
            rng.shuffle(idx)
            Xs = X.iloc[idx] if hasattr(X, "iloc") else X[idx]
            ys = y[idx] if y is not None else None
            gs = groups[idx] if groups is not None else None
            for tr_s, va_s in self._gkf.split(Xs, y=ys, groups=gs):
                tr = idx[tr_s]
                va = idx[va_s]
                yield tr, va

        def get_n_splits(self, X=None, y=None, groups=None):
            return self.n_splits

    stack_cv_shuffled = _ShuffledGroupKFold(n_splits=FOLDS, random_state=SEED)

    SCF = SklearnStackingClassifier(
        estimators=[("lgbm_or_hgb", clf1), ("rf", clf5)],
        final_estimator=Pipeline(
            steps=[
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ("lr", LogisticRegression(random_state=SEED, max_iter=200, n_jobs=4)),
            ]
        ),
        stack_method="predict_proba",
        passthrough=False,
        cv=stack_cv_shuffled,
        n_jobs=4,
    )



## === cell 23
if test_pipeline:
    pipelines = []
    pipelines.append(("Booster", Pipeline([("Booster", clf1)])))
    pipelines.append(("LogisticRegression", Pipeline([("LogisticRegression", clf2)])))
    pipelines.append(("XGB_or_HGB", Pipeline([("XGB_or_HGB", clf3)])))
    pipelines.append(("GaussianNB", Pipeline([("GaussianNB", clf4)])))
    pipelines.append(
        ("RandomForestClassifier", Pipeline([("RandomForestClassifier", clf5)]))
    )
    pipelines.append(
        ("KNeighborsClassifier", Pipeline([("KNeighborsClassifier", clf9)]))
    )
    pipelines.append(
        ("DecisionTreeClassifier", Pipeline([("DecisionTreeClassifier", clf10)]))
    )
    pipelines.append(
        (
            "GradientBoostingClassifier",
            Pipeline([("GradientBoostingClassifier", clf11)]),
        )
    )
    pipelines.append(("StackingClassifier", Pipeline([("StackingClassifier", SCF)])))

    X_train = train_coded.drop(["target", "tfrecord", "image_name"], axis=1)
    y_train = train_coded["target"].astype(int)

    M_name = []
    M_auc_score = []
    for name, model in pipelines:
        cv_results = cross_val_score(
            model, X_train, y_train, cv=FOLDS, scoring="roc_auc"
        )
        M_auc_score.append(np.mean(cv_results))
        M_name.append(name)
        print("%s: %f +/- %f" % (name, cv_results.mean(), cv_results.std()))



## === cell 24
select_classifier = clf2
try:
    print(select_classifier.get_params())
except Exception:
    pass



## === cell 25
if Setup_Parameters:
    X_train = train_coded.drop(["target", "tfrecord", "image_name"], axis=1)
    y_train = train_coded["target"].astype(int)

    params = dict(
        lr__C=[0.001, 0.01, 0.1, 1, 10],
        lr__max_iter=[100, 150, 200],
    )

    grid = GridSearchCV(
        estimator=select_classifier,
        param_grid=params,
        cv=FOLDS,
        scoring="roc_auc",
        refit=True,
        n_jobs=4,
    )

    grid.fit(X_train, y_train)

    for r, _ in enumerate(grid.cv_results_["mean_test_score"]):
        print(
            "AUC : %0.5f +/- %0.5f %r"
            % (
                grid.cv_results_["mean_test_score"][r],
                grid.cv_results_["std_test_score"][r] / 2.0,
                grid.cv_results_["params"][r],
            )
        )

    print(f"\nBest parameters: {grid.best_params_}")

    oof, submission, auc_score, model_name = crossValidate(
        grid,
        X=train_coded,
        X_test=test_coded,
        FOLDS=FOLDS,
        SEED=SEED,
        show_roc_curve=False,
        pesudo_label=False,
    )



## === cell 26
CLF = SCF

if pesudo_label:
    (
        oof,
        submission,
        auc_score,
        model_name,
        pesudo_oof,
        pesudo_submission,
        pesudo_auc_score,
        pesudo_model_name,
    ) = crossValidate(
        CLF,
        X=train_coded,
        X_test=test_coded,
        FOLDS=FOLDS,
        SEED=SEED,
        show_roc_curve=False,
        pesudo_label=True,
    )
else:
    oof, submission, auc_score, model_name = crossValidate(
        CLF,
        X=train_coded,
        X_test=test_coded,
        FOLDS=FOLDS,
        SEED=SEED,
        show_roc_curve=False,
        pesudo_label=False,
    )

oof.to_csv(f"oof_{model_name}_{auc_score:0.4f}.csv", index=False)
submission.to_csv(f"submission_{model_name}_{auc_score:0.4f}.csv", index=False)
print("Finished model:", model_name, "CV AUC:", auc_score)

pred_map = submission.set_index("image_name")["target"]
sub = sample_submission[["image_name"]].copy()
sub["target"] = sub["image_name"].map(pred_map)

sub["target"] = (
    sub["target"]
    .fillna(sub["target"].mean() if sub["target"].notna().any() else 0.5)
    .clip(0.0, 1.0)
)

out_path = "submission.csv"
sub.to_csv(out_path, index=False)
print("Wrote:", out_path, "shape:", sub.shape)
print(sub.head())
print("Saved to:", os.path.abspath(out_path))
