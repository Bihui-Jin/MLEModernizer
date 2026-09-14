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

0.940525624083034

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.66203) has done: 'I make the smallest changes needed to (1) ensure your TFRecord-based fold split is correct and stable, (2) prevent your pseudo-label branch from accidentally reusing the same estimator object across folds (which can corrupt CV and final test predictions), and (3) always produce a valid `submission.csv` with probabilities clipped to `[0,1]` and matching `sample_submission` order. These fixes preserve your core approach (metadata one-hot + a base LGB OOF feature + model zoo + optional pseudo-labeling) but remove two logic bugs that can prevent a valid/consistent submission and can materially hurt AUC. I not change model types, losses, or add early stopping; I only adjust fold splitting and model cloning and make the final fit/predict consistent with CV.'
- What this solution (achieved 0.66294) has done: 'Your current 0.66203 is far below the 0.9405 target (higher-is-better), and the biggest score limiter is that the “tfrecord” fold assignment is currently using Python’s `hash()`, which is intentionally non-deterministic across processes and not tied to the real TFRecord shard layout; this makes folds effectively random/noisy and hurts generalization. I replace that with a deterministic mapping that matches the known TFRecord naming convention for this competition (from `image_name` → shard index), while keeping the rest of your pipeline (metadata features, base LGB OOF feature, model zoo, pseudo-label option, training loops) unchanged. I also make the pseudo-label selection thresholds slightly less extreme (0.99/0.01 → 0.98/0.02) to add a bit more high-confidence data per fold, which typically improves AUC without changing the approach. Finally, I keep submission ordering aligned to `sample_submission` and continue writing a valid `submission.csv`.'
- What this solution (achieved 0.66304) has done: 'I fix the fold-splitting crash by making TFRecord group assignment always produce multiple groups (the current `_assign_tfrecord_ids` can collapse to a single group due to clipping), and I make `crossValidate()` robust by falling back to `StratifiedKFold` when there are fewer than 2 unique groups. I also ensure every model’s test prediction output is interpreted as a probability (especially regressors) by applying a sigmoid when `predict_proba` is unavailable, which is score-improving but does not change the modeling approach. Finally, I harden the submission creation to always exactly match `sample_submission` order and row count, writing a valid `submission.csv` in the working directory.'

# 9. Code solution

## === cell 0
import os
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    get_ipython  # type: ignore
    try:
        get_ipython().run_line_magic("matplotlib", "inline")  # type: ignore
    except Exception:
        pass
except Exception:
    pass

try:
    from colorama import Fore, Back, Style
except Exception:

    class _Dummy:
        def __getattr__(self, name):
            return ""

    Fore = Back = Style = _Dummy()

import lightgbm as lgb  # CLF1
from sklearn.linear_model import LogisticRegression  # CLF2
from xgboost import XGBRegressor  # CLF3
from sklearn.naive_bayes import GaussianNB  # CLF4
from sklearn.ensemble import RandomForestClassifier  # CLF5
from sklearn.linear_model import LinearRegression  # CLF6
from sklearn.linear_model import Lasso  # CLF7
from sklearn.linear_model import ElasticNet  # CLF8
from sklearn.neighbors import KNeighborsRegressor  # CLF9
from sklearn.tree import DecisionTreeRegressor  # CLF10
from sklearn.ensemble import GradientBoostingRegressor  # CLF11
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis  # CLF12

from sklearn.ensemble import StackingClassifier as SklearnStackingClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, roc_curve

from sklearn.base import clone

warnings.filterwarnings(action="ignore", category=DeprecationWarning, module="sklearn")
warnings.simplefilter("ignore")


def seed_everything(SEED: int):
    np.random.seed(SEED)
    os.environ["PYTHONHASHSEED"] = str(SEED)




## === cell 1
FOLDS = 3
SEED = 123
Setup_Parameters = False
seed_everything(SEED)
file_add_list = [1, 2, 3, 4, 5]
pesudo_label = True

JITTER_SCALE = 0.0




## === cell 2
BASE_PATH_CANDIDATES = [
    "../input/siim-isic-melanoma-classification",
    "/kaggle/input/siim-isic-melanoma-classification",
    "/kaggle/data/siim-isic-melanoma-classification",
    "../kaggle/data/siim-isic-melanoma-classification",
    "/kaggle/data",
    "/kaggle/input",
]
BASE_PATH = None
for p in BASE_PATH_CANDIDATES:
    if os.path.exists(p) and os.path.exists(os.path.join(p, "train.csv")):
        BASE_PATH = p
        break
if BASE_PATH is None:
    BASE_PATH = "../input"

print("Using BASE_PATH:", BASE_PATH)




## === cell 3
train_metadata = pd.read_csv(os.path.join(BASE_PATH, "train.csv"))
test_metadata = pd.read_csv(os.path.join(BASE_PATH, "test.csv"))
sample_submission = pd.read_csv(os.path.join(BASE_PATH, "sample_submission.csv"))


def _assign_tfrecord_ids(image_names: pd.Series, n_tf: int = 15) -> pd.DataFrame:
    img = image_names.astype(str)

    import zlib

    ids = img.apply(lambda s: zlib.crc32(s.encode("utf-8")) & 0xFFFFFFFF).astype(
        np.int64
    )
    tf_ids = (ids % int(n_tf)).astype(int)

    return pd.DataFrame({"image_name": image_names.values, "tfrecord": tf_ids.values})


tfrecord_number_df = _assign_tfrecord_ids(train_metadata["image_name"], n_tf=15)




## === cell 4
assert "image_name" in train_metadata.columns and "target" in train_metadata.columns
assert "image_name" in test_metadata.columns
assert list(sample_submission.columns) == ["image_name", "target"]




## === cell 5
print("Train data shape : ", train_metadata.shape)
print("Test data shape : ", test_metadata.shape)




## === cell 6
def align_columns(train_df: pd.DataFrame, test_df: pd.DataFrame, drop_cols=()):
    train_df = train_df.copy()
    test_df = test_df.copy()
    for c in drop_cols:
        if c in train_df.columns:
            train_df = train_df.drop(columns=[c])
        if c in test_df.columns:
            test_df = test_df.drop(columns=[c])

    all_cols = sorted(set(train_df.columns).union(set(test_df.columns)))
    for c in all_cols:
        if c not in train_df.columns:
            train_df[c] = 0
        if c not in test_df.columns:
            test_df[c] = 0
    return train_df[all_cols], test_df[all_cols]




## === cell 7
print("Unique values in column with frequency : ")

print("\nsex : ", dict(train_metadata.sex.value_counts(dropna=False)))
print(
    "\nage_approx : ",
    dict(train_metadata.age_approx.value_counts(dropna=False).head(10)),
)
print(
    "\nanatom_site_general_challenge : ",
    dict(train_metadata.anatom_site_general_challenge.value_counts(dropna=False)),
)
print(
    "\ndiagnosis : ", dict(train_metadata.diagnosis.value_counts(dropna=False).head(10))
)
print(
    "\nbenign_malignant : ",
    dict(train_metadata.benign_malignant.value_counts(dropna=False)),
)
print("\ntarget : ", dict(train_metadata.target.value_counts(dropna=False)))




## === cell 8
def build_metadata_features(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    d = df.copy()
    d["age_approx"] = d["age_approx"].astype(float)
    d["age_approx"] = d["age_approx"].fillna(d["age_approx"].mean())
    sex_code = pd.get_dummies(d["sex"], prefix="sex")
    anatom_code = pd.get_dummies(
        d["anatom_site_general_challenge"], prefix="anatom_site"
    )
    age_norm = (d["age_approx"] - d["age_approx"].mean()) / (
        d["age_approx"].std() + 1e-9
    )
    out = pd.concat(
        [d["image_name"], sex_code, age_norm.rename("age_approx_norm"), anatom_code],
        axis=1,
    )
    if is_train:
        out = pd.concat([out, d["target"]], axis=1)
    return out


train_coded = build_metadata_features(train_metadata, is_train=True)
test_coded = build_metadata_features(test_metadata, is_train=False)

train_feat = train_coded.drop(columns=["image_name", "target"])
test_feat = test_coded.drop(columns=["image_name"])
train_feat_aligned, test_feat_aligned = align_columns(train_feat, test_feat)

train_coded = pd.concat(
    [train_coded[["image_name"]], train_feat_aligned, train_coded[["target"]]], axis=1
)
test_coded = pd.concat([test_coded[["image_name"]], test_feat_aligned], axis=1)

print("train_coded shape:", train_coded.shape)
print("test_coded shape:", test_coded.shape)




## === cell 9
print("Unique values in column with frequency : ")

print("\nsex : ", dict(test_metadata.sex.value_counts(dropna=False)))
print(
    "\nage_approx : ",
    dict(test_metadata.age_approx.value_counts(dropna=False).head(10)),
)
print(
    "\nanatom_site_general_challenge : ",
    dict(test_metadata.anatom_site_general_challenge.value_counts(dropna=False)),
)




## === cell 10
def make_oof_and_test_preds_via_lgb(
    train_df: pd.DataFrame, test_df: pd.DataFrame, seed: int, n_splits: int = 5
):
    X = train_df.drop(columns=["image_name", "target"])
    y = train_df["target"].astype(int).values
    X_test = test_df.drop(columns=["image_name"])

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    oof = np.zeros(len(train_df), dtype=float)
    test_pred = np.zeros(len(test_df), dtype=float)

    params = dict(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        reg_lambda=0.0,
        objective="binary",
    )
    for tr_idx, va_idx in skf.split(X, y):
        clf = lgb.LGBMClassifier(**params)
        clf.fit(X.iloc[tr_idx], y[tr_idx])
        oof[va_idx] = clf.predict_proba(X.iloc[va_idx])[:, 1]
        test_pred += clf.predict_proba(X_test)[:, 1] / n_splits
    return oof, test_pred


_base_oof, _base_test = make_oof_and_test_preds_via_lgb(
    train_coded, test_coded, seed=SEED, n_splits=5
)




## === cell 11
train_coded.head()




## === cell 12
train_coded = pd.merge(tfrecord_number_df, train_coded, on="image_name", how="left")
assert "tfrecord" in train_coded.columns




## === cell 13
def add_OOF_pred(train_coded_in: pd.DataFrame, num: int):
    train_coded_local = train_coded_in.copy()
    rng = np.random.RandomState(SEED)
    for n in file_add_list:
        jitter = rng.normal(
            loc=0.0, scale=JITTER_SCALE * n, size=len(train_coded_local)
        )
        preds = np.clip(_base_oof + jitter, 0.0, 1.0)
        train_coded_local[f"pred_{n}"] = preds
    return train_coded_local


train_coded = add_OOF_pred(train_coded, 5)
train_coded.to_csv("train_coded.csv", index=False)
train_coded.tail()




## === cell 14
test_coded = test_coded.copy()
test_coded["tfrecord"] = _assign_tfrecord_ids(test_coded["image_name"], n_tf=15)[
    "tfrecord"
].values




## === cell 15
test_coded.head()




## === cell 16
pass




## === cell 17
def add_submission_pred(test_coded_in: pd.DataFrame, num: int):
    test_coded_local = test_coded_in.copy()
    rng = np.random.RandomState(SEED + 999)
    for n in file_add_list:
        jitter = rng.normal(loc=0.0, scale=JITTER_SCALE * n, size=len(test_coded_local))
        preds = np.clip(_base_test + jitter, 0.0, 1.0)
        test_coded_local[f"pred_{n}"] = preds
    return test_coded_local


test_coded = add_submission_pred(test_coded, 5)
test_coded.to_csv("test_coded.csv", index=False)
test_coded.tail()




## === cell 18
def _sigmoid(x):
    x = np.asarray(x, dtype=float)
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))


def _predict_proba_1d(model, X):
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)[:, 1]
        return np.asarray(p, dtype=float)
    raw = model.predict(X)
    raw = np.asarray(raw, dtype=float)
    if np.nanmin(raw) >= -1e-6 and np.nanmax(raw) <= 1.0 + 1e-6:
        return np.clip(raw, 0.0, 1.0)
    return _sigmoid(raw)




## === cell 19
def crossValidate(
    CLF,
    X=train_coded,
    X_test=test_coded,
    FOLDS=3,
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

    X_local = X.reset_index(drop=True).copy()
    groups = X_local["tfrecord"].astype(int).values
    y_all = X_local["target"].astype(int).values

    unique_groups = np.unique(groups)
    use_group = len(unique_groups) >= 2
    if use_group:
        n_splits = int(min(FOLDS, len(unique_groups)))
        if n_splits < 2:
            use_group = False
        else:
            splitter = GroupKFold(n_splits=n_splits)
            split_iter = splitter.split(X_local, y_all, groups)
    if not use_group:
        splitter = StratifiedKFold(
            n_splits=max(2, int(FOLDS)), shuffle=True, random_state=SEED
        )
        split_iter = splitter.split(X_local, y_all)

    fold_models = []
    fold_models_pseudo = []

    best_fold_auc = -np.inf
    best_model = None
    best_model_pesudo = None

    for fold, (idxT, idxV) in enumerate(split_iter):
        X_train_main = X_local.iloc[idxT].copy()
        y_train_main = X_local.iloc[idxT].copy()
        X_val_main = X_local.iloc[idxV].copy()
        y_val_main = X_local.iloc[idxV].copy()

        print(Fore.MAGENTA)
        print("#" * 60, Style.RESET_ALL)
        print(Fore.BLUE)
        print("FOLD : ", fold)
        if use_group:
            print("Train TFrecords : ", sorted(X_train_main.tfrecord.unique().tolist()))
            print(
                "Validation TFrecords : ", sorted(X_val_main.tfrecord.unique().tolist())
            )
        else:
            print("Split method: StratifiedKFold (fallback; insufficient groups)")
        image_names = list(X_val_main["image_name"])

        X_train = X_train_main.drop(["target", "tfrecord"], axis=1).iloc[:, 1:]
        y_train = y_train_main["target"].astype(int)

        X_val = X_val_main.drop(["target", "tfrecord"], axis=1).iloc[:, 1:]
        y_val = y_val_main["target"].astype(int)

        if X_train.shape[0] == 0 or X_val.shape[0] == 0:
            raise RuntimeError(
                f"Empty fold encountered: fold={fold}, X_train={X_train.shape}, X_val={X_val.shape}. "
                f"Check split configuration."
            )

        fold_model = clone(CLF)
        fold_model.fit(X_train, y_train)
        fold_models.append(fold_model)

        y_train_pred = _predict_proba_1d(fold_model, X_train)
        print("Train AUC : ", roc_auc_score(y_train, y_train_pred))

        Val_pred = _predict_proba_1d(fold_model, X_val)
        Val_auc = roc_auc_score(y_val, Val_pred)
        print("Val AUC : ", Val_auc)

        CV_Score.append(Val_auc)
        Val_preds.append(Val_pred)
        Val_imagenames.append(image_names)
        val_targets.append(list(y_val))

        if Val_auc > best_fold_auc:
            best_fold_auc = Val_auc
            best_model = fold_model

        if pesudo_label:
            CLF_pesudo = clone(CLF)

            train2_Pesudo = X_train_main.copy()
            train2_Pesudo["target_label"] = y_train_main["target"].astype(int)

            X_test_full = X_test.copy()
            X_test_feat = X_test_full.drop(columns=["tfrecord"]).iloc[:, 1:]
            test_pred_for_pseudo = _predict_proba_1d(fold_model, X_test_feat)

            test2_Pesudo = X_test_full.copy()
            test2_Pesudo["target_label"] = test_pred_for_pseudo

            test2_Pesudo = test2_Pesudo[
                (test2_Pesudo["target_label"] >= 0.98)
                | (test2_Pesudo["target_label"] <= 0.02)
            ]
            test2_Pesudo.loc[test2_Pesudo["target_label"] >= 0.5, "target_label"] = 1
            test2_Pesudo.loc[test2_Pesudo["target_label"] < 0.5, "target_label"] = 0

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

            train_pesudo = pd.concat([train2_Pesudo, test2_Pesudo], axis=0)

            X_train_pesudo = train_pesudo.drop(
                ["target_label", "target", "tfrecord"], axis=1, errors="ignore"
            ).iloc[:, 1:]
            y_train_pesudo = train_pesudo["target_label"].astype(int)

            CLF_pesudo.fit(X_train_pesudo, y_train_pesudo)
            fold_models_pseudo.append(CLF_pesudo)

            y_train_pred_pesudo = _predict_proba_1d(CLF_pesudo, X_train_pesudo)
            print(
                "Pesudo Train AUC : ",
                roc_auc_score(y_train_pesudo, y_train_pred_pesudo),
            )

            Val_pred_pesudo = _predict_proba_1d(CLF_pesudo, X_val)
            Val_auc_pesudo = roc_auc_score(y_val, Val_pred_pesudo)
            print("Pesudo Val AUC : ", Val_auc_pesudo)
            print(Style.RESET_ALL)

            CV_Score_pesudo.append(Val_auc_pesudo)
            Val_preds_pesudo.append(Val_pred_pesudo)
            Val_imagenames_pesudo.append(image_names)
            val_targets_pesudo.append(list(y_val))

            if Val_auc_pesudo >= (
                best_fold_auc if best_fold_auc is not None else -np.inf
            ):
                best_model_pesudo = CLF_pesudo

    valtargets = np.concatenate(val_targets)
    valpreds = np.concatenate(Val_preds)
    valimagenames = np.concatenate(Val_imagenames)

    auc_score = roc_auc_score(valtargets, valpreds)

    print(Fore.YELLOW)
    print("#" * 60)
    print("\nCV(auc_score) : ", auc_score)
    print(f"Mean CV : {np.mean(CV_Score)} +/- {np.std(CV_Score)}\n")

    oof = pd.DataFrame()
    oof["image_name"] = valimagenames
    oof["pred"] = valpreds
    oof["target"] = valtargets

    Test_imagenames = X_test["image_name"]
    X_test_feat = X_test.drop(columns=["tfrecord"]).iloc[:, 1:]

    if len(fold_models) == 0:
        best_model = clone(CLF).fit(
            X_local.drop(["target", "tfrecord"], axis=1).iloc[:, 1:],
            X_local["target"].astype(int),
        )
        test_pred = _predict_proba_1d(best_model, X_test_feat)
    else:
        test_pred = np.zeros(len(X_test_feat), dtype=float)
        for m in fold_models:
            test_pred += _predict_proba_1d(m, X_test_feat) / len(fold_models)

    submission = pd.DataFrame()
    submission["image_name"] = Test_imagenames
    submission["target"] = np.clip(test_pred.astype(float), 0.0, 1.0)

    if pesudo_label:
        valtargets_pesudo = (
            np.concatenate(val_targets_pesudo)
            if len(val_targets_pesudo)
            else valtargets
        )
        valpreds_pesudo = (
            np.concatenate(Val_preds_pesudo) if len(Val_preds_pesudo) else valpreds
        )
        valimagenames_pesudo = (
            np.concatenate(Val_imagenames_pesudo)
            if len(Val_imagenames_pesudo)
            else valimagenames
        )

        auc_score_pesudo = roc_auc_score(valtargets_pesudo, valpreds_pesudo)

        print("Pesudo CV(auc_score) : ", auc_score_pesudo)
        print(
            f"Pesudo Mean CV : {np.mean(CV_Score_pesudo) if len(CV_Score_pesudo) else np.mean(CV_Score)} +/- "
            f"{np.std(CV_Score_pesudo) if len(CV_Score_pesudo) else np.std(CV_Score)}\n"
        )

        oof_pesudo = pd.DataFrame()
        oof_pesudo["image_name"] = valimagenames_pesudo
        oof_pesudo["pred"] = valpreds_pesudo
        oof_pesudo["target"] = valtargets_pesudo

        if len(fold_models_pseudo) == 0:
            model_for_pseudo_test = (
                best_model_pesudo if best_model_pesudo is not None else best_model
            )
            test_pred_pesudo = _predict_proba_1d(model_for_pseudo_test, X_test_feat)
        else:
            test_pred_pesudo = np.zeros(len(X_test_feat), dtype=float)
            for m in fold_models_pseudo:
                test_pred_pesudo += _predict_proba_1d(m, X_test_feat) / len(
                    fold_models_pseudo
                )

        submission_pesudo = pd.DataFrame()
        submission_pesudo["image_name"] = Test_imagenames
        submission_pesudo["target"] = np.clip(test_pred_pesudo.astype(float), 0.0, 1.0)

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
        if pesudo_label and len(Val_preds_pesudo):
            fpr_p, tpr_p, _ = roc_curve(valtargets_pesudo, valpreds_pesudo)
            plt.plot(
                fpr_p,
                tpr_p,
                color="red",
                lw=lw,
                label=f"Pesudo ROC (area = {auc_score_pesudo:0.4f})",
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




## === cell 20
pass




## === cell 21
pass




## === cell 22
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
    random_state=SEED,
)




## === cell 23
pass




## === cell 24
clf2 = LogisticRegression(
    C=1.0,
    class_weight=None,
    dual=False,
    fit_intercept=True,
    intercept_scaling=1,
    l1_ratio=None,
    max_iter=200,
    multi_class="auto",
    n_jobs=None,
    penalty="l2",
    random_state=SEED,
    solver="lbfgs",
    tol=0.0001,
    verbose=0,
    warm_start=False,
)




## === cell 25
pass




## === cell 26
clf3 = XGBRegressor(
    base_score=0.5,
    booster=None,
    colsample_bylevel=1,
    colsample_bynode=1,
    colsample_bytree=0.8,
    gamma=1,
    gpu_id=-1,
    importance_type="gain",
    learning_rate=0.002,
    max_depth=10,
    min_child_weight=1,
    n_estimators=700,
    n_jobs=-1,
    nthread=-1,
    num_parallel_tree=1,
    objective="binary:logistic",
    random_state=SEED,
    reg_alpha=0,
    reg_lambda=1,
    scale_pos_weight=1,
    subsample=0.8,
    tree_method=None,
    validate_parameters=False,
    verbosity=0,
)




## === cell 27
pass




## === cell 28
clf4 = GaussianNB(priors=None, var_smoothing=1e-09)




## === cell 29
pass




## === cell 30
clf5 = RandomForestClassifier(
    bootstrap=True,
    ccp_alpha=0.0,
    class_weight=None,
    criterion="gini",
    max_depth=5,
    max_features="sqrt",
    max_leaf_nodes=30,
    max_samples=None,
    min_impurity_decrease=0.0,
    min_samples_leaf=2,
    min_samples_split=100,
    min_weight_fraction_leaf=0.0,
    n_estimators=300,
    n_jobs=-1,
    oob_score=False,
    random_state=SEED,
    verbose=0,
    warm_start=False,
)




## === cell 31
pass




## === cell 32
clf6 = LinearRegression()




## === cell 33
clf7 = Lasso()




## === cell 34
clf8 = ElasticNet()




## === cell 35
pass




## === cell 36
clf12 = QuadraticDiscriminantAnalysis()




## === cell 37
pass




## === cell 38
clf9 = KNeighborsRegressor(
    algorithm="auto",
    leaf_size=30,
    metric="minkowski",
    metric_params=None,
    n_jobs=-1,
    n_neighbors=10,
    p=5,
    weights="uniform",
)




## === cell 39
pass




## === cell 40
clf10 = DecisionTreeRegressor(
    ccp_alpha=0.0,
    criterion="squared_error",
    max_depth=5,
    max_features="sqrt",
    max_leaf_nodes=30,
    min_impurity_decrease=0.0,
    min_samples_leaf=2,
    min_samples_split=100,
    min_weight_fraction_leaf=0.0,
    random_state=SEED,
    splitter="best",
)




## === cell 41
pass




## === cell 42
clf11 = GradientBoostingRegressor(random_state=SEED)




## === cell 43
pass




## === cell 44
SCF = SklearnStackingClassifier(
    estimators=[("lgbm", clf1), ("rf", clf5)],
    final_estimator=LogisticRegression(max_iter=300, random_state=SEED),
    stack_method="predict_proba",
    passthrough=False,
    n_jobs=None,
    cv=3,
)




## === cell 45
if False:
    pipelines = []
    pipelines.append(("LGBMClassifier", Pipeline([("LGBMClassifier", clf1)])))
    pipelines.append(("LogisticRegression", Pipeline([("LogisticRegression", clf2)])))
    pipelines.append(("XGBRegressor", Pipeline([("XGBRegressor", clf3)])))
    pipelines.append(("GaussianNB", Pipeline([("GaussianNB", clf4)])))
    pipelines.append(
        ("RandomForestClassifier", Pipeline([("RandomForestClassifier", clf5)]))
    )
    pipelines.append(("KNeighborsRegressor", Pipeline([("KNeighborsRegressor", clf9)])))
    pipelines.append(
        ("DecisionTreeRegressor", Pipeline([("DecisionTreeRegressor", clf10)]))
    )
    pipelines.append(
        ("GradientBoostingRegressor", Pipeline([("GradientBoostingRegressor", clf11)]))
    )
    pipelines.append(("StackingClassifier", Pipeline([("StackingClassifier", SCF)])))

    X_train = train_coded.drop("target", axis=1).iloc[:, 1:]
    y_train = train_coded["target"]

    from sklearn.model_selection import cross_val_score

    M_name = []
    M_auc_score = []
    for name, model in pipelines:
        cv_results = cross_val_score(
            model, X_train, y_train, cv=FOLDS, scoring="roc_auc"
        )
        M_auc_score.append(np.mean(cv_results))
        M_name.append(name)
        print("%s: %f +/- %f" % (name, cv_results.mean(), cv_results.std()))




## === cell 46
if False:
    fig, ax = plt.subplots(figsize=(7, 7))
    bars = ax.bar(x=M_name, height=M_auc_score, tick_label=M_name)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#DDDDDD")
    ax.tick_params(bottom=False, left=False)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#EEEEEE")
    ax.xaxis.grid(False)
    bar_color = bars[0].get_facecolor()
    plt.xticks(rotation=90)
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            round(bar.get_height(), 4),
            horizontalalignment="center",
            color=bar_color,
            weight="bold",
        )
    fig.tight_layout()




## === cell 47
pass




## === cell 48
select_classifier = clf2
select_classifier.get_params()




## === cell 49
if Setup_Parameters:
    X_train = train_coded.drop("target", axis=1).iloc[:, 1:]
    y_train = train_coded["target"]

    params = dict()

    grid = GridSearchCV(
        estimator=select_classifier,
        param_grid=params,
        cv=FOLDS,
        scoring="roc_auc",
        refit=False,
        n_jobs=-1,
    )

    grid.fit(X_train, y_train)




## === cell 50
pass




## === cell 51
M_name = []
M_auc_score = []
last_submission = None
last_model_name = None
last_auc = None

for CLF in [clf1, clf2, clf4, clf5, clf10, clf11, SCF]:
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
    M_name.append(model_name)
    M_auc_score.append(auc_score)

    if (last_auc is None) or (auc_score > last_auc):
        last_auc = auc_score
        last_submission = submission.copy()
        last_model_name = model_name

    if pesudo_label:
        pesudo_oof.to_csv(
            f"Pesudo_oof_{model_name}_{pesudo_auc_score:0.4f}.csv", index=False
        )
        pesudo_submission.to_csv(
            f"Pesudo_submission_{model_name}_{pesudo_auc_score:0.4f}.csv", index=False
        )
        M_name.append(pesudo_model_name)
        M_auc_score.append(pesudo_auc_score)

        if pesudo_auc_score > (last_auc if last_auc is not None else -np.inf):
            last_auc = pesudo_auc_score
            last_submission = pesudo_submission.copy()
            last_model_name = pesudo_model_name

if last_submission is None:
    last_submission = sample_submission.copy()
    last_submission["target"] = float(train_metadata["target"].mean())
    last_model_name = "fallback_mean"
    last_auc = np.nan

sub = sample_submission[["image_name"]].merge(
    last_submission[["image_name", "target"]],
    on="image_name",
    how="left",
    validate="one_to_one",
)
sub["target"] = pd.to_numeric(sub["target"], errors="coerce")
if sub["target"].isna().any():
    sub["target"] = sub["target"].fillna(float(train_metadata["target"].mean()))
sub["target"] = sub["target"].clip(0.0, 1.0)

assert len(sub) == len(
    sample_submission
), "Submission row count mismatch vs sample_submission"
assert (
    sub["image_name"].tolist() == sample_submission["image_name"].tolist()
), "Submission order mismatch"

sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv using model:", last_model_name, "CV AUC:", last_auc)




## === cell 52
pass




## === cell 53
if len(M_name) > 0:
    fig, ax = plt.subplots(figsize=(15, 7))
    bars = ax.bar(x=M_name, height=M_auc_score, tick_label=M_name)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#DDDDDD")
    ax.tick_params(bottom=False, left=False)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#EEEEEE")
    ax.xaxis.grid(False)

    bar_color = bars[0].get_facecolor()
    plt.xticks(rotation=90)

    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            round(bar.get_height(), 5),
            horizontalalignment="center",
            color=bar_color,
            weight="bold",
        )
    fig.tight_layout()
else:
    print("No models were evaluated; skipping bar plot.")
