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
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scipy==1.15.3
seaborn==0.12.2
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

0.9445852127725416

# 6. Current score

0.7673

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.66764) has done: 'Your notebook is failing because it expects an external Kaggle dataset `../input/pseudolabelmodels/` (and specific CSV filenames) that does not exist in your environment, so `glob` returns nothing and every downstream step crashes. To keep the core “rank-average ensemble of multiple submission CSVs” logic intact, I make the code robust to missing external pseudo-label submissions by instead discovering any available `sample_submission.csv`/`submission*.csv` under `../input/` and, if none exist besides the provided sample, falling back to a simple metadata-only model trained from `train.csv` to produce valid probabilities for `test.csv`. This guarantees an end-to-end run and a valid `submission.csv` with correct columns and row alignment. The fallback model is lightweight (LogisticRegression on cleaned/encoded metadata) and should produce a non-trivial AUC, moving you upward from “no submission” toward the target.'
- What this solution (achieved 0.66764) has done: 'Your current 0.66764 score is far below the target 0.9446 (higher-is-better), and the main reason is that the pipeline falls back to a weak metadata-only LogisticRegression because no real external “pseudolabelmodels” submissions exist in this environment. To move toward the target with minimal semantic change, I keep your ensemble logic intact but expand the discovery to also include any `*.csv` under `../input/**` that look like predictions (have `image_name` + `target`, correct row count), while explicitly excluding `train.csv`/`test.csv`/`sample_submission.csv` to avoid ensembling nonsense. If still nothing is found, the fallback stays the same (so it always produces a valid submission), but in typical Kaggle environments this broader search pick up your own imported submission files and markedly improve AUC toward the target. I also enforce alignment to `sample_submission` order before ensembling/writing, preventing silent row-mismatch score drops.'
- What this solution (achieved 0.7673) has done: 'Your current score (0.66764) is far below the target AUC (0.9446), and the limiting factor is the weak metadata-only fallback model that activates when no external prediction CSVs are found. With minimal changes and the same core approach (linear model on metadata with CV), I (1) add safe, high-signal metadata features that exist in your CSVs (notably `patient_id`, plus missing/unknown indicators), (2) switch to patient-grouped CV to reduce leakage (more realistic generalization that typically improves public AUC for this competition), and (3) tune LogisticRegression regularization slightly while keeping the same model family/training loop. The ensemble logic and submission-writing remain unchanged; if usable external prediction CSVs exist, they still be used as before.'

# 9. Code solution

## === cell 0
import os
import glob
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE_INPUT = "../input"
COMP_DIR = os.path.join(BASE_INPUT, "siim-isic-melanoma-classification")

LABELS = ["target"]

print("Listing ../input:")
try:
    print("\n".join(sorted(os.listdir(BASE_INPUT))[:50]))
except Exception as e:
    print("Could not list ../input:", repr(e))



## === cell 1
from scipy.stats import rankdata

PSEUDO_DIR = os.path.join(BASE_INPUT, "pseudolabelmodels")
all_files = glob.glob(os.path.join(PSEUDO_DIR, "*.csv"))

print(f"Found {len(all_files)} files under {PSEUDO_DIR}")


def _load_csv_safe(path1, path2):
    if os.path.exists(path1):
        return pd.read_csv(path1)
    return pd.read_csv(path2)


train_path = os.path.join(COMP_DIR, "train.csv")
test_path = os.path.join(COMP_DIR, "test.csv")
sample_path = os.path.join(COMP_DIR, "sample_submission.csv")

train_df = _load_csv_safe(os.path.join(BASE_INPUT, "train.csv"), train_path)
test_df = _load_csv_safe(os.path.join(BASE_INPUT, "test.csv"), test_path)
sample_sub = _load_csv_safe(
    os.path.join(BASE_INPUT, "sample_submission.csv"), sample_path
)

test_df = test_df.merge(sample_sub[["image_name"]], on="image_name", how="right")
test_image_order = sample_sub["image_name"].tolist()
test_image_set = set(test_image_order)
expected_n_test = len(sample_sub)


def _is_excluded_basename(p: str) -> bool:
    bn = os.path.basename(p).lower()
    return bn in {
        "train.csv",
        "test.csv",
        "sample_submission.csv",
        "train.csv.zip",
        "test.csv.zip",
        "sample_submission.csv.zip",
    }


if len(all_files) == 0:
    candidates = []
    candidates += glob.glob(os.path.join(BASE_INPUT, "*", "submission*.csv"))
    candidates += glob.glob(os.path.join(BASE_INPUT, "*", "*", "submission*.csv"))
    candidates += glob.glob(os.path.join(BASE_INPUT, "*", "sample_submission.csv"))
    candidates += glob.glob(os.path.join(BASE_INPUT, "*", "*", "sample_submission.csv"))
    candidates += glob.glob(os.path.join(BASE_INPUT, "**", "*.csv"), recursive=True)

    seen = set()
    all_files = []
    for f in candidates:
        if f not in seen and os.path.isfile(f) and (not _is_excluded_basename(f)):
            seen.add(f)
            all_files.append(f)

print("Discovered CSV candidates (first 40):")
print("\n".join(all_files[:40]))
print("Total discovered candidates:", len(all_files))



## === cell 2
concat_sub = None
usable_files = []


def _read_pred_csv(path):
    try:
        df = pd.read_csv(path)
    except Exception:
        return None

    cols = set(df.columns)
    if not {"image_name", "target"}.issubset(cols):
        return None

    df = df[["image_name", "target"]].copy()
    df = df.drop_duplicates(subset=["image_name"], keep="first")

    if len(df) != expected_n_test:
        return None
    if set(df["image_name"].tolist()) != test_image_set:
        return None

    df = df.set_index("image_name").loc[test_image_order].reset_index()
    return df


for f in all_files:
    dfp = _read_pred_csv(f)
    if dfp is None:
        continue
    usable_files.append(f)

print(
    f"Usable prediction-like CSVs with image_name+target matching test set: {len(usable_files)}"
)
print("Usable files (up to 20):")
print("\n".join(usable_files[:20]))

outs = []
if len(usable_files) > 0:
    for f in usable_files:
        dfp = _read_pred_csv(f)
        if dfp is None:
            continue
        dfp = dfp.set_index("image_name")
        outs.append(dfp[["target"]].rename(columns={"target": os.path.basename(f)}))

if len(outs) >= 2:
    concat_sub = pd.concat(outs, axis=1)
    cols = [f"m{i}" for i in range(len(concat_sub.columns))]
    concat_sub.columns = cols
    concat_sub.reset_index(inplace=True)
    if concat_sub.columns[0] != "image_name":
        concat_sub.rename(columns={concat_sub.columns[0]: "image_name"}, inplace=True)
    print("concat_sub shape:", concat_sub.shape)
else:
    print("Not enough external prediction files to ensemble; will use fallback model.")
    concat_sub = None



## === cell 3
m_gmean = None
if concat_sub is not None:
    rank = np.tril(concat_sub.iloc[:, 1:].corr().values, -1)
    m = (rank > 0).sum()
    m_gmean_acc, s = 0.0, 0.0
    for n in range(min(rank.shape[0], m if m > 0 else rank.shape[0])):
        mx = np.unravel_index(rank.argmin(), rank.shape)
        w = (m - n) / (m + n / 10) if (m + n / 10) != 0 else 1.0
        a = concat_sub.iloc[:, mx[0] + 1].astype(float).clip(1e-12, 1.0)
        b = concat_sub.iloc[:, mx[1] + 1].astype(float).clip(1e-12, 1.0)
        m_gmean_acc += w * (np.log(a) + np.log(b)) / 2.0
        s += w
        rank[mx] = 1
    m_gmean = np.exp(m_gmean_acc / s).clip(0.0, 1.0)
    print("Computed m_gmean from ensemble files.")
else:
    print("Skipping m_gmean computation (no concat_sub).")



## === cell 4
predictions = None
if concat_sub is not None:
    preds_mat = concat_sub.iloc[:, 1:].to_numpy(dtype=float)  # shape (N, K)
    n, k = preds_mat.shape
    ranked = np.zeros_like(preds_mat, dtype=float)
    for j in range(k):
        ranked[:, j] = rankdata(preds_mat[:, j]) / n
    avg_rank = ranked.mean(axis=1).clip(0.0, 1.0)

    if m_gmean is not None and len(m_gmean) == len(avg_rank):
        avg_rank = (0.7 * avg_rank + 0.3 * np.asarray(m_gmean, dtype=float)).clip(
            0.0, 1.0
        )

    predictions = avg_rank.reshape(-1, 1)
    print("Ensemble predictions computed:", predictions.shape)
else:
    print("No ensemble predictions computed in this cell.")



## === cell 5
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

if predictions is None:
    y = train_df["target"].astype(int).values

    for df in (train_df, test_df):
        if "sex" in df.columns:
            df["sex"] = df["sex"].astype("object")
        if "anatom_site_general_challenge" in df.columns:
            df["anatom_site_general_challenge"] = df[
                "anatom_site_general_challenge"
            ].astype("object")
        if "patient_id" in df.columns:
            df["patient_id"] = df["patient_id"].astype("object")

    def _add_flags(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["age_missing"] = out["age_approx"].isna().astype(int)
        out["sex_missing"] = out["sex"].isna().astype(int)
        out["site_missing"] = out["anatom_site_general_challenge"].isna().astype(int)
        out["sex_unknown"] = (
            out["sex"]
            .fillna("")
            .astype(str)
            .str.strip()
            .isin(["", "unknown"])
            .astype(int)
        )
        out["site_unknown"] = (
            out["anatom_site_general_challenge"]
            .fillna("")
            .astype(str)
            .str.strip()
            .isin(["", "unknown"])
            .astype(int)
        )
        return out

    train_fe = _add_flags(train_df)
    test_fe = _add_flags(test_df)

    feature_cols_num = [
        "age_approx",
        "age_missing",
        "sex_missing",
        "site_missing",
        "sex_unknown",
        "site_unknown",
    ]
    feature_cols_cat = ["sex", "anatom_site_general_challenge", "patient_id"]

    X_train = train_fe[feature_cols_num + feature_cols_cat].copy()
    X_test = test_fe[feature_cols_num + feature_cols_cat].copy()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, feature_cols_num),
            ("cat", categorical_transformer, feature_cols_cat),
        ],
        remainder="drop",
    )

    groups = train_df["patient_id"].astype(str).values
    try:
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        splits = cv.split(X_train, y, groups=groups)
        print("Using StratifiedGroupKFold with patient_id groups.")
    except Exception as e:
        print(
            "Could not use StratifiedGroupKFold; falling back to StratifiedKFold. Reason:",
            repr(e),
        )
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        splits = cv.split(X_train, y)

    test_pred = np.zeros(len(X_test), dtype=float)

    for fold, (tr_idx, va_idx) in enumerate(splits, 1):
        X_tr, y_tr = X_train.iloc[tr_idx], y[tr_idx]
        clf = LogisticRegression(
            solver="lbfgs",
            max_iter=400,
            n_jobs=None,
            class_weight="balanced",
            C=0.5,
            random_state=42,
        )
        model = Pipeline(steps=[("preprocess", preprocessor), ("clf", clf)])
        model.fit(X_tr, y_tr)
        test_pred += model.predict_proba(X_test)[:, 1] / 5.0

    predictions = test_pred.reshape(-1, 1)
    print("Fallback metadata model predictions computed:", predictions.shape)
else:
    print("Using ensemble predictions; fallback model not used.")



## === cell 6
submission = sample_sub.copy()
submission["target"] = np.asarray(predictions).reshape(-1).astype(float)
submission["target"] = submission["target"].clip(0.0, 1.0)
submission = submission[["image_name", "target"]]

out_path = "submission.csv"
submission.to_csv(out_path, index=False)
print("Wrote:", out_path, "shape:", submission.shape)
print(submission.head())
