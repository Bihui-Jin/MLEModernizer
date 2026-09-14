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

0.942396872030668

# 6. Current score

0.73577

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.67969) has done: 'Your notebook fails because it relies on an `../input/pseudolabelmodels/` dataset that does not exist in this environment, so no CSVs are found and all downstream variables are undefined. I make the smallest change that preserves the ensembling “core logic” (rank-averaging submissions) by instead using the provided competition files and generating a simple, deterministic metadata-based prediction when no external model CSVs are available. This guarantees the notebook runs end-to-end and writes a valid `submission.csv` with the required columns and row order. Since there is no current score, the aim is to produce a reasonable baseline AUC (not necessarily near the target) without changing the intent of the pipeline.'
- What this solution (achieved 0.72202) has done: 'Your current pipeline falls back to a single deterministic metadata baseline because no external pseudo-label model CSVs exist, which caps AUC around your current score. To move toward the target with minimal change and without altering the “ensemble over model outputs” core logic, I keep the rank-averaging submission mechanism but add additional legitimate “model columns” derived from metadata using (1) a train-fitted logistic regression and (2) a train-fitted random forest, both trained only on train.csv metadata and then applied to test.csv. These extra diverse predictors give the rank-ensemble more signal while preserving the same downstream ensembling semantics. I also ensure strict row alignment to `sample_submission.csv` and keep the final `submission.csv` format unchanged.'
- What this solution (achieved 0.73577) has done: 'Your current score (0.72202) is far below the target (0.9424), so we should cautiously add signal while keeping your ensemble core logic (generate multiple prediction vectors → rank-average → write submission) intact. The biggest safe gain with minimal conceptual change is to add one more metadata-derived “model column” that captures non-linear interactions without changing the overall pipeline: a GradientBoostingClassifier on the same preprocessed metadata. I also make the existing LR slightly more class-imbalance aware (class_weight="balanced"), which typically improves AUC for this dataset without changing semantics. Finally, I keep strict row alignment to `sample_submission.csv` and only add these extra predictors into the same rank-averaging ensemble.'

# 9. Code solution

## === cell 0
import os
import glob
import numpy as np
import pandas as pd



## === cell 1
CANDIDATE_INPUTS = [
    "/kaggle/data",
    "/kaggle/input",
    "../input",
    "../input/siim-isic-melanoma-classification",
    "/kaggle/data/siim-isic-melanoma-classification",
]
BASE = None
for p in CANDIDATE_INPUTS:
    if os.path.exists(p):
        if os.path.exists(os.path.join(p, "train.csv")) and os.path.exists(
            os.path.join(p, "test.csv")
        ):
            BASE = p
            break

if BASE is None:
    for p in CANDIDATE_INPUTS:
        if os.path.exists(
            os.path.join(p, "siim-isic-melanoma-classification", "train.csv")
        ):
            BASE = os.path.join(p, "siim-isic-melanoma-classification")
            break

if BASE is None:
    raise FileNotFoundError(
        "Could not locate train.csv/test.csv under expected Kaggle input paths."
    )

TRAIN_CSV = os.path.join(BASE, "train.csv")
TEST_CSV = os.path.join(BASE, "test.csv")
SAMPLE_SUB = os.path.join(BASE, "sample_submission.csv")

print("Using BASE:", BASE)
print(
    "Found files:",
    os.path.exists(TRAIN_CSV),
    os.path.exists(TEST_CSV),
    os.path.exists(SAMPLE_SUB),
)




## === cell 2
def ls(path, max_items=50):
    if not os.path.exists(path):
        print(f"{path} does not exist")
        return
    items = sorted(os.listdir(path))
    print(f"Listing {path} (showing up to {max_items}):")
    for x in items[:max_items]:
        print(" -", x)


ls("/kaggle", 50)
ls("/kaggle/data", 50)
ls("/kaggle/input", 50)
ls(BASE, 50)



## === cell 3
from scipy.stats import rankdata

LABELS = ["target"]

PSEUDO_DIR_CANDIDATES = [
    os.path.join("/kaggle/input", "pseudolabelmodels"),
    os.path.join("/kaggle/data", "pseudolabelmodels"),
    "../input/pseudolabelmodels",
]
PSEUDO_DIR = next((p for p in PSEUDO_DIR_CANDIDATES if os.path.exists(p)), None)

all_files = []
if PSEUDO_DIR is not None:
    all_files = glob.glob(os.path.join(PSEUDO_DIR, "*.csv"))

print("PSEUDO_DIR:", PSEUDO_DIR)
print("Found pseudo CSV files:", len(all_files))



## === cell 4
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

test_df = pd.read_csv(TEST_CSV)
train_df = pd.read_csv(TRAIN_CSV)
sub_df = pd.read_csv(SAMPLE_SUB)

if "image_name" not in sub_df.columns or "target" not in sub_df.columns:
    raise ValueError(
        "sample_submission.csv does not have required columns: image_name,target"
    )

test_df = test_df.merge(
    sub_df[["image_name"]], on="image_name", how="right", validate="one_to_one"
)


def _metadata_baseline_predict(df: pd.DataFrame) -> np.ndarray:
    """
    Simple deterministic baseline using metadata only.
    Produces probabilities in (0,1), aligned with melanoma prevalence trends.
    """
    d = df.copy()

    age = pd.to_numeric(
        d.get("age_approx", pd.Series(np.nan, index=d.index)), errors="coerce"
    )
    age_med = (
        float(np.nanmedian(age.values))
        if np.isfinite(np.nanmedian(age.values))
        else 50.0
    )
    age = age.fillna(age_med).clip(0, 100)
    age_z = (age - 50.0) / 15.0

    sex = d.get("sex", pd.Series("", index=d.index)).fillna("").astype(str).str.lower()
    sex_m = (sex == "male").astype(float)
    sex_f = (sex == "female").astype(float)

    site = (
        d.get("anatom_site_general_challenge", pd.Series("", index=d.index))
        .fillna("")
        .astype(str)
        .str.lower()
    )
    site_w = pd.Series(0.0, index=d.index)
    site_w += site.str.contains("head|neck").astype(float) * 0.15
    site_w += site.str.contains("upper extremity").astype(float) * 0.05
    site_w += site.str.contains("lower extremity").astype(float) * 0.03
    site_w += site.str.contains("torso").astype(float) * 0.08
    site_w += site.str.contains("palms|soles").astype(float) * 0.10

    logit = -3.98 + 0.55 * age_z + 0.20 * sex_m - 0.05 * sex_f + site_w
    prob = 1.0 / (1.0 + np.exp(-logit))
    return prob.astype(np.float32).values.reshape(-1, 1)


def _train_metadata_models_and_predict(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> dict:
    """
    Minimal improvement toward the target AUC:
    - Keep the same metadata feature set and same ensemble core logic.
    - Add one extra diverse metadata model (GradientBoostingClassifier).
    - Make LR class-imbalance aware via class_weight='balanced' (often improves AUC).
    """
    feat_cols = ["sex", "age_approx", "anatom_site_general_challenge", "patient_id"]
    X_train = train_df[feat_cols].copy()
    y_train = train_df["target"].astype(int).values
    X_test = test_df[feat_cols].copy()

    numeric_features = ["age_approx"]
    categorical_features = ["sex", "anatom_site_general_challenge", "patient_id"]

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
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

    lr = Pipeline(
        steps=[
            ("pre", preprocessor),
            (
                "clf",
                LogisticRegression(
                    max_iter=300,
                    solver="lbfgs",
                    n_jobs=None,
                    class_weight="balanced",
                ),
            ),
        ]
    )
    lr.fit(X_train, y_train)
    lr_pred = lr.predict_proba(X_test)[:, 1].astype(np.float32)

    rf = Pipeline(
        steps=[
            ("pre", preprocessor),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=250,
                    random_state=0,
                    n_jobs=-1,
                    min_samples_leaf=20,
                ),
            ),
        ]
    )
    rf.fit(X_train, y_train)
    rf_pred = rf.predict_proba(X_test)[:, 1].astype(np.float32)

    gbdt = Pipeline(
        steps=[
            ("pre", preprocessor),
            (
                "clf",
                GradientBoostingClassifier(
                    random_state=0,
                    n_estimators=120,
                    learning_rate=0.08,
                    max_depth=3,
                ),
            ),
        ]
    )
    gbdt.fit(X_train, y_train)
    gbdt_pred = gbdt.predict_proba(X_test)[:, 1].astype(np.float32)

    return {"meta_lr": lr_pred, "meta_rf": rf_pred, "meta_gbdt": gbdt_pred}


meta_models_pred = _train_metadata_models_and_predict(train_df, test_df)

if len(all_files) > 0:
    outs = []
    for f in all_files:
        df = pd.read_csv(f)
        if "image_name" in df.columns and "target" in df.columns:
            df = df[["image_name", "target"]].copy()
            df = df.merge(
                sub_df[["image_name"]],
                on="image_name",
                how="right",
                validate="one_to_one",
            )
            outs.append(
                df.set_index("image_name")[["target"]].rename(
                    columns={"target": os.path.basename(f)}
                )
            )
        else:
            df2 = pd.read_csv(f, index_col=0)
            if "target" in df2.columns:
                df2 = df2.loc[sub_df["image_name"].values]
                outs.append(
                    df2[["target"]].rename(columns={"target": os.path.basename(f)})
                )
    if len(outs) == 0:
        concat_sub = pd.DataFrame(
            {
                "image_name": sub_df["image_name"].values,
                "m0": _metadata_baseline_predict(test_df)[:, 0],
                "m1": meta_models_pred["meta_lr"],
                "m2": meta_models_pred["meta_rf"],
                "m3": meta_models_pred["meta_gbdt"],
            }
        )
    else:
        concat_sub = pd.concat(outs, axis=1)
        concat_sub.reset_index(inplace=True)
        concat_sub.rename(columns={"index": "image_name"}, inplace=True)
        model_cols = [c for c in concat_sub.columns if c != "image_name"]
        new_cols = ["m" + str(i) for i in range(len(model_cols))]
        rename_map = {old: new for old, new in zip(model_cols, new_cols)}
        concat_sub.rename(columns=rename_map, inplace=True)

        concat_sub = concat_sub.merge(
            sub_df[["image_name"]], on="image_name", how="right", validate="one_to_one"
        )
        concat_sub["m_meta_lr"] = meta_models_pred["meta_lr"]
        concat_sub["m_meta_rf"] = meta_models_pred["meta_rf"]
        concat_sub["m_meta_gbdt"] = meta_models_pred["meta_gbdt"]
else:
    base_pred = _metadata_baseline_predict(test_df)[:, 0]
    concat_sub = pd.DataFrame(
        {
            "image_name": sub_df["image_name"].values,
            "m0": base_pred,
            "m1": meta_models_pred["meta_lr"],
            "m2": meta_models_pred["meta_rf"],
            "m3": meta_models_pred["meta_gbdt"],
        }
    )

for c in concat_sub.columns:
    if c != "image_name":
        concat_sub[c] = (
            pd.to_numeric(concat_sub[c], errors="coerce")
            .fillna(0.5)
            .clip(1e-6, 1 - 1e-6)
        )

print("concat_sub shape:", concat_sub.shape)
print("concat_sub head:\n", concat_sub.head())



## === cell 5
import warnings

warnings.filterwarnings("ignore")

model_mat = concat_sub.iloc[:, 1:]  # exclude image_name
if model_mat.shape[1] == 1:
    m_gmean = model_mat.iloc[:, 0].values
else:
    rank = np.tril(model_mat.corr().values, -1)
    m = (rank > 0).sum()
    m_gmean_acc, s = 0.0, 0.0
    for n in range(min(rank.shape[0], m)):
        mx = np.unravel_index(rank.argmin(), rank.shape)
        w = (m - n) / (m + n / 10)
        m_gmean_acc += (
            w
            * (
                np.log(model_mat.iloc[:, mx[0]].values)
                + np.log(model_mat.iloc[:, mx[1]].values)
            )
            / 2.0
        )
        s += w
        rank[mx] = 1
    m_gmean = np.exp(m_gmean_acc / s).clip(0.0, 1.0)

m_gmean = np.asarray(m_gmean).reshape(-1, 1)
print(
    "m_gmean stats:", float(m_gmean.min()), float(m_gmean.max()), float(m_gmean.mean())
)



## === cell 6
predict_list = []
for c in concat_sub.columns:
    if c != "image_name":
        predict_list.append(concat_sub[[c]].values.astype(np.float32))

predict_list_2 = list(predict_list)

print("predict_list models:", len(predict_list))
print("predict_list_2 models:", len(predict_list_2))



## === cell 7
import warnings

warnings.filterwarnings("ignore")

print("Rank averaging on ", len(predict_list), " files")
if len(predict_list) == 0:
    raise ValueError("No predictions available to ensemble.")

predictions = np.zeros_like(predict_list[0], dtype=np.float32)
for predict in predict_list:
    predictions[:, 0] = predictions[:, 0] + (
        rankdata(predict[:, 0]) / predictions.shape[0]
    ).astype(np.float32)

predictions = predictions / float(len(predict_list))

print(
    "predictions stats:",
    float(predictions.min()),
    float(predictions.max()),
    float(predictions.mean()),
)



## === cell 8
import warnings

warnings.filterwarnings("ignore")

print("Rank averaging on ", len(predict_list_2), " files")
if len(predict_list_2) == 0:
    raise ValueError("No predictions available to ensemble (predict_list_2).")

predictions_2 = np.zeros_like(predict_list_2[0], dtype=np.float32)
for predict_2 in predict_list_2:
    predictions_2[:, 0] = predictions_2[:, 0] + (
        rankdata(predict_2[:, 0]) / predictions_2.shape[0]
    ).astype(np.float32)

predictions_2 = predictions_2 / float(len(predict_list_2))

print(
    "predictions_2 stats:",
    float(predictions_2.min()),
    float(predictions_2.max()),
    float(predictions_2.mean()),
)



## === cell 9
submission_1 = pd.read_csv(SAMPLE_SUB)
submission_1[LABELS] = predictions

submission_2 = pd.read_csv(SAMPLE_SUB)
submission_2[LABELS] = predictions_2

submission = pd.read_csv(SAMPLE_SUB)
submission[LABELS] = submission_2[LABELS] * 0.55 + submission_1[LABELS] * 0.45

submission["target"] = submission["target"].astype(float).clip(0.0, 1.0)

out_path = "submission.csv"
submission.to_csv(out_path, index=False)

print("Wrote:", out_path)
print(submission.head())
print("Rows:", len(submission), "Cols:", list(submission.columns))
