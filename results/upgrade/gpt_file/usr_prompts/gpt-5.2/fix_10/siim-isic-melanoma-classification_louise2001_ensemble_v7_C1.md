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

0.9015931091476356

# 6. Current score

0.77386

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'I fix the hard failure caused by referencing a non-existent input folder (`/kaggle/input/melanoma`) by removing/guarding the post-processing cells that try to read from it, since they are not needed to generate a valid submission. I also make the ensemble-merge logic robust when the optional folder doesn’t exist, so the notebook still runs end-to-end and always writes `submission.csv` with the required `image_name,target` columns. Finally, I ensure the target column computation works even when no extra prediction files are found (fallback to a constant 0.5) and keep the rest of the logic unchanged.'
- What this solution (achieved 0.66776) has done: 'Your current 0.5 AUC comes from outputting a constant 0.5 for every test row when the optional ensemble folder is missing, so we need to generate a non-constant probability using only the provided train/test metadata. To keep the core “simple tabular prediction” logic minimal and fast, I fit a lightweight scikit-learn logistic regression on the metadata (sex, age, site) with one-hot encoding and median imputation, then use its predicted probabilities for the submission. This directly targets ROC-AUC improvement without changing any image/model pipeline (since there isn’t one here) and still fall back safely to 0.5 if something unexpected happens. The rest of your optional ensemble-merge/error-analysis cells are preserved and guarded exactly as before so the notebook always writes a valid `submission.csv`.'
- What this solution (achieved 0.6672) has done: 'Your current approach is a simple metadata-only model, so the safest way to move AUC upward toward the target is to keep the same feature set and logistic regression, but make the prediction less noisy and better calibrated by reducing variance and using a patient-group-aware validation for out-of-fold stacking (still logistic regression, same loss/semantics). I add out-of-fold predicted probabilities for the training set using `GroupKFold` on `patient_id`, then fit a second logistic regression on that single meta-feature (“stacking”), which often improves ROC-AUC with minimal logic change and low risk of leakage. I also set `random_state` where applicable for stability and switch the solver to `liblinear` (more robust on small/medium sparse one-hot), keeping everything else the same. The submission writing stays identical (`submission.csv` with `image_name,target`) and all optional ensemble cells remain guarded.'
- What this solution (achieved 0.66479) has done: 'Your current score (0.6672) is far below the target (0.9016), so we should improve AUC while keeping the same metadata-only logistic-regression core. The main minimal upgrade is to add the strong, competition-standard `patient_id` leakage-aware feature: out-of-fold (GroupKFold) patient-level target mean encoding, then include it as an additional numeric feature alongside the existing sex/age/site preprocessing and logistic regression. This preserves the same model family (logistic regression), same evaluation semantics (probabilities for ROC-AUC), and stays fast. I also keep your existing optional ensemble-folder logic untouched and still always write a valid `submission.csv`.'
- What this solution (achieved 0.66485) has done: 'I keep your metadata-only logistic-regression + stacking core intact and make two small, score-relevant fixes: (1) prevent patient target-mean leakage into the test set by computing `patient_target_mean` for test using only out-of-fold fold maps (so the model can’t “peek” at a test patient’s label distribution via the full training set), and (2) improve the stacker’s calibration by fitting it on logit-transformed OOF probabilities (still logistic regression, same semantics) rather than raw probabilities, which typically helps AUC a bit with minimal change. I also make `age_approx` explicitly numeric to avoid silent object-type issues that can reduce model quality. Submission writing and all optional ensemble-merge/error-analysis behavior remain unchanged, and the script still always writes `submission.csv`.'
- What this solution (achieved 0.77389) has done: 'Your score gap to the target is large (0.66485 → 0.90159), so we need a real AUC lift while keeping your metadata-only logistic-regression core intact. The most effective minimal change here is to add one competition-standard categorical feature (`patient_id`) with proper out-of-fold target mean encoding (to avoid leakage) and a small amount of smoothing, then feed it into the same preprocessing + logistic regression + logit-stacking pipeline you already use. This preserves your model family, training loop style, and probability semantics, but typically improves ranking signal substantially versus sex/age/site alone. I also keep the optional ensemble-folder logic and still always write a valid `submission.csv`.'
- What this solution (achieved 0.77388) has done: 'We keep your metadata-only logistic regression + logit-stacking core exactly the same, but fix a subtle leakage/weak-signal issue in the `patient_target_mean` feature computation: it currently uses the global target mean from the full dataset inside each fold, which slightly leaks fold information and can reduce generalization. We recompute `global_mean` inside each fold using only the fold’s training portion, and we also compute the smoothed mean consistently from the same fold-only data. This is a minimal change (no new model types, no new features, no tuning loops) and is directly aimed at improving ROC-AUC toward your target by making the patient encoding more faithful out-of-fold. Submission writing remains unchanged and still produces `submission.csv`.'
- What this solution (achieved 0.77241) has done: 'Your current score (0.77388) is well below the target (0.90159), so we should cautiously improve AUC while keeping the same metadata-only logistic regression + patient target-mean encoding + logit-stacking core. The smallest high-impact change is to make the base model’s out-of-fold feature (`patient_target_mean`) consistent with the folds used in `cross_val_predict`: we precompute fold indices once, reuse them for both the target-mean encoding and OOF predictions, and thus remove a subtle mismatch that can degrade stacking/generalization. I also add a tiny amount of smoothing to the per-fold `patient_id` one-hot expansion by setting `min_frequency` in `OneHotEncoder`, which reduces extreme sparsity/overfitting without changing the model family or training approach. Submission writing remains identical and still produces `submission.csv`.'
- What this solution (achieved 0.77386) has done: 'Your current AUC (0.77241) is far below the target (0.90159), so we need a cautious, minimal upgrade that adds ranking signal without changing the model family or training semantics. I keep the same metadata-only logistic regression + patient target-mean encoding + logit-stacking pipeline, but fix two score-relevant weaknesses: (1) the `patient_target_mean` uses only per-fold maps, which underutilizes stable patient priors at inference, and (2) rare patient IDs are being grouped by `min_frequency`, reducing the benefit of the patient feature. Concretely, I compute a leakage-safe test-time patient mean by averaging fold-specific encodings and also add a stable “full-train smoothed patient mean” feature (OOF for train, full-map for test), then include both numeric patient-mean features in the same logistic regression pipeline. This stays fast, preserves the core approach, and should move AUC upward toward your target while still producing `submission.csv` exactly as required.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict

SAMPLE_SUB_PATH = (
    "/kaggle/input/siim-isic-melanoma-classification/sample_submission.csv"
)
TRAIN_CSV_PATH = "/kaggle/input/siim-isic-melanoma-classification/train.csv"
TEST_CSV_PATH = "/kaggle/input/siim-isic-melanoma-classification/test.csv"

f = pd.read_csv(SAMPLE_SUB_PATH)[["image_name"]]
print(f.shape)

ENSEMBLE_DIR = (
    "/kaggle/input/melanoma"  # optional directory containing other submissions
)
cols = {}

if os.path.isdir(ENSEMBLE_DIR):
    for dirname, _, filenames in os.walk(ENSEMBLE_DIR):
        for i, filename in enumerate(sorted(filenames)):
            if not filename.lower().endswith(".csv"):
                continue
            cols[filename] = f"target_{i}"
            ff = pd.read_csv(os.path.join(dirname, filename))
            ff = ff.iloc[:, :2].copy()
            ff.columns = ["image_name", f"target_{i}"]
            f = f.merge(ff, on="image_name", how="left")
else:
    print(
        f"Optional ensemble dir not found: {ENSEMBLE_DIR}. Proceeding without extra predictions."
    )



## === cell 1
f.head()



## === cell 2
print(f.shape)



## === cell 3
f.head()



## === cell 4
pred_cols = list(cols.values())

if len(pred_cols) > 0:
    f["target"] = f[pred_cols].mean(axis=1)
else:
    try:
        train_df = pd.read_csv(TRAIN_CSV_PATH)
        test_df = pd.read_csv(TEST_CSV_PATH)

        feature_cols_base = ["sex", "age_approx", "anatom_site_general_challenge"]
        X_train = train_df[feature_cols_base].copy()
        y_train = train_df["target"].astype(int).values
        X_test = test_df[feature_cols_base].copy()

        X_train["age_approx"] = pd.to_numeric(X_train["age_approx"], errors="coerce")
        X_test["age_approx"] = pd.to_numeric(X_test["age_approx"], errors="coerce")

        train_pat = train_df["patient_id"].astype(str).fillna("NA")
        test_pat = test_df["patient_id"].astype(str).fillna("NA")

        n_splits = 5
        gkf = GroupKFold(n_splits=n_splits)

        folds = list(gkf.split(train_df, y_train, groups=train_pat.values))

        smoothing_m = 20.0  # fixed; no tuning

        oof_pat_mean = np.zeros(len(train_df), dtype=float)
        test_pat_mean_folds = np.zeros((len(test_df), n_splits), dtype=float)

        oof_pat_mean_fullmap = np.zeros(len(train_df), dtype=float)

        for fold, (tr_idx, va_idx) in enumerate(folds):
            fold_global_mean = float(np.mean(y_train[tr_idx]))

            tr_pat = train_pat.iloc[tr_idx]
            tr_y = pd.Series(y_train[tr_idx], index=train_df.index[tr_idx])

            grp = tr_y.groupby(tr_pat)
            pat_mean = grp.mean()
            pat_cnt = grp.size().astype(float)

            pat_smoothed = (pat_cnt * pat_mean + smoothing_m * fold_global_mean) / (
                pat_cnt + smoothing_m
            )

            oof_pat_mean[va_idx] = (
                train_pat.iloc[va_idx].map(pat_smoothed).fillna(fold_global_mean).values
            )

            oof_pat_mean_fullmap[va_idx] = (
                train_pat.iloc[va_idx].map(pat_smoothed).fillna(fold_global_mean).values
            )

            test_pat_mean_folds[:, fold] = (
                test_pat.map(pat_smoothed).fillna(fold_global_mean).values.astype(float)
            )

        test_pat_mean = test_pat_mean_folds.mean(axis=1).astype(float)

        full_global_mean = float(np.mean(y_train))
        full_y = pd.Series(y_train, index=train_df.index)
        full_grp = full_y.groupby(train_pat)
        full_pat_mean = full_grp.mean()
        full_pat_cnt = full_grp.size().astype(float)
        full_pat_smoothed = (
            full_pat_cnt * full_pat_mean + smoothing_m * full_global_mean
        ) / (full_pat_cnt + smoothing_m)
        test_pat_mean_fullmap = (
            test_pat.map(full_pat_smoothed)
            .fillna(full_global_mean)
            .values.astype(float)
        )

        X_train["patient_target_mean"] = oof_pat_mean
        X_test["patient_target_mean"] = test_pat_mean

        X_train["patient_target_mean_fullmap"] = oof_pat_mean_fullmap
        X_test["patient_target_mean_fullmap"] = test_pat_mean_fullmap

        X_train["patient_id"] = train_pat.values
        X_test["patient_id"] = test_pat.values

        numeric_features = [
            "age_approx",
            "patient_target_mean",
            "patient_target_mean_fullmap",
        ]
        categorical_features = [
            "sex",
            "anatom_site_general_challenge",
            "patient_id",
        ]

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", SimpleImputer(strategy="median"), numeric_features),
                (
                    "cat",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore")),
                        ]
                    ),
                    categorical_features,
                ),
            ],
            remainder="drop",
        )

        base_clf = LogisticRegression(
            max_iter=500,
            class_weight="balanced",
            solver="liblinear",
            random_state=42,
        )

        base_model = Pipeline(steps=[("preprocess", preprocessor), ("clf", base_clf)])

        oof_proba = cross_val_predict(
            base_model,
            X_train,
            y_train,
            cv=folds,
            method="predict_proba",
            n_jobs=None,
        )[:, 1].astype(float)

        base_model.fit(X_train, y_train)
        test_proba = base_model.predict_proba(X_test)[:, 1].astype(float)

        eps = 1e-6
        oof_proba_clip = np.clip(oof_proba, eps, 1.0 - eps)
        test_proba_clip = np.clip(test_proba, eps, 1.0 - eps)
        stack_X_train = np.log(oof_proba_clip / (1.0 - oof_proba_clip)).reshape(-1, 1)
        stack_X_test = np.log(test_proba_clip / (1.0 - test_proba_clip)).reshape(-1, 1)

        stack_clf = LogisticRegression(
            max_iter=500,
            class_weight="balanced",
            solver="lbfgs",
            random_state=42,
        )
        stack_clf.fit(stack_X_train, y_train)
        proba = stack_clf.predict_proba(stack_X_test)[:, 1].astype(float)

        pred_df = pd.DataFrame(
            {"image_name": test_df["image_name"].values, "target": proba}
        )
        f = f.merge(pred_df, on="image_name", how="left")

        f["target"] = f["target"].fillna(0.5).clip(0.0, 1.0)
    except Exception as e:
        print("Metadata model failed; falling back to constant 0.5. Error:", repr(e))
        f["target"] = 0.5

f_ = f[["image_name", "target"]].copy()
f_.head()



## === cell 5
f_.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", f_.shape)
print(f_.head())



## === cell 6
dic_errors = {}
biggest_error = []
min_vals = []

if os.path.isdir(ENSEMBLE_DIR) and len(pred_cols) > 0:
    files = [x for x in os.listdir(ENSEMBLE_DIR) if x.lower().endswith(".csv")]

    def calculate_mse(y, df):
        df = df.copy()
        for col in df.columns:
            df[col] = (df[col] - y) ** 2
        return df.sum(axis=1)

    for file in files:
        y = pd.read_csv(os.path.join(ENSEMBLE_DIR, file))["target"]
        error = calculate_mse(
            y, f.drop(columns=["image_name"], inplace=False)[pred_cols]
        )
        dic_errors[file] = error
else:
    print(
        "Skipping error-analysis cells because ensemble directory/predictions are unavailable."
    )



## === cell 7
if len(dic_errors) > 0:
    display(pd.DataFrame(dic_errors).head())
else:
    print("dic_errors is empty.")



## === cell 8
if len(dic_errors) > 0:
    biggest_error = (
        pd.DataFrame(dic_errors)
        .transpose()
        .sum(axis=1)
        .sort_values(ascending=False)
        .index.tolist()[:2]
    )
    print("biggest_error:", biggest_error)
else:
    biggest_error = []
    print("biggest_error not computed (dic_errors empty).")



## === cell 9
biggest_error



## === cell 10
if len(pred_cols) > 0 and len(biggest_error) > 0:
    f["target_wo_2"] = f[[c for k, c in cols.items() if k not in biggest_error]].mean(
        axis=1
    )
    f[["image_name", "target_wo_2"]].to_csv(
        "sub_wo_2.csv", index=False, header=["image_name", "target"]
    )
    print("Wrote sub_wo_2.csv")
else:
    print("Skipping sub_wo_2.csv (insufficient ensemble data).")



## === cell 11
if len(dic_errors) > 0 and len(pred_cols) > 0:
    min_dist = pd.DataFrame(dic_errors).idxmin(axis=1)
    min_vals = []
    for i, sub in min_dist.items():
        min_vals.append(f.loc[i, cols[sub]])
else:
    min_vals = []
    print("Skipping argmin computation (dic_errors empty).")



## === cell 12
if len(min_vals) > 0:
    f["target_arg_min"] = min_vals
    f[["image_name", "target_arg_min"]].to_csv(
        "sub_argmin.csv", index=False, header=["image_name", "target"]
    )
    print("Wrote sub_argmin.csv")
else:
    print("Skipping sub_argmin.csv (min_vals empty).")
