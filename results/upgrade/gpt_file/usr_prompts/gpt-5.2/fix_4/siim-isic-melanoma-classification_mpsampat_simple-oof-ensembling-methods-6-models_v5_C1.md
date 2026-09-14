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

bayesian-optimization==3.1.0
geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
scipy==1.15.3
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

0.9413178099846004

# 6. Current score

0.67415

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.75687) has done: 'I fix the pipeline so it always runs end-to-end even when no external `oof.csv/submission.csv` prediction folders exist in `/kaggle/input`. The main bug is that your code hard-fails when it can’t find precomputed predictions, which also causes downstream NaNs and AUC computation errors. I add a minimal fallback that trains a simple scikit-learn model on the provided metadata (same target semantics) and produces `submission.csv` in the required format, while keeping your ensembling logic intact when predictions are available. I also harden merges/AUC steps by ensuring predictions are finite and filling any remaining missing values deterministically.'
- What this solution (achieved 0.67415) has done: 'Your current score (0.75687) is far below the target (0.9413), so we should improve the fallback path that’s actually generating your submission. The biggest issue is that the fallback model is dominated by `patient_id` one-hot leakage-like memorization that does not generalize to test; removing `patient_id` from features typically boosts public AUC for this competition’s metadata baselines. With minimal changes, I switch the fallback to a stronger but still simple scikit-learn model (HistGradientBoostingClassifier) using only `sex`, `age_approx`, and `anatom_site_general_challenge`, and I also apply a tiny isotonic calibration (fit on train) to improve AUC via better probability ranking. Everything else (external prediction discovery + ensembling + BO) is kept intact, and the script still always writes a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import glob
import numpy as np
import pandas as pd

from sklearn import metrics
from bayes_opt import BayesianOptimization

np.random.seed(42)

BASE_COMP_PATH = "/kaggle/input/siim-isic-melanoma-classification"
train = pd.read_csv(os.path.join(BASE_COMP_PATH, "train.csv"))
test = pd.read_csv(os.path.join(BASE_COMP_PATH, "test.csv"))
sub = pd.read_csv(os.path.join(BASE_COMP_PATH, "sample_submission.csv"))



## === cell 1
models = [
    "384-E6-with-2018",
    "512-E6",
    "768-E2",
    "512-E5",
    "effb2-fulldata-upsample",
    "effb1-fulldata-upsample",
]


def discover_prediction_dirs(requested_models):
    candidates = []
    for p in glob.glob("/kaggle/input/**/oof.csv", recursive=True):
        d = os.path.dirname(p)
        sub_path = os.path.join(d, "submission.csv")
        if os.path.exists(sub_path):
            candidates.append(d)

    by_name = {}
    for d in candidates:
        name = os.path.basename(d)
        by_name[name] = d

    found = []
    for m in requested_models:
        if m in by_name:
            found.append((m, by_name[m]))

    if len(found) == 0 and len(candidates) > 0:
        candidates_sorted = sorted(candidates)
        found = [(os.path.basename(d), d) for d in candidates_sorted[:6]]

    return found


found_models = discover_prediction_dirs(models)

print(f"Discovered {len(found_models)} usable model prediction folders.")
for name, d in found_models:
    print(f" - {name}: {d}")

if len(found_models) == 0:
    print(
        "WARNING: No usable prediction folders found under /kaggle/input containing both oof.csv and submission.csv.\n"
        "Falling back to a metadata-only model trained from train.csv."
    )



## === cell 2
usable_model_names = []

if len(found_models) > 0:
    for model_name, dirname in found_models:
        oof_path = os.path.join(dirname, "oof.csv")
        sub_path = os.path.join(dirname, "submission.csv")

        _oof = pd.read_csv(oof_path)
        if (
            "pred" not in _oof.columns
            or "target" not in _oof.columns
            or "image_name" not in _oof.columns
        ):
            print(f"Skipping {model_name}: oof.csv missing required columns.")
            continue

        try:
            oof_pred = pd.to_numeric(_oof["pred"], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )
            if oof_pred.isna().any():
                print(f"Skipping {model_name}: oof.csv has non-finite predictions.")
                continue

            score = metrics.roc_auc_score(_oof["target"], oof_pred)
            print(f"{model_name}: OOF auc:{score:.6f}")
        except Exception as e:
            print(f"Skipping {model_name}: failed to compute AUC due to {e}")
            continue

        _oof = _oof.rename(columns={"pred": model_name}).drop(["target"], axis=1)
        if "fold" in _oof.columns:
            _oof = _oof.drop(["fold"], axis=1)

        if _oof["image_name"].duplicated().any():
            print(f"Skipping {model_name}: oof.csv has duplicate image_name values.")
            continue

        train = train.merge(_oof, on="image_name", how="left")

        _sub = pd.read_csv(sub_path)
        if _sub.shape[1] >= 2:
            _sub = _sub.iloc[:, :2].copy()
        _sub.columns = ["image_name", model_name]

        if _sub["image_name"].duplicated().any():
            print(
                f"Skipping {model_name}: submission.csv has duplicate image_name values."
            )
            train = train.drop(columns=[model_name])
            continue

        test = test.merge(_sub, on="image_name", how="left")

        for df_ in (train, test):
            df_[model_name] = pd.to_numeric(df_[model_name], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )

        if (
            train[model_name].isna().mean() > 0.01
            or test[model_name].isna().mean() > 0.01
        ):
            print(f"Skipping {model_name}: too many missing predictions after merge.")
            train = train.drop(columns=[model_name])
            test = test.drop(columns=[model_name])
            continue

        usable_model_names.append(model_name)

print(f"Using {len(usable_model_names)} models for ensembling: {usable_model_names}")



## === cell 3
train.head()



## === cell 4
models = usable_model_names  # keep later cells consistent

if len(models) > 0:
    train["pred_rank"] = 0.0
    train["pred_power"] = 0.0
    train["pred_avg"] = 0.0

    for c in models:
        r = train[c].rank()
        rmax = r.max() if r.max() != 0 else 1.0
        train["pred_rank"] += r / rmax

        p2 = np.power(train[c].to_numpy(dtype=float), 2)
        p2max = p2.max() if p2.max() != 0 else 1.0
        train["pred_power"] += p2 / p2max

        vmax = train[c].max() if train[c].max() != 0 else 1.0
        train["pred_avg"] += train[c] / vmax

    train["pred_rank"] /= len(models)
    train["pred_power"] /= len(models)
    train["pred_avg"] /= len(models)

    for col in ["pred_avg", "pred_rank", "pred_power"]:
        train[col] = pd.to_numeric(train[col], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        if train[col].isna().any():
            train[col] = train[col].fillna(train[col].mean())

    print(
        f"OOF avg_auc:{metrics.roc_auc_score(train['target'], train['pred_avg']):.6f}"
    )
    print(
        f"OOF rank_auc:{metrics.roc_auc_score(train['target'], train['pred_rank']):.6f}"
    )
    print(
        f"OOF pow_auc:{metrics.roc_auc_score(train['target'], train['pred_power']):.6f}"
    )
else:
    print(
        "No external prediction columns available; skipping ensemble feature creation."
    )



## === cell 5
if len(models) > 0:
    test_rank = test.copy()
    test_rank["target"] = 0.0
    for c in models:
        r = test_rank[c].rank()
        rmax = r.max() if r.max() != 0 else 1.0
        test_rank["target"] += r / rmax
    test_rank["target"] /= len(models)

    sub_rank = test_rank[["image_name", "target"]]
    sub_rank.to_csv("submission_rank.csv", index=False)
    print(sub_rank.head())
else:
    print("Skipping submission_rank.csv (no external model predictions).")




## === cell 6
def dim_optimizer(df_oof, features, init_points=20, n_iter=30):
    pbounds = {f"c{i}": (0.0, 1.0) for i in range(len(features))}

    def q(**params):
        x = np.zeros(len(df_oof), dtype=float)
        for i, f in enumerate(features):
            x += params[f"c{i}"] * df_oof[f].to_numpy(dtype=float)
        return metrics.roc_auc_score(df_oof["target"], x)

    optimizer = BayesianOptimization(
        f=q,
        pbounds=pbounds,
        random_state=42,
    )

    optimizer.maximize(init_points=init_points, n_iter=n_iter)

    best_params = optimizer.max["params"]
    best_auc = optimizer.max["target"]
    coeffs = [best_params[f"c{i}"] for i in range(len(features))]

    msg = "bo auc:{:.6f}, ".format(best_auc) + ", ".join(
        [f"c{i}:{coeffs[i]:.6f}" for i in range(len(coeffs))]
    )
    print(msg)

    return coeffs, best_auc


coeffs, bo_auc = None, None
if len(models) > 0:
    try:
        coeffs, bo_auc = dim_optimizer(train, models, init_points=20, n_iter=20)
    except Exception as e:
        print(f"Bayesian optimization failed due to: {e}")
        coeffs, bo_auc = None, None
else:
    print("Skipping Bayesian optimization (no external model predictions).")




## === cell 7
def bo_pred(df, features, coeffs):
    x = np.zeros(len(df), dtype=float)
    for i, f in enumerate(features):
        x += coeffs[i] * df[f].to_numpy(dtype=float)
    return x


if len(models) > 0:
    if coeffs is not None:
        train["pred_bo"] = bo_pred(train, models, coeffs)
        train["pred_bo"] = pd.to_numeric(train["pred_bo"], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        if train["pred_bo"].isna().any():
            train["pred_bo"] = train["pred_bo"].fillna(train["pred_bo"].mean())
        print(f"auc bo:{metrics.roc_auc_score(train['target'], train['pred_bo']):.6f}")
    else:
        train["pred_bo"] = train["pred_avg"]
        train["pred_bo"] = pd.to_numeric(train["pred_bo"], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        if train["pred_bo"].isna().any():
            train["pred_bo"] = train["pred_bo"].fillna(train["pred_bo"].mean())
        print(
            f"auc fallback(avg):{metrics.roc_auc_score(train['target'], train['pred_bo']):.6f}"
        )
else:
    print(
        "No ensemble predictions to evaluate; will train metadata-only model for submission."
    )




## === cell 8
def train_metadata_fallback(train_df, test_df, random_state=42):
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.impute import SimpleImputer
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.isotonic import IsotonicRegression

    feature_cols = ["sex", "age_approx", "anatom_site_general_challenge"]
    X_train = train_df[feature_cols].copy()
    y_train = train_df["target"].astype(int).copy()
    X_test = test_df[feature_cols].copy()

    numeric_features = ["age_approx"]
    categorical_features = ["sex", "anatom_site_general_challenge"]

    pre = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([("imp", SimpleImputer(strategy="median"))]),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="most_frequent")),
                        ("ohe", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ],
        remainder="drop",
    )

    clf = HistGradientBoostingClassifier(
        random_state=random_state,
        learning_rate=0.05,
        max_depth=3,
        max_iter=200,
    )

    pipe = Pipeline([("pre", pre), ("clf", clf)])
    pipe.fit(X_train, y_train)

    p_train = pipe.predict_proba(X_train)[:, 1]
    p_test = pipe.predict_proba(X_test)[:, 1]

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_train, y_train)
    p_test_cal = iso.transform(p_test)

    return p_test_cal


if len(models) > 0:
    if coeffs is not None:
        test["target"] = bo_pred(test, models, coeffs)
    else:
        test["target"] = 0.0
        for c in models:
            vmax = test[c].max() if test[c].max() != 0 else 1.0
            test["target"] += test[c] / vmax
        test["target"] /= len(models)
else:
    test["target"] = train_metadata_fallback(train, test, random_state=42)

test["target"] = pd.to_numeric(test["target"], errors="coerce").replace(
    [np.inf, -np.inf], np.nan
)
if test["target"].isna().any():
    test["target"] = test["target"].fillna(test["target"].mean())
test["target"] = test["target"].clip(0.0, 1.0)

sub_out = sub[["image_name"]].merge(
    test[["image_name", "target"]], on="image_name", how="left"
)
if sub_out["target"].isna().any():
    sub_out["target"] = sub_out["target"].fillna(sub_out["target"].mean())
sub_out["target"] = sub_out["target"].clip(0.0, 1.0)

sub_out.to_csv("submission.csv", index=False)
print(sub_out.head())
print(
    f"Wrote submission.csv with shape {sub_out.shape} and columns {list(sub_out.columns)}"
)
print(
    f"target stats: min={sub_out['target'].min():.6f}, max={sub_out['target'].max():.6f}, mean={sub_out['target'].mean():.6f}"
)
