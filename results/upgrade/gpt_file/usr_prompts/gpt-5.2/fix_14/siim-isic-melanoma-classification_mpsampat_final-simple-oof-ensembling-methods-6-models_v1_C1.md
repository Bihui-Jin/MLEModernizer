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

0.26887

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.66789) has done: 'I remove the hard “exit” behavior when no external OOF/submission files are found and replace it with a safe metadata-only baseline model so the notebook always runs end-to-end and writes `submission.csv`. I also fix the NaN generation in the blending features by guarding against division by zero and by ensuring `models` is non-empty before computing AUC/blends. Finally, I make Bayesian Optimization optional: it run only when there is at least 1 usable model feature; otherwise it be skipped cleanly so `coeffs` is never undefined. These changes preserve the intended ensemble/blending logic when model files exist, while producing a valid submission in environments where they don’t.'
- What this solution (achieved 0.75689) has done: 'You’re far below the target AUC (0.6679 vs 0.9413), so we should improve the score with the smallest changes that keep your current “metadata-only fallback” core logic intact. The biggest gain, without changing the model family, is to add the strong `patient_id` leakage-style signal (it’s available in both train/test and is known to be very predictive in this competition) as a categorical feature in the same LogisticRegression pipeline. To avoid any accidental misalignment that could tank AUC, we also ensure the final `submission.csv` is ordered exactly like `sample_submission.csv` (same `image_name` order) before writing. Everything else (CV loop, model, preprocessing approach) stays the same.'
- What this solution (achieved 0.74637) has done: 'Your current score (0.75689) is far below the target AUC (0.9413), so we should increase performance with minimal, low-risk changes that preserve your metadata-only LogisticRegression pipeline and CV loop. The main improvement is to avoid train/validation leakage from `patient_id` by switching the CV splitter to `GroupKFold` on `patient_id` while keeping the same model/feature set; this usually improves generalization and LB AUC for this competition. To recover some of the signal lost by removing leakage, we add two safe, classic metadata features (`age_isna` and `n_images_per_patient`) without changing the model family. Finally, we keep your strict submission ordering to `sample_submission.csv` to avoid any alignment-related score drops.'
- What this solution (achieved 0.7848) has done: 'Your current AUC (0.746) is far below the target (0.941), so we should improve with the smallest safe changes that keep your metadata-only LogisticRegression + CV core intact. The biggest low-risk gain here is fixing a subtle but important leakage/shift bug: `n_images_per_patient` is currently computed separately in train vs test, which makes the same `patient_id` have inconsistent values between train and test and hurts generalization. We compute `n_images_per_patient` from the combined (train+test) `patient_id` counts so the feature is consistent across both splits, while keeping GroupKFold, the same model, and the same preprocessing. Everything else remains the same, including submission ordering to `sample_submission.csv`.'
- What this solution (achieved 0.78472) has done: 'We fix the runtime KeyError by ensuring `patient_target_rate` exists before it’s used to build `X`/`X_test` in the metadata-only fallback path (it was referenced in `features_num` before being created). This change is minimal and preserves the exact model family (LogisticRegression), CV approach (GroupKFold/StratifiedKFold), and feature engineering intent; it only reorders the feature creation to match the code’s dependencies. We also keep the submission alignment to `sample_submission.csv` so the output order is correct and stable. With this fix, the notebook run end-to-end and always write a valid `submission.csv`.'
- What this solution (achieved 0.78453) has done: 'Your current AUC (0.78472) is far below the target (0.94132), so we should make a small, low-risk improvement within the same metadata-only LogisticRegression + GroupKFold framework. The biggest issue limiting your feature strength is that `patient_target_rate` only uses `patient_id`, which is intentionally de-leaked by GroupKFold; a minimal and legitimate upgrade is to compute the OOF rate on a slightly broader, still-available grouping key (`patient_id + anatom_site`) to capture consistent per-patient/per-site tendencies without changing the model family or training loop. We keep your existing `patient_target_rate` feature for stability, add the new OOF rate feature, and also apply tiny additive smoothing to both rates to reduce noise for rare groups. Submission writing and ordering remain exactly aligned to `sample_submission.csv`.'
- What this solution (achieved 0.78483) has done: 'We’re far below the target AUC, so we should make a small, low-risk improvement inside your existing metadata-only fallback (LogisticRegression + GroupKFold + target-encoding features) rather than changing the overall approach. The simplest gain is to add one more smoothed OOF target-encoding feature based on `anatom_site_general_challenge` alone, which is a strong metadata signal and complements your existing `patient_id` and `patient_id||site` encodings. This keeps the same CV splits, same model family, same preprocessing, and the same evaluation semantics; it only adds a single engineered feature and its corresponding test-time mapping. Submission writing and ordering remain aligned to `sample_submission.csv` exactly as before.'
- What this solution (achieved 0.79191) has done: 'Your current AUC (0.78483) is still far below the target (0.94132), so we should cautiously increase performance with very small, low-risk changes inside the same metadata-only fallback (LogisticRegression + GroupKFold + smoothed OOF target-encoding). The main limitation now is that the smoothed encodings use a fixed alpha=20 regardless of group frequency; we keep the same encoding logic but add one additional, more stable signal: a smoothed “logit” transform of each target-rate (patient/site/patient_site) which often linearizes the relationship for LogisticRegression. This does not change the model family, CV, loss, or training loop; it only adds three numeric features derived from already-computed rates. Submission writing and ordering remain aligned to `sample_submission.csv`.'
- What this solution (achieved 0.79375) has done: 'Your current AUC (0.79191) is well below the target (0.9413), so we should nudge performance upward with the smallest legitimate change inside your existing metadata-only LogisticRegression + GroupKFold + smoothed target-encoding framework. The most impactful low-risk improvement here is to add one more OOF smoothed target-encoding feature for `patient_id` alone: the (log) count of images per patient computed consistently from combined train+test (you already compute the raw count; we add its log1p transform). This often helps linear models because the effect of “many images for a patient” is non-linear, and it doesn’t change the training loop, CV, model family, or loss—just one extra numeric feature derived from an existing one. We also compute the same feature for test and include it in `features_num`, keeping submission ordering aligned to `sample_submission.csv` exactly as before.'
- What this solution (achieved 0.7943) has done: 'We’re far below the target AUC, so the safest way to increase score without changing your core model/training loop is to add one more small, legitimate metadata signal that is known to help in this competition: a smoothed OOF target-encoding for `age_approx` after binning it (so LogisticRegression can use a non-linear age effect). This preserves your existing LogisticRegression + (Group)KFold CV and the same feature pipeline; it only adds one engineered categorical feature plus its corresponding OOF/test mapping. I implement the binned-age encoding with the same smoothing (alpha=20) and add it both as a rate and its logit transform (analogous to your existing encodings), then include them as numeric features. Submission writing and strict ordering to `sample_submission.csv` remain unchanged.'
- What this solution (achieved 0.26887) has done: 'We’re well below the target AUC (0.7943 vs 0.9413), so we should carefully increase performance with very small, low-risk changes that preserve your existing metadata-only LogisticRegression + GroupKFold + smoothed OOF target-encoding core. The main tweak is to add one additional, competition-relevant OOF target-encoding feature for the `diagnosis` column (train-only), computed in the same leakage-safe OOF manner and mapped to test as the global mean; this often provides a meaningful lift because diagnosis is strongly correlated with the target. We also add its logit transform, matching your existing “rate + logit” pattern, without changing the model family, CV, or training loop. Submission writing and strict ordering to `sample_submission.csv` remain unchanged.'

# 9. Code solution

## === cell 0
import os
import glob
import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

from bayes_opt import BayesianOptimization

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

DATA_DIR = "/kaggle/input/siim-isic-melanoma-classification"
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
sub_sample = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

print("train:", train.shape, "test:", test.shape, "sample:", sub_sample.shape)




## === cell 1
def discover_model_dirs(search_root="/kaggle/input"):
    model_dirs = []
    for oof_path in glob.glob(
        os.path.join(search_root, "**", "oof.csv"), recursive=True
    ):
        d = os.path.dirname(oof_path)
        sub_path = os.path.join(d, "submission.csv")
        if os.path.exists(sub_path):
            model_dirs.append(d)
    seen = set()
    uniq = []
    for d in sorted(model_dirs):
        if d not in seen:
            uniq.append(d)
            seen.add(d)
    return uniq


model_dirs = discover_model_dirs("/kaggle/input")
print(f"Discovered {len(model_dirs)} model dirs with both oof.csv and submission.csv.")
for d in model_dirs[:20]:
    print(" -", d)



## === cell 2
USE_EXTERNAL_MODELS = len(model_dirs) > 0
if not USE_EXTERNAL_MODELS:
    print("WARNING: No ensemble model inputs found under /kaggle/input.")
    print(
        "Falling back to a metadata-only LogisticRegression baseline to produce a valid submission.csv."
    )



## === cell 3
models = []
dir_to_model = {}
name_counts = {}

for d in model_dirs:
    base = os.path.basename(d.rstrip("/"))
    if base in name_counts:
        name_counts[base] += 1
        name = f"{base}__{name_counts[base]}"
    else:
        name_counts[base] = 0
        name = base
    models.append(name)
    dir_to_model[name] = d

print(f"Using {len(models)} discovered models (before validation).")
print(models[:30])



## === cell 4
kept_models = []

if USE_EXTERNAL_MODELS:
    for model in models:
        dirname = dir_to_model[model]
        oof_path = os.path.join(dirname, "oof.csv")
        sub_path = os.path.join(dirname, "submission.csv")

        try:
            _oof = pd.read_csv(oof_path)
            _sub = pd.read_csv(sub_path)
        except Exception as e:
            print(f"Skipping {model} due to read error: {e}")
            continue

        if "image_name" not in _oof.columns or "target" not in _oof.columns:
            print(f"Skipping {model}: oof.csv missing required columns.")
            continue
        if "pred" not in _oof.columns:
            alt = None
            for cand in ["oof", "prediction", "preds", "y_pred", "target_pred"]:
                if cand in _oof.columns:
                    alt = cand
                    break
            if alt is None:
                print(
                    f"Skipping {model}: oof.csv has no 'pred' (or known alternatives)."
                )
                continue
            _oof = _oof.rename(columns={alt: "pred"})

        _oof = _oof[
            ["image_name", "target", "pred"]
            + (["fold"] if "fold" in _oof.columns else [])
        ].copy()
        _oof = _oof.drop_duplicates(subset=["image_name"], keep="first")

        try:
            score = metrics.roc_auc_score(_oof["target"].values, _oof["pred"].values)
            print(f"{model}: OOF auc:{score:.6f}")
        except Exception as e:
            print(f"Skipping {model}: cannot compute AUC ({e})")
            continue

        _oof = _oof.rename(columns={"pred": model}).drop(["target"], axis=1)
        if "fold" in _oof.columns:
            _oof = _oof.drop(["fold"], axis=1)

        train = train.merge(_oof, on="image_name", how="left", validate="one_to_one")
        if train[model].isna().mean() > 0.05:
            print(
                f"Skipping {model}: too many missing after merge (NaN rate={train[model].isna().mean():.3f})."
            )
            train = train.drop(columns=[model])
            continue

        if "image_name" not in _sub.columns:
            print(f"Skipping {model}: submission.csv missing image_name.")
            train = train.drop(columns=[model])
            continue

        if "target" in _sub.columns:
            pred_col = "target"
        else:
            candidates = [c for c in _sub.columns if c != "image_name"]
            if not candidates:
                print(f"Skipping {model}: submission.csv has no prediction column.")
                train = train.drop(columns=[model])
                continue
            pred_col = candidates[0]

        _sub = _sub[["image_name", pred_col]].copy()
        _sub = _sub.rename(columns={pred_col: model})
        _sub = _sub.drop_duplicates(subset=["image_name"], keep="first")

        test = test.merge(_sub, on="image_name", how="left", validate="one_to_one")
        if test[model].isna().mean() > 0.05:
            print(
                f"Skipping {model}: too many missing in test after merge (NaN rate={test[model].isna().mean():.3f})."
            )
            test = test.drop(columns=[model])
            train = train.drop(columns=[model])
            continue

        kept_models.append(model)

    models = kept_models
    print(f"Kept {len(models)} models after validation.")
else:
    models = []




## === cell 5
def _safe_div(x, denom, eps=1e-12):
    denom = np.asarray(denom)
    return x / np.maximum(denom, eps)


if len(models) > 0:
    for c in models:
        if train[c].isna().any() or test[c].isna().any():
            med = np.nanmedian(train[c].values)
            if not np.isfinite(med):
                med = 0.5
            train[c] = train[c].fillna(med)
            test[c] = test[c].fillna(med)

train.head()



## === cell 6
if len(models) > 0:
    train["pred_rank"] = 0.0
    train["pred_power"] = 0.0
    train["pred_avg"] = 0.0

    for c in models:
        r = train[c].rank()
        train["pred_rank"] += _safe_div(r, r.max())
        p2 = np.power(train[c].values, 2)
        train["pred_power"] += _safe_div(p2, np.nanmax(p2))
        train["pred_avg"] += _safe_div(train[c].values, np.nanmax(train[c].values))

    train["pred_rank"] /= len(models)
    train["pred_power"] /= len(models)
    train["pred_avg"] /= len(models)

    for col in ["pred_rank", "pred_power", "pred_avg"]:
        train[col] = train[col].replace([np.inf, -np.inf], np.nan).fillna(0.5)

    score = metrics.roc_auc_score(train["target"], train["pred_avg"])
    print(f"OOF avg_auc:{score:.6f}")

    score = metrics.roc_auc_score(train["target"], train["pred_rank"])
    print(f"OOF rank_auc:{score:.6f}")

    score = metrics.roc_auc_score(train["target"], train["pred_power"])
    print(f"OOF pow_auc:{score:.6f}")
else:
    print("No external models available; skipping blend AUC computations.")




## === cell 7
def make_submission_from_blend(blend_name: str, series: pd.Series, out_name: str):
    df = pd.DataFrame(
        {"image_name": test["image_name"].values, "target": series.values}
    )
    df.to_csv(out_name, index=False)
    print(f"Wrote {out_name} ({blend_name}) with shape {df.shape}")
    return df


if len(models) > 0:
    test_rank = np.zeros(len(test), dtype=float)
    for c in models:
        r = test[c].rank()
        test_rank += _safe_div(r.values, r.max())
    test_rank /= len(models)
    sub_rank = make_submission_from_blend(
        "rank", pd.Series(test_rank), "submission_rank.csv"
    )

    test_pow = np.zeros(len(test), dtype=float)
    for c in models:
        p2 = np.power(test[c].values, 2)
        test_pow += _safe_div(p2, np.nanmax(p2))
    test_pow /= len(models)
    sub_pow = make_submission_from_blend(
        "power", pd.Series(test_pow), "submission_pow.csv"
    )

    test_avg = np.zeros(len(test), dtype=float)
    for c in models:
        test_avg += _safe_div(test[c].values, np.nanmax(test[c].values))
    test_avg /= len(models)
    sub_avg = make_submission_from_blend(
        "avg", pd.Series(test_avg), "submission_avg.csv"
    )

    sub_rank.to_csv("submission.csv", index=False)
    print(sub_rank.head())
else:
    print(
        "No external models available; will create submission.csv from metadata-only baseline later."
    )




## === cell 8
def dim_optimizer(df_oof, features, init_points=20, n_iter=30):
    pbounds = {f"c{i}": (0.0, 1.0) for i in range(len(features))}

    def q(**params):
        x = np.zeros(len(df_oof), dtype=float)
        for i, f in enumerate(features):
            x += params[f"c{i}"] * df_oof[f].values
        x = np.nan_to_num(x, nan=0.5, posinf=1.0, neginf=0.0)
        return metrics.roc_auc_score(df_oof["target"].values, x)

    optimizer = BayesianOptimization(
        f=q,
        pbounds=pbounds,
        random_state=RANDOM_STATE,
        verbose=0,
    )
    optimizer.maximize(init_points=init_points, n_iter=n_iter)

    best_params = optimizer.max["params"]
    best_auc = optimizer.max["target"]
    coeffs = np.array([best_params[f"c{i}"] for i in range(len(features))], dtype=float)

    print(f"bo auc:{best_auc:.6f}")
    for f, c in zip(features, coeffs):
        print(f"  {f}: {c:.6f}")
    return coeffs, best_auc


coeffs, bo_auc = None, None
models_for_bo = []

if len(models) > 0:
    MAX_MODELS_FOR_BO = 12
    models_for_bo = models[:MAX_MODELS_FOR_BO]
    if len(models) > MAX_MODELS_FOR_BO:
        print(
            f"Note: limiting BayesOpt to first {MAX_MODELS_FOR_BO} models for runtime safety."
        )
    if len(models_for_bo) >= 1:
        coeffs, bo_auc = dim_optimizer(train, models_for_bo, init_points=10, n_iter=15)
else:
    print("No external models available; skipping BayesOpt.")




## === cell 9
def bo_pred(df, features, coeffs):
    x = np.zeros(len(df), dtype=float)
    for f, c in zip(features, coeffs):
        x += c * df[f].values
    return np.nan_to_num(x, nan=0.5, posinf=1.0, neginf=0.0)


if coeffs is not None and len(models_for_bo) > 0:
    train["pred_bo"] = bo_pred(train, models_for_bo, coeffs)
    score_bo = metrics.roc_auc_score(train["target"], train["pred_bo"])
    print(f"auc bo:{score_bo:.6f}")

    test["target"] = bo_pred(test, models_for_bo, coeffs)
    submission_bo = test[["image_name", "target"]].copy()
    submission_bo.to_csv("submission_bo.csv", index=False)

    rank_auc = (
        metrics.roc_auc_score(train["target"], train["pred_rank"])
        if "pred_rank" in train.columns
        else -np.inf
    )
    if score_bo >= rank_auc:
        submission_bo.to_csv("submission.csv", index=False)
        print("BO blend selected as submission.csv (better/equal OOF than rank).")
    else:
        print("Rank blend kept as submission.csv (better OOF than BO).")

    print(submission_bo.head())
else:
    print("No BO blend produced.")



## === cell 10
if not os.path.exists("submission.csv"):

    def _logit(p, eps=1e-5):
        p = np.asarray(p, dtype=float)
        p = np.clip(p, eps, 1.0 - eps)
        return np.log(p / (1.0 - p)).astype(np.float32)

    def _make_age_bins(s: pd.Series) -> pd.Series:
        s = pd.to_numeric(s, errors="coerce")
        bins = [-np.inf, 30, 45, 55, 65, 75, np.inf]
        labels = ["<=30", "31-45", "46-55", "56-65", "66-75", "76+"]
        out = pd.cut(s, bins=bins, labels=labels)
        return out.astype(object).where(~out.isna(), "NA_BIN")

    features_num = [
        "age_approx",
        "age_isna",
        "n_images_per_patient",
        "log1p_n_images_per_patient",
        "patient_target_rate",
        "patient_site_target_rate",
        "site_target_rate",
        "patient_target_logit",
        "patient_site_target_logit",
        "site_target_logit",
        "age_bin_target_rate",
        "age_bin_target_logit",
        "diagnosis_target_rate",
        "diagnosis_target_logit",
    ]
    features_cat = ["sex", "anatom_site_general_challenge", "patient_id"]

    train = train.copy()
    test = test.copy()

    train["age_isna"] = train["age_approx"].isna().astype(np.float32)
    test["age_isna"] = test["age_approx"].isna().astype(np.float32)

    train["age_bin"] = _make_age_bins(train["age_approx"])
    test["age_bin"] = _make_age_bins(test["age_approx"])

    all_pid = pd.concat(
        [train[["patient_id"]], test[["patient_id"]]],
        axis=0,
        ignore_index=True,
    )
    pid_counts = (
        all_pid["patient_id"]
        .astype(str)
        .fillna("NA")
        .value_counts(dropna=False)
        .astype(np.float32)
    )
    train["n_images_per_patient"] = (
        train["patient_id"].astype(str).fillna("NA").map(pid_counts).astype(np.float32)
    )
    test["n_images_per_patient"] = (
        test["patient_id"].astype(str).fillna("NA").map(pid_counts).astype(np.float32)
    )

    train["log1p_n_images_per_patient"] = np.log1p(
        train["n_images_per_patient"].astype(np.float32)
    ).astype(np.float32)
    test["log1p_n_images_per_patient"] = np.log1p(
        test["n_images_per_patient"].astype(np.float32)
    ).astype(np.float32)

    y = train["target"].values
    groups = train["patient_id"].astype(str).fillna("NA").values
    if train["patient_id"].isna().all():
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        splits = list(splitter.split(train, y))
        print("Using StratifiedKFold (patient_id missing).")
    else:
        splitter = GroupKFold(n_splits=5)
        splits = list(splitter.split(train, y, groups=groups))
        print(
            "Using GroupKFold on patient_id to reduce leakage and improve LB generalization."
        )

    global_mean = float(np.mean(y))
    alpha = 20.0  # smoothing strength (kept same for stability vs prior runs)

    train["patient_target_rate"] = global_mean
    for tr_idx, va_idx in splits:
        pid_tr = train.iloc[tr_idx]["patient_id"].astype(str).fillna("NA")
        y_tr = train.iloc[tr_idx]["target"].astype(float)

        grp = pd.DataFrame({"pid": pid_tr.values, "y": y_tr.values}).groupby("pid")["y"]
        sum_map = grp.sum()
        cnt_map = grp.count()
        rate_map = (sum_map + alpha * global_mean) / (cnt_map + alpha)

        pid_va = train.iloc[va_idx]["patient_id"].astype(str).fillna("NA")
        train.loc[train.index[va_idx], "patient_target_rate"] = (
            pid_va.map(rate_map).fillna(global_mean).astype(np.float32).values
        )

    train["patient_site_target_rate"] = global_mean
    for tr_idx, va_idx in splits:
        pid_tr = train.iloc[tr_idx]["patient_id"].astype(str).fillna("NA")
        site_tr = (
            train.iloc[tr_idx]["anatom_site_general_challenge"].astype(str).fillna("NA")
        )
        key_tr = (pid_tr + "||" + site_tr).values
        y_tr = train.iloc[tr_idx]["target"].astype(float)

        grp = pd.DataFrame({"k": key_tr, "y": y_tr.values}).groupby("k")["y"]
        sum_map = grp.sum()
        cnt_map = grp.count()
        rate_map = (sum_map + alpha * global_mean) / (cnt_map + alpha)

        pid_va = train.iloc[va_idx]["patient_id"].astype(str).fillna("NA")
        site_va = (
            train.iloc[va_idx]["anatom_site_general_challenge"].astype(str).fillna("NA")
        )
        key_va = (pid_va + "||" + site_va).values
        train.loc[train.index[va_idx], "patient_site_target_rate"] = (
            pd.Series(key_va)
            .map(rate_map)
            .fillna(global_mean)
            .astype(np.float32)
            .values
        )

    train["site_target_rate"] = global_mean
    for tr_idx, va_idx in splits:
        site_tr = (
            train.iloc[tr_idx]["anatom_site_general_challenge"].astype(str).fillna("NA")
        )
        y_tr = train.iloc[tr_idx]["target"].astype(float)

        grp = pd.DataFrame({"site": site_tr.values, "y": y_tr.values}).groupby("site")[
            "y"
        ]
        sum_map = grp.sum()
        cnt_map = grp.count()
        rate_map = (sum_map + alpha * global_mean) / (cnt_map + alpha)

        site_va = (
            train.iloc[va_idx]["anatom_site_general_challenge"].astype(str).fillna("NA")
        )
        train.loc[train.index[va_idx], "site_target_rate"] = (
            site_va.map(rate_map).fillna(global_mean).astype(np.float32).values
        )

    train["age_bin_target_rate"] = global_mean
    for tr_idx, va_idx in splits:
        agebin_tr = train.iloc[tr_idx]["age_bin"].astype(str).fillna("NA_BIN")
        y_tr = train.iloc[tr_idx]["target"].astype(float)

        grp = pd.DataFrame({"ab": agebin_tr.values, "y": y_tr.values}).groupby("ab")[
            "y"
        ]
        sum_map = grp.sum()
        cnt_map = grp.count()
        rate_map = (sum_map + alpha * global_mean) / (cnt_map + alpha)

        agebin_va = train.iloc[va_idx]["age_bin"].astype(str).fillna("NA_BIN")
        train.loc[train.index[va_idx], "age_bin_target_rate"] = (
            agebin_va.map(rate_map).fillna(global_mean).astype(np.float32).values
        )

    diag_series_train = train["diagnosis"].astype(str).fillna("NA_DIAG")
    train["diagnosis_target_rate"] = global_mean
    for tr_idx, va_idx in splits:
        diag_tr = diag_series_train.iloc[tr_idx]
        y_tr = train.iloc[tr_idx]["target"].astype(float)

        grp = pd.DataFrame({"d": diag_tr.values, "y": y_tr.values}).groupby("d")["y"]
        sum_map = grp.sum()
        cnt_map = grp.count()
        rate_map = (sum_map + alpha * global_mean) / (cnt_map + alpha)

        diag_va = diag_series_train.iloc[va_idx]
        train.loc[train.index[va_idx], "diagnosis_target_rate"] = (
            diag_va.map(rate_map).fillna(global_mean).astype(np.float32).values
        )
    test["diagnosis_target_rate"] = np.float32(global_mean)

    test_pid = test["patient_id"].astype(str).fillna("NA")
    train_pid = train["patient_id"].astype(str).fillna("NA")

    grp_full = train["target"].astype(float).groupby(train_pid)
    sum_full = grp_full.sum()
    cnt_full = grp_full.count()
    full_rate_map = (sum_full + alpha * global_mean) / (cnt_full + alpha)
    test["patient_target_rate"] = (
        test_pid.map(full_rate_map).fillna(global_mean).astype(np.float32)
    )

    test_site = test["anatom_site_general_challenge"].astype(str).fillna("NA")
    train_site = train["anatom_site_general_challenge"].astype(str).fillna("NA")
    train_key = train_pid + "||" + train_site

    grp_full2 = train["target"].astype(float).groupby(train_key)
    sum_full2 = grp_full2.sum()
    cnt_full2 = grp_full2.count()
    full_rate_map2 = (sum_full2 + alpha * global_mean) / (cnt_full2 + alpha)

    test_key = test_pid + "||" + test_site
    test["patient_site_target_rate"] = (
        test_key.map(full_rate_map2).fillna(global_mean).astype(np.float32)
    )

    grp_full3 = train["target"].astype(float).groupby(train_site)
    sum_full3 = grp_full3.sum()
    cnt_full3 = grp_full3.count()
    full_rate_map3 = (sum_full3 + alpha * global_mean) / (cnt_full3 + alpha)
    test["site_target_rate"] = (
        test_site.map(full_rate_map3).fillna(global_mean).astype(np.float32)
    )

    train_agebin = train["age_bin"].astype(str).fillna("NA_BIN")
    test_agebin = test["age_bin"].astype(str).fillna("NA_BIN")
    grp_full_ab = train["target"].astype(float).groupby(train_agebin)
    sum_full_ab = grp_full_ab.sum()
    cnt_full_ab = grp_full_ab.count()
    full_rate_map_ab = (sum_full_ab + alpha * global_mean) / (cnt_full_ab + alpha)
    test["age_bin_target_rate"] = (
        test_agebin.map(full_rate_map_ab).fillna(global_mean).astype(np.float32)
    )

    train["patient_target_logit"] = _logit(train["patient_target_rate"].values)
    train["patient_site_target_logit"] = _logit(
        train["patient_site_target_rate"].values
    )
    train["site_target_logit"] = _logit(train["site_target_rate"].values)
    train["age_bin_target_logit"] = _logit(train["age_bin_target_rate"].values)
    train["diagnosis_target_logit"] = _logit(train["diagnosis_target_rate"].values)

    test["patient_target_logit"] = _logit(test["patient_target_rate"].values)
    test["patient_site_target_logit"] = _logit(test["patient_site_target_rate"].values)
    test["site_target_logit"] = _logit(test["site_target_rate"].values)
    test["age_bin_target_logit"] = _logit(test["age_bin_target_rate"].values)
    test["diagnosis_target_logit"] = _logit(test["diagnosis_target_rate"].values)

    X = train[features_num + features_cat].copy()
    X_test = test[features_num + features_cat].copy()

    pre = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(steps=[("imp", SimpleImputer(strategy="median"))]),
                features_num,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imp", SimpleImputer(strategy="most_frequent")),
                        ("oh", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                features_cat,
            ),
        ],
        remainder="drop",
    )

    clf = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )

    pipe = Pipeline(steps=[("pre", pre), ("clf", clf)])

    oof = np.zeros(len(train), dtype=float)
    for tr_idx, va_idx in splits:
        pipe.fit(X.iloc[tr_idx], y[tr_idx])
        oof[va_idx] = pipe.predict_proba(X.iloc[va_idx])[:, 1]
    auc = metrics.roc_auc_score(y, oof)
    print(
        f"Metadata baseline (GroupKFold + small FE + smoothed OOF target encodings + logit TE + log1p count + age-bin TE + diagnosis OOF TE) 5-fold OOF AUC: {auc:.6f}"
    )

    pipe.fit(X, y)
    preds = pipe.predict_proba(X_test)[:, 1]
    submission = pd.DataFrame(
        {"image_name": test["image_name"].values, "target": preds}
    )

    submission = sub_sample[["image_name"]].merge(
        submission, on="image_name", how="left", validate="one_to_one"
    )
    submission["target"] = submission["target"].fillna(0.5)

    submission.to_csv("submission.csv", index=False)
    print(
        "Wrote submission.csv from metadata baseline (with diagnosis OOF target-encoding added)."
    )
    print(submission.head())
else:
    sub = pd.read_csv("submission.csv")
    if list(sub.columns) != ["image_name", "target"]:
        if "image_name" in sub.columns and "target" in sub.columns:
            sub = sub[["image_name", "target"]]
        else:
            sub = sub_sample.copy()
            sub["target"] = 0.5

    sub = sub_sample[["image_name"]].merge(
        sub, on="image_name", how="left", validate="one_to_one"
    )
    sub["target"] = sub["target"].fillna(0.5)
    sub.to_csv("submission.csv", index=False)

    print("submission.csv already exists (re-ordered to sample_submission). Head:")
    print(pd.read_csv("submission.csv").head())
