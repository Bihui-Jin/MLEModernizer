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

0.50517

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.75687) has done: 'I fix the pipeline so it always runs end-to-end even when no external `oof.csv/submission.csv` prediction folders exist in `/kaggle/input`. The main bug is that your code hard-fails when it can’t find precomputed predictions, which also causes downstream NaNs and AUC computation errors. I add a minimal fallback that trains a simple scikit-learn model on the provided metadata (same target semantics) and produces `submission.csv` in the required format, while keeping your ensembling logic intact when predictions are available. I also harden merges/AUC steps by ensuring predictions are finite and filling any remaining missing values deterministically.'
- What this solution (achieved 0.67415) has done: 'Your current score (0.75687) is far below the target (0.9413), so we should improve the fallback path that’s actually generating your submission. The biggest issue is that the fallback model is dominated by `patient_id` one-hot leakage-like memorization that does not generalize to test; removing `patient_id` from features typically boosts public AUC for this competition’s metadata baselines. With minimal changes, I switch the fallback to a stronger but still simple scikit-learn model (HistGradientBoostingClassifier) using only `sex`, `age_approx`, and `anatom_site_general_challenge`, and I also apply a tiny isotonic calibration (fit on train) to improve AUC via better probability ranking. Everything else (external prediction discovery + ensembling + BO) is kept intact, and the script still always writes a valid `submission.csv`.'
- What this solution (achieved 0.67956) has done: 'Your current score (0.67415) is far below the target (0.9413), and in your environment you’re almost certainly on the metadata-only fallback path; the biggest issue there is that you “calibrate” test predictions using an isotonic model fit on *train* predictions but then (incorrectly) apply it to *test* predictions, which can badly distort ranking and hurt AUC. With minimal changes and identical core approach (same features + HistGradientBoosting), I replace that with proper out-of-fold (OOF) isotonic calibration using StratifiedKFold so calibration is learned without target leakage and is applied consistently to test. I also add `class_weight="balanced"` to the booster to better handle the strong class imbalance typical of this competition, which usually improves ROC-AUC for metadata baselines without changing the overall pipeline structure. Everything else (external prediction discovery + ensembling/BO when available + submission writing) is kept intact.'
- What this solution (achieved 0.67104) has done: 'Your current score (0.67956) is far below the target (0.9413), so we should focus on improving the metadata-fallback path that is likely producing your submission. The biggest minimal win for ROC-AUC here is to make the fallback model generalize better by (1) using a patient-level split (GroupKFold on `patient_id`) to avoid leakage across folds during OOF calibration and (2) slightly strengthening the HistGradientBoosting settings (more iterations/leaves) without changing the overall modeling approach. I also apply the isotonic calibration correctly by fitting it on OOF predictions from the *same* group-split and then applying it to test predictions. Everything else (external prediction discovery/ensembling/BO, submission writing, column semantics) stays the same.'
- What this solution (achieved 0.67458) has done: 'We’re far below the target AUC, and since you’re almost certainly on the metadata-only fallback path, the smallest safe way to move the score up is to improve the fallback model’s *ranking* without changing the overall approach. I keep the same preprocessing + HistGradientBoosting + OOF isotonic calibration, but (1) switch to StratifiedGroupKFold (still patient-grouped, but also stratified) for more stable OOF calibration, and (2) let the booster use a bit more capacity (more iterations/leaves) while staying in the same model family. I also (3) add a very small amount of label smoothing before isotonic fitting to reduce overfitting to noisy OOF ranks (often improves ROC-AUC slightly), and (4) ensure the produced submission stays aligned and valid.'
- What this solution (achieved 0.66604) has done: 'Your current score (0.67458) is far below the target (0.9413), so we should improve the metadata-only fallback that is likely producing your submission. With minimal core-logic change (still: same metadata features → HistGradientBoostingClassifier → OOF isotonic calibration), I (1) strengthen the model a bit in a controlled way (slightly more trees/leaves, add L2 regularization) and (2) make the isotonic calibration less overfit by fitting it on OOF predictions but with stronger, simpler label smoothing (no leakage). I also (3) optionally blend a small amount of the raw model probability with the calibrated probability to preserve ranking if isotonic over-corrects (a tiny, fixed blend; no tuning). Everything else (external prediction discovery + ensembling/BO when available, submission alignment/format) stays the same and still writes a valid `submission.csv`.'
- What this solution (achieved 0.66274) has done: 'Your current score (0.66604) is far below the target (0.9413), and given the code path it’s very likely you’re using the metadata-only fallback; the biggest minimal win is to remove the OOF isotonic calibration step, which can easily distort ranking and hurt ROC-AUC when the base model is weak/noisy. I keep the same features and the same HistGradientBoostingClassifier core model, but instead do a simple GroupKFold OOF evaluation only for sanity and then train once on full train and predict raw probabilities for test (no calibration, no blending). I also reduce regularization slightly and use a slightly larger learning rate (same model family, same approach) to improve ranking a bit without changing the overall logic. The script still run end-to-end and always write a valid `submission.csv` with the correct columns and alignment.'
- What this solution (achieved 0.67218) has done: 'We’re far below the target AUC, and you’re almost certainly on the metadata-only fallback path (no external `oof.csv/submission.csv` discovered), so the smallest meaningful improvement is to strengthen that fallback while keeping the same core logic (metadata → HistGradientBoostingClassifier → predict_proba). I keep the exact same features and model family, but replace the plain full-fit prediction with a patient-grouped cross-fitting blend: train 5 grouped models (GroupKFold by `patient_id`) and average their test probabilities, which usually improves generalization and ranking without changing semantics. I also slightly tune the existing HGB settings (still HGB; no new model types) toward a bit more capacity with mild regularization to lift AUC, and keep all submission alignment/NaN guards unchanged. The ensemble/BO path for external predictions is left intact.'
- What this solution (achieved 0.66014) has done: 'Your current AUC (0.67218) is far below the target (0.9413), so we should cautiously improve the *metadata fallback* path that is producing your submission. The smallest likely win without changing the core approach (metadata → HistGradientBoostingClassifier → predict_proba) is to add a couple of strong, competition-standard metadata features (`log1p(age)` and per-category malignant rate priors from train) and to use a stratified, patient-grouped cross-fitting ensemble (still GroupKFold-based) for a more stable ranking. These changes keep the same model family and prediction semantics, but typically improve ROC-AUC for this competition’s metadata-only baselines. Everything else (external prediction discovery, BO ensembling when available, submission alignment/format) is left intact.'
- What this solution (achieved 0.51253) has done: 'Your current AUC (0.66014) is far below the target (0.9413), so we should improve the *metadata fallback* path that is likely producing your submission while keeping the same overall approach (metadata → HistGradientBoostingClassifier → averaged fold predictions). The smallest high-impact change for this specific competition is to add a couple of well-known “strong” tabular priors (per-category *and* per-category+agebin target-rate encodings) computed on train only, which improves ranking without changing the model family or training loop structure. To keep it robust (and avoid leakage), those encodings are computed in an out-of-fold manner for train (for sanity AUC reporting) but are fit on full train for test-time features. Everything else (external prediction discovery/ensembling/BO and submission writing/alignment) is kept intact.'
- What this solution (achieved 0.51253) has done: 'Your current AUC (0.51253) is far below the target (0.9413), so we should focus on the metadata-fallback path (since external prediction folders are usually absent). The biggest minimal fix is to correct the target-encoding bug: your “full” encodings for test are accidentally computed on the already OOF-overwritten training columns, which injects fold-wise noise and can severely damage ranking (AUC). I keep the exact same model family (HistGradientBoostingClassifier), same features, and same grouped fold-averaging training loop, but compute OOF target encodings for train and separately compute “full-train” encodings for test from the original train targets only. This should move the score upward toward the target while preserving your overall pipeline and still writing a valid `submission.csv`.'
- What this solution (achieved 0.51253) has done: 'I fix the `KeyError: 'sex_rate'` by making the target-encoding merge helper robust to pandas’ `merge` suffix behavior (your fold dataframe already contains placeholder `sex_rate/site_rate/site_age_rate`, so `merge` creates `_x/_y` columns and the expected name disappears). I do this with a minimal change: `_merge_rate` drop any existing rate column before merging so the merged column name is stable, and it also handle accidental suffixes defensively. This is a pure bugfix that restores the intended fallback logic (metadata + OOF target encodings + HistGradientBoosting + fold-averaged test preds) and allow the pipeline to run end-to-end and write a valid `submission.csv`. No model/feature/training-loop changes beyond this merge bugfix.'
- What this solution (achieved 0.50517) has done: 'Your current score (0.51253) is far below the target (0.9413), so we should make the smallest changes that improve ranking quality in the metadata-fallback path (the only path likely used when no external oof/sub folders exist). The biggest low-risk fix is to compute the target-encoding features (`sex_rate/site_rate/site_age_rate`) strictly out-of-fold for training and strictly from full-train for test, but also to smooth them with a simple empirical-Bayes prior to reduce overfitting/noise that can collapse AUC toward ~0.5. I keep your exact model family (HistGradientBoostingClassifier), same fold-averaging loop, same features, and same submission-writing semantics, only adjusting how the rate features are computed (smoothed, leak-free) and adding a tiny deterministic noise to break ties (helps ROC-AUC when many identical priors occur). Everything else (external prediction discovery/ensembling/BO) remains unchanged.'

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
    """
    Keep the same core approach (metadata -> HistGradientBoostingClassifier -> avg fold preds).

    Score fix (minimal, directly relevant):
    - The rate features are powerful but can easily overfit / become too noisy, collapsing AUC ~0.5.
      We keep the exact same target-encoding features, but compute them with simple empirical-Bayes
      smoothing toward the global prior (on train-only stats), which typically improves ranking.
    - We keep OOF computation for train encodings and full-train computation for test encodings
      (leak-free), and we add tiny deterministic noise to the *rate* features only to break ties.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.impute import SimpleImputer
    from sklearn.ensemble import HistGradientBoostingClassifier

    tr = train_df.copy()
    te = test_df.copy()

    for df_ in (tr, te):
        df_["age_approx"] = pd.to_numeric(df_["age_approx"], errors="coerce")
        df_["age_log1p"] = np.log1p(df_["age_approx"].clip(lower=0))
        df_["age_bin"] = pd.cut(
            df_["age_approx"],
            bins=[-np.inf, 20, 30, 40, 50, 60, 70, 80, np.inf],
            labels=["<20", "20s", "30s", "40s", "50s", "60s", "70s", "80+"],
        ).astype(str)

        df_["sex"] = df_["sex"].fillna("missing").astype(str)
        df_["anatom_site_general_challenge"] = (
            df_["anatom_site_general_challenge"].fillna("missing").astype(str)
        )

    y = tr["target"].astype(int).to_numpy()
    global_rate = float(np.mean(y))

    groups = tr["patient_id"].astype(str).fillna("UNK")
    grp = pd.DataFrame({"patient_id": groups, "target": y})
    grp_stats = grp.groupby("patient_id")["target"].agg(["mean", "size"])
    grp_stats = grp_stats.rename(columns={"mean": "pos_rate", "size": "n"})

    order = grp_stats.sort_values(
        ["pos_rate", "n"], ascending=[False, False]
    ).reset_index()
    k = 5
    fold_pos = np.zeros(k, dtype=float)
    fold_n = np.zeros(k, dtype=float)
    group_to_fold = {}

    for _, row in order.iterrows():
        g = row["patient_id"]
        n = float(row["n"])
        pos = float(row["pos_rate"]) * n
        scores = (fold_pos + pos) / (fold_n + n + 1e-9)
        target_score = global_rate
        best = np.lexsort((fold_n, np.abs(scores - target_score)))[0]
        group_to_fold[g] = int(best)
        fold_pos[best] += pos
        fold_n[best] += n

    fold_id = groups.map(group_to_fold).to_numpy()

    tr_full_for_te = tr.copy()

    def _merge_rate(df_left, rates_df, key_cols, rate_col_name, default_rate):
        left = df_left.copy()
        if rate_col_name in left.columns:
            left = left.drop(columns=[rate_col_name])

        out = left.merge(rates_df, on=key_cols, how="left")

        if rate_col_name not in out.columns:
            for cand in (f"{rate_col_name}_y", f"{rate_col_name}_x"):
                if cand in out.columns:
                    out[rate_col_name] = out[cand]
                    break

        out[rate_col_name] = pd.to_numeric(out[rate_col_name], errors="coerce").fillna(
            default_rate
        )
        return out

    def _smoothed_rate_table(tr_part, group_cols, out_name, prior, alpha=50.0):
        g = tr_part.groupby(group_cols)["target"].agg(["sum", "count"]).reset_index()
        g[out_name] = (g["sum"] + alpha * prior) / (g["count"] + alpha)
        g = g.drop(columns=["sum", "count"])
        return g

    def _add_oof_te(tr_df, te_df, tr_full_df):
        tr_df = tr_df.copy()
        te_df = te_df.copy()
        tr_full_df = tr_full_df.copy()

        for col in ["sex_rate", "site_rate", "site_age_rate"]:
            tr_df[col] = global_rate
            te_df[col] = global_rate

        for f in range(k):
            idx_tr = np.where(fold_id != f)[0]
            idx_va = np.where(fold_id == f)[0]
            tr_part = tr_df.iloc[idx_tr]

            sex_rate = _smoothed_rate_table(
                tr_part, ["sex"], "sex_rate", prior=global_rate, alpha=50.0
            )
            site_rate = _smoothed_rate_table(
                tr_part,
                ["anatom_site_general_challenge"],
                "site_rate",
                prior=global_rate,
                alpha=50.0,
            )
            site_age_rate = _smoothed_rate_table(
                tr_part,
                ["anatom_site_general_challenge", "age_bin"],
                "site_age_rate",
                prior=global_rate,
                alpha=100.0,
            )

            va = tr_df.iloc[idx_va].copy()
            va = _merge_rate(va, sex_rate, ["sex"], "sex_rate", global_rate)
            va = _merge_rate(
                va,
                site_rate,
                ["anatom_site_general_challenge"],
                "site_rate",
                global_rate,
            )
            va = _merge_rate(
                va,
                site_age_rate,
                ["anatom_site_general_challenge", "age_bin"],
                "site_age_rate",
                global_rate,
            )

            tr_df.iloc[idx_va, tr_df.columns.get_loc("sex_rate")] = va[
                "sex_rate"
            ].to_numpy(dtype=float)
            tr_df.iloc[idx_va, tr_df.columns.get_loc("site_rate")] = va[
                "site_rate"
            ].to_numpy(dtype=float)
            tr_df.iloc[idx_va, tr_df.columns.get_loc("site_age_rate")] = va[
                "site_age_rate"
            ].to_numpy(dtype=float)

        sex_rate_full = _smoothed_rate_table(
            tr_full_df, ["sex"], "sex_rate", prior=global_rate, alpha=50.0
        )
        site_rate_full = _smoothed_rate_table(
            tr_full_df,
            ["anatom_site_general_challenge"],
            "site_rate",
            prior=global_rate,
            alpha=50.0,
        )
        site_age_rate_full = _smoothed_rate_table(
            tr_full_df,
            ["anatom_site_general_challenge", "age_bin"],
            "site_age_rate",
            prior=global_rate,
            alpha=100.0,
        )

        te2 = te_df.copy()
        te2 = _merge_rate(te2, sex_rate_full, ["sex"], "sex_rate", global_rate)
        te2 = _merge_rate(
            te2,
            site_rate_full,
            ["anatom_site_general_challenge"],
            "site_rate",
            global_rate,
        )
        te2 = _merge_rate(
            te2,
            site_age_rate_full,
            ["anatom_site_general_challenge", "age_bin"],
            "site_age_rate",
            global_rate,
        )

        return tr_df, te2

    tr, te = _add_oof_te(tr, te, tr_full_for_te)

    eps = 1e-6
    for df_ in (tr, te):
        for col in ["sex_rate", "site_rate", "site_age_rate"]:
            df_[col] = pd.to_numeric(df_[col], errors="coerce").fillna(global_rate)
            df_[col] = (
                df_[col] + eps * np.random.RandomState(42).randn(len(df_))
            ).clip(0.0, 1.0)

    feature_cols = [
        "sex",
        "age_approx",
        "age_log1p",
        "age_bin",
        "anatom_site_general_challenge",
        "sex_rate",
        "site_rate",
        "site_age_rate",
    ]
    X = tr[feature_cols].copy()
    X_test = te[feature_cols].copy()

    numeric_features = [
        "age_approx",
        "age_log1p",
        "sex_rate",
        "site_rate",
        "site_age_rate",
    ]
    categorical_features = ["sex", "age_bin", "anatom_site_general_challenge"]

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
        max_iter=1300,
        max_leaf_nodes=127,
        min_samples_leaf=20,
        l2_regularization=0.1,
        class_weight="balanced",
    )

    oof = np.zeros(len(X), dtype=float)
    test_pred_accum = np.zeros(len(X_test), dtype=float)

    for f in range(k):
        tr_idx = np.where(fold_id != f)[0]
        va_idx = np.where(fold_id == f)[0]

        pipe_fold = Pipeline([("pre", pre), ("clf", clf)])
        pipe_fold.fit(X.iloc[tr_idx], y[tr_idx])

        oof[va_idx] = pipe_fold.predict_proba(X.iloc[va_idx])[:, 1]
        test_pred_accum += pipe_fold.predict_proba(X_test)[:, 1] / float(k)

    try:
        print(f"Metadata fallback grouped OOF AUC: {metrics.roc_auc_score(y, oof):.6f}")
    except Exception as e:
        print(f"Could not compute grouped-OOF AUC due to: {e}")

    return test_pred_accum


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
