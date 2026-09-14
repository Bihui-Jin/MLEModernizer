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

0.9032724995722342

# 6. Current score

0.77116

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'Your script currently doesn’t yield a score because it points to `/kaggle/input/melanoma`, which doesn’t exist in your provided filesystem, so it never finds prediction files to merge and `cols` stays empty. I make the input path robust by searching a small set of likely directories (including `/kaggle/working` and `/kaggle/input`) for any `.csv` files containing `image_name` and a prediction column, then merge only those that match the sample submission IDs. I also add a safe fallback: if no external prediction files are found, create a valid submission using the sample submission template (all zeros) so you always get a submit-ready `submission.csv`. These changes preserve your core ensembling-by-mean logic and only fix the file-discovery/alignment issues so a valid submission is always produced.'
- What this solution (achieved 0.66666) has done: 'Your current pipeline produces a valid CSV but (score=0.5) it’s effectively constant predictions because no real model outputs are being found/used. To move toward the target AUC with minimal change to your existing “merge prediction CSVs then mean-ensemble” logic, I (1) stop scanning arbitrary `/kaggle/*` trees and instead explicitly look for the competition’s known baseline OOF/test prediction artifacts (often stored as `.npy` in public notebooks), and (2) if none are found, create a lightweight metadata-only model (logistic regression) trained on `train.csv` and applied to `test.csv` to generate non-constant probabilities. This keeps the “simple ensembling of available predictors into submission” semantics, but ensures we always have at least one meaningful predictor instead of all-zeros. The result should raise AUC substantially above 0.5 and closer to your 0.903 target without changing any deep learning architecture/training loops (you currently have none).'
- What this solution (achieved 0.75264) has done: 'Your current fallback is a single in-sample logistic regression on noisy/limited metadata, which tends to underperform (AUC ~0.66) because it overfits and doesn’t capture patient-level leakage patterns correctly. To move the score upward toward your 0.903 target while keeping the same core “metadata logistic regression fallback + mean ensembling of any found prediction CSVs” logic, I replace the in-sample training with out-of-fold (OOF) target encoding for `patient_id` and `anatom_site_general_challenge`, then fit the same LogisticRegression on these encoded features plus the existing ones. This is a minimal modeling change (still logistic regression; no new training loops beyond a simple KFold) and is specifically aimed at improving AUC by giving the model a stronger, properly-regularized signal without leaking the label. I also ensure train/test feature alignment stays consistent and that we still always write a valid `submission.csv`.'
- What this solution (achieved 0.76085) has done: 'I keep your overall “discover any prediction CSVs → merge → mean-ensemble, else train metadata LogisticRegression” flow intact, but make two minimal, score-relevant upgrades to push AUC upward toward 0.903. First, I add one more high-signal, leak-safe feature: an out-of-fold target encoding for `diagnosis` (train-only) and map it to test as the global mean (since test lacks diagnosis), which improves train-time separation without leaking and can modestly improve generalization via better calibration/regularization of the linear model. Second, I make the fallback LogisticRegression slightly better calibrated for AUC by using `class_weight="balanced"` (this often helps in this highly imbalanced dataset) while keeping the same model family/solver and no extra training loops. These are small changes that should increase your current 0.75264 toward the target band without altering your ensembling semantics or adding heavy dependencies.'
- What this solution (achieved 0.66098) has done: 'Your current score (0.76085) is below the target (0.90327), so we should improve AUC with minimal risk while keeping your existing “find external prediction CSVs → else metadata LogisticRegression with OOF target encoding → mean-ensemble” logic. The biggest low-risk gain here is to prevent leakage across the same `patient_id` in the OOF target encoding: switching from `KFold` to `GroupKFold(patient_id)` keeps the encoding leak-safe and typically improves test AUC for this competition. I also ensure the diagnosis target encoding uses the same group splits (still train-only; test remains mapped to global mean), and keep everything else (features, model, ensembling, submission writing) unchanged. These changes are small, metric-aligned, and should move your score upward toward the target band.'
- What this solution (achieved 0.5) has done: 'Your current score (0.66098) is far below the target (0.90327), so we should improve AUC while keeping your overall “use any found prediction CSVs else fallback metadata LogisticRegression with OOF target encoding → mean-ensemble” logic intact. The biggest low-risk gain is to stop training the fallback classifier on leakage-prone, per-image rows and instead aggregate training to one row per `patient_id` (mean target and mean/majority metadata) so the model learns patient-level risk more robustly; then map predictions back to test images by `patient_id`. This keeps the same model family (LogisticRegression), the same target encodings idea (now on patient-level categories), and the same mean-ensemble mechanics, but typically boosts generalization on this competition. I also keep your file discovery/merging unchanged and still always write a valid `submission.csv`.'
- What this solution (achieved 0.69383) has done: 'I fix the runtime error by ensuring the fallback logistic regression is trained on a proper binary classification target (0/1) instead of a continuous patient-mean target, while keeping the same patient-level aggregation and feature pipeline. Specifically, I change the patient-level `target` aggregation from mean to max (equivalent to “any malignant for that patient”), which matches the classification objective and removes the `Unknown label type: 'continuous'` error. I also keep probabilities well-formed (fill missing with global mean and clip to [0,1]) so the submission is always valid. These changes are minimal, preserve your overall “find external preds else metadata fallback → mean ensemble” logic, and should improve AUC versus constant/near-constant predictions.'
- What this solution (achieved 0.6941) has done: 'Your current score (0.69383) is well below the target (0.90327), so we should improve AUC with minimal, low-risk changes while keeping your overall “external prediction CSV mean-ensemble, else metadata LogisticRegression fallback” logic intact. The biggest issue is that your fallback model only uses very weak patient-level metadata; we can add leak-safe, high-signal out-of-fold target encodings for `patient_id` and `sex` (in addition to your existing `anatom_site` TE) using `GroupKFold(patient_id)` to avoid leakage. This preserves the same model family (LogisticRegression), same basic feature pipeline (metadata + TE + one-hot + scaling), and only strengthens features to move the score upward toward the target. We also keep the submission alignment unchanged and still always write a valid `submission.csv`.'
- What this solution (achieved 0.67542) has done: 'Your current 0.6941 AUC is far below the 0.90327 target (higher is better), so we should improve the fallback predictor while keeping your core “use external prediction CSVs if found, else metadata LogisticRegression, then mean-ensemble” logic unchanged. The biggest issue is that your patient-level model is trained on one row per patient (losing per-image signal) and then broadcast back to images; in this competition, per-image metadata + leak-safe patient-level target encoding usually works better. I switch the fallback to train at the image level with `GroupKFold(patient_id)`-based target encodings for `patient_id`, `anatom_site_general_challenge`, and `sex`, then fit the same LogisticRegression pipeline and predict per image. I also tighten CSV discovery slightly to prefer files that cover most test IDs to avoid ensembling in broken/partial prediction files that can drag AUC down.'
- What this solution (achieved 0.67609) has done: 'Your current score (0.67542) is well below the target (0.90327), so we should cautiously improve AUC while keeping your same overall pipeline: “use any external prediction CSVs if found, else fall back to metadata logistic regression, then mean-ensemble.” The lowest-risk uplift is to strengthen the fallback model without changing its family by (1) adding a leak-safe, out-of-fold target encoding for `anatom_site_general_challenge` conditioned on `sex` (interaction signal often helps), and (2) fitting the LogisticRegression on out-of-fold predictions (via GroupKFold) to reduce overfitting and improve ranking on test. We keep the same feature set plus the added encoded feature, still use LogisticRegression with the same solver/loss, and still output a valid `submission.csv` with the required columns. No changes are made to I/O paths, and the external-prediction discovery/mean-ensemble logic remains intact.'
- What this solution (achieved 0.67217) has done: 'We keep your existing pipeline (search external prediction CSVs → otherwise fallback metadata logistic regression with GroupKFold target encodings → mean-ensemble) but make two minimal, score-relevant fixes to push AUC upward toward the 0.903 target. First, we correct a subtle preprocessing issue: `StandardScaler(with_mean=False)` is not appropriate for dense numeric features (and can weaken the logistic regression fit); switching to `with_mean=True` is a small change that typically improves ranking/AUC without changing the model family or training approach. Second, we add a very lightweight, leak-safe normalization for `age_approx` by creating an `age_missing` indicator (missingness is informative here) and keep the rest identical; this often improves AUC with negligible added complexity. Everything else (paths, file discovery, OOF target encodings, GroupKFold averaging, ensembling, and writing `submission.csv`) remains the same.'
- What this solution (achieved 0.67216) has done: 'We keep your existing “external CSV mean-ensemble, else metadata LogisticRegression with GroupKFold target encodings” core logic, but make two small, score-relevant upgrades aimed at improving AUC ranking without changing the modeling family. First, we fix a subtle bug in your current setup where the target encoding for `patient_id` is computed with the same `patient_id` as the grouping variable, which makes the OOF encoding collapse toward the global mean (weak signal); we instead compute patient-id target encoding using a different, leak-safe grouping (`patient_id` groups still protect model CV, but the encoding be built with folds grouped by `patient_id` only for non-patient features, and for `patient_id` itself we use standard KFold on patients). Second, we reduce variance and improve calibration slightly by averaging two LogisticRegression strengths (two `C` values) inside the same CV loop and then still mean-ensemble as before; this is minimal and stays within the same approach. Everything else (paths, file discovery, preprocessing, submission writing) remains unchanged and still produces a valid `submission.csv`.'
- What this solution (achieved 0.73873) has done: 'We need to increase AUC from 0.67216 toward 0.90327 (higher is better), but keep the same overall “external prediction CSV mean-ensemble, else metadata LogisticRegression with target encodings” logic. The smallest high-impact, still-lightweight change is to add a strong, leak-safe numeric feature derived from the training set only: patient-level lesion count (`n_images_per_patient`) and map it to test by `patient_id` (with an “unknown” fallback), since this competition has strong patient-level structure. I also make the final ensemble slightly more robust by using a coverage-weighted mean across sources (still a mean-ensemble, but reduces the damage from weaker/partial sources), while keeping identical I/O and submission semantics. These two changes should improve ranking/AUC without changing the model family, loss, or introducing heavy dependencies.'
- What this solution (achieved 0.77116) has done: 'Your current score (0.73873) is well below the target (0.90327), so we should improve AUC with the smallest, low-risk change that preserves your existing “external prediction CSV mean-ensemble else metadata LogisticRegression with target encodings” core logic. The biggest likely issue is that your target-encoding features are fit once on the full training set and then used inside a GroupKFold-averaged model, which leaks validation labels through the encodings and tends to hurt true generalization; we recompute the target encodings *inside each GroupKFold training fold* (OOF-within-CV) and then average fold test predictions as you already do. This keeps the same model family (LogisticRegression), same feature set/engineering (age, missing flag, patient image counts, sex/site one-hots, and the same TE columns), same training approach (GroupKFold averaging), and the same submission semantics, but makes the encodings leak-safe and typically increases leaderboard AUC. The external CSV discovery/merging and coverage-weighted mean ensemble remain unchanged, and the script still always writes a valid `submission.csv`.'
- What this solution (achieved 0.77116) has done: 'Your current AUC (0.77116) is below the target (0.90327), so we should improve ranking with the smallest, low-risk change while preserving your “external prediction CSV mean-ensemble, else metadata LogisticRegression fallback” core logic. The main weakness is that in each GroupKFold loop you train the model on `X_train_enc.iloc[tr_idx]` but `X_tr_fold`/encodings were built using *all* training rows, which re-introduces fold leakage through feature construction; I rebuild fold features from only the fold’s train/val splits, and fit on the fold-train rows only. I also add out-of-fold (fold-specific) target encodings for the fold’s validation rows (not just train/test), so the model learns on leak-safe encodings and generalizes better. Everything else (paths, ensembling, LogisticRegression, GroupKFold averaging, submission writing) stays the same.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import os
from pathlib import Path

SS_CANDIDATES = [
    "/kaggle/input/siim-isic-melanoma-classification/sample_submission.csv",
    "/kaggle/data/siim-isic-melanoma-classification/sample_submission.csv",
    "/kaggle/input/sample_submission.csv",
    "/kaggle/data/sample_submission.csv",
]
ss_path = next((p for p in SS_CANDIDATES if Path(p).exists()), None)
if ss_path is None:
    raise FileNotFoundError(f"Could not find sample_submission.csv in: {SS_CANDIDATES}")

f = pd.read_csv(ss_path)[["image_name"]]
print("sample_submission shape:", f.shape, "path:", ss_path)




## === cell 1
search_roots = [
    "/kaggle/working",
    "/kaggle/input/siim-isic-melanoma-classification",
    "/kaggle/data/siim-isic-melanoma-classification",
]


def _is_prob_column(s: pd.Series) -> bool:
    if not pd.api.types.is_numeric_dtype(s):
        return False
    s2 = s.dropna()
    if len(s2) == 0:
        return True
    return (s2.min() >= -0.5) and (s2.max() <= 1.5)


def find_prediction_csvs(roots, max_files=500):
    """
    Keep your CSV-ensemble core logic; only search likely directories to find external preds.
    """
    candidates = []
    seen = 0
    for root in roots:
        rootp = Path(root)
        if not rootp.exists():
            continue
        for p in rootp.rglob("*.csv"):
            seen += 1
            if seen > max_files:
                break
            name = p.name.lower()
            if name in {
                "train.csv",
                "test.csv",
                "sample_submission.csv",
                "submission.csv",
            }:
                continue
            candidates.append(str(p))
    return candidates


pred_csvs = find_prediction_csvs(search_roots)

usable = []
test_ids = set(f["image_name"].values)
min_coverage = (
    0.90  # keep: avoid ensembling partial/broken prediction files that can hurt AUC.
)

for p in pred_csvs:
    try:
        ff = pd.read_csv(p)
        if "image_name" not in ff.columns:
            continue

        pred_col = None
        if "target" in ff.columns:
            pred_col = "target"
        else:
            for c in ff.columns:
                if c == "image_name":
                    continue
                if _is_prob_column(ff[c]):
                    pred_col = c
                    break
        if pred_col is None:
            continue

        ff = ff[["image_name", pred_col]].copy()
        ff = ff.rename(columns={pred_col: "target"})
        ff = ff[ff["image_name"].isin(test_ids)]
        if ff.shape[0] == 0:
            continue

        coverage = ff["image_name"].nunique() / len(test_ids)
        if coverage < min_coverage:
            continue

        usable.append((p, ff))
    except Exception:
        continue

print(f"Found {len(pred_csvs)} csv files; usable prediction files: {len(usable)}")
for p, ff in usable[:10]:
    print("  ", p, ff.shape)




## === cell 2
if len(usable) == 0:
    TRAIN_CANDIDATES = [
        "/kaggle/input/siim-isic-melanoma-classification/train.csv",
        "/kaggle/data/siim-isic-melanoma-classification/train.csv",
        "/kaggle/input/train.csv",
        "/kaggle/data/train.csv",
    ]
    TEST_CANDIDATES = [
        "/kaggle/input/siim-isic-melanoma-classification/test.csv",
        "/kaggle/data/siim-isic-melanoma-classification/test.csv",
        "/kaggle/input/test.csv",
        "/kaggle/data/test.csv",
    ]
    train_path = next((p for p in TRAIN_CANDIDATES if Path(p).exists()), None)
    test_path = next((p for p in TEST_CANDIDATES if Path(p).exists()), None)
    if train_path is None or test_path is None:
        raise FileNotFoundError(
            "Could not locate train.csv/test.csv in expected Kaggle paths."
        )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    test_df = test_df.merge(f[["image_name"]], on="image_name", how="right")

    from sklearn.model_selection import GroupKFold

    def te_fit_map(train_cat: pd.Series, train_y: pd.Series, smoothing: float):
        """
        Why: build smoothed mean encoding map on a training fold only (leak-safe).
        """
        train_cat = train_cat.fillna("unknown").astype(str)
        train_y = train_y.astype(float)
        global_mean = float(train_y.mean())
        stats = (
            pd.DataFrame({"cat": train_cat.values, "y": train_y.values})
            .groupby("cat")["y"]
            .agg(["mean", "count"])
        )
        smooth = (stats["mean"] * stats["count"] + global_mean * smoothing) / (
            stats["count"] + smoothing
        )
        return smooth, global_mean

    def te_apply_map(cat: pd.Series, smooth_map: pd.Series, global_mean: float):
        cat = cat.fillna("unknown").astype(str)
        return cat.map(smooth_map).fillna(global_mean).astype(np.float64).values

    def patient_te_fit_map_kfold(patient_id: pd.Series, y: pd.Series, smoothing: float):
        """
        Why: patient_id TE computed from patient-level max target (fold-train only), then mapped to rows.
        """
        pid = patient_id.fillna("unknown").astype(str)
        y = y.astype(float)
        global_mean = float(y.mean())

        pat = pd.DataFrame({"patient_id": pid.values, "y": y.values})
        pat_agg = pat.groupby("patient_id")["y"].max().astype(float)

        stats = pat_agg.to_frame("y").groupby(level=0)["y"].agg(["mean", "count"])
        smooth = (stats["mean"] * stats["count"] + global_mean * smoothing) / (
            stats["count"] + smoothing
        )
        return smooth, global_mean

    for df in (train_df, test_df):
        df["age_approx"] = pd.to_numeric(df["age_approx"], errors="coerce")
        df["sex"] = df["sex"].fillna("unknown").astype(str)
        df["anatom_site_general_challenge"] = (
            df["anatom_site_general_challenge"].fillna("unknown").astype(str)
        )
        df["patient_id"] = df["patient_id"].fillna("unknown").astype(str)
        df["site_x_sex"] = (
            df["anatom_site_general_challenge"].astype(str)
            + "___"
            + df["sex"].astype(str)
        )

    train_df["age_missing"] = train_df["age_approx"].isna().astype(np.int8)
    test_df["age_missing"] = test_df["age_approx"].isna().astype(np.int8)

    age_median = float(np.nanmedian(train_df["age_approx"].values))
    train_df["age_approx"] = train_df["age_approx"].fillna(age_median)
    test_df["age_approx"] = test_df["age_approx"].fillna(age_median)

    y = train_df["target"].astype(int)
    groups = train_df["patient_id"]

    pid_counts = train_df["patient_id"].value_counts(dropna=False)
    train_df["n_images_per_patient"] = (
        train_df["patient_id"].map(pid_counts).astype(np.float64)
    )
    test_df["n_images_per_patient"] = (
        test_df["patient_id"].map(pid_counts).fillna(1.0).astype(np.float64)
    )

    base_cols = [
        "age_approx",
        "age_missing",
        "n_images_per_patient",
        "sex",
        "anatom_site_general_challenge",
    ]

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    C_LIST = [0.5, 1.0]
    gkf = GroupKFold(n_splits=5)

    test_pred = np.zeros(len(test_df), dtype=np.float64)

    for fold, (tr_idx, val_idx) in enumerate(
        gkf.split(train_df, y, groups=groups), start=1
    ):
        tr = train_df.iloc[tr_idx].copy()
        tr_y = y.iloc[tr_idx].copy()

        site_map, site_gm = te_fit_map(
            tr["anatom_site_general_challenge"], tr_y, smoothing=20.0
        )
        sex_map, sex_gm = te_fit_map(tr["sex"], tr_y, smoothing=20.0)
        sxs_map, sxs_gm = te_fit_map(tr["site_x_sex"], tr_y, smoothing=20.0)
        pid_map, pid_gm = patient_te_fit_map_kfold(
            tr["patient_id"], tr_y, smoothing=50.0
        )

        def build_X(df_slice: pd.DataFrame) -> pd.DataFrame:
            X = df_slice[base_cols].copy()
            X["te_anatom_site"] = te_apply_map(
                df_slice["anatom_site_general_challenge"], site_map, site_gm
            )
            X["te_sex"] = te_apply_map(df_slice["sex"], sex_map, sex_gm)
            X["te_site_x_sex"] = te_apply_map(df_slice["site_x_sex"], sxs_map, sxs_gm)
            X["te_patient_id"] = (
                df_slice["patient_id"]
                .map(pid_map)
                .fillna(pid_gm)
                .astype(np.float64)
                .values
            )
            return X

        X_tr_raw = build_X(train_df.iloc[tr_idx])
        X_val_raw = build_X(train_df.iloc[val_idx])
        X_te_raw = build_X(test_df)

        X_all = pd.concat([X_tr_raw, X_val_raw, X_te_raw], axis=0, ignore_index=True)
        X_all = pd.get_dummies(
            X_all, columns=["sex", "anatom_site_general_challenge"], dummy_na=False
        )
        n_tr = len(X_tr_raw)
        n_val = len(X_val_raw)
        X_tr_enc = X_all.iloc[:n_tr].copy()
        X_val_enc = X_all.iloc[n_tr : n_tr + n_val].copy()
        X_te_enc = X_all.iloc[n_tr + n_val :].copy()

        fold_test_pred = np.zeros(len(X_te_enc), dtype=np.float64)
        for C in C_LIST:
            clf = Pipeline(
                steps=[
                    ("scaler", StandardScaler(with_mean=True)),
                    (
                        "lr",
                        LogisticRegression(
                            max_iter=2000,
                            solver="lbfgs",
                            C=C,
                            class_weight="balanced",
                        ),
                    ),
                ]
            )
            clf.fit(X_tr_enc, y.iloc[tr_idx])
            fold_test_pred += clf.predict_proba(X_te_enc)[:, 1].astype(
                np.float64
            ) / len(C_LIST)

        test_pred += fold_test_pred / gkf.n_splits
        print(f"Fold {fold}/{gkf.n_splits} done.")

    global_mean = float(y.mean())
    test_pred = np.clip(test_pred, 0.0, 1.0)

    ff = pd.DataFrame({"image_name": test_df["image_name"].values, "target": test_pred})
    ff["target"] = ff["target"].fillna(global_mean).clip(0.0, 1.0)

    usable.append(("__metadata_logreg_image_level_gkf_leaksafe_te__", ff))
    print(
        "Added fallback predictor from metadata (image-level logistic regression, GroupKFold avg, leak-safe fold-built features):",
        ff.shape,
    )




## === cell 3
cols = []
src_meta = []  # (col_name, source_path, coverage)
n_test = len(f)

for i, (p, ff) in enumerate(usable):
    col = f"target_{i}"
    cols.append(col)

    cov = float(ff["image_name"].nunique() / n_test) if n_test > 0 else 0.0
    src_meta.append((col, p, cov))

    ff = ff.rename(columns={"target": col})
    f = f.merge(ff, on="image_name", how="left")

print("Merged shape:", f.shape)
print("Ensemble columns:", len(cols))
if len(cols) > 0:
    print("Ensemble sources (first 10):", [p for p, _ in usable[:10]])
    print("Coverage (first 10):", src_meta[:10])




## === cell 4
if len(cols) == 0:
    f["target"] = 0.0
else:
    weights = np.array([max(1e-6, m[2]) for m in src_meta], dtype=np.float64)

    X = f[cols].to_numpy(dtype=np.float64)
    mask = np.isfinite(X)
    X_filled = np.where(mask, X, 0.0)

    num = (X_filled * weights.reshape(1, -1)).sum(axis=1)
    den = (mask.astype(np.float64) * weights.reshape(1, -1)).sum(axis=1)
    pred = np.where(den > 0, num / den, np.nan)

    f["target"] = pd.Series(pred, index=f.index)
    f["target"] = f["target"].fillna(
        f["target"].mean() if f["target"].notna().any() else 0.0
    )
    f["target"] = f["target"].clip(0.0, 1.0)
    f.drop(columns=cols, inplace=True)

f.head()




## === cell 5
sub = f[["image_name", "target"]].copy()
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.dtypes)
print(sub.head())
print("target summary:", sub["target"].describe())
