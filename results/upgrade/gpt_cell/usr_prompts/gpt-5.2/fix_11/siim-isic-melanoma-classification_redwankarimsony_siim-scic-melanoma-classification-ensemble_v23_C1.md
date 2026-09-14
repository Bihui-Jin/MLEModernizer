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

0.9421

# 6. Current score

0.65889

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'Diagnosis: Cell 1 crashes because it tries to read multiple CSVs from `../input/public-submission-melanoma-95/`, a directory that does not exist in this environment, causing a `FileNotFoundError` on the first `pd.read_csv`. Those external submission files are not part of the provided dataset paths, so the code cannot proceed as written. To unblock execution while preserving the expected downstream interface, we need to ensure the DataFrames referenced in cell 2 exist and have the required `target` column aligned with `sub`.

Patch summary: In cell 1, keep loading `sub` from the available `../input/siim-isic-melanoma-classification/sample_submission.csv`. For each missing external submission file, fall back to a copy of `sub` with `target` filled with zeros, so that cell 2’s weighted ensemble calculation runs deterministically without changing its logic.

Updated cells: only cell 1 is modified.

Compatibility notes for cell k+1: Cell 2 expects `public_sub_9619`, `public_sub_9606`, `public_sub_9603`, `public_sub_tabular`, and `sub` to exist and each have a `.target` series of identical length/order; the patch guarantees this by aligning all fallbacks to `sub`.

Assumptions: The competition’s `sample_submission.csv` is present at `../input/siim-isic-melanoma-classification/sample_submission.csv` (as shown in the provided file listing), and cell 2 does not require any other columns besides `target`.'
- What this solution (achieved 0.66696) has done: 'Your current 0.5 score is because all “public_sub_*” inputs are falling back to constant zeros, so the ensemble produces a constant prediction (AUC≈0.5). To move toward the 0.9421 target with minimal change and without altering the ensemble logic, I make the fallbacks generate a simple, legitimate tabular-only probability model trained from `train.csv` metadata and applied to `test.csv`. This keeps the same downstream interface (`public_sub_*.target` aligned to `sub`) while producing non-constant predictions that should substantially improve AUC. I also hard-align `image_name` ordering to the sample submission to avoid any accidental row misalignment. The submission path/format remains unchanged (`submission.csv` with `image_name,target`).'
- What this solution (achieved 0.66743) has done: 'Your score gap is large (0.66696 vs 0.9421), and right now the ensemble is unintentionally averaging multiple *identical* fallback models, which collapses diversity and limits AUC. I keep the ensemble formula exactly the same, but make the four fallback “public_sub_*” inputs legitimately different by training the same metadata-only logistic regression with different (deterministic) patient-group CV splits and using out-of-fold stacking to generate test probabilities. This preserves the core logic (a weighted blend of four `target` columns) while increasing predictive power through mild diversity, and keeps alignment strictly to `sample_submission.csv` order to avoid any AUC loss from row misordering. The code still runs end-to-end within constraints and writes a valid `submission.csv` with `image_name,target`.'
- What this solution (achieved 0.67157) has done: 'I keep your ensemble equation exactly as-is, but make each fallback “public_sub_*” file legitimately different so the blend gains diversity and AUC rises toward your 0.9421 target. The smallest safe change is to keep the same metadata-only LogisticRegression pipeline, yet train multiple variants with different deterministic feature sets and CV strategies (still group-aware by patient), then use their test probabilities as the stand-ins for the missing external submissions. I also add a lightweight probability clipping to [0,1] and keep strict `image_name` alignment to `sample_submission.csv` to avoid accidental row-order AUC loss. The script still run end-to-end and write a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.66345) has done: 'To move your AUC upward toward 0.9421 without changing the ensemble equation or switching away from the same LogisticRegression-on-metadata core logic, I make the fallback models genuinely stronger and more diverse by (1) training on `diagnosis` when available (train-only) using target-mean encoding computed out-of-fold by `patient_id` GroupKFold (leakage-safe), and (2) giving each fallback variant a different deterministic encoding strength/noise to preserve diversity for the fixed-weight blend. This is a minimal extension of the existing tabular fallback (still logistic regression, still group-aware CV), but it adds a high-signal feature that metadata-only models otherwise miss. I also hard-align train/test categorical levels and keep strict `image_name` ordering via the sample submission (as you already do) to avoid accidental misalignment loss. The script still run end-to-end and write `submission.csv` in the required format.'
- What this solution (achieved 0.66867) has done: 'Your current score is far below the target, so we should cautiously increase AUC with minimal logic changes. The biggest issue in your fallback is that `test_diag_te` is never actually mapped using `diag_map` (it’s currently just the global mean), which wastes the strongest tabular signal and keeps the ensemble weak. I fix `test_diag_te` to use the learned smoothed means (and keep the existing optional noise for diversity), and I also make the train/test “unknown” handling consistent. This preserves your overall approach (same weighted blend, same logistic regression fallback structure) while producing a meaningful improvement without changing the submission format or paths.'
- What this solution (achieved 0.6612) has done: 'Your gap to the target is large (0.66867 vs 0.9421), so we should increase AUC while keeping your fixed-weight blend and the same LogisticRegression fallback core logic. The biggest boost available with minimal disruption is to make the fallback model better match AUC by adding a small, leakage-safe out-of-fold target-encoding for `anatom_site_general_challenge` (in addition to your existing `diagnosis` target encoding), which is strong signal in this dataset. I also slightly increase model capacity only via `max_iter` (same solver/model) to reduce underfitting/failed convergence without changing the approach. All outputs remain aligned to `sample_submission.csv` order and the script still writes a valid `submission.csv`.'
- What this solution (achieved 0.65878) has done: 'Your current score (0.6612) is far below the target (0.9421), so we should make a small, legitimate improvement that increases AUC while keeping your fixed-weight ensemble and LogisticRegression-based fallback core logic unchanged. The biggest safe gain is to strengthen the fallback models by adding a leakage-safe out-of-fold target-encoding for `sex` and a simple interaction target-encoding for `sex|site`, both computed with GroupKFold by `patient_id` like your existing encodings. This adds signal with minimal disruption (same training approach and same ensemble equation), and keeps strict `image_name` alignment to the sample submission. The rest of the pipeline (file reading, blending, clipping, submission writing) remains the same.'
- What this solution (achieved 0.65878) has done: 'Your current AUC is far below the target, so we should improve prediction quality with minimal, metric-aligned changes while keeping your fixed-weight ensemble and LogisticRegression fallback approach intact. The biggest issue is that your fallback trains five separate fold-models but averages only the **test** predictions; it never uses out-of-fold (OOF) calibration, so the probabilities can be poorly calibrated and less rank-optimal. I add a lightweight, leakage-safe **OOF Platt calibration** (a second LogisticRegression on the fold OOF predictions) and apply it to the averaged test probabilities; this preserves the same core model family and training loop structure, but typically improves AUC ranking. I also ensure all fallbacks and the final submission are strictly aligned to `sample_submission.csv` order and keep clipping to `[0,1]`.'
- What this solution (achieved 0.65889) has done: 'Your current gap is large (0.65878 vs 0.9421), so we should increase AUC with minimal changes while keeping the same fixed-weight ensemble in cell 2 and the same LogisticRegression-based fallback approach. The main weakness is that the fallback’s calibration is trained with `class_weight="balanced"`, which can distort probability ranking for AUC; we keep calibration but train the Platt calibrator without class weights. We also make the fold-averaging more AUC-friendly by averaging logits (then applying sigmoid) instead of averaging probabilities (this is still the same CV+LR approach, just a numerically different aggregation). Finally, we keep strict `image_name` alignment and output format unchanged so you get a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd



## === cell 1
import os

sub = pd.read_csv("../input/siim-isic-melanoma-classification/sample_submission.csv")


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def _oof_target_mean_encode(
    train_series: pd.Series,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    seed: int,
    noise: float = 0.0,
    smooth: float = 20.0,
):
    """
    Leakage-safe target mean encoding computed OOF using GroupKFold by patient_id.

    Returns:
      enc_oof: np.ndarray aligned to train_series
      global_mean: float
      full_means: pd.Series mapping category -> smoothed mean using full train (for test transform)
    """
    from sklearn.model_selection import GroupKFold

    s = train_series.fillna("unknown").astype(str)
    y = y.astype(np.float64)
    global_mean = float(np.mean(y))

    gkf = GroupKFold(n_splits=n_splits)
    enc_oof = np.zeros(len(s), dtype=np.float64)

    rng = np.random.RandomState(seed)

    for tr_idx, va_idx in gkf.split(np.zeros(len(s)), y, groups=groups):
        s_tr = s.iloc[tr_idx]
        y_tr = y[tr_idx]

        stats = (
            pd.DataFrame({"cat": s_tr.values, "y": y_tr})
            .groupby("cat")["y"]
            .agg(["mean", "count"])
        )
        smooth_mean = (stats["count"] * stats["mean"] + smooth * global_mean) / (
            stats["count"] + smooth
        )

        s_va = s.iloc[va_idx].values
        enc = (
            pd.Series(s_va)
            .map(smooth_mean)
            .fillna(global_mean)
            .values.astype(np.float64)
        )
        if noise > 0:
            enc = np.clip(enc + rng.normal(0.0, noise, size=enc.shape), 0.0, 1.0)
        enc_oof[va_idx] = enc

    stats_full = (
        pd.DataFrame({"cat": s.values, "y": y})
        .groupby("cat")["y"]
        .agg(["mean", "count"])
    )
    full_means = (stats_full["count"] * stats_full["mean"] + smooth * global_mean) / (
        stats_full["count"] + smooth
    )

    return enc_oof, global_mean, full_means


def _build_tabular_fallback_predictions(
    template_sub: pd.DataFrame,
    seed: int = 0,
    variant: str = "full",
    n_splits: int = 5,
) -> pd.DataFrame:
    train_path = "../input/siim-isic-melanoma-classification/train.csv"
    test_path = "../input/siim-isic-melanoma-classification/test.csv"

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    test_df = template_sub[["image_name"]].merge(test_df, on="image_name", how="left")

    age_median = pd.to_numeric(train_df["age_approx"], errors="coerce").median()
    train_age = pd.to_numeric(train_df["age_approx"], errors="coerce").fillna(
        age_median
    )
    test_age = pd.to_numeric(test_df["age_approx"], errors="coerce").fillna(age_median)

    train_sex = train_df["sex"].fillna("unknown").replace("", "unknown").astype(str)
    test_sex = test_df["sex"].fillna("unknown").replace("", "unknown").astype(str)
    train_site = (
        train_df["anatom_site_general_challenge"]
        .fillna("unknown")
        .replace("", "unknown")
        .astype(str)
    )
    test_site = (
        test_df["anatom_site_general_challenge"]
        .fillna("unknown")
        .replace("", "unknown")
        .astype(str)
    )

    train_diag = (
        train_df.get("diagnosis", pd.Series(["unknown"] * len(train_df)))
        .fillna("unknown")
        .replace("", "unknown")
        .astype(str)
    )
    test_diag = (
        test_df.get("diagnosis", pd.Series(["unknown"] * len(test_df)))
        .fillna("unknown")
        .replace("", "unknown")
        .astype(str)
    )

    y_train = train_df["target"].astype(int).values
    groups = train_df["patient_id"].astype(str).fillna("unknown").values

    te_cfg = {
        "full": dict(smooth=25.0, noise=0.00),
        "full_polyage": dict(smooth=18.0, noise=0.005),
        "sex_site": dict(smooth=35.0, noise=0.00),
        "sex_site_diag": dict(smooth=12.0, noise=0.010),
        "age_only": dict(smooth=45.0, noise=0.00),
    }
    cfg = te_cfg.get(variant, dict(smooth=25.0, noise=0.00))

    diag_te_oof, diag_global, diag_map = _oof_target_mean_encode(
        train_series=train_diag,
        y=y_train,
        groups=groups,
        n_splits=n_splits,
        seed=seed,
        noise=float(cfg["noise"]),
        smooth=float(cfg["smooth"]),
    )
    site_te_oof, site_global, site_map = _oof_target_mean_encode(
        train_series=train_site.astype(str),
        y=y_train,
        groups=groups,
        n_splits=n_splits,
        seed=seed + 101,
        noise=0.0,
        smooth=max(10.0, float(cfg["smooth"]) * 0.8),
    )

    sex_te_oof, sex_global, sex_map = _oof_target_mean_encode(
        train_series=train_sex,
        y=y_train,
        groups=groups,
        n_splits=n_splits,
        seed=seed + 202,
        noise=0.0,
        smooth=max(10.0, float(cfg["smooth"]) * 1.2),
    )
    train_sex_site = (train_sex.astype(str) + "|" + train_site.astype(str)).astype(str)
    test_sex_site = (test_sex.astype(str) + "|" + test_site.astype(str)).astype(str)
    sex_site_te_oof, sex_site_global, sex_site_map = _oof_target_mean_encode(
        train_series=train_sex_site,
        y=y_train,
        groups=groups,
        n_splits=n_splits,
        seed=seed + 303,
        noise=0.0,
        smooth=max(10.0, float(cfg["smooth"]) * 0.9),
    )

    test_diag_te = test_diag.map(diag_map).fillna(diag_global).values.astype(np.float64)
    if float(cfg["noise"]) > 0:
        rng = np.random.RandomState(seed)
        test_diag_te = np.clip(
            test_diag_te
            + rng.normal(0.0, float(cfg["noise"]), size=test_diag_te.shape),
            0.0,
            1.0,
        )

    test_site_te = (
        test_site.astype(str)
        .map(site_map)
        .fillna(site_global)
        .values.astype(np.float64)
    )
    test_sex_te = (
        test_sex.astype(str).map(sex_map).fillna(sex_global).values.astype(np.float64)
    )
    test_sex_site_te = (
        test_sex_site.astype(str)
        .map(sex_site_map)
        .fillna(sex_site_global)
        .values.astype(np.float64)
    )

    if variant == "age_only":
        X_train = pd.DataFrame(
            {
                "age_approx": train_age.values,
                "diag_te": diag_te_oof,
                "site_te": site_te_oof,
                "sex_te": sex_te_oof,
                "sex_site_te": sex_site_te_oof,
            }
        )
        X_test = pd.DataFrame(
            {
                "age_approx": test_age.values,
                "diag_te": test_diag_te,
                "site_te": test_site_te,
                "sex_te": test_sex_te,
                "sex_site_te": test_sex_site_te,
            }
        )
    else:
        train_cat = pd.DataFrame({"sex": train_sex, "site": train_site})
        test_cat = pd.DataFrame({"sex": test_sex, "site": test_site})

        all_cat = pd.concat([train_cat, test_cat], axis=0, ignore_index=True)
        dummies = pd.get_dummies(all_cat, columns=["sex", "site"], dummy_na=False)

        X_train_cat = dummies.iloc[: len(train_df)].reset_index(drop=True)
        X_test_cat = dummies.iloc[len(train_df) :].reset_index(drop=True)

        if variant == "full_polyage":
            train_age_df = pd.DataFrame(
                {"age_approx": train_age.values, "age_approx_sq": (train_age.values**2)}
            )
            test_age_df = pd.DataFrame(
                {"age_approx": test_age.values, "age_approx_sq": (test_age.values**2)}
            )
        else:
            train_age_df = pd.DataFrame({"age_approx": train_age.values})
            test_age_df = pd.DataFrame({"age_approx": test_age.values})

        X_train = pd.concat(
            [
                train_age_df.reset_index(drop=True),
                pd.DataFrame(
                    {
                        "diag_te": diag_te_oof,
                        "site_te": site_te_oof,
                        "sex_te": sex_te_oof,
                        "sex_site_te": sex_site_te_oof,
                    }
                ),
                X_train_cat,
            ],
            axis=1,
        )
        X_test = pd.concat(
            [
                test_age_df.reset_index(drop=True),
                pd.DataFrame(
                    {
                        "diag_te": test_diag_te,
                        "site_te": test_site_te,
                        "sex_te": test_sex_te,
                        "sex_site_te": test_sex_site_te,
                    }
                ),
                X_test_cat,
            ],
            axis=1,
        )

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold

    def _make_clf(C: float):
        return Pipeline(
            steps=[
                ("scaler", StandardScaler(with_mean=False)),
                (
                    "lr",
                    LogisticRegression(
                        max_iter=1200,
                        class_weight="balanced",
                        solver="lbfgs",
                        random_state=seed,
                        C=C,
                    ),
                ),
            ]
        )

    uniq_groups = pd.unique(groups)
    rng2 = np.random.RandomState(seed)
    rng2.shuffle(uniq_groups)
    group_to_rank = {g: i for i, g in enumerate(uniq_groups)}
    group_rank = np.array([group_to_rank[g] for g in groups], dtype=np.int64)
    order = np.argsort(group_rank, kind="mergesort")

    X_train_ord = X_train.iloc[order].reset_index(drop=True)
    y_train_ord = y_train[order]
    groups_ord = groups[order]

    gkf = GroupKFold(n_splits=n_splits)

    C_map = {
        "full": 1.0,
        "full_polyage": 0.7,
        "sex_site": 1.2,
        "sex_site_diag": 0.9,
        "age_only": 0.6,
    }
    C = C_map.get(variant, 1.0)

    test_logit_accum = np.zeros(len(X_test), dtype=np.float64)
    oof_pred = np.zeros(len(X_train_ord), dtype=np.float64)

    for tr_idx, va_idx in gkf.split(X_train_ord, y_train_ord, groups=groups_ord):
        clf = _make_clf(C=C)
        clf.fit(X_train_ord.iloc[tr_idx], y_train_ord[tr_idx])

        oof_pred[va_idx] = clf.predict_proba(X_train_ord.iloc[va_idx])[:, 1].astype(
            np.float64
        )

        fold_test_p = clf.predict_proba(X_test)[:, 1].astype(np.float64)
        test_logit_accum += _logit(fold_test_p)

    test_pred = _sigmoid(test_logit_accum / float(n_splits))

    try:
        from sklearn.linear_model import LogisticRegression as LR

        calibrator = LR(
            solver="lbfgs",
            max_iter=500,
            random_state=seed,
            class_weight=None,
        )
        calibrator.fit(oof_pred.reshape(-1, 1), y_train_ord.astype(int))
        test_pred = calibrator.predict_proba(test_pred.reshape(-1, 1))[:, 1].astype(
            np.float64
        )
    except Exception:
        test_pred = test_pred.astype(np.float64)

    test_pred = np.clip(test_pred, 0.0, 1.0)

    fb = template_sub.copy()
    fb["target"] = test_pred
    return fb


def _read_or_fallback(
    path, template_df, seed: int = 0, variant: str = "full", n_splits: int = 5
):
    if os.path.exists(path):
        df = pd.read_csv(path)
        if "image_name" in df.columns:
            df = template_df[["image_name"]].merge(
                df[["image_name", "target"]], on="image_name", how="left"
            )
        else:
            df = df.copy()
            if len(df) != len(template_df):
                df = template_df.copy()
                df["target"] = np.nan
        df["target"] = pd.to_numeric(df["target"], errors="coerce").fillna(0.0)
        df["target"] = np.clip(df["target"].values.astype(np.float64), 0.0, 1.0)
        return df

    return _build_tabular_fallback_predictions(
        template_df, seed=seed, variant=variant, n_splits=n_splits
    )


public_sub_mean_9533 = _read_or_fallback(
    "../input/public-submission-melanoma-95/submission_mean.csv",
    sub,
    seed=11,
    variant="sex_site",
    n_splits=5,
)
public_sub_median_9533 = _read_or_fallback(
    "../input/public-submission-melanoma-95/submission_median.csv",
    sub,
    seed=13,
    variant="age_only",
    n_splits=5,
)
public_sub_meta_ens_9577 = _read_or_fallback(
    "../input/public-submission-melanoma-95/external_meta_ensembled.csv",
    sub,
    seed=17,
    variant="full_polyage",
    n_splits=5,
)
public_sub_9581 = _read_or_fallback(
    "../input/public-submission-melanoma-95/submission_9581.csv",
    sub,
    seed=19,
    variant="full",
    n_splits=5,
)

public_sub_tabular = _read_or_fallback(
    "../input/public-submission-melanoma-95/submission_tabular_only.csv",
    sub,
    seed=23,
    variant="age_only",
    n_splits=5,
)
public_sub_9619 = _read_or_fallback(
    "../input/public-submission-melanoma-95/submission_9619.csv",
    sub,
    seed=29,
    variant="full",
    n_splits=5,
)
public_sub_9606 = _read_or_fallback(
    "../input/public-submission-melanoma-95/submission_9606.csv",
    sub,
    seed=31,
    variant="full_polyage",
    n_splits=5,
)
public_sub_9603 = _read_or_fallback(
    "../input/public-submission-melanoma-95/submission_9603.csv",
    sub,
    seed=37,
    variant="sex_site",
    n_splits=5,
)



## === cell 2
sub.target = (
    public_sub_9619.target * 0.40
    + public_sub_9606.target * 0.20
    + public_sub_9603.target * 0.20
    + public_sub_tabular.target * 0.20
)

sub["target"] = np.clip(sub["target"].values.astype(np.float64), 0.0, 1.0)



## === cell 3
sub.head()
sub.to_csv("submission.csv", index=False)



## === cell 4
sub.head
