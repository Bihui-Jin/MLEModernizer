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

0.925558552950128

# 6. Current score

0.73442

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'The current notebook fails immediately because it tries to read OOF/submission blend files from a Kaggle Dataset path that doesn’t exist in your environment, so none of the downstream variables are defined and the optimizer crashes with empty inputs. I keep the same blending/weight-optimization core logic, but make it robust by (1) detecting whether those external blend files exist and (2) falling back to a valid, score-neutral baseline submission (all targets = mean train prevalence) when they don’t. I also fix the optimizer objective to actually *minimize* (negative AUC) rather than minimize AUC directly, and ensure the output CSV always matches `sample_submission.csv` ordering and has the correct columns and `.csv` suffix. This run end-to-end and always produce `submission.csv`.'
- What this solution (achieved 0.75821) has done: 'Your current 0.5 score comes from the fallback path that submits a constant probability (train prevalence), which yields random-ranking AUC. To move toward the 0.9256 target without changing your blending/weight-optimization core logic, I keep the same script but add a robust, metadata-only model fallback (still legitimate) that uses the provided CSV features to create non-constant predictions. This replaces only the “missing external blend files” behavior, producing a meaningful ranking (and thus higher AUC) while still writing a valid `submission.csv` aligned to `sample_submission.csv`. I also ensure categorical handling and missing values are consistent between train/test so the submission is stable.'
- What this solution (achieved 0.74671) has done: 'Your current score (0.75821) is far below the target (0.92556), so we should improve the fallback path (used when blend files are missing) while keeping your blending core logic unchanged. The smallest high-impact fix is to make the metadata fallback model ranking-stronger by (1) using a stratified CV OOF scheme to generate a calibrated meta-prediction without leakage and (2) using class-balancing for the rare-positive target, which typically improves AUC a lot for this dataset. We keep the same overall approach (metadata-only logistic regression pipeline) but make it more robust by adding a simple `sex` cleanup, using `saga` (supports sparse + class_weight well), and averaging test probabilities across folds. The output submission writing logic and file paths remain the same and still always produce `submission.csv`.'
- What this solution (achieved 0.75791) has done: 'To move your AUC closer to the 0.9256 target while preserving your core “blend-if-available else metadata-logreg” logic, I strengthen only the metadata fallback path (which is what you’re currently using) with a minimal, competition-appropriate change: add a second LogisticRegression with different regularization strength and average their fold-averaged test probabilities. This keeps the same model family, same preprocessing, same CV training loop, and same evaluation semantics, but typically improves ranking signal on this dataset without adding heavy computation. I also ensure `patient_id` is treated as categorical consistently (string) to avoid accidental numeric casting differences between train/test, which can silently hurt AUC. Submission writing remains identical and still produces `submission.csv` aligned to `sample_submission.csv`.'
- What this solution (achieved 0.35214) has done: 'To move your AUC upward toward the 0.9256 target without changing the overall “blend if available else metadata-logreg” core logic, I’m only strengthening the metadata fallback path (which you’re currently using). The smallest high-signal change is to add `diagnosis` as an additional categorical feature (train-only column) while keeping the exact same preprocessing + LogisticRegression + StratifiedKFold loop and submission-writing semantics. This competition’s AUC benefits a lot from `diagnosis` because it’s strongly correlated with malignancy, so this should close much of the gap while staying within your existing model family and training approach. I also keep all file paths and the output `submission.csv` schema/order aligned to `sample_submission.csv`.'
- What this solution (achieved 0.75791) has done: 'Your current big drop to 0.35214 is almost certainly coming from the new `diagnosis` feature: it exists only in train, so in test it becomes a constant “unknown”, which makes the model overfit to train-only categories and destroys ranking on the real test distribution. To move back up toward the 0.9256 target with minimal change and identical core logic, I keep the same metadata-only LogisticRegression + StratifiedKFold + micro-ensemble fallback, but remove `diagnosis` from the fallback features (and keep the rest unchanged). I also add a tiny safety guard to ensure `image_name` ordering matches `sample_submission.csv` exactly and that predictions are always finite/clipped. This should restore the previous ~0.75-ish behavior (closer to target than 0.35) without changing your blending path at all.'
- What this solution (achieved 0.74124) has done: 'Your current score (0.75791) is well below the target (0.92556), so we should cautiously improve the fallback (metadata-only) path that you’re actually using when blend files are missing. With minimal changes and the same preprocessing + LogisticRegression + StratifiedKFold training loop, the highest-impact improvement is to add a second LR with a different regularization strength *and* a mild interaction feature (`age_approx` binned) that often improves ranking AUC without changing the model family. I also switch the CV splitter to `StratifiedGroupKFold` grouped by `patient_id` to reduce leakage across folds (keeps the same CV training approach, but makes the learned ranking more robust and typically improves generalization). Submission writing/ordering stays identical and still writes a valid `submission.csv`.'
- What this solution (achieved 0.74337) has done: 'We keep your existing “blend if external files exist, else metadata-only LR with StratifiedGroupKFold” logic unchanged, and only make two small, score-relevant adjustments to strengthen the fallback ranking signal. First, we add two simple, non-leaky numeric features derived from `age_approx` (a missingness indicator and a scaled version) that LogisticRegression can exploit without changing the model family or training loop. Second, we slightly extend the LR micro-ensemble with one additional nearby regularization strength, which often improves AUC modestly while staying within the same approach and time limits. Submission writing, ordering, and file paths remain identical to ensure a valid `submission.csv`.'
- What this solution (achieved 0.7404) has done: 'We keep your overall “blend if available else metadata-only LR+StratifiedGroupKFold micro-ensemble” logic unchanged, but tighten a couple of score-relevant details in the fallback path to improve ranking AUC toward your target. Specifically, we (1) fix a subtle preprocessing issue by scaling numerics with `with_mean=True` (safe here because the numeric branch is dense), and (2) add two very small, non-leaky interaction features derived from existing metadata (`age_x_sex` and `site_x_sex`) to give LogisticRegression extra linear separability without changing the model family or training loop. We also increase `max_iter` slightly to ensure `saga` reliably converges (improves stability, not an approximation) and keep submission writing identical and aligned to `sample_submission.csv`. These are minimal changes focused on improving the fallback model’s AUC (your current score indicates you’re on the fallback path).'
- What this solution (achieved 0.74385) has done: 'Your current score (0.7404) is far below the target (0.9256), and since the external blend files are missing you’re relying entirely on the metadata fallback; the only practical way to move toward the target (without changing the core LR+CV approach) is to strengthen signal inside that same fallback. I keep the exact same preprocessing + LogisticRegression micro-ensemble + StratifiedGroupKFold training loop, but (1) add a few simple, non-leaky numeric transforms from existing columns (`log1p(age)`, squared age, and sex missingness) that LR can exploit, and (2) slightly widen the existing LR C-grid with one additional nearby value to improve ranking robustness. I also switch the numeric scaler to `with_mean=False` to avoid sparse/dense interaction issues inside the ColumnTransformer output (this is a stability/correctness fix that can otherwise silently hurt performance). Submission writing stays identical and still guarantees `submission.csv` aligned to `sample_submission.csv`.'
- What this solution (achieved 0.68391) has done: 'We keep your existing “blend if external files exist, else metadata-only LR micro-ensemble with StratifiedGroupKFold” core logic unchanged, and only make a couple of small, score-relevant adjustments inside the metadata fallback (which is what you’re using). Specifically, we add `PolynomialFeatures(degree=2, interaction_only=True)` on the numeric branch to capture a few extra linearizable interactions (still a linear LR model downstream), and we slightly extend the LR C-grid with one additional nearby value to improve ranking robustness. We also ensure the numeric branch outputs are consistently scaled and sparse-safe, and keep the exact same submission writing/alignment to `sample_submission.csv`. These minimal changes are aimed at nudging AUC upward from ~0.744 toward your target without changing the overall approach.'
- What this solution (achieved 0.73442) has done: 'I remove the heavy 200× L-BFGS blending optimization (which repeatedly calls AUC on ~29k rows) and replace it with a provably equivalent result for this specific objective: because AUC depends only on ranking, any strictly-positive scaling of weights is identical, so we can directly compute the AUC-optimal nonnegative weights by fitting a nonnegative logistic regression on the OOF predictions (same inputs, same target) and then renormalizing to sum to 1. For the metadata fallback, I eliminate repeated feature transformations inside each fold by pre-fitting the `ColumnTransformer` once on all rows (this is deterministic and does not change the model family/loop), and I reuse the already-transformed sparse matrices for every C value and fold. I also avoid repeated disk I/O (re-reading CSVs) and avoid unnecessary DataFrame sorts/prints that add overhead but don’t affect outputs. These changes keep the same prediction semantics (linear blend of base model predictions; LR fallback with same CV loop and same solvers) while cutting runtime to comfortably fit under 600 seconds.'

# 9. Code solution

## === cell 0
import os
import pandas as pd

"""
Same Seed: 42, 
Base Model: E6
Top Model - GAP, ATTENTION, GEM

This notebook originally depends on an external Kaggle Dataset:
'../input/efficientnetb6seed-42-oof-prediction/'.
In this environment that directory may not exist, so we:
- try to load the blend files if present
- otherwise fall back to a valid metadata-only submission to ensure a .csv is produced
"""

BLEND_DIR = "../input/efficientnetb6seed-42-oof-prediction/"

BASE_DATA_DIR = "/kaggle/data"
TRAIN_CSV = os.path.join(BASE_DATA_DIR, "train.csv")
TEST_CSV = os.path.join(BASE_DATA_DIR, "test.csv")
SAMPLE_SUB_CSV = os.path.join(BASE_DATA_DIR, "sample_submission.csv")


def _safe_read_csv(path: str):
    return pd.read_csv(path) if os.path.exists(path) else None


oof_one = _safe_read_csv(os.path.join(BLEND_DIR, "oof_e6_attn_gap_seed_42.csv"))
test_one = _safe_read_csv(os.path.join(BLEND_DIR, "s_e6_attn_gap_seed_42.csv"))

oof_two = _safe_read_csv(os.path.join(BLEND_DIR, "oof_e6_attn_seed_42.csv"))
test_two = _safe_read_csv(os.path.join(BLEND_DIR, "s_e6_attn_seed_42.csv"))

oof_three = _safe_read_csv(os.path.join(BLEND_DIR, "oof_e6_gem_seed_42.csv"))
test_three = _safe_read_csv(os.path.join(BLEND_DIR, "s_e6_gem_seed_42.csv"))

oof_four = _safe_read_csv(os.path.join(BLEND_DIR, "oof_e6_our_attn_seed_42.csv"))
test_four = _safe_read_csv(os.path.join(BLEND_DIR, "s_e6_our_attn_seed_42.csv"))

oof_five = _safe_read_csv(os.path.join(BLEND_DIR, "oof_e6_gap_seed_42.csv"))
test_five = _safe_read_csv(os.path.join(BLEND_DIR, "s_e6_gap_seed_42.csv"))

blend_files_available = all(
    df is not None
    for df in [
        oof_one,
        test_one,
        oof_two,
        test_two,
        oof_three,
        test_three,
        oof_four,
        test_four,
        oof_five,
        test_five,
    ]
)

print("Blend files available:", blend_files_available)
if not blend_files_available:
    print(
        f"WARNING: Missing external blend files under {BLEND_DIR}. Will fit a metadata-only fallback model and write submission.csv."
    )



## === cell 1
if blend_files_available:
    print("Blend files loaded; skipping head() prints to save time.")
else:
    print("Blend files not available; will use fallback model.")



## === cell 2
import numpy as np
from sklearn.metrics import roc_auc_score


def _get_col(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


if blend_files_available:
    base_order = oof_one[["image_name"]].copy()
    base_order["__row__"] = np.arange(len(base_order), dtype=np.int32)
    base_order = base_order.set_index("image_name")

    def _align(df: pd.DataFrame) -> pd.DataFrame:
        return (
            df.set_index("image_name")
            .join(base_order, how="inner")
            .sort_values("__row__")
            .drop(columns="__row__")
            .reset_index()
        )

    oof_one = _align(oof_one)
    oof_two = _align(oof_two)
    oof_three = _align(oof_three)
    oof_four = _align(oof_four)
    oof_five = _align(oof_five)

    test_order = test_one[["image_name"]].copy()
    test_order["__row__"] = np.arange(len(test_order), dtype=np.int32)
    test_order = test_order.set_index("image_name")

    def _align_test(df: pd.DataFrame) -> pd.DataFrame:
        return (
            df.set_index("image_name")
            .join(test_order, how="inner")
            .sort_values("__row__")
            .drop(columns="__row__")
            .reset_index()
        )

    test_one = _align_test(test_one)
    test_two = _align_test(test_two)
    test_three = _align_test(test_three)
    test_four = _align_test(test_four)
    test_five = _align_test(test_five)

    y_col = _get_col(oof_one, ["target"])
    p1_col = _get_col(oof_one, ["pred", "oof", "prediction", "target_pred"])
    p2_col = _get_col(oof_two, ["pred", "oof", "prediction", "target_pred"])
    p3_col = _get_col(oof_three, ["pred", "oof", "prediction", "target_pred"])
    p4_col = _get_col(oof_four, ["pred", "oof", "prediction", "target_pred"])
    p5_col = _get_col(oof_five, ["pred", "oof", "prediction", "target_pred"])

    t1_col = _get_col(test_one, ["target", "pred", "prediction"])
    t2_col = _get_col(test_two, ["target", "pred", "prediction"])
    t3_col = _get_col(test_three, ["target", "pred", "prediction"])
    t4_col = _get_col(test_four, ["target", "pred", "prediction"])
    t5_col = _get_col(test_five, ["target", "pred", "prediction"])

    missing = [
        name
        for name, col in [
            ("y_col", y_col),
            ("p1_col", p1_col),
            ("p2_col", p2_col),
            ("p3_col", p3_col),
            ("p4_col", p4_col),
            ("p5_col", p5_col),
            ("t1_col", t1_col),
            ("t2_col", t2_col),
            ("t3_col", t3_col),
            ("t4_col", t4_col),
            ("t5_col", t5_col),
        ]
        if col is None
    ]
    if missing:
        raise ValueError(
            f"Required columns not found in blend files: {missing}. Available cols example: {oof_one.columns.tolist()}"
        )

    y_train = oof_one[y_col].to_numpy(dtype=np.int8)

    blend_train = np.vstack(
        [
            oof_one[p1_col].to_numpy(dtype=np.float64),
            oof_two[p2_col].to_numpy(dtype=np.float64),
            oof_three[p3_col].to_numpy(dtype=np.float64),
            oof_four[p4_col].to_numpy(dtype=np.float64),
            oof_five[p5_col].to_numpy(dtype=np.float64),
        ]
    )

    blend_test = np.vstack(
        [
            test_one[t1_col].to_numpy(dtype=np.float64),
            test_two[t2_col].to_numpy(dtype=np.float64),
            test_three[t3_col].to_numpy(dtype=np.float64),
            test_four[t4_col].to_numpy(dtype=np.float64),
            test_five[t5_col].to_numpy(dtype=np.float64),
        ]
    )
else:
    y_train = None
    blend_train = None
    blend_test = None



## === cell 3
bestWght = None
bestSC = None

if blend_files_available:
    from sklearn.linear_model import LogisticRegression

    X_blend = blend_train.T  # shape (n_samples, n_models)
    lr_blend = LogisticRegression(
        penalty="l2",
        solver="lbfgs",
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
    )
    lr_blend.fit(X_blend, y_train)

    w = lr_blend.coef_.ravel().astype(np.float64)
    w = np.clip(w, 0.0, None)  # enforce nonnegative weights as in original bounds

    if not np.isfinite(w).all() or w.sum() <= 0:
        w = np.ones(X_blend.shape[1], dtype=np.float64)

    bestWght = w / w.sum()

    train_blend_pred = X_blend @ bestWght
    bestSC = roc_auc_score(y_train, train_blend_pred)
    print("\n Ensemble Score (OOF AUC): {best_score}".format(best_score=float(bestSC)))
    print("\n Best Weights (sum=1): {weights}".format(weights=bestWght))

    test_prices = (blend_test.T @ bestWght).astype(np.float64)
    test_prices = np.clip(test_prices, 0.0, 1.0)

else:
    try:
        from sklearnex import patch_sklearn

        patch_sklearn()
        _SKLEX_OK = True
    except Exception:
        _SKLEX_OK = False

    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedGroupKFold
    from scipy import sparse

    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)

    feature_cols = [
        "sex",
        "age_approx",
        "anatom_site_general_challenge",
        "patient_id",
        "age_bin",
        "age_missing",
        "age_scaled",
        "age_log",
        "age_sq",
        "sex_missing",
        "age_x_sex",
        "site_x_sex",
    ]

    for _df in (train_df, test_df):
        if "sex" in _df.columns:
            _df["sex"] = _df["sex"].replace("", np.nan)
        if "patient_id" in _df.columns:
            _df["patient_id"] = _df["patient_id"].astype(str)

        age = pd.to_numeric(_df.get("age_approx"), errors="coerce")
        _df["age_bin"] = pd.cut(
            age,
            bins=[-np.inf, 20, 30, 40, 50, 60, 70, 80, np.inf],
            labels=[
                "<=20",
                "21-30",
                "31-40",
                "41-50",
                "51-60",
                "61-70",
                "71-80",
                "81+",
            ],
        ).astype("object")

        _df["age_missing"] = age.isna().astype(np.int8)
        _df["age_scaled"] = age
        _df["age_log"] = np.log1p(age.clip(lower=0))
        _df["age_sq"] = age * age
        _df["sex_missing"] = _df["sex"].isna().astype(np.int8)

        sex_s = _df.get("sex").astype("object")
        _df["age_x_sex"] = (
            _df["age_bin"].astype("object").astype(str) + "_" + sex_s.astype(str)
        ).astype("object")
        _df["site_x_sex"] = (
            _df["anatom_site_general_challenge"].astype("object").astype(str)
            + "_"
            + sex_s.astype(str)
        ).astype("object")

        _df["age_x_sex"] = _df["age_x_sex"].replace(
            ["nan_nan", "nan_None", "None_nan"], np.nan
        )
        _df["site_x_sex"] = _df["site_x_sex"].replace(
            ["nan_nan", "nan_None", "None_nan"], np.nan
        )

    X_all = train_df[feature_cols].copy()
    y = train_df["target"].astype(np.int8).to_numpy()
    X_test = test_df[feature_cols].copy()

    numeric_cont_features = ["age_approx", "age_scaled", "age_log", "age_sq"]
    numeric_indicator_features = ["age_missing", "sex_missing"]
    categorical_features = [
        c
        for c in feature_cols
        if c not in (numeric_cont_features + numeric_indicator_features)
    ]

    numeric_cont_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False, with_std=True)),
            (
                "poly",
                PolynomialFeatures(degree=2, include_bias=False, interaction_only=True),
            ),
        ]
    )

    numeric_ind_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num_cont", numeric_cont_transformer, numeric_cont_features),
            ("num_ind", numeric_ind_transformer, numeric_indicator_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

    def _make_lr(C: float) -> LogisticRegression:
        return LogisticRegression(
            max_iter=1400,
            solver="saga",
            class_weight="balanced",
            n_jobs=None,
            random_state=42,
            C=C,
        )

    Cs = [1.0, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
    models = [_make_lr(C) for C in Cs]

    groups = train_df["patient_id"].astype(str).to_numpy()
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    X_all_tr = preprocessor.fit_transform(X_all, y)
    X_test_tr = preprocessor.transform(X_test)

    if sparse.issparse(X_all_tr):
        X_all_tr = X_all_tr.tocsr()
    if sparse.issparse(X_test_tr):
        X_test_tr = X_test_tr.tocsr()

    test_pred = np.zeros(len(test_df), dtype=np.float64)
    oof_pred = np.zeros(len(train_df), dtype=np.float64)

    for fold, (tr_idx, va_idx) in enumerate(
        sgkf.split(X_all, y, groups=groups), start=1
    ):
        X_tr = X_all_tr[tr_idx]
        y_tr = y[tr_idx]
        X_va = X_all_tr[va_idx]
        y_va = y[va_idx]

        va_proba_ens = np.zeros(len(va_idx), dtype=np.float64)
        te_proba_ens = np.zeros(len(test_df), dtype=np.float64)

        for clf in models:
            clf.fit(X_tr, y_tr)
            va_proba_ens += clf.predict_proba(X_va)[:, 1].astype(np.float64) / len(
                models
            )
            te_proba_ens += clf.predict_proba(X_test_tr)[:, 1].astype(np.float64) / len(
                models
            )

        oof_pred[va_idx] = va_proba_ens
        test_pred += te_proba_ens / sgkf.n_splits

        fold_auc = roc_auc_score(y_va, va_proba_ens)
        print(f"Fold {fold} AUC: {fold_auc:.5f}")

    oof_auc = roc_auc_score(y, oof_pred)
    print(f"OOF AUC (metadata fallback): {oof_auc:.5f}")

    test_prices = np.nan_to_num(
        test_pred, nan=float(np.mean(y)), posinf=1.0, neginf=0.0
    ).astype(np.float64)
    test_prices = np.clip(test_prices, 0.0, 1.0)



## === cell 4
sample_sub = pd.read_csv(SAMPLE_SUB_CSV)

if blend_files_available:
    pred_df = pd.DataFrame(
        {"image_name": test_one["image_name"].values, "target": test_prices}
    )
    sub = sample_sub[["image_name"]].merge(pred_df, on="image_name", how="left")
else:
    pred_df = pd.DataFrame(
        {"image_name": test_df["image_name"].values, "target": test_prices}
    )
    sub = sample_sub[["image_name"]].merge(pred_df, on="image_name", how="left")

if sub["target"].isna().any():
    sub["target"] = sub["target"].fillna(float(sub["target"].mean()))

sub = sub.set_index("image_name").loc[sample_sub["image_name"].values].reset_index()

sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())



## === cell 5
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 4))
plt.hist(sub["target"].values, bins=100)
plt.ylim((0, max(10, int(len(sub) * 0.05))))
plt.title("Submission target distribution")
plt.show()
