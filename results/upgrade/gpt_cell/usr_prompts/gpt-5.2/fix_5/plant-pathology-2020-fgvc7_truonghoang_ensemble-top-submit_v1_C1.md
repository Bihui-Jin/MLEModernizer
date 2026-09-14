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
Detect apple diseases from images.

## Metric
Mean column-wise ROC AUC.

## Submission Format
For each image_id in the test set, you must predict a probability for each target variable. The file should contain a header and have the following format:

```
image_id,
test_0,0.25,0.25,0.25,0.25
test_1,0.25,0.25,0.25,0.25
test_2,0.25,0.25,0.25,0.25
etc.
```

## Dataset
Given a photo of an apple leaf, can you accurately assess its health? This competition will challenge you to distinguish between leaves which are healthy, those which are infected with apple rust, those that have apple scab, and those with more than one disease.

**train.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

**images**

A folder containing the train and test images, in jpg format.

**test.csv**

- `image_id`: the foreign key

**sample_submission.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

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
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        input/
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        working/
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
```

-> data/plant-pathology-2020-fgvc7/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/plant-pathology-2020-fgvc7/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/plant-pathology-2020-fgvc7/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> (stopped after 10 files for performance)

# 5. Target score

0.96796

# 6. Current score

0.64645

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'Diagnosis: Cell 1 crashes with `FileNotFoundError` because it tries to load five external ensemble submission CSVs from `../input/plantpathology/`, a directory/files that do not exist in this environment. The available dataset provides only the competition files (e.g., `../input/plant-pathology-2020-fgvc7/sample_submission.csv`) but not those precomputed model submissions. To keep cell 2’s averaging logic intact (it expects `sub1..sub4` with the standard columns), we should create fallback DataFrames with the correct schema and deterministic values when those CSVs are missing.  

Patch summary: In cell 1 only, wrap each `pd.read_csv` in a safe loader that falls back to copying `sample_submission.csv` and setting class probabilities to a uniform distribution (0.25 each) if the target file is absent. This preserves the interface (`sub1..sub5` exist with expected columns) so cell 2 can run unchanged.  

Updated cells: Only cell 1 is modified.  

Compatibility notes for cell k+1: Cell 2 still find `sub1`, `sub2`, `sub3`, `sub4` with columns `healthy`, `multiple_diseases`, `rust`, `scab`, and aligned row order/length matching `sample_submission.csv`. `sub5` is also created to preserve the original variable, though it is not used in cell 2.  

Assumptions: When external ensemble CSVs are unavailable, using a uniform probability fallback is acceptable to unblock execution and maintain deterministic behavior without changing downstream averaging semantics.'
- What this solution (achieved 0.64273) has done: 'Your current 0.5 score comes from the uniform 0.25 fallback predictions when the external ensemble CSVs are missing. To move toward the 0.96796 target without changing the “average submissions” core logic, the minimal legitimate improvement is to replace the fallback with a lightweight image-based baseline trained on the provided train.csv/images and then use its test probabilities as stand-ins for the missing `sub1..sub4`. This preserves the downstream averaging semantics (cell 2 remains an averaging step) while producing informative probabilities from available data. The model is a simple scikit-learn multinomial logistic regression on color-histogram features extracted from the JPGs, which is fast enough and uses only installed packages plus PIL (available in Kaggle). The submission format and column alignment are kept identical to sample_submission.csv.'
- What this solution (achieved 0.59161) has done: 'Your current baseline uses only global RGB histograms; that’s often too weak for this leaf-disease task, keeping AUC far below your 0.96796 target. To move the score up while preserving the same core “train a simple scikit-learn model and average sub1..sub4” logic, I minimally strengthen the feature extractor by adding coarse spatial information (2×2 grid RGB histograms) and a small amount of regularization tuning consistent with the same LogisticRegression approach. I also fit a StandardScaler (within a Pipeline) so the classifier sees better-conditioned features without changing the modeling family. The rest of the pipeline—including the safe fallback mechanism and the averaging in cell 2—stays intact and still writes a valid `submission.csv`.'
- What this solution (achieved 0.64645) has done: 'I keep your core approach (extract histogram features → multinomial LogisticRegression → use that as a stand-in for missing ensemble CSVs → average sub1..sub4) exactly the same, but make the fallback baseline slightly more predictive with minimal changes. Specifically, I add a second feature block based on HSV histograms (global + same 2×2 grid) alongside your existing RGB histograms; this still remains “simple histogram features → LR” but usually improves separability for plant disease colors. I also switch the LR regularization to a slightly stronger value (small C reduction) to improve generalization, and I compute the trained baseline once and reuse it for sub1..sub5 to avoid redundant retraining (same predictions, faster/more stable runtime). The submission format, column order, and averaging semantics remain unchanged.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd


DATA_ROOT = "../input/plant-pathology-2020-fgvc7"
IMG_DIR = os.path.join(DATA_ROOT, "images")
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
TEST_CSV = os.path.join(DATA_ROOT, "test.csv")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")

TARGET_COLS = ["healthy", "multiple_diseases", "rust", "scab"]

_BASELINE_PRED_CACHE = None


def _load_image_rgb(path):
    from PIL import Image

    with Image.open(path) as im:
        im = im.convert("RGB")
        return im


def _rgb_hist(arr, bins=16):
    feats = []
    for ch in range(3):
        h, _ = np.histogram(arr[..., ch], bins=bins, range=(0, 256))
        feats.append(h.astype(np.float32))
    x = np.concatenate(feats)
    s = float(x.sum())
    if s > 0:
        x /= s
    return x


def _hsv_hist(arr_rgb_uint8, bins_h=16, bins_sv=16):
    """
    Change (score-improving, same core logic):
    Add HSV histogram block (global/grid), which often better captures disease coloration.
    This preserves the same "histograms -> LogisticRegression" modeling family.
    Output dim: bins_h + bins_sv + bins_sv (H, S, V hist lengths).
    """
    from PIL import Image

    im = Image.fromarray(arr_rgb_uint8, mode="RGB").convert("HSV")
    arr = np.asarray(im, dtype=np.uint8)

    h_hist, _ = np.histogram(arr[..., 0], bins=bins_h, range=(0, 256))
    s_hist, _ = np.histogram(arr[..., 1], bins=bins_sv, range=(0, 256))
    v_hist, _ = np.histogram(arr[..., 2], bins=bins_sv, range=(0, 256))

    x = np.concatenate(
        [
            h_hist.astype(np.float32),
            s_hist.astype(np.float32),
            v_hist.astype(np.float32),
        ]
    )
    s = float(x.sum())
    if s > 0:
        x /= s
    return x


def _rgb_hsv_hist_features(
    img_path, size=160, bins_rgb=16, grid=2, bins_h=16, bins_sv=16
):
    """
    Change (score-improving, same core logic):
    Keep your RGB global + grid histograms and append HSV global + grid histograms.
    This is still a lightweight, deterministic feature extractor.
    """
    im = _load_image_rgb(img_path)
    im = im.resize((size, size))
    arr = np.asarray(im, dtype=np.uint8)

    feats = []
    feats.append(_rgb_hist(arr, bins=bins_rgb))
    feats.append(_hsv_hist(arr, bins_h=bins_h, bins_sv=bins_sv))

    if grid and grid > 1:
        h = size // grid
        w = size // grid
        for gy in range(grid):
            for gx in range(grid):
                y0, y1 = gy * h, (gy + 1) * h
                x0, x1 = gx * w, (gx + 1) * w
                patch = arr[y0:y1, x0:x1, :]
                feats.append(_rgb_hist(patch, bins=bins_rgb))
                feats.append(_hsv_hist(patch, bins_h=bins_h, bins_sv=bins_sv))

    x = np.concatenate(feats).astype(np.float32)
    return x


def _build_features(
    df, img_dir=IMG_DIR, size=160, bins_rgb=16, grid=2, bins_h=16, bins_sv=16
):
    per_block = (3 * bins_rgb) + (bins_h + 2 * bins_sv)
    n_blocks = 1 + (grid * grid if grid and grid > 1 else 0)
    feat_dim = n_blocks * per_block

    X = np.zeros((len(df), feat_dim), dtype=np.float32)
    missing = 0
    for i, image_id in enumerate(df["image_id"].values):
        img_path = os.path.join(img_dir, f"{image_id}.jpg")
        if not os.path.exists(img_path):
            missing += 1
            continue
        X[i] = _rgb_hsv_hist_features(
            img_path,
            size=size,
            bins_rgb=bins_rgb,
            grid=grid,
            bins_h=bins_h,
            bins_sv=bins_sv,
        )
    return X, missing


def _train_and_predict_probs():
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)

    X_train, _ = _build_features(
        train_df, size=160, bins_rgb=16, grid=2, bins_h=16, bins_sv=16
    )
    X_test, _ = _build_features(
        test_df, size=160, bins_rgb=16, grid=2, bins_h=16, bins_sv=16
    )

    y = train_df[TARGET_COLS].values.astype(np.int64)
    y_class = y.argmax(axis=1)

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    clf = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            (
                "lr",
                LogisticRegression(
                    multi_class="multinomial",
                    solver="lbfgs",
                    max_iter=800,
                    C=1.0,
                    n_jobs=1,
                    random_state=42,
                ),
            ),
        ]
    )

    clf.fit(X_train, y_class)
    proba = clf.predict_proba(X_test)  # shape (n_test, 4)

    classes = clf.named_steps["lr"].classes_

    aligned = np.zeros((len(test_df), len(TARGET_COLS)), dtype=np.float32)
    for j, cls in enumerate(classes):
        aligned[:, int(cls)] = proba[:, j]

    out = pd.DataFrame({"image_id": test_df["image_id"].values})
    for k, c in enumerate(TARGET_COLS):
        out[c] = aligned[:, k].astype(np.float32)

    out[TARGET_COLS] = out[TARGET_COLS].clip(0.0, 1.0)
    return out


def _safe_read_submission(path, template_path=SAMPLE_SUB):
    """
    Preserve original interface: try reading external submission; if missing,
    use our trained baseline predictions (informative) instead of uniform 0.25.

    Change (stability/runtime, same semantics):
    Train the baseline only once and reuse cached predictions for all missing subs.
    """
    global _BASELINE_PRED_CACHE

    if os.path.exists(path):
        df = pd.read_csv(path)
        for c in ["image_id"] + TARGET_COLS:
            if c not in df.columns:
                raise ValueError(f"Submission at {path} missing required column: {c}")
        return df[["image_id"] + TARGET_COLS].copy()

    if _BASELINE_PRED_CACHE is None:
        _BASELINE_PRED_CACHE = _train_and_predict_probs()

    pred = _BASELINE_PRED_CACHE.copy()

    tmpl = pd.read_csv(template_path)[["image_id"]].copy()
    pred = tmpl.merge(pred, on="image_id", how="left")
    pred[TARGET_COLS] = pred[TARGET_COLS].fillna(0.25)
    return pred[["image_id"] + TARGET_COLS].copy()


sub1 = _safe_read_submission(
    "../input/plantpathology/effnet-fastai-folds-x5_version3.csv"
)
sub2 = _safe_read_submission(
    "../input/plantpathology/fork-of-plant-2020-tpu-915e9c_version1.csv"
)
sub3 = _safe_read_submission(
    "../input/plantpathology/plant-pathology-pytorch-efficientnet-b4-gpu_version_7.csv"
)
sub4 = _safe_read_submission(
    "../input/plantpathology/public-first-score-tpu-incepresnetv2-enb7_version8.csv"
)
sub5 = _safe_read_submission(
    "../input/plantpathology/tpu-ensemble-effnb7-effnb6-inceptresnetv2-etc_verion13.csv"
)



## === cell 1
import pandas as pd

sub = pd.read_csv("../input/plant-pathology-2020-fgvc7/sample_submission.csv")

sub.healthy = (sub1.healthy + sub2.healthy + sub3.healthy + sub4.healthy) / 4
sub.multiple_diseases = (
    sub1.multiple_diseases
    + sub2.multiple_diseases
    + sub3.multiple_diseases
    + sub4.multiple_diseases
) / 4
sub.rust = (sub1.rust + sub2.rust + sub3.rust + sub4.rust) / 4
sub.scab = (sub1.scab + sub2.scab + sub3.scab + sub4.scab) / 4

sub = sub[["image_id", "healthy", "multiple_diseases", "rust", "scab"]]
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
