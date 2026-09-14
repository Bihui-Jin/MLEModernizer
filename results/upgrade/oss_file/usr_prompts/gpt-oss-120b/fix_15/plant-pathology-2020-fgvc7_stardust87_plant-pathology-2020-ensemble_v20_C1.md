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

0.9694615556143084

# 6. Current score

0.5

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'The script now safely collects any existing CSV submissions; if none are found it falls back to a simple baseline that predicts each disease probability as the mean label frequency from the training set. This guarantees a valid `submission.csv` with the correct columns, fixing the original index error and ensuring the pipeline finishes successfully. The changes are minimal and keep the original ensemble logic intact when prior submissions exist.'
- What this solution (achieved 0.52496) has done: 'I add a very lightweight image‑size based model that is used only when no prior submissions exist. It extracts the file size of each image (a cheap numeric feature), fits a separate logistic‑regression for each target column on the training set, and then predicts probabilities for the test set. This keeps the original fallback logic while giving the model a small amount of signal that can lift the ROC‑AUC from the constant‑mean baseline toward the target score.'
- What this solution (achieved 0.56898) has done: 'I keep the overall pipeline unchanged but replace the very lightweight logistic‑regression fallback with a slightly richer feature set (image file size + numeric part of the image id) and a GradientBoosting model, which usually captures more signal while still being fast. This change is confined to the `model_based_submission` function and therefore preserves the original logic and file outputs, but it should raise the ROC‑AUC from the current ~0.52 toward the target score.'
- What this solution (achieved 0.53879) has done: 'I add a cheap but informative image feature (average pixel intensity) using Pillow, and use it together with file size and numeric ID to build a richer feature set. The GradientBoosting model be given a few more trees (n_estimators = 300) to capture more patterns, and the logistic‑regression fallback also use all available features. These changes keep the overall fallback pipeline unchanged while providing extra signal, which should raise the ROC‑AUC toward the target score.'
- What this solution (achieved 0.58758) has done: 'I enrich the cheap image‑derived feature set by adding per‑channel mean colors, image dimensions and grayscale variance, and increase the GradientBoosting capacity (more trees and a deeper tree). These extra, still inexpensive visual cues give the model more signal to distinguish diseases, which should raise the ROC‑AUC toward the target while keeping the overall pipeline unchanged.'
- What this solution (achieved 0.62769) has done: 'I enrich the cheap image‑derived feature set by adding per‑channel standard deviations (std R, std G, std B) to capture colour variability, and I give the GradientBoostingClassifier a bit more capacity (800 trees, depth 5) which still respects the original lightweight design. These extra numeric cues should improve the model’s discriminative power and raise the ROC‑AUC toward the target while keeping the overall pipeline unchanged.'
- What this solution (achieved 0.66157) has done: 'I add richer image‑derived features (8‑bin colour histograms for each RGB channel) to the cheap feature set and slightly increase the GradientBoosting capacity (1200 trees, depth 6). These changes keep the overall lightweight design while giving the model more signal, which should raise the ROC‑AUC toward the target without altering the core workflow.'
- What this solution (achieved 0.5) has done: 'We speed up the heavy feature‑extraction step by (1) switching to a ProcessPoolExecutor so CPU‑bound image processing runs in true parallel across all cores, and (2) allocating the feature matrix once and filling it directly instead of building many intermediate Python lists and then stacking them, which removes costly list/zip handling while preserving the exact same feature order and values. These changes keep the model‑training logic unchanged, guarantee deterministic ordering, and dramatically cut the runtime without affecting prediction accuracy.'
- What this solution (achieved 0.5) has done: 'I simplify the image‑feature extraction in `model_based_submission` by removing the large flattened‑pixel vector and keeping only the inexpensive numeric cues (size, id, dimensions, colour means/stds, grayscale stats, and 8‑bin colour histograms). This reduces the feature dimension from ≈12 k to ≈36, avoids over‑parameterisation that was driving the model to near‑random performance, and keeps the same GradientBoosting classifier (with a modestly lower number of trees). The rest of the pipeline stays identical, and a valid `submission.csv` is still written.'
- What this solution (achieved 0.5) has done: 'The patch adds two inexpensive visual cues (aspect‑ratio and size‑per‑pixel) to the cheap feature set, expands the feature dimension accordingly, and switches the core model to a higher‑capacity HistGradientBoostingClassifier with a deeper tree, slower learning rate and subsampling – all still within the original gradient‑boosting workflow. These adjustments are expected to capture more discriminative signal from the images and raise the ROC‑AUC toward the target without altering the overall pipeline or submission format. Additionally, the feature‑extraction code now fills the new columns safely.'
- What this solution (achieved 0.5) has done: 'I add a more robust image‑file lookup (try “.JPG” if “.jpg” fails) and introduce a RandomForestClassifier as a first‑line model (it handles class imbalance well). If it fails, the existing HistGradientBoosting and GradientBoosting fall‑backs remain unchanged. These small tweaks keep the overall pipeline intact while giving the predictions stronger discriminative power, moving the ROC‑AUC toward the target.'
- What this solution (achieved 0.5) has done: 'I improve the predictive power while keeping the overall pipeline intact by (1) increasing the forest size for a stronger RandomForest, and (2) averaging the RandomForest and HistGradientBoosting predictions for each label when both models train successfully. This adds only a lightweight ensemble step inside the existing loop, preserving the original fallback logic and file outputs, and is expected to raise the ROC‑AUC toward the target score.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)

try:
    from PIL import Image

    _PIL_AVAILABLE = True
except Exception:
    _PIL_AVAILABLE = False

DATA_ROOT = "/kaggle/input/plant-pathology-2020-fgvc7"

TRAIN_PATH = os.path.join(DATA_ROOT, "train.csv")
TEST_PATH = os.path.join(DATA_ROOT, "test.csv")
SAMPLE_SUB_PATH = os.path.join(DATA_ROOT, "sample_submission.csv")
IMAGES_DIR = os.path.join(DATA_ROOT, "images")



## === cell 1
SUBMISSIONS_PATH = "/kaggle/input/submissions/"

submissions_all = []
if os.path.isdir(SUBMISSIONS_PATH):
    for dirname, _, filenames in os.walk(SUBMISSIONS_PATH):
        for filename in filenames:
            if filename.lower().endswith(".csv"):
                submissions_all.append(os.path.join(dirname, filename))
submissions_all.sort()
print("Found submissions:", submissions_all)




## === cell 2
def model_based_submission():
    """
    Light model using several cheap image‑derived features plus 8‑bin RGB histograms
    and two extra size‑based cues (aspect ratio and bytes‑per‑pixel).  A
    RandomForestClassifier is tried first; if it also succeeds we also fit a
    HistGradientBoostingClassifier and average both predictions.  If any step
    fails we fall back to the existing hierarchical models.
    """

    import concurrent.futures

    FEATURE_DIM = 38

    def _process_one(img_id):
        """Extract all required features for a single image."""
        path = os.path.join(IMAGES_DIR, f"{img_id}.jpg")
        if not os.path.exists(path):
            path = os.path.join(IMAGES_DIR, f"{img_id}.JPG")
        try:
            size = os.path.getsize(path)
        except Exception:
            size = 0
        num_str = "".join(filter(str.isdigit, str(img_id)))
        num_id = int(num_str) if num_str else 0

        width = height = 0
        mr = mg = mb = 0.0
        sr = sg = sb = 0.0
        gray_mean = gray_var = 0.0
        hist_r = np.zeros(8, dtype=float)
        hist_g = np.zeros(8, dtype=float)
        hist_b = np.zeros(8, dtype=float)

        if _PIL_AVAILABLE and os.path.exists(path):
            try:
                img = Image.open(path).convert("RGB")
                img_resized = img.resize((64, 64))
                img_array = np.array(img_resized)  # (64, 64, 3)

                width, height = img_resized.width, img_resized.height

                r = img_array[:, :, 0]
                g = img_array[:, :, 1]
                b = img_array[:, :, 2]

                mr = r.mean()
                mg = g.mean()
                mb = b.mean()

                sr = r.std()
                sg = g.std()
                sb = b.std()

                hist_r, _ = np.histogram(r, bins=8, range=(0, 255), density=True)
                hist_g, _ = np.histogram(g, bins=8, range=(0, 255), density=True)
                hist_b, _ = np.histogram(b, bins=8, range=(0, 255), density=True)

                gray = img.convert("L")
                gray_arr = np.array(gray.resize((64, 64)))
                gray_mean = gray_arr.mean()
                gray_var = gray_arr.var()
            except Exception:
                pass

        aspect_ratio = (width / height) if height > 0 else 0.0
        size_per_pixel = (size / (width * height)) if (width * height) > 0 else 0.0

        feats = np.empty(FEATURE_DIM, dtype=float)
        feats[0] = size
        feats[1] = num_id
        feats[2] = width
        feats[3] = height
        feats[4] = mr
        feats[5] = mg
        feats[6] = mb
        feats[7] = sr
        feats[8] = sg
        feats[9] = sb
        feats[10] = gray_mean
        feats[11] = gray_var
        feats[12:20] = hist_r
        feats[20:28] = hist_g
        feats[28:36] = hist_b
        feats[36] = aspect_ratio
        feats[37] = size_per_pixel
        return feats

    def get_features(df):
        """Parallel feature extraction preserving original order, returning a dense matrix."""
        img_ids = df["image_id"].tolist()
        n = len(img_ids)
        X = np.empty((n, FEATURE_DIM), dtype=float)

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=os.cpu_count()
        ) as executor:
            for i, feats in enumerate(executor.map(_process_one, img_ids, chunksize=8)):
                X[i, :] = feats
        return X

    prob_cols = ["healthy", "multiple_diseases", "rust", "scab"]

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    X_train = get_features(train_df)
    X_test = get_features(test_df)

    preds = np.empty((len(test_df), len(prob_cols)), dtype=float)

    for i, col in enumerate(prob_cols):
        y = train_df[col].values
        try:
            rf = RandomForestClassifier(
                n_estimators=1000,  # stronger forest
                max_depth=None,
                min_samples_leaf=1,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )
            rf.fit(X_train, y)
            rf_pred = rf.predict_proba(X_test)[:, 1]
        except Exception as e:
            print(f"RF failed for {col} ({e}), falling back to next model.")
            rf_pred = None

        try:
            hgb = HistGradientBoostingClassifier(
                max_iter=1200,
                learning_rate=0.03,
                max_depth=6,
                min_samples_leaf=20,
                l2_regularization=0.1,
                random_state=42,
            )
            hgb.fit(X_train, y)
            hgb_pred = hgb.predict_proba(X_test)[:, 1]
        except Exception as e:
            print(f"HGB failed for {col} ({e}), falling back.")
            hgb_pred = None

        if rf_pred is not None and hgb_pred is not None:
            preds[:, i] = (rf_pred + hgb_pred) / 2.0
            continue

        if rf_pred is not None:
            preds[:, i] = rf_pred
            continue

        if hgb_pred is not None:
            preds[:, i] = hgb_pred
            continue

        try:
            gb = GradientBoostingClassifier(
                n_estimators=800,
                learning_rate=0.05,
                max_depth=5,
                random_state=42,
            )
            gb.fit(X_train, y)
            preds[:, i] = gb.predict_proba(X_test)[:, 1]
            continue
        except Exception as e:
            print(f"GB also failed for {col} ({e}), using LogisticRegression.")

        lr = LogisticRegression(
            solver="liblinear",
            class_weight="balanced",
            max_iter=1000,
        )
        lr.fit(X_train, y)
        preds[:, i] = lr.predict_proba(X_test)[:, 1]

    sub_df = test_df.copy()
    sub_df[prob_cols] = preds
    sub_df.to_csv("submission.csv", index=False)
    print("Model‑based submission written to submission.csv")




## === cell 3
def ensemble(submissions_all, sub_idx, weights=[]):
    """
    Weighted average of the probabilities from a list of submission files.
    If the requested indices are out of range we raise a clear error.
    """
    if not submissions_all:
        raise ValueError("No submission files available for ensembling.")
    if len(sub_idx) != len(weights):
        raise ValueError("sub_idx and weights must have the same length.")
    submission_with_weight = []
    for i in range(len(sub_idx)):
        idx = sub_idx[i]
        if idx >= len(submissions_all):
            raise IndexError(
                f"sub_idx[{i}] == {idx} is out of bounds for submissions_all of length {len(submissions_all)}"
            )
        print(f"I'm taking submission {submissions_all[idx]} with weight {weights[i]}")
        sub = pd.read_csv(submissions_all[idx])
        sub_vals = sub.loc[:, ["healthy", "multiple_diseases", "rust", "scab"]].values
        submission_with_weight.append(sub_vals * weights[i])
    submission_avg = sum(submission_with_weight)
    return submission_avg




## === cell 4
def make_submission_file(submission_avg, reference_csv_path):
    """
    Write the averaged probabilities to a CSV file using the structure of a reference submission.
    """
    ref_df = pd.read_csv(reference_csv_path)
    prob_cols = ["healthy", "multiple_diseases", "rust", "scab"]
    ref_df.loc[:, prob_cols] = submission_avg
    ref_df.to_csv("submission.csv", index=False)
    print("Submission written to submission.csv")


def baseline_submission():
    """
    Simple baseline: predict each class probability as the mean label frequency
    observed in the training data.
    """
    train_df = pd.read_csv(TRAIN_PATH)
    prob_cols = ["healthy", "multiple_diseases", "rust", "scab"]
    means = train_df[prob_cols].mean().values.reshape(1, -1)
    test_df = pd.read_csv(TEST_PATH)
    baseline_preds = np.repeat(means, repeats=len(test_df), axis=0)
    sub_df = test_df.copy()
    sub_df[prob_cols] = baseline_preds
    sub_df.to_csv("submission.csv", index=False)
    print("Baseline submission written to submission.csv")




## === cell 5
import numpy as np

if submissions_all:
    try:
        model_based_submission()
        model_preds = (
            pd.read_csv("submission.csv")
            .loc[:, ["healthy", "multiple_diseases", "rust", "scab"]]
            .values
        )
    except Exception as e:
        print(f"Model‑based failed ({e}), falling back to baseline.")
        baseline_submission()
        model_preds = (
            pd.read_csv("submission.csv")
            .loc[:, ["healthy", "multiple_diseases", "rust", "scab"]]
            .values
        )

    n_subs = min(3, len(submissions_all))
    idxs = list(range(n_subs))
    ensemble_weights = [0.5 / n_subs] * n_subs
    ensemble_avg = ensemble(submissions_all, idxs, ensemble_weights)
    combined_avg = (ensemble_avg + model_preds) / 2.0
    make_submission_file(combined_avg, submissions_all[0])
else:
    try:
        model_based_submission()
    except Exception as e:
        print(f"Model‑based submission failed ({e}), falling back to baseline.")
        baseline_submission()
