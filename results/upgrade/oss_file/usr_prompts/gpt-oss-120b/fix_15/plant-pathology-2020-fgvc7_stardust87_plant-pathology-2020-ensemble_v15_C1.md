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

0.9700013841179632

# 6. Current score

0.59816

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'I correct the path used to locate submission CSV files, safely collect any CSVs present, and adjust the ensemble and write‑out functions to work with explicit file lists rather than index look‑ups (which caused the IndexError). The script now falls back to using the single available sample submission with a weight of 1.0, builds the weighted average correctly, and writes a proper `submission.csv` containing the required columns.'
- What this solution (achieved 0.57121) has done: 'I add a lightweight image‑based model that creates a prediction CSV (using resized pixel values and a multi‑output logistic regression). The script then include this file in the ensemble step, so the final `submission.csv` contains realistic probabilities rather than a placeholder, moving the validation AUC from ~0.5 toward the target score.'
- What this solution (achieved 0.57121) has done: 'I fixed the KeyError by filtering the discovered CSV files so only those that actually contain the prediction columns are kept, and I guaranteed that the model’s own prediction file is used. I also changed the reference file for the final submission to the official test CSV, ensuring the image order is correct. The ensemble now works with a single valid prediction file, preserving the original model logic while producing a proper `submission.csv`.'
- What this solution (achieved 0.58403) has done: 'I filter the discovered CSV files so only those that match the test set row count are kept, preventing the shape‑mismatch error during ensembling. I also enhance the image feature extraction by using a larger 128×128 size and train the logistic regression with balanced class weights, which should modestly improve the validation AUC while keeping the core modelling approach unchanged. The script now runs end‑to‑end and writes a correct `submission.csv`.'
- What this solution (achieved 0.67551) has done: 'I replace the linear logistic‑regression base model with a balanced RandomForest classifier, which can capture non‑linear patterns in the pixel data while keeping the same multi‑output structure and overall pipeline. This change is expected to raise the validation ROC‑AUC significantly and move the competition score closer to the target, without altering the data handling, ensembling, or submission logic.'
- What this solution (achieved 0.5703) has done: 'I minimally extend the feature extraction to include per‑channel mean and std, increase the forest size, and add a simple LogisticRegression model whose predictions are also ensembled with the RandomForest. These changes keep the same overall pipeline and should raise the validation ROC‑AUC, moving the score closer to the target while still producing a correct `submission.csv`.'
- What this solution (achieved 0.60659) has done: 'I improve the feature extraction by adding per‑channel histograms (16 bins each) and increase the RandomForest size to 1000 trees for stronger modeling. I also simplify the ensemble to use only the RandomForest predictions (the strongest model) so the final submission reflects its better AUC. These minimal changes keep the overall pipeline intact while moving the validation score closer to the target.'
- What this solution (achieved 0.63969) has done: 'I include the LogisticRegression predictions in the final ensemble (instead of using only the RandomForest) and give each model equal weight. This adds complementary information from the second model, which should raise the ROC‑AUC toward the target while keeping the overall pipeline unchanged.'
- What this solution (achieved 0.62737) has done: 'I add a PCA dimensionality‑reduction step to clean up the very high‑dimensional pixel features, keep only the RandomForest model (its predictions performed best in prior runs) and drop the LogisticRegression ensemble, and adjust the submission handling to use the single, stronger model. This minimal preprocessing is expected to raise the ROC‑AUC toward the target without altering the core modeling logic.'
- What this solution (achieved 0.60209) has done: 'I slightly boost the feature representation by keeping a bit more variance in the PCA (0.98 instead of 0.95) and add a lightweight LogisticRegression model alongside the existing RandomForest. Both models generate prediction CSVs and be ensembled with equal weights, preserving the original pipeline while aiming to raise the validation AUC toward the target score.'
- What this solution (achieved 0.60024) has done: 'I slightly enhance the image feature extraction (larger resize and more histogram bins) and retain a little more variance in the PCA, increase the RandomForest strength, and give it higher ensemble weight because it consistently out‑performs the logistic model. These modest tweaks keep the original pipeline intact while are expected to raise the ROC‑AUC toward the target score.'
- What this solution (achieved 0.60024) has done: 'I add a lightweight validation split to evaluate each model’s ROC‑AUC on the training data, then set the ensemble weights proportionally to the observed scores (instead of fixed 0.7/0.3). This keeps the existing RandomForest + LogisticRegression pipeline unchanged while moving the final AUC closer to the target. I also import the needed utilities and compute the weights dynamically before creating the final submission file.'
- What this solution (achieved 0.5756) has done: 'I slightly improve the image feature extraction by averaging original and horizontally‑flipped images, increase the resize to 256 px and histogram bins to 64, and reduce PCA variance to 0.95 to remove noisy components. These modest changes keep the same RandomForest + LogisticRegression pipeline while expectedly raising the validation AUC and moving the score closer to the target. The script is re‑ordered into four numbered cells, preserving the original workflow and still writing a correct `submission.csv`.'
- What this solution (achieved 0.59816) has done: 'I keep the overall pipeline but improve the feature extraction (larger resize and more histogram bins) and retain more variance in the PCA (0.99). I also compute validation scores with models trained only on the training split (avoiding leakage) and use those more reliable AUCs to set the ensemble weights. Finally, I increase the LogisticRegression regularisation strength (C=2.0) to give it a bit more flexibility. These modest, focused tweaks should raise the validation AUC and move the Kaggle score closer to the target while preserving the original workflow.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
from PIL import Image, ImageOps
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

BASE_DIR = "/kaggle/input/plant-pathology-2020-fgvc7"
TRAIN_CSV = os.path.join(BASE_DIR, "train.csv")
TEST_CSV = os.path.join(BASE_DIR, "test.csv")
IMAGES_DIR = os.path.join(BASE_DIR, "images")

train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)


def extract_features(path, size=(300, 300), hist_bins=128):
    """Load an image, resize, compute features for original and flipped version,
    then average them to obtain a more robust representation."""
    img = Image.open(path).convert("RGB")
    img = img.resize(size, Image.BILINEAR)
    img_flip = ImageOps.mirror(img)

    def feat(im):
        arr = np.asarray(im) / 255.0  # (H, W, 3)
        flat = arr.flatten()
        channel_means = arr.mean(axis=(0, 1))
        channel_stds = arr.std(axis=(0, 1))
        hist_features = []
        for c in range(3):
            hist, _ = np.histogram(
                arr[:, :, c], bins=hist_bins, range=(0.0, 1.0), density=True
            )
            hist_features.append(hist)
        hist_features = np.concatenate(hist_features)
        return np.concatenate([flat, channel_means, channel_stds, hist_features])

    f_orig = feat(img)
    f_flip = feat(img_flip)
    return (f_orig + f_flip) / 2.0


train_features = [
    extract_features(os.path.join(IMAGES_DIR, f"{img_id}.jpg"))
    for img_id in train_df["image_id"]
]
X_train_raw = np.stack(train_features)

test_features = [
    extract_features(os.path.join(IMAGES_DIR, f"{img_id}.jpg"))
    for img_id in test_df["image_id"]
]
X_test_raw = np.stack(test_features)

pca = PCA(n_components=0.99, svd_solver="full", random_state=42)
X_train = pca.fit_transform(X_train_raw)
X_test = pca.transform(X_test_raw)

train_labels = train_df[["healthy", "multiple_diseases", "rust", "scab"]].values




## === cell 1
rf_base = RandomForestClassifier(
    n_estimators=4000,
    max_depth=None,
    n_jobs=-1,
    class_weight="balanced",
    random_state=42,
)
rf_clf = MultiOutputClassifier(rf_base)
rf_clf.fit(X_train, train_labels)

rf_test_probs = np.stack(
    [estimator.predict_proba(X_test)[:, 1] for estimator in rf_clf.estimators_],
    axis=1,
)

rf_pred_path = "model_predictions_rf.csv"
rf_pred_df = test_df.copy()
rf_pred_df[["healthy", "multiple_diseases", "rust", "scab"]] = rf_test_probs
rf_pred_df.to_csv(rf_pred_path, index=False)
print(f"RandomForest predictions saved to {rf_pred_path}")

logreg_base = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    n_jobs=-1,
    solver="lbfgs",
    C=2.0,  # slightly weaker regularisation
)
logreg_clf = MultiOutputClassifier(logreg_base)
logreg_clf.fit(X_train, train_labels)

logreg_test_probs = np.stack(
    [estimator.predict_proba(X_test)[:, 1] for estimator in logreg_clf.estimators_],
    axis=1,
)

logreg_pred_path = "model_predictions_logreg.csv"
logreg_pred_df = test_df.copy()
logreg_pred_df[["healthy", "multiple_diseases", "rust", "scab"]] = logreg_test_probs
logreg_pred_df.to_csv(logreg_pred_path, index=False)
print(f"LogisticRegression predictions saved to {logreg_pred_path}")




## === cell 2
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, train_labels, test_size=0.2, random_state=42
)

rf_val_clf = MultiOutputClassifier(
    RandomForestClassifier(
        n_estimators=4000,
        max_depth=None,
        n_jobs=-1,
        class_weight="balanced",
        random_state=42,
    )
)
rf_val_clf.fit(X_tr, y_tr)
rf_val_probs = np.stack(
    [estimator.predict_proba(X_val)[:, 1] for estimator in rf_val_clf.estimators_],
    axis=1,
)

logreg_val_clf = MultiOutputClassifier(
    LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        n_jobs=-1,
        solver="lbfgs",
        C=2.0,
    )
)
logreg_val_clf.fit(X_tr, y_tr)
logreg_val_probs = np.stack(
    [estimator.predict_proba(X_val)[:, 1] for estimator in logreg_val_clf.estimators_],
    axis=1,
)


def mean_auc(y_true, probs):
    """Average ROC‑AUC over all four targets."""
    aucs = [roc_auc_score(y_true[:, i], probs[:, i]) for i in range(y_true.shape[1])]
    return np.mean(aucs)


rf_val_auc = mean_auc(y_val, rf_val_probs)
logreg_val_auc = mean_auc(y_val, logreg_val_probs)

total = rf_val_auc + logreg_val_auc
rf_weight = rf_val_auc / total
logreg_weight = logreg_val_auc / total
weights = [rf_weight, logreg_weight]

print(f"Validation AUC – RF: {rf_val_auc:.5f}, LogReg: {logreg_val_auc:.5f}")
print(f"Derived ensemble weights – RF: {rf_weight:.3f}, LogReg: {logreg_weight:.3f}")

submissions_all = [rf_pred_path, logreg_pred_path]
print("Using submission files:", submissions_all)




## === cell 3
def ensemble(submission_paths, weights):
    """
    Return the weighted average of the prediction columns from the given CSV files.
    Each CSV must contain the columns: healthy, multiple_diseases, rust, scab.
    """
    if len(submission_paths) != len(weights):
        raise ValueError("Number of paths and weights must match")
    weighted_sum = None
    for path, w in zip(submission_paths, weights):
        print(f"Reading {path} with weight {w:.4f}")
        df = pd.read_csv(path)
        vals = df.loc[:, ["healthy", "multiple_diseases", "rust", "scab"]].values
        if weighted_sum is None:
            weighted_sum = vals * w
        else:
            weighted_sum += vals * w
    return weighted_sum


def make_submission_file(submission_avg, reference_path):
    """
    Write `submission.csv` using the image_id column from a reference file
    and the averaged predictions supplied in `submission_avg`.
    """
    ref_df = pd.read_csv(reference_path)
    if submission_avg.shape[0] != ref_df.shape[0]:
        raise ValueError(
            "Shape mismatch between averaged predictions and reference file"
        )
    submission_df = ref_df.copy()
    submission_df[["healthy", "multiple_diseases", "rust", "scab"]] = submission_avg
    submission_df.to_csv("submission.csv", index=False)
    print("submission.csv written successfully.")


if not submissions_all:
    raise FileNotFoundError("No valid CSV submission files found.")

submission_avg = ensemble(submissions_all, weights)
make_submission_file(submission_avg, TEST_CSV)
