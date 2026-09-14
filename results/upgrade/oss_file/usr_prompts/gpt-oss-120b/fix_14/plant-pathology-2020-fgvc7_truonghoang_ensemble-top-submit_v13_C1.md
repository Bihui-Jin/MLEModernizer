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

0.9709023663875812

# 6. Current score

0.62201

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'The fix removes the hard‑coded reads of missing CSV files, loads only the prediction files that actually exist, and safely averages whatever predictions are available (or falls back to the original sample submission). This resolves the `FileNotFoundError` and the `NameError` caused by the undefined `dsub`, and ensures a valid `submission.csv` is written with the correct column names.'
- What this solution (achieved 0.52496) has done: 'I replace the empty‑ensemble logic with a tiny, reproducible model that uses the image file size as a single feature. The script now loads the train and test tables, extracts each image’s byte size, fits a separate logistic‑regression model for every target column, and writes the predicted probabilities to `submission.csv`. This adds predictive signal (instead of the constant sample submission) and should raise the ROC‑AUC well above the current 0.5, moving the score toward the target while keeping the overall workflow unchanged.'
- What this solution (achieved 0.52496) has done: 'I added lightweight image‑metadata features (width, height and size in kilobytes) using Pillow when it is available, and included them together with the original file‑size feature for the logistic‑regression models. These extra signals give the classifiers more information than a single scalar, so the predicted probabilities should move the ROC‑AUC closer to the target score while preserving the overall workflow.'
- What this solution (achieved 0.59223) has done: 'I enrich the image‑based features by adding the mean RGB channel values (computed with Pillow when available) to the existing size/width/height metadata. These additional visual cues give the logistic‑regression models more predictive signal, which should raise the ROC‑AUC toward the target while keeping the overall workflow and model architecture unchanged.'
- What this solution (achieved 0.64814) has done: 'I enrich the image‑metadata features by adding per‑channel standard deviations and a simple green‑to‑red ratio, then include these new columns in the training matrix.  The logistic‑regression model is kept unchanged, but the extra statistical descriptors give it more predictive signal, which should raise the ROC‑AUC toward the target while preserving the overall pipeline.'
- What this solution (achieved 0.70327) has done: 'I replace the logistic‑regression models with a balanced RandomForest classifier, which can capture non‑linear patterns in the image‑metadata features and is expected to raise the ROC‑AUC toward the target while keeping the overall pipeline unchanged. The only additions are the necessary import and the model swap inside the training loop.'
- What this solution (achieved 0.7469) has done: 'I enrich the image‑metadata features by adding HSV statistics and additional colour‑ratio metrics, then feed these extra columns into the existing RandomForest model (with a slightly larger forest) so the classifier gets more discriminative signal and the ROC‑AUC moves closer to the target. The overall pipeline, model type, and training loop remain unchanged, only the feature extraction and model hyper‑parameter count are adjusted.'
- What this solution (achieved 0.734) has done: 'I add a lightweight Logistic Regression model for each target and average its probabilities with those from the existing RandomForest. This ensemble often yields a modest boost in ROC‑AUC without changing the core feature extraction or training pipeline. I also increase the forest size slightly for more stable estimates.'
- What this solution (achieved 0.73287) has done: 'I add a simple numeric identifier extracted from each image_id as an additional feature and give the RandomForest a stronger configuration (more trees, leaf size = 1). I also shift the ensemble to weight the RandomForest predictions more heavily (70 % RF + 30 % LR). These minimal tweaks keep the overall pipeline unchanged while providing extra predictive signal and a slightly more powerful model, which should raise the ROC‑AUC toward the target.'
- What this solution (achieved 0.73217) has done: 'I give the RandomForest a slightly stronger configuration and let it dominate the ensemble (increase trees and weight it 90 % vs 10 % for Logistic Regression). This keeps the overall pipeline unchanged while nudging the ROC‑AUC upward toward the target.'
- What this solution (achieved 0.72777) has done: 'I increase the RandomForest capacity (more trees) and let it dominate the ensemble by removing the Logistic‑Regression contribution – this adds predictive power without altering the overall pipeline or feature set, moving the ROC‑AUC closer to the target score.'
- What this solution (achieved 0.73147) has done: 'I slightly strengthen the random‑forest models (more trees, limited leaf size and a reduced max_features) and add probability calibration, then blend a modest share of the logistic‑regression predictions (80 % RF + 20 % LR). These minimal tweaks keep the overall pipeline unchanged while providing a calibrated, slightly more powerful ensemble that should raise the ROC‑AUC toward the target.'
- What this solution (achieved 0.62201) has done: 'I tighten the ensemble by using only the calibrated RandomForest predictions (removing the Logistic‑Regression blend) and switch its calibration to the isotonic method, which often yields better probability estimates for AUC. This small adjustment keeps the core pipeline unchanged while providing a more accurate scoring model, moving the ROC‑AUC closer to the target.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

try:
    from PIL import Image

    _PIL_AVAILABLE = True
except Exception:
    _PIL_AVAILABLE = False




## === cell 1
DATA_ROOT = "../input/plant-pathology-2020-fgvc7"
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
TEST_CSV = os.path.join(DATA_ROOT, "test.csv")
IMG_DIR = os.path.join(DATA_ROOT, "images")

train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)




## === cell 2
def get_image_features(img_id):
    """
    Return a tuple with extended colour and HSV statistics:
    (file_size_bytes, width, height,
     mean_R, mean_G, mean_B,
     std_R, std_G, std_B,
     green_red_ratio, green_blue_ratio, red_blue_ratio,
     mean_H, mean_S, mean_V,
     std_H, std_S, std_V)
    Missing values are set to NaN.
    """
    path = os.path.join(IMG_DIR, f"{img_id}.jpg")
    try:
        size = os.path.getsize(path)
    except OSError:
        size = np.nan

    if _PIL_AVAILABLE:
        try:
            with Image.open(path) as im:
                width, height = im.size
                im = im.convert("RGB")
                arr = np.asarray(im, dtype=np.float32)

                mean_r = arr[..., 0].mean()
                mean_g = arr[..., 1].mean()
                mean_b = arr[..., 2].mean()
                std_r = arr[..., 0].std()
                std_g = arr[..., 1].std()
                std_b = arr[..., 2].std()
                green_red_ratio = mean_g / (mean_r + 1e-6)
                green_blue_ratio = mean_g / (mean_b + 1e-6)
                red_blue_ratio = mean_r / (mean_b + 1e-6)

                hsv = np.asarray(im.convert("HSV"), dtype=np.float32)
                mean_h = hsv[..., 0].mean()
                mean_s = hsv[..., 1].mean()
                mean_v = hsv[..., 2].mean()
                std_h = hsv[..., 0].std()
                std_s = hsv[..., 1].std()
                std_v = hsv[..., 2].std()
        except Exception:
            width, height = np.nan, np.nan
            mean_r = mean_g = mean_b = np.nan
            std_r = std_g = std_b = np.nan
            green_red_ratio = green_blue_ratio = red_blue_ratio = np.nan
            mean_h = mean_s = mean_v = np.nan
            std_h = std_s = std_v = np.nan
    else:
        width, height = np.nan, np.nan
        mean_r = mean_g = mean_b = np.nan
        std_r = std_g = std_b = np.nan
        green_red_ratio = green_blue_ratio = red_blue_ratio = np.nan
        mean_h = mean_s = mean_v = np.nan
        std_h = std_s = std_v = np.nan

    return (
        size,
        width,
        height,
        mean_r,
        mean_g,
        mean_b,
        std_r,
        std_g,
        std_b,
        green_red_ratio,
        green_blue_ratio,
        red_blue_ratio,
        mean_h,
        mean_s,
        mean_v,
        std_h,
        std_s,
        std_v,
    )


train_meta = train_df["image_id"].apply(get_image_features)
test_meta = test_df["image_id"].apply(get_image_features)

train_df[
    [
        "img_size",
        "img_width",
        "img_height",
        "img_mean_r",
        "img_mean_g",
        "img_mean_b",
        "img_std_r",
        "img_std_g",
        "img_std_b",
        "img_green_red_ratio",
        "img_green_blue_ratio",
        "img_red_blue_ratio",
        "img_mean_h",
        "img_mean_s",
        "img_mean_v",
        "img_std_h",
        "img_std_s",
        "img_std_v",
    ]
] = pd.DataFrame(train_meta.tolist(), index=train_df.index)

test_df[
    [
        "img_size",
        "img_width",
        "img_height",
        "img_mean_r",
        "img_mean_g",
        "img_mean_b",
        "img_std_r",
        "img_std_g",
        "img_std_b",
        "img_green_red_ratio",
        "img_green_blue_ratio",
        "img_red_blue_ratio",
        "img_mean_h",
        "img_mean_s",
        "img_mean_v",
        "img_std_h",
        "img_std_s",
        "img_std_v",
    ]
] = pd.DataFrame(test_meta.tolist(), index=test_df.index)

train_df["img_id_num"] = train_df["image_id"].str.extract(r"(\d+)").astype(int)
test_df["img_id_num"] = test_df["image_id"].str.extract(r"(\d+)").astype(int)

train_df = train_df.dropna(subset=["img_size"])
train_df = train_df.fillna(-1)
test_df = test_df.fillna(-1)

feature_cols = [
    "img_size",
    "img_width",
    "img_height",
    "img_mean_r",
    "img_mean_g",
    "img_mean_b",
    "img_std_r",
    "img_std_g",
    "img_std_b",
    "img_green_red_ratio",
    "img_green_blue_ratio",
    "img_red_blue_ratio",
    "img_mean_h",
    "img_mean_s",
    "img_mean_v",
    "img_std_h",
    "img_std_s",
    "img_std_v",
    "img_id_num",
]

X_train = train_df[feature_cols].values
X_test = test_df[feature_cols].values

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)




## === cell 3
target_cols = ["healthy", "multiple_diseases", "rust", "scab"]
rf_models = {}
lr_models = {}

for col in target_cols:
    y = train_df[col].values

    rf = RandomForestClassifier(
        n_estimators=6000,  # more trees for a modest gain
        max_features=0.7,  # limit features per split to reduce variance
        min_samples_leaf=2,  # prevent over‑fitting on tiny leaves
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        max_depth=None,
    )
    rf.fit(X_train, y)

    rf_calibrated = CalibratedClassifierCV(rf, method="isotonic", cv="prefit")
    rf_calibrated.fit(X_train, y)
    rf_models[col] = rf_calibrated

    lr = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="liblinear",
        random_state=42,
    )
    lr.fit(X_train, y)
    lr_models[col] = lr




## === cell 4
sub_path = os.path.join(DATA_ROOT, "sample_submission.csv")
submission = pd.read_csv(sub_path)

rf_weight = 1.0
lr_weight = 0.0

for col in target_cols:
    rf_probs = rf_models[col].predict_proba(X_test)[:, 1]
    lr_probs = lr_models[col].predict_proba(X_test)[:, 1]
    combined = rf_weight * rf_probs + lr_weight * lr_probs
    submission[col] = np.clip(combined, 0.0, 1.0)

submission = submission.set_index("image_id").loc[test_df["image_id"]].reset_index()
submission.to_csv("submission.csv", index=False)
