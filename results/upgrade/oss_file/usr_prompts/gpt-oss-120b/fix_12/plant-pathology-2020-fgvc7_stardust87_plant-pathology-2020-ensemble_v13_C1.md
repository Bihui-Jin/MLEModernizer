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

0.9699585629562538

# 6. Current score

0.45167

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'I replace the faulty ensemble logic with a simple baseline that reads the training labels, computes their overall mean probabilities, and writes those constant predictions into the required `submission.csv` using the provided sample submission template. This removes the out‑of‑range list access, ensures the file is saved with the correct columns and a `.csv` extension, and produces a valid submission that can be evaluated.'
- What this solution (achieved 0.54806) has done: 'I replace the constant‑mean baseline with a tiny linear model that uses the numeric part of each image filename as a feature. By fitting a simple slope + intercept for each disease column on the training data and applying it to the test IDs, the predictions gain a meaningful ordering, which should raise the ROC‑AUC from the 0.5 baseline toward the target while keeping the original workflow essentially unchanged.'
- What this solution (achieved 0.47528) has done: 'I replace the simple linear‑trend model with a small GradientBoostingRegressor for each disease label, using the numeric part of the image filename as the only feature. This model can capture non‑linear patterns in the IDs, which often correlate with disease classes, and should raise the ROC‑AUC substantially while keeping the overall pipeline unchanged. I also renumber the cells to start from 1 as required and add the necessary import.'
- What this solution (achieved 0.47528) has done: 'I enrich the simple numeric ID feature with two inexpensive yet potentially informative attributes – a “train‑vs‑test” flag derived from the image_id prefix and a log‑scaled version of the numeric ID. These additional columns are created both for the training and test sets and used as input to the existing GradientBoostingRegressor models. This small feature expansion keeps the overall pipeline unchanged while giving the models slightly more signal, which should improve the ranking and move the ROC‑AUC closer to the target score.'
- What this solution (achieved 0.48711) has done: 'I added two simple polynomial features (`num_id_sq` and `log_id_sq`) to give the model more expressive power, and I strengthened the GradientBoostingRegressor by increasing `n_estimators` to 500 and `max_depth` to 5 (keeping the same learning rate and randomness). The feature list is updated accordingly in both training and test processing, so the models can capture any non‑linear relationship between the image ID numbers and the disease probabilities, which should raise the ROC‑AUC toward the target while preserving the overall pipeline.'
- What this solution (achieved 0.48285) has done: 'I added a few inexpensive numeric features derived from the image_id (mod 2, 3, 5) and increased the number of boosting trees so the model can capture a bit more pattern while keeping the original GradientBoostingRegressor setup. These changes are tiny yet give the model more expressive power, which should raise the ROC‑AUC and move the score closer to the target without altering the overall pipeline.'
- What this solution (achieved 0.45626) has done: 'I add a few inexpensive but potentially informative numeric features derived from the image IDs (reversed ID, digit sum, modulo 7, and a cube‑root transform) and include them in the feature list used by the existing GradientBoostingRegressor models. I also increase the number of boosting trees slightly to give the model more capacity, while keeping the same regressor type and overall pipeline. The cells are renumbered to start at 1 and the script now writes a valid `submission.csv` at the end.'
- What this solution (achieved 0.46987) has done: 'I replace the GradientBoostingRegressor with a GradientBoostingClassifier (which is better suited for binary probability prediction) and adjust the prediction step to use `predict_proba`. This small change keeps the overall pipeline and feature set intact while improving the ranking quality, moving the ROC‑AUC score closer to the target.'
- What this solution (achieved 0.47057) has done: 'I add inexpensive image‑based features (average R, G, B channels) to the existing numeric ID features and include them in the GradientBoostingClassifier training. This keeps the original model type and workflow while giving the learner visual information that is highly predictive, so the ROC‑AUC should move noticeably toward the target. The script is otherwise unchanged and still writes a valid `submission.csv`.'
- What this solution (achieved 0.5) has done: 'I renumber the notebook cells to start at 1 and lightly regularize the GradientBoostingClassifier by reducing its depth and number of trees and adding class‑weight balancing. This should reduce over‑fitting on the many ID‑based features, yielding better probability ranking and moving the ROC‑AUC upward toward the target while keeping the overall pipeline unchanged.'
- What this solution (achieved 0.45167) has done: 'I fix the `GradientBoostingClassifier` initialization by switching to `GradientBoostingRegressor` (which does not support `class_weight`) and adjust the prediction step accordingly. This resolves the runtime errors, ensures the model dictionary contains all target columns, and allows the trained models to output probability‑like scores that are clipped to [0, 1]. These minimal changes keep the overall pipeline intact while improving the ranking ability, moving the ROC‑AUC toward the target score.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from PIL import Image  # lightweight image handling for new features




## === cell 1
BASE_PATH = "/kaggle/input/plant-pathology-2020-fgvc7"
TRAIN_PATH = os.path.join(BASE_PATH, "train.csv")
TEST_PATH = os.path.join(BASE_PATH, "test.csv")
SAMPLE_SUB_PATH = os.path.join(BASE_PATH, "sample_submission.csv")
SUBMISSION_OUTPUT = "submission.csv"


def _image_color_stats(image_id):
    img_path = os.path.join(BASE_PATH, "images", image_id)
    try:
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            arr = np.array(img).astype(np.float32) / 255.0  # normalise
            mean_r = arr[:, :, 0].mean()
            mean_g = arr[:, :, 1].mean()
            mean_b = arr[:, :, 2].mean()
            std_r = arr[:, :, 0].std()
            std_g = arr[:, :, 1].std()
            std_b = arr[:, :, 2].std()
    except Exception:
        mean_r = mean_g = mean_b = std_r = std_g = std_b = 0.0
    return pd.Series(
        {
            "mean_r": mean_r,
            "mean_g": mean_g,
            "mean_b": mean_b,
            "std_r": std_r,
            "std_g": std_g,
            "std_b": std_b,
        }
    )




## === cell 2
train_df = pd.read_csv(TRAIN_PATH)

train_df["num_id"] = train_df["image_id"].str.extract(r"(\d+)").astype(int)
train_df["is_train"] = train_df["image_id"].str.startswith("Train").astype(int)
train_df["log_id"] = np.log1p(train_df["num_id"])

train_df["num_id_sq"] = train_df["num_id"] ** 2
train_df["log_id_sq"] = train_df["log_id"] ** 2

train_df["mod2"] = train_df["num_id"] % 2
train_df["mod3"] = train_df["num_id"] % 3
train_df["mod5"] = train_df["num_id"] % 5

train_df["rev_id"] = train_df["num_id"].astype(str).apply(lambda s: int(s[::-1]))
train_df["digit_sum"] = (
    train_df["num_id"].astype(str).apply(lambda s: sum(int(ch) for ch in s))
)
train_df["mod7"] = train_df["num_id"] % 7
train_df["log_id_cbrt"] = np.cbrt(train_df["num_id"] + 1)

train_img_feats = train_df["image_id"].apply(_image_color_stats)
train_df = pd.concat([train_df, train_img_feats], axis=1)

TARGET_COLS = ["healthy", "multiple_diseases", "rust", "scab"]
models = {}

feature_cols = [
    "num_id",
    "is_train",
    "log_id",
    "num_id_sq",
    "log_id_sq",
    "mod2",
    "mod3",
    "mod5",
    "rev_id",
    "digit_sum",
    "mod7",
    "log_id_cbrt",
    "mean_r",
    "mean_g",
    "mean_b",
    "std_r",
    "std_g",
    "std_b",
]

for col in TARGET_COLS:
    X = train_df[feature_cols].values
    y = train_df[col].values
    gbr = GradientBoostingRegressor(
        n_estimators=800,
        learning_rate=0.05,
        max_depth=5,
        random_state=42,
    )
    gbr.fit(X, y)
    models[col] = gbr




## === cell 3
test_df = pd.read_csv(TEST_PATH)

test_df["num_id"] = test_df["image_id"].str.extract(r"(\d+)").astype(int)
test_df["is_train"] = test_df["image_id"].str.startswith("Train").astype(int)
test_df["log_id"] = np.log1p(test_df["num_id"])

test_df["num_id_sq"] = test_df["num_id"] ** 2
test_df["log_id_sq"] = test_df["log_id"] ** 2

test_df["mod2"] = test_df["num_id"] % 2
test_df["mod3"] = test_df["num_id"] % 3
test_df["mod5"] = test_df["num_id"] % 5

test_df["rev_id"] = test_df["num_id"].astype(str).apply(lambda s: int(s[::-1]))
test_df["digit_sum"] = (
    test_df["num_id"].astype(str).apply(lambda s: sum(int(ch) for ch in s))
)
test_df["mod7"] = test_df["num_id"] % 7
test_df["log_id_cbrt"] = np.cbrt(test_df["num_id"] + 1)

test_img_feats = test_df["image_id"].apply(_image_color_stats)
test_df = pd.concat([test_df, test_img_feats], axis=1)

submission = pd.read_csv(SAMPLE_SUB_PATH)

for col in TARGET_COLS:
    model = models[col]
    preds = model.predict(test_df[feature_cols].values)
    probs = np.clip(preds, 0.0, 1.0)  # ensure valid probability range
    submission[col] = probs




## === cell 4
submission.to_csv(SUBMISSION_OUTPUT, index=False)
print(f"Submission written to {SUBMISSION_OUTPUT}")
