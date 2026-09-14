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

0.48675

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'The fix removes the missing external submission files, reads the provided training data, computes average label probabilities, fills the sample submission with these averages, and writes a valid `submission.csv`. This resolves the FileNotFound and NameError issues and guarantees a correctly‑formatted CSV output.'
- What this solution (achieved 0.5) has done: 'I add a simple lookup so that any test image that also appears in the training set receives its exact training label (giving perfect AUC for those rows), while all other images fall back to the overall class‑wise mean probabilities. This modest change introduces variability into the predictions and moves the score much closer to the target without altering the overall modeling approach.'
- What this solution (achieved 0.53632) has done: 'I add a tiny numeric feature set derived from the `image_id` string (length, digit count, ASCII sum, first‑character code) and train a separate LogisticRegression model for each target column using these features. The models replace the constant class‑mean predictions, giving each test image a distinct probability that is still cheap to compute. This variability should raise the ROC‑AUC above the baseline 0.5 and move the score toward the target while keeping the overall pipeline unchanged. The script now reads the data, builds the features, fits the models, generates predictions (falling back to class means only if a model fails), and writes a valid `submission.csv`.'
- What this solution (achieved 0.49249) has done: 'I added a deterministic lookup that copies the exact training label for any test image appearing in the training set, which gives perfect AUC for those rows. I also enriched the simple ID‑based features with a numeric “id_int” derived from the digits in the image_id, giving the logistic models a bit more signal. The rest of the pipeline stays unchanged, and the script still writes a correctly‑formatted `submission.csv`.'
- What this solution (achieved 0.49456) has done: 'I enhance the ID‑based features (add modulus and a binary flag) and feed them through a `StandardScaler` + `PolynomialFeatures` pipeline before the existing `LogisticRegression`. This keeps the core model type unchanged while giving it richer, normalized inputs that can capture non‑linear patterns, which should raise the ROC‑AUC toward the target. The rest of the script (overlap handling, class‑mean fallback, submission writing) stays the same.'
- What this solution (achieved 0.49669) has done: 'I boost the model’s expressive power with a modest change: use a third‑degree polynomial (instead of degree 2) and allow more iterations for convergence. This keeps the same logistic‑regression pipeline while giving it richer interaction features, which should improve the ROC‑AUC and move the score closer to the target without altering the overall approach.'
- What this solution (achieved 0.49669) has done: 'I add a lightweight nearest‑neighbor fallback based on the numeric part of the image_id. For any test image that isn’t exactly in the training set, the script now finds the training image with the closest id_int value and copies its true labels (probability 1 or 0). This provides more informative predictions than the constant class‑means while keeping the existing logistic‑regression pipeline untouched. The rest of the code (model training, overlap handling, class‑mean fill‑in, and CSV writing) stays the same.'
- What this solution (achieved 0.49669) has done: 'I keep the overall pipeline unchanged but improve the fallback for images that are not in the training set. Instead of copying the single nearest training label, the code now average the labels of the k closest training IDs (k=5) to give a smoother, more informative prediction. I also raise the logistic regression `max_iter` to 1000 to ensure convergence. These minimal tweaks add predictive signal without altering the core model architecture, moving the ROC‑AUC closer to the target.'
- What this solution (achieved 0.52263) has done: 'I keep the overall pipeline unchanged but make two small tweaks that should raise the ROC‑AUC toward the target:  
1. increase the polynomial degree to 4 (giving the logistic model a richer feature set) and switch to the more stable **lbfgs** solver with a larger max_iter;  
2. remove the unconditional class‑mean fallback (the predictions are already defined for every test row, so filling with the overall mean only drags performance toward 0.5).  

These changes preserve the original architecture while providing a modest, deterministic boost in predictive signal.'
- What this solution (achieved 0.47218) has done: 'I replace the logistic‑regression pipeline with a GradientBoostingClassifier (which can capture richer non‑linear patterns from the same ID‑based features) and keep the overlap‑handling that copies exact training labels. The fallback “nearest‑neighbour” block is unnecessary because the model now provides predictions for every test row, so it is left unchanged but never be triggered. This change preserves the overall workflow while giving a stronger model that should raise the ROC‑AUC toward the target.'
- What this solution (achieved 0.48675) has done: 'I enrich the ID‑based numeric features (adding digit‑sum, last‑char code, extra modulus columns) and give the GradientBoosting model a bit more capacity (more estimators and slightly deeper trees). These tweaks keep the overall pipeline and model type unchanged but provide the learner with extra signal, which should raise the ROC‑AUC and move the score closer to the target.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

BASE_INPUT = "../input/plant-pathology-2020-fgvc7"

assert os.path.isfile(os.path.join(BASE_INPUT, "train.csv")), "train.csv not found"
assert os.path.isfile(os.path.join(BASE_INPUT, "test.csv")), "test.csv not found"
assert os.path.isfile(
    os.path.join(BASE_INPUT, "sample_submission.csv")
), "sample_submission.csv not found"

train_df = pd.read_csv(os.path.join(BASE_INPUT, "train.csv"))
test_df = pd.read_csv(os.path.join(BASE_INPUT, "test.csv"))
sample_sub = pd.read_csv(os.path.join(BASE_INPUT, "sample_submission.csv"))

target_cols = ["healthy", "multiple_diseases", "rust", "scab"]




## === cell 1
def extract_id_features(df):
    """Create richer numeric features from the image_id string."""
    ids = df["image_id"].astype(str)

    features = pd.DataFrame()
    features["len_id"] = ids.str.len()
    features["num_digits"] = ids.str.replace(r"\D", "", regex=True).str.len()
    features["digit_sum"] = ids.str.replace(r"\D", "", regex=True).apply(
        lambda x: sum(int(d) for d in x) if x else 0
    )
    features["ascii_sum"] = ids.apply(lambda x: sum(ord(ch) for ch in x))
    features["first_char_ord"] = ids.apply(lambda x: ord(x[0]) if x else 0)
    features["last_char_ord"] = ids.apply(lambda x: ord(x[-1]) if x else 0)
    features["id_int"] = ids.str.replace(r"\D", "", regex=True).fillna("0").astype(int)
    features["id_mod_5"] = features["id_int"] % 5
    features["id_mod_10"] = features["id_int"] % 10
    features["id_mod_100"] = features["id_int"] % 100
    features["is_train_prefix"] = ids.str.startswith("Train_").astype(int)
    return features


X_train_raw = extract_id_features(train_df)
X_test_raw = extract_id_features(test_df)


def build_model():
    return GradientBoostingClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=4,
        random_state=42,
    )


models = {}
for col in target_cols:
    y = train_df[col]
    model = build_model()
    model.fit(X_train_raw, y)
    models[col] = model

test_pred = pd.DataFrame({"image_id": test_df["image_id"]})
for col in target_cols:
    test_pred[col] = models[col].predict_proba(X_test_raw)[:, 1]

overlap_mask = test_pred["image_id"].isin(train_df["image_id"])
if overlap_mask.any():
    train_labels = train_df.set_index("image_id")[target_cols]
    mapped = (
        test_pred.loc[overlap_mask, "image_id"]
        .map(train_labels.to_dict(orient="index"))
        .apply(pd.Series)
    )
    test_pred.loc[overlap_mask, target_cols] = mapped.values

missing_mask = test_pred[target_cols].isna().any(axis=1)
if missing_mask.any():
    train_id_int = X_train_raw["id_int"].values
    sorted_idx = np.argsort(train_id_int)
    sorted_id = train_id_int[sorted_idx]
    sorted_labels = train_df[target_cols].iloc[sorted_idx].reset_index(drop=True)

    k = 5
    for idx in test_pred[missing_mask].index:
        test_int = X_test_raw.loc[idx, "id_int"]
        pos = np.searchsorted(sorted_id, test_int)

        left = max(0, pos - k // 2)
        right = min(len(sorted_id), left + k)
        left = max(0, right - k)

        nearest_labels = sorted_labels.iloc[left:right]
        avg_vals = nearest_labels.mean().values
        test_pred.loc[idx, target_cols] = avg_vals




## === cell 2
submission = sample_sub.copy()
submission = submission.set_index("image_id")
submission.loc[test_pred["image_id"], target_cols] = test_pred[target_cols].values
submission = submission.reset_index()

output_path = "submission.csv"
submission.to_csv(output_path, index=False)
print(f"Submission written to {output_path}")
