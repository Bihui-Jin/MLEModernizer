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

0.9700998924276286

# 6. Current score

0.62512

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'I replace the nonexistent submissions directory with the actual data paths, add a safeguard for empty submission lists, and implement a simple baseline that uses the mean label values from the training set to create a valid `submission.csv`. This resolves the IndexError, ensures the script runs end‑to‑end, and produces a correctly formatted submission file.'
- What this solution (achieved 0.5) has done: 'The fix keeps the original workflow but adds a simple calibration step to the baseline mean‑based predictions: each class probability is stretched away from 0.5 using a configurable `ALPHA` factor (greater than 1). This modest increase in prediction variance often boosts ROC‑AUC without changing the core logic or requiring any new libraries. The rest of the code remains unchanged, and a valid `submission.csv` is still written.'
- What this solution (achieved 0.54806) has done: 'I add a lightweight numeric‑feature model that uses the numeric part of each image_id as a predictor for each disease label. This keeps the original mean‑baseline as a fallback, introduces only a simple scikit‑learn LogisticRegression (which is already available in the Kaggle environment), and gives per‑image varied probabilities so the ROC‑AUC can improve toward the target. The rest of the script remains unchanged.'
- What this solution (achieved 0.51082) has done: 'I keep the overall workflow unchanged but enhance the numeric‑feature baseline: I add polynomial features (degree 3) to capture non‑linear patterns in the image‑id numbers, and I blend each model’s probability with a calibrated mean‑baseline (weight 0.7 model + 0.3 mean). This modest change stays within the original simple logistic‑regression per‑label approach while providing a realistic chance to raise the ROC‑AUC toward the target score.'
- What this solution (achieved 0.50514) has done: 'The update keeps the same numeric‑id baseline but makes it more expressive: it uses a higher‑degree polynomial (5 instead of 3), stretches the mean calibration with a larger ALPHA, gives the logistic model a larger contribution, and adds `class_weight="balanced"` to help rare classes. These tweaks stay within the original logistic‑regression‑on‑id framework while increasing prediction variance, which should raise the ROC‑AUC and move the score closer to the target.'
- What this solution (achieved 0.45194) has done: 'I keep the overall workflow unchanged but add a StandardScaler after creating the polynomial features, increase the logistic‑regression capacity (more iterations and weaker regularisation), and give the model a larger share in the final blend (95 % model + 5 % calibrated mean). These small, targeted tweaks are expected to raise the ROC‑AUC toward the target without altering the core logic of the baseline.'
- What this solution (achieved 0.47471) has done: 'I add a few lightweight numeric features derived from the image ID (modulo 10, modulo 100, and digit‑sum) and increase the polynomial degree slightly to give the existing LogisticRegression a richer signal while keeping the overall workflow unchanged. The blend weight is also softened (0.8 model + 0.2 calibrated) to add a bit more variance, which should improve the ROC‑AUC toward the target without altering the core logic.'
- What this solution (achieved 0.45502) has done: 'I keep the overall workflow unchanged but make the model a bit more expressive and rely more on its predictions: increase the polynomial degree, use a much weaker regularisation (larger C), allow more optimisation iterations, and give the LogisticRegression output a larger contribution in the final blend (95 % model + 5 % calibrated mean). These minimal tweaks stay within the existing logic while increasing prediction variance, which should raise the ROC‑AUC toward the target score.'
- What this solution (achieved 0.57762) has done: 'I add simple image‑based features (mean RGB values) to the existing numeric‑id features and lower the polynomial degree to keep the model tractable. These extra visual cues give the logistic‑regression models more predictive signal, which should raise the ROC‑AUC toward the target while preserving the overall workflow.'
- What this solution (achieved 0.62512) has done: 'I added extra image‑based features (standard‑deviation of each RGB channel) to give the model more visual signal, and increased the polynomial degree from 2 to 3 to let the logistic regressions capture richer interactions. These tweaks stay within the original numeric‑id + logistic‑regression framework, keep the same blending approach, and simply expand the feature matrix, which is expected to raise the ROC‑AUC toward the target while preserving all existing logic.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import re
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.exceptions import ConvergenceWarning
import warnings
from PIL import Image
import numpy as np




## === cell 1
DATA_ROOT = "/kaggle/input/plant-pathology-2020-fgvc7"
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
TEST_CSV = os.path.join(DATA_ROOT, "test.csv")
SAMPLE_SUBMISSION_CSV = os.path.join(DATA_ROOT, "sample_submission.csv")




## === cell 2
def ensemble(submissions_all, sub_idx, weights=[]):
    """
    Combine existing submission files with given weights.
    If submissions_all is empty this function will simply return None.
    """
    if not submissions_all:
        return None
    submission_with_weight = []
    for i in range(len(sub_idx)):
        print(
            f"I'm taking submission {submissions_all[sub_idx[i]]} with weight {weights[i]}"
        )
        submission = pd.read_csv(submissions_all[sub_idx[i]])
        submission = submission.loc[
            :, ["healthy", "multiple_diseases", "rust", "scab"]
        ].values
        submission_with_weight.append(submission * weights[i])
    return sum(submission_with_weight)


def make_submission_file_from_ensemble(submission_avg, submissions_all):
    """Create submission.csv from an ensemble matrix."""
    submission_df = pd.read_csv(submissions_all[0])
    submission_df.iloc[:, 1:] = 0
    submission_df[["healthy", "multiple_diseases", "rust", "scab"]] = submission_avg
    submission_df.to_csv("submission.csv", index=False)




## === cell 3
SUBMISSIONS_PATH = "/kaggle/input/submissions"
submissions_all = []
if os.path.isdir(SUBMISSIONS_PATH):
    for dirname, _, filenames in os.walk(SUBMISSIONS_PATH):
        for filename in filenames:
            submissions_all.append(os.path.join(dirname, filename))
    submissions_all.sort()
print("Found submissions:", submissions_all)




## === cell 4
submission_avg = None
if len(submissions_all) >= 3:
    submission_avg = ensemble(submissions_all, [0, 1, 2], [0.05, 0.9, 0.05])
    if submission_avg is not None:
        make_submission_file_from_ensemble(submission_avg, submissions_all)




## === cell 5
if submission_avg is None:
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)
    label_cols = ["healthy", "multiple_diseases", "rust", "scab"]

    def extract_id_num(img_id):
        nums = re.findall(r"\d+", str(img_id))
        return int(nums[0]) if nums else 0

    def digit_sum(n):
        return sum(int(d) for d in str(abs(n)))

    def compute_rgb_stats(img_id):
        """Return mean and std for each RGB channel."""
        img_path = os.path.join(DATA_ROOT, "images", f"{img_id}.jpg")
        try:
            with Image.open(img_path) as im:
                im = im.convert("RGB")
                arr = np.asarray(im) / 255.0  # shape (H, W, 3)
                mean_vals = arr.mean(axis=(0, 1))  # (R, G, B)
                std_vals = arr.std(axis=(0, 1))  # (R, G, B)
                return list(mean_vals) + list(
                    std_vals
                )  # [mean_r, mean_g, mean_b, std_r, std_g, std_b]
        except Exception:
            return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    for df in (train_df, test_df):
        df["id_num"] = df["image_id"].apply(extract_id_num)
        df["id_mod_10"] = df["id_num"] % 10
        df["id_mod_100"] = df["id_num"] % 100
        df["id_digit_sum"] = df["id_num"].apply(digit_sum)
        rgb_stats = df["image_id"].apply(compute_rgb_stats)
        df["mean_r"] = rgb_stats.apply(lambda x: x[0])
        df["mean_g"] = rgb_stats.apply(lambda x: x[1])
        df["mean_b"] = rgb_stats.apply(lambda x: x[2])
        df["std_r"] = rgb_stats.apply(lambda x: x[3])
        df["std_g"] = rgb_stats.apply(lambda x: x[4])
        df["std_b"] = rgb_stats.apply(lambda x: x[5])

    feature_cols = [
        "id_num",
        "id_mod_10",
        "id_mod_100",
        "id_digit_sum",
        "mean_r",
        "mean_g",
        "mean_b",
        "std_r",
        "std_g",
        "std_b",
    ]

    poly = PolynomialFeatures(degree=3, include_bias=False)
    X_train_poly = poly.fit_transform(train_df[feature_cols].values)
    X_test_poly = poly.transform(test_df[feature_cols].values)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_poly)
    X_test = scaler.transform(X_test_poly)

    preds = pd.DataFrame()
    preds["image_id"] = test_df["image_id"]

    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    for col in label_cols:
        y_train = train_df[col].values
        mean_val = y_train.mean()

        calibrated = mean_val

        if y_train.min() == y_train.max():
            preds[col] = calibrated
            continue

        model = LogisticRegression(
            solver="lbfgs",
            max_iter=2000,
            C=100.0,
            class_weight="balanced",
        )
        model.fit(X_train, y_train)
        prob_pos = model.predict_proba(X_test)[:, 1]

        blended = 0.95 * prob_pos + 0.05 * calibrated
        preds[col] = blended.clip(0.0, 1.0)

    sample_sub = pd.read_csv(SAMPLE_SUBMISSION_CSV)
    preds = preds[sample_sub.columns]

    preds.to_csv("submission.csv", index=False)
    print(
        "Added RGB std features and increased polynomial degree to 3; "
        "trained per‑label logistic models and wrote submission.csv."
    )
