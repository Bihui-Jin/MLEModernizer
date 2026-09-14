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

0.9697621918404656

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'I correct the file paths so the script can actually locate the training and sample‑submission CSVs inside the Kaggle input directory, and point the submissions search to the working directory (where previous submissions would be saved). These minimal fixes ensure the fallback baseline runs without a FileNotFoundError and creates a proper `submission.csv` that Kaggle accept.'
- What this solution (achieved 0.5) has done: 'I add a lightweight image‑based logistic‑regression model that uses the mean RGB values of each leaf photo as features. This model is trained on the provided training CSV and its image files, then predicts probabilities for the test set. The original fallback‑baseline remains as a safety net, and the rest of the script (ensemble handling, submission writing) is unchanged. This should give the classifier some discriminative power, moving the ROC‑AUC score significantly closer to the target.'
- What this solution (achieved 0.5) has done: 'I fix the missing numpy import, extend the image features to include per‑channel standard deviation (giving six simple features per image), and replace the linear logistic‑regression model with a MultiOutput RandomForest classifier, which usually yields higher ROC‑AUC while keeping the overall pipeline unchanged. These changes are minimal, preserve the original workflow, and should raise the validation score toward the target.'
- What this solution (achieved 0.5) has done: 'The changes parallelize image feature extraction using a thread pool (which speeds up the I/O‑bound loading without altering any model or ensemble logic) and replace the slower `DataFrame.iterrows()` loops with `itertuples()` for deterministic, faster iteration. All other logic, including model architecture, training, ensembling, and fallbacks, remains unchanged.'
- What this solution (achieved 0.69795) has done: 'I fix the training routine so that the label matrix `y` is built correctly from the dataframe (the previous code tried to index a tuple with a list, causing an exception and forcing the fallback baseline, which only yields ~0.5 AUC). By extracting the target columns directly from `train_df`, the RandomForest model actually be trained and its predictions used, moving the ROC‑AUC score upward toward the target.'
- What this solution (achieved 0.69307) has done: 'I boost the RandomForest by using more trees (n_estimators = 800) and enrich the image feature vector with an extra edge‑sharpness statistic (variance of the Laplacian computed from the grayscale image). Both tweaks keep the same overall pipeline and model type while giving the classifier more expressive power, which should raise the ROC‑AUC toward the target score.'

# 9. Code solution

## === cell 0
import pandas as pd
import os
from pathlib import Path
import numpy as np  # needed for image feature calculations
from PIL import Image  # import once to avoid repeated imports in workers

DATA_ROOT = "/kaggle/input/plant-pathology-2020-fgvc7"
SUBMISSIONS_PATH = "/kaggle/working"




## === cell 1
submissions_all = []
for dirname, _, filenames in os.walk(SUBMISSIONS_PATH):
    for filename in filenames:
        if filename.lower().endswith(".csv"):
            submissions_all.append(os.path.join(dirname, filename))
submissions_all.sort()
print("Found submissions:", submissions_all)




## === cell 2
def ensemble(submissions_all, sub_idx, weights=[]):
    """Ensemble a set of existing submission CSVs using provided weights."""
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
    submission_avg = sum(submission_with_weight)
    return submission_avg




## === cell 3
def make_submission_file(submission_avg, submissions_all):
    """Write the averaged predictions to a CSV matching the required format."""
    submission_df = pd.read_csv(submissions_all[0])
    submission_df[["healthy", "multiple_diseases", "rust", "scab"]] = submission_avg
    submission_df.to_csv("submission.csv", index=False)
    print("Submission written to submission.csv")




## === cell 4
def _load_image_features(image_path):
    """Return extended RGB, HSV, grayscale, edge‑sharpness, plus skew/kurtosis statistics."""
    try:
        img_rgb = Image.open(image_path).convert("RGB")
        arr_rgb = np.asarray(img_rgb).astype("float32") / 255.0  # (H, W, 3)

        rgb_mean = arr_rgb.mean(axis=(0, 1))
        rgb_std = arr_rgb.std(axis=(0, 1))

        rgb_centered = arr_rgb - rgb_mean
        rgb_std_adj = np.where(rgb_std == 0, 1e-6, rgb_std)
        rgb_skew = (rgb_centered**3).mean(axis=(0, 1)) / (rgb_std_adj**3)
        rgb_kurt = (rgb_centered**4).mean(axis=(0, 1)) / (rgb_std_adj**4) - 3

        img_hsv = img_rgb.convert("HSV")
        arr_hsv = np.asarray(img_hsv).astype("float32") / 255.0
        hsv_mean = arr_hsv.mean(axis=(0, 1))
        hsv_std = arr_hsv.std(axis=(0, 1))

        hsv_centered = arr_hsv - hsv_mean
        hsv_std_adj = np.where(hsv_std == 0, 1e-6, hsv_std)
        hsv_skew = (hsv_centered**3).mean(axis=(0, 1)) / (hsv_std_adj**3)
        hsv_kurt = (hsv_centered**4).mean(axis=(0, 1)) / (hsv_std_adj**4) - 3

        gray = (
            0.2989 * arr_rgb[..., 0]
            + 0.5870 * arr_rgb[..., 1]
            + 0.1140 * arr_rgb[..., 2]
        )
        gray_mean = gray.mean()
        gray_std = gray.std()

        laplacian = np.gradient(np.gradient(gray, axis=0), axis=0) + np.gradient(
            np.gradient(gray, axis=1), axis=1
        )
        lap_var = laplacian.var()

        return np.concatenate(
            [
                rgb_mean,
                rgb_std,
                rgb_skew,
                rgb_kurt,
                hsv_mean,
                hsv_std,
                hsv_skew,
                hsv_kurt,
                [gray_mean, gray_std, lap_var],
            ]
        )  # (23,)
    except Exception as e:
        print(f"Warning: could not read {image_path}: {e}")
        return np.zeros(23, dtype="float32")


def _batch_load_features(paths):
    """Load image features for a list of paths using a process pool to avoid GIL contention."""
    import concurrent.futures

    max_workers = os.cpu_count() or 4
    results = [None] * len(paths)
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(_load_image_features, p): i for i, p in enumerate(paths)
        }
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            results[idx] = future.result()
    return np.stack(results)




## === cell 5
def train_simple_model():
    """
    Train a basic multi‑label classifier using extended color features.
    Returns a DataFrame with predictions for the test set.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.multioutput import MultiOutputClassifier

    train_path = os.path.join(DATA_ROOT, "train.csv")
    train_df = pd.read_csv(train_path)

    img_dir = Path(DATA_ROOT) / "images"

    train_image_paths = [
        str(img_dir / f"{row.image_id}.jpg") for row in train_df.itertuples(index=False)
    ]
    X = _batch_load_features(train_image_paths)  # (n_samples, 23)

    y = train_df[
        ["healthy", "multiple_diseases", "rust", "scab"]
    ].values  # (n_samples, 4)

    rf = RandomForestClassifier(
        n_estimators=1200,  # increased trees for better performance
        max_depth=None,
        n_jobs=-1,
        random_state=42,
        min_samples_leaf=1,
        class_weight="balanced",  # help with label imbalance
    )
    clf = MultiOutputClassifier(rf)
    clf.fit(X, y)

    test_path = os.path.join(DATA_ROOT, "test.csv")
    test_df = pd.read_csv(test_path)

    test_image_paths = [
        str(img_dir / f"{row.image_id}.jpg") for row in test_df.itertuples(index=False)
    ]
    X_test = _batch_load_features(test_image_paths)

    prob_arrays = [est.predict_proba(X_test)[:, 1] for est in clf.estimators_]
    probs = np.column_stack(prob_arrays)  # shape (n_test, 4)

    sample_sub_path = os.path.join(DATA_ROOT, "sample_submission.csv")
    submission_df = pd.read_csv(sample_sub_path)
    submission_df[["healthy", "multiple_diseases", "rust", "scab"]] = probs
    return submission_df




## === cell 6
def fallback_baseline():
    """Create a simple baseline submission using the mean label frequencies from training data."""
    train_path = os.path.join(DATA_ROOT, "train.csv")
    sample_sub_path = os.path.join(DATA_ROOT, "sample_submission.csv")
    print(f"Loading train data from {train_path}")
    train_df = pd.read_csv(train_path)

    class_means = (
        train_df[["healthy", "multiple_diseases", "rust", "scab"]].mean().values
    )
    print("Class mean probabilities:", class_means)

    sample_sub = pd.read_csv(sample_sub_path)
    for col, prob in zip(["healthy", "multiple_diseases", "rust", "scab"], class_means):
        sample_sub[col] = prob

    sample_sub.to_csv("submission.csv", index=False)
    print("Baseline submission written to submission.csv")




## === cell 7
if submissions_all:
    try:
        submission_avg = ensemble(submissions_all, [0, 1], [0.3, 0.7])
        make_submission_file(submission_avg, submissions_all)
    except Exception as e:
        print("Ensembling failed:", e)
        print("Falling back to model‑based submission.")
        try:
            model_submission = train_simple_model()
            model_submission.to_csv("submission.csv", index=False)
            print("Model‑based submission written to submission.csv")
        except Exception as me:
            print("Model training failed:", me)
            print("Falling back to baseline submission.")
            fallback_baseline()
else:
    print("No previous submissions found; using model‑based submission.")
    try:
        model_submission = train_simple_model()
        model_submission.to_csv("submission.csv", index=False)
        print("Model‑based submission written to submission.csv")
    except Exception as me:
        print("Model training failed:", me)
        print("Falling back to baseline submission.")
        fallback_baseline()
