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

0.9693194356375324

# 6. Current score

0.5291

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'I replace the faulty ensemble logic with a simple baseline that reads the training labels, computes the overall mean probability for each disease, and assigns these constant predictions to every test image. This removes the out‑of‑range index error, guarantees a correctly formatted `submission.csv`, and provides a reasonable score without altering any core modeling approach.'
- What this solution (achieved 0.47271) has done: 'I replace the constant‑mean predictions with a simple nearest‑neighbor lookup based on the numeric part of each image filename: for every test image I find the training image whose number is closest and copy its exact label values (treated as probabilities). This keeps the overall pipeline unchanged while providing more informative, image‑specific predictions, which should raise the ROC‑AUC toward the target score.'
- What this solution (achieved 0.50188) has done: 'I replace the single‑nearest‑neighbor lookup with a small‑k nearest‑neighbor average (k=5). This keeps the overall “lookup‑based” idea but provides smoother, more informative probability estimates, which should raise the ROC‑AUC toward the target while preserving the existing pipeline structure. I also import NumPy for efficient distance handling.'
- What this solution (achieved 0.5271) has done: 'I keep the original k‑nearest‑neighbour logic but increase k to smooth predictions, and add a lightweight logistic‑regression model using the numeric part of the image IDs as a single feature. The two sets of probabilities are then blended (50 % each) to give more calibrated predictions, which should raise the ROC‑AUC toward the target while preserving the overall pipeline.'
- What this solution (achieved 0.50652) has done: 'I replace the simple un‑weighted k‑nearest average with a distance‑weighted average (giving closer images more influence) and adjust the blend to rely more on the logistic‑regression predictions, which tend to capture the monotonic trend of the numeric IDs better. I also set `class_weight="balanced"` for each logistic model to improve calibration for the minority disease classes. These changes keep the overall pipeline (numeric‑ID‑based lookup + logistic blend) while providing a more informative prediction that should raise the ROC‑AUC toward the target.'
- What this solution (achieved 0.50652) has done: 'I keep the same numeric‑ID‑based k‑nearest‑neighbour + logistic‑regression pipeline but adjust the hyper‑parameters so the model leans more on the neighbour average (which captures the stronger ID‑related patterns) and uses a larger neighbourhood. Specifically, I increase k to 200 and change the blending weight to 70 % average + 30 % logistic. These modest changes preserve the original logic while giving the predictions a better calibrated signal, which should raise the ROC‑AUC toward the target.'
- What this solution (achieved 0.5139) has done: 'I replace the numeric‑ID based neighbour model with a tiny image‑based logistic regression that uses low‑resolution grayscale pixel values as features. This keeps the overall structure (train → predict → blend) but adds a far more informative feature set, which is expected to raise the ROC‑AUC toward the target. The rest of the script (paths, CSV handling, and submission writing) remains unchanged.'
- What this solution (achieved 0.49909) has done: 'I increase the image resolution to capture more detail, raise the logistic regression iterations for better convergence, and add a lightweight numeric‑ID based model whose predictions are blended 50/50 with the image‑based predictions. This preserves the original pipeline while giving the model extra signal, which should raise the ROC‑AUC toward the target score.'
- What this solution (achieved 0.48752) has done: 'I add a lightweight k‑nearest‑neighbour (k = 20) predictor that uses the same flattened grayscale image features already computed. For each target column it averages the labels of the 20 closest training images and blends this neighbor estimate with the existing image‑based and ID‑based logistic predictions (40 % neighbor, 30 % image logistic, 30 % ID logistic). This small addition keeps the original pipeline intact while giving the model more informative, image‑specific signals, which should raise the ROC‑AUC toward the target.'
- What this solution (achieved 0.45407) has done: 'I add a StandardScaler to normalize the pixel features, increase the neighbour size to k=50 and replace the simple mean with a distance‑weighted average, then give the neighbour predictions a larger share in the final blend (0.6 vs 0.3). These tweaks keep the original pipeline but provide more informative, calibrated predictions, which should raise the ROC‑AUC toward the target score.'
- What this solution (achieved 0.53044) has done: 'I keep the overall pipeline unchanged but make two small adjustments that are expected to raise the ROC‑AUC toward the target: (1) reduce the neighbour size k from 50 to 5 so that each test image is averaged over its most similar neighbours rather than a very large, overly‑smoothed set, and (2) increase the neighbour contribution in the final blend to 0.8 while decreasing the image‑logistic and ID‑logistic weights to 0.1 each. These minimal changes preserve the core logic and model architecture while giving the more informative neighbour signal a stronger influence, which should improve the ranking quality and move the score closer to the target.'
- What this solution (achieved 0.51683) has done: 'I increase the image resolution to capture more detail, add a PCA step to create a compact yet informative feature space, and modestly adjust the blending weights and neighbour count. These changes keep the overall pipeline (image‑logistic, id‑logistic, k‑NN blend) intact while giving the models richer signals, which should raise the ROC‑AUC toward the target.'
- What this solution (achieved 0.51926) has done: 'I keep the overall pipeline (image‑based logistic regression, ID‑based logistic regression, distance‑weighted k‑NN on image features and blending) but make a few targeted tweaks that are expected to raise the ROC‑AUC: increase image resolution to capture more detail, keep more PCA components, use a smaller, more focused neighbourhood (k = 5), and shift the blending weights toward the image‑based model (0.5 img + 0.3 neighbor + 0.2 id). These adjustments are minimal, preserve the core logic, and should move the score closer to the target.'
- What this solution (achieved 0.4949) has done: 'I increase the image resolution and retain more variance in the PCA step so the image‑based logistic model gets richer features, raise the neighbour count to 10 and give the distance‑weighted neighbour average a much larger share in the final blend (80 %). The image‑logistic and ID‑logistic contributions are reduced accordingly. These tweaks keep the original pipeline unchanged while providing stronger, better‑calibrated signals, which should move the ROC‑AUC score significantly closer to the target.'
- What this solution (achieved 0.5291) has done: 'I slightly adjust the neighbour‑based part to use a smaller, more focused neighbourhood (k = 5) and rebalance the blend so the image‑logistic model contributes more strongly while still keeping the neighbour signal. This modest change is expected to give clearer, less‑over‑smoothed predictions and raise the ROC‑AUC toward the target without altering the overall pipeline.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA




## === cell 1
DATA_ROOT = "/kaggle/input/plant-pathology-2020-fgvc7"

TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
TEST_CSV = os.path.join(DATA_ROOT, "test.csv")
SAMPLE_SUBMISSION_CSV = os.path.join(DATA_ROOT, "sample_submission.csv")
IMAGE_DIR = os.path.join(DATA_ROOT, "images")




## === cell 2
train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

target_cols = ["healthy", "multiple_diseases", "rust", "scab"]


def load_image_features(df, img_dir, size=(300, 300)):
    """
    Load images referenced by `image_id` in `df`, resize to `size`,
    convert to grayscale, flatten, and scale to [0, 1].
    Returns a NumPy array of shape (len(df), size[0]*size[1]).
    """
    features = []
    for img_id in df["image_id"]:
        img_path = os.path.join(img_dir, f"{img_id}.jpg")
        if not os.path.exists(img_path):
            img_path = os.path.join(img_dir, f"{img_id}.JPG")
        with Image.open(img_path) as im:
            im = im.convert("L")  # grayscale
            im = im.resize(size, Image.BILINEAR)
            arr = np.asarray(im, dtype=np.float32) / 255.0
            features.append(arr.ravel())
    return np.stack(features)


X_train_img = load_image_features(train_df, IMAGE_DIR, size=(300, 300))
X_test_img = load_image_features(test_df, IMAGE_DIR, size=(300, 300))

scaler = StandardScaler()
X_train_img = scaler.fit_transform(X_train_img)
X_test_img = scaler.transform(X_test_img)

pca = PCA(n_components=400, random_state=42)
X_train_img = pca.fit_transform(X_train_img)
X_test_img = pca.transform(X_test_img)




## === cell 3
logistic_img_preds = {}
for col in target_cols:
    y = train_df[col].values
    lr = LogisticRegression(solver="liblinear", class_weight="balanced", max_iter=1000)
    lr.fit(X_train_img, y)
    logistic_img_preds[col] = lr.predict_proba(X_test_img)[:, 1]


def extract_numeric_id(series):
    return series.str.split("_").str[1].astype(int).values.reshape(-1, 1)


X_train_id = extract_numeric_id(train_df["image_id"])
X_test_id = extract_numeric_id(test_df["image_id"])

logistic_id_preds = {}
for col in target_cols:
    y = train_df[col].values
    lr_id = LogisticRegression(
        solver="liblinear", class_weight="balanced", max_iter=1000
    )
    lr_id.fit(X_train_id, y)
    logistic_id_preds[col] = lr_id.predict_proba(X_test_id)[:, 1]

k = 5

dists = np.sum(
    (X_test_img[:, np.newaxis, :] - X_train_img[np.newaxis, :, :]) ** 2, axis=2
)

neighbor_preds = {}
n_test = X_test_img.shape[0]
idx_k = np.argpartition(dists, kth=k, axis=1)[:, :k]  # (n_test, k)

for col in target_cols:
    y_train_col = train_df[col].values
    neigh_dist = dists[np.arange(n_test)[:, None], idx_k]  # (n_test, k)
    weights = 1.0 / (np.sqrt(neigh_dist) + 1e-5)
    weighted_sum = np.sum(weights * y_train_col[idx_k], axis=1)
    weight_total = np.sum(weights, axis=1)
    neighbor_means = weighted_sum / weight_total
    neighbor_preds[col] = neighbor_means

blended_preds = {}
for col in target_cols:
    blended_preds[col] = (
        0.5 * neighbor_preds[col]  # neighbour average
        + 0.4 * logistic_img_preds[col]  # image‑based logistic
        + 0.1 * logistic_id_preds[col]  # ID‑based logistic
    )
    blended_preds[col] = np.clip(blended_preds[col], 0.0, 1.0)

logistic_img_df = pd.DataFrame(blended_preds)




## === cell 4
submission_df = pd.DataFrame()
submission_df["image_id"] = test_df["image_id"]
for col in target_cols:
    submission_df[col] = logistic_img_df[col]




## === cell 5
submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")
