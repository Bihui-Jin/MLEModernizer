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
Classify each cassava image into four disease categories or a fifth category indicating a healthy leaf.

## Metric
Categorization accuracy.

## Submission Format
```
image_id,label
1000471002.jpg,4
1000840542.jpg,4
etc.
```

## Dataset
**[train/test]_images** the image files.

**train.csv**

- `image_id` the image file name.

- `label` the ID code for the disease.

**sample_submission.csv** A properly formatted sample submission, given the disclosed test set content.

- `image_id` the image file name.

- `label` the predicted ID code for the disease.

**[train/test]_tfrecords** the image files in tfrecord format.

**label_num_to_disease_map.json** The mapping between each disease code and the real disease name.

# 2. Python version

3.13

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        input/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        working/
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
```

-> data/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/cassava-leaf-disease-classification/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/cassava-leaf-disease-classification/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> data/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> input/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> (stopped after 10 files for performance)

# 5. Target score

0.8445149592021759

# 6. Current score

0.61435

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.61099) has done: 'The fix removes the failing model loading and TensorFlow code, replaces it with a simple baseline that predicts the most common label from the training data for every test image, and ensures the script runs end‑to‑end and writes a correctly formatted `submission.csv`.'
- What this solution (achieved 0.05531) has done: 'I replace the naive majority‑label baseline with a tiny image‑based classifier that uses each image’s average RGB colour as a feature. By computing the mean colour for every training image and averaging these per class, we obtain a quick “colour prototype” for each disease. Each test image is then assigned to the class whose prototype is closest in Euclidean distance. This adds only a few lines, keeps the overall pipeline unchanged, and is expected to raise the accuracy from ~0.61 toward the target 0.84 while still writing a correct `submission.csv`.'
- What this solution (achieved 0.47608) has done: 'The script is rewritten to parallelize the RGB‑stat extraction for both training and test images using a thread pool, pre‑allocate the feature arrays, and replace the per‑image Python loop with a single NumPy distance‑matrix computation. This removes the costly Python‑level iteration over 2 600 test images while keeping the exact K‑nearest‑neighbor logic, and the memory usage of the distance matrix (~400 MB) fits comfortably in the environment. The overall I/O work is unchanged, but the parallel extraction and vectorized nearest‑neighbor search bring the runtime well under the 600‑second limit.'
- What this solution (achieved 0.51158) has done: 'Implemented a small yet effective tweak: switched from a 1‑nearest‑neighbor rule to a 3‑nearest‑neighbor majority vote on the same RGB mean‑std features. This modest change keeps the original feature extraction and overall pipeline intact while typically boosting classification accuracy, moving the score closer to the target. The rest of the script (data loading, feature computation, and CSV output) remains unchanged.'
- What this solution (achieved 0.58707) has done: 'I expand the image feature vector by also adding mean and standard‑deviation of the HSV colour space, increase the neighbour count to 5, and replace the majority‑vote with a distance‑weighted vote. These adjustments keep the overall pipeline intact while giving the classifier richer colour information and a more nuanced K‑NN decision, which should raise the validation accuracy toward the target.'
- What this solution (achieved 0.58146) has done: 'I normalize the RGB‑HSV feature vectors so that each dimension has zero mean and unit variance across the training set before computing distances. This simple scaling often makes Euclidean nearest‑neighbour similarity more meaningful and should raise the validation accuracy, moving the score closer to the target while keeping the overall pipeline unchanged.'
- What this solution (achieved 0.59081) has done: 'I replace the Euclidean‑distance based K‑NN with a cosine‑similarity version while keeping the same RGB‑HSV feature extraction and overall pipeline. After standardising the features I L2‑normalise them, compute similarity scores, pick the top k (nearest 7) neighbours and use their similarity as voting weights. This small change usually yields a more meaningful neighbourhood for colour‑based features and is expected to raise the validation accuracy toward the target without altering the core logic.'
- What this solution (achieved 0.61435) has done: 'I increase the number of neighbours used in the cosine‑similarity weighted voting from 7 to 15. A larger k generally provides a more stable vote in this colour‑statistics‑based K‑NN, which should raise the validation accuracy and move the leaderboard score closer to the target while keeping the original pipeline unchanged.'
- What this solution (achieved 0.5994) has done: 'I remove the z‑score standardisation of the RGB‑HSV statistics and keep only L2 normalisation before computing cosine similarity, and I lower the neighbour count from 15 to 7 (a smaller k often yields higher accuracy for this colour‑based K‑NN). These tiny adjustments keep the overall pipeline unchanged while expectedly moving the validation accuracy upward toward the target score.'
- What this solution (achieved 0.61697) has done: 'I increase the neighbour count from 7 to 15, which in earlier tests slightly raised the validation accuracy while keeping the same feature extraction, cosine‑similarity and weighted voting logic. This small change keeps the core pipeline intact and should move the score closer to the target.'
- What this solution (achieved 0.61435) has done: 'I add simple z‑score standardisation of the 12‑dim colour statistics before the L2‑normalisation. Scaling the features makes the cosine similarity compare comparable RGB and HSV information, which should raise the validation accuracy a bit and move the score closer to the target while keeping the overall K‑NN pipeline unchanged.'
- What this solution (achieved 0.61846) has done: 'Implemented a modest tweak to the K‑NN inference: increased the neighbourhood size from 15 to 30 while keeping the existing cosine‑similarity weighting. This small adjustment preserves the entire feature extraction and normalization pipeline but allows the classifier to aggregate information from more neighbours, which is expected to move the validation accuracy closer to the target score.'
- What this solution (achieved 0.61435) has done: 'I adjust the neighbour count to a more moderate k = 15 and amplify the influence of closer neighbours by squaring the cosine‑similarity weights before the weighted vote. This keeps the overall pipeline unchanged while giving the classifier a sharper focus on the most similar images, which should raise the validation accuracy toward the target score.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
from PIL import Image
import concurrent.futures

train_csv_path = "/kaggle/input/cassava-leaf-disease-classification/train.csv"
train_images_dir = "/kaggle/input/cassava-leaf-disease-classification/train_images"
test_images_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images"

train_df = pd.read_csv(train_csv_path)


def rgb_hsv_stats(image_path: str) -> np.ndarray:
    """Return mean and std of RGB and HSV channels as a 12‑dim vector."""
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        arr_rgb = np.array(img).astype(np.float32) / 255.0
        mean_rgb = arr_rgb.mean(axis=(0, 1))
        std_rgb = arr_rgb.std(axis=(0, 1))

        img_hsv = img.convert("HSV")
        arr_hsv = np.array(img_hsv).astype(np.float32) / 255.0
        mean_hsv = arr_hsv.mean(axis=(0, 1))
        std_hsv = arr_hsv.std(axis=(0, 1))

        return np.concatenate([mean_rgb, std_rgb, mean_hsv, std_hsv])


def compute_features(image_paths):
    """Compute rgb_hsv_stats for a list of image paths using a thread pool."""
    features = np.empty((len(image_paths), 12), dtype=np.float32)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for idx, feat in enumerate(executor.map(rgb_hsv_stats, image_paths)):
            features[idx] = feat
    return features


train_image_paths = [
    os.path.join(train_images_dir, img_name)
    for img_name in train_df["image_id"]
    if os.path.exists(os.path.join(train_images_dir, img_name))
]

train_features = compute_features(train_image_paths)
train_labels = (
    train_df.loc[
        train_df["image_id"].isin([os.path.basename(p) for p in train_image_paths]),
        "label",
    ]
    .astype(int)
    .to_numpy()
)

test_image_names = sorted(
    [
        entry.name
        for entry in os.scandir(test_images_dir)
        if entry.is_file() and entry.name.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
)

test_image_paths = [os.path.join(test_images_dir, name) for name in test_image_names]
test_features = compute_features(test_image_paths)

feat_mean = train_features.mean(axis=0, keepdims=True)
feat_std = train_features.std(axis=0, keepdims=True) + 1e-6

train_features_std = (train_features - feat_mean) / feat_std
test_features_std = (test_features - feat_mean) / feat_std

train_norm = train_features_std / (
    np.linalg.norm(train_features_std, axis=1, keepdims=True) + 1e-6
)
test_norm = test_features_std / (
    np.linalg.norm(test_features_std, axis=1, keepdims=True) + 1e-6
)

similarity = train_norm @ test_norm.T

k = 15  # reduced from 30
nearest_idxs = np.argpartition(-similarity, kth=k - 1, axis=0)[:k]  # (k, N_test)
nearest_labels = train_labels[nearest_idxs]  # (k, N_test)
nearest_sims = np.take_along_axis(similarity, nearest_idxs, axis=0)  # (k, N_test)

weights = nearest_sims**2

N_test = nearest_labels.shape[1]
pred_labels = np.empty(N_test, dtype=int)

for i in range(N_test):
    labels = nearest_labels[:, i]
    w = weights[:, i]
    pred_labels[i] = np.argmax(np.bincount(labels, weights=w, minlength=5))

submission = pd.DataFrame({"image_id": test_image_names, "label": pred_labels})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path} with {len(submission)} rows.")



## === cell 1
submission.head()
