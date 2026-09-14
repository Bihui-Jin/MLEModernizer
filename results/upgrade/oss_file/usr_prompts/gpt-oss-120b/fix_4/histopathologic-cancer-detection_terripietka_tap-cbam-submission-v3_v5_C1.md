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
Given a dataset of images from digital pathology scans, predict if the center 32x32px region of a patch contains at least one pixel of tumor tissue. Tumor tissue in the outer region of the patch does not influence the label. 

## Metric
Area under the ROC curve.

## Submission Format
For each `id` in the test set, you must predict a probability that center 32x32px region of a patch contains at least one pixel of tumor tissue. The file should contain a header and have the following format:

```
id,label
0b2ea2a822ad23fdb1b5dd26653da899fbd2c0d5,0
95596b92e5066c5c52466c90b69ff089b39f2737,0
248e6738860e2ebcf6258cdc1f32f299e0c76914,0
etc.
```

## Dataset
Files are named with an image `id`. The `train_labels.csv` file provides the ground truth for the images in the `train` folder. You are predicting the labels for the images in the `test` folder.

# 2. Python version

3.13

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (63 lines)
            sample_submission.csv (45562 lines)
            sample_submission.csv.zip (1.1 MB)
            test.zip (1.1 GB)
            train.zip (4.2 GB)
            train_labels.csv (174465 lines)
            train_labels.csv.zip (4.2 MB)
            histopathologic-cancer-detection/
                description.md (63 lines)
                sample_submission.csv (45562 lines)
                ... and 5 other files
                histopathologic-cancer-detection/
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
            test/
                7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                ... and 45559 other files
                test/
            train/
                bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                ... and 174462 other files
                train/
        input/
            description.md (63 lines)
            sample_submission.csv (45562 lines)
            sample_submission.csv.zip (1.1 MB)
            test.zip (1.1 GB)
            train.zip (4.2 GB)
            train_labels.csv (174465 lines)
            train_labels.csv.zip (4.2 MB)
            histopathologic-cancer-detection/
                description.md (63 lines)
                sample_submission.csv (45562 lines)
                ... and 5 other files
                histopathologic-cancer-detection/
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
            test/
                7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                ... and 45559 other files
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
            train/
                bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                ... and 174462 other files
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
        working/
            histopathologic-cancer-detection/
                description.md (63 lines)
                sample_submission.csv (45562 lines)
                ... and 5 other files
                histopathologic-cancer-detection/
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
```

-> data/histopathologic-cancer-detection/sample_submission.csv has 45561 rows and 2 columns.
The columns are: id, label

-> data/histopathologic-cancer-detection/train_labels.csv has 174464 rows and 2 columns.
The columns are: id, label

-> data/sample_submission.csv has 45561 rows and 2 columns.
The columns are: id, label

-> data/train_labels.csv has 174464 rows and 2 columns.
The columns are: id, label

-> input/histopathologic-cancer-detection/sample_submission.csv has 45561 rows and 2 columns.
The columns are: id, label

-> input/histopathologic-cancer-detection/train_labels.csv has 174464 rows and 2 columns.
The columns are: id, label

-> (stopped after 10 files for performance)

# 5. Target score

0.753736212726282

# 6. Current score

0.64228

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.43721) has done: 'I remove the failing TensorFlow model loading and instead compute a simple probability for each test image based on its average pixel intensity. This avoids the protobuf import error, ensures a valid `submission.csv` is created, and keeps the rest of the workflow (data loading, visualization) intact.'
- What this solution (achieved 0.43721) has done: 'Implemented a simple calibration step using the training images: we sample a subset of the training set, compute each image’s mean pixel intensity, and derive min‑max bounds. The probability function now linearly rescales the test‑image mean intensity based on these bounds (clipped to [0,1]), which better aligns the naive intensity feature with the true label distribution and should raise the AUC toward the target. The rest of the pipeline (data loading, visualization, CSV export) remains unchanged.'
- What this solution (achieved 0.64228) has done: 'I add a lightweight feature extractor (mean + standard‑deviation of pixel intensities) and fit a simple logistic‑regression on a sampled subset of the training data. The model replace the crude min‑max scaling, keeping the overall workflow unchanged while giving a probability that is better aligned with the true labels, which should raise the AUC toward the target.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import random
from sklearn.linear_model import LogisticRegression

SEED = 42
np.random.seed(SEED)
random.seed(SEED)



## === cell 1
test_images = "/kaggle/input/histopathologic-cancer-detection/test/"
train_images = "/kaggle/input/histopathologic-cancer-detection/train/"

test_df = pd.read_csv(
    "/kaggle/input/histopathologic-cancer-detection/sample_submission.csv"
)
test_df["id_file"] = test_df["id"] + ".tif"
print("Test Set Size:", test_df.shape)



## === cell 2
train_labels_path = "/kaggle/input/histopathologic-cancer-detection/train_labels.csv"
train_labels = pd.read_csv(train_labels_path)
train_labels["id_file"] = train_labels["id"] + ".tif"

SAMPLE_SIZE = 20000  # adjust if memory/time permits
train_sample = train_labels.sample(n=SAMPLE_SIZE, random_state=SEED)


def _mean_intensity(image_path):
    """Mean pixel intensity (0‑255) of a grayscale image."""
    img = np.array(Image.open(image_path).convert("L"))
    return img.mean()


def _std_intensity(image_path):
    """Standard deviation of pixel intensities."""
    img = np.array(Image.open(image_path).convert("L"))
    return img.std()


def _extract_features(image_path):
    """Return a tuple (mean, std) for the given image."""
    img = np.array(Image.open(image_path).convert("L"))
    return img.mean(), img.std()


train_means = []
train_stds = []
valid_labels = []
for _, row in train_sample.iterrows():
    img_path = os.path.join(train_images, row["id_file"])
    try:
        mean, std = _extract_features(img_path)
        train_means.append(mean)
        train_stds.append(std)
        valid_labels.append(row["label"])
    except Exception:
        continue

X_train = np.column_stack([train_means, train_stds])
y_train = np.array(valid_labels)

model = LogisticRegression(max_iter=1000, n_jobs=5, solver="lbfgs")
model.fit(X_train, y_train)
print(f"Logistic regression trained on {X_train.shape[0]} images.")




## === cell 3
def compute_probability(image_path):
    """
    Predict probability using the logistic‑regression model on mean+std features.
    """
    mean, std = _extract_features(image_path)
    prob = model.predict_proba([[mean, std]])[0, 1]
    return float(prob)




## === cell 4
probabilities = []
for _, row in test_df.iterrows():
    img_path = os.path.join(test_images, row["id_file"])
    prob = compute_probability(img_path)
    probabilities.append(prob)

test_df["pred_prob"] = probabilities
submission = pd.DataFrame({"id": test_df["id"], "label": test_df["pred_prob"]})

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")
print(submission.head())



## === cell 5
plt.figure(figsize=(6, 4))
plt.hist(submission["label"], bins=50, color="steelblue", edgecolor="black")
plt.title("Distribution of Predicted Probabilities")
plt.xlabel("Predicted probability of tumor in centre")
plt.ylabel("Number of images")
plt.show()
