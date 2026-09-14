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

0.43721

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.43721) has done: 'I remove the failing TensorFlow model loading and instead compute a simple probability for each test image based on its average pixel intensity. This avoids the protobuf import error, ensures a valid `submission.csv` is created, and keeps the rest of the workflow (data loading, visualization) intact.'
- What this solution (achieved 0.43721) has done: 'Implemented a simple calibration step using the training images: we sample a subset of the training set, compute each image’s mean pixel intensity, and derive min‑max bounds. The probability function now linearly rescales the test‑image mean intensity based on these bounds (clipped to [0,1]), which better aligns the naive intensity feature with the true label distribution and should raise the AUC toward the target. The rest of the pipeline (data loading, visualization, CSV export) remains unchanged.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import random

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
test_df.head()



## === cell 2
train_labels_path = "/kaggle/input/histopathologic-cancer-detection/train_labels.csv"
train_labels = pd.read_csv(train_labels_path)
train_labels["id_file"] = train_labels["id"] + ".tif"

SAMPLE_SIZE = 20000  # adjust if memory/time permits
train_sample = train_labels.sample(n=SAMPLE_SIZE, random_state=SEED)


def _mean_intensity(image_path):
    """Return the mean pixel intensity (0‑255) of a grayscale image."""
    img = np.array(Image.open(image_path).convert("L"))
    return img.mean()


train_means = []
for _, row in train_sample.iterrows():
    img_path = os.path.join(train_images, row["id_file"])
    try:
        train_means.append(_mean_intensity(img_path))
    except Exception:
        continue

train_means = np.array(train_means)
MIN_MEAN = train_means.min()
MAX_MEAN = train_means.max()
print(f"Calibration obtained from {len(train_means)} training images:")
print(f"  min mean intensity = {MIN_MEAN:.2f}")
print(f"  max mean intensity = {MAX_MEAN:.2f}")




## === cell 3
def compute_probability(image_path):
    """
    Compute a probability using the calibrated min‑max scaling of mean intensity.
    The result is clipped to the [0, 1] interval.
    """
    mean_intensity = _mean_intensity(image_path)
    prob = (mean_intensity - MIN_MEAN) / (MAX_MEAN - MIN_MEAN)
    return float(np.clip(prob, 0.0, 1.0))




## === cell 4
probabilities = []
for idx, row in test_df.iterrows():
    img_path = os.path.join(test_images, row["id_file"])
    prob = compute_probability(img_path)
    probabilities.append(prob)

test_df["pred_prob"] = probabilities



## === cell 5
submission = pd.DataFrame({"id": test_df["id"], "label": test_df["pred_prob"]})

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")
print(submission.head())



## === cell 6
plt.figure(figsize=(6, 4))
plt.hist(submission["label"], bins=50, color="steelblue", edgecolor="black")
plt.title("Distribution of Predicted Probabilities")
plt.xlabel("Predicted probability of tumor in centre")
plt.ylabel("Number of images")
plt.show()
