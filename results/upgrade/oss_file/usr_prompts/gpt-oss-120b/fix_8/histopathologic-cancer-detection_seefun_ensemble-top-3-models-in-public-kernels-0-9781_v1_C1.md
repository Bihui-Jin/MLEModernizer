# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

3.7

# 3. Installed packages

geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scipy==1.15.3
sklearn-pandas==2.2.0

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

# 5. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import scipy.special
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sigmoid = lambda x: scipy.special.expit(x)




## === cell 1
sample_path_candidates = [
    "../input/histopathologic-cancer-detection/sample_submission.csv",
    "../input/sample_submission.csv",
    "../input/histopathologic-cancer-detection/sample_submission.csv.zip",
    "../input/sample_submission.csv.zip",
]

for path in sample_path_candidates:
    if os.path.exists(path):
        sample_sub = pd.read_csv(path)
        break
else:
    raise FileNotFoundError("Sample submission file not found in expected locations.")

train_labels_path = None
for path in [
    "../input/histopathologic-cancer-detection/train_labels.csv",
    "../input/train_labels.csv",
    "../input/histopathologic-cancer-detection/train_labels.csv.zip",
    "../input/train_labels.csv.zip",
]:
    if os.path.exists(path):
        train_labels_path = path
        break

if train_labels_path is None:
    raise FileNotFoundError("train_labels.csv not found in expected locations.")

train_labels = pd.read_csv(train_labels_path)




## === cell 2
def find_dir(possible_paths):
    for p in possible_paths:
        if os.path.isdir(p):
            return p
    raise FileNotFoundError("Directory not found among candidates.")


train_dir = find_dir(
    [
        "../input/histopathologic-cancer-detection/train",
        "../input/train",
        "../input/histopathologic-cancer-detection/train/",
        "../input/train/",
    ]
)

test_dir = find_dir(
    [
        "../input/histopathologic-cancer-detection/test",
        "../input/test",
        "../input/histopathologic-cancer-detection/test/",
        "../input/test/",
    ]
)




## === cell 3
def image_channel_stats(filepath):
    """
    Returns per‑channel mean, std, median (R, G, B) plus overall mean, overall std, overall median.
    If the image is single‑channel, the same value is replicated for each channel.
    """
    try:
        img = Image.open(filepath)
        arr = np.asarray(img).astype(np.float32)
        if arr.ndim == 2:  # grayscale
            arr = np.stack([arr, arr, arr], axis=2)
        elif arr.ndim == 3 and arr.shape[2] == 4:  # RGBA -> drop alpha
            arr = arr[:, :, :3]

        means = arr.mean(axis=(0, 1))
        stds = arr.std(axis=(0, 1))
        medians = np.median(arr, axis=(0, 1))

        overall_mean = means.mean()
        overall_std = stds.mean()
        overall_median = medians.mean()

        return (
            list(means)
            + list(stds)
            + list(medians)
            + [overall_mean, overall_std, overall_median]
        )  # length 12
    except Exception:
        return [np.nan] * 12


subset_size = len(train_labels)  # process all available images
np.random.seed(42)
subset_ids = np.random.choice(
    train_labels["id"].values, size=min(subset_size, len(train_labels)), replace=False
)

features = []
labels = []

tumor_means = []
non_tumor_means = []

for img_id in subset_ids:
    lbl = train_labels.loc[train_labels["id"] == img_id, "label"].values[0]
    img_path = os.path.join(train_dir, f"{img_id}.tif")
    stats = image_channel_stats(img_path)
    if np.isnan(stats).any():
        continue
    features.append(stats)
    labels.append(lbl)
    overall_mean = np.mean(stats[:3])  # mean of the per‑channel means
    if lbl == 1:
        tumor_means.append(overall_mean)
    else:
        non_tumor_means.append(overall_mean)

tumor_mean = np.mean(tumor_means) if tumor_means else 0.0
non_tumor_mean = np.mean(non_tumor_means) if non_tumor_means else 0.0
global_min = min(tumor_mean, non_tumor_mean, 0.0)
global_max = max(tumor_mean, non_tumor_mean, 255.0)

if len(features) >= 2:
    X = np.array(features)  # shape (n_samples, 12)
    y = np.array(labels)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lr_model = LogisticRegression(
        solver="lbfgs",
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
        C=1.0,
    )
    lr_model.fit(X_scaled, y)
else:
    lr_model = None
    scaler = None  # fallback will use simple scaling


def stats_to_prob(stats):
    """
    Map image statistics to a probability.
    Uses the calibrated logistic‑regression model when possible,
    otherwise falls back to overall tumor rate or min‑max scaling.
    """
    if np.isnan(stats).any():
        return float(train_labels["label"].mean())
    if lr_model is not None:
        stats_arr = np.array(stats).reshape(1, -1)
        stats_scaled = scaler.transform(stats_arr)
        prob = lr_model.predict_proba(stats_scaled)[0, 1]
        return float(prob)
    overall_mean = np.mean(stats[:3])
    return (overall_mean - global_min) / (global_max - global_min)




## === cell 4
preds = []
for idx, row in sample_sub.iterrows():
    img_id = row["id"]
    img_path = os.path.join(test_dir, f"{img_id}.tif")
    stats = np.array(image_channel_stats(img_path))
    prob = stats_to_prob(stats)
    preds.append(prob)

sample_sub["label"] = preds

print(sample_sub.head())




## === cell 5
output_path = "ensemble.csv"
sample_sub.to_csv(output_path, index=False)
print(f"Ensemble submission written to {output_path}")
