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
Mean F1-Score

## Submission Format
labels should be a space-delimited list.

The file should contain a header and have the following format:

```
image, labels
85f8cb619c66b863.jpg,healthy
ad8770db05586b59.jpg,healthy
c7b03e718489f3ca.jpg,healthy
```

## Dataset
**train.csv** - the training set metadata.

- `image` - the image ID.
- `labels` - the target classes, a space delimited list of all diseases found in the image. Unhealthy leaves with too many diseases to classify visually will have the `complex` class, and may also have a subset of the diseases identified.

**sample_submission.csv** - A sample submission file in the correct format.

- `image`
- `labels`

**train_images** - The training set images.

**test_images** - The test set images. This competition has a hidden test set: only three images are provided here as samples while the remaining 5,000 images will be available to your notebook once it is submitted.

# 2. Python version

3.9

# 3. Installed packages

geopandas==0.14.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
        input/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
        working/
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
```

-> data/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> data/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> (stopped after 10 files for performance)

# 5. Target score

0.7322807017543864

# 6. Current score

0.4397

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.28656) has done: 'The fix adds a safe import of TensorFlow (falling back if it fails) and replaces the missing model loading with a simple baseline that predicts the most frequent label from the training data for every test image. It also corrects the data paths to use the Kaggle `/kaggle/input` directory and ensures the submission CSV is written with the required columns and `.csv` extension. This resolves the protobuf import error, the missing SavedModel error, and guarantees a valid submission file is produced.'
- What this solution (achieved 0.24249) has done: 'The changes keep the exact prediction logic but replace the per‑pixel Python loop with a NumPy‑based mean calculation, which is orders of magnitude faster. The image processing is performed in parallel using a thread pool (I/O‑bound work), preserving the original ordering of filenames so the submission remains identical. Only imports and the main loop are altered; all model‑free baseline logic stays unchanged.'
- What this solution (achieved 0.2769) has done: 'The changes speed up image feature extraction by using Pillow’s fast `ImageStat` (avoiding costly NumPy conversions) and eliminate the O(N²) lookup in the prediction loop by iterating with indices. These tweaks keep the exact same mean‑RGB values and model logic, so predictions remain identical while dramatically reducing runtime.'
- What this solution (achieved 0.38622) has done: 'I keep the overall workflow (mean‑RGB feature extraction, logistic regression with One‑Vs‑Rest, and the fallback rule) but make two lightweight tweaks that are known to boost F1 for imbalanced multi‑label data: (1) use `class_weight='balanced'` in the LogisticRegression so rare disease classes influence the model more, and (2) lower the probability threshold from 0.5 to 0.35, which usually improves recall without overly harming precision. These changes stay within the original logic and should raise the mean F1 from ≈0.28 toward the target ≈0.73 while still producing a valid `submission.csv`.'
- What this solution (achieved 0.38906) has done: 'I add a simple but potentially more discriminative feature set (mean + standard‑deviation of each RGB channel) and lower the prediction probability threshold slightly. These changes keep the original linear‑model pipeline while giving the classifier extra information and improving recall, which should raise the mean F1 toward the target score.'
- What this solution (achieved 0.4094) has done: 'I add feature scaling with StandardScaler to normalise the mean‑and‑std RGB features before training the logistic regression, and I raise the probability threshold slightly to 0.35 to improve the precision‑recall balance. These small, targeted tweaks keep the original pipeline intact while giving the model a better chance to raise the mean F1 toward the target.'
- What this solution (achieved 0.42345) has done: 'I add a quick validation split to tune the probability threshold rather than using the fixed 0.35. After training the One‑Vs‑Rest logistic regression on a training subset, I evaluate a few thresholds on a held‑out validation set and pick the one that yields the highest macro F1. The chosen threshold is then used for the final test predictions. This small adjustment keeps the original feature extraction, scaling, and model unchanged while likely moving the mean F1 closer to the target.'
- What this solution (achieved 0.4397) has done: 'The changes add a lightweight but informative feature (the green‑to‑total‑mean ratio) to the existing mean‑and‑std RGB vector, expand the threshold search to a finer grid, and keep the same overall modeling pipeline; this modest enrichment is expected to raise the validation macro F1 and therefore move the Kaggle score closer to the target while preserving the core logic.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
from collections import Counter
from PIL import Image, ImageStat
import numpy as np
import concurrent.futures

from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score



## === cell 1
BASE_INPUT = "/kaggle/input/plant-pathology-2021-fgvc8"
TRAIN_CSV = os.path.join(BASE_INPUT, "train.csv")
TEST_DIR = os.path.join(BASE_INPUT, "test_images")

train_df = pd.read_csv(TRAIN_CSV)

all_labels = train_df["labels"].str.split().explode()
most_common_label = Counter(all_labels).most_common(1)[0][0]

non_healthy_labels = [lbl for lbl in all_labels if lbl != "healthy"]
if non_healthy_labels:
    most_common_nonhealthy = Counter(non_healthy_labels).most_common(1)[0][0]
else:
    most_common_nonhealthy = most_common_label

print(
    f"Baseline fallback: 'healthy' when green is high, otherwise '{most_common_nonhealthy}'"
)


def mean_std_rgb_with_green_ratio(path):
    """
    Return concatenated mean, std for R,G,B channels (6 values) plus
    green‑to‑total‑mean ratio (1 value) → length 7.
    """
    try:
        with Image.open(path) as img:
            img = img.convert("RGB")
            stat = ImageStat.Stat(img)
            mean = stat.mean  # list of 3 floats
            std = stat.stddev  # list of 3 floats
            total_mean = sum(mean) + 1e-6
            green_ratio = mean[1] / total_mean
            return np.array(mean + std + [green_ratio], dtype=np.float32)
    except Exception as e:
        print(f"Warning: could not read {path}: {e}")
        return np.zeros(7, dtype=np.float32)


train_images = train_df["image"].tolist()
train_paths = [os.path.join(BASE_INPUT, "train_images", name) for name in train_images]

with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
    train_features = list(executor.map(mean_std_rgb_with_green_ratio, train_paths))

X_train_raw = np.vstack(train_features)  # (n_samples, 7)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train_raw)

mlb = MultiLabelBinarizer()
y = mlb.fit_transform(train_df["labels"].str.split())

X_tr, X_val, y_tr, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

clf = OneVsRestClassifier(
    LogisticRegression(
        solver="liblinear",
        max_iter=300,
        random_state=42,
        class_weight="balanced",
    )
)
clf.fit(X_tr, y_tr)

probs_val = clf.predict_proba(X_val)  # (n_val, n_classes)

candidate_thresholds = np.arange(0.20, 0.51, 0.01)
best_thr = 0.35
best_f1 = 0.0
for thr in candidate_thresholds:
    y_pred_bin = (probs_val >= thr).astype(int)
    f1 = f1_score(y_val, y_pred_bin, average="macro")
    if f1 > best_f1:
        best_f1 = f1
        best_thr = thr

THRESHOLD = best_thr
print(
    f"Selected probability threshold: {THRESHOLD:.2f} (validation macro F1={best_f1:.4f})"
)


def predict_labels(paths, filenames):
    """Predict space‑separated label strings for given image paths."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        feats_raw = list(executor.map(mean_std_rgb_with_green_ratio, paths))
    X_test_raw = np.vstack(feats_raw)

    X_test = scaler.transform(X_test_raw)

    probs = clf.predict_proba(X_test)  # (n_samples, n_classes)

    preds = []
    for i, (prob_vec, fname) in enumerate(zip(probs, filenames)):
        label_idxs = np.where(prob_vec >= THRESHOLD)[0]
        if len(label_idxs) == 0:
            mean_green = X_test_raw[i][1] / 255.0
            pred = "healthy" if mean_green > 0.5 else most_common_nonhealthy
            preds.append(pred)
        else:
            pred_labels = mlb.classes_[label_idxs]
            preds.append(" ".join(pred_labels))
    return preds




## === cell 2
if __name__ == "__main__":
    test_images = sorted(
        [
            f
            for f in os.listdir(TEST_DIR)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
    )
    test_paths = [os.path.join(TEST_DIR, name) for name in test_images]

    predicted_labels = predict_labels(test_paths, test_images)

    submission_df = pd.DataFrame({"image": test_images, "labels": predicted_labels})
    output_path = os.path.join(".", "submission.csv")
    submission_df.to_csv(output_path, index=False)
    print(f"Submission written to {output_path}")
