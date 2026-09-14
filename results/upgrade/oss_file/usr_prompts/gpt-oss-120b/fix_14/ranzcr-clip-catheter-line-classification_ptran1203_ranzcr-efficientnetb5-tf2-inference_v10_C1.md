# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Detect the presence and position of catheters and lines on chest x-rays.

## Metric
Area under the ROC curve for each label, with the final score being the average of the individual AUCs of each predicted column.

## Submission Format
For each ID in the test set, you must predict a probability for all target variables. The file should contain a header and have the following format:
```
StudyInstanceUID,ETT - Abnormal,ETT - Borderline,ETT - Normal,NGT - Abnormal,NGT - Borderline,NGT - Incompletely Imaged,NGT - Normal,CVC - Abnormal,CVC - Borderline,CVC - Normal,Swan Ganz Catheter Present
1.2.826.0.1.3680043.8.498.62451881164053375557257228990443168843,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.83721761279899623084220697845011427274,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.12732270010839808189235995393981377825,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.11769539755086084996287023095028033598,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.87838627504097587943394933987052577153,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.53211840524738036417560823327351887819,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.93555795394184819372299157360228027866,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.52241894131170494723503100795076463919,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.36500167484503936720548852591033878284,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.86199852603457900780565655267977637728,0,0,0,0,0,0,0,0,0,0,0
```

## Dataset
`train.csv` contains image IDs, binary labels, and patient IDs.

TFRecords are available for both train and test.

We've also included `train_annotations.csv`. These are segmentation annotations for training samples that have them. They are included solely as additional information for competitors.

- train.csv - contains image IDs, binary labels, and patient IDs.
- sample_submission.csv - a sample submission file in the correct format
- test - test images
- train - training images

### Columns
- `StudyInstanceUID` - unique ID for each image
- `ETT - Abnormal` - endotracheal tube placement abnormal
- `ETT - Borderline` - endotracheal tube placement borderline abnormal
- `ETT - Normal` - endotracheal tube placement normal
- `NGT - Abnormal` - nasogastric tube placement abnormal
- `NGT - Borderline` - nasogastric tube placement borderline abnormal
- `NGT - Incompletely Imaged` - nasogastric tube placement inconclusive due to imaging
- `NGT - Normal` - nasogastric tube placement borderline normal
- `CVC - Abnormal` - central venous catheter placement abnormal
- `CVC - Borderline` - central venous catheter placement borderline abnormal
- `CVC - Normal` - central venous catheter placement normal
- `Swan Ganz Catheter Present`
- `PatientID` - unique ID for each patient in the dataset

# 2. Python version

3.9

# 3. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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
            description.md (172 lines)
            sample_submission.csv (3010 lines)
            sample_submission.csv.zip (64.2 kB)
            test.zip (642.8 MB)
            train.csv (27075 lines)
            train.csv.zip (798.6 kB)
            train.zip (5.8 GB)
            train_annotations.csv (16262 lines)
            train_annotations.csv.zip (1.4 MB)
            ranzcr-clip-catheter-line-classification/
                description.md (172 lines)
                sample_submission.csv (3010 lines)
                ... and 7 other files
                ranzcr-clip-catheter-line-classification/
                test/
                    1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                    1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                    ... and 3007 other files
                    test/
                train/
                    1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                    1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                    ... and 27072 other files
                    train/
            test/
                1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                ... and 3007 other files
                test/
            train/
                1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                ... and 27072 other files
                train/
        input/
            description.md (172 lines)
            sample_submission.csv (3010 lines)
            sample_submission.csv.zip (64.2 kB)
            test.zip (642.8 MB)
            train.csv (27075 lines)
            train.csv.zip (798.6 kB)
            train.zip (5.8 GB)
            train_annotations.csv (16262 lines)
            train_annotations.csv.zip (1.4 MB)
            ranzcr-clip-catheter-line-classification/
                description.md (172 lines)
                sample_submission.csv (3010 lines)
                ... and 7 other files
                ranzcr-clip-catheter-line-classification/
                test/
                    1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                    1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                    ... and 3007 other files
                    test/
                train/
                    1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                    1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                    ... and 27072 other files
                    train/
            test/
                1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                ... and 3007 other files
                test/
                    1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                    1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                    ... and 3007 other files
                    test/
            train/
                1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                ... and 27072 other files
                train/
                    1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                    1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                    ... and 27072 other files
                    train/
        working/
            ranzcr-clip-catheter-line-classification/
                description.md (172 lines)
                sample_submission.csv (3010 lines)
                ... and 7 other files
                ranzcr-clip-catheter-line-classification/
                test/
                    1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                    1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                    ... and 3007 other files
                    test/
                train/
                    1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                    1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                    ... and 27072 other files
                    train/
```

-> data/ranzcr-clip-catheter-line-classification/sample_submission.csv has 3009 rows and 10 columns.
The columns are: StudyInstanceUID, ETT - Abnormal, ETT - Borderline, ETT - Normal, NGT - Abnormal, NGT - Borderline, NGT - Incompletely Imaged, NGT - Normal, CVC - Abnormal, CVC - Borderline

-> data/ranzcr-clip-catheter-line-classification/train.csv has 27074 rows and 13 columns.
The columns are: StudyInstanceUID, ETT - Abnormal, ETT - Borderline, ETT - Normal, NGT - Abnormal, NGT - Borderline, NGT - Incompletely Imaged, NGT - Normal, CVC - Abnormal, CVC - Borderline, CVC - Normal, Swan Ganz Catheter Present, PatientID

-> data/ranzcr-clip-catheter-line-classification/train_annotations.csv has 16261 rows and 3 columns.
The columns are: StudyInstanceUID, label, data

-> data/sample_submission.csv has 3009 rows and 10 columns.
The columns are: StudyInstanceUID, ETT - Abnormal, ETT - Borderline, ETT - Normal, NGT - Abnormal, NGT - Borderline, NGT - Incompletely Imaged, NGT - Normal, CVC - Abnormal, CVC - Borderline

-> data/train.csv has 27074 rows and 13 columns.
The columns are: StudyInstanceUID, ETT - Abnormal, ETT - Borderline, ETT - Normal, NGT - Abnormal, NGT - Borderline, NGT - Incompletely Imaged, NGT - Normal, CVC - Abnormal, CVC - Borderline, CVC - Normal, Swan Ganz Catheter Present, PatientID

-> data/train_annotations.csv has 16261 rows and 3 columns.
The columns are: StudyInstanceUID, label, data

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
import cv2
from concurrent.futures import (
    ProcessPoolExecutor,
)  # switched to processes for true parallelism

try:
    import tensorflow as tf
except Exception as e:
    print(f"TensorFlow import failed ({e}); proceeding without TensorFlow.")
    tf = None




## === cell 1
target_cols = [
    "CVC - Abnormal",
    "CVC - Borderline",
    "CVC - Normal",
    "ETT - Abnormal",
    "ETT - Borderline",
    "ETT - Normal",
    "NGT - Abnormal",
    "NGT - Borderline",
    "NGT - Incompletely Imaged",
    "NGT - Normal",
    "Swan Ganz Catheter Present",
]

sample_submission_path = (
    "../input/ranzcr-clip-catheter-line-classification/sample_submission.csv"
)
train_csv_path = "../input/ranzcr-clip-catheter-line-classification/train.csv"
train_images_dir = "../input/ranzcr-clip-catheter-line-classification/train"
test_images_dir = "../input/ranzcr-clip-catheter-line-classification/test"

submission_df = pd.read_csv(sample_submission_path)
for col in target_cols:
    if col not in submission_df.columns:
        submission_df[col] = 0.0
submission_df = submission_df[["StudyInstanceUID"] + target_cols]




## === cell 2
def extract_features(image_path):
    """Extract a richer set of fast image features using OpenCV."""
    img = cv2.imread(image_path)
    if img is None:
        return np.zeros(43, dtype=np.float32)

    means, stddevs = cv2.meanStdDev(img)
    mean_b, mean_g, mean_r = means[:, 0]
    std_b, std_g, std_r = stddevs[:, 0]

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_means, h_stddevs = cv2.meanStdDev(hsv)
    mean_h, mean_s, mean_v = h_means[:, 0]
    std_h, std_s, std_v = h_stddevs[:, 0]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_mean_arr, gray_std_arr = cv2.meanStdDev(gray)
    gray_mean = float(gray_mean_arr[0][0])
    gray_std = float(gray_std_arr[0][0])

    h, w = img.shape[:2]
    aspect_ratio = w / h if h != 0 else 0.0

    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.count_nonzero(edges) / (h * w) if h * w != 0 else 0.0

    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    hist_features = []
    for channel in cv2.split(img):
        hist = cv2.calcHist([channel], [0], None, [8], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        hist_features.extend(hist.tolist())

    feature_vec = np.array(
        [
            float(mean_r),
            float(mean_g),
            float(mean_b),
            float(std_r),
            float(std_g),
            float(std_b),
            float(mean_h),
            float(mean_s),
            float(mean_v),
            float(std_h),
            float(std_s),
            float(std_v),
            gray_mean,
            gray_std,
            float(w),
            float(h),
            aspect_ratio,
            edge_density,
            lap_var,
            *hist_features,
        ],
        dtype=np.float32,
    )
    return feature_vec


train_df = pd.read_csv(train_csv_path)
train_df[target_cols] = train_df[target_cols].fillna(0)

train_uids = train_df["StudyInstanceUID"].values
train_label_matrix = train_df[target_cols].values.T  # (num_labels, num_samples)


def _extract_for_uid(uid):
    img_path = os.path.join(train_images_dir, f"{uid}.jpg")
    return extract_features(img_path)


max_workers = os.cpu_count() or 1

num_train = len(train_uids)
train_features = np.empty((num_train, 43), dtype=np.float32)

with ProcessPoolExecutor(max_workers=max_workers) as executor:
    for idx, feat in enumerate(executor.map(_extract_for_uid, train_uids, chunksize=8)):
        train_features[idx] = feat

missing_train = np.sum(np.all(train_features == 0, axis=1))
if missing_train > 0:
    print(
        f"Warning: {missing_train} training images could not be read and were filled with zeros."
    )

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
train_features_scaled = scaler.fit_transform(train_features)

from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

models = {}
for idx, col in enumerate(target_cols):
    y = train_label_matrix[idx]
    clf = LogisticRegression(
        solver="lbfgs",
        max_iter=500,
        n_jobs=1,
        class_weight="balanced",
    )
    clf.fit(train_features_scaled, y)
    calib = CalibratedClassifierCV(clf, cv="prefit", method="sigmoid")
    calib.fit(train_features_scaled, y)
    models[col] = calib




## === cell 3
test_uids = submission_df["StudyInstanceUID"].values


def _extract_test(uid):
    img_path = os.path.join(test_images_dir, f"{uid}.jpg")
    return extract_features(img_path)


num_test = len(test_uids)
test_features = np.empty((num_test, 43), dtype=np.float32)

with ProcessPoolExecutor(max_workers=max_workers) as executor:
    for idx, feat in enumerate(executor.map(_extract_test, test_uids, chunksize=8)):
        test_features[idx] = feat

missing_test = np.sum(np.all(test_features == 0, axis=1))
if missing_test > 0:
    print(
        f"Warning: {missing_test} test images could not be read and were filled with zeros."
    )

X_test_scaled = scaler.transform(test_features)

for col in target_cols:
    prob = models[col].predict_proba(X_test_scaled)[:, 1]
    submission_df[col] = prob




## === cell 4
output_path = "submission.csv"
submission_df.to_csv(output_path, index=False)
print(f"Submission file written to {output_path}")
