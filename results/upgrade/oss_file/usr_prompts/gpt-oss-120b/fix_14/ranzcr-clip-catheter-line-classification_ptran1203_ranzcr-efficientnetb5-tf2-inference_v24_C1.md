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
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import concurrent.futures

try:
    import tensorflow as tf

    tf_available = True
except Exception as e:
    tf = None
    tf_available = False
    print("TensorFlow import failed (forced fallback):", e)

tf_available = False

W = H = 338
N_CLASSES = 11

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

test_image_dir = "../input/ranzcr-clip-catheter-line-classification/test"
train_image_dir = "../input/ranzcr-clip-catheter-line-classification/train"
sample_submission_path = (
    "../input/ranzcr-clip-catheter-line-classification/sample_submission.csv"
)

sample_df = pd.read_csv(sample_submission_path, nrows=0)
target_cols = list(sample_df.columns)[1:]


def get_filenames_and_ids(image_dir):
    """Return sorted filenames and corresponding StudyInstanceUIDs."""
    files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(".jpg")])
    paths = [os.path.join(image_dir, f) for f in files]
    ids = [os.path.splitext(f)[0] for f in files]  # UID is filename without extension
    return paths, ids


_HIST_BINS = 32
_FLATTEN_SIZE = 64 * 64 * 3  # resized image flatten dimension
_FEATURE_DIM = _HIST_BINS * 3 + 3 + 3 + _FLATTEN_SIZE  # hist + mean + std + flatten


def extract_features(path):
    """Compute rich features: resized flatten, histograms, mean and std per channel."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return np.zeros(_FEATURE_DIM, dtype=np.float32)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    small = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
    flat = (small.astype(np.float32) / 255.0).reshape(-1)  # (12288,)

    channel_means = img.mean(axis=(0, 1))  # (3,)
    channel_stds = img.std(axis=(0, 1))  # (3,)

    hist_features = []
    for ch in range(3):
        hist, _ = np.histogram(
            img[:, :, ch], bins=_HIST_BINS, range=(0, 256), density=True
        )
        hist_features.append(hist.astype(np.float32))
    hist_features = np.concatenate(hist_features)  # (96,)

    return np.concatenate(
        [
            flat,
            hist_features,
            channel_means.astype(np.float32),
            channel_stds.astype(np.float32),
        ]
    ).astype(np.float32)


def compute_features_parallel(paths):
    """Compute feature vectors for a list of image paths using a process pool."""
    n = len(paths)
    feats = np.empty((n, _FEATURE_DIM), dtype=np.float32)
    with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        for i, f in enumerate(executor.map(extract_features, paths, chunksize=32)):
            feats[i] = f
    return feats




## === cell 1
train_df = pd.read_csv("../input/ranzcr-clip-catheter-line-classification/train.csv")
train_paths, train_ids = get_filenames_and_ids(train_image_dir)

train_features = compute_features_parallel(train_paths)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(train_features)

X_tr, X_val, y_tr_idx, y_val_idx = train_test_split(
    np.arange(X_scaled.shape[0]),
    np.arange(X_scaled.shape[0]),
    test_size=0.2,
    random_state=42,
)

chosen_models = {}


def _train_one_label(col):
    """Train RF and LR for a single target column and return the better final model."""
    y = train_df[col].values
    y_train = y[X_tr]
    y_val = y[X_val]

    rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        n_jobs=1,
        class_weight="balanced",
        random_state=42,
    )
    rf.fit(X_scaled[X_tr], y_train)
    rf_val_pred = rf.predict_proba(X_scaled[X_val])[:, 1]
    rf_auc = roc_auc_score(y_val, rf_val_pred)

    lr = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="saga",
        max_iter=200,
        n_jobs=1,
        class_weight="balanced",
        random_state=42,
    )
    lr.fit(X_scaled[X_tr], y_train)
    lr_val_pred = lr.predict_proba(X_scaled[X_val])[:, 1]
    lr_auc = roc_auc_score(y_val, lr_val_pred)

    if lr_auc > rf_auc:
        final_lr = LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="saga",
            max_iter=300,
            n_jobs=1,
            class_weight="balanced",
            random_state=42,
        )
        final_lr.fit(X_scaled, y)
        return col, final_lr
    else:
        final_rf = RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            n_jobs=1,
            class_weight="balanced",
            random_state=42,
        )
        final_rf.fit(X_scaled, y)
        return col, final_rf


with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
    futures = {executor.submit(_train_one_label, col): col for col in target_cols}
    for future in concurrent.futures.as_completed(futures):
        col, model = future.result()
        chosen_models[col] = model




## === cell 2
test_paths, test_ids = get_filenames_and_ids(test_image_dir)

if tf_available:
    test_ds = tf.data.Dataset.from_tensor_slices(test_paths)
    test_ds = test_ds.map(
        lambda p: tf.image.resize(
            tf.cast(tf.image.decode_jpeg(tf.io.read_file(p), channels=3), tf.float32)
            / 255.0,
            (H, W),
        ),
        num_parallel_calls=1,
    )
    test_ds = test_ds.batch(16).prefetch(1)
else:
    test_features = compute_features_parallel(test_paths)
    test_X = scaler.transform(test_features)




## === cell 3
if tf_available:
    preds_list = []
    for batch in test_ds:
        batch_preds = model.predict_on_batch(batch)
        preds_list.append(batch_preds[:, : len(target_cols)])
    preds = np.concatenate(preds_list, axis=0)
else:
    preds = np.column_stack(
        [chosen_models[col].predict_proba(test_X)[:, 1] for col in target_cols]
    )

submission = pd.DataFrame(preds, columns=target_cols)
submission.insert(0, "StudyInstanceUID", test_ids)

submission.to_csv("submission.csv", index=False)
print("Submission file saved as submission.csv")
