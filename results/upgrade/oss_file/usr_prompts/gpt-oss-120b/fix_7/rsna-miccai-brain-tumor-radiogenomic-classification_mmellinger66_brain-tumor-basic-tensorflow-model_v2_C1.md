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
Predict the genetic subtype of glioblastoma using MRI (magnetic resonance imaging) scans to detect for the presence of MGMT promoter methylation.

## Metric
Area under the ROC curve between the predicted probability and the observed target.

## Submission Format
For each `BraTS21ID` in the test set, you must predict a probability for the target `MGMT_value`. The file should contain a header and have the following format:

```
BraTS21ID,MGMT_value
00001,0.5
00013,0.5
00015,0.5
etc.
```

## Dataset
- **train/** - folder containing the training files, with each top-level folder representing a subject. **NOTE:** There are some unexpected issues with the following three cases in the training dataset, participants can exclude the cases during training: `[00109, 00123, 00709]`. We have checked and confirmed that the testing dataset is free from such issues.
- **train_labels.csv** - file containing the target `MGMT_value` for each subject in the training data (e.g. the presence of MGMT promoter methylation)
- **test/** - the test files, which use the same structure as `train/`; your task is to predict the `MGMT_value` for each subject in the test data. **NOTE**: the total size of the rerun test set (Public and Private) is ~5x the size of the Public test set
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.10

# 3. Installed packages

geopandas==0.14.4
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
pydicom==3.0.1
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
tqdm==4.67.1

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (202 lines)
            sample_submission.csv (60 lines)
            sample_submission.csv.zip (382 Bytes)
            test.zip (1.3 GB)
            train.zip (10.2 GB)
            train_labels.csv (527 lines)
            train_labels.csv.zip (1.4 kB)
            rsna-miccai-brain-tumor-radiogenomic-classification/
                description.md (202 lines)
                sample_submission.csv (60 lines)
                ... and 5 other files
                rsna-miccai-brain-tumor-radiogenomic-classification/
                test/
                    00002/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00019/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 58 other folders
                train/
                    00000/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00003/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 525 other folders
            test/
                00002/
                    FLAIR/
                        Image-387.dcm (525.4 kB)
                        Image-388.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 29 other files
                    T1wCE/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 382 other files
                00019/
                    FLAIR/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 30 other files
                    T1wCE/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T2w/
                        Image-258.dcm (525.4 kB)
                        Image-259.dcm (525.4 kB)
                        ... and 127 other files
                ... and 58 other folders
            train/
                00000/
                    FLAIR/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 398 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 31 other files
                    T1wCE/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 406 other files
                00003/
                    FLAIR/
                        Image-387.dcm (525.4 kB)
                        Image-388.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 31 other files
                    T1wCE/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 406 other files
                ... and 525 other folders
        input/
            description.md (202 lines)
            sample_submission.csv (60 lines)
            sample_submission.csv.zip (382 Bytes)
            test.zip (1.3 GB)
            train.zip (10.2 GB)
            train_labels.csv (527 lines)
            train_labels.csv.zip (1.4 kB)
            rsna-miccai-brain-tumor-radiogenomic-classification/
                description.md (202 lines)
                sample_submission.csv (60 lines)
                ... and 5 other files
                rsna-miccai-brain-tumor-radiogenomic-classification/
                test/
                    00002/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00019/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 58 other folders
                train/
                    00000/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00003/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 525 other folders
            test/
                00002/
                    FLAIR/
                        Image-387.dcm (525.4 kB)
                        Image-388.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 29 other files
                    T1wCE/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 382 other files
                00019/
                    FLAIR/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 30 other files
                    T1wCE/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T2w/
                        Image-258.dcm (525.4 kB)
                        Image-259.dcm (525.4 kB)
                        ... and 127 other files
                ... and 58 other folders
            train/
                00000/
                    FLAIR/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 398 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 31 other files
                    T1wCE/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 406 other files
                00003/
                    FLAIR/
                        Image-387.dcm (525.4 kB)
                        Image-388.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 31 other files
                    T1wCE/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 406 other files
                ... and 525 other folders
        working/
            rsna-miccai-brain-tumor-radiogenomic-classification/
                description.md (202 lines)
                sample_submission.csv (60 lines)
                ... and 5 other files
                rsna-miccai-brain-tumor-radiogenomic-classification/
                test/
                    00002/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00019/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 58 other folders
                train/
                    00000/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00003/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 525 other folders
```

-> data/rsna-miccai-brain-tumor-radiogenomic-classification/sample_submission.csv has 59 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> data/rsna-miccai-brain-tumor-radiogenomic-classification/train_labels.csv has 526 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> data/sample_submission.csv has 59 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> data/train_labels.csv has 526 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> input/rsna-miccai-brain-tumor-radiogenomic-classification/sample_submission.csv has 59 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> input/rsna-miccai-brain-tumor-radiogenomic-classification/train_labels.csv has 526 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> (stopped after 10 files for performance)

# 5. Target score

-1.0

# 6. Current score

0.46235

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.56) has done: 'Implemented fixes to resolve import errors, updated TensorFlow layer usage, added fallback to scikit‑learn if TensorFlow cannot be imported, corrected model checkpoint handling, and ensured the final CSV matches the required submission format.'
- What this solution (achieved 0.56) has done: 'Implemented fixes to avoid TensorFlow import crashes by safely handling missing TF dependencies and wrapping all TF‑specific model code in guards. Added fallback placeholders for `tf` and `keras` when TF cannot be imported, and moved the `get_model02` definition inside a conditional block so it’s only defined when TensorFlow is available. This ensures the script runs using the scikit‑learn fallback without raising NameErrors, and still produces a valid `submission.csv` in the required format.'
- What this solution (achieved 0.53294) has done: 'The fix adds a small adjustment to use more training subjects (improving the AUC a bit) while keeping the same model logic and ensuring the script always creates a valid `submission.csv`. The rest of the code remains unchanged.'
- What this solution (achieved 0.46) has done: 'The fix removes the artificial limits on how many subjects are loaded for training and testing, allowing the model to use the full dataset (which can modestly improve the validation AUC) while keeping all existing logic intact. No other changes are made, so the script runs end‑to‑end and still writes a correctly formatted `submission.csv`.'
- What this solution (achieved 0.46235) has done: 'The fix disables the TensorFlow import that crashes due to protobuf incompatibility, safely defaults to the scikit‑learn fallback, and retains all other logic unchanged so the script runs end‑to‑end and writes a correct `submission.csv`. This resolves the runtime error while keeping the current validation AUC unchanged (still above the target).'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

TF_AVAILABLE = False
tf = None
keras = None
to_categorical = None

import glob
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pydicom
from tqdm.notebook import tqdm

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split



## === cell 1
data_dir = Path("../input/rsna-miccai-brain-tumor-radiogenomic-classification/")

mri_types = ["FLAIR", "T1w", "T2w", "T1wCE"]
excluded_images = [str(x).zfill(5) for x in [109, 123, 709]]  # Bad images



## === cell 2
train_df = pd.read_csv(data_dir / "train_labels.csv")
test_df = pd.read_csv(data_dir / "sample_submission.csv")
sample_submission = pd.read_csv(data_dir / "sample_submission.csv")

train_df = train_df[~train_df.BraTS21ID.isin(excluded_images)].reset_index(drop=True)

print(f"train data: Rows={train_df.shape[0]}, Columns={train_df.shape[1]}")




## === cell 3
def load_dicom(path, size=224):
    """
    Reads a DICOM image, normalizes pixel values to [0,1] and rescales to 0‑255,
    then resizes to the requested square size.
    """
    dicom = pydicom.dcmread(path)
    data = dicom.pixel_array.astype(np.float32)
    if np.max(data) != 0:
        data = data / np.max(data)
    data = (data * 255).astype(np.uint8)
    return cv2.resize(data, (size, size))




## === cell 4
def get_all_image_paths(brats21id, image_type, folder="train"):
    """
    Returns an array of selected image file paths for a given patient and modality.
    """
    assert image_type in mri_types
    patient_path = os.path.join(
        "../input/rsna-miccai-brain-tumor-radiogenomic-classification",
        folder,
        str(brats21id).zfill(5),
    )
    if not os.path.isdir(patient_path):
        return np.array([])  # gracefully handle missing folders

    paths = sorted(
        glob.glob(os.path.join(patient_path, image_type, "*")),
        key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split("-")[-1]),
    )
    num_images = len(paths)
    start = int(num_images * 0.25)
    end = int(num_images * 0.75)
    interval = 3 if num_images >= 10 else 1
    return np.array(paths[start:end:interval])


def get_all_images(brats21id, image_type, folder="train", size=225):
    return [
        load_dicom(p, size) for p in get_all_image_paths(brats21id, image_type, folder)
    ]




## === cell 5
def get_all_data_for_train(image_type, image_size=32, max_subjects=None):
    """
    Loads images for training. To keep runtime reasonable we optionally limit the
    number of subjects processed (max_subjects). Returns stacked arrays and label list.
    """
    X, y, ids = [], [], []
    subjects = train_df.index.tolist()
    if max_subjects is not None:
        subjects = subjects[:max_subjects]

    for i in tqdm(subjects, desc="Loading train images"):
        row = train_df.loc[i]
        imgs = get_all_images(int(row["BraTS21ID"]), image_type, "train", image_size)
        if not imgs:
            continue
        X.extend(imgs)
        y.extend([row["MGMT_value"]] * len(imgs))
        ids.extend([int(row["BraTS21ID"])] * len(imgs))

    return np.array(X), np.array(y), np.array(ids)


def get_all_data_for_test(image_type, image_size=32, max_subjects=None):
    X, ids = [], []
    subjects = test_df.index.tolist()
    if max_subjects is not None:
        subjects = subjects[:max_subjects]

    for i in tqdm(subjects, desc="Loading test images"):
        row = test_df.loc[i]
        imgs = get_all_images(int(row["BraTS21ID"]), image_type, "test", image_size)
        if not imgs:
            continue
        X.extend(imgs)
        ids.extend([int(row["BraTS21ID"])] * len(imgs))

    return np.array(X), np.array(ids)




## === cell 6
X, y, trainidt = get_all_data_for_train("T1wCE", image_size=32, max_subjects=None)
X_test, testidt = get_all_data_for_test("T1wCE", image_size=32, max_subjects=None)

if X.size == 0:
    X = np.random.randint(0, 256, size=(10, 32, 32), dtype=np.uint8)
    y = np.random.randint(0, 2, size=(10,))
    trainidt = np.arange(10)

if X_test.size == 0:
    X_test = np.random.randint(0, 256, size=(5, 32, 32), dtype=np.uint8)
    testidt = np.arange(1000, 1000 + len(X_test))

print("Shapes after loading:", X.shape, y.shape, X_test.shape)



## === cell 7
X_train, X_valid, y_train, y_valid, trainidt_train, trainidt_valid = train_test_split(
    X, y, trainidt, test_size=0.2, random_state=42, stratify=y
)

if TF_AVAILABLE:
    X_train = tf.expand_dims(X_train, axis=-1)
    X_valid = tf.expand_dims(X_valid, axis=-1)
    X_test = tf.expand_dims(X_test, axis=-1)
    print("After channel expand (TF):", X_train.shape, X_valid.shape, X_test.shape)

    y_train_cat = to_categorical(y_train, num_classes=2)
    y_valid_cat = to_categorical(y_valid, num_classes=2)



## === cell 8
if TF_AVAILABLE:

    def get_model02(input_shape):
        np.random.seed(0)
        random.seed(12)
        tf.random.set_seed(12)

        inpt = keras.Input(shape=input_shape)  # (height, width, 1)

        h = keras.layers.Rescaling(1.0 / 255)(inpt)

        h = keras.layers.Conv2D(
            64, kernel_size=(4, 4), activation="relu", name="Conv_1"
        )(h)
        h = keras.layers.MaxPool2D(pool_size=(2, 2))(h)

        h = keras.layers.Conv2D(
            32, kernel_size=(2, 2), activation="relu", name="Conv_2"
        )(h)
        h = keras.layers.MaxPool2D(pool_size=(1, 1))(h)

        h = keras.layers.Dropout(0.1)(h)

        h = keras.layers.Flatten()(h)
        h = keras.layers.Dense(32, activation="relu")(h)

        output = keras.layers.Dense(2, activation="softmax")(h)

        model = keras.Model(inpt, output)

        model.compile(
            loss="categorical_crossentropy",
            optimizer="adam",
            metrics=[tf.keras.metrics.AUC(name="auc")],
        )
        return model

else:
    get_model02 = None



## === cell 9
if TF_AVAILABLE:
    checkpoint_filepath = "best_model.h5"
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=False,
        monitor="val_auc",
        mode="max",
        save_best_only=True,
        verbose=1,
    )
else:
    checkpoint_filepath = None
    model_checkpoint_callback = None



## === cell 10
if TF_AVAILABLE:
    model = get_model02(X_train.shape[1:])  # input shape without batch dim
    history = model.fit(
        x=X_train,
        y=y_train_cat,
        epochs=5,  # modest epochs for quick run
        validation_data=(X_valid, y_valid_cat),
        callbacks=[model_checkpoint_callback],
        verbose=2,
    )
    model_best = tf.keras.models.load_model(checkpoint_filepath)
else:
    from sklearn.linear_model import LogisticRegression

    X_train_flat = X_train.reshape((X_train.shape[0], -1))
    X_valid_flat = X_valid.reshape((X_valid.shape[0], -1))
    clf = LogisticRegression(max_iter=200, class_weight="balanced", n_jobs=-1)
    clf.fit(X_train_flat, y_train)
    model_best = clf
    X_valid = X_valid_flat  # keep flattened for evaluation later



## === cell 11
if TF_AVAILABLE:
    y_pred_valid = model_best.predict(X_valid)
    prob_valid = y_pred_valid[:, 1]
else:
    prob_valid = model_best.predict_proba(X_valid)[:, 1]

val_auc = roc_auc_score(y_valid, prob_valid)
print(f"Validation AUC = {val_auc:.4f}")



## === cell 12
if TF_AVAILABLE:
    X_test_input = X_test  # already has channel dim
    y_pred_test = model_best.predict(X_test_input)
    prob_test = y_pred_test[:, 1]
else:
    X_test_flat = X_test.reshape((X_test.shape[0], -1))
    prob_test = model_best.predict_proba(X_test_flat)[:, 1]

result = pd.DataFrame({"BraTS21ID": testidt, "MGMT_value": prob_test})
submission = result.groupby("BraTS21ID", as_index=False).mean()

submission = sample_submission[["BraTS21ID"]].merge(
    submission, on="BraTS21ID", how="left"
)
submission["MGMT_value"] = submission["MGMT_value"].fillna(
    0.5
)  # fallback for missing IDs

submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")
