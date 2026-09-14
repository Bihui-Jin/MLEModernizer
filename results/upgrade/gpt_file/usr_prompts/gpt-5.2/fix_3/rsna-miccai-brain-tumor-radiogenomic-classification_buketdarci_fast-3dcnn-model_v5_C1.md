# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

3.9

# 3. Installed packages

geopandas==0.14.4
google-api-python-client==2.177.0
ipython==7.34.0
ipython-genutils==0.2.0
ipython_pygments_lexers==1.1.1
ipython-sql==0.5.0
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
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
tf_keras==2.18.0

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

0.58439

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
from glob import glob
import numpy as np
import pandas as pd
import re
import cv2
import pydicom
import matplotlib.pyplot as plt

np.random.seed(1)

DATA_DIR = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
LABELS_CSV = os.path.join(DATA_DIR, "train_labels.csv")
SAMPLE_SUB_CSV = os.path.join(DATA_DIR, "sample_submission.csv")

print("Train dir exists:", os.path.isdir(TRAIN_DIR))
print("Test dir exists:", os.path.isdir(TEST_DIR))
print("Labels exists:", os.path.isfile(LABELS_CSV))
print("Sample submission exists:", os.path.isfile(SAMPLE_SUB_CSV))



## === cell 1
dicom_files = glob(os.path.join(TRAIN_DIR, "*", "T2w", "*.dcm"))
if len(dicom_files) == 0:
    raise RuntimeError("No DICOM files found under train/*/T2w/*.dcm (unexpected).")

image_path = sorted(dicom_files)[0]
ds = pydicom.dcmread(image_path)
plt.figure(figsize=(4, 4))
plt.imshow(ds.pixel_array, cmap="gray")
plt.title(os.path.basename(image_path))
plt.axis("off")
plt.show()




## === cell 2
def load_dicom(path: str) -> np.ndarray:
    """Read DICOM and return uint8 image scaled to [0,255]."""
    data = pydicom.dcmread(path)
    img = data.pixel_array.astype(np.float32)
    mx = float(np.max(img))
    if mx > 0:
        img = img / mx
    img = (img * 255.0).astype(np.uint8)
    return img


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split(r"(\d+)", str(text))]


def all_slice(sequence, list_name):
    """Sort each subject's list of dicoms and append."""
    for x in range(len(sequence)):
        sequence[x].sort(key=natural_keys)
        list_name.append(sequence[x][0 : len(sequence[x])])
    return list_name




## === cell 3
df = pd.read_csv(LABELS_CSV)
df = df.set_index("BraTS21ID")
df = df.drop([109, 123, 709], axis=0, errors="ignore")
df = df.reset_index()

labels = np.array(df["MGMT_value"]).astype(np.float32)
print("Train labels shape:", labels.shape, "positives:", labels.mean())



## === cell 4
imagePatches = glob(
    os.path.join(TRAIN_DIR, "*", "")
)  # ensures exactly one level under TRAIN_DIR
imagePatches = [p for p in imagePatches if os.path.isdir(p)]
imagePatches = [
    p for p in imagePatches if os.path.basename(os.path.normpath(p)).isdigit()
]
imagePatches.sort(key=natural_keys)

exclude_ids = {"00109", "00123", "00709"}
imagePatches = [
    p for p in imagePatches if os.path.basename(os.path.normpath(p)) not in exclude_ids
]

print("Num train subjects (folders):", len(imagePatches))
print("Num labels after drop:", len(df))

folder_ids = [int(os.path.basename(os.path.normpath(p))) for p in imagePatches]
label_ids = df["BraTS21ID"].astype(int).tolist()
id_to_folder = {i: p for i, p in zip(folder_ids, imagePatches)}

missing_folders = [i for i in label_ids if i not in id_to_folder]
extra_folders = [i for i in folder_ids if i not in set(label_ids)]
print(
    "Missing folders for some labels:",
    missing_folders[:5],
    "count:",
    len(missing_folders),
)
print("Extra folders without labels:", extra_folders[:5], "count:", len(extra_folders))

common_ids = [i for i in label_ids if i in id_to_folder]
df = df[df["BraTS21ID"].astype(int).isin(common_ids)].copy()
df = df.sort_values("BraTS21ID").reset_index(drop=True)
labels = df["MGMT_value"].astype(np.float32).to_numpy()

imagePatches = [id_to_folder[int(i)] for i in df["BraTS21ID"].astype(int).tolist()]
print("Aligned train subjects:", len(imagePatches), "labels:", len(labels))



## === cell 5
flair_patches, t1w_patches, t1wce_patches, t2w_patches = [], [], [], []

for subfolder in imagePatches:
    flair_patches.append(glob(os.path.join(subfolder, "FLAIR", "*.dcm")))
    t1w_patches.append(glob(os.path.join(subfolder, "T1w", "*.dcm")))
    t1wce_patches.append(glob(os.path.join(subfolder, "T1wCE", "*.dcm")))
    t2w_patches.append(glob(os.path.join(subfolder, "T2w", "*.dcm")))

all_flair_patches = all_slice(flair_patches, [])
all_t1w_patches = all_slice(t1w_patches, [])
all_t1wce_patches = all_slice(t1wce_patches, [])
all_t2w_patches = all_slice(t2w_patches, [])

print("Example slices per subject (T2w):", [len(x) for x in all_t2w_patches[:3]])




## === cell 6
def create_input_3d(patches, number_image=None):
    """
    Bugfix: Keras Conv3D expects channels-last NDHWC: (H, W, D, C) for 3D volumes.
    We build shape (N, 256, 256, 3, 1) where D=3 slices, C=1 channel.
    """
    if number_image is None:
        number_image = len(patches)
    inputs = np.zeros((number_image, 256, 256, 3, 1), dtype=np.uint8)

    for i in range(min(number_image, len(patches))):
        seq = patches[i]
        if len(seq) == 0:
            continue

        mid = len(seq) // 2
        idxs = [
            max(0, min(len(seq) - 1, mid - 1)),
            max(0, min(len(seq) - 1, mid)),
            max(0, min(len(seq) - 1, mid + 1)),
        ]

        for d, j in enumerate(idxs):
            img = load_dicom(seq[j])
            image_array = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
            inputs[i, :, :, d, 0] = image_array
    return inputs




## === cell 7
t2w_inputs = create_input_3d(all_t2w_patches)
flair_inputs = create_input_3d(all_flair_patches)
t1wce_inputs = create_input_3d(all_t1wce_patches)
t1w_inputs = create_input_3d(all_t1w_patches)

print("flair_inputs:", flair_inputs.shape, flair_inputs.dtype)



## === cell 8
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

print("TF:", tf.__version__, "tf.keras:", keras.__version__)




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 9
def get_model(optimizer):
    inputs = keras.Input((256, 256, 3, 1))

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu", padding="same")(
        inputs
    )
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=128, kernel_size=3, activation="relu", padding="same")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(units=512, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(units=256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(units=1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs, name="3dcnn")
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model


model = get_model(keras.optimizers.Adam())
model.summary()



## === cell 10
x_train, x_valid, y_train, y_valid = train_test_split(
    flair_inputs, labels, test_size=0.2, random_state=1, stratify=labels
)

x_train = x_train.astype(np.float32) / 255.0
x_valid = x_valid.astype(np.float32) / 255.0

os.makedirs("/kaggle/working/models", exist_ok=True)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss", factor=0.3, patience=3, mode="auto", verbose=1
)
early_stop = EarlyStopping(
    monitor="val_loss", min_delta=0.1, patience=3, mode="min", restore_best_weights=True
)

model = get_model(keras.optimizers.Adam())
history = model.fit(
    x_train,
    y_train,
    epochs=50,
    batch_size=8,
    shuffle=True,
    validation_data=(x_valid, y_valid),
    callbacks=[reduce_lr, early_stop],
    verbose=2,
)



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1811166502.py in <cell line: 0>()
     17 
     18 model = get_model(keras.optimizers.Adam())
---> 19 history = model.fit(
     20     x_train,
     21     y_train,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    122             raise e.with_traceback(filtered_tb) from None
    123         finally:
--> 124             del filtered_tb
    125 
    126     return error_handler

ValueError: Exception encountered when calling MaxPooling3D.call().

Negative dimension size caused by subtracting 2 from 1 for '{{node 3dcnn_1/max_pooling3d_3_1/MaxPool3D}} = MaxPool3D[T=DT_FLOAT, data_format="NDHWC", ksize=[1, 2, 2, 2, 1], padding="VALID", strides=[1, 2, 2, 2, 1]](3dcnn_1/conv3d_3_1/Relu)' with input shapes: [?,128,128,1,128].

Arguments received by MaxPooling3D.call():
  • inputs=tf.Tensor(shape=(None, 128, 128, 1, 128), dtype=float32)

## === cell 11
model.save("/kaggle/working/t1winputs.keras")
model_t1w = keras.models.load_model("/kaggle/working/t1winputs.keras")

model.save("/kaggle/working/t1wceinputs.keras")
model_t1wce = keras.models.load_model("/kaggle/working/t1wceinputs.keras")

model.save("/kaggle/working/t2winputs.keras")
model_t2w = keras.models.load_model("/kaggle/working/t2winputs.keras")

model.save("/kaggle/working/flairinputs.keras")
model_flair = keras.models.load_model("/kaggle/working/flairinputs.keras")



## === cell 12
preds_flair = model_flair.predict(x_valid, verbose=0).reshape(-1)
preds_t1w = model_t1w.predict(x_valid, verbose=0).reshape(-1)
preds_t2w = model_t2w.predict(x_valid, verbose=0).reshape(-1)
preds_t1wce = model_t1wce.predict(x_valid, verbose=0).reshape(-1)

mean_valid = (preds_flair + preds_t1w + preds_t2w + preds_t1wce) / 4.0
auc_score = roc_auc_score(y_valid, mean_valid)
print("Validation AUC:", float(auc_score))



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/760261287.py in <cell line: 0>()
----> 1 preds_flair = model_flair.predict(x_valid, verbose=0).reshape(-1)
      2 preds_t1w = model_t1w.predict(x_valid, verbose=0).reshape(-1)
      3 preds_t2w = model_t2w.predict(x_valid, verbose=0).reshape(-1)
      4 preds_t1wce = model_t1wce.predict(x_valid, verbose=0).reshape(-1)
      5 

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    122             raise e.with_traceback(filtered_tb) from None
    123         finally:
--> 124             del filtered_tb
    125 
    126     return error_handler

ValueError: Exception encountered when calling MaxPooling3D.call().

Negative dimension size caused by subtracting 2 from 1 for '{{node 3dcnn_1/max_pooling3d_3_1/MaxPool3D}} = MaxPool3D[T=DT_FLOAT, data_format="NDHWC", ksize=[1, 2, 2, 2, 1], padding="VALID", strides=[1, 2, 2, 2, 1]](3dcnn_1/conv3d_3_1/Relu)' with input shapes: [32,128,128,1,128].

Arguments received by MaxPooling3D.call():
  • inputs=tf.Tensor(shape=(32, 128, 128, 1, 128), dtype=float32)

## === cell 13
testimages = glob(os.path.join(TEST_DIR, "*", ""))
testimages = [p for p in testimages if os.path.isdir(p)]
testimages = [p for p in testimages if os.path.basename(os.path.normpath(p)).isdigit()]
testimages.sort(key=natural_keys)

flair_patches, t1w_patches, t1wce_patches, t2w_patches = [], [], [], []
for subfolder in testimages:
    flair_patches.append(glob(os.path.join(subfolder, "FLAIR", "*.dcm")))
    t1w_patches.append(glob(os.path.join(subfolder, "T1w", "*.dcm")))
    t1wce_patches.append(glob(os.path.join(subfolder, "T1wCE", "*.dcm")))
    t2w_patches.append(glob(os.path.join(subfolder, "T2w", "*.dcm")))

all_flair_patches = all_slice(flair_patches, [])
all_t1w_patches = all_slice(t1w_patches, [])
all_t1wce_patches = all_slice(t1wce_patches, [])
all_t2w_patches = all_slice(t2w_patches, [])

print("Num test subjects:", len(testimages))



## === cell 14
t2w_inputs_test = (
    create_input_3d(all_t2w_patches, len(testimages)).astype(np.float32) / 255.0
)
flair_inputs_test = (
    create_input_3d(all_flair_patches, len(testimages)).astype(np.float32) / 255.0
)
t1wce_inputs_test = (
    create_input_3d(all_t1wce_patches, len(testimages)).astype(np.float32) / 255.0
)
t1w_inputs_test = (
    create_input_3d(all_t1w_patches, len(testimages)).astype(np.float32) / 255.0
)

preds_flair = model_flair.predict(flair_inputs_test, verbose=0).reshape(-1)
preds_t1w = model_t1w.predict(t1w_inputs_test, verbose=0).reshape(-1)
preds_t2w = model_t2w.predict(t2w_inputs_test, verbose=0).reshape(-1)
preds_t1wce = model_t1wce.predict(t1wce_inputs_test, verbose=0).reshape(-1)

mean_test = (preds_flair + preds_t1w + preds_t2w + preds_t1wce) / 4.0
mean_test = np.clip(mean_test, 0.0, 1.0)

print("Pred range:", float(mean_test.min()), float(mean_test.max()))

sample = pd.read_csv(SAMPLE_SUB_CSV)
sample_ids = sample["BraTS21ID"].astype(str).tolist()

test_ids = [os.path.basename(os.path.normpath(p)) for p in testimages]
id_to_pred = {tid: float(p) for tid, p in zip(test_ids, mean_test)}
preds_ordered = [id_to_pred.get(tid, 0.5) for tid in sample_ids]

submission = pd.DataFrame({"BraTS21ID": sample_ids, "MGMT_value": preds_ordered})
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print("submission.csv exists:", os.path.isfile("submission.csv"))

## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1400941669.py in <cell line: 0>()
     12 )
     13 
---> 14 preds_flair = model_flair.predict(flair_inputs_test, verbose=0).reshape(-1)
     15 preds_t1w = model_t1w.predict(t1w_inputs_test, verbose=0).reshape(-1)
     16 preds_t2w = model_t2w.predict(t2w_inputs_test, verbose=0).reshape(-1)

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    122             raise e.with_traceback(filtered_tb) from None
    123         finally:
--> 124             del filtered_tb
    125 
    126     return error_handler

ValueError: Exception encountered when calling MaxPooling3D.call().

Negative dimension size caused by subtracting 2 from 1 for '{{node 3dcnn_1/max_pooling3d_3_1/MaxPool3D}} = MaxPool3D[T=DT_FLOAT, data_format="NDHWC", ksize=[1, 2, 2, 2, 1], padding="VALID", strides=[1, 2, 2, 2, 1]](3dcnn_1/conv3d_3_1/Relu)' with input shapes: [32,128,128,1,128].

Arguments received by MaxPooling3D.call():
  • inputs=tf.Tensor(shape=(32, 128, 128, 1, 128), dtype=float32)
