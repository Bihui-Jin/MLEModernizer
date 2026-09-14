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

No external packages required in the script and installed.

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

0.5278344382117967

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import warnings
from glob import glob

import numpy as np
import pandas as pd

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import pydicom as dicom

import cv2

import tensorflow as tf
from tensorflow.keras.utils import Sequence

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
BASE_PATH = "../input/rsna-miccai-brain-tumor-radiogenomic-classification"
TRAIN_LABELS_CSV = f"{BASE_PATH}/train_labels.csv"
SAMPLE_SUB_CSV = f"{BASE_PATH}/sample_submission.csv"
TEST_DIR = f"{BASE_PATH}/test"

train_df = pd.read_csv(TRAIN_LABELS_CSV)
test_df = pd.read_csv(SAMPLE_SUB_CSV)

train_array = np.array(train_df["BraTS21ID"])
train_label = np.array(train_df["MGMT_value"])
test_array = np.array(test_df["BraTS21ID"])
test_label = np.array(test_df["MGMT_value"])

DIM = 512
NB_CHANNELS = 1
BATCH_SIZE = 16

train_ds = pd.DataFrame(columns=["path", "label"])
test_ds = pd.DataFrame(columns=["path", "label"])

print("train_df:", train_df.shape, "test_df(sample_submission):", test_df.shape)



## === cell 2
dataframe_values = []
for a, b in zip(train_df["BraTS21ID"], train_df["MGMT_value"]):
    pid = str(a).zfill(5)
    pdir = f"{BASE_PATH}/train/{pid}"
    if not os.path.isdir(pdir):
        continue
    folders_list = os.listdir(pdir)
    for folder in folders_list:
        fdir = f"{pdir}/{folder}"
        if not os.path.isdir(fdir):
            continue
        img_list = os.listdir(fdir)
        for img in img_list:
            dataframe_values.append({"path": f"{fdir}/{img}", "label": b})

train_ds = pd.DataFrame.from_dict(dataframe_values)
print("train_ds slices:", len(train_ds))



## === cell 3
dataframe_values = []
for a, b in zip(test_df["BraTS21ID"], test_df["MGMT_value"]):
    pid = str(a).zfill(5)
    pdir = f"{BASE_PATH}/test/{pid}"
    if not os.path.isdir(pdir):
        continue
    folders_list = os.listdir(pdir)
    for folder in folders_list:
        fdir = f"{pdir}/{folder}"
        if not os.path.isdir(fdir):
            continue
        img_list = os.listdir(fdir)
        for img in img_list:
            dataframe_values.append({"path": f"{fdir}/{img}", "label": b})

test_ds = pd.DataFrame.from_dict(dataframe_values)
print("test_ds slices:", len(test_ds))




## === cell 4
class Frames_Generator(Sequence):
    def __init__(
        self,
        Sample_df,
        batch_size=1,
        dim=(512, 512),
        shuffle=True,
        train_ds=None,
        nb_steps=0,
    ):
        self.batch_size = batch_size
        self.Sample_df = np.array(Sample_df)
        self.shuffle = shuffle
        self.dim = dim
        self.train_ds = (
            train_ds.reset_index(drop=True) if train_ds is not None else None
        )
        self.nb_steps = int(nb_steps)
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.Sample_df) / self.batch_size))

    def __getitem__(self, index):
        _ = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]

        frames, labels = self.__data_generation()
        X = np.asarray(frames, dtype=np.float32) / 255.0
        y = np.asarray(labels, dtype=np.float32)
        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.Sample_df))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self):
        label_train = []
        frame_train = []

        while len(frame_train) < 16:
            if self.nb_steps >= len(self.train_ds):
                self.nb_steps = 0

            path = self.train_ds.loc[self.nb_steps, "path"]
            label = self.train_ds.loc[self.nb_steps, "label"]

            try:
                ds = dicom.dcmread(path)
                arr = ds.pixel_array
                img = cv2.resize(arr, self.dim, interpolation=cv2.INTER_AREA)
                img = img.reshape(self.dim[0], self.dim[1], 1)
            except Exception:
                self.nb_steps += 1
                continue

            frame_train.append(img)
            label_train.append(label)
            self.nb_steps += 1

        return frame_train, label_train




## === cell 5
train_steps_arr = np.arange(max(1, len(train_ds) // BATCH_SIZE - 16))
test_steps_arr = np.arange(max(1, len(test_ds) // BATCH_SIZE - 16))

train_params = {
    "Sample_df": train_steps_arr,
    "dim": (DIM, DIM),
    "batch_size": 1,
    "shuffle": True,
    "train_ds": train_ds,
    "nb_steps": 0,
}
test_params = {
    "Sample_df": test_steps_arr,
    "dim": (DIM, DIM),
    "batch_size": 1,
    "shuffle": False,  # keep deterministic ordering for test predictions aggregation
    "train_ds": test_ds,
    "nb_steps": 0,
}

training_generator = Frames_Generator(**train_params)
validation_generator = Frames_Generator(**test_params)

print("len(training_generator):", len(training_generator))
print("len(validation_generator):", len(validation_generator))



## === cell 6
MODEL_PATH = "../input/tumor-model/model.h5"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Expected pretrained model at {MODEL_PATH}. "
        f"Please ensure the dataset 'tumor-model' is attached."
    )

model = tf.keras.models.load_model(MODEL_PATH)
print("Loaded model.")



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/2596787368.py in <cell line: 0>()
      3 MODEL_PATH = "../input/tumor-model/model.h5"
      4 if not os.path.exists(MODEL_PATH):
----> 5     raise FileNotFoundError(
      6         f"Expected pretrained model at {MODEL_PATH}. "
      7         f"Please ensure the dataset 'tumor-model' is attached."

FileNotFoundError: Expected pretrained model at ../input/tumor-model/model.h5. Please ensure the dataset 'tumor-model' is attached.

## === cell 7
result = model.predict(validation_generator, verbose=1)
result = np.asarray(result).reshape(-1)
print("Pred slices predicted:", result.shape)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2387171467.py in <cell line: 0>()
      1 # Predict on all test slices in the same order as test_ds (generator iterates sequentially via nb_steps).
      2 # Fix: Keras in Kaggle can error with workers>0 depending on environment; use single-process predict.
----> 3 result = model.predict(validation_generator, verbose=1)
      4 result = np.asarray(result).reshape(-1)
      5 print("Pred slices predicted:", result.shape)

NameError: name 'model' is not defined

## === cell 8
data_dir = TEST_DIR
patients_test = sorted(
    [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
)

lengths = []
for patient in patients_test:
    seq_dirs = [
        os.path.join(data_dir, patient, s)
        for s in os.listdir(os.path.join(data_dir, patient))
    ]
    length = 0
    for sd in seq_dirs:
        if os.path.isdir(sd):
            length += len(glob(os.path.join(sd, "*.dcm")))
    lengths.append(length)

total_slices = int(np.sum(lengths))
if total_slices != len(test_ds):
    print(
        f"Warning: total_slices from disk={total_slices} differs from test_ds rows={len(test_ds)}"
    )

usable = min(len(result), total_slices)
result = result[:usable]

final_result = []
i = 0
for length in lengths:
    if length <= 0:
        final_result.append(0.5)
        continue
    j = min(i + length, usable)
    if j <= i:
        final_result.append(0.5)
    else:
        final_result.append(float(result[i:j].mean()))
    i += length

if len(final_result) != len(patients_test):
    raise RuntimeError(
        f"final_result length {len(final_result)} != patients_test length {len(patients_test)}"
    )

print("Patients:", len(patients_test), "Final preds:", len(final_result))



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/61185701.py in <cell line: 0>()
     27 
     28 # If predictions length doesn't equal number of slices, clamp safely.
---> 29 usable = min(len(result), total_slices)
     30 result = result[:usable]
     31 

NameError: name 'result' is not defined

## === cell 9
sub_template = pd.read_csv(SAMPLE_SUB_CSV)
sub_template["BraTS21ID"] = sub_template["BraTS21ID"].apply(lambda x: str(x).zfill(5))

pred_map = {pid: p for pid, p in zip(patients_test, final_result)}
submission = pd.DataFrame(
    {
        "BraTS21ID": sub_template["BraTS21ID"],
        "MGMT_value": sub_template["BraTS21ID"].map(pred_map).fillna(0.5).astype(float),
    }
)

submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2903689311.py in <cell line: 0>()
      4 
      5 # patients_test are already zfilled folder names; align to template order
----> 6 pred_map = {pid: p for pid, p in zip(patients_test, final_result)}
      7 submission = pd.DataFrame(
      8     {

NameError: name 'final_result' is not defined
