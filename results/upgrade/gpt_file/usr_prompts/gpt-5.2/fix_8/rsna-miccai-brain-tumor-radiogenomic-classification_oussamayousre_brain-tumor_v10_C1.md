# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import warnings
from collections import OrderedDict
from glob import glob

import numpy as np
import pandas as pd

import cv2

import tensorflow as tf
from tensorflow.keras.utils import Sequence

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
except Exception:
    pass


def read_dcm_cv2(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
    if img is None:
        raise ValueError(f"cv2.imread failed for {path}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img




## === cell 1
BASE_PATH = "../input/rsna-miccai-brain-tumor-radiogenomic-classification"
TRAIN_LABELS_CSV = f"{BASE_PATH}/train_labels.csv"
SAMPLE_SUB_CSV = f"{BASE_PATH}/sample_submission.csv"
TRAIN_DIR = f"{BASE_PATH}/train"
TEST_DIR = f"{BASE_PATH}/test"

train_df = pd.read_csv(TRAIN_LABELS_CSV)
test_df = pd.read_csv(SAMPLE_SUB_CSV)

train_df["BraTS21ID"] = train_df["BraTS21ID"].astype(str).str.zfill(5)
test_df["BraTS21ID"] = test_df["BraTS21ID"].astype(str).str.zfill(5)

DIM = 512
NB_CHANNELS = 1
BATCH_SIZE = 16

print("train_df:", train_df.shape, "test_df(sample_submission):", test_df.shape)



## === cell 2
bad_cases = set(["00109", "00123", "00709"])


def list_dcm_paths_two_levels(patient_dir: str):
    return glob(os.path.join(patient_dir, "*", "*.dcm"))


paths = []
labels = []
pids = []

pid_to_label = dict(zip(train_df["BraTS21ID"].values, train_df["MGMT_value"].values))

for pid in train_df["BraTS21ID"].values:
    if pid in bad_cases:
        continue
    pdir = f"{TRAIN_DIR}/{pid}"
    if not os.path.isdir(pdir):
        continue

    files = list_dcm_paths_two_levels(pdir)
    if not files:
        continue

    lb = float(pid_to_label[pid])
    paths.extend(files)
    labels.extend([lb] * len(files))
    pids.extend([pid] * len(files))

train_ds = pd.DataFrame(
    {"path": paths, "label": np.asarray(labels, dtype=np.float32), "pid": pids}
)
print("train_ds slices:", len(train_ds), "unique patients:", train_ds["pid"].nunique())



## === cell 3
paths = []
labels = []
pids = []

for pid in test_df["BraTS21ID"].values:
    pdir = f"{TEST_DIR}/{pid}"
    if not os.path.isdir(pdir):
        continue

    files = list_dcm_paths_two_levels(pdir)
    if not files:
        continue

    paths.extend(files)
    labels.extend([0.0] * len(files))
    pids.extend([pid] * len(files))

test_ds = pd.DataFrame(
    {"path": paths, "label": np.asarray(labels, dtype=np.float32), "pid": pids}
)
print("test_ds slices:", len(test_ds), "unique patients:", test_ds["pid"].nunique())




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
        cache_size=4096,
    ):
        self.batch_size = int(batch_size)
        self.Sample_df = np.array(Sample_df)
        self.shuffle = bool(shuffle)
        self.dim = tuple(dim)
        self.nb_steps = int(nb_steps)

        if train_ds is None:
            raise ValueError("train_ds must be provided")

        train_ds = train_ds.reset_index(drop=True)

        self._paths = train_ds["path"].to_numpy()
        self._labels = train_ds["label"].to_numpy(dtype=np.float32, copy=False)

        self._cache_size = int(cache_size) if cache_size is not None else 0
        self._cache = OrderedDict()  # path -> np.ndarray (H,W,1)

        self._inv255 = np.float32(1.0 / 255.0)

        self.on_epoch_end()

    def __len__(self):
        return int(len(self.Sample_df))

    def __getitem__(self, index):
        frames, labels = self.__data_generation(index)
        X = np.asarray(frames, dtype=np.float32) * self._inv255
        y = np.asarray(labels, dtype=np.float32)
        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.Sample_df))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def _read_resize_cached(self, path):
        if self._cache_size > 0:
            hit = self._cache.get(path, None)
            if hit is not None:
                self._cache.move_to_end(path, last=True)
                return hit

        arr = read_dcm_cv2(path)

        arr = arr.astype(np.float32, copy=False)
        mn, mx, _, _ = cv2.minMaxLoc(arr)
        if mx > mn:
            arr = (arr - mn) * (255.0 / (mx - mn))
        else:
            arr = np.zeros_like(arr, dtype=np.float32)

        img = cv2.resize(arr, self.dim, interpolation=cv2.INTER_AREA)
        img = img.reshape(self.dim[0], self.dim[1], 1)

        if self._cache_size > 0:
            self._cache[path] = img
            if len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return img

    def __data_generation(self, index):
        label_train = []
        frame_train = []

        target = 16  # keep original per-step target
        step = int(self.indexes[index])
        start = int((self.nb_steps + step * target) % max(1, len(self._paths)))

        j = 0
        while len(frame_train) < target and j < (target * 4):
            idx = (start + j) % len(self._paths)
            path = self._paths[idx]
            label = self._labels[idx]
            try:
                img = self._read_resize_cached(path)
            except Exception:
                j += 1
                continue
            frame_train.append(img)
            label_train.append(label)
            j += 1

        if len(frame_train) == 0:
            frame_train = [
                np.zeros((self.dim[0], self.dim[1], 1), dtype=np.float32)
            ] * target
            label_train = [0.0] * target
        elif len(frame_train) < target:
            pad_n = target - len(frame_train)
            frame_train.extend([frame_train[-1]] * pad_n)
            label_train.extend([label_train[-1]] * pad_n)

        return frame_train, label_train




## === cell 5
unique_pids = np.array(sorted(train_ds["pid"].unique()))
rng = np.random.RandomState(SEED)
rng.shuffle(unique_pids)

val_frac = 0.2
n_val = max(1, int(len(unique_pids) * val_frac))
val_pids = set(unique_pids[:n_val])
trn_pids = set(unique_pids[n_val:])

trn_ds = train_ds[train_ds["pid"].isin(trn_pids)].reset_index(drop=True)
val_ds = train_ds[train_ds["pid"].isin(val_pids)].reset_index(drop=True)

print("Train slices:", len(trn_ds), "Val slices:", len(val_ds))
print("Train patients:", len(trn_pids), "Val patients:", len(val_pids))

train_steps_arr = np.arange(max(1, len(trn_ds) // BATCH_SIZE - 16))
val_steps_arr = np.arange(max(1, len(val_ds) // BATCH_SIZE - 16))

train_params = {
    "Sample_df": train_steps_arr,
    "dim": (DIM, DIM),
    "batch_size": 1,
    "shuffle": True,
    "train_ds": trn_ds,
    "nb_steps": 0,
    "cache_size": 4096,
}
val_params = {
    "Sample_df": val_steps_arr,
    "dim": (DIM, DIM),
    "batch_size": 1,
    "shuffle": True,
    "train_ds": val_ds,
    "nb_steps": 0,
    "cache_size": 4096,
}

training_generator = Frames_Generator(**train_params)
validation_generator = Frames_Generator(**val_params)

print("len(training_generator):", len(training_generator))
print("len(validation_generator):", len(validation_generator))

inputs = tf.keras.Input(shape=(DIM, DIM, 1))
x = tf.keras.layers.Conv2D(8, 3, padding="same", activation="relu")(inputs)
x = tf.keras.layers.MaxPool2D()(x)
x = tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu")(x)
x = tf.keras.layers.MaxPool2D()(x)
x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(32, activation="relu")(x)
outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

model = tf.keras.Model(inputs, outputs)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="binary_crossentropy")

EPOCHS = 2

model.fit(
    training_generator,
    validation_data=validation_generator,
    epochs=EPOCHS,
    verbose=1,
)



## === cell 6
test_steps_arr = np.arange(max(1, len(test_ds) // BATCH_SIZE - 16))
test_params = {
    "Sample_df": test_steps_arr,
    "dim": (DIM, DIM),
    "batch_size": 1,
    "shuffle": False,
    "train_ds": test_ds,
    "nb_steps": 0,
    "cache_size": 4096,
}
test_generator = Frames_Generator(**test_params)

result = model.predict(
    test_generator,
    verbose=1,
)
result = np.asarray(result).reshape(-1)
print("Pred slices predicted:", result.shape)



## === cell 7
patients_test = list(test_df["BraTS21ID"].values)

pid_to_len = test_ds["pid"].value_counts().to_dict()
lengths = [int(pid_to_len.get(pid, 0)) for pid in patients_test]

total_slices = int(np.sum(lengths))
if total_slices != len(test_ds):
    print(
        f"Warning: total_slices from test_ds={total_slices} differs from test_ds rows={len(test_ds)}"
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
        final_result.append(float(np.mean(result[i:j])))
    i += length

if len(final_result) < len(patients_test):
    final_result.extend([0.5] * (len(patients_test) - len(final_result)))

final_result = final_result[: len(patients_test)]
print("Patients:", len(patients_test), "Final preds:", len(final_result))



## === cell 8
submission = pd.DataFrame(
    {
        "BraTS21ID": test_df["BraTS21ID"].values,
        "MGMT_value": np.clip(np.asarray(final_result, dtype=float), 0.0, 1.0),
    }
)

submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print("submission.csv exists:", os.path.exists("submission.csv"))
