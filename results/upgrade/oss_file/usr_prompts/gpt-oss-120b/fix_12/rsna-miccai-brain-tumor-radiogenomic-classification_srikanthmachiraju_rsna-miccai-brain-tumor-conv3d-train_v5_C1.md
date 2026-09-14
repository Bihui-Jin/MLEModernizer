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
pass



## === cell 1
pass



## === cell 2
import os
import re
import glob
import numpy as np
import pandas as pd
import cv2
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")
import random as rn
import matplotlib.pyplot as plt
import pydicom
import math
from concurrent.futures import ThreadPoolExecutor

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf

tf.config.optimizer.set_jit(True)

from tensorflow.keras.callbacks import *
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow import keras

from sklearn.model_selection import StratifiedKFold
from tensorflow.keras import backend as K, optimizers, regularizers
from random import shuffle

rn.seed(30)
np.random.seed(30)
tf.compat.v1.random.set_random_seed(30)
from pydicom.pixel_data_handlers.util import apply_voi_lut



## === cell 3
config = {
    "images_source_path": "../input/rsna-miccai-brain-tumor-radiogenomic-classification/train",
    "test_images_source_path": "../input/rsna-miccai-brain-tumor-radiogenomic-classification/test",
    "csv_path": "../input/rsna-miccai-brain-tumor-radiogenomic-classification/train_labels.csv",
    "data_path": "../input/rsna-miccai-brain-tumor-radiogenomic-classification",
    "output_path": "./crnn/",
    "nfolds": 3,
    "global_seed": 42,
    "batch_size": 32,
    "frames_per_seq": 12,
    "img_size": 224,
    "learning_rate": 0.0001,
    "num_epochs": 10,
    "channels": 3,
    "scale": 0.75,
}

mri_types = ["T2w"]



## === cell 4
df_data = pd.read_csv(config["csv_path"])
df_data["folder_name"] = [format(x, "05d") for x in df_data["BraTS21ID"]]
df_data["folder_path"] = [
    os.path.join(config["images_source_path"], x) for x in df_data["folder_name"]
]

skf = StratifiedKFold(
    n_splits=config["nfolds"], shuffle=True, random_state=config["global_seed"]
)

for index, (train_index, val_index) in enumerate(
    skf.split(X=df_data.index, y=df_data.MGMT_value)
):
    df_data.loc[val_index, "fold"] = index

df_train = df_data[df_data.fold != 0].reset_index(drop=True)
df_val = df_data[df_data.fold == 0].reset_index(drop=True)




## === cell 5
class Dataset(tf.keras.utils.Sequence):
    def __init__(
        self, df, is_train=True, batch_size=config["batch_size"], shuffle=True
    ):
        self.paths = df["folder_path"].values
        self.y = df["MGMT_value"].values if "MGMT_value" in df.columns else None
        self.is_train = is_train
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.file_lists = [self.get_img_path_3d(p, mri_types[0]) for p in self.paths]

        self._cache = {}

        max_workers = min(8, (os.cpu_count() or 1))
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            for idx, img in enumerate(
                ex.map(self._load_from_filelist, self.file_lists)
            ):
                self._cache[idx] = img

        self.order = np.arange(len(self.paths))
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.paths) / self.batch_size)

    def __getitem__(self, idx):
        batch_start = idx * self.batch_size
        batch_end = (idx + 1) * self.batch_size
        batch_idx = self.order[batch_start:batch_end]

        batch_imgs = [self._cache[i] for i in batch_idx]

        batch_X = np.stack(batch_imgs, axis=0).astype(np.float32)

        if self.is_train:
            batch_y = self.y[batch_idx]
            return batch_X, batch_y
        else:
            return batch_X

    def _load_from_filelist(self, file_list):
        """Load a single subject given its pre‑computed DICOM file list."""
        img3d = np.stack([self.read_mri(f) for f in file_list], axis=-1)  # (H, W, D)
        if img3d.shape[-1] < config["frames_per_seq"]:
            n_zero = np.zeros(
                (
                    config["img_size"],
                    config["img_size"],
                    config["frames_per_seq"] - img3d.shape[-1],
                ),
                dtype=img3d.dtype,
            )
            img3d = np.concatenate((img3d, n_zero), axis=-1)
        return np.expand_dims(img3d, -1)  # (H, W, D, 1)

    def read_mri(self, path, voi_lut=True, fix_monochrome=True):
        dicom = pydicom.dcmread(path)
        if voi_lut:
            data = apply_voi_lut(dicom.pixel_array, dicom)
        else:
            data = dicom.pixel_array
        if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
            data = np.amax(data) - data
        data = data - np.min(data)
        data = data / np.max(data)
        data = (data * 255).astype(np.uint8)
        data = cv2.resize(data, (config["img_size"], config["img_size"]))
        return data

    def get_img_path_3d(self, scan_id, mri_type):
        """Return a list of selected slice file paths for one subject."""
        modality_path = Path(scan_id) / mri_type
        files = sorted(
            modality_path.glob("*.dcm"),
            key=lambda p: (
                int(re.findall(r"\d+", p.name)[0]) if re.findall(r"\d+", p.name) else 0
            ),
        )
        total_img_num = len(files)
        mid_num = total_img_num // 2
        num_3d2 = config["frames_per_seq"] // 2
        start_idx = max(0, mid_num - num_3d2)
        end_idx = min(total_img_num, mid_num + num_3d2)
        return [str(f) for f in files[start_idx:end_idx]]

    def on_epoch_end(self):
        if self.shuffle and self.is_train:
            np.random.shuffle(self.order)


train_dataset = Dataset(
    df_train, is_train=True, batch_size=config["batch_size"], shuffle=True
)
valid_dataset = Dataset(
    df_val, is_train=True, batch_size=config["batch_size"], shuffle=False
)

for i in range(1):
    images, label = train_dataset[i]
    print("Dimension of the CT scan is:", images.shape)
    print("label=", label)
    plt.imshow(images[0, :, :, 3, 0], cmap="gray")
    plt.show()




## === cell 6
def get_3d_model(
    width=config["img_size"], height=config["img_size"], depth=config["frames_per_seq"]
):
    """Build a 3D convolutional neural network model."""
    inputs = keras.Input((width, height, depth, 1))
    x = Conv3D(32, 3, padding="same", activation="relu")(inputs)
    x = MaxPool3D(pool_size=(2, 2, 1))(x)
    x = BatchNormalization()(x)

    x = Conv3D(32, 3, padding="same", activation="relu")(x)
    x = MaxPool3D(pool_size=(2, 2, 1))(x)
    x = BatchNormalization()(x)

    x = Conv3D(64, 3, padding="same", activation="relu")(x)
    x = MaxPool3D(pool_size=(2, 2, 1))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.01)(x)

    x = Conv3D(128, 3, padding="same", activation="relu")(x)
    x = MaxPool3D(pool_size=(2, 2, 1))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.02)(x)

    x = Conv3D(256, 3, padding="same", activation="relu")(x)
    x = MaxPool3D(pool_size=(2, 2, 1))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.03)(x)

    x = GlobalAveragePooling3D()(x)
    x = Dense(1024, activation="relu")(x)
    x = Dropout(0.08)(x)
    outputs = Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs, name="3dcnn")
    return model


model = get_3d_model()
model.summary()



## === cell 7
from tensorflow.keras.metrics import AUC

initial_learning_rate = 0.0001
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
)
model.compile(
    loss="binary_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
    metrics=[AUC(name="auc"), "accuracy"],
)

model.fit(
    train_dataset,
    validation_data=valid_dataset,
    epochs=config["num_epochs"],
    shuffle=True,
    verbose=1,
)



## === cell 8
sample_submission_path = os.path.join(config["data_path"], "sample_submission.csv")
sample_df = pd.read_csv(sample_submission_path)

test_df = sample_df.copy()
test_df["folder_name"] = [format(x, "05d") for x in test_df.BraTS21ID]
test_df["folder_path"] = [
    os.path.join(config["data_path"], "test", x) for x in test_df["folder_name"]
]
test_dataset = Dataset(test_df, is_train=False, batch_size=1, shuffle=False)

preds = model.predict(test_dataset, verbose=0)
preds = preds.reshape(-1)
submission = pd.DataFrame({"BraTS21ID": sample_df["BraTS21ID"], "MGMT_value": preds})



## === cell 9
print(submission.head())



## === cell 10
submission.to_csv("submission.csv", index=False)
print("Saved submission to submission.csv")
