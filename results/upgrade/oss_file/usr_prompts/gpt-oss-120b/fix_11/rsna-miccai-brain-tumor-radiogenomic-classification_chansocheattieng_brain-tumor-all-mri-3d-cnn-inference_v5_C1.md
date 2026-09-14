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
import math
import numpy as np
import pandas as pd
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import glob
import re
import sys

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

tf = None
keras = None
layers = None
KerasSequence = None

if KerasSequence is not None:
    BaseSequence = KerasSequence
else:

    class BaseSequence:
        """Fallback base class when Keras is unavailable."""

        pass


from sklearn import model_selection as sk_model_selection
from sklearn.linear_model import LogisticRegression




## === cell 1
DATA_DIR = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification"

train_labels_path = os.path.join(DATA_DIR, "train_labels.csv")
sample_submission_path = os.path.join(DATA_DIR, "sample_submission.csv")

train_df = pd.read_csv(train_labels_path)
sample_submission = pd.read_csv(sample_submission_path)

to_exclude = [109, 123, 709]
train_df["BraTS21ID_int"] = train_df["BraTS21ID"].astype(int)
train_df = train_df[~train_df["BraTS21ID_int"].isin(to_exclude)]

train_df["BraTS21ID5"] = train_df["BraTS21ID"].apply(lambda x: f"{int(x):05d}")
sample_submission["BraTS21ID5"] = sample_submission["BraTS21ID"].apply(
    lambda x: f"{int(x):05d}"
)




## === cell 2
IMAGE_SIZE = 128  # target 2‑D size after resize (kept for compatibility)
NUM_IMAGES_PER_TYPE = 32  # number of slices per MRI modality
NUM_IMAGES = NUM_IMAGES_PER_TYPE * 4
BATCH_SIZE = 4
MRI_TYPES = ["FLAIR", "T1w", "T1wCE", "T2w"]


def load_dicom_image(path, img_size=IMAGE_SIZE, voi_lut=True, rotate=0):
    """Read a single DICOM file and return a 2‑D numpy array."""
    dicom = pydicom.dcmread(path)
    data = dicom.pixel_array.astype(np.float32)
    if voi_lut:
        data = apply_voi_lut(data, dicom)
    if data.max() > data.min():
        data = (data - data.min()) / (data.max() - data.min())
    if data.shape != (img_size, img_size):
        data = np.resize(data, (img_size, img_size))
    return data




## === cell 3
def load_dicom_images_3d(
    scan_id,
    split="train",
    mri_type="FLAIR",
    num_imgs=NUM_IMAGES_PER_TYPE,
    img_size=IMAGE_SIZE,
    rotate=0,
):
    """Load a stack of slices for a single MRI modality."""
    folder = os.path.join(DATA_DIR, split, scan_id, mri_type)
    files = sorted(
        [f for f in glob.glob(os.path.join(folder, "*.dcm"))],
        key=lambda var: [
            int(x) if x.isdigit() else x for x in re.findall(r"[^0-9]|[0-9]+", var)
        ],
    )
    if not files:  # safety guard
        return np.zeros((img_size, img_size, num_imgs), dtype=np.float32)

    middle = len(files) // 2
    p1 = max(0, middle - num_imgs)
    p2 = min(len(files), middle + num_imgs)
    selected = files[p1:p2:2]  # take every second slice to reduce count

    slices = [load_dicom_image(f, rotate=rotate) for f in selected]
    img3d = np.stack(slices, axis=-1)  # shape (H, W, depth)

    if img3d.shape[-1] < num_imgs:
        pad_width = num_imgs - img3d.shape[-1]
        pad_front = pad_width // 2
        pad_back = pad_width - pad_front
        img3d = np.concatenate(
            [
                np.zeros((img_size, img_size, pad_front), dtype=np.float32),
                img3d,
                np.zeros((img_size, img_size, pad_back), dtype=np.float32),
            ],
            axis=-1,
        )
    elif img3d.shape[-1] > num_imgs:
        img3d = img3d[..., :num_imgs]

    return img3d


def load_dicom_images_3d_all(scan_id, split="train"):
    """Concatenate all four MRI modalities along the channel dimension."""
    arrays = [load_dicom_images_3d(scan_id, split, mri_type) for mri_type in MRI_TYPES]
    return np.concatenate(arrays, axis=-1)  # shape (H, W, depth_total)




## === cell 4
class Dataset(BaseSequence):
    """Keras Sequence that yields 3‑D volumes (with a singleton channel)."""

    def __init__(
        self, df, split="train", is_train=True, batch_size=BATCH_SIZE, shuffle=True
    ):
        self.ids = df["BraTS21ID5"].values
        self.labels = df["MGMT_value"].values if "MGMT_value" in df.columns else None
        self.split = split
        self.is_train = is_train
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.ids) / self.batch_size)

    def __getitem__(self, idx):
        batch_ids = self.ids[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_vols = [load_dicom_images_3d_all(sid, self.split) for sid in batch_ids]
        batch_X = np.stack(batch_vols, axis=0)  # (B, H, W, D)
        batch_X = np.expand_dims(batch_X, -1)  # (B, H, W, D, 1)

        if self.is_train:
            batch_y = self.labels[idx * self.batch_size : (idx + 1) * self.batch_size]
            return batch_X, batch_y
        else:
            return batch_X

    def on_epoch_end(self):
        if self.shuffle and self.is_train:
            perm = np.random.permutation(len(self.ids))
            self.ids = self.ids[perm]
            if self.labels is not None:
                self.labels = self.labels[perm]




## === cell 5
df_train, df_valid = sk_model_selection.train_test_split(
    train_df,
    test_size=0.2,
    random_state=12,
    stratify=train_df["MGMT_value"],
)


def extract_mean_feature(df, split):
    feats = []
    for sid in df["BraTS21ID5"].values:
        vol = load_dicom_images_3d_all(sid, split=split)
        feats.append(vol.mean())
    return np.array(feats).reshape(-1, 1)


X_train = extract_mean_feature(df_train, split="train")
y_train = df_train["MGMT_value"].values

X_valid = extract_mean_feature(df_valid, split="train")
y_valid = df_valid["MGMT_value"].values

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

X_test = extract_mean_feature(sample_submission, split="test")
test_probs = clf.predict_proba(X_test)[:, 1].astype(np.float32)

predictions = 1.0 - test_probs  # keep original transformation




## === cell 6
test_dataset = Dataset(
    sample_submission, split="test", is_train=False, batch_size=1, shuffle=False
)




## === cell 7
def get_model(width=IMAGE_SIZE, height=IMAGE_SIZE, depth=NUM_IMAGES * 4, channels=1):
    """3‑D CNN model compatible with the data shape."""
    if keras is None:
        return None
    inputs = keras.Input((width, height, depth, channels))

    x = layers.Conv3D(32, 3, activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool3D(2)(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv3D(64, 3, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool3D(2)(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv3D(128, 3, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool3D(2)(x)
    x = layers.Dropout(0.2)(x)

    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs, name="3D_CNN")
    return model


model = get_model()
if model is not None:
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc")],
    )




## === cell 8
submission = pd.DataFrame(
    {"BraTS21ID": sample_submission["BraTS21ID"], "MGMT_value": predictions}
)
submission.to_csv("submission.csv", index=False)
