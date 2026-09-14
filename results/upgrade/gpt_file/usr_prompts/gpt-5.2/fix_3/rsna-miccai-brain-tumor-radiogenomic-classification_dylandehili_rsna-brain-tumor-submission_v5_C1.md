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

3.10

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
import os, math, glob, re, random
import numpy as np
import pandas as pd
import cv2
import pydicom

import tensorflow as tf
from sklearn.model_selection import train_test_split

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")




## === cell 1
IMAGE_SIZE = 256
IMAGE_DEPTH = 64

mri_types = ["FLAIR", "T1w", "T1wCE", "T2w"]
CHANNELS = len(mri_types)

local_directory = "../input/rsna-miccai-brain-tumor-radiogenomic-classification"
local_label_path = local_directory + "/train_labels.csv"
local_submission_path = local_directory + "/sample_submission.csv"

train_root = local_directory + "/train"
test_root = local_directory + "/test"




## === cell 2

from functools import lru_cache


def load_dicom_slice(path, img_size=256):
    dicom = pydicom.dcmread(path)
    data = dicom.pixel_array.astype(np.float32)

    if data.max() > data.min():
        data = (data - data.min()) / (data.max() - data.min())
    else:
        data = data * 0.0

    data = cv2.resize(data, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    return data.astype(np.float32)


def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    return sorted(data, key=alphanum_key)


@lru_cache(maxsize=None)
def _sorted_dicom_files(split, scan_id, mri_type):
    pattern = f"{local_directory}/{split}/{scan_id}/{mri_type}/*.dcm"
    files = tf.io.gfile.glob(pattern)
    if not files:
        return tuple()

    meta = []
    ok = True
    for f in files:
        try:
            ds = pydicom.dcmread(
                f,
                stop_before_pixels=True,
                specific_tags=["InstanceNumber", "ImagePositionPatient"],
            )
            inst = getattr(ds, "InstanceNumber", None)
            ipp = getattr(ds, "ImagePositionPatient", None)
            if inst is not None:
                key = (0, int(inst))
            elif ipp is not None and len(ipp) >= 3:
                key = (1, float(ipp[2]))
            else:
                ok = False
                break
            meta.append((key, f))
        except Exception:
            ok = False
            break

    if ok and len(meta) == len(files):
        meta.sort(key=lambda x: x[0])
        return tuple([f for _, f in meta])

    return tuple(
        sorted(
            files,
            key=lambda var: [
                int(x) if x.isdigit() else x for x in re.findall(r"[^0-9]|[0-9]+", var)
            ],
        )
    )


def load_dicom_modality(mri_type, scan_id, img_depth, img_size, split):
    files = list(_sorted_dicom_files(split, scan_id, mri_type))

    num_files = len(files)
    if num_files == 0:
        return np.zeros((img_size, img_size, img_depth), dtype=np.float32)

    num_files_middle = num_files // 2
    img_depth_middle = img_depth // 2

    start_depth = max(0, num_files_middle - img_depth_middle)
    end_depth = min(num_files, num_files_middle + img_depth_middle)

    img3d = np.stack(
        [
            load_dicom_slice(dicom_img, img_size=img_size)
            for dicom_img in files[start_depth:end_depth]
        ]
    ).T

    if img3d.shape[-1] < img_depth:
        n_zero = np.zeros(
            (img_size, img_size, img_depth - img3d.shape[-1]), dtype=np.float32
        )
        img3d = np.concatenate((img3d, n_zero), axis=-1)

    return img3d.astype(np.float32)


def load_dicom_3D(scan_id, img_depth=128, img_size=256, split="test", verbose=False):
    if verbose:
        print(scan_id, end=" ")

    dicom_channels = [
        load_dicom_modality(
            scan_id=scan_id,
            img_depth=img_depth,
            img_size=img_size,
            split=split,
            mri_type=mtype,
        )
        for mtype in mri_types
    ]

    img = np.stack(dicom_channels, axis=-1).astype(np.float32)

    mn, mx = np.min(img), np.max(img)
    if mn < mx:
        img = (img - mn) / (mx - mn)
    else:
        img = img * 0.0

    return img.astype(np.float32)




## === cell 3
def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[float(value)]))


def serialize_example(image, label=None):
    feature = {"image": _bytes_feature(image.tobytes())}
    if label is not None:
        feature["label"] = _float_feature(label)
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    return sorted(data, key=alphanum_key)




## === cell 4
labels = pd.read_csv(local_label_path)
labels["BraTS21ID"] = labels["BraTS21ID"].astype(str).str.zfill(5)

bad_ids = set(["00109", "00123", "00709"])
labels = labels[~labels["BraTS21ID"].isin(bad_ids)].reset_index(drop=True)

train_ids, val_ids = train_test_split(
    labels["BraTS21ID"].values,
    test_size=0.15,
    random_state=SEED,
    stratify=labels["MGMT_value"].values,
)

id_to_label = dict(zip(labels["BraTS21ID"].values, labels["MGMT_value"].values))

print("Train n:", len(train_ids), "Val n:", len(val_ids))




## === cell 5
os.makedirs("./tfrecords", exist_ok=True)

from concurrent.futures import ThreadPoolExecutor


def _build_serialized_for_id(sid, include_labels, split_for_load):
    img = load_dicom_3D(
        sid,
        img_size=IMAGE_SIZE,
        img_depth=IMAGE_DEPTH,
        split=split_for_load,
    )
    y = id_to_label[sid] if include_labels else None
    return sid, serialize_example(img, y)


def write_tfrec(ids, split_name, include_labels):
    outpath = f"./tfrecords/brain_{split_name}.tfrec"
    split_for_load = "train" if split_name in ["train", "val"] else "test"

    max_workers = min(8, (os.cpu_count() or 4))
    futures = {}

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for sid in ids:
            futures[sid] = ex.submit(
                _build_serialized_for_id, sid, include_labels, split_for_load
            )

        with tf.io.TFRecordWriter(
            outpath, options=tf.io.TFRecordOptions(compression_type="GZIP")
        ) as writer:
            for sid in ids:
                _, ser = futures[sid].result()
                writer.write(ser)

    return outpath


train_tfrec = write_tfrec(train_ids, "train", include_labels=True)
val_tfrec = write_tfrec(val_ids, "val", include_labels=True)

sub = pd.read_csv(local_submission_path)
sub["BraTS21ID"] = sub["BraTS21ID"].astype(str).str.zfill(5)
test_ids = sub["BraTS21ID"].tolist()

test_tfrec = write_tfrec(test_ids, "test", include_labels=False)

print("TFRecords written:", train_tfrec, val_tfrec, test_tfrec)




## === cell 6
AUTO = tf.data.AUTOTUNE


def deserialize_example_with_label(serialized_string):
    image_feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.float32),
    }
    parsed = tf.io.parse_single_example(serialized_string, image_feature_description)
    image = tf.io.decode_raw(parsed["image"], tf.float32)
    image = tf.reshape(image, [IMAGE_SIZE, IMAGE_SIZE, IMAGE_DEPTH, CHANNELS])
    label = parsed["label"]
    return image, label


def deserialize_example_image_only(serialized_string):
    image_feature_description = {"image": tf.io.FixedLenFeature([], tf.string)}
    parsed = tf.io.parse_single_example(serialized_string, image_feature_description)
    image = tf.io.decode_raw(parsed["image"], tf.float32)
    image = tf.reshape(image, [IMAGE_SIZE, IMAGE_SIZE, IMAGE_DEPTH, CHANNELS])
    return image


BATCH_SIZE = 1

options = tf.data.Options()
options.experimental_deterministic = True

train_ds = (
    tf.data.TFRecordDataset(
        train_tfrec, compression_type="GZIP", num_parallel_reads=AUTO
    )
    .with_options(options)
    .map(deserialize_example_with_label, num_parallel_calls=AUTO)
    .shuffle(256, seed=SEED, reshuffle_each_iteration=True)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

val_ds = (
    tf.data.TFRecordDataset(val_tfrec, compression_type="GZIP", num_parallel_reads=AUTO)
    .with_options(options)
    .map(deserialize_example_with_label, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

test_ds = (
    tf.data.TFRecordDataset(
        test_tfrec, compression_type="GZIP", num_parallel_reads=AUTO
    )
    .with_options(options)
    .map(deserialize_example_image_only, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)




## === cell 7
def build_model():
    inputs = tf.keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, IMAGE_DEPTH, CHANNELS))
    x = tf.keras.layers.Conv3D(
        8, (3, 3, 3), padding="same", activation=tf.nn.leaky_relu
    )(inputs)
    x = tf.keras.layers.MaxPool3D((2, 2, 2))(x)

    x = tf.keras.layers.Conv3D(
        16, (3, 3, 3), padding="same", activation=tf.nn.leaky_relu
    )(x)
    x = tf.keras.layers.MaxPool3D((2, 2, 2))(x)

    x = tf.keras.layers.Conv3D(
        32, (3, 3, 3), padding="same", activation=tf.nn.leaky_relu
    )(x)
    x = tf.keras.layers.GlobalAveragePooling3D()(x)

    x = tf.keras.layers.Dense(64, activation=tf.nn.leaky_relu)(x)
    x = tf.keras.layers.Dropout(0.2, seed=SEED)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="auc")],
    )
    return model


model = build_model()
model.summary()




## === cell 8
EPOCHS = 3
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=1)




## === cell 9
pred = model.predict(test_ds, verbose=1)
pred = pred.reshape(-1).astype(np.float32)

print("Pred shape:", pred.shape, "min/max:", float(pred.min()), float(pred.max()))




## === cell 10
sub = pd.read_csv(local_submission_path)
sub["BraTS21ID"] = sub["BraTS21ID"].astype(str).str.zfill(5)

if len(pred) != len(sub):
    raise RuntimeError(
        f"Prediction length {len(pred)} does not match submission length {len(sub)}"
    )

sub["MGMT_value"] = pred
sub.to_csv("submission.csv", index=False)
print(sub.head())
print("Wrote submission.csv with", len(sub), "rows")
