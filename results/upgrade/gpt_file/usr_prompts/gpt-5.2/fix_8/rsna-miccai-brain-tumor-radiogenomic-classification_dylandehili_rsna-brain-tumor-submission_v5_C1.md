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

# 5. Target score

0.5515043345232025

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

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

try:
    tf.config.threading.set_intra_op_parallelism_threads(
        max(1, (os.cpu_count() or 4) // 2)
    )
    tf.config.threading.set_inter_op_parallelism_threads(
        max(1, (os.cpu_count() or 4) // 2)
    )
except Exception:
    pass




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

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

try:
    pydicom.config.image_handlers = ["gdcm", "pylibjpeg", "numpy"]
except Exception:
    pass

_num_token_re = re.compile(r"\d+")


def _numeric_key_from_path(p: str) -> int:
    m = _num_token_re.findall(os.path.basename(p))
    return int(m[-1]) if m else -1


@lru_cache(maxsize=None)
def _sorted_dicom_files(split, scan_id, mri_type):
    pattern = f"{local_directory}/{split}/{scan_id}/{mri_type}/*.dcm"
    files = tf.io.gfile.glob(pattern)
    if not files:
        return tuple()
    return tuple(sorted(files, key=_numeric_key_from_path))


def _minmax_norm_inplace(data: np.ndarray) -> np.ndarray:
    mn = float(data.min())
    mx = float(data.max())
    if mx > mn:
        data = (data - mn) / (mx - mn)
    else:
        data = data * 0.0
    return data


_DCM_READ_KW = dict(
    force=True,
    stop_before_pixels=False,
    specific_tags=["PixelData", "RescaleSlope", "RescaleIntercept"],
)
_DCM_READ_KW_FAST = dict(
    force=True, stop_before_pixels=False, specific_tags=["PixelData"]
)


def load_dicom_slice(path, img_size=256):
    ds = pydicom.dcmread(path, **_DCM_READ_KW_FAST)
    data = ds.pixel_array.astype(np.float32, copy=False)

    slope = getattr(ds, "RescaleSlope", None)
    intercept = getattr(ds, "RescaleIntercept", None)
    if slope is None and intercept is None:
        ds2 = pydicom.dcmread(path, **_DCM_READ_KW)
        slope = getattr(ds2, "RescaleSlope", None)
        intercept = getattr(ds2, "RescaleIntercept", None)

    if slope is not None or intercept is not None:
        s = float(slope) if slope is not None else 1.0
        itc = float(intercept) if intercept is not None else 0.0
        if s != 1.0 or itc != 0.0:
            data = data * s + itc

    data = _minmax_norm_inplace(data)
    data = cv2.resize(data, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    return data.astype(np.float32, copy=False)


def load_dicom_modality(mri_type, scan_id, img_depth, img_size, split):
    files = _sorted_dicom_files(split, scan_id, mri_type)
    num_files = len(files)
    if num_files == 0:
        return np.zeros((img_size, img_size, img_depth), dtype=np.float32)

    num_files_middle = num_files // 2
    img_depth_middle = img_depth // 2

    start_depth = max(0, num_files_middle - img_depth_middle)
    end_depth = min(num_files, num_files_middle + img_depth_middle)

    slice_files = files[start_depth:end_depth]

    vol = np.zeros((img_size, img_size, img_depth), dtype=np.float32)
    _lds = load_dicom_slice
    for i, f in enumerate(slice_files):
        vol[:, :, i] = _lds(f, img_size=img_size)
    return vol


def load_dicom_3D(scan_id, img_depth=128, img_size=256, split="test", verbose=False):
    if verbose:
        print(scan_id, end=" ")

    img = np.empty((img_size, img_size, img_depth, CHANNELS), dtype=np.float32)
    for c, mtype in enumerate(mri_types):
        img[:, :, :, c] = load_dicom_modality(
            scan_id=scan_id,
            img_depth=img_depth,
            img_size=img_size,
            split=split,
            mri_type=mtype,
        )

    img = _minmax_norm_inplace(img)
    return img.astype(np.float32, copy=False)




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
    return serialize_example(img, y)


def write_tfrec(ids, split_name, include_labels):
    outpath = f"./tfrecords/brain_{split_name}.tfrec"

    if tf.io.gfile.exists(outpath) and tf.io.gfile.stat(outpath).length > 0:
        return outpath

    split_for_load = "train" if split_name in ["train", "val"] else "test"

    cpu = os.cpu_count() or 8
    max_workers = max(4, min(16, cpu))  # keep same paradigm; tuned for I/O latency

    _fn = _build_serialized_for_id

    with ThreadPoolExecutor(max_workers=max_workers) as ex, tf.io.TFRecordWriter(
        outpath, options=tf.io.TFRecordOptions(compression_type="GZIP")
    ) as writer:
        for ser in ex.map(
            lambda sid: _fn(sid, include_labels, split_for_load), ids, chunksize=64
        ):
            writer.write(ser)

    return outpath


train_tfrec = write_tfrec(train_ids, "train", include_labels=True)
val_tfrec = write_tfrec(val_ids, "val", include_labels=True)

sub = pd.read_csv(local_submission_path)
sub["BraTS21ID"] = sub["BraTS21ID"].astype(str).str.zfill(5)
test_ids = sub["BraTS21ID"].tolist()

test_tfrec = write_tfrec(test_ids, "test", include_labels=False)

print("TFRecords written:", train_tfrec, val_tfrec, test_tfrec)




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/2232865886.py in <cell line: 0>()
     41 
     42 
---> 43 train_tfrec = write_tfrec(train_ids, "train", include_labels=True)
     44 val_tfrec = write_tfrec(val_ids, "val", include_labels=True)
     45 

/tmp/ipykernel_11/2232865886.py in write_tfrec(ids, split_name, include_labels)
     33     ) as writer:
     34         # Speed: larger chunksize reduces executor overhead; correctness unaffected (order doesn't matter).
---> 35         for ser in ex.map(
     36             lambda sid: _fn(sid, include_labels, split_for_load), ids, chunksize=64
     37         ):

/usr/lib/python3.11/concurrent/futures/_base.py in result_iterator()
    617                     # Careful not to keep a reference to the popped future
    618                     if timeout is None:
--> 619                         yield _result_or_cancel(fs.pop())
    620                     else:
    621                         yield _result_or_cancel(fs.pop(), end_time - time.monotonic())

/usr/lib/python3.11/concurrent/futures/_base.py in _result_or_cancel(***failed resolving arguments***)
    315     try:
    316         try:
--> 317             return fut.result(timeout)
    318         finally:
    319             fut.cancel()

/usr/lib/python3.11/concurrent/futures/_base.py in result(self, timeout)
    454                     raise CancelledError()
    455                 elif self._state == FINISHED:
--> 456                     return self.__get_result()
    457                 else:
    458                     raise TimeoutError()

/usr/lib/python3.11/concurrent/futures/_base.py in __get_result(self)
    399         if self._exception:
    400             try:
--> 401                 raise self._exception
    402             finally:
    403                 # Break a reference cycle with the exception in self._exception

/usr/lib/python3.11/concurrent/futures/thread.py in run(self)
     56 
     57         try:
---> 58             result = self.fn(*self.args, **self.kwargs)
     59         except BaseException as exc:
     60             self.future.set_exception(exc)

/tmp/ipykernel_11/2232865886.py in <lambda>(sid)
     34         # Speed: larger chunksize reduces executor overhead; correctness unaffected (order doesn't matter).
     35         for ser in ex.map(
---> 36             lambda sid: _fn(sid, include_labels, split_for_load), ids, chunksize=64
     37         ):
     38             writer.write(ser)

/tmp/ipykernel_11/2232865886.py in _build_serialized_for_id(sid, include_labels, split_for_load)
      5 
      6 def _build_serialized_for_id(sid, include_labels, split_for_load):
----> 7     img = load_dicom_3D(
      8         sid,
      9         img_size=IMAGE_SIZE,

/tmp/ipykernel_11/490233845.py in load_dicom_3D(scan_id, img_depth, img_size, split, verbose)
     99     img = np.empty((img_size, img_size, img_depth, CHANNELS), dtype=np.float32)
    100     for c, mtype in enumerate(mri_types):
--> 101         img[:, :, :, c] = load_dicom_modality(
    102             scan_id=scan_id,
    103             img_depth=img_depth,

/tmp/ipykernel_11/490233845.py in load_dicom_modality(mri_type, scan_id, img_depth, img_size, split)
     89     _lds = load_dicom_slice
     90     for i, f in enumerate(slice_files):
---> 91         vol[:, :, i] = _lds(f, img_size=img_size)
     92     return vol
     93 

/tmp/ipykernel_11/490233845.py in load_dicom_slice(path, img_size)
     50     # Try fast read first (pixel data only); if slope/intercept needed and missing, re-read with tags.
     51     ds = pydicom.dcmread(path, **_DCM_READ_KW_FAST)
---> 52     data = ds.pixel_array.astype(np.float32, copy=False)
     53 
     54     slope = getattr(ds, "RescaleSlope", None)

/usr/local/lib/python3.11/dist-packages/pydicom/dataset.py in __getattr__(self, name)
    916             return {}
    917         # Try the base class attribute getter (fix for issue 332)
--> 918         return object.__getattribute__(self, name)
    919 
    920     @property

/usr/local/lib/python3.11/dist-packages/pydicom/dataset.py in pixel_array(self)
   2191             that iterates through the image frames.
   2192         """
-> 2193         self.convert_pixel_data()
   2194         return cast("numpy.ndarray", self._pixel_array)
   2195 

/usr/local/lib/python3.11/dist-packages/pydicom/dataset.py in convert_pixel_data(self, handler_name)
   1724             # Use 'pydicom.pixels' backend
   1725             opts["decoding_plugin"] = name
-> 1726             self._pixel_array = pixel_array(self, **opts)
   1727             self._pixel_id = get_image_pixel_ids(self)
   1728         else:

/usr/local/lib/python3.11/dist-packages/pydicom/pixels/utils.py in pixel_array(src, ds_out, specific_tags, index, raw, decoding_plugin, **kwargs)
   1428 
   1429         opts = as_pixel_options(ds, **kwargs)
-> 1430         return decoder.as_array(
   1431             ds,
   1432             index=index,

/usr/local/lib/python3.11/dist-packages/pydicom/pixels/decoders/base.py in as_array(self, src, index, validate, raw, decoding_plugin, **kwargs)
    988 
    989         if validate:
--> 990             runner.validate()
    991 
    992         if self.is_native:

/usr/local/lib/python3.11/dist-packages/pydicom/pixels/decoders/base.py in validate(self)
    751     def validate(self) -> None:
    752         """Validate the decoding options and source buffer (if any)."""
--> 753         self._validate_options()
    754         if self.is_dataset or self.is_buffer:
    755             self._validate_buffer()

/usr/local/lib/python3.11/dist-packages/pydicom/pixels/decoders/base.py in _validate_options(self)
    820     def _validate_options(self) -> None:
    821         """Validate the supplied options to ensure they meet minimum requirements."""
--> 822         super()._validate_options()
    823 
    824         # The Extended Offset Table is optional

/usr/local/lib/python3.11/dist-packages/pydicom/pixels/common.py in _validate_options(self)
    573         prefix = "Missing required element: (0028"
    574         if self._opts.get("bits_allocated") is None:
--> 575             raise AttributeError(f"{prefix},0100) 'Bits Allocated'")
    576 
    577         if not 1 <= self.bits_allocated <= 64 or (

AttributeError: Missing required element: (0028,0100) 'Bits Allocated'

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
options.experimental_optimization.apply_default_optimizations = True

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




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3269320991.py in <cell line: 0>()
     30 train_ds = (
     31     tf.data.TFRecordDataset(
---> 32         train_tfrec, compression_type="GZIP", num_parallel_reads=AUTO
     33     )
     34     .with_options(options)

NameError: name 'train_tfrec' is not defined

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




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2101019971.py in <cell line: 0>()
      1 EPOCHS = 3
----> 2 history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=1)
      3 
      4 

NameError: name 'train_ds' is not defined

## === cell 9
pred = model.predict(test_ds, verbose=1)
pred = pred.reshape(-1).astype(np.float32)

print("Pred shape:", pred.shape, "min/max:", float(pred.min()), float(pred.max()))




## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1509847922.py in <cell line: 0>()
----> 1 pred = model.predict(test_ds, verbose=1)
      2 pred = pred.reshape(-1).astype(np.float32)
      3 
      4 print("Pred shape:", pred.shape, "min/max:", float(pred.min()), float(pred.max()))
      5 

NameError: name 'test_ds' is not defined

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

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1605039584.py in <cell line: 0>()
      2 sub["BraTS21ID"] = sub["BraTS21ID"].astype(str).str.zfill(5)
      3 
----> 4 if len(pred) != len(sub):
      5     raise RuntimeError(
      6         f"Prediction length {len(pred)} does not match submission length {len(sub)}"

NameError: name 'pred' is not defined
