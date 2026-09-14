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
import os, math, glob, re
import numpy as np
import pandas as pd
import cv2

import matplotlib.pyplot as plt
import pydicom

from kaggle_datasets import KaggleDatasets

from sklearn.model_selection import train_test_split


from random import shuffle
import tensorflow as tf

## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
IMAGE_SIZE  = 256
IMAGE_DEPTH = 64

mri_types = ['FLAIR','T1w','T1wCE','T2w']
CHANNELS  = len(mri_types)

local_directory = "../input/rsna-miccai-brain-tumor-radiogenomic-classification"
local_label_path = local_directory+"/train_labels.csv"
local_submission_path = local_directory+"/sample_submission.csv"

## === cell 3
def load_dicom_slice(path, img_size=256):

    dicom = pydicom.read_file(path)
    data = dicom.pixel_array
    data = cv2.resize(data, (img_size, img_size))
    
    return data

def load_dicom_modality(mri_type, scan_id, img_depth, img_size, split):

    files = sorted(tf.io.gfile.glob(f"{local_directory}/{split}/{scan_id}/{mri_type}/*.dcm"), 
               key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])


    
    num_files = len(files)
    num_files_middle = num_files//2
    img_depth_middle = img_depth//2
    
    start_depth = max(0, num_files_middle - img_depth_middle)
    end_depth   = min(num_files, num_files_middle + img_depth_middle)
    img3d = np.stack([load_dicom_slice(dicom_img, img_size=img_size) 
                      for dicom_img in files[start_depth:end_depth]]).T
    
    if img3d.shape[-1] < img_depth:
        n_zero = np.zeros((img_size, img_size, img_depth - img3d.shape[-1]))
        img3d  = np.concatenate((img3d, n_zero), axis=-1)        

    return img3d

def load_dicom_3D(scan_id, img_depth=128, img_size=256, split="test"):
    print(scan_id, end=" ")
    dicom_channels = [load_dicom_modality(scan_id=scan_id,
                                         img_depth=img_depth, 
                                         img_size=img_size, 
                                         split=split,
                                         mri_type=mtype) 
                      for mtype in mri_types]
        
    img = np.stack(dicom_channels, axis=-1)
    if np.min(img) < np.max(img):
        img = img - np.min(img)
        img = img / np.max(img)
    return img

## === cell 5
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() 
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

## === cell 6
def serialize_example(image):
    feature = {
        'image': _bytes_feature(image.tobytes()),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

## === cell 7
def sorted_alphanumeric(data):  #
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(data, key=alphanum_key)



## === cell 8
! mkdir -p ./tfrecords/test/
outpath_train = "./tfrecords/test"
with tf.io.TFRecordWriter(str(outpath_train + os.sep + 'brain_test.tfrec'),
                          options=tf.io.TFRecordOptions(compression_type="GZIP")) as writer:
    Path = "../input/rsna-miccai-brain-tumor-radiogenomic-classification/test"
    train_dir = os.listdir(Path)
    train_dir = sorted_alphanumeric(train_dir)
    print(f"traindir : {train_dir}")
    
    for x in train_dir:
        img = load_dicom_3D(x, img_size=IMAGE_SIZE, img_depth=IMAGE_DEPTH, split="test")
        example = serialize_example(img)
        writer.write(example)

        


## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/1311820228.py in <cell line: 0>()
      9 
     10     for x in train_dir:
---> 11         img = load_dicom_3D(x, img_size=IMAGE_SIZE, img_depth=IMAGE_DEPTH, split="test")
     12         example = serialize_example(img)
     13         writer.write(example)

/tmp/ipykernel_11/1866149945.py in load_dicom_3D(scan_id, img_depth, img_size, split)
     34 def load_dicom_3D(scan_id, img_depth=128, img_size=256, split="test"):
     35     print(scan_id, end=" ")
---> 36     dicom_channels = [load_dicom_modality(scan_id=scan_id,
     37                                          img_depth=img_depth,
     38                                          img_size=img_size,

/tmp/ipykernel_11/1866149945.py in <listcomp>(.0)
     34 def load_dicom_3D(scan_id, img_depth=128, img_size=256, split="test"):
     35     print(scan_id, end=" ")
---> 36     dicom_channels = [load_dicom_modality(scan_id=scan_id,
     37                                          img_depth=img_depth,
     38                                          img_size=img_size,

/tmp/ipykernel_11/1866149945.py in load_dicom_modality(mri_type, scan_id, img_depth, img_size, split)
     22     start_depth = max(0, num_files_middle - img_depth_middle)
     23     end_depth   = min(num_files, num_files_middle + img_depth_middle)
---> 24     img3d = np.stack([load_dicom_slice(dicom_img, img_size=img_size) 
     25                       for dicom_img in files[start_depth:end_depth]]).T
     26 

/tmp/ipykernel_11/1866149945.py in <listcomp>(.0)
     22     start_depth = max(0, num_files_middle - img_depth_middle)
     23     end_depth   = min(num_files, num_files_middle + img_depth_middle)
---> 24     img3d = np.stack([load_dicom_slice(dicom_img, img_size=img_size) 
     25                       for dicom_img in files[start_depth:end_depth]]).T
     26 

/tmp/ipykernel_11/1866149945.py in load_dicom_slice(path, img_size)
      2 def load_dicom_slice(path, img_size=256):
      3 
----> 4     dicom = pydicom.read_file(path)
      5     data = dicom.pixel_array
      6     data = cv2.resize(data, (img_size, img_size))

AttributeError: module 'pydicom' has no attribute 'read_file'

## === cell 9
AUTO = tf.data.AUTOTUNE
def deserialize_example(serialized_string):
    image_feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
    }
    parsed_record = tf.io.parse_single_example(serialized_string, image_feature_description)
    image = tf.io.decode_raw(parsed_record['image'], tf.float64)

    image = tf.reshape(image,[256, 256, 64, 4])


    return image

BATCH_SIZE = 1 
train_set = tf.data.TFRecordDataset("./tfrecords/test/brain_test.tfrec",compression_type="GZIP", num_parallel_reads=AUTO).map(deserialize_example).batch(BATCH_SIZE).prefetch(AUTO)


## === cell 10
model = tf.keras.models.load_model('../input/test-2/Test2.h5',  custom_objects={'leaky_relu': tf.nn.leaky_relu})
pred = model.predict(train_set)

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/1129931563.py in <cell line: 0>()
----> 1 model = tf.keras.models.load_model('../input/test-2/Test2.h5',  custom_objects={'leaky_relu': tf.nn.leaky_relu})
      2 pred = model.predict(train_set)

/usr/local/lib/python3.11/dist-packages/keras/src/saving/saving_api.py in load_model(filepath, custom_objects, compile, safe_mode)
    194         )
    195     if str(filepath).endswith((".h5", ".hdf5")):
--> 196         return legacy_h5_format.load_model_from_hdf5(
    197             filepath, custom_objects=custom_objects, compile=compile
    198         )

/usr/local/lib/python3.11/dist-packages/keras/src/legacy/saving/legacy_h5_format.py in load_model_from_hdf5(filepath, custom_objects, compile)
    114     opened_new_file = not isinstance(filepath, h5py.File)
    115     if opened_new_file:
--> 116         f = h5py.File(filepath, mode="r")
    117     else:
    118         f = filepath

/usr/local/lib/python3.11/dist-packages/h5py/_hl/files.py in __init__(self, name, mode, driver, libver, userblock_size, swmr, rdcc_nslots, rdcc_nbytes, rdcc_w0, track_order, fs_strategy, fs_persist, fs_threshold, fs_page_size, page_buf_size, min_meta_keep, min_raw_keep, locking, alignment_threshold, alignment_interval, meta_block_size, **kwds)
    562                                  fs_persist=fs_persist, fs_threshold=fs_threshold,
    563                                  fs_page_size=fs_page_size)
--> 564                 fid = make_fid(name, mode, userblock_size, fapl, fcpl, swmr=swmr)
    565 
    566             if isinstance(libver, tuple):

/usr/local/lib/python3.11/dist-packages/h5py/_hl/files.py in make_fid(name, mode, userblock_size, fapl, fcpl, swmr)
    236         if swmr and swmr_support:
    237             flags |= h5f.ACC_SWMR_READ
--> 238         fid = h5f.open(name, flags, fapl=fapl)
    239     elif mode == 'r+':
    240         fid = h5f.open(name, h5f.ACC_RDWR, fapl=fapl)

h5py/_objects.pyx in h5py._objects.with_phil.wrapper()

h5py/_objects.pyx in h5py._objects.with_phil.wrapper()

h5py/h5f.pyx in h5py.h5f.open()

FileNotFoundError: [Errno 2] Unable to synchronously open file (unable to open file: name = '../input/test-2/Test2.h5', errno = 2, error message = 'No such file or directory', flags = 0, o_flags = 0)

## === cell 11
print(pred.shape)
print("dab")
sub = pd.read_csv("../input/rsna-miccai-brain-tumor-radiogenomic-classification/sample_submission.csv")
print(sub)
sub['MGMT_value'] = pred
sub.to_csv('submission.csv', index=False)
print(sub)

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4248412805.py in <cell line: 0>()
----> 1 print(pred.shape)
      2 print("dab")
      3 sub = pd.read_csv("../input/rsna-miccai-brain-tumor-radiogenomic-classification/sample_submission.csv")
      4 print(sub)
      5 sub['MGMT_value'] = pred

NameError: name 'pred' is not defined
