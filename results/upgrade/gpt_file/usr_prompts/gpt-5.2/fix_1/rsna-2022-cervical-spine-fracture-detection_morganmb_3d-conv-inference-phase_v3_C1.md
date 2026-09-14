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
Identify fractures in CT scans of the cervical spine (neck) at both the level of a single vertebrae and the entire patient.

## Metric
Weighted multi-label logarithmic loss. Each fracture sub-type is its own row for every exam, and you are expected to predict a probability for a fracture at each of the seven cervical vertebrae designated as C1, C2, C3, C4, C5, C6 and C7. There is also an any label, `patient_overall`, which indicates that a fracture of ANY kind described before exists in the examination. Fractures in the skull base, thoracic spine, ribs, and clavicles are ignored. The any label is weighted more highly than specific fracture level sub-types.

For each exam Id, you must submit a set of predicted probabilities (a separate row for each cervical level subtype). We then take the log loss for each predicted probability versus its true label.

The binary weighted log loss function for label j on exam i is specified as:

$$
L_{i j}=-w_j *\left[y_{i j} * \log \left(p_{i j}\right)+\left(1-y_{i j}\right) * \log \left(1-p_{i j}\right)\right]
$$

Finally, loss is averaged across all rows.

## Submission Format
There will be 8 rows per image Id. The label indicated by a particular row will look like [image Id]_[Sub-type Name], as follows. There is also a target column, `fractured`, indicating the probability of whether a fracture exists at the specified level. For each image ID in the test set, you must predict a probability for each of the different possible sub-types and the patient overall. The file should contain a header and have the following format:

```
row_id,fractured
1_C1,0
1_C2,0
1_C3,0
1_C4,0.6
1_C5,0
1_C6,0.9
1_C7,0.01
1_patient_overall,0.99
2_C1,0
etc.
```

## Dataset
**train.csv** Metadata for the train test set.

- `StudyInstanceUID` - The study ID. There is one unique study ID for each patient scan.
- `patient_overall` - One of the target columns. The patient level outcome, i.e. if any of the vertebrae are fractured.
- `C[1-7]` - The other target columns. Whether the given vertebrae is fractured. See [this diagram](https://en.wikipedia.org/wiki/Vertebral_column#/media/File:Gray_111_-_Vertebral_column-coloured.png) for the real location of each vertbrae in the spine.

**test.csv** Metadata for the test set prediction structure. Only the first few rows of the test set are available for download.

- `row_id` - The row ID. This will match the same column in the sample submission file.
- `StudyInstanceUID` - The study ID.
- `prediction_type` - Which one of the eight target columns needs a prediction in this row.

**[train/test]_images/[StudyInstanceUID]/[slice_number].dcm** The image data, organized with one folder per scan. Expect to see roughly 1,500 scans in the hidden test set.\

Each image is in [the dicom file format](https://www.dicomstandard.org/). The DICOM image files are ≤ 1 mm slice thickness, axial orientation, and bone kernel. Note that some of the DICOM files are JPEG compressed. You may require additional resources to read the pixel array of these files, such as GDCM and pylibjpeg.

**sample_submission.csv** A valid sample submission.

- `row_id` - The row ID. See the test.csv for what prediction needs to be filed in that row.
- `fractured` - The target column.

**train_bounding_boxes.csv** Bounding boxes for a subset of the training set.

**segmentations/** Pixel level annotations for a subset of the training set. This data is provided in the [nifti file format](https://nifti.nimh.nih.gov/).

A portion of the imaging datasets have been segmented automatically using a 3D UNET model, and radiologists modified and approved the segmentations. The provided segmentation labels have values of 1 to 7 for C1 to C7 (seven cervical vertebrae) and 8 to 19 for T1 to T12 (twelve thoracic vertebrae are located in the center of your upper and middle back), and 0 for everything else. As we focused on the cervical spine, all scans have C1 to C7 labels but not all thoracic labels.

Please be aware that the NIFTI files consist of segmentation in the sagittal plane, while the DICOM files are in the axial plane. Please use the NIFTI header information to determine the appropriate orientation such that the DICOM images and segmentation match. Otherwise, you run the risk of having the segmentations flipped in the Z axis and mirrored in the X axis.

# 2. Python version

3.10

# 3. Installed packages

geopandas==0.14.4
joblib==1.5.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
pydicom==3.0.1
scipy==1.15.3
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
            description.md (276 lines)
            sample_submission.csv (14537 lines)
            sample_submission.csv.zip (50.3 kB)
            segmentations.zip (3.0 MB)
            test.csv (14537 lines)
            test.csv.zip (94.5 kB)
            test.zip (160 Bytes)
            test_images.zip (181.9 GB)
            train.csv (203 lines)
            train.csv.zip (1.2 kB)
            train.zip (162 Bytes)
            train_bounding_boxes.csv (691 lines)
            train_bounding_boxes.csv.zip (11.5 kB)
            train_images.zip (20.3 GB)
            rsna-2022-cervical-spine-fracture-detection/
                description.md (276 lines)
                sample_submission.csv (14537 lines)
                ... and 12 other files
                rsna-2022-cervical-spine-fracture-detection/
                segmentations/
                    1.2.826.0.1.3680043.12292.nii (89.4 MB)
                    1.2.826.0.1.3680043.24617.nii (307.2 MB)
                    ... and 7 other files
                test_images/
                    1.2.826.0.1.3680043.10001/
                        1.dcm (525.0 kB)
                        10.dcm (525.0 kB)
                        ... and 266 other files
                    1.2.826.0.1.3680043.10005/
                        1.dcm (525.1 kB)
                        10.dcm (525.1 kB)
                        ... and 257 other files
                    ... and 1816 other folders
                train_images/
                    1.2.826.0.1.3680043.10014/
                        1.dcm (240.9 kB)
                        10.dcm (250.8 kB)
                        ... and 256 other files
                    1.2.826.0.1.3680043.10058/
                        1.dcm (525.0 kB)
                        10.dcm (525.0 kB)
                        ... and 574 other files
                    ... and 201 other folders
            segmentations/
                1.2.826.0.1.3680043.12292.nii (89.4 MB)
                1.2.826.0.1.3680043.24617.nii (307.2 MB)
                ... and 7 other files
            test_images/
                1.2.826.0.1.3680043.10001/
                    1.dcm (525.0 kB)
                    10.dcm (525.0 kB)
                    ... and 266 other files
                1.2.826.0.1.3680043.10005/
                    1.dcm (525.1 kB)
                    10.dcm (525.1 kB)
                    ... and 257 other files
                ... and 1816 other folders
            train_images/
                1.2.826.0.1.3680043.10014/
                    1.dcm (240.9 kB)
                    10.dcm (250.8 kB)
                    ... and 256 other files
                1.2.826.0.1.3680043.10058/
                    1.dcm (525.0 kB)
                    10.dcm (525.0 kB)
                    ... and 574 other files
                ... and 201 other folders
        input/
            description.md (276 lines)
            sample_submission.csv (14537 lines)
            sample_submission.csv.zip (50.3 kB)
            segmentations.zip (3.0 MB)
            test.csv (14537 lines)
            test.csv.zip (94.5 kB)
            test.zip (160 Bytes)
            test_images.zip (181.9 GB)
            train.csv (203 lines)
            train.csv.zip (1.2 kB)
            train.zip (162 Bytes)
            train_bounding_boxes.csv (691 lines)
            train_bounding_boxes.csv.zip (11.5 kB)
            train_images.zip (20.3 GB)
            rsna-2022-cervical-spine-fracture-detection/
                description.md (276 lines)
                sample_submission.csv (14537 lines)
                ... and 12 other files
                rsna-2022-cervical-spine-fracture-detection/
                segmentations/
                    1.2.826.0.1.3680043.12292.nii (89.4 MB)
                    1.2.826.0.1.3680043.24617.nii (307.2 MB)
                    ... and 7 other files
                test_images/
                    1.2.826.0.1.3680043.10001/
                        1.dcm (525.0 kB)
                        10.dcm (525.0 kB)
                        ... and 266 other files
                    1.2.826.0.1.3680043.10005/
                        1.dcm (525.1 kB)
                        10.dcm (525.1 kB)
                        ... and 257 other files
                    ... and 1816 other folders
                train_images/
                    1.2.826.0.1.3680043.10014/
                        1.dcm (240.9 kB)
                        10.dcm (250.8 kB)
                        ... and 256 other files
                    1.2.826.0.1.3680043.10058/
                        1.dcm (525.0 kB)
                        10.dcm (525.0 kB)
                        ... and 574 other files
                    ... and 201 other folders
            segmentations/
                1.2.826.0.1.3680043.12292.nii (89.4 MB)
                1.2.826.0.1.3680043.24617.nii (307.2 MB)
                ... and 7 other files
            test_images/
                1.2.826.0.1.3680043.10001/
                    1.dcm (525.0 kB)
                    10.dcm (525.0 kB)
                    ... and 266 other files
                1.2.826.0.1.3680043.10005/
                    1.dcm (525.1 kB)
                    10.dcm (525.1 kB)
                    ... and 257 other files
                ... and 1816 other folders
            train_images/
                1.2.826.0.1.3680043.10014/
                    1.dcm (240.9 kB)
                    10.dcm (250.8 kB)
                    ... and 256 other files
                1.2.826.0.1.3680043.10058/
                    1.dcm (525.0 kB)
                    10.dcm (525.0 kB)
                    ... and 574 other files
                ... and 201 other folders
        working/
            rsna-2022-cervical-spine-fracture-detection/
                description.md (276 lines)
                sample_submission.csv (14537 lines)
                ... and 12 other files
                rsna-2022-cervical-spine-fracture-detection/
                segmentations/
                    1.2.826.0.1.3680043.12292.nii (89.4 MB)
                    1.2.826.0.1.3680043.24617.nii (307.2 MB)
                    ... and 7 other files
                test_images/
                    1.2.826.0.1.3680043.10001/
                        1.dcm (525.0 kB)
                        10.dcm (525.0 kB)
                        ... and 266 other files
                    1.2.826.0.1.3680043.10005/
                        1.dcm (525.1 kB)
                        10.dcm (525.1 kB)
                        ... and 257 other files
                    ... and 1816 other folders
                train_images/
                    1.2.826.0.1.3680043.10014/
                        1.dcm (240.9 kB)
                        10.dcm (250.8 kB)
                        ... and 256 other files
                    1.2.826.0.1.3680043.10058/
                        1.dcm (525.0 kB)
                        10.dcm (525.0 kB)
                        ... and 574 other files
                    ... and 201 other folders
```

-> data/rsna-2022-cervical-spine-fracture-detection/sample_submission.csv has 14536 rows and 2 columns.
The columns are: row_id, fractured

-> data/rsna-2022-cervical-spine-fracture-detection/test.csv has 14536 rows and 3 columns.
The columns are: StudyInstanceUID, prediction_type, row_id

-> data/rsna-2022-cervical-spine-fracture-detection/train.csv has 202 rows and 9 columns.
The columns are: StudyInstanceUID, patient_overall, C1, C2, C3, C4, C5, C6, C7

-> data/rsna-2022-cervical-spine-fracture-detection/train_bounding_boxes.csv has 690 rows and 6 columns.
The columns are: StudyInstanceUID, x, y, width, height, slice_number

-> data/sample_submission.csv has 14536 rows and 2 columns.
The columns are: row_id, fractured

-> data/test.csv has 14536 rows and 3 columns.
The columns are: StudyInstanceUID, prediction_type, row_id

-> data/train.csv has 202 rows and 9 columns.
The columns are: StudyInstanceUID, patient_overall, C1, C2, C3, C4, C5, C6, C7

-> data/train_bounding_boxes.csv has 690 rows and 6 columns.
The columns are: StudyInstanceUID, x, y, width, height, slice_number

-> (stopped after 10 files for performance)

# 5. Target score

0.6523345712247187

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 2

import pkgutil
check_module = True if pkgutil.find_loader("hydra") else False
if not check_module:
    import subprocess    
    subprocess.run('python -m pip install --no-index --find-links=/kaggle/input/notebook-to-download-packages python-gdcm'.split())
    subprocess.run('python -m pip install --no-index --find-links=/kaggle/input/notebook-to-download-packages pylibjpeg'.split())
    subprocess.run('python -m pip install --no-index --find-links=/kaggle/input/notebook-to-download-packages pylibjpeg-libjpeg'.split())
    subprocess.run('python -m pip install --no-index --find-links=/kaggle/input/notebook-to-download-packages pydicom'.split())    
    subprocess.run('python -m pip install --no-index --find-links=/kaggle/input/notebook-to-download-packages livelossplot'.split())    
else:
    print("Environment is already setup")

del check_module

## === cell 3
import os
import glob
import gc

import numpy as np
import pandas as pd
from scipy import ndimage

from joblib import Parallel, delayed
from tqdm.notebook import tqdm

import pydicom as dicom
import gdcm

import tensorflow as tf
from tensorflow import keras
from livelossplot import PlotLossesKeras

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/2801503563.py in <cell line: 0>()
     11 
     12 import pydicom as dicom
---> 13 import gdcm
     14 
     15 import tensorflow as tf

ModuleNotFoundError: No module named 'gdcm'

## === cell 5

TRAIN_IMAGES_PATH = '/kaggle/input/rsna-2022-cervical-spine-fracture-detection/train_images/'
TEST_IMAGES_PATH = '/kaggle/input/rsna-2022-cervical-spine-fracture-detection/test_images/'

TRAIN_CSV_PATH = '/kaggle/input/rsna-2022-cervical-spine-fracture-detection/train.csv'
TEST_CSV_PATH = '/kaggle/input/rsna-2022-cervical-spine-fracture-detection/test.csv'

TEST_OUTPUT_PATH = './test_arrays/'
if not os.path.exists(TEST_OUTPUT_PATH): os.mkdir(TEST_OUTPUT_PATH)  
    
MODEL_PATH = "/kaggle/input/3d-conv-training-phase"

## === cell 6


desired_width = 64
desired_height = 64
desired_depth = 64


train_images = os.listdir(TRAIN_IMAGES_PATH)
test_images = os.listdir(TEST_IMAGES_PATH)


train_df = pd.read_csv(TRAIN_CSV_PATH)
test_df = pd.read_csv(TEST_CSV_PATH)


## === cell 8

def load_dicom(path: str):
    """Load a dicom file (.dcm) even if it is compressed."""
    try:
        file_as_array = dicom.dcmread(path).pixel_array
    except:
        decompressed_file = gdcm.ImageReader().SetFileName(path).Read()
        file_as_array = decompressed_file.pixel_array
    return(file_as_array)

## === cell 9

def preprocessing_slice(slice_path: str):
    """Load dicom of a slice, normalize and resize it"""
    slice_array = load_dicom(slice_path)
    slice_array = slice_array.astype(np.uint8)
    
    slice_array = slice_array - np.min(slice_array)
    if np.max(slice_array) != 0:
        slice_array = slice_array / np.max(slice_array)
    slice_array = (slice_array * 255).astype(np.uint8)
        
    width_factor = desired_width / slice_array.shape[0]
    height_factor = desired_height / slice_array.shape[1]
    
    slice_array = ndimage.zoom(slice_array, (width_factor, height_factor), order=3) # resize with spline interpolation of order 3
    
    return(slice_array)

## === cell 10
def resize_depth(numpy_volume: np.array, desired_depth=desired_depth):
    """Resize across z-axis"""
    current_depth = numpy_volume.shape[0]
    depth_factor = desired_depth / current_depth
    
    numpy_volume = ndimage.rotate(numpy_volume, 90, reshape=False)
    volume = ndimage.zoom(numpy_volume, (depth_factor, 1, 1), order=1) # resize with spline interpolation of order 1
    return volume

## === cell 11

def load_and_stack_dicom_parallel(scan_path: str):
    """Load all dicom files from a scan and stack them all in a numpy array"""
    slice_paths = sorted(glob.glob(os.path.join(scan_path, "*")),
                         key=lambda x: int(x.split('/')[-1].split(".")[0]))
    
    images = Parallel(n_jobs=-1)(delayed(preprocessing_slice)(filename) for filename in slice_paths)
    
    return(tf.expand_dims(resize_depth(np.array(images)), axis=3))

## === cell 12

def save_3D_arrays(scan_path: str, output_path: str):
    """Create and save the 3D arrays corresponding to a scan"""
    
    volume = load_and_stack_dicom_parallel(scan_path=scan_path)
    
    volume_file_name = output_path + scan_path.split('/')[-1] + '.npy'
    np.save(volume_file_name, volume)
    
    del volume
    
    return None

## === cell 13

for i in tqdm(range(len(test_images))):
    case_path = TEST_IMAGES_PATH + test_images[i]
    save_3D_arrays(case_path, TEST_OUTPUT_PATH)



gc.collect()

## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
_RemoteTraceback                          Traceback (most recent call last)
_RemoteTraceback: 
"""
Traceback (most recent call last):
  File "/tmp/ipykernel_11/3013495376.py", line 6, in load_dicom
  File "/usr/local/lib/python3.11/dist-packages/pydicom/dataset.py", line 2193, in pixel_array
    self.convert_pixel_data()
  File "/usr/local/lib/python3.11/dist-packages/pydicom/dataset.py", line 1726, in convert_pixel_data
    self._pixel_array = pixel_array(self, **opts)
                        ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/pydicom/pixels/utils.py", line 1430, in pixel_array
    return decoder.as_array(
           ^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/pydicom/pixels/decoders/base.py", line 982, in as_array
    self._validate_plugins(decoding_plugin),
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/pydicom/pixels/common.py", line 257, in _validate_plugins
    raise RuntimeError(
RuntimeError: Unable to decompress 'JPEG Lossless, Non-Hierarchical, First-Order Prediction (Process 14 [Selection Value 1])' pixel data because all plugins are missing dependencies:
	gdcm - requires gdcm>=3.0.10
	pylibjpeg - requires pylibjpeg>=2.0 and pylibjpeg-libjpeg>=2.1

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/joblib/externals/loky/process_executor.py", line 490, in _process_worker
    r = call_item()
        ^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/joblib/externals/loky/process_executor.py", line 291, in __call__
    return self.fn(*self.args, **self.kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/joblib/parallel.py", line 607, in __call__
    return [func(*args, **kwargs) for func, args, kwargs in self.items]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/joblib/parallel.py", line 607, in <listcomp>
    return [func(*args, **kwargs) for func, args, kwargs in self.items]
            ^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/ipykernel_11/1345018422.py", line 6, in preprocessing_slice
  File "/tmp/ipykernel_11/3013495376.py", line 8, in load_dicom
NameError: name 'gdcm' is not defined
"""

The above exception was the direct cause of the following exception:

NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/513348083.py in <cell line: 0>()
      3 for i in tqdm(range(len(test_images))):
      4     case_path = TEST_IMAGES_PATH + test_images[i]
----> 5     save_3D_arrays(case_path, TEST_OUTPUT_PATH)
      6 
      7 

/tmp/ipykernel_11/1257537960.py in save_3D_arrays(scan_path, output_path)
      5 
      6     # Preprocessing and creation :
----> 7     volume = load_and_stack_dicom_parallel(scan_path=scan_path)
      8 
      9     # Saving the numpy array :

/tmp/ipykernel_11/914074542.py in load_and_stack_dicom_parallel(scan_path)
      8 
      9     # Preprocessing slices :
---> 10     images = Parallel(n_jobs=-1)(delayed(preprocessing_slice)(filename) for filename in slice_paths)
     11 
     12     # Returning stacked slices as a resized on depth (3rd dimension) volume :

/usr/local/lib/python3.11/dist-packages/joblib/parallel.py in __call__(self, iterable)
   2070         next(output)
   2071 
-> 2072         return output if self.return_generator else list(output)
   2073 
   2074     def __repr__(self):

/usr/local/lib/python3.11/dist-packages/joblib/parallel.py in _get_outputs(self, iterator, pre_dispatch)
   1680 
   1681             with self._backend.retrieval_context():
-> 1682                 yield from self._retrieve()
   1683 
   1684         except GeneratorExit:

/usr/local/lib/python3.11/dist-packages/joblib/parallel.py in _retrieve(self)
   1782             # worker traceback.
   1783             if self._aborting:
-> 1784                 self._raise_error_fast()
   1785                 break
   1786 

/usr/local/lib/python3.11/dist-packages/joblib/parallel.py in _raise_error_fast(self)
   1857         # called directly or if the generator is gc'ed.
   1858         if error_job is not None:
-> 1859             error_job.get_result(self.timeout)
   1860 
   1861     def _warn_exit_early(self):

/usr/local/lib/python3.11/dist-packages/joblib/parallel.py in get_result(self, timeout)
    756             # callback thread, and is stored internally. It's just waiting to
    757             # be returned.
--> 758             return self._return_or_raise()
    759 
    760         # For other backends, the main thread needs to run the retrieval step.

/usr/local/lib/python3.11/dist-packages/joblib/parallel.py in _return_or_raise(self)
    771         try:
    772             if self.status == TASK_ERROR:
--> 773                 raise self._result
    774             return self._result
    775         finally:

NameError: name 'gdcm' is not defined

## === cell 15

model = keras.models.load_model(os.path.join(MODEL_PATH,"InceptionV3-b-64x64x64"))

## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/530463961.py in <cell line: 0>()
      1 # Loading the trained model :
      2 
----> 3 model = keras.models.load_model(os.path.join(MODEL_PATH,"InceptionV3-b-64x64x64"))

NameError: name 'keras' is not defined

## === cell 16

batch_x = np.empty((len(test_images), 64, 64, 64, 1))
for i_slice in range(0, len(test_images)):
    batch_x[i_slice,] = np.load(os.path.join(TEST_OUTPUT_PATH, os.listdir(TEST_OUTPUT_PATH)[i_slice]))

predictions = model.predict(batch_x)
print(predictions)

## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
/tmp/ipykernel_11/2285743369.py in <cell line: 0>()
      4 batch_x = np.empty((len(test_images), 64, 64, 64, 1))
      5 for i_slice in range(0, len(test_images)):
----> 6     batch_x[i_slice,] = np.load(os.path.join(TEST_OUTPUT_PATH, os.listdir(TEST_OUTPUT_PATH)[i_slice]))
      7 
      8 predictions = model.predict(batch_x)

IndexError: list index out of range

## === cell 18

submission = predictions.copy()

for sequence_index in range(submission.shape[0]):
    submission[sequence_index] = np.concatenate((submission[sequence_index][1:], [submission[sequence_index][0]]))
    
submission

## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1714008359.py in <cell line: 0>()
      1 # Putting predictions outputs in the right order for the submission file ('patient_overall' at the end of each sequence) :
      2 
----> 3 submission = predictions.copy()
      4 
      5 for sequence_index in range(submission.shape[0]):

NameError: name 'predictions' is not defined

## === cell 19


fractured_col_values = np.concatenate(submission.copy())
    

row_id_values = []
for ID in test_images:
    row_id_values.append(ID+'_C1')
    row_id_values.append(ID+'_C2')
    row_id_values.append(ID+'_C3')
    row_id_values.append(ID+'_C4')
    row_id_values.append(ID+'_C5')
    row_id_values.append(ID+'_C6')
    row_id_values.append(ID+'_C7')
    row_id_values.append(ID+'_patient_overall')
    

submissions_df = pd.DataFrame({"row_id": row_id_values, "fractured": fractured_col_values})

submissions_df

## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1484777252.py in <cell line: 0>()
      3 # Putting every predictions output as a single vector (corresponding to the column 'fractured' in the dataframe) :
      4 
----> 5 fractured_col_values = np.concatenate(submission.copy())
      6 
      7 # Creating the 'row_id' column values :

NameError: name 'submission' is not defined

## === cell 20

submissions_df.to_csv("submission.csv", index=False)

## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/383119429.py in <cell line: 0>()
      1 # Uploading submission csv file :
      2 
----> 3 submissions_df.to_csv("submission.csv", index=False)

NameError: name 'submissions_df' is not defined
