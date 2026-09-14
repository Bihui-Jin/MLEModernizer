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
Detect breast cancer in mammograms.

## Metric
[Probabilistic F1 score](https://aclanthology.org/2020.eval4nlp-1.9.pdf) (pF1). This extension of the traditional F score accepts probabilities instead of binary classifications. 

With pX as the probabilistic version of X:

$$
pF_1 = 2 \frac{pPrecision \cdot pRecall}{pPrecision + pRecall}
$$

where:

$$
pPrecision = \frac{pTP}{pTP + pFP}
$$

$$
pRecall = \frac{pTP}{TP + FN}
$$

## Submission Format
For each `prediction_id`, you should predict the likelihood of cancer in the corresponding `cancer` column. The submission file should have the following format:

```
prediction_id,cancer
0-L,0
0-R,0.5
0-R,0.5
1-L,1
...
# Dataset

**[train/test]_images/[patient_id]/[image_id].dcm** The mammograms, in dicom format. You can expect roughly 8,000 patients in the hidden test set. There are usually but not always 4 images per patient. Note that many of the images use the jpeg 2000 format which may you may need special libraries to load.

**sample_submission.csv** A valid sample submission.

**[train/test].csv** Metadata for each patient and image. Only the first few rows of the test set are available for download.

- `site_id` - ID code for the source hospital.
- `patient_id` - ID code for the patient.
- `image_id` - ID code for the image.
- `laterality` - Whether the image is of the left or right breast.
- `view` - The orientation of the image. The default for a screening exam is to capture two views per breast.
- `age` - The patient's age in years.
- `implant` - Whether or not the patient had breast implants. Site 1 only provides breast implant information at the patient level, not at the breast level.
- `density` - A rating for how dense the breast tissue is, with A being the least dense and D being the most dense. Extremely dense tissue can make diagnosis more difficult. Only provided for train.
- `machine_id` - An ID code for the imaging device.
- `cancer` - Whether or not the breast was positive for malignant cancer. The target value. Only provided for train.
- `biopsy` - Whether or not a follow-up biopsy was performed on the breast. Only provided for train.
- `invasive` - If the breast is positive for cancer, whether or not the cancer proved to be invasive. Only provided for train.
- `BIRADS` - 0 if the breast required follow-up, 1 if the breast was rated as negative for cancer, and 2 if the breast was rated as normal. Only provided for train.
- `prediction_id` - The ID for the matching submission row. Multiple images will share the same prediction ID. Test only.
- `difficult_negative_case` - True if the case was unusually difficult. Only provided for train.

# 2. Python version

3.12

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (190 lines)
            sample_submission.csv (2385 lines)
            sample_submission.csv.zip (6.5 kB)
            test.csv (5475 lines)
            test.csv.zip (60.6 kB)
            test.zip (160 Bytes)
            test_images.zip (29.1 GB)
            train.csv (49233 lines)
            train.csv.zip (513.7 kB)
            train.zip (162 Bytes)
            train_images.zip (260.7 GB)
            rsna-breast-cancer-detection/
                description.md (190 lines)
                sample_submission.csv (2385 lines)
                ... and 9 other files
                rsna-breast-cancer-detection/
                test_images/
                    10116/
                        1470873094.dcm (8.5 MB)
                        472095321.dcm (5.6 MB)
                        ... and 2 other files
                    10130/
                        1013166704.dcm (9.7 MB)
                        1165309236.dcm (8.7 MB)
                        ... and 5 other files
                    ... and 1191 other folders
                train_images/
                    10006/
                        1459541791.dcm (4.4 MB)
                        1864590858.dcm (4.0 MB)
                        ... and 2 other files
                    10011/
                        1031443799.dcm (2.1 MB)
                        220375232.dcm (1.7 MB)
                        ... and 2 other files
                    ... and 10720 other folders
            test_images/
                10116/
                    1470873094.dcm (8.5 MB)
                    472095321.dcm (5.6 MB)
                    ... and 2 other files
                10130/
                    1013166704.dcm (9.7 MB)
                    1165309236.dcm (8.7 MB)
                    ... and 5 other files
                ... and 1191 other folders
            train_images/
                10006/
                    1459541791.dcm (4.4 MB)
                    1864590858.dcm (4.0 MB)
                    ... and 2 other files
                10011/
                    1031443799.dcm (2.1 MB)
                    220375232.dcm (1.7 MB)
                    ... and 2 other files
                ... and 10720 other folders
        input/
            description.md (190 lines)
            sample_submission.csv (2385 lines)
            sample_submission.csv.zip (6.5 kB)
            test.csv (5475 lines)
            test.csv.zip (60.6 kB)
            test.zip (160 Bytes)
            test_images.zip (29.1 GB)
            train.csv (49233 lines)
            train.csv.zip (513.7 kB)
            train.zip (162 Bytes)
            train_images.zip (260.7 GB)
            rsna-breast-cancer-detection/
                description.md (190 lines)
                sample_submission.csv (2385 lines)
                ... and 9 other files
                rsna-breast-cancer-detection/
                test_images/
                    10116/
                        1470873094.dcm (8.5 MB)
                        472095321.dcm (5.6 MB)
                        ... and 2 other files
                    10130/
                        1013166704.dcm (9.7 MB)
                        1165309236.dcm (8.7 MB)
                        ... and 5 other files
                    ... and 1191 other folders
                train_images/
                    10006/
                        1459541791.dcm (4.4 MB)
                        1864590858.dcm (4.0 MB)
                        ... and 2 other files
                    10011/
                        1031443799.dcm (2.1 MB)
                        220375232.dcm (1.7 MB)
                        ... and 2 other files
                    ... and 10720 other folders
            test_images/
                10116/
                    1470873094.dcm (8.5 MB)
                    472095321.dcm (5.6 MB)
                    ... and 2 other files
                10130/
                    1013166704.dcm (9.7 MB)
                    1165309236.dcm (8.7 MB)
                    ... and 5 other files
                ... and 1191 other folders
            train_images/
                10006/
                    1459541791.dcm (4.4 MB)
                    1864590858.dcm (4.0 MB)
                    ... and 2 other files
                10011/
                    1031443799.dcm (2.1 MB)
                    220375232.dcm (1.7 MB)
                    ... and 2 other files
                ... and 10720 other folders
        working/
            rsna-breast-cancer-detection/
                description.md (190 lines)
                sample_submission.csv (2385 lines)
                ... and 9 other files
                rsna-breast-cancer-detection/
                test_images/
                    10116/
                        1470873094.dcm (8.5 MB)
                        472095321.dcm (5.6 MB)
                        ... and 2 other files
                    10130/
                        1013166704.dcm (9.7 MB)
                        1165309236.dcm (8.7 MB)
                        ... and 5 other files
                    ... and 1191 other folders
                train_images/
                    10006/
                        1459541791.dcm (4.4 MB)
                        1864590858.dcm (4.0 MB)
                        ... and 2 other files
                    10011/
                        1031443799.dcm (2.1 MB)
                        220375232.dcm (1.7 MB)
                        ... and 2 other files
                    ... and 10720 other folders
```

-> data/rsna-breast-cancer-detection/sample_submission.csv has 2384 rows and 2 columns.
The columns are: prediction_id, cancer

-> data/rsna-breast-cancer-detection/test.csv has 5474 rows and 9 columns.
The columns are: site_id, patient_id, image_id, laterality, view, age, implant, machine_id, prediction_id

-> data/rsna-breast-cancer-detection/train.csv has 49232 rows and 14 columns.
The columns are: site_id, patient_id, image_id, laterality, view, age, cancer, biopsy, invasive, BIRADS, implant, density, machine_id, difficult_negative_case

-> data/sample_submission.csv has 2384 rows and 2 columns.
The columns are: prediction_id, cancer

-> data/test.csv has 5474 rows and 9 columns.
The columns are: site_id, patient_id, image_id, laterality, view, age, implant, machine_id, prediction_id

-> data/train.csv has 49232 rows and 14 columns.
The columns are: site_id, patient_id, image_id, laterality, view, age, cancer, biopsy, invasive, BIRADS, implant, density, machine_id, difficult_negative_case

-> (stopped after 10 files for performance)

# 5. Target score

0.139917695473251

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 4
!ls -d -1 "/kaggle/input/rsna-mammography-breast-cancer-tensorflow-model"/*

## === cell 5
%%writefile requirements.txt
/kaggle/input/rsna-mammography-breast-cancer-tensorflow-model/dicomsdl-0.109.2-cp310-cp310-manylinux_2_12_x86_64.manylinux2010_x86_64.whl
/kaggle/input/rsna-mammography-breast-cancer-tensorflow-model/keras_cv_attention_models-1.3.22-py3-none-any.whl
/kaggle/input/rsna-mammography-breast-cancer-tensorflow-model/pylibjpeg-1.4.0-py3-none-any.whl
/kaggle/input/rsna-mammography-breast-cancer-tensorflow-model/pylibjpeg_libjpeg-1.3.4-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
/kaggle/input/rsna-mammography-breast-cancer-tensorflow-model/python_gdcm-3.0.22-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

## === cell 6
import os, sys, platform

!{sys.executable} -m pip install -r requirements.txt --no-cache-dir -Uq --no-index --no-deps
print("Platform:", platform.system())  # platform.platform()
print("Python  :", platform.python_version())  # sys.version
print("Actv Env:", os.getenv('CONDA_DEFAULT_ENV', 'Not Found Conda Env'))

## === cell 8
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
print("Tensorflow version \t\t:", tf.__version__)

tf.config.threading.set_inter_op_parallelism_threads(num_threads=1)
print("Number of threads \t\t:", tf.config.threading.get_inter_op_parallelism_threads())

print("Available devices:")
for i, device in enumerate(tf.config.list_logical_devices()):
    print("%d) %s" % (i, device))

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 9
try:
    import tensorflow as tf
    cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(cluster_resolver)
    tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
    strategy = tf.distribute.TPUStrategy(cluster_resolver)
    print('Running on TPU ', cluster_resolver.master(), len(tf.config.list_logical_devices('TPU')))
except ValueError:
    gpus = tf.config.list_logical_devices('GPU')
    if len(gpus) > 1:
        strategy = tf.distribute.MirroredStrategy([gpu.name for gpu in gpus])
        print('Running on multiple GPUs ', gpus)
    elif len(gpus) == 1:
        strategy = tf.distribute.get_strategy()
        print('Running on single GPU ', gpus[0].name)
    else:
        strategy = tf.distribute.get_strategy()
        print('Running on CPU')
finally:
    print("Number of accelerators: ", strategy.num_replicas_in_sync)



## === cell 10
tf.config.optimizer.set_jit(True)
tf.config.optimizer.get_jit()

## === cell 12
import cv2; print('cv2', cv2.__version__), cv2.setNumThreads(1)
print('cv2 cuda:', cv2.cuda.getCudaEnabledDeviceCount())

import pylibjpeg; print("pylibjpeg", pylibjpeg.__version__)
import libjpeg; print("libjpeg", libjpeg.__version__)
import dicomsdl; print("dicomsdl", dicomsdl.DICOMSDL_VERSION)
import pydicom; print("pydicom", pydicom.__version__);
from pydicom.pixel_data_handlers.util import apply_voi_lut, apply_voi
from pydicom.pixel_data_handlers.util import apply_color_lut, apply_modality_lut
import gdcm; print("gdcm", gdcm.GDCM_VERSION);

## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/737708500.py in <cell line: 0>()
      4 print('cv2 cuda:', cv2.cuda.getCudaEnabledDeviceCount())
      5 
----> 6 import pylibjpeg; print("pylibjpeg", pylibjpeg.__version__)
      7 import libjpeg; print("libjpeg", libjpeg.__version__)
      8 import dicomsdl; print("dicomsdl", dicomsdl.DICOMSDL_VERSION)

ModuleNotFoundError: No module named 'pylibjpeg'

## === cell 13
from kaggle_datasets import KaggleDatasets
from keras_cv_attention_models import convnext
import kecam; print(f'keras_cv_attention_models: {kecam.__version__}')

## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/3899279624.py in <cell line: 0>()
      1 from kaggle_datasets import KaggleDatasets
----> 2 from keras_cv_attention_models import convnext
      3 import kecam; print(f'keras_cv_attention_models: {kecam.__version__}')

ModuleNotFoundError: No module named 'keras_cv_attention_models'

## === cell 14
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import scikitplot as skplt

import re
import time
import random
import datetime
import tempfile
import importlib
from glob import glob
from typing import cast
from pathlib import Path
from tqdm.auto import tqdm
from multiprocessing import cpu_count
import joblib
import pickle

import gc
gc.collect()

## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/3828974855.py in <cell line: 0>()
      4 import matplotlib.pyplot as plt
      5 # !pip install scikit-plot -Uq
----> 6 import scikitplot as skplt
      7 
      8 import re

/usr/local/lib/python3.11/dist-packages/scikitplot/__init__.py in <module>
      1 from __future__ import absolute_import, division, print_function, unicode_literals
----> 2 from . import metrics, cluster, decomposition, estimators
      3 __version__ = '0.3.7'
      4 
      5 

/usr/local/lib/python3.11/dist-packages/scikitplot/metrics.py in <module>
     25 from sklearn.utils import deprecated
     26 
---> 27 from scipy import interp
     28 
     29 from scikitplot.helpers import binary_ks_curve, validate_labels

ImportError: cannot import name 'interp' from 'scipy' (/usr/local/lib/python3.11/dist-packages/scipy/__init__.py)

## === cell 16
IMAGE_FORMAT  = 'JPG'
IMAGE_QUALITY = 100
TARGET_HEIGHT, TARGET_WIDTH, N_CHANNELS = ( 624, 512, 1 )
INPUT_SHAPE   = (TARGET_HEIGHT, TARGET_WIDTH, N_CHANNELS)

THRESHOLD_BEST = 0.857292

## === cell 17
DATA_DIR  = "/kaggle/input/rsna-breast-cancer-detection"
print('DATA_DIR :', DATA_DIR)

## === cell 20
test_df              = pd.read_csv(f'{DATA_DIR}/test.csv')
sample_submission_df = pd.read_csv(f'{DATA_DIR}/sample_submission.csv')

test_df.shape, sample_submission_df.shape

## === cell 21
display(test_df.head(), sample_submission_df.head())

## === cell 22
test_df.info(), print('\n'), sample_submission_df.info(), 

## === cell 24
test_file_paths = tf.io.gfile.glob(f'{DATA_DIR}/test_images/*/*.dcm')

print(f'Test size images : {len(test_file_paths):<10}')

## === cell 26
def apply_voi_lut_dicomsdl(image_data, ds, isDicomsdl=True):
    meta_data = ds.getPixelDataInfo().keys() if isDicomsdl else ds
        
    if all(i in meta_data for i in ['WindowCenter', 'WindowWidth',
                                    'BitsStored', 'RescaleSlope', 'RescaleIntercept',
                                    'PixelRepresentation']):
        center = np.array(ds.WindowCenter, dtype=np.float64).flatten()[0]
        width  = np.array(ds.WindowWidth, dtype=np.float64).flatten()[0]
        
        y_min, y_max = 0.0, float(2**ds.BitsStored - 1)
        slope     = ds.RescaleSlope
        intercept = ds.RescaleIntercept
        if slope is not None and intercept is not None:
            y_min = y_min * float(slope) + float(intercept)
            y_max = y_max * float(slope) + float(intercept)
        y_range = y_max - y_min
        
        try:
            voi_func = ds.VOILUTFunction
        except:
            voi_func = None
        finally:
            if voi_func is None: voi_func = 'LINEAR'

        arr = image_data.astype('float64')
        
        if ds.PhotometricInterpretation in ['MONOCHROME1', 'MONOCHROME2']:
            if voi_func in ['LINEAR', 'LINEAR_EXACT']:
                if voi_func == 'LINEAR':
                    if width < 1:
                        raise ValueError(
                            "The (0028,1051) Window Width must be greater than or "
                            "equal to 1 for a 'LINEAR' windowing operation"
                        )
                    center -= 0.5
                    width -= 1
                elif width <= 0:
                    raise ValueError(
                        "The (0028,1051) Window Width must be greater than 0 "
                        "for a 'LINEAR_EXACT' windowing operation"
                    )

                below = arr <= (center - width / 2)
                above = arr > (center + width / 2)
                between = np.logical_and(~below, ~above)

                arr[below] = y_min
                arr[above] = y_max
                if between.any():
                    arr[between] = (
                        ((arr[between] - center) / width + 0.5) * y_range + y_min
                    )
            elif voi_func == 'SIGMOID':
                if width <= 0:
                    raise ValueError(
                        "The (0028,1051) Window Width must be greater than 0 "
                        "for a 'SIGMOID' windowing operation"
                    )

                arr = y_range / (1 + np.exp(-4 * (arr - center) / width)) + y_min
            else:
                raise ValueError(
                    f"Unsupported (0028,1056) VOI LUT Function value '{voi_func}'"
                )
        
        if ds.PhotometricInterpretation == 'MONOCHROME1':
            arr = np.amax(arr) - arr
    return arr.astype('uint16')

## === cell 27
def analyze_components(img_data_voi, ds, filtering=False, threshold=cv2.THRESH_BINARY, debug=False):
    img_data_8u = cv2.normalize(img_data_voi, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    if filtering:
        blur = cv2.GaussianBlur(src=img_data_8u, ksize=(5,5), sigmaX=0)
    else:
        blur = img_data_8u
    
    if threshold in range(255):
        retval, black_background_mask = cv2.threshold(
            src=blur,       # Input image (should be grayscale)
            thresh=25,      # Threshold value
            maxval=255,     # Value assigned to pixels exceeding the threshold
            type=threshold,  # Binary thresholding type
        )
    else:
        black_background_mask = (blur > 25).astype(np.uint8)
    
    output = cv2.connectedComponentsWithStats(image=black_background_mask, connectivity=8, ltype=cv2.CV_32S)
    
    retval, labels, stats, centroids = output
    
    if retval > 1:
        largest_component_index = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
        x, y, width, height, area = stats[largest_component_index]
        img_data_roi = img_data_8u[y:y+height, x:x+width]
    else:
        img_data_roi = img_data_8u
    
    if debug:
        plt.subplot(121)
        plt.imshow(black_background_mask);
        plt.title(f'image mask\nthreshold: {threshold}\nfiltering {filtering}')
        
        plt.subplot(122)
        plt.imshow(img_data_roi);
        plt.colorbar()
        plt.title(f'cropped image\nshape: {img_data_roi.shape}')
        
        plt.tight_layout()
        plt.show();        
        for label in range(1, retval)[: 2]:
            area     = stats[label, cv2.CC_STAT_AREA]
            centroid = centroids[label]
            x, y, width, height, _ = stats[label]
            print(f"""Component {label}: 
            Area={area}, Centroid=({centroid[0]:.2f}, {centroid[1]:.2f}), 
            Bounding Box=({x}, {y}, {width}, {height})
            """)
    
    return img_data_roi

## === cell 28
def image_findNonZero(img_data_voi):
    img_data_8u = cv2.normalize(img_data_voi, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    coords = cv2.findNonZero(img_data_8u)
    x, y, width, height = cv2.boundingRect(coords)
    
    img_data_roi = img_data_8u[y:y+height, x:x+width]
    return img_data_roi

## === cell 30
def load_and_preprocess_image(file_path, scale_image=False, save_image=True, debug=False):    
    img_dicom    = dicomsdl.open(file_path)
    img_data_16u = img_dicom.pixelData(storedvalue=True)    
    img_data_voi = apply_voi_lut_dicomsdl(img_data_16u, img_dicom, isDicomsdl=True)    
    img_data_voi_roi = analyze_components(img_data_voi, img_dicom)
    
    img_data_resized = cv2.resize(
        src   = img_data_voi_roi,
        dsize = INPUT_SHAPE[:2][::-1], # attention cv2 size reverse
        interpolation = cv2.INTER_LANCZOS4,
    )
    img_data_resized = np.expand_dims(img_data_resized, 2)
    
    if save_image:
        patient_id, image_id = re.split('[\./]', file_path)[-3:-1]
        if not os.path.exists(patient_id):
            os.makedirs(patient_id, exist_ok=True)
            
        if IMAGE_FORMAT == 'PNG':
            cv2.imwrite(f'{patient_id}/{image_id}.png', img_data_resized)
        else:
            cv2.imwrite(f'{patient_id}/{image_id}.jpg', img_data_resized, [cv2.IMWRITE_JPEG_QUALITY, IMAGE_QUALITY])
            
    if debug:
        plt.imshow(img_data_resized, cmap='bone')

## === cell 34
MODEL_DIR = "/kaggle/input/rsna-mammography-breast-cancer-tensorflow-model/rsna_cancer_convnext_v2_tiny_model_legacy.h5"
print('MODEL_DIR:', MODEL_DIR)

## === cell 35
tf.keras.backend.clear_session()
gc.collect()

with strategy.scope():
    model = tf.keras.models.load_model(MODEL_DIR, compile=False)    
    model.trainable = False
    model.compile()    
    model.summary()

## --- ERROR in cell 35, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2222416526.py in <cell line: 0>()
      1 tf.keras.backend.clear_session()
----> 2 gc.collect()
      3 
      4 with strategy.scope():
      5     # Load the model (architecture + weights)

NameError: name 'gc' is not defined

## === cell 39
def preprocess_and_save_image(args):
    (patient_id, laterality), group = args    
    for row_idx, row_data in group.iterrows():
        image_path = f"{DATA_DIR}/test_images/{row_data['patient_id']}/{row_data['image_id']}.dcm"
        load_and_preprocess_image(image_path)  
        
        
def convert_images_dcm2jpg():   
    jobs = [joblib.delayed(preprocess_and_save_image)(args) for args in test_df.groupby(['patient_id', 'laterality'])]
    SUBMISSION_ROWS = joblib.Parallel(
        n_jobs = -1,                 # -1, cpu_count()
        verbose= 0,
        backend= 'multiprocessing',  # 'multiprocessing', 'threading'
        prefer = 'threads',          # processes
    )(jobs)
    if np.random.rand() > 0.95:
        gc.collect()

## === cell 40
convert_images_dcm2jpg()

## --- ERROR in cell 40, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1318358515.py in <cell line: 0>()
      1 # 1.92 s ± 58.9 ms per loop (mean ± std. dev. of 7 runs, 2 loops each)
      2 # %timeit -r 7 -n 2 -p 3 convert_images_dcm2jpg()
----> 3 convert_images_dcm2jpg()

/tmp/ipykernel_11/979935879.py in convert_images_dcm2jpg()
     11 def convert_images_dcm2jpg():
     12     # Preprocess all images in parallel using Joblib
---> 13     jobs = [joblib.delayed(preprocess_and_save_image)(args) for args in test_df.groupby(['patient_id', 'laterality'])]
     14     SUBMISSION_ROWS = joblib.Parallel(
     15         n_jobs = -1,                 # -1, cpu_count()

/tmp/ipykernel_11/979935879.py in <listcomp>(.0)
     11 def convert_images_dcm2jpg():
     12     # Preprocess all images in parallel using Joblib
---> 13     jobs = [joblib.delayed(preprocess_and_save_image)(args) for args in test_df.groupby(['patient_id', 'laterality'])]
     14     SUBMISSION_ROWS = joblib.Parallel(
     15         n_jobs = -1,                 # -1, cpu_count()

NameError: name 'joblib' is not defined

## === cell 42
SUBMISSION_ROWS = []

for idx, ((patient_id, laterality), group) in enumerate(tqdm(test_df.groupby(['patient_id', 'laterality']))):
    image_datas, cancer = [], []
    for row_idx, row in group.iterrows():
        patient_id, image_id = row['patient_id'], row['image_id']
        image_data = cv2.imread(f'{patient_id}/{image_id}.{IMAGE_FORMAT.lower()}', -1)
        image_datas.append(image_data)
        image_data = np.expand_dims(image_data, [0, 3])
        cancer.append( model.predict_on_batch({'image': image_data}).squeeze().tolist() )
        os.remove(f'{patient_id}/{image_id}.{IMAGE_FORMAT.lower()}')
        
    if idx < 2:
        fig, axes = plt.subplots(nrows=1, ncols=len(image_datas), figsize=(5,5))
        fig.subplots_adjust(hspace=0.1, wspace=0.05)
        for n, ax in enumerate(axes.flat):
            ax.imshow(image_datas[n], cmap='bone') # binary, gray, bone
            ax.set_title(f'{laterality}: {cancer[n]:.1e}')
            ax.axis('off')            
        plt.show()
        fig.tight_layout()
        
        
    SUBMISSION_ROWS.append({
        'prediction_id': f'{patient_id}_{laterality}',
        'cancer': np.int8(np.mean(cancer) > THRESHOLD_BEST),
    })    
    if np.random.rand() > 0.95:
        gc.collect()

## --- ERROR in cell 42, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3882181877.py in <cell line: 0>()
      2 
      3 # Iterate over all patient_id/laterality combinations groups
----> 4 for idx, ((patient_id, laterality), group) in enumerate(tqdm(test_df.groupby(['patient_id', 'laterality']))):
      5     # Cancer target is mean of predicted cancer values
      6     image_datas, cancer = [], []

NameError: name 'tqdm' is not defined

## === cell 44
submission_df = pd.DataFrame(SUBMISSION_ROWS)

display(submission_df.head()), print('\n'), submission_df.info(),

## === cell 45
submission_df.to_csv("submission.csv", index=False)

## --- ERROR in outputing the csv:
Invalid submission: prediction_id not in submission
