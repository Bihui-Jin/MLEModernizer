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

3.11

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

0.2375478927203065

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 3
%%writefile requirements.txt


/kaggle/input/rsna-breast-cancer-class-convnextv2-model-weights/pylibjpeg-1.4.0-py3-none-any.whl
/kaggle/input/rsna-breast-cancer-class-convnextv2-model-weights/pylibjpeg_libjpeg-1.3.2-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
/kaggle/input/rsna-breast-cancer-class-convnextv2-model-weights/dicomsdl-0.109.1-cp37-cp37m-manylinux_2_12_x86_64.manylinux2010_x86_64.whl 
/kaggle/input/rsna-breast-cancer-class-convnextv2-model-weights/pydicom-2.3.0-py3-none-any.whl
/kaggle/input/rsna-breast-cancer-class-convnextv2-model-weights/python_gdcm-3.0.10-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

/kaggle/input/rsna-breast-cancer-class-convnextv2-model-weights/keras_cv_attention_models-1.3.5-py3-none-any.whl

## === cell 4
import os, sys, platform
print("Python  :", sys.version)
print("Platform:", platform.platform())

!{sys.executable} -m pip install -Uq -r requirements.txt --no-deps --no-index  # --force-reinstall

## === cell 6
import tensorflow as tf
print("Tensorflow version \t\t:" + tf.__version__)


tf.compat.v1.enable_v2_behavior()
tf.config.optimizer.set_jit(True)

tf.config.threading.set_inter_op_parallelism_threads(num_threads=1)

import cv2; print('cv2', cv2.__version__), cv2.setNumThreads(1)
import pylibjpeg; print("pylibjpeg", pylibjpeg.__version__)
import libjpeg; print("libjpeg", libjpeg.__version__)
import dicomsdl; print("dicomsdl", dicomsdl.DICOMSDL_VERSION)
import pydicom; print("pydicom", pydicom.__version__);
from pydicom.pixel_data_handlers.util import apply_voi_lut, apply_modality_lut
import gdcm; print("gdcm", gdcm.GDCM_VERSION);

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 7
try:
    import tensorflow as tf
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()
    strategy = tf.distribute.TPUStrategy(tpu)    
    print('Running on TPU ', tpu.master(), len(tf.config.list_logical_devices('TPU')))
    gpus = None    
except ValueError:
    gpus = tf.config.list_logical_devices('GPU')
    if len(gpus) > 1:      
        strategy = tf.distribute.MirroredStrategy(devices=[f"/gpu:{gpu.name.split(':')[-1]}" for gpu in gpus])
        print('Running on multiple GPUs', [gpu.name for gpu in gpus])
    elif len(gpus) == 1:
        strategy = tf.distribute.get_strategy() # default strategy that works on CPU and single GPU
        print('Running on single GPU', gpus[0].name)
    else:
        strategy = tf.distribute.get_strategy() # default strategy that works on CPU and single GPU
        print('Running on CPU')
        
print("Number of accelerators: ", strategy.num_replicas_in_sync)  

## === cell 8
import tensorflow.python as tf_python

if gpus:
    print("changed shard policy...")
    tf.data.experimental.DistributeOptions.auto_shard_policy = tf_python.data.util.options.create_option(
        docstring="The type of sharding to use. See "
        "`tf.data.experimental.AutoShardPolicy` for additional information.",
        ty=tf.data.experimental.AutoShardPolicy,
        default_factory=lambda: tf.data.experimental.AutoShardPolicy.DATA,
        name="auto_shard_policy",
    )
    


## === cell 10
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

import random, re
import scipy.stats as stats

import scikitplot as skplt

import time
from glob import glob
from pathlib import Path
from tqdm.notebook import tqdm 
from joblib import Parallel, delayed
from multiprocessing import cpu_count

from keras_cv_attention_models.convnext import ConvNeXtV2Tiny
from kaggle_datasets import KaggleDatasets

import gc


## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/3344812652.py in <cell line: 0>()
      9 
     10 # !pip install scikit-plot -Uq
---> 11 import scikitplot as skplt
     12 
     13 import time

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

## === cell 12
IMAGE_FORMAT  = 'JPG'
IMAGE_QUALITY = 95
TARGET_HEIGHT, TARGET_WIDTH, N_CHANNELS = ( 1024, 768, 1 )
INPUT_SHAPE   = (TARGET_HEIGHT, TARGET_WIDTH, N_CHANNELS)

THRESHOLD_BEST = 0.849352

## === cell 13
DATA_DIR       = "/kaggle/input/rsna-breast-cancer-detection"
MODEL_DATA_DIR = "/kaggle/input/rsna-breast-cancer-class-convnextv2-model-weights/rsna_cancer_convnext_v2_tiny_model_weights.h5"


print('DATA_DIR      :', DATA_DIR)
print('MODEL_DATA_DIR:', MODEL_DATA_DIR)

## === cell 16
test_df              = pd.read_csv(f'{DATA_DIR}/test.csv')

sample_submission_df = pd.read_csv(f'{DATA_DIR}/sample_submission.csv')


test_df.shape, sample_submission_df.shape

## === cell 17
display(test_df.head(), sample_submission_df.head())

## === cell 18
test_df.info(), print('\n'), sample_submission_df.info(), 

## === cell 20
test_file_paths = tf.io.gfile.glob(f'{DATA_DIR}/test_images/*/*.dcm')

print(f'Test size images : {len(test_file_paths):<10}')

## === cell 23
def image_findNonZero(img_data_uint8):
    coords = cv2.findNonZero(img_data_uint8)
    x1, y1, w, h = cv2.boundingRect(coords)
    
    img_data_cropped = img_data_uint8[y1:y1+h, x1:x1+w]      
    return img_data_cropped


def load_and_preprocess_image(file_path, save_image=True, debug=False):        
    img_dicom       = dicomsdl.open(file_path)              # dicomsdl - more faster
    img_data_uint16 = img_dicom.pixelData(storedvalue=True)       
    img_data_uint8  = cv2.normalize(img_data_uint16, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)     
    if img_dicom.PhotometricInterpretation == 'MONOCHROME1':
        img_data_uint8 = np.amax(img_data_uint8) - img_data_uint8  
        
            
    img_data_cropped         = image_findNonZero(img_data_uint8)   
    img_data_cropped_resized = cv2.resize(
        img_data_cropped,
        dsize=INPUT_SHAPE[:2][::-1], # attention cv2 size reverse
        interpolation=cv2.INTER_LANCZOS4,       
    )    
    img_data_cropped_resized = np.expand_dims(img_data_cropped_resized, 2)    
    
       
    if save_image:
        patient_id, image_id = re.split('[\./]', file_path)[-3:-1]
        if not os.path.exists(patient_id):
            os.makedirs(patient_id, exist_ok=True)
            
        if IMAGE_FORMAT == 'PNG':
            cv2.imwrite(f'{patient_id}/{image_id}.png', img_data_cropped_resized)
        else:
            cv2.imwrite(f'{patient_id}/{image_id}.jpg', img_data_cropped_resized, [cv2.IMWRITE_JPEG_QUALITY, IMAGE_QUALITY])
            
    if debug:            
        plt.imshow(img_data_cropped_resized, cmap='bone')

## === cell 24
%timeit with strategy.scope(): load_and_preprocess_image(test_file_paths[0], debug=True)

## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3292787859.py in <cell line: 0>()
      1 # samples
----> 2 get_ipython().run_line_magic('timeit', 'with strategy.scope(): load_and_preprocess_image(test_file_paths[0], debug=True)')

/usr/local/lib/python3.11/dist-packages/IPython/core/interactiveshell.py in run_line_magic(self, magic_name, line, _stack_depth)
   2416                 kwargs['local_ns'] = self.get_local_scope(stack_depth)
   2417             with self.builtin_trap:
-> 2418                 result = fn(*args, **kwargs)
   2419             return result
   2420 

<decorator-gen-53> in timeit(self, line, cell, local_ns)

/usr/local/lib/python3.11/dist-packages/IPython/core/magic.py in <lambda>(f, *a, **k)
    185     # but it's overkill for just that one bit of state.
    186     def magic_deco(arg):
--> 187         call = lambda f, *a, **k: f(*a, **k)
    188 
    189         if callable(arg):

/usr/local/lib/python3.11/dist-packages/IPython/core/magics/execution.py in timeit(self, line, cell, local_ns)
   1178             for index in range(0, 10):
   1179                 number = 10 ** index
-> 1180                 time_number = timer.timeit(number)
   1181                 if time_number >= 0.2:
   1182                     break

/usr/local/lib/python3.11/dist-packages/IPython/core/magics/execution.py in timeit(self, number)
    167         gc.disable()
    168         try:
--> 169             timing = self.inner(it, self.timer)
    170         finally:
    171             if gcold:

<magic-timeit> in inner(_it, _timer)

/tmp/ipykernel_11/3193753708.py in load_and_preprocess_image(file_path, save_image, debug)
     13 def load_and_preprocess_image(file_path, save_image=True, debug=False):
     14     # Load the raw data from the file as dicomsdl image array uint16
---> 15     img_dicom       = dicomsdl.open(file_path)              # dicomsdl - more faster
     16     img_data_uint16 = img_dicom.pixelData(storedvalue=True)
     17     img_data_uint8  = cv2.normalize(img_data_uint16, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

NameError: name 'dicomsdl' is not defined

## === cell 27
print("Model Defined Shape: ", INPUT_SHAPE)         # Input Layer Shape

def build_classifier_model(
        input_shape    = INPUT_SHAPE,
        model_weights  = MODEL_DATA_DIR,  # Pretrained Model Weights File Path
) -> tf.keras.models.Model:    
    import tensorflow as tf    
    
    input_image      = tf.keras.layers.Input(shape=input_shape, name='image', dtype=tf.uint8)
    input_patient_id = tf.keras.layers.Input(shape=(1,), name='patient_id')
    input_image_id   = tf.keras.layers.Input(shape=(1,), name='image_id')
    
    x = tf.image.grayscale_to_rgb(input_image)
    x = tf.cast(x, tf.float32)
    x = tf.keras.applications.imagenet_utils.preprocess_input(x, mode='tf')   
    x = ConvNeXtV2Tiny(input_shape=(input_shape[:-1] + (3,)), pretrained=None, num_classes=0)(x) 
    x = tf.keras.layers.SpatialDropout2D(0.3)(x)
    

    x = tf.keras.layers.GlobalAveragePooling2D()(x)     
    x = tf.keras.layers.Dense(512, activation=tf.keras.layers.LeakyReLU(alpha=0.3), kernel_regularizer=tf.keras.regularizers.l1_l2(l2=0.005))(x)
    x = tf.keras.layers.Dropout(0.3)(x) 
    x = tf.keras.layers.Dense(256, activation='gelu', kernel_regularizer=tf.keras.regularizers.l1(0.005))(x)
    x = tf.keras.layers.Dropout(0.3)(x)   
    x = tf.keras.layers.Dense(128, activation='gelu', kernel_regularizer=tf.keras.regularizers.l1(0.005))(x)
    x = tf.keras.layers.Dropout(0.3)(x)   
    
    output = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.models.Model(inputs=[input_image, input_patient_id, input_image_id], outputs=output)
    
    
    model.load_weights(model_weights)

    model.trainable = False

    model.compile()

    return model

## === cell 28
tf.keras.backend.clear_session()

with strategy.scope():  
    model = build_classifier_model()
    model.summary()

## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/463086840.py in <cell line: 0>()
      2 
      3 with strategy.scope():
----> 4     model = build_classifier_model()
      5     model.summary()

/tmp/ipykernel_11/4082781074.py in build_classifier_model(input_shape, model_weights)
     15     # convolutional layers - feature extraction
     16     # Repeat channels to create 3 channel images required by pretrained ConvNextV2 models
---> 17     x = tf.image.grayscale_to_rgb(input_image)
     18     # tf: will scale pixels between -1 and 1, sample-wise.
     19     x = tf.cast(x, tf.float32)

/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/traceback_utils.py in error_handler(*args, **kwargs)
    151     except Exception as e:
    152       filtered_tb = _process_traceback_frames(e.__traceback__)
--> 153       raise e.with_traceback(filtered_tb) from None
    154     finally:
    155       del filtered_tb

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/image_ops_impl.py in _CheckGrayscaleImage(image, require_static)
    293     else:
    294       image_shape = image.get_shape().with_rank_at_least(2)
--> 295   except ValueError:
    296     raise ValueError('A grayscale image (shape %s) must be at least '
    297                      'two-dimensional.' % image.shape)

AttributeError: 'KerasTensor' object has no attribute 'get_shape'

## === cell 32
def preprocess_and_save_image(args):
    (patient_id, laterality), group = args    
    for row_idx, row_data in group.iterrows():    
        image_path = f"{DATA_DIR}/test_images/{patient_id}/{row_data['image_id']}.dcm"
        load_and_preprocess_image(image_path)  
        
        
def convert_images_dicom2jpg():   
    jobs = [delayed(preprocess_and_save_image)(args) for args in tqdm(test_df.groupby(['patient_id', 'laterality']))]
    SUBMISSION_ROWS = Parallel(
        n_jobs = cpu_count(), #-1,
        verbose= 0,
        backend= 'multiprocessing',
        prefer = 'threads',
    )(jobs)

## === cell 34
with strategy.scope():
    convert_images_dicom2jpg() 
    
    if np.random.rand() > 0.90:
        gc.collect()

## --- ERROR in cell 34, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4136810703.py in <cell line: 0>()
      1 with strategy.scope():
----> 2     convert_images_dicom2jpg()
      3 
      4     if np.random.rand() > 0.90:
      5         gc.collect()

/tmp/ipykernel_11/3767537029.py in convert_images_dicom2jpg()
     10 def convert_images_dicom2jpg():
     11     # Preprocess all images in parallel using Joblib
---> 12     jobs = [delayed(preprocess_and_save_image)(args) for args in tqdm(test_df.groupby(['patient_id', 'laterality']))]
     13     SUBMISSION_ROWS = Parallel(
     14         n_jobs = cpu_count(), #-1,

NameError: name 'tqdm' is not defined

## === cell 35
SUBMISSION_ROWS = []

for idx, ((patient_id, laterality), group) in enumerate(tqdm(test_df.groupby(['patient_id', 'laterality']))):    
    image_datas, cancer = [], []    
    for row_idx, row in group.iterrows():
        patient_id, image_id = row['patient_id'], row['image_id']
        image_data = cv2.imread(f'{patient_id}/{image_id}.{IMAGE_FORMAT.lower()}', -1)
        image_datas.append(image_data)
        image_data = np.expand_dims(image_data, [0, 3])
        
        X_batch = {'image': image_data, 'patient_id':np.array([patient_id]), 'image_id':np.array([image_id])}
        cancer.append( model.predict_on_batch(X_batch).squeeze().tolist() )
        
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
    
    if np.random.rand() > 0.99:
        gc.collect()

## --- ERROR in cell 35, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2062811185.py in <cell line: 0>()
      2 
      3 # Iterate over all patient_id/laterality combinations groups
----> 4 for idx, ((patient_id, laterality), group) in enumerate(tqdm(test_df.groupby(['patient_id', 'laterality']))):
      5     image_datas, cancer = [], []
      6     # Iterate over all scans in group

NameError: name 'tqdm' is not defined

## === cell 37
submission_df = pd.DataFrame(SUBMISSION_ROWS)

display(submission_df.head()), print('\n'), submission_df.info(),

## === cell 38
submission_df.to_csv("submission.csv", index=False)

## --- ERROR in outputing the csv:
Invalid submission: prediction_id not in submission
