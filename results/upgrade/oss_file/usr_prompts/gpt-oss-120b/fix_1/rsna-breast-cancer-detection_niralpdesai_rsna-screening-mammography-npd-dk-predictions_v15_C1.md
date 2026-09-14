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

9.291604409103898e-05

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
!pip install --no-index --no-deps /kaggle/input/extrapackages-dicomsdl-gdcm-pylibjpeg/wheelhouse/dicomsdl-0.109.1-cp37-cp37m-manylinux_2_12_x86_64.manylinux2010_x86_64.whl
!pip install --no-index --no-deps /kaggle/input/extrapackages-dicomsdl-gdcm-pylibjpeg/wheelhouse/numpy-1.21.6-cp37-cp37m-manylinux_2_12_x86_64.manylinux2010_x86_64.whl
!pip install --no-index --no-deps /kaggle/input/extrapackages-dicomsdl-gdcm-pylibjpeg/wheelhouse/pylibjpeg-1.4.0-py3-none-any.whl
!pip install --no-index --no-deps /kaggle/input/extrapackages-dicomsdl-gdcm-pylibjpeg/wheelhouse/python_gdcm-3.0.21-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl


## === cell 1

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # for making plots
from tqdm.notebook import tqdm
import sys
from joblib import Parallel, delayed
from multiprocessing import cpu_count
import dicomsdl


import os

import glob



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/3362383539.py in <cell line: 0>()
     10 from joblib import Parallel, delayed
     11 from multiprocessing import cpu_count
---> 12 import dicomsdl
     13 
     14 # Input data files are available in the read-only "../input/" directory

ModuleNotFoundError: No module named 'dicomsdl'

## === cell 2
csv_train = pd.read_csv("/kaggle/input/rsna-breast-cancer-detection/train.csv")
csv_test = pd.read_csv("/kaggle/input/rsna-breast-cancer-detection/test.csv")

## === cell 3
csv_train.groupby('cancer').cancer.count()


## === cell 4
testdir = '/kaggle/input/rsna-breast-cancer-detection/test_images'
for i in glob.iglob('/kaggle/input/rsna-breast-cancer-detection/test_images/**/**'):
    print(i)

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3585149662.py in <cell line: 0>()
      1 testdir = '/kaggle/input/rsna-breast-cancer-detection/test_images'
----> 2 for i in glob.iglob('/kaggle/input/rsna-breast-cancer-detection/test_images/**/**'):
      3     print(i)

NameError: name 'glob' is not defined

## === cell 5
columns_in_test = []
for col in csv_test.columns:
    if col != 'prediction_id': # This column is not in the training set.
        columns_in_test.append(col)
columns_in_test.append('cancer') # Make sure we don't drop the target column yet!
print(columns_in_test)

data_train = csv_train[columns_in_test]

bad_ids = [1942326353]
data_train = data_train.drop(data_train[data_train['image_id'].isin(bad_ids)].index)
data_train.shape

data_train = data_train.dropna(subset=['age'])
data_train['age'] = data_train['age'] / data_train['age'].max()

## === cell 6
data_test = csv_test.copy()
data_test['age'] = data_test['age'].fillna(data_test['age'].mean())

## === cell 7
cancer_train = data_train.loc[csv_train.cancer == 1]
nocancer_train = data_train.loc[csv_train.cancer == 0]

cancer_train.head()

## === cell 8
nocancer_train.head()
print(len(data_train['patient_id'].unique()) )
print(len(data_train['image_id'].unique()) - len(data_train))


## === cell 9
import pydicom as dcm
import pydicom.filereader
import pylibjpeg

train_images_folder = "/kaggle/input/rsna-breast-cancer-detection/train_images"
test_images_folder = "/kaggle/input/rsna-breast-cancer-detection/test_images"

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/10697191.py in <cell line: 0>()
      2 import pydicom as dcm
      3 import pydicom.filereader
----> 4 import pylibjpeg
      5 
      6 train_images_folder = "/kaggle/input/rsna-breast-cancer-detection/train_images"

ModuleNotFoundError: No module named 'pylibjpeg'

## === cell 10

train_im_dir = '/kaggle/input/rsna-breast-cancer-detection/train_images/'
test_im_dir = '/kaggle/input/rsna-breast-cancer-detection/test_images/'

path = [test_im_dir + str(patient) + '/' + str(image) + '.dcm'
        for patient, image in zip(data_test['patient_id'], data_test['image_id'])]

## === cell 11


import cv2
import time
import ray

def process_im_ray(im, image_size = 256, show = False):
    if show:
        try:
            print('begin process')
            plt.imshow(im.pixelData(), cmap='gray')
            plt.show(); plt.close()
        except: 
            print('begin process')
            plt.imshow(im.pixel_array, cmap='gray')
            plt.show(); plt.close()
    try:
        out = im.pixelData()
        phototype = im.getPixelDataInfo()['PhotometricInterpretation']
    except:
        out = im.pixel_array
        phototype = im.PhotometricInterpretation
        
    if phototype == 'MONOCHROME1':
        out = out.max() - out
        
    minval = out.min()
    maxval = out.max()
    
    Threshold = maxval/5
    out = cut_empty_space_ray(out, T = Threshold, show = show)
    
    out = (out-minval) / (maxval-minval)
    
    out = cv2.resize(out, (image_size, image_size))
    
    if show:
        plt.imshow(out, cmap='gray')
        plt.show(); plt.close()
        
    return out


def cut_empty_space_ray(im, T = 100, cutedge = 10, show = False):
    impx_cv2_raw = im[cutedge:-cutedge, cutedge:-cutedge] 
    
    _, impx_cv2 = cv2.threshold(impx_cv2_raw, T, 255, cv2.THRESH_BINARY) 
    
    contours, _ = cv2.findContours(impx_cv2.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
    boundary = max(contours, key=cv2.contourArea) # Get the contour enclosing the most space.
    
    impx_mask = np.zeros(impx_cv2.shape, dtype="uint8") 
    cv2.drawContours(impx_mask, [boundary], -1, 255, cv2.FILLED) 
    impx_cv2_masked = cv2.bitwise_and(impx_cv2_raw, impx_cv2_raw, mask = impx_mask) 
    
    s = time.time()
    x, y, w, h = cv2.boundingRect(boundary)

    
    
    out = impx_cv2_masked[y:y+h, x:x+w]
    if show:        
        plt.imshow(out)
        plt.show(); plt.close()
        print('end crop')
    return out

## === cell 12
ray.shutdown()
ray.init(log_to_driver=False, num_gpus=1, num_cpus=cpu_count())




def to_iterator(obj_ids):
    while obj_ids:
        done,  obj_ids = ray.wait(obj_ids)
        yield ray.get(done[0])

@ray.remote
def get_images_new(imagepath, image_size = 256):
    
    im = dicomsdl.open(imagepath)
    image = process_im_ray(im, image_size)
    image = np.ndarray.flatten(image)
    del im
    return image
    
@ray.remote
def save_images(imagepath, image_size = 256, savedir = None):
    
    im = dicomsdl.open(imagepath)
    image = process_im_ray(im, image_size)
    
    directory_split = imagepath.split('/')
    image_id = directory_split[-1].split('.')[0]
    patient_id = directory_split[-2]
    if savedir is not None:
        newfolder = savedir + '/' + patient_id
        fname = newfolder + '/' + image_id + '.png'
        os.makedirs(newfolder, exist_ok=True)
        image_save = (image * 255).astype(np.uint8)
        cv2.imwrite(fname, image_save)
    
start = time.time()
image_size = 512

run_dcm_to_png = True
show_prog_bar = False
endpoint = 100

if run_dcm_to_png:

    workdir = '/kaggle/working/test/'
    save_dir = workdir + 'processed_' + str(image_size)
    os.makedirs(save_dir, exist_ok=True)

    Save_Im_futures = [save_images.remote(p, image_size = image_size, savedir = save_dir) for p in path]
    if show_prog_bar:
        Save_Im = [x for x in tqdm(to_iterator(Save_Im_futures), total=len(Save_Im_futures))]
    else:
        Save_Im = [x for x in (to_iterator(Save_Im_futures))]
    

ray.shutdown()
print(time.time() - start)


## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/473786882.py in <cell line: 0>()
     57     workdir = '/kaggle/working/test/'
     58     save_dir = workdir + 'processed_' + str(image_size)
---> 59     os.makedirs(save_dir, exist_ok=True)
     60 
     61     Save_Im_futures = [save_images.remote(p, image_size = image_size, savedir = save_dir) for p in path]

NameError: name 'os' is not defined

## === cell 13
def onehot(df, encode):
    temp = pd.get_dummies(df[encode])
    df = df.drop([encode], axis=1)
    df = pd.concat([df, temp], axis=1)
    return df

irrelevant_train = ['machine_id', 'site_id'] 
data_train_relevant = data_train.drop(irrelevant_train, axis=1)

irrelevant_test = ['machine_id', 'site_id', 'prediction_id']
data_test_relevant = data_test.drop(irrelevant_test, axis=1)

onehotcols = ['laterality', 'view']
for col in onehotcols:
    data_test_relevant = onehot(data_test_relevant, col)
    data_train_relevant = onehot(data_train_relevant, col)

data_test_relevant['implant'] = data_test_relevant['implant'].astype('uint8')

## === cell 14
data_train_relevant.info()
data_test_relevant.info()

add_onehot_cols = []
for c in data_train_relevant.columns:
    if c not in data_test_relevant.columns and c != 'cancer':
        data_test_relevant[c] = (0)
        data_test_relevant[c] = data_test_relevant[c].astype('uint8')

## === cell 15
data_train_relevant.info()
data_test_relevant.info()

## === cell 16
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator


pngfolder = '/kaggle/working/test/processed_512'

data_test_relevant['file'] = data_test_relevant.apply(
    lambda x: pngfolder + '/' + str(int(x['patient_id'])) + '/' + str(int(x['image_id'])) 
    + '.png', axis=1)


def test_from_dataframe(directory, generator, subset='training', batch_size = 64,
                        data = data_train_relevant, columns = ['cancer'], seed = None):
    
    gendat = generator.flow_from_dataframe(data, directory=directory, shuffle = True, 
                                           target_size = (ImageSize, ImageSize),
                                           subset = subset, batch_size = batch_size,
                                           x_col = 'file', y_col = columns, 
                                           class_mode = 'multi_output', 
                                           color_mode = 'grayscale', 
                                           validate_filenames=False)
    N = gendat.n
    i = 0
    while i < N:
        data = gendat.next()
        x_im = np.array(data[0]).astype(np.float16) # reference to image
        x_info = np.array(data[1][:]).T.astype(np.float16) # data columns
        i += batch_size
        yield [x_im, x_info]
ImageSize = 512 # i don't know if we should be resizing this
val_split = 0.0 
batch_size = 64 # No idea what a good number for this is, i'm using something close to what i saw on Google

cols = ['age', 'implant', 'L', 'R', 'AT', 'CC', 'LM', 'LMO', 'ML', 'MLO']
imagegen = ImageDataGenerator(rescale = 1./255., validation_split = val_split)

test_gen = test_from_dataframe(None, imagegen, subset='training', batch_size = batch_size,
                                data = data_test_relevant, columns = cols)

## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 17
data_test_relevant.head()

## === cell 18
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

with tf.device('/gpu:0'):
    
    input_Image = tf.keras.layers.Input(shape=(512,512, 1))
    
    print(input_Image.get_shape())
    
    conv_Image = Conv2D(10, 13, activation='relu', 
                        input_shape = (512, 512, 1))(input_Image) # convolution neural network

    
    
    print(conv_Image.get_shape())
    
    pool_Image = MaxPooling2D(pool_size=10)(conv_Image) # to reduce the size of the image for input to the rest of the ML
                        
   
    
    print(pool_Image.get_shape())
    


    conv_Image = Conv2D(10, 5, activation='relu', 
                        input_shape = (50, 50, 10))(pool_Image)
    
    print(conv_Image.get_shape())

    
    pool_Image = MaxPooling2D(pool_size=4)(conv_Image)
    
    print(pool_Image.get_shape())
    
    
    
    flatten_Image = Flatten()(pool_Image)
    
    input_Tags = Input(shape=(10,))
    input_all = Concatenate()([flatten_Image, input_Tags])
    
    
    CNN = Dense(10, activation='relu')(input_all)
    CNN = Dense(10, activation='relu')(CNN)
    CNN = Dense(10, activation='relu')(CNN)
    CNN = Dense(10, activation='relu')(CNN)
    CNN = Dense(1, activation='sigmoid')(CNN)


    ML = Model(inputs=[input_Image, input_Tags], outputs=CNN)
    ML.summary()
    
    ML.compile(optimizer='adam',
               loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
               metrics=['accuracy']
              )



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/3670880434.py in <cell line: 0>()
     18     input_Image = tf.keras.layers.Input(shape=(512,512, 1))
     19 
---> 20     print(input_Image.get_shape())
     21 
     22     # relu is a transformation applied to the output of a node

AttributeError: 'KerasTensor' object has no attribute 'get_shape'

## === cell 19
model_import_path = '/kaggle/input/breastcancerpredictor/breastcancerpredictor.h5'
ML.load_weights(model_import_path)

## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/705751977.py in <cell line: 0>()
      1 model_import_path = '/kaggle/input/breastcancerpredictor/breastcancerpredictor.h5'
----> 2 ML.load_weights(model_import_path)

NameError: name 'ML' is not defined

## === cell 20
test_pred = ML.predict(test_gen, steps=None)

## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2466071006.py in <cell line: 0>()
----> 1 test_pred = ML.predict(test_gen, steps=None)

NameError: name 'ML' is not defined

## === cell 21
print(test_pred)
test_pred_series = pd.Series(test_pred.T[0])
print(test_pred_series)
predictions = pd.DataFrame()
predictions['image_id'] = pd.Series(data_test_relevant['image_id']).tolist()
predictions['cancer'] = (test_pred_series).tolist()
predict_test = data_test.copy()
predict_test['cancer'] = (test_pred_series).tolist()
print(predict_test)

## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/976825558.py in <cell line: 0>()
----> 1 print(test_pred)
      2 test_pred_series = pd.Series(test_pred.T[0])
      3 print(test_pred_series)
      4 predictions = pd.DataFrame()
      5 predictions['image_id'] = pd.Series(data_test_relevant['image_id']).tolist()

NameError: name 'test_pred' is not defined

## === cell 22
def sigmoid(x, m = 10):
    y = (2*m*x-m)
    out = (1/(1+np.exp(-y)))
    return out


predict_merge = predict_test.groupby(['patient_id', 'laterality'])['cancer'].mean()
print(predict_merge)
predict_merge = sigmoid(predict_merge)
predict_merge = predict_merge.reset_index()
predict_merge['prediction_id'] = predict_merge.apply(
    lambda x: str(x['patient_id']) + '_' + x['laterality'], axis = 1
)
print(predict_merge)
output_cols = ['predict_merge', 'cancer']
predict_merge = predict_merge.reindex(columns=['prediction_id', 'cancer'])
print(predict_merge)

## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/623659065.py in <cell line: 0>()
      5 
      6 
----> 7 predict_merge = predict_test.groupby(['patient_id', 'laterality'])['cancer'].mean()
      8 print(predict_merge)
      9 predict_merge = sigmoid(predict_merge)

NameError: name 'predict_test' is not defined

## === cell 23
predict_merge.to_csv('submission.csv', index=False)

## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3212370506.py in <cell line: 0>()
----> 1 predict_merge.to_csv('submission.csv', index=False)

NameError: name 'predict_merge' is not defined
