# Goal

You will receive environment details and a partial notebook export.

# Requirements

- Fix the bug that causes the error in cell k.
- Do NOT adjust any other non-buggy cells.
- You may reference cell k+1 only to preserve variable/interface compatibility.
- Do not complete or extend code logic in cell k, k+1, or later cells.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (bug fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Output must follow your strict format: Diagnosis / Patch summary / Updated cells / Compatibility notes for cell k+1 / Assumptions.


# 1. Python version

3.9

# 2. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
seaborn==0.12.2
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

# 3. Data file paths

```
/
    kaggle/
        data/
            description.md (59 lines)
            sampleSubmission.csv (5789881 lines)
            sampleSubmission.csv.zip (12.0 MB)
            test.zip (4.0 MB)
            train.zip (15.5 MB)
            train_cleaned.zip (5.2 MB)
            denoising-dirty-documents/
                description.md (59 lines)
                sampleSubmission.csv (5789881 lines)
                ... and 4 other files
                denoising-dirty-documents/
                test/
                    110.png (149.2 kB)
                    111.png (146.9 kB)
                    ... and 27 other files
                    test/
                train/
                    116.png (152.2 kB)
                    201.png (156.1 kB)
                    ... and 113 other files
                    train/
                train_cleaned/
                    173.png (60.4 kB)
                    47.png (35.5 kB)
                    ... and 113 other files
            test/
                110.png (149.2 kB)
                111.png (146.9 kB)
                ... and 27 other files
                test/
            train/
                116.png (152.2 kB)
                201.png (156.1 kB)
                ... and 113 other files
                train/
            train_cleaned/
                173.png (60.4 kB)
                47.png (35.5 kB)
                ... and 113 other files
        input/
            description.md (59 lines)
            sampleSubmission.csv (5789881 lines)
            sampleSubmission.csv.zip (12.0 MB)
            test.zip (4.0 MB)
            train.zip (15.5 MB)
            train_cleaned.zip (5.2 MB)
            denoising-dirty-documents/
                description.md (59 lines)
                sampleSubmission.csv (5789881 lines)
                ... and 4 other files
                denoising-dirty-documents/
                test/
                    110.png (149.2 kB)
                    111.png (146.9 kB)
                    ... and 27 other files
                    test/
                train/
                    116.png (152.2 kB)
                    201.png (156.1 kB)
                    ... and 113 other files
                    train/
                train_cleaned/
                    173.png (60.4 kB)
                    47.png (35.5 kB)
                    ... and 113 other files
            test/
                110.png (149.2 kB)
                111.png (146.9 kB)
                ... and 27 other files
                test/
                    110.png (149.2 kB)
                    111.png (146.9 kB)
                    ... and 27 other files
                    test/
            train/
                116.png (152.2 kB)
                201.png (156.1 kB)
                ... and 113 other files
                train/
                    116.png (152.2 kB)
                    201.png (156.1 kB)
                    ... and 113 other files
                    train/
            train_cleaned/
                173.png (60.4 kB)
                47.png (35.5 kB)
                ... and 113 other files
        working/
            denoising-dirty-documents/
                description.md (59 lines)
                sampleSubmission.csv (5789881 lines)
                ... and 4 other files
                denoising-dirty-documents/
                test/
                    110.png (149.2 kB)
                    111.png (146.9 kB)
                    ... and 27 other files
                    test/
                train/
                    116.png (152.2 kB)
                    201.png (156.1 kB)
                    ... and 113 other files
                    train/
                train_cleaned/
                    173.png (60.4 kB)
                    47.png (35.5 kB)
                    ... and 113 other files
```

-> data/denoising-dirty-documents/sampleSubmission.csv has 5789880 rows and 2 columns.
The columns are: id, value

-> data/sampleSubmission.csv has 5789880 rows and 2 columns.
The columns are: id, value

-> input/denoising-dirty-documents/sampleSubmission.csv has 5789880 rows and 2 columns.
The columns are: id, value

-> input/sampleSubmission.csv has 5789880 rows and 2 columns.
The columns are: id, value

-> working/denoising-dirty-documents/sampleSubmission.csv has 5789880 rows and 2 columns.
The columns are: id, value

# 4. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

try:
    from google.protobuf import message_factory as _message_factory  # type: ignore

    _mf_cls = getattr(_message_factory, "MessageFactory", None)
    if _mf_cls is not None and getattr(_mf_cls, "GetPrototype", None) is None:

        def _GetPrototype(self, descriptor):  # type: ignore
            get_message_class = getattr(self, "GetMessageClass", None)
            if callable(get_message_class):
                return get_message_class(descriptor)
            raise AttributeError(
                "MessageFactory.GetPrototype is not available and cannot be emulated "
                "because MessageFactory.GetMessageClass is also missing."
            )

        try:
            setattr(_mf_cls, "GetPrototype", _GetPrototype)  # type: ignore[attr-defined]
        except Exception:
            pass
except Exception:
    pass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import zipfile, cv2
from tqdm.auto import tqdm
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras import layers, callbacks, utils

try:
    import imgaug as ia  # type: ignore
    import imgaug.augmenters as iaa  # type: ignore

    _imgaug_import_error = None
except Exception as e:
    ia = None
    iaa = None
    _imgaug_import_error = e

sns.set_style("darkgrid")


## === cell 1
path_zip = '../input/denoising-dirty-documents/'
path = '/kaggle/working/'

with zipfile.ZipFile(path_zip + 'train.zip', 'r') as zip_ref:
    zip_ref.extractall(path)

with zipfile.ZipFile(path_zip + 'test.zip', 'r') as zip_ref:
    zip_ref.extractall(path)  
    
with zipfile.ZipFile(path_zip + 'train_cleaned.zip', 'r') as zip_ref:
    zip_ref.extractall(path)  
    
with zipfile.ZipFile(path_zip + 'sampleSubmission.csv.zip', 'r') as zip_ref:
    zip_ref.extractall(path)
    
train_img = sorted(os.listdir(path + '/train'))
train_cleaned_img = sorted(os.listdir(path + '/train_cleaned'))
test_img = sorted(os.listdir(path + '/test'))


## === cell 2
class config():
    IMG_SIZE = (420, 540)

imgs = [cv2.imread(path + 'train/' + f) for f in sorted(os.listdir(path + 'train/'))]
print('Median Dimensions:', np.median([len(img) for img in imgs]), np.median([len(img[0]) for img in imgs]))
del imgs


## === cell 3
def process_image(path):
    img = cv2.imread(path)
    img = np.asarray(img, dtype="float32")
    img = cv2.resize(img, config.IMG_SIZE[::-1])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img/255.0
    img = np.reshape(img, (*config.IMG_SIZE, 1))
    
    return img


## === cell 4
train = []
train_cleaned = []
test = []

for f in sorted(os.listdir(path + 'train/')):
    train.append(process_image(path + 'train/' + f))

for f in sorted(os.listdir(path + 'train_cleaned/')):
    train_cleaned.append(process_image(path + 'train_cleaned/' + f))
    
for f in sorted(os.listdir(path + 'test/')):
    test.append(process_image(path + 'test/' + f))
    
train = np.asarray(train)
train_cleaned = np.asarray(train_cleaned)
test = np.asarray(test)


## === cell 5
train.shape, train_cleaned.shape, test.shape


## === cell 6
fig, ax = plt.subplots(4, 2, figsize=(15,25))
for i in range(4):
    ax[i][0].imshow(tf.squeeze(train[i]), cmap='gray')
    ax[i][0].set_title('Noise image: {}'.format(train_img[i]))
    
    ax[i][1].imshow(tf.squeeze(train_cleaned[i]), cmap='gray')
    ax[i][1].set_title('Denoised image: {}'.format(train_img[i]))
    
    ax[i][0].get_xaxis().set_visible(False)
    ax[i][0].get_yaxis().set_visible(False)
    ax[i][1].get_xaxis().set_visible(False)
    ax[i][1].get_yaxis().set_visible(False)


## === cell 7
def augment_pipeline(pipeline, images, seed=19):
    ia.seed(seed)
    processed_images = images.copy()
    for step in pipeline:
        temp = np.array(step.augment_images(images))
        processed_images = np.append(processed_images, temp, axis=0)
    return(processed_images)


## === cell 8
if iaa is None:
    raise ImportError(
        "imgaug is required for the augmentation pipeline in this notebook, but it could not be imported. "
        f"Original import error: {_imgaug_import_error!r}"
    )

rotate90 = iaa.Rot90(1)  # rotate image 90 degrees
rotate180 = iaa.Rot90(2)  # rotate image 180 degrees
rotate270 = iaa.Rot90(3)  # rotate image 270 degrees
random_rotate = iaa.Rot90((1, 3))  # randomly rotate image from 90,180,270 degrees
perc_transform = iaa.PerspectiveTransform(
    scale=(0.02, 0.1)
)  # Skews and transform images without black bg
rotate10 = iaa.Affine(rotate=(10))  # rotate image 10 degrees
rotate10r = iaa.Affine(rotate=(-10))  # rotate image 30 degrees in reverse
crop = iaa.Crop(px=(5, 32))  # Crop between 5 to 32 pixels
hflip = iaa.Fliplr(1)  # horizontal flips for 100% of images
vflip = iaa.Flipud(1)  # vertical flips for 100% of images
gblur = iaa.GaussianBlur(
    sigma=(1, 1.5)
)  # gaussian blur images with a sigma of 1.0 to 1.5
motionblur = iaa.MotionBlur(8)  # motion blur images with a kernel size 8

seq_rp = iaa.Sequential(
    [
        iaa.Rot90((1, 3)),  # randomly rotate image from 90,180,270 degrees
        iaa.PerspectiveTransform(
            scale=(0.02, 0.1)
        ),  # Skews and transform images without black bg
    ]
)

seq_cfg = iaa.Sequential(
    [
        iaa.Crop(
            px=(5, 32)
        ),  # crop images from each side by 5 to 32px (randomly chosen)
        iaa.Fliplr(0.5),  # horizontally flip 50% of the images
        iaa.GaussianBlur(sigma=(0, 1.5)),  # blur images with a sigma of 0 to 1.5
    ]
)

seq_fm = iaa.Sequential(
    [
        iaa.Flipud(1),  # vertical flips all the images
        iaa.MotionBlur(k=6),  # motion blur images with a kernel size 6
    ]
)


## --- ERROR in cell 8, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mImportError[0m                               Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/2747940079.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      2[0m [0;31m# Fail fast here with a clear error instead of `'NoneType' object has no attribute ...'`.[0m[0;34m[0m[0;34m[0m[0m
[1;32m      3[0m [0;32mif[0m [0miaa[0m [0;32mis[0m [0;32mNone[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 4[0;31m     raise ImportError(
[0m[1;32m      5[0m         [0;34m"imgaug is required for the augmentation pipeline in this notebook, but it could not be imported. "[0m[0;34m[0m[0;34m[0m[0m
[1;32m      6[0m         [0;34mf"Original import error: {_imgaug_import_error!r}"[0m[0;34m[0m[0;34m[0m[0m

[0;31mImportError[0m: imgaug is required for the augmentation pipeline in this notebook, but it could not be imported. Original import error: ModuleNotFoundError("No module named 'imgaug'")

## === cell 9
pipeline = [
    rotate90, rotate180, rotate270, hflip, vflip
]
