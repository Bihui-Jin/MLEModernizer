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

import sys, subprocess

subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q", "protobuf==4.25.3"]
)

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "imgaug==0.4.0"])

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import seaborn as sns

    sns.set_style("darkgrid")
except Exception:
    sns = None
    plt.style.use("dark_background")

import zipfile, cv2
from tqdm.auto import tqdm

import builtins as _builtins

_real_import = _builtins.__import__


def _import_block_sklearn(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "sklearn" or name.startswith("sklearn."):
        raise ImportError(
            "Blocked sklearn import to avoid SciPy/NumPy incompatibility."
        )
    return _real_import(name, globals, locals, fromlist, level)


_builtins.__import__ = _import_block_sklearn
try:
    import tensorflow as tf
finally:
    _builtins.__import__ = _real_import

from tensorflow.keras.models import Model
from tensorflow.keras import layers, callbacks, utils

try:
    import imgaug as ia
    from imgaug import augmenters as iaa
except Exception:
    ia = None
    iaa = None


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
def _list_image_files(dir_path):
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    files = []
    for fn in sorted(os.listdir(dir_path)):
        fp = os.path.join(dir_path, fn)
        if not (os.path.isfile(fp) and fn.lower().endswith(exts)):
            continue

        try:
            probe = cv2.imread(fp)
        except Exception:
            probe = None
        if probe is None:
            continue

        files.append(fn)
    return files


_process_image_orig = process_image


def process_image(path):
    img = cv2.imread(path)
    if img is None or not isinstance(img, np.ndarray) or img.size == 0:
        return None

    img = np.asarray(img, dtype="float32")
    if not isinstance(img, np.ndarray) or img.size == 0:
        return None

    img = cv2.resize(img, config.IMG_SIZE[::-1])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img / 255.0
    img = np.reshape(img, (*config.IMG_SIZE, 1))
    return img


train = []
train_cleaned = []
test = []

train_files = _list_image_files(path + "train/")
train_cleaned_files = _list_image_files(path + "train_cleaned/")
test_files = _list_image_files(path + "test/")

for f in train_files:
    img = process_image(path + "train/" + f)
    if img is None:
        continue
    train.append(img)

for f in train_cleaned_files:
    img = process_image(path + "train_cleaned/" + f)
    if img is None:
        continue
    train_cleaned.append(img)

for f in test_files:
    img = process_image(path + "test/" + f)
    if img is None:
        continue
    test.append(img)

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


## --- ERROR in cell 6, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mIndexError[0m                                Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/2340813209.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      1[0m [0mfig[0m[0;34m,[0m [0max[0m [0;34m=[0m [0mplt[0m[0;34m.[0m[0msubplots[0m[0;34m([0m[0;36m4[0m[0;34m,[0m [0;36m2[0m[0;34m,[0m [0mfigsize[0m[0;34m=[0m[0;34m([0m[0;36m15[0m[0;34m,[0m[0;36m25[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      2[0m [0;32mfor[0m [0mi[0m [0;32min[0m [0mrange[0m[0;34m([0m[0;36m4[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 3[0;31m     [0max[0m[0;34m[[0m[0mi[0m[0;34m][0m[0;34m[[0m[0;36m0[0m[0;34m][0m[0;34m.[0m[0mimshow[0m[0;34m([0m[0mtf[0m[0;34m.[0m[0msqueeze[0m[0;34m([0m[0mtrain[0m[0;34m[[0m[0mi[0m[0;34m][0m[0;34m)[0m[0;34m,[0m [0mcmap[0m[0;34m=[0m[0;34m'gray'[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      4[0m     [0max[0m[0;34m[[0m[0mi[0m[0;34m][0m[0;34m[[0m[0;36m0[0m[0;34m][0m[0;34m.[0m[0mset_title[0m[0;34m([0m[0;34m'Noise image: {}'[0m[0;34m.[0m[0mformat[0m[0;34m([0m[0mtrain_img[0m[0;34m[[0m[0mi[0m[0;34m][0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      5[0m [0;34m[0m[0m

[0;31mIndexError[0m: index 0 is out of bounds for axis 0 with size 0

## === cell 7
def augment_pipeline(pipeline, images, seed=19):
    ia.seed(seed)
    processed_images = images.copy()
    for step in pipeline:
        temp = np.array(step.augment_images(images))
        processed_images = np.append(processed_images, temp, axis=0)
    return(processed_images)
