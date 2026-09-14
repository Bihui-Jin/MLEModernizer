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
def _find_extracted_root(base="/kaggle/working/"):
    candidates = [
        base,
        os.path.join(base, "denoising-dirty-documents"),
        os.path.join(base, "denoising-dirty-documents", "denoising-dirty-documents"),
    ]
    for cand in candidates:
        if all(
            os.path.isdir(os.path.join(cand, d)) for d in ("train", "train_cleaned")
        ):
            return cand if cand.endswith(os.sep) else cand + os.sep
    return base if base.endswith(os.sep) else base + os.sep


root = _find_extracted_root(path if isinstance(path, str) else "/kaggle/working/")
train_dir = os.path.join(root, "train")
train_cleaned_dir = os.path.join(root, "train_cleaned")

train_files_for_plot = _list_image_files(train_dir) if os.path.isdir(train_dir) else []
train_cleaned_files_for_plot = (
    _list_image_files(train_cleaned_dir) if os.path.isdir(train_cleaned_dir) else []
)

n_show = min(4, len(train_files_for_plot), len(train_cleaned_files_for_plot))
if n_show == 0:
    raise ValueError(
        "No training images available to plot. Checked directories:\n"
        f"- {train_dir}\n- {train_cleaned_dir}\n"
        "This typically means image extraction/loading failed or paths are incorrect."
    )

fig, ax = plt.subplots(n_show, 2, figsize=(15, 6 * n_show))
if n_show == 1:
    ax = np.array([ax])

for i in range(n_show):
    noisy_fp = os.path.join(train_dir, train_files_for_plot[i])
    clean_fp = os.path.join(train_cleaned_dir, train_cleaned_files_for_plot[i])

    noisy_img = process_image(noisy_fp)
    clean_img = process_image(clean_fp)
    if noisy_img is None or clean_img is None:
        continue

    ax[i][0].imshow(tf.squeeze(noisy_img), cmap="gray")
    ax[i][0].set_title("Noise image: {}".format(train_files_for_plot[i]))

    ax[i][1].imshow(tf.squeeze(clean_img), cmap="gray")
    ax[i][1].set_title("Denoised image: {}".format(train_cleaned_files_for_plot[i]))

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
    try:
        import imgaug as ia  # noqa: F401
        from imgaug import augmenters as iaa  # noqa: F401
    except Exception:
        ia = None
        iaa = None

if iaa is None:

    class _NoOpAugmenter:
        def augment_images(self, images):
            return images

    def _noop(*args, **kwargs):
        return _NoOpAugmenter()

    rotate90 = _noop()
    rotate180 = _noop()
    rotate270 = _noop()
    random_rotate = _noop()
    perc_transform = _noop()
    rotate10 = _noop()
    rotate10r = _noop()
    crop = _noop()
    hflip = _noop()
    vflip = _noop()
    gblur = _noop()
    motionblur = _noop()

    seq_rp = _noop()
    seq_cfg = _noop()
    seq_fm = _noop()
else:
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


## === cell 9
pipeline = [
    rotate90, rotate180, rotate270, hflip, vflip
]


## === cell 10
processed_train = augment_pipeline(pipeline, train)
processed_train_cleaned = augment_pipeline(pipeline, train_cleaned)

processed_train.shape, processed_train_cleaned.shape


## --- ERROR in cell 10, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mAttributeError[0m                            Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/3350348689.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[0;32m----> 1[0;31m [0mprocessed_train[0m [0;34m=[0m [0maugment_pipeline[0m[0;34m([0m[0mpipeline[0m[0;34m,[0m [0mtrain[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      2[0m [0mprocessed_train_cleaned[0m [0;34m=[0m [0maugment_pipeline[0m[0;34m([0m[0mpipeline[0m[0;34m,[0m [0mtrain_cleaned[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      3[0m [0;34m[0m[0m
[1;32m      4[0m [0mprocessed_train[0m[0;34m.[0m[0mshape[0m[0;34m,[0m [0mprocessed_train_cleaned[0m[0;34m.[0m[0mshape[0m[0;34m[0m[0;34m[0m[0m

[0;32m/tmp/ipykernel_11/2255021665.py[0m in [0;36maugment_pipeline[0;34m(pipeline, images, seed)[0m
[1;32m      1[0m [0;32mdef[0m [0maugment_pipeline[0m[0;34m([0m[0mpipeline[0m[0;34m,[0m [0mimages[0m[0;34m,[0m [0mseed[0m[0;34m=[0m[0;36m19[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 2[0;31m     [0mia[0m[0;34m.[0m[0mseed[0m[0;34m([0m[0mseed[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      3[0m     [0mprocessed_images[0m [0;34m=[0m [0mimages[0m[0;34m.[0m[0mcopy[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      4[0m     [0;32mfor[0m [0mstep[0m [0;32min[0m [0mpipeline[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m      5[0m         [0mtemp[0m [0;34m=[0m [0mnp[0m[0;34m.[0m[0marray[0m[0;34m([0m[0mstep[0m[0;34m.[0m[0maugment_images[0m[0;34m([0m[0mimages[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;31mAttributeError[0m: 'NoneType' object has no attribute 'seed'

## === cell 11
class DenoisingAutoencoder(Model):
  def __init__(self):
    super(DenoisingAutoencoder, self).__init__()
    self.encoder = tf.keras.Sequential([
        layers.Input(shape=(*config.IMG_SIZE, 1)), 
        layers.Conv2D(48, (5, 5), activation='relu', padding='same'),
        layers.Conv2D(72, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(144, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2), padding='same'),
        layers.Dropout(0.5),
    ])

    self.decoder = tf.keras.Sequential([
        layers.Conv2D(144, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(72, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(48, (5, 5), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.UpSampling2D((2, 2)),
        layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')
    ])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

autoencoder = DenoisingAutoencoder()
autoencoder.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])
