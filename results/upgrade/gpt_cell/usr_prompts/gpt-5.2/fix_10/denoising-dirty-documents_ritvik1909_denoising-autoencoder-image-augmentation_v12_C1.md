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
    from google.protobuf import message_factory as _message_factory

    if hasattr(_message_factory, "MessageFactory") and not hasattr(
        _message_factory.MessageFactory, "GetPrototype"
    ):

        def _GetPrototype(self, descriptor):
            return self.GetMessageClass(descriptor)

        _message_factory.MessageFactory.GetPrototype = _GetPrototype
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

ia = None


class _ImgAugUnavailable:
    def __getattr__(self, name):
        raise ImportError(
            "imgaug could not be imported due to a protobuf incompatibility in this environment. "
            "Augmentation is unavailable unless the environment is changed."
        )


iaa = _ImgAugUnavailable()

sns.set_style("darkgrid")

SEED = 19
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass



## === cell 1
path_zip = "../input/denoising-dirty-documents/"
path = "/kaggle/working/"


def _ensure_extracted(zip_path, out_dir, sentinel_dir):
    if os.path.isdir(sentinel_dir) and len(os.listdir(sentinel_dir)) > 0:
        return
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)


_ensure_extracted(path_zip + "train.zip", path, os.path.join(path, "train"))
_ensure_extracted(path_zip + "test.zip", path, os.path.join(path, "test"))
_ensure_extracted(
    path_zip + "train_cleaned.zip", path, os.path.join(path, "train_cleaned")
)
_ensure_extracted(
    path_zip + "sampleSubmission.csv.zip",
    path,
    os.path.join(path, "sampleSubmission.csv"),
)

train_img = sorted(os.listdir(path + "/train"))
train_cleaned_img = sorted(os.listdir(path + "/train_cleaned"))
test_img = sorted(os.listdir(path + "/test"))




## === cell 2
class config:
    IMG_SIZE = (420, 540)


imgs = [cv2.imread(path + "train/" + f) for f in sorted(os.listdir(path + "train/"))]
print(
    "Median Dimensions:",
    np.median([len(img) for img in imgs]),
    np.median([len(img[0]) for img in imgs]),
)
del imgs




## === cell 3
def process_image(path_):
    img = cv2.imread(path_, cv2.IMREAD_GRAYSCALE)  # (H,W), uint8
    img = cv2.resize(
        img, config.IMG_SIZE[::-1]
    )  # (540,420) -> (420,540) after resize target
    img = img.astype("float32") / 255.0
    img = np.reshape(img, (*config.IMG_SIZE, 1))
    return img




## === cell 4
train_files = sorted(os.listdir(path + "train/"))
train_cleaned_files = sorted(os.listdir(path + "train_cleaned/"))
test_files = sorted(os.listdir(path + "test/"))

n_train = len(train_files)
n_train_cleaned = len(train_cleaned_files)
n_test = len(test_files)

train = np.empty((n_train, *config.IMG_SIZE, 1), dtype=np.float32)
train_cleaned = np.empty((n_train_cleaned, *config.IMG_SIZE, 1), dtype=np.float32)
test = np.empty((n_test, *config.IMG_SIZE, 1), dtype=np.float32)

for i, f in enumerate(train_files):
    train[i] = process_image(path + "train/" + f)

for i, f in enumerate(train_cleaned_files):
    train_cleaned[i] = process_image(path + "train_cleaned/" + f)

for i, f in enumerate(test_files):
    test[i] = process_image(path + "test/" + f)



## === cell 5
train.shape, train_cleaned.shape, test.shape



## === cell 6
fig, ax = plt.subplots(4, 2, figsize=(15, 25))
for i in range(4):
    ax[i][0].imshow(tf.squeeze(train[i]), cmap="gray")
    ax[i][0].set_title("Noise image: {}".format(train_img[i]))

    ax[i][1].imshow(tf.squeeze(train_cleaned[i]), cmap="gray")
    ax[i][1].set_title("Denoised image: {}".format(train_img[i]))

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
    return processed_images




## === cell 8
try:
    _ = iaa.Rot90  # will raise ImportError via _ImgAugUnavailable.__getattr__
    _IMG_AUG_AVAILABLE = True
except Exception:
    _IMG_AUG_AVAILABLE = False

if not _IMG_AUG_AVAILABLE:
    class _NP_Rot90:
        def __init__(self, k):
            self.k = int(k)

        def augment_images(self, images):
            return np.rot90(images, k=self.k, axes=(1, 2)).copy()

    class _NP_Fliplr:
        def __init__(self, p=1.0):
            self.p = float(p)

        def augment_images(self, images):
            if self.p >= 1.0:
                return images[:, :, ::-1, :].copy()
            rng = np.random.RandomState(SEED)
            out = images.copy()
            m = rng.rand(images.shape[0]) < self.p
            out[m] = out[m, :, ::-1, :]
            return out

    class _NP_Flipud:
        def __init__(self, p=1.0):
            self.p = float(p)

        def augment_images(self, images):
            if self.p >= 1.0:
                return images[:, ::-1, :, :].copy()
            rng = np.random.RandomState(SEED)
            out = images.copy()
            m = rng.rand(images.shape[0]) < self.p
            out[m] = out[m, ::-1, :, :]
            return out

    rotate90 = _NP_Rot90(1)
    rotate180 = _NP_Rot90(2)
    rotate270 = _NP_Rot90(3)

    random_rotate = _NP_Rot90(1)
    perc_transform = _NP_Rot90(0)
    rotate10 = _NP_Rot90(0)
    rotate10r = _NP_Rot90(0)
    crop = _NP_Rot90(0)

    hflip = _NP_Fliplr(1.0)
    vflip = _NP_Flipud(1.0)

    gblur = _NP_Rot90(0)
    motionblur = _NP_Rot90(0)

    class _NoOpSequential:
        def __init__(self, *args, **kwargs):
            pass

        def augment_images(self, images):
            return np.array(images, copy=True)

    seq_rp = _NoOpSequential()
    seq_cfg = _NoOpSequential()
    seq_fm = _NoOpSequential()
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
pipeline = [rotate90, rotate180, rotate270, hflip, vflip]




## === cell 10
def augment_pipeline(pipeline, images, seed=19):
    if ia is not None and hasattr(ia, "seed"):
        ia.seed(seed)
    n = images.shape[0]
    k = len(pipeline)
    out = np.empty((n * (k + 1),) + images.shape[1:], dtype=images.dtype)
    out[:n] = images
    write = n
    for step in pipeline:
        temp = np.asarray(step.augment_images(images), dtype=images.dtype)
        out[write : write + n] = temp
        write += n
    return out


processed_train = augment_pipeline(pipeline, train, seed=SEED)
processed_train_cleaned = augment_pipeline(pipeline, train_cleaned, seed=SEED)

processed_train.shape, processed_train_cleaned.shape




## --- ERROR in cell 10, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mValueError[0m                                Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/2017471364.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     14[0m [0;34m[0m[0m
[1;32m     15[0m [0;34m[0m[0m
[0;32m---> 16[0;31m [0mprocessed_train[0m [0;34m=[0m [0maugment_pipeline[0m[0;34m([0m[0mpipeline[0m[0;34m,[0m [0mtrain[0m[0;34m,[0m [0mseed[0m[0;34m=[0m[0mSEED[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     17[0m [0mprocessed_train_cleaned[0m [0;34m=[0m [0maugment_pipeline[0m[0;34m([0m[0mpipeline[0m[0;34m,[0m [0mtrain_cleaned[0m[0;34m,[0m [0mseed[0m[0;34m=[0m[0mSEED[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     18[0m [0;34m[0m[0m

[0;32m/tmp/ipykernel_11/2017471364.py[0m in [0;36maugment_pipeline[0;34m(pipeline, images, seed)[0m
[1;32m      9[0m     [0;32mfor[0m [0mstep[0m [0;32min[0m [0mpipeline[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m     10[0m         [0mtemp[0m [0;34m=[0m [0mnp[0m[0;34m.[0m[0masarray[0m[0;34m([0m[0mstep[0m[0;34m.[0m[0maugment_images[0m[0;34m([0m[0mimages[0m[0;34m)[0m[0;34m,[0m [0mdtype[0m[0;34m=[0m[0mimages[0m[0;34m.[0m[0mdtype[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 11[0;31m         [0mout[0m[0;34m[[0m[0mwrite[0m [0;34m:[0m [0mwrite[0m [0;34m+[0m [0mn[0m[0;34m][0m [0;34m=[0m [0mtemp[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     12[0m         [0mwrite[0m [0;34m+=[0m [0mn[0m[0;34m[0m[0;34m[0m[0m
[1;32m     13[0m     [0;32mreturn[0m [0mout[0m[0;34m[0m[0;34m[0m[0m

[0;31mValueError[0m: could not broadcast input array from shape (115,540,420,1) into shape (115,420,540,1)

## === cell 11
class DenoisingAutoencoder(Model):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        self.encoder = tf.keras.Sequential(
            [
                layers.Input(shape=(*config.IMG_SIZE, 1)),
                layers.Conv2D(48, (3, 3), activation="relu", padding="same"),
                layers.Conv2D(72, (3, 3), activation="relu", padding="same"),
                layers.Conv2D(144, (3, 3), activation="relu", padding="same"),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2), padding="same"),
                layers.Dropout(0.5),
            ]
        )

        self.decoder = tf.keras.Sequential(
            [
                layers.Conv2D(144, (3, 3), activation="relu", padding="same"),
                layers.Conv2D(72, (3, 3), activation="relu", padding="same"),
                layers.Conv2D(48, (3, 3), activation="relu", padding="same"),
                layers.BatchNormalization(),
                layers.UpSampling2D((2, 2)),
                layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same"),
            ]
        )

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


autoencoder = DenoisingAutoencoder()
autoencoder.compile(
    optimizer="adam", loss="mean_squared_error", metrics=["mean_absolute_error"]
)
