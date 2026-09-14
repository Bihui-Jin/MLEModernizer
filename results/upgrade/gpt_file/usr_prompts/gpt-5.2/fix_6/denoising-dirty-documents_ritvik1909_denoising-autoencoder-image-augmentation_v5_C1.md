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
Given a dataset of images of scanned text that is noisy, remove the noise.

## Metric
Root mean squared error between the cleaned pixel intensities and the actual grayscale pixel intensities.

## Submission Format
Form the submission file by melting each images into a set of pixels, assigning each pixel an id of image_row_col (e.g. 1_2_1 is image 1, row 2, column 1). Intensity values range from 0 (black) to 1 (white). The file should contain a header and have the following format:

```
id,value
1_1_1,1
1_2_1,1
1_3_1,1
etc.
```

## Dataset
You are provided two sets of images, train and test. These images contain various styles of text, to which synthetic noise has been added to simulate real-world, messy artifacts. The training set includes the test without the noise (train_cleaned).

# 2. Python version

3.9

# 3. Installed packages

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

# 4. Data file paths

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

# 5. Target score

0.02838

# 6. Current score

0.28616

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 0.28616) has done: 'I fix the import-time crash coming from `imgaug` (it’s incompatible with the current `protobuf`/TF stack in this environment), which is why `iaa` never gets defined and all later augmentation/training cells fail. To keep the core training and model logic intact, I replace the `imgaug` augmentation pipeline with an equivalent, minimal NumPy-based augmentation that performs the same flips/90-degree rotations. I also add a small safety fix to ensure predictions are clipped to `[0, 1]` before writing the submission (score-neutral but prevents invalid values). Finally, I keep the same file paths and ensure `submission.csv` is produced with the required `id,value` columns.'
- What this solution (achieved 0.28616) has done: 'I fix the TensorFlow/protobuf import-time crash by forcing the pure-Python protobuf implementation before importing TensorFlow, which resolves the `MessageFactory.GetPrototype` error in this Kaggle environment. Then I fix the augmentation bug that swaps height/width during `np.rot90`, which currently causes shape mismatches and prevents training from running. After that, the training and inference run unchanged, and I keep the submission writing logic but ensure values remain clipped to `[0,1]` and the CSV is written as `submission.csv` with the required `id,value` columns. These changes are execution-unblocking and should materially improve the score versus the current broken/incorrect augmentation pipeline.'
- What this solution (achieved 0.28616) has done: 'I fix the TensorFlow/protobuf crash by switching to the C++ protobuf implementation (the `GetPrototype` AttributeError happens when TF ends up using the pure-Python protobuf runtime). Next, I fix the augmentation rotation bug: `np.rot90` swaps height/width for 90/270-degree rotations, so we must resize back to the expected `(420,540)` shape to keep the same training pipeline semantics and unblock training. Finally, I keep the model/training loop intact but ensure inference writes a correctly formatted `submission.csv` with values clipped to `[0,1]` (already present) and with robust id construction aligned to the original image shapes.'
- What this solution (achieved 0.28616) has done: 'I fix the TensorFlow/protobuf import-time crash by setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` before importing TensorFlow (this resolves the `MessageFactory.GetPrototype` error in this environment). Then I keep your existing data loading, augmentation, model, and training logic intact, only adding small robustness tweaks (safe directory creation and consistent path joining) that don’t change the learning semantics. Finally, I ensure the submission is always written as `submission.csv` with the required `id,value` columns and values clipped to `[0,1]` (already present) so Kaggle accepts it. These changes are execution-unblocking and should let the model actually train/infer properly, moving RMSE down toward your target.'
- What this solution (achieved 0.28616) has done: 'You’re currently crashing at import time because TensorFlow 2.18 expects a newer protobuf API (`MessageFactory.GetPrototype`), and forcing `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` triggers the incompatible pure-Python protobuf runtime. I remove that override and instead force the C++ protobuf runtime before importing TensorFlow, which fixes the import error while keeping the rest of your pipeline identical. Then I make sure the extracted data path is consistent (`/kaggle/working/` as you already use) and keep your augmentation/model/training logic unchanged. Finally, I keep your submission generation logic but add a small safety sort by numeric image id to ensure stable ordering and always write `submission.csv` with the required `id,value` columns.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "cpp"
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import zipfile, cv2
from tqdm.auto import tqdm
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras import layers, callbacks, utils

sns.set_style("darkgrid")

np.random.seed(19)
tf.random.set_seed(19)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/4265732877.py in <cell line: 0>()
     14 import zipfile, cv2
     15 from tqdm.auto import tqdm
---> 16 import tensorflow as tf
     17 
     18 from tensorflow.keras.models import Model

/usr/local/lib/python3.11/dist-packages/tensorflow/__init__.py in <module>
     47 _tf2.enable()
     48 
---> 49 from tensorflow._api.v2 import __internal__
     50 from tensorflow._api.v2 import __operators__
     51 from tensorflow._api.v2 import audio

/usr/local/lib/python3.11/dist-packages/tensorflow/_api/v2/__internal__/__init__.py in <module>
      6 import sys as _sys
      7 
----> 8 from tensorflow._api.v2.__internal__ import autograph
      9 from tensorflow._api.v2.__internal__ import decorator
     10 from tensorflow._api.v2.__internal__ import dispatch

/usr/local/lib/python3.11/dist-packages/tensorflow/_api/v2/__internal__/autograph/__init__.py in <module>
      6 import sys as _sys
      7 
----> 8 from tensorflow.python.autograph.core.ag_ctx import control_status_ctx # line: 34
      9 from tensorflow.python.autograph.impl.api import tf_convert # line: 493

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/core/ag_ctx.py in <module>
     19 import threading
     20 
---> 21 from tensorflow.python.autograph.utils import ag_logging
     22 from tensorflow.python.util.tf_export import tf_export
     23 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/utils/__init__.py in <module>
     15 """Utility module that contains APIs usable in the generated code."""
     16 
---> 17 from tensorflow.python.autograph.utils.context_managers import control_dependency_on_returns
     18 from tensorflow.python.autograph.utils.misc import alias_tensors
     19 from tensorflow.python.autograph.utils.tensor_list import dynamic_list_append

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/utils/context_managers.py in <module>
     17 import contextlib
     18 
---> 19 from tensorflow.python.framework import ops
     20 from tensorflow.python.ops import tensor_array_ops
     21 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py in <module>
     31 
     32 from google.protobuf import message
---> 33 from tensorflow.core.framework import attr_value_pb2
     34 from tensorflow.core.framework import full_type_pb2
     35 from tensorflow.core.framework import function_pb2

/usr/local/lib/python3.11/dist-packages/tensorflow/core/framework/attr_value_pb2.py in <module>
      3 # source: tensorflow/core/framework/attr_value.proto
      4 """Generated protocol buffer code."""
----> 5 from google.protobuf.internal import builder as _builder
      6 from google.protobuf import descriptor as _descriptor
      7 from google.protobuf import descriptor_pool as _descriptor_pool

/usr/local/lib/python3.11/dist-packages/google/protobuf/internal/builder.py in <module>
     16 
     17 from google.protobuf.internal import enum_type_wrapper
---> 18 from google.protobuf.internal import python_message
     19 from google.protobuf import message as _message
     20 from google.protobuf import reflection as _reflection

/usr/local/lib/python3.11/dist-packages/google/protobuf/internal/python_message.py in <module>
     36 import weakref
     37 
---> 38 from google.protobuf import descriptor as descriptor_mod
     39 from google.protobuf import message as message_mod
     40 from google.protobuf import text_format

/usr/local/lib/python3.11/dist-packages/google/protobuf/descriptor.py in <module>
     27   # TODO: Remove this import after fix api_implementation
     28   if _message is None:
---> 29     from google.protobuf.pyext import _message
     30   _USE_C_DESCRIPTORS = True
     31 

ImportError: cannot import name '_message' from 'google.protobuf.pyext' (/usr/local/lib/python3.11/dist-packages/google/protobuf/pyext/__init__.py)

## === cell 1
path_zip = "../input/denoising-dirty-documents/"
path = "/kaggle/working/"

os.makedirs(path, exist_ok=True)

with zipfile.ZipFile(os.path.join(path_zip, "train.zip"), "r") as zip_ref:
    zip_ref.extractall(path)

with zipfile.ZipFile(os.path.join(path_zip, "test.zip"), "r") as zip_ref:
    zip_ref.extractall(path)

with zipfile.ZipFile(os.path.join(path_zip, "train_cleaned.zip"), "r") as zip_ref:
    zip_ref.extractall(path)

with zipfile.ZipFile(
    os.path.join(path_zip, "sampleSubmission.csv.zip"), "r"
) as zip_ref:
    zip_ref.extractall(path)

train_img = sorted(os.listdir(os.path.join(path, "train")))
train_cleaned_img = sorted(os.listdir(os.path.join(path, "train_cleaned")))
test_img = sorted(os.listdir(os.path.join(path, "test")))




## === cell 2
class config:
    IMG_SIZE = (420, 540)


imgs = [
    cv2.imread(os.path.join(path, "train", f))
    for f in sorted(os.listdir(os.path.join(path, "train")))
]
print(
    "Median Dimensions:",
    np.median([len(img) for img in imgs]),
    np.median([len(img[0]) for img in imgs]),
)
del imgs




## === cell 3
def process_image(path_):
    img = cv2.imread(path_)
    img = np.asarray(img, dtype="float32")
    img = cv2.resize(img, config.IMG_SIZE[::-1])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img / 255.0
    img = np.reshape(img, (*config.IMG_SIZE, 1))
    return img




## === cell 4
train = []
train_cleaned = []
test = []

for f in sorted(os.listdir(os.path.join(path, "train"))):
    train.append(process_image(os.path.join(path, "train", f)))

for f in sorted(os.listdir(os.path.join(path, "train_cleaned"))):
    train_cleaned.append(process_image(os.path.join(path, "train_cleaned", f)))

for f in sorted(os.listdir(os.path.join(path, "test"))):
    test.append(process_image(os.path.join(path, "test", f)))

train = np.asarray(train)
train_cleaned = np.asarray(train_cleaned)
test = np.asarray(test)



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




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1446037899.py in <cell line: 0>()
      1 fig, ax = plt.subplots(4, 2, figsize=(15, 25))
      2 for i in range(4):
----> 3     ax[i][0].imshow(tf.squeeze(train[i]), cmap="gray")
      4     ax[i][0].set_title("Noise image: {}".format(train_img[i]))
      5 

NameError: name 'tf' is not defined

## === cell 7
def _resize_back(images, target_hw):
    out = np.empty(
        (images.shape[0], target_hw[0], target_hw[1], images.shape[-1]),
        dtype=images.dtype,
    )
    for i in range(images.shape[0]):
        out[i, ..., 0] = cv2.resize(
            images[i, ..., 0],
            (target_hw[1], target_hw[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    return out


def _rot90(images, k):
    rotated = np.rot90(images, k=k, axes=(1, 2)).copy()
    if rotated.shape[1] != config.IMG_SIZE[0] or rotated.shape[2] != config.IMG_SIZE[1]:
        rotated = _resize_back(rotated, config.IMG_SIZE)
    return rotated


def _hflip(images):
    return np.flip(images, axis=2).copy()  # flip width


def _vflip(images):
    return np.flip(images, axis=1).copy()  # flip height


def augment_pipeline(pipeline, images, seed=19):
    processed_images = images.copy()
    for step in pipeline:
        temp = np.asarray(step(images), dtype=processed_images.dtype)
        if temp.shape[1:] != processed_images.shape[1:]:
            raise ValueError(
                f"Augmentation produced shape {temp.shape} but expected {processed_images.shape}"
            )
        processed_images = np.concatenate([processed_images, temp], axis=0)
    return processed_images




## === cell 8
rotate90 = lambda x: _rot90(x, 1)
rotate180 = lambda x: _rot90(x, 2)
rotate270 = lambda x: _rot90(x, 3)
hflip = lambda x: _hflip(x)
vflip = lambda x: _vflip(x)



## === cell 9
pipeline = [rotate90, rotate180, rotate270, hflip, vflip]



## === cell 10
processed_train = augment_pipeline(pipeline, train)
processed_train_cleaned = augment_pipeline(pipeline, train_cleaned)

processed_train.shape, processed_train_cleaned.shape




## === cell 11
class DenoisingAutoencoder(Model):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        self.encoder = tf.keras.Sequential(
            [
                layers.Input(shape=(*config.IMG_SIZE, 1)),
                layers.Conv2D(96, (3, 3), activation="relu", padding="same"),
                layers.Conv2D(192, (3, 3), activation="relu", padding="same"),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2), padding="same"),
                layers.Dropout(0.5),
            ]
        )

        self.decoder = tf.keras.Sequential(
            [
                layers.Conv2D(192, (3, 3), activation="relu", padding="same"),
                layers.Conv2D(96, (3, 3), activation="relu", padding="same"),
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



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1724573819.py in <cell line: 0>()
----> 1 class DenoisingAutoencoder(Model):
      2     def __init__(self):
      3         super(DenoisingAutoencoder, self).__init__()
      4         self.encoder = tf.keras.Sequential(
      5             [

NameError: name 'Model' is not defined

## === cell 12
es = callbacks.EarlyStopping(
    monitor="loss", patience=30, verbose=1, restore_best_weights=True
)

history = autoencoder.fit(
    processed_train,
    processed_train_cleaned,
    shuffle=True,
    callbacks=[es],
    epochs=500,
    batch_size=12,
)



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3245784305.py in <cell line: 0>()
----> 1 es = callbacks.EarlyStopping(
      2     monitor="loss", patience=30, verbose=1, restore_best_weights=True
      3 )
      4 
      5 history = autoencoder.fit(

NameError: name 'callbacks' is not defined

## === cell 13
fig, ax = plt.subplots(figsize=(20, 6))
pd.DataFrame(history.history).plot(ax=ax)
del history



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2278981582.py in <cell line: 0>()
      1 fig, ax = plt.subplots(figsize=(20, 6))
----> 2 pd.DataFrame(history.history).plot(ax=ax)
      3 del history
      4 

NameError: name 'history' is not defined

## === cell 14
autoencoder.encoder.summary()
autoencoder.decoder.summary()



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2619533403.py in <cell line: 0>()
----> 1 autoencoder.encoder.summary()
      2 autoencoder.decoder.summary()
      3 

NameError: name 'autoencoder' is not defined

## === cell 15
decoded_imgs = autoencoder(train[:4]).numpy()

fig, ax = plt.subplots(4, 2, figsize=(15, 25))
for i in range(4):
    ax[i][0].imshow(tf.squeeze(train_cleaned[i]), cmap="gray")
    ax[i][0].set_title("Denoised image: {}".format(train_img[i]))

    ax[i][1].imshow(tf.squeeze(decoded_imgs[i]), cmap="gray")
    ax[i][1].set_title("Predicted image: {}".format(train_img[i]))

    ax[i][0].get_xaxis().set_visible(False)
    ax[i][0].get_yaxis().set_visible(False)
    ax[i][1].get_xaxis().set_visible(False)
    ax[i][1].get_yaxis().set_visible(False)

del decoded_imgs



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1481402213.py in <cell line: 0>()
----> 1 decoded_imgs = autoencoder(train[:4]).numpy()
      2 
      3 fig, ax = plt.subplots(4, 2, figsize=(15, 25))
      4 for i in range(4):
      5     ax[i][0].imshow(tf.squeeze(train_cleaned[i]), cmap="gray")

NameError: name 'autoencoder' is not defined

## === cell 16
test_img = sorted(test_img, key=lambda x: int(os.path.splitext(x)[0]))

ids = []
vals = []
for i, f in tqdm(list(enumerate(test_img))):
    file = os.path.join(path, "test", f)
    imgid = int(f[:-4])
    img = cv2.imread(file, 0)
    img_shape = img.shape  # (H,W)

    decoded_img = autoencoder(test[i : i + 1], training=False).numpy()
    decoded_img = np.squeeze(decoded_img)  # (420,540)

    preds_reshaped = cv2.resize(
        decoded_img, (img_shape[1], img_shape[0]), interpolation=cv2.INTER_LINEAR
    )
    preds_reshaped = np.clip(preds_reshaped, 0.0, 1.0)

    for r in range(img_shape[0]):
        for c in range(img_shape[1]):
            ids.append(str(imgid) + "_" + str(r + 1) + "_" + str(c + 1))
            vals.append(float(preds_reshaped[r, c]))

print("Length of IDs: {}".format(len(ids)))
pd.DataFrame({"id": ids, "value": vals}).to_csv("submission.csv", index=False)
print("Results saved to submission.csv!")

## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3944408070.py in <cell line: 0>()
     10     img_shape = img.shape  # (H,W)
     11 
---> 12     decoded_img = autoencoder(test[i : i + 1], training=False).numpy()
     13     decoded_img = np.squeeze(decoded_img)  # (420,540)
     14 

NameError: name 'autoencoder' is not defined
