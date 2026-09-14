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
Given a dataset of images of dogs, predict the breed of each image.

## Metric
Multi Class Log Loss.

## Submission Format
For each image in the test set, you must predict a probability for each of the different breeds. The file should contain a header and have the following format:
```
id,affenpinscher,afghan_hound,..,yorkshire_terrier
000621fb3cbb32d8935728e48679680e,0.0083,0.0,...,0.0083
etc.
```

## Dataset Description
- `train.zip` - the training set, you are provided the breed for these dogs
- `test.zip` - the test set, you must predict the probability of each breed for each image
- `sample_submission.csv` - a sample submission file in the correct format
- `labels.csv` - the breeds for the images in the train set

# 2. Python version

3.8

# 3. Installed packages

geopandas==0.14.4
google-api-python-client==2.177.0
ipython==7.34.0
ipython-genutils==0.2.0
ipython_pygments_lexers==1.1.1
ipython-sql==0.5.0
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
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
seaborn==0.12.2
sklearn-pandas==2.2.0
tf_keras==2.18.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (169 lines)
            labels.csv (9200 lines)
            labels.csv.zip (201.6 kB)
            sample_submission.csv (1024 lines)
            sample_submission.csv.zip (32.1 kB)
            test.zip (36.4 MB)
            train.zip (324.7 MB)
            dog-breed-identification/
                description.md (169 lines)
                labels.csv (9200 lines)
                ... and 5 other files
                dog-breed-identification/
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
            test/
                bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                ... and 1021 other files
                test/
            train/
                868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                ... and 9197 other files
                train/
        input/
            description.md (169 lines)
            labels.csv (9200 lines)
            labels.csv.zip (201.6 kB)
            sample_submission.csv (1024 lines)
            sample_submission.csv.zip (32.1 kB)
            test.zip (36.4 MB)
            train.zip (324.7 MB)
            dog-breed-identification/
                description.md (169 lines)
                labels.csv (9200 lines)
                ... and 5 other files
                dog-breed-identification/
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
            test/
                bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                ... and 1021 other files
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
            train/
                868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                ... and 9197 other files
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
        working/
            dog-breed-identification/
                description.md (169 lines)
                labels.csv (9200 lines)
                ... and 5 other files
                dog-breed-identification/
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
```

-> data/dog-breed-identification/labels.csv has 9199 rows and 2 columns.
The columns are: id, breed

-> data/dog-breed-identification/sample_submission.csv has 1023 rows and 121 columns.
The columns are: id, affenpinscher, afghan_hound, african_hunting_dog, airedale, american_staffordshire_terrier, appenzeller, australian_terrier, basenji, basset, beagle, bedlington_terrier, bernese_mountain_dog, black-and-tan_coonhound, blenheim_spaniel... and 106 more columns

-> data/labels.csv has 9199 rows and 2 columns.
The columns are: id, breed

-> data/sample_submission.csv has 1023 rows and 121 columns.
The columns are: id, affenpinscher, afghan_hound, african_hunting_dog, airedale, american_staffordshire_terrier, appenzeller, australian_terrier, basenji, basset, beagle, bedlington_terrier, bernese_mountain_dog, black-and-tan_coonhound, blenheim_spaniel... and 106 more columns

-> input/dog-breed-identification/labels.csv has 9199 rows and 2 columns.
The columns are: id, breed

-> input/dog-breed-identification/sample_submission.csv has 1023 rows and 121 columns.
The columns are: id, affenpinscher, afghan_hound, african_hunting_dog, airedale, american_staffordshire_terrier, appenzeller, australian_terrier, basenji, basset, beagle, bedlington_terrier, bernese_mountain_dog, black-and-tan_coonhound, blenheim_spaniel... and 106 more columns

-> (stopped after 10 files for performance)

# 5. Target score

0.28106

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 5.06376) has done: 'The timeout is dominated by two things: (1) copying ~9k images into 3 directory trees (very slow filesystem I/O), and (2) training/evaluating/predicting at 512×512 with Python-based generators without any input pipelining. To keep the exact same model and training semantics, the main speed fix is to avoid copying by creating the same directory structure but using hardlinks/symlinks (falling back to copy only if linking fails), and to ensure the split is deterministic and done once. Then we speed up the input pipeline by enabling generator multiprocessing/workers in `fit/evaluate/predict` (doesn’t change the algorithm, only parallelizes loading/augmentation). Finally, we remove expensive per-epoch plotting/`clear_output` overhead from the training loop (the training itself is unchanged; only visualization is deferred to the end).'
- What this solution (achieved 5.06378) has done: 'I fix two blockers that prevent the notebook from running end-to-end: the protobuf/Keras import crash in the first cell and the broken train/valid/test directory split that creates links to paths that don’t exist (causing `FileNotFoundError` during `flow_from_directory`). The split bug comes from using only the `.jpg` filename while the actual source directory is organized by breed subfolders, so the code must build correct source paths per breed. These changes are score-neutral by themselves (they restore the intended training data pipeline), but they should also move the logloss substantially toward the target by ensuring the model is actually trained on real images instead of failing mid-epoch. Finally, I keep the model/training logic intact and ensure a valid `submission.csv` is always written in `/kaggle/working/`.'
- What this solution (achieved 5.06377) has done: 'I fix the two root causes preventing the model from training: (1) the protobuf crash caused by forcing the pure-Python protobuf implementation, and (2) incorrect train image paths because this dataset’s `/train/` directory is flat (no breed subfolders), so the split builder must link/copy from `/train/<id>.jpg`. Then I make the split deterministic and stable (per-image RNG derived from the image id) so that you always get non-empty train/valid/test directories and `steps_per_epoch` is positive. These changes keep the same model/training logic intact but should drastically reduce logloss toward the target because the model finally train and produce meaningful probabilities. Finally, I ensure a valid `/kaggle/working/submission.csv` is always written with columns exactly matching `sample_submission.csv`.'
- What this solution (achieved 5.06249) has done: 'I remove the expensive per-breed directory split/linking step and instead build the same train/valid split directly from `labels.csv` using the exact same deterministic CRC32 rule, which preserves the core split semantics while eliminating thousands of filesystem operations. I also remove `tf.py_function` and `ImageDataGenerator` from the `tf.data` pipeline and replace them with equivalent TensorFlow-native augmentation ops (same transforms: rotate, shifts, horizontal flip, rescale), which keeps the model and training loop identical but makes input processing vectorized and parallelizable. Finally, I cache and optimize the validation/test input pipelines (no augmentation) and ensure deterministic behavior via explicit seeds in the stateless random ops.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "cpp"
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import shutil
import numpy as np
import pandas as pd
import cv2

import seaborn as sns
import matplotlib.pyplot as plt

import tensorflow as tf  # must import TF before tf_keras in this environment

import tf_keras as keras
from tf_keras import backend as K
from tf_keras.layers import (
    Dense,
    Activation,
    Dropout,
    BatchNormalization,
    Input,
    Flatten,
    MaxPooling2D,
)
from tf_keras.models import Model
from tf_keras.optimizers import Adam
from tf_keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from tf_keras.applications.inception_resnet_v2 import InceptionResNetV2
from tf_keras.initializers import he_normal
from tf_keras.preprocessing.image import ImageDataGenerator

np.random.seed(42)
tf.random.set_seed(42)
try:
    keras.utils.set_random_seed(42)
except Exception:
    pass

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    cpu_cnt = os.cpu_count() or 2
    tf.config.threading.set_intra_op_parallelism_threads(min(8, cpu_cnt))
    tf.config.threading.set_inter_op_parallelism_threads(min(4, cpu_cnt))
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

print("Imports OK")
print("TF version:", tf.__version__)
print("CPU count:", os.cpu_count())



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/503901070.py in <cell line: 0>()
     14 import matplotlib.pyplot as plt
     15 
---> 16 import tensorflow as tf  # must import TF before tf_keras in this environment
     17 
     18 import tf_keras as keras

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
train_path = "../input/dog-breed-identification/train"
print("Train dir exists:", os.path.isdir(train_path))
print("First 5 files/entries:", sorted(os.listdir(train_path))[:5])



## === cell 2
labels = pd.read_csv("../input/dog-breed-identification/labels.csv")
labels.head(5)



## === cell 3
classes = np.unique(labels.breed)
classes_num = classes.size
classes_num



## === cell 4
train_dir = "../input/dog-breed-identification/train"
images_num = labels.shape[0]
print(f"Number of images (from labels.csv): {images_num}")



## === cell 5
new_train_dir = "/kaggle/working/new_train/"  # kept for compatibility with later prints
new_test_dir = "/kaggle/working/new_test/"
new_valid_dir = "/kaggle/working/new_valid/"
print(
    "Skipping creation of new_train/new_valid/new_test directories (not used by pipeline)."
)
print(new_train_dir, new_test_dir, new_valid_dir)



## === cell 6
print("Skipping per-class subdir creation (not used). Classes:", len(classes))



## === cell 7
labels_jpg = labels.copy(deep=True)
labels_jpg["id"] += ".jpg"  # add .jpg to each image id to get its filename

grouped_ids = labels_jpg.groupby("breed")["id"].apply(list).to_dict()
print(classes[0], grouped_ids[classes[0]][:5])



## === cell 8
test_split = 0.1
valid_split = 0.2



## === cell 9
import zlib


def _stable_rand01_from_str(s: str) -> float:
    v = zlib.crc32(s.encode("utf-8")) & 0xFFFFFFFF
    return v / 4294967296.0


train_size = 0
valid_size = 0
test_size = 0

for breed, breed_images in grouped_ids.items():
    for img in breed_images:
        rnd_prob = _stable_rand01_from_str(img)
        if rnd_prob <= test_split:
            test_size += 1
        elif rnd_prob <= (test_split + valid_split):
            valid_size += 1
        else:
            train_size += 1

print("Skipping on-disk split (building datasets from file lists instead).")
print("Split sizes:", train_size, valid_size, test_size)

if train_size == 0 or valid_size == 0:
    raise RuntimeError(
        f"Invalid split produced empty directory: train_size={train_size}, valid_size={valid_size}. "
        "Check train_dir path and split logic."
    )



## === cell 10
test_breed = classes[0]
print("Example breed:", test_breed)
print(
    "First 5 files in original train dir:",
    sorted([f for f in os.listdir(train_dir) if f.lower().endswith(".jpg")])[:5],
)



## === cell 11
width, height, channels = 512, 512, 3



## === cell 12
images_samples = np.zeros((4, height, width, 3), dtype=np.float32)
samples_labels = []

rnd_breeds = np.random.choice(classes, size=4, replace=True)
for i, b in enumerate(rnd_breeds):
    breed_ids = grouped_ids[b]
    img_filename = np.random.choice(breed_ids)
    img_path = os.path.join(train_dir, img_filename)
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        ok = False
        for _ in range(min(10, len(breed_ids))):
            img_filename = np.random.choice(breed_ids)
            img_path = os.path.join(train_dir, img_filename)
            img_bgr = cv2.imread(img_path)
            if img_bgr is not None:
                ok = True
                break
        if not ok:
            raise RuntimeError(
                f"Failed to read any image for breed '{b}' from {train_dir}"
            )

    img_rgb = img_bgr[:, :, [2, 1, 0]]
    images_samples[i] = (
        cv2.resize(src=img_rgb, dsize=(width, height)).astype(np.float32) / 255.0
    )
    samples_labels.append(b)



## === cell 13
DO_PLOTS = False

if DO_PLOTS:
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    for ax, img, label in zip(axs.ravel(), images_samples, samples_labels):
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(f"Class: {label}", size=15)
    plt.show()



## === cell 14
norm_factor = 1 / 255

transform_params = {
    "featurewise_center": False,
    "featurewise_std_normalization": False,
    "samplewise_center": False,
    "samplewise_std_normalization": False,
    "rotation_range": 30,
    "width_shift_range": 0.15,
    "height_shift_range": 0.15,
    "horizontal_flip": True,
    "rescale": norm_factor,
}

img_gen = ImageDataGenerator(**transform_params)



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2261100777.py in <cell line: 0>()
     13 }
     14 
---> 15 img_gen = ImageDataGenerator(**transform_params)
     16 

NameError: name 'ImageDataGenerator' is not defined

## === cell 15
img_feed = ImageDataGenerator(rescale=1 / 255)



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1566107731.py in <cell line: 0>()
----> 1 img_feed = ImageDataGenerator(rescale=1 / 255)
      2 

NameError: name 'ImageDataGenerator' is not defined

## === cell 16
if DO_PLOTS:
    fig, axs = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle("Augmentation Results", size=32)

    for axs_col, img in enumerate(images_samples):
        viz_transoform_params = {
            "theta": np.random.randint(
                -transform_params["rotation_range"], transform_params["rotation_range"]
            ),
            "tx": np.random.uniform(
                -transform_params["width_shift_range"],
                transform_params["width_shift_range"],
            ),
            "ty": np.random.uniform(
                -transform_params["height_shift_range"],
                transform_params["height_shift_range"],
            ),
            "flip_horizontal": np.random.choice([True, False], p=[0.5, 0.5]),
        }
        aug_img = img_gen.apply_transform(img, viz_transoform_params)

        axs[0, axs_col].imshow(img)
        axs[0, axs_col].axis("off")
        axs[0, axs_col].set_title("Original Image", size=15)

        axs[1, axs_col].imshow(aug_img)
        axs[1, axs_col].axis("off")
        axs[1, axs_col].set_title("Augmented Image", size=15)

    plt.show()




## === cell 17
class Plotter(Callback):
    def plot(self):
        if not DO_PLOTS:
            if len(self.epochs) > 0:
                print(
                    f"Epoch #{self.epochs[-1]+1} >> train_acc={self.acc[-1]*100:.3f}%, train_loss={self.losses[-1]:.5f}"
                )
                print(
                    f"Epoch #{self.epochs[-1]+1} >> val_acc={self.val_acc[-1]*100:.3f}%, val_loss={self.val_losses[-1]:.5f}"
                )
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        ax1.plot(self.epochs, self.losses, label="train_loss")
        ax1.plot(self.epochs, self.val_losses, label="val_loss")

        ax2.plot(self.epochs, self.acc, label="train_acc")
        ax2.plot(self.epochs, self.val_acc, label="val_acc")

        ax1.set_title("Loss vs Epochs")
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Loss")

        ax2.set_title("Accuracy vs Epochs")
        ax2.set_xlabel("Epochs")
        ax2.set_ylabel("Accuracy")

        ax1.legend()
        ax2.legend()
        plt.show()

        if len(self.epochs) > 0:
            print(
                f"Epoch #{self.epochs[-1]+1} >> train_acc={self.acc[-1]*100:.3f}%, train_loss={self.losses[-1]:.5f}"
            )
            print(
                f"Epoch #{self.epochs[-1]+1} >> val_acc={self.val_acc[-1]*100:.3f}%, val_loss={self.val_losses[-1]:.5f}"
            )

    def on_train_begin(self, logs=None):
        self.losses = []
        self.val_losses = []
        self.epochs = []
        self.acc = []
        self.val_acc = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.losses.append(logs.get("loss"))
        self.val_losses.append(logs.get("val_loss"))
        self.acc.append(logs.get("accuracy", logs.get("acc")))
        self.val_acc.append(logs.get("val_accuracy", logs.get("val_acc")))
        self.epochs.append(epoch)

    def on_train_end(self, logs=None):
        self.plot()


plotter = Plotter()



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/37187235.py in <cell line: 0>()
----> 1 class Plotter(Callback):
      2     def plot(self):
      3         if not DO_PLOTS:
      4             if len(self.epochs) > 0:
      5                 print(

NameError: name 'Callback' is not defined

## === cell 18
plateau_reduce = ReduceLROnPlateau(
    monitor="val_loss", factor=0.002, patience=1, min_lr=1e-50
)



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1803105361.py in <cell line: 0>()
----> 1 plateau_reduce = ReduceLROnPlateau(
      2     monitor="val_loss", factor=0.002, patience=1, min_lr=1e-50
      3 )
      4 

NameError: name 'ReduceLROnPlateau' is not defined

## === cell 19
e_stop = EarlyStopping(
    monitor="val_loss", patience=25, mode="min", restore_best_weights=True
)



## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1581723736.py in <cell line: 0>()
----> 1 e_stop = EarlyStopping(
      2     monitor="val_loss", patience=25, mode="min", restore_best_weights=True
      3 )
      4 

NameError: name 'EarlyStopping' is not defined

## === cell 20
callbacks = [plotter, plateau_reduce, e_stop]




## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1516817019.py in <cell line: 0>()
----> 1 callbacks = [plotter, plateau_reduce, e_stop]
      2 
      3 

NameError: name 'plotter' is not defined

## === cell 21
def dense_block(x, neurons, layer_no):
    x = Dense(
        neurons, kernel_initializer=he_normal(layer_no), name=f"topDense{layer_no}"
    )(x)
    x = Activation("relu", name=f"Relu{layer_no}")(x)
    x = BatchNormalization(name=f"BatchNorm{layer_no}")(x)
    x = Dropout(0.5, name=f"Dropout{layer_no}")(x)
    return x




## === cell 22
def create_model(shape):
    input_layer = Input(shape, name="input_layer")
    incep_res = InceptionResNetV2(
        include_top=False, weights="imagenet", input_tensor=input_layer
    )
    for layer in incep_res.layers:
        layer.trainable = False

    pool = MaxPooling2D(pool_size=[3, 3], strides=[3, 3], padding="same")(
        incep_res.output
    )
    flat1 = Flatten(name="Flatten1")(pool)
    flat1_bn = BatchNormalization()(flat1)

    dens1 = dense_block(flat1_bn, neurons=512, layer_no=1)
    dens2 = dense_block(dens1, neurons=512, layer_no=2)
    dens3 = dense_block(dens2, neurons=1024, layer_no=3)

    dens_final = Dense(classes_num, name="Dense6")(dens3)
    output_layer = Activation("softmax")(dens_final)

    model = Model(inputs=[input_layer], outputs=[output_layer])
    return model




## === cell 23
height, width, channels_num = 512, 512, 3
learning_rate = 0.001
epochs = 25
batch_size = 32



## === cell 24
model = create_model((height, width, channels_num))
optimizer = Adam(learning_rate)

try:
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
        run_eagerly=False,
    )
except TypeError:
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )
model.summary()



## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1324242100.py in <cell line: 0>()
----> 1 model = create_model((height, width, channels_num))
      2 optimizer = Adam(learning_rate)
      3 
      4 try:
      5     model.compile(

/tmp/ipykernel_11/1685357566.py in create_model(shape)
      1 def create_model(shape):
----> 2     input_layer = Input(shape, name="input_layer")
      3     incep_res = InceptionResNetV2(
      4         include_top=False, weights="imagenet", input_tensor=input_layer
      5     )

NameError: name 'Input' is not defined

## === cell 25
AUTOTUNE = tf.data.AUTOTUNE

one_hot_map = {c: i for i, c in enumerate(classes)}


def _build_split_file_lists_from_labels(labels_df):
    ids = labels_df["id"].to_numpy()
    breeds = labels_df["breed"].to_numpy()

    base = train_dir
    join = os.path.join

    train_files = []
    train_labels = []
    valid_files = []
    valid_labels = []
    test_files_local = []
    test_labels_local = []

    for img_id, breed in zip(ids, breeds):
        fname = img_id + ".jpg"
        rnd_prob = _stable_rand01_from_str(fname)
        fpath = join(base, fname)
        y_idx = one_hot_map[breed]
        if rnd_prob <= test_split:
            test_files_local.append(fpath)
            test_labels_local.append(y_idx)
        elif rnd_prob <= (test_split + valid_split):
            valid_files.append(fpath)
            valid_labels.append(y_idx)
        else:
            train_files.append(fpath)
            train_labels.append(y_idx)

    return (
        np.array(train_files, dtype=object),
        np.array(train_labels, dtype=np.int32),
        np.array(valid_files, dtype=object),
        np.array(valid_labels, dtype=np.int32),
        np.array(test_files_local, dtype=object),
        np.array(test_labels_local, dtype=np.int32),
    )


train_files, train_labels_idx, valid_files, valid_labels_idx, _, _ = (
    _build_split_file_lists_from_labels(labels)
)

if train_files.size == 0 or valid_files.size == 0:
    raise RuntimeError("No training/validation files found after split.")


@tf.function(reduce_retracing=True)
def _decode_resize_uint8(path):
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img_bytes, channels=3)  # uint8
    img = tf.image.resize(img, [height, width], method="nearest")  # float32
    img = tf.clip_by_value(img, 0.0, 255.0)  # float32 clip is valid
    img = tf.cast(tf.round(img), tf.uint8)
    img.set_shape([height, width, 3])
    return img


@tf.function(reduce_retracing=True)
def _uint8_to_float01(img_u8):
    img = tf.cast(img_u8, tf.float32) * (1.0 / 255.0)
    img.set_shape([height, width, 3])
    return img


_ROT_FACTOR = float(transform_params["rotation_range"]) / 360.0
_WSHIFT = float(transform_params["width_shift_range"])
_HSHIFT = float(transform_params["height_shift_range"])

epoch_counter = tf.Variable(0, dtype=tf.int64, trainable=False)


def _inc_epoch():
    epoch_counter.assign_add(1)
    return 0


@tf.function(reduce_retracing=True)
def _train_augment_tf(img_f32, seed2):
    seed = tf.stack([tf.constant(42, tf.int32), tf.cast(seed2, tf.int32)], axis=0)

    img_f32 = tf.image.stateless_random_flip_left_right(img_f32, seed=seed)

    angle = tf.random.stateless_uniform(
        [],
        seed=seed + tf.constant([101, 103], tf.int32),
        minval=-_ROT_FACTOR,
        maxval=_ROT_FACTOR,
    )
    angle = angle * (2.0 * np.pi)

    img_f32 = tf.image.rotate(img_f32, angles=angle, interpolation="BILINEAR")

    tx_frac = tf.random.stateless_uniform(
        [], seed=seed + tf.constant([5, 7], tf.int32), minval=-_WSHIFT, maxval=_WSHIFT
    )
    ty_frac = tf.random.stateless_uniform(
        [], seed=seed + tf.constant([11, 13], tf.int32), minval=-_HSHIFT, maxval=_HSHIFT
    )

    dx = tf.cast(tf.round(tx_frac * tf.cast(width, tf.float32)), tf.float32)
    dy = tf.cast(tf.round(ty_frac * tf.cast(height, tf.float32)), tf.float32)

    dx_i = tf.cast(dx, tf.int32)
    dy_i = tf.cast(dy, tf.int32)
    img_f32 = tf.roll(img_f32, shift=[dy_i, dx_i], axis=[0, 1])

    img_f32.set_shape([height, width, 3])
    return img_f32


@tf.function(reduce_retracing=True)
def _train_map_decoded(img_u8, y_idx, path):
    img = _uint8_to_float01(img_u8)
    path_h = tf.cast(tf.strings.to_hash_bucket_fast(path, 2**31 - 1), tf.int64)
    seed2 = tf.cast(
        (path_h + tf.cast(y_idx, tf.int64) * 1315423911 + epoch_counter * 2654435761)
        % (2**31 - 1),
        tf.int32,
    )
    img = _train_augment_tf(img, seed2)
    y = tf.one_hot(y_idx, depth=classes_num, dtype=tf.float32)
    y.set_shape([classes_num])
    return img, y


@tf.function(reduce_retracing=True)
def _valid_map(path, y_idx):
    img_u8 = _decode_resize_uint8(path)
    img = _uint8_to_float01(img_u8)
    y = tf.one_hot(y_idx, depth=classes_num, dtype=tf.float32)
    y.set_shape([classes_num])
    return img, y


options = tf.data.Options()
options.experimental_deterministic = True

train_paths_ds = tf.data.Dataset.from_tensor_slices((train_files, train_labels_idx))
train_paths_ds = train_paths_ds.with_options(options)
train_paths_ds = train_paths_ds.shuffle(
    buffer_size=min(4096, int(train_files.size)), seed=42, reshuffle_each_iteration=True
)

train_decoded = train_paths_ds.map(
    lambda p, y: (_decode_resize_uint8(p), y, p), num_parallel_calls=AUTOTUNE
).cache()

train_ds = train_decoded.map(_train_map_decoded, num_parallel_calls=AUTOTUNE)
train_ds = train_ds.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)

valid_ds = tf.data.Dataset.from_tensor_slices((valid_files, valid_labels_idx))
valid_ds = valid_ds.with_options(options)
valid_ds = valid_ds.map(_valid_map, num_parallel_calls=AUTOTUNE).cache()
valid_ds = valid_ds.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)

steps_per_epoch = max(1, int(np.ceil(train_files.size / batch_size)))
validation_steps = max(1, int(np.ceil(valid_files.size / batch_size)))

print("train samples:", train_files.size, "valid samples:", valid_files.size)
print("steps_per_epoch:", steps_per_epoch, "validation_steps:", validation_steps)




## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2369335866.py in <cell line: 0>()
----> 1 AUTOTUNE = tf.data.AUTOTUNE
      2 
      3 one_hot_map = {c: i for i, c in enumerate(classes)}
      4 
      5 

NameError: name 'tf' is not defined

## === cell 26
class _EpochCounterCallback(Callback):
    def on_epoch_begin(self, epoch, logs=None):
        _inc_epoch()


callbacks = [_EpochCounterCallback()] + callbacks

history = model.fit(
    train_ds,
    epochs=epochs,
    steps_per_epoch=steps_per_epoch,
    validation_data=valid_ds,
    validation_steps=validation_steps,
    callbacks=callbacks,
    verbose=1,
)



## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3294312846.py in <cell line: 0>()
----> 1 class _EpochCounterCallback(Callback):
      2     def on_epoch_begin(self, epoch, logs=None):
      3         _inc_epoch()
      4 
      5 

NameError: name 'Callback' is not defined

## === cell 27
print("Skipping new_test_dir evaluation to save time (not needed for submission).")



## === cell 28
list(one_hot_map.items())[:5]



## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4045751800.py in <cell line: 0>()
----> 1 list(one_hot_map.items())[:5]
      2 

NameError: name 'one_hot_map' is not defined

## === cell 29
sample_sub_path = "../input/dog-breed-identification/sample_submission.csv"
sample_sub = pd.read_csv(sample_sub_path)
breed_cols = [c for c in sample_sub.columns if c != "id"]

kaggle_test_dir = "../input/dog-breed-identification/test"

id_to_file = {}
with os.scandir(kaggle_test_dir) as it:
    for e in it:
        if e.is_file() and e.name.lower().endswith(".jpg"):
            img_id = os.path.splitext(e.name)[0]
            id_to_file[img_id] = e.path

sub = sample_sub.copy()
sub_ids = sub["id"].to_numpy()

missing = [i for i in sub_ids.tolist() if i not in id_to_file]
if missing:
    raise FileNotFoundError(
        f"Missing {len(missing)} test images referenced by sample_submission. Example: {missing[:3]}"
    )

test_files = np.array([id_to_file[i] for i in sub_ids], dtype=object)


@tf.function(reduce_retracing=True)
def _pred_map(path):
    img_u8 = _decode_resize_uint8(path)
    return _uint8_to_float01(img_u8)


pred_options = tf.data.Options()
pred_options.experimental_deterministic = True

pred_ds = tf.data.Dataset.from_tensor_slices(test_files).with_options(pred_options)
pred_ds = pred_ds.map(_pred_map, num_parallel_calls=AUTOTUNE).cache()
pred_ds = pred_ds.batch(batch_size).prefetch(AUTOTUNE)

pred = model.predict(
    pred_ds,
    steps=int(np.ceil(len(test_files) / batch_size)),
    verbose=1,
)
pred = pred[: len(test_files), :]

inv_map = {v: k for k, v in one_hot_map.items()}
model_class_order = [inv_map[i] for i in range(classes_num)]

pred_df = pd.DataFrame(pred, columns=model_class_order)
pred_df = pred_df[breed_cols]  # reorder to match submission template exactly

sub_out = pd.concat(
    [pd.DataFrame({"id": sub_ids}), pred_df.reset_index(drop=True)], axis=1
)

assert sub_out.shape == sample_sub.shape, (sub_out.shape, sample_sub.shape)
assert list(sub_out.columns) == list(
    sample_sub.columns
), "Submission columns do not match sample_submission."

sub_path = "/kaggle/working/submission.csv"
sub_out.to_csv(sub_path, index=False)
print("Wrote:", sub_path)
print(sub_out.head())

## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4276645308.py in <cell line: 0>()
     24 
     25 
---> 26 @tf.function(reduce_retracing=True)
     27 def _pred_map(path):
     28     img_u8 = _decode_resize_uint8(path)

NameError: name 'tf' is not defined
