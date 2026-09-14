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
Create a classifier to predict the severity of diabetic retinopathy.

## Metric
Quadratic weighted kappa, which measures the agreement between two ratings. This metric typically varies from 0 (random agreement between raters) to 1 (complete agreement between raters). In the event that there is less agreement between the raters than expected by chance, this metric may go below 0. The quadratic weighted kappa is calculated between the scores assigned by the human rater and the predicted scores.

Images have five possible ratings, 0,1,2,3,4.  Each image is characterized by a tuple *(e*,*e)*, which corresponds to its scores by *Rater A* (human) and *Rater B* (predicted).  The quadratic weighted kappa is calculated as follows. First, an N x N histogram matrix *O* is constructed, such that *O* corresponds to the number of images that received a rating *i* by *A* and a rating *j* by *B*. An *N-by-N* matrix of weights, *w*, is calculated based on the difference between raters' scores:

An *N-by-N* histogram matrix of expected ratings, *E*, is calculated, assuming that there is no correlation between rating scores.  This is calculated as the outer product between each rater's histogram vector of ratings, normalized such that *E* and *O* have the same sum.

## Submission Format
```
id_code,diagnosis
0005cfc8afb6,0
003f0afdcd15,0
etc.
```

## Dataset
You are provided with a large set of retina images taken using [fundus photography](https://en.wikipedia.org/wiki/Fundus_photography) under a variety of imaging conditions.

Labels are on a scale of 0 to 4:

> 0 - No DR
> 1 - Mild
> 2 - Moderate
> 3 - Severe
> 4 - Proliferative DR

Images may contain artifacts, be out of focus, underexposed, or overexposed. The images were gathered from multiple clinics using a variety of cameras over an extended period of time, which will introduce further variation.

- **train.csv** - the training labels
- **test.csv** - the test set (you must predict the `diagnosis` value for these variables)
- **sample_submission.csv** - a sample submission file in the correct format
- **train.zip** - the training set images
- **test.zip** - the public test set images

# 2. Python version

3.7

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (118 lines)
            sample_submission.csv (368 lines)
            sample_submission.csv.zip (3.2 kB)
            test.csv (368 lines)
            test.csv.zip (2.9 kB)
            test.zip (160 Bytes)
            test_images.zip (902.9 MB)
            train.csv (3296 lines)
            train.csv.zip (27.5 kB)
            train.zip (162 Bytes)
            train_images.zip (7.7 GB)
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
            test_images/
                218c822a3dd9.png (5.7 MB)
                0e82bcacc475.png (5.2 MB)
                ... and 365 other files
                test_images/
            train_images/
                184a185e7447.png (337.5 kB)
                c4aef0d88d1b.png (876.6 kB)
                ... and 3293 other files
                train_images/
        input/
            description.md (118 lines)
            sample_submission.csv (368 lines)
            sample_submission.csv.zip (3.2 kB)
            test.csv (368 lines)
            test.csv.zip (2.9 kB)
            test.zip (160 Bytes)
            test_images.zip (902.9 MB)
            train.csv (3296 lines)
            train.csv.zip (27.5 kB)
            train.zip (162 Bytes)
            train_images.zip (7.7 GB)
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
            test_images/
                218c822a3dd9.png (5.7 MB)
                0e82bcacc475.png (5.2 MB)
                ... and 365 other files
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
            train_images/
                184a185e7447.png (337.5 kB)
                c4aef0d88d1b.png (876.6 kB)
                ... and 3293 other files
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
        working/
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
```

-> data/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/aptos2019-blindness-detection/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/aptos2019-blindness-detection/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> input/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> (stopped after 10 files for performance)

# 5. Target score

0.5556703829117215

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved -0.02659) has done: 'I fix the runtime blockers by removing the invalid `scipy.ndarray` import/type usage, ensuring `deepcopy` is imported where it’s used, and making the notebook compatible with the Kaggle environment’s TensorFlow/Keras versions by avoiding TF1 session APIs and notebook magics. Since the provided pre-trained model path is missing/unsupported, I keep the same DenseNet121-based core model logic but actually build/train it briefly so `model.predict()` works end-to-end and a non-empty `submission.csv` is produced. I also make prediction order match `test.csv` (not filesystem glob order) to avoid id/label misalignment that would destroy kappa. Finally, I write the submission using the exact required columns and ensure the file has a `.csv` suffix in the working directory.'
- What this solution (achieved 0.01238) has done: 'The timeout is dominated by Python-side image I/O + augmentation inside `from_generator`, which repeatedly calls OpenCV and allocates new arrays while TensorFlow waits, plus an extra full pre-processing pass for test prediction. I keep the exact same preprocessing, augmentations (original + random gamma + hflip), model, optimizer, and training loop semantics, but remove repeated work by (1) precomputing deterministic per-sample random gammas per epoch, (2) caching the base preprocessed images once (already present) and generating augmented variants with far fewer Python allocations, and (3) reusing a base-cache for test prediction instead of re-reading from disk. These changes are provably equivalent (same transforms applied with the same distributions) and primarily reduce constant factors and host overhead so the GPU/CPU can stay busy. I also set `workers=1,use_multiprocessing=False` for Keras to avoid generator overhead/pickling costs and keep determinism.'
- What this solution (achieved 0.01238) has done: 'The timeout is dominated by Python-side image loading/preprocessing and per-sample augmentation happening inside a generator every epoch, plus unnecessary recomputation/copying of batches. I keep the exact same model, loss, training loop, and augmentation semantics (base image + random-gamma + horizontal flip per sample), but make it faster by (1) caching preprocessed base images once, (2) vectorizing gamma augmentation per batch using precomputed LUTs instead of calling `cv2.LUT` per image, (3) switching the generator to index-based shuffling while still producing the same 3x-expanded dataset, and (4) using a `Sequence` (still used through `tf.data`) to reduce generator overhead and avoid repeated allocations. These are provably equivalent transformations (same operations, just batched/cached), deterministic under the same seed scheme, and they remove the main Python overhead that causes the 10-minute timeout.'
- What this solution (achieved -0.04374) has done: 'Main bottlenecks are (1) repeatedly doing expensive OpenCV preprocessing + gamma augmentation inside Python for every batch, and (2) driving training through a Python generator path that can’t keep the GPU/TF runtime fed. The refactor below keeps the exact same model, loss, epochs, and the same “triplet” augmentation stream per image (base, random-gamma, horizontal flip), but moves it into a `tf.keras.utils.Sequence` that pre-caches the expensive base preprocessing once and vectorizes the gamma LUT augmentation per batch. It also removes slow/unused debug visualization work and avoids building redundant lists/loops, while preserving determinism via fixed seeds and per-epoch seeded gamma generation. Net effect: far less Python overhead and disk I/O during `fit()`, typically bringing runtime under the 600s limit without changing core training semantics.'
- What this solution (achieved -0.04374) has done: 'The main timeout comes from repeatedly doing expensive OpenCV preprocessing and per-image gamma correction inside the training generators, plus rebuilding small caches during prediction. I keep the exact same model, loss, epochs, and the same “triplet per image” augmentation semantics, but make the data pipeline faster by (1) caching the base-preprocessed images once, (2) using a `keras.utils.Sequence` end-to-end (so TensorFlow can pipeline batches), (3) vectorizing the per-batch gamma LUT application (exactly equivalent to `cv2.LUT` per image on the same nearest gamma grid already used), and (4) caching the test base-preprocessed images once to avoid repeated disk I/O during prediction. These are provably equivalent transformations of the existing operations and preserve determinism via fixed seeds.'

# 9. Code solution

## === cell 0
import os
import glob
import random
from copy import deepcopy
from functools import lru_cache

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "cpp")

import numpy as np
import pandas as pd

import cv2
import matplotlib.pyplot as plt

SEED = 1337
random.seed(SEED)
np.random.seed(SEED)

try:
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)
except Exception:
    pass

print("Listing ../input:")
print(os.listdir("../input")[:20])



## === cell 1
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

tf.random.set_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

print("TensorFlow:", tf.__version__)
print("tf.keras:", keras.__version__)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/1849864157.py in <cell line: 0>()
----> 1 import tensorflow as tf
      2 from tensorflow import keras
      3 from tensorflow.keras import backend as K
      4 
      5 tf.random.set_seed(SEED)

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

## === cell 2
plt.rcParams.update({"axes.titlesize": "small"})



## === cell 3
IMG_SIZE = 128



## === cell 4
DATA_DIR = "../input/aptos2019-blindness-detection"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")

print("Train csv exists:", os.path.exists(TRAIN_CSV))
print("Test csv exists:", os.path.exists(TEST_CSV))
print("Train img dir exists:", os.path.isdir(TRAIN_IMG_DIR))
print("Test img dir exists:", os.path.isdir(TEST_IMG_DIR))



## === cell 5
train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)
print(train_df.head())
print(test_df.head())



## === cell 6
labels_df = train_df.set_index("id_code")
print("Num train:", len(labels_df))
print(labels_df["diagnosis"].value_counts())



## === cell 7
images = sorted(glob.glob(os.path.join(TRAIN_IMG_DIR, "*.png")))
ids = [os.path.splitext(os.path.basename(f))[0] for f in images]
labels = labels_df.loc[ids, "diagnosis"].astype(int).tolist()

print("Found train images:", len(images))
print("Found labels:", len(labels))



## === cell 8
n = len(images)
k = min(6000, n)  # keeps "about 6000" intent while remaining valid
perm = np.arange(n)
rng = np.random.RandomState(SEED)
rng.shuffle(perm)

train_mask = np.zeros(n, dtype=bool)
train_mask[perm[:k]] = True

images_arr = np.array(images, dtype=object)
labels_arr = np.array(labels, dtype=np.int64)

train_images = images_arr[train_mask].tolist()
train_labels = labels_arr[train_mask].tolist()
test_images = images_arr[~train_mask].tolist()
test_labels = labels_arr[~train_mask].tolist()

print(len(train_labels), len(train_images))
print(len(test_labels), len(test_images))



## === cell 9
for img, lb in list(zip(images[:5], labels[:5])):
    print(img, lb)



## === cell 10
label_text = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]

try:
    if False:
        images_to_display = []
        for lb in range(5):
            candidates = [
                (images[ix], labels[ix])
                for ix in range(len(images))
                if labels[ix] == lb
            ]
            if len(candidates) > 0:
                images_to_display += random.sample(
                    candidates, k=min(3, len(candidates))
                )

        fig = plt.figure(figsize=(12, 8))
        for ii, (img_path, label) in enumerate(images_to_display):
            ax = fig.add_subplot(2, 8, ii + 1, xticks=[], yticks=[])
            img = cv2.imread(img_path)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(img)
            ax.set_title(label_text[label])
        plt.show()
except Exception as e:
    print("Plot skipped:", repr(e))



## === cell 11
_GAMMA_GRID = np.round(np.linspace(0.8, 1.8, 101), 3)  # inclusive grid

_GAMMA_LUTS_ARR = np.empty((len(_GAMMA_GRID), 256), dtype=np.uint8)
for i, g in enumerate(_GAMMA_GRID):
    invGamma = 1.0 / float(g)
    _GAMMA_LUTS_ARR[i] = (
        (np.arange(256, dtype=np.float32) / 255.0) ** invGamma * 255.0
    ).astype(np.uint8)


def _nearest_gamma_index(gamma: float) -> int:
    idx = int(
        np.clip(
            np.round((gamma - 0.8) / (1.8 - 0.8) * (len(_GAMMA_GRID) - 1)),
            0,
            len(_GAMMA_GRID) - 1,
        )
    )
    return idx


def adjust_gamma(image, gamma=1.0):
    idx = _nearest_gamma_index(float(gamma))
    return cv2.LUT(image, _GAMMA_LUTS_ARR[idx])


def _adjust_gamma_batch_uint8(imgs_uint8: np.ndarray, gammas: np.ndarray) -> np.ndarray:
    """
    Equivalent of applying adjust_gamma(img, gamma) per image, using nearest LUT.
    imgs_uint8: (B,H,W,3) uint8
    gammas: (B,) float32
    returns: (B,H,W,3) uint8
    """
    B = imgs_uint8.shape[0]
    idx = np.rint((gammas - 0.8) / (1.8 - 0.8) * (len(_GAMMA_GRID) - 1)).astype(
        np.int32
    )
    np.clip(idx, 0, len(_GAMMA_GRID) - 1, out=idx)
    luts = _GAMMA_LUTS_ARR[idx]  # (B,256)
    b_ix = np.arange(B, dtype=np.int32)[:, None, None, None]
    return luts[b_ix, imgs_uint8]


def add_contrast(img, contrast):
    buf = img.copy()
    f = float(131 * (contrast + 127)) / (127 * (131 - contrast))
    alpha_c = f
    gamma_c = 127 * (1 - f)
    buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)
    return buf


def preproces_image(img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = adjust_gamma(img, 1.5)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = add_contrast(img, 20)
    return img




## === cell 12
def random_rotation(image_array: np.ndarray):
    raise NotImplementedError("random_rotation is unused in this pipeline.")


def random_noise(image_array: np.ndarray):
    raise NotImplementedError("random_noise is unused in this pipeline.")


def horizontal_flip(image_array: np.ndarray):
    return image_array[:, ::-1]




## === cell 13
wts = [0.1, 0.4, 0.2, 0.90, 0.60]
num_of_class = 5



## === cell 14
try:
    if False:
        img0 = cv2.imread(train_images[0])
        img0p = preproces_image(img0)
        plt.imshow(img0p)
        plt.axis("off")
        plt.show()
except Exception:
    pass




## === cell 15
def _to_model_input(batch_imgs_uint8):
    return batch_imgs_uint8.astype(np.float32) / 255.0


@lru_cache(maxsize=65536)
def _read_and_base_preprocess(path: str):
    im = cv2.imread(path)
    if im is None:
        im = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    else:
        im = preproces_image(im)
    return np.ascontiguousarray(im)


def _build_base_cache(image_paths):
    cache = np.empty((len(image_paths), IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    read_pp = _read_and_base_preprocess
    for i, p in enumerate(image_paths):
        cache[i] = read_pp(p)
    return cache


train_base_cache = _build_base_cache(train_images)
val_base_cache = _build_base_cache(test_images)
print("Cached train/val base images:", train_base_cache.shape, val_base_cache.shape)


def _epoch_gammas(num_samples: int, epoch_seed: int) -> np.ndarray:
    rs = np.random.RandomState(int(epoch_seed))
    return rs.uniform(0.8, 1.8, size=num_samples).astype(np.float32)


class TripletAugSequence(tf.keras.utils.Sequence):
    """
    Produces exactly the same stream as the old generator:
    for each epoch, for each sample in order (or shuffled indices), emit:
      1) base image
      2) gamma-augmented image (random gamma per sample per epoch)
      3) horizontal flip of base image
    """

    def __init__(
        self,
        base_cache_uint8,
        labels,
        batch_size,
        with_sample_weights,
        seed_offset=0,
        shuffle=False,
    ):
        self.base = base_cache_uint8
        self.labels = np.asarray(labels, dtype=np.int64)
        self.batch_size = int(batch_size)
        self.with_sw = bool(with_sample_weights)
        self.seed_offset = int(seed_offset)
        self.shuffle = bool(shuffle)

        self.n = len(self.labels)
        self.onehot = tf.keras.utils.to_categorical(self.labels, num_of_class).astype(
            np.float32
        )
        if self.with_sw:
            self.sample_w = np.asarray(
                [wts[int(t)] for t in self.labels], dtype=np.float32
            )

        self.m = self.n * 3

        self._base_rs = np.random.RandomState(SEED + self.seed_offset)
        self._epoch_seed = None
        self._gammas = None
        self._order = np.arange(self.n, dtype=np.int32)
        self.on_epoch_end()

        self._bx = np.empty((self.batch_size, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        self._by = np.empty((self.batch_size, num_of_class), dtype=np.float32)
        if self.with_sw:
            self._bw = np.empty((self.batch_size,), dtype=np.float32)

    def __len__(self):
        return int(np.ceil(self.m / self.batch_size))

    def on_epoch_end(self):
        self._epoch_seed = int(self._base_rs.randint(0, 2**31 - 1))
        self._gammas = _epoch_gammas(self.n, self._epoch_seed)
        if self.shuffle:
            rs = np.random.RandomState(self._epoch_seed)
            rs.shuffle(self._order)

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = min(start + self.batch_size, self.m)
        bs = end - start

        exp = np.arange(start, end, dtype=np.int32)
        sample_ix = exp // 3
        aug_type = exp - sample_ix * 3

        sample_ix = self._order[sample_ix]

        self._by[:bs] = self.onehot[sample_ix]
        if self.with_sw:
            self._bw[:bs] = self.sample_w[sample_ix]

        bx = self._bx

        mask0 = aug_type == 0
        if np.any(mask0):
            pos = np.where(mask0)[0]
            sx = sample_ix[mask0]
            bx[pos] = self.base[sx]

        mask2 = aug_type == 2
        if np.any(mask2):
            pos = np.where(mask2)[0]
            sx = sample_ix[mask2]
            bx[pos] = self.base[sx][:, :, ::-1, :]

        mask1 = aug_type == 1
        if np.any(mask1):
            pos = np.where(mask1)[0]
            sx = sample_ix[mask1]
            g = self._gammas[sx]
            bx[pos] = _adjust_gamma_batch_uint8(self.base[sx], g)

        x = _to_model_input(bx[:bs])
        y = self._by[:bs]

        if self.with_sw:
            return x, y, self._bw[:bs]
        return x, y


def generate_training_images(cur_images, cur_tags, batch_size=500):
    cur_tags_arr = np.asarray(cur_tags, dtype=np.int64)
    onehot = tf.keras.utils.to_categorical(cur_tags_arr, num_of_class)
    sample_w = np.asarray([wts[int(t)] for t in cur_tags_arr], dtype=np.float32)

    to_inp = _to_model_input
    hflip = horizontal_flip

    base_rs = np.random.RandomState(SEED)
    base_cache = (
        train_base_cache
        if cur_images is train_images
        else _build_base_cache(cur_images)
    )
    n = len(cur_images)

    batch_x = np.empty((batch_size, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    batch_y = np.empty((batch_size, num_of_class), dtype=np.float32)
    batch_w = np.empty((batch_size,), dtype=np.float32)

    while True:
        epoch_seed = int(base_rs.randint(0, 2**31 - 1))
        gammas = _epoch_gammas(n, epoch_seed)

        bi = 0
        for ix in range(n):
            label_oh = onehot[ix]
            wt = sample_w[ix]
            img = base_cache[ix]

            batch_x[bi] = img
            batch_y[bi] = label_oh
            batch_w[bi] = wt
            bi += 1
            if bi == batch_size:
                yield to_inp(batch_x), batch_y, batch_w
                bi = 0

            batch_x[bi] = adjust_gamma(img, float(gammas[ix]))
            batch_y[bi] = label_oh
            batch_w[bi] = wt
            bi += 1
            if bi == batch_size:
                yield to_inp(batch_x), batch_y, batch_w
                bi = 0

            batch_x[bi] = hflip(img)
            batch_y[bi] = label_oh
            batch_w[bi] = wt
            bi += 1
            if bi == batch_size:
                yield to_inp(batch_x), batch_y, batch_w
                bi = 0

        if bi > 0:
            yield to_inp(batch_x[:bi]), batch_y[:bi], batch_w[:bi]




## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/132309720.py in <cell line: 0>()
     31 
     32 
---> 33 class TripletAugSequence(tf.keras.utils.Sequence):
     34     """
     35     Produces exactly the same stream as the old generator:

NameError: name 'tf' is not defined

## === cell 16
def generate_testing_images(cur_images, cur_tags, batch_size=500):
    cur_tags_arr = np.asarray(cur_tags, dtype=np.int64)
    onehot = tf.keras.utils.to_categorical(cur_tags_arr, num_of_class)

    to_inp = _to_model_input
    hflip = horizontal_flip

    base_rs = np.random.RandomState(SEED + 1)
    base_cache = (
        val_base_cache if cur_images is test_images else _build_base_cache(cur_images)
    )
    n = len(cur_images)

    batch_x = np.empty((batch_size, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    batch_y = np.empty((batch_size, num_of_class), dtype=np.float32)

    while True:
        epoch_seed = int(base_rs.randint(0, 2**31 - 1))
        gammas = _epoch_gammas(n, epoch_seed)

        bi = 0
        for ix in range(n):
            label_oh = onehot[ix]
            img = base_cache[ix]

            batch_x[bi] = img
            batch_y[bi] = label_oh
            bi += 1
            if bi == batch_size:
                yield to_inp(batch_x), batch_y
                bi = 0

            batch_x[bi] = adjust_gamma(img, float(gammas[ix]))
            batch_y[bi] = label_oh
            bi += 1
            if bi == batch_size:
                yield to_inp(batch_x), batch_y
                bi = 0

            batch_x[bi] = hflip(img)
            batch_y[bi] = label_oh
            bi += 1
            if bi == batch_size:
                yield to_inp(batch_x), batch_y
                bi = 0

        if bi > 0:
            yield to_inp(batch_x[:bi]), batch_y[:bi]




## === cell 17
if False:
    for batch in generate_training_images(train_images, train_labels, batch_size=50):
        tmp_images, tmp_labels, tmp_w = batch
        print(
            tmp_images.shape,
            tmp_labels.shape,
            tmp_w.shape,
            tmp_images.dtype,
            tmp_images.min(),
            tmp_images.max(),
        )
        plt.imshow((tmp_images[0] * 255).astype(np.uint8))
        plt.axis("off")
        plt.show()
        break



## === cell 18
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler




## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/3141945872.py in <cell line: 0>()
----> 1 from tensorflow.keras.applications.densenet import DenseNet121
      2 from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D
      3 from tensorflow.keras.models import Model
      4 from tensorflow.keras.optimizers import Adam
      5 from tensorflow.keras.callbacks import LearningRateScheduler

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

## === cell 19
def reset_tf_session():
    K.clear_session()
    try:
        tf.keras.backend.clear_session()
    except Exception:
        pass


reset_tf_session()
input_shape = (IMG_SIZE, IMG_SIZE, 3)



## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1093513407.py in <cell line: 0>()
      7 
      8 
----> 9 reset_tf_session()
     10 input_shape = (IMG_SIZE, IMG_SIZE, 3)
     11 

/tmp/ipykernel_11/1093513407.py in reset_tf_session()
      1 def reset_tf_session():
----> 2     K.clear_session()
      3     try:
      4         tf.keras.backend.clear_session()
      5     except Exception:

NameError: name 'K' is not defined

## === cell 20
inp = Input(shape=input_shape)
base = DenseNet121(include_top=False, weights="imagenet", input_tensor=inp)
x = GlobalAveragePooling2D()(base.output)
out = Dense(num_of_class, activation="softmax")(x)
model = Model(inputs=inp, outputs=out)

INIT_LR = 5e-3
BATCH_SIZE = 200
EPOCHS = 2  # keep core training loop approach unchanged


def lr_scheduler(epoch):
    return max(INIT_LR * (0.9**epoch), 0.00001)


model.compile(
    optimizer=Adam(learning_rate=INIT_LR),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()



## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4120879811.py in <cell line: 0>()
----> 1 inp = Input(shape=input_shape)
      2 base = DenseNet121(include_top=False, weights="imagenet", input_tensor=inp)
      3 x = GlobalAveragePooling2D()(base.output)
      4 out = Dense(num_of_class, activation="softmax")(x)
      5 model = Model(inputs=inp, outputs=out)

NameError: name 'Input' is not defined

## === cell 21
steps_per_epoch = max(1, int(np.ceil((len(train_images) * 3) / BATCH_SIZE)))
val_steps = max(1, int(np.ceil((len(test_images) * 3) / BATCH_SIZE)))

train_seq = TripletAugSequence(
    train_base_cache,
    train_labels,
    batch_size=BATCH_SIZE,
    with_sample_weights=True,
    seed_offset=0,
    shuffle=False,
)
val_seq = TripletAugSequence(
    val_base_cache,
    test_labels,
    batch_size=BATCH_SIZE,
    with_sample_weights=False,
    seed_offset=1,
    shuffle=False,
)

history = model.fit(
    train_seq,
    steps_per_epoch=steps_per_epoch,
    epochs=EPOCHS,
    validation_data=val_seq,
    validation_steps=val_steps,
    callbacks=[LearningRateScheduler(lr_scheduler)],
    verbose=2,
)



## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3058791127.py in <cell line: 0>()
----> 1 steps_per_epoch = max(1, int(np.ceil((len(train_images) * 3) / BATCH_SIZE)))
      2 val_steps = max(1, int(np.ceil((len(test_images) * 3) / BATCH_SIZE)))
      3 
      4 train_seq = TripletAugSequence(
      5     train_base_cache,

NameError: name 'BATCH_SIZE' is not defined

## === cell 22
test_ids = test_df["id_code"].astype(str).tolist()
predict_images = [os.path.join(TEST_IMG_DIR, f"{id_code}.png") for id_code in test_ids]

missing = [p for p in predict_images if not os.path.exists(p)]
print("Num test images:", len(predict_images))
print("Missing test images:", len(missing))
if len(missing) > 0:
    print("Example missing:", missing[:5])


def predict_stream_tta_triplet(model, image_paths, batch_size=32):
    """
    Minimal score improvement while preserving core augmentation semantics:
    apply the same triplet transforms used in training (base, gamma, hflip) at test time
    and average predicted probabilities (TTA).
    """
    base_uint8 = _build_base_cache(image_paths)
    x0 = _to_model_input(base_uint8)
    p0 = model.predict(x0, batch_size=batch_size, verbose=0).astype(np.float32)

    gammas = _epoch_gammas(len(image_paths), SEED + 12345)
    x1_uint8 = _adjust_gamma_batch_uint8(base_uint8, gammas)
    x1 = _to_model_input(x1_uint8)
    p1 = model.predict(x1, batch_size=batch_size, verbose=0).astype(np.float32)

    x2_uint8 = base_uint8[:, :, ::-1, :]
    x2 = _to_model_input(x2_uint8)
    p2 = model.predict(x2, batch_size=batch_size, verbose=0).astype(np.float32)

    return (p0 + p1 + p2) / 3.0


predictions = predict_stream_tta_triplet(model, predict_images, batch_size=32)
print("Predictions:", predictions.shape, predictions.dtype)



## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3626281029.py in <cell line: 0>()
     32 
     33 
---> 34 predictions = predict_stream_tta_triplet(model, predict_images, batch_size=32)
     35 print("Predictions:", predictions.shape, predictions.dtype)
     36 

NameError: name 'model' is not defined

## === cell 23
pred_labels = predictions.argmax(axis=1).astype(int)

subm = pd.DataFrame({"id_code": test_ids, "diagnosis": pred_labels})
submission_path = "submission.csv"
subm.to_csv(submission_path, index=False)

print("Wrote", submission_path, "with shape", subm.shape)
print(subm.head())
print("Unique predictions:", np.unique(pred_labels, return_counts=True))

## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1543355118.py in <cell line: 0>()
----> 1 pred_labels = predictions.argmax(axis=1).astype(int)
      2 
      3 subm = pd.DataFrame({"id_code": test_ids, "diagnosis": pred_labels})
      4 submission_path = "submission.csv"
      5 subm.to_csv(submission_path, index=False)

NameError: name 'predictions' is not defined
