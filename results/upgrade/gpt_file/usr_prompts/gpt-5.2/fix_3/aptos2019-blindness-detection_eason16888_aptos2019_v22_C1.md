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

3.10

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

0.8335036836078675

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import gc
import random
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

print("TF:", tf.__version__)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
"""
Config + image preprocessing (kept core logic: Ben Graham style preprocessing + cropping)

Speed fix: add a deterministic on-disk cache of preprocessed images.
- Preprocessing is CPU-heavy and was being repeated every epoch via tf.numpy_function.
- Caching stores the exact float32/255.0 result of preprocess_path() as .npy once per image,
  then subsequent epochs load it (identical values), preserving correctness.
"""
IMG_SIZE = 224
BATCH_SIZE = 16

DATA_ROOT = "/kaggle/input/aptos2019-blindness-detection"
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
TEST_CSV = os.path.join(DATA_ROOT, "test.csv")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train_images")
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test_images")

CACHE_DIR = "/kaggle/working/preprocessed_cache_224"
TRAIN_CACHE_DIR = os.path.join(CACHE_DIR, "train")
TEST_CACHE_DIR = os.path.join(CACHE_DIR, "test")
os.makedirs(TRAIN_CACHE_DIR, exist_ok=True)
os.makedirs(TEST_CACHE_DIR, exist_ok=True)


def crop_image_from_gray(img, tol=7):
    if img is None:
        return None
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol

        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if check_shape == 0:
            return img
        img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
        img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
        img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
        img = np.stack([img1, img2, img3], axis=-1)
        return img
    return img


def load_ben_color_bgr(image_bgr, sigmaX=10):
    if image_bgr is None:
        return None
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image = crop_image_from_gray(image).astype("uint8")
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)
    return image


def preprocess_path(path: str):
    img = cv2.imread(path)
    img = load_ben_color_bgr(img)
    if img is None:
        img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    img = img.astype("float32") / 255.0
    return img


def preprocess_path_cached(path: str, cache_dir: str):
    base = os.path.splitext(os.path.basename(path))[0]
    cache_path = os.path.join(cache_dir, base + ".npy")
    try:
        arr = np.load(cache_path, allow_pickle=False)
        if arr.shape != (IMG_SIZE, IMG_SIZE, 3) or arr.dtype != np.float32:
            raise ValueError("bad cache")
        return arr
    except Exception:
        arr = preprocess_path(path)
        tmp_path = cache_path + ".tmp"
        np.save(tmp_path, arr, allow_pickle=False)
        os.replace(tmp_path, cache_path)
        return arr




## === cell 2
train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)
sub_df = pd.read_csv(SAMPLE_SUB)

assert set(["id_code", "diagnosis"]).issubset(train_df.columns)
assert set(["id_code"]).issubset(test_df.columns)
assert set(["id_code", "diagnosis"]).issubset(sub_df.columns)

print("Train:", train_df.shape, " Test:", test_df.shape, " Sample:", sub_df.shape)

train_idx, val_idx = train_test_split(
    np.arange(len(train_df)),
    test_size=0.15,
    random_state=SEED,
    stratify=train_df["diagnosis"].values,
)
tr_df = train_df.iloc[train_idx].reset_index(drop=True)
va_df = train_df.iloc[val_idx].reset_index(drop=True)

print("Train split:", tr_df.shape, "Val split:", va_df.shape)

cw = class_weight.compute_class_weight(
    class_weight="balanced",
    classes=np.array([0, 1, 2, 3, 4]),
    y=tr_df["diagnosis"].values,
)
class_weights = {i: float(w) for i, w in enumerate(cw)}
print("Class weights:", class_weights)




## === cell 3
AUTOTUNE = tf.data.AUTOTUNE

"""
Speed fix in tf.data:
- Use cached preprocessed .npy files to avoid repeating OpenCV work across epochs.
- Avoid per-sample path.decode('utf-8') inside the hot path; numpy_function receives bytes,
  so we decode once in a small wrapper.
- Add ds.cache() for val/test to avoid repeated reading even within/among predict/validation.
  (Train is shuffled each epoch; caching AFTER shuffle would freeze order, so we don't.)
"""


def make_dataset(df, img_dir, training, cache_dir=None, with_labels=True):
    paths = (img_dir + "/" + df["id_code"].values + ".png").astype(str)
    if with_labels:
        labels = df["diagnosis"].values.astype(np.int32)
    else:
        labels = None

    path_ds = tf.data.Dataset.from_tensor_slices(paths)
    if labels is not None:
        label_ds = tf.data.Dataset.from_tensor_slices(labels)
        ds = tf.data.Dataset.zip((path_ds, label_ds))
    else:
        ds = path_ds

    def _py_preprocess(p_bytes):
        p = p_bytes.decode("utf-8")
        if cache_dir is None:
            return preprocess_path(p)
        return preprocess_path_cached(p, cache_dir)

    def _load(path, label=None):
        img = tf.numpy_function(_py_preprocess, [path], tf.float32)
        img.set_shape([IMG_SIZE, IMG_SIZE, 3])
        if label is None:
            return img
        return img, tf.one_hot(label, 5)

    if labels is not None:
        ds = ds.map(_load, num_parallel_calls=AUTOTUNE)
    else:
        ds = ds.map(lambda p: _load(p, None), num_parallel_calls=AUTOTUNE)

    if training:
        ds = ds.shuffle(1024, seed=SEED, reshuffle_each_iteration=True)
    else:
        ds = ds.cache()

    ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return ds


train_ds = make_dataset(
    tr_df, TRAIN_IMG_DIR, training=True, cache_dir=TRAIN_CACHE_DIR, with_labels=True
)
val_ds = make_dataset(
    va_df, TRAIN_IMG_DIR, training=False, cache_dir=TRAIN_CACHE_DIR, with_labels=True
)

test_ds = make_dataset(
    test_df, TEST_IMG_DIR, training=False, cache_dir=TEST_CACHE_DIR, with_labels=False
)




## === cell 4
inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
base = tf.keras.applications.DenseNet121(
    include_top=False, weights="imagenet", input_tensor=inputs
)
x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
x = tf.keras.layers.Dropout(0.5)(x)
outputs = tf.keras.layers.Dense(5, activation="softmax")(x)
model = tf.keras.Model(inputs, outputs)

for layer in base.layers:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()




## === cell 5
callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=1, verbose=1, min_lr=1e-6
    )
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=3,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=2,
)

for layer in base.layers[-60:]:
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

history_ft = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=2,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=2,
)




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
UnknownError                              Traceback (most recent call last)
/tmp/ipykernel_11/56238865.py in <cell line: 0>()
      5 ]
      6 
----> 7 history = model.fit(
      8     train_ds,
      9     validation_data=val_ds,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/execute.py in quick_execute(op_name, num_outputs, inputs, attrs, ctx, name)
     57       e.message += " name: " + name
     58     raise core._status_to_exception(e) from None
---> 59   except TypeError as e:
     60     keras_symbolic_tensors = [x for x in inputs if _is_keras_symbolic_tensor(x)]
     61     if keras_symbolic_tensors:

UnknownError: Graph execution error:

Detected at node PyFunc defined at (most recent call last):
<stack traces unavailable>
Error in user-defined function passed to ParallelMapDatasetV2:3 transformation with iterator: Iterator::Root::Prefetch::Map::Prefetch::BatchV2::Shuffle::Map: FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/working/preprocessed_cache_224/train/5814cbd2e9bf.npy.tmp' -> '/kaggle/working/preprocessed_cache_224/train/5814cbd2e9bf.npy'
Traceback (most recent call last):

  File "/tmp/ipykernel_11/2788041081.py", line 73, in preprocess_path_cached
    arr = np.load(cache_path, allow_pickle=False)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/numpy/lib/npyio.py", line 427, in load
    fid = stack.enter_context(open(os_fspath(file), "rb"))
                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/working/preprocessed_cache_224/train/5814cbd2e9bf.npy'


During handling of the above exception, another exception occurred:


Traceback (most recent call last):

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/script_ops.py", line 269, in __call__
    ret = func(*args)
          ^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py", line 643, in wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^

  File "/tmp/ipykernel_11/1257191890.py", line 32, in _py_preprocess
    return preprocess_path_cached(p, cache_dir)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/tmp/ipykernel_11/2788041081.py", line 83, in preprocess_path_cached
    os.replace(tmp_path, cache_path)

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/working/preprocessed_cache_224/train/5814cbd2e9bf.npy.tmp' -> '/kaggle/working/preprocessed_cache_224/train/5814cbd2e9bf.npy'


	 [[{{node PyFunc}}]]
	 [[IteratorGetNext]] [Op:__inference_multi_step_on_iterator_21986]

## === cell 6
pred_proba = model.predict(test_ds, verbose=1)
test_pred = np.argmax(pred_proba, axis=1).astype(np.int64)

submission = pd.DataFrame(
    {"id_code": test_df["id_code"].values, "diagnosis": test_pred}
)

submission = sub_df[["id_code"]].merge(submission, on="id_code", how="left")
assert submission["diagnosis"].isna().sum() == 0

submission.to_csv("submission.csv", index=False)
print(submission.head())
unique, counts = np.unique(test_pred, return_counts=True)
print("Pred label distribution:", dict(zip(unique.tolist(), counts.tolist())))
print("Wrote submission.csv")

gc.collect()

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
UnknownError                              Traceback (most recent call last)
/tmp/ipykernel_11/1138538900.py in <cell line: 0>()
----> 1 pred_proba = model.predict(test_ds, verbose=1)
      2 test_pred = np.argmax(pred_proba, axis=1).astype(np.int64)
      3 
      4 submission = pd.DataFrame(
      5     {"id_code": test_df["id_code"].values, "diagnosis": test_pred}

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py in raise_from_not_ok_status(e, name)
   6000 def raise_from_not_ok_status(e, name) -> NoReturn:
   6001   e.message += (" name: " + str(name if name is not None else ""))
-> 6002   raise core._status_to_exception(e) from None  # pylint: disable=protected-access
   6003 
   6004 

UnknownError: {{function_node __wrapped__IteratorGetNext_output_types_1_device_/job:localhost/replica:0/task:0/device:CPU:0}} Error in user-defined function passed to ParallelMapDatasetV2:15 transformation with iterator: Iterator::Root::Prefetch::BatchV2::MemoryCacheImpl::Map: FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/working/preprocessed_cache_224/test/b460ca9fa26f.npy.tmp' -> '/kaggle/working/preprocessed_cache_224/test/b460ca9fa26f.npy'
Traceback (most recent call last):

  File "/tmp/ipykernel_11/2788041081.py", line 73, in preprocess_path_cached
    arr = np.load(cache_path, allow_pickle=False)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/numpy/lib/npyio.py", line 427, in load
    fid = stack.enter_context(open(os_fspath(file), "rb"))
                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/working/preprocessed_cache_224/test/b460ca9fa26f.npy'


During handling of the above exception, another exception occurred:


Traceback (most recent call last):

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/script_ops.py", line 269, in __call__
    ret = func(*args)
          ^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py", line 643, in wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^

  File "/tmp/ipykernel_11/1257191890.py", line 32, in _py_preprocess
    return preprocess_path_cached(p, cache_dir)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/tmp/ipykernel_11/2788041081.py", line 83, in preprocess_path_cached
    os.replace(tmp_path, cache_path)

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/working/preprocessed_cache_224/test/b460ca9fa26f.npy.tmp' -> '/kaggle/working/preprocessed_cache_224/test/b460ca9fa26f.npy'


	 [[{{node PyFunc}}]] [Op:IteratorGetNext] name:
