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
Detect apple diseases from images.

## Metric
Mean column-wise ROC AUC.

## Submission Format
For each image_id in the test set, you must predict a probability for each target variable. The file should contain a header and have the following format:

```
image_id,
test_0,0.25,0.25,0.25,0.25
test_1,0.25,0.25,0.25,0.25
test_2,0.25,0.25,0.25,0.25
etc.
```

## Dataset
Given a photo of an apple leaf, can you accurately assess its health? This competition will challenge you to distinguish between leaves which are healthy, those which are infected with apple rust, those that have apple scab, and those with more than one disease.

**train.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

**images**

A folder containing the train and test images, in jpg format.

**test.csv**

- `image_id`: the foreign key

**sample_submission.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

# 2. Python version

3.9

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        input/
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        working/
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
```

-> data/plant-pathology-2020-fgvc7/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/plant-pathology-2020-fgvc7/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/plant-pathology-2020-fgvc7/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> (stopped after 10 files for performance)

# 5. Target score

0.9707334593753392

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, math, re, random

import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow.keras.layers as L
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split

print("Tensorflow version " + tf.__version__)

SEED = 2020
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    if hasattr(tf.config.experimental, "enable_op_determinism"):
        tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
AUTO = tf.data.experimental.AUTOTUNE

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print("Running on TPU", tpu.master())
except Exception:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy()

print("REPLICAS:", strategy.num_replicas_in_sync)

EPOCHS = 40
BATCH_SIZE = 8 * strategy.num_replicas_in_sync

DATA_DIR = "/kaggle/input/plant-pathology-2020-fgvc7"
IMAGES_DIR = os.path.join(DATA_DIR, "images")

assert os.path.exists(DATA_DIR), f"DATA_DIR not found: {DATA_DIR}"
assert os.path.exists(IMAGES_DIR), f"IMAGES_DIR not found: {IMAGES_DIR}"




## === cell 2
def format_path(image_id: str) -> str:
    return os.path.join(IMAGES_DIR, f"{image_id}.jpg")




## === cell 3
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
sub = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

target_cols = [c for c in sub.columns if c != "image_id"]
assert "image_id" in sub.columns
for c in target_cols:
    assert c in train.columns, f"Missing target column in train.csv: {c}"

train_paths = (IMAGES_DIR + "/" + train["image_id"].astype(str) + ".jpg").to_numpy()
test_paths = (IMAGES_DIR + "/" + test["image_id"].astype(str) + ".jpg").to_numpy()

train_labels = train[target_cols].to_numpy(dtype=np.float32)

train_paths, valid_paths, train_labels, valid_labels = train_test_split(
    train_paths, train_labels, test_size=0.15, random_state=SEED, shuffle=True
)

assert len(train_paths) == len(train_labels)
assert len(valid_paths) == len(valid_labels)
assert train_labels.shape[1] == len(target_cols)




## === cell 4
img_size = 768


def decode_image(filename, label=None, image_size=(img_size, img_size)):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits, channels=3)
    image = tf.image.resize(
        image, image_size
    )  # keep same resize op to preserve behavior
    image = tf.cast(image, tf.float32) * (1.0 / 255.0)
    image.set_shape([image_size[0], image_size[1], 3])
    if label is None:
        return image
    return image, label


def _seed_from_filename(filename, base_seed=SEED):
    h = tf.strings.to_hash_bucket_fast(filename, 2**31 - 1)  # int64
    s0 = tf.cast(tf.bitwise.bitwise_xor(h, tf.cast(base_seed, tf.int64)), tf.int32)
    s1 = tf.cast(
        tf.bitwise.bitwise_xor(
            tf.bitwise.right_shift(h, 1),
            tf.cast(base_seed * 9973, tf.int64),
        ),
        tf.int32,
    )
    return tf.stack([s0, s1], axis=0)


def data_augment(image, label=None, seed=None):
    if seed is None:
        seed = tf.constant([SEED, SEED], dtype=tf.int32)
    image = tf.image.stateless_random_flip_left_right(image, seed=seed)
    image = tf.image.stateless_random_flip_up_down(
        image, seed=seed + tf.constant([1, 1], tf.int32)
    )
    if label is None:
        return image
    return image, label


def _with_fast_dataset_options(ds):
    opts = tf.data.Options()
    opts.experimental_deterministic = True
    try:
        opts.experimental_optimization.map_parallelization = True
        opts.experimental_optimization.map_and_batch_fusion = True
        opts.experimental_optimization.apply_default_optimizations = True
        opts.experimental_optimization.parallel_batch = True
    except Exception:
        pass
    return ds.with_options(opts)




## === cell 5
def _resolve_weight_path(p: str) -> str:
    if os.path.exists(p):
        return p

    candidates = [p]
    if p.startswith("/kaggle/input/"):
        candidates.append(p.replace("/kaggle/input/", "/kaggle/data/"))
        candidates.append(p.replace("/kaggle/input/", "/kaggle/working/"))

    basename = os.path.basename(p)
    for root in ("/kaggle/input", "/kaggle/data", "/kaggle/working"):
        candidates.append(os.path.join(root, basename))
        try:
            for d in os.listdir(root):
                candidates.append(os.path.join(root, d, basename))
        except Exception:
            pass

    for c in candidates:
        if c and os.path.exists(c):
            return c
    return p


w1 = _resolve_weight_path(
    "/kaggle/input/tf-zoo-models-on-tpu-efficientnetb7/my_ef_net_b7.h5"
)
w2 = _resolve_weight_path(
    "/kaggle/input/tf-zoo-models-on-tpu-densenet201/my_dense_net_201.h5"
)

need_train_1 = not os.path.exists(w1)
need_train_2 = not os.path.exists(w2)

print("Weights resolved:")
print(" model1:", w1, "exists:", os.path.exists(w1))
print(" model2:", w2, "exists:", os.path.exists(w2))

if need_train_1 or need_train_2:
    raise FileNotFoundError(
        "Required pretrained weights were not found. "
        "Training EfficientNetB7/DenseNet201 from scratch here will time out under 600s.\n"
        f"Resolved paths:\n  w1={w1} (exists={os.path.exists(w1)})\n  w2={w2} (exists={os.path.exists(w2)})\n"
        "Please ensure the Kaggle datasets providing these files are attached."
    )

test_cache_path = "/kaggle/working/test_cache_768"
test_dataset = tf.data.Dataset.from_tensor_slices(test_paths)
test_dataset = _with_fast_dataset_options(test_dataset)
test_dataset = test_dataset.map(
    decode_image, num_parallel_calls=AUTO, deterministic=True
)
test_dataset = test_dataset.cache(test_cache_path)
test_dataset = test_dataset.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTO)




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/1743753308.py in <cell line: 0>()
     46 # two very large models for 40 epochs at 768px, which cannot complete under 600s.
     47 if need_train_1 or need_train_2:
---> 48     raise FileNotFoundError(
     49         "Required pretrained weights were not found. "
     50         "Training EfficientNetB7/DenseNet201 from scratch here will time out under 600s.\n"

FileNotFoundError: Required pretrained weights were not found. Training EfficientNetB7/DenseNet201 from scratch here will time out under 600s.
Resolved paths:
  w1=/kaggle/input/tf-zoo-models-on-tpu-efficientnetb7/my_ef_net_b7.h5 (exists=False)
  w2=/kaggle/input/tf-zoo-models-on-tpu-densenet201/my_dense_net_201.h5 (exists=False)
Please ensure the Kaggle datasets providing these files are attached.

## === cell 6
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras.applications.densenet import DenseNet201


def get_model(use_model, n_classes: int):
    base_model = use_model(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(img_size, img_size, 3),
    )
    x = base_model.output
    predictions = Dense(n_classes, activation="softmax")(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(
        optimizer="nadam",
        loss="categorical_crossentropy",
        metrics=["categorical_accuracy"],
    )
    return model


n_classes = train_labels.shape[1]

with strategy.scope():
    model1 = get_model(EfficientNetB7, n_classes=n_classes)
with strategy.scope():
    model2 = get_model(DenseNet201, n_classes=n_classes)




## === cell 7
model1.load_weights(w1)
print("Loaded model1 weights:", w1)

model2.load_weights(w2)
print("Loaded model2 weights:", w2)




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/2787012927.py in <cell line: 0>()
      1 # Weights must exist (validated earlier); loading is fast vs training.
----> 2 model1.load_weights(w1)
      3 print("Loaded model1 weights:", w1)
      4 
      5 model2.load_weights(w2)

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/h5py/_hl/files.py in __init__(self, name, mode, driver, libver, userblock_size, swmr, rdcc_nslots, rdcc_nbytes, rdcc_w0, track_order, fs_strategy, fs_persist, fs_threshold, fs_page_size, page_buf_size, min_meta_keep, min_raw_keep, locking, alignment_threshold, alignment_interval, meta_block_size, **kwds)
    562                                  fs_persist=fs_persist, fs_threshold=fs_threshold,
    563                                  fs_page_size=fs_page_size)
--> 564                 fid = make_fid(name, mode, userblock_size, fapl, fcpl, swmr=swmr)
    565 
    566             if isinstance(libver, tuple):

/usr/local/lib/python3.11/dist-packages/h5py/_hl/files.py in make_fid(name, mode, userblock_size, fapl, fcpl, swmr)
    236         if swmr and swmr_support:
    237             flags |= h5f.ACC_SWMR_READ
--> 238         fid = h5f.open(name, flags, fapl=fapl)
    239     elif mode == 'r+':
    240         fid = h5f.open(name, h5f.ACC_RDWR, fapl=fapl)

h5py/_objects.pyx in h5py._objects.with_phil.wrapper()

h5py/_objects.pyx in h5py._objects.with_phil.wrapper()

h5py/h5f.pyx in h5py.h5f.open()

FileNotFoundError: [Errno 2] Unable to synchronously open file (unable to open file: name = '/kaggle/input/tf-zoo-models-on-tpu-efficientnetb7/my_ef_net_b7.h5', errno = 2, error message = 'No such file or directory', flags = 0, o_flags = 0)

## === cell 8
best_alpha = 0.52

test_steps = math.ceil(len(test_paths) / BATCH_SIZE)

print("Computing predictions...")
probabilities1 = model1.predict(test_dataset, steps=test_steps, verbose=1)
probabilities2 = model2.predict(test_dataset, steps=test_steps, verbose=1)

probabilities = best_alpha * probabilities1 + (1.0 - best_alpha) * probabilities2

probabilities = np.clip(probabilities, 0.0, 1.0)
assert probabilities.shape == (len(test), len(target_cols)), (
    probabilities.shape,
    len(test),
    len(target_cols),
)

sub = sub.copy()
sub["image_id"] = test["image_id"].values
sub.loc[:, target_cols] = probabilities

sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2105806265.py in <cell line: 0>()
      4 
      5 print("Computing predictions...")
----> 6 probabilities1 = model1.predict(test_dataset, steps=test_steps, verbose=1)
      7 probabilities2 = model2.predict(test_dataset, steps=test_steps, verbose=1)
      8 

NameError: name 'test_dataset' is not defined
