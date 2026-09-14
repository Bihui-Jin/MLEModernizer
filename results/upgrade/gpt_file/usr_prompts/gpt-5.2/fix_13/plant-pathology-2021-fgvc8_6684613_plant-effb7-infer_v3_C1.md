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
Mean F1-Score

## Submission Format
labels should be a space-delimited list.

The file should contain a header and have the following format:

```
image, labels
85f8cb619c66b863.jpg,healthy
ad8770db05586b59.jpg,healthy
c7b03e718489f3ca.jpg,healthy
```

## Dataset
**train.csv** - the training set metadata.

- `image` - the image ID.
- `labels` - the target classes, a space delimited list of all diseases found in the image. Unhealthy leaves with too many diseases to classify visually will have the `complex` class, and may also have a subset of the diseases identified.

**sample_submission.csv** - A sample submission file in the correct format.

- `image`
- `labels`

**train_images** - The training set images.

**test_images** - The test set images. This competition has a hidden test set: only three images are provided here as samples while the remaining 5,000 images will be available to your notebook once it is submitted.

# 2. Python version

3.9

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
        input/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
        working/
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
```

-> data/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> data/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> (stopped after 10 files for performance)

# 5. Target score

0.7740904893813486

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import tensorflow as tf

import matplotlib.pyplot as plt  # noqa: F401

from glob import glob  # noqa: F401
from random import seed, randint, random, choice  # noqa: F401
from PIL import Image  # noqa: F401

print("TF version:", tf.__version__)

SEED = 42
tf.keras.utils.set_random_seed(SEED)

os.environ.pop("XLA_FLAGS", None)
try:
    tf.config.optimizer.set_jit(False)
except Exception:
    pass

try:
    for _gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(_gpu, True)
except Exception:
    pass




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def auto_select_accelerator():
    """
    Tries TPU, otherwise defaults to the available strategy (GPU/CPU).
    """
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.TPUStrategy(tpu)
        print("Running on TPU:", tpu.master())
    except Exception as e:
        strategy = tf.distribute.get_strategy()
        print("TPU not found/usable, using default strategy. Reason:", repr(e))
    print(f"Running on {strategy.num_replicas_in_sync} replicas")
    return strategy




## === cell 2
IMSIZES = (224, 240, 260, 300, 380, 456, 528, 600)
im_size = IMSIZES[7]  # 600

load_dir = "/kaggle/input/plant-pathology-2021-fgvc8/"
df = pd.read_csv(os.path.join(load_dir, "train.csv"))

all_tokens = set(" ".join(df["labels"].astype(str).values).split())
class_name = sorted(list(all_tokens))

print("Classes:", class_name)
print("Num classes:", len(class_name))

n_labels = len(class_name)




## === cell 3
strategy = auto_select_accelerator()
BATCH_SIZE = strategy.num_replicas_in_sync * 16

test_dir = "/kaggle/input/plant-pathology-2021-fgvc8/test_images/"
test_df = pd.DataFrame({"image": sorted(os.listdir(test_dir))})

print("Test images:", len(test_df))




## === cell 4
AUTOTUNE = tf.data.AUTOTUNE

test_filenames = test_df["image"].values
test_paths = (test_dir + test_df["image"]).values


def _load_decode_resize_preprocess(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)  # matches PIL/keras default RGB
    img = tf.image.resize(
        img, [im_size, im_size], method=tf.image.ResizeMethod.BILINEAR
    )
    img = tf.cast(img, tf.float32)
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    return img


base_ds = tf.data.Dataset.from_tensor_slices(test_paths)

options = tf.data.Options()
try:
    options.experimental_deterministic = True
except Exception:
    pass
try:
    options.experimental_optimization.map_and_batch_fusion = True
except Exception:
    pass
try:
    options.experimental_optimization.parallel_batch = True
except Exception:
    pass
try:
    options.experimental_optimization.apply_default_optimizations = True
except Exception:
    pass
base_ds = base_ds.with_options(options)

base_ds = base_ds.map(
    _load_decode_resize_preprocess, num_parallel_calls=AUTOTUNE, deterministic=True
)
base_ds = base_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)

test_steps = int(np.ceil(len(test_df) / BATCH_SIZE))

test_imgs = np.empty((len(test_df), im_size, im_size, 3), dtype=np.float32)
offset = 0
for batch in base_ds:
    bsz = int(batch.shape[0])
    test_imgs[offset : offset + bsz] = batch.numpy()
    offset += bsz
assert offset == len(test_df), (offset, len(test_df))




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
FailedPreconditionError                   Traceback (most recent call last)
/tmp/ipykernel_11/1278832031.py in <cell line: 0>()
     50 test_imgs = np.empty((len(test_df), im_size, im_size, 3), dtype=np.float32)
     51 offset = 0
---> 52 for batch in base_ds:
     53     bsz = int(batch.shape[0])
     54     test_imgs[offset : offset + bsz] = batch.numpy()

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/iterator_ops.py in __next__(self)
    824   def __next__(self):
    825     try:
--> 826       return self._next_internal()
    827     except errors.OutOfRangeError:
    828       raise StopIteration

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/iterator_ops.py in _next_internal(self)
    774     # to communicate that there is no more data to iterate over.
    775     with context.execution_mode(context.SYNC):
--> 776       ret = gen_dataset_ops.iterator_get_next(
    777           self._iterator_resource,
    778           output_types=self._flat_output_types,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/gen_dataset_ops.py in iterator_get_next(iterator, output_types, output_shapes, name)
   3084       return _result
   3085     except _core._NotOkStatusException as e:
-> 3086       _ops.raise_from_not_ok_status(e, name)
   3087     except _core._FallbackException:
   3088       pass

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py in raise_from_not_ok_status(e, name)
   6000 def raise_from_not_ok_status(e, name) -> NoReturn:
   6001   e.message += (" name: " + str(name if name is not None else ""))
-> 6002   raise core._status_to_exception(e) from None  # pylint: disable=protected-access
   6003 
   6004 

FailedPreconditionError: {{function_node __wrapped__IteratorGetNext_output_types_1_device_/job:localhost/replica:0/task:0/device:CPU:0}} /kaggle/input/plant-pathology-2021-fgvc8/test_images/test_images; Is a directory
	 [[{{node ReadFile}}]] [Op:IteratorGetNext] name: 

## === cell 5
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalMaxPooling2D

with strategy.scope():
    base = tf.keras.applications.EfficientNetB7(
        weights=None,
        include_top=False,
        input_shape=(im_size, im_size, 3),
    )

    model = Sequential()
    model.add(base)
    model.add(GlobalMaxPooling2D())
    model.add(Dense(n_labels, activation="softmax"))

    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(learning_rate=4e-4),
        metrics=["accuracy"],
        jit_compile=False,
    )

model.summary()




## === cell 6
weights_path = "/kaggle/input/model111/bestmodel_tpu_aug.h5"
if tf.io.gfile.exists(weights_path):
    model.load_weights(weights_path)
    print("Loaded weights:", weights_path)
else:
    print("WARNING: Weights file not found:", weights_path)
    print(
        "Proceeding with randomly initialized weights (submission will be valid but score will be low)."
    )




## === cell 7
TTA = 5

n_test = len(test_df)
pred_sum = np.zeros((n_test, n_labels), dtype=np.float32)

test_imgs_tf = tf.constant(test_imgs)  # fixed tensor for deterministic stateless ops


@tf.function(reduce_retracing=True)
def _tta_flip_all(x, tta_id):
    n = tf.shape(x)[0]
    idx = tf.range(n, dtype=tf.int32)
    seed0 = tf.fill(
        [n], tf.cast(SEED, tf.int32) ^ (tf.cast(tta_id, tf.int32) * 1000003)
    )
    seeds = tf.stack([seed0, idx], axis=1)  # [n, 2]
    return tf.image.stateless_random_flip_left_right(x, seed=seeds)


@tf.function(reduce_retracing=True)
def _predict_tensor(x):
    return model(x, training=False)


for tta_id in range(TTA):
    x_aug = _tta_flip_all(test_imgs_tf, tf.cast(tta_id, tf.int32))
    preds_list = []
    for i in range(0, n_test, BATCH_SIZE):
        preds_list.append(_predict_tensor(x_aug[i : i + BATCH_SIZE]))
    preds = tf.concat(preds_list, axis=0).numpy()
    pred_sum += preds.astype(np.float32, copy=False)

pred = pred_sum / float(TTA)

argpred = np.argmax(pred, axis=1)
test_df["labels"] = [class_name[i] for i in argpred]

sub_path = "submission.csv"
test_df[["image", "labels"]].to_csv(sub_path, index=False)

print("Wrote:", sub_path)
print(test_df.head())

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3131924077.py in <cell line: 0>()
     27 
     28 for tta_id in range(TTA):
---> 29     x_aug = _tta_flip_all(test_imgs_tf, tf.cast(tta_id, tf.int32))
     30     # Predict in batches to control memory; identical semantics to predict over a dataset.
     31     preds_list = []

/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/traceback_utils.py in error_handler(*args, **kwargs)
    151     except Exception as e:
    152       filtered_tb = _process_traceback_frames(e.__traceback__)
--> 153       raise e.with_traceback(filtered_tb) from None
    154     finally:
    155       del filtered_tb

/tmp/__autograph_generated_filex3hyeg0y.py in tf___tta_flip_all(x, tta_id)
     17                 except:
     18                     do_return = False
---> 19                     raise
     20                 return fscope.ret(retval_, do_return)
     21         return tf___tta_flip_all

ValueError: in user code:

    File "/tmp/ipykernel_11/3131924077.py", line 20, in _tta_flip_all  *
        return tf.image.stateless_random_flip_left_right(x, seed=seeds)

    ValueError: Shape must be rank 1 but is rank 2 for '{{node stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter}} = StatelessRandomGetKeyCounter[Tseed=DT_INT32](stack)' with input shapes: [3728,2].
