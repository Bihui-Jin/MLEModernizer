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

3.8

# 2. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
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

# 3. Data file paths

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

# 4. Code solution

## === cell 0
import os
import sys
import subprocess

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "protobuf<5"])

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "3")

import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt

print("TF:", tf.__version__)
print("Num GPUs:", len(tf.config.list_physical_devices("GPU")))

tf.config.optimizer.set_jit(True)

tf.keras.utils.set_random_seed(2020)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass


## === cell 1
AUTO = tf.data.experimental.AUTOTUNE
try:
    from kaggle_datasets import KaggleDatasets

    GCS_DS_PATH = KaggleDatasets().get_gcs_path()
except Exception:
    GCS_DS_PATH = "/kaggle/input/plant-pathology-2020-fgvc7"

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print("Running on TPU", tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy()

print("REPLICAS:", strategy.num_replicas_in_sync)
print("GCS_DS_PATH:", GCS_DS_PATH)



## === cell 2
path = GCS_DS_PATH.rstrip("/") + "/"

train = pd.read_csv(path + "train.csv")
test = pd.read_csv(path + "test.csv")
sub = pd.read_csv(path + "sample_submission.csv")

label_cols = [c for c in sub.columns if c != "image_id"]

missing = [c for c in label_cols if c not in train.columns]
if missing:
    raise ValueError(
        f"Train is missing expected label columns from sample_submission.csv: {missing}. "
        f"Train columns: {list(train.columns)}"
    )

train = train[["image_id"] + label_cols]
train_labels = train[label_cols].values.astype(np.float32)

train_paths = (path + "images/" + train["image_id"].astype(str)).values
test_paths = (path + "images/" + test["image_id"].astype(str)).values

print(train.shape, test.shape, sub.shape)
print("Label columns:", label_cols)
print("Example train path:", train_paths[0])
print("Example test path:", test_paths[0])



## === cell 3
nb_classes = 4
BATCH_SIZE = 8 * strategy.num_replicas_in_sync
img_size = 768
EPOCHS = 40

img = plt.imread(path + "images/Train_0.jpg")
print("Example image shape:", img.shape)




## === cell 4
def decode_image(filename, label=None, image_size=(img_size, img_size)):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, image_size)
    if label is None:
        return image
    return image, label


def data_augment(image, label=None, seed=2020):
    s = tf.stack(
        [
            tf.cast(seed, tf.int32),
            tf.cast(tf.reduce_sum(tf.cast(image * 255.0, tf.int32)), tf.int32),
        ]
    )
    image = tf.image.stateless_random_flip_left_right(image, seed=s)
    image = tf.image.stateless_random_flip_up_down(
        image, seed=s + tf.constant([1, 1], tf.int32)
    )

    image = tf.image.adjust_saturation(image, 1.2)

    if label is None:
        return image
    return image, label




## === cell 5
options = tf.data.Options()
options.experimental_deterministic = True

try:
    if hasattr(options, "experimental_optimization") and hasattr(
        options.experimental_optimization, "autotune_buffers"
    ):
        options.experimental_optimization.autotune_buffers = True
except (AttributeError, TypeError):
    pass

try:
    if hasattr(options, "experimental_optimization") and hasattr(
        options.experimental_optimization, "map_parallelization"
    ):
        options.experimental_optimization.map_parallelization = True
except (AttributeError, TypeError):
    pass

train_ds_base = tf.data.Dataset.from_tensor_slices(
    (train_paths, train_labels)
).with_options(options)
test_ds_base = tf.data.Dataset.from_tensor_slices(test_paths).with_options(options)

train_dataset = (
    train_ds_base.map(decode_image, num_parallel_calls=AUTO)
    .cache()
    .shuffle(512, seed=2020, reshuffle_each_iteration=True)
    .repeat()
    .map(data_augment, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(AUTO)
)

test_dataset = (
    test_ds_base.map(decode_image, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTO)
)


## === cell 6
LR_START = 0.00001
LR_MAX = 0.0001 * strategy.num_replicas_in_sync
LR_MIN = 0.00001
LR_RAMPUP_EPOCHS = 15
LR_SUSTAIN_EPOCHS = 3
LR_EXP_DECAY = 0.8


def lrfn(epoch):
    if epoch < LR_RAMPUP_EPOCHS:
        lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START
    elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:
        lr = LR_MAX
    else:
        lr = (LR_MAX - LR_MIN) * LR_EXP_DECAY ** (
            epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS
        ) + LR_MIN
    return lr


lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=True)

rng = list(range(16))
y = [lrfn(x) for x in rng]
plt.plot(rng, y)
plt.title("Learning rate schedule (first 16 epochs)")
plt.show()
print("LR: {:.3g} -> {:.3g} -> {:.3g}".format(y[0], max(y), y[-1]))




## === cell 7
def get_model():
    pretrained_model = tf.keras.applications.EfficientNetB7(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(img_size, img_size, 3),
    )
    pretrained_model.trainable = True

    model = tf.keras.Sequential(
        [
            pretrained_model,
            tf.keras.layers.Dense(4, activation="sigmoid"),
        ]
    )
    return model


with strategy.scope():
    model = get_model()

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.AUC(curve="ROC", multi_label=True, num_labels=4, name="auc"),
    ],
)
model.summary()



## === cell 8
history = model.fit(
    train_dataset,
    steps_per_epoch=train_labels.shape[0] // BATCH_SIZE,
    callbacks=[lr_callback],
    epochs=16,
)



## --- ERROR in cell 8, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mNotFoundError[0m                             Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/779360443.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[0;32m----> 1[0;31m history = model.fit(
[0m[1;32m      2[0m     [0mtrain_dataset[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m      3[0m     [0msteps_per_epoch[0m[0;34m=[0m[0mtrain_labels[0m[0;34m.[0m[0mshape[0m[0;34m[[0m[0;36m0[0m[0;34m][0m [0;34m//[0m [0mBATCH_SIZE[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m      4[0m     [0mcallbacks[0m[0;34m=[0m[0;34m[[0m[0mlr_callback[0m[0;34m][0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m      5[0m     [0mepochs[0m[0;34m=[0m[0;36m16[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py[0m in [0;36merror_handler[0;34m(*args, **kwargs)[0m
[1;32m    120[0m             [0;31m# To get the full stack trace, call:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    121[0m             [0;31m# `keras.config.disable_traceback_filtering()`[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 122[0;31m             [0;32mraise[0m [0me[0m[0;34m.[0m[0mwith_traceback[0m[0;34m([0m[0mfiltered_tb[0m[0;34m)[0m [0;32mfrom[0m [0;32mNone[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    123[0m         [0;32mfinally[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    124[0m             [0;32mdel[0m [0mfiltered_tb[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/execute.py[0m in [0;36mquick_execute[0;34m(op_name, num_outputs, inputs, attrs, ctx, name)[0m
[1;32m     57[0m       [0me[0m[0;34m.[0m[0mmessage[0m [0;34m+=[0m [0;34m" name: "[0m [0;34m+[0m [0mname[0m[0;34m[0m[0;34m[0m[0m
[1;32m     58[0m     [0;32mraise[0m [0mcore[0m[0;34m.[0m[0m_status_to_exception[0m[0;34m([0m[0me[0m[0;34m)[0m [0;32mfrom[0m [0;32mNone[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 59[0;31m   [0;32mexcept[0m [0mTypeError[0m [0;32mas[0m [0me[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     60[0m     [0mkeras_symbolic_tensors[0m [0;34m=[0m [0;34m[[0m[0mx[0m [0;32mfor[0m [0mx[0m [0;32min[0m [0minputs[0m [0;32mif[0m [0m_is_keras_symbolic_tensor[0m[0;34m([0m[0mx[0m[0;34m)[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[1;32m     61[0m     [0;32mif[0m [0mkeras_symbolic_tensors[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;31mNotFoundError[0m: Graph execution error:

Detected at node ReadFile defined at (most recent call last):
<stack traces unavailable>
Detected at node ReadFile defined at (most recent call last):
<stack traces unavailable>
2 root error(s) found.
  (0) NOT_FOUND:  Error in user-defined function passed to ParallelMapDatasetV2:4 transformation with iterator: Iterator::Root::Prefetch::MapAndBatch::ShuffleAndRepeat::MemoryCacheImpl::ParallelMapV2: /kaggle/input/plant-pathology-2020-fgvc7/images/Train_0; No such file or directory
	 [[{{node ReadFile}}]]
	 [[IteratorGetNext]]
	 [[IteratorGetNext/_2]]
  (1) NOT_FOUND:  Error in user-defined function passed to ParallelMapDatasetV2:4 transformation with iterator: Iterator::Root::Prefetch::MapAndBatch::ShuffleAndRepeat::MemoryCacheImpl::ParallelMapV2: /kaggle/input/plant-pathology-2020-fgvc7/images/Train_0; No such file or directory
	 [[{{node ReadFile}}]]
	 [[IteratorGetNext]]
0 successful operations.
0 derived errors ignored. [Op:__inference_multi_step_on_iterator_159334]

## === cell 9
probs = model.predict(test_dataset, verbose=1)
probs = probs[: len(sub)]

if probs.ndim != 2 or probs.shape[1] != len(label_cols):
    raise ValueError(
        f"Expected probs with shape (n,{len(label_cols)}); got {probs.shape}. "
        f"label_cols={label_cols}"
    )
if len(sub) != probs.shape[0]:
    raise ValueError(f"Row mismatch: sub {len(sub)} vs probs {probs.shape[0]}")

sub_out = sub.copy()
sub_out.loc[:, label_cols] = probs.astype(np.float32)

sub_out = sub_out[["image_id"] + label_cols]
sub_out.to_csv("submission.csv", index=False)

print(sub_out.head())
print("Wrote submission.csv with shape:", sub_out.shape)
print("Columns:", list(sub_out.columns))
