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
Given a dataset of images from digital pathology scans, predict if the center 32x32px region of a patch contains at least one pixel of tumor tissue. Tumor tissue in the outer region of the patch does not influence the label. 

## Metric
Area under the ROC curve.

## Submission Format
For each `id` in the test set, you must predict a probability that center 32x32px region of a patch contains at least one pixel of tumor tissue. The file should contain a header and have the following format:

```
id,label
0b2ea2a822ad23fdb1b5dd26653da899fbd2c0d5,0
95596b92e5066c5c52466c90b69ff089b39f2737,0
248e6738860e2ebcf6258cdc1f32f299e0c76914,0
etc.
```

## Dataset
Files are named with an image `id`. The `train_labels.csv` file provides the ground truth for the images in the `train` folder. You are predicting the labels for the images in the `test` folder.

# 2. Python version

3.13

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (63 lines)
            sample_submission.csv (45562 lines)
            sample_submission.csv.zip (1.1 MB)
            test.zip (1.1 GB)
            train.zip (4.2 GB)
            train_labels.csv (174465 lines)
            train_labels.csv.zip (4.2 MB)
            histopathologic-cancer-detection/
                description.md (63 lines)
                sample_submission.csv (45562 lines)
                ... and 5 other files
                histopathologic-cancer-detection/
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
            test/
                7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                ... and 45559 other files
                test/
            train/
                bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                ... and 174462 other files
                train/
        input/
            description.md (63 lines)
            sample_submission.csv (45562 lines)
            sample_submission.csv.zip (1.1 MB)
            test.zip (1.1 GB)
            train.zip (4.2 GB)
            train_labels.csv (174465 lines)
            train_labels.csv.zip (4.2 MB)
            histopathologic-cancer-detection/
                description.md (63 lines)
                sample_submission.csv (45562 lines)
                ... and 5 other files
                histopathologic-cancer-detection/
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
            test/
                7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                ... and 45559 other files
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
            train/
                bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                ... and 174462 other files
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
        working/
            histopathologic-cancer-detection/
                description.md (63 lines)
                sample_submission.csv (45562 lines)
                ... and 5 other files
                histopathologic-cancer-detection/
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
```

-> data/histopathologic-cancer-detection/sample_submission.csv has 45561 rows and 2 columns.
The columns are: id, label

-> data/histopathologic-cancer-detection/train_labels.csv has 174464 rows and 2 columns.
The columns are: id, label

-> data/sample_submission.csv has 45561 rows and 2 columns.
The columns are: id, label

-> data/train_labels.csv has 174464 rows and 2 columns.
The columns are: id, label

-> input/histopathologic-cancer-detection/sample_submission.csv has 45561 rows and 2 columns.
The columns are: id, label

-> input/histopathologic-cancer-detection/train_labels.csv has 174464 rows and 2 columns.
The columns are: id, label

-> (stopped after 10 files for performance)

# 5. Target score

0.753736212726282

# 6. Current score

0.66296

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.66296) has done: 'I fix the Keras runtime/import issue that triggers the `MessageFactory.GetPrototype` error by forcing Keras to use the TensorFlow backend (compatible in Kaggle) before importing `keras`. I also remove the now-unsupported `workers/use_multiprocessing/max_queue_size` arguments from `fit()` and `predict()` (new Keras API), which currently prevents training/inference from running at all. Finally, I keep the model/training logic the same and ensure predictions are generated and written to a valid `submission.csv` with the required `id,label` columns.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import random

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

print("Python:", os.sys.version)



## === cell 1
BASE = "/kaggle/input/histopathologic-cancer-detection"
train_images = f"{BASE}/train/"
test_images = f"{BASE}/test/"
train_csv = f"{BASE}/train_labels.csv"
sample_sub_csv = f"{BASE}/sample_submission.csv"

assert os.path.exists(train_images), f"Missing path: {train_images}"
assert os.path.exists(test_images), f"Missing path: {test_images}"
assert os.path.exists(train_csv), f"Missing path: {train_csv}"
assert os.path.exists(sample_sub_csv), f"Missing path: {sample_sub_csv}"



## === cell 2
test_df = pd.read_csv(sample_sub_csv)
print("Test Set Size:", test_df.shape)
test_df.head()



## === cell 3
train_df = pd.read_csv(train_csv)
train_df["label"] = train_df["label"].astype(int)

print("Train Set Size:", train_df.shape)
train_df.head()



## === cell 4
BATCH_SIZE = 64
TARGET_SIZE = (96, 96)




## === cell 5
def standardize_np(img_float_0_1: np.ndarray) -> np.ndarray:
    mean = img_float_0_1.mean(dtype=np.float32)
    var = img_float_0_1.var(dtype=np.float32)
    std = np.sqrt(var, dtype=np.float32)
    return (img_float_0_1 - mean) / (std + 1e-7)




## === cell 6
from PIL import Image
from sklearn.model_selection import train_test_split

train_split_df, val_split_df = train_test_split(
    train_df, test_size=0.10, random_state=SEED, stratify=train_df["label"]
)


def load_image(path, target_size=(96, 96)):
    with Image.open(path) as im:
        im = im.convert("RGB")
        if im.size != (target_size[1], target_size[0]):
            im = im.resize((target_size[1], target_size[0]), resample=Image.BILINEAR)
        arr = np.asarray(im, dtype=np.float32) * (1.0 / 255.0)
    arr = standardize_np(arr).astype(np.float32, copy=False)
    return arr


MAX_TRAIN = 30000
MAX_VAL = 6000

train_split_df_sub = train_split_df.sample(
    n=min(MAX_TRAIN, len(train_split_df)), random_state=SEED
).reset_index(drop=True)
val_split_df_sub = val_split_df.sample(
    n=min(MAX_VAL, len(val_split_df)), random_state=SEED
).reset_index(drop=True)

print("Train subset:", train_split_df_sub.shape, "Val subset:", val_split_df_sub.shape)



## === cell 7
import keras
from keras import layers
from keras.layers import (
    Layer,
    GlobalAveragePooling2D,
    GlobalMaxPooling2D,
    Dense,
    Conv2D,
)
from keras.initializers import lecun_normal
from keras.utils import register_keras_serializable


@register_keras_serializable(package="Custom")
class CBAM(Layer):
    def __init__(self, channels, reduction_ratio=16, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.reduction_ratio = reduction_ratio

        self.global_avg_pool = GlobalAveragePooling2D()
        self.global_max_pool = GlobalMaxPooling2D()
        self.fc1 = Dense(
            units=channels // reduction_ratio,
            activation="selu",
            kernel_initializer=lecun_normal(),
        )
        self.fc2 = Dense(
            units=channels,
            activation="sigmoid",
            kernel_initializer=lecun_normal(),
        )

        self.conv = Conv2D(
            filters=1,
            kernel_size=7,
            padding="same",
            activation="sigmoid",
            kernel_initializer=lecun_normal(),
        )

    def call(self, inputs, **kwargs):
        avg_pooled = self.global_avg_pool(inputs)
        max_pooled = self.global_max_pool(inputs)
        avg_fc = self.fc2(self.fc1(avg_pooled))
        max_fc = self.fc2(self.fc1(max_pooled))
        channel_attention = avg_fc + max_fc
        channel_attention = keras.ops.expand_dims(channel_attention, axis=1)
        channel_attention = keras.ops.expand_dims(channel_attention, axis=1)
        channel_refined = inputs * channel_attention

        avg_spatial = keras.ops.mean(channel_refined, axis=-1, keepdims=True)
        max_spatial = keras.ops.max(channel_refined, axis=-1, keepdims=True)
        spatial_attention = self.conv(
            keras.ops.concatenate([avg_spatial, max_spatial], axis=-1)
        )
        spatial_refined = channel_refined * spatial_attention
        return spatial_refined

    def get_config(self):
        config = super().get_config()
        config.update(
            {"channels": self.channels, "reduction_ratio": self.reduction_ratio}
        )
        return config


@register_keras_serializable(package="Custom")
class CustomAlphaDropout(Layer):
    def __init__(self, rate, **kwargs):
        super().__init__(**kwargs)
        self.rate = rate

    def call(self, inputs, training=None):
        if training is None:
            training = False
        if not training:
            return inputs
        keep_prob = 1.0 - self.rate
        rnd = keras.random.uniform(
            shape=keras.ops.shape(inputs), minval=0.0, maxval=1.0, seed=SEED
        )
        binary_tensor = keras.ops.floor(keep_prob + rnd)
        outputs = inputs * binary_tensor / keep_prob
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({"rate": self.rate})
        return config


def build_model(input_shape=(96, 96, 3)):
    inputs = keras.Input(shape=input_shape)

    x = layers.Conv2D(32, 3, padding="same", kernel_initializer=lecun_normal())(inputs)
    x = layers.Activation("selu")(x)
    x = CBAM(32)(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, 3, padding="same", kernel_initializer=lecun_normal())(x)
    x = layers.Activation("selu")(x)
    x = CBAM(64)(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, 3, padding="same", kernel_initializer=lecun_normal())(x)
    x = layers.Activation("selu")(x)
    x = CBAM(128)(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(256, 3, padding="same", kernel_initializer=lecun_normal())(x)
    x = layers.Activation("selu")(x)
    x = CBAM(256)(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="selu", kernel_initializer=lecun_normal())(x)
    x = CustomAlphaDropout(0.2)(x)
    outputs = layers.Dense(2, activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    return model


cnn = build_model(input_shape=(96, 96, 3))
cnn.summary()



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 8
import math


class HCDSequence(keras.utils.Sequence):
    def __init__(
        self,
        df,
        img_dir,
        target_size=(96, 96),
        batch_size=64,
        shuffle=False,
        seed=42,
        with_labels=True,
    ):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.target_size = target_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.with_labels = with_labels

        self.ids = self.df["id"].astype(str).values
        self.labels = (
            self.df["label"].values.astype(np.int64, copy=False)
            if with_labels and "label" in self.df.columns
            else None
        )

        self.indexes = np.arange(len(self.ids), dtype=np.int64)
        self.rng = np.random.RandomState(self.seed)
        self.on_epoch_end()

    def __len__(self):
        return (len(self.indexes) + self.batch_size - 1) // self.batch_size

    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self.indexes)

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = min(len(self.indexes), start + self.batch_size)
        batch_idx = self.indexes[start:end]
        b = len(batch_idx)

        x = np.empty((b, self.target_size[0], self.target_size[1], 3), dtype=np.float32)
        if self.with_labels:
            y = np.empty((b,), dtype=np.int64)

        for i, j in enumerate(batch_idx):
            img_id = self.ids[j]
            x[i] = load_image(
                os.path.join(self.img_dir, f"{img_id}.tif"),
                target_size=self.target_size,
            )
            if self.with_labels:
                y[i] = self.labels[j]

        return (x, y) if self.with_labels else x


train_seq = HCDSequence(
    train_split_df_sub,
    train_images,
    target_size=TARGET_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED,
    with_labels=True,
)
val_seq = HCDSequence(
    val_split_df_sub,
    train_images,
    target_size=TARGET_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
    seed=SEED,
    with_labels=True,
)

cnn.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=[keras.metrics.AUC(name="auc")],
)

EPOCHS = 2  # unchanged

history = cnn.fit(
    train_seq,
    validation_data=val_seq,
    epochs=EPOCHS,
    verbose=1,
)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
InvalidArgumentError                      Traceback (most recent call last)
/tmp/ipykernel_11/2897434513.py in <cell line: 0>()
     89 
     90 # Fix: New Keras Trainer API doesn't accept workers/use_multiprocessing/max_queue_size here.
---> 91 history = cnn.fit(
     92     train_seq,
     93     validation_data=val_seq,

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

InvalidArgumentError: Graph execution error:

Detected at node UnsortedSegmentSum defined at (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main

  File "<frozen runpy>", line 88, in _run_code

  File "/usr/local/lib/python3.11/dist-packages/colab_kernel_launcher.py", line 37, in <module>

  File "/usr/local/lib/python3.11/dist-packages/traitlets/config/application.py", line 992, in launch_instance

  File "/usr/local/lib/python3.11/dist-packages/ipykernel/kernelapp.py", line 712, in start

  File "/usr/local/lib/python3.11/dist-packages/tornado/platform/asyncio.py", line 211, in start

  File "/usr/lib/python3.11/asyncio/base_events.py", line 608, in run_forever

  File "/usr/lib/python3.11/asyncio/base_events.py", line 1936, in _run_once

  File "/usr/lib/python3.11/asyncio/events.py", line 84, in _run

  File "/usr/local/lib/python3.11/dist-packages/ipykernel/kernelbase.py", line 510, in dispatch_queue

  File "/usr/local/lib/python3.11/dist-packages/ipykernel/kernelbase.py", line 499, in process_one

  File "/usr/local/lib/python3.11/dist-packages/ipykernel/kernelbase.py", line 406, in dispatch_shell

  File "/usr/local/lib/python3.11/dist-packages/ipykernel/kernelbase.py", line 730, in execute_request

  File "/usr/local/lib/python3.11/dist-packages/ipykernel/ipkernel.py", line 383, in do_execute

  File "/usr/local/lib/python3.11/dist-packages/ipykernel/zmqshell.py", line 528, in run_cell

  File "/usr/local/lib/python3.11/dist-packages/IPython/core/interactiveshell.py", line 2975, in run_cell

  File "/usr/local/lib/python3.11/dist-packages/IPython/core/interactiveshell.py", line 3030, in _run_cell

  File "/usr/local/lib/python3.11/dist-packages/IPython/core/async_helpers.py", line 78, in _pseudo_sync_runner

  File "/usr/local/lib/python3.11/dist-packages/IPython/core/interactiveshell.py", line 3257, in run_cell_async

  File "/usr/local/lib/python3.11/dist-packages/IPython/core/interactiveshell.py", line 3473, in run_ast_nodes

  File "/usr/local/lib/python3.11/dist-packages/IPython/core/interactiveshell.py", line 3553, in run_code

  File "/tmp/ipykernel_11/2897434513.py", line 91, in <cell line: 0>

  File "/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py", line 117, in error_handler

  File "/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/trainer.py", line 371, in fit

  File "/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/trainer.py", line 219, in function

  File "/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/trainer.py", line 132, in multi_step_on_iterator

  File "/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/trainer.py", line 113, in one_step_on_data

  File "/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/trainer.py", line 84, in train_step

  File "/usr/local/lib/python3.11/dist-packages/keras/src/trainers/trainer.py", line 490, in compute_metrics

  File "/usr/local/lib/python3.11/dist-packages/keras/src/trainers/compile_utils.py", line 334, in update_state

  File "/usr/local/lib/python3.11/dist-packages/keras/src/trainers/compile_utils.py", line 21, in update_state

  File "/usr/local/lib/python3.11/dist-packages/keras/src/metrics/confusion_metrics.py", line 1376, in update_state

  File "/usr/local/lib/python3.11/dist-packages/keras/src/metrics/metrics_utils.py", line 481, in update_confusion_matrix_variables

  File "/usr/local/lib/python3.11/dist-packages/keras/src/metrics/metrics_utils.py", line 272, in _update_confusion_matrix_variables_optimized

  File "/usr/local/lib/python3.11/dist-packages/keras/src/ops/math.py", line 86, in segment_sum

  File "/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/math.py", line 23, in segment_sum

data.shape = [64] does not start with segment_ids.shape = [128]
	 [[{{node UnsortedSegmentSum}}]] [Op:__inference_multi_step_on_iterator_8048]

## === cell 9
test_ids = test_df["id"].astype(str).values

test_only_df = pd.DataFrame({"id": test_ids})
test_seq = HCDSequence(
    test_only_df,
    test_images,
    target_size=TARGET_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
    seed=SEED,
    with_labels=False,
)

test_preds = cnn.predict(
    test_seq,
    verbose=1,
)

print(
    "test_preds shape:",
    test_preds.shape,
    "min/max:",
    float(test_preds.min()),
    float(test_preds.max()),
)



## === cell 10
tumor_probs = test_preds[:, 1].astype(np.float32)



## === cell 11
submission = pd.DataFrame({"id": test_ids, "label": tumor_probs})
submission.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())



## === cell 12
frequency_distribution = (submission["label"].round(3).describe()).to_frame(
    name="label_stats"
)
print(frequency_distribution)
