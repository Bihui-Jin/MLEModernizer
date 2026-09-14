# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.90752

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.66296) has done: 'I fix the Keras runtime/import issue that triggers the `MessageFactory.GetPrototype` error by forcing Keras to use the TensorFlow backend (compatible in Kaggle) before importing `keras`. I also remove the now-unsupported `workers/use_multiprocessing/max_queue_size` arguments from `fit()` and `predict()` (new Keras API), which currently prevents training/inference from running at all. Finally, I keep the model/training logic the same and ensure predictions are generated and written to a valid `submission.csv` with the required `id,label` columns.'
- What this solution (achieved 0.90752) has done: 'I fix the two runtime blockers: (1) the protobuf `MessageFactory.GetPrototype` crash by pinning a compatible pure-Python protobuf implementation before importing TensorFlow/Keras, and (2) the `AUC` metric shape mismatch by changing the metric to binary AUC while keeping the same 2-class softmax output/loss. These are execution/correctness fixes that should also improve AUC because the metric now be computed correctly during training/validation (previously it crashed). I keep your model architecture, preprocessing, data sampling, and training loop intact, and ensure the script always writes a valid `submission.csv` with `id,label`.'
- What this solution (achieved 0.90752) has done: 'I fix the protobuf/TensorFlow import crash by forcing protobuf to use the pure-Python implementation *before* importing Keras/TensorFlow, instead of trying to load the incompatible C++ protobuf extension. Then I make the Keras import deterministic (no try/except that leaves `keras` undefined) so the later cells can compile/train/predict without cascading `NameError`s. Finally, I keep your model, preprocessing, subset sizes, and 2-epoch training unchanged, and ensure we always write a valid `submission.csv` with `id,label` aligned to `sample_submission.csv`.'
- What this solution (achieved 0.90752) has done: 'We fix the protobuf/TensorFlow/Keras import crash by setting the protobuf implementation and backend flags before any Keras/TensorFlow import, and by importing `tensorflow` explicitly first (this prevents the `MessageFactory.GetPrototype` failure seen in your cell 7). To avoid changing score (your current 0.90752 is well above the 0.7537 target), we not change the model, data sampling, epochs, optimizer, or preprocessing; the only functional edits are import-order stability and a small robustness tweak to the custom AUC metric to ensure correct dtype/shape handling. Finally, we ensure the pipeline always reaches the end and writes a valid `submission.csv` with the required `id,label` columns aligned to `sample_submission.csv`.'
- What this solution (achieved 0.90752) has done: 'I fix the protobuf/TensorFlow import crash that currently prevents the model from even building by forcing a compatible protobuf version and pure-Python implementation before importing TensorFlow/Keras. To keep your score from overshooting the target (your current 0.90752 is far above 0.7537), I not change your model, data sampling, preprocessing, epochs, optimizer, or prediction logic—only the import/runtime stability. I also keep the submission creation exactly aligned to `sample_submission.csv` and ensure `submission.csv` is always written with the required `id,label` columns.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

try:
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "protobuf<5"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
except Exception:
    pass

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
import tensorflow as tf  # noqa: F401

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



## === cell 8
import math  # noqa: F401


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

binary_auc = keras.metrics.AUC(name="auc")


def auc_on_positive_class(y_true, y_pred):
    y_true = keras.ops.cast(y_true, "float32")
    return binary_auc(y_true, y_pred[:, 1])


cnn.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=[auc_on_positive_class],
)

EPOCHS = 2  # unchanged

history = cnn.fit(
    train_seq,
    validation_data=val_seq,
    epochs=EPOCHS,
    verbose=1,
)



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

submission = test_df[["id"]].merge(submission, on="id", how="left")

assert submission.shape[0] == test_df.shape[0]
assert submission["label"].isna().sum() == 0, "Missing predictions for some test ids"

submission.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())



## === cell 12
frequency_distribution = (submission["label"].round(3).describe()).to_frame(
    name="label_stats"
)
print(frequency_distribution)
