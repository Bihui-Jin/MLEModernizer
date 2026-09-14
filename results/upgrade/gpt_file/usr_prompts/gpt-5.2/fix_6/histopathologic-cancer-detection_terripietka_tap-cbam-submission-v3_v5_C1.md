# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os

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
    mean = img_float_0_1.mean(axis=(0, 1, 2), keepdims=True)
    var = img_float_0_1.var(axis=(0, 1, 2), keepdims=True)
    std = np.sqrt(var)
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
        arr = np.asarray(im, dtype=np.float32) / 255.0
    arr = standardize_np(arr).astype(np.float32)
    return arr


def make_xy(df, img_dir, target_size=(96, 96)):
    n = len(df)
    x = np.empty((n, target_size[0], target_size[1], 3), dtype=np.float32)
    y = np.empty((n,), dtype=np.int64)
    ids = df["id"].astype(str).values
    labels = df["label"].values
    for i, (img_id, lab) in enumerate(zip(ids, labels)):
        x[i] = load_image(
            os.path.join(img_dir, f"{img_id}.tif"), target_size=target_size
        )
        y[i] = lab
    return x, y


MAX_TRAIN = 30000
MAX_VAL = 6000

train_split_df_sub = train_split_df.sample(
    n=min(MAX_TRAIN, len(train_split_df)), random_state=SEED
).reset_index(drop=True)
val_split_df_sub = val_split_df.sample(
    n=min(MAX_VAL, len(val_split_df)), random_state=SEED
).reset_index(drop=True)

X_train, y_train = make_xy(train_split_df_sub, train_images, target_size=TARGET_SIZE)
X_val, y_val = make_xy(val_split_df_sub, train_images, target_size=TARGET_SIZE)

print("Loaded X_train:", X_train.shape, "y_train:", y_train.shape)
print("Loaded X_val:", X_val.shape, "y_val:", y_val.shape)



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



## === cell 8
cnn.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=[keras.metrics.AUC(name="auc")],
)

EPOCHS = 2  # unchanged

history = cnn.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1,
)



## === cell 9
test_ids = test_df["id"].astype(str).values


def make_x_from_ids(ids, img_dir, target_size=(96, 96), batch_size=64):
    n = len(ids)
    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)
        batch_ids = ids[start:end]
        x = np.empty(
            (len(batch_ids), target_size[0], target_size[1], 3), dtype=np.float32
        )
        for i, img_id in enumerate(batch_ids):
            x[i] = load_image(
                os.path.join(img_dir, f"{img_id}.tif"), target_size=target_size
            )
        yield x


preds_list = []
for xb in make_x_from_ids(
    test_ids, test_images, target_size=TARGET_SIZE, batch_size=BATCH_SIZE
):
    preds_list.append(cnn.predict(xb, verbose=0))
test_preds = np.concatenate(preds_list, axis=0)

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
