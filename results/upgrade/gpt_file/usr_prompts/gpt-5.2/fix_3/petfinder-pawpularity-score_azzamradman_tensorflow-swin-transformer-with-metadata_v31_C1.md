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
Predict engagement with a pet's profile based on the photograph for that profile.

## Metric
Root mean squared error.

## Submission Format
For each `Id` in the test set, you must predict a probability for the target variable, `Pawpularity`. The file should contain a header and have the following format:

```
Id, Pawpularity
0008dbfb52aa1dc6ee51ee02adf13537, 99.24
0014a7b528f1682f0cf3b73a991c17a0, 61.71
0019c1388dfcd30ac8b112fb4250c251, 6.23
00307b779c82716b240a24f028b0031b, 9.43
00320c6dd5b4223c62a9670110d47911, 70.89
etc.
```

## Dataset
- **train/** - Folder containing training set photos of the form **{id}.jpg**, where **{id}** is a unique Pet Profile ID.
- **train.csv** - Metadata (described below) for each photo in the training set as well as the target, the photo's Pawpularity score. The Id column gives the photo's unique Pet Profile ID corresponding the photo's file name.

The train.csv and test.csv files contain metadata for photos in the training set and test set, respectively. Each pet photo is labeled with the value of 1 (Yes) or 0 (No) for each of the following features:

- **Focus** - Pet stands out against uncluttered background, not too close / far.
- **Eyes** - Both eyes are facing front or near-front, with at least 1 eye / pupil decently clear.
- **Face** - Decently clear face, facing front or near-front.
- **Near** - Single pet taking up significant portion of photo (roughly over 50% of photo width or height).
- **Action** - Pet in the middle of an action (e.g., jumping).
- **Accessory** - Accompanying physical or digital accessory / prop (i.e. toy, digital sticker), excluding collar and leash.
- **Group** - More than 1 pet in the photo.
- **Collage** - Digitally-retouched photo (i.e. with digital photo frame, combination of multiple photos).
- **Human** - Human in the photo.
- **Occlusion** - Specific undesirable objects blocking part of the pet (i.e. human, cage or fence). Note that not all blocking objects are considered occlusion.
- **Info** - Custom-added text or labels (i.e. pet name, description).
- **Blur** - Noticeably out of focus or noisy, especially for the pet's eyes and face. For Blur entries, "Eyes" column is always set to 0.

# 2. Python version

3.10

# 3. Installed packages

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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (132 lines)
            sample_submission.csv (993 lines)
            sample_submission.csv.zip (22.7 kB)
            test.csv (993 lines)
            test.csv.zip (22.5 kB)
            test.zip (102.2 MB)
            train.csv (8921 lines)
            train.csv.zip (213.0 kB)
            train.zip (926.9 MB)
            petfinder-pawpularity-score/
                description.md (132 lines)
                sample_submission.csv (993 lines)
                ... and 7 other files
                petfinder-pawpularity-score/
                test/
                    a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                    2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                    ... and 990 other files
                    test/
                train/
                    e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                    cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                    ... and 8918 other files
                    train/
            test/
                a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                ... and 990 other files
                test/
            train/
                e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                ... and 8918 other files
                train/
        input/
            description.md (132 lines)
            sample_submission.csv (993 lines)
            sample_submission.csv.zip (22.7 kB)
            test.csv (993 lines)
            test.csv.zip (22.5 kB)
            test.zip (102.2 MB)
            train.csv (8921 lines)
            train.csv.zip (213.0 kB)
            train.zip (926.9 MB)
            petfinder-pawpularity-score/
                description.md (132 lines)
                sample_submission.csv (993 lines)
                ... and 7 other files
                petfinder-pawpularity-score/
                test/
                    a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                    2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                    ... and 990 other files
                    test/
                train/
                    e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                    cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                    ... and 8918 other files
                    train/
            test/
                a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                ... and 990 other files
                test/
                    a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                    2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                    ... and 990 other files
                    test/
            train/
                e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                ... and 8918 other files
                train/
                    e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                    cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                    ... and 8918 other files
                    train/
        working/
            petfinder-pawpularity-score/
                description.md (132 lines)
                sample_submission.csv (993 lines)
                ... and 7 other files
                petfinder-pawpularity-score/
                test/
                    a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                    2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                    ... and 990 other files
                    test/
                train/
                    e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                    cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                    ... and 8918 other files
                    train/
```

-> data/petfinder-pawpularity-score/sample_submission.csv has 992 rows and 2 columns.
The columns are: Id, Pawpularity

-> data/petfinder-pawpularity-score/test.csv has 992 rows and 13 columns.
The columns are: Id, Subject Focus, Eyes, Face, Near, Action, Accessory, Group, Collage, Human, Occlusion, Info, Blur

-> data/petfinder-pawpularity-score/train.csv has 8920 rows and 14 columns.
The columns are: Id, Subject Focus, Eyes, Face, Near, Action, Accessory, Group, Collage, Human, Occlusion, Info, Blur, Pawpularity

-> data/sample_submission.csv has 992 rows and 2 columns.
The columns are: Id, Pawpularity

-> data/test.csv has 992 rows and 13 columns.
The columns are: Id, Subject Focus, Eyes, Face, Near, Action, Accessory, Group, Collage, Human, Occlusion, Info, Blur

-> data/train.csv has 8920 rows and 14 columns.
The columns are: Id, Subject Focus, Eyes, Face, Near, Action, Accessory, Group, Collage, Human, Occlusion, Info, Blur, Pawpularity

-> (stopped after 10 files for performance)

# 5. Target score

20.62399

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import sys
import math
import numpy as np
import pandas as pd

import subprocess

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "protobuf==4.25.3"],
    check=False,
)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras import layers
from tensorflow.keras.models import Model

from sklearn.model_selection import StratifiedKFold

SEED = 2021
tf.keras.utils.set_random_seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

print("TF:", tf.__version__)
print("Eager:", tf.executing_eagerly())



## === cell 1
DATA_DIR = "../input/petfinder-pawpularity-score"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")
SAMPLE_SUB = os.path.join(DATA_DIR, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test")

train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)
sample = pd.read_csv(SAMPLE_SUB)

print(train_df.shape, test_df.shape, sample.shape)
print(train_df.columns.tolist())



## === cell 2
train_df = train_df.copy()
test_df = test_df.copy()

train_df["filename"] = train_df["Id"].astype(str) + ".jpg"
test_df["filename"] = test_df["Id"].astype(str) + ".jpg"

train_df["path"] = train_df["filename"].apply(lambda x: os.path.join(TRAIN_IMG_DIR, x))
test_df["path"] = test_df["filename"].apply(lambda x: os.path.join(TEST_IMG_DIR, x))

missing_train = (~train_df["path"].apply(os.path.exists)).sum()
missing_test = (~test_df["path"].apply(os.path.exists)).sum()
print("Missing train images:", missing_train, "Missing test images:", missing_test)



## === cell 3
n_splits = 10
dirs_df = pd.DataFrame({"dirs": train_df["filename"]})
train_labels = train_df["Pawpularity"].astype(np.float32)

n_bins = 20
binned = pd.qcut(train_labels, q=n_bins, labels=False, duplicates="drop")

skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
dirs_df["fold"] = -1
train_df["fold"] = -1

for fold, (_, val_idx) in enumerate(skf.split(dirs_df, binned)):
    dirs_df.loc[val_idx, "fold"] = fold
    train_df.loc[val_idx, "fold"] = fold

print("Fold counts:\n", train_df["fold"].value_counts().sort_index())



## === cell 4
train_mask = ~train_df["fold"].isin([5, 6, 9])
valid_mask = train_df["fold"].isin([5, 6])
holdout_mask = train_df["fold"].isin([9])

feature_cols = [
    c
    for c in train_df.columns
    if c not in ["Id", "filename", "path", "fold", "Pawpularity"]
]
assert (
    len(feature_cols) == 12
), f"Expected 12 tabular features, got {len(feature_cols)}: {feature_cols}"

train_paths = train_df.loc[train_mask, "path"].tolist()
valid_paths = train_df.loc[valid_mask, "path"].tolist()
holdout_paths = train_df.loc[holdout_mask, "path"].tolist()

X_train = train_df.loc[train_mask, feature_cols].astype(np.float32).to_numpy()
y_train = train_df.loc[train_mask, "Pawpularity"].astype(np.float32).to_numpy()

X_valid = train_df.loc[valid_mask, feature_cols].astype(np.float32).to_numpy()
y_valid = train_df.loc[valid_mask, "Pawpularity"].astype(np.float32).to_numpy()

X_holdout = train_df.loc[holdout_mask, feature_cols].astype(np.float32).to_numpy()
y_holdout = train_df.loc[holdout_mask, "Pawpularity"].astype(np.float32).to_numpy()

print("Train:", X_train.shape, y_train.shape, len(train_paths))
print("Valid:", X_valid.shape, y_valid.shape, len(valid_paths))
print("Holdout:", X_holdout.shape, y_holdout.shape, len(holdout_paths))



## === cell 5
BATCH_SIZE = 32
EPOCHS = 10  # preserve final training loop usage below
AUTOTUNE = tf.data.AUTOTUNE


def process_image(file_path):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [224, 224], method="bilinear")
    img = tf.cast(img, tf.float32) / 255.0
    return img


def make_dataset(paths, table, labels=None, shuffle=False, batch_size=BATCH_SIZE):
    path_ds = tf.data.Dataset.from_tensor_slices(paths)
    img_ds = path_ds.map(process_image, num_parallel_calls=AUTOTUNE)
    table_ds = tf.data.Dataset.from_tensor_slices(table)
    if labels is not None:
        y_ds = tf.data.Dataset.from_tensor_slices(labels)
        ds = tf.data.Dataset.zip(({"img": img_ds, "table": table_ds}, y_ds))
    else:
        ds = tf.data.Dataset.zip({"img": img_ds, "table": table_ds})
    if shuffle:
        ds = ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds


train_tensor = make_dataset(
    train_paths, X_train, y_train, shuffle=True, batch_size=BATCH_SIZE
)
valid_tensor = make_dataset(
    valid_paths, X_valid, y_valid, shuffle=False, batch_size=BATCH_SIZE
)
holdout_tensor = make_dataset(
    holdout_paths, X_holdout, y_holdout, shuffle=False, batch_size=BATCH_SIZE
)

for xb, yb in train_tensor.take(1):
    print("img:", xb["img"].shape, "table:", xb["table"].shape, "y:", yb.shape)



## === cell 6
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.TPUStrategy(tpu)
    BATCH_SIZE = strategy.num_replicas_in_sync * 64
    print("Running on TPU:", tpu.master())
    print(f"Batch Size: {BATCH_SIZE}")
except Exception as e:
    strategy = tf.distribute.get_strategy()
    print(f"Running on {strategy.num_replicas_in_sync} replicas")
    print(f"Batch Size: {BATCH_SIZE}")



## === cell 7
input_shape = (224, 224, 3)
patch_size = (4, 4)
dropout_rate = 0.03
num_heads = 8
embed_dim = 64
num_mlp = 256
qkv_bias = True
window_size = 2
shift_size = 1
image_dimension = 224

num_patch_x = input_shape[0] // patch_size[0]
num_patch_y = input_shape[1] // patch_size[1]

learning_rate = 3e-4
weight_decay = 1e-5




## === cell 8
def window_partition(x, window_size):
    shape = tf.shape(x)
    B, H, W, C = shape[0], shape[1], shape[2], shape[3]
    patch_num_y = H // window_size
    patch_num_x = W // window_size
    x = tf.reshape(x, shape=(B, patch_num_y, window_size, patch_num_x, window_size, C))
    x = tf.transpose(x, (0, 1, 3, 2, 4, 5))
    windows = tf.reshape(x, shape=(-1, window_size, window_size, C))
    return windows


def window_reverse(windows, window_size, height, width, channels):
    patch_num_y = height // window_size
    patch_num_x = width // window_size
    x = tf.reshape(
        windows,
        shape=(-1, patch_num_y, patch_num_x, window_size, window_size, channels),
    )
    x = tf.transpose(x, perm=(0, 1, 3, 2, 4, 5))
    x = tf.reshape(x, shape=(-1, height, width, channels))
    return x


class DropPath(layers.Layer):
    def __init__(self, drop_prob=None, **kwargs):
        super(DropPath, self).__init__(**kwargs)
        self.drop_prob = drop_prob if drop_prob is not None else 0.0

    def call(self, x, training=None):
        if (training is False) or self.drop_prob == 0.0:
            return x
        input_shape = tf.shape(x)
        batch_size = input_shape[0]
        rank = x.shape.rank
        shape = (batch_size,) + (1,) * (rank - 1)
        random_tensor = (1 - self.drop_prob) + tf.random.uniform(shape, dtype=x.dtype)
        path_mask = tf.floor(random_tensor)
        output = tf.math.divide(x, 1 - self.drop_prob) * path_mask
        return output




## === cell 9
class WindowAttention(layers.Layer):
    def __init__(
        self, dim, window_size, num_heads, qkv_bias=True, dropout_rate=0.0, **kwargs
    ):
        super(WindowAttention, self).__init__(**kwargs)
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = layers.Dense(dim * 3, use_bias=qkv_bias)
        self.dropout = layers.Dropout(dropout_rate)
        self.proj = layers.Dense(dim)

        self.relative_position_bias_table = None
        self.relative_position_index = None

    def build(self, input_shape):
        num_window_elements = (2 * self.window_size[0] - 1) * (
            2 * self.window_size[1] - 1
        )

        self.relative_position_bias_table = self.add_weight(
            name="relative_position_bias_table",
            shape=(num_window_elements, self.num_heads),
            initializer=tf.initializers.Zeros(),
            trainable=True,
        )

        coords_h = np.arange(self.window_size[0])
        coords_w = np.arange(self.window_size[1])
        coords_matrix = np.meshgrid(coords_h, coords_w, indexing="ij")
        coords = np.stack(coords_matrix)
        coords_flatten = coords.reshape(2, -1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.transpose([1, 2, 0])
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1).astype(np.int32)

        self.relative_position_index = self.add_weight(
            name="relative_position_index",
            shape=relative_position_index.shape,
            initializer=tf.constant_initializer(relative_position_index),
            trainable=False,
            dtype=tf.int32,
        )
        super().build(input_shape)

    def call(self, x, mask=None, training=None):
        x_shape = tf.shape(x)
        size = x_shape[1]
        channels = x.shape[-1]
        head_dim = channels // self.num_heads

        x_qkv = self.qkv(x)
        x_qkv = tf.reshape(x_qkv, shape=(-1, size, 3, self.num_heads, head_dim))
        x_qkv = tf.transpose(x_qkv, perm=(2, 0, 3, 1, 4))
        q, k, v = x_qkv[0], x_qkv[1], x_qkv[2]
        q = q * self.scale
        k = tf.transpose(k, perm=(0, 1, 3, 2))
        attn = q @ k

        num_window_elements = self.window_size[0] * self.window_size[1]
        relative_position_index_flat = tf.reshape(self.relative_position_index, (-1,))
        relative_position_bias = tf.gather(
            self.relative_position_bias_table, relative_position_index_flat
        )
        relative_position_bias = tf.reshape(
            relative_position_bias, shape=(num_window_elements, num_window_elements, -1)
        )
        relative_position_bias = tf.transpose(relative_position_bias, perm=(2, 0, 1))
        attn = attn + tf.expand_dims(relative_position_bias, axis=0)

        if mask is not None:
            nW = tf.shape(mask)[0]
            mask_float = tf.cast(
                tf.expand_dims(tf.expand_dims(mask, axis=1), axis=0), tf.float32
            )
            attn = (
                tf.reshape(attn, shape=(-1, nW, self.num_heads, size, size))
                + mask_float
            )
            attn = tf.reshape(attn, shape=(-1, self.num_heads, size, size))

        attn = keras.activations.softmax(attn, axis=-1)
        attn = self.dropout(attn, training=training)

        x_out = attn @ v
        x_out = tf.transpose(x_out, perm=(0, 2, 1, 3))
        x_out = tf.reshape(x_out, shape=(-1, size, channels))
        x_out = self.proj(x_out)
        x_out = self.dropout(x_out, training=training)
        return x_out




## === cell 10
class SwinTransformer(layers.Layer):
    def __init__(
        self,
        dim,
        num_patch,
        num_heads,
        window_size=7,
        shift_size=0,
        num_mlp=1024,
        qkv_bias=True,
        dropout_rate=0.0,
        **kwargs,
    ):
        super(SwinTransformer, self).__init__(**kwargs)

        self.dim = dim
        self.num_patch = num_patch
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.num_mlp = num_mlp

        self.norm1 = layers.LayerNormalization(epsilon=1e-5)
        self.attn = WindowAttention(
            dim,
            window_size=(self.window_size, self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            dropout_rate=dropout_rate,
        )
        self.drop_path = DropPath(dropout_rate)
        self.norm2 = layers.LayerNormalization(epsilon=1e-5)

        self.mlp = keras.Sequential(
            [
                layers.Dense(num_mlp),
                layers.Activation(keras.activations.gelu),
                layers.Dropout(dropout_rate),
                layers.Dense(dim),
                layers.Dropout(dropout_rate),
            ]
        )

        if min(self.num_patch) < self.window_size:
            self.shift_size = 0
            self.window_size = min(self.num_patch)

        self.attn_mask = None

    def build(self, input_shape):
        if self.shift_size == 0:
            self.attn_mask = None
        else:
            height, width = self.num_patch
            h_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            w_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            mask_array = np.zeros((1, height, width, 1), dtype=np.int32)
            count = 0
            for h in h_slices:
                for w in w_slices:
                    mask_array[:, h, w, :] = count
                    count += 1
            mask_tensor = tf.convert_to_tensor(mask_array)

            mask_windows = window_partition(mask_tensor, self.window_size)
            mask_windows = tf.reshape(
                mask_windows, shape=[-1, self.window_size * self.window_size]
            )
            attn_mask = tf.expand_dims(mask_windows, axis=1) - tf.expand_dims(
                mask_windows, axis=2
            )
            attn_mask = tf.where(attn_mask != 0, -100.0, attn_mask)
            attn_mask = tf.where(attn_mask == 0, 0.0, attn_mask)

            self.attn_mask = self.add_weight(
                name="attn_mask",
                shape=attn_mask.shape,
                initializer=tf.constant_initializer(attn_mask.numpy()),
                trainable=False,
                dtype=tf.float32,
            )

        super().build(input_shape)

    def call(self, x, training=None):
        height, width = self.num_patch
        channels = x.shape[-1]

        x_skip = x
        x = self.norm1(x)
        x = tf.reshape(x, shape=(-1, height, width, channels))

        if self.shift_size > 0:
            shifted_x = tf.roll(
                x, shift=[-self.shift_size, -self.shift_size], axis=[1, 2]
            )
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = tf.reshape(
            x_windows, shape=(-1, self.window_size * self.window_size, channels)
        )
        attn_windows = self.attn(x_windows, mask=self.attn_mask, training=training)

        attn_windows = tf.reshape(
            attn_windows, shape=(-1, self.window_size, self.window_size, channels)
        )
        shifted_x = window_reverse(
            attn_windows, self.window_size, height, width, channels
        )

        if self.shift_size > 0:
            x = tf.roll(
                shifted_x, shift=[self.shift_size, self.shift_size], axis=[1, 2]
            )
        else:
            x = shifted_x

        x = tf.reshape(x, shape=(-1, height * width, channels))
        x = self.drop_path(x, training=training)
        x = x_skip + x

        x_skip = x
        x = self.norm2(x)
        x = self.mlp(x, training=training)
        x = self.drop_path(x, training=training)
        x = x_skip + x
        return x




## === cell 11
class PatchExtract(layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super(PatchExtract, self).__init__(**kwargs)
        self.patch_size_x = patch_size[0]
        self.patch_size_y = patch_size[1]

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=(1, self.patch_size_x, self.patch_size_y, 1),
            strides=(1, self.patch_size_x, self.patch_size_y, 1),
            rates=(1, 1, 1, 1),
            padding="VALID",
        )
        patch_dim = patches.shape[-1]
        patch_num = patches.shape[1]
        return tf.reshape(patches, (batch_size, patch_num * patch_num, patch_dim))


class PatchEmbedding(layers.Layer):
    def __init__(self, num_patch, embed_dim, **kwargs):
        super(PatchEmbedding, self).__init__(**kwargs)
        self.num_patch = num_patch
        self.proj = layers.Dense(embed_dim)
        self.pos_embed = layers.Embedding(input_dim=num_patch, output_dim=embed_dim)

    def call(self, patch):
        pos = tf.range(start=0, limit=self.num_patch, delta=1)
        return self.proj(patch) + self.pos_embed(pos)


class PatchMerging(tf.keras.layers.Layer):
    def __init__(self, num_patch, embed_dim):
        super(PatchMerging, self).__init__()
        self.num_patch = num_patch
        self.embed_dim = embed_dim
        self.linear_trans = layers.Dense(2 * embed_dim, use_bias=False)

    def call(self, x):
        height, width = self.num_patch
        _, _, C = x.get_shape().as_list()
        x = tf.reshape(x, shape=(-1, height, width, C))
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = tf.concat((x0, x1, x2, x3), axis=-1)
        x = tf.reshape(x, shape=(-1, (height // 2) * (width // 2), 4 * C))
        return self.linear_trans(x)




## === cell 12
def swin_transformer_with_mlp_model():
    input_1 = layers.Input(input_shape, name="img")
    x = PatchExtract(patch_size)(input_1)
    x = PatchEmbedding(num_patch_x * num_patch_y, embed_dim)(x)
    x = SwinTransformer(
        dim=embed_dim,
        num_patch=(num_patch_x, num_patch_y),
        num_heads=num_heads,
        window_size=window_size,
        shift_size=0,
        num_mlp=num_mlp,
        qkv_bias=qkv_bias,
        dropout_rate=dropout_rate,
    )(x)
    x = SwinTransformer(
        dim=embed_dim,
        num_patch=(num_patch_x, num_patch_y),
        num_heads=num_heads,
        window_size=window_size,
        shift_size=shift_size,
        num_mlp=num_mlp,
        qkv_bias=qkv_bias,
        dropout_rate=dropout_rate,
    )(x)
    x = PatchMerging((num_patch_x, num_patch_y), embed_dim=embed_dim)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x_1 = tf.keras.layers.LeakyReLU()(x)

    input_2 = Input(shape=(12,), name="table")
    x = Dense(256)(input_2)
    x = tf.keras.layers.LeakyReLU()(x)
    x = Dense(128)(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = Dense(128)(x)
    x_2 = tf.keras.layers.LeakyReLU()(x)

    concat = Concatenate()([x_1, x_2])

    x = Dense(64)(concat)
    x = tf.keras.layers.LeakyReLU()(x)
    output = layers.Dense(1, activation="relu")(x)

    model = Model(inputs=[input_1, input_2], outputs=output)
    return model




## === cell 13
optimizer = tf.keras.optimizers.AdamW(
    learning_rate=learning_rate, weight_decay=weight_decay
)
loss_fn = tf.keras.losses.MeanSquaredError()
acc_metric = tf.keras.metrics.RootMeanSquaredError()
acc_metric_valid = tf.keras.metrics.RootMeanSquaredError()

with strategy.scope():
    model = swin_transformer_with_mlp_model()

print(model.summary())



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1111773898.py in <cell line: 0>()
      7 
      8 with strategy.scope():
----> 9     model = swin_transformer_with_mlp_model()
     10 
     11 print(model.summary())

/tmp/ipykernel_11/2959005523.py in swin_transformer_with_mlp_model()
     13         dropout_rate=dropout_rate,
     14     )(x)
---> 15     x = SwinTransformer(
     16         dim=embed_dim,
     17         num_patch=(num_patch_x, num_patch_y),

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/tmp/ipykernel_11/1877678093.py in build(self, input_shape)
     79                 mask_windows, axis=2
     80             )
---> 81             attn_mask = tf.where(attn_mask != 0, -100.0, attn_mask)
     82             attn_mask = tf.where(attn_mask == 0, 0.0, attn_mask)
     83 

TypeError: Cannot convert -100.0 to EagerTensor of dtype int32

## === cell 14
num_epochs = EPOCHS
for epoch in range(num_epochs):
    print(f"\nStart of Training Epoch {epoch+1}/{num_epochs}")
    for batch_idx, ((x_batch, y_batch), (x_batch_valid, y_batch_valid)) in enumerate(
        zip(train_tensor, valid_tensor)
    ):
        img = x_batch["img"]
        table = x_batch["table"]

        img_valid = x_batch_valid["img"]
        table_valid = x_batch_valid["table"]

        with tf.GradientTape() as tape:
            y_pred = model((img, table), training=True)
            loss = loss_fn(y_batch, y_pred)

        gradients = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))
        acc_metric.update_state(y_batch, y_pred)

        y_pred_valid = model((img_valid, table_valid), training=False)
        acc_metric_valid.update_state(y_batch_valid, y_pred_valid)

    train_acc = acc_metric.result()
    valid_acc = acc_metric_valid.result()
    print(f"Train RMSE: {float(train_acc.numpy()):.5f}")
    print(f"Valid RMSE: {float(valid_acc.numpy()):.5f}")

    acc_metric.reset_state()
    acc_metric_valid.reset_state()



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/262656228.py in <cell line: 0>()
     12 
     13         with tf.GradientTape() as tape:
---> 14             y_pred = model((img, table), training=True)
     15             loss = loss_fn(y_batch, y_pred)
     16 

NameError: name 'model' is not defined

## === cell 15
test_paths = test_df["path"].tolist()
X_test = test_df[feature_cols].astype(np.float32).to_numpy()

test_ds_tensor = make_dataset(
    test_paths, X_test, labels=None, shuffle=False, batch_size=BATCH_SIZE
)

preds = []
for xb in test_ds_tensor:
    img = xb["img"]
    table = xb["table"]
    y_pred = model((img, table), training=False)
    preds.append(y_pred.numpy().reshape(-1))

y_test_preds_array = np.concatenate(preds, axis=0)
print("Pred shape:", y_test_preds_array.shape)



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/477199146.py in <cell line: 0>()
     10     img = xb["img"]
     11     table = xb["table"]
---> 12     y_pred = model((img, table), training=False)
     13     preds.append(y_pred.numpy().reshape(-1))
     14 

NameError: name 'model' is not defined

## === cell 16
y_test_preds_array = np.clip(y_test_preds_array, 1.0, 100.0)

sub = pd.DataFrame(
    {"Id": test_df["Id"].values, "Pawpularity": y_test_preds_array.astype(np.float32)}
)
sub.to_csv("submission.csv", index=False)

print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
print("submission.csv exists:", os.path.exists("submission.csv"))
print(
    "submission.csv size (bytes):",
    os.path.getsize("submission.csv") if os.path.exists("submission.csv") else None,
)

## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1721581109.py in <cell line: 0>()
----> 1 y_test_preds_array = np.clip(y_test_preds_array, 1.0, 100.0)
      2 
      3 sub = pd.DataFrame(
      4     {"Id": test_df["Id"].values, "Pawpularity": y_test_preds_array.astype(np.float32)}
      5 )

NameError: name 'y_test_preds_array' is not defined
