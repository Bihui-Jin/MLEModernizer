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

20.7676

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from tensorflow.keras.optimizers import AdamW

np.random.seed(997)
tf.random.set_seed(997)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
class Config:
    image_size = 128
    input_shape = [image_size, image_size, 3]
    patch_size = [8, 8]
    num_patch_x = input_shape[0] // patch_size[0]
    num_patch_y = input_shape[1] // patch_size[1]

    learning_rate = 1e-3
    num_mlp = 256
    dropout_rate = 0.03
    weight_decay = 0.0001
    batch_size = 128
    num_classes = 1
    num_epochs = 30
    num_heads = 8
    label_smoothing = 0.1
    embed_dim = 64
    window_size = 2

    tabular_columns = [
        "Subject Focus",
        "Eyes",
        "Face",
        "Near",
        "Action",
        "Accessory",
        "Group",
        "Collage",
        "Human",
        "Occlusion",
        "Info",
        "Blur",
    ]


config = Config()




## === cell 2
def display_images(images, row_count, column_count):
    fig, axs = plt.subplots(row_count, column_count, figsize=(10, 10))
    for i in range(row_count):
        for j in range(column_count):
            axs[i, j].imshow(images[i * column_count + j])
            axs[i, j].axis("off")
    plt.show()




## === cell 3
def random_erasing(img, sl=0.1, sh=0.2, rl=0.4, p=0.3):
    h = tf.shape(img)[0]
    w = tf.shape(img)[1]
    c = tf.shape(img)[2]
    origin_area = tf.cast(h * w, tf.float32)

    e_size_l = tf.cast(tf.round(tf.sqrt(origin_area * sl * rl)), tf.int32)
    e_size_h = tf.cast(tf.round(tf.sqrt(origin_area * sh / rl)), tf.int32)

    e_height_h = tf.minimum(e_size_h, h)
    e_width_h = tf.minimum(e_size_h, w)

    erase_height = tf.random.uniform(
        shape=[], minval=e_size_l, maxval=e_height_h, dtype=tf.int32
    )
    erase_width = tf.random.uniform(
        shape=[], minval=e_size_l, maxval=e_width_h, dtype=tf.int32
    )

    erase_area = tf.zeros(shape=[erase_height, erase_width, c])
    erase_area = tf.cast(erase_area, tf.uint8)

    pad_h = h - erase_height
    pad_top = tf.random.uniform(shape=[], minval=0, maxval=pad_h, dtype=tf.int32)
    pad_bottom = pad_h - pad_top

    pad_w = w - erase_width
    pad_left = tf.random.uniform(shape=[], minval=0, maxval=pad_w, dtype=tf.int32)
    pad_right = pad_w - pad_left

    erase_mask = tf.pad(
        [erase_area],
        [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]],
        constant_values=1,
    )
    erase_mask = tf.squeeze(erase_mask, axis=0)
    erased_img = tf.multiply(tf.cast(img, tf.float32), tf.cast(erase_mask, tf.float32))

    return tf.cond(
        tf.random.uniform([], 0, 1) > p,
        lambda: tf.cast(img, img.dtype),
        lambda: tf.cast(erased_img, img.dtype),
    )


def data_augment(image):
    image = tf.image.random_flip_left_right(image)
    image = random_erasing(image)
    return image




## === cell 4
def preprocess_image(image_url, augment):
    image_string = tf.io.read_file(image_url)
    image = tf.image.decode_jpeg(image_string, channels=3)

    if augment == True:
        image = data_augment(image)
    image = tf.image.resize(image, (Config.image_size, Config.image_size))
    image = tf.cast(image, tf.float32) / 255.0
    return image


def preprocess_train(image_url, tabular):
    tabular = tf.cast(tabular, tf.float32)
    return (preprocess_image(image_url, True), tabular[1:]), tabular[0]


def preprocess_valid(image_url, tabular):
    tabular = tf.cast(tabular, tf.float32)
    return (preprocess_image(image_url, False), tabular[1:]), tabular[0]


def preprocess_test(image_url, tabular):
    tabular = tf.cast(tabular, tf.float32)
    return (preprocess_image(image_url, False), tabular), 0




## === cell 5
def rmse(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))




## === cell 6
train = pd.read_csv("../input/petfinder-pawpularity-score/train.csv")
test = pd.read_csv("../input/petfinder-pawpularity-score/test.csv")



## === cell 7
train["file_path"] = train["Id"].apply(
    lambda identifier: "../input/petfinder-pawpularity-score/train/"
    + identifier
    + ".jpg"
)
test["file_path"] = test["Id"].apply(
    lambda identifier: "../input/petfinder-pawpularity-score/test/"
    + identifier
    + ".jpg"
)



## === cell 8
train["Pawpularity"].hist()



## === cell 9
item_width = 5
data = train[train.Pawpularity >= 90]
for item in (
    tf.data.Dataset.from_tensor_slices(
        (data["file_path"], data[["Pawpularity"] + Config.tabular_columns])
    )
    .map(preprocess_valid)
    .batch(item_width**2)
    .take(1)
):
    display_images(item[0][0].numpy(), item_width, item_width)



## === cell 10
data = train[train.Pawpularity <= 10]
for item in (
    tf.data.Dataset.from_tensor_slices(
        (data["file_path"], data[["Pawpularity"] + Config.tabular_columns])
    )
    .map(preprocess_valid)
    .batch(item_width**2)
    .take(1)
):
    display_images(item[0][0].numpy(), item_width, item_width)



## === cell 11
data = train[(train.Pawpularity >= 40) & (train.Pawpularity <= 60)]
for item in (
    tf.data.Dataset.from_tensor_slices(
        (data["file_path"], data[["Pawpularity"] + Config.tabular_columns])
    )
    .map(preprocess_valid)
    .batch(item_width**2)
    .take(1)
):
    display_images(item[0][0].numpy(), item_width, item_width)




## === cell 12
def window_partition(x, window_size):
    _, height, width, channels = x.shape
    patch_num_y = height // window_size
    patch_num_x = width // window_size
    x = tf.reshape(
        x, shape=(-1, patch_num_y, window_size, patch_num_x, window_size, channels)
    )
    x = tf.transpose(x, (0, 1, 3, 2, 4, 5))
    windows = tf.reshape(x, shape=(-1, window_size, window_size, channels))
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
        self.drop_prob = drop_prob

    def call(self, x):
        input_shape = tf.shape(x)
        batch_size = input_shape[0]
        rank = x.shape.rank
        shape = (batch_size,) + (1,) * (rank - 1)
        random_tensor = (1 - self.drop_prob) + tf.random.uniform(shape, dtype=x.dtype)
        path_mask = tf.floor(random_tensor)
        output = tf.math.divide(x, 1 - self.drop_prob) * path_mask
        return output




## === cell 13
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

    def build(self, input_shape):
        num_window_elements = (2 * self.window_size[0] - 1) * (
            2 * self.window_size[1] - 1
        )
        self.relative_position_bias_table = self.add_weight(
            shape=(num_window_elements, self.num_heads),
            initializer=tf.initializers.Zeros(),
            trainable=True,
            name="relative_position_bias_table",
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

        self.relative_position_index = tf.constant(
            relative_position_index, dtype=tf.int32, name="relative_position_index"
        )

    def call(self, x, mask=None):
        _, size, channels = x.shape
        head_dim = channels // self.num_heads
        x_qkv = self.qkv(x)
        x_qkv = tf.reshape(x_qkv, shape=(-1, size, 3, self.num_heads, head_dim))
        x_qkv = tf.transpose(x_qkv, perm=(2, 0, 3, 1, 4))
        q, k, v = x_qkv[0], x_qkv[1], x_qkv[2]
        q = q * self.scale
        k = tf.transpose(k, perm=(0, 1, 3, 2))
        attn = q @ k

        num_window_elements = self.window_size[0] * self.window_size[1]
        relative_position_index_flat = tf.reshape(
            self.relative_position_index, shape=(-1,)
        )
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
        else:
            attn = keras.activations.softmax(attn, axis=-1)

        attn = self.dropout(attn)

        x_qkv = attn @ v
        x_qkv = tf.transpose(x_qkv, perm=(0, 2, 1, 3))
        x_qkv = tf.reshape(x_qkv, shape=(-1, size, channels))
        x_qkv = self.proj(x_qkv)
        x_qkv = self.dropout(x_qkv)
        return x_qkv




## === cell 14
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
            mask_array = np.zeros((1, height, width, 1), dtype=np.float32)
            count = 0
            for h in h_slices:
                for w in w_slices:
                    mask_array[:, h, w, :] = count
                    count += 1

            mask_windows = window_partition(
                tf.convert_to_tensor(mask_array), self.window_size
            )
            mask_windows = tf.reshape(
                mask_windows, shape=[-1, self.window_size * self.window_size]
            )
            attn_mask = tf.expand_dims(mask_windows, axis=1) - tf.expand_dims(
                mask_windows, axis=2
            )
            attn_mask = tf.where(attn_mask != 0, -100.0, attn_mask)
            attn_mask = tf.where(attn_mask == 0, 0.0, attn_mask)

            self.attn_mask = tf.constant(
                attn_mask.numpy(), dtype=tf.float32, name="attn_mask"
            )

    def call(self, x):
        height, width = self.num_patch
        _, _, channels = x.shape
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
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

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
        x = self.drop_path(x)
        x = x_skip + x
        x_skip = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path(x)
        x = x_skip + x
        return x




## === cell 15
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




## === cell 16
def get_swin_transformer(config, inputs):
    x = PatchExtract(config.patch_size)(inputs)
    x = PatchEmbedding(config.num_patch_x * config.num_patch_y, config.embed_dim)(x)
    for shift_size in range(2):
        x = SwinTransformer(
            dim=config.embed_dim,
            num_patch=(config.num_patch_x, config.num_patch_y),
            num_heads=config.num_heads,
            window_size=config.window_size,
            shift_size=shift_size,
            num_mlp=config.num_mlp,
            qkv_bias=True,
            dropout_rate=config.dropout_rate,
        )(x)
    x = PatchMerging(
        (config.num_patch_x, config.num_patch_y), embed_dim=config.embed_dim
    )(x)
    x = layers.GlobalAveragePooling1D()(x)
    return x




## === cell 17
def get_tabular_model(inputs):
    width = 8
    depth = 9
    activation = "relu"
    for i in range(depth):
        if i == 0:
            x = inputs
        x = keras.layers.Dense(width, activation=activation)(x)
        if (i + 1) % 3 == 0:
            x = keras.layers.Concatenate()([x, inputs])
    return x




## === cell 18
def get_model(config, use_tabular_input=True, is_training=True):
    image_inputs = keras.Input(shape=config.input_shape)
    tabular_inputs = keras.Input(shape=(len(config.tabular_columns),))
    image_x = get_swin_transformer(config, image_inputs)
    inputs = [image_inputs, tabular_inputs]
    if use_tabular_input:
        tabular_x = get_tabular_model(tabular_inputs)
        x = layers.Concatenate()([image_x, tabular_x])
    else:
        x = image_x
    output = layers.Dense(1)(x)
    model = keras.Model(inputs, output)
    if is_training:
        model.compile(
            loss=rmse,
            optimizer=AdamW(
                learning_rate=config.learning_rate, weight_decay=config.weight_decay
            ),
            metrics=["mae"],
        )
    return model




## === cell 19
model = get_model(config, is_training=False)
model.summary()



## === cell 20
image = np.random.normal(size=(1, Config.image_size, Config.image_size, 3)).astype(
    np.float32
)
tabular = np.random.normal(size=(1, len(Config.tabular_columns))).astype(np.float32)
print(image.shape, tabular.shape)
print(model((image, tabular)).shape)



## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1459952102.py in <cell line: 0>()
      4 tabular = np.random.normal(size=(1, len(Config.tabular_columns))).astype(np.float32)
      5 print(image.shape, tabular.shape)
----> 6 print(model((image, tabular)).shape)
      7 

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/tmp/ipykernel_11/3271626507.py in call(self, x)
    102             x_windows, shape=(-1, self.window_size * self.window_size, channels)
    103         )
--> 104         attn_windows = self.attn(x_windows, mask=self.attn_mask)
    105 
    106         attn_windows = tf.reshape(

/tmp/ipykernel_11/4204639828.py in call(self, x, mask)
     53 
     54         num_window_elements = self.window_size[0] * self.window_size[1]
---> 55         relative_position_index_flat = tf.reshape(
     56             self.relative_position_index, shape=(-1,)
     57         )

TypeError: Exception encountered when calling WindowAttention.call().

<tf.Tensor 'window_attention/relative_position_index:0' shape=(4, 4) dtype=int32> is out of scope and cannot be used here. Use return values, explicit Python locals or TensorFlow collections to access it.
Please see https://www.tensorflow.org/guide/function#all_outputs_of_a_tffunction_must_be_return_values for more information.

<tf.Tensor 'window_attention/relative_position_index:0' shape=(4, 4) dtype=int32> was defined here:
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
    File "/tmp/ipykernel_11/3688633874.py", line 1, in <cell line: 0>
    File "/tmp/ipykernel_11/1502862427.py", line 4, in get_model
    File "/tmp/ipykernel_11/1903798741.py", line 5, in get_swin_transformer
    File "/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py", line 117, in error_handler
    File "/usr/local/lib/python3.11/dist-packages/keras/src/layers/layer.py", line 908, in __call__
    File "/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py", line 117, in error_handler
    File "/usr/local/lib/python3.11/dist-packages/keras/src/ops/operation.py", line 46, in __call__
    File "/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py", line 156, in error_handler
    File "/usr/local/lib/python3.11/dist-packages/keras/src/ops/operation.py", line 58, in symbolic_call
    File "/usr/local/lib/python3.11/dist-packages/keras/src/layers/layer.py", line 1061, in compute_output_spec
    File "/usr/local/lib/python3.11/dist-packages/keras/src/ops/operation.py", line 78, in compute_output_spec
    File "/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/core.py", line 224, in compute_output_spec
    File "/tmp/ipykernel_11/3271626507.py", line 104, in call
    File "/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py", line 117, in error_handler
    File "/usr/local/lib/python3.11/dist-packages/keras/src/layers/layer.py", line 829, in __call__
    File "/usr/local/lib/python3.11/dist-packages/keras/src/layers/layer.py", line 1371, in _maybe_build
    File "/usr/local/lib/python3.11/dist-packages/keras/src/layers/layer.py", line 228, in build_wrapper
    File "/tmp/ipykernel_11/4204639828.py", line 39, in build
    File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/weak_tensor_ops.py", line 142, in wrapper
    File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/constant_op.py", line 276, in constant
    File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/constant_op.py", line 291, in _constant_impl
    File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py", line 289, in _create_graph_constant
    File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/func_graph.py", line 670, in _create_op_internal
    File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py", line 2701, in _create_op_internal
    File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py", line 1196, in from_node_def

The tensor <tf.Tensor 'window_attention/relative_position_index:0' shape=(4, 4) dtype=int32> cannot be accessed from here, because it was defined in FuncGraph(name=scratch_graph_3, id=139883084621504), which is out of scope.

Arguments received by WindowAttention.call():
  • x=tf.Tensor(shape=(64, 4, 64), dtype=float32)
  • mask=None

## === cell 21
tf.keras.backend.clear_session()
models = []
historys = []
kfold = KFold(n_splits=5, shuffle=True, random_state=997)
train_on_fold = 4

for index, (train_indices, val_indices) in enumerate(kfold.split(train)):
    if train_on_fold is not None and index != train_on_fold:
        continue

    x_train = train.loc[train_indices, "file_path"]
    tabular_train = train.loc[train_indices, ["Pawpularity"] + Config.tabular_columns]
    x_val = train.loc[val_indices, "file_path"]
    tabular_val = train.loc[val_indices, ["Pawpularity"] + Config.tabular_columns]

    train_ds = (
        tf.data.Dataset.from_tensor_slices((x_train, tabular_train))
        .map(preprocess_train, num_parallel_calls=tf.data.AUTOTUNE)
        .shuffle(512)
        .batch(Config.batch_size)
        .cache()
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = (
        tf.data.Dataset.from_tensor_slices((x_val, tabular_val))
        .map(preprocess_valid, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(Config.batch_size)
        .cache()
        .prefetch(tf.data.AUTOTUNE)
    )

    model = get_model(config)
    history = model.fit(
        train_ds, epochs=config.num_epochs, validation_data=val_ds, callbacks=[]
    )

    for metrics in [("loss", "val_loss"), ("mae", "val_mae")]:
        pd.DataFrame(history.history, columns=metrics).plot()
        plt.show()

    historys.append(history)
    models.append(model)



## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3925006296.py in <cell line: 0>()
     31 
     32     model = get_model(config)
---> 33     history = model.fit(
     34         train_ds, epochs=config.num_epochs, validation_data=val_ds, callbacks=[]
     35     )

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/execute.py in quick_execute(op_name, num_outputs, inputs, attrs, ctx, name)
     63           "Inputs to eager execution function cannot be Keras symbolic "
     64           "tensors, but found {}".format(keras_symbolic_tensors))
---> 65     raise e
     66   # pylint: enable=protected-access
     67   return tensors

TypeError: <tf.Tensor 'window_attention/relative_position_index:0' shape=(4, 4) dtype=int32> is out of scope and cannot be used here. Use return values, explicit Python locals or TensorFlow collections to access it.
Please see https://www.tensorflow.org/guide/function#all_outputs_of_a_tffunction_must_be_return_values for more information.

<tf.Tensor 'window_attention/relative_position_index:0' shape=(4, 4) dtype=int32> was defined here:
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
    File "/tmp/ipykernel_11/3925006296.py", line 32, in <cell line: 0>
    File "/tmp/ipykernel_11/1502862427.py", line 4, in get_model
    File "/tmp/ipykernel_11/1903798741.py", line 5, in get_swin_transformer
    File "/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py", line 117, in error_handler
    File "/usr/local/lib/python3.11/dist-packages/keras/src/layers/layer.py", line 908, in __call__
    File "/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py", line 117, in error_handler
    File "/usr/local/lib/python3.11/dist-packages/keras/src/ops/operation.py", line 46, in __call__
    File "/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py", line 156, in error_handler
    File "/usr/local/lib/python3.11/dist-packages/keras/src/ops/operation.py", line 58, in symbolic_call
    File "/usr/local/lib/python3.11/dist-packages/keras/src/layers/layer.py", line 1061, in compute_output_spec
    File "/usr/local/lib/python3.11/dist-packages/keras/src/ops/operation.py", line 78, in compute_output_spec
    File "/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/core.py", line 224, in compute_output_spec
    File "/tmp/ipykernel_11/3271626507.py", line 104, in call
    File "/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py", line 117, in error_handler
    File "/usr/local/lib/python3.11/dist-packages/keras/src/layers/layer.py", line 829, in __call__
    File "/usr/local/lib/python3.11/dist-packages/keras/src/layers/layer.py", line 1371, in _maybe_build
    File "/usr/local/lib/python3.11/dist-packages/keras/src/layers/layer.py", line 228, in build_wrapper
    File "/tmp/ipykernel_11/4204639828.py", line 39, in build
    File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/weak_tensor_ops.py", line 142, in wrapper
    File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/constant_op.py", line 276, in constant
    File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/constant_op.py", line 291, in _constant_impl
    File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py", line 289, in _create_graph_constant
    File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/func_graph.py", line 670, in _create_op_internal
    File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py", line 2701, in _create_op_internal
    File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py", line 1196, in from_node_def

The tensor <tf.Tensor 'window_attention/relative_position_index:0' shape=(4, 4) dtype=int32> cannot be accessed from here, because it was defined in FuncGraph(name=scratch_graph_3, id=139881320724288), which is out of scope.

## === cell 22
sample_submission = pd.read_csv(
    "../input/petfinder-pawpularity-score/sample_submission.csv"
)
test_ds = (
    tf.data.Dataset.from_tensor_slices(
        (test["file_path"], test[Config.tabular_columns])
    )
    .map(preprocess_test, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(Config.batch_size)
    .cache()
    .prefetch(tf.data.AUTOTUNE)
)



## === cell 23
total_results = []
for model in models:
    preds = model.predict(test_ds, verbose=0).reshape(-1)
    total_results.append(preds)

results = np.mean(total_results, axis=0).astype(np.float32)

results = np.clip(results, 1.0, 100.0)

sample_submission["Pawpularity"] = results
sample_submission.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", sample_submission.shape)
print(sample_submission.head())
print(
    "Pawpularity min/max:",
    float(sample_submission["Pawpularity"].min()),
    float(sample_submission["Pawpularity"].max()),
)

## --- ERROR in outputing the csv:
Invalid submission: Pawpularity in submission should be between 1 and 100
