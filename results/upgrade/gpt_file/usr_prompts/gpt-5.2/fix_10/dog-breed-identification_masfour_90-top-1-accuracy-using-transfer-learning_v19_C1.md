# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Given a dataset of images of dogs, predict the breed of each image.

## Metric
Multi Class Log Loss.

## Submission Format
For each image in the test set, you must predict a probability for each of the different breeds. The file should contain a header and have the following format:
```
id,affenpinscher,afghan_hound,..,yorkshire_terrier
000621fb3cbb32d8935728e48679680e,0.0083,0.0,...,0.0083
etc.
```

## Dataset Description
- `train.zip` - the training set, you are provided the breed for these dogs
- `test.zip` - the test set, you must predict the probability of each breed for each image
- `sample_submission.csv` - a sample submission file in the correct format
- `labels.csv` - the breeds for the images in the train set

# 2. Python version

3.8

# 3. Installed packages

geopandas==0.14.4
google-api-python-client==2.177.0
ipython==7.34.0
ipython-genutils==0.2.0
ipython_pygments_lexers==1.1.1
ipython-sql==0.5.0
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
seaborn==0.12.2
sklearn-pandas==2.2.0
tf_keras==2.18.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (169 lines)
            labels.csv (9200 lines)
            labels.csv.zip (201.6 kB)
            sample_submission.csv (1024 lines)
            sample_submission.csv.zip (32.1 kB)
            test.zip (36.4 MB)
            train.zip (324.7 MB)
            dog-breed-identification/
                description.md (169 lines)
                labels.csv (9200 lines)
                ... and 5 other files
                dog-breed-identification/
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
            test/
                bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                ... and 1021 other files
                test/
            train/
                868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                ... and 9197 other files
                train/
        input/
            description.md (169 lines)
            labels.csv (9200 lines)
            labels.csv.zip (201.6 kB)
            sample_submission.csv (1024 lines)
            sample_submission.csv.zip (32.1 kB)
            test.zip (36.4 MB)
            train.zip (324.7 MB)
            dog-breed-identification/
                description.md (169 lines)
                labels.csv (9200 lines)
                ... and 5 other files
                dog-breed-identification/
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
            test/
                bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                ... and 1021 other files
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
            train/
                868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                ... and 9197 other files
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
        working/
            dog-breed-identification/
                description.md (169 lines)
                labels.csv (9200 lines)
                ... and 5 other files
                dog-breed-identification/
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
```

-> data/dog-breed-identification/labels.csv has 9199 rows and 2 columns.
The columns are: id, breed

-> data/dog-breed-identification/sample_submission.csv has 1023 rows and 121 columns.
The columns are: id, affenpinscher, afghan_hound, african_hunting_dog, airedale, american_staffordshire_terrier, appenzeller, australian_terrier, basenji, basset, beagle, bedlington_terrier, bernese_mountain_dog, black-and-tan_coonhound, blenheim_spaniel... and 106 more columns

-> data/labels.csv has 9199 rows and 2 columns.
The columns are: id, breed

-> data/sample_submission.csv has 1023 rows and 121 columns.
The columns are: id, affenpinscher, afghan_hound, african_hunting_dog, airedale, american_staffordshire_terrier, appenzeller, australian_terrier, basenji, basset, beagle, bedlington_terrier, bernese_mountain_dog, black-and-tan_coonhound, blenheim_spaniel... and 106 more columns

-> input/dog-breed-identification/labels.csv has 9199 rows and 2 columns.
The columns are: id, breed

-> input/dog-breed-identification/sample_submission.csv has 1023 rows and 121 columns.
The columns are: id, affenpinscher, afghan_hound, african_hunting_dog, airedale, american_staffordshire_terrier, appenzeller, australian_terrier, basenji, basset, beagle, bedlington_terrier, bernese_mountain_dog, black-and-tan_coonhound, blenheim_spaniel... and 106 more columns

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os
import shutil
import random
import numpy as np
import pandas as pd
import cv2

import matplotlib.pyplot as plt
from IPython.display import clear_output

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import tf_keras as keras
from tf_keras import backend as K
from tf_keras.layers import (
    Dense,
    Activation,
    Dropout,
    BatchNormalization,
    Input,
    Flatten,
    MaxPooling2D,
)
from tf_keras.models import Model
from tf_keras.optimizers import Adam
from tf_keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from tf_keras.applications.inception_resnet_v2 import (
    InceptionResNetV2,
    preprocess_input,
)
from tf_keras.initializers import he_normal
from tf_keras.preprocessing.image import ImageDataGenerator

SEED = 1337
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
try:
    keras.utils.set_random_seed(SEED)
except Exception:
    pass

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
try:
    import tensorflow as tf  # tf_keras backend

    try:
        tf.config.optimizer.set_jit(False)
    except Exception:
        pass

    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass

except Exception:
    tf = None

if tf is not None:
    try:
        tf.config.threading.set_intra_op_parallelism_threads(
            max(1, (os.cpu_count() or 4) // 2)
        )
        tf.config.threading.set_inter_op_parallelism_threads(
            max(1, (os.cpu_count() or 4) // 2)
        )
    except Exception:
        pass




## === cell 1
INPUT_DIR = "../input/dog-breed-identification"
if not os.path.exists(INPUT_DIR):
    INPUT_DIR = "/kaggle/input/dog-breed-identification"

TRAIN_DIR = os.path.join(INPUT_DIR, "train")
TEST_DIR = os.path.join(INPUT_DIR, "test")
LABELS_CSV = os.path.join(INPUT_DIR, "labels.csv")
SAMPLE_SUB_CSV = os.path.join(INPUT_DIR, "sample_submission.csv")

print("INPUT_DIR:", INPUT_DIR)
print("TRAIN_DIR exists:", os.path.exists(TRAIN_DIR))
print("TEST_DIR exists:", os.path.exists(TEST_DIR))




## === cell 2
labels = pd.read_csv(LABELS_CSV)
labels.head(5)




## === cell 3
classes = np.unique(labels.breed)
classes_num = classes.size
classes_num




## === cell 4
train_dir = TRAIN_DIR  # the images directory
images_names = os.listdir(train_dir)
images_num = len(images_names)
print(f"Number of images: {images_num}")




## === cell 5
WORK_DIR = "/kaggle/working" if os.path.exists("/kaggle/working") else "."
new_train_dir = os.path.join(WORK_DIR, "new_train")
new_test_dir = os.path.join(WORK_DIR, "new_test")
new_valid_dir = os.path.join(WORK_DIR, "new_valid")

print("WORK_DIR:", WORK_DIR)
print(
    "Skipping creation of unused directory trees:",
    new_train_dir,
    new_test_dir,
    new_valid_dir,
)




## === cell 6
labels_jpg = labels.copy(deep=True)
labels_jpg["filename"] = labels_jpg["id"].astype(str) + ".jpg"
labels_jpg.head(3)




## === cell 7
test_split = 0.1
valid_split = 0.2




## === cell 8
rng = np.random.RandomState(SEED)

u = rng.rand(len(labels_jpg))
is_holdout = u <= test_split
is_valid = (u > test_split) & (u <= (test_split + valid_split))
is_train = u > (test_split + valid_split)

train_df = labels_jpg.loc[is_train, ["filename", "breed", "id"]].reset_index(drop=True)
valid_df = labels_jpg.loc[is_valid, ["filename", "breed", "id"]].reset_index(drop=True)
holdout_df = labels_jpg.loc[is_holdout, ["filename", "breed", "id"]].reset_index(
    drop=True
)

train_size, valid_size, holdout_size = len(train_df), len(valid_df), len(holdout_df)
print("Split sizes:", train_size, valid_size, holdout_size)
print("Breeds:", classes_num)




## === cell 9
pass




## === cell 10
transform_params = {
    "featurewise_center": False,
    "featurewise_std_normalization": False,
    "samplewise_center": False,
    "samplewise_std_normalization": False,
    "rotation_range": 30,
    "width_shift_range": 0.15,
    "height_shift_range": 0.15,
    "horizontal_flip": True,
    "preprocessing_function": preprocess_input,
}

img_gen = ImageDataGenerator(**transform_params)
img_feed = ImageDataGenerator(preprocessing_function=preprocess_input)




## === cell 11
pass




## === cell 12
class Plotter(Callback):
    def plot(self):
        return

    def on_train_begin(self, logs=None):
        self.losses = []
        self.val_losses = []
        self.epochs = []
        self.acc = []
        self.val_acc = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.losses.append(logs.get("loss"))
        self.val_losses.append(logs.get("val_loss"))
        self.acc.append(logs.get("acc", logs.get("accuracy")))
        self.val_acc.append(logs.get("val_acc", logs.get("val_accuracy")))
        self.epochs.append(epoch)


plotter = Plotter()




## === cell 13
plateau_reduce = ReduceLROnPlateau(
    monitor="val_loss", factor=0.01, patience=1, min_lr=1e-20
)
e_stop = EarlyStopping(
    monitor="val_loss", patience=15, mode="min", restore_best_weights=True
)
callbacks = [plotter, plateau_reduce, e_stop]




## === cell 14
def dense_block(x, neurons, layer_no):
    x = Dense(
        neurons, kernel_initializer=he_normal(seed=layer_no), name=f"topDense{layer_no}"
    )(x)
    x = Activation("relu", name=f"Relu{layer_no}")(x)
    x = BatchNormalization(name=f"BatchNorm{layer_no}")(x)
    x = Dropout(0.5, name=f"Dropout{layer_no}")(x)
    return x




## === cell 15
def create_model(shape):
    input_layer = Input(shape, name="input_layer")
    incep_res = InceptionResNetV2(
        include_top=False, weights="imagenet", input_tensor=input_layer
    )
    for layer in incep_res.layers:
        layer.trainable = False

    pool = MaxPooling2D(pool_size=[3, 3], strides=[3, 3], padding="same")(
        incep_res.output
    )
    flat1 = Flatten(name="Flatten1")(pool)
    flat1_bn = BatchNormalization(name="BatchNormFlat")(flat1)

    dens1 = dense_block(flat1_bn, neurons=512, layer_no=1)
    dens2 = dense_block(dens1, neurons=512, layer_no=2)
    dens3 = dense_block(dens2, neurons=1024, layer_no=3)

    dens_final = Dense(classes_num, name="Dense4")(dens3)
    output_layer = Activation("softmax", name="Softmax")(dens_final)

    model = Model(inputs=[input_layer], outputs=[output_layer])
    return model




## === cell 16
height, width, channels_num = 512, 512, 3
learning_rate = 0.004
epochs = 15
batch_size = 32

model = create_model((height, width, channels_num))
optimizer = Adam(learning_rate=learning_rate)

model.compile(
    optimizer=optimizer,
    loss="categorical_crossentropy",
    metrics=["acc"],
    jit_compile=False,
)
model.summary()




## === cell 17
AUTOTUNE = tf.data.AUTOTUNE

class_to_idx = {c: i for i, c in enumerate(classes)}
idx_to_class = {i: c for c, i in class_to_idx.items()}


def _paths_and_labels_from_df(df, directory, with_labels=True):
    paths = (directory.rstrip("/") + "/" + df["filename"].astype(str)).to_numpy()
    if not with_labels:
        return paths, None
    y_idx = df["breed"].map(class_to_idx).to_numpy(np.int32)
    return paths, y_idx


@tf.function
def _decode_resize_preprocess(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(
        img, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )
    img = tf.cast(img, tf.float32)
    img = preprocess_input(img)
    return img


@tf.function
def _augment(img, seedpair):
    s1, s2 = seedpair[0], seedpair[1]
    img = tf.image.stateless_random_flip_left_right(img, seed=[s1, s2])

    max_dx = int(round(width * 0.15))
    max_dy = int(round(height * 0.15))
    dx = tf.random.stateless_uniform(
        [], seed=[s1 + 11, s2 + 17], minval=-max_dx, maxval=max_dx + 1, dtype=tf.int32
    )
    dy = tf.random.stateless_uniform(
        [], seed=[s1 + 23, s2 + 29], minval=-max_dy, maxval=max_dy + 1, dtype=tf.int32
    )
    pad_x = max_dx
    pad_y = max_dy
    img = tf.pad(img, [[pad_y, pad_y], [pad_x, pad_x], [0, 0]], mode="REFLECT")
    start_y = pad_y + dy
    start_x = pad_x + dx
    img = tf.image.crop_to_bounding_box(img, start_y, start_x, height, width)

    angle = tf.random.stateless_uniform(
        [], seed=[s1 + 37, s2 + 41], minval=-30.0, maxval=30.0, dtype=tf.float32
    )
    angle = angle * (np.pi / 180.0)

    cos_a = tf.cos(angle)
    sin_a = tf.sin(angle)
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    a0 = cos_a
    a1 = -sin_a
    a2 = cx - cos_a * cx + sin_a * cy
    b0 = sin_a
    b1 = cos_a
    b2 = cy - sin_a * cx - cos_a * cy
    transform = tf.stack([a0, a1, a2, b0, b1, b2, 0.0, 0.0])[None, :]
    img = tf.raw_ops.ImageProjectiveTransformV3(
        images=img[None, ...],
        transforms=transform,
        output_shape=tf.constant([height, width], dtype=tf.int32),
        interpolation="BILINEAR",
        fill_mode="REFLECT",
        fill_value=0.0,
    )[0]
    return img


def make_dataset(df, directory, training, with_labels=True):
    paths, y_idx = _paths_and_labels_from_df(df, directory, with_labels=with_labels)
    ds = tf.data.Dataset.from_tensor_slices(paths if y_idx is None else (paths, y_idx))

    if training:
        shuffle_buf = min(len(df), 2048)
        ds = ds.shuffle(
            buffer_size=shuffle_buf, seed=SEED, reshuffle_each_iteration=True
        )

    cache_tag = f"{'train' if training else 'eval'}_{with_labels}_{height}x{width}_b{batch_size}_s{SEED}"
    cache_path = os.path.join(WORK_DIR, f"cache_decoded_{cache_tag}")

    options = tf.data.Options()
    options.deterministic = True
    try:
        options.threading.private_threadpool_size = max(4, (os.cpu_count() or 4))
        options.threading.max_intra_op_parallelism = 1
    except Exception:
        pass
    ds = ds.with_options(options)

    if with_labels:

        @tf.function
        def _decode_map(path, y):
            img = _decode_resize_preprocess(path)
            return path, img, y

        ds = ds.map(_decode_map, num_parallel_calls=AUTOTUNE, deterministic=True)
        ds = ds.cache(cache_path)

        @tf.function
        def _final_map(path, img, y):
            if training:
                h = tf.strings.to_hash_bucket_fast(path, 2**31 - 1)
                seedpair = tf.stack(
                    [
                        tf.cast(h % 2147483647, tf.int32),
                        tf.cast((h // 2147483647) + SEED, tf.int32),
                    ]
                )
                img = _augment(img, seedpair)
            y_oh = tf.one_hot(y, depth=classes_num, dtype=tf.float32)
            return img, y_oh

        ds = ds.map(_final_map, num_parallel_calls=AUTOTUNE, deterministic=True)

    else:

        @tf.function
        def _decode_map(path):
            img = _decode_resize_preprocess(path)
            return img

        ds = ds.map(_decode_map, num_parallel_calls=AUTOTUNE, deterministic=True)
        ds = ds.cache(cache_path)

    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_ds = make_dataset(train_df, TRAIN_DIR, training=True, with_labels=True)
valid_ds = make_dataset(valid_df, TRAIN_DIR, training=False, with_labels=True)




## === cell 18
history = model.fit(
    train_ds,
    epochs=epochs,
    validation_data=valid_ds,
    callbacks=callbacks,
    verbose=1,
)




## === cell 19
holdout_ds = make_dataset(holdout_df, TRAIN_DIR, training=False, with_labels=True)

metrics = model.evaluate(holdout_ds, verbose=1)
m_names = model.metrics_names
print(f"{m_names[0]} = {metrics[0]}\n{m_names[1]} = {metrics[1]}")




## === cell 20
one_hot_map = class_to_idx
list(one_hot_map.items())[:5], len(one_hot_map)




## === cell 21
sample_sub = pd.read_csv(SAMPLE_SUB_CSV)
breed_cols = [c for c in sample_sub.columns if c != "id"]

test_ids = sample_sub["id"].astype(str).tolist()
test_df = pd.DataFrame({"filename": [f"{tid}.jpg" for tid in test_ids]})

test_paths, _ = _paths_and_labels_from_df(test_df, TEST_DIR, with_labels=False)
test_ds = tf.data.Dataset.from_tensor_slices(test_paths)

options = tf.data.Options()
options.deterministic = True
try:
    options.threading.private_threadpool_size = max(4, (os.cpu_count() or 4))
    options.threading.max_intra_op_parallelism = 1
except Exception:
    pass
test_ds = test_ds.with_options(options)

test_ds = test_ds.map(
    _decode_resize_preprocess, num_parallel_calls=AUTOTUNE, deterministic=True
)

test_cache_path = os.path.join(
    WORK_DIR, f"cache_decoded_test_{height}x{width}_b{batch_size}_s{SEED}"
)
test_ds = test_ds.cache(test_cache_path)

test_ds = test_ds.batch(32, drop_remainder=False).prefetch(AUTOTUNE)

pred = model.predict(
    test_ds,
    verbose=1,
)
pred = pred[: len(sample_sub)]  # safety

idx_to_class = {v: k for k, v in one_hot_map.items()}
model_class_order = [idx_to_class[i] for i in range(classes_num)]

pred_df = pd.DataFrame(pred, columns=model_class_order)

pred_df = pred_df.reindex(columns=breed_cols).fillna(1.0 / len(breed_cols))

eps = 1e-7
pred_df = pred_df.clip(lower=eps, upper=1.0 - eps)
pred_df = pred_df.div(pred_df.sum(axis=1), axis=0)

submission = pd.concat([sample_sub[["id"]].copy(), pred_df], axis=1)
assert submission.shape == sample_sub.shape, (submission.shape, sample_sub.shape)
assert list(submission.columns) == list(sample_sub.columns)

out_path = os.path.join(WORK_DIR, "submission.csv")
submission.to_csv(out_path, index=False)
print("Wrote:", out_path)
submission.head()
