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

# 5. Target score

0.28106

# 6. Current score

5.08502

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 5.08483) has done: 'The timeout is dominated by training InceptionResNetV2 on 512×512 images with heavy augmentation and slow Python-side generators, plus multiprocessing overhead. To keep the exact same model/training logic but make it faster, I (1) enable XLA/JIT compilation for the TF graph, (2) switch the generators to use OpenCV (much faster decode/resize than PIL) without changing augmentation semantics, and (3) tune `workers/max_queue_size` for this workload while avoiding excess IPC overhead. I also remove a small O(N) label lookup inside the preview-sampling loop and keep determinism settings intact.'
- What this solution (achieved 5.08502) has done: 'The timeout is dominated by decoding/resizing 512×512 JPEGs every epoch and (more importantly) doing expensive per-image geometric augmentation (projective transform) inside `tf.map_fn` for every training step. To keep the exact same model and augmentation logic, the main speedups are: (1) replace `tf.map_fn` with fully-vectorized stateless ops across the whole batch (same math/semantics, far less overhead), and (2) remove unnecessary `@tf.function` wrappers that force retracing and add overhead, while keeping deterministic options and seeds. Additionally, we keep the on-disk cache for decoded images (so augmentation still changes every epoch) but ensure the cache directory is unique and created once to avoid repeated filesystem work.'
- What this solution (achieved 5.08502) has done: 'I fix the protobuf-related crash by removing the forced pure-Python protobuf implementation that is incompatible with the Kaggle runtime, while keeping the same TF/Keras stack. Then I fix the tf.data augmentation bug: `stateless_random_flip_left_right` requires a single shape-(2,) seed per call, so we apply per-image flips with `tf.map_fn` (same augmentation semantics as intended, just correct) to unblock dataset creation and training. Finally, I ensure the training/prediction pipeline runs end-to-end and always writes a valid `submission.csv` matching `sample_submission.csv` columns and order.'

# 9. Code solution

## === cell 0
import os
import shutil
import random
import numpy as np
import pandas as pd
import cv2

import matplotlib.pyplot as plt
from IPython.display import clear_output

if "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION" in os.environ:
    os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION")

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

print("TF version:", getattr(tf, "__version__", None))



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

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
from tf_keras.layers import Reshape


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

    flat1 = Reshape((38400,), name="Flatten1")(pool)
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


def _add_stateless_seeds(df, directory):
    paths = (directory.rstrip("/") + "/" + df["filename"].astype(str)).to_numpy()
    h = pd.util.hash_pandas_object(pd.Series(paths), index=False).to_numpy(np.uint64)
    s1 = (h % np.uint64(2147483647)).astype(np.int32)
    s2 = ((h // np.uint64(2147483647)) % np.uint64(2147483647)).astype(np.int32)
    df = df.copy()
    df["seed1"] = s1
    df["seed2"] = s2
    return df


train_df = _add_stateless_seeds(train_df, TRAIN_DIR)
valid_df = _add_stateless_seeds(valid_df, TRAIN_DIR)
holdout_df = _add_stateless_seeds(holdout_df, TRAIN_DIR)


def _decode_resize_preprocess(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(
        img, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )
    img = tf.cast(img, tf.float32)
    img = preprocess_input(img)
    return img


def _augment_batch(images, y_idx, seed1, seed2):
    seed1 = tf.cast(seed1, tf.int32)
    seed2 = tf.cast(seed2, tf.int32) + tf.cast(SEED, tf.int32)

    def _flip_one(args):
        img, s1, s2 = args
        return tf.image.stateless_random_flip_left_right(img, seed=tf.stack([s1, s2]))

    images = tf.map_fn(
        _flip_one,
        (images, seed1, seed2),
        fn_output_signature=tf.float32,
        parallel_iterations=32,
    )

    max_dx = int(round(width * 0.15))
    max_dy = int(round(height * 0.15))

    dx = tf.random.stateless_uniform(
        [tf.shape(images)[0]],
        seed=tf.stack([seed1 + 11, seed2 + 17], axis=1),
        minval=-max_dx,
        maxval=max_dx + 1,
        dtype=tf.int32,
    )
    dy = tf.random.stateless_uniform(
        [tf.shape(images)[0]],
        seed=tf.stack([seed1 + 23, seed2 + 29], axis=1),
        minval=-max_dy,
        maxval=max_dy + 1,
        dtype=tf.int32,
    )

    pad_x = max_dx
    pad_y = max_dy
    images = tf.pad(
        images, [[0, 0], [pad_y, pad_y], [pad_x, pad_x], [0, 0]], mode="REFLECT"
    )
    start_y = pad_y + dy
    start_x = pad_x + dx
    boxes = tf.stack(
        [
            tf.cast(start_y, tf.float32) / tf.cast(height + 2 * pad_y, tf.float32),
            tf.cast(start_x, tf.float32) / tf.cast(width + 2 * pad_x, tf.float32),
            tf.cast(start_y + height, tf.float32)
            / tf.cast(height + 2 * pad_y, tf.float32),
            tf.cast(start_x + width, tf.float32)
            / tf.cast(width + 2 * pad_x, tf.float32),
        ],
        axis=1,
    )
    box_ind = tf.range(tf.shape(images)[0], dtype=tf.int32)
    images = tf.image.crop_and_resize(
        images, boxes, box_ind, crop_size=[height, width], method="bilinear"
    )

    angle = tf.random.stateless_uniform(
        [tf.shape(images)[0]],
        seed=tf.stack([seed1 + 37, seed2 + 41], axis=1),
        minval=-30.0,
        maxval=30.0,
        dtype=tf.float32,
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
    transforms = tf.stack(
        [a0, a1, a2, b0, b1, b2, tf.zeros_like(a0), tf.zeros_like(a0)], axis=1
    )

    images = tf.raw_ops.ImageProjectiveTransformV3(
        images=images,
        transforms=transforms,
        output_shape=tf.constant([height, width], dtype=tf.int32),
        interpolation="BILINEAR",
        fill_mode="REFLECT",
        fill_value=0.0,
    )

    y_oh = tf.one_hot(y_idx, depth=classes_num, dtype=tf.float32)
    return images, y_oh


def _noaug_batch(images, y_idx):
    y_oh = tf.one_hot(y_idx, depth=classes_num, dtype=tf.float32)
    return images, y_oh


_base_options = tf.data.Options()
_base_options.deterministic = True
try:
    _base_options.threading.private_threadpool_size = max(4, (os.cpu_count() or 4))
    _base_options.threading.max_intra_op_parallelism = 1
    _base_options.experimental_optimization.map_parallelization = True
    _base_options.experimental_optimization.parallel_batch = True
except Exception:
    pass


def make_dataset(df, directory, training, with_labels=True):
    if with_labels:
        paths = (directory.rstrip("/") + "/" + df["filename"].astype(str)).to_numpy()
        y_idx = df["breed"].map(class_to_idx).to_numpy(np.int32)
        s1 = df["seed1"].to_numpy(np.int32)
        s2 = df["seed2"].to_numpy(np.int32)
        ds = tf.data.Dataset.from_tensor_slices((paths, y_idx, s1, s2))
    else:
        paths = (directory.rstrip("/") + "/" + df["filename"].astype(str)).to_numpy()
        ds = tf.data.Dataset.from_tensor_slices(paths)

    if training:
        shuffle_buf = min(len(df), 2048)
        ds = ds.shuffle(
            buffer_size=shuffle_buf, seed=SEED, reshuffle_each_iteration=True
        )

    cache_tag = f"{'train' if training else 'eval'}_{with_labels}_{height}x{width}_b{batch_size}_s{SEED}"
    cache_path = os.path.join(WORK_DIR, f"cache_decoded_{cache_tag}")
    os.makedirs(WORK_DIR, exist_ok=True)

    ds = ds.with_options(_base_options)

    if with_labels:

        def _decode_map(path, y, seed1, seed2):
            img = _decode_resize_preprocess(path)
            return img, y, seed1, seed2

        ds = ds.map(_decode_map, num_parallel_calls=AUTOTUNE, deterministic=True)
        ds = ds.cache(cache_path)
        ds = ds.batch(batch_size, drop_remainder=False)

        if training:
            ds = ds.map(_augment_batch, num_parallel_calls=AUTOTUNE, deterministic=True)
        else:

            def _eval_batch(images, y, seed1, seed2):
                return _noaug_batch(images, y)

            ds = ds.map(_eval_batch, num_parallel_calls=AUTOTUNE, deterministic=True)

    else:

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

print("Datasets ready:", train_ds, valid_ds)



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3172343825.py in <cell line: 0>()
    206 
    207 
--> 208 train_ds = make_dataset(train_df, TRAIN_DIR, training=True, with_labels=True)
    209 valid_ds = make_dataset(valid_df, TRAIN_DIR, training=False, with_labels=True)
    210 

/tmp/ipykernel_11/3172343825.py in make_dataset(df, directory, training, with_labels)
    184 
    185         if training:
--> 186             ds = ds.map(_augment_batch, num_parallel_calls=AUTOTUNE, deterministic=True)
    187         else:
    188 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/dataset_ops.py in map(self, map_func, num_parallel_calls, deterministic, synchronous, use_unbounded_threadpool, name)
   2339     from tensorflow.python.data.ops import map_op
   2340 
-> 2341     return map_op._map_v2(
   2342         self,
   2343         map_func,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/map_op.py in _map_v2(input_dataset, map_func, num_parallel_calls, deterministic, synchronous, use_unbounded_threadpool, name)
     55           num_parallel_calls,
     56       )
---> 57     return _ParallelMapDataset(
     58         input_dataset,
     59         map_func,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/map_op.py in __init__(self, input_dataset, map_func, num_parallel_calls, deterministic, use_inter_op_parallelism, preserve_cardinality, use_legacy_function, use_unbounded_threadpool, name)
    200     self._input_dataset = input_dataset
    201     self._use_inter_op_parallelism = use_inter_op_parallelism
--> 202     self._map_func = structured_function.StructuredFunctionWrapper(
    203         map_func,
    204         self._transformation_name(),

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/structured_function.py in __init__(self, func, transformation_name, dataset, input_classes, input_shapes, input_types, input_structure, add_to_graph, use_legacy_function, defun_kwargs)
    263         fn_factory = trace_tf_function(defun_kwargs)
    264 
--> 265     self._function = fn_factory()
    266     # There is no graph to add in eager mode.
    267     add_to_graph &= not context.executing_eagerly()

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py in get_concrete_function(self, *args, **kwargs)
   1249   def get_concrete_function(self, *args, **kwargs):
   1250     # Implements PolymorphicFunction.get_concrete_function.
-> 1251     concrete = self._get_concrete_function_garbage_collected(*args, **kwargs)
   1252     concrete._garbage_collector.release()  # pylint: disable=protected-access
   1253     return concrete

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py in _get_concrete_function_garbage_collected(self, *args, **kwargs)
   1219       if self._variable_creation_config is None:
   1220         initializers = []
-> 1221         self._initialize(args, kwargs, add_initializers_to=initializers)
   1222         self._initialize_uninitialized_variables(initializers)
   1223 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py in _initialize(self, args, kwds, add_initializers_to)
    694     )
    695     # Force the definition of the function for these arguments
--> 696     self._concrete_variable_creation_fn = tracing_compilation.trace_function(
    697         args, kwds, self._variable_creation_config
    698     )

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/tracing_compilation.py in trace_function(args, kwargs, tracing_options)
    176       kwargs = {}
    177 
--> 178     concrete_function = _maybe_define_function(
    179         args, kwargs, tracing_options
    180     )

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/tracing_compilation.py in _maybe_define_function(args, kwargs, tracing_options)
    281         else:
    282           target_func_type = lookup_func_type
--> 283         concrete_function = _create_concrete_function(
    284             target_func_type, lookup_func_context, func_graph, tracing_options
    285         )

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/tracing_compilation.py in _create_concrete_function(function_type, type_context, func_graph, tracing_options)
    308       attributes_lib.DISABLE_ACD, False
    309   )
--> 310   traced_func_graph = func_graph_module.func_graph_from_py_func(
    311       tracing_options.name,
    312       tracing_options.python_function,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/func_graph.py in func_graph_from_py_func(name, python_func, args, kwargs, signature, func_graph, add_control_dependencies, arg_names, op_return_value, collections, capture_by_value, create_placeholders)
   1057 
   1058     _, original_func = tf_decorator.unwrap(python_func)
-> 1059     func_outputs = python_func(*func_args, **func_kwargs)
   1060 
   1061     # invariant: `func_outputs` contains only Tensors, CompositeTensors,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py in wrapped_fn(*args, **kwds)
    597         # the function a weak reference to itself to avoid a reference cycle.
    598         with OptionalXlaContext(compile_with_xla):
--> 599           out = weak_wrapped_fn().__wrapped__(*args, **kwds)
    600         return out
    601 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/structured_function.py in wrapped_fn(*args)
    229       # Note: wrapper_helper will apply autograph based on context.
    230       def wrapped_fn(*args):  # pylint: disable=missing-docstring
--> 231         ret = wrapper_helper(*args)
    232         ret = structure.to_tensor_list(self._output_structure, ret)
    233         return [ops.convert_to_tensor(t) for t in ret]

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/structured_function.py in wrapper_helper(*args)
    159       if not _should_unpack(nested_args):
    160         nested_args = (nested_args,)
--> 161       ret = autograph.tf_convert(self._func, ag_ctx)(*nested_args)
    162       ret = variable_utils.convert_variables_to_tensors(ret)
    163       if _should_pack(ret):

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in wrapper(*args, **kwargs)
    691       except Exception as e:  # pylint:disable=broad-except
    692         if hasattr(e, 'ag_error_metadata'):
--> 693           raise e.ag_error_metadata.to_exception(e)
    694         else:
    695           raise

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in wrapper(*args, **kwargs)
    688       try:
    689         with conversion_ctx:
--> 690           return converted_call(f, args, kwargs, options=options)
    691       except Exception as e:  # pylint:disable=broad-except
    692         if hasattr(e, 'ag_error_metadata'):

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in converted_call(f, args, kwargs, caller_fn_scope, options)
    437     try:
    438       if kwargs is not None:
--> 439         result = converted_f(*effective_args, **kwargs)
    440       else:
    441         result = converted_f(*effective_args)

/tmp/__autograph_generated_fileqo8irhgz.py in tf___augment_batch(images, y_idx, seed1, seed2)
     27                 max_dx = ag__.converted_call(ag__.ld(int), (ag__.converted_call(ag__.ld(round), (ag__.ld(width) * 0.15,), None, fscope),), None, fscope)
     28                 max_dy = ag__.converted_call(ag__.ld(int), (ag__.converted_call(ag__.ld(round), (ag__.ld(height) * 0.15,), None, fscope),), None, fscope)
---> 29                 dx = ag__.converted_call(ag__.ld(tf).random.stateless_uniform, ([ag__.converted_call(ag__.ld(tf).shape, (ag__.ld(images),), None, fscope)[0]],), dict(seed=ag__.converted_call(ag__.ld(tf).stack, ([ag__.ld(seed1) + 11, ag__.ld(seed2) + 17],), dict(axis=1), fscope), minval=-ag__.ld(max_dx), maxval=ag__.ld(max_dx) + 1, dtype=ag__.ld(tf).int32), fscope)
     30                 dy = ag__.converted_call(ag__.ld(tf).random.stateless_uniform, ([ag__.converted_call(ag__.ld(tf).shape, (ag__.ld(images),), None, fscope)[0]],), dict(seed=ag__.converted_call(ag__.ld(tf).stack, ([ag__.ld(seed1) + 23, ag__.ld(seed2) + 29],), dict(axis=1), fscope), minval=-ag__.ld(max_dy), maxval=ag__.ld(max_dy) + 1, dtype=ag__.ld(tf).int32), fscope)
     31                 pad_x = ag__.ld(max_dx)

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in converted_call(f, args, kwargs, caller_fn_scope, options)
    375 
    376   if not options.user_requested and conversion.is_allowlisted(f):
--> 377     return _call_unconverted(f, args, kwargs, options)
    378 
    379   # internal_convert_user_code is for example turned off when issuing a dynamic

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in _call_unconverted(f, args, kwargs, options, update_cache)
    457 
    458   if kwargs is not None:
--> 459     return f(*args, **kwargs)
    460   return f(*args)
    461 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/traceback_utils.py in error_handler(*args, **kwargs)
    151     except Exception as e:
    152       filtered_tb = _process_traceback_frames(e.__traceback__)
--> 153       raise e.with_traceback(filtered_tb) from None
    154     finally:
    155       del filtered_tb

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py in _create_c_op(graph, node_def, inputs, control_inputs, op_def, extract_traceback)
   1054   except errors.InvalidArgumentError as e:
   1055     # Convert to ValueError for backwards compatibility.
-> 1056     raise ValueError(e.message)
   1057 
   1058   # Record the current Python stack trace as the creating stacktrace of this

ValueError: in user code:

    File "/tmp/ipykernel_11/3172343825.py", line 62, in _augment_batch  *
        dx = tf.random.stateless_uniform(

    ValueError: Shape must be rank 1 but is rank 2 for '{{node stateless_random_uniform/StatelessRandomGetKeyCounter}} = StatelessRandomGetKeyCounter[Tseed=DT_INT32](stack)' with input shapes: [?,2].


## === cell 18
history = model.fit(
    train_ds,
    epochs=epochs,
    validation_data=valid_ds,
    callbacks=callbacks,
    verbose=1,
)



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/42800090.py in <cell line: 0>()
      1 history = model.fit(
----> 2     train_ds,
      3     epochs=epochs,
      4     validation_data=valid_ds,
      5     callbacks=callbacks,

NameError: name 'train_ds' is not defined

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
test_ds = tf.data.Dataset.from_tensor_slices(test_paths).with_options(_base_options)

test_ds = test_ds.map(
    _decode_resize_preprocess, num_parallel_calls=AUTOTUNE, deterministic=True
)

test_cache_path = os.path.join(
    WORK_DIR, f"cache_decoded_test_{height}x{width}_b{batch_size}_s{SEED}"
)
test_ds = test_ds.cache(test_cache_path)

test_ds = test_ds.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)

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
