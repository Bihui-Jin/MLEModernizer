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

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import random
import numpy as np
import pandas as pd
import cv2

import matplotlib.pyplot as plt
from IPython.display import clear_output

import tensorflow as tf
import tf_keras as keras
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
from tf_keras.applications.inception_resnet_v2 import InceptionResNetV2
from tf_keras.initializers import he_normal
from tf_keras.preprocessing.image import ImageDataGenerator

SEED = 1337
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

print("OK imports. Using tf_keras:", keras.__version__)
print("TF version:", tf.__version__)
print("KERAS_BACKEND:", os.environ.get("KERAS_BACKEND"))
print(
    "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION:",
    os.environ.get("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"),
)

try:
    tf.config.threading.set_intra_op_parallelism_threads(
        max(1, (os.cpu_count() or 2) // 2)
    )
    tf.config.threading.set_inter_op_parallelism_threads(2)
except Exception:
    pass

try:
    tf.config.run_functions_eagerly(False)
except Exception:
    pass




## === cell 1
input_dir = "../input/dog-breed-identification"
train_dir = os.path.join(input_dir, "train")
test_dir = os.path.join(input_dir, "test")

labels_path = os.path.join(input_dir, "labels.csv")
sample_sub_path = os.path.join(input_dir, "sample_submission.csv")

print(train_dir, test_dir, labels_path, sample_sub_path)




## === cell 2
labels = pd.read_csv(labels_path)
labels.head()




## === cell 3
classes = np.unique(labels.breed.values)
classes_num = classes.size
print("Num classes:", classes_num)




## === cell 4
images_names = os.listdir(train_dir)
images_num = len(images_names)
print(f"Number of training images: {images_num}")




## === cell 5
work_base = "/kaggle/working/dog_split"
new_train_dir = os.path.join(work_base, "train")
new_valid_dir = os.path.join(work_base, "valid")
new_test_dir = os.path.join(work_base, "test")

print(
    "Skipping on-disk split folder creation to eliminate heavy I/O. Using in-memory split instead."
)




## === cell 6
labels_jpg = labels.copy(deep=True)
labels_jpg["filename"] = labels_jpg["id"].astype(str) + ".jpg"
print(
    classes[0],
    labels_jpg[labels_jpg["breed"] == classes[0]]["filename"].head().tolist(),
)




## === cell 7
test_split = 0.1
valid_split = 0.2




## === cell 8
def deterministic_bucket_vec(filenames, seed=SEED):
    s = pd.Series(filenames, dtype="string")
    h = (
        pd.util.hash_pandas_object((str(seed) + "_" + s), index=False)
        .astype("uint64")
        .to_numpy()
    )
    return (h / np.float64(2**64)).astype(np.float64)


buckets = deterministic_bucket_vec(labels_jpg["filename"].tolist(), seed=SEED)
labels_jpg = labels_jpg.assign(_bucket=buckets)

labels_jpg["_split"] = np.where(
    labels_jpg["_bucket"] <= test_split,
    "test",
    np.where(labels_jpg["_bucket"] <= (test_split + valid_split), "valid", "train"),
)

train_df = labels_jpg[labels_jpg["_split"] == "train"][
    ["filename", "breed"]
].reset_index(drop=True)
valid_df = labels_jpg[labels_jpg["_split"] == "valid"][
    ["filename", "breed"]
].reset_index(drop=True)
test_df_internal = labels_jpg[labels_jpg["_split"] == "test"][
    ["filename", "breed"]
].reset_index(drop=True)

print(
    "Split sizes:",
    len(train_df),
    len(valid_df),
    len(test_df_internal),
    "missing/skipped:",
    0,
)




## === cell 9
test_breed = classes[0]
example_files = (
    train_df.loc[train_df["breed"] == test_breed, "filename"].head(5).tolist()
)
print("Example files:", example_files)




## === cell 10
width, height, channels_num = 512, 512, 3




## === cell 11
do_quick_viz = False
if do_quick_viz:
    images_samples = np.zeros((4, height, width, 3), dtype=float)
    samples_labels = []

    rnd_indexes = np.random.randint(0, images_num, 4)
    for i, rnd_idx in enumerate(rnd_indexes):
        img_filename = images_names[rnd_idx]
        img_id = img_filename[:-4]
        img_bgr = cv2.imread(os.path.join(train_dir, img_filename))
        img_rgb = img_bgr[:, :, [2, 1, 0]]
        images_samples[i] = cv2.resize(src=img_rgb, dsize=(width, height)) / 255.0
        img_label = labels.breed[labels.id == img_id].values[0]
        samples_labels.append(img_label)

    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    for ax, img, label in zip(axs.ravel(), images_samples, samples_labels):
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(f"Class: {label}", size=15)
    plt.show()
else:
    print("Skipping visualization to save runtime.")




## === cell 12
norm_factor = 1 / 255.0
transform_params = {
    "featurewise_center": False,
    "featurewise_std_normalization": False,
    "samplewise_center": False,
    "samplewise_std_normalization": False,
    "rotation_range": 30,
    "width_shift_range": 0.15,
    "height_shift_range": 0.15,
    "horizontal_flip": True,
    "rescale": norm_factor,
}
img_gen = ImageDataGenerator(**transform_params)
img_feed = ImageDataGenerator(rescale=1 / 255.0)




## === cell 13
class Plotter(Callback):
    def __init__(self, plot_every=5):
        super().__init__()
        self.plot_every = int(plot_every)

    def plot(self):
        clear_output(wait=True)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        ax1.plot(self.epochs, self.losses, label="train_loss")
        ax1.plot(self.epochs, self.val_losses, label="val_loss")

        ax2.plot(self.epochs, self.acc, label="train_acc")
        ax2.plot(self.epochs, self.val_acc, label="val_acc")

        ax1.set_title("Loss vs Epochs")
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Loss")

        ax2.set_title("Accuracy vs Epochs")
        ax2.set_xlabel("Epochs")
        ax2.set_ylabel("Accuracy")

        ax1.legend()
        ax2.legend()
        plt.show()

        if self.epochs:
            print(
                f"Epoch #{self.epochs[-1]+1} >> train_acc={self.acc[-1]*100:.3f}%, train_loss={self.losses[-1]:.5f}"
            )
            print(
                f"Epoch #{self.epochs[-1]+1} >> val_acc={self.val_acc[-1]*100:.3f}%, val_loss={self.val_losses[-1]:.5f}"
            )

    def on_train_begin(self, logs=None):
        self.losses, self.val_losses, self.epochs = [], [], []
        self.acc, self.val_acc = [], []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.losses.append(logs.get("loss"))
        self.val_losses.append(logs.get("val_loss"))
        self.acc.append(logs.get("acc", logs.get("accuracy")))
        self.val_acc.append(logs.get("val_acc", logs.get("val_accuracy")))
        self.epochs.append(epoch)

        if (
            (epoch == 0)
            or ((epoch + 1) % self.plot_every == 0)
            or (epoch + 1 == getattr(self.params, "epochs", epoch + 1))
        ):
            self.plot()


plotter = Plotter(plot_every=5)




## === cell 14
plateau_reduce = ReduceLROnPlateau(
    monitor="val_loss", factor=0.01, patience=1, min_lr=1e-20
)
e_stop = EarlyStopping(
    monitor="val_loss", patience=15, mode="min", restore_best_weights=True
)

callbacks = [plateau_reduce, e_stop]




## === cell 15
def dense_block(x, neurons, layer_no):
    x = Dense(
        neurons, kernel_initializer=he_normal(layer_no), name=f"topDense{layer_no}"
    )(x)
    x = Activation("relu", name=f"Relu{layer_no}")(x)
    x = BatchNormalization(name=f"BatchNorm{layer_no}")(x)
    x = Dropout(0.5, name=f"Dropout{layer_no}")(x)
    return x


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
learning_rate = 0.004
epochs = 15
batch_size = 32

model = create_model((height, width, channels_num))
optimizer = Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["acc"])
model.summary()




## === cell 17
AUTOTUNE = tf.data.AUTOTUNE

class_to_index = {c: i for i, c in enumerate(list(classes))}
index_to_class = {i: c for c, i in class_to_index.items()}


def _build_paths_and_labels(df, directory, labeled=True):
    paths = (directory.rstrip("/") + "/" + df["filename"].astype(str)).tolist()
    if labeled:
        y = df["breed"].map(class_to_index).astype(np.int32).to_numpy()
        return paths, y
    return paths, None


train_paths, train_y = _build_paths_and_labels(train_df, train_dir, labeled=True)
valid_paths, valid_y = _build_paths_and_labels(valid_df, train_dir, labeled=True)

HEIGHT_T = tf.constant(height, dtype=tf.int32)
WIDTH_T = tf.constant(width, dtype=tf.int32)
INV_255 = tf.constant(1.0 / 255.0, dtype=tf.float32)
SEED_T = tf.constant(SEED, dtype=tf.int32)
MAX_ROT = tf.constant(30.0 * (np.pi / 180.0), dtype=tf.float32)
MAX_SHIFT = tf.constant(0.15, dtype=tf.float32)


@tf.function(reduce_retracing=True)
def _decode_resize(path):
    img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(
        img, [HEIGHT_T, WIDTH_T], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )
    img = tf.cast(img, tf.float32) * INV_255
    return img


@tf.function(reduce_retracing=True)
def _rotate_image(img, angle_rad):
    angle = tf.cast(angle_rad, tf.float32)
    cos_a = tf.math.cos(angle)
    sin_a = tf.math.sin(angle)

    h = tf.cast(tf.shape(img)[0], tf.float32)
    w = tf.cast(tf.shape(img)[1], tf.float32)
    cx = (w - 1.0) / 2.0
    cy = (h - 1.0) / 2.0

    a0 = cos_a
    a1 = sin_a
    a2 = (1.0 - cos_a) * cx - sin_a * cy
    b0 = -sin_a
    b1 = cos_a
    b2 = sin_a * cx + (1.0 - cos_a) * cy

    transform = tf.stack([a0, a1, a2, b0, b1, b2, 0.0, 0.0])[tf.newaxis, :]
    img4 = img[tf.newaxis, ...]
    out = tf.raw_ops.ImageProjectiveTransformV3(
        images=img4,
        transforms=transform,
        output_shape=tf.stack([HEIGHT_T, WIDTH_T]),
        fill_value=0.0,
        interpolation="BILINEAR",
        fill_mode="REFLECT",
    )
    return out[0]


@tf.function(reduce_retracing=True)
def _augment(img, y, epoch_i, index_i):
    seed_pair = tf.stack(
        [SEED_T + tf.cast(epoch_i, tf.int32), tf.cast(index_i, tf.int32)], axis=0
    )

    img = tf.image.stateless_random_flip_left_right(img, seed_pair)

    seed_pair2 = seed_pair + tf.constant([1, 0], dtype=tf.int32)
    seed_pair3 = seed_pair + tf.constant([2, 0], dtype=tf.int32)
    dx = tf.random.stateless_uniform(
        [], seed_pair2, minval=-MAX_SHIFT, maxval=MAX_SHIFT, dtype=tf.float32
    )
    dy = tf.random.stateless_uniform(
        [], seed_pair3, minval=-MAX_SHIFT, maxval=MAX_SHIFT, dtype=tf.float32
    )

    shift_x = tf.cast(tf.round(dx * tf.cast(WIDTH_T, tf.float32)), tf.int32)
    shift_y = tf.cast(tf.round(dy * tf.cast(HEIGHT_T, tf.float32)), tf.int32)

    pad_x = tf.abs(shift_x)
    pad_y = tf.abs(shift_y)
    img = tf.pad(img, [[pad_y, pad_y], [pad_x, pad_x], [0, 0]], mode="REFLECT")
    offset_y = pad_y - shift_y
    offset_x = pad_x - shift_x
    img = tf.image.crop_to_bounding_box(img, offset_y, offset_x, HEIGHT_T, WIDTH_T)

    seed_pair4 = seed_pair + tf.constant([3, 0], dtype=tf.int32)
    angle = tf.random.stateless_uniform(
        [], seed_pair4, minval=-MAX_ROT, maxval=MAX_ROT, dtype=tf.float32
    )
    img = _rotate_image(img, angle)

    y_oh = tf.one_hot(y, depth=classes_num, dtype=tf.float32)
    return img, y_oh


@tf.function(reduce_retracing=True)
def _eval_map(img, y):
    y_oh = tf.one_hot(y, depth=classes_num, dtype=tf.float32)
    return img, y_oh


options = tf.data.Options()
options.experimental_deterministic = True
options.experimental_optimization.apply_default_optimizations = True
options.experimental_optimization.map_parallelization = True
options.experimental_optimization.map_and_batch_fusion = True

train_base = tf.data.Dataset.from_tensor_slices((train_paths, train_y))
train_base = train_base.map(
    lambda p, y: (_decode_resize(p), y), num_parallel_calls=AUTOTUNE
)
train_base = train_base.cache()

valid_base = tf.data.Dataset.from_tensor_slices((valid_paths, valid_y))
valid_base = valid_base.map(
    lambda p, y: (_decode_resize(p), y), num_parallel_calls=AUTOTUNE
)
valid_base = valid_base.cache()

train_n = len(train_paths)
shuffle_buf = min(train_n, 2048)

epoch_var = tf.Variable(0, dtype=tf.int32, trainable=False)


class _EpochSetter(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        epoch_var.assign(epoch)


epoch_setter = _EpochSetter()
callbacks = callbacks + [epoch_setter]

train_ds = train_base.shuffle(
    buffer_size=shuffle_buf, seed=SEED, reshuffle_each_iteration=True
)
train_ds = train_ds.enumerate()
train_ds = train_ds.map(
    lambda idx, xy: _augment(xy[0], xy[1], epoch_var, idx), num_parallel_calls=AUTOTUNE
)
train_ds = train_ds.batch(batch_size, drop_remainder=False)
train_ds = train_ds.prefetch(AUTOTUNE)
train_ds = train_ds.with_options(options)

valid_ds = valid_base.map(_eval_map, num_parallel_calls=AUTOTUNE)
valid_ds = valid_ds.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)
valid_ds = valid_ds.with_options(options)

steps_per_epoch = int(np.ceil(len(train_paths) / batch_size))
validation_steps = int(np.ceil(len(valid_paths) / batch_size))

print("tf.data train/valid ready:", steps_per_epoch, validation_steps)
print("Shuffle buffer:", shuffle_buf)




## === cell 18
history = model.fit(
    train_ds,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    validation_data=valid_ds,
    validation_steps=validation_steps,
    callbacks=callbacks,
    verbose=1,
)




## === cell 19
unfreeze_last_n = 60  # keep as-is (score-improving but core logic-preserving)
trainable_layers = 0

for lyr in model.layers[::-1]:
    if "inception_resnet_v2" in lyr.name.lower():
        if ("batch_normalization" in lyr.name.lower()) or ("bn" in lyr.name.lower()):
            lyr.trainable = False
            continue
        if trainable_layers < unfreeze_last_n:
            lyr.trainable = True
            trainable_layers += 1
        else:
            lyr.trainable = False

print("Unfroze backbone layers:", trainable_layers)

fine_tune_lr = learning_rate * 0.1
model.compile(
    optimizer=Adam(learning_rate=fine_tune_lr),
    loss="categorical_crossentropy",
    metrics=["acc"],
)

history_ft = model.fit(
    train_ds,
    steps_per_epoch=steps_per_epoch,
    epochs=max(1, epochs // 2),
    validation_data=valid_ds,
    validation_steps=validation_steps,
    callbacks=callbacks,
    verbose=1,
)




## === cell 20
one_hot_map = {c: i for i, c in enumerate(list(classes))}
print("First 10 class indices:", list(one_hot_map.items())[:10])




## === cell 21
sample_sub = pd.read_csv(sample_sub_path)
test_ids = sample_sub["id"].astype(str).tolist()  # required length/order

test_paths = [os.path.join(test_dir, f"{i}.jpg") for i in test_ids]
flow_ids = test_ids

test_base = (
    tf.data.Dataset.from_tensor_slices(test_paths)
    .map(_decode_resize, num_parallel_calls=AUTOTUNE)
    .cache()
)
test_ds = (
    test_base.batch(batch_size, drop_remainder=False)
    .prefetch(AUTOTUNE)
    .with_options(options)
)

print("Test samples:", len(test_paths))
print("Sample submission rows:", len(test_ids))




## === cell 22
preds = model.predict(
    test_ds,
    steps=int(np.ceil(len(test_paths) / batch_size)),
    verbose=1,
)

if preds.shape[1] != (sample_sub.shape[1] - 1):
    raise ValueError(
        f"Pred dim {preds.shape[1]} != submission classes {sample_sub.shape[1]-1}"
    )

eps = 1e-7
preds = np.clip(preds, eps, 1.0 - eps)
preds = preds / preds.sum(axis=1, keepdims=True)

pred_df = pd.DataFrame(preds, columns=sample_sub.columns[1:])
pred_df.insert(0, "id", flow_ids)

submission = pred_df
submission.columns = sample_sub.columns  # enforce exact header order

out_path = "submission.csv"
submission.to_csv(out_path, index=False)
print("Wrote:", out_path, "shape:", submission.shape)
print(submission.head())




## === cell 23
assert out_path.endswith(".csv")
assert submission.shape[0] == sample_sub.shape[0], "Submission row count mismatch"
assert list(submission.columns) == list(
    sample_sub.columns
), "Submission columns mismatch"
print("Submission validated.")
