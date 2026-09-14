# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Detect the presence and position of catheters and lines on chest x-rays.

## Metric
Area under the ROC curve for each label, with the final score being the average of the individual AUCs of each predicted column.

## Submission Format
For each ID in the test set, you must predict a probability for all target variables. The file should contain a header and have the following format:
```
StudyInstanceUID,ETT - Abnormal,ETT - Borderline,ETT - Normal,NGT - Abnormal,NGT - Borderline,NGT - Incompletely Imaged,NGT - Normal,CVC - Abnormal,CVC - Borderline,CVC - Normal,Swan Ganz Catheter Present
1.2.826.0.1.3680043.8.498.62451881164053375557257228990443168843,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.83721761279899623084220697845011427274,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.12732270010839808189235995393981377825,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.11769539755086084996287023095028033598,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.87838627504097587943394933987052577153,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.53211840524738036417560823327351887819,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.93555795394184819372299157360228027866,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.52241894131170494723503100795076463919,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.36500167484503936720548852591033878284,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.86199852603457900780565655267977637728,0,0,0,0,0,0,0,0,0,0,0
```

## Dataset
`train.csv` contains image IDs, binary labels, and patient IDs.

TFRecords are available for both train and test.

We've also included `train_annotations.csv`. These are segmentation annotations for training samples that have them. They are included solely as additional information for competitors.

- train.csv - contains image IDs, binary labels, and patient IDs.
- sample_submission.csv - a sample submission file in the correct format
- test - test images
- train - training images

### Columns
- `StudyInstanceUID` - unique ID for each image
- `ETT - Abnormal` - endotracheal tube placement abnormal
- `ETT - Borderline` - endotracheal tube placement borderline abnormal
- `ETT - Normal` - endotracheal tube placement normal
- `NGT - Abnormal` - nasogastric tube placement abnormal
- `NGT - Borderline` - nasogastric tube placement borderline abnormal
- `NGT - Incompletely Imaged` - nasogastric tube placement inconclusive due to imaging
- `NGT - Normal` - nasogastric tube placement borderline normal
- `CVC - Abnormal` - central venous catheter placement abnormal
- `CVC - Borderline` - central venous catheter placement borderline abnormal
- `CVC - Normal` - central venous catheter placement normal
- `Swan Ganz Catheter Present`
- `PatientID` - unique ID for each patient in the dataset

# 2. Python version

3.9

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (172 lines)
            sample_submission.csv (3010 lines)
            sample_submission.csv.zip (64.2 kB)
            test.zip (642.8 MB)
            train.csv (27075 lines)
            train.csv.zip (798.6 kB)
            train.zip (5.8 GB)
            train_annotations.csv (16262 lines)
            train_annotations.csv.zip (1.4 MB)
            ranzcr-clip-catheter-line-classification/
                description.md (172 lines)
                sample_submission.csv (3010 lines)
                ... and 7 other files
                ranzcr-clip-catheter-line-classification/
                test/
                    1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                    1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                    ... and 3007 other files
                    test/
                train/
                    1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                    1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                    ... and 27072 other files
                    train/
            test/
                1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                ... and 3007 other files
                test/
            train/
                1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                ... and 27072 other files
                train/
        input/
            description.md (172 lines)
            sample_submission.csv (3010 lines)
            sample_submission.csv.zip (64.2 kB)
            test.zip (642.8 MB)
            train.csv (27075 lines)
            train.csv.zip (798.6 kB)
            train.zip (5.8 GB)
            train_annotations.csv (16262 lines)
            train_annotations.csv.zip (1.4 MB)
            ranzcr-clip-catheter-line-classification/
                description.md (172 lines)
                sample_submission.csv (3010 lines)
                ... and 7 other files
                ranzcr-clip-catheter-line-classification/
                test/
                    1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                    1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                    ... and 3007 other files
                    test/
                train/
                    1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                    1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                    ... and 27072 other files
                    train/
            test/
                1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                ... and 3007 other files
                test/
                    1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                    1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                    ... and 3007 other files
                    test/
            train/
                1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                ... and 27072 other files
                train/
                    1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                    1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                    ... and 27072 other files
                    train/
        working/
            ranzcr-clip-catheter-line-classification/
                description.md (172 lines)
                sample_submission.csv (3010 lines)
                ... and 7 other files
                ranzcr-clip-catheter-line-classification/
                test/
                    1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                    1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                    ... and 3007 other files
                    test/
                train/
                    1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                    1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                    ... and 27072 other files
                    train/
```

-> data/ranzcr-clip-catheter-line-classification/sample_submission.csv has 3009 rows and 10 columns.
The columns are: StudyInstanceUID, ETT - Abnormal, ETT - Borderline, ETT - Normal, NGT - Abnormal, NGT - Borderline, NGT - Incompletely Imaged, NGT - Normal, CVC - Abnormal, CVC - Borderline

-> data/ranzcr-clip-catheter-line-classification/train.csv has 27074 rows and 13 columns.
The columns are: StudyInstanceUID, ETT - Abnormal, ETT - Borderline, ETT - Normal, NGT - Abnormal, NGT - Borderline, NGT - Incompletely Imaged, NGT - Normal, CVC - Abnormal, CVC - Borderline, CVC - Normal, Swan Ganz Catheter Present, PatientID

-> data/ranzcr-clip-catheter-line-classification/train_annotations.csv has 16261 rows and 3 columns.
The columns are: StudyInstanceUID, label, data

-> data/sample_submission.csv has 3009 rows and 10 columns.
The columns are: StudyInstanceUID, ETT - Abnormal, ETT - Borderline, ETT - Normal, NGT - Abnormal, NGT - Borderline, NGT - Incompletely Imaged, NGT - Normal, CVC - Abnormal, CVC - Borderline

-> data/train.csv has 27074 rows and 13 columns.
The columns are: StudyInstanceUID, ETT - Abnormal, ETT - Borderline, ETT - Normal, NGT - Abnormal, NGT - Borderline, NGT - Incompletely Imaged, NGT - Normal, CVC - Abnormal, CVC - Borderline, CVC - Normal, Swan Ganz Catheter Present, PatientID

-> data/train_annotations.csv has 16261 rows and 3 columns.
The columns are: StudyInstanceUID, label, data

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications import Xception
from tensorflow.keras.optimizers import Adam

import warnings

warnings.simplefilter("ignore")

tf.keras.utils.set_random_seed(42)
try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

try:
    for _gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(_gpu, True)
except Exception:
    pass

print("TF version:", tf.__version__)
print("Num GPUs:", len(tf.config.list_physical_devices("GPU")))



## === cell 1
pass



## === cell 2
pass



## === cell 3
WORK_DIR = "../input/ranzcr-clip-catheter-line-classification"
os.listdir(WORK_DIR)[:10]



## === cell 4
pass



## === cell 5
train_dir = os.path.join(WORK_DIR, "train")
test_dir = os.path.join(WORK_DIR, "test")
print("Train dir exists:", os.path.isdir(train_dir))
print("Test dir exists:", os.path.isdir(test_dir))



## === cell 6
train = pd.read_csv(os.path.join(WORK_DIR, "train.csv"))
train_images = WORK_DIR + "/train/" + train["StudyInstanceUID"] + ".jpg"

ss = pd.read_csv(os.path.join(WORK_DIR, "sample_submission.csv"))
test_images = WORK_DIR + "/test/" + ss["StudyInstanceUID"] + ".jpg"

label_cols = ss.columns[1:]
labels = train[label_cols].values.astype(np.float32)

train_annot = pd.read_csv(os.path.join(WORK_DIR, "train_annotations.csv"))

print("Train images (from CSV): %d" % len(train))
print("Test images (from sample_submission): %d" % len(ss))
print("Labels:\n", "*" * 20, "\n", label_cols.values)
print("*" * 50)
train.head()



## === cell 7
pass



## === cell 8
pass



## === cell 9
pass



## === cell 10
BATCH_SIZE = 8 * 1
EPOCHS = 30
TARGET_SIZE = 750

print("BATCH_SIZE:", BATCH_SIZE)
print("EPOCHS:", EPOCHS)
print("TARGET_SIZE:", TARGET_SIZE)




## === cell 11
def build_decoder(with_labels=True, target_size=(TARGET_SIZE, TARGET_SIZE), ext="jpg"):
    target_size = tuple(target_size)

    @tf.function
    def decode(path):
        file_bytes = tf.io.read_file(path)
        if ext == "png":
            img = tf.image.decode_png(file_bytes, channels=3)
        elif ext in ["jpg", "jpeg"]:
            img = tf.io.decode_jpeg(file_bytes, channels=3)
        else:
            raise ValueError("Image extension not supported")

        img = tf.image.convert_image_dtype(img, tf.float32)  # == cast/255.0
        img = tf.image.resize(
            img, target_size, method=tf.image.ResizeMethod.BILINEAR, antialias=False
        )
        return img

    @tf.function
    def decode_with_labels(path, label):
        return decode(path), label

    return decode_with_labels if with_labels else decode


def build_augmenter(with_labels=True):
    @tf.function
    def augment(img):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        img = tf.image.adjust_brightness(img, 0.1)
        return img

    @tf.function
    def augment_with_labels(img, label):
        return augment(img), label

    return augment_with_labels if with_labels else augment


def build_dataset(
    paths,
    labels=None,
    bsize=32,
    cache=True,
    decode_fn=None,
    augment_fn=None,
    augment=True,
    repeat=True,
    shuffle=1024,
    cache_dir="",
):
    AUTO = tf.data.AUTOTUNE

    if decode_fn is None:
        decode_fn = build_decoder(labels is not None)

    if augment_fn is None:
        augment_fn = build_augmenter(labels is not None)

    if isinstance(paths, (pd.Series, pd.Index)):
        paths = paths.values
    if labels is not None and isinstance(labels, (pd.Series, pd.Index)):
        labels = labels.values

    if labels is None:
        dset = tf.data.Dataset.from_tensor_slices(paths)
        dset = dset.map(decode_fn, num_parallel_calls=AUTO)
    else:
        dset = tf.data.Dataset.from_tensor_slices((paths, labels))
        dset = dset.map(decode_fn, num_parallel_calls=AUTO)

    options = tf.data.Options()
    options.experimental_deterministic = True
    try:
        options.experimental_optimization.apply_default_optimizations = True
        options.experimental_optimization.map_parallelization = True
        options.experimental_optimization.parallel_batch = True
        options.experimental_slack = True
        options.threading.private_threadpool_size = 8
        options.threading.max_intra_op_parallelism = 0
    except Exception:
        pass
    dset = dset.with_options(options)

    dset = dset.apply(tf.data.experimental.ignore_errors())

    if cache:
        if cache_dir:
            os.makedirs(os.path.dirname(cache_dir), exist_ok=True)
            dset = dset.cache(cache_dir)
        else:
            dset = dset.cache()

    if augment:
        dset = dset.map(augment_fn, num_parallel_calls=AUTO)

    if shuffle:
        dset = dset.shuffle(shuffle, reshuffle_each_iteration=True)

    dset = dset.repeat() if repeat else dset
    dset = dset.batch(bsize, drop_remainder=False)

    dset = dset.prefetch(AUTO)
    return dset




## === cell 12
train_paths, val_paths, y_train, y_val = train_test_split(
    train_images.values, labels, test_size=0.15, random_state=42, shuffle=True
)

train_cache_path = os.path.join("/kaggle/working", f"cache_train_{TARGET_SIZE}.tfdata")
val_cache_path = os.path.join("/kaggle/working", f"cache_val_{TARGET_SIZE}.tfdata")
test_cache_path = os.path.join("/kaggle/working", f"cache_test_{TARGET_SIZE}.tfdata")

train_ds = build_dataset(
    train_paths,
    y_train,
    bsize=BATCH_SIZE,
    repeat=False,
    shuffle=2048,
    augment=True,
    cache=True,
    cache_dir=train_cache_path,
)

val_ds = build_dataset(
    val_paths,
    y_val,
    bsize=BATCH_SIZE,
    repeat=False,
    shuffle=False,
    augment=False,
    cache=True,
    cache_dir=val_cache_path,
)

steps_per_epoch = int(np.ceil(len(train_paths) / BATCH_SIZE))
validation_steps = int(np.ceil(len(val_paths) / BATCH_SIZE))

print("Train size:", len(train_paths), "Val size:", len(val_paths))
print("steps_per_epoch:", steps_per_epoch, "validation_steps:", validation_steps)



## === cell 13
test_df = build_dataset(
    test_images.values,
    bsize=BATCH_SIZE,
    repeat=False,
    shuffle=False,
    augment=False,
    cache=True,
    cache_dir=test_cache_path,
)

test_df



## === cell 14
inp = layers.Input(shape=(TARGET_SIZE, TARGET_SIZE, 3))
base = Xception(include_top=False, weights="imagenet", input_tensor=inp, pooling="avg")

base.trainable = False

x = base.output
x = layers.Dropout(0.2)(x)
out = layers.Dense(len(label_cols), activation="sigmoid")(x)
model = models.Model(inputs=inp, outputs=out)

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    steps_per_execution=16,
)

model.summary()



## === cell 15
ckpt_path = "xception_best.keras"
callbacks = [
    ModelCheckpoint(
        ckpt_path, monitor="val_loss", save_best_only=True, save_weights_only=False
    ),
    ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=2, verbose=1, min_lr=1e-6
    ),
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
)

model = models.load_model(ckpt_path)



## === cell 16
print("Our Xception CNN has %d layers" % len(model.layers))



## === cell 17
pass




## === cell 18
def activation_layer_vis(img, activation_layer=0, layers_n=10):
    import matplotlib.pyplot as plt

    layer_outputs = [layer.output for layer in model.layers[:layers_n]]
    activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
    activations = activation_model.predict(img)

    rows = int(activations[activation_layer].shape[3] / 3)
    rows = max(rows, 1)
    cols = int(np.ceil(activations[activation_layer].shape[3] / rows))
    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
    axes = np.array(axes).reshape(-1)

    for i, ax in zip(range(activations[activation_layer].shape[3]), axes):
        ax.matshow(activations[activation_layer][0, :, :, i], cmap="viridis")
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def all_activations_vis(img, layers_n=10):
    import matplotlib.pyplot as plt

    layer_outputs = [layer.output for layer in model.layers[:layers_n]]
    activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
    activations = activation_model.predict(img)

    layer_names = [layer.name for layer in model.layers[:layers_n]]

    images_per_row = 3
    for layer_name, layer_activation in zip(layer_names, activations):
        n_features = layer_activation.shape[-1]
        size = layer_activation.shape[1]
        n_cols = max(1, n_features // images_per_row)

        display_grid = np.zeros((size * n_cols, images_per_row * size))

        for col in range(n_cols):
            for row in range(images_per_row):
                idx = col * images_per_row + row
                if idx >= n_features:
                    continue
                channel_image = layer_activation[0, :, :, idx]
                channel_image = channel_image - channel_image.mean()
                std = channel_image.std()
                if std > 1e-6:
                    channel_image = channel_image / std
                channel_image = channel_image * 64 + 128
                channel_image = np.clip(channel_image, 0, 255).astype("uint8")
                display_grid[
                    col * size : (col + 1) * size, row * size : (row + 1) * size
                ] = channel_image

        scale = 1.0 / size
        plt.figure(
            figsize=(
                scale * 5 * display_grid.shape[1],
                scale * 5 * display_grid.shape[0],
            )
        )
        plt.title(layer_name)
        plt.grid(False)
        plt.axis("off")
        plt.imshow(display_grid, aspect="auto", cmap="viridis")
        plt.show()




## === cell 19
pass



## === cell 20
pass



## === cell 21
pass



## === cell 22
pred = model.predict(test_df, verbose=1)
pred = np.clip(pred, 0.0, 1.0)

ss.loc[:, label_cols] = pred
ss.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", ss.shape)
print("Columns:", list(ss.columns))



## === cell 23
ss.head()
