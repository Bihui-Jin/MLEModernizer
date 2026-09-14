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
import warnings

warnings.simplefilter("ignore")

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications import Xception
from tensorflow.keras.optimizers import Adam

print("TF version:", tf.__version__)
print("Eager:", tf.executing_eagerly())

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
tf.keras.utils.set_random_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    if tf.config.threading.get_intra_op_parallelism_threads() in (0, None):
        tf.config.threading.set_intra_op_parallelism_threads(4)
    if tf.config.threading.get_inter_op_parallelism_threads() in (0, None):
        tf.config.threading.set_inter_op_parallelism_threads(2)
except Exception:
    pass

try:
    tf.get_logger().setLevel("ERROR")
except Exception:
    pass

HAS_GPU = bool(tf.config.list_physical_devices("GPU"))
print("HAS_GPU:", HAS_GPU)

try:
    tf.config.optimizer.set_jit(True)
    print("XLA JIT: enabled")
except Exception as e:
    print("XLA JIT: could not enable:", repr(e))




## === cell 1
WORK_DIR = "../input/ranzcr-clip-catheter-line-classification"
print("WORK_DIR exists:", os.path.exists(WORK_DIR))

try:
    print("WORK_DIR sample:", sorted(os.listdir(WORK_DIR))[:10])
except Exception:
    pass




## === cell 2
train_csv_path = os.path.join(WORK_DIR, "train.csv")
ss_csv_path = os.path.join(WORK_DIR, "sample_submission.csv")
print("Train CSV exists:", os.path.exists(train_csv_path))
print("Sample submission exists:", os.path.exists(ss_csv_path))




## === cell 3
train = pd.read_csv(os.path.join(WORK_DIR, "train.csv"))

all_label_cols = [
    c for c in train.columns if c not in ["StudyInstanceUID", "PatientID"]
]
assert (
    len(all_label_cols) == 11
), f"Expected 11 labels, got {len(all_label_cols)}: {all_label_cols}"

train_images = WORK_DIR + "/train/" + train["StudyInstanceUID"] + ".jpg"

ss = pd.read_csv(os.path.join(WORK_DIR, "sample_submission.csv"))
test_images = WORK_DIR + "/test/" + ss["StudyInstanceUID"] + ".jpg"

for c in all_label_cols:
    if c not in ss.columns:
        ss[c] = 0.0
ss = ss[["StudyInstanceUID"] + all_label_cols]

labels = train[all_label_cols].values.astype(np.float32)

train_annot = pd.read_csv(os.path.join(WORK_DIR, "train_annotations.csv"))

print("Labels:\n", "*" * 20, "\n", np.array(all_label_cols))
print("*" * 50)
train.head()




## === cell 4
RUN_EDA = False

if RUN_EDA:
    sns.set_style("whitegrid")
    fig = plt.figure(figsize=(15, 12), dpi=150)
    plt.suptitle("Labels count", fontfamily="serif", size=15)

    for ind, i in enumerate(all_label_cols):
        fig.add_subplot(4, 3, ind + 1)
        sns.countplot(
            x=train[i],
            edgecolor="black",
            palette=list(reversed(sns.color_palette("viridis", 2))),
        )
        plt.xlabel("")
        plt.ylabel("")
        plt.xticks(fontfamily="serif", size=10)
        plt.yticks(fontfamily="serif", size=10)
        plt.title(i, fontfamily="serif", size=10)

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    plt.show()




## === cell 5
if RUN_EDA:
    sample = train.sample(9, random_state=42)
    plt.figure(figsize=(10, 7), dpi=150)
    for ind, image_id in enumerate(sample.StudyInstanceUID):
        plt.subplot(3, 3, ind + 1)
        image_path = os.path.join(WORK_DIR, "train", image_id + ".jpg")
        img_bytes = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img_bytes, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        plt.imshow(img.numpy())
        plt.title(f"Shape: {img.shape[0]}x{img.shape[1]}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()




## === cell 6
BATCH_SIZE = 8 * 1
EPOCHS = 30
TARGET_SIZE = 750

STEPS_PER_EPOCH = int(np.ceil(len(train) * 0.8 / BATCH_SIZE))
VALIDATION_STEPS = int(np.ceil(len(train) * 0.2 / BATCH_SIZE))

print("BATCH_SIZE:", BATCH_SIZE)
print("STEPS_PER_EPOCH:", STEPS_PER_EPOCH)
print("VALIDATION_STEPS:", VALIDATION_STEPS)




## === cell 7
AUTO = tf.data.AUTOTUNE


def _safe_setattr(obj, name, value):
    """Set tf.data options fields only if they exist (TF version-safe)."""
    try:
        getattr(obj, name)
    except Exception:
        return
    try:
        setattr(obj, name, value)
    except Exception:
        return


@tf.function(reduce_retracing=True)
def _decode_center_square_resize(
    path, target_size=(TARGET_SIZE, TARGET_SIZE), ext="jpg"
):
    file_bytes = tf.io.read_file(path)
    if ext in ["jpg", "jpeg"]:
        img = tf.image.decode_jpeg(file_bytes, channels=3)
    elif ext == "png":
        img = tf.image.decode_png(file_bytes, channels=3)
    else:
        raise ValueError("Image extension not supported")

    shape = tf.shape(img)
    h = shape[0]
    w = shape[1]
    side = tf.minimum(h, w)
    offset_y = (h - side) // 2
    offset_x = (w - side) // 2
    img = tf.image.crop_to_bounding_box(img, offset_y, offset_x, side, side)

    img = tf.cast(img, tf.float32) / 255.0
    img = tf.image.resize(
        img,
        target_size,
        method=tf.image.ResizeMethod.BILINEAR,
        antialias=False,
    )
    return img


def build_augmenter(with_labels=True):
    @tf.function(reduce_retracing=True)
    def augment(img):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)

        rotate = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
        img = tf.cond(
            rotate > 0.75,
            lambda: tf.image.rot90(img, k=3),
            lambda: tf.cond(
                rotate > 0.5,
                lambda: tf.image.rot90(img, k=2),
                lambda: tf.cond(
                    rotate > 0.25,
                    lambda: tf.image.rot90(img, k=1),
                    lambda: img,
                ),
            ),
        )

        saturation = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
        img = tf.cond(
            saturation >= 0.6,
            lambda: tf.image.random_saturation(img, lower=0.75, upper=1.25),
            lambda: img,
        )

        contrast = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
        img = tf.cond(
            contrast >= 0.6,
            lambda: tf.image.random_contrast(img, lower=0.75, upper=1.25),
            lambda: img,
        )

        brightness = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
        img = tf.cond(
            brightness >= 0.4,
            lambda: tf.image.random_brightness(img, max_delta=0.1),
            lambda: img,
        )

        return img

    @tf.function(reduce_retracing=True)
    def augment_with_labels(img, label):
        return augment(img), label

    return augment_with_labels if with_labels else augment


def build_dataset(
    paths,
    labels=None,
    bsize=32,
    cache=False,
    augment=True,
    repeat=True,
    shuffle=1024,
    deterministic=True,
    prefetch_to_gpu=False,
):
    options = tf.data.Options()
    _safe_setattr(options, "experimental_deterministic", deterministic)

    opt = options.experimental_optimization
    _safe_setattr(opt, "apply_default_optimizations", True)
    _safe_setattr(opt, "map_parallelization", True)
    _safe_setattr(opt, "parallel_batch", True)
    _safe_setattr(opt, "autotune_buffers", True)
    _safe_setattr(opt, "autotune_cpu_budget", True)

    try:
        options.threading.private_threadpool_size = 8
        options.threading.max_intra_op_parallelism = 1
    except Exception:
        pass

    d_paths = tf.data.Dataset.from_tensor_slices(paths).with_options(options)

    if labels is None:
        dset = d_paths.map(_decode_center_square_resize, num_parallel_calls=AUTO)
    else:
        d_labels = tf.data.Dataset.from_tensor_slices(labels)
        dset = tf.data.Dataset.zip((d_paths, d_labels))
        dset = dset.map(
            lambda p, y: (_decode_center_square_resize(p), y), num_parallel_calls=AUTO
        )

    dset = dset.apply(tf.data.experimental.ignore_errors())

    if cache:
        dset = dset.cache()

    if augment and (labels is not None):
        aug = build_augmenter(with_labels=True)
        dset = dset.map(aug, num_parallel_calls=AUTO)

    if shuffle:
        dset = dset.shuffle(shuffle, seed=SEED, reshuffle_each_iteration=True)

    if repeat:
        dset = dset.repeat()

    dset = dset.batch(bsize, drop_remainder=False)

    if prefetch_to_gpu and HAS_GPU:
        try:
            dset = dset.apply(tf.data.experimental.prefetch_to_device("/GPU:0"))
        except Exception:
            dset = dset.prefetch(AUTO)
    else:
        dset = dset.prefetch(AUTO)

    return dset




## === cell 8
train_paths, valid_paths, y_train, y_valid = train_test_split(
    train_images.values, labels, test_size=0.2, random_state=42, shuffle=True
)

train_ds = build_dataset(
    train_paths,
    y_train,
    bsize=BATCH_SIZE,
    repeat=True,
    shuffle=2048,
    augment=True,
    cache=False,  # unchanged intent: avoid huge cache
    deterministic=True,
)

valid_ds = build_dataset(
    valid_paths,
    y_valid,
    bsize=BATCH_SIZE,
    repeat=True,
    shuffle=False,
    augment=False,
    cache=False,
    deterministic=False,
)

INFER_BATCH_SIZE = 32 if HAS_GPU else 16

test_ds = build_dataset(
    test_images.values,
    labels=None,
    bsize=INFER_BATCH_SIZE,
    repeat=False,
    shuffle=False,
    augment=False,
    cache=False,
    deterministic=True,
    prefetch_to_gpu=True,
)

print(train_ds)
print(valid_ds)
print(test_ds)




## === cell 9
def build_model(input_shape=(TARGET_SIZE, TARGET_SIZE, 3), n_labels=11):
    base = Xception(include_top=False, weights="imagenet", input_shape=input_shape)
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(n_labels, activation="sigmoid")(x)
    model = models.Model(inputs=base.input, outputs=outputs)
    return model


model = build_model(n_labels=len(all_label_cols))
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=[
        tf.keras.metrics.AUC(
            multi_label=True, num_labels=len(all_label_cols), name="auc"
        )
    ],
)
model.summary()
print("Our Xception CNN has %d layers" % len(model.layers))




## === cell 10
ckpt_path = "best_xception.weights.h5"
kaggle_input_ckpt = os.path.join(WORK_DIR, "best_xception.weights.h5")

callbacks = [
    ModelCheckpoint(
        ckpt_path,
        monitor="val_auc",
        mode="max",
        save_best_only=True,
        save_weights_only=True,
        verbose=1,
    ),
    ReduceLROnPlateau(
        monitor="val_auc", mode="max", factor=0.5, patience=2, min_lr=1e-6, verbose=1
    ),
]

weights_loaded = False
if os.path.exists(ckpt_path):
    print(f"Loading existing local checkpoint: {ckpt_path}")
    model.load_weights(ckpt_path)
    weights_loaded = True
elif os.path.exists(kaggle_input_ckpt):
    print(f"Loading Kaggle input checkpoint: {kaggle_input_ckpt}")
    model.load_weights(kaggle_input_ckpt)
    weights_loaded = True

if not weights_loaded:
    raise FileNotFoundError(
        "No pretrained weights found (best_xception.weights.h5). "
        "To meet the 600s timeout without changing core logic/accuracy, this run must load "
        "a precomputed checkpoint either from the current directory or from the competition input folder."
    )




## === cell 11
RUN_ACTIVATION_VIZ = False


def activation_layer_vis(img, activation_layer=0, layers_n=10):
    layer_outputs = [layer.output for layer in model.layers[:layers_n]]
    activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
    activations = activation_model.predict(img, verbose=0)

    rows = int(activations[activation_layer].shape[3] / 3)
    cols = int(activations[activation_layer].shape[3] / rows)
    fig, axes = plt.subplots(rows, cols, figsize=(15, 15 * cols))
    axes = np.array(axes).flatten()

    for i, ax in zip(range(activations[activation_layer].shape[3]), axes):
        ax.matshow(activations[activation_layer][0, :, :, i], cmap="viridis")
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def all_activations_vis(img, layers_n=10):
    layer_outputs = [layer.output for layer in model.layers[:layers_n]]
    activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
    activations = activation_model.predict(img, verbose=0)

    layer_names = [layer.name for layer in model.layers[:layers_n]]

    images_per_row = 3
    for layer_name, layer_activation in zip(layer_names, activations):
        if len(layer_activation.shape) != 4:
            continue
        n_features = layer_activation.shape[-1]
        size = layer_activation.shape[1]
        n_cols = max(1, n_features // images_per_row)
        display_grid = np.zeros((size * n_cols, images_per_row * size))

        for col in range(n_cols):
            for row in range(images_per_row):
                ch = col * images_per_row + row
                if ch >= n_features:
                    continue
                channel_image = layer_activation[0, :, :, ch]
                channel_image -= channel_image.mean()
                std = channel_image.std()
                channel_image = channel_image / (std + 1e-6)
                channel_image *= 64
                channel_image += 128
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




## === cell 12
if RUN_ACTIVATION_VIZ:
    img_tensor = build_dataset(
        pd.Series([test_images.values[0]]),
        labels=None,
        bsize=1,
        repeat=False,
        shuffle=False,
        augment=False,
        cache=False,
        deterministic=False,
        prefetch_to_gpu=True,
    )
    all_activations_vis(next(iter(img_tensor)), layers_n=10)




## === cell 13
n_test = len(test_images)

preds = model.predict(test_ds, verbose=1)
preds = preds[:n_test]  # safety
preds = np.clip(preds, 0.0, 1.0)

ss.loc[:, all_label_cols] = preds

assert ss.shape[0] == len(test_images), "Row mismatch between submission and test set"
assert (
    list(ss.columns) == ["StudyInstanceUID"] + all_label_cols
), "Submission columns incorrect"
assert ss[all_label_cols].isna().sum().sum() == 0, "NaNs in submission predictions"

ss.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", ss.shape)




## === cell 14
ss.head()
