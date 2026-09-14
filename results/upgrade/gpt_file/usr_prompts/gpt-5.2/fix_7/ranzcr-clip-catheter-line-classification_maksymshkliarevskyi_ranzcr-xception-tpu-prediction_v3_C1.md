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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.applications import Xception
from tensorflow.keras.optimizers import Adam

tf.get_logger().setLevel("ERROR")
print("TensorFlow:", tf.__version__)
print("Num GPUs:", len(tf.config.list_physical_devices("GPU")))

try:
    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(
        False
    )  # keep default math; avoid any potential XLA-induced numeric diffs
except Exception:
    pass

try:
    cpu = os.cpu_count() or 2
    tf.config.threading.set_intra_op_parallelism_threads(max(1, cpu // 2))
    tf.config.threading.set_inter_op_parallelism_threads(max(1, cpu // 2))
except Exception:
    pass




## === cell 1
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)




## === cell 2
WORK_DIR = "../input/ranzcr-clip-catheter-line-classification"
os.listdir(WORK_DIR)[:10]




## === cell 3
train_dir = os.path.join(WORK_DIR, "train")
test_dir = os.path.join(WORK_DIR, "test")

print("Train dir:", train_dir)
print("Test dir:", test_dir)




## === cell 4
train = pd.read_csv(os.path.join(WORK_DIR, "train.csv"))
ss = pd.read_csv(os.path.join(WORK_DIR, "sample_submission.csv"))

target_cols = [c for c in train.columns if c not in ["StudyInstanceUID", "PatientID"]]
label_cols = target_cols  # keep the original variable name used downstream

train_images = (WORK_DIR + "/train/" + train["StudyInstanceUID"] + ".jpg").astype(str)
test_images = (WORK_DIR + "/test/" + ss["StudyInstanceUID"] + ".jpg").astype(str)

labels = train[label_cols].values.astype(np.float32)

train_annot = pd.read_csv(os.path.join(WORK_DIR, "train_annotations.csv"))

print("Num train:", len(train), "Num test:", len(ss))
print("Labels:\n", "*" * 20, "\n", np.array(label_cols))
print("*" * 50)
train.head()




## === cell 5
if False:
    sns.set_style("whitegrid")
    fig = plt.figure(figsize=(15, 12), dpi=200)
    plt.suptitle("Labels count", fontfamily="serif", size=15)

    for ind, i in enumerate(label_cols):
        fig.add_subplot(4, 3, ind + 1)
        sns.countplot(
            x=train[i],
            edgecolor="black",
            palette=reversed(sns.color_palette("viridis", 2)),
        )
        plt.xlabel("")
        plt.ylabel("")
        plt.xticks(fontfamily="serif", size=9)
        plt.yticks(fontfamily="serif", size=9)
        plt.title(i, fontfamily="serif", size=9)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()




## === cell 6
if False:
    sample = train.sample(9, random_state=SEED)
    plt.figure(figsize=(10, 7), dpi=200)

    for ind, image_id in enumerate(sample.StudyInstanceUID):
        plt.subplot(3, 3, ind + 1)
        image_path = os.path.join(WORK_DIR, "train", f"{image_id}.jpg")
        img = plt.imread(image_path)
        plt.imshow(img)
        plt.title(f"Shape: {img.shape[:2]}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()




## === cell 7
BATCH_SIZE = 8 * 1
EPOCHS = 30
TARGET_SIZE = 750

TRAIN_FRACTION = 0.8




## === cell 8
def build_decoder(with_labels=True, target_size=(TARGET_SIZE, TARGET_SIZE), ext="jpg"):
    @tf.function(reduce_retracing=True)
    def decode(path):
        file_bytes = tf.io.read_file(path)
        if ext == "png":
            img = tf.image.decode_png(file_bytes, channels=3)
        elif ext in ["jpg", "jpeg"]:
            img = tf.image.decode_jpeg(file_bytes, channels=3)
        else:
            raise ValueError("Image extension not supported")

        img = tf.cast(img, tf.float32) / 255.0
        img = tf.image.resize(img, target_size)
        return img

    @tf.function(reduce_retracing=True)
    def decode_with_labels(path, label):
        return decode(path), label

    return decode_with_labels if with_labels else decode


def build_augmenter(with_labels=True):
    @tf.function(reduce_retracing=True)
    def augment(img):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        img = tf.image.adjust_brightness(img, 0.1)
        return img

    @tf.function(reduce_retracing=True)
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
    drop_remainder=False,
):
    AUTO = tf.data.AUTOTUNE

    options = tf.data.Options()
    options.deterministic = True
    options.experimental_optimization.apply_default_optimizations = True
    options.experimental_optimization.map_parallelization = True
    options.experimental_optimization.parallel_batch = True

    with_labels = labels is not None

    if decode_fn is None:
        decode_fn = build_decoder(
            with_labels=with_labels, target_size=(TARGET_SIZE, TARGET_SIZE), ext="jpg"
        )

    if with_labels:
        dset = tf.data.Dataset.from_tensor_slices((paths, labels))
        dset = dset.with_options(options)
        dset = dset.map(decode_fn, num_parallel_calls=AUTO)
    else:
        dset = tf.data.Dataset.from_tensor_slices(paths)
        dset = dset.with_options(options)
        dset = dset.map(decode_fn, num_parallel_calls=AUTO)

    if cache and not augment:
        dset = dset.cache()

    if augment:
        if augment_fn is None:
            augment_fn = build_augmenter(with_labels)
        dset = dset.map(augment_fn, num_parallel_calls=AUTO)

    if repeat:
        dset = dset.repeat()
    if shuffle:
        dset = dset.shuffle(shuffle, seed=SEED, reshuffle_each_iteration=True)

    dset = dset.batch(bsize, drop_remainder=drop_remainder)
    dset = dset.prefetch(AUTO)
    return dset




## === cell 9
unique_patients = train["PatientID"].unique()
train_pats, val_pats = train_test_split(
    unique_patients, test_size=0.2, random_state=SEED
)

trn_idx = train["PatientID"].isin(train_pats)
val_idx = train["PatientID"].isin(val_pats)

train_paths = train_images[trn_idx].reset_index(drop=True)
val_paths = train_images[val_idx].reset_index(drop=True)

train_labels = labels[trn_idx]
val_labels = labels[val_idx]

print("Train samples:", len(train_paths), "Val samples:", len(val_paths))

train_df = build_dataset(
    train_paths.values,
    train_labels,
    bsize=BATCH_SIZE,
    repeat=True,
    shuffle=2048,
    augment=True,
    cache=False,  # keep semantics: no in-memory cache for augmented training stream
    cache_dir="",
    drop_remainder=True,  # training uses fixed batch shapes
)
val_df = build_dataset(
    val_paths.values,
    val_labels,
    bsize=BATCH_SIZE,
    repeat=False,
    shuffle=False,
    augment=False,
    cache=True,  # keep semantics
    cache_dir="",
    drop_remainder=False,
)

test_df = build_dataset(
    test_images.values,
    bsize=BATCH_SIZE,
    repeat=False,
    shuffle=False,
    augment=False,
    cache=True,
    cache_dir="",
    drop_remainder=False,
)

steps_per_epoch = max(1, len(train_paths) // BATCH_SIZE)
validation_steps = int(np.ceil(len(val_paths) / BATCH_SIZE))

steps_per_epoch, validation_steps




## === cell 10
def build_model(input_shape=(TARGET_SIZE, TARGET_SIZE, 3), n_classes=len(label_cols)):
    base = Xception(
        include_top=False, weights="imagenet", input_shape=input_shape, pooling=None
    )

    base.trainable = False

    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(n_classes, activation="sigmoid")(x)
    model = models.Model(inputs=base.input, outputs=out)
    return model


model = build_model()
model.compile(
    optimizer=Adam(learning_rate=1e-4), loss="binary_crossentropy", metrics=[]
)
print("Our Xception CNN has %d layers" % len(model.layers))
model.summary()




## === cell 11
ckpt_path = "xception_ranzcr_best.keras"
callbacks = [
    ModelCheckpoint(
        ckpt_path,
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=False,
        verbose=1,
    ),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1),
    EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True, verbose=1),
]

history = model.fit(
    train_df,
    validation_data=val_df,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1,
)




## === cell 12
if os.path.exists(ckpt_path):
    model = models.load_model(ckpt_path, compile=False)
    model.compile(optimizer=Adam(learning_rate=1e-4), loss="binary_crossentropy")
print("Model ready for inference.")




## === cell 13
def activation_layer_vis(img, activation_layer=0, layers_n=10):
    layer_outputs = [layer.output for layer in model.layers[:layers_n]]
    activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
    activations = activation_model.predict(img, verbose=0)

    rows = int(activations[activation_layer].shape[3] / 3)
    rows = max(rows, 1)
    cols = int(np.ceil(activations[activation_layer].shape[3] / rows))
    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
    axes = np.array(axes).reshape(-1)

    for i in range(min(activations[activation_layer].shape[3], len(axes))):
        axes[i].matshow(activations[activation_layer][0, :, :, i], cmap="viridis")
        axes[i].axis("off")
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

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
                idx = col * images_per_row + row
                if idx >= n_features:
                    continue
                channel_image = layer_activation[0, :, :, idx]
                std = channel_image.std()
                if std > 1e-6:
                    channel_image = (channel_image - channel_image.mean()) / std
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




## === cell 14
if False:
    img_tensor = build_dataset(
        pd.Series([test_images.iloc[0]]).values,
        bsize=1,
        repeat=False,
        shuffle=False,
        augment=False,
        cache=False,
    )
    all_activations_vis(img_tensor, layers_n=3)




## === cell 15
test_steps = int(np.ceil(len(ss) / BATCH_SIZE))

preds = model.predict(test_df, steps=test_steps, verbose=1)
preds = preds[: len(ss)]
preds = np.clip(preds, 0.0, 1.0)

submission = pd.DataFrame({"StudyInstanceUID": ss["StudyInstanceUID"].values})
for i, c in enumerate(label_cols):
    submission[c] = preds[:, i]

submission = submission[["StudyInstanceUID"] + label_cols]
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
submission.head()




## === cell 16
assert submission.shape[0] == ss.shape[0]
assert submission.isna().sum().sum() == 0
assert submission.columns.tolist()[0] == "StudyInstanceUID"
print("Submission columns:", submission.columns.tolist())
