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

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import Xception
from tensorflow.keras.optimizers import Adam

import warnings

warnings.simplefilter("ignore")

import cv2



## === cell 1
WORK_DIR = "../input/ranzcr-clip-catheter-line-classification"
os.listdir(WORK_DIR)



## === cell 2
print("Train images: %d" % len(os.listdir(os.path.join(WORK_DIR, "train"))))



## === cell 3
train = pd.read_csv(os.path.join(WORK_DIR, "train.csv"))
train_images = WORK_DIR + "/train/" + train["StudyInstanceUID"] + ".jpg"

ss = pd.read_csv(os.path.join(WORK_DIR, "sample_submission.csv"))
test_images = WORK_DIR + "/test/" + ss["StudyInstanceUID"] + ".jpg"

label_cols = ss.columns[1:]
labels = train[label_cols].values

train_annot = pd.read_csv(os.path.join(WORK_DIR, "train_annotations.csv"))

print("Labels:\n", "*" * 20, "\n", label_cols.values)
print("*" * 50)
train.head()



## === cell 4
sns.set_style("whitegrid")
fig = plt.figure(figsize=(15, 12), dpi=300)
plt.suptitle("Labels count", fontfamily="serif", size=15)

for ind, i in enumerate(label_cols):
    fig.add_subplot(4, 3, ind + 1)

    sns.countplot(
        train[i], edgecolor="black", palette=reversed(sns.color_palette("viridis", 2))
    )

    plt.xlabel("")
    plt.ylabel("")
    plt.xticks(fontfamily="serif", size=10)
    plt.yticks(fontfamily="serif", size=10)
    plt.title(i, fontfamily="serif", size=10)
plt.show()



## === cell 5
sample = train.sample(9)
plt.figure(figsize=(10, 7), dpi=300)
for ind, image_id in enumerate(sample.StudyInstanceUID):
    plt.subplot(3, 3, ind + 1)
    image = image_id + ".jpg"
    img = cv2.imread(os.path.join(WORK_DIR, "train", image))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.title("Shape: {}".format(img.shape[:2]))
    plt.axis("off")
plt.show()



## === cell 6
BATCH_SIZE = 8 * 1
STEPS_PER_EPOCH = len(train) * 0.85 / BATCH_SIZE
VALIDATION_STEPS = len(train) * 0.15 / BATCH_SIZE
EPOCHS = 5  # modest number to keep runtime low
TARGET_SIZE = 750




## === cell 7
def build_decoder(with_labels=True, target_size=(TARGET_SIZE, TARGET_SIZE), ext="jpg"):
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

    def decode_with_labels(path, label):
        return decode(path), label

    return decode_with_labels if with_labels else decode


def build_augmenter(with_labels=True):
    def augment(img):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        img = tf.image.adjust_brightness(img, 0.1)
        return img

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
    if cache_dir != "" and cache is True:
        os.makedirs(cache_dir, exist_ok=True)

    if decode_fn is None:
        decode_fn = build_decoder(labels is not None)

    if augment_fn is None:
        augment_fn = build_augmenter(labels is not None)

    AUTO = tf.data.experimental.AUTOTUNE
    slices = paths if labels is None else (paths, labels)

    dset = tf.data.Dataset.from_tensor_slices(slices)
    dset = dset.map(decode_fn, num_parallel_calls=AUTO)
    dset = dset.cache(cache_dir) if cache else dset
    dset = dset.map(augment_fn, num_parallel_calls=AUTO) if augment else dset
    dset = dset.repeat() if repeat else dset
    dset = dset.shuffle(shuffle) if shuffle else dset
    dset = dset.batch(bsize).prefetch(AUTO)

    return dset




## === cell 8
test_df = build_dataset(
    test_images,
    bsize=BATCH_SIZE,
    repeat=False,
    shuffle=False,
    augment=False,
    cache=False,
)

test_df



## === cell 9
model_path = "../input/ranzcr-xception-tpu-baseline/Xception_750_TPU.h5"
if os.path.exists(model_path):
    model = models.load_model(model_path)
else:
    base = Xception(
        weights="imagenet",
        include_top=False,
        input_shape=(TARGET_SIZE, TARGET_SIZE, 3),
        pooling="avg",
    )
    x = layers.Dense(1024, activation="relu")(base.output)
    output = layers.Dense(len(label_cols), activation="sigmoid")(x)
    model = models.Model(inputs=base.input, outputs=output)
    model.compile(optimizer=Adam(1e-4), loss="binary_crossentropy")
model.summary()



## === cell 10
print("Our Xception CNN has %d layers" % len(model.layers))



## === cell 11
train_idx, val_idx = train_test_split(
    np.arange(len(train)), test_size=0.15, random_state=42, stratify=labels
)
train_paths = train_images.iloc[train_idx].reset_index(drop=True)
val_paths = train_images.iloc[val_idx].reset_index(drop=True)
train_labels = labels[train_idx]
val_labels = labels[val_idx]

train_ds = build_dataset(
    train_paths,
    train_labels,
    bsize=BATCH_SIZE,
    augment=True,
    repeat=True,
    shuffle=1024,
    cache=False,
)
val_ds = build_dataset(
    val_paths,
    val_labels,
    bsize=BATCH_SIZE,
    augment=False,
    repeat=False,
    shuffle=False,
    cache=False,
)

ckpt_path = "best_model.h5"
checkpoint = ModelCheckpoint(
    ckpt_path, monitor="val_loss", save_best_only=True, verbose=0
)
early_stop = EarlyStopping(
    monitor="val_loss", patience=3, verbose=0, restore_best_weights=True
)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=0)

model.fit(
    train_ds,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_data=val_ds,
    validation_steps=VALIDATION_STEPS,
    epochs=EPOCHS,
    callbacks=[checkpoint, early_stop, reduce_lr],
    verbose=1,
)




## === cell 12
def activation_layer_vis(img, activation_layer=0, layers=10):
    layer_outputs = [layer.output for layer in model.layers[:layers]]
    activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
    activations = activation_model.predict(img)

    rows = int(activations[activation_layer].shape[3] / 3)
    cols = int(activations[activation_layer].shape[3] / rows)
    fig, axes = plt.subplots(rows, cols, figsize=(15, 15 * cols))
    axes = axes.flatten()

    for i, ax in zip(range(activations[activation_layer].shape[3]), axes):
        ax.matshow(activations[activation_layer][0, :, :, i], cmap="viridis")
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def all_activations_vis(img, layers=10):
    layer_outputs = [layer.output for layer in model.layers[:layers]]
    activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
    activations = activation_model.predict(img)

    layer_names = []
    for layer in model.layers[:layers]:
        layer_names.append(layer.name)

    images_per_row = 3
    for layer_name, layer_activation in zip(layer_names, activations):
        n_features = layer_activation.shape[-1]
        size = layer_activation.shape[1]
        n_cols = n_features // images_per_row
        display_grid = np.zeros((size * n_cols, images_per_row * size))

        for col in range(n_cols):
            for row in range(images_per_row):
                channel_image = layer_activation[0, :, :, col * images_per_row + row]
                channel_image -= channel_image.mean()
                channel_image /= channel_image.std()
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




## === cell 13
img_tensor = build_dataset(
    pd.Series(test_images[0]),
    bsize=1,
    repeat=False,
    shuffle=False,
    augment=False,
    cache=False,
)



## === cell 14
all_activations_vis(img_tensor, 3)



## === cell 15
ss[label_cols] = model.predict(test_df, verbose=1)
ss.to_csv("submission.csv", index=False)



## === cell 16
ss.head()
