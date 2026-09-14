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
import random
import warnings

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications import Xception
from tensorflow.keras.optimizers import Adam

warnings.simplefilter("ignore")

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

try:
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(False)
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

print("TF version:", tf.__version__)



## === cell 1
pass



## === cell 2
pass



## === cell 3
WORK_DIR = "../input/ranzcr-clip-catheter-line-classification"
print("WORK_DIR exists:", os.path.exists(WORK_DIR))
print("WORK_DIR listing (head):", sorted(os.listdir(WORK_DIR))[:10])



## === cell 4
pass



## === cell 5
train_dir = os.path.join(WORK_DIR, "train")
test_dir = os.path.join(WORK_DIR, "test")


def _count_jpgs_fast(d, max_check=50):
    n = 0
    with os.scandir(d) as it:
        for e in it:
            if e.is_file() and e.name.endswith(".jpg"):
                n += 1
            if n >= max_check:
                break
    return f">= {max_check}" if n >= max_check else str(n)


print("Train images (quick check):", _count_jpgs_fast(train_dir))
print("Test images (quick check):", _count_jpgs_fast(test_dir))



## === cell 6
train = pd.read_csv(os.path.join(WORK_DIR, "train.csv"))

ss_path = os.path.join(WORK_DIR, "sample_submission.csv")
ss_ref = pd.read_csv(ss_path)
if "StudyInstanceUID" not in ss_ref.columns:
    raise ValueError("sample_submission.csv missing StudyInstanceUID column")

label_cols = [c for c in ss_ref.columns if c != "StudyInstanceUID"]
missing = [c for c in label_cols if c not in train.columns]
if missing:
    raise ValueError(f"Expected label columns missing from train.csv: {missing}")

train_uids = train["StudyInstanceUID"].astype(str).tolist()
train_images = [os.path.join(WORK_DIR, "train", f"{uid}.jpg") for uid in train_uids]
labels = train[label_cols].values.astype(np.float32)

test_uids = ss_ref["StudyInstanceUID"].astype(str).copy()
test_images = [
    os.path.join(WORK_DIR, "test", f"{uid}.jpg") for uid in test_uids.tolist()
]

train_annot = pd.read_csv(os.path.join(WORK_DIR, "train_annotations.csv"))

print("Label columns used (from sample_submission):", label_cols)
print(
    "train:",
    train.shape,
    "test_uids:",
    test_uids.shape,
    "train_annot:",
    train_annot.shape,
)
train.head()



## === cell 7
SKIP_PLOTS = True
if not SKIP_PLOTS:
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_style("whitegrid")
    fig = plt.figure(figsize=(15, 12), dpi=150)
    plt.suptitle("Labels count", fontfamily="serif", size=15)

    for ind, i in enumerate(label_cols):
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

    plt.tight_layout()
    plt.show()



## === cell 8
if not SKIP_PLOTS:
    import matplotlib.pyplot as plt

    sample = train.sample(9, random_state=SEED)
    plt.figure(figsize=(10, 7), dpi=150)

    for ind, image_id in enumerate(sample.StudyInstanceUID):
        plt.subplot(3, 3, ind + 1)
        image_path = os.path.join(WORK_DIR, "train", f"{image_id}.jpg")
        img_bytes = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img_bytes, channels=3)
        img_np = img.numpy()
        plt.imshow(img_np)
        plt.title("Shape: {}".format(img_np.shape[:2]))
        plt.axis("off")

    plt.tight_layout()
    plt.show()



## === cell 9
pass



## === cell 10
BATCH_SIZE = 8 * 1
EPOCHS = 30
TARGET_SIZE = 750

STEPS_PER_EPOCH = int(np.ceil(len(train) * 0.8 / BATCH_SIZE))
VALIDATION_STEPS = int(np.ceil(len(train) * 0.2 / BATCH_SIZE))

print("BATCH_SIZE:", BATCH_SIZE)
print("STEPS_PER_EPOCH:", STEPS_PER_EPOCH)
print("VALIDATION_STEPS:", VALIDATION_STEPS)
print("EPOCHS:", EPOCHS, "TARGET_SIZE:", TARGET_SIZE)




## === cell 11
def build_decoder(with_labels=True, target_size=(TARGET_SIZE, TARGET_SIZE), ext="jpg"):
    target_size = tuple(target_size)

    def decode(path):
        file_bytes = tf.io.read_file(path)
        img = tf.io.decode_jpeg(file_bytes, channels=3, dct_method="INTEGER_FAST")
        img = tf.image.convert_image_dtype(img, tf.float32)  # cast/255.0
        img = tf.image.resize(
            img, target_size, method=tf.image.ResizeMethod.BILINEAR, antialias=False
        )
        img.set_shape([target_size[0], target_size[1], 3])
        return img

    def decode_with_labels(path, label):
        return decode(path), label

    return decode_with_labels if with_labels else decode


def build_augmenter(with_labels=True):
    def augment(img):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        img = tf.image.random_brightness(img, max_delta=0.1)
        img = tf.image.random_contrast(img, 0.9, 1.0)
        img = tf.image.random_saturation(img, 0.9, 1.0)
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
    deterministic=True,
):
    if decode_fn is None:
        decode_fn = build_decoder(labels is not None)

    if augment_fn is None:
        augment_fn = build_augmenter(labels is not None)

    AUTO = tf.data.AUTOTUNE

    paths = (
        tf.convert_to_tensor(paths, dtype=tf.string)
        if not tf.is_tensor(paths)
        else paths
    )

    if labels is None:
        dset = tf.data.Dataset.from_tensor_slices(paths)
    else:
        labels = tf.convert_to_tensor(labels, dtype=tf.float32)
        dset = tf.data.Dataset.from_tensor_slices((paths, labels))

    options = tf.data.Options()
    try:
        options.experimental_deterministic = bool(deterministic)
    except Exception:
        pass
    try:
        options.experimental_optimization.apply_default_optimizations = True
    except Exception:
        pass
    dset = dset.with_options(options)

    if shuffle:
        dset = dset.shuffle(shuffle, seed=SEED, reshuffle_each_iteration=True)
    if repeat:
        dset = dset.repeat()

    dset = dset.map(decode_fn, num_parallel_calls=AUTO)

    if cache:
        dset = dset.cache(cache_dir) if cache_dir else dset.cache()

    if augment:
        dset = dset.map(augment_fn, num_parallel_calls=AUTO)

    dset = dset.batch(bsize, drop_remainder=False)
    dset = dset.prefetch(AUTO)
    return dset




## === cell 12
DO_TRAIN = False

if DO_TRAIN:
    train_paths, valid_paths, train_y, valid_y = train_test_split(
        train_images, labels, test_size=0.2, random_state=SEED, shuffle=True
    )

    train_ds = build_dataset(
        train_paths,
        train_y,
        bsize=BATCH_SIZE,
        repeat=True,
        shuffle=1024,
        augment=True,
        cache=False,
        cache_dir="",
        deterministic=True,
    )

    valid_ds = build_dataset(
        valid_paths,
        valid_y,
        bsize=BATCH_SIZE,
        repeat=True,
        shuffle=False,
        augment=False,
        cache=False,
        cache_dir="",
        deterministic=True,
    )

    one_batch = next(iter(train_ds))
    print("Train batches example:", one_batch[0].shape, one_batch[1].shape)
else:
    train_ds = None
    valid_ds = None



## === cell 13
INFER_BATCH_SIZE = max(BATCH_SIZE, 32)

test_df = build_dataset(
    tf.convert_to_tensor(test_images, dtype=tf.string),
    labels=None,
    bsize=INFER_BATCH_SIZE,
    repeat=False,
    shuffle=False,
    augment=False,
    cache=False,
    cache_dir="",
    deterministic=False,
)
test_df




## === cell 14
def build_model(input_shape=(TARGET_SIZE, TARGET_SIZE, 3), n_classes=len(label_cols)):
    base = Xception(
        include_top=False, weights="imagenet", input_shape=input_shape, pooling="avg"
    )
    x = base.output
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(n_classes, activation="sigmoid")(x)
    m = models.Model(inputs=base.input, outputs=out)

    m.compile(optimizer=Adam(learning_rate=1e-4), loss="binary_crossentropy")
    return m


model = build_model()
model.summary()



## === cell 15
ckpt_path = "xception_750_best.keras"
callbacks = [
    ModelCheckpoint(
        ckpt_path, monitor="val_loss", save_best_only=True, save_weights_only=False
    ),
    ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=2, verbose=1, min_lr=1e-6
    ),
]

if DO_TRAIN:
    history = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=EPOCHS,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_steps=VALIDATION_STEPS,
        callbacks=callbacks,
        verbose=1,
    )
    model = models.load_model(ckpt_path)
else:
    history = None
    if os.path.exists(ckpt_path):
        model = models.load_model(ckpt_path)
        print(f"Loaded existing checkpoint: {ckpt_path}")
    else:
        print("No checkpoint found; using freshly initialized ImageNet Xception head.")



## === cell 16
print("Our Xception CNN has %d layers" % len(model.layers))



## === cell 17
pass




## === cell 18
def activation_layer_vis(img, activation_layer=0, layers_n=10):
    import matplotlib.pyplot as plt

    layer_outputs = [layer.output for layer in model.layers[:layers_n]]
    activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
    activations = activation_model.predict(img, verbose=0)

    rows = int(activations[activation_layer].shape[3] / 3)
    rows = max(rows, 1)
    cols = int(np.ceil(activations[activation_layer].shape[3] / rows))
    fig, axes = plt.subplots(rows, cols, figsize=(15, 15 * cols / max(rows, 1)))
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
    activations = activation_model.predict(img, verbose=0)

    layer_names = [layer.name for layer in model.layers[:layers_n]]

    images_per_row = 3
    for layer_name, layer_activation in zip(layer_names, activations):
        if len(layer_activation.shape) != 4:
            continue

        n_features = layer_activation.shape[-1]
        size = layer_activation.shape[1]
        n_cols = max(n_features // images_per_row, 1)
        display_grid = np.zeros(
            (size * n_cols, images_per_row * size), dtype=np.float32
        )

        for col in range(n_cols):
            for row in range(images_per_row):
                idx = col * images_per_row + row
                if idx >= n_features:
                    continue
                channel_image = layer_activation[0, :, :, idx]
                channel_image -= channel_image.mean()
                std = channel_image.std() + 1e-6
                channel_image /= std
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype("uint8")
                display_grid[
                    col * size : (col + 1) * size, row * size : (row + 1) * size
                ] = channel_image

        scale = 1.0 / max(size, 1)
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




## === cell 19
pass



## === cell 20
if not SKIP_PLOTS:
    img_tensor = build_dataset(
        pd.Series([test_images[0]]),
        labels=None,
        bsize=1,
        repeat=False,
        shuffle=False,
        augment=False,
        cache=False,
        deterministic=True,
    )



## === cell 21
if not SKIP_PLOTS:
    all_activations_vis(img_tensor, layers_n=3)



## === cell 22
pass



## === cell 23
n_test = len(test_images)
steps = int(np.ceil(n_test / INFER_BATCH_SIZE))

preds = model.predict(test_df, steps=steps, verbose=0)
preds = preds[:n_test]  # safety: ensure exact length if any extra batching occurs
preds = np.clip(preds, 0.0, 1.0)

submission = ss_ref.copy()
submission[label_cols] = preds

submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print("Submission columns:", submission.columns.tolist())



## === cell 24
submission.head()
