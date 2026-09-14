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
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

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

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

AUTO = tf.data.AUTOTUNE

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

tf.keras.backend.clear_session()

try:
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception:
    pass



## === cell 1
pass



## === cell 2
WORK_DIR = "../input/ranzcr-clip-catheter-line-classification"
_ = os.path.exists(WORK_DIR)



## === cell 3
pass



## === cell 4
train_dir = os.path.join(WORK_DIR, "train")
try:
    n_train_imgs = len([f for f in os.scandir(train_dir) if f.is_file()])
except Exception:
    n_train_imgs = -1
print("Train images: %d" % n_train_imgs)



## === cell 5
train = pd.read_csv(os.path.join(WORK_DIR, "train.csv"))
train_images = WORK_DIR + "/train/" + train["StudyInstanceUID"] + ".jpg"

ss = pd.read_csv(os.path.join(WORK_DIR, "sample_submission.csv"))
test_images = WORK_DIR + "/test/" + ss["StudyInstanceUID"] + ".jpg"

label_cols = [c for c in train.columns if c not in ["StudyInstanceUID", "PatientID"]]
labels = train[label_cols].values.astype(np.float32)

print("Labels:\n", "*" * 20, "\n", np.array(label_cols))
print("*" * 50)
train.head()



## === cell 6
pass



## === cell 7
pass



## === cell 8
pass



## === cell 9
BATCH_SIZE = 8 * 1
EPOCHS = 30
TARGET_SIZE = 750




## === cell 10
def build_decoder(with_labels=True, target_size=(TARGET_SIZE, TARGET_SIZE), ext="jpg"):
    @tf.function
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

    @tf.function
    def decode_with_labels(path, label):
        return decode(path), label

    return decode_with_labels if with_labels else decode


def build_augmenter(with_labels=True):
    @tf.function
    def augment(img):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        return img

    @tf.function
    def augment_with_labels(img, label):
        return augment(img), label

    return augment_with_labels if with_labels else augment


def build_dataset(
    paths,
    labels=None,
    bsize=32,
    cache=False,
    decode_fn=None,
    augment_fn=None,
    augment=True,
    repeat=True,
    shuffle=1024,
    cache_dir="",
    drop_remainder=None,
):
    """
    Speed changes (semantics preserved):
    - Cache for val/test (no augmentation) to avoid repeated JPEG decode/resize.
    - Keep repeat() before shuffle() to avoid reinitialization overhead; randomness preserved
      via reshuffle_each_iteration=True and fixed seed.
    - Use ignore_errors() to prevent rare decode errors from stalling.
    - Enable tf.data autotune/optimizations while keeping determinism.
    """
    if decode_fn is None:
        decode_fn = build_decoder(labels is not None)

    if augment_fn is None:
        augment_fn = build_augmenter(labels is not None)

    slices = paths if labels is None else (paths, labels)
    dset = tf.data.Dataset.from_tensor_slices(slices)

    dset = dset.map(decode_fn, num_parallel_calls=AUTO, deterministic=True)
    dset = dset.apply(tf.data.experimental.ignore_errors())

    if cache:
        if cache_dir:
            tf.io.gfile.makedirs(os.path.dirname(cache_dir))
            dset = dset.cache(cache_dir)
        else:
            dset = dset.cache()

    if repeat:
        dset = dset.repeat()

    if shuffle:
        dset = dset.shuffle(shuffle, seed=SEED, reshuffle_each_iteration=True)

    if augment:
        dset = dset.map(augment_fn, num_parallel_calls=AUTO, deterministic=True)

    if drop_remainder is None:
        drop_remainder = bool(repeat)

    dset = dset.batch(bsize, drop_remainder=drop_remainder)

    opts = tf.data.Options()
    opts.experimental_deterministic = True
    try:
        opts.experimental_optimization.apply_default_optimizations = True
        opts.experimental_optimization.autotune_buffers = True
        opts.experimental_optimization.autotune_cpu_budget = 0
        opts.threading.private_threadpool_size = 0
        opts.threading.max_intra_op_parallelism = 0
    except Exception:
        pass
    dset = dset.with_options(opts)

    dset = dset.prefetch(AUTO)
    return dset




## === cell 11
tr_paths, va_paths, tr_y, va_y = train_test_split(
    train_images.values, labels, test_size=0.2, random_state=SEED, shuffle=True
)

STEPS_PER_EPOCH = len(tr_paths) // BATCH_SIZE
VALIDATION_STEPS = int(np.ceil(len(va_paths) / BATCH_SIZE))

train_df = build_dataset(
    tr_paths,
    tr_y,
    bsize=BATCH_SIZE,
    repeat=True,
    shuffle=2048,
    augment=True,
    cache=False,  # caching augmented stream doesn't help and can blow memory
    cache_dir="",
    drop_remainder=True,
)

valid_df = build_dataset(
    va_paths,
    va_y,
    bsize=BATCH_SIZE,
    repeat=False,
    shuffle=False,
    augment=False,
    cache=True,
    cache_dir="",
    drop_remainder=False,
)



## === cell 12
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
test_df



## === cell 13
pass



## === cell 14
inp = layers.Input(shape=(TARGET_SIZE, TARGET_SIZE, 3))
base = Xception(include_top=False, weights="imagenet", input_tensor=inp)
x = layers.GlobalAveragePooling2D()(base.output)
out = layers.Dense(len(label_cols), activation="sigmoid")(x)
model = models.Model(inputs=inp, outputs=out)

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    steps_per_execution=16,
)
model.summary()



## === cell 15
print("Our Xception CNN has %d layers" % len(model.layers))



## === cell 16
ckpt_path = "xception_local_best.keras"

callbacks = [
    ModelCheckpoint(
        ckpt_path,
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=False,
        verbose=1,
    ),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1),
]

if os.path.exists(ckpt_path):
    model = models.load_model(ckpt_path)
else:
    history = model.fit(
        train_df,
        validation_data=valid_df,
        epochs=EPOCHS,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_steps=VALIDATION_STEPS,
        callbacks=callbacks,
        verbose=1,
    )
    if os.path.exists(ckpt_path):
        model = models.load_model(ckpt_path)




## === cell 17
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

    for i in range(activations[activation_layer].shape[3]):
        ax = axes[i]
        ax.matshow(activations[activation_layer][0, :, :, i], cmap="viridis")
        ax.axis("off")
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
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
        n_cols = max(n_features // images_per_row, 1)
        display_grid = np.zeros((size * n_cols, images_per_row * size))

        for col in range(n_cols):
            for row in range(images_per_row):
                idx = col * images_per_row + row
                if idx >= n_features:
                    continue
                channel_image = layer_activation[0, :, :, idx]
                channel_image -= channel_image.mean()
                std = channel_image.std()
                if std > 0:
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
        plt.show()




## === cell 18
pass



## === cell 19
pass



## === cell 20
pass



## === cell 21
pass



## === cell 22
preds = model.predict(test_df, verbose=1)
preds = np.clip(preds, 0.0, 1.0)

submission = pd.DataFrame({"StudyInstanceUID": ss["StudyInstanceUID"].values})
for i, col in enumerate(label_cols):
    submission[col] = preds[:, i].astype(np.float32)

submission = submission[["StudyInstanceUID"] + label_cols]
submission.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", submission.shape)
print("Columns:", submission.columns.tolist())



## === cell 23
submission.head()
