# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Classify each cassava image into four disease categories or a fifth category indicating a healthy leaf.

## Metric
Categorization accuracy.

## Submission Format
```
image_id,label
1000471002.jpg,4
1000840542.jpg,4
etc.
```

## Dataset
**[train/test]_images** the image files.

**train.csv**

- `image_id` the image file name.

- `label` the ID code for the disease.

**sample_submission.csv** A properly formatted sample submission, given the disclosed test set content.

- `image_id` the image file name.

- `label` the predicted ID code for the disease.

**[train/test]_tfrecords** the image files in tfrecord format.

**label_num_to_disease_map.json** The mapping between each disease code and the real disease name.

# 2. Python version

3.9

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        input/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        working/
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
```

-> data/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/cassava-leaf-disease-classification/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/cassava-leaf-disease-classification/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> data/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> input/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os, json, warnings, random

warnings.simplefilter("ignore")

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.optimizers import Adam

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

print("TensorFlow:", tf.__version__)
print("Num GPUs Available:", len(tf.config.list_physical_devices("GPU")))

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

tf.config.threading.set_intra_op_parallelism_threads(0)
tf.config.threading.set_inter_op_parallelism_threads(0)

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

tf.keras.backend.clear_session()



## === cell 1
WORK_DIR = "../input/cassava-leaf-disease-classification"

print("Files in WORK_DIR:", os.listdir(WORK_DIR)[:10])

with open(os.path.join(WORK_DIR, "label_num_to_disease_map.json")) as file:
    print(json.dumps(json.loads(file.read()), indent=4))

train_labels = pd.read_csv(os.path.join(WORK_DIR, "train.csv"))
train_labels.head()



## === cell 2
BATCH_SIZE = 8
EPOCHS = 20
TARGET_SIZE = 350

TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.2

STEPS_PER_EPOCH = int(np.ceil(len(train_labels) * TRAIN_SPLIT / BATCH_SIZE))
VALIDATION_STEPS = int(np.ceil(len(train_labels) * VAL_SPLIT / BATCH_SIZE))

print("STEPS_PER_EPOCH:", STEPS_PER_EPOCH, "VALIDATION_STEPS:", VALIDATION_STEPS)



## === cell 3
from pathlib import Path

base_path = Path(WORK_DIR)
train_img_dir = base_path / "train_images"
test_img_dir = base_path / "test_images"

train_tfrec_dir = base_path / "train_tfrecords"
test_tfrec_dir = base_path / "test_tfrecords"



## === cell 4
train_df = train_labels.copy()
diseaseMapping = pd.read_json(base_path / "label_num_to_disease_map.json", typ="series")

print(train_df.shape)



## === cell 5
mappingDict = diseaseMapping.to_dict()
mappingDict



## === cell 6
train_df.head()



## === cell 7
train_df_explore = None
labelCounts = None
healthyImages = cbbImages = cbsdImages = cgmImages = cmdImages = None
print(
    "Exploratory EDA (replace/value_counts/class image lists/plots) skipped to meet 600s runtime."
)




## === cell 8
def showImages(images):
    if plt is None:
        print("matplotlib not available; skipping showImages.")
        return
    if len(images) == 0:
        print("No images provided.")
        return

    random_images = [np.random.choice(images) for _ in range(9)]
    plt.figure(figsize=(10, 8))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        img = plt.imread(str(train_img_dir / random_images[i]))
        plt.imshow(img)
        plt.axis("off")
    plt.tight_layout()




## === cell 9
def showHistogram(sample_img, title):
    if plt is None:
        print("matplotlib not available; skipping showHistogram.")
        return

    raw_image = plt.imread(str(train_img_dir / sample_img))
    f = plt.figure(figsize=(16, 8))
    f.add_subplot(1, 2, 1)
    plt.imshow(raw_image)
    plt.colorbar()
    plt.title(title)
    print(f"Image dimensions: {(raw_image.shape[0], raw_image.shape[1])}")
    print(
        f"Maximum pixel value : {raw_image.max():.1f} ; Minimum pixel value:{raw_image.min():.1f}"
    )
    print(
        f"Mean value of the pixels : {raw_image.mean():.1f} ; Standard deviation : {raw_image.std():.1f}"
    )

    f.add_subplot(1, 2, 2)
    _ = plt.hist(raw_image[:, :, 0].ravel(), bins=256, color="red", alpha=0.5)
    _ = plt.hist(raw_image[:, :, 1].ravel(), bins=256, color="green", alpha=0.5)
    _ = plt.hist(raw_image[:, :, 2].ravel(), bins=256, color="blue", alpha=0.5)
    _ = plt.xlabel("Intensity Value")
    _ = plt.ylabel("Count")
    _ = plt.legend(["Red_Channel", "Green_Channel", "Blue_Channel"])
    plt.show()




## === cell 10
import cv2


def load_image(image_id):
    image = cv2.imread(str(train_img_dir / image_id))
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def showChannelDistribution(images, leafType):
    print(
        "Channel distribution visualization skipped (plotly not used in this runtime-stable version)."
    )
    return None




## === cell 11
def showBoxPlot(histData, leafType):
    print(
        "Box plot visualization skipped (plotly not used in this runtime-stable version)."
    )




## === cell 12
channelIntensityDf = pd.DataFrame(
    {
        "Leaf Type": ["Healthy", "CBB", "CBSD", "CGM", "CMD"],
        "Red Channel Mean": [108, 102, 106, 113, 110],
        "Green Channel Mean": [126, 117, 123, 128, 128],
        "Blue Channel Mean": [80, 66, 72, 85, 80],
    }
)
channelIntensityDf



## === cell 13
train_labels = train_labels.copy()
train_labels["label"] = train_labels["label"].astype(str)
train_labels.head()



## === cell 14
AUTOTUNE = tf.data.AUTOTUNE

idx = np.arange(len(train_labels))
rng = np.random.RandomState(SEED)
rng.shuffle(idx)
split = int((1.0 - VAL_SPLIT) * len(idx))
train_idx = idx[:split]
val_idx = idx[split:]

train_df_split = train_labels.iloc[train_idx].reset_index(drop=True)
val_df_split = train_labels.iloc[val_idx].reset_index(drop=True)

train_y = train_df_split["label"].astype(np.int32).values
val_y = val_df_split["label"].astype(np.int32).values

train_x = (train_img_dir.as_posix() + "/" + train_df_split["image_id"].values).astype(
    "U"
)
val_x = (train_img_dir.as_posix() + "/" + val_df_split["image_id"].values).astype("U")

data_augmentation = tf.keras.Sequential(
    [
        layers.RandomFlip(mode="horizontal_and_vertical", seed=SEED),
        layers.RandomRotation(factor=45.0 / 360.0, fill_mode="nearest", seed=SEED),
        layers.RandomZoom(
            height_factor=(-0.2, 0.2),
            width_factor=(-0.2, 0.2),
            fill_mode="nearest",
            seed=SEED,
        ),
        layers.RandomTranslation(
            height_factor=0.1, width_factor=0.1, fill_mode="nearest", seed=SEED
        ),
        layers.RandomShear(x_factor=0.1, y_factor=0.1, fill_mode="nearest", seed=SEED),
    ],
    name="augmentation",
)

TFREC_FEATURES = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
}


@tf.function
def _decode_resize_from_bytes(img_bytes):
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(
        img, [TARGET_SIZE, TARGET_SIZE], method=tf.image.ResizeMethod.BILINEAR
    )
    img = tf.cast(img, tf.float32)
    img.set_shape([TARGET_SIZE, TARGET_SIZE, 3])
    return img


@tf.function
def _parse_train_tfrecord(example_proto):
    x = tf.io.parse_single_example(example_proto, TFREC_FEATURES)
    img = _decode_resize_from_bytes(x["image"])
    label = tf.cast(x["target"], tf.int32)
    return img, label


@tf.function
def _parse_test_tfrecord(example_proto):
    x = tf.io.parse_single_example(example_proto, TFREC_FEATURES)
    img = _decode_resize_from_bytes(x["image"])
    return img


@tf.function
def _augment(img, label):
    img = data_augmentation(img, training=True)
    return img, label


@tf.function
def _identity(img, label):
    return img, label


def _list_tfrec_files(tfrec_dir: Path, prefix: str):
    files = tf.io.gfile.glob((tfrec_dir / f"{prefix}*.tfrec").as_posix())
    files = sorted(files)
    if not files:
        raise FileNotFoundError(
            f"No TFRecord files found in {tfrec_dir} with {prefix}*.tfrec"
        )
    return files


def _make_cache_path(name: str):
    return os.path.join("/kaggle/working", name)


def make_trainval_ds_from_tfrecords(train_files, val_files):
    options = tf.data.Options()
    options.experimental_deterministic = True

    def make(files, training, cache_path):
        ds = tf.data.TFRecordDataset(files, num_parallel_reads=AUTOTUNE)
        ds = ds.with_options(options)
        ds = ds.map(_parse_train_tfrecord, num_parallel_calls=AUTOTUNE)
        ds = ds.cache(cache_path)
        if training:
            ds = ds.shuffle(buffer_size=8192, seed=SEED, reshuffle_each_iteration=True)
            ds = ds.map(_augment, num_parallel_calls=AUTOTUNE)
        else:
            ds = ds.map(_identity, num_parallel_calls=AUTOTUNE)
        ds = ds.batch(BATCH_SIZE, drop_remainder=False)
        ds = ds.prefetch(AUTOTUNE)
        return ds

    train_ds = make(train_files, True, _make_cache_path("train_cache.tf-data"))
    val_ds = make(val_files, False, _make_cache_path("val_cache.tf-data"))
    return train_ds, val_ds


all_train_tfrecs = _list_tfrec_files(train_tfrec_dir, "ld_train")
rng2 = np.random.RandomState(SEED)
perm = np.arange(len(all_train_tfrecs))
rng2.shuffle(perm)
all_train_tfrecs = [all_train_tfrecs[i] for i in perm]
split_files = max(1, int(round((1.0 - VAL_SPLIT) * len(all_train_tfrecs))))
train_tfrecs = all_train_tfrecs[:split_files]
val_tfrecs = all_train_tfrecs[split_files:]

train_dataset, val_dataset = make_trainval_ds_from_tfrecords(train_tfrecs, val_tfrecs)

print("Train/Val sizes (csv rows):", len(train_df_split), len(val_df_split))
print("TFRecord files train/val:", len(train_tfrecs), len(val_tfrecs))




## === cell 15
def create_model():
    conv_base = EfficientNetB3(
        include_top=False, weights=None, input_shape=(TARGET_SIZE, TARGET_SIZE, 3)
    )
    x = conv_base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(5, activation="softmax")(x)
    model = models.Model(conv_base.input, x)

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["acc"],
    )
    return model




## === cell 16
model = create_model()
model.summary()

ckpt_path = "EffNetB3_TARGET350_BS8_best.keras"
callbacks = [
    ModelCheckpoint(
        ckpt_path, monitor="val_acc", save_best_only=True, mode="max", verbose=1
    ),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1),
]

history = model.fit(
    train_dataset,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_data=val_dataset,
    validation_steps=VALIDATION_STEPS,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1,
)

best_model = tf.keras.models.load_model(ckpt_path)
model = best_model



## === cell 17
pass



## === cell 18
print("Our EfficientNet CNN has %d layers" % len(model.layers))



## === cell 19
ss = pd.read_csv(os.path.join(WORK_DIR, "sample_submission.csv"))
ss.head()



## === cell 20
test_tfrecs = _list_tfrec_files(test_tfrec_dir, "ld_test")

test_options = tf.data.Options()
test_options.experimental_deterministic = True

test_dataset = tf.data.TFRecordDataset(test_tfrecs, num_parallel_reads=AUTOTUNE)
test_dataset = test_dataset.with_options(test_options)
test_dataset = test_dataset.map(_parse_test_tfrecord, num_parallel_calls=AUTOTUNE)
test_dataset = test_dataset.cache(_make_cache_path("test_cache.tf-data"))
test_dataset = test_dataset.batch(BATCH_SIZE, drop_remainder=False)
test_dataset = test_dataset.prefetch(AUTOTUNE)



## === cell 21
pred_probs = model.predict(test_dataset, verbose=1)
preds = np.argmax(pred_probs, axis=1).astype(int).tolist()

print("Predictions:", len(preds), "Expected:", len(ss))



## === cell 22
pred_probs[0]



## === cell 23
print("First 20 predicted labels:", preds[:20])



## === cell 24
submission = ss.copy()
submission["label"] = preds
submission.head()



## === cell 25
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.dtypes)



## === cell 26
assert list(submission.columns) == ["image_id", "label"]
assert submission["image_id"].nunique() == len(submission)
assert len(submission) == len(ss)
print("Submission format looks valid.")
