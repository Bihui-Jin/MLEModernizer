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
import math
import warnings

warnings.simplefilter("ignore")

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=2")

import numpy as np
import pandas as pd

import cv2

import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.applications import Xception
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

tf.config.run_functions_eagerly(False)

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

try:
    tf.data.experimental.enable_debug_mode(False)
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

try:
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception:
    pass




## === cell 1
print("TF version:", tf.__version__)
print("Eager:", tf.executing_eagerly())




## === cell 2
WORK_DIR = "../input/ranzcr-clip-catheter-line-classification"
print("WORK_DIR exists:", os.path.exists(WORK_DIR))
try:
    print("WORK_DIR sample:", os.listdir(WORK_DIR)[:10])
except Exception as e:
    print("Could not list WORK_DIR:", e)




## === cell 3
print("Train dir exists:", os.path.exists(os.path.join(WORK_DIR, "train")))
print("Test dir exists:", os.path.exists(os.path.join(WORK_DIR, "test")))




## === cell 4
def fast_count_files(path):
    try:
        with os.scandir(path) as it:
            return sum(1 for _ in it)
    except Exception:
        return -1


print("Train images: %d" % fast_count_files(os.path.join(WORK_DIR, "train")))
print("Test images: %d" % fast_count_files(os.path.join(WORK_DIR, "test")))




## === cell 5
train = pd.read_csv(os.path.join(WORK_DIR, "train.csv"))
ss = pd.read_csv(os.path.join(WORK_DIR, "sample_submission.csv"))

label_cols = [c for c in train.columns if c not in ["StudyInstanceUID", "PatientID"]]
print("Number of label columns:", len(label_cols))
print("Labels:", label_cols)

train_images = (WORK_DIR + "/train/" + train["StudyInstanceUID"] + ".jpg").astype(str)
test_images = (WORK_DIR + "/test/" + ss["StudyInstanceUID"] + ".jpg").astype(str)

labels = train[label_cols].values.astype(np.float32)

train.head()




## === cell 6
DO_PLOTS = False
if DO_PLOTS:
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




## === cell 7
DO_SAMPLE_IMAGES = False
if DO_SAMPLE_IMAGES:
    import matplotlib.pyplot as plt

    sample = train.sample(9, random_state=SEED)
    plt.figure(figsize=(10, 7), dpi=150)
    for ind, image_id in enumerate(sample.StudyInstanceUID):
        plt.subplot(3, 3, ind + 1)
        image = image_id + ".jpg"
        img = cv2.imread(os.path.join(WORK_DIR, "train", image))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.title("Shape: {}".format(img.shape[:2]))
        plt.axis("off")
    plt.tight_layout()
    plt.show()




## === cell 8
BATCH_SIZE = 32  # was 8
EPOCHS = 3  # keep same training loop count

TARGET_SIZE = 299

STEPS_PER_EPOCH = None
VALIDATION_STEPS = None

print("BATCH_SIZE:", BATCH_SIZE)
print("TARGET_SIZE:", TARGET_SIZE)
print("EPOCHS:", EPOCHS)
print("STEPS_PER_EPOCH:", STEPS_PER_EPOCH)
print("VALIDATION_STEPS:", VALIDATION_STEPS)




## === cell 9
def build_decoder(with_labels=True, target_size=(TARGET_SIZE, TARGET_SIZE), ext="jpg"):
    target_h, target_w = int(target_size[0]), int(target_size[1])

    def decode(path):
        file_bytes = tf.io.read_file(path)
        if ext == "png":
            img = tf.image.decode_png(file_bytes, channels=3)
        elif ext in ["jpg", "jpeg"]:
            img = tf.image.decode_jpeg(file_bytes, channels=3)
        else:
            raise ValueError("Image extension not supported")

        img = tf.image.convert_image_dtype(img, dtype=tf.float32)
        img = tf.image.resize(
            img,
            [target_h, target_w],
            method=tf.image.ResizeMethod.BILINEAR,
            antialias=False,
        )
        img.set_shape([target_h, target_w, 3])
        return img

    def decode_with_labels(path, label):
        return decode(path), label

    return decode_with_labels if with_labels else decode


def build_augmenter(with_labels=True):
    @tf.function
    def augment(img):
        img = tf.image.random_flip_left_right(img, seed=SEED)
        img = tf.image.random_flip_up_down(img, seed=SEED)
        img = tf.image.adjust_brightness(img, 0.1)
        return img

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
):
    if decode_fn is None:
        decode_fn = build_decoder(labels is not None)

    if augment_fn is None:
        augment_fn = build_augmenter(labels is not None)

    AUTO = tf.data.AUTOTUNE
    slices = paths if labels is None else (paths, labels)

    dset = tf.data.Dataset.from_tensor_slices(slices)

    options = tf.data.Options()
    options.experimental_optimization.apply_default_optimizations = True
    options.experimental_optimization.map_parallelization = True
    options.experimental_optimization.parallel_batch = True
    try:
        options.experimental_slack = True
    except Exception:
        pass
    dset = dset.with_options(options)

    if shuffle:
        dset = dset.shuffle(shuffle, seed=SEED, reshuffle_each_iteration=True)

    dset = dset.map(decode_fn, num_parallel_calls=AUTO)

    if cache:
        if cache_dir:
            os.makedirs(os.path.dirname(cache_dir), exist_ok=True)
            dset = dset.cache(cache_dir)
        else:
            dset = dset.cache()

    if augment:
        dset = dset.map(augment_fn, num_parallel_calls=AUTO)

    if repeat:
        dset = dset.repeat()

    dset = dset.batch(bsize, drop_remainder=False)
    dset = dset.prefetch(AUTO)

    return dset




## === cell 10
idx = np.arange(len(train))
train_idx, valid_idx = train_test_split(
    idx, test_size=0.15, random_state=SEED, shuffle=True
)

train_paths = train_images.iloc[train_idx].values
valid_paths = train_images.iloc[valid_idx].values
train_labels = labels[train_idx]
valid_labels = labels[valid_idx]

CACHE_ROOT = "/kaggle/working/tfdata_cache"
train_cache = os.path.join(CACHE_ROOT, f"train_{TARGET_SIZE}.cache")
valid_cache = os.path.join(CACHE_ROOT, f"valid_{TARGET_SIZE}.cache")
test_cache = os.path.join(CACHE_ROOT, f"test_{TARGET_SIZE}.cache")

train_ds = build_dataset(
    train_paths,
    train_labels,
    bsize=BATCH_SIZE,
    cache=False,
    cache_dir=train_cache,
    augment=True,
    repeat=False,
    shuffle=2048,
)
valid_ds = build_dataset(
    valid_paths,
    valid_labels,
    bsize=BATCH_SIZE,
    cache=False,
    cache_dir=valid_cache,
    augment=False,
    repeat=False,
    shuffle=False,
)

test_ds = build_dataset(
    test_images.values,
    labels=None,
    bsize=BATCH_SIZE,
    repeat=False,
    shuffle=False,
    augment=False,
    cache=False,
    cache_dir=test_cache,
)

STEPS_PER_EPOCH = int(math.ceil(len(train_paths) / BATCH_SIZE))
VALIDATION_STEPS = int(math.ceil(len(valid_paths) / BATCH_SIZE))

print("Datasets ready.")
print("Computed STEPS_PER_EPOCH:", STEPS_PER_EPOCH)
print("Computed VALIDATION_STEPS:", VALIDATION_STEPS)




## === cell 11
def build_model(input_shape=(TARGET_SIZE, TARGET_SIZE, 3), n_classes=len(label_cols)):
    base = Xception(include_top=False, weights="imagenet", input_shape=input_shape)
    base.trainable = False
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(n_classes, activation="sigmoid")(x)
    model = models.Model(inputs=base.input, outputs=outputs)
    return model


model = build_model()

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=[],
    jit_compile=True,
)
print("Our Xception CNN has %d layers" % len(model.layers))
print(
    "Trainable variables:",
    len(model.trainable_variables),
    "| Non-trainable:",
    len(model.non_trainable_variables),
)




## === cell 12
history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_steps=VALIDATION_STEPS,
    verbose=2,
)




## === cell 13
pred = model.predict(test_ds, verbose=1)
print("Pred shape:", pred.shape)




## === cell 14
required_cols = [
    "StudyInstanceUID",
    "ETT - Abnormal",
    "ETT - Borderline",
    "ETT - Normal",
    "NGT - Abnormal",
    "NGT - Borderline",
    "NGT - Incompletely Imaged",
    "NGT - Normal",
    "CVC - Abnormal",
    "CVC - Borderline",
    "CVC - Normal",
    "Swan Ganz Catheter Present",
]

sub = pd.DataFrame({"StudyInstanceUID": ss["StudyInstanceUID"].values})
pred_df = pd.DataFrame(pred, columns=label_cols)
sub = pd.concat([sub, pred_df], axis=1)

for c in required_cols[1:]:
    if c not in sub.columns:
        sub[c] = 0.5

sub = sub.reindex(columns=required_cols)
sub.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", sub.shape)
sub.head()




## === cell 15
assert os.path.exists("submission.csv")
chk = pd.read_csv("submission.csv")
print("Submission columns:", chk.columns.tolist())
print("Any NaNs:", chk.isna().any().any())
print(chk.head(2))
