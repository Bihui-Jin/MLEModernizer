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

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import math, re, warnings, random, glob
import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow.keras.layers as L
import tensorflow.keras.backend as K
from tensorflow.keras import Sequential



## === cell 1
SEED = 555
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
warnings.filterwarnings("ignore")

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass



## === cell 2
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print(f"Running on TPU {tpu.master()}")
except Exception:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy()

AUTO = tf.data.experimental.AUTOTUNE
REPLICAS = strategy.num_replicas_in_sync
print(f"REPLICAS: {REPLICAS}")



## === cell 3
database_base_path = "/kaggle/input/ranzcr-clip-catheter-line-classification/"



## === cell 4
BATCH_SIZE = 16 * REPLICAS
HEIGHT = 512
WIDTH = 512
CHANNELS = 3
N_CLASSES = 11
TTA_STEPS = 0  # keep deterministic/stable and within runtime
IMAGE_SIZE = [512, 512]
AUG_BATCH = BATCH_SIZE




## === cell 5
def data_augment(image, label):
    image = tf.image.rot90(
        image, k=tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32, seed=SEED)
    )
    image = tf.image.random_flip_left_right(image, seed=SEED)
    image = tf.image.random_flip_up_down(image, seed=SEED)
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_saturation(image, 0.8, 1.2, seed=SEED)
    return image, label




## === cell 6
train_csv_path = os.path.join(database_base_path, "train.csv")
sample_sub_path = os.path.join(database_base_path, "sample_submission.csv")
train_img_dir = os.path.join(database_base_path, "train")
test_img_dir = os.path.join(database_base_path, "test")

train_df = pd.read_csv(train_csv_path)
sample_submission = pd.read_csv(sample_sub_path)

print("train_df:", train_df.shape)
print("sample_submission:", sample_submission.shape)

target_cols_full = [
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
missing = [c for c in target_cols_full if c not in train_df.columns]
if missing:
    raise ValueError(f"train.csv missing expected target columns: {missing}")

submission_cols = ["StudyInstanceUID"] + target_cols_full




## === cell 7
def decode_image_from_path(path):
    image_bytes = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image_bytes, channels=3)
    image = tf.image.resize(image, [IMAGE_SIZE[0], IMAGE_SIZE[1]])
    image = tf.cast(image, tf.float32) / 255.0
    return image


def make_path(uid, base_dir):
    return tf.strings.join([base_dir, "/", uid, ".jpg"])


def read_labeled_from_uid(uid, label_vec):
    path = make_path(uid, tf.constant(train_img_dir))
    image = decode_image_from_path(path)
    label_vec = tf.cast(label_vec, tf.float32)
    return image, label_vec


def read_unlabeled_from_uid(uid):
    path = make_path(uid, tf.constant(test_img_dir))
    image = decode_image_from_path(path)
    return image, uid




## === cell 8
uids = train_df["StudyInstanceUID"].astype(str).values
labels = train_df[target_cols_full].astype(np.float32).values

n_total = len(train_df)
val_size = int(0.1 * n_total)
train_size = n_total - val_size
print("Split sizes:", train_size, val_size)

rng = np.random.RandomState(SEED)
idx = np.arange(n_total)
rng.shuffle(idx)

train_idx = idx[:train_size]
val_idx = idx[train_size:]

train_uids = uids[train_idx]
val_uids = uids[val_idx]
train_labels = labels[train_idx]
val_labels = labels[val_idx]

train_ds = tf.data.Dataset.from_tensor_slices((train_uids, train_labels))
val_ds = tf.data.Dataset.from_tensor_slices((val_uids, val_labels))

train_ds = train_ds.map(read_labeled_from_uid, num_parallel_calls=AUTO)
val_ds = val_ds.map(read_labeled_from_uid, num_parallel_calls=AUTO)

train_ds = train_ds.map(data_augment, num_parallel_calls=AUTO)

val_ds = val_ds.cache()

train_ds = train_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTO)
val_ds = val_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTO)

test_uids = sorted(
    [
        os.path.splitext(os.path.basename(p))[0]
        for p in glob.glob(os.path.join(test_img_dir, "*.jpg"))
    ]
)
print("Found test images:", len(test_uids))

test_ds = tf.data.Dataset.from_tensor_slices(test_uids)
test_ds = test_ds.map(read_unlabeled_from_uid, num_parallel_calls=AUTO)
test_ds = test_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTO)



## === cell 9
with strategy.scope():
    K.clear_session()
    model = Sequential(
        [
            L.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
            L.Conv2D(16, 3, padding="same", activation="relu"),
            L.MaxPool2D(),
            L.Conv2D(32, 3, padding="same", activation="relu"),
            L.MaxPool2D(),
            L.Conv2D(64, 3, padding="same", activation="relu"),
            L.GlobalAveragePooling2D(),
            L.Dense(64, activation="relu"),
            L.Dense(N_CLASSES, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[],
    )

EPOCHS = 1
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=1)



## === cell 10
models = [model]
print("Using in-notebook trained model (no external .h5 found).")




## === cell 11
def predict_proba(models, test_ds, tta_steps=0):
    if tta_steps and tta_steps > 0:
        probs_steps = []
        for step in range(tta_steps):
            print(f"TTA step {step+1}/{tta_steps}")
            aug_ds = test_ds.map(lambda img, uid: (img, uid), num_parallel_calls=AUTO)
            aug_ds = aug_ds.map(
                lambda img, uid: (
                    data_augment(
                        img, tf.zeros([tf.shape(img)[0], N_CLASSES], dtype=tf.float32)
                    )[0],
                    uid,
                ),
                num_parallel_calls=AUTO,
            )
            imgs_only = aug_ds.map(lambda img, uid: img, num_parallel_calls=AUTO)
            probs = np.average(
                [m.predict(imgs_only, verbose=0) for m in models], axis=0
            )
            probs_steps.append(probs)
        probabilities = np.mean(probs_steps, axis=0)
    else:
        imgs_only = test_ds.map(lambda img, uid: img, num_parallel_calls=AUTO)
        probabilities = np.average(
            [m.predict(imgs_only, verbose=0) for m in models], axis=0
        )

    probabilities = np.asarray(probabilities, dtype=np.float32)
    probabilities = np.clip(probabilities, 0.0, 1.0)
    return probabilities




## === cell 12
print(" TTA_STEPS = {} ".format(TTA_STEPS))
probabilities = predict_proba(models, test_ds, tta_steps=TTA_STEPS)
print("Predictions shape:", probabilities.shape)



## === cell 13
if probabilities.ndim != 2 or probabilities.shape[1] != N_CLASSES:
    raise ValueError(
        f"Unexpected probabilities shape {probabilities.shape}; expected (*, {N_CLASSES})"
    )

probs_full = probabilities  # shape [N_test, 11]



## === cell 14
id_col = "StudyInstanceUID"
target_cols = target_cols_full

sub_df = pd.DataFrame({id_col: np.array(test_uids, dtype=str)})

for j, c in enumerate(target_cols):
    sub_df[c] = probs_full[:, j].astype(np.float32)

sub_df = sub_df[[id_col] + target_cols]



## === cell 15
sub_path = "submission.csv"
sub_df.to_csv(sub_path, index=False)
print(f"Wrote {sub_path} with shape {sub_df.shape} and columns: {list(sub_df.columns)}")

with open(sub_path, "r") as f:
    for _ in range(5):
        print(f.readline().rstrip("\n"))
