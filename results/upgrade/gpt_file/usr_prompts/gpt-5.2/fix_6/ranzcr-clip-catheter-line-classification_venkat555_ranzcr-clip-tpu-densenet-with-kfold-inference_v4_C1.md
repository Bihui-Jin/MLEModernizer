# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


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

# 5. Target score

0.6294664873796718

# 6. Current score

0.52735

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.55957) has done: 'I fix the protobuf/TensorFlow import crash by removing the forced pure-Python protobuf implementation and using the Kaggle default (this is the direct cause of the `MessageFactory.GetPrototype` error). Then I fix the data input path logic: your dataset doesn’t contain `*_tfrecords/*.tfrec`, so I switch to reading the provided JPEGs using `sample_submission.csv` (test) and `train.csv` (train), keeping the same label encoding and the same simple CNN fallback training. Finally, I make prediction deterministic and stable by predicting directly from the `(image, uid)` dataset (no intermediate `.map` that can confuse Keras), and I build a submission that matches the competition’s required 11 target columns by starting from the official sample submission and adding the missing columns when necessary. The result run end-to-end and write `submission.csv`.'
- What this solution (achieved 0.56413) has done: 'I fix the immediate TensorFlow/protobuf crash by explicitly forcing TensorFlow to use the pure-Python protobuf implementation before importing TensorFlow, which avoids the `MessageFactory.GetPrototype` incompatibility. Then I fix the submission schema mismatch by always starting from the competition’s `sample_submission.csv` (which defines the required columns/order) and filling its label columns with our predictions, adding any missing targets with safe defaults so the output is always valid. Finally, to nudge score upward toward your target while preserving the same simple CNN/softmax core logic, I correctly convert the single-class softmax output into multi-label probabilities by mapping each study’s predicted class probability into the appropriate target column (instead of incorrectly writing the 11-class softmax vector directly as 11 independent label probabilities). These changes keep the model/training approach intact, unblock runtime, and should materially improve AUC because the submission probabilities now be semantically aligned with the metric.'
- What this solution (achieved 0.52735) has done: 'I first fix the TensorFlow/protobuf crash by removing the forced pure-Python protobuf environment variables that trigger the `MessageFactory.GetPrototype` incompatibility in this Kaggle image, while keeping TF logging suppression intact. Next, I fix the TTA pipeline bug where augmentation is incorrectly applied after batching (it currently passes a batch tensor into `tf.image.rot90`, which expects rank-3 images), by applying TTA augmentation per-image before batching. Finally, I keep your existing single-class softmax training/prediction core logic unchanged, but correctly map the 11-way softmax outputs into the 11 required submission columns (cell 10 currently incorrectly assumes the model outputs already match the 11 targets one-to-one). These changes should both unblock execution and improve AUC toward your target because the predictions become semantically aligned with the multilabel metric.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import re, math, warnings, random, glob
import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow.keras.layers as L
import tensorflow.keras.backend as K
from tensorflow.keras import Sequential

warnings.filterwarnings("ignore")

print("TF version:", tf.__version__)
print("Imports loaded.")



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
SEED = 555
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)



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
BATCH_SIZE = 16 * REPLICAS
HEIGHT = 512
WIDTH = 512
CHANNELS = 3

N_CLASSES = 11

TTA_STEPS = 3  # Do TTA if > 0
IMAGE_SIZE = [512, 512]
AUG_BATCH = BATCH_SIZE



## === cell 4
database_base_path = "/kaggle/input/ranzcr-clip-catheter-line-classification/"
submission_sample_path = f"{database_base_path}sample_submission.csv"
train_csv_path = f"{database_base_path}train.csv"
train_img_dir = f"{database_base_path}train"
test_img_dir = f"{database_base_path}test"

submission = pd.read_csv(submission_sample_path)
train_df = pd.read_csv(train_csv_path)

print("Sample submission shape:", submission.shape)
print("Train CSV shape:", train_df.shape)
print("Sample submission columns:", submission.columns.tolist())
print("Train columns:", train_df.columns.tolist())

if not tf.io.gfile.exists(train_img_dir) or not tf.io.gfile.exists(test_img_dir):
    raise FileNotFoundError(
        f"Expected train/test image dirs not found: {train_img_dir}, {test_img_dir}"
    )

test_ids_list = submission["StudyInstanceUID"].astype(str).tolist()
NUM_TEST_IMAGES = len(test_ids_list)
print("NUM_TEST_IMAGES:", NUM_TEST_IMAGES)




## === cell 5
def to_float32_2(image, label):
    max_val = tf.reduce_max(label, axis=-1, keepdims=True)
    cond = tf.equal(label, max_val)
    label = tf.where(cond, tf.ones_like(label), tf.zeros_like(label))
    return tf.cast(image, tf.float32), tf.cast(label, tf.int32)


def to_float32(image, label):
    return tf.cast(image, tf.float32), label


def decode_image(image_data):
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    return image


REQUIRED_TARGETS = [
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
REQUIRED_COLUMNS = ["StudyInstanceUID"] + REQUIRED_TARGETS


def encode_multilabel_to_single_class(row_values):
    label = 0
    for i, v in enumerate(row_values):
        if v == 1:
            label = i
    return label


def load_jpeg(path):
    img_bytes = tf.io.read_file(path)
    img = decode_image(img_bytes)
    img = tf.image.resize(img, [IMAGE_SIZE[0], IMAGE_SIZE[1]])
    img = tf.ensure_shape(img, [IMAGE_SIZE[0], IMAGE_SIZE[1], 3])
    return img


def data_augment(image, label):
    k = tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32, seed=SEED)
    image = tf.image.rot90(image, k=k)
    image = tf.image.random_flip_left_right(image, seed=SEED)
    image = tf.image.random_flip_up_down(image, seed=SEED)
    image = tf.image.random_brightness(image, max_delta=0.5)
    image = tf.image.random_saturation(image, 0, 2, seed=SEED)
    image = tf.image.adjust_saturation(image, 3)
    return image, label


def get_training_dataset(dataset, do_aug=True, do_onehot=False):
    if do_aug:
        dataset = dataset.map(data_augment, num_parallel_calls=AUTO)
    dataset = dataset.repeat()
    dataset = dataset.batch(AUG_BATCH)
    if do_onehot:
        raise NameError(
            "onehot() was requested but is not defined in the provided script."
        )
    dataset = dataset.unbatch()
    dataset = dataset.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    dataset = dataset.prefetch(AUTO)
    return dataset


def get_test_dataset(ordered=False, tta=False):
    ids = tf.constant(test_ids_list)
    paths = tf.strings.join([test_img_dir, "/", ids, ".jpg"])
    ds = tf.data.Dataset.from_tensor_slices((paths, ids))

    def _read(path, uid):
        img = load_jpeg(path)
        return img, uid

    ds = ds.map(_read, num_parallel_calls=AUTO)

    if tta:
        ds = ds.map(
            lambda image, uid: (data_augment(image, uid)[0], uid),
            num_parallel_calls=AUTO,
        )

    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(AUTO)
    return ds


def load_train_dataset_from_csv(df):
    ids = df["StudyInstanceUID"].astype(str).values
    labels_matrix = df[REQUIRED_TARGETS].astype(np.int32).values
    labels = np.array(
        [encode_multilabel_to_single_class(row) for row in labels_matrix],
        dtype=np.int32,
    )

    paths = np.array([f"{train_img_dir}/{uid}.jpg" for uid in ids], dtype=object)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def _read(path, label):
        img = load_jpeg(path)
        return img, tf.cast(label, tf.int32)

    ds = ds.map(_read, num_parallel_calls=AUTO)
    return ds


print("Data pipeline functions ready.")



## === cell 6
model_path_list = glob.glob("/kaggle/input/ranzcr-clip/*.h5")
model_path_list.sort()
print("Models to predict (external .h5):")
print(*model_path_list, sep="\n")

USE_EXTERNAL_MODELS = len(model_path_list) > 0
print("USE_EXTERNAL_MODELS:", USE_EXTERNAL_MODELS)



## === cell 7
from tensorflow import keras

models = []
if USE_EXTERNAL_MODELS:
    for model_path in model_path_list:
        print("Loading:", model_path)
        K.clear_session()
        models.append(keras.models.load_model(model_path))
    print("Loaded models:", len(models))
else:
    with strategy.scope():
        model = Sequential(
            [
                L.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
                L.Conv2D(16, 3, padding="same", activation="relu"),
                L.MaxPooling2D(),
                L.Conv2D(32, 3, padding="same", activation="relu"),
                L.MaxPooling2D(),
                L.Conv2D(64, 3, padding="same", activation="relu"),
                L.GlobalAveragePooling2D(),
                L.Dense(64, activation="relu"),
                L.Dense(N_CLASSES, activation="softmax"),
            ]
        )
        model.compile(
            optimizer=keras.optimizers.Adam(1e-3),
            loss=keras.losses.SparseCategoricalCrossentropy(),
            metrics=[],
        )

    train_ds = load_train_dataset_from_csv(train_df)
    train_ds = get_training_dataset(train_ds, do_aug=True, do_onehot=False)

    STEPS_PER_EPOCH = 200
    EPOCHS = 2
    print(
        f"Training fallback model: steps_per_epoch={STEPS_PER_EPOCH}, epochs={EPOCHS}, batch={BATCH_SIZE}"
    )
    model.fit(train_ds, steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS, verbose=1)

    models = [model]
    print("Trained fallback model(s):", len(models))



## === cell 8
print(f" TTA_STEPS = {TTA_STEPS} ")
all_probs = []


def ds_images_only(ds):
    return ds.map(lambda image, uid: image, num_parallel_calls=AUTO)


if TTA_STEPS > 0:
    for step in range(TTA_STEPS):
        test_ds = get_test_dataset(ordered=True, tta=True)
        print(f"TTA step {step+1}/{TTA_STEPS}")
        test_images_ds = ds_images_only(test_ds)
        probs_step = np.average(
            [m.predict(test_images_ds, verbose=0) for m in models], axis=0
        )
        all_probs.append(probs_step)
    probabilities = np.mean(np.stack(all_probs, axis=0), axis=0)
else:
    test_ds = get_test_dataset(ordered=True, tta=False)
    test_images_ds = ds_images_only(test_ds)
    probabilities = np.average(
        [m.predict(test_images_ds, verbose=0) for m in models], axis=0
    )

print("probabilities shape:", probabilities.shape)



## === cell 9
test_ids = np.array(test_ids_list, dtype=str)

if probabilities.ndim != 2:
    raise ValueError(
        f"Expected 2D probabilities (n_samples, n_classes) but got shape {probabilities.shape}"
    )
if len(test_ids) != probabilities.shape[0]:
    raise ValueError(
        f"Mismatch: got {len(test_ids)} ids but {probabilities.shape[0]} predictions"
    )



## === cell 10
n_model_outputs = probabilities.shape[1]
if n_model_outputs != N_CLASSES:
    raise ValueError(
        f"Expected model to output {N_CLASSES} classes but got {n_model_outputs}."
    )

pred_df = pd.DataFrame({"StudyInstanceUID": test_ids})
for i, col in enumerate(REQUIRED_TARGETS):
    pred_df[col] = probabilities[:, i].astype(np.float32)

sub_base = submission.copy()
sub_base["StudyInstanceUID"] = sub_base["StudyInstanceUID"].astype(str)

for col in REQUIRED_TARGETS:
    if col not in sub_base.columns:
        sub_base[col] = 0.0

out = sub_base.merge(pred_df, on="StudyInstanceUID", how="left", suffixes=("", "_pred"))
for col in REQUIRED_TARGETS:
    pred_col = f"{col}_pred"
    if pred_col in out.columns:
        out[col] = out[pred_col].astype(np.float32)
        out.drop(columns=[pred_col], inplace=True)

for col in REQUIRED_COLUMNS:
    if col not in out.columns:
        out[col] = 0.0
out = out[REQUIRED_COLUMNS]

print("Prepared submission dataframe:", out.shape)
print("Output columns:", out.columns.tolist())



## === cell 11
print("Generating submission.csv file...")
out.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", out.shape)
print(out.head())



## === cell 12
import subprocess

subprocess.run(["bash", "-lc", "head -n 3 submission.csv"], check=False)
subprocess.run(["bash", "-lc", "ls -lah submission.csv"], check=False)
