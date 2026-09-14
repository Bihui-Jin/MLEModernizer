# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1

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

0.7011379184728057

# 6. Current score

0.57544

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.50057) has done: 'I fixed the import‑related protobuf error, removed the unused TFRecord handling, and rewrote the pipeline to load the actual JPG test images, run them through an EfficientNet‑B3 backbone (with ImageNet weights) and write a correctly‑formatted `submission.csv`. The changes keep the original model architecture and preprocessing while ensuring the script runs end‑to‑end and produces a valid submission file.'
- What this solution (achieved 0.46998) has done: 'The script now sets a compatible protobuf implementation before importing TensorFlow to eliminate the import error, adds a lightweight training stage that loads the JPG training images and their 11 target labels, fits the original EfficientNet‑B3 model for a short number of epochs (which reliably raises the AUC above the previous 0.50 baseline), and then runs inference on the test set and writes a correctly‑formatted `submission.csv`. All changes keep the original model architecture and preprocessing unchanged while fixing the runtime crash and nudging the score toward the target.'
- What this solution (achieved 0.49045) has done: 'The changes add mixed‑precision training and cache the TF‑data pipeline so images are read‑and‑decoded only once, which dramatically cuts the I/O and computation time per epoch while leaving the model architecture, loss, optimizer and training loops unchanged.  The `make_dataset` function now caches after the map step, and mixed‑precision is enabled before the model is built.  These optimisations keep the exact same training‑validation split and prediction logic, preserving result accuracy.'
- What this solution (achieved 0.49021) has done: 'I add shuffling to the training dataset (so the validation split isn’t biased), introduce a learning‑rate reducer and early‑stopping callbacks, and raise the epoch count to give the model more opportunity to learn. These changes keep the EfficientNet‑B3 backbone and all preprocessing unchanged while modestly improving generalisation, which should raise the AUC toward the target score.'
- What this solution (achieved 0.57544) has done: 'The update speeds up execution by cutting the maximum number of training epochs from 30 to 15 (early‑stopping still stop earlier if possible) and adds a short comment explaining the change. All other logic, model architecture, preprocessing, and data handling remain unchanged, preserving the original behavior and results.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
try:
    from google.protobuf.message_factory import MessageFactory

    if not hasattr(MessageFactory, "GetPrototype"):

        def _GetPrototype(self, descriptor):
            return self.GetMessageClass(descriptor)

        MessageFactory.GetPrototype = _GetPrototype
except Exception:
    pass

import tensorflow as tf
import pandas as pd
import numpy as np
from glob import glob

from tensorflow.keras import mixed_precision, callbacks

tf.random.set_seed(42)

mixed_precision.set_global_policy("mixed_float16")

cpu_count = os.cpu_count() or 1
tf.config.threading.set_inter_op_parallelism_threads(cpu_count)
tf.config.threading.set_intra_op_parallelism_threads(cpu_count)

W = H = 224
N_CLASSES = 11
BATCH_SIZE = 32
EPOCHS = 15
TRAIN_IMG_DIR = "/kaggle/input/ranzcr-clip-catheter-line-classification/train"
TEST_IMG_DIR = "/kaggle/input/ranzcr-clip-catheter-line-classification/test"
TRAIN_CSV_PATH = "/kaggle/input/ranzcr-clip-catheter-line-classification/train.csv"

TARGET_COLS = [
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

_MEAN = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
_STD = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)


def load_and_preprocess_image(path):
    """Read an image, decode, resize and normalise."""
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (H, W))
    image = tf.cast(image, tf.float32) / 255.0
    image = (image - _MEAN) / _STD
    return image


def augment_image(image):
    """Simple random augmentations for training."""
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, 0.9, 1.1)
    return image


def make_dataset(file_list, label_dict=None):
    """
    Validation / test dataset creator.
    If label_dict is provided, yields (image, label) pairs;
    otherwise yields (image, study_id) for test.
    """
    if label_dict is not None:
        labels = []
        for fp in file_list:
            study_id = os.path.basename(fp).replace(".jpg", "")
            lbl = label_dict.get(study_id, [0.0] * N_CLASSES)
            labels.append(lbl)
        label_tensor = tf.constant(labels, dtype=tf.float32)

        ds = tf.data.Dataset.from_tensor_slices((file_list, label_tensor))
        ds = ds.map(
            lambda img_path, label: (load_and_preprocess_image(img_path), label),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
        ds = ds.shuffle(buffer_size=10000, seed=42, reshuffle_each_iteration=False)
    else:
        ds = tf.data.Dataset.from_tensor_slices(file_list)
        ds = ds.map(
            lambda f: (
                load_and_preprocess_image(f),
                tf.strings.regex_replace(tf.strings.split(f, os.sep)[-1], ".jpg", ""),
            ),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
    ds = ds.cache()
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
    return ds


def make_train_dataset(file_list, label_dict):
    """Training dataset with augmentation (cached after basic preprocessing)."""
    labels = []
    for fp in file_list:
        study_id = os.path.basename(fp).replace(".jpg", "")
        lbl = label_dict.get(study_id, [0.0] * N_CLASSES)
        labels.append(lbl)
    label_tensor = tf.constant(labels, dtype=tf.float32)

    ds = tf.data.Dataset.from_tensor_slices((file_list, label_tensor))
    ds = ds.map(
        lambda img_path, label: (load_and_preprocess_image(img_path), label),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    cache_path = "/tmp/train_cache"
    ds = ds.cache(cache_path)  # on‑disk cache, safe for large datasets
    ds = ds.map(
        lambda img, label: (augment_image(img), label),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    ds = ds.shuffle(buffer_size=10000, seed=42, reshuffle_each_iteration=True)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
    return ds




## === cell 1
def build_model():
    """EfficientNet‑B3 backbone + dropout + sigmoid dense head."""
    base = tf.keras.applications.EfficientNetB3(
        include_top=False, input_shape=(H, W, 3), pooling="avg", weights="imagenet"
    )
    x = tf.keras.layers.Dropout(0.3)(base.output)
    out = tf.keras.layers.Dense(N_CLASSES, activation="sigmoid")(x)
    model = tf.keras.Model(inputs=base.input, outputs=out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="auc")],
    )
    return model




## === cell 2
train_df = pd.read_csv(TRAIN_CSV_PATH)
label_cols = TARGET_COLS

label_dict = (
    train_df.set_index("StudyInstanceUID")[label_cols]
    .apply(lambda row: row.astype(np.float32).tolist(), axis=1)
    .to_dict()
)

train_files = sorted(glob(os.path.join(TRAIN_IMG_DIR, "*.jpg")))
rng = np.random.default_rng(seed=42)
rng.shuffle(train_files)

val_size = int(0.1 * len(train_files))
val_files = train_files[:val_size]
train_files_split = train_files[val_size:]

train_dataset = make_train_dataset(train_files_split, label_dict)
val_dataset = make_dataset(val_files, label_dict=label_dict)

model = build_model()
cb_list = [
    callbacks.ReduceLROnPlateau(
        monitor="val_auc", factor=0.5, patience=2, mode="max", verbose=1
    ),
    callbacks.EarlyStopping(
        monitor="val_auc", patience=6, mode="max", restore_best_weights=True, verbose=1
    ),
]

model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=val_dataset,
    verbose=2,
    callbacks=cb_list,
)

test_files = sorted(glob(os.path.join(TEST_IMG_DIR, "*.jpg")))

test_image_ds = tf.data.Dataset.from_tensor_slices(test_files)
test_image_ds = test_image_ds.map(
    load_and_preprocess_image,
    num_parallel_calls=tf.data.experimental.AUTOTUNE,
)
test_image_ds = test_image_ds.batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)

pred_array = model.predict(test_image_ds, verbose=0)

study_ids = [os.path.basename(fp).replace(".jpg", "") for fp in test_files]

submission = pd.DataFrame(columns=["StudyInstanceUID"] + TARGET_COLS)
submission["StudyInstanceUID"] = study_ids
submission[TARGET_COLS] = pred_array

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")
