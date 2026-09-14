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
            ranzcr-clip-catheter-line-classification/
                description.md (172 lines)
                sample_submission.csv (3010 lines)
                ... and 2 other files
                test/
                    1.2.826.0.1.3680043.8.498.11377666469497047083697986859684478212.jpg (162.5 kB)
                    1.2.826.0.1.3680043.8.498.51816564183045831083926719815892052332.jpg (322.2 kB)
                    ... and 3007 other files
                train/
                    1.2.826.0.1.3680043.8.498.86126240554015133345121504023675730690.jpg (136.3 kB)
                    1.2.826.0.1.3680043.8.498.85192731373652359757496590545802262751.jpg (189.5 kB)
                    ... and 27072 other files
        input/
            ranzcr-clip-catheter-line-classification/
                description.md (172 lines)
                sample_submission.csv (3010 lines)
                ... and 2 other files
                test/
                    1.2.826.0.1.3680043.8.498.11377666469497047083697986859684478212.jpg (162.5 kB)
                    1.2.826.0.1.3680043.8.498.51816564183045831083926719815892052332.jpg (322.2 kB)
                    ... and 3007 other files
                train/
                    1.2.826.0.1.3680043.8.498.86126240554015133345121504023675730690.jpg (136.3 kB)
                    1.2.826.0.1.3680043.8.498.85192731373652359757496590545802262751.jpg (189.5 kB)
                    ... and 27072 other files
            test/
                test/
                    1.2.826.0.1.3680043.8.498.11377666469497047083697986859684478212.jpg (162.5 kB)
                    1.2.826.0.1.3680043.8.498.51816564183045831083926719815892052332.jpg (322.2 kB)
                    ... and 3007 other files
            train/
                train/
                    1.2.826.0.1.3680043.8.498.86126240554015133345121504023675730690.jpg (136.3 kB)
                    1.2.826.0.1.3680043.8.498.85192731373652359757496590545802262751.jpg (189.5 kB)
                    ... and 27072 other files
        working/
            ranzcr-clip-catheter-line-classification/
                description.md (172 lines)
                sample_submission.csv (3010 lines)
                ... and 2 other files
                test/
                    1.2.826.0.1.3680043.8.498.11377666469497047083697986859684478212.jpg (162.5 kB)
                    1.2.826.0.1.3680043.8.498.51816564183045831083926719815892052332.jpg (322.2 kB)
                    ... and 3007 other files
                train/
                    1.2.826.0.1.3680043.8.498.86126240554015133345121504023675730690.jpg (136.3 kB)
                    1.2.826.0.1.3680043.8.498.85192731373652359757496590545802262751.jpg (189.5 kB)
                    ... and 27072 other files
```

-> data/ranzcr-clip-catheter-line-classification/sample_submission.csv has 3009 rows and 10 columns.
The columns are: StudyInstanceUID, ETT - Abnormal, ETT - Borderline, ETT - Normal, NGT - Abnormal, NGT - Borderline, NGT - Incompletely Imaged, NGT - Normal, CVC - Abnormal, CVC - Borderline

-> data/ranzcr-clip-catheter-line-classification/train.csv has 27074 rows and 13 columns.
The columns are: StudyInstanceUID, ETT - Abnormal, ETT - Borderline, ETT - Normal, NGT - Abnormal, NGT - Borderline, NGT - Incompletely Imaged, NGT - Normal, CVC - Abnormal, CVC - Borderline, CVC - Normal, Swan Ganz Catheter Present, PatientID

-> data/ranzcr-clip-catheter-line-classification/train_annotations.csv has 16261 rows and 3 columns.
The columns are: StudyInstanceUID, label, data

-> input/ranzcr-clip-catheter-line-classification/sample_submission.csv has 3009 rows and 10 columns.
The columns are: StudyInstanceUID, ETT - Abnormal, ETT - Borderline, ETT - Normal, NGT - Abnormal, NGT - Borderline, NGT - Incompletely Imaged, NGT - Normal, CVC - Abnormal, CVC - Borderline

-> input/ranzcr-clip-catheter-line-classification/train.csv has 27074 rows and 13 columns.
The columns are: StudyInstanceUID, ETT - Abnormal, ETT - Borderline, ETT - Normal, NGT - Abnormal, NGT - Borderline, NGT - Incompletely Imaged, NGT - Normal, CVC - Abnormal, CVC - Borderline, CVC - Normal, Swan Ganz Catheter Present, PatientID

-> input/ranzcr-clip-catheter-line-classification/train_annotations.csv has 16261 rows and 3 columns.
The columns are: StudyInstanceUID, label, data

-> working/ranzcr-clip-catheter-line-classification/sample_submission.csv has 3009 rows and 10 columns.
The columns are: StudyInstanceUID, ETT - Abnormal, ETT - Borderline, ETT - Normal, NGT - Abnormal, NGT - Borderline, NGT - Incompletely Imaged, NGT - Normal, CVC - Abnormal, CVC - Borderline

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB5, EfficientNetB3

BASE_DIR = "/workspace/kaggle/data/ranzcr-clip-catheter-line-classification"
if not os.path.isdir(BASE_DIR):
    BASE_DIR = "/kaggle/input/ranzcr-clip-catheter-line-classification"
    if not os.path.isdir(BASE_DIR):
        raise FileNotFoundError("Dataset directory not found.")

TRAIN_CSV = os.path.join(BASE_DIR, "train.csv")
SAMPLE_SUB_PATH = os.path.join(BASE_DIR, "sample_submission.csv")
TRAIN_IMAGE_DIR = os.path.join(BASE_DIR, "train")
TEST_IMAGE_DIR = os.path.join(BASE_DIR, "test")

train_df = pd.read_csv(TRAIN_CSV)
all_cols = list(train_df.columns)
target_cols = [c for c in all_cols if c not in ("StudyInstanceUID", "PatientID")]
NUM_CLASSES = len(target_cols)

AUTOTUNE = tf.data.AUTOTUNE
IMG_SIZE_B5 = (456, 456)  # EfficientNet‑B5 default
IMG_SIZE_B3 = (300, 300)  # EfficientNet‑B3 default


def load_and_preprocess(file_path):
    """Read an image file, decode, resize to EfficientNet‑B5 input size, and normalize."""
    filename = tf.strings.split(file_path, os.sep)[-1]
    uid = tf.strings.regex_replace(filename, ".jpg$", "")
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE_B5)
    img = img / 255.0
    return img, uid


def load_image(file_path):
    """Decode and resize a single image for training."""
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE_B5)
    img = img / 255.0
    return img


def get_model(base_cls, baseline_weight="imagenet", init_weight=None):
    """
    Build a classification model that uses the given EfficientNet as backbone
    and adds a dense head for the target columns.
    """
    base = base_cls(
        weights=baseline_weight, include_top=False, input_shape=IMG_SIZE_B5 + (3,)
    )
    base.trainable = False  # inference only; will be unfrozen for training later
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="sigmoid")(x)
    model = Model(inputs=base.input, outputs=outputs)
    if init_weight is not None and os.path.isfile(init_weight):
        model.load_weights(init_weight)
    return model


train_file_paths = [
    os.path.join(TRAIN_IMAGE_DIR, f"{uid}.jpg") for uid in train_df["StudyInstanceUID"]
]
train_labels = train_df[target_cols].values.astype(np.float32)

train_ds = tf.data.Dataset.from_tensor_slices((train_file_paths, train_labels))


def _load_img_and_label(path, label):
    img = load_image(path)
    return img, label


train_ds = train_ds.map(_load_img_and_label, num_parallel_calls=AUTOTUNE)

VAL_SPLIT = 0.2
train_size = int((1 - VAL_SPLIT) * len(train_file_paths))
val_size = len(train_file_paths) - train_size

train_ds = train_ds.shuffle(10000, reshuffle_each_iteration=False)
train_dataset = train_ds.take(train_size).batch(32).prefetch(AUTOTUNE)
val_dataset = train_ds.skip(train_size).batch(32).prefetch(AUTOTUNE)

model_b5 = get_model(EfficientNetB5, baseline_weight="imagenet", init_weight=None)

model_b5.layers[0].trainable = True

model_b5.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=[tf.keras.metrics.AUC(name="auc")],
)

model_b5.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=3,
    verbose=2,
)

model_b3 = get_model(EfficientNetB3, baseline_weight="imagenet", init_weight=None)

test_files = [
    os.path.join(TEST_IMAGE_DIR, f)
    for f in os.listdir(TEST_IMAGE_DIR)
    if f.lower().endswith(".jpg")
]
test_dataset = tf.data.Dataset.from_tensor_slices(test_files)
test_dataset = test_dataset.map(load_and_preprocess, num_parallel_calls=AUTOTUNE)
test_dataset = test_dataset.batch(16).prefetch(AUTOTUNE)



## === cell 1
submission_df = pd.read_csv(SAMPLE_SUB_PATH).head(0)

for col in target_cols:
    if col not in submission_df.columns:
        submission_df[col] = np.nan

preds_list = []
ids_list = []

for batch_images, batch_ids in test_dataset:
    preds_b5 = model_b5.predict_on_batch(batch_images)
    preds_b3 = model_b3.predict_on_batch(batch_images)

    flipped_images = tf.image.flip_left_right(batch_images)
    preds_b5_flip = model_b5.predict_on_batch(flipped_images)
    preds_b3_flip = model_b3.predict_on_batch(flipped_images)

    avg_b5 = (preds_b5 + preds_b5_flip) / 2.0
    avg_b3 = (preds_b3 + preds_b3_flip) / 2.0
    final_preds = (avg_b5 + avg_b3) / 2.0

    preds_list.extend(final_preds.tolist())
    ids_list.extend([uid.numpy().decode("utf-8") for uid in batch_ids])

preds_np = np.array(preds_list)  # (num_samples, NUM_CLASSES)

submission_df = submission_df.reindex(range(len(preds_np))).reset_index(drop=True)

submission_df[target_cols] = preds_np
submission_df["StudyInstanceUID"] = ids_list

submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
