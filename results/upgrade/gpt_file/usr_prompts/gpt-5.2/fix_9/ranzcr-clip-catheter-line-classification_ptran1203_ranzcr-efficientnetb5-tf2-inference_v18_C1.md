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

0.9285447549366652

# 6. Current score

0.50175

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.50372) has done: 'I remove the `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` override because it triggers a protobuf/tensorflow incompatibility (`MessageFactory.GetPrototype` missing) in this environment. Then I fix the inference loop bug by not calling `.numpy()` on the already-numpy output of `predict_on_batch`, while keeping the model, weights, preprocessing, and batching unchanged. Finally, I make submission creation robust to the provided `sample_submission.csv` missing two target columns by adding any missing required columns from `target_cols` and outputting a correctly ordered 11-label submission.'
- What this solution (achieved 0.49589) has done: 'The immediate blocker is that TensorFlow can’t import because protobuf’s C-extension (`google.protobuf.pyext._message`) isn’t available in this environment; forcing `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=cpp` triggers that failure. I remove the protobuf override entirely so TensorFlow can choose a compatible implementation, which unblocks all downstream `tf` usage and fixes the cascading `NameError`s. Next, I make the dataset base-path resolution include your actual `/kaggle/data/...` layout and fall back to the non-TFRecord JPG pipeline (since `test_tfrecords/` likely isn’t present). Finally, I make submission creation robust to the provided `sample_submission.csv` missing some target columns by adding any missing required columns and outputting an 11-label submission in the exact required order.'
- What this solution (achieved 0.49467) has done: 'I fix the TensorFlow import crash caused by an incompatibility between TensorFlow 2.18 and protobuf 6.x by forcing the pure-Python protobuf implementation *before* importing TensorFlow. Then I make submission creation robust to the fact that your provided `sample_submission.csv` has only 10 target columns by adding the missing required columns (notably `CVC - Normal` and `Swan Ganz Catheter Present`) and outputting all 11 targets in the correct order. These changes are execution-critical and score-improving because the previous run couldn’t actually produce valid model predictions end-to-end. The rest of the model, preprocessing, and inference loop logic is kept unchanged.'
- What this solution (achieved 0.52606) has done: 'I fix the TensorFlow import crash by removing the protobuf implementation override that triggers the `MessageFactory.GetPrototype` incompatibility in this environment. Then I correct a shape bug in `get_model()` where `input_shape` uses swapped `(W, H, 3)` instead of `(H, W, 3)`, which can silently misconfigure the model and hurt AUC while keeping the same architecture and weights. Finally, I make submission creation robust to the provided `sample_submission.csv` having only 10 target columns by always outputting all 11 required columns in the correct order, ensuring a valid `submission.csv` is written end-to-end.'
- What this solution (achieved 0.5224) has done: 'We fix the immediate TensorFlow import crash caused by the protobuf/TensorFlow incompatibility by forcing the pure-Python protobuf implementation *before* importing TensorFlow (this is required in this environment). Then we keep your model/inference logic intact, but make submission writing robust to the provided `sample_submission.csv` having only 10 target columns by always outputting all 11 required columns in the exact order. Finally, we add a couple of small safety checks (existence of test images, prediction/image-id length alignment) that are execution-stability changes and should be score-neutral while ensuring an end-to-end valid `submission.csv`.'
- What this solution (achieved 0.50175) has done: 'I fix the TensorFlow/protobuf import crash by removing the forced pure-Python protobuf override that’s incompatible with TF 2.18 + protobuf 6 in this environment. Then I make the input pipeline and submission-writing more robust but score-neutral: ensure IDs are extracted consistently from JPG paths, ensure prediction ordering/shape alignment is validated, and always write all 11 required target columns in the correct order. I also add a safe fallback for missing external weight directories so the notebook still runs end-to-end (while keeping the same EfficientNetB5 architecture and inference logic). These changes should unblock execution and prevent silent submission-format issues that can depress AUC.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
import pandas as pd
import numpy as np
import cv2  # kept to preserve original imports/core environment parity

from datetime import datetime as dt



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
W = H = 338
N_CLASSES = 11
autotune = tf.data.AUTOTUNE

features = {
    "StudyInstanceUID": tf.io.FixedLenFeature([], tf.string),
    "image": tf.io.FixedLenFeature([], tf.string),
}

target_cols = [
    "CVC - Abnormal",
    "CVC - Borderline",
    "CVC - Normal",
    "ETT - Abnormal",
    "ETT - Borderline",
    "ETT - Normal",
    "NGT - Abnormal",
    "NGT - Borderline",
    "NGT - Incompletely Imaged",
    "NGT - Normal",
    "Swan Ganz Catheter Present",
]

mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
std = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)

CANDIDATE_BASES = [
    "../input/ranzcr-clip-catheter-line-classification",
    "/kaggle/input/ranzcr-clip-catheter-line-classification",
    "/kaggle/data/ranzcr-clip-catheter-line-classification",
    "/kaggle/data/input/ranzcr-clip-catheter-line-classification",
    "/kaggle/data/ranzcr-clip-catheter-line-classification/ranzcr-clip-catheter-line-classification",
    "/kaggle/data/input/ranzcr-clip-catheter-line-classification/ranzcr-clip-catheter-line-classification",
]
BASE_INPUT = None
for p in CANDIDATE_BASES:
    if os.path.exists(p):
        BASE_INPUT = p
        break
if BASE_INPUT is None:
    raise FileNotFoundError(f"Could not find dataset base dir in: {CANDIDATE_BASES}")

test_tfrecords_dir = os.path.join(BASE_INPUT, "test_tfrecords")
test_images_dir = os.path.join(BASE_INPUT, "test")

CANDIDATE_WEIGHT_DIRS = [
    "../input/cassava2020weights",
    "/kaggle/input/cassava2020weights",
    "/kaggle/data/cassava2020weights",
]
weight_dir = None
for p in CANDIDATE_WEIGHT_DIRS:
    if os.path.exists(p):
        weight_dir = p
        break
if weight_dir is None:
    weight_dir = CANDIDATE_WEIGHT_DIRS[-1]
    print(
        f"WARNING: weight directory not found; expected one of {CANDIDATE_WEIGHT_DIRS}"
    )

model_map = {
    "efficientb3": [
        tf.keras.applications.EfficientNetB3,
        os.path.join(weight_dir, "ranzcr_efficientb3.h5"),
    ],
    "efficientb5": [
        tf.keras.applications.EfficientNetB5,
        os.path.join(weight_dir, "ranzcr_efficientb5.h5"),
    ],
    "efficientb7": [
        tf.keras.applications.EfficientNetB7,
        os.path.join(weight_dir, "ranzcr_efficientb7.h5"),
    ],
}




## === cell 2
def triple_image(image):
    return tf.concat([image] * 3, axis=-1)


def parse_example(sample):
    sample = tf.io.parse_single_example(sample, features)
    image = tf.image.decode_png(sample["image"], channels=1)
    image = tf.image.resize(image, (H, W))
    image = triple_image(image)
    image_id = sample["StudyInstanceUID"]
    return image, image_id


def preprocess(images, image_ids):
    images = tf.cast(images, tf.float32) / 255.0
    images = (images - mean) / std
    return images, image_ids


def get_model(
    base_model,
    baseline_weight=None,
    init_weight=None,
    lr=0.001,
    optimizer=tf.optimizers.Adam,
):
    base_model = base_model(
        include_top=False,
        input_shape=(H, W, 3),
        pooling="avg",
        weights=baseline_weight,
    )
    base_out = base_model.output
    out = tf.keras.layers.Dropout(0.3)(base_out)
    out = tf.keras.layers.Dense(N_CLASSES, activation="sigmoid")(out)
    model = tf.keras.models.Model(inputs=base_model.input, outputs=out)

    model.compile(
        optimizer=optimizer(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC()],
    )

    if init_weight:
        try:
            model.load_weights(init_weight)
            print(f"Weight loaded from {init_weight}")
        except Exception as e:
            print(f"Load weight from {init_weight} failed, {e}")
    return model




## === cell 3
use_tfrecords = os.path.isdir(test_tfrecords_dir)
test_data = None

if use_tfrecords:
    test_tfrecords = sorted(
        [
            f
            for f in os.listdir(test_tfrecords_dir)
            if f.endswith(".tfrec") or f.endswith(".tfrecord")
        ]
    )
    if len(test_tfrecords) == 0:
        use_tfrecords = False

if use_tfrecords:
    files = [os.path.join(test_tfrecords_dir, c) for c in test_tfrecords]
    test_data = tf.data.TFRecordDataset(files, num_parallel_reads=autotune)
    test_data = test_data.map(parse_example, num_parallel_calls=autotune)
    test_data = test_data.batch(16)
    test_data = test_data.map(preprocess, num_parallel_calls=autotune)
    test_data = test_data.prefetch(autotune)
    print(f"Using TFRecords from: {test_tfrecords_dir} ({len(files)} files)")
else:
    if not os.path.isdir(test_images_dir):
        raise FileNotFoundError(f"Test images dir not found: {test_images_dir}")

    test_files = sorted(
        [f for f in os.listdir(test_images_dir) if f.lower().endswith(".jpg")]
    )
    if len(test_files) == 0:
        raise FileNotFoundError(f"No .jpg files found in: {test_images_dir}")

    test_paths = [os.path.join(test_images_dir, f) for f in test_files]

    def _read_jpg(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=1)
        img = tf.image.resize(img, (H, W))
        img = triple_image(img)
        fname = tf.strings.split(path, "/")[-1]
        uid = tf.strings.regex_replace(fname, r"\.jpg$", "")
        return img, uid

    test_data = tf.data.Dataset.from_tensor_slices(test_paths)
    test_data = test_data.map(_read_jpg, num_parallel_calls=autotune)
    test_data = test_data.batch(16)
    test_data = test_data.map(preprocess, num_parallel_calls=autotune)
    test_data = test_data.prefetch(autotune)
    print(f"Using JPGs from: {test_images_dir} ({len(test_paths)} files)")

sample_path = os.path.join(BASE_INPUT, "sample_submission.csv")
if not os.path.exists(sample_path):
    alt_sample = "/kaggle/data/sample_submission.csv"
    if os.path.exists(alt_sample):
        sample_path = alt_sample
    else:
        raise FileNotFoundError(f"sample_submission.csv not found at: {sample_path}")

sample_sub = pd.read_csv(sample_path)

for col in target_cols:
    if col not in sample_sub.columns:
        sample_sub[col] = 0.0
sample_sub = sample_sub[["StudyInstanceUID"] + target_cols]



## === cell 4
base_mode, weight_path = model_map["efficientb5"]
model = get_model(base_mode, init_weight=weight_path)

preds = []
image_ids = []

for batch_images, batch_ids in test_data:
    batch_ids_np = batch_ids.numpy()
    batch_ids_str = [
        bid.decode("utf-8") if isinstance(bid, (bytes, np.bytes_)) else str(bid)
        for bid in batch_ids_np
    ]
    image_ids.extend(batch_ids_str)

    batch_pred = model.predict_on_batch(batch_images)
    preds.append(batch_pred)

preds = np.concatenate(preds, axis=0)

if preds.shape[0] != len(image_ids):
    raise RuntimeError(
        f"Prediction rows ({preds.shape[0]}) != number of image ids ({len(image_ids)})"
    )

pred_df = pd.DataFrame(preds, columns=target_cols)
pred_df.insert(0, "StudyInstanceUID", image_ids)

sub = sample_sub[["StudyInstanceUID"]].merge(pred_df, on="StudyInstanceUID", how="left")

for col in target_cols:
    if col not in sub.columns:
        sub[col] = 0.0

sub[target_cols] = sub[target_cols].fillna(0.0).astype(np.float32)
sub = sub[["StudyInstanceUID"] + target_cols]

sub.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", sub.shape)
print("Submission columns:", list(sub.columns))
print(sub.head())
