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

0.7669549828163248

# 6. Current score

0.50074

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.47088) has done: 'I first fix the import/runtime crash caused by an incompatible protobuf version by pinning protobuf to a TF-compatible range at runtime (this is required for the notebook to even start). Then I fix the broken variable dependencies so `test_tfrecords`, `test_data`, and `model_map` are always defined (cells currently fail after the first exception). Finally, I correct the sample submission/template mismatch by building the submission from the provided `sample_submission.csv` and ensuring all 11 required target columns exist (adding missing ones if needed) before writing `submission.csv`.'
- What this solution (achieved 0.47708) has done: 'Your current score is far below the target, and the biggest issue is that the submission is being built from a 10-target `sample_submission.csv` while the competition requires 11 targets; that mismatch (and the current “append image_ids in dataset order” approach) can severely damage AUC due to column/row misalignment. I keep your model and weights exactly the same, but (1) normalize inputs with the correct `tf.keras.applications.*.preprocess_input` for the chosen backbone (needed because the pretrained weights expect it), (2) force predictions to align to the official test ID order by merging on `StudyInstanceUID` instead of overwriting it, and (3) ensure all 11 target columns are present in the final submission with correct ordering. These are minimal, evaluation-relevant fixes that should move the score up toward your target without changing the core modeling approach.'
- What this solution (achieved 0.52569) has done: 'Your score is far below the target, so the biggest “minimal-change” gains are from fixing avoidable data/IO mismatches rather than changing the model. I (1) ensure we read the correct `sample_submission.csv` with all 11 required targets (the earlier 10-column template can silently drop/omit targets), (2) fix a subtle but important bug in `get_model()` where the input shape uses `(W, H, 3)` instead of `(H, W, 3)`, and (3) make TFRecord file discovery robust by globbing for common `.tfrec/.tfrecord` patterns so we actually use the competition TFRecords when present. These changes keep the same backbone, head, weights, loss, and prediction logic, but should move AUC upward toward your target by correcting input geometry and submission schema/alignment.'
- What this solution (achieved 0.49326) has done: 'Your current score (0.52569) is well below the target (0.76695), so we should improve AUC with minimal, evaluation-relevant fixes rather than changing the model. The biggest likely drag here is that test predictions may not be aligned to the template `StudyInstanceUID`s because TFRecord iteration order is not guaranteed to match the sample submission; your current merge then fills missing rows with 0.0, which can heavily hurt AUC. I make the TFRecord input deterministic (disable non-deterministic parallelism), explicitly decode `StudyInstanceUID` robustly, and add a safety check that reports overlap rate so we don’t silently submit lots of zeros. I also ensure the correct official sample submission is used (the one that includes all 11 targets if present), while still keeping your model, preprocessing, and inference logic the same.'
- What this solution (achieved 0.51979) has done: 'Your current score is far below the target, so we should improve AUC with minimal, evaluation-relevant fixes rather than changing the model. The biggest likely remaining issue is UID mismatch: TFRecords often store `StudyInstanceUID` as bytes, and your current `_decode_uid` can stringify tensors/arrays in a way that doesn’t exactly match the template IDs, causing many rows to be filled with 0.0 and tanking AUC. I make UID decoding strictly UTF-8 for scalar tensors/bytes, and (crucially) I also extract and enforce the correct test ID order directly from the TFRecords (or filenames fallback) so prediction rows are aligned deterministically without relying on merge/fill-na behavior. The model, preprocessing, weights, and inference logic remain the same; this only fixes alignment so your predictions actually land on the right rows/columns in `submission.csv`.'
- What this solution (achieved 0.49193) has done: 'Your current score (0.51979) is far below the target (0.76695), so we should only make small, evaluation-relevant fixes that plausibly increase AUC without changing the model/training core. The biggest remaining risk is still ID alignment: your TFRecord UID extraction is fine, but your TFRecord *dataset order* can differ from the required submission order, and any UID decode/whitespace issues cause many rows to be filled with 0.0 (which crushes AUC). I (1) make TFRecord parsing robust to both `png/jpg` encodings and ensure UID is stripped/normalized, (2) build the prediction DataFrame with guaranteed one-row-per-UID (drop duplicates defensively), and (3) enforce full column coverage by always starting from `train.csv`’s 11-label schema (since your sample file is 10 columns in this environment) and writing predictions in that exact order.'
- What this solution (achieved 0.50242) has done: 'Your score is far below the target, so we should focus on a minimal, evaluation-relevant fix that improves AUC without changing your model/training logic. The biggest remaining likely issue is that your TFRecord parser only reads `StudyInstanceUID` and `image`, so TFRecords that store the ID under other common keys (or store image bytes under `jpg/png`) yield wrong/empty UIDs, causing many rows to be filled with 0.0 and crushing AUC. I make TFRecord parsing robust by trying multiple known feature keys for both UID and image (without changing preprocessing, model, or inference), and I add a hard safety check to fail early if UID overlap is too low (so we don’t silently submit mostly zeros). Everything else—backbone, weights, loss, prediction, and submission schema—stays the same.'
- What this solution (achieved 0.47575) has done: 'Your current score is far below target, so we should make the smallest change that plausibly increases AUC without changing your model/training core. The biggest likely remaining issue is that TFRecords in this competition commonly store the test UID under the feature key `id` (and sometimes `study_uid`), which your parser doesn’t read—this can silently produce empty/incorrect UIDs and then you end up filling many rows with 0.0, crushing AUC. I minimally extend the TFRecord feature spec + UID selection logic to include these common keys while keeping the same decode, resize, preprocessing, model, and prediction flow. I also keep your overlap safety check so we fail loudly if alignment is still broken, ensuring we only write a meaningful submission.'
- What this solution (achieved 0.46212) has done: 'Your current score is far below the target, so we should aim for a small, high-impact fix that increases AUC without changing the model/training logic. The most likely remaining drag is that TFRecord test images can be 16-bit grayscale DICOM-converted PNGs/JPEGs, and `tf.image.decode_image(..., channels=3)` can yield poorly scaled intensities versus what the pretrained backbone expects, harming predictions even if IDs align. I minimally improve decoding by detecting non-uint8 images and scaling them to uint8 (0–255) before resizing/preprocess, while keeping the same backbone, weights, head, loss, and inference loop. I also add a tiny safety clamp to keep predictions in (0,1) to avoid any downstream numeric edge cases (does not change semantics for valid sigmoid outputs).'
- What this solution (achieved 0.50074) has done: 'Your score is far below the target, so the smallest high-impact, metric-relevant fix is to correct the submission schema: you are currently using a 10-column `sample_submission.csv` template in this environment, then adding missing columns, but Kaggle expects **all 11 targets** in the exact competition order; any mismatch can severely damage AUC. I keep your model, weights, preprocessing, and inference loop unchanged, but I build the submission template from `train.csv`’s 11-label schema (and then align it to the official test ID list), ensuring we always write all 11 columns. I also add a strict sanity check that the written submission has exactly 3009 rows and 12 columns (`StudyInstanceUID` + 11 targets) before saving, so we don’t accidentally submit an invalid/partial file. These are minimal I/O/schema fixes that directly affect the leaderboard metric without changing modeling core logic.'

# 9. Code solution

## === cell 0
import os
import sys
import subprocess
import glob


def _ensure_protobuf_compat():
    try:
        import google.protobuf  # noqa: F401
        import google.protobuf.__version__ as pbv  # type: ignore
    except Exception:
        pbv = None

    def _major(v):
        try:
            return int(str(v).split(".")[0])
        except Exception:
            return None

    if pbv is None or (_major(pbv) is not None and _major(pbv) >= 5):
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "protobuf<5"]
        )


_ensure_protobuf_compat()

import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

W = H = 338
N_CLASSES = 11
autotune = tf.data.experimental.AUTOTUNE

tf.random.set_seed(123)

features = {
    "StudyInstanceUID": tf.io.FixedLenFeature([], tf.string, default_value=b""),
    "study_instance_uid": tf.io.FixedLenFeature([], tf.string, default_value=b""),
    "id": tf.io.FixedLenFeature([], tf.string, default_value=b""),
    "study_uid": tf.io.FixedLenFeature([], tf.string, default_value=b""),
    "image": tf.io.FixedLenFeature([], tf.string, default_value=b""),
    "jpg": tf.io.FixedLenFeature([], tf.string, default_value=b""),
    "png": tf.io.FixedLenFeature([], tf.string, default_value=b""),
}

target_cols = [
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

BASE_INPUT = "../input/ranzcr-clip-catheter-line-classification"
test_tfrecords_dir = os.path.join(BASE_INPUT, "test_tfrecords")
weight_dir = "../input/cassava2020weights"

test_tfrecords = []
if os.path.isdir(test_tfrecords_dir):
    files = []
    files += glob.glob(os.path.join(test_tfrecords_dir, "*.tfrec"))
    files += glob.glob(os.path.join(test_tfrecords_dir, "*.tfrecord"))
    test_tfrecords = sorted(set(files))

model_map = {
    "efficientb3": [
        tf.keras.applications.EfficientNetB3,
        os.path.join(weight_dir, "ranzcr_efficientb3.h5"),
        tf.keras.applications.efficientnet.preprocess_input,
    ],
    "efficientb5": [
        tf.keras.applications.EfficientNetB5,
        os.path.join(weight_dir, "ranzcr_efficientb5.h5"),
        tf.keras.applications.efficientnet.preprocess_input,
    ],
    "efficientb7": [
        tf.keras.applications.EfficientNetB7,
        os.path.join(weight_dir, "ranzcr_efficientb7.h5"),
        tf.keras.applications.efficientnet.preprocess_input,
    ],
    "resnet101": [
        tf.keras.applications.ResNet101,
        os.path.join(weight_dir, "ranzcr_resnet101.h5"),
        tf.keras.applications.resnet.preprocess_input,
    ],
}

print("TF version:", tf.__version__)
print("Found test tfrecords:", len(test_tfrecords), "in", test_tfrecords_dir)




## === cell 1
def _decode_any_image(encoded_bytes):
    img = tf.image.decode_image(encoded_bytes, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])

    if img.dtype != tf.uint8:
        img_f = tf.cast(img, tf.float32)
        mn = tf.reduce_min(img_f)
        mx = tf.reduce_max(img_f)
        img_f = tf.cond(
            mx > mn,
            lambda: (img_f - mn) / (mx - mn) * 255.0,
            lambda: tf.zeros_like(img_f),
        )
        img = tf.cast(tf.clip_by_value(img_f, 0.0, 255.0), tf.uint8)

    return img


def parse_example(sample):
    sample = tf.io.parse_single_example(sample, features)

    img_bytes = sample["image"]
    img_bytes = tf.cond(
        tf.size(img_bytes) > 0, lambda: img_bytes, lambda: sample["jpg"]
    )
    img_bytes = tf.cond(
        tf.size(img_bytes) > 0, lambda: img_bytes, lambda: sample["png"]
    )

    image = _decode_any_image(img_bytes)
    image = tf.image.resize(image, (H, W))

    uid = sample["StudyInstanceUID"]
    uid = tf.cond(
        tf.strings.length(uid) > 0, lambda: uid, lambda: sample["study_instance_uid"]
    )
    uid = tf.cond(tf.strings.length(uid) > 0, lambda: uid, lambda: sample["id"])
    uid = tf.cond(tf.strings.length(uid) > 0, lambda: uid, lambda: sample["study_uid"])

    image_id = tf.strings.strip(uid)
    return image, image_id


BACKBONE_KEY = "resnet101"
_preprocess_input = model_map[BACKBONE_KEY][2]


def preprocess(images, labels):
    images = tf.cast(images, tf.float32)
    images = _preprocess_input(images)
    return images, labels


def get_model(
    base_model,
    baseline_weight=None,
    init_weight=None,
    lr=0.001,
    optimizer=tf.optimizers.Adam,
):
    base_model = base_model(
        include_top=False, input_shape=(H, W, 3), pooling="avg", weights=baseline_weight
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




## === cell 2
if len(test_tfrecords) > 0:
    options = tf.data.Options()
    options.experimental_deterministic = True

    test_data = tf.data.TFRecordDataset(test_tfrecords, num_parallel_reads=1)
    test_data = test_data.with_options(options)
    test_data = test_data.map(parse_example, num_parallel_calls=1)
    test_data = test_data.map(preprocess, num_parallel_calls=1)
    test_data = test_data.prefetch(1)
    USING_TFRECORDS = True
else:
    test_img_dir = os.path.join(BASE_INPUT, "test")
    test_paths = sorted(
        [
            os.path.join(test_img_dir, f)
            for f in os.listdir(test_img_dir)
            if f.lower().endswith(".jpg")
        ]
    )

    def _load_jpg(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (H, W))
        img = tf.cast(img, tf.float32)
        img = _preprocess_input(img)
        uid = tf.strings.regex_replace(
            tf.strings.split(path, os.sep)[-1], r"\.jpg$", ""
        )
        uid = tf.strings.strip(uid)
        return img, uid

    test_data = tf.data.Dataset.from_tensor_slices(test_paths)
    test_data = test_data.map(_load_jpg, num_parallel_calls=autotune)
    test_data = test_data.prefetch(1)
    USING_TFRECORDS = False

print("USING_TFRECORDS:", USING_TFRECORDS)




## === cell 3
def show_samples(dataset):
    rows = cols = 2
    fig = plt.figure(figsize=(10, 10))
    for i, (img, label) in enumerate(dataset.shuffle(100).take(rows * cols)):
        fig.add_subplot(rows, cols, i + 1)
        disp = tf.clip_by_value(img, -128.0, 255.0)
        disp = (disp - tf.reduce_min(disp)) / (
            tf.reduce_max(disp) - tf.reduce_min(disp) + 1e-6
        )
        plt.imshow(disp.numpy())
        plt.title(str(label.numpy())[:30])
        plt.axis("off")
    plt.show()


show_samples(test_data)



## === cell 4
base_mode, weight_path, _ = model_map[BACKBONE_KEY]
model = get_model(base_mode, init_weight=weight_path)



## === cell 5
preds = []
image_ids = []

train_csv_candidates = [
    os.path.join(BASE_INPUT, "train.csv"),
    "../input/train.csv",
]
train_csv_path = None
for p in train_csv_candidates:
    if os.path.exists(p):
        train_csv_path = p
        break
if train_csv_path is None:
    raise FileNotFoundError("Could not find train.csv in expected locations.")

train_df = pd.read_csv(train_csv_path)

required_targets = [c for c in target_cols if c in train_df.columns]
if len(required_targets) != len(target_cols):
    missing = [c for c in target_cols if c not in required_targets]
    raise RuntimeError(f"train.csv is missing expected target columns: {missing}")

candidate_templates = [
    os.path.join(BASE_INPUT, "sample_submission.csv"),
    "../input/sample_submission.csv",
]
sample_path = None
for p in candidate_templates:
    if os.path.exists(p):
        sample_path = p
        break
if sample_path is None:
    raise FileNotFoundError(
        "Could not find sample_submission.csv in expected locations."
    )

sub_template = pd.read_csv(sample_path)

sub_template["StudyInstanceUID"] = (
    sub_template["StudyInstanceUID"].astype(str).str.strip()
)
for col in target_cols:
    if col not in sub_template.columns:
        sub_template[col] = 0.0
sub_template = sub_template[["StudyInstanceUID"] + target_cols].copy()


def TTA(image):
    return tf.stack([image, tf.image.flip_left_right(image)])


c = 0
use_tta = False


def _decode_uid(x):
    if isinstance(x, tf.Tensor):
        x = x.numpy()
    if isinstance(x, np.ndarray):
        x = x.item() if x.shape == () else x.reshape(-1)[0]
    if isinstance(x, (bytes, bytearray)):
        s = x.decode("utf-8", errors="strict")
    else:
        s = str(x)
    return s.strip()


for image, img_id in test_data:
    uid = _decode_uid(img_id)
    image_ids.append(uid)

    if use_tta:
        tta_imgs = TTA(image)
        pred = model.predict_on_batch(tta_imgs).astype(np.float32)
        pred = np.max(pred, axis=0)
    else:
        pred = model.predict(image[tf.newaxis, ...], verbose=0).astype(np.float32)[0]

    pred = np.clip(pred, 1e-7, 1.0 - 1e-7)
    preds.append(pred)

    c += 1
    if c % 1000 == 0:
        print(c)

preds = np.asarray(preds, dtype=np.float32)

pred_df = pd.DataFrame(preds[:, : len(target_cols)], columns=target_cols)
pred_df.insert(0, "StudyInstanceUID", image_ids)
pred_df["StudyInstanceUID"] = pred_df["StudyInstanceUID"].astype(str).str.strip()

pred_df = pred_df.drop_duplicates(
    subset=["StudyInstanceUID"], keep="first"
).reset_index(drop=True)

merged = sub_template[["StudyInstanceUID"]].merge(
    pred_df, on="StudyInstanceUID", how="left"
)
missing_mask = merged[target_cols].isna().any(axis=1)
merged[target_cols] = merged[target_cols].fillna(0.0)
merged = merged[["StudyInstanceUID"] + target_cols]

if merged.shape[1] != 1 + len(target_cols):
    raise RuntimeError(
        f"Bad submission column count: got {merged.shape[1]}, expected {1+len(target_cols)}"
    )
if "StudyInstanceUID" not in merged.columns:
    raise RuntimeError("Submission missing StudyInstanceUID column.")
if list(merged.columns) != ["StudyInstanceUID"] + target_cols:
    raise RuntimeError("Submission columns are not in the required order.")
if (
    merged["StudyInstanceUID"].isna().any()
    or (merged["StudyInstanceUID"].astype(str).str.len() == 0).any()
):
    raise RuntimeError("Submission has empty StudyInstanceUID values.")

out_path = "submission.csv"
merged.to_csv(out_path, index=False)

overlap = 1.0 - float(missing_mask.mean())
print("Wrote", out_path, "with shape", merged.shape)
print(merged.head())
print("Template/pred UID overlap rate:", overlap)
print("Missing predictions filled with 0.0 rows:", int(missing_mask.sum()))

if overlap < 0.95:
    raise RuntimeError(
        f"Low UID overlap rate ({overlap:.3f}). This will severely hurt AUC; check TFRecord UID/image keys."
    )
