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

0.8872297608897167

# 6. Current score

0.50624

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.51828) has done: 'I fix the protobuf/TensorFlow import crash by removing the forced pure‑python protobuf setting (it’s causing the `MessageFactory.GetPrototype` error with protobuf 6.x), while keeping the rest of the pipeline unchanged. Then I fix inference to work in TF 2.x eager mode by not calling `.numpy()` on the output of `predict_on_batch`, which is already a NumPy array. Finally, I make the submission robust to the provided sample submission missing some target columns by always producing all 11 required columns in the correct order and writing `submission.csv`.'
- What this solution (achieved 0.48957) has done: 'I fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf implementation before importing TensorFlow (this avoids the `MessageFactory.GetPrototype` issue seen with protobuf 6.x). I also fix a subtle input-shape bug in the model (`input_shape` should be `(H, W, 3)` not `(W, H, 3)`), which can otherwise degrade predictions and score without changing the core architecture. Finally, I keep the existing inference/submission logic but make the TFRecord PNG decode more robust (handle both PNG/JPEG bytes) so the pipeline runs end-to-end and always writes a valid `submission.csv` with all 11 required columns.'
- What this solution (achieved 0.50636) has done: 'We fix the immediate TensorFlow/protobuf crash by removing the forced pure-Python protobuf setting, which is incompatible with protobuf 6.x in this environment and triggers the `MessageFactory.GetPrototype` error. Then we make the submission schema robust to the provided sample submission missing some of the 11 required label columns by always adding and ordering all targets exactly as the competition expects. Finally, we keep the existing model/inference pipeline intact but ensure the data pipeline (especially TFRecord image decoding) is graph-safe and doesn’t rely on Python `try` inside `tf.data` mapping, so it runs end-to-end and writes a valid `submission.csv`.'
- What this solution (achieved 0.52035) has done: 'I first fix the TensorFlow import crash by removing the forced protobuf “cpp” setting, which is causing the `_message` import error with protobuf 6.x in this environment. Then I make the notebook execution order consistent by ensuring all globals (paths, constants) are defined after TensorFlow successfully imports, so later cells don’t hit `NameError`. Finally, I keep your modeling/inference logic intact but make submission writing robust by always outputting all 11 required target columns in the exact order and filling any missing predictions with 0.0, producing a valid `submission.csv`.'
- What this solution (achieved 0.49602) has done: 'We fix the protobuf/TensorFlow import crash by forcing the Python protobuf implementation *before* importing TensorFlow (this is the most reliable workaround for the `MessageFactory.GetPrototype` issue in this environment). Then we fix a submission-schema bug by always producing all 11 target columns even if the provided `sample_submission.csv` is missing some (your sample file here has only 10). Finally, we keep your model/inference core logic unchanged but make inference ID handling robust and ensure the output CSV is correctly aligned to the sample submission order and written as `submission.csv`.'
- What this solution (achieved 0.75487) has done: 'The crash happens before any training/inference because forcing `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` is incompatible with TensorFlow 2.18 + protobuf 6.x here, triggering `MessageFactory.GetPrototype` errors; removing that environment override fixes runtime. Then, to move the score toward the target, we load the provided training labels and fine-tune the same EfficientNetB3 head (same architecture/loss) for a short, deterministic run before predicting test, which should materially improve AUC over random-ish outputs. Finally, we keep the existing robust submission schema logic but ensure the output CSV always contains all 11 required columns in the correct order and aligns exactly to the sample submission IDs.'
- What this solution (achieved 0.75487) has done: 'I remove the protobuf environment override that forces the broken C++ protobuf path in this Kaggle image, which is currently preventing TensorFlow from importing and cascading into all the later `NameError`s. Then I keep your same model/data/training/inference logic but make the path resolution more robust by auto-detecting the dataset root (`/kaggle/input/...` vs `../input/...`) so `train.csv`, images, and `sample_submission.csv` are always found. Finally, I ensure the submission always contains all 11 required target columns in the correct order (even though the provided `sample_submission.csv` here only has 10), and write `submission.csv`.'
- What this solution (achieved 0.50624) has done: 'I fix the TensorFlow/protobuf import crash by forcing the pure-Python protobuf implementation *before* importing TensorFlow (this is the reliable workaround for TF 2.18 + protobuf 6.x). Then I keep your exact model/training/inference logic, but add a small safety fallback so the training runs even if no external weights directory exists (using EfficientNet ImageNet weights instead of random init). Finally, I keep the robust submission schema logic but ensure we always output all 11 required target columns in the correct order and write a valid `submission.csv`.'
- What this solution (achieved 0.50624) has done: 'I remove the protobuf environment override that forces pure-Python protobuf, because it’s what’s triggering the `MessageFactory.GetPrototype` crash with TensorFlow 2.18 + protobuf 6.x here. Then I keep the same model/training/inference pipeline, but fix the submission schema to always output all 11 required target columns in the exact competition order (your provided `sample_submission.csv` appears to be missing some columns). Finally, I make prediction-to-ID alignment robust by always reindexing predictions to the sample submission IDs and filling missing rows with 0.0, ensuring a valid `submission.csv` is written end-to-end.'
- What this solution (achieved 0.50624) has done: 'I remove the forced protobuf “cpp” setting that is causing TensorFlow to fail importing in this environment, which currently prevents every later cell from running. Then I keep your exact model/training/inference logic but make the dataset root autodetection include the `/kaggle/data/...` layout you actually have, so the CSVs/images are found reliably. Finally, I keep the existing robust submission-schema handling (add any missing target columns) and ensure `submission.csv` is always written with all 11 required columns in the correct order.'
- What this solution (achieved 0.50624) has done: 'We fix the immediate TensorFlow/protobuf crash by forcing the pure-Python protobuf implementation *before* importing TensorFlow (this avoids the `MessageFactory.GetPrototype` issue with protobuf 6.x in this environment). Then we fix a submission-format bug by always outputting all 11 competition target columns in the required order, even if `sample_submission.csv` is missing some of them. Finally, we keep your exact model/data/training/inference flow intact, but add a tiny safety clamp to ensure image IDs are decoded consistently so predictions align correctly to the submission IDs.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"

import tensorflow as tf
import pandas as pd
import numpy as np

W = H = 224
N_CLASSES = 11
autotune = tf.data.experimental.AUTOTUNE

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

mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
std = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)

CANDIDATE_ROOTS = [
    "/kaggle/input/ranzcr-clip-catheter-line-classification",
    "/kaggle/data/ranzcr-clip-catheter-line-classification",
    "/kaggle/data/input/ranzcr-clip-catheter-line-classification",
    "../input/ranzcr-clip-catheter-line-classification",
]
DATA_ROOT = next((p for p in CANDIDATE_ROOTS if os.path.exists(p)), CANDIDATE_ROOTS[0])

TEST_DIR = os.path.join(DATA_ROOT, "test")
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
SAMPLE_SUB_PATH = os.path.join(DATA_ROOT, "sample_submission.csv")
TRAIN_CSV_PATH = os.path.join(DATA_ROOT, "train.csv")

test_tfrecords_dir = os.path.join(DATA_ROOT, "test_tfrecords")  # may not exist
weight_dir = "../input/cassava2020weights"  # may not exist

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

print("TF version:", tf.__version__)
print("DATA_ROOT:", DATA_ROOT)
print("TRAIN_CSV_PATH exists:", os.path.exists(TRAIN_CSV_PATH))
print("SAMPLE_SUB_PATH exists:", os.path.exists(SAMPLE_SUB_PATH))
print("TRAIN_DIR exists:", os.path.isdir(TRAIN_DIR))
print("TEST_DIR exists:", os.path.isdir(TEST_DIR))

tf.keras.utils.set_random_seed(42)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def triple_image(image):
    ch = tf.shape(image)[-1]
    image = tf.cond(
        tf.equal(ch, 1),
        lambda: tf.concat([image, image, image], axis=-1),
        lambda: image,
    )
    return image


def preprocess(images, labels):
    images = tf.cast(images, tf.float32) / 255.0
    images = (images - mean) / std
    return images, labels


def parse_jpeg_path(path):
    img_bytes = tf.io.read_file(path)
    image = tf.image.decode_jpeg(img_bytes, channels=1)  # force grayscale
    image = tf.image.resize(image, (H, W), method=tf.image.ResizeMethod.BILINEAR)
    image = triple_image(image)
    image_id = tf.strings.regex_replace(
        tf.strings.split(path, os.sep)[-1], r"\.jpg$", ""
    )
    return image, image_id


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
        metrics=[
            tf.keras.metrics.AUC(curve="ROC", multi_label=True, num_labels=N_CLASSES)
        ],
    )

    if init_weight:
        if os.path.exists(init_weight):
            try:
                model.load_weights(init_weight)
                print(f"Weight loaded from {init_weight}")
            except Exception as e:
                print(f"Load weight from {init_weight} failed, {e}")
        else:
            print(
                f"Init weight not found at {init_weight}. Proceeding with baseline weights={baseline_weight}."
            )
    return model




## === cell 2
train_df = pd.read_csv(TRAIN_CSV_PATH)
missing = [c for c in (["StudyInstanceUID"] + target_cols) if c not in train_df.columns]
if missing:
    raise ValueError(f"train.csv missing columns: {missing}")

train_df["StudyInstanceUID"] = train_df["StudyInstanceUID"].astype(str)
train_df[target_cols] = train_df[target_cols].astype(np.float32)

train_paths = [
    os.path.join(TRAIN_DIR, f"{uid}.jpg")
    for uid in train_df["StudyInstanceUID"].tolist()
]
train_labels = train_df[target_cols].values.astype(np.float32)

exists_mask = np.array([os.path.exists(p) for p in train_paths])
if exists_mask.mean() < 0.95:
    train_paths = list(np.array(train_paths)[exists_mask])
    train_labels = train_labels[exists_mask]
    print(f"Filtered training images to existing files: {len(train_paths)}")


def parse_train_jpeg_path(path, label):
    img_bytes = tf.io.read_file(path)
    image = tf.image.decode_jpeg(img_bytes, channels=1)
    image = tf.image.resize(image, (H, W), method=tf.image.ResizeMethod.BILINEAR)
    image = triple_image(image)
    return image, label


BATCH = 16

train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
train_ds = train_ds.shuffle(
    min(len(train_paths), 8192), seed=42, reshuffle_each_iteration=True
)
train_ds = train_ds.map(parse_train_jpeg_path, num_parallel_calls=autotune)
train_ds = train_ds.batch(BATCH, drop_remainder=False)
train_ds = train_ds.map(preprocess, num_parallel_calls=autotune)
train_ds = train_ds.prefetch(1)

base_mode, weight_path = model_map["efficientb3"]

baseline_weight = None
if not (weight_path and os.path.exists(weight_path)):
    baseline_weight = "imagenet"

model = get_model(base_mode, baseline_weight=baseline_weight, init_weight=weight_path)

EPOCHS = 1
model.fit(train_ds, epochs=EPOCHS, verbose=2)



## === cell 3
if os.path.isdir(test_tfrecords_dir):
    tfrec_files = sorted(
        [
            os.path.join(test_tfrecords_dir, f)
            for f in os.listdir(test_tfrecords_dir)
            if f.endswith(".tfrec") or f.endswith(".tfrecord")
        ]
    )
else:
    tfrec_files = []

if len(tfrec_files) > 0:
    features = {
        "StudyInstanceUID": tf.io.FixedLenFeature([], tf.string),
        "image": tf.io.FixedLenFeature([], tf.string),
    }

    def decode_image_bytes(img_bytes):
        is_png = tf.equal(
            tf.strings.substr(img_bytes, 0, 8),
            tf.constant(b"\x89PNG\r\n\x1a\n"),
        )

        def _decode_png():
            return tf.image.decode_png(img_bytes, channels=1)

        def _decode_jpg():
            return tf.image.decode_jpeg(img_bytes, channels=1)

        return tf.cond(is_png, _decode_png, _decode_jpg)

    def parse_example(sample):
        sample = tf.io.parse_single_example(sample, features)
        image = decode_image_bytes(sample["image"])
        image = tf.image.resize(image, (H, W))
        image = triple_image(image)
        image_id = sample["StudyInstanceUID"]
        return image, image_id

    test_data = tf.data.TFRecordDataset(tfrec_files, num_parallel_reads=autotune)
    test_data = test_data.map(parse_example, num_parallel_calls=autotune)
else:
    test_paths = sorted(
        [
            os.path.join(TEST_DIR, f)
            for f in os.listdir(TEST_DIR)
            if f.lower().endswith(".jpg")
        ]
    )
    if len(test_paths) == 0:
        raise FileNotFoundError(f"No test images found in {TEST_DIR}")
    test_data = tf.data.Dataset.from_tensor_slices(test_paths)
    test_data = test_data.map(parse_jpeg_path, num_parallel_calls=autotune)

test_data = test_data.batch(BATCH)
test_data = test_data.map(preprocess, num_parallel_calls=autotune)
test_data = test_data.prefetch(1)

sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
if "StudyInstanceUID" not in sample_sub.columns:
    raise ValueError("sample_submission.csv must contain StudyInstanceUID")

for c in target_cols:
    if c not in sample_sub.columns:
        sample_sub[c] = 0.0

sample_sub = sample_sub[["StudyInstanceUID"] + target_cols]
test_uids = sample_sub["StudyInstanceUID"].astype(str).tolist()



## === cell 4
preds = []
image_ids = []

for batch_images, batch_image_ids in test_data:
    batch_ids = batch_image_ids.numpy()
    if batch_ids.dtype.kind in ("S", "O"):
        batch_ids = [
            x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else str(x)
            for x in batch_ids.tolist()
        ]
    else:
        batch_ids = batch_ids.astype("U").tolist()

    batch_ids = [x.strip() for x in batch_ids]

    image_ids.extend(batch_ids)

    batch_pred = model.predict_on_batch(batch_images)  # already numpy
    preds.append(batch_pred)

preds = np.concatenate(preds, axis=0)

preds = np.nan_to_num(preds, nan=0.0, posinf=1.0, neginf=0.0)
preds = np.clip(preds, 0.0, 1.0)

pred_df = pd.DataFrame(preds, columns=target_cols)
pred_df["StudyInstanceUID"] = pd.Series(image_ids, dtype="string")

pred_df = pred_df.set_index("StudyInstanceUID")
out = pred_df.reindex(test_uids).reset_index()

out[target_cols] = out[target_cols].fillna(0.0).astype(float)
out = out[["StudyInstanceUID"] + target_cols]

out.to_csv("submission.csv", index=False)

print(out.shape)
print(out.head())
print("Wrote submission.csv")
