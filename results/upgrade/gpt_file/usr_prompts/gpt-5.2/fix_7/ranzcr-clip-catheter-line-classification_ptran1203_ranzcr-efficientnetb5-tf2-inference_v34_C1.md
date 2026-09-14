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

0.9378210302387324

# 6. Current score

0.48789

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.50431) has done: 'I fix the immediate runtime blockers by (1) removing the incompatible protobuf downgrade (it conflicts with TF 2.18), and (2) making the input pipeline robust to the absence of TFRecords by falling back to reading the provided JPG test images while keeping the same preprocessing and model. I also correct the sample submission handling: your provided sample has only 10 target columns, so we use the official 11-label target list from `train.csv` and ensure the submission has all required columns. Finally, I keep the core model (EfficientNet + dropout + sigmoid) and inference logic the same, just ensuring weights and paths are resolved safely so a valid `submission.csv` is always written end-to-end.'
- What this solution (achieved 0.48197) has done: 'I fix the immediate crash caused by an incompatible `protobuf`/TensorFlow combination by forcing TensorFlow to use the pure-Python protobuf implementation before importing TensorFlow. Then I correct the TFRecord parsing so it can handle both PNG/JPEG-encoded bytes robustly and prevent dtype issues in preprocessing. Finally, I fix the submission-column mismatch by always using the 11 official label columns from `train.csv` (the provided `sample_submission.csv` in this environment is missing some columns), ensuring the output CSV is valid and aligned to the expected format.'
- What this solution (achieved 0.52802) has done: 'I fix the TensorFlow/protobuf crash by avoiding the forced pure-Python protobuf setting that triggers the `MessageFactory.GetPrototype` error in this environment, while keeping TensorFlow 2.18 intact. I also make the dataset path resolution robust for both `/kaggle/input/...` and the provided `/kaggle/data/...` layout so the notebook always finds `train.csv` and `sample_submission.csv`. Finally, I harden TFRecord parsing to accept either `image` or `JPEG` feature keys (common in this competition’s TFRecords) and ensure the submission always contains all 11 required target columns even if the local sample submission is missing some columns.'
- What this solution (achieved 0.49336) has done: 'We fix the immediate TensorFlow/protobuf crash by forcing TensorFlow to use the pure-Python protobuf implementation *and* pinning protobuf to the compatible 4.x runtime **inside the notebook** before importing TensorFlow (this avoids the `MessageFactory.GetPrototype` failure seen with protobuf 6.x). Then we keep your model and preprocessing intact, but harden dataset base-path resolution to prefer `/kaggle/input/...` first (more Kaggle-standard) and ensure the submission always includes all 11 required target columns (even if the provided sample submission is missing some). Finally, we keep inference identical but ensure stable UID decoding and always write a valid `submission.csv`.'
- What this solution (achieved 0.48789) has done: 'Your current score is far below the target, and the biggest issue is that you’re likely predicting only 9/11 labels correctly because the local `sample_submission.csv` in this environment is missing `CVC - Normal` and `Swan Ganz Catheter Present`, which causes the test dataset IDs (and thus predictions) to be misaligned/incomplete. I fix this by building the test ID list directly from the filenames in the `test/` folder (or TFRecords if present) rather than relying on the broken sample submission, while keeping your model, preprocessing, and inference logic the same. I also ensure the submission always has all 11 required columns in the exact order from `train.csv`, and that every test UID gets exactly one prediction row. These are minimal, semantics-preserving changes that should substantially increase AUC toward your target by correcting label/row alignment rather than “improving the model.”'

# 9. Code solution

## === cell 0
import os
import sys
import subprocess

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

try:
    import google.protobuf  # noqa: F401
    from google.protobuf import __version__ as pb_ver

    if int(pb_ver.split(".")[0]) >= 5:
        raise ImportError(f"protobuf runtime too new: {pb_ver}")
except Exception:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "protobuf<5"])
    import importlib

    importlib.invalidate_caches()

import numpy as np
import pandas as pd

import tensorflow as tf
import matplotlib.pyplot as plt

tf.random.set_seed(42)
np.random.seed(42)

W = H = 456
N_CLASSES = 11
autotune = tf.data.AUTOTUNE

mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
std = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)


def decode_image_bytes(img_bytes):
    img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
    img = tf.image.resize(img, (H, W), method="bilinear")
    img = tf.cast(img, tf.float32)
    return img


features = {
    "StudyInstanceUID": tf.io.FixedLenFeature([], tf.string),
    "image": tf.io.FixedLenFeature([], tf.string, default_value=b""),
    "JPEG": tf.io.FixedLenFeature([], tf.string, default_value=b""),
}

CANDIDATES = [
    "/kaggle/input/ranzcr-clip-catheter-line-classification",
    "/kaggle/data/ranzcr-clip-catheter-line-classification",
    "/kaggle/data/input/ranzcr-clip-catheter-line-classification",
    "../input/ranzcr-clip-catheter-line-classification",
]
DATA_BASE = None
for p in CANDIDATES:
    if os.path.exists(p):
        DATA_BASE = p
        break
if DATA_BASE is None:
    raise FileNotFoundError(f"Could not find dataset base. Tried: {CANDIDATES}")

test_tfrecords_dir = os.path.join(DATA_BASE, "test_tfrecords")
test_img_dir = os.path.join(DATA_BASE, "test")
train_csv_path = os.path.join(DATA_BASE, "train.csv")
sample_path = os.path.join(DATA_BASE, "sample_submission.csv")

train_df = pd.read_csv(train_csv_path)
target_cols = [
    c for c in train_df.columns if c not in ["StudyInstanceUID", "PatientID"]
]
assert (
    len(target_cols) == N_CLASSES
), f"Expected {N_CLASSES} target columns, got {len(target_cols)}"

weight_dir_1 = "../input/cassava2020weights"
weight_dir_2 = "/kaggle/input/cassava2020weights"
weight_dir_3 = "/kaggle/data/cassava2020weights"
weight_dir = (
    weight_dir_1
    if os.path.exists(weight_dir_1)
    else (weight_dir_2 if os.path.exists(weight_dir_2) else weight_dir_3)
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

print("TF version:", tf.__version__)
print("DATA_BASE:", DATA_BASE)
print("test_tfrecords_dir exists:", os.path.isdir(test_tfrecords_dir))
print("test_img_dir exists:", os.path.isdir(test_img_dir))
print("sample submission exists:", os.path.exists(sample_path))
print("train.csv exists:", os.path.exists(train_csv_path))
print("weight_dir exists:", os.path.isdir(weight_dir))
print("N target cols:", len(target_cols))
print("Targets:", target_cols)




## === cell 1
def parse_example(sample):
    sample = tf.io.parse_single_example(sample, features)
    img_bytes = sample["image"]
    img_bytes = tf.cond(
        tf.size(img_bytes) > 0, lambda: img_bytes, lambda: sample["JPEG"]
    )
    image = decode_image_bytes(img_bytes)
    image_id = sample["StudyInstanceUID"]
    return image, image_id


def preprocess(images, labels):
    images = tf.cast(images, tf.float32) / 255.0
    images = (images - mean) / std
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
    if init_weight and isinstance(init_weight, str) and os.path.exists(init_weight):
        try:
            model.load_weights(init_weight)
            print(f"Weight loaded from {init_weight}")
        except Exception as e:
            print(f"Load weight from {init_weight} failed, {e}")
    else:
        print(
            f"Init weight not found; training-free inference will be weak. Looked for: {init_weight}"
        )
    return model




## === cell 2
def _list_test_ids_from_images(test_dir):
    files = [f for f in os.listdir(test_dir) if f.lower().endswith(".jpg")]
    ids = [os.path.splitext(f)[0] for f in files]
    ids = sorted(ids)
    return ids


def make_test_dataset_and_ids():
    if os.path.isdir(test_tfrecords_dir):
        test_tfrecords = sorted(
            [
                f
                for f in os.listdir(test_tfrecords_dir)
                if f.endswith((".tfrec", ".tfrecord", ".tfrec.gz", ".tfrecord.gz"))
            ]
        )
        files = [os.path.join(test_tfrecords_dir, c) for c in test_tfrecords]
        if len(files) > 0:
            ds = tf.data.TFRecordDataset(files, num_parallel_reads=autotune)
            ds = ds.map(parse_example, num_parallel_calls=autotune)
            ds = ds.map(preprocess, num_parallel_calls=autotune)
            ds = ds.batch(1).prefetch(autotune)
            return ds, None

    if os.path.isdir(test_img_dir):
        ids = _list_test_ids_from_images(test_img_dir)
        paths = [os.path.join(test_img_dir, f"{uid}.jpg") for uid in ids]

        def _load_jpg(path, uid):
            img_bytes = tf.io.read_file(path)
            img = decode_image_bytes(img_bytes)
            return img, uid

        ds = tf.data.Dataset.from_tensor_slices((paths, ids))
        ds = ds.map(_load_jpg, num_parallel_calls=autotune)
        ds = ds.map(preprocess, num_parallel_calls=autotune)
        ds = ds.batch(1).prefetch(autotune)
        return ds, ids

    raise FileNotFoundError(
        f"Neither TFRecords nor test image directory found. Checked: {test_tfrecords_dir} and {test_img_dir}"
    )


test_data, test_ids_from_images = make_test_dataset_and_ids()
print("Test dataset built.")




## === cell 3
def show_samples(dataset):
    rows = cols = 2
    fig = plt.figure(figsize=(10, 10))
    for i, (img, _) in enumerate(dataset.unbatch().shuffle(100).take(rows * cols)):
        fig.add_subplot(rows, cols, i + 1)
        plt.imshow(tf.clip_by_value((img * std + mean), 0.0, 1.0))
        plt.axis("off")
    plt.show()




## === cell 4
base_mode, weight_path = model_map["efficientb5"]
model = get_model(base_mode, init_weight=weight_path)



## === cell 5
if test_ids_from_images is None:
    submission_ids = None
else:
    submission_ids = test_ids_from_images

preds = []
image_ids = []


def TTA(image):
    return tf.stack([image, tf.image.flip_left_right(image)])


c = 0
use_tta = False

for image, img_id in test_data:
    raw = img_id.numpy()[0]
    if isinstance(raw, (bytes, np.bytes_)):
        sid = raw.decode("utf-8")
    else:
        sid = str(raw)
    image_ids.append(sid)

    if use_tta:
        tta_imgs = TTA(image[0])
        pred = model.predict_on_batch(tta_imgs).astype(np.float32)
        pred = np.max(pred, axis=0)
    else:
        pred = model.predict_on_batch(image).astype(np.float32)[0]

    preds.append(pred)

    c += 1
    if c % 1000 == 0:
        print("Predicted", c)

preds = np.asarray(preds, dtype=np.float32)
pred_df = pd.DataFrame(preds, columns=target_cols)
pred_df["StudyInstanceUID"] = image_ids

if submission_ids is None:
    base_ids = pd.Series(image_ids, dtype=str).drop_duplicates().tolist()
else:
    base_ids = submission_ids

sub = pd.DataFrame({"StudyInstanceUID": pd.Series(base_ids, dtype=str).values})
sub = sub.merge(pred_df, on="StudyInstanceUID", how="left")

sub[target_cols] = sub[target_cols].astype(np.float32).fillna(0.5).clip(0.0, 1.0)

sub = sub[["StudyInstanceUID"] + target_cols].copy()

sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print("Columns:", list(sub.columns))
print(sub.head())
