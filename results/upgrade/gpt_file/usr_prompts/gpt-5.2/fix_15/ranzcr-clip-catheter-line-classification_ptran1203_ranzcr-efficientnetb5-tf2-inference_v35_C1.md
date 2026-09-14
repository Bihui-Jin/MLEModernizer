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
import sys
import subprocess
import glob


def _ensure_protobuf_compat():
    """
    Bugfix: some Kaggle TF notebooks crash on import with protobuf>=5 depending on TF build.
    Keep as-is (minimal) to make runtime reliable.
    """
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

ENSEMBLE_KEYS = ["resnet101", "efficientb3", "efficientb5", "efficientb7"]
ENSEMBLE_KEYS = [k for k in ENSEMBLE_KEYS if k in model_map]
print("Requested ensemble backbones:", ENSEMBLE_KEYS)

AVAILABLE_WEIGHT_KEYS = [k for k in ENSEMBLE_KEYS if os.path.exists(model_map[k][1])]
MISSING_WEIGHT_KEYS = [k for k in ENSEMBLE_KEYS if k not in AVAILABLE_WEIGHT_KEYS]

print("Backbones with provided .h5 weights:", AVAILABLE_WEIGHT_KEYS)
print("Backbones missing .h5 weights (will use ImageNet init):", MISSING_WEIGHT_KEYS)

if len(ENSEMBLE_KEYS) == 0:
    raise RuntimeError("No known backbones configured in model_map.")




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




## === cell 2
test_img_dir = os.path.join(BASE_INPUT, "test")
use_jpg_fallback = os.path.isdir(test_img_dir)

FORCE_JPG_INFERENCE = True  # deterministic UID alignment via filenames

if (not FORCE_JPG_INFERENCE) and len(test_tfrecords) > 0:
    options = tf.data.Options()
    options.experimental_deterministic = True

    test_data = tf.data.TFRecordDataset(test_tfrecords, num_parallel_reads=1)
    test_data = test_data.with_options(options)
    test_data = test_data.map(parse_example, num_parallel_calls=1)
    test_data = test_data.prefetch(1)
    USING_TFRECORDS = True
    test_paths = None
else:
    if not use_jpg_fallback:
        raise FileNotFoundError(
            f"Could not find test image directory at: {test_img_dir}"
        )

    test_paths = sorted(
        [
            os.path.join(test_img_dir, f)
            for f in os.listdir(test_img_dir)
            if f.lower().endswith(".jpg")
        ]
    )

    test_data = tf.data.Dataset.from_tensor_slices(test_paths)
    test_data = test_data.prefetch(1)
    USING_TFRECORDS = False

print("USING_TFRECORDS:", USING_TFRECORDS)
print("Test items:", len(test_paths) if not USING_TFRECORDS else "unknown (tfrecord)")




## === cell 3
def show_samples_from_paths(paths, n=4):
    fig = plt.figure(figsize=(10, 10))
    for i, p in enumerate(paths[:n]):
        img = tf.io.read_file(p)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (H, W))
        fig.add_subplot(2, 2, i + 1)
        disp = tf.cast(img, tf.float32) / 255.0
        plt.imshow(disp.numpy())
        plt.title(os.path.basename(p)[:30])
        plt.axis("off")
    plt.show()


if not USING_TFRECORDS and len(test_paths) >= 4:
    show_samples_from_paths(test_paths, n=4)



## === cell 4
models = []
preprocess_fns = []

for key in ENSEMBLE_KEYS:
    base_mode, weight_path, prep_fn = model_map[key]
    if os.path.exists(weight_path):
        m = get_model(base_mode, baseline_weight=None, init_weight=weight_path)
        print(f"{key}: using provided weights {weight_path}")
    else:
        m = get_model(base_mode, baseline_weight="imagenet", init_weight=None)
        print(f"{key}: provided weights not found, using ImageNet init")
    models.append(m)
    preprocess_fns.append(prep_fn)

if len(models) == 0:
    raise RuntimeError("No models loaded; cannot run inference.")

print("Loaded models:", [k for k in ENSEMBLE_KEYS], "count:", len(models))



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

if USING_TFRECORDS:
    raise RuntimeError(
        "This notebook is configured for deterministic JPG inference to ensure correct StudyInstanceUID alignment."
    )

test_uids = [os.path.basename(p)[:-4].strip() for p in test_paths]  # remove ".jpg"
sub_template = pd.DataFrame({"StudyInstanceUID": test_uids})
for col in target_cols:
    sub_template[col] = 0.0
sub_template = sub_template[["StudyInstanceUID"] + target_cols].copy()


def TTA(image):
    return tf.stack([image, tf.image.flip_left_right(image)])


c = 0
use_tta = False  # keep as provided


for path in test_paths:
    uid = os.path.basename(path)[:-4].strip()
    image_ids.append(uid)

    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (H, W))
    img = tf.cast(img, tf.float32)

    per_model_preds = []
    for m, prep_fn in zip(models, preprocess_fns):
        x = prep_fn(img)
        if use_tta:
            tta_imgs = TTA(x)
            pr = m.predict_on_batch(tta_imgs).astype(np.float32)
            pr = np.max(pr, axis=0)
        else:
            pr = m.predict(x[tf.newaxis, ...], verbose=0).astype(np.float32)[0]
        per_model_preds.append(pr)

    if len(per_model_preds) == 0:
        raise RuntimeError("No per-model predictions produced; models list is empty.")

    pred = np.mean(np.stack(per_model_preds, axis=0), axis=0).astype(np.float32)
    pred = np.clip(pred, 1e-7, 1.0 - 1e-7)
    preds.append(pred)

    c += 1
    if c % 500 == 0:
        print("Predicted:", c, "/", len(test_paths))

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
if list(merged.columns) != ["StudyInstanceUID"] + target_cols:
    raise RuntimeError("Submission columns are not in the required order.")
if merged.shape[0] != len(test_paths):
    raise RuntimeError(
        f"Bad submission row count: got {merged.shape[0]}, expected {len(test_paths)}"
    )

out_path = "submission.csv"
merged.to_csv(out_path, index=False)

overlap = 1.0 - float(missing_mask.mean())
print("Wrote", out_path, "with shape", merged.shape)
print(merged.head())
print("Predicted UIDs covered in template rate:", overlap)
print("Missing predictions filled with 0.0 rows:", int(missing_mask.sum()))
