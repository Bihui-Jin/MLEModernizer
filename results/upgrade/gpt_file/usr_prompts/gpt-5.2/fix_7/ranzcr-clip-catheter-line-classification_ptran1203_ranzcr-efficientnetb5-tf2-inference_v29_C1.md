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


def _ensure_compatible_protobuf():
    try:
        import google.protobuf  # noqa: F401
        from google.protobuf import __version__ as pb_ver
    except Exception:
        pb_ver = None

    def _major(v):
        try:
            return int(str(v).split(".", 1)[0])
        except Exception:
            return None

    if pb_ver is None or (_major(pb_ver) is not None and _major(pb_ver) >= 5):
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "protobuf<5"]
        )
        import importlib
        import google.protobuf

        importlib.reload(google.protobuf)


_ensure_compatible_protobuf()

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

tf.keras.utils.set_random_seed(42)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

W = H = 456
N_CLASSES = 11
AUTOTUNE = tf.data.AUTOTUNE

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


def first_existing_path(paths):
    for p in paths:
        if tf.io.gfile.exists(p):
            return p
    return None


base_data_dir = first_existing_path(
    [
        "/kaggle/input/ranzcr-clip-catheter-line-classification",
        "../input/ranzcr-clip-catheter-line-classification",
        "/kaggle/data/ranzcr-clip-catheter-line-classification",
        "../data/ranzcr-clip-catheter-line-classification",
    ]
)
if base_data_dir is None:
    raise FileNotFoundError(
        "Competition data directory not found in expected locations."
    )

test_img_dir = first_existing_path([os.path.join(base_data_dir, "test")])
train_img_dir = first_existing_path([os.path.join(base_data_dir, "train")])
train_csv_path = first_existing_path([os.path.join(base_data_dir, "train.csv")])
sample_sub_path = first_existing_path(
    [os.path.join(base_data_dir, "sample_submission.csv")]
)

if test_img_dir is None:
    raise FileNotFoundError(f"Missing test image dir (expected {base_data_dir}/test).")
if train_img_dir is None:
    raise FileNotFoundError(
        f"Missing train image dir (expected {base_data_dir}/train)."
    )
if train_csv_path is None:
    raise FileNotFoundError("Missing train.csv in expected locations.")
if sample_sub_path is None:
    raise FileNotFoundError("Missing sample_submission.csv in expected locations.")

print("TF version:", tf.__version__)
print("Base data dir:", base_data_dir)
print("Train image dir:", train_img_dir)
print("Test image dir:", test_img_dir)
print("Train CSV:", train_csv_path)
print("Sample sub:", sample_sub_path)

train_df = pd.read_csv(train_csv_path)
missing_targets = [c for c in target_cols if c not in train_df.columns]
if missing_targets:
    raise KeyError(f"train.csv missing target columns: {missing_targets}")

sub_template = pd.read_csv(sample_sub_path)
print("Sample submission columns:", list(sub_template.columns))
print("Sample submission shape:", sub_template.shape)

for c in target_cols:
    if c not in sub_template.columns:
        sub_template[c] = 0.0
sub_template = sub_template[["StudyInstanceUID"] + target_cols].copy()




## === cell 1
def _decode_image_from_path(path):
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, (H, W), method="bilinear")
    img.set_shape([H, W, 3])
    return img


def preprocess(image, y_or_uid):
    image = tf.cast(image, tf.float32) / 255.0
    image = (image - mean) / std
    return image, y_or_uid


test_uids = sub_template["StudyInstanceUID"].astype(str).tolist()
test_paths = [os.path.join(test_img_dir, f"{uid}.jpg") for uid in test_uids]

missing_files = [p for p in test_paths if not tf.io.gfile.exists(p)]
if missing_files:
    raise FileNotFoundError(
        f"Missing {len(missing_files)} test images; example: {missing_files[0]}"
    )

test_ds = tf.data.Dataset.from_tensor_slices((test_paths, test_uids))
test_ds = test_ds.map(
    lambda p, u: (_decode_image_from_path(p), u), num_parallel_calls=AUTOTUNE
)
test_ds = (
    test_ds.batch(16).map(preprocess, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
)

for batch_images, batch_ids in test_ds.take(1):
    print("Batch images:", batch_images.shape, batch_images.dtype)
    print("Batch ids:", batch_ids.shape, batch_ids.dtype)
    print("First uid:", batch_ids[0].numpy().decode("utf-8"))




## === cell 2
def show_samples(dataset, n=9):
    rows = cols = int(np.ceil(np.sqrt(n)))
    fig = plt.figure(figsize=(12, 12))
    i = 0
    for img, uid in dataset.unbatch().take(n):
        i += 1
        ax = fig.add_subplot(rows, cols, i)
        disp = img * std + mean
        disp = tf.clip_by_value(disp, 0.0, 1.0)
        ax.imshow(disp.numpy())
        ax.set_title(uid.numpy().decode("utf-8")[:18] + "...")
        ax.axis("off")
    plt.tight_layout()
    plt.show()




## === cell 3
from sklearn.model_selection import GroupShuffleSplit

gss = GroupShuffleSplit(n_splits=1, test_size=0.10, random_state=42)
idx_tr, idx_va = next(gss.split(train_df, groups=train_df["PatientID"].values))
tr_df = train_df.iloc[idx_tr].reset_index(drop=True)
va_df = train_df.iloc[idx_va].reset_index(drop=True)


def _make_paths_and_labels(df):
    uids = df["StudyInstanceUID"].astype(str).tolist()
    paths = [os.path.join(train_img_dir, f"{uid}.jpg") for uid in uids]
    y = df[target_cols].astype(np.float32).values
    return uids, paths, y


tr_uids, tr_paths, tr_y = _make_paths_and_labels(tr_df)
va_uids, va_paths, va_y = _make_paths_and_labels(va_df)

for p in tr_paths[:5]:
    if not tf.io.gfile.exists(p):
        raise FileNotFoundError(f"Missing train image: {p}")


def make_train_ds(paths, labels, batch_size=8, training=True):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(min(len(paths), 2048), seed=42, reshuffle_each_iteration=True)
    ds = ds.map(
        lambda p, y: (_decode_image_from_path(p), y), num_parallel_calls=AUTOTUNE
    )
    ds = (
        ds.batch(batch_size)
        .map(preprocess, num_parallel_calls=AUTOTUNE)
        .prefetch(AUTOTUNE)
    )
    return ds


train_ds = make_train_ds(tr_paths, tr_y, batch_size=8, training=True)
val_ds = make_train_ds(va_paths, va_y, batch_size=8, training=False)

print("Train/Val sizes:", len(tr_df), len(va_df))




## === cell 4
def get_model(
    base_model,
    baseline_weight="imagenet",
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
            tf.keras.metrics.AUC(multi_label=True, num_labels=N_CLASSES, name="auc")
        ],
    )

    if init_weight:
        try:
            model.load_weights(init_weight)
            print(f"Weight loaded from {init_weight}")
        except Exception as e:
            print(f"Load weight from {init_weight} failed: {e}")
    return model


base_mode = tf.keras.applications.EfficientNetB5
model = get_model(base_mode, baseline_weight="imagenet", init_weight=None)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=2,
    verbose=1,
)



## === cell 5
preds = np.zeros((len(test_uids), N_CLASSES), dtype=np.float32)

i0 = 0
for images, ids in test_ds:
    p = model.predict_on_batch(images)
    bs = p.shape[0]
    preds[i0 : i0 + bs] = p
    i0 += bs

if i0 != len(test_uids):
    raise RuntimeError(f"Prediction size mismatch: got {i0}, expected {len(test_uids)}")

sub = pd.DataFrame({"StudyInstanceUID": test_uids})
for j, c in enumerate(target_cols):
    sub[c] = preds[:, j].astype(np.float32)

out_path = "submission.csv"
sub.to_csv(out_path, index=False)

print("Wrote:", out_path, "shape:", sub.shape)
print(sub.head())
print("Columns:", list(sub.columns))
