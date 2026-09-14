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

0.6908938481576787

# 6. Current score

0.49992

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.49992) has done: 'I fix the environment crash in the first cell by pinning a compatible `protobuf` version at runtime (the `MessageFactory.GetPrototype` error is a known incompatibility between TF 2.18 and protobuf 6.x), which unblocks TensorFlow import. Then I fix the inference crash by removing the invalid `.numpy()` call on the output of `predict_on_batch` (it already returns a NumPy array), so the script can finish and write `submission.csv`. Finally, I make the sample submission column handling robust (the provided sample has only 10 target columns) by always adding any missing targets and outputting exactly the required 11 targets, preserving the core model and preprocessing.'

# 9. Code solution

## === cell 0
import os
import sys
import subprocess

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")


def _ensure_protobuf_compatible():
    try:
        import google.protobuf  # noqa: F401
        from google.protobuf import __version__ as pb_ver

        major = int(pb_ver.split(".")[0])
        if major >= 5:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", "protobuf==4.25.3"]
            )
            import importlib

            importlib.invalidate_caches()
    except Exception as e:
        print("Warning: protobuf compatibility step failed:", repr(e))


_ensure_protobuf_compatible()

import tensorflow as tf
import pandas as pd
import numpy as np

W = H = 338
N_CLASSES = 11
autotune = tf.data.AUTOTUNE

features = {
    "StudyInstanceUID": tf.io.FixedLenFeature([], tf.string),
    "image": tf.io.FixedLenFeature([], tf.string),
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

mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
std = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)


def _find_base_input_dir():
    candidates = [
        "/kaggle/input/ranzcr-clip-catheter-line-classification",
        "../input/ranzcr-clip-catheter-line-classification",
        "/kaggle/data/ranzcr-clip-catheter-line-classification",
        "../data/ranzcr-clip-catheter-line-classification",
        "/kaggle/input",
        "../input",
        "/kaggle/data",
        "../data",
    ]
    for c in candidates:
        if os.path.isdir(c):
            if os.path.basename(c) == "input" or os.path.basename(c) == "data":
                comp = os.path.join(c, "ranzcr-clip-catheter-line-classification")
                if os.path.isdir(comp):
                    return comp
            if os.path.exists(os.path.join(c, "sample_submission.csv")):
                return c
    raise FileNotFoundError(
        "Could not locate ranzcr-clip-catheter-line-classification directory."
    )


BASE_INPUT_DIR = _find_base_input_dir()

test_tfrecords_dir = os.path.join(BASE_INPUT_DIR, "test_tfrecords")
test_img_dir = os.path.join(BASE_INPUT_DIR, "test")

weight_path = "../input/cassava2020weights/ranzcr_efficientb3.h5"
if not os.path.exists(weight_path):
    alt_weight = os.path.join(BASE_INPUT_DIR, "ranzcr_efficientb3.h5")
    if os.path.exists(alt_weight):
        weight_path = alt_weight
    else:
        weight_path = None

print("TF version:", tf.__version__)
print("BASE_INPUT_DIR:", BASE_INPUT_DIR)
print("Has test TFRecords dir:", os.path.isdir(test_tfrecords_dir))
print("Has test image dir:", os.path.isdir(test_img_dir))
print("Weight path:", weight_path)




## === cell 1
def triple_image(image):
    return tf.concat([image] * 3, axis=-1)


def parse_example(sample):
    sample = tf.io.parse_single_example(sample, features)
    image = tf.image.decode_png(sample["image"], channels=1)
    image = tf.image.resize(image, (H, W), method="bilinear")
    image = triple_image(image)  # (H,W,3)
    image_id = sample["StudyInstanceUID"]
    return image, image_id


def preprocess(images, labels):
    images = tf.cast(images, tf.float32) / 255.0
    images = (images - mean) / std
    return images, labels


def get_model(
    baseline_weight=None, init_weight=None, lr=0.001, optimizer=tf.optimizers.Adam
):
    base_model = tf.keras.applications.EfficientNetB3(
        include_top=False,
        input_shape=(W, H, 3),
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
        metrics=[tf.keras.metrics.AUC(name="auc")],
    )

    if init_weight:
        if os.path.exists(init_weight):
            try:
                model.load_weights(init_weight)
                print(f"Weight loaded from {init_weight}")
            except Exception as e:
                print(f"Load weight from {init_weight} failed: {e}")
        else:
            print(f"Weight path not found, using random/imagenet init: {init_weight}")
    return model


def _build_test_dataset(batch_size=16):
    if os.path.isdir(test_tfrecords_dir):
        test_tfrecords = sorted(
            [f for f in os.listdir(test_tfrecords_dir) if f.endswith(".tfrec")]
        )
        if len(test_tfrecords) > 0:
            files = [os.path.join(test_tfrecords_dir, c) for c in test_tfrecords]
            ds = tf.data.TFRecordDataset(files, num_parallel_reads=autotune)
            ds = ds.map(parse_example, num_parallel_calls=autotune)
            ds = ds.batch(batch_size)
            ds = ds.map(preprocess, num_parallel_calls=autotune)
            ds = ds.prefetch(autotune)
            return ds, "tfrecord"
        else:
            print(
                f"Warning: No .tfrec files found in {test_tfrecords_dir}, falling back to jpg."
            )
    else:
        print(f"Warning: Missing directory: {test_tfrecords_dir}, falling back to jpg.")

    if not os.path.isdir(test_img_dir):
        raise FileNotFoundError(f"Missing test images directory: {test_img_dir}")

    jpgs = sorted([f for f in os.listdir(test_img_dir) if f.lower().endswith(".jpg")])
    if len(jpgs) == 0:
        raise FileNotFoundError(f"No .jpg files found in {test_img_dir}")

    paths = [os.path.join(test_img_dir, f) for f in jpgs]
    ids = [os.path.splitext(os.path.basename(f))[0] for f in jpgs]

    def _load(path, sid):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=1)
        img = tf.image.resize(img, (H, W), method="bilinear")
        img = triple_image(img)
        return img, sid

    ds = tf.data.Dataset.from_tensor_slices((paths, ids))
    ds = ds.map(_load, num_parallel_calls=autotune)
    ds = ds.batch(batch_size)
    ds = ds.map(preprocess, num_parallel_calls=autotune)
    ds = ds.prefetch(autotune)
    return ds, "jpg"




## === cell 2
test_data, data_mode = _build_test_dataset(batch_size=16)

sub_path = os.path.join(BASE_INPUT_DIR, "sample_submission.csv")
sub_df = pd.read_csv(sub_path)

for c in target_cols:
    if c not in sub_df.columns:
        sub_df[c] = 0.0
sub_df = sub_df[["StudyInstanceUID"] + target_cols]

model = get_model(None, weight_path)

preds = []
image_ids = []

for images, image_id in test_data:
    image_ids.extend([x.decode("utf-8") for x in image_id.numpy().tolist()])
    batch_pred = model.predict_on_batch(images)
    preds.append(batch_pred)

preds = np.concatenate(preds, axis=0)

if preds.shape[0] != len(image_ids):
    raise RuntimeError(
        f"Pred/ID length mismatch: preds={preds.shape[0]} ids={len(image_ids)}"
    )

pred_df = pd.DataFrame(preds, columns=target_cols)
pred_df.insert(0, "StudyInstanceUID", image_ids)

out = sub_df[["StudyInstanceUID"]].merge(pred_df, on="StudyInstanceUID", how="left")
out[target_cols] = out[target_cols].astype(np.float32).fillna(0.5)

out = out[["StudyInstanceUID"] + target_cols]

out.to_csv("submission.csv", index=False)
print(f"Wrote submission.csv ({data_mode}) with shape:", out.shape)
print(out.head())



## === cell 3
assert os.path.exists("submission.csv")
chk = pd.read_csv("submission.csv")
assert chk.shape[1] == 1 + len(target_cols)
assert chk.columns[0] == "StudyInstanceUID"
for c in target_cols:
    assert c in chk.columns
assert chk["StudyInstanceUID"].isna().sum() == 0
print("Submission OK:", chk.shape)
