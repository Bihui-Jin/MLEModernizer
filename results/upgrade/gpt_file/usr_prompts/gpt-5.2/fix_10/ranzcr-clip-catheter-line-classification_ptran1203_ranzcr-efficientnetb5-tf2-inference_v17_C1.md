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

0.8981493718195671

# 6. Current score

0.46517

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.51637) has done: 'We fix the initial TensorFlow/protobuf crash by removing the forced pure-Python protobuf implementation (it breaks TF 2.18 with protobuf 6.x in this environment). Next, we fix inference by not calling `.numpy()` on outputs that are already NumPy arrays from `predict_on_batch`, which currently prevents any submission from being written. Finally, we make the submission columns robust to the provided `sample_submission.csv` (which may have missing target columns) by adding any missing columns and ensuring the final CSV has exactly the 11 required targets in the right order.'
- What this solution (achieved 0.49414) has done: 'I fix the TensorFlow/protobuf crash that happens at import time by forcing the Python protobuf implementation before importing TensorFlow (this avoids the `MessageFactory.GetPrototype` error seen with TF 2.18 + protobuf 6.x). I also add a small path fallback so the script reliably finds the dataset in this environment (either `../input/...` or `/kaggle/data/...`) without changing the rest of the pipeline. Finally, I keep the model/inference logic the same, but make submission column handling robust so the written `submission.csv` always has exactly the 11 required targets in the correct order.'
- What this solution (achieved 0.49603) has done: 'I fix the TensorFlow/protobuf import crash by removing the forced pure-Python protobuf override, which is what triggers the `MessageFactory.GetPrototype` error in this TF 2.18 + protobuf 6.x environment. I also add a small, safe fallback that uses the JPEG test folder if TFRecords aren’t present, and ensure the submission always contains exactly the 11 required target columns in the correct order even if `sample_submission.csv` is missing some columns. These changes are runtime/stability fixes and should also improve score versus the currently broken/unstable run by allowing the intended pretrained weights inference to execute. No model architecture, loss, or inference semantics are changed.'
- What this solution (achieved 0.48727) has done: 'I fix the TensorFlow import crash by removing the incompatible protobuf environment override that forces the C++ implementation (it triggers the missing `_message` ImportError in this environment). Next, I keep the same model/inference logic but make data-root and submission handling robust so the pipeline always finds `sample_submission.csv` and writes a valid `submission.csv` with exactly the 11 required target columns in the correct order. Finally, I add a safe fallback to load test JPEGs if TFRecords are absent, ensuring `test_data` is always defined and the script runs end-to-end without changing model architecture or loss/metrics.'
- What this solution (achieved 0.49003) has done: 'We fix the TensorFlow/protobuf import crash by forcing the pure-Python protobuf implementation *before* importing TensorFlow (this is the direct cause of `MessageFactory.GetPrototype` under TF 2.18 + protobuf 6.x). Then we keep your model and inference logic intact, but make the submission creation robust to the provided `sample_submission.csv` having only 10 target columns by always outputting the full 11 required targets in the correct order. Finally, we keep your TFRecords/JPEG fallback and ensure `submission.csv` is always written end-to-end.'
- What this solution (achieved 0.53397) has done: 'I fix the immediate runtime crash by removing the forced pure-Python protobuf override, which is incompatible with TensorFlow 2.18 + protobuf 6.x here and triggers the `MessageFactory.GetPrototype` error at import time. I keep the model/inference logic unchanged, but add a small, safe fallback to locate TFRecords/test images and the weights directory more reliably in this filesystem layout so the intended pretrained weights can actually be loaded. Finally, I make submission generation robust to the provided `sample_submission.csv` sometimes having only 10 target columns by always writing all 11 required targets in the correct order to `submission.csv`. These changes are execution/stability fixes and should improve score from the current low value by ensuring the correct weights/inference path runs end-to-end.'
- What this solution (achieved 0.46517) has done: 'I fix the TensorFlow/protobuf import crash that prevents the notebook from running by pinning protobuf to the pure-Python implementation *before* importing TensorFlow (this is the minimal environment fix for the `MessageFactory.GetPrototype` issue in this setup). Then I correct two small but score-critical issues: the EfficientNet input shape is currently swapped (W,H instead of H,W), and the model map points EfficientNetB7 to the wrong weights file (B5). Finally, I make `sample_submission.csv` handling robust by ensuring we always output all 11 target columns (even if the provided sample has only 10), so the produced `submission.csv` is always valid.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

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

DATA_ROOT_CANDIDATES = [
    "/kaggle/data/ranzcr-clip-catheter-line-classification",
    "/kaggle/data/input/ranzcr-clip-catheter-line-classification",
    "/kaggle/input/ranzcr-clip-catheter-line-classification",
    "../input/ranzcr-clip-catheter-line-classification",
    "/kaggle/data/ranzcr-clip-catheter-line-classification/ranzcr-clip-catheter-line-classification",
    "/kaggle/data/input/ranzcr-clip-catheter-line-classification/ranzcr-clip-catheter-line-classification",
    "/kaggle/input/ranzcr-clip-catheter-line-classification/ranzcr-clip-catheter-line-classification",
]
DATA_ROOT = None
for p in DATA_ROOT_CANDIDATES:
    if tf.io.gfile.exists(p):
        DATA_ROOT = p
        break
if DATA_ROOT is None:
    DATA_ROOT = DATA_ROOT_CANDIDATES[0]  # keep original string for error context

test_tfrecords_dir = f"{DATA_ROOT}/test_tfrecords"

weight_dir_candidates = [
    "../input/cassava2020weights",
    "/kaggle/input/cassava2020weights",
    "/kaggle/data/cassava2020weights",
    "/kaggle/data/input/cassava2020weights",
]
weight_dir = None
for p in weight_dir_candidates:
    if tf.io.gfile.exists(p):
        weight_dir = p
        break
if weight_dir is None:
    weight_dir = weight_dir_candidates[0]

has_tfrecords = tf.io.gfile.exists(test_tfrecords_dir)

model_map = {
    "efficientb3": [
        tf.keras.applications.EfficientNetB3,
        f"{weight_dir}/ranzcr_efficientb3.h5",
    ],
    "efficientb5": [
        tf.keras.applications.EfficientNetB5,
        f"{weight_dir}/ranzcr_efficientb5.h5",
    ],
    "efficientb7": [
        tf.keras.applications.EfficientNetB7,
        f"{weight_dir}/ranzcr_efficientb7.h5",
    ],
}

print("TensorFlow:", tf.__version__)
print("DATA_ROOT:", DATA_ROOT)
print("Has TFRecords:", has_tfrecords)
print("Weight dir exists:", tf.io.gfile.exists(weight_dir))




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def triple_image(image):
    return tf.concat([image, image, image], axis=-1)


def parse_example(sample):
    sample = tf.io.parse_single_example(sample, features)
    image = tf.io.decode_jpeg(sample["image"], channels=1)
    image = tf.image.resize(image, (H, W), method="bilinear")
    image = triple_image(image)
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
        include_top=False,
        input_shape=(H, W, 3),
        pooling="avg",
        weights=baseline_weight,
    )
    x = base_model.output
    x = tf.keras.layers.Dropout(0.3)(x)
    out = tf.keras.layers.Dense(N_CLASSES, activation="sigmoid")(x)
    model = tf.keras.models.Model(inputs=base_model.input, outputs=out)

    model.compile(
        optimizer=optimizer(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.AUC(curve="ROC", multi_label=True, num_labels=N_CLASSES)
        ],
    )

    if init_weight:
        try:
            model.load_weights(init_weight)
            print(f"Weight loaded from {init_weight}")
        except Exception as e:
            print(f"Load weight from {init_weight} failed: {e}")
    return model




## === cell 2
if has_tfrecords:
    test_tfrecords = sorted(
        [
            p
            for p in tf.io.gfile.listdir(test_tfrecords_dir)
            if p.endswith(".tfrec") or p.endswith(".tfrecord")
        ]
    )
    if len(test_tfrecords) == 0:
        raise FileNotFoundError(f"No TFRecord files found in: {test_tfrecords_dir}")

    files = [f"{test_tfrecords_dir}/{c}" for c in test_tfrecords]
    test_data = tf.data.TFRecordDataset(files, num_parallel_reads=autotune)
    test_data = test_data.map(parse_example, num_parallel_calls=autotune)
    test_data = test_data.batch(16)
    test_data = test_data.map(preprocess, num_parallel_calls=autotune)
    test_data = test_data.prefetch(autotune)
else:
    test_img_dir = f"{DATA_ROOT}/test"
    if not tf.io.gfile.exists(test_img_dir):
        alt_img_dirs = [
            "/kaggle/data/test",
            "/kaggle/input/test",
            "../input/test",
            "/kaggle/data/ranzcr-clip-catheter-line-classification/test",
            "/kaggle/data/input/ranzcr-clip-catheter-line-classification/test",
            "/kaggle/input/ranzcr-clip-catheter-line-classification/test",
        ]
        for d in alt_img_dirs:
            if tf.io.gfile.exists(d):
                test_img_dir = d
                break
    if not tf.io.gfile.exists(test_img_dir):
        raise FileNotFoundError(f"Test image directory not found: {test_img_dir}")

    test_files = sorted(
        [f for f in tf.io.gfile.listdir(test_img_dir) if f.lower().endswith(".jpg")]
    )

    def parse_jpg(path):
        uid = tf.strings.regex_replace(
            tf.strings.split(path, os.sep)[-1], r"\.jpg$", ""
        )
        img = tf.io.read_file(path)
        img = tf.io.decode_jpeg(img, channels=1)
        img = tf.image.resize(img, (H, W), method="bilinear")
        img = triple_image(img)
        return img, uid

    test_paths = [f"{test_img_dir}/{f}" for f in test_files]
    test_data = tf.data.Dataset.from_tensor_slices(test_paths)
    test_data = test_data.map(parse_jpg, num_parallel_calls=autotune)
    test_data = test_data.batch(16)
    test_data = test_data.map(preprocess, num_parallel_calls=autotune)
    test_data = test_data.prefetch(autotune)

sub_path = f"{DATA_ROOT}/sample_submission.csv"
if not tf.io.gfile.exists(sub_path):
    alt_candidates = [
        "/kaggle/data/sample_submission.csv",
        "/kaggle/input/sample_submission.csv",
        "../input/sample_submission.csv",
        "/kaggle/data/ranzcr-clip-catheter-line-classification/sample_submission.csv",
        "/kaggle/data/input/ranzcr-clip-catheter-line-classification/sample_submission.csv",
        "/kaggle/input/ranzcr-clip-catheter-line-classification/sample_submission.csv",
    ]
    for alt in alt_candidates:
        if tf.io.gfile.exists(alt):
            sub_path = alt
            break

test_df = pd.read_csv(sub_path)

base_mode, weight_path = model_map["efficientb5"]
model = get_model(base_mode, init_weight=weight_path)



## === cell 3
preds = []
image_ids = []

for images, uids in test_data:
    uids_np = uids.numpy()
    if isinstance(uids_np, np.ndarray) and uids_np.dtype.kind in ("S", "O"):
        uids_np = np.array(
            [
                u.decode("utf-8") if isinstance(u, (bytes, bytearray)) else str(u)
                for u in uids_np
            ]
        )
    else:
        uids_np = uids_np.astype("U")

    image_ids.extend(uids_np.tolist())

    batch_pred = model.predict_on_batch(images)
    preds.append(batch_pred)

preds = np.concatenate(preds, axis=0)

pred_df = pd.DataFrame(preds, columns=target_cols)
pred_df.insert(0, "StudyInstanceUID", image_ids)

if "StudyInstanceUID" not in test_df.columns:
    raise ValueError("sample_submission.csv must contain StudyInstanceUID")

test_df = test_df[["StudyInstanceUID"]].merge(
    pred_df, on="StudyInstanceUID", how="left"
)

for c in target_cols:
    if c not in test_df.columns:
        test_df[c] = 0.5
    test_df[c] = test_df[c].fillna(0.5).clip(0.0, 1.0).astype(np.float32)

test_df = test_df[["StudyInstanceUID"] + target_cols]
test_df.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", test_df.shape)
print(test_df.head())
