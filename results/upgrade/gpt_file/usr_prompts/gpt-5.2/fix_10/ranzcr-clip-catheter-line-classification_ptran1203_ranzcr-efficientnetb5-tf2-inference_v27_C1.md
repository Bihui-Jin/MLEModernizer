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

0.916984158104694

# 6. Current score

0.49912

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.52501) has done: 'I fix the runtime error by removing the unnecessary `.numpy()` call on the output of `model.predict_on_batch`, which already returns a NumPy array in TF/Keras. I also make the sample submission handling robust: your provided `sample_submission.csv` is missing two required columns (`CVC - Normal`, `Swan Ganz Catheter Present`), so I build the submission starting from the test IDs we predict and then enforce the exact required column set/order. Finally, I keep your model and preprocessing logic unchanged, only adding safe checks and column alignment so a valid `submission.csv` is always produced end-to-end.'
- What this solution (achieved 0.52283) has done: 'Your score is far below the target, so we need a real (but still minimal) correctness fix rather than tuning: your pipeline is feeding *unnormalized 0–255 images* into EfficientNet weights that expect a specific `preprocess_input`, which typically collapses AUC to near-random. I keep your model, weights, and prediction loop unchanged, and only change preprocessing to apply the matching EfficientNet preprocessing (and ensure float32). I also remove the `pip install protobuf==3.20.3` because it can silently break TF 2.18 behavior; this should improve stability and avoid dependency mismatches. These changes should move the score upward substantially toward the target while preserving the core logic and submission format.'
- What this solution (achieved 0.50357) has done: 'I remove the protobuf-related runtime failure by forcing TensorFlow to use the pure-Python protobuf implementation before importing TensorFlow, which avoids the `MessageFactory.GetPrototype` crash in this environment. I also fix the dataset input paths: this competition dataset provides JPGs and a sample submission but not `test_tfrecords`, so the code consistently read from the existing `test/` folder. Finally, I keep your model and EfficientNet preprocessing intact (since that’s the key score fix already applied), while ensuring the submission has exactly the required 11 target columns in the correct order even if the provided `sample_submission.csv` is missing some columns.'
- What this solution (achieved 0.52259) has done: 'I fix the protobuf/TensorFlow crash that’s currently preventing the notebook from running by enforcing a compatible protobuf version and forcing the pure-Python protobuf runtime before importing TensorFlow. I also correct a real logic issue that hurts AUC badly: your TFRecord parser decodes the image bytes as PNG even though the dataset images are JPEG; this silently corrupts inputs when TFRecords are used. Finally, I keep your model, weights, preprocessing, and prediction loop intact, but make the TFRecord/JPG selection robust and ensure we always write a valid `submission.csv` with all 11 required target columns in the correct order.'
- What this solution (achieved 0.53554) has done: 'Your current score is far below the target, so the most likely issue is not “tuning” but correctness: you are using EfficientNetB5 *ImageNet* preprocessing (`efficientnet.preprocess_input`) while building the model with `tf.keras.applications.EfficientNetB5` (which in TF uses the EfficientNetV2-style `-1..1` preprocessing). I keep your model, weights loading, and prediction loop unchanged, but switch preprocessing to the matching `tf.keras.applications.efficientnet_v2.preprocess_input` to align inputs with what these backbones/weights expect. I also enforce deterministic decoding/resizing and ensure the submission includes all 11 required columns in the correct order (keeping your robust column-fill behavior). These are minimal changes aimed at materially increasing AUC toward the target without changing the modeling approach.'
- What this solution (achieved 0.52985) has done: 'Your score is far below the target, so we should focus on a likely correctness issue rather than tuning: your preprocessing is currently using `efficientnet_v2.preprocess_input`, but your backbone is `tf.keras.applications.EfficientNetB5` (EfficientNet v1) which expects its own `tf.keras.applications.efficientnet.preprocess_input`. I make that single preprocessing swap (keeping the model, weights, and inference loop the same) to align inputs with the pretrained/finetuned weights and typically recover a large amount of AUC. I also add a small safety fix to ensure test IDs are always converted from `tf.Tensor` to proper Python strings robustly (avoids rare misalignment) while keeping the exact submission schema with all 11 required target columns.'
- What this solution (achieved 0.51171) has done: 'Your current score is far below the target, so the most likely blocker is not “tuning” but a correctness mismatch between the backbone and its preprocessing. I keep your model (EfficientNetB5), weights, and inference loop identical, but switch preprocessing to the matching `tf.keras.applications.efficientnet.preprocess_input` for EfficientNet v1 and also ensure that the sample-visualization is not inadvertently showing already-preprocessed tensors. I also make ID decoding robust for both TFRecord and JPG paths and enforce that the submission always contains all 11 required label columns (even when the provided sample submission is missing some), preserving your current schema/alignment logic. These are minimal changes aimed at materially increasing AUC toward your target without changing the modeling approach.'
- What this solution (achieved 0.49912) has done: 'Your score is far below the target, so we focus on a small correctness fix that can materially lift AUC without changing your model or training approach: ensure the EfficientNetB5 preprocessing exactly matches the backbone by using the `tf.keras.applications.efficientnet.preprocess_input` **from the same module as the model** (and keep images as float32). We also make the test-ID ordering deterministic and robust by emitting IDs directly from the dataset in the same order as predictions (no shuffle anywhere) and asserting row counts match. Finally, we harden the submission schema to always include all 11 required target columns in the correct order, even if the provided `sample_submission.csv` is missing columns.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import sys
import subprocess


def _ensure_protobuf_compat():
    try:
        import google.protobuf  # noqa: F401
        from importlib.metadata import version

        v = version("protobuf")
        major = int(v.split(".")[0])
        if major >= 6:
            print(
                "Detected protobuf",
                v,
                "-> installing protobuf==4.25.3 for TensorFlow compatibility",
            )
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", "protobuf==4.25.3"]
            )
    except Exception as e:
        print("protobuf compatibility check skipped/failed:", repr(e))


_ensure_protobuf_compat()

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

W = H = 456
N_CLASSES = 11
autotune = tf.data.experimental.AUTOTUNE

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

label_list = target_cols[:]  # 11 labels

DATA_ROOT = "../input/ranzcr-clip-catheter-line-classification"

test_images_dir = os.path.join(DATA_ROOT, "test")
test_tfrecords_dir = os.path.join(DATA_ROOT, "test_tfrecords")  # may or may not exist

weight_dir = "../input/cassava2020weights"
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
print("Test TFRecord dir exists:", os.path.isdir(test_tfrecords_dir))
print("Test image dir exists:", os.path.isdir(test_images_dir))




## === cell 1
def parse_example(sample):
    sample = tf.io.parse_single_example(sample, features)
    image = tf.image.decode_jpeg(sample["image"], channels=3)
    image = tf.image.resize(image, (H, W), method="bilinear", antialias=True)
    image_id = sample["StudyInstanceUID"]
    return image, image_id


def preprocess(images, labels):
    images = tf.cast(images, tf.float32)
    images = tf.keras.applications.efficientnet.preprocess_input(images)
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
            print(f"Load weight from {init_weight} failed: {e}")
    return model




## === cell 2
def make_test_dataset_from_tfrecords(tfrecord_dir, batch_size=16):
    test_tfrecords = sorted(
        [
            f
            for f in os.listdir(tfrecord_dir)
            if f.endswith(".tfrec") or f.endswith(".tfrecord")
        ]
    )
    files = [os.path.join(tfrecord_dir, f) for f in test_tfrecords]
    ds = tf.data.TFRecordDataset(files, num_parallel_reads=autotune)
    ds = ds.map(parse_example, num_parallel_calls=autotune)
    opts = tf.data.Options()
    opts.experimental_deterministic = True
    ds = ds.with_options(opts)

    ds = ds.batch(batch_size)
    ds = ds.map(preprocess, num_parallel_calls=autotune)
    ds = ds.prefetch(1)
    return ds, len(files)


def make_test_dataset_from_jpgs(image_dir, batch_size=16):
    jpgs = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(".jpg")])
    paths = [os.path.join(image_dir, f) for f in jpgs]
    ids = [os.path.splitext(os.path.basename(p))[0] for p in paths]

    def _load(path, sid):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (H, W), method="bilinear", antialias=True)
        return img, sid

    ds = tf.data.Dataset.from_tensor_slices((paths, ids))
    opts = tf.data.Options()
    opts.experimental_deterministic = True
    ds = ds.with_options(opts)

    ds = ds.map(_load, num_parallel_calls=autotune)
    ds = ds.batch(batch_size)
    ds = ds.map(preprocess, num_parallel_calls=autotune)
    ds = ds.prefetch(1)
    return ds, len(paths)


use_tfrecords = False
if os.path.isdir(test_tfrecords_dir):
    try:
        tfrec_files = [
            f
            for f in os.listdir(test_tfrecords_dir)
            if f.endswith(".tfrec") or f.endswith(".tfrecord")
        ]
        use_tfrecords = len(tfrec_files) > 0
    except Exception:
        use_tfrecords = False

if use_tfrecords:
    test_data, n_files = make_test_dataset_from_tfrecords(
        test_tfrecords_dir, batch_size=16
    )
    print("Using TFRecords for test. n_files:", n_files)
else:
    if not os.path.isdir(test_images_dir):
        raise FileNotFoundError(f"Test image directory not found: {test_images_dir}")
    test_data, n_imgs = make_test_dataset_from_jpgs(test_images_dir, batch_size=16)
    print("Using JPGs for test. n_images:", n_imgs)




## === cell 3
def show_samples(dataset, n=9):
    rows = cols = int(np.ceil(np.sqrt(n)))
    fig = plt.figure(figsize=(12, 12))
    for i, (img, _sid) in enumerate(dataset.unbatch().shuffle(100).take(rows * cols)):
        ax = fig.add_subplot(rows, cols, i + 1)
        disp = (img + 1.0) * 127.5
        ax.imshow(tf.cast(tf.clip_by_value(disp, 0, 255), tf.uint8).numpy())
        ax.axis("off")
    plt.tight_layout()
    plt.show()




## === cell 4
base_mode, weight_path = model_map["efficientb5"]
model = get_model(base_mode, init_weight=weight_path)



## === cell 5
required_cols = ["StudyInstanceUID"] + target_cols

preds = []
image_ids = []


def _to_str_id(sid):
    if isinstance(sid, tf.Tensor):
        sid = sid.numpy()
    if isinstance(sid, (bytes, bytearray, np.bytes_)):
        return sid.decode("utf-8")
    return str(sid)


c = 0
for images, image_id in test_data:
    for sid in image_id:
        image_ids.append(_to_str_id(sid))

    batch_pred = model.predict_on_batch(images)  # already numpy
    preds.append(batch_pred)

    c += 1
    if c % 100 == 0:
        print("batches:", c)

preds = np.concatenate(preds, axis=0)

if len(image_ids) != preds.shape[0]:
    raise RuntimeError(
        f"Mismatch between collected ids ({len(image_ids)}) and predictions ({preds.shape[0]})."
    )

pred_df = pd.DataFrame(preds, columns=target_cols)
pred_df.insert(0, "StudyInstanceUID", image_ids)

sample_path = os.path.join(DATA_ROOT, "sample_submission.csv")
if os.path.exists(sample_path):
    sub = pd.read_csv(sample_path)
    if "StudyInstanceUID" not in sub.columns:
        sub = pred_df[["StudyInstanceUID"]].copy()
else:
    sub = pred_df[["StudyInstanceUID"]].copy()

if "StudyInstanceUID" in sub.columns and len(sub) > 0:
    sub_ids = sub[["StudyInstanceUID"]].copy()
else:
    sub_ids = pred_df[["StudyInstanceUID"]].copy()

sub = sub_ids.merge(pred_df, on="StudyInstanceUID", how="left")

for col in target_cols:
    if col not in sub.columns:
        sub[col] = np.nan

sub[target_cols] = sub[target_cols].astype(np.float32).fillna(0.5)
sub = sub[required_cols]

out_path = "submission.csv"
sub.to_csv(out_path, index=False)
print("Wrote", out_path, "shape:", sub.shape)
print("Columns:", list(sub.columns))
print(sub.head())
