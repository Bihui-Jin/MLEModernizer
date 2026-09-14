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

0.9397651414001412

# 6. Current score

0.54377

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.49167) has done: 'I fix the protobuf/TensorFlow import crash by removing the `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` override (it triggers the `MessageFactory.GetPrototype` error with the installed protobuf version). Then I make the dataset construction robust by discovering TFRecords if present, otherwise falling back to reading the provided test JPGs directly, so `test_data` is always defined. Finally, I ensure the submission matches `train.csv`’s 11 target columns (the bundled sample has only 10) and that predictions are aligned to the sample submission’s `StudyInstanceUID` order, producing a valid `submission.csv`.'
- What this solution (achieved 0.5793) has done: 'I fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf implementation (which avoids the `MessageFactory.GetPrototype` AttributeError in this environment) and setting it before importing TensorFlow. I also correct the test TFRecord directory path to be optional (so it won’t break if TFRecords aren’t present) and keep the existing robust JPG fallback. Finally, I keep the model/inference logic intact but make the submission schema strictly match the 11 target columns from `train.csv` regardless of the provided sample submission’s missing columns, ensuring a valid `submission.csv` is always written.'
- What this solution (achieved 0.48008) has done: 'I fix the crash happening before TensorFlow import by setting the protobuf implementation in a way that works with TensorFlow 2.18 + protobuf 6.x, and I make that selection automatic (python vs upb) so the notebook runs in this environment. I also correct the TFRecord parsing to decode JPEG bytes (the competition TFRecords store JPEG), and fix the base model input shape ordering (height, width) to match how images are resized, both of which are correctness fixes that should improve the score toward your target without changing the overall modeling approach. Finally, I make the submission schema always include all 11 target columns (adding any missing ones from the provided sample) and write `submission.csv` end-to-end.'
- What this solution (achieved 0.54594) has done: 'I fix the TensorFlow/protobuf import crash by defaulting to the `upb` protobuf backend (which is the compatible default for protobuf 6.x) and only falling back to `python` if needed, before importing TensorFlow. I also correct the submission schema to always include all 11 target columns (the provided sample has only 10), and ensure the test UID ordering matches the sample submission exactly. Finally, I keep your model, weights, TTA, and inference logic intact, only making execution-stability changes so the notebook runs end-to-end and writes a valid `submission.csv`.'
- What this solution (achieved 0.46776) has done: 'I fix the protobuf/TensorFlow import crash by forcing the pure-Python protobuf backend *before* any TensorFlow import, which avoids the `MessageFactory.GetPrototype` error in this environment. I also make the submission schema always match the 11 target columns from `train.csv` (the provided sample file in this environment is missing 2 columns), and keep the test UID ordering exactly aligned to the sample submission. Finally, I keep your model, TTA, and inference logic intact, only making stability/correctness fixes so the notebook runs end-to-end and writes a valid `submission.csv`.'
- What this solution (achieved 0.49011) has done: 'I fix the immediate crash by making the TensorFlow/protobuf backend selection robust for TF 2.18 + protobuf 6.x (try default `upb`, fall back to pure-Python only if needed) and ensuring the env var is set before importing TensorFlow. I also correct the submission schema to always include all 11 target columns (your bundled sample file is missing some columns) and align predictions to the sample submission UID order exactly. These changes are execution/correctness fixes (not architectural changes) and should move the score substantially upward from the current near-random level by ensuring the intended pretrained weights can actually load and the model outputs map to the correct labels. The rest of the model, TTA, and inference loop remain the same.'
- What this solution (achieved 0.54377) has done: 'I fix the immediate TensorFlow/protobuf import crash by forcing the pure-Python protobuf implementation *before* importing TensorFlow (this is the only reliable workaround for the `MessageFactory.GetPrototype` error with TF 2.18 + protobuf 6.x in this environment). Then I correct the submission schema to always include all 11 target columns in the exact `train.csv` order, regardless of the incomplete `sample_submission.csv` bundled here (your current sample has only 10 columns, which can silently misalign evaluation). Finally, I keep your model/TTA/inference logic intact, but make the UID decoding robust for both TFRecord and JPG pipelines so every test UID gets a prediction and `submission.csv` is always written.'

# 9. Code solution

## === cell 0
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import tensorflow as tf  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

W = H = 456
N_CLASSES = 11
autotune = tf.data.experimental.AUTOTUNE

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

test_tfrecords_dir = "../input/ranzcr-clip-catheter-line-classification/test_tfrecords"
weight_dir = "../input/cassava2020weights"
test_images_dir = "../input/ranzcr-clip-catheter-line-classification/test"


def _safe_list_tfrecords(tfrec_dir):
    if os.path.isdir(tfrec_dir):
        return sorted(
            [
                f
                for f in os.listdir(tfrec_dir)
                if f.endswith(".tfrec") or f.endswith(".tfrecord")
            ]
        )
    return []


test_tfrecords = _safe_list_tfrecords(test_tfrecords_dir)

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
print(
    "Protobuf impl:",
    os.environ.get("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "default"),
)
print("Found test tfrecords:", len(test_tfrecords))
print("Test images dir exists:", os.path.isdir(test_images_dir))




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def parse_example(sample):
    sample = tf.io.parse_single_example(sample, features)
    image = tf.image.decode_jpeg(sample["image"], channels=3)
    image = tf.image.resize(image, (H, W))
    image_id = sample["StudyInstanceUID"]
    return image, image_id


def preprocess(images, labels):
    images = tf.cast(images, tf.float32)
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
def _build_test_dataset_from_tfrecords(tfrec_dir, tfrec_files):
    files = [os.path.join(tfrec_dir, c) for c in tfrec_files]
    ds = tf.data.TFRecordDataset(files, num_parallel_reads=autotune)
    ds = ds.map(parse_example, num_parallel_calls=autotune)
    ds = ds.map(preprocess, num_parallel_calls=autotune)
    ds = ds.batch(1).prefetch(autotune)
    return ds


def _decode_jpg(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, (H, W))
    img = tf.cast(img, tf.float32)
    return img


def _build_test_dataset_from_images(image_dir, uids_in_order):
    paths = [os.path.join(image_dir, f"{uid}.jpg") for uid in uids_in_order]
    path_ds = tf.data.Dataset.from_tensor_slices(paths)
    uid_ds = tf.data.Dataset.from_tensor_slices(uids_in_order)

    img_ds = path_ds.map(_decode_jpg, num_parallel_calls=autotune)
    ds = tf.data.Dataset.zip((img_ds, uid_ds))
    ds = ds.batch(1).prefetch(autotune)
    return ds


sample_path = "../input/ranzcr-clip-catheter-line-classification/sample_submission.csv"
sub_df = pd.read_csv(sample_path)

for col in target_cols:
    if col not in sub_df.columns:
        sub_df[col] = 0.0

test_uids = sub_df["StudyInstanceUID"].astype(str).tolist()

if len(test_tfrecords) > 0:
    test_data = _build_test_dataset_from_tfrecords(test_tfrecords_dir, test_tfrecords)
    for img, uid in test_data.take(1):
        print(
            "Using TFRecords. One batch image shape:",
            img.shape,
            "uid shape:",
            uid.shape,
        )
else:
    if not os.path.isdir(test_images_dir):
        raise FileNotFoundError(
            f"Neither TFRecords found in {test_tfrecords_dir} nor test image dir exists at {test_images_dir}"
        )
    test_data = _build_test_dataset_from_images(test_images_dir, test_uids)
    for img, uid in test_data.take(1):
        print("Using JPGs. One batch image shape:", img.shape, "uid shape:", uid.shape)




## === cell 3
def show_samples(dataset):
    rows = cols = 2
    fig = plt.figure(figsize=(12, 12))
    for i, (img, _) in enumerate(dataset.unbatch().shuffle(100).take(rows * cols)):
        fig.add_subplot(rows, cols, i + 1)
        plt.imshow(img / 255.0)
        plt.axis("off")
    plt.show()


try:
    show_samples(test_data)
except Exception as e:
    print("show_samples skipped:", e)



## === cell 4
base_mode, weight_path = model_map["efficientb5"]
model = get_model(base_mode, init_weight=weight_path)



## === cell 5
preds = []
image_ids = []


def TTA(image_batch):
    image = image_batch[0]
    tta = tf.stack([image, tf.image.flip_left_right(image)], axis=0)  # (2,H,W,3)
    return tta


def _to_uid(x):
    if isinstance(x, (bytes, np.bytes_)):
        return x.decode("utf-8")
    if hasattr(x, "numpy"):
        x = x.numpy()
        if isinstance(x, (bytes, np.bytes_)):
            return x.decode("utf-8")
        return str(x)
    return str(x)


c = 0
total = len(sub_df)

for image_b, img_id_b in test_data:
    img_id = _to_uid(img_id_b[0])
    image_ids.append(img_id)

    tta_imgs = TTA(image_b)
    pred = model.predict_on_batch(tta_imgs).tolist()
    pred = np.max(pred, axis=0)
    preds.append(pred)

    c += 1
    if c % 1000 == 0:
        print(c, f"/{total}")

preds = np.asarray(preds, dtype=np.float32)

pred_map = {uid: p for uid, p in zip(image_ids, preds)}
missing = [uid for uid in test_uids if uid not in pred_map]
if missing:
    raise RuntimeError(
        f"Missing predictions for {len(missing)} UIDs (e.g. {missing[:3]})"
    )

preds_ordered = np.stack([pred_map[uid] for uid in test_uids], axis=0)

if preds_ordered.shape[0] != len(sub_df):
    raise RuntimeError(
        f"Pred count mismatch: got {preds_ordered.shape[0]} preds, expected {len(sub_df)}"
    )

sub_df["StudyInstanceUID"] = test_uids
sub_df[target_cols] = preds_ordered

sub_path = "submission.csv"
sub_df[["StudyInstanceUID"] + target_cols].to_csv(sub_path, index=False)
print("Wrote:", sub_path, "shape:", sub_df[["StudyInstanceUID"] + target_cols].shape)
print(sub_df.head())
