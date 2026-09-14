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

0.7104867514736114

# 6. Current score

0.56778

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'I fix the protobuf-related crash by setting a compatible protobuf implementation mode before importing TensorFlow, which avoids the `MessageFactory`/`GetPrototype` error in this environment. Then I fix broken variables/paths (the code referenced a non-existent `test_tfrecords` due to the earlier crash and also points to TFRecords/weights that aren’t in your provided data tree). Finally, I keep the same EfficientNet + sigmoid multi-label prediction core, but switch the input pipeline to read the provided JPG test images (since they exist), and ensure the submission has all 11 required target columns by aligning to `train.csv` rather than the (incomplete) sample submission you inspected.'
- What this solution (achieved 0.56778) has done: 'I fix the TensorFlow/protobuf crash by setting the correct env var *before* importing TensorFlow and also setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION=2`, which is the common Kaggle-safe combination for TF 2.18 + protobuf 6.x. Then I fix the prediction loop error by removing the unnecessary `.numpy()` call, because `predict_on_batch` already returns a NumPy array in eager mode. Finally, to move the score up from ~0.5 toward your target, I add a minimal training step on `train.csv` using the same EfficientNet+sigmoid multi-label setup (no architecture/loss change), training only the head for 1 epoch to keep runtime reasonable and improve calibration away from constant-ish outputs.'

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
autotune = tf.data.AUTOTUNE

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

DATA_ROOT = "../input/ranzcr-clip-catheter-line-classification"
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train")
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test")

weight_path = "../input/cassava2020weights/ranzcr_efficientb5.h5"

print("TensorFlow:", tf.__version__)
print(
    "Train dir exists:",
    os.path.isdir(TRAIN_IMG_DIR),
    "num files:",
    len(os.listdir(TRAIN_IMG_DIR)) if os.path.isdir(TRAIN_IMG_DIR) else 0,
)
print(
    "Test dir exists:",
    os.path.isdir(TEST_IMG_DIR),
    "num files:",
    len(os.listdir(TEST_IMG_DIR)) if os.path.isdir(TEST_IMG_DIR) else 0,
)
print("Train CSV exists:", tf.io.gfile.exists(TRAIN_CSV))




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def triple_image(image):
    return tf.concat([image] * 3, axis=-1)


def load_jpg(image_path):
    img_bytes = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img_bytes, channels=1)  # grayscale
    img = tf.image.resize(img, (H, W))
    img = triple_image(img)  # -> 3 channels
    img = tf.cast(img, tf.float32) / 255.0
    return img


def parse_path(image_path):
    uid = tf.strings.regex_replace(
        tf.strings.split(image_path, os.sep)[-1], r"\.jpg$", ""
    )
    img = load_jpg(image_path)
    return img, uid


def get_model(
    baseline_weight=None, init_weight=None, lr=0.001, optimizer=tf.optimizers.Adam
):
    base_model = tf.keras.applications.EfficientNetB7(
        include_top=False, input_shape=(W, H, 3), pooling="avg", weights=baseline_weight
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

    if init_weight and tf.io.gfile.exists(init_weight):
        try:
            model.load_weights(init_weight)
            print(f"Weight loaded from {init_weight}")
        except Exception as e:
            print(f"Load weight from {init_weight} failed, {e}")
    else:
        print(
            "Init weight not found; using baseline ImageNet weights only."
            if baseline_weight
            else "No weights provided; using random/init defaults."
        )
    return model




## === cell 2
train_df = pd.read_csv(TRAIN_CSV)
train_df["image_path"] = train_df["StudyInstanceUID"].apply(
    lambda x: os.path.join(TRAIN_IMG_DIR, f"{x}.jpg")
)

exists_mask = train_df["image_path"].apply(tf.io.gfile.exists)
train_df = train_df.loc[exists_mask].reset_index(drop=True)

y = train_df[target_cols].astype(np.float32).values
x_paths = train_df["image_path"].values

print("Train rows (with images):", len(train_df))

train_ds = tf.data.Dataset.from_tensor_slices((x_paths, y))


def parse_train(path, label):
    img = load_jpg(path)
    return img, label


train_ds = train_ds.shuffle(2048, seed=42, reshuffle_each_iteration=True)
train_ds = train_ds.map(parse_train, num_parallel_calls=autotune)
train_ds = train_ds.batch(16).prefetch(autotune)

model = get_model(baseline_weight="imagenet", init_weight=weight_path, lr=1e-3)

for layer in model.layers:
    if isinstance(layer, tf.keras.Model):  # base model itself is a Model in some graphs
        layer.trainable = False
for layer in model.layers:
    if "efficientnet" in layer.name.lower():
        layer.trainable = False

model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=[tf.keras.metrics.AUC()],
)

history = model.fit(train_ds, epochs=1, verbose=1)



## === cell 3
test_paths = sorted(tf.io.gfile.glob(os.path.join(TEST_IMG_DIR, "*.jpg")))
if len(test_paths) == 0:
    raise FileNotFoundError(f"No test images found in {TEST_IMG_DIR}")

test_data = tf.data.Dataset.from_tensor_slices(test_paths)
test_data = test_data.map(parse_path, num_parallel_calls=autotune)
test_data = test_data.batch(16).prefetch(autotune)

sub = pd.DataFrame(
    {"StudyInstanceUID": [os.path.splitext(os.path.basename(p))[0] for p in test_paths]}
)
for c in target_cols:
    sub[c] = 0.0

preds = []
uids = []
for batch_imgs, batch_uids in test_data:
    batch_pred = model.predict_on_batch(batch_imgs)
    preds.append(batch_pred)
    uids.extend([u.decode("utf-8") for u in batch_uids.numpy().tolist()])

preds = np.concatenate(preds, axis=0)
pred_df = pd.DataFrame(preds, columns=target_cols)
pred_df.insert(0, "StudyInstanceUID", uids)

sub = sub.merge(pred_df, on="StudyInstanceUID", how="left", suffixes=("_base", ""))
for c in target_cols:
    if c + "_base" in sub.columns:
        sub.drop(columns=[c + "_base"], inplace=True)

sub[target_cols] = sub[target_cols].astype(np.float32).fillna(0.5)

print(sub.head())
print("Submission shape:", sub.shape)



## === cell 4
out_path = "submission.csv"
sub = sub[["StudyInstanceUID"] + target_cols]
sub.to_csv(out_path, index=False)
print("Wrote:", out_path, "rows:", len(sub), "cols:", len(sub.columns))
print(sub.columns.tolist())
