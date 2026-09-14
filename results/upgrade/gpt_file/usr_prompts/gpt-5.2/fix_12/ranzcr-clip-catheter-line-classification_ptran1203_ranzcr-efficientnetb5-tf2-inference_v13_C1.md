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

0.9030920693473292

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.50753) has done: 'I fix the TensorFlow/protobuf crash by removing the forced pure-Python protobuf setting that breaks TF 2.18 in this environment, and add a small compatibility guard so the notebook still runs if the variable is set externally. I also fix the inference error by treating `predict_on_batch` as already returning a NumPy array (so `.numpy()` is invalid) while keeping the model and pipeline unchanged. Finally, I ensure the submission matches the required `sample_submission.csv` columns even though the provided sample file is missing some targets, by backfilling the missing required columns from `train.csv` and defaulting any truly absent ones to 0.0, so a valid `submission.csv` is always produced end-to-end.'

# 9. Code solution

## === cell 0
import os

if os.environ.get("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "").lower() == "python":
    print(
        "Warning: PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python is set. "
        "This may cause TensorFlow/protobuf incompatibilities; consider unsetting it."
    )

import tensorflow as tf
import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import GroupShuffleSplit

tf.keras.utils.set_random_seed(42)
try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

W = H = 448
autotune = tf.data.AUTOTUNE

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

_CANDIDATE_DATA_DIRS = [
    "/kaggle/input/ranzcr-clip-catheter-line-classification",
    "/kaggle/data/ranzcr-clip-catheter-line-classification",
    "../input/ranzcr-clip-catheter-line-classification",
]
DATA_DIR = None
for d in _CANDIDATE_DATA_DIRS:
    if tf.io.gfile.exists(os.path.join(d, "train.csv")):
        DATA_DIR = d
        break
if DATA_DIR is None:
    raise FileNotFoundError(
        "Could not find ranzcr-clip-catheter-line-classification dataset directory. "
        f"Tried: {_CANDIDATE_DATA_DIRS}"
    )

TEST_IMG_DIR = os.path.join(DATA_DIR, "test")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train")

model_map = {
    "efficientb3": tf.keras.applications.EfficientNetB3,
    "efficientb5": tf.keras.applications.EfficientNetB5,
    "efficientb7": tf.keras.applications.EfficientNetB7,
}

print("TensorFlow:", tf.__version__)
print("DATA_DIR:", DATA_DIR)
print("Train image dir exists:", tf.io.gfile.exists(TRAIN_IMG_DIR))
print("Test image dir exists:", tf.io.gfile.exists(TEST_IMG_DIR))

try:
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception:
    pass




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
@tf.function
def _read_and_preprocess_image(path):
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img_bytes, channels=1)  # grayscale
    img = tf.image.resize(
        img, (H, W), method=tf.image.ResizeMethod.BILINEAR, antialias=False
    )
    img = tf.image.grayscale_to_rgb(img)  # H,W,3
    img = tf.image.convert_image_dtype(img, tf.float32)  # /255.0
    img = (img - mean) / std
    img.set_shape((H, W, 3))
    return img


@tf.function
def parse_jpg_from_path(path):
    image = _read_and_preprocess_image(path)
    uid = tf.strings.split(path, os.sep)[-1]
    uid = tf.strings.regex_replace(uid, r"\.jpg$", "")
    return image, uid


@tf.function
def parse_train_from_path(path, label_vec):
    image = _read_and_preprocess_image(path)
    label_vec = tf.cast(label_vec, tf.float32)
    return image, label_vec


def get_model(
    base_model,
    baseline_weight="imagenet",
    init_weight=None,
    lr=0.001,
    optimizer=tf.optimizers.Adam,
):
    base_model = base_model(
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
id_col = "StudyInstanceUID"
train_csv_path = os.path.join(DATA_DIR, "train.csv")
train_df = pd.read_csv(train_csv_path)

target_cols = [c for c in target_cols if c in train_df.columns]
N_CLASSES = len(target_cols)
if N_CLASSES != 11:
    print(
        f"Warning: expected 11 targets, found {N_CLASSES} in train.csv. Using available targets."
    )

train_df["path"] = TRAIN_IMG_DIR + "/" + train_df[id_col].astype(str) + ".jpg"

gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
train_idx, val_idx = next(gss.split(train_df, groups=train_df["PatientID"]))
tr_df = train_df.iloc[train_idx].reset_index(drop=True)
va_df = train_df.iloc[val_idx].reset_index(drop=True)

print("Train/val sizes:", len(tr_df), len(va_df))

y_tr = tr_df[target_cols].astype(np.float32).values
y_va = va_df[target_cols].astype(np.float32).values

BATCH_SIZE = 16

CACHE_DIR = "/kaggle/working/tfdata_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
TRAIN_CACHE = os.path.join(CACHE_DIR, "train_decoded.cache")
VAL_CACHE = os.path.join(CACHE_DIR, "val_decoded.cache")

options = tf.data.Options()
options.experimental_deterministic = True
options.experimental_optimization.map_parallelization = True
options.experimental_optimization.parallel_batch = True
options.experimental_optimization.map_fusion = True
options.experimental_optimization.autotune_buffers = True
options.experimental_optimization.autotune = True
try:
    cpu = os.cpu_count() or 8
    options.threading.private_threadpool_size = max(8, min(32, cpu))
    options.threading.max_intra_op_parallelism = 0
except Exception:
    pass

train_paths = tr_df["path"].values
val_paths = va_df["path"].values

train_path_ds = tf.data.Dataset.from_tensor_slices(train_paths).with_options(options)
train_label_ds = tf.data.Dataset.from_tensor_slices(y_tr).with_options(options)
train_ds = (
    tf.data.Dataset.zip((train_path_ds, train_label_ds))
    .shuffle(min(len(tr_df), 4096), seed=42, reshuffle_each_iteration=True)
    .map(parse_train_from_path, num_parallel_calls=autotune, deterministic=True)
    .cache(TRAIN_CACHE)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(autotune)
)

val_path_ds = tf.data.Dataset.from_tensor_slices(val_paths).with_options(options)
val_label_ds = tf.data.Dataset.from_tensor_slices(y_va).with_options(options)
val_ds = (
    tf.data.Dataset.zip((val_path_ds, val_label_ds))
    .map(parse_train_from_path, num_parallel_calls=autotune, deterministic=True)
    .cache(VAL_CACHE)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(autotune)
)

test_paths = tf.io.gfile.glob(os.path.join(TEST_IMG_DIR, "*.jpg"))
test_paths = sorted(test_paths)
if len(test_paths) == 0:
    raise FileNotFoundError(f"No test .jpg files found in: {TEST_IMG_DIR}")

test_path_ds = tf.data.Dataset.from_tensor_slices(test_paths).with_options(options)
test_data = (
    test_path_ds.map(
        parse_jpg_from_path, num_parallel_calls=autotune, deterministic=True
    )
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(autotune)
)

print("Test images:", len(test_paths))
print("Num targets:", N_CLASSES)




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_55/2652962043.py in <cell line: 0>()
     36 options.experimental_optimization.parallel_batch = True
     37 options.experimental_optimization.map_fusion = True
---> 38 options.experimental_optimization.autotune_buffers = True
     39 options.experimental_optimization.autotune = True
     40 try:

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/util/options.py in __setattr__(self, name, value)
     59       object.__setattr__(self, name, value)
     60     else:
---> 61       raise AttributeError("Cannot set the property {} on {}.".format(
     62           name,
     63           type(self).__name__))

AttributeError: Cannot set the property autotune_buffers on OptimizationOptions.

## === cell 3
base_mode = model_map["efficientb3"]
model = get_model(base_mode, baseline_weight="imagenet", init_weight=None, lr=1e-3)

for layer in model.layers:
    if isinstance(layer, tf.keras.Model) and "efficientnet" in layer.name.lower():
        layer.trainable = False

model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=[tf.keras.metrics.AUC()],
)

model.fit(train_ds, validation_data=val_ds, epochs=1, verbose=1)

for layer in model.layers:
    if isinstance(layer, tf.keras.Model) and "efficientnet" in layer.name.lower():
        layer.trainable = True

model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=[tf.keras.metrics.AUC()],
)

model.fit(train_ds, validation_data=val_ds, epochs=1, verbose=1)




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3182423277.py in <cell line: 0>()
     12 )
     13 
---> 14 model.fit(train_ds, validation_data=val_ds, epochs=1, verbose=1)
     15 
     16 for layer in model.layers:

NameError: name 'train_ds' is not defined

## === cell 4
preds = model.predict(test_data, verbose=1)
preds = np.asarray(preds, dtype=np.float32)

image_ids = np.char.replace(
    np.array([os.path.basename(p) for p in test_paths], dtype=str), ".jpg", ""
).tolist()

assert (
    preds.shape[0] == len(image_ids) == len(test_paths)
), "Prediction/ID count mismatch"
assert (
    preds.shape[1] == N_CLASSES
), f"Expected {N_CLASSES} outputs, got {preds.shape[1]}"

sub_cols = [id_col] + target_cols
sub = pd.DataFrame({id_col: image_ids})
for j, c in enumerate(target_cols):
    sub[c] = preds[:, j].astype(np.float32)

sub = sub[sub_cols]
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3075031468.py in <cell line: 0>()
----> 1 preds = model.predict(test_data, verbose=1)
      2 preds = np.asarray(preds, dtype=np.float32)
      3 
      4 # Speed-up (correctness-preserving): vectorized UID extraction (avoids Python loop over 3009 items).
      5 # Produces identical IDs as prior rsplit/[:-4].

NameError: name 'test_data' is not defined
