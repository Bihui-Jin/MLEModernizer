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

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import GroupShuffleSplit

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

mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
std = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)

BASE_DIR = "../input/ranzcr-clip-catheter-line-classification"
TEST_IMG_DIR = os.path.join(BASE_DIR, "test")
TRAIN_IMG_DIR = os.path.join(BASE_DIR, "train")
sample_path = os.path.join(BASE_DIR, "sample_submission.csv")
train_csv_path = os.path.join(BASE_DIR, "train.csv")

weight_path = "../input/cassava2020weights/ranzcr_efficientb5.h5"

tf.random.set_seed(42)
np.random.seed(42)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

print("TF:", tf.__version__)
print("Sample path exists:", os.path.exists(sample_path))
print("Train CSV exists:", os.path.exists(train_csv_path))
print("Train dir exists:", os.path.isdir(TRAIN_IMG_DIR))
print("Test dir exists:", os.path.isdir(TEST_IMG_DIR))
print("External weight path exists:", os.path.exists(weight_path))

CACHE_DIR = "/kaggle/working/tfdata_cache"
os.makedirs(CACHE_DIR, exist_ok=True)




## === cell 1
def triple_image(image):
    return tf.concat([image] * 3, axis=-1)


@tf.function(reduce_retracing=True)
def decode_resize_and_preprocess_jpeg(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=1)  # grayscale
    img = tf.image.resize(img, (H, W), method="bilinear")
    img = tf.cast(img, tf.float32)
    img = triple_image(img)  # (H, W, 3)
    img = img / 255.0
    img = (img - mean) / std
    return img


@tf.function(reduce_retracing=True)
def parse_train_and_preprocess(path, label_vec):
    image = decode_resize_and_preprocess_jpeg(path)
    return image, tf.cast(label_vec, tf.float32)


@tf.function(reduce_retracing=True)
def parse_path_and_preprocess(path):
    fname = tf.strings.split(path, os.sep)[-1]
    uid = tf.strings.regex_replace(fname, r"\.jpg$", "")
    image = decode_resize_and_preprocess_jpeg(path)
    return image, uid


class WeightedBinaryCrossentropy(tf.keras.losses.Loss):
    def __init__(self, pos_weight, name="weighted_bce"):
        super().__init__(name=name)
        self.pos_weight = tf.constant(pos_weight, dtype=tf.float32)
        self.eps = tf.constant(1e-7, dtype=tf.float32)

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(tf.cast(y_pred, tf.float32), self.eps, 1.0 - self.eps)
        w = 1.0 + y_true * (self.pos_weight - 1.0)
        bce = -(
            y_true * tf.math.log(y_pred) + (1.0 - y_true) * tf.math.log(1.0 - y_pred)
        )
        return tf.reduce_mean(w * bce)


def get_model(
    baseline_weight=None,
    init_weight=None,
    lr=0.001,
    optimizer=tf.optimizers.Adam,
    loss_obj="binary_crossentropy",
    metrics=None,
):
    base_model = tf.keras.applications.EfficientNetB7(
        include_top=False, input_shape=(H, W, 3), pooling="avg", weights=baseline_weight
    )
    base_out = base_model.output
    out = tf.keras.layers.Dropout(0.3)(base_out)
    out = tf.keras.layers.Dense(N_CLASSES, activation="sigmoid")(out)
    model = tf.keras.models.Model(inputs=base_model.input, outputs=out)

    model.compile(
        optimizer=optimizer(learning_rate=lr),
        loss=loss_obj,
        metrics=metrics,
    )

    if init_weight:
        try:
            model.load_weights(init_weight)
            print(f"Weight loaded from {init_weight}")
        except Exception as e:
            print(f"Load weight from {init_weight} failed, {e}")
    return model




## === cell 2
train_df = pd.read_csv(train_csv_path)

for c in target_cols:
    if c not in train_df.columns:
        train_df[c] = 0

train_df["img_path"] = (
    TRAIN_IMG_DIR + os.sep + train_df["StudyInstanceUID"].astype(str) + ".jpg"
)

X_paths = train_df["img_path"].values.astype(str)
y = train_df[target_cols].values.astype(np.float32)
groups = train_df["PatientID"].values

gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
tr_idx, va_idx = next(gss.split(X_paths, y, groups=groups))

train_paths, val_paths = X_paths[tr_idx], X_paths[va_idx]
train_y, val_y = y[tr_idx], y[va_idx]

pos = train_y.mean(axis=0)
neg = 1.0 - pos
pos_weight = (neg / np.clip(pos, 1e-6, 1.0)).astype(np.float32)
pos_weight = np.clip(pos_weight, 1.0, 20.0)
print("Per-label pos_weight (clipped):", dict(zip(target_cols, pos_weight.round(3))))

BATCH_SIZE = 32

options = tf.data.Options()
options.deterministic = False
try:
    options.threading.max_intra_op_parallelism = 0
    options.threading.private_threadpool_size = 0
except Exception:
    pass

train_cache_path = os.path.join(CACHE_DIR, "train_cache")
val_cache_path = os.path.join(CACHE_DIR, "val_cache")

train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_y)).with_options(
    options
)
train_ds = train_ds.shuffle(
    min(4096, len(train_paths)), seed=42, reshuffle_each_iteration=True
)
train_ds = train_ds.map(parse_train_and_preprocess, num_parallel_calls=autotune)
train_ds = train_ds.ignore_errors()
train_ds = train_ds.cache(train_cache_path)
train_ds = train_ds.batch(BATCH_SIZE, drop_remainder=False)
train_ds = train_ds.prefetch(autotune)

val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_y)).with_options(options)
val_ds = val_ds.map(parse_train_and_preprocess, num_parallel_calls=autotune)
val_ds = val_ds.ignore_errors()
val_ds = val_ds.cache(val_cache_path)
val_ds = val_ds.batch(BATCH_SIZE, drop_remainder=False)
val_ds = val_ds.prefetch(autotune)

baseline = "imagenet"
init_w = None
if isinstance(weight_path, str) and os.path.exists(weight_path):
    baseline = None
    init_w = weight_path
    print("Using provided weights:", weight_path)
else:
    print("Using EfficientNetB7 ImageNet weights (will fit head + brief fine-tune).")

loss_obj = WeightedBinaryCrossentropy(pos_weight=pos_weight)

model = get_model(
    baseline_weight=baseline,
    init_weight=init_w,
    lr=0.001,
    loss_obj=loss_obj,
    metrics=None,
)

base_model = model.get_layer(index=1) if len(model.layers) > 1 else None
if isinstance(base_model, tf.keras.Model) and base_model.name.startswith(
    "efficientnet"
):
    base_model.trainable = False
else:
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and layer.name.startswith("efficientnet"):
            layer.trainable = False
            break

model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.001),
    loss=loss_obj,
    metrics=None,
)

steps_per_epoch = min(800, max(1, len(train_paths) // BATCH_SIZE))
print("steps_per_epoch:", steps_per_epoch)

history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=1,
    steps_per_epoch=steps_per_epoch,
    verbose=2,
)

model.trainable = True
model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=2e-4),
    loss=loss_obj,
    metrics=None,
)

history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=1,
    steps_per_epoch=steps_per_epoch,
    verbose=2,
)

auc_metric_model = tf.keras.metrics.AUC(multi_label=True, num_labels=N_CLASSES)
model.compile(optimizer=model.optimizer, loss=loss_obj, metrics=[auc_metric_model])
_ = model.evaluate(val_ds, verbose=2)




## === cell 3
sub_df = pd.read_csv(sample_path)

if "StudyInstanceUID" not in sub_df.columns:
    raise ValueError("sample_submission.csv must contain StudyInstanceUID")

sub_df = sub_df[["StudyInstanceUID"]].copy()
for c in target_cols:
    sub_df[c] = 0.0
sub_df = sub_df[["StudyInstanceUID"] + target_cols]

uids = sub_df["StudyInstanceUID"].astype(str).to_numpy()
test_paths = (TEST_IMG_DIR + os.sep + uids + ".jpg").astype(str)

test_cache_path = os.path.join(CACHE_DIR, "test_cache")

ds = tf.data.Dataset.from_tensor_slices(test_paths).with_options(options)
ds = ds.map(parse_path_and_preprocess, num_parallel_calls=autotune)
ds = ds.ignore_errors()
ds = ds.cache(test_cache_path)
ds = ds.batch(BATCH_SIZE, drop_remainder=False)
ds = ds.prefetch(autotune)

preds = model.predict(ds, verbose=0)

pred_df = pd.DataFrame(preds, columns=target_cols)
pred_df["StudyInstanceUID"] = uids[: len(pred_df)]

sub_out = sub_df[["StudyInstanceUID"]].merge(pred_df, on="StudyInstanceUID", how="left")
sub_out[target_cols] = sub_out[target_cols].astype(np.float32).fillna(0.5)

sub_out = sub_out[["StudyInstanceUID"] + target_cols]
sub_out.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", sub_out.shape)
print("Columns:", list(sub_out.columns))
print(sub_out.head())
