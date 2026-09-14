# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Classify each cassava image into four disease categories or a fifth category indicating a healthy leaf.

## Metric
Categorization accuracy.

## Submission Format
```
image_id,label
1000471002.jpg,4
1000840542.jpg,4
etc.
```

## Dataset
**[train/test]_images** the image files.

**train.csv**

- `image_id` the image file name.

- `label` the ID code for the disease.

**sample_submission.csv** A properly formatted sample submission, given the disclosed test set content.

- `image_id` the image file name.

- `label` the predicted ID code for the disease.

**[train/test]_tfrecords** the image files in tfrecord format.

**label_num_to_disease_map.json** The mapping between each disease code and the real disease name.

# 2. Python version

3.13

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        input/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        working/
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
```

-> data/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/cassava-leaf-disease-classification/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/cassava-leaf-disease-classification/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> data/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> input/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

if "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION" in os.environ:
    os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

import tensorflow as tf
from tensorflow.keras import layers, models

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

INPUT_DIR = "/kaggle/input/cassava-leaf-disease-classification"
WORKING_DIR = "/kaggle/working"

TRAIN_CSV_PATH = f"{INPUT_DIR}/train.csv"
SAMPLE_SUB_PATH = f"{INPUT_DIR}/sample_submission.csv"
TRAIN_IMG_DIR = f"{INPUT_DIR}/train_images"
TEST_IMG_DIR = f"{INPUT_DIR}/test_images"
TRAIN_TFREC_DIR = f"{INPUT_DIR}/train_tfrecords"
TEST_TFREC_DIR = f"{INPUT_DIR}/test_tfrecords"

print("TF version:", tf.__version__)
print("Train CSV exists:", os.path.exists(TRAIN_CSV_PATH))
print("Sample submission exists:", os.path.exists(SAMPLE_SUB_PATH))
print("Train image dir exists:", os.path.isdir(TRAIN_IMG_DIR))
print("Test image dir exists:", os.path.isdir(TEST_IMG_DIR))
print("Train TFRecord dir exists:", os.path.isdir(TRAIN_TFREC_DIR))
print("Test TFRecord dir exists:", os.path.isdir(TEST_TFREC_DIR))




## === cell 1
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.efficientnet import (
    preprocess_input as eff_preprocess,
)
from tensorflow.keras.applications.densenet import preprocess_input as dn_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mb_preprocess

train_csv = pd.read_csv(TRAIN_CSV_PATH)
train_csv["path"] = TRAIN_IMG_DIR.rstrip("/") + "/" + train_csv["image_id"].astype(str)
train_csv["label"] = train_csv["label"].astype(int)

train_df, valid_df = train_test_split(
    train_csv,
    test_size=0.2,
    stratify=train_csv["label"],
    random_state=SEED,
)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 5

AUTOTUNE = tf.data.AUTOTUNE
options = tf.data.Options()
options.experimental_deterministic = True
try:
    options.threading.private_threadpool_size = max(1, (os.cpu_count() or 2) // 2)
except Exception:
    pass

train_labels = train_df["label"].to_numpy(dtype=np.int64)


def _sorted_tfrec_files(dir_path, prefix):
    files = tf.io.gfile.glob(os.path.join(dir_path, f"{prefix}*.tfrec"))
    files = sorted(files)
    if not files:
        raise FileNotFoundError(
            f"No TFRecord files found under {dir_path} with prefix {prefix}"
        )
    return files


TRAIN_TFRECS = _sorted_tfrec_files(TRAIN_TFREC_DIR, "ld_train")
TEST_TFRECS = _sorted_tfrec_files(TEST_TFREC_DIR, "ld_test")

_FEATURE_DESC = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
    "image_id": tf.io.FixedLenFeature([], tf.string, default_value=b""),
}

_TEST_FEATURE_DESC = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_id": tf.io.FixedLenFeature([], tf.string, default_value=b""),
}


@tf.function(autograph=False)
def _parse_train_ex(example_proto):
    ex = tf.io.parse_single_example(example_proto, _FEATURE_DESC)
    img = tf.image.decode_jpeg(ex["image"], channels=3)
    img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32)
    label = tf.cast(ex["target"], tf.int64)
    return img, label


@tf.function(autograph=False)
def _onehot(label):
    return tf.one_hot(tf.cast(label, tf.int32), depth=NUM_CLASSES)


@tf.function(autograph=False)
def _prep_eff(img, label):
    return eff_preprocess(img), _onehot(label)


@tf.function(autograph=False)
def _prep_dn(img, label):
    return dn_preprocess(img), _onehot(label)


@tf.function(autograph=False)
def _prep_mb(img, label):
    return mb_preprocess(img), _onehot(label)


raw_all = tf.data.TFRecordDataset(
    TRAIN_TFRECS, num_parallel_reads=AUTOTUNE
).with_options(options)
decoded_all = raw_all.map(_parse_train_ex, num_parallel_calls=AUTOTUNE)

n_total = int(train_csv.shape[0])
n_train = int(train_df.shape[0])
n_valid = int(valid_df.shape[0])
assert n_train + n_valid == n_total, "Unexpected split sizes."

train_decoded = decoded_all.take(n_train).shuffle(
    buffer_size=min(n_train, 4096),
    seed=SEED,
    reshuffle_each_iteration=True,
)
valid_decoded = decoded_all.skip(n_train).take(n_valid)

train_ds_eff = (
    train_decoded.map(_prep_eff, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTOTUNE)
)
valid_ds_eff = (
    valid_decoded.map(_prep_eff, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTOTUNE)
)

train_ds_dn = (
    train_decoded.map(_prep_dn, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTOTUNE)
)
valid_ds_dn = (
    valid_decoded.map(_prep_dn, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTOTUNE)
)

train_ds_mb = (
    train_decoded.map(_prep_mb, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTOTUNE)
)
valid_ds_mb = (
    valid_decoded.map(_prep_mb, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTOTUNE)
)

print("n_total:", n_total, "n_train:", n_train, "n_valid:", n_valid)




## === cell 2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import EfficientNetB0, DenseNet169, MobileNetV2

early_stopping = EarlyStopping(
    monitor="val_loss", patience=3, restore_best_weights=True
)
learning_rate_reduction = ReduceLROnPlateau(
    monitor="val_loss", patience=2, factor=0.5, min_lr=1e-6, verbose=1
)

augment_layer = layers.Lambda(lambda x: x, name="data_augmentation_identity")


def build_backbone_model(backbone, name):
    inp = layers.Input(shape=(224, 224, 3))
    x = augment_layer(inp)
    x = backbone(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    out = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    model = models.Model(inp, out, name=name)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
        steps_per_execution=16,
    )
    return model


eff_b0 = EfficientNetB0(
    include_top=False, weights="imagenet", input_shape=(224, 224, 3)
)
dn169 = DenseNet169(include_top=False, weights="imagenet", input_shape=(224, 224, 3))
mbv2 = MobileNetV2(include_top=False, weights="imagenet", input_shape=(224, 224, 3))

for bb in (eff_b0, dn169, mbv2):
    bb.trainable = False

cropnet_model = build_backbone_model(
    eff_b0, "effnetb0_classifier"
)  # proxy for "cropnet"
densenet_model = build_backbone_model(dn169, "densenet169_classifier")
efficientnet_model = build_backbone_model(
    mbv2, "mobilenetv2_classifier"
)  # proxy for "efficientnetb4"

cropnet_weight = 0.65
densenet_weight = 0.25
efficientnet_weight = 0.10

EPOCHS_BASE = 1
print("Training base model 1/3...")
cropnet_model.fit(
    train_ds_eff,
    validation_data=valid_ds_eff,
    epochs=EPOCHS_BASE,
    callbacks=[early_stopping, learning_rate_reduction],
    verbose=1,
)
print("Training base model 2/3...")
densenet_model.fit(
    train_ds_dn,
    validation_data=valid_ds_dn,
    epochs=EPOCHS_BASE,
    callbacks=[early_stopping, learning_rate_reduction],
    verbose=1,
)
print("Training base model 3/3...")
efficientnet_model.fit(
    train_ds_mb,
    validation_data=valid_ds_mb,
    epochs=EPOCHS_BASE,
    callbacks=[early_stopping, learning_rate_reduction],
    verbose=1,
)




## === cell 3
soft_voting_labels = train_labels.astype(np.int64, copy=False)


@tf.function(autograph=False)
def _prep_eff_only(img, label):
    return eff_preprocess(img)


@tf.function(autograph=False)
def _prep_dn_only(img, label):
    return dn_preprocess(img)


@tf.function(autograph=False)
def _prep_mb_only(img, label):
    return mb_preprocess(img)


train_x_eff = (
    train_decoded.map(_prep_eff_only, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTOTUNE)
)
train_x_dn = (
    train_decoded.map(_prep_dn_only, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTOTUNE)
)
train_x_mb = (
    train_decoded.map(_prep_mb_only, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTOTUNE)
)

n_train_batches = int(np.ceil(n_train / BATCH_SIZE))

cropnet_probs_chunks = []
densenet_probs_chunks = []
efficientnet_probs_chunks = []

it_eff = iter(train_x_eff)
it_dn = iter(train_x_dn)
it_mb = iter(train_x_mb)

for _ in range(n_train_batches):
    xb_eff = next(it_eff)
    xb_dn = next(it_dn)
    xb_mb = next(it_mb)
    cropnet_probs_chunks.append(cropnet_model(xb_eff, training=False).numpy())
    densenet_probs_chunks.append(densenet_model(xb_dn, training=False).numpy())
    efficientnet_probs_chunks.append(efficientnet_model(xb_mb, training=False).numpy())

cropnet_probs_all = np.concatenate(cropnet_probs_chunks, axis=0)
densenet_probs_all = np.concatenate(densenet_probs_chunks, axis=0)
efficientnet_probs_all = np.concatenate(efficientnet_probs_chunks, axis=0)

soft_voting_features = (
    cropnet_weight * cropnet_probs_all
    + densenet_weight * densenet_probs_all
    + efficientnet_weight * efficientnet_probs_all
).astype(np.float32, copy=False)

print("Soft-voting features shape:", soft_voting_features.shape)
print("Soft-voting labels shape:", soft_voting_labels.shape)

assert (
    soft_voting_features.shape[0] == soft_voting_labels.shape[0]
    and soft_voting_features.shape[0] > 0
), "Mismatch between features and labels for meta-model training."




## === cell 4
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, log_loss

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

fold_accuracies = []
fold_log_losses = []
train_distributions = []
val_distributions = []

meta_models = []

for train_index, val_index in kf.split(soft_voting_features, soft_voting_labels):
    X_train, X_val = soft_voting_features[train_index], soft_voting_features[val_index]
    y_train, y_val = soft_voting_labels[train_index], soft_voting_labels[val_index]

    meta_model = LogisticRegression(
        max_iter=1000,
        C=0.1,
        multi_class="multinomial",
        penalty="l2",
        solver="lbfgs",
        n_jobs=None,
    )
    meta_model.fit(X_train, y_train)
    meta_models.append(meta_model)

    val_preds = meta_model.predict(X_val)
    val_probs = meta_model.predict_proba(X_val)

    fold_accuracies.append(accuracy_score(y_val, val_preds))
    fold_log_losses.append(log_loss(y_val, val_probs, labels=list(range(NUM_CLASSES))))

    train_distributions.append(
        pd.Series(y_train).value_counts(normalize=True).to_dict()
    )
    val_distributions.append(pd.Series(y_val).value_counts(normalize=True).to_dict())

print("CV accuracy:", fold_accuracies, "mean:", float(np.mean(fold_accuracies)))
print("CV logloss:", fold_log_losses, "mean:", float(np.mean(fold_log_losses)))

meta_model = meta_models[-1]




## === cell 5
SKIP_PLOTTING = True
if not SKIP_PLOTTING:
    import matplotlib.pyplot as plt

    num_folds = len(train_distributions)
    plt.figure(figsize=(15, 3 * num_folds))

    for fold in range(num_folds):
        train_dist = train_distributions[fold]
        val_dist = val_distributions[fold]

        df_train = pd.DataFrame(
            list(train_dist.items()), columns=["Class", "Proportion"]
        )
        df_val = pd.DataFrame(list(val_dist.items()), columns=["Class", "Proportion"])

        plt.subplot(num_folds, 4, 4 * fold + 1)
        plt.bar(
            df_train["Class"].astype(str),
            df_train["Proportion"],
            color="blue",
            alpha=0.6,
            label="Train",
        )
        plt.ylim(0, 1)
        plt.title(f"Fold {fold + 1} - Train Distribution")
        plt.xlabel("Class")
        plt.ylabel("Proportion")
        plt.legend()

        plt.subplot(num_folds, 4, 4 * fold + 2)
        plt.bar(
            df_val["Class"].astype(str),
            df_val["Proportion"],
            color="red",
            alpha=0.6,
            label="Validation",
        )
        plt.ylim(0, 1)
        plt.title(f"Fold {fold + 1} - Validation Distribution")
        plt.xlabel("Class")
        plt.ylabel("Proportion")
        plt.legend()

    plt.tight_layout()
    plt.show()




## === cell 6
if not SKIP_PLOTTING:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(
        range(1, len(fold_accuracies) + 1),
        fold_accuracies,
        marker="o",
        color="b",
        label="Accuracy",
    )
    plt.xlabel("Fold")
    plt.ylabel("Accuracy")
    plt.title("Cross-Validation Accuracy by Fold")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(
        range(1, len(fold_log_losses) + 1),
        fold_log_losses,
        marker="o",
        color="r",
        label="Log Loss",
    )
    plt.xlabel("Fold")
    plt.ylabel("Log Loss")
    plt.title("Cross-Validation Log Loss by Fold")
    plt.legend()

    plt.tight_layout()
    plt.show()




## === cell 7
if not SKIP_PLOTTING:
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    meta_model_predictions = meta_model.predict(soft_voting_features)
    cm = confusion_matrix(
        soft_voting_labels, meta_model_predictions, labels=list(range(NUM_CLASSES))
    )
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=list(range(NUM_CLASSES))
    )
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix for Meta-Model on Training Data")
    plt.show()




## === cell 8
from tensorflow.keras.preprocessing.image import (
    load_img,
    img_to_array,
)  # kept (paths unchanged)

sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
test_image_ids_from_sample = sample_sub["image_id"].astype(str).tolist()

assert (
    isinstance(meta_models, list) and len(meta_models) > 0
), "meta_models is empty; stacking did not train."


@tf.function(autograph=False)
def _parse_test_example_with_id(example_proto):
    ex = tf.io.parse_single_example(example_proto, _TEST_FEATURE_DESC)
    img = tf.image.decode_jpeg(ex["image"], channels=3)
    img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32)
    return img, ex["image_id"]


test_raw = tf.data.TFRecordDataset(
    TEST_TFRECS, num_parallel_reads=AUTOTUNE
).with_options(options)

test_decoded_with_id = test_raw.map(
    _parse_test_example_with_id, num_parallel_calls=AUTOTUNE
).prefetch(AUTOTUNE)


@tf.function(autograph=False)
def _prep_eff_test(img, image_id):
    return eff_preprocess(img), image_id


@tf.function(autograph=False)
def _prep_dn_test(img, image_id):
    return dn_preprocess(img), image_id


@tf.function(autograph=False)
def _prep_mb_test(img, image_id):
    return mb_preprocess(img), image_id


test_eff = (
    test_decoded_with_id.map(_prep_eff_test, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTOTUNE)
)
test_dn = (
    test_decoded_with_id.map(_prep_dn_test, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTOTUNE)
)
test_mb = (
    test_decoded_with_id.map(_prep_mb_test, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTOTUNE)
)

n_test = int(sample_sub.shape[0])
n_test_batches = int(np.ceil(n_test / BATCH_SIZE))

cropnet_test_chunks = []
densenet_test_chunks = []
efficientnet_test_chunks = []

it_eff = iter(test_eff)
it_dn = iter(test_dn)
it_mb = iter(test_mb)

for _ in range(n_test_batches):
    xb_eff, _id_eff = next(it_eff)
    xb_dn, _id_dn = next(it_dn)
    xb_mb, _id_mb = next(it_mb)
    cropnet_test_chunks.append(cropnet_model(xb_eff, training=False).numpy())
    densenet_test_chunks.append(densenet_model(xb_dn, training=False).numpy())
    efficientnet_test_chunks.append(efficientnet_model(xb_mb, training=False).numpy())

cropnet_test_probs = np.concatenate(cropnet_test_chunks, axis=0)
densenet_test_probs = np.concatenate(densenet_test_chunks, axis=0)
efficientnet_test_probs = np.concatenate(efficientnet_test_chunks, axis=0)

test_soft_voting = (
    cropnet_weight * cropnet_test_probs
    + densenet_weight * densenet_test_probs
    + efficientnet_weight * efficientnet_test_probs
).astype(np.float32, copy=False)

fold_pred_matrix = np.stack([m.predict(test_soft_voting) for m in meta_models], axis=0)

F, N = fold_pred_matrix.shape
counts = np.zeros((N, NUM_CLASSES), dtype=np.int16)
flat_rows = np.repeat(np.arange(N), F)
flat_cols = fold_pred_matrix.T.reshape(-1)
np.add.at(counts, (flat_rows, flat_cols), 1)
predictions = counts.argmax(axis=1).astype(np.int64, copy=False)

test_image_ids = test_image_ids_from_sample
assert len(test_image_ids) == N, (
    f"image_id length ({len(test_image_ids)}) does not match predictions ({N}). "
    "Cannot create a valid submission."
)

submission_df = pd.DataFrame(
    {"image_id": test_image_ids, "label": predictions.astype(int)}
)
out_path = os.path.join(WORKING_DIR, "submission.csv")
submission_df.to_csv(out_path, index=False)

print("Submission file created:", out_path)
print(submission_df.head())
print("Submission shape:", submission_df.shape)
assert out_path.endswith(".csv") and submission_df.shape[0] == sample_sub.shape[0]
