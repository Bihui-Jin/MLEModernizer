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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"
try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass
try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

tf.random.set_seed(SEED)
np.random.seed(SEED)

plt.ioff()

DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV_PATH = f"{DATA_DIR}/train.csv"
SAMPLE_SUB_PATH = f"{DATA_DIR}/sample_submission.csv"
TRAIN_IMG_DIR = f"{DATA_DIR}/train_images"
TEST_IMG_DIR = f"{DATA_DIR}/test_images"
TRAIN_TFREC_DIR = f"{DATA_DIR}/train_tfrecords"
TEST_TFREC_DIR = f"{DATA_DIR}/test_tfrecords"

NUM_CLASSES = 5
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

AUTOTUNE = tf.data.AUTOTUNE

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

_TF_DATA_OPTIONS = tf.data.Options()
_TF_DATA_OPTIONS.deterministic = True
try:
    _TF_DATA_OPTIONS.experimental_optimization.apply_default_optimizations = True
    _TF_DATA_OPTIONS.experimental_optimization.map_parallelization = True
    _TF_DATA_OPTIONS.experimental_optimization.parallel_batch = True
except Exception:
    pass




## === cell 1
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.preprocessing import LabelEncoder

_FEATURE_DESCRIPTION = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
}


def _parse_tfrec(example_proto):
    ex = tf.io.parse_single_example(example_proto, _FEATURE_DESCRIPTION)
    img = tf.image.decode_jpeg(ex["image"], channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = preprocess_input(tf.cast(img, tf.float32))  # preserve preprocessing semantics
    label = tf.cast(ex["target"], tf.int32)
    label_oh = tf.one_hot(label, NUM_CLASSES, dtype=tf.float32)
    return img, label_oh, label


def load_and_preprocess_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


label_to_disease = pd.read_json(
    f"{DATA_DIR}/label_num_to_disease_map.json", typ="series"
)
train_csv = pd.read_csv(TRAIN_CSV_PATH)

train_csv["disease"] = train_csv["label"].map(label_to_disease)
train_csv["path"] = TRAIN_IMG_DIR + "/" + train_csv["image_id"]

le = LabelEncoder()
train_csv["label_encoded"] = le.fit_transform(train_csv["disease"].astype(str))

train_csv["disease"] = train_csv["disease"].astype(str)
train_csv["label"] = train_csv["label"].astype(str)

train, valid = train_test_split(
    train_csv,
    test_size=0.2,
    stratify=train_csv["label"],
    random_state=SEED,
)

train_tfrec_files = sorted(
    [
        os.path.join(TRAIN_TFREC_DIR, f)
        for f in os.listdir(TRAIN_TFREC_DIR)
        if f.endswith(".tfrec")
    ]
)
if not train_tfrec_files:
    raise FileNotFoundError(f"No .tfrec files found under {TRAIN_TFREC_DIR}")

num_train_files = max(1, int(round(0.8 * len(train_tfrec_files))))
tr_files = train_tfrec_files[:num_train_files]
va_files = (
    train_tfrec_files[num_train_files:]
    if num_train_files < len(train_tfrec_files)
    else train_tfrec_files[-1:]
)

_CACHE_DIR = "/kaggle/working/tfdata_cache"
os.makedirs(_CACHE_DIR, exist_ok=True)


def _count_records_fast(tfrec_files):
    ds = tf.data.TFRecordDataset(tfrec_files, num_parallel_reads=AUTOTUNE)
    ds = ds.with_options(_TF_DATA_OPTIONS)
    c = 0
    for _ in ds:
        c += 1
    return c


def _make_ds(files, training: bool, cache_tag: str):
    ds = tf.data.TFRecordDataset(files, num_parallel_reads=AUTOTUNE)
    ds = ds.with_options(_TF_DATA_OPTIONS)
    ds = ds.map(_parse_tfrec, num_parallel_calls=AUTOTUNE)

    cache_path = os.path.join(_CACHE_DIR, f"{cache_tag}.cache")
    ds = ds.cache(cache_path)

    if training:

        def _augment(img, y_oh, y_int):
            img = tf.image.random_flip_left_right(img, seed=SEED)
            img = tf.image.random_flip_up_down(img, seed=SEED)
            return img, y_oh, y_int

        ds = ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
        ds = ds.map(_augment, num_parallel_calls=AUTOTUNE)

    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_ds_full = _make_ds(tr_files, training=True, cache_tag="train_parsed")
valid_ds_full = _make_ds(va_files, training=False, cache_tag="valid_parsed")

train_fit_ds = train_ds_full.map(
    lambda x, y_oh, y_int: (x, y_oh), num_parallel_calls=AUTOTUNE
)
valid_fit_ds = valid_ds_full.map(
    lambda x, y_oh, y_int: (x, y_oh), num_parallel_calls=AUTOTUNE
)

n_train_examples = _count_records_fast(tr_files)
n_valid_examples = _count_records_fast(va_files)
steps_per_epoch = int(np.ceil(n_train_examples / BATCH_SIZE))
val_steps = int(np.ceil(n_valid_examples / BATCH_SIZE))




## === cell 2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

early_stopping = EarlyStopping(
    monitor="val_loss", patience=3, restore_best_weights=True
)
learning_rate_reduction = ReduceLROnPlateau(
    monitor="val_loss", patience=2, factor=0.5, min_lr=1e-6, verbose=1
)




## === cell 3
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetB0, DenseNet169


def build_backbone_model(
    backbone, input_shape=(224, 224, 3), num_classes=5, name="model"
):
    inp = Input(shape=input_shape)
    x = backbone(include_top=False, weights="imagenet", input_tensor=inp)
    y = x.output
    y = GlobalAveragePooling2D()(y)
    y = Dense(num_classes, activation="softmax")(y)
    m = Model(inputs=inp, outputs=y, name=name)
    return m


cropnet_model = build_backbone_model(
    EfficientNetB0,
    input_shape=(*IMG_SIZE, 3),
    num_classes=NUM_CLASSES,
    name="cropnet_model",
)
densenet_model = build_backbone_model(
    DenseNet169,
    input_shape=(*IMG_SIZE, 3),
    num_classes=NUM_CLASSES,
    name="densenet_model",
)
efficientnet_model = build_backbone_model(
    EfficientNetB0,
    input_shape=(*IMG_SIZE, 3),
    num_classes=NUM_CLASSES,
    name="efficientnet_model",
)

cropnet_weight = 0.65
densenet_weight = 0.25
efficientnet_weight = 0.1

for m in (cropnet_model, densenet_model, efficientnet_model):
    m.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

EPOCHS = 2  # keep identical as provided

fit_kwargs = dict(
    validation_data=valid_fit_ds,
    epochs=EPOCHS,
    callbacks=[early_stopping, learning_rate_reduction],
    verbose=1,
    steps_per_epoch=max(1, steps_per_epoch),
    validation_steps=max(1, val_steps),
)

history_crop = cropnet_model.fit(train_fit_ds, **fit_kwargs)
history_dense = densenet_model.fit(train_fit_ds, **fit_kwargs)
history_eff = efficientnet_model.fit(train_fit_ds, **fit_kwargs)




## === cell 4
def _as_probs(pred):
    if isinstance(pred, dict):
        return np.asarray(pred[next(iter(pred.keys()))])
    return np.asarray(pred)




## === cell 5
@tf.function(jit_compile=True)
def _batch_predict_3models(xb):
    p1 = cropnet_model(xb, training=False)
    p2 = densenet_model(xb, training=False)
    p3 = efficientnet_model(xb, training=False)
    return p1, p2, p3


stack_ds = train_ds_full  # yields (img, y_oh, y_int)

n_train = int(n_train_examples)
cropnet_probs = np.empty((n_train, NUM_CLASSES), dtype=np.float32)
densenet_probs = np.empty((n_train, NUM_CLASSES), dtype=np.float32)
efficientnet_probs = np.empty((n_train, NUM_CLASSES), dtype=np.float32)
soft_voting_labels = np.empty((n_train,), dtype=np.int32)

write_idx = 0
for xb, _, yb in stack_ds:
    bs = int(xb.shape[0])
    p1, p2, p3 = _batch_predict_3models(xb)
    cropnet_probs[write_idx : write_idx + bs] = p1.numpy()
    densenet_probs[write_idx : write_idx + bs] = p2.numpy()
    efficientnet_probs[write_idx : write_idx + bs] = p3.numpy()
    soft_voting_labels[write_idx : write_idx + bs] = yb.numpy().astype(
        np.int32, copy=False
    )
    write_idx += bs

cropnet_probs = cropnet_probs[:write_idx]
densenet_probs = densenet_probs[:write_idx]
efficientnet_probs = efficientnet_probs[:write_idx]
soft_voting_labels = soft_voting_labels[:write_idx]

soft_voting_features = (
    cropnet_weight * cropnet_probs
    + densenet_weight * densenet_probs
    + efficientnet_weight * efficientnet_probs
)

print("soft_voting_features shape:", soft_voting_features.shape)
print("soft_voting_labels shape:", soft_voting_labels.shape)




## === cell 6
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold

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
        max_iter=1000, C=0.1, multi_class="multinomial", penalty="l2"
    )
    meta_model.fit(X_train, y_train)
    meta_models.append(meta_model)

    val_preds = meta_model.predict(X_val)
    val_probs = meta_model.predict_proba(X_val)

    fold_accuracies.append(accuracy_score(y_val, val_preds))
    fold_log_losses.append(log_loss(y_val, val_probs))

    train_distributions.append(
        pd.Series(y_train).value_counts(normalize=True).to_dict()
    )
    val_distributions.append(pd.Series(y_val).value_counts(normalize=True).to_dict())

print("CV accuracy (mean):", float(np.mean(fold_accuracies)))
print("CV logloss (mean):", float(np.mean(fold_log_losses)))




## === cell 7
if False:
    num_folds = len(train_distributions)
    plt.figure(figsize=(15, 3 * max(1, num_folds)))

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




## === cell 8
if False:
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




## === cell 9
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

meta_model = meta_models[-1]
meta_model_predictions = meta_model.predict(soft_voting_features)

cm = confusion_matrix(soft_voting_labels, meta_model_predictions)
true_class_3_correct_predictions = int(cm[3, 3]) if cm.shape[0] > 3 else 0

if False:
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(cmap="Blues", ax=ax)
    plt.title("Confusion Matrix for Meta-Model on Training Data")
    plt.gcf().text(
        0.5,
        -0.0001,
        f"True positive predictions for class 3: {true_class_3_correct_predictions}",
        ha="center",
        va="center",
        fontsize=12,
    )
    plt.show()




## === cell 10
print("True positive predictions for class 3:", true_class_3_correct_predictions)




## === cell 11
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

test_tfrec_files = sorted(
    [
        os.path.join(TEST_TFREC_DIR, f)
        for f in os.listdir(TEST_TFREC_DIR)
        if f.endswith(".tfrec")
    ]
)
if not test_tfrec_files:
    raise FileNotFoundError(f"No .tfrec files found under {TEST_TFREC_DIR}")

test_ds_full = tf.data.TFRecordDataset(test_tfrec_files, num_parallel_reads=AUTOTUNE)
test_ds_full = test_ds_full.with_options(_TF_DATA_OPTIONS)

_TEST_FEATURE_DESCRIPTION = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
}


def _parse_test_tfrec(example_proto):
    ex = tf.io.parse_single_example(example_proto, _TEST_FEATURE_DESCRIPTION)
    img = tf.image.decode_jpeg(ex["image"], channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = preprocess_input(tf.cast(img, tf.float32))
    name = ex["image_name"]
    return img, name


test_ds = test_ds_full.map(_parse_test_tfrec, num_parallel_calls=AUTOTUNE)
test_ds = test_ds.cache(os.path.join(_CACHE_DIR, "test_parsed.cache"))
test_ds = test_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

n_test_examples = _count_records_fast(test_tfrec_files)
n_test = int(n_test_examples)
soft_voting_test_features = np.empty((n_test, NUM_CLASSES), dtype=np.float32)
test_names = np.empty((n_test,), dtype=object)

write_idx = 0
for xb, nameb in test_ds:
    bs = int(xb.shape[0])
    p1, p2, p3 = _batch_predict_3models(xb)
    soft_voting_test_features[write_idx : write_idx + bs] = (
        cropnet_weight * p1.numpy()
        + densenet_weight * p2.numpy()
        + efficientnet_weight * p3.numpy()
    )
    test_names[write_idx : write_idx + bs] = nameb.numpy().astype(object)
    write_idx += bs

soft_voting_test_features = soft_voting_test_features[:write_idx]
test_names = test_names[:write_idx]

fold_preds = np.stack(
    [m.predict(soft_voting_test_features) for m in meta_models], axis=1
)  # (N, K)

num_classes = NUM_CLASSES
counts = np.zeros((fold_preds.shape[0], num_classes), dtype=np.int32)
rows = np.repeat(np.arange(fold_preds.shape[0]), fold_preds.shape[1])
cols = fold_preds.reshape(-1)
np.add.at(counts, (rows, cols), 1)
predictions = counts.argmax(axis=1).astype(int).tolist()

test_image_ids = sample_sub["image_id"].tolist()
if len(test_image_ids) != len(predictions):
    test_paths = [os.path.join(TEST_IMG_DIR, image_id) for image_id in test_image_ids]
    test_fallback_ds = tf.data.Dataset.from_tensor_slices(
        (np.array(test_paths, dtype=object), np.zeros(len(test_paths), dtype=np.int64))
    )
    test_fallback_ds = (
        test_fallback_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(AUTOTUNE)
    )

    fb_feats = []
    for xb, _ in test_fallback_ds:
        p1, p2, p3 = _batch_predict_3models(xb)
        fb_feats.append(
            cropnet_weight * p1.numpy()
            + densenet_weight * p2.numpy()
            + efficientnet_weight * p3.numpy()
        )
    soft_voting_test_features = np.concatenate(fb_feats, axis=0)

    fold_preds = np.stack(
        [m.predict(soft_voting_test_features) for m in meta_models], axis=1
    )
    counts = np.zeros((fold_preds.shape[0], num_classes), dtype=np.int32)
    rows = np.repeat(np.arange(fold_preds.shape[0]), fold_preds.shape[1])
    cols = fold_preds.reshape(-1)
    np.add.at(counts, (rows, cols), 1)
    predictions = counts.argmax(axis=1).astype(int).tolist()

submission_df = pd.DataFrame({"image_id": test_image_ids, "label": predictions})
out_path = "/kaggle/working/submission.csv"
submission_df.to_csv(out_path, index=False)
print(f"Submission file created: {out_path}")
print(submission_df.head())
print("Submission shape:", submission_df.shape)
