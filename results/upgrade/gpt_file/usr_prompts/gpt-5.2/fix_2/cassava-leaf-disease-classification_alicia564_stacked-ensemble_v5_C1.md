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

# 5. Target score

0.8961922030825022

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras import layers, models

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

INPUT_DIR = "/kaggle/input/cassava-leaf-disease-classification"
WORKING_DIR = "/kaggle/working"

TRAIN_CSV_PATH = f"{INPUT_DIR}/train.csv"
SAMPLE_SUB_PATH = f"{INPUT_DIR}/sample_submission.csv"
TRAIN_IMG_DIR = f"{INPUT_DIR}/train_images"
TEST_IMG_DIR = f"{INPUT_DIR}/test_images"

print("TF version:", tf.__version__)
print("Train CSV exists:", os.path.exists(TRAIN_CSV_PATH))
print("Sample submission exists:", os.path.exists(SAMPLE_SUB_PATH))
print("Train image dir exists:", os.path.isdir(TRAIN_IMG_DIR))
print("Test image dir exists:", os.path.isdir(TEST_IMG_DIR))



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import (
    preprocess_input as eff_preprocess,
)

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

datagen_aug = ImageDataGenerator(
    preprocessing_function=eff_preprocess,
    rotation_range=45,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode="nearest",
)

datagen_valid = ImageDataGenerator(preprocessing_function=eff_preprocess)

train_generator = datagen_aug.flow_from_dataframe(
    dataframe=train_df,
    x_col="path",
    y_col="label",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True,
    seed=SEED,
)

valid_generator = datagen_valid.flow_from_dataframe(
    dataframe=valid_df,
    x_col="path",
    y_col="label",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False,
)


def load_and_preprocess_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32)
    image = eff_preprocess(image)  # match generator preprocessing
    return image, label


train_ds = tf.data.Dataset.from_tensor_slices(
    (train_df["path"].values, train_df["label"].values)
)
valid_ds = tf.data.Dataset.from_tensor_slices(
    (valid_df["path"].values, valid_df["label"].values)
)

train_ds = (
    train_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)
valid_ds = (
    valid_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2277570932.py in <cell line: 0>()
     43 datagen_valid = ImageDataGenerator(preprocessing_function=eff_preprocess)
     44 
---> 45 train_generator = datagen_aug.flow_from_dataframe(
     46     dataframe=train_df,
     47     x_col="path",

/usr/local/lib/python3.11/dist-packages/keras/src/legacy/preprocessing/image.py in flow_from_dataframe(self, dataframe, directory, x_col, y_col, weight_col, target_size, color_mode, classes, class_mode, batch_size, shuffle, seed, save_to_dir, save_prefix, save_format, subset, interpolation, validate_filenames, **kwargs)
   1206             )
   1207 
-> 1208         return DataFrameIterator(
   1209             dataframe,
   1210             directory,

/usr/local/lib/python3.11/dist-packages/keras/src/legacy/preprocessing/image.py in __init__(self, dataframe, directory, image_data_generator, x_col, y_col, weight_col, target_size, color_mode, classes, class_mode, batch_size, shuffle, seed, data_format, save_to_dir, save_prefix, save_format, subset, interpolation, keep_aspect_ratio, dtype, validate_filenames)
    749         self.dtype = dtype
    750         # check that inputs match the required class_mode
--> 751         self._check_params(df, x_col, y_col, weight_col, classes)
    752         if (
    753             validate_filenames

/usr/local/lib/python3.11/dist-packages/keras/src/legacy/preprocessing/image.py in _check_params(self, df, x_col, y_col, weight_col, classes)
    839             types = (str, list, tuple)
    840             if not all(df[y_col].apply(lambda x: isinstance(x, types))):
--> 841                 raise TypeError(
    842                     'If class_mode="{}", y_col="{}" column '
    843                     "values must be type string, list or tuple.".format(

TypeError: If class_mode="categorical", y_col="label" column values must be type string, list or tuple.

## === cell 2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

early_stopping = EarlyStopping(
    monitor="val_loss", patience=3, restore_best_weights=True
)
learning_rate_reduction = ReduceLROnPlateau(
    monitor="val_loss", patience=2, factor=0.5, min_lr=1e-6, verbose=1
)

from tensorflow.keras.applications import EfficientNetB0, DenseNet169, MobileNetV2


def build_backbone_model(backbone, name):
    inp = layers.Input(shape=(224, 224, 3))
    x = backbone(inp, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    out = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    model = models.Model(inp, out, name=name)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


eff_b0 = EfficientNetB0(
    include_top=False, weights="imagenet", input_shape=(224, 224, 3)
)
dn169 = DenseNet169(include_top=False, weights="imagenet", input_shape=(224, 224, 3))
mbv2 = MobileNetV2(include_top=False, weights="imagenet", input_shape=(224, 224, 3))

for bb in (eff_b0, dn169, mbv2):
    bb.trainable = True

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
    train_generator,
    validation_data=valid_generator,
    epochs=EPOCHS_BASE,
    callbacks=[early_stopping, learning_rate_reduction],
    verbose=1,
)
print("Training base model 2/3...")
densenet_model.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=EPOCHS_BASE,
    callbacks=[early_stopping, learning_rate_reduction],
    verbose=1,
)
print("Training base model 3/3...")
efficientnet_model.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=EPOCHS_BASE,
    callbacks=[early_stopping, learning_rate_reduction],
    verbose=1,
)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1245199556.py in <cell line: 0>()
     57 print("Training base model 1/3...")
     58 cropnet_model.fit(
---> 59     train_generator,
     60     validation_data=valid_generator,
     61     epochs=EPOCHS_BASE,

NameError: name 'train_generator' is not defined

## === cell 3
soft_voting_features = []
soft_voting_labels = []

for img_batch, label_batch in train_ds:
    img_batch_np = img_batch.numpy()

    cropnet_probs = cropnet_model.predict(img_batch_np, verbose=0)
    densenet_probs = densenet_model.predict(img_batch_np, verbose=0)
    efficientnet_probs = efficientnet_model.predict(img_batch_np, verbose=0)

    soft_voting_probs = (
        cropnet_weight * cropnet_probs
        + densenet_weight * densenet_probs
        + efficientnet_weight * efficientnet_probs
    )

    soft_voting_features.append(soft_voting_probs)
    soft_voting_labels.append(label_batch.numpy())

soft_voting_features = (
    np.concatenate(soft_voting_features, axis=0)
    if len(soft_voting_features)
    else np.empty((0, NUM_CLASSES))
)
soft_voting_labels = (
    np.concatenate(soft_voting_labels, axis=0)
    if len(soft_voting_labels)
    else np.empty((0,), dtype=int)
)

print("Soft-voting features shape:", soft_voting_features.shape)
print("Soft-voting labels shape:", soft_voting_labels.shape)
assert (
    soft_voting_features.shape[0] == soft_voting_labels.shape[0]
    and soft_voting_features.shape[0] > 0
)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/738512688.py in <cell line: 0>()
      4 soft_voting_labels = []
      5 
----> 6 for img_batch, label_batch in train_ds:
      7     img_batch_np = img_batch.numpy()
      8 

NameError: name 'train_ds' is not defined

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



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/861690856.py in <cell line: 0>()
     12 meta_models = []
     13 
---> 14 for train_index, val_index in kf.split(soft_voting_features, soft_voting_labels):
     15     X_train, X_val = soft_voting_features[train_index], soft_voting_features[val_index]
     16     y_train, y_val = soft_voting_labels[train_index], soft_voting_labels[val_index]

/usr/local/lib/python3.11/dist-packages/sklearn/model_selection/_split.py in split(self, X, y, groups)
    769         to an integer.
    770         """
--> 771         y = check_array(y, input_name="y", ensure_2d=False, dtype=None)
    772         return super().split(X, y, groups)
    773 

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in check_array(array, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, ensure_min_samples, ensure_min_features, estimator, input_name)
    929         n_samples = _num_samples(array)
    930         if n_samples < ensure_min_samples:
--> 931             raise ValueError(
    932                 "Found array with %d sample(s) (shape=%s) while a"
    933                 " minimum of %d is required%s."

ValueError: Found array with 0 sample(s) (shape=(0,)) while a minimum of 1 is required.

## === cell 5
import matplotlib.pyplot as plt

num_folds = len(train_distributions)
plt.figure(figsize=(15, 3 * num_folds))

for fold in range(num_folds):
    train_dist = train_distributions[fold]
    val_dist = val_distributions[fold]

    df_train = pd.DataFrame(list(train_dist.items()), columns=["Class", "Proportion"])
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



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4035718734.py in <cell line: 0>()
      1 from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
      2 
----> 3 meta_model_predictions = meta_model.predict(soft_voting_features)
      4 cm = confusion_matrix(
      5     soft_voting_labels, meta_model_predictions, labels=list(range(NUM_CLASSES))

NameError: name 'meta_model' is not defined

## === cell 8
from tensorflow.keras.preprocessing.image import load_img, img_to_array

sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
test_image_ids = sample_sub["image_id"].astype(str).tolist()

predictions = []

for image_id in test_image_ids:
    img_path = os.path.join(TEST_IMG_DIR, image_id)
    img = load_img(img_path, target_size=IMG_SIZE)
    img_array = img_to_array(img).astype(np.float32)
    img_array = eff_preprocess(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    cropnet_probs = cropnet_model.predict(img_array, verbose=0)
    densenet_probs = densenet_model.predict(img_array, verbose=0)
    efficientnet_probs = efficientnet_model.predict(img_array, verbose=0)

    soft_voting_probs = (
        cropnet_weight * cropnet_probs
        + densenet_weight * densenet_probs
        + efficientnet_weight * efficientnet_probs
    ).reshape(1, -1)

    fold_preds = [m.predict(soft_voting_probs)[0] for m in meta_models]
    final_pred = int(max(set(fold_preds), key=fold_preds.count))
    predictions.append(final_pred)

submission_df = pd.DataFrame({"image_id": test_image_ids, "label": predictions})
out_path = os.path.join(WORKING_DIR, "submission.csv")
submission_df.to_csv(out_path, index=False)

print("Submission file created:", out_path)
print(submission_df.head())
print("Submission shape:", submission_df.shape)
assert out_path.endswith(".csv") and submission_df.shape[0] == sample_sub.shape[0]

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/26845778.py in <cell line: 0>()
     26     # Majority vote across fold meta-models (core logic preserved)
     27     fold_preds = [m.predict(soft_voting_probs)[0] for m in meta_models]
---> 28     final_pred = int(max(set(fold_preds), key=fold_preds.count))
     29     predictions.append(final_pred)
     30 

ValueError: max() arg is an empty sequence
