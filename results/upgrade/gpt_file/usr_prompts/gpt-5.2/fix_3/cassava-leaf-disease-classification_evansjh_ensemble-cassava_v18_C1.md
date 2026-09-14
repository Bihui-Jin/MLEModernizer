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

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import shutil
from collections import Counter

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

print("TF version:", tf.__version__)
print("Eager:", tf.executing_eagerly())



## === cell 1
BASE = "/kaggle/input/cassava-leaf-disease-classification"
test_image_dir = f"{BASE}/test_images"
train_image_dir = f"{BASE}/train_images"
train_csv_path = f"{BASE}/train.csv"
sample_path = f"{BASE}/sample_submission.csv"

assert os.path.exists(test_image_dir), f"Missing test image dir: {test_image_dir}"
assert os.path.exists(sample_path), f"Missing sample submission: {sample_path}"
assert os.path.exists(train_image_dir), f"Missing train image dir: {train_image_dir}"
assert os.path.exists(train_csv_path), f"Missing train.csv: {train_csv_path}"

sample_csv = pd.read_csv(sample_path)
train_csv = pd.read_csv(train_csv_path)
print("sample_submission shape:", sample_csv.shape)
print("train_csv shape:", train_csv.shape)
sample_csv.head()



## === cell 2
source_dir = "/kaggle/input/cp-model"
dest_dir = "/kaggle/working/cp-model"
if os.path.exists(source_dir):
    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir)
    shutil.copytree(source_dir, dest_dir)
    os.environ["TFHUB_CACHE_DIR"] = dest_dir
    print("TFHUB_CACHE_DIR set to:", dest_dir)
else:
    print(
        "Optional TFHub cache dataset not found at",
        source_dir,
        "- continuing without TFHub.",
    )



## === cell 3
model_path_1 = (
    "/kaggle/input/combinedmodel3/tensorflow2/default/1/BestModel_3454_8937.h5"
)
model_path_2 = (
    "/kaggle/input/combinedmodel3/tensorflow2/default/1/best_model_0.37458707.h5"
)
model_path_3 = (
    "/kaggle/input/combinedmodel3/tensorflow2/default/1/googlenet_inceptionv3.h5"
)
model_path_4 = (
    "/kaggle/input/bestmodel_550_2/tensorflow2/default/1/BestModel_3577_8940.h5"
)
model_path_5 = (
    "/kaggle/input/bestmodel_8878/tensorflow2/default/1/BestModel_8878_0358.h5"
)
model_path_6 = "/kaggle/input/bestmodel_8875/tensorflow2/default/1/BestModel_8875.h5"
model_path_7 = "/kaggle/input/googlenet_512/tensorflow2/default/1/GOOG1.h5"

classifier = None  # placeholder to keep variable names consistent
model_path_8 = classifier

paths = [
    model_path_1,
    model_path_2,
    model_path_3,
    model_path_4,
    model_path_5,
    model_path_6,
    model_path_7,
]
exists = {p: os.path.exists(p) for p in paths}
print("Model files found:", sum(exists.values()), "of", len(paths))
for p, ok in exists.items():
    if ok:
        print("  OK:", p)
    else:
        print("  MISSING:", p)



## === cell 4
models_info = [
    (model_path_1, (550, 550)),
    (model_path_2, (550, 550)),
    (model_path_3, (299, 299)),  # common for InceptionV3
    (model_path_4, (550, 550)),
    (model_path_5, (550, 550)),
    (model_path_6, (550, 550)),
    (model_path_7, (512, 512)),
]

models = []
for path, input_size in models_info:
    if not os.path.exists(path):
        continue
    try:
        m = load_model(path, compile=False)
        models.append((m, input_size))
        print("Loaded model:", os.path.basename(path), "input_size:", input_size)
    except Exception as e:
        print("Failed to load model:", path, "error:", repr(e))

if classifier is not None:
    models.append((classifier, (224, 224)))

print("Total loaded models:", len(models))

NEED_FALLBACK = len(models) == 0
print("Need fallback model:", NEED_FALLBACK)



## === cell 5

fallback_model = None
fallback_input_size = (224, 224)

if NEED_FALLBACK:
    seed = 1337
    tf.keras.utils.set_random_seed(seed)

    img_size = fallback_input_size
    batch_size = 32
    num_classes = 5

    train_csv["filepath"] = train_csv["image_id"].apply(
        lambda x: os.path.join(train_image_dir, x)
    )
    train_csv = train_csv[train_csv["filepath"].apply(os.path.exists)].reset_index(
        drop=True
    )

    val_frac = 0.10
    parts = []
    val_parts = []
    for lbl, grp in train_csv.groupby("label"):
        grp = grp.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        n_val = max(1, int(len(grp) * val_frac))
        val_parts.append(grp.iloc[:n_val])
        parts.append(grp.iloc[n_val:])
    tr_df = (
        pd.concat(parts, axis=0)
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )
    va_df = (
        pd.concat(val_parts, axis=0)
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )

    print("Fallback train/val sizes:", len(tr_df), len(va_df))
    print("Train label dist:", tr_df["label"].value_counts().sort_index().to_dict())
    print("Val label dist:", va_df["label"].value_counts().sort_index().to_dict())

    def decode_image(path, label=None):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, img_size, method="bilinear")
        img = tf.cast(img, tf.float32) / 255.0
        if label is None:
            return img
        return img, tf.one_hot(tf.cast(label, tf.int32), num_classes)

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = tf.data.Dataset.from_tensor_slices(
        (tr_df["filepath"].values, tr_df["label"].values)
    )
    train_ds = train_ds.shuffle(
        min(len(tr_df), 8192), seed=seed, reshuffle_each_iteration=True
    )
    train_ds = (
        train_ds.map(decode_image, num_parallel_calls=AUTOTUNE)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )

    val_ds = tf.data.Dataset.from_tensor_slices(
        (va_df["filepath"].values, va_df["label"].values)
    )
    val_ds = (
        val_ds.map(decode_image, num_parallel_calls=AUTOTUNE)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )

    inputs = tf.keras.Input(shape=(img_size[0], img_size[1], 3))
    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    fallback_model = tf.keras.Model(inputs, outputs)

    fallback_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    epochs = 5
    history = fallback_model.fit(
        train_ds, validation_data=val_ds, epochs=epochs, verbose=2
    )

    models = [(fallback_model, fallback_input_size)]
    print("Fallback model trained and attached. Total models:", len(models))



## === cell 6
image_predictions = []

test_ids = set(sample_csv["image_id"].tolist())
available_files = [
    f
    for f in os.listdir(test_image_dir)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]
available_set = set(available_files)

missing = [
    img_id for img_id in sample_csv["image_id"].tolist() if img_id not in available_set
]
if missing:
    print(
        "Warning: missing",
        len(missing),
        "test images referenced in sample_submission (showing up to 5):",
        missing[:5],
    )

for image_id in sample_csv["image_id"].tolist():
    if image_id not in available_set:
        image_predictions.append({"image_id": image_id, "label": 0})
        continue

    model_predictions = []
    confidence_scores = {}

    img_path = os.path.join(test_image_dir, image_id)

    for model, input_size in models:
        img = load_img(img_path, target_size=input_size)
        img_array = np.expand_dims(img_to_array(img) / 255.0, axis=0)

        if callable(model) and not hasattr(model, "predict"):
            preds = model(img_array, training=False)
            preds = tf.convert_to_tensor(preds)
            predicted_class = int(tf.math.argmax(preds, axis=-1).numpy()[0])
            preds_np = preds.numpy()
        else:
            preds_np = model.predict(img_array, verbose=0)
            predicted_class = int(np.argmax(preds_np, axis=1)[0])

        confidence_score = float(preds_np[0][predicted_class])

        model_predictions.append(predicted_class)
        confidence_scores.setdefault(predicted_class, []).append(confidence_score)

    if len(model_predictions) == 0:
        final_predicted_class = 0
    else:
        class_votes = Counter(model_predictions)
        most_common = class_votes.most_common()
        final_predicted_class = most_common[0][0]

        if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
            tied_classes = [
                cls for cls, count in most_common if count == most_common[0][1]
            ]
            final_predicted_class = max(
                tied_classes,
                key=lambda cls: sum(confidence_scores[cls])
                / len(confidence_scores[cls]),
            )

    image_predictions.append(
        {"image_id": image_id, "label": int(final_predicted_class)}
    )

submission_df = pd.DataFrame(image_predictions)
submission_df = submission_df[["image_id", "label"]]

assert (
    submission_df.shape[0] == sample_csv.shape[0]
), "Submission row count mismatch vs sample_submission"
assert list(submission_df.columns) == ["image_id", "label"], "Wrong submission columns"
assert (
    submission_df["label"].between(0, 4).all()
), "Labels must be in [0,4] for this competition."

out_path = "/kaggle/working/submission.csv"
submission_df.to_csv(out_path, index=False)
print("Wrote:", out_path, "shape:", submission_df.shape)



## === cell 7
submission_df.head()



## === cell 8
print("Label value counts:")
print(submission_df["label"].value_counts().sort_index())
print(
    "Min label:",
    submission_df["label"].min(),
    "Max label:",
    submission_df["label"].max(),
)
submission_df.tail()
