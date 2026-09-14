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
from collections import Counter

import numpy as np
import pandas as pd

import tf_keras as keras
import tensorflow as tf  # keep for low-level ops used with tf.data; if this fails, Kaggle runtime is broken anyway.

from keras.preprocessing.image import load_img, img_to_array

print("TF:", tf.__version__)
print("Keras (tf_keras):", keras.__version__)

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)



## === cell 1
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

test_image_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images"
train_image_dir = "/kaggle/input/cassava-leaf-disease-classification/train_images"
sample = "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
train_csv_path = "/kaggle/input/cassava-leaf-disease-classification/train.csv"

assert os.path.exists(sample), f"Missing sample_submission.csv at {sample}"
assert os.path.isdir(test_image_dir), f"Missing test image dir at {test_image_dir}"
assert os.path.exists(train_csv_path), f"Missing train.csv at {train_csv_path}"
assert os.path.isdir(train_image_dir), f"Missing train image dir at {train_image_dir}"



## === cell 2
sample_csv = pd.read_csv(sample)
print(sample_csv.head())
print("sample_submission rows:", len(sample_csv))
print("sample_submission columns:", list(sample_csv.columns))

train_df = pd.read_csv(train_csv_path)
print(train_df.head())
print("train rows:", len(train_df))
print("train columns:", list(train_df.columns))
print("label distribution:\n", train_df["label"].value_counts().sort_index())



## === cell 3
models_info = [
    (model_path_3, (448, 448)),
    (model_path_7, (512, 512)),
    (model_path_1, (512, 512)),
    (model_path_2, (512, 512)),
    (model_path_4, (512, 512)),
    (model_path_5, (512, 512)),
    (model_path_6, (512, 512)),
]

existing_models_info = [(p, s) for (p, s) in models_info if os.path.exists(p)]
missing_models = [p for (p, _) in models_info if not os.path.exists(p)]

print("Missing model files (will be skipped):")
for p in missing_models:
    print(" -", p)

models = []
for path, input_size in existing_models_info:
    try:
        m = keras.models.load_model(path, compile=False)
    except TypeError:
        m = keras.models.load_model(path)
    models.append((m, input_size))
    print(f"Loaded model: {path} with input size {input_size}")

print("Number of external models loaded:", len(models))



## === cell 4

NUM_CLASSES = 5


def make_fallback_model(input_size=(224, 224), num_classes=5):
    inputs = keras.Input(shape=(input_size[0], input_size[1], 3))
    base = keras.applications.MobileNetV2(
        input_shape=(input_size[0], input_size[1], 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False
    x = base(inputs, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.2)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_dataset(df, image_dir, input_size=(224, 224), batch_size=32, training=True):
    paths = (image_dir.rstrip("/") + "/" + df["image_id"].astype(str)).values
    labels = df["label"].values.astype(np.int32) if "label" in df.columns else None

    def _load(path, label=None):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, input_size, method="bilinear")
        img = tf.cast(img, tf.float32) / 255.0
        if training:
            img = tf.image.random_flip_left_right(img, seed=SEED)
        if label is None:
            return img
        return img, label

    if labels is None:
        ds = tf.data.Dataset.from_tensor_slices(paths)
        ds = ds.map(lambda p: _load(p, None), num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        ds = ds.map(lambda p, y: _load(p, y), num_parallel_calls=tf.data.AUTOTUNE)

    if training:
        ds = ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


if len(models) == 0:
    fallback_input_size = (224, 224)
    fallback_model = make_fallback_model(
        input_size=fallback_input_size, num_classes=NUM_CLASSES
    )

    train_df_shuffled = train_df.sample(frac=1.0, random_state=SEED).reset_index(
        drop=True
    )
    val_frac = 0.1
    n_val = int(len(train_df_shuffled) * val_frac)
    val_df = train_df_shuffled.iloc[:n_val].copy()
    tr_df = train_df_shuffled.iloc[n_val:].copy()

    train_ds = build_dataset(
        tr_df,
        train_image_dir,
        input_size=fallback_input_size,
        batch_size=32,
        training=True,
    )
    val_ds = build_dataset(
        val_df,
        train_image_dir,
        input_size=fallback_input_size,
        batch_size=32,
        training=False,
    )

    fallback_model.fit(train_ds, validation_data=val_ds, epochs=3, verbose=2)

    models = [(fallback_model, fallback_input_size)]
    print("Trained and added fallback model. Number of models loaded:", len(models))
else:
    print("Using external models only; fallback training skipped.")



## === cell 5
class_labels = {
    0: "Cassava Bacterial Blight (CBB)",
    1: "Cassava Brown Streak Disease (CBSD)",
    2: "Cassava Green Mottle (CGM)",
    3: "Cassava Mosaic Disease (CMD)",
    4: "Healthy",
}

image_ids = sample_csv["image_id"].tolist()

missing_imgs = [
    img for img in image_ids if not os.path.exists(os.path.join(test_image_dir, img))
]
print("Missing test images referenced by sample_submission:", len(missing_imgs))
if len(missing_imgs) > 0:
    print("Example missing images:", missing_imgs[:5])



## === cell 6
image_predictions = []

for idx, image_id in enumerate(image_ids):
    img_path = os.path.join(test_image_dir, image_id)

    if not os.path.exists(img_path):
        image_predictions.append({"image_id": image_id, "label": 4})
        continue

    model_predictions = []
    confidence_scores = {}

    for model, input_size in models:
        img = load_img(img_path, target_size=input_size)
        img_array = np.expand_dims(img_to_array(img) / 255.0, axis=0)

        predictions = model.predict(img_array, verbose=0)
        predicted_class = int(np.argmax(predictions, axis=1)[0])
        confidence_score = float(predictions[0][predicted_class])

        model_predictions.append(predicted_class)
        confidence_scores.setdefault(predicted_class, []).append(confidence_score)

    class_votes = Counter(model_predictions)
    most_common = class_votes.most_common()
    final_predicted_class = most_common[0][0]

    if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
        tied_classes = [cls for cls, count in most_common if count == most_common[0][1]]
        final_predicted_class = max(
            tied_classes,
            key=lambda cls: sum(confidence_scores[cls]) / len(confidence_scores[cls]),
        )

    image_predictions.append(
        {"image_id": image_id, "label": int(final_predicted_class)}
    )

    if (idx + 1) % 200 == 0:
        print(f"Predicted {idx+1}/{len(image_ids)} images...")

submission_df = pd.DataFrame(image_predictions)



## === cell 7
submission_df = submission_df.merge(
    sample_csv[["image_id"]], on="image_id", how="right"
)
submission_df["label"] = submission_df["label"].fillna(4).astype(int)  # safety fallback
submission_df = submission_df[["image_id", "label"]]

print(submission_df.head())
print("submission rows:", len(submission_df), "expected:", len(sample_csv))
assert len(submission_df) == len(sample_csv), "Submission row count mismatch."
assert list(submission_df.columns) == [
    "image_id",
    "label",
], "Submission columns mismatch."

out_path = "/kaggle/working/submission.csv"
submission_df.to_csv(out_path, index=False)
print("Wrote:", out_path)



## === cell 8
submission_df
