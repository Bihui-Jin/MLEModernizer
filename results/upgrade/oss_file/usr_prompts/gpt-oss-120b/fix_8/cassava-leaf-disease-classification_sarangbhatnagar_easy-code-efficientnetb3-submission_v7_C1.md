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

3.9

# 3. Installed packages

geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
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

0.8549410698096102

# 6. Current score

0.76607

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.61099) has done: 'I replace the failing TensorFlow /Keras import with a safe `import tensorflow as tf` alias, remove the broken model loading (the .h5 file is missing), and fall back to a simple baseline that predicts the most frequent label from the training set for every test image. This eliminates the import and file‑not‑found errors, guarantees a valid `submission.csv` with the correct columns, and keeps the core logic minimal while still producing a runnable notebook.'
- What this solution (achieved 0.53737) has done: 'Implemented a protobuf compatibility shim before importing TensorFlow to avoid the `MessageFactory.GetPrototype` error, and converted the label column to strings so Keras’ `flow_from_dataframe` accepts it for categorical mode. The rest of the logic is unchanged, preserving the original model architecture and training loop, while ensuring a valid `submission.csv` is always written.'
- What this solution (achieved 0.6648) has done: 'Implemented a safe import for TensorFlow Addons (used for image rotation) with a fallback that simply returns the original image when the library is unavailable. This resolves the `NameError: name 'tfa' is not defined` and allows the dataset pipeline to run, ensuring `my_submission` is created and the final preview prints correctly. Minor restructuring keeps the original workflow unchanged while guaranteeing a valid `submission.csv` is produced.'
- What this solution (achieved 0.76607) has done: 'I increase the image resolution to 224 × 224 (the native size for MobileNetV2), train a bit longer (15 epochs per phase) and keep the same architecture and augmentations. These minimal changes should raise validation accuracy and move the Kaggle score closer to the target while preserving the original workflow.'

# 9. Code solution

## === cell 0
import os

try:
    from google.protobuf.message_factory import MessageFactory

    if not hasattr(MessageFactory, "GetPrototype"):

        def _get_prototype(self, descriptor):
            return self.GetMessageClass(descriptor)

        MessageFactory.GetPrototype = _get_prototype
except Exception:
    pass

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_ALLOW_GLOBAL_ASSIGNMENTS"] = "1"

import numpy as np
import pandas as pd
from pathlib import Path

try:
    import tensorflow as tf
except Exception as e:
    print("TensorFlow import failed:", e)
    tf = None

try:
    import tensorflow_addons as tfa
except Exception:

    class _DummyTFA:
        @staticmethod
        def image_rotate(image, angle, interpolation="BILINEAR"):
            return image

    tfa = _DummyTFA()




## === cell 1
train_path = Path("../input/cassava-leaf-disease-classification/train.csv")
if not train_path.exists():
    train_path = Path("/kaggle/input/cassava-leaf-disease-classification/train.csv")
train_df = pd.read_csv(train_path)

sample_sub_path = Path(
    "../input/cassava-leaf-disease-classification/sample_submission.csv"
)
if not sample_sub_path.exists():
    sample_sub_path = Path(
        "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
    )
sample_sub_df = pd.read_csv(sample_sub_path)

train_df["label"] = train_df["label"].astype(str)

most_common_label = train_df["label"].mode()[0]




## === cell 2
if tf is not None:
    import sklearn.model_selection

    IMG_SIZE = 224  # use the native MobileNetV2 size for better feature extraction

    train_img_dir = Path("../input/cassava-leaf-disease-classification/train_images")
    if not train_img_dir.exists():
        train_img_dir = Path(
            "/kaggle/input/cassava-leaf-disease-classification/train_images"
        )

    test_img_dir = Path("../input/cassava-leaf-disease-classification/test_images")
    if not test_img_dir.exists():
        test_img_dir = Path(
            "/kaggle/input/cassava-leaf-disease-classification/test_images"
        )

    train_df_shuf = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    train_split, val_split = sklearn.model_selection.train_test_split(
        train_df_shuf, test_size=0.2, random_state=42, stratify=train_df_shuf["label"]
    )

    label_to_index = {
        label: idx for idx, label in enumerate(sorted(train_df["label"].unique()))
    }
    num_classes = len(label_to_index)

    def df_to_ds(df, training):
        file_paths = tf.convert_to_tensor(
            [str(train_img_dir / fname) for fname in df["image_id"]], dtype=tf.string
        )
        labels = tf.convert_to_tensor(
            [label_to_index[lbl] for lbl in df["label"]], dtype=tf.int32
        )
        ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))

        def _load_and_preprocess(path, label):
            img = tf.io.read_file(path)
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
            img = img / 255.0  # rescale

            if training:
                img = tf.image.random_flip_left_right(img)
                img = tf.image.random_flip_up_down(img)
                img = tf.image.random_brightness(img, max_delta=0.1)
                angle = tf.random.uniform([], -20, 20) * tf.constant(np.pi / 180)
                img = tfa.image_rotate(img, angle, interpolation="BILINEAR")
                scales = tf.random.uniform([], 0.8, 1.2)
                new_size = tf.cast(scales * IMG_SIZE, tf.int32)
                img = tf.image.resize(img, [new_size, new_size])
                img = tf.image.resize_with_crop_or_pad(img, IMG_SIZE, IMG_SIZE)
            return img, tf.one_hot(label, depth=num_classes)

        ds = ds.map(_load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        if training:
            ds = ds.shuffle(1024, seed=42)
        ds = ds.batch(32).prefetch(tf.data.AUTOTUNE)
        return ds

    train_dataset = df_to_ds(train_split, training=True)
    val_dataset = df_to_ds(val_split, training=False)

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False  # freeze pretrained weights

    model = tf.keras.Sequential(
        [
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(5, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(
        train_dataset,
        epochs=15,
        validation_data=val_dataset,
        verbose=2,
    )

    base_model.trainable = True
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        train_dataset,
        epochs=15,
        validation_data=val_dataset,
        verbose=2,
    )

    test_paths = tf.convert_to_tensor(
        [str(test_img_dir / fname) for fname in sample_sub_df["image_id"]],
        dtype=tf.string,
    )
    test_ds = tf.data.Dataset.from_tensor_slices(test_paths)

    def _load_test(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
        img = img / 255.0
        return img

    test_ds = test_ds.map(_load_test, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.batch(32).prefetch(tf.data.AUTOTUNE)

    preds_prob = model.predict(test_ds, verbose=0)
    preds = np.argmax(preds_prob, axis=1)

    my_submission = pd.DataFrame(
        {"image_id": sample_sub_df["image_id"], "label": preds}
    )
    my_submission.to_csv("submission.csv", index=False)
else:
    preds = [int(most_common_label)] * len(sample_sub_df)
    my_submission = pd.DataFrame(
        {"image_id": sample_sub_df["image_id"], "label": preds}
    )
    my_submission.to_csv("submission.csv", index=False)




## === cell 3
print("Submission saved. Preview:")
print(my_submission.head())
