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

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import pandas as pd
import numpy as np
from collections import Counter




## === cell 1
test_image_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images"
train_image_dir = "/kaggle/input/cassava-leaf-disease-classification/train_images"
train_csv_path = "/kaggle/input/cassava-leaf-disease-classification/train.csv"
sample_submission_path = (
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)

train_df = pd.read_csv(train_csv_path)
most_common_label = train_df["label"].mode()[0]




## === cell 2
model_paths_and_sizes = []  # no external models provided

models = []
for path, input_size in model_paths_and_sizes:
    try:
        from tensorflow.keras.models import load_model

        model = load_model(path)
        models.append((model, input_size))
    except Exception as e:
        print(f"Could not load model {path}: {e}")

if not models:
    try:
        import tensorflow as tf
        from tensorflow.keras import layers, Model

        tf.config.optimizer.set_jit(True)

        if tf.config.list_physical_devices("GPU"):
            from tensorflow.keras.mixed_precision import experimental as mixed_precision

            policy = mixed_precision.Policy("mixed_float16")
            mixed_precision.set_policy(policy)

        IMG_SIZE = (224, 224)
        BATCH_SIZE = 256  # <<< reduced from 2048 to enable GPU training efficiently
        EPOCHS_FROZEN = 3
        EPOCHS_UNFROZEN = 3
        NUM_CLASSES = 5
        SEED = 42

        tf.random.set_seed(SEED)

        def _parse_function(filename, label):
            img_raw = tf.io.read_file(filename)
            img = tf.image.decode_jpeg(img_raw, channels=3)
            img = tf.image.resize(img, IMG_SIZE)
            img = img / 255.0
            return img, label

        image_paths = [
            os.path.join(train_image_dir, img_id) for img_id in train_df["image_id"]
        ]
        labels = train_df["label"].values.astype(np.int32)

        rng = np.random.default_rng(SEED)
        indices = rng.permutation(len(image_paths))
        split_idx = int(0.9 * len(indices))
        train_idx, val_idx = indices[:split_idx], indices[split_idx:]

        train_paths = np.array(image_paths)[train_idx]
        train_labels = labels[train_idx]
        val_paths = np.array(image_paths)[val_idx]
        val_labels = labels[val_idx]

        train_ds = (
            tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
            .map(_parse_function, num_parallel_calls=tf.data.AUTOTUNE)
            .cache()  # cache in RAM after first epoch
            .shuffle(1000, seed=SEED)
            .batch(BATCH_SIZE)
            .prefetch(tf.data.AUTOTUNE)
        )
        val_ds = (
            tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
            .map(_parse_function, num_parallel_calls=tf.data.AUTOTUNE)
            .cache()
            .batch(BATCH_SIZE)
            .prefetch(tf.data.AUTOTUNE)
        )

        tf.keras.backend.clear_session()

        base = tf.keras.applications.MobileNetV2(
            input_shape=IMG_SIZE + (3,), include_top=False, weights=None
        )
        base.trainable = False  # frozen phase

        inputs = layers.Input(shape=IMG_SIZE + (3,))
        x = base(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
        model = Model(inputs, outputs)

        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_FROZEN, verbose=0)

        base.trainable = True
        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-4),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_UNFROZEN, verbose=0)

        models.append((model, IMG_SIZE))
    except Exception as e:
        print(f"Training fallback model failed: {e}")
        models = []




## === cell 3
image_predictions = []

test_filenames = sorted(
    [
        f
        for f in os.listdir(test_image_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
)

if models:
    import tensorflow as tf

    test_paths = [os.path.join(test_image_dir, f) for f in test_filenames]

    def _parse_test(filename):
        img_raw = tf.io.read_file(filename)
        img = tf.image.decode_jpeg(img_raw, channels=3)
        img = tf.image.resize(img, IMG_SIZE)
        img = img / 255.0
        return img

    test_ds = (
        tf.data.Dataset.from_tensor_slices(test_paths)
        .map(_parse_test, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    all_prob_arrays = []  # list of (N, NUM_CLASSES) arrays
    for model, input_size in models:
        probs = model.predict(test_ds, verbose=0)  # shape (N, NUM_CLASSES)
        all_prob_arrays.append(probs)

    all_class_arrays = [np.argmax(p, axis=1) for p in all_prob_arrays]

    for idx, image_id in enumerate(test_filenames):
        model_predictions = [cls_arr[idx] for cls_arr in all_class_arrays]
        votes = Counter(model_predictions)
        most_common = votes.most_common()
        final_predicted_class = most_common[0][0]

        if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
            tied_classes = [cls for cls, cnt in most_common if cnt == most_common[0][1]]
            avg_conf = {}
            for cls in tied_classes:
                confidences = [prob_arr[idx][cls] for prob_arr in all_prob_arrays]
                avg_conf[cls] = sum(confidences) / len(confidences)
            final_predicted_class = max(tied_classes, key=lambda c: avg_conf[c])

        image_predictions.append({"image_id": image_id, "label": final_predicted_class})
else:
    for image_id in test_filenames:
        image_predictions.append({"image_id": image_id, "label": most_common_label})




## === cell 4
submission_df = pd.DataFrame(image_predictions)
submission_path = "/kaggle/working/submission.csv"
submission_df.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")




## === cell 5
submission_df.head()
