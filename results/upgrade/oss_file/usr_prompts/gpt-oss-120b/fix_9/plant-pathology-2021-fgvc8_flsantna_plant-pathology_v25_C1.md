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
Detect apple diseases from images.

## Metric
Mean F1-Score

## Submission Format
labels should be a space-delimited list.

The file should contain a header and have the following format:

```
image, labels
85f8cb619c66b863.jpg,healthy
ad8770db05586b59.jpg,healthy
c7b03e718489f3ca.jpg,healthy
```

## Dataset
**train.csv** - the training set metadata.

- `image` - the image ID.
- `labels` - the target classes, a space delimited list of all diseases found in the image. Unhealthy leaves with too many diseases to classify visually will have the `complex` class, and may also have a subset of the diseases identified.

**sample_submission.csv** - A sample submission file in the correct format.

- `image`
- `labels`

**train_images** - The training set images.

**test_images** - The test set images. This competition has a hidden test set: only three images are provided here as samples while the remaining 5,000 images will be available to your notebook once it is submitted.

# 2. Python version

3.9

# 3. Installed packages

geopandas==0.14.4
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
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
        input/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
        working/
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
```

-> data/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> data/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> (stopped after 10 files for performance)

# 5. Target score

0.7860203139427543

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
try:
    from google.protobuf import message_factory

    if not hasattr(message_factory.MessageFactory, "GetPrototype"):

        def _get_proto(self, *args, **kwargs):
            return self.GetMessageClass(*args, **kwargs)

        message_factory.MessageFactory.GetPrototype = _get_proto
except Exception:
    pass

import pandas as pd
import tensorflow as tf
import os
from sklearn.model_selection import train_test_split
import numpy as np




## === cell 1
base_input = "/kaggle/input"

test_dir = os.path.join(base_input, "plant-pathology-2021-fgvc8", "test_images")
model_dir = os.path.join(base_input, "model-effb7-01", "epoch-5")

image_dims = (300, 300, 3)

data_set = pd.read_csv(
    os.path.join(base_input, "plant-pathology-2021-fgvc8", "train.csv")
)
df_labels = data_set["labels"]
one_hot = df_labels.str.get_dummies(sep=" ")
dataset_labels = one_hot.columns.to_list()
num_classes = len(dataset_labels)




## === cell 2
if __name__ == "__main__":
    if os.path.isdir(model_dir) and (
        os.path.isfile(os.path.join(model_dir, "saved_model.pb"))
        or os.path.isfile(os.path.join(model_dir, "saved_model.pbtxt"))
    ):
        model = tf.keras.Sequential(
            [tf.keras.layers.TFSMLayer(model_dir, call_endpoint="serving_default")]
        )
    else:
        backbone = tf.keras.applications.EfficientNetB7(
            include_top=False,
            input_shape=image_dims,
            pooling="avg",
            weights="imagenet",
        )
        backbone.trainable = False  # freeze for initial training

        outputs = tf.keras.layers.Dense(num_classes, activation="sigmoid")(
            backbone.output
        )
        model = tf.keras.Model(inputs=backbone.input, outputs=outputs)

        train_images_dir = os.path.join(
            base_input, "plant-pathology-2021-fgvc8", "train_images"
        )

        def _load_image(path):
            raw = tf.io.read_file(path)
            img = tf.io.decode_jpeg(raw, channels=3)
            img = tf.image.resize(img, [image_dims[0], image_dims[1]])
            img = tf.cast(img, tf.float32) / 255.0
            return img

        image_paths = [
            os.path.join(train_images_dir, fname) for fname in data_set["image"].values
        ]
        label_arrays = one_hot.values.astype(np.float32)

        train_idx, val_idx = train_test_split(
            np.arange(len(image_paths)),
            test_size=0.1,
            random_state=42,
            stratify=one_hot.values,
        )

        train_paths = [image_paths[i] for i in train_idx]
        train_labels = label_arrays[train_idx]
        val_paths = [image_paths[i] for i in val_idx]
        val_labels = label_arrays[val_idx]

        def _make_dataset(paths, labels, shuffle=True):
            ds = tf.data.Dataset.from_tensor_slices((paths, labels))
            ds = ds.map(
                lambda p, l: (_load_image(p), l),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
            if shuffle:
                ds = ds.shuffle(buffer_size=1024, seed=42)
            ds = ds.batch(32).prefetch(tf.data.AUTOTUNE)
            return ds

        train_ds = _make_dataset(train_paths, train_labels, shuffle=True)
        val_ds = _make_dataset(val_paths, val_labels, shuffle=False)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss="binary_crossentropy",
            metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy")],
        )
        model.fit(train_ds, validation_data=val_ds, epochs=3, verbose=2)

    image_paths = tf.io.gfile.glob(os.path.join(test_dir, "**/*.jpg"))
    image_paths = sorted(image_paths)  # deterministic order
    images_filenames = [os.path.basename(p) for p in image_paths]

    def _load_and_preprocess(path):
        raw = tf.io.read_file(path)
        img = tf.io.decode_jpeg(raw, channels=3)
        img = tf.cast(img, tf.float32) / 255.0
        img = tf.image.resize(img, [image_dims[0], image_dims[1]])
        img = tf.ensure_shape(img, [image_dims[0], image_dims[1], 3])
        return img

    batch_size = 32
    threshold = 0.5
    values = []

    for start_idx in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start_idx : start_idx + batch_size]
        batch_filenames = images_filenames[start_idx : start_idx + batch_size]

        batch_images = tf.stack([_load_and_preprocess(p) for p in batch_paths])

        preds = model(batch_images, training=False)
        if isinstance(preds, dict):
            preds = list(preds.values())[0]
        preds_np = preds.numpy()  # (batch, num_labels)

        for fname, row in zip(batch_filenames, preds_np):
            index_values = [idx for idx, v in enumerate(row) if v > threshold]
            if not index_values:  # fallback to highest probability
                index_values = [int(tf.argmax(row).numpy())]
            classes_img = " ".join([dataset_labels[idx] for idx in index_values])
            values.append([fname, classes_img.strip()])

    csv_pd = pd.DataFrame(values, columns=["image", "labels"])
    csv_pd.to_csv(os.path.join("/kaggle/working", "submission.csv"), index=False)
