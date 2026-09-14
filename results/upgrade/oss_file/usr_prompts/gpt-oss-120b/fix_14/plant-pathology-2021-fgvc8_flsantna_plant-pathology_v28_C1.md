# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "3"

try:
    from google.protobuf import message_factory as _mf

    if not hasattr(_mf.MessageFactory, "GetPrototype"):
        def _fallback_get_prototype(self, *args, **kwargs):
            if hasattr(self, "GetMessageClass"):
                return self.GetMessageClass(*args, **kwargs)
            raise AttributeError(
                "MessageFactory has no GetPrototype or GetMessageClass"
            )

        _mf.MessageFactory.GetPrototype = _fallback_get_prototype
except Exception:
    pass

import pandas as pd
import numpy as np

BASE_INPUT = "/kaggle/input/plant-pathology-2021-fgvc8"
output_dir = "./"
test_dir = os.path.join(BASE_INPUT, "test_images")
sample_submission_path = os.path.join(BASE_INPUT, "sample_submission.csv")

image_dims = (300, 300, 3)

data_set = pd.read_csv(os.path.join(BASE_INPUT, "train.csv"))
df_labels = data_set["labels"]
one_hot = df_labels.str.get_dummies(sep=" ")
dataset_labels = one_hot.columns.to_list()

label_counts = one_hot.sum()
freq_threshold = 0.01  # 1% frequency instead of 5%
freq_labels = (
    (label_counts / label_counts.sum())
    .loc[lambda s: s >= freq_threshold]
    .index.tolist()
)
default_label_str = " ".join(freq_labels) if freq_labels else "healthy"




## === cell 1
if __name__ == "__main__":
    try:
        import tensorflow as tf

        tf_available = True
    except Exception as import_err:
        print(
            f"TensorFlow import failed ({import_err}); falling back to heuristic predictions."
        )
        tf = None
        tf_available = False

    use_tf = False
    model = None

    if tf_available:
        try:
            subset_df = data_set.sample(n=5000, random_state=42).reset_index(drop=True)
            subset_paths = (
                subset_df["image"]
                .apply(lambda x: os.path.join(BASE_INPUT, "train_images", x))
                .tolist()
            )
            subset_labels = one_hot.loc[subset_df.index].values.astype(np.float32)

            AUTOTUNE = tf.data.AUTOTUNE

            def _load_image(path):
                img_bytes = tf.io.read_file(path)
                img = tf.io.decode_jpeg(img_bytes, channels=3)
                img = tf.image.resize(img, [image_dims[0], image_dims[1]])
                img = tf.cast(img, tf.float32) / 255.0
                return img

            train_ds = tf.data.Dataset.from_tensor_slices((subset_paths, subset_labels))
            train_ds = train_ds.map(
                lambda p, l: (_load_image(p), l), num_parallel_calls=AUTOTUNE
            )
            train_ds = train_ds.shuffle(1024).batch(32).prefetch(AUTOTUNE)

            base = tf.keras.applications.MobileNetV2(
                input_shape=image_dims, weights="imagenet", include_top=False
            )
            base.trainable = False

            inputs = tf.keras.Input(shape=image_dims)
            x = base(inputs, training=False)
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            outputs = tf.keras.layers.Dense(len(dataset_labels), activation="sigmoid")(
                x
            )
            model = tf.keras.Model(inputs, outputs)

            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                loss="binary_crossentropy",
            )

            model.fit(train_ds, epochs=20, verbose=0)
            use_tf = True
            print("Fine‑tuned MobileNetV2 model ready for inference.")
        except Exception as train_err:
            print(
                f"Model training/loading failed ({train_err}); falling back to heuristic predictions."
            )
            model = None
            use_tf = False

    sample_sub = pd.read_csv(sample_submission_path)
    images_list = sample_sub["image"].tolist()

    prob_threshold = 0.1  # lower threshold to capture more classes
    predictions = []

    if use_tf:

        def _load_test_image(path):
            img_bytes = tf.io.read_file(path)
            img = tf.io.decode_jpeg(img_bytes, channels=3)
            img = tf.image.resize(img, [image_dims[0], image_dims[1]])
            img = tf.cast(img, tf.float32) / 255.0
            return img

        test_paths = [os.path.join(test_dir, img_name) for img_name in images_list]
        test_ds = tf.data.Dataset.from_tensor_slices(test_paths)
        test_ds = test_ds.map(_load_test_image, num_parallel_calls=tf.data.AUTOTUNE)
        test_ds = test_ds.batch(32)

        all_probs = model.predict(test_ds, verbose=0)
        for img_name, probs in zip(images_list, all_probs):
            selected_idx = [i for i, p in enumerate(probs) if p > prob_threshold]
            label_str = " ".join(dataset_labels[i] for i in selected_idx).strip()
            if not label_str:
                label_str = default_label_str
            predictions.append([img_name, label_str])
    else:
        for img_name in images_list:
            predictions.append([img_name, default_label_str])

    submission_df = pd.DataFrame(predictions, columns=["image", "labels"])
    submission_path = os.path.join(output_dir, "submission.csv")
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission written to {submission_path}")
