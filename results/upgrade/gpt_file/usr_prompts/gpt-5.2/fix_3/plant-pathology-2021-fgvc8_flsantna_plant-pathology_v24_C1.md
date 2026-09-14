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
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import keras

tf.random.set_seed(42)
np.random.seed(42)

output_dir = "./"
base_dir = "../input/plant-pathology-2021-fgvc8"
train_csv_path = os.path.join(base_dir, "train.csv")
test_dir = os.path.join(base_dir, "test_images")
sample_sub_path = os.path.join(base_dir, "sample_submission.csv")

image_dims = (300, 300, 3)

data_set = pd.read_csv(train_csv_path)
df_labels = data_set["labels"].astype(str)
one_hot = df_labels.str.get_dummies(sep=" ")
dataset_labels = one_hot.columns.to_list()
num_classes = len(dataset_labels)

print(f"Found {len(data_set)} training rows and {num_classes} classes.")
print("First classes:", dataset_labels[:10])




## === cell 1
def decode_and_resize_image(path):
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_image(
        img_bytes, channels=3, dtype=tf.float32, expand_animations=False
    )
    img = tf.image.resize(img, [image_dims[0], image_dims[1]])
    return img


def to_label_string(probs, threshold=0.8):
    idx = np.where(probs > threshold)[0].tolist()
    if len(idx) == 0:
        idx = [int(np.argmax(probs))]
    return " ".join([dataset_labels[i] for i in idx])




## === cell 2


def build_model():
    inp = keras.Input(shape=image_dims, dtype=tf.float32)

    x = keras.applications.efficientnet.preprocess_input(inp)

    base = keras.applications.EfficientNetB7(
        include_top=False,
        weights="imagenet",
        input_tensor=x,
        pooling="avg",
    )
    base.trainable = True

    x = base.output
    x = keras.layers.Dropout(0.2)(x)
    logits = keras.layers.Dense(num_classes, name="logits")(x)

    model = keras.Model(inp, logits)
    return model


model = build_model()
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
)

print(model.output_shape)



## === cell 3


def make_train_dataset(df, batch_size=8, shuffle=True):
    img_paths = (
        os.path.join(base_dir, "train_images") + "/" + df["image"].astype(str)
    ).values
    y = (
        df["labels"]
        .astype(str)
        .str.get_dummies(sep=" ")
        .reindex(columns=dataset_labels, fill_value=0)
        .values.astype(np.float32)
    )

    ds = tf.data.Dataset.from_tensor_slices((img_paths, y))

    if shuffle:
        ds = ds.shuffle(
            buffer_size=min(len(df), 2048), seed=42, reshuffle_each_iteration=True
        )

    def _load(path, label):
        img = decode_and_resize_image(path)
        img = img * 255.0
        return img, label

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


perm = np.random.RandomState(42).permutation(len(data_set))
val_size = int(0.1 * len(data_set))
val_idx = perm[:val_size]
train_idx = perm[val_size:]

train_df = data_set.iloc[train_idx].reset_index(drop=True)
val_df = data_set.iloc[val_idx].reset_index(drop=True)

train_ds = make_train_dataset(train_df, batch_size=8, shuffle=True)
val_ds = make_train_dataset(val_df, batch_size=8, shuffle=False)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=1,
    verbose=1,
)



## === cell 4

if __name__ == "__main__":
    images_path_list = sorted(
        [f for f in os.listdir(test_dir) if f.lower().endswith(".jpg")]
    )

    values = []
    for fname in images_path_list:
        img = decode_and_resize_image(os.path.join(test_dir, fname))
        img = img * 255.0
        batch = tf.expand_dims(img, axis=0)

        pred = model(batch, training=False)
        pred = tf.convert_to_tensor(pred)

        pred_probs = tf.sigmoid(pred)
        probs_1d = pred_probs[0].numpy()
        label_str = to_label_string(probs_1d, threshold=0.8)

        values.append([fname, label_str])

    submission = pd.DataFrame(values, columns=["image", "labels"])

    if os.path.exists(sample_sub_path):
        sample_sub = pd.read_csv(sample_sub_path)
        submission = sample_sub[["image"]].merge(submission, on="image", how="left")
        submission["labels"] = submission["labels"].fillna("healthy")

    out_path = os.path.join(output_dir, "submission.csv")
    submission.to_csv(out_path, index=False)

    assert (
        submission.shape[0] == len(pd.read_csv(sample_sub_path))
        if os.path.exists(sample_sub_path)
        else len(images_path_list)
    )
    assert list(submission.columns) == ["image", "labels"]
    assert submission["labels"].isna().sum() == 0
    print(f"Wrote {out_path} with {len(submission)} rows.")
