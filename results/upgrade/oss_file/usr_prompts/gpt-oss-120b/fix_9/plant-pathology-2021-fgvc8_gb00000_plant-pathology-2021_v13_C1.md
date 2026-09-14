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

No external packages required in the script and installed.

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
try:
    import google.protobuf.message_factory as _mf

    if not hasattr(_mf.MessageFactory, "GetPrototype"):

        def _GetPrototype(self, descriptor):
            return self.pool.GetPrototype(descriptor)

        _mf.MessageFactory.GetPrototype = _GetPrototype  # type: ignore
except Exception:
    pass

try:
    import tensorflow as tf

    print("Num GPUs Available:", len(tf.config.list_physical_devices("GPU")))
except Exception as e:
    tf = None
    print("TensorFlow import failed:", e)




## === cell 1
import os
import pandas as pd
import numpy as np
from collections import Counter

train_path = "/kaggle/input/plant-pathology-2021-fgvc8/train.csv"
train_df = pd.read_csv(train_path)

all_tokens = []
for lbl in train_df["labels"].astype(str):
    all_tokens.extend(lbl.split())
token_counts = Counter(all_tokens)
most_common_tokens = [tok for tok, _ in token_counts.most_common(3)]
default_prediction = " ".join(most_common_tokens)  # used when model is unavailable

unique_tokens = sorted(token_counts.keys())
token_to_idx = {tok: i for i, tok in enumerate(unique_tokens)}
num_classes = len(unique_tokens)




## === cell 2
train_images_dir = "/kaggle/input/plant-pathology-2021-fgvc8/train_images"
train_df["image_path"] = train_df["image"].apply(
    lambda x: os.path.join(train_images_dir, x)
)


def multilabel_vector(label_str):
    vec = np.zeros(num_classes, dtype=np.float32)
    for tok in label_str.split():
        idx = token_to_idx.get(tok)
        if idx is not None:
            vec[idx] = 1.0
    return vec


train_df["label_vec"] = train_df["labels"].astype(str).apply(multilabel_vector)

train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
split_idx = int(0.8 * len(train_df))
df_train = train_df.iloc[:split_idx]
df_val = train_df.iloc[split_idx:]




## === cell 3
if tf is not None:
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    IMG_SIZE = (224, 224)

    def load_and_preprocess(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, IMG_SIZE)
        img = img / 255.0  # normalize to [0,1]
        return img, label

    train_ds = tf.data.Dataset.from_tensor_slices(
        (df_train["image_path"].values, np.stack(df_train["label_vec"].values))
    )
    val_ds = tf.data.Dataset.from_tensor_slices(
        (df_val["image_path"].values, np.stack(df_val["label_vec"].values))
    )

    train_ds = (
        train_ds.map(load_and_preprocess, num_parallel_calls=AUTOTUNE)
        .shuffle(1024)
        .batch(32)
        .prefetch(AUTOTUNE)
    )
    val_ds = (
        val_ds.map(load_and_preprocess, num_parallel_calls=AUTOTUNE)
        .batch(32)
        .prefetch(AUTOTUNE)
    )

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet"
    )
    base_model.trainable = False  # freeze backbone

    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="sigmoid")(x)
    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy")],
    )

    model.fit(train_ds, validation_data=val_ds, epochs=3, verbose=2)
else:
    model = None  # will trigger fallback later




## === cell 4
sample_sub_path = "/kaggle/input/plant-pathology-2021-fgvc8/sample_submission.csv"
test_df = pd.read_csv(sample_sub_path)
test_ids = test_df["image"].astype(str).tolist()
test_images_dir = "/kaggle/input/plant-pathology-2021-fgvc8/test_images"
test_paths = [os.path.join(test_images_dir, img_id) for img_id in test_ids]

predictions = []

if tf is not None and model is not None:
    def load_test(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, IMG_SIZE)
        img = img / 255.0
        return img

    test_ds = tf.data.Dataset.from_tensor_slices(test_paths)
    test_ds = test_ds.map(load_test, num_parallel_calls=AUTOTUNE).batch(32)

    prob_arrays = model.predict(test_ds, verbose=0)
    for probs in prob_arrays:
        token_idxs = np.where(probs > 0.5)[0]
        if len(token_idxs) == 0:
            predictions.append(default_prediction)
        else:
            tokens = [unique_tokens[i] for i in token_idxs]
            predictions.append(" ".join(tokens))
else:
    predictions = [default_prediction] * len(test_ids)




## === cell 5
submission = pd.DataFrame({"image": test_ids, "labels": predictions})
submission.to_csv("submission.csv", index=False)
