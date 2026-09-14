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

0.7491043397968614

# 6. Current score

0.57709

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.57709) has done: 'Your script currently can’t yield a Kaggle score because it exits early and writes a fallback submission when the external `model_dir` isn’t present in the provided data paths. I make it always produce a valid `submission.csv` by (1) removing the protobuf downgrade/restart logic (it can break TF 2.18 runtime) and (2) replacing the missing external model dependency with an in-notebook TF image model that keeps the same “predict probabilities then threshold then space-delimited labels” core semantics. To move toward the target score without changing the overall approach, I keep the same image size, use a standard pretrained backbone, train briefly on the provided `train.csv`/`train_images` with multi-label BCE, then predict on test and apply a fixed threshold (0.7) like your current code. This is minimal but turns “Not yielded” into a real submission and should produce a materially better score than the fallback “all healthy”.'

# 9. Code solution

## === cell 0
import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf

SEED = 42
tf.keras.utils.set_random_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

print("TensorFlow:", tf.__version__)
print("Pandas:", pd.__version__)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
output_dir = "./"
data_root = "../input/plant-pathology-2021-fgvc8/"
train_csv_path = os.path.join(data_root, "train.csv")
sample_path = os.path.join(data_root, "sample_submission.csv")
train_dir = os.path.join(data_root, "train_images/")
test_dir = os.path.join(data_root, "test_images/")

image_dims = (300, 300, 3)

data_set = pd.read_csv(train_csv_path)
df_labels = data_set["labels"].fillna("")
one_hot = df_labels.str.get_dummies(sep=" ")
dataset_labels = one_hot.columns.to_list()

print("Train rows:", len(data_set))
print("Num classes:", len(dataset_labels))
print("First classes:", dataset_labels[:10])



## === cell 2
AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 16

train_files = data_set["image"].astype(str).tolist()
train_paths = [os.path.join(train_dir, f) for f in train_files]
y = one_hot.values.astype("float32")


def _load_and_preprocess(path, label=None):
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, [image_dims[0], image_dims[1]])
    img = tf.cast(img, tf.float32)  # 0..255
    if label is None:
        return img
    return img, label


idx = np.arange(len(train_paths))
rng = np.random.default_rng(SEED)
rng.shuffle(idx)
val_frac = 0.1
val_size = int(len(idx) * val_frac)
val_idx = idx[:val_size]
tr_idx = idx[val_size:]

tr_paths = [train_paths[i] for i in tr_idx]
tr_y = y[tr_idx]
val_paths = [train_paths[i] for i in val_idx]
val_y = y[val_idx]

train_ds = tf.data.Dataset.from_tensor_slices((tr_paths, tr_y))
train_ds = train_ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
train_ds = (
    train_ds.map(_load_and_preprocess, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_y))
val_ds = (
    val_ds.map(_load_and_preprocess, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

print("Train batches:", tf.data.experimental.cardinality(train_ds).numpy())
print("Val batches:", tf.data.experimental.cardinality(val_ds).numpy())



## === cell 3
import keras

num_classes = len(dataset_labels)

base = keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=image_dims,
)
base.trainable = False  # keeps training stable and fast within time limit

inp = keras.Input(shape=image_dims, dtype=tf.float32, name="image")
x = keras.applications.efficientnet.preprocess_input(inp)
x = base(x, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.2, seed=SEED)(x)
out = keras.layers.Dense(num_classes, activation="sigmoid", name="predictions")(x)
model = keras.Model(inp, out)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
)

EPOCHS = 2
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=2)



## === cell 4
images_path_list = sorted(
    [f for f in os.listdir(test_dir) if f.lower().endswith(".jpg")]
)
test_paths = [os.path.join(test_dir, f) for f in images_path_list]

test_ds = tf.data.Dataset.from_tensor_slices(test_paths)
test_ds = (
    test_ds.map(_load_and_preprocess, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

preds = model.predict(test_ds, verbose=1)
preds = np.asarray(preds)

thr = 0.7
values = []
for name, pred_row in zip(images_path_list, preds):
    index_values = np.where(pred_row > thr)[0].tolist()
    classes_img_list = [
        dataset_labels[i] for i in index_values if i < len(dataset_labels)
    ]
    classes_img = " ".join(classes_img_list).strip()
    if classes_img == "":
        classes_img = "healthy"
    values.append([name, classes_img])

csv_pd = pd.DataFrame(values, columns=["image", "labels"])

sample = pd.read_csv(sample_path)
csv_pd = sample[["image"]].merge(csv_pd, on="image", how="left")
csv_pd["labels"] = csv_pd["labels"].fillna("healthy")

out_path = os.path.join(output_dir, "submission.csv")
csv_pd.to_csv(out_path, index=False)
print("Wrote submission.csv with shape:", csv_pd.shape)
print(csv_pd.head())
print("Saved to:", out_path)
