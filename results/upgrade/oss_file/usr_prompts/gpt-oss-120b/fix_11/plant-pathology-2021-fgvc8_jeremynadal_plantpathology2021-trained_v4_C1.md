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
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

import glob, numpy as np, pandas as pd, matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from tensorflow.keras import mixed_precision

print("Using tensorflow", tf.__version__)

tf.config.optimizer.set_jit(False)
tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.threading.set_intra_op_parallelism_threads(2)
mixed_precision.set_global_policy("mixed_float16")




## === cell 1
BASE_DIR = os.path.abspath(".")
possible_dirs = [
    os.path.join(BASE_DIR, "input", "plant-pathology-2021-fgvc8"),
    os.path.join(BASE_DIR, "data", "plant-pathology-2021-fgvc8"),
    os.path.join("/kaggle", "input", "plant-pathology-2021-fgvc8"),
]
DATA_DIR = next((d for d in possible_dirs if os.path.isdir(d)), None)
if DATA_DIR is None:
    raise FileNotFoundError(
        "Could not find the plant-pathology-2021-fgvc8 data directory."
    )

TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV = os.path.join(DATA_DIR, "sample_submission.csv")  # contains test image list
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")

df_train = pd.read_csv(TRAIN_CSV)
df_test = pd.read_csv(TEST_CSV)  # only 'image' column needed for submission

mlb = MultiLabelBinarizer()
train_labels = mlb.fit_transform(df_train["labels"].str.split())
class_names = mlb.classes_.tolist()
num_classes = len(class_names)

for idx, cls in enumerate(class_names):
    df_train[cls] = train_labels[:, idx]

df_train["stratify_key"] = df_train["labels"].apply(lambda x: x.split()[0])

train_df, val_df = train_test_split(
    df_train,
    test_size=0.1,
    random_state=42,
    stratify=df_train["stratify_key"],
)

train_df = train_df.drop(columns=["stratify_key"]).reset_index(drop=True)
val_df = val_df.drop(columns=["stratify_key"]).reset_index(drop=True)


class CFG:
    img_size = 224
    batch_size = 32
    epochs = 5
    lr = 1e-4
    classes = class_names


AUTOTUNE = tf.data.AUTOTUNE




## === cell 2
def parse_image(filename):
    img = tf.io.read_file(filename)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [CFG.img_size, CFG.img_size])
    img = tf.cast(img, tf.float32) / 255.0
    return img


def data_augmentation(image):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    return image


def prepare_dataset(df, img_dir, augment=False):
    paths = (
        df["image"].apply(lambda x: os.path.join(img_dir, str(x))).values.astype(str)
    )
    labels = df[CFG.classes].values.astype(np.float32)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(lambda x, y: (parse_image(x), y), num_parallel_calls=AUTOTUNE)
    if augment:
        ds = ds.map(lambda x, y: (data_augmentation(x), y), num_parallel_calls=AUTOTUNE)
    ds = ds.shuffle(1024).batch(CFG.batch_size).prefetch(AUTOTUNE).repeat()
    return ds


ds_train = prepare_dataset(train_df, TRAIN_IMG_DIR, augment=True)
ds_val = prepare_dataset(val_df, TRAIN_IMG_DIR, augment=False)




## === cell 3
base_model = tf.keras.applications.EfficientNetB0(
    include_top=False,
    input_shape=(CFG.img_size, CFG.img_size, 3),
    weights="imagenet",
    pooling="avg",
)

inputs = tf.keras.layers.Input(shape=(CFG.img_size, CFG.img_size, 3))
x = base_model(inputs, training=False)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(num_classes, activation="sigmoid", dtype="float32")(x)
model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(CFG.lr),
    loss="binary_crossentropy",
    metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy")],
)

model.summary()




## === cell 4
steps_per_epoch = max(1, len(train_df) // CFG.batch_size)
validation_steps = max(1, len(val_df) // CFG.batch_size)

model.fit(
    ds_train,
    epochs=CFG.epochs,
    steps_per_epoch=steps_per_epoch,
    validation_data=ds_val,
    validation_steps=validation_steps,
    verbose=2,
)




## === cell 5
test_paths = (
    df_test["image"]
    .apply(lambda x: os.path.join(TEST_IMG_DIR, str(x)))
    .values.astype(str)
)


def test_dataset(paths):
    ds = tf.data.Dataset.from_tensor_slices(paths)
    ds = ds.map(parse_image, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(CFG.batch_size).prefetch(AUTOTUNE)
    return ds


ds_test = test_dataset(test_paths)

pred_probs = model.predict(ds_test, verbose=0)
pred_labels = (pred_probs >= 0.5).astype(int)


def probs_to_labels(row):
    idxs = np.where(row == 1)[0]
    if len(idxs) == 0:
        return "healthy"
    return " ".join([class_names[i] for i in idxs])


pred_strings = [probs_to_labels(row) for row in pred_labels]

submission = pd.DataFrame({"image": df_test["image"], "labels": pred_strings})

submission_path = os.path.join(BASE_DIR, "submission.csv")
submission.to_csv(submission_path, index=False)
print(f"Submission file written to {submission_path}")
