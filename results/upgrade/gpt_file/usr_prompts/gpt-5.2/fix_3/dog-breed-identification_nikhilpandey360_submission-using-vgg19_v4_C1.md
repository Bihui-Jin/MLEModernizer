# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Given a dataset of images of dogs, predict the breed of each image.

## Metric
Multi Class Log Loss.

## Submission Format
For each image in the test set, you must predict a probability for each of the different breeds. The file should contain a header and have the following format:
```
id,affenpinscher,afghan_hound,..,yorkshire_terrier
000621fb3cbb32d8935728e48679680e,0.0083,0.0,...,0.0083
etc.
```

## Dataset Description
- `train.zip` - the training set, you are provided the breed for these dogs
- `test.zip` - the test set, you must predict the probability of each breed for each image
- `sample_submission.csv` - a sample submission file in the correct format
- `labels.csv` - the breeds for the images in the train set

# 2. Python version

3.7

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (169 lines)
            labels.csv (9200 lines)
            labels.csv.zip (201.6 kB)
            sample_submission.csv (1024 lines)
            sample_submission.csv.zip (32.1 kB)
            test.zip (36.4 MB)
            train.zip (324.7 MB)
            dog-breed-identification/
                description.md (169 lines)
                labels.csv (9200 lines)
                ... and 5 other files
                dog-breed-identification/
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
            test/
                bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                ... and 1021 other files
                test/
            train/
                868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                ... and 9197 other files
                train/
        input/
            description.md (169 lines)
            labels.csv (9200 lines)
            labels.csv.zip (201.6 kB)
            sample_submission.csv (1024 lines)
            sample_submission.csv.zip (32.1 kB)
            test.zip (36.4 MB)
            train.zip (324.7 MB)
            dog-breed-identification/
                description.md (169 lines)
                labels.csv (9200 lines)
                ... and 5 other files
                dog-breed-identification/
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
            test/
                bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                ... and 1021 other files
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
            train/
                868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                ... and 9197 other files
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
        working/
            dog-breed-identification/
                description.md (169 lines)
                labels.csv (9200 lines)
                ... and 5 other files
                dog-breed-identification/
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
```

-> data/dog-breed-identification/labels.csv has 9199 rows and 2 columns.
The columns are: id, breed

-> data/dog-breed-identification/sample_submission.csv has 1023 rows and 121 columns.
The columns are: id, affenpinscher, afghan_hound, african_hunting_dog, airedale, american_staffordshire_terrier, appenzeller, australian_terrier, basenji, basset, beagle, bedlington_terrier, bernese_mountain_dog, black-and-tan_coonhound, blenheim_spaniel... and 106 more columns

-> data/labels.csv has 9199 rows and 2 columns.
The columns are: id, breed

-> data/sample_submission.csv has 1023 rows and 121 columns.
The columns are: id, affenpinscher, afghan_hound, african_hunting_dog, airedale, american_staffordshire_terrier, appenzeller, australian_terrier, basenji, basset, beagle, bedlington_terrier, bernese_mountain_dog, black-and-tan_coonhound, blenheim_spaniel... and 106 more columns

-> input/dog-breed-identification/labels.csv has 9199 rows and 2 columns.
The columns are: id, breed

-> input/dog-breed-identification/sample_submission.csv has 1023 rows and 121 columns.
The columns are: id, affenpinscher, afghan_hound, african_hunting_dog, airedale, american_staffordshire_terrier, appenzeller, australian_terrier, basenji, basset, beagle, bedlington_terrier, bernese_mountain_dog, black-and-tan_coonhound, blenheim_spaniel... and 106 more columns

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print("TF version:", tf.__version__)

try:
    tf.keras.utils.set_random_seed(42)
except Exception:
    tf.random.set_seed(42)
np.random.seed(42)

BASE_INPUT_CANDIDATES = [
    "../input/dog-breed-identification",
    "/kaggle/input/dog-breed-identification",
    "../input",
    "/kaggle/input",
]
BASE = None
for p in BASE_INPUT_CANDIDATES:
    if os.path.exists(p):
        if os.path.basename(p) == "input":
            nested = os.path.join(p, "dog-breed-identification")
            if os.path.exists(nested):
                BASE = nested
                break
        BASE = p
        break

if BASE is None:
    raise FileNotFoundError(
        "Could not locate Kaggle input directory. Checked: "
        + str(BASE_INPUT_CANDIDATES)
    )

LABELS_CSV = os.path.join(BASE, "labels.csv")
SAMPLE_SUB_CSV = os.path.join(BASE, "sample_submission.csv")
TRAIN_DIR = os.path.join(BASE, "train")
TEST_DIR = os.path.join(BASE, "test")

print("Using BASE:", BASE)
print("Found labels:", os.path.exists(LABELS_CSV), LABELS_CSV)
print("Found sample_submission:", os.path.exists(SAMPLE_SUB_CSV), SAMPLE_SUB_CSV)
print("Found train dir:", os.path.exists(TRAIN_DIR), TRAIN_DIR)
print("Found test dir:", os.path.exists(TEST_DIR), TEST_DIR)

labels_df = pd.read_csv(LABELS_CSV)
sample_sub = pd.read_csv(SAMPLE_SUB_CSV)

breed_cols = [c for c in sample_sub.columns if c != "id"]
num_classes = len(breed_cols)

print("labels_df shape:", labels_df.shape)
print("num_classes:", num_classes)

breed_to_idx = {b: i for i, b in enumerate(breed_cols)}
labels_df = labels_df[labels_df["breed"].isin(breed_to_idx)].copy()
labels_df["label"] = labels_df["breed"].map(breed_to_idx).astype(int)




## === cell 1
labels_df["filepath"] = TRAIN_DIR + "/" + labels_df["id"].astype(str) + ".jpg"

fps = labels_df["filepath"].to_numpy()
exists_mask = np.fromiter((os.path.exists(p) for p in fps), dtype=bool, count=len(fps))
missing = int((~exists_mask).sum())
if missing:
    labels_df = labels_df.loc[exists_mask].copy()

train_df, val_df = train_test_split(
    labels_df[["id", "breed", "label", "filepath"]],
    test_size=0.15,
    random_state=42,
    stratify=labels_df["label"],
)

print("Train/Val sizes:", train_df.shape, val_df.shape)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE


@tf.function
def decode_and_resize(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    return img, label


@tf.function
def decode_and_resize_nolabel(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    return img


def make_ds(df, training, cache_name=None):
    paths = df["filepath"].values
    labels = df["label"].values.astype(np.int32)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(min(len(df), 4096), seed=42, reshuffle_each_iteration=True)

    opts = tf.data.Options()
    opts.experimental_deterministic = (
        True  # preserve stable iteration ordering behavior
    )
    ds = ds.with_options(opts)

    ds = ds.map(decode_and_resize, num_parallel_calls=AUTOTUNE)
    ds = ds.apply(
        tf.data.experimental.ignore_errors()
    )  # avoid stalls if a file is unreadable

    if cache_name is None:
        cache_name = "train" if training else "val"
    cache_path = os.path.join("/kaggle/working", f"tfds_cache_{cache_name}")
    ds = ds.cache(cache_path)

    return ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)


train_ds = make_ds(train_df, training=True, cache_name="train_224")
val_ds = make_ds(val_df, training=False, cache_name="val_224")




## === cell 2
base_model = tf.keras.applications.ResNet50(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
)
base_model.trainable = False

inputs = keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = tf.keras.applications.resnet.preprocess_input(inputs)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)
model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()




## === cell 3
EPOCHS = 3

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    verbose=1,
)




## === cell 4
test_files = [f for f in os.listdir(TEST_DIR) if f.lower().endswith(".jpg")]
test_files.sort()
test_ids = [os.path.splitext(f)[0] for f in test_files]
test_paths = [os.path.join(TEST_DIR, f) for f in test_files]

test_ds = tf.data.Dataset.from_tensor_slices(test_paths)

opts = tf.data.Options()
opts.experimental_deterministic = True
test_ds = test_ds.with_options(opts)

test_ds = test_ds.map(decode_and_resize_nolabel, num_parallel_calls=AUTOTUNE)
test_ds = test_ds.apply(tf.data.experimental.ignore_errors())
test_ds = test_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

probs = model.predict(test_ds, verbose=1)
probs = np.asarray(probs, dtype=np.float64)

if probs.shape != (len(test_ids), num_classes):
    raise ValueError(
        f"Pred shape {probs.shape} does not match expected {(len(test_ids), num_classes)}"
    )

row_sums = probs.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1.0
probs = probs / row_sums

sub = pd.DataFrame(probs, columns=breed_cols)
sub.insert(0, "id", test_ids)
sub = sub[sample_sub.columns]

sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())
