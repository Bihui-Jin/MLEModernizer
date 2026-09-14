# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Create a classifier to predict whether an image contains a cactus.

## Metric
Area under the ROC curve.

## Submission Format
For each ID in the test set, you must predict a probability for the `has_cactus` variable. The file should contain a header and have the following format:

```
id,has_cactus
000940378805c44108d287872b2f04ce.jpg,0.5
0017242f54ececa4512b4d7937d1e21e.jpg,0.5
001ee6d8564003107853118ab87df407.jpg,0.5
etc.
```

## Dataset
This dataset contains a large number of 32 x 32 thumbnail images containing aerial photos of a cactus. The file name of an image corresponds to its `id`.

- **train/** - the training set images
- **test/** - the test set images (you must predict the labels of these)
- **train.csv** - the training set labels, indicates whether the image has a cactus (`has_cactus = 1`)
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.13

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
        input/
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
        working/
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
```

-> data/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> data/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

shown = 0
for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        shown += 1
        if shown >= 50:
            break
    if shown >= 50:
        break



## === cell 1
import zipfile

extract_dir = "/kaggle/working"

with zipfile.ZipFile(
    "/kaggle/input/aerial-cactus-identification/train.zip", "r"
) as zip_ref:
    zip_ref.extractall(extract_dir)

with zipfile.ZipFile(
    "/kaggle/input/aerial-cactus-identification/test.zip", "r"
) as zip_ref:
    zip_ref.extractall(extract_dir)



## === cell 2
for dirname, _, _ in os.walk("/kaggle/working"):
    print(dirname)



## === cell 3
pass




## === cell 4
def resolve_data_dirs(base="/kaggle/working"):
    candidates = [
        (os.path.join(base, "train"), os.path.join(base, "test")),
        (
            os.path.join(base, "aerial-cactus-identification", "train"),
            os.path.join(base, "aerial-cactus-identification", "test"),
        ),
    ]
    for tr, te in candidates:
        if os.path.isdir(tr) and os.path.isdir(te):
            return tr, te
    found_train, found_test = None, None
    for root, dirs, _ in os.walk(base):
        if "train" in dirs and found_train is None:
            found_train = os.path.join(root, "train")
        if "test" in dirs and found_test is None:
            found_test = os.path.join(root, "test")
    if found_train and found_test:
        return found_train, found_test
    raise FileNotFoundError(
        "Could not locate extracted train/test directories under /kaggle/working"
    )


train_dir, test_dir = resolve_data_dirs("/kaggle/working")

train_df = pd.read_csv("/kaggle/input/aerial-cactus-identification/train.csv")
print(train_df.head())
print("Resolved train_dir:", train_dir)
print("Resolved test_dir:", test_dir)




## === cell 5
def count_files(directory):
    return len(
        [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    )


train_count = count_files(train_dir)
test_count = count_files(test_dir)

print(f"Train images: {train_count}")
print(f"Test images: {test_count}")

assert (
    train_count > 0 and test_count > 0
), "Train/test image folders not found or empty."



## === cell 6
class_ratio = train_df["has_cactus"].value_counts(normalize=True) * 100
print(class_ratio)



## === cell 7
import matplotlib.pyplot as plt

counts = train_df["has_cactus"].value_counts()
labels = ["Has Cactus (1)", "No Cactus (0)"]
colors = ["lightgreen", "lightcoral"]

plt.figure(figsize=(6, 6))
plt.pie(counts, labels=labels, autopct="%1.1f%%", startangle=90, colors=colors)
plt.title("Distribution of Cactus Presence (has_cactus)")
plt.axis("equal")
plt.show()



## === cell 8
import cv2

imgs = []
for idx in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]:
    path = os.path.join(train_dir, train_df.loc[idx, "id"])
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    imgs.append(img)

plt.figure(figsize=(10, 10))
for i in range(12):
    plt.subplot(4, 3, i + 1)
    plt.imshow(imgs[i])
    plt.axis("off")
plt.tight_layout()
plt.show()



## === cell 9
train_df["has_cactus"] = train_df["has_cactus"].astype("str")



## === cell 10
import random


def custom_preprocessing(image):
    k = random.randint(0, 3)
    image = np.rot90(image, k)

    if random.random() > 0.5:
        image = np.fliplr(image)

    if random.random() > 0.5:
        image = np.flipud(image)

    factor = random.uniform(0.8, 1.2)
    image = np.clip(image.astype(np.float32) * factor, 0.0, 255.0) / 255.0
    return image.astype(np.float32)




## === cell 11
import tensorflow as tf

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

df = train_df.copy()
df["_label_int"] = df["has_cactus"].astype(int)

idx = np.arange(len(df))
rng = np.random.default_rng(SEED)
rng.shuffle(idx)
val_size = int(round(0.10 * len(df)))
val_idx = idx[:val_size]
train_idx = idx[val_size:]

train_split = df.iloc[train_idx].reset_index(drop=True)
val_split = df.iloc[val_idx].reset_index(drop=True)

print("Train split:", train_split.shape, "Val split:", val_split.shape)


def _read_image_bytes(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, [32, 32], method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32)
    return img


def _augment_np(img):
    out = tf.numpy_function(custom_preprocessing, [img], tf.float32)
    out.set_shape([32, 32, 3])
    return out


def make_ds(split_df, training, batch_size):
    paths = tf.constant(
        [os.path.join(train_dir, fid) for fid in split_df["id"].tolist()]
    )
    labels = tf.constant(split_df["_label_int"].astype(np.float32).values)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(
            buffer_size=len(split_df), seed=SEED, reshuffle_each_iteration=True
        )

    def _map(path, label):
        img = _read_image_bytes(path)
        if training:
            img = _augment_np(img)
        else:
            img = img / 255.0
        return img, label

    ds = ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
    return ds


train_dataset = make_ds(train_split, training=True, batch_size=128)
val_dataset = make_ds(val_split, training=False, batch_size=64)

for xb, yb in train_dataset.take(1):
    print("Train batch:", xb.shape, yb.shape, xb.dtype, yb.dtype)



## === cell 12
import cv2
import matplotlib.pyplot as plt

cactus = []
for i in range(12):
    path = os.path.join(train_dir, train_df.loc[i, "id"])
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cactus.append(img)

cactus_augmented = [custom_preprocessing(img) for img in cactus]

plt.figure(figsize=(10, 10))
for i in range(12):
    plt.subplot(4, 3, i + 1)
    plt.imshow(cactus_augmented[i])
    plt.title(f"Image {i+1}")
    plt.axis("off")
plt.tight_layout()
plt.show()



## === cell 13
sample_sub = pd.read_csv(
    "/kaggle/input/aerial-cactus-identification/sample_submission.csv"
)

test_paths = tf.constant(
    [os.path.join(test_dir, fid) for fid in sample_sub["id"].tolist()]
)
test_ds = tf.data.Dataset.from_tensor_slices(test_paths)


def _map_test(path):
    img = _read_image_bytes(path)
    img = img / 255.0
    return img


test_dataset = (
    test_ds.map(_map_test, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(64)
    .prefetch(tf.data.AUTOTUNE)
)
assert len(sample_sub) > 0, "Sample submission is empty."



## === cell 14
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

efficient_net = EfficientNetB3(
    weights="imagenet",
    input_shape=(32, 32, 3),
    include_top=False,
    pooling="max",
)

model = Sequential()
model.add(efficient_net)
model.add(Dense(units=120, activation="relu"))
model.add(Dense(units=120, activation="relu"))
model.add(Dense(units=1, activation="sigmoid"))
model.summary()



## === cell 15
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)



## === cell 16
history = model.fit(
    train_dataset,
    epochs=50,
    validation_data=val_dataset,
)



## === cell 17
acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, "bo", label="Training Accuracy")
plt.plot(epochs, val_acc, "b", label="Validation Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.figure()

plt.plot(epochs, loss, "bo", label="Training loss")
plt.plot(epochs, val_loss, "b", label="Validation Loss")
plt.title("Training and Validation Loss")
plt.legend()

plt.show()



## === cell 18
preds = model.predict(test_dataset, verbose=1)
preds = preds.reshape(-1)

preds = np.clip(preds, 0.0, 1.0)

submission = pd.DataFrame({"id": sample_sub["id"].values, "has_cactus": preds})
print(submission.head(10))
print(submission.shape)
assert (
    submission.shape[0] == sample_sub.shape[0]
), "Row count mismatch vs sample_submission."



## === cell 19
out_path = "/kaggle/working/submission.csv"
submission.to_csv(out_path, index=False)
print(f"Wrote: {out_path}")

print(os.listdir("/kaggle/working"))
print(pd.read_csv(out_path).head())
