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
import os, zipfile, random
import numpy as np, pandas as pd

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.optimizers import Adam

try:
    from tensorflow.keras.applications.efficientnet import EfficientNetB3
except Exception:
    EfficientNetB3 = None
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split

extract_dir = "/kaggle/working"
with zipfile.ZipFile("/kaggle/input/aerial-cactus-identification/train.zip", "r") as z:
    z.extractall(os.path.join(extract_dir, "train"))
with zipfile.ZipFile("/kaggle/input/aerial-cactus-identification/test.zip", "r") as z:
    z.extractall(os.path.join(extract_dir, "test"))

train_dir = os.path.join(extract_dir, "train")
test_dir = os.path.join(extract_dir, "test")



## === cell 1
train_df = pd.read_csv("/kaggle/input/aerial-cactus-identification/train.csv")
sample_sub = pd.read_csv(
    "/kaggle/input/aerial-cactus-identification/sample_submission.csv"
)
train_df["has_cactus"] = train_df["has_cactus"].astype(int)



## === cell 2
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_df["has_cactus"]),
    y=train_df["has_cactus"],
)
class_weights_dict = dict(zip(np.unique(train_df["has_cactus"]), class_weights))
print("Class weights:", class_weights_dict)




## === cell 3
def tf_preprocess(image):
    """TensorFlow‑only augmentation matching the original NumPy logic."""
    image = tf.cast(image, tf.uint8)  # ensure uint8 input
    k = tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32)
    image = tf.image.rot90(image, k)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    factor = tf.random.uniform([], 0.8, 1.2, dtype=tf.float32)
    image = tf.cast(image, tf.float32) * factor
    image = tf.clip_by_value(image, 0.0, 255.0)
    image = image / 255.0  # scale to [0,1]
    return tf.cast(image, tf.float32)




## === cell 4
def load_images_from_df(df, base_dir):
    imgs = []
    for fname in df["id"]:
        path = os.path.join(base_dir, fname)
        img = load_img(path, target_size=(32, 32))
        arr = img_to_array(img)  # (32,32,3), dtype float32 0‑255
        imgs.append(arr)
    return np.stack(imgs, axis=0)  # (N,32,32,3)


print("Loading training images into memory …")
train_images_raw = load_images_from_df(train_df, train_dir)  # float32 0‑255
train_images_raw = train_images_raw.astype(
    np.uint8
)  # keep as uint8 for TF augmentation

train_idx, val_idx = train_test_split(
    np.arange(len(train_df)),
    test_size=0.10,
    stratify=train_df["has_cactus"],
    random_state=42,
)

x_train = train_images_raw[train_idx]
y_train = train_df["has_cactus"].values[train_idx].astype(np.float32)

x_val = train_images_raw[val_idx] / 255.0  # only rescale, no aug
y_val = train_df["has_cactus"].values[val_idx].astype(np.float32)

train_ds = (
    tf.data.Dataset.from_tensor_slices((x_train, y_train))
    .cache()
    .shuffle(buffer_size=len(x_train), seed=42)
    .map(
        lambda img, label: (tf_preprocess(img), label),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    .batch(512)
    .prefetch(tf.data.AUTOTUNE)
)

val_ds = (
    tf.data.Dataset.from_tensor_slices((x_val, y_val))
    .batch(256)
    .prefetch(tf.data.AUTOTUNE)
)



## === cell 5
print("Building model …")
if EfficientNetB3 is not None:
    efficient_net = EfficientNetB3(
        weights="imagenet", input_shape=(32, 32, 3), include_top=False, pooling="max"
    )
    model = Sequential(
        [
            efficient_net,
            Dense(120, activation="relu"),
            Dense(120, activation="relu"),
            Dense(1, activation="sigmoid"),
        ]
    )
else:
    model = Sequential(
        [
            Conv2D(
                32, (3, 3), activation="relu", padding="same", input_shape=(32, 32, 3)
            ),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation="relu", padding="same"),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation="relu", padding="same"),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation="relu"),
            Dropout(0.5),
            Dense(1, activation="sigmoid"),
        ]
    )
model.summary()



## === cell 6
model.compile(
    optimizer=Adam(learning_rate=5e-5),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)



## === cell 7
history = model.fit(
    train_ds,
    epochs=30,  # increased epochs for better convergence
    validation_data=val_ds,
    class_weight=class_weights_dict,
    verbose=0,
)



## === cell 8
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="train")
plt.plot(history.history["val_accuracy"], label="val")
plt.title("Accuracy")
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="val")
plt.title("Loss")
plt.legend()
plt.show()



## === cell 9
preds = model.predict(test_ds, verbose=0).flatten()



## === cell 10
submission = pd.DataFrame({"id": sample_sub["id"], "has_cactus": preds})
submission_path = "/kaggle/working/submission.csv"
submission.to_csv(submission_path, index=False)
print("Saved submission to:", submission_path)
print(submission.head())



## === cell 11
print("Files in /kaggle/working:", os.listdir("/kaggle/working"))
