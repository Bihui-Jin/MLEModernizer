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

3.7

# 3. Installed packages

geopandas==0.14.4
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
import concurrent.futures  # added for parallel image loading

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dropout, Flatten, Dense
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

keras = tf.keras

tf.config.optimizer.set_jit(True)
from tensorflow.keras import mixed_precision

mixed_precision.set_global_policy("mixed_float16")

tf.config.threading.set_intra_op_parallelism_threads(os.cpu_count())
tf.config.threading.set_inter_op_parallelism_threads(os.cpu_count())

tf.random.set_seed(42)
np.random.seed(42)

print("Input directory contents:", os.listdir("../input"))




## === cell 1
train_img_dir = "../input/aerial-cactus-identification/train/"
train_img_pathes = [
    os.path.join(train_img_dir, f)
    for f in sorted(os.listdir(train_img_dir))
    if os.path.isfile(os.path.join(train_img_dir, f))
]

df = pd.read_csv("../input/aerial-cactus-identification/train.csv")
print("Number of training images:", len(train_img_pathes))
print(df.head())




## === cell 2
lst = sorted(
    [
        f
        for f in os.listdir(train_img_dir)
        if os.path.isfile(os.path.join(train_img_dir, f))
    ]
)
err = False
for i, idx in enumerate(df["id"]):
    if idx != lst[i]:
        print("mismatch after %d iterations" % i)
        err = True
        break
if not err:
    print("1:1 correspondence between train_img_pathes and df labels")




## === cell 3
img_size = 32


def read_and_prep_images(img_paths, img_height=img_size, img_width=img_size):
    """
    Parallel image loader using a ThreadPoolExecutor.
    Reads each JPEG, decodes, resizes, and scales to [0,1].
    Returns a NumPy array of shape (N, img_height, img_width, 3).
    """

    def _load(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [img_height, img_width])
        img = tf.cast(img, tf.float32) / 255.0
        return img.numpy()

    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as ex:
        imgs = list(ex.map(_load, img_paths))
    return np.stack(imgs, axis=0)


train_imgs = read_and_prep_images(train_img_pathes)




## === cell 4
out_y = df["has_cactus"].values.astype(np.float32)

print("Shape of a single training image:", train_imgs[0].shape)
print("Shape of training data before augmentation:", train_imgs.shape)
print("Shape of label vector before augmentation:", out_y.shape)

h_flip = np.flip(train_imgs, axis=2)  # flip width dimension
v_flip = np.flip(train_imgs, axis=1)  # flip height dimension

rot90 = np.rot90(train_imgs, k=1, axes=(1, 2))
rot180 = np.rot90(train_imgs, k=2, axes=(1, 2))

train_imgs = np.concatenate([train_imgs, h_flip, v_flip, rot90, rot180], axis=0)
out_y = np.concatenate([out_y, out_y, out_y, out_y, out_y], axis=0)

print("Shape of training data after augmentation:", train_imgs.shape)
print("Shape of label vector after augmentation:", out_y.shape)

perm = np.random.permutation(train_imgs.shape[0])
train_imgs = train_imgs[perm]
out_y = out_y[perm]




## === cell 5
model = Sequential()
model.add(
    Conv2D(filters=256, kernel_size=(3, 3), input_shape=(32, 32, 3), activation="relu")
)
model.add(Dropout(0.2))
model.add(Conv2D(filters=256, kernel_size=(3, 3), activation="relu"))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(512, activation="relu"))  # larger dense layer
model.add(Dense(1, activation="sigmoid"))

model.compile(
    loss=keras.losses.BinaryCrossentropy(),
    optimizer="adam",
    metrics=["accuracy", keras.metrics.AUC(name="auc")],
)




## === cell 6
classes = np.unique(out_y)
class_weights_array = compute_class_weight(
    class_weight="balanced", classes=classes, y=out_y.astype(int)
)
class_weight_dict = {int(cls): float(w) for cls, w in zip(classes, class_weights_array)}
print("Class weights:", class_weight_dict)

split_idx = int(0.8 * train_imgs.shape[0])
train_X, val_X = train_imgs[:split_idx], train_imgs[split_idx:]
train_y, val_y = out_y[:split_idx], out_y[split_idx:]

batch_size = 1024
train_ds = (
    tf.data.Dataset.from_tensor_slices((train_X, train_y))
    .shuffle(1024, reshuffle_each_iteration=True)
    .batch(batch_size)
    .cache()
    .prefetch(tf.data.AUTOTUNE)
)

val_ds = (
    tf.data.Dataset.from_tensor_slices((val_X, val_y))
    .batch(batch_size)
    .cache()
    .prefetch(tf.data.AUTOTUNE)
)

model.fit(
    train_ds,
    epochs=120,
    validation_data=val_ds,
    class_weight=class_weight_dict,
    verbose=2,
)




## === cell 7
test_img_dir = "../input/aerial-cactus-identification/test/"
test_img_pathes = [
    os.path.join(test_img_dir, f)
    for f in sorted(os.listdir(test_img_dir))
    if os.path.isfile(os.path.join(test_img_dir, f))
]
test_imgs = read_and_prep_images(test_img_pathes)
print("Number of test images:", len(test_imgs))




## === cell 8
prediction = model.predict(test_imgs, batch_size=1024).flatten()

ids = [os.path.basename(p) for p in test_img_pathes]
answer = pd.DataFrame({"id": ids, "has_cactus": prediction.astype(float)})

print("Submission preview:")
print(answer.head())




## === cell 9
answer.to_csv("submission.csv", index=False)
print("Submission file written to submission.csv")
