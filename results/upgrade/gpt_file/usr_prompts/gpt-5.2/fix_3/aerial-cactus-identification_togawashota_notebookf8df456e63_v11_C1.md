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
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
import random

try:
    import tensorflow as tf
except Exception as e:
    tf = None
    print("WARNING: TensorFlow import failed; cannot train CNN in this environment.")
    print("Import error:", repr(e))

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
if tf is not None:
    try:
        tf.random.set_seed(SEED)
    except Exception as e:
        print("WARNING: tf.random.set_seed failed:", repr(e))



## === cell 2
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



## === cell 3
for dirname, _, _ in os.walk("/kaggle/working"):
    print(dirname)




## === cell 4
def find_image_dir(root, target_name):
    candidates = []
    for dirpath, dirnames, filenames in os.walk(root):
        if os.path.basename(dirpath) == target_name:
            jpgs = [f for f in filenames if f.lower().endswith(".jpg")]
            if len(jpgs) > 0:
                candidates.append(dirpath)
    if not candidates:
        raise FileNotFoundError(
            f"Could not find extracted '{target_name}' directory with jpgs under: {root}"
        )
    candidates.sort(key=lambda p: (p.count(os.sep), len(p)))
    return candidates[0]


train_dir = find_image_dir("/kaggle/working", "train")
test_dir = find_image_dir("/kaggle/working", "test")

print("Detected train_dir:", train_dir)
print("Detected test_dir:", test_dir)

train_df = pd.read_csv("/kaggle/input/aerial-cactus-identification/train.csv")
train_df.head(20)




## === cell 5
def count_files(directory):
    return len(
        [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    )


train_count = count_files(train_dir)
test_count = count_files(test_dir)

print(f"Train images: {train_count}")
print(f"Test images: {test_count}")



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
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_df["has_cactus"]),
    y=train_df["has_cactus"],
)
class_weights_dict = dict(enumerate(class_weights))
print(class_weights_dict)



## === cell 9
import cv2

cactus = []
idxs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
for i in idxs:
    fp = os.path.join(train_dir, train_df.loc[i, "id"])
    img = cv2.imread(fp)
    if img is None:
        raise FileNotFoundError(f"cv2.imread failed for: {fp}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cactus.append(img)

plt.figure(figsize=(10, 10))
for x in range(12):
    plt.subplot(4, 3, x + 1)
    plt.imshow(cactus[x])
    plt.axis("off")

plt.tight_layout()
plt.show()



## === cell 10
train_df["has_cactus"] = train_df["has_cactus"].astype("str")



## === cell 11
if tf is not None:
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    def custom_preprocessing(image):
        k = random.randint(0, 3)
        image = np.rot90(image, k)

        if random.random() > 0.5:
            image = np.fliplr(image)

        if random.random() > 0.5:
            image = np.flipud(image)

        factor = random.uniform(0.8, 1.2)
        image = np.clip(image * factor, 0, 255).astype(np.uint32) / 255.0
        return image

    train_datagen = ImageDataGenerator(
        validation_split=0.10,
        preprocessing_function=custom_preprocessing,
    )

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=train_dir,
        x_col="id",
        y_col="has_cactus",
        target_size=(32, 32),
        subset="training",
        batch_size=512,
        shuffle=True,
        class_mode="binary",
        seed=SEED,
    )

    val_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=train_dir,
        x_col="id",
        y_col="has_cactus",
        target_size=(32, 32),
        subset="validation",
        batch_size=256,
        shuffle=True,
        class_mode="binary",
        seed=SEED,
    )



## === cell 12
if tf is not None:
    cactus = []
    for i in range(12):
        fp = os.path.join(train_dir, train_df.loc[i, "id"])
        img = cv2.imread(fp)
        if img is None:
            raise FileNotFoundError(f"cv2.imread failed for: {fp}")
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
if tf is not None:
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    test_datagen = ImageDataGenerator(rescale=1 / 255)

    test_parent = os.path.dirname(test_dir)

    test_generator = test_datagen.flow_from_directory(
        directory=test_parent,
        classes=["test"],
        target_size=(32, 32),
        batch_size=1,
        shuffle=False,
        class_mode=None,
    )



## === cell 14
if tf is not None:
    from tensorflow.keras.applications import EfficientNetB3
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.optimizers import Adam

    efficient_net = EfficientNetB3(
        weights="imagenet", input_shape=(32, 32, 3), include_top=False, pooling="max"
    )

    model = Sequential()
    model.add(efficient_net)
    model.add(Dense(units=120, activation="relu"))
    model.add(Dense(units=120, activation="relu"))
    model.add(Dense(units=1, activation="sigmoid"))
    model.summary()



## === cell 15
if tf is not None:
    model.compile(
        optimizer=Adam(learning_rate=0.00005),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )



## === cell 16
if tf is not None:
    history = model.fit(
        train_generator,
        epochs=70,
        steps_per_epoch=30,
        validation_data=val_generator,
        validation_steps=7,
        class_weight=class_weights_dict,
    )



## === cell 17
if tf is not None:
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
if tf is not None:
    preds = model.predict(
        test_generator, steps=len(test_generator.filenames), verbose=1
    )



## === cell 19
sample_sub = pd.read_csv(
    "/kaggle/input/aerial-cactus-identification/sample_submission.csv"
)

if tf is not None:
    pred_map = {
        os.path.basename(fn): float(p)
        for fn, p in zip(test_generator.filenames, preds.flatten())
    }

    submission = sample_sub.copy()
    submission["has_cactus"] = submission["id"].map(pred_map).astype(float)

    missing = submission["has_cactus"].isna().sum()
    if missing:
        raise ValueError(
            f"{missing} test ids in sample_submission were not found in generated predictions mapping."
        )
else:
    submission = sample_sub.copy()
    submission["has_cactus"] = 0.5

print(submission.head(20))
print(submission.shape)



## === cell 20
submission.to_csv("/kaggle/working/submission.csv", index=False)



## === cell 21
print(os.listdir("/kaggle/working"))
print("Wrote:", "/kaggle/working/submission.csv")
