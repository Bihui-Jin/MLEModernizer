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

os.environ["PYTHONHASHSEED"] = "0"
np.random.seed(0)

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
print("PWD:", os.getcwd())
print("Listing /kaggle/input:", os.listdir("/kaggle/input")[:10])



## === cell 2
import zipfile

extract_dir = "/kaggle/working/aerial_cactus_extracted"
os.makedirs(extract_dir, exist_ok=True)

train_zip = "/kaggle/input/aerial-cactus-identification/train.zip"
test_zip = "/kaggle/input/aerial-cactus-identification/test.zip"

with zipfile.ZipFile(train_zip, "r") as zip_ref:
    zip_ref.extractall(os.path.join(extract_dir, "train"))
with zipfile.ZipFile(test_zip, "r") as zip_ref:
    zip_ref.extractall(os.path.join(extract_dir, "test"))

print("Extracted train to:", os.path.join(extract_dir, "train"))
print("Extracted test to :", os.path.join(extract_dir, "test"))



## === cell 3
for dirname, subdirs, files in os.walk(extract_dir):
    if dirname.count(os.sep) - extract_dir.count(os.sep) <= 3:
        print(dirname, "| subdirs:", subdirs[:5], "| files:", len(files))




## === cell 4
def find_image_dir(root, must_contain_ext=".jpg"):
    for dirpath, _, filenames in os.walk(root):
        if any(f.lower().endswith(must_contain_ext) for f in filenames):
            return dirpath
    return None


train_dir = find_image_dir(os.path.join(extract_dir, "train"))
test_images_dir = find_image_dir(os.path.join(extract_dir, "test"))

if train_dir is None or test_images_dir is None:
    raise FileNotFoundError(
        f"Could not locate image dirs. train_dir={train_dir}, test_images_dir={test_images_dir}"
    )

print("Resolved train_dir:", train_dir)
print("Resolved test_images_dir:", test_images_dir)



## === cell 5
import tensorflow as tf

from tensorflow.keras import callbacks
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

try:
    from tensorflow.keras.preprocessing.image import (
        ImageDataGenerator,
    )  # TF <= 2.13 style
except Exception:
    from tensorflow.keras.legacy.preprocessing.image import (
        ImageDataGenerator,
    )  # TF/Keras 3 style

from tensorflow.keras.applications import EfficientNetB3

print("TensorFlow:", tf.__version__)
print("Eager execution:", tf.executing_eagerly())



## === cell 6
train_df = pd.read_csv("/kaggle/input/aerial-cactus-identification/train.csv")
print(train_df.head())
print(train_df.dtypes)
print("Train rows:", len(train_df))



## === cell 7
missing = 0
for fn in train_df["id"].head(50).tolist():
    if not os.path.exists(os.path.join(train_dir, fn)):
        missing += 1
print("Missing among first 50 train images:", missing)



## === cell 8
import cv2
import matplotlib.pyplot as plt

idxs = [0, 1, 2, 6, 7, 11]
imgs = []
labels = []
for i in idxs:
    img_path = os.path.join(train_dir, train_df.loc[i, "id"])
    img = cv2.imread(img_path)
    if img is None:
        imgs.append(np.zeros((32, 32, 3), dtype=np.uint8))
        labels.append(f"missing: {train_df.loc[i, 'has_cactus']}")
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgs.append(img)
        labels.append("cactus" if train_df.loc[i, "has_cactus"] == 1 else "no cactus")

plt.figure(figsize=[10, 6])
for x in range(len(imgs)):
    plt.subplot(2, 3, x + 1)
    plt.imshow(imgs[x])
    plt.title(labels[x])
    plt.axis("off")
plt.tight_layout()
plt.show()



## === cell 9
train_df["has_cactus"] = train_df["has_cactus"].astype(str)



## === cell 10
train_datagen = ImageDataGenerator(
    rescale=1 / 255,
    validation_split=0.10,
)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=train_dir,
    x_col="id",
    y_col="has_cactus",
    target_size=(32, 32),
    subset="training",
    batch_size=256,
    shuffle=True,
    class_mode="binary",
    seed=0,
)

val_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=train_dir,
    x_col="id",
    y_col="has_cactus",
    target_size=(32, 32),
    subset="validation",
    batch_size=256,
    shuffle=False,
    class_mode="binary",
    seed=0,
)

print("train_generator batches:", len(train_generator))
print("val_generator batches  :", len(val_generator))



## === cell 11
test_parent = os.path.dirname(test_images_dir)
test_subdir_name = os.path.basename(test_images_dir)

test_datagen = ImageDataGenerator(rescale=1 / 255)

test_generator = test_datagen.flow_from_directory(
    directory=test_parent,
    classes=[test_subdir_name],  # ensures only that folder is read
    target_size=(32, 32),
    batch_size=32,
    shuffle=False,
    class_mode=None,
)

print("test_generator batches:", len(test_generator))
print("test images:", test_generator.samples)



## === cell 12
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



## === cell 13
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)



## === cell 14
steps_per_epoch = min(15, len(train_generator))
validation_steps = min(7, len(val_generator))

if steps_per_epoch == 0 or validation_steps == 0:
    raise ValueError(
        f"Generator length is 0. steps_per_epoch={steps_per_epoch}, validation_steps={validation_steps}."
    )

history = model.fit(
    train_generator,
    epochs=50,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_generator,
    validation_steps=validation_steps,
    verbose=2,
)



## === cell 15
import matplotlib.pyplot as plt

acc = history.history.get("accuracy", [])
val_acc = history.history.get("val_accuracy", [])
loss = history.history.get("loss", [])
val_loss = history.history.get("val_loss", [])

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



## === cell 16
print(history.history.keys())



## === cell 17
preds = model.predict(
    test_generator,
    steps=len(test_generator),
    verbose=1,
)
predictions = preds.reshape(-1)

print("Predictions shape:", predictions.shape)
print("First 5 preds:", predictions[:5])



## === cell 18
sample_sub = pd.read_csv(
    "/kaggle/input/aerial-cactus-identification/sample_submission.csv"
)
sample_ids = sample_sub["id"].tolist()

gen_ids = [os.path.basename(p) for p in test_generator.filenames]

if len(gen_ids) != len(predictions):
    raise ValueError(f"Mismatch: gen_ids={len(gen_ids)} preds={len(predictions)}")

pred_map = dict(zip(gen_ids, predictions))
missing_ids = [i for i in sample_ids if i not in pred_map]
if missing_ids:
    raise ValueError(
        f"Some sample_submission ids not found in generator: {missing_ids[:5]} (and more)"
    )

ordered_preds = np.array([pred_map[i] for i in sample_ids], dtype=np.float32)

submission = pd.DataFrame({"id": sample_ids, "has_cactus": ordered_preds})
print(submission.head())
print("Submission rows:", len(submission))

out_path = "/kaggle/working/submission.csv"
submission.to_csv(out_path, index=False)
print("Wrote:", out_path)
print(os.listdir("/kaggle/working")[:20])
print("File size:", os.path.getsize(out_path), "bytes")
