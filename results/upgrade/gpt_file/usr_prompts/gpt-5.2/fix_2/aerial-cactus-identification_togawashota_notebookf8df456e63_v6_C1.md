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

# 5. Target score

0.839517

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
import zipfile

extract_root = "/kaggle/working"

train_zip_path = "/kaggle/input/aerial-cactus-identification/train.zip"
test_zip_path = "/kaggle/input/aerial-cactus-identification/test.zip"

train_extract_dir = os.path.join(extract_root, "train_extracted")
test_extract_dir = os.path.join(extract_root, "test_extracted")

os.makedirs(train_extract_dir, exist_ok=True)
os.makedirs(test_extract_dir, exist_ok=True)

with zipfile.ZipFile(train_zip_path, "r") as z:
    z.extractall(train_extract_dir)

with zipfile.ZipFile(test_zip_path, "r") as z:
    z.extractall(test_extract_dir)

train_dir = os.path.join(train_extract_dir, "train")
test_dir = os.path.join(test_extract_dir, "test")

print("Resolved train_dir:", train_dir, "exists:", os.path.isdir(train_dir))
print("Resolved test_dir :", test_dir, "exists:", os.path.isdir(test_dir))



## === cell 2
for dirname, _, _ in os.walk("/kaggle/working"):
    if dirname.count(os.sep) - "/kaggle/working".count(os.sep) <= 3:
        print(dirname)



## === cell 3
pass



## === cell 4
train_df = pd.read_csv("/kaggle/input/aerial-cactus-identification/train.csv")
print(train_df.head())

sample_sub = pd.read_csv(
    "/kaggle/input/aerial-cactus-identification/sample_submission.csv"
)
print(sample_sub.head())




## === cell 5
def count_files(directory):
    return len(
        [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    )


train_count = count_files(train_dir)
test_count = count_files(test_dir)

print(f"Train images: {train_count}")
print(f"Test images:  {test_count}")



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/875076756.py in <cell line: 0>()
      5 
      6 
----> 7 train_count = count_files(train_dir)
      8 test_count = count_files(test_dir)
      9 

/tmp/ipykernel_11/875076756.py in count_files(directory)
      1 def count_files(directory):
      2     return len(
----> 3         [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
      4     )
      5 

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/working/train_extracted/train'

## === cell 6
class_ratio = train_df["has_cactus"].value_counts(normalize=True) * 100
print(class_ratio)



## === cell 7
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

counts = train_df["has_cactus"].value_counts()
labels = ["Has Cactus (1)", "No Cactus (0)"]
colors = ["lightgreen", "lightcoral"]

plt.figure(figsize=(6, 6))
plt.pie(counts, labels=labels, autopct="%1.1f%%", startangle=90, colors=colors)
plt.title("Distribution of Cactus Presence (has_cactus)")
plt.axis("equal")
plt.savefig("/kaggle/working/class_distribution.png")
plt.close()



## === cell 8
import cv2

cactus = []
for idx in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]:
    img_path = os.path.join(train_dir, train_df.loc[idx, "id"])
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Failed to read image at {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cactus.append(img)

plt.figure(figsize=(10, 10))
for i in range(12):
    plt.subplot(4, 3, i + 1)
    plt.imshow(cactus[i])
    plt.axis("off")
plt.tight_layout()
plt.savefig("/kaggle/working/sample_train_images.png")
plt.close()



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/1671327599.py in <cell line: 0>()
      7     img = cv2.imread(img_path)
      8     if img is None:
----> 9         raise FileNotFoundError(f"Failed to read image at {img_path}")
     10     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
     11     cactus.append(img)

FileNotFoundError: Failed to read image at /kaggle/working/train_extracted/train/2de8f189f1dce439766637e75df0ee27.jpg

## === cell 9
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras import callbacks
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import random

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)




## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 10
def custom_preprocessing(image):
    k = random.randint(0, 3)
    image = np.rot90(image, k)

    if random.random() > 0.5:
        image = np.fliplr(image)

    if random.random() > 0.5:
        image = np.flipud(image)

    factor = random.uniform(0.8, 1.2)
    image = np.clip(image * factor, 0, 255).astype(np.float32) / 255.0
    return image


train_df["has_cactus"] = train_df["has_cactus"].astype(str)

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



## === cell 11
cactus_augmented = [custom_preprocessing(img) for img in cactus]

plt.figure(figsize=(10, 10))
for i in range(12):
    plt.subplot(4, 3, i + 1)
    plt.imshow(cactus_augmented[i])
    plt.axis("off")
plt.tight_layout()
plt.savefig("/kaggle/working/sample_augmented_images.png")
plt.close()



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
/tmp/ipykernel_11/431790985.py in <cell line: 0>()
      5 for i in range(12):
      6     plt.subplot(4, 3, i + 1)
----> 7     plt.imshow(cactus_augmented[i])
      8     plt.axis("off")
      9 plt.tight_layout()

IndexError: list index out of range

## === cell 12
test_datagen = ImageDataGenerator(rescale=1 / 255.0)

test_generator = test_datagen.flow_from_directory(
    directory=test_extract_dir,  # contains subfolder "test" with images
    target_size=(32, 32),
    batch_size=1,
    shuffle=False,
    class_mode=None,
)

print(
    "Test batches:", len(test_generator), "Test files:", len(test_generator.filenames)
)



## === cell 13
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



## === cell 14
model.compile(
    optimizer=Adam(learning_rate=0.00005),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)



## === cell 15
history = model.fit(
    train_generator,
    epochs=50,
    steps_per_epoch=30,
    validation_data=val_generator,
    validation_steps=7,
)



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2379813777.py in <cell line: 0>()
      1 # Fit (core loop preserved). With correct paths, generators are non-empty and training runs.
----> 2 history = model.fit(
      3     train_generator,
      4     epochs=50,
      5     steps_per_epoch=30,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/trainers/data_adapters/py_dataset_adapter.py in get_tf_dataset(self)
    293             ]
    294             if len(batches) == 0:
--> 295                 raise ValueError("The PyDataset has length 0")
    296             self._output_signature = data_adapter_utils.get_tensor_spec(batches)
    297 

ValueError: The PyDataset has length 0

## === cell 16
acc = history.history.get("accuracy", [])
val_acc = history.history.get("val_accuracy", [])
loss = history.history.get("loss", [])
val_loss = history.history.get("val_loss", [])

epochs = range(1, len(acc) + 1)

plt.figure()
plt.plot(epochs, acc, "bo", label="Training Accuracy")
plt.plot(epochs, val_acc, "b", label="Validation Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.savefig("/kaggle/working/acc_curve.png")
plt.close()

plt.figure()
plt.plot(epochs, loss, "bo", label="Training loss")
plt.plot(epochs, val_loss, "b", label="Validation Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.savefig("/kaggle/working/loss_curve.png")
plt.close()



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/139304165.py in <cell line: 0>()
      1 # Training curves (robust; save to file)
----> 2 acc = history.history.get("accuracy", [])
      3 val_acc = history.history.get("val_accuracy", [])
      4 loss = history.history.get("loss", [])
      5 val_loss = history.history.get("val_loss", [])

NameError: name 'history' is not defined

## === cell 17
preds = model.predict(
    test_generator,
    steps=len(test_generator.filenames),
    verbose=1,
)

predictions = preds.reshape(-1)

image_ids = [os.path.basename(name) for name in test_generator.filenames]
pred_df = pd.DataFrame({"id": image_ids, "has_cactus": predictions})

submission = sample_sub[["id"]].merge(pred_df, on="id", how="left")

if submission["has_cactus"].isna().any():
    submission["has_cactus"] = submission["has_cactus"].fillna(
        float(np.nanmean(predictions))
    )

print(submission.head(10))
print("Submission rows:", len(submission))



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/4114911408.py in <cell line: 0>()
----> 1 preds = model.predict(
      2     test_generator,
      3     steps=len(test_generator.filenames),
      4     verbose=1,
      5 )

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/trainers/data_adapters/py_dataset_adapter.py in get_tf_dataset(self)
    293             ]
    294             if len(batches) == 0:
--> 295                 raise ValueError("The PyDataset has length 0")
    296             self._output_signature = data_adapter_utils.get_tensor_spec(batches)
    297 

ValueError: The PyDataset has length 0

## === cell 18
out_path = "/kaggle/working/submission.csv"
submission.to_csv(out_path, index=False)
print("Wrote:", out_path)
print(os.listdir("/kaggle/working"))

## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2301576977.py in <cell line: 0>()
      1 out_path = "/kaggle/working/submission.csv"
----> 2 submission.to_csv(out_path, index=False)
      3 print("Wrote:", out_path)
      4 print(os.listdir("/kaggle/working"))

NameError: name 'submission' is not defined
