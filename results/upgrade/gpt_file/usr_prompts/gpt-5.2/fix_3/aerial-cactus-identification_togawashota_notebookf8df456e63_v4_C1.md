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

0.7480851666666667

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
    for filename in filenames[:5]:
        print(os.path.join(dirname, filename))



## === cell 1
import zipfile

extract_dir = "/kaggle/working/aerial_cactus"
os.makedirs(extract_dir, exist_ok=True)

train_zip_path = "/kaggle/input/aerial-cactus-identification/train.zip"
test_zip_path = "/kaggle/input/aerial-cactus-identification/test.zip"

with zipfile.ZipFile(train_zip_path, "r") as zip_ref:
    zip_ref.extractall(extract_dir)

with zipfile.ZipFile(test_zip_path, "r") as zip_ref:
    zip_ref.extractall(extract_dir)

print("Extracted to:", extract_dir)
print("Top-level extracted folders:", os.listdir(extract_dir))




## === cell 2
def find_dir_containing_jpg(root_dir: str, must_contain: str):
    candidates = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        base = os.path.basename(dirpath).lower()
        if must_contain in base:
            if any(f.lower().endswith(".jpg") for f in filenames):
                candidates.append(dirpath)
    candidates.sort(key=lambda p: (p.count(os.sep), len(p)))
    return candidates[0] if candidates else None


train_dir = os.path.join(extract_dir, "train")
test_dir = os.path.join(extract_dir, "test")

if not (
    os.path.isdir(train_dir)
    and any(f.lower().endswith(".jpg") for f in os.listdir(train_dir))
):
    found = find_dir_containing_jpg(extract_dir, "train")
    if found:
        train_dir = found

if not (
    os.path.isdir(test_dir)
    and any(f.lower().endswith(".jpg") for f in os.listdir(test_dir))
):
    found = find_dir_containing_jpg(extract_dir, "test")
    if found:
        test_dir = found

print("Resolved train_dir:", train_dir)
print("Resolved test_dir :", test_dir)

assert os.path.isdir(train_dir), f"train_dir not found: {train_dir}"
assert os.path.isdir(test_dir), f"test_dir not found: {test_dir}"
assert any(
    f.lower().endswith(".jpg") for f in os.listdir(train_dir)
), f"No jpgs in train_dir: {train_dir}"
assert any(
    f.lower().endswith(".jpg") for f in os.listdir(test_dir)
), f"No jpgs in test_dir: {test_dir}"



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AssertionError                            Traceback (most recent call last)
/tmp/ipykernel_11/3768293789.py in <cell line: 0>()
     34 print("Resolved test_dir :", test_dir)
     35 
---> 36 assert os.path.isdir(train_dir), f"train_dir not found: {train_dir}"
     37 assert os.path.isdir(test_dir), f"test_dir not found: {test_dir}"
     38 assert any(

AssertionError: train_dir not found: /kaggle/working/aerial_cactus/train

## === cell 3
train_df = pd.read_csv("/kaggle/input/aerial-cactus-identification/train.csv")
print(train_df.head())
print("Train df shape:", train_df.shape)




## === cell 4
def count_files(directory):
    return len(
        [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    )


train_count = count_files(train_dir)
test_count = count_files(test_dir)

print(f"Train images: {train_count}")
print(f"Test images: {test_count}")



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/3786933762.py in <cell line: 0>()
      5 
      6 
----> 7 train_count = count_files(train_dir)
      8 test_count = count_files(test_dir)
      9 

/tmp/ipykernel_11/3786933762.py in count_files(directory)
      1 def count_files(directory):
      2     return len(
----> 3         [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
      4     )
      5 

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/working/aerial_cactus/train'

## === cell 5
import random
import cv2
import matplotlib.pyplot as plt



## === cell 6
class_ratio = train_df["has_cactus"].value_counts(normalize=True) * 100
print(class_ratio)



## === cell 7
cactus_imgs = []
for i in range(min(12, len(train_df))):
    p = os.path.join(train_dir, train_df.loc[i, "id"])
    img = cv2.imread(p)
    if img is None:
        continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cactus_imgs.append(img)

if len(cactus_imgs) > 0:
    plt.figure(figsize=(10, 10))
    for i in range(len(cactus_imgs)):
        plt.subplot(4, 3, i + 1)
        plt.imshow(cactus_imgs[i])
        plt.axis("off")
    plt.tight_layout()
    plt.show()



## === cell 8
train_df["has_cactus"] = train_df["has_cactus"].astype(str)



## === cell 9
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

from tensorflow.keras.preprocessing.image import ImageDataGenerator


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
    batch_size=64,
    shuffle=True,
    class_mode="binary",
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
)



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 10
cactus = []
for i in range(min(12, len(train_df))):
    img_path = os.path.join(train_dir, train_df.loc[i, "id"])
    img = cv2.imread(img_path)
    if img is None:
        continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cactus.append(img)

cactus_augmented = [custom_preprocessing(img) for img in cactus]

if len(cactus_augmented) > 0:
    plt.figure(figsize=(10, 10))
    for i in range(len(cactus_augmented)):
        plt.subplot(4, 3, i + 1)
        plt.imshow(cactus_augmented[i])
        plt.title(f"Image {i+1}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()



## === cell 11
sample_sub_path = "/kaggle/input/aerial-cactus-identification/sample_submission.csv"
sample_sub = pd.read_csv(sample_sub_path)

test_datagen = ImageDataGenerator(rescale=1 / 255)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=sample_sub[["id"]].copy(),
    directory=test_dir,
    x_col="id",
    y_col=None,
    target_size=(32, 32),
    batch_size=256,
    shuffle=False,
    class_mode=None,
)



## === cell 12
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



## === cell 13
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)



## === cell 14
steps_per_epoch = len(train_generator)
validation_steps = len(val_generator)

print("steps_per_epoch:", steps_per_epoch)
print("validation_steps:", validation_steps)

history = model.fit(
    train_generator,
    epochs=50,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_generator,
    validation_steps=validation_steps,
)



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1701905869.py in <cell line: 0>()
      5 print("validation_steps:", validation_steps)
      6 
----> 7 history = model.fit(
      8     train_generator,
      9     epochs=50,

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

## === cell 15
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



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/763515286.py in <cell line: 0>()
----> 1 acc = history.history.get("accuracy", [])
      2 val_acc = history.history.get("val_accuracy", [])
      3 loss = history.history.get("loss", [])
      4 val_loss = history.history.get("val_loss", [])
      5 

NameError: name 'history' is not defined

## === cell 16
preds = model.predict(
    test_generator,
    steps=len(test_generator),
    verbose=1,
)



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3520289635.py in <cell line: 0>()
----> 1 preds = model.predict(
      2     test_generator,
      3     steps=len(test_generator),
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

## === cell 17
image_ids = list(sample_sub["id"].values)
predictions = preds.reshape(-1)

assert len(image_ids) == len(predictions), (len(image_ids), len(predictions))
submission = pd.DataFrame({"id": image_ids, "has_cactus": predictions.astype(float)})

print(submission.head(10))
print("Submission shape:", submission.shape)



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2341322233.py in <cell line: 0>()
      1 # Fix: When using flow_from_dataframe, filenames are already the ids (no subdir prefix).
      2 image_ids = list(sample_sub["id"].values)
----> 3 predictions = preds.reshape(-1)
      4 
      5 # Safety: ensure same length and float dtype

NameError: name 'preds' is not defined

## === cell 18
out_path = "/kaggle/working/submission.csv"
submission.to_csv(out_path, index=False)
print("Wrote:", out_path)



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4127143202.py in <cell line: 0>()
      1 out_path = "/kaggle/working/submission.csv"
----> 2 submission.to_csv(out_path, index=False)
      3 print("Wrote:", out_path)
      4 

NameError: name 'submission' is not defined

## === cell 19
print("Working directory contents:", os.listdir("/kaggle/working"))
print("Extract dir contents:", os.listdir(extract_dir))
