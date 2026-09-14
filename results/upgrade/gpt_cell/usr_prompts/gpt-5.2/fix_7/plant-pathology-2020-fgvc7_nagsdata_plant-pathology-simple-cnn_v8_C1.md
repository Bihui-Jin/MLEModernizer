# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


# 1. Kaggle task description

## Task
Detect apple diseases from images.

## Metric
Mean column-wise ROC AUC.

## Submission Format
For each image_id in the test set, you must predict a probability for each target variable. The file should contain a header and have the following format:

```
image_id,
test_0,0.25,0.25,0.25,0.25
test_1,0.25,0.25,0.25,0.25
test_2,0.25,0.25,0.25,0.25
etc.
```

## Dataset
Given a photo of an apple leaf, can you accurately assess its health? This competition will challenge you to distinguish between leaves which are healthy, those which are infected with apple rust, those that have apple scab, and those with more than one disease.

**train.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

**images**

A folder containing the train and test images, in jpg format.

**test.csv**

- `image_id`: the foreign key

**sample_submission.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

# 2. Python version

3.8

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
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        input/
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        working/
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
```

-> data/plant-pathology-2020-fgvc7/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/plant-pathology-2020-fgvc7/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/plant-pathology-2020-fgvc7/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> (stopped after 10 files for performance)

# 5. Target score

0.87157

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.46995) has done: 'Your pipeline didn’t yield a Kaggle score mainly because it’s likely failing before producing a submission due to the forced protobuf downgrade (which conflicts with your installed TensorFlow 2.18), and because `EarlyStopping` can stop training before completing your intended epochs. I remove the protobuf pinning so TensorFlow can import reliably, and I keep your exact model/loss/loop while making training deterministic and stable by setting seeds and ensuring the generator shuffles consistently. I also keep your current output semantics (4-class softmax with categorical_crossentropy) and ensure the submission columns/order match `sample_submission.csv` exactly so the CSV is always valid. These are minimal changes aimed at getting a valid submission and a reasonable score movement toward the 0.87157 target without changing the core approach.'
- What this solution (achieved 0.56253) has done: 'Diagnosis: The crash happens immediately when importing TensorFlow in cell 1, before any of your notebook logic runs. In this environment you have `tensorflow==2.18.0` paired with `protobuf==6.33.0`, and that combination is incompatible: TensorFlow 2.18 expects the protobuf Python runtime API that still provides `MessageFactory.GetPrototype`, but protobuf 6 removed/changed it, triggering `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'` during TensorFlow import. Since we can’t change installed packages, the minimal in-notebook fix is to force TensorFlow to use the pure-Python protobuf implementation, which avoids the incompatible compiled runtime path. We must set `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` before importing TensorFlow.

Patch summary: In cell 1 only, set the protobuf implementation environment variable (and optionally its version) before `import tensorflow as tf`, then keep the rest of the cell identical to preserve seeds and downstream variables.

Updated cells: (cell 1 only)

Compatibility notes for cell k+1: `tf`, `img_to_array`, `ImageDataGenerator`, `plt`, and `SEED` are still defined exactly as before, so cell 2 and later cells remain compatible.

Assumptions: The runtime permits setting environment variables at notebook execution time and using the pure-Python protobuf implementation; this may be slightly slower but is deterministic and fixes the import crash without changing modeling logic.'
- What this solution (achieved 0.61456) has done: 'Diagnosis: The crash happens during `import tensorflow as tf` in cell 1, before any model code runs. You force `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`, which makes TensorFlow use the pure-Python protobuf backend; with the installed `protobuf==6.33.0` this triggers an incompatibility (`MessageFactory` missing `GetPrototype`). The fix is to stop forcing the pure-Python protobuf implementation so TensorFlow can use the faster C++ backend that remains compatible in this environment. This change is localized to cell 1 and does not alter any training/model logic.

Patch summary: Remove the lines that set `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION` (and its version) to `"python"`/`"2"` so TensorFlow imports cleanly with the installed protobuf version.

Updated cells: Only cell 1 is changed.

Compatibility notes for cell k+1: All symbols created in cell 1 (`tf`, `img_to_array`, `ImageDataGenerator`, `plt`, `SEED`) remain defined exactly as before; subsequent cells can run unchanged.

Assumptions: The environment supports the default (C++-backed) protobuf implementation, and no other notebook cell relies on protobuf being forced to the pure-Python backend.'
- What this solution (achieved 0.51618) has done: 'Diagnosis: Cell 1 crashes during `import tensorflow as tf` because the environment has an incompatible `protobuf==6.33.0` with TensorFlow 2.18’s protobuf expectations, leading to `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'`. This is a known protobuf API change issue that can be avoided by forcing TensorFlow to use the pure-Python protobuf implementation at runtime. The smallest safe fix is to set `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` (and version to `3`) before importing TensorFlow. This keeps the rest of the logic identical and unblocks execution for subsequent cells.

Patch summary: Modify only cell 1 by setting the protobuf environment variables before `import tensorflow as tf`, without changing any downstream logic or variables.

Updated cells: (cell 1 only)

Compatibility notes for cell k+1: No variables or interfaces are changed; `tf`, `ImageDataGenerator`, `plt`, `SEED`, and the random seeds are still defined exactly as before, so cell 2 run unchanged.

Assumptions: The crash happens at TensorFlow import time due to protobuf runtime incompatibility, and setting the protobuf implementation environment variables is sufficient in this environment.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os



## === cell 1
import os

try:
    import google.protobuf  # noqa: F401
    from google.protobuf import __version__ as _pb_version

    _pb_major = int(_pb_version.split(".", 1)[0])
    if _pb_major >= 6:
        import sys
        import subprocess

        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "protobuf<6"]
        )
        os.execv(sys.executable, [sys.executable] + sys.argv)
except Exception:
    pass

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "3")

import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, ImageDataGenerator
import matplotlib.pyplot as plt

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


## === cell 2
train_csv = pd.read_csv("/kaggle/input/plant-pathology-2020-fgvc7/train.csv")
train_csv.head()



## === cell 3
test_csv = pd.read_csv("/kaggle/input/plant-pathology-2020-fgvc7/test.csv")
test_csv.head()



## === cell 4
image = plt.imread("/kaggle/input/plant-pathology-2020-fgvc7/images/Train_0.jpg")
plt.imshow(image)
print(image.shape)




## === cell 5
def image_resize(img, size=(None, None), ratio=3):
    if size[0] is None:
        resize_ratio = ratio
        resize_height = int(img.shape[0] / resize_ratio)
        resize_width = int(img.shape[1] / resize_ratio)
        print(f"height: {resize_height}, width: {resize_width}")
    else:
        resize_height = size[0]
        resize_width = size[1]

    img_resize = tf.image.resize(img, [resize_height, resize_width]).numpy()
    img_resize = img_resize.astype(np.uint8)
    return img_resize




## === cell 6
plt.figure(1, figsize=(10, 10))
plt.subplot(221)
plt.imshow(image_resize(image, ratio=3))

plt.subplot(222)
plt.imshow(image_resize(image, ratio=4))
plt.show()

plt.subplot(223)
plt.imshow(image_resize(image, ratio=5))

plt.subplot(224)
plt.imshow(image_resize(image, ratio=6))
plt.show()



## === cell 7
img_height = 227
img_width = 341
plt.imshow(image_resize(image, size=(img_height, img_width)))



## === cell 8
train_resized = []
for img_id in train_csv["image_id"].to_list():
    image = plt.imread(f"/kaggle/input/plant-pathology-2020-fgvc7/images/{img_id}.jpg")
    train_resized.append(image_resize(image, (img_height, img_width)))
print(len(train_resized))

test_resized = []
for img_id in test_csv["image_id"].to_list():
    image = plt.imread(f"/kaggle/input/plant-pathology-2020-fgvc7/images/{img_id}.jpg")
    test_resized.append(image_resize(image, (img_height, img_width)))
print(len(test_resized))



## === cell 9
x_train = np.ndarray(
    shape=(len(train_resized), img_height, img_width, 3), dtype=np.float32
)
x_test = np.ndarray(
    shape=(len(test_resized), img_height, img_width, 3), dtype=np.float32
)

for i in range(len(train_resized)):
    x_train[i] = img_to_array(train_resized[i])

for i in range(len(test_resized)):
    x_test[i] = img_to_array(test_resized[i])

x_train = x_train / 255.0
x_test = x_test / 255.0

print(x_train.shape)
print(x_test.shape)



## === cell 10
y_train = train_csv.iloc[:, 1:]
y_train.head()



## === cell 11
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications import InceptionResNetV2

resnet = InceptionResNetV2(weights="imagenet", include_top=False, pooling="avg")

model = Sequential()
model.add(resnet)
model.add(Dense(128, activation="relu"))
model.add(Dense(64, activation="relu"))
model.add(Dense(y_train.shape[1], activation="softmax"))

model.layers[0].trainable = False
model.summary()



## === cell 12
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics="accuracy")



## === cell 13
from sklearn.model_selection import train_test_split

train_x, val_x, train_y, val_y = train_test_split(
    x_train, y_train, test_size=0.2, random_state=SEED, shuffle=True
)

print(train_x.shape)
print(train_y.shape)
print(val_x.shape)
print(val_y.shape)



## === cell 14
datagen = ImageDataGenerator(
    rotation_range=25,
    shear_range=0.20,
    zoom_range=0.20,
    width_shift_range=0.20,
    height_shift_range=0.20,
    horizontal_flip=True,
    vertical_flip=True,
)

batch_size = 24
datagen_without_aug = ImageDataGenerator()

train_datagen = datagen_without_aug.flow(
    train_x, train_y, batch_size=batch_size, shuffle=True, seed=SEED
)
val_datagen = datagen_without_aug.flow(
    val_x, val_y, batch_size=batch_size, shuffle=False
)



## === cell 15
history = model.fit(
    train_datagen,
    epochs=6,
    steps_per_epoch=train_x.shape[0] // batch_size,
    validation_data=val_datagen,
    validation_steps=val_x.shape[0] // batch_size,
    verbose=1,
)



## === cell 16
history_df = pd.DataFrame(history.history)
history_df.head()



## === cell 17
plt.plot(history_df.index, history_df["accuracy"])
plt.plot(history_df.index, history_df["val_accuracy"])
plt.show()



## === cell 18
plt.plot(history_df.index, history_df["loss"])
plt.plot(history_df.index, history_df["val_loss"])
plt.show()



## === cell 19
y_preds = model.predict(x_test, verbose=0)



## === cell 20
sample_sub = pd.read_csv(
    "/kaggle/input/plant-pathology-2020-fgvc7/sample_submission.csv"
)
res = pd.DataFrame({"image_id": test_csv["image_id"]})

for i, col in enumerate(sample_sub.columns[1:]):
    res[col] = y_preds[:, i]

res = res.set_index("image_id").loc[test_csv["image_id"]].reset_index()

res.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", res.shape)
print(res.head())
