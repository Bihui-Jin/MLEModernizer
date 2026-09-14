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

3.10

# 3. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
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

# 5. Target score

0.9909

# 6. Current score

0.5

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.5) has done: 'I fix the missing `train/` and `test/` folder issue by extracting the zip files into a known working directory and then consistently referencing those absolute paths everywhere (counting, plotting, tf.data loaders, and prediction). I also fix the TensorFlow dataset shape error by explicitly setting the tensor shape after `tf.py_function`, which removes the “unknown TensorShape” crash during `model.fit`. The protobuf `MessageFactory.GetPrototype` crash be avoided by using a simple, stable `tf.keras.utils.set_random_seed` setup and removing the notebook-only `%matplotlib inline` magic that can trigger environment-dependent issues. Finally, I ensure predictions are flattened to 1D and written to a valid `submission.csv` with the exact required columns.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os

for dirname, _, filenames in os.walk("/kaggle/input/aerial-cactus-identification"):
    for filename in filenames[:5]:
        print(os.path.join(dirname, filename))



## === cell 1
import pandas as pd

PATH = "/kaggle/input/aerial-cactus-identification/"
labels = pd.read_csv(PATH + "train.csv")
submissions = pd.read_csv(PATH + "sample_submission.csv")
labels.head()



## === cell 2
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rc("font", size=15)
plt.figure(figsize=(7, 7))

label = ["Has catus", "Hasn't cactus"]
plt.pie(labels["has_cactus"].value_counts(), labels=label, autopct="%.1f%%")
plt.show()



## === cell 3
from zipfile import ZipFile as zf
import os

WORK_DIR = "/kaggle/working/aerial-cactus-identification-extracted"
TRAIN_DIR = os.path.join(WORK_DIR, "train")
TEST_DIR = os.path.join(WORK_DIR, "test")
os.makedirs(WORK_DIR, exist_ok=True)

with zf(PATH + "train.zip") as zipper:
    zipper.extractall(WORK_DIR)

with zf(PATH + "test.zip") as zipper:
    zipper.extractall(WORK_DIR)

print("Extracted to:", WORK_DIR)
print("Train dir exists:", os.path.isdir(TRAIN_DIR))
print("Test dir exists:", os.path.isdir(TEST_DIR))



## === cell 4
import os

n_t = len(os.listdir(TRAIN_DIR))
n_test = len(os.listdir(TEST_DIR))
print(n_t, n_test, sep="\t")



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/3659184864.py in <cell line: 0>()
      2 
      3 # Fix: use extracted absolute paths
----> 4 n_t = len(os.listdir(TRAIN_DIR))
      5 n_test = len(os.listdir(TEST_DIR))
      6 print(n_t, n_test, sep="\t")

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/working/aerial-cactus-identification-extracted/train'

## === cell 5
import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rc("font", size=7)
plt.figure(figsize=(15, 6))

cac_img_name = labels.loc[labels["has_cactus"] == 1, "id"].iloc[-12:].tolist()

for idx, img_name in enumerate(cac_img_name):
    img_path = os.path.join(TRAIN_DIR, img_name)
    img = cv2.imread(img_path)
    if img is None:
        continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax = plt.subplot(2, 6, idx + 1)
    ax.imshow(img)
    ax.axis("off")
plt.show()



## === cell 6
from sklearn.model_selection import train_test_split

train, val = train_test_split(
    labels, test_size=0.1, stratify=labels["has_cactus"], random_state=50
)
print(train.shape)



## === cell 7
import tensorflow as tf
import cv2
import numpy as np
import os

tf.keras.utils.set_random_seed(50)

IMG_SHAPE = (32, 32, 3)


def _load_image_py(img_name_bytes, base_dir):
    img_name = img_name_bytes.numpy().decode("utf-8")
    img_path = os.path.join(base_dir, img_name)
    img = cv2.imread(img_path)
    if img is None:
        return np.zeros(IMG_SHAPE, dtype=np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    return img


def create_img_data_train(img_name):
    return _load_image_py(img_name, TRAIN_DIR)


def create_img_data_test(img_name):
    return _load_image_py(img_name, TEST_DIR)


def tf_load_train(x, y):
    img = tf.py_function(create_img_data_train, [x], tf.float32)
    img.set_shape(IMG_SHAPE)
    return img, tf.cast(y, tf.int16)


def tf_load_val(x, y):
    img = tf.py_function(create_img_data_train, [x], tf.float32)
    img.set_shape(IMG_SHAPE)
    return img, tf.cast(y, tf.int16)


ds_train = tf.data.Dataset.from_tensor_slices(
    (train["id"].values, train["has_cactus"].values)
).map(tf_load_train, num_parallel_calls=tf.data.AUTOTUNE)

ds_val = tf.data.Dataset.from_tensor_slices(
    (val["id"].values, val["has_cactus"].values)
).map(tf_load_val, num_parallel_calls=tf.data.AUTOTUNE)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 8
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras import Input
from tensorflow.keras import metrics


def model_1(shape):
    input = Input(shape=shape)
    conv_activation = "relu"
    x = Conv2D(32, 3, padding="same", activation=conv_activation)(input)
    x = MaxPooling2D(2)(x)
    x = Conv2D(64, 3, padding="same", activation=conv_activation)(input)
    x = MaxPooling2D(2)(x)
    x = Flatten()(x)
    x = Dense(1, activation="sigmoid")(x)
    model = Model(input, x)
    return model


model = model_1((32, 32, 3))
model.compile(
    loss="binary_crossentropy", optimizer="adam", metrics=[metrics.binary_accuracy]
)

print("done")



## === cell 9
from tensorflow.keras.utils import plot_model

try:
    plot_model(model, show_shapes=True)
except Exception as e:
    print("plot_model skipped:", repr(e))



## === cell 10
batch_size = 32
epochs = 10

ds_train_batched = ds_train.batch(batch_size).prefetch(tf.data.AUTOTUNE)
ds_val_batched = ds_val.batch(batch_size).prefetch(tf.data.AUTOTUNE)

hist = model.fit(ds_train_batched, validation_data=ds_val_batched, epochs=epochs)



## === cell 11
import pandas as pd
import tensorflow as tf

test_labels = pd.read_csv(PATH + "sample_submission.csv")
ds_test = tf.data.Dataset.from_tensor_slices(test_labels["id"].values).map(
    lambda x: tf.py_function(create_img_data_test, [x], tf.float32),
    num_parallel_calls=tf.data.AUTOTUNE,
)


def _set_shape(img):
    img.set_shape(IMG_SHAPE)
    return img


ds_test = (
    ds_test.map(_set_shape, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(32)
    .prefetch(tf.data.AUTOTUNE)
)

preds = model.predict(ds_test)
preds.shape



## === cell 12
submissions = pd.read_csv(PATH + "sample_submission.csv")
submissions["has_cactus"] = preds.reshape(-1)
submissions.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submissions.shape)
print(submissions.head())



## === cell 13
import shutil
import os

if os.path.isdir(WORK_DIR):
    shutil.rmtree(WORK_DIR)
print("Cleanup done.")
