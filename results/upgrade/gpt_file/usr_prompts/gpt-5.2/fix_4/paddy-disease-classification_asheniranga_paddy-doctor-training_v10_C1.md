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
Develop a model to classify paddy leaf images into one of the nine disease categories or normal leaf.

## Metric
Categorization accuracy.

## Submission Format
```
image_id,label
200001.jpg,normal
200002.jpg,blast
etc.
```

## Dataset
**train.csv** - The training set

- `image_id` - Unique image identifier corresponds to image file names (.jpg) found in the train_images directory.
- `label` - Type of paddy disease, also the target class. There are ten categories, including the normal leaf.
- `variety` - The name of the paddy variety.
- `age` - Age of the paddy in days.

**sample_submission.csv** - Sample submission file.

**train_images** - Training images stored under different sub-directories corresponding to ten target classes. Filename corresponds to the `image_id` column of `train.csv`.

**test_images** - Test set images.

# 2. Python version

3.10

# 3. Installed packages

albumentations==2.0.8
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
seaborn==0.12.2
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
            description.md (72 lines)
            sample_submission.csv (2603 lines)
            sample_submission.csv.zip (7.4 kB)
            test.zip (160 Bytes)
            test_images.zip (205.2 MB)
            train.csv (7806 lines)
            train.csv.zip (40.1 kB)
            train.zip (162 Bytes)
            train_images.zip (614.5 MB)
            paddy-disease-classification/
                description.md (72 lines)
                sample_submission.csv (2603 lines)
                ... and 7 other files
                paddy-disease-classification/
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
                train_images/
                    bacterial_leaf_blight/
                        109831.jpg (91.1 kB)
                        109785.jpg (81.8 kB)
                        ... and 356 other files
                    bacterial_leaf_streak/
                        100394.jpg (99.7 kB)
                        103308.jpg (104.3 kB)
                        ... and 295 other files
                    ... and 9 other folders
            test_images/
                102916.jpg (92.1 kB)
                100596.jpg (83.9 kB)
                ... and 2600 other files
                test_images/
            train_images/
                bacterial_leaf_blight/
                    109831.jpg (91.1 kB)
                    109785.jpg (81.8 kB)
                    ... and 356 other files
                bacterial_leaf_streak/
                    100394.jpg (99.7 kB)
                    103308.jpg (104.3 kB)
                    ... and 295 other files
                ... and 9 other folders
        input/
            description.md (72 lines)
            sample_submission.csv (2603 lines)
            sample_submission.csv.zip (7.4 kB)
            test.zip (160 Bytes)
            test_images.zip (205.2 MB)
            train.csv (7806 lines)
            train.csv.zip (40.1 kB)
            train.zip (162 Bytes)
            train_images.zip (614.5 MB)
            paddy-disease-classification/
                description.md (72 lines)
                sample_submission.csv (2603 lines)
                ... and 7 other files
                paddy-disease-classification/
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
                train_images/
                    bacterial_leaf_blight/
                        109831.jpg (91.1 kB)
                        109785.jpg (81.8 kB)
                        ... and 356 other files
                    bacterial_leaf_streak/
                        100394.jpg (99.7 kB)
                        103308.jpg (104.3 kB)
                        ... and 295 other files
                    ... and 9 other folders
            test_images/
                102916.jpg (92.1 kB)
                100596.jpg (83.9 kB)
                ... and 2600 other files
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
            train_images/
                bacterial_leaf_blight/
                    109831.jpg (91.1 kB)
                    109785.jpg (81.8 kB)
                    ... and 356 other files
                bacterial_leaf_streak/
                    100394.jpg (99.7 kB)
                    103308.jpg (104.3 kB)
                    ... and 295 other files
                ... and 9 other folders
        working/
            paddy-disease-classification/
                description.md (72 lines)
                sample_submission.csv (2603 lines)
                ... and 7 other files
                paddy-disease-classification/
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
                train_images/
                    bacterial_leaf_blight/
                        109831.jpg (91.1 kB)
                        109785.jpg (81.8 kB)
                        ... and 356 other files
                    bacterial_leaf_streak/
                        100394.jpg (99.7 kB)
                        103308.jpg (104.3 kB)
                        ... and 295 other files
                    ... and 9 other folders
```

-> data/paddy-disease-classification/sample_submission.csv has 2602 rows and 2 columns.
The columns are: image_id, label

-> data/paddy-disease-classification/train.csv has 7805 rows and 4 columns.
The columns are: image_id, label, variety, age

-> data/sample_submission.csv has 2602 rows and 2 columns.
The columns are: image_id, label

-> data/train.csv has 7805 rows and 4 columns.
The columns are: image_id, label, variety, age

-> input/paddy-disease-classification/sample_submission.csv has 2602 rows and 2 columns.
The columns are: image_id, label

-> input/paddy-disease-classification/train.csv has 7805 rows and 4 columns.
The columns are: image_id, label, variety, age

-> (stopped after 10 files for performance)

# 5. Target score

0.96313

# 6. Current score

0.04766

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.0465) has done: 'I fix the import/runtime issues that prevent any cells from running by removing the incompatible `tensorflow_addons` import (it triggers the protobuf `GetPrototype` crash in TF 2.18) and by switching to the supported Keras preprocessing imports for TF 2.18 so `ImageDataGenerator`, `load_img`, and `img_to_array` are defined. I also fix the incorrect dataset paths (use `/kaggle/input/...`), add the missing `plt` import usage (already imported but never executed due to earlier crash), and make the generator statistics `fit()` receive a proper NumPy array. Finally, I ensure test-time prediction uses the true test set size (not a hardcoded 3469) and that the written submission has the required `image_id,label` columns with correct `image_id` values and a `.csv` filename.'
- What this solution (achieved 0.06188) has done: 'I fix the environment-breaking imports that cause the protobuf `GetPrototype` crash by removing `tensorflow_hub` usage and relying on `tf.keras.applications.EfficientNetB4` (same model family and training semantics). I also fix the Keras incompatibility that makes `tf_keras.preprocessing.image.DirectoryIterator` unrecognized by `keras` in TF 2.18 by switching to `tf.keras.preprocessing.image.ImageDataGenerator` everywhere. Finally, I make the generator `.fit()` robust by sampling existing images from disk (instead of hardcoded missing filenames), update the weights filename to the required `.weights.h5` suffix, and ensure a correctly formatted `submission.csv` is always written.'
- What this solution (achieved 0.04766) has done: 'I fix the protobuf `GetPrototype` crash by removing the unused `albumentations` import path that triggers protobuf internals in this TF 2.18 environment, while keeping the modeling and training logic unchanged. Then I fix the `DirectoryIterator` API usage by replacing `.next()` with Python’s `next(iter(...))` so the visualization/debug cells run. Finally, I fix the label shape mismatch by making the generators produce one-hot labels explicitly (`class_mode="categorical"`), aligning with your `categorical_crossentropy` and `classes=10`, which should also substantially improve accuracy toward your target. All paths and submission writing stay the same, and the script always emit a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
import cv2


from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import StratifiedKFold, train_test_split

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Add, Activation
from tensorflow.keras.layers import (
    MaxPooling2D,
    AveragePooling2D,
    GlobalAveragePooling2D,
)
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten

from tensorflow.keras.activations import relu, softmax

from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator,
    load_img,
    img_to_array,
    array_to_img,
)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_meta_data = "/kaggle/input/paddy-disease-classification/train.csv"
train_data_dir = "/kaggle/input/paddy-disease-classification/train_images"
epochs = 25
lr = 1e-4
valid_split = 0.2
input_size = 128
batch_size = 16
classes = 10
initializer = tf.keras.initializers.HeUniform()
optimizer = tf.keras.optimizers.Nadam(learning_rate=lr)
loss = tf.keras.losses.categorical_crossentropy



## === cell 2
early_stop = tf.keras.callbacks.EarlyStopping(
    patience=20, monitor="val_loss", restore_best_weights=True, verbose=1
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    patience=5, monitor="val_loss", factor=0.5, verbose=1
)



## === cell 3
src = "/kaggle/input/paddy-disease-classification/train_images/dead_heart/100008.jpg"
img = img_to_array(load_img(src), dtype="uint8")
img.shape




## === cell 4
def random_cutout(image, patch_size=16, patches=16):
    if random.choice([True, False]):
        anchors_x = []
        anchors_y = []

        for _ in range(patches):
            rv = np.random.randint(0, image.shape[0])
            if rv not in anchors_x:
                anchors_x.append(rv)

        for _ in range(patches):
            rv = np.random.randint(0, image.shape[1])
            if rv not in anchors_y:
                anchors_y.append(rv)

        for x, y in zip(anchors_x, anchors_y):
            image[x : x + patch_size, y : y + patch_size, :] = 0

        return image
    else:
        return image


def random_gaus_blur(image):
    if random.choice([True, False]):
        return cv2.GaussianBlur(image, (7, 7), 0)
    else:
        return image


def random_displacment(image):
    if random.choice([True, False]):
        ax = random.choice([0, 1])

        if ax == 0:
            slices = np.split(image, 8, axis=ax)
            np.random.shuffle(slices)
            return np.row_stack(slices)
        else:
            slices = np.split(image, 8, axis=ax)
            np.random.shuffle(slices)
            return np.column_stack(slices)
    else:
        return image


def center_crop_and_random_augmentations_fn(image):
    image = tf.convert_to_tensor(image)
    image = tf.image.random_crop(image, (input_size, input_size, 3)).numpy()
    image = random_cutout(image, 8, 16)
    image = random_displacment(image)
    image = random_gaus_blur(image)
    image = tf.image.random_brightness(image, 0.2)
    image = tf.image.random_contrast(image, 0.5, 2.0)
    image = tf.image.random_saturation(image, 0.75, 1.25)
    image = tf.image.random_hue(image, 0.1).numpy()
    return image


def test_time_augmentation_fn(image):
    image = tf.convert_to_tensor(image)
    image = tf.image.random_crop(image, (input_size, input_size, 3)).numpy()
    image = tf.image.random_brightness(image, 0.2)
    image = tf.image.random_contrast(image, 0.5, 2.0)
    return image




## === cell 5
generator = ImageDataGenerator(
    rescale=1 / 255,
    rotation_range=5,
    width_shift_range=0.1,
    height_shift_range=0.1,
    featurewise_center=True,
    featurewise_std_normalization=True,
    horizontal_flip=True,
    vertical_flip=True,
    validation_split=valid_split,
    preprocessing_function=center_crop_and_random_augmentations_fn,
)

train_datagen = generator.flow_from_directory(
    train_data_dir,
    target_size=(input_size, input_size),
    batch_size=batch_size,
    subset="training",
    seed=SEED,
    class_mode="categorical",
)

valid_datagen = generator.flow_from_directory(
    train_data_dir,
    target_size=(input_size, input_size),
    batch_size=batch_size,
    subset="validation",
    seed=SEED,
    class_mode="categorical",
)




## === cell 6
def _list_all_jpgs(root_dir):
    out = []
    for root, _, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith(".jpg"):
                out.append(os.path.join(root, f))
    return out


all_train_jpgs = _list_all_jpgs(train_data_dir)
if len(all_train_jpgs) == 0:
    raise FileNotFoundError(f"No .jpg images found under {train_data_dir}")

rng = np.random.RandomState(SEED)
n_fit = min(128, len(all_train_jpgs))
fit_files = rng.choice(all_train_jpgs, size=n_fit, replace=False).tolist()

to_gen_fit = []
for file in fit_files:
    to_gen_fit.append(img_to_array(load_img(file), dtype="uint8"))

to_gen_fit = np.stack(to_gen_fit, axis=0)
to_gen_fit.shape



## === cell 7
generator.fit(to_gen_fit)



## === cell 8
train_batch = next(iter(train_datagen))
valid_batch = next(iter(valid_datagen))
len(train_batch[0]), len(valid_batch[0])



## === cell 9
fig, axes = plt.subplots(nrows=2, ncols=8, figsize=[32, 10], dpi=200)
axes = axes.ravel()

train_batch = next(iter(train_datagen))
for i, arr in enumerate(train_batch[0][: len(axes)]):
    img = array_to_img(arr)
    axes[i].imshow(img)
    axes[i].axis("off")

plt.show()



## === cell 10
fig, axes = plt.subplots(nrows=2, ncols=8, figsize=[32, 10], dpi=200)
axes = axes.ravel()

valid_batch = next(iter(valid_datagen))
for i, arr in enumerate(valid_batch[0][: len(axes)]):
    img = array_to_img(arr)
    axes[i].imshow(img)
    axes[i].axis("off")

plt.show()



## === cell 11
base = tf.keras.applications.EfficientNetB4(
    include_top=False,
    weights="imagenet",
    input_shape=(input_size, input_size, 3),
    pooling="avg",
)
base.trainable = True
model = tf.keras.Sequential(
    [base, tf.keras.layers.Dense(classes, activation="softmax")]
)

model.build([None, input_size, input_size, 3])
model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
_using_hub = False
_using_hub



## === cell 12
model.summary()



## === cell 13
history = model.fit(
    train_datagen,
    validation_data=valid_datagen,
    batch_size=batch_size,
    epochs=epochs,
    callbacks=[early_stop, reduce_lr],
    verbose=1,
)



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3837220834.py in <cell line: 0>()
----> 1 history = model.fit(
      2     train_datagen,
      3     validation_data=valid_datagen,
      4     batch_size=batch_size,
      5     epochs=epochs,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/nn.py in categorical_crossentropy(target, output, from_logits, axis)
    658     for e1, e2 in zip(target.shape, output.shape):
    659         if e1 is not None and e2 is not None and e1 != e2:
--> 660             raise ValueError(
    661                 "Arguments `target` and `output` must have the same shape. "
    662                 "Received: "

ValueError: Arguments `target` and `output` must have the same shape. Received: target.shape=(None, 11), output.shape=(None, 10)

## === cell 14
tta_datagen = generator.flow_from_directory(
    train_data_dir,
    target_size=(input_size, input_size),
    batch_size=batch_size,
    subset="validation",
    shuffle=False,
    class_mode="categorical",
)



## === cell 15
n_val = tta_datagen.samples
eve_encodings = np.zeros((n_val, classes), dtype=np.float32)

for _ in range(5):
    encodings = model.predict(tta_datagen, verbose=1)
    eve_encodings += encodings

eve_encodings /= 5.0
eve_encodings[:3]



## === cell 16
pred_classes = np.argmax(eve_encodings, axis=1)
true_classes = tta_datagen.classes
accuracy_score(true_classes, pred_classes)



## === cell 17
plt.figure(figsize=[12, 6], dpi=300)
sns.lineplot(
    x=list(range(len(history.history["accuracy"]))),
    y=history.history["accuracy"],
    label="train",
)
sns.lineplot(
    x=list(range(len(history.history["val_accuracy"]))),
    y=history.history["val_accuracy"],
    label="validation",
)
plt.show()



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3060732024.py in <cell line: 0>()
      1 plt.figure(figsize=[12, 6], dpi=300)
      2 sns.lineplot(
----> 3     x=list(range(len(history.history["accuracy"]))),
      4     y=history.history["accuracy"],
      5     label="train",

NameError: name 'history' is not defined

## === cell 18
plt.figure(figsize=[12, 6], dpi=300)
sns.lineplot(
    x=list(range(len(history.history["loss"]))),
    y=history.history["loss"],
    label="train",
)
sns.lineplot(
    x=list(range(len(history.history["val_loss"]))),
    y=history.history["val_loss"],
    label="validation",
)
plt.show()



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/106012946.py in <cell line: 0>()
      1 plt.figure(figsize=[12, 6], dpi=300)
      2 sns.lineplot(
----> 3     x=list(range(len(history.history["loss"]))),
      4     y=history.history["loss"],
      5     label="train",

NameError: name 'history' is not defined

## === cell 19
temp_hist = pd.DataFrame(history.history)
temp_hist.to_csv("model_effnetb4_history.csv", index=False)
temp_hist.tail()



## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2185903327.py in <cell line: 0>()
----> 1 temp_hist = pd.DataFrame(history.history)
      2 temp_hist.to_csv("model_effnetb4_history.csv", index=False)
      3 temp_hist.tail()
      4 

NameError: name 'history' is not defined

## === cell 20
model.save("model_effnet_b4.hdf5")



## === cell 21
model.save_weights("model_effnet_b4_weights.weights.h5")



## === cell 22
test_loc = "/kaggle/input/paddy-disease-classification/test_images"

test_generator = ImageDataGenerator(
    rescale=1.0 / 255,
    featurewise_center=True,
    featurewise_std_normalization=True,
    horizontal_flip=True,
    vertical_flip=True,
    preprocessing_function=test_time_augmentation_fn,
)



## === cell 23
test_generator.fit(to_gen_fit)



## === cell 24
test_datagen = test_generator.flow_from_directory(
    directory=test_loc,
    target_size=(input_size, input_size),
    batch_size=batch_size,
    classes=["."],
    shuffle=False,
    class_mode=None,
)



## === cell 25
fig, axes = plt.subplots(nrows=2, ncols=8, figsize=[32, 10], dpi=200)
axes = axes.ravel()

test_batch = next(iter(test_datagen))
for i, arr in enumerate(test_batch[: len(axes)]):
    img = array_to_img(arr)
    axes[i].imshow(img)
    axes[i].axis("off")

plt.show()



## === cell 26
n_test = test_datagen.samples
test_encodings = np.zeros((n_test, classes), dtype=np.float32)

for _ in range(5):
    encodings = model.predict(test_datagen, verbose=1)
    test_encodings += encodings

test_encodings /= 5.0
test_encodings[:3]



## === cell 27
predict_max = np.argmax(test_encodings, axis=1)
predict_max[:10]



## === cell 28
inverse_map = {v: k for k, v in train_datagen.class_indices.items()}
predictions = [inverse_map[k] for k in predict_max]
predictions[:10]



## === cell 29
files = test_datagen.filenames
sub = pd.DataFrame({"image_id": files, "label": predictions})
sub["image_id"] = sub["image_id"].str.replace("./", "", regex=False)

sample_path = "/kaggle/input/paddy-disease-classification/sample_submission.csv"
if os.path.exists(sample_path):
    sample = pd.read_csv(sample_path)
    sub = sample[["image_id"]].merge(sub, on="image_id", how="left")
    if sub["label"].isna().any():
        sub["label"] = sub["label"].fillna(pd.Series(predictions).mode().iloc[0])

sub.to_csv("submission.csv", index=False)
sub.head()



## === cell 30
sub.label.value_counts()
