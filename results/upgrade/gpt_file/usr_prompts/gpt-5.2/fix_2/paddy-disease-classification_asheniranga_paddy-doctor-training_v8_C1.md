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

0.96543

# 6. Current score

0.06764

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.06764) has done: 'I fix the runtime blockers that prevent the notebook from importing/defining key objects and from writing a valid submission CSV. Specifically: remove the incompatible `tensorflow_addons` dependency (it triggers the protobuf `MessageFactory` error) and replace its cutout augmentation with an equivalent pure-TensorFlow implementation; update Keras checkpoint/saving filenames to the required `.keras`/`.weights.h5` extensions for TF/Keras 2.18; and make `Compose`/`ImageDataGenerator`/`plt` reliably available by using their current import locations. I also skip `plot_model()` if Graphviz/dot isn’t available (non-critical) and ensure test `image_id` formatting matches the sample submission (no directory prefixes). The core model (Xception + GAP + Dense softmax) and training loop remain the same, and the script finish by writing `model_submission_v5.csv`.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
import cv2
import albumentations as A

from albumentations import Compose

from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
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

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import load_img, img_to_array, array_to_img

SEED = 42
tf.keras.utils.set_random_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

print("TF:", tf.__version__)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_meta_data = "../input/paddy-disease-classification/train.csv"
train_data_dir = "../input/paddy-disease-classification/train_images"
epochs = 100
lr = 1e-4
valid_split = 0.2
input_size = 224
batch_size = 32
classes = 10
initializer = tf.keras.initializers.HeUniform()
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
loss = tf.keras.losses.categorical_crossentropy



## === cell 2
early_stop = tf.keras.callbacks.EarlyStopping(
    patience=15, monitor="val_loss", restore_best_weights=True, verbose=1
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    patience=5, monitor="val_loss", factor=0.75, verbose=1
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath="best_chp.keras", monitor="val_loss", verbose=1, save_best_only=True
)




## === cell 3
def resize(image, size):
    return tf.image.resize(image, size)


def blur(img, blur_limit):
    return cv2.blur(img, ksize=[blur_limit, blur_limit])


def gaussian_blur(img, blur_limit=(3, 7), sigma_limit=0):
    return cv2.GaussianBlur(img, ksize=blur_limit, sigmaX=sigma_limit)


def motion_blur(img, blur_limit=7):
    kmb = np.zeros((blur_limit, blur_limit))
    kmb[(blur_limit - 1) // 2, :] = np.ones(blur_limit)
    kmb = kmb / blur_limit
    return cv2.filter2D(img, -1, kernel=kmb)


def random_cut_out(images, cutout_size=(32, 32), constant_values=0.0):
    """
    images: Tensor [B,H,W,C]
    """
    images = tf.convert_to_tensor(images)
    b = tf.shape(images)[0]
    h = tf.shape(images)[1]
    w = tf.shape(images)[2]
    c = tf.shape(images)[3]

    cut_h = tf.cast(cutout_size[0], tf.int32)
    cut_w = tf.cast(cutout_size[1], tf.int32)

    cy = tf.random.uniform([b], 0, h, dtype=tf.int32)
    cx = tf.random.uniform([b], 0, w, dtype=tf.int32)

    y1 = tf.clip_by_value(cy - cut_h // 2, 0, h)
    y2 = tf.clip_by_value(y1 + cut_h, 0, h)
    x1 = tf.clip_by_value(cx - cut_w // 2, 0, w)
    x2 = tf.clip_by_value(x1 + cut_w, 0, w)

    yy = tf.range(h)[None, :, None, None]  # [1,H,1,1]
    xx = tf.range(w)[None, None, :, None]  # [1,1,W,1]

    y1_ = y1[:, None, None, None]
    y2_ = y2[:, None, None, None]
    x1_ = x1[:, None, None, None]
    x2_ = x2[:, None, None, None]

    mask_y = tf.logical_and(yy >= y1_, yy < y2_)  # [B,H,1,1]
    mask_x = tf.logical_and(xx >= x1_, xx < x2_)  # [B,1,W,1]
    mask = tf.logical_and(mask_y, mask_x)  # [B,H,W,1]
    mask = tf.tile(mask, [1, 1, 1, c])  # [B,H,W,C]

    fill = tf.cast(constant_values, images.dtype)
    return tf.where(mask, fill, images)


def aug_fn(image):
    data = {"image": image}
    aug_data = get_transform(**data)
    aug_img = aug_data["image"]
    aug_img = tf.cast(aug_img / 255.0, tf.float32)
    aug_img = tf.image.resize(aug_img, size=[224, 224])
    return aug_img


get_transform = Compose(
    [
        A.CoarseDropout(
            max_holes=16,
            min_holes=8,
            max_height=16,
            max_width=16,
            min_height=8,
            min_width=8,
            p=0.2,
        )
    ]
)




## === cell 4
def get_transforms_train(image):
    if np.random.choice([True, False], p=[0.45, 0.55]):
        crop_side = int(224 * random.uniform(0.5, 1))
        temp = tf.image.random_crop(image, size=(crop_side, crop_side, 3)).numpy()
        temp = resize(temp, size=(224, 224)).numpy()

        temp = tf.image.random_flip_left_right(temp).numpy()

        if np.random.choice([True, False], p=[0.45, 0.55]):
            if random.choice([True, False]):
                delta = random.uniform(-0.3, 0.3)
                cf = random.uniform(-1.0, 1.0)
                temp = tf.image.adjust_brightness(temp, delta=delta).numpy()
                temp = tf.image.adjust_contrast(temp, contrast_factor=cf).numpy()

        if np.random.choice([True, False], p=[0.25, 0.75]):
            delta = random.uniform(-0.1, 0.2)
            temp = tf.image.adjust_hue(temp, delta=delta).numpy()

        if np.random.choice([True, False], p=[0.3, 0.7]):
            temp = temp.reshape([1, temp.shape[0], temp.shape[1], 3])
            temp = random_cut_out(
                temp, cutout_size=(32, 32), constant_values=0.0
            ).numpy()
            return tf.convert_to_tensor(temp[0], dtype=tf.float32)

        temp = aug_fn(temp).numpy()
        return tf.convert_to_tensor(temp, dtype=tf.float32)
    else:
        return image




## === cell 5
generator = ImageDataGenerator(
    rescale=1 / 255,
    rotation_range=10,
    shear_range=0.25,
    zoom_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    validation_split=valid_split,
)

train_datagen = generator.flow_from_directory(
    "../input/paddy-disease-classification/train_images/",
    target_size=(input_size, input_size),
    batch_size=batch_size,
    subset="training",
    seed=SEED,
)

valid_datagen = generator.flow_from_directory(
    "../input/paddy-disease-classification/train_images/",
    target_size=(input_size, input_size),
    batch_size=batch_size,
    subset="validation",
    seed=SEED,
)



## === cell 6
len(train_datagen.next()[0]), len(valid_datagen.next()[0])



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/3027733224.py in <cell line: 0>()
----> 1 len(train_datagen.next()[0]), len(valid_datagen.next()[0])
      2 

AttributeError: 'DirectoryIterator' object has no attribute 'next'

## === cell 7
fig, axes = plt.subplots(nrows=4, ncols=8, figsize=[32, 16], dpi=200)
axes = axes.ravel()

for i, arr in enumerate(train_datagen.next()[0]):
    img = array_to_img(arr)
    axes[i].imshow(img)
    axes[i].axis("off")

plt.show()



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/1040565158.py in <cell line: 0>()
      2 axes = axes.ravel()
      3 
----> 4 for i, arr in enumerate(train_datagen.next()[0]):
      5     img = array_to_img(arr)
      6     axes[i].imshow(img)

AttributeError: 'DirectoryIterator' object has no attribute 'next'

## === cell 8
fig, axes = plt.subplots(nrows=4, ncols=8, figsize=[32, 16], dpi=200)
axes = axes.ravel()

for i, arr in enumerate(valid_datagen.next()[0]):
    img = array_to_img(arr)
    axes[i].imshow(img)
    axes[i].axis("off")

plt.show()



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/3661813812.py in <cell line: 0>()
      2 axes = axes.ravel()
      3 
----> 4 for i, arr in enumerate(valid_datagen.next()[0]):
      5     img = array_to_img(arr)
      6     axes[i].imshow(img)

AttributeError: 'DirectoryIterator' object has no attribute 'next'

## === cell 9
meta = pd.read_csv("../input/paddy-disease-classification/train.csv")
meta.head()



## === cell 10
plt.figure(figsize=[24, 20], dpi=200)
sns.barplot(x="age", y="label", hue="variety", data=meta, palette="OrRd_r")
plt.show()



## === cell 11
plt.figure(figsize=[12, 6], dpi=200)
sns.barplot(
    x="age",
    y="label",
    hue="variety",
    data=meta.groupby(by=["age", "variety"])[["label"]].count().reset_index(),
    palette="OrRd_r",
)
plt.show()



## === cell 12
back_bone = tf.keras.applications.Xception(
    weights="imagenet", input_shape=(input_size, input_size, 3), include_top=False
)
back_bone.summary()



## === cell 13
try:
    tf.keras.utils.plot_model(back_bone, to_file="xception.png")
    print("Saved model plot to xception.png")
except Exception as e:
    print("Skipping plot_model due to environment limitation:", repr(e))



## === cell 14
input_layer = Input(shape=(input_size, input_size, 3))
x = back_bone(input_layer)
x = GlobalAveragePooling2D()(x)
output_layer = Dense(10, activation="softmax")(x)

model = Model(input_layer, output_layer)

model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])



## === cell 15
model.summary()



## === cell 16
history = model.fit(
    train_datagen,
    validation_data=valid_datagen,
    batch_size=batch_size,
    epochs=epochs,
    callbacks=[early_stop, reduce_lr, checkpoint],
)



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3841833056.py in <cell line: 0>()
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

## === cell 17
model.evaluate(valid_datagen, verbose=1)



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1444370453.py in <cell line: 0>()
----> 1 model.evaluate(valid_datagen, verbose=1)
      2 

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

## === cell 18
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



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3060732024.py in <cell line: 0>()
      1 plt.figure(figsize=[12, 6], dpi=300)
      2 sns.lineplot(
----> 3     x=list(range(len(history.history["accuracy"]))),
      4     y=history.history["accuracy"],
      5     label="train",

NameError: name 'history' is not defined

## === cell 19
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



## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/106012946.py in <cell line: 0>()
      1 plt.figure(figsize=[12, 6], dpi=300)
      2 sns.lineplot(
----> 3     x=list(range(len(history.history["loss"]))),
      4     y=history.history["loss"],
      5     label="train",

NameError: name 'history' is not defined

## === cell 20
temp_hist = pd.DataFrame(history.history)
temp_hist.to_csv("model_xception_history.csv", index=False)
temp_hist.head()



## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2498000146.py in <cell line: 0>()
----> 1 temp_hist = pd.DataFrame(history.history)
      2 temp_hist.to_csv("model_xception_history.csv", index=False)
      3 temp_hist.head()
      4 

NameError: name 'history' is not defined

## === cell 21
model.save("model_xception.keras")



## === cell 22
model.save_weights("model_xception_weights.weights.h5")



## === cell 23
test_loc = "../input/paddy-disease-classification/test_images"

test_data = ImageDataGenerator(rescale=1.0 / 255).flow_from_directory(
    directory=test_loc,
    target_size=(input_size, input_size),
    batch_size=batch_size,
    classes=["."],
    shuffle=False,
)



## === cell 24
train_datagen.class_indices



## === cell 25
predict_max = np.argmax(model.predict(test_data, verbose=1), axis=1)



## === cell 26
inverse_map = {v: k for k, v in train_datagen.class_indices.items()}
predictions = [inverse_map[k] for k in predict_max]



## === cell 27
files = test_data.filenames
sub = pd.DataFrame({"image_id": files, "label": predictions})

sub["image_id"] = sub["image_id"].str.replace("./", "", regex=False)
sub["image_id"] = sub["image_id"].str.replace(".\\", "", regex=False)
sub["image_id"] = sub["image_id"].str.replace("/", "", regex=False)

sample = pd.read_csv("../input/paddy-disease-classification/sample_submission.csv")
sub = sample[["image_id"]].merge(sub, on="image_id", how="left")

if sub["label"].isna().any():
    most_common_label = max(
        train_datagen.class_indices.keys(), key=lambda k: (k == "normal", k)
    )
    sub["label"] = sub["label"].fillna(most_common_label)

sub.to_csv("model_submission_v5.csv", index=False)
sub.head()



## === cell 28
sub.label.value_counts()
