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

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.18332) has done: 'The fixes address the import error, incorrect ImageDataGenerator arguments, wrong file paths, misuse of TensorFlow Hub, and the loss of file‑paths after dataset mapping. Paths are switched to the Kaggle `/kaggle/input` location, the EfficientNet B4 model is built directly from `tf.keras.applications`, and the test dataset keeps its original file list for submission. All cells now run sequentially and produce a valid `model_submission_v17.csv` file.'

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
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from tensorflow.keras import mixed_precision

mixed_precision.set_global_policy("mixed_float16")

train_meta_data = "/kaggle/input/paddy-disease-classification/train.csv"
train_data_dir = "/kaggle/input/paddy-disease-classification/train_images"
test_data_dir = "/kaggle/input/paddy-disease-classification/test_images"

train_df = pd.read_csv(train_meta_data)
class_names = sorted(train_df["label"].unique())  # ensures 10 classes in correct order

epochs = 30
lr = 1e-4
valid_split = 0.2
input_size = 128
batch_size = 16
classes = len(class_names)  # should be 10
optimizer = tf.keras.optimizers.Nadam(learning_rate=lr)
loss = tf.keras.losses.CategoricalCrossentropy()
initializer = tf.keras.initializers.HeUniform()




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
early_stop = tf.keras.callbacks.EarlyStopping(
    patience=20, monitor="val_loss", restore_best_weights=True, verbose=1
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    patience=5, monitor="val_loss", factor=0.5, verbose=1
)




## === cell 2
src = os.path.join(train_data_dir, "dead_heart", "100008.jpg")
if os.path.exists(src):
    img = tf.keras.preprocessing.image.load_img(src)
    img_arr = tf.keras.preprocessing.image.img_to_array(img, dtype="uint8")
else:
    img_arr = None




## === cell 3
def random_cutout(image, patch_size=16, patches=16):
    if random.choice([True, False]):
        anchors_x, anchors_y = [], []
        for _ in range(patches):
            rv = np.random.randint(0, image.shape[0])
            if rv not in anchors_x:
                anchors_x.append(rv)
        for _ in range(patches):
            rv = np.random.randint(0, image.shape[0])
            if rv not in anchors_y:
                anchors_y.append(rv)
        for x, y in zip(anchors_x, anchors_y):
            image[x : x + patch_size, y : y + patch_size, :] = 0
    return image


def random_gaus_blur(image):
    if random.choice([True, False]):
        return cv2.GaussianBlur(image, (7, 7), 0)
    return image


def random_displacement(image):
    if random.choice([True, False]):
        ax = random.choice([0, 1])
        slices = np.split(image, 8, axis=ax)
        np.random.shuffle(slices)
        return np.row_stack(slices) if ax == 0 else np.column_stack(slices)
    return image


def center_crop_and_random_augmentations_fn(image):
    image = tf.image.random_crop(image, (input_size, input_size, 3)).numpy()
    image = random_cutout(image, 8, 16)
    image = random_displacement(image)
    image = random_gaus_blur(image)
    image = tf.image.random_brightness(image, 0.2).numpy()
    image = tf.image.random_contrast(image, 0.5, 2.0).numpy()
    image = tf.image.random_saturation(image, 0.75, 1.25).numpy()
    image = tf.image.random_hue(image, 0.1).numpy()
    return image


def test_time_augmentation_fn(image):
    image = tf.image.random_crop(image, (input_size, input_size, 3)).numpy()
    image = tf.image.random_brightness(image, 0.2).numpy()
    image = tf.image.random_contrast(image, 0.5, 2.0).numpy()
    return image




## === cell 4
generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1 / 255,
    rotation_range=5,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    validation_split=valid_split,
    preprocessing_function=center_crop_and_random_augmentations_fn,
)

train_datagen = generator.flow_from_directory(
    train_data_dir,
    target_size=(input_size, input_size),
    batch_size=batch_size,
    classes=class_names,
    subset="training",
    shuffle=True,
    seed=42,
    class_mode="categorical",
    workers=4,
    use_multiprocessing=True,
)

valid_datagen = generator.flow_from_directory(
    train_data_dir,
    target_size=(input_size, input_size),
    batch_size=batch_size,
    classes=class_names,
    subset="validation",
    shuffle=False,
    seed=42,
    class_mode="categorical",
    workers=4,
    use_multiprocessing=True,
)




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1116440961.py in <cell line: 0>()
     10 )
     11 
---> 12 train_datagen = generator.flow_from_directory(
     13     train_data_dir,
     14     target_size=(input_size, input_size),

TypeError: ImageDataGenerator.flow_from_directory() got an unexpected keyword argument 'workers'

## === cell 5
pass




## === cell 6
print(
    "Train batch shape:",
    next(train_datagen)[0].shape,
    "Validation batch shape:",
    next(valid_datagen)[0].shape,
)




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3137579304.py in <cell line: 0>()
      1 print(
      2     "Train batch shape:",
----> 3     next(train_datagen)[0].shape,
      4     "Validation batch shape:",
      5     next(valid_datagen)[0].shape,

NameError: name 'train_datagen' is not defined

## === cell 7
fig, axes = plt.subplots(nrows=2, ncols=8, figsize=[32, 10], dpi=200)
axes = axes.ravel()
for i, arr in enumerate(next(train_datagen)[0][:16]):
    img = tf.keras.preprocessing.image.array_to_img(arr)
    axes[i].imshow(img)
plt.show()




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/320876115.py in <cell line: 0>()
      1 fig, axes = plt.subplots(nrows=2, ncols=8, figsize=[32, 10], dpi=200)
      2 axes = axes.ravel()
----> 3 for i, arr in enumerate(next(train_datagen)[0][:16]):
      4     img = tf.keras.preprocessing.image.array_to_img(arr)
      5     axes[i].imshow(img)

NameError: name 'train_datagen' is not defined

## === cell 8
fig, axes = plt.subplots(nrows=2, ncols=8, figsize=[32, 10], dpi=200)
axes = axes.ravel()
for i, arr in enumerate(next(valid_datagen)[0][:16]):
    img = tf.keras.preprocessing.image.array_to_img(arr)
    axes[i].imshow(img)
plt.show()




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1887891722.py in <cell line: 0>()
      1 fig, axes = plt.subplots(nrows=2, ncols=8, figsize=[32, 10], dpi=200)
      2 axes = axes.ravel()
----> 3 for i, arr in enumerate(next(valid_datagen)[0][:16]):
      4     img = tf.keras.preprocessing.image.array_to_img(arr)
      5     axes[i].imshow(img)

NameError: name 'valid_datagen' is not defined

## === cell 9
base_model = tf.keras.applications.EfficientNetB4(
    include_top=False,
    weights="imagenet",
    input_shape=(input_size, input_size, 3),
    pooling="avg",
)

model = tf.keras.Sequential(
    [
        base_model,
        tf.keras.layers.Dense(classes, activation="softmax"),
    ]
)




## === cell 10
model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])




## === cell 11
model.summary()




## === cell 12
history = model.fit(
    train_datagen,
    validation_data=valid_datagen,
    epochs=epochs,
    callbacks=[early_stop, reduce_lr],
    verbose=1,
)




## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2775695056.py in <cell line: 0>()
      1 history = model.fit(
----> 2     train_datagen,
      3     validation_data=valid_datagen,
      4     epochs=epochs,
      5     callbacks=[early_stop, reduce_lr],

NameError: name 'train_datagen' is not defined

## === cell 13
tta_datagen = generator.flow_from_directory(
    train_data_dir,
    target_size=(input_size, input_size),
    batch_size=batch_size,
    classes=class_names,
    subset="validation",
    shuffle=False,
    seed=42,
    class_mode=None,
    workers=4,
    use_multiprocessing=True,
)




## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_55/143155845.py in <cell line: 0>()
----> 1 tta_datagen = generator.flow_from_directory(
      2     train_data_dir,
      3     target_size=(input_size, input_size),
      4     batch_size=batch_size,
      5     classes=class_names,

TypeError: ImageDataGenerator.flow_from_directory() got an unexpected keyword argument 'workers'

## === cell 14
eve_encodings = np.zeros((len(tta_datagen.filenames), classes))
for _ in range(5):
    enc = model.predict(tta_datagen, verbose=0)
    eve_encodings += enc
eve_encodings /= 5




## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2260989702.py in <cell line: 0>()
----> 1 eve_encodings = np.zeros((len(tta_datagen.filenames), classes))
      2 for _ in range(5):
      3     enc = model.predict(tta_datagen, verbose=0)
      4     eve_encodings += enc
      5 eve_encodings /= 5

NameError: name 'tta_datagen' is not defined

## === cell 15
pred_classes = np.argmax(eve_encodings, axis=1)
true_classes = tta_datagen.classes
print("Validation accuracy:", accuracy_score(true_classes, pred_classes))




## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2567983394.py in <cell line: 0>()
----> 1 pred_classes = np.argmax(eve_encodings, axis=1)
      2 true_classes = tta_datagen.classes
      3 print("Validation accuracy:", accuracy_score(true_classes, pred_classes))
      4 
      5 

NameError: name 'eve_encodings' is not defined

## === cell 16
plt.figure(figsize=[12, 6], dpi=300)
sns.lineplot(
    x=range(len(history.history["accuracy"])),
    y=history.history["accuracy"],
    label="train",
)
sns.lineplot(
    x=range(len(history.history["val_accuracy"])),
    y=history.history["val_accuracy"],
    label="validation",
)
plt.show()




## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/456603417.py in <cell line: 0>()
      1 plt.figure(figsize=[12, 6], dpi=300)
      2 sns.lineplot(
----> 3     x=range(len(history.history["accuracy"])),
      4     y=history.history["accuracy"],
      5     label="train",

NameError: name 'history' is not defined

## === cell 17
plt.figure(figsize=[12, 6], dpi=300)
sns.lineplot(
    x=range(len(history.history["loss"])), y=history.history["loss"], label="train"
)
sns.lineplot(
    x=range(len(history.history["val_loss"])),
    y=history.history["val_loss"],
    label="validation",
)
plt.show()




## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1312310631.py in <cell line: 0>()
      1 plt.figure(figsize=[12, 6], dpi=300)
      2 sns.lineplot(
----> 3     x=range(len(history.history["loss"])), y=history.history["loss"], label="train"
      4 )
      5 sns.lineplot(

NameError: name 'history' is not defined

## === cell 18
pd.DataFrame(history.history).to_csv("model_effnetb4_history.csv", index=False)




## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3619690965.py in <cell line: 0>()
----> 1 pd.DataFrame(history.history).to_csv("model_effnetb4_history.csv", index=False)
      2 
      3 

NameError: name 'history' is not defined

## === cell 19
model.save("model_effnet_b4.hdf5")
model.save_weights("model_effnet_b4.weights.h5")




## === cell 20
test_dataset = tf.keras.utils.image_dataset_from_directory(
    test_data_dir,
    labels=None,
    image_size=(input_size, input_size),
    batch_size=batch_size,
    shuffle=False,
)
test_file_paths = test_dataset.file_paths




## === cell 21
def normalize_batch(batch):
    batch = tf.cast(batch, tf.float32) / 255.0
    return batch


test_dataset = test_dataset.map(lambda x: normalize_batch(x))




## === cell 22
test_encodings = np.zeros((len(test_file_paths), classes))
for _ in range(5):
    enc = model.predict(test_dataset, verbose=0)
    test_encodings += enc
test_encodings /= 5




## === cell 23
predict_max = np.argmax(test_encodings, axis=1)




## === cell 24
inverse_map = {v: k for k, v in train_datagen.class_indices.items()}
predictions = [inverse_map[k] for k in predict_max]




## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2727795600.py in <cell line: 0>()
----> 1 inverse_map = {v: k for k, v in train_datagen.class_indices.items()}
      2 predictions = [inverse_map[k] for k in predict_max]
      3 
      4 

NameError: name 'train_datagen' is not defined

## === cell 25
files = [os.path.basename(p) for p in test_file_paths]
submission = pd.DataFrame({"image_id": files, "label": predictions})
submission.to_csv("model_submission_v17.csv", index=False)
print(submission.head())

## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/824998782.py in <cell line: 0>()
      1 files = [os.path.basename(p) for p in test_file_paths]
----> 2 submission = pd.DataFrame({"image_id": files, "label": predictions})
      3 submission.to_csv("model_submission_v17.csv", index=False)
      4 print(submission.head())

NameError: name 'predictions' is not defined
