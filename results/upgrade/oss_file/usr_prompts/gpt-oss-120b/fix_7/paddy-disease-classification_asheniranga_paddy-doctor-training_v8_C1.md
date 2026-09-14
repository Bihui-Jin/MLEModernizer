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

0.103

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.05304) has done: 'The fix corrects the broken imports, replaces the incompatible `tensorflow_addons` call with TensorFlow’s own `random_cutout`, updates the checkpoint file extension, imports missing utilities, guards the optional model‑plotting step, and rewrites the test data loader so it works with a flat image folder. These changes let the notebook run end‑to‑end and generate a proper `model_submission_v5.csv` file while preserving the original model architecture.'
- What this solution (achieved 0.04689) has done: 'I fix the import that triggers the protobuf error, replace the nonexistent `tf.image.random_cutout` with a no‑op function, correct the use of the data generator iterator, and give the model‑weights file a proper `.weights.h5` extension. These changes unblock the script, let it train and evaluate, and produce a correctly‑named CSV submission while preserving the original model architecture.'
- What this solution (achieved 0.12836) has done: 'Implemented fixes to unblock the notebook and improve model performance:

1. Set the protobuf implementation environment variable **before** importing TensorFlow to avoid the `MessageFactory` error.  
2. Adjusted `random_cut_out` to be a no‑op that simply returns the input image (keeps the original shape).  
3. Fixed `get_transforms_train` by removing the erroneous `.numpy()` call after `random_cut_out` and ensuring the returned tensor is correctly constructed.  
4. Bumped training epochs slightly to give the model more learning opportunity while keeping the original architecture and augmentation strategy intact.'
- What this solution (achieved 0.103) has done: 'Implemented minimal fixes to unblock the pipeline and generate a valid CSV submission:

* Moved the protobuf environment setting to the very top (before any imports) to avoid the `MessageFactory` error.
* Removed unsupported `workers`, `max_queue_size`, and `use_multiprocessing` arguments from `flow_from_directory` (they caused a `TypeError`).
* Kept the rest of the logic unchanged, preserving the model architecture, training regime, and augmentation pipeline.
* Ensured the script now creates the generators, trains the model, evaluates, predicts on the test set, and writes `model_submission_v5.csv` with the correct columns.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import random
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
import cv2
import albumentations as A

from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Input,
    Dense,
    Conv2D,
    Add,
    Activation,
    MaxPooling2D,
    AveragePooling2D,
    GlobalAveragePooling2D,
    BatchNormalization,
    concatenate,
    Dropout,
    Flatten,
)
from tensorflow.keras.activations import relu, softmax
from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator,
    load_img,
    img_to_array,
    array_to_img,
)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_meta_data = "../train.csv"
train_data_dir = "../input/paddy-disease-classification/train_images"
epochs = 30  # increased slightly for better learning
lr = 1e-4
valid_split = 0.2
input_size = 224
batch_size = 32
initializer = tf.keras.initializers.HeUniform()
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
loss = tf.keras.losses.CategoricalCrossentropy()



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


def random_cut_out(images):
    return images


def aug_fn(image):
    data = {"image": image}
    aug_data = get_transform(**data)
    aug_img = aug_data["image"]
    aug_img = tf.cast(aug_img / 255.0, tf.float32)
    aug_img = tf.image.resize(aug_img, size=[224, 224])
    return aug_img


get_transform = A.Compose(
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
            temp = random_cut_out(temp)  # returns numpy array now
            return tf.convert_to_tensor(temp[0], dtype=tf.float32)

        temp = aug_fn(temp).numpy()
        return tf.convert_to_tensor(temp, dtype=tf.float32)
    else:
        return image




## === cell 4
train_datagen = ImageDataGenerator(
    rescale=1 / 255,
    rotation_range=10,
    shear_range=0.25,
    zoom_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    validation_split=valid_split,
    preprocessing_function=get_transforms_train,
)

train_generator = train_datagen.flow_from_directory(
    "../input/paddy-disease-classification/train_images/",
    target_size=(input_size, input_size),
    batch_size=batch_size,
    class_mode="categorical",
    subset="training",
    shuffle=True,
    seed=42,
)

valid_generator = train_datagen.flow_from_directory(
    "../input/paddy-disease-classification/train_images/",
    target_size=(input_size, input_size),
    batch_size=batch_size,
    class_mode="categorical",
    subset="validation",
    shuffle=False,
    seed=42,
)



## === cell 5
print(f"Training batches per epoch: {len(train_generator)}")
print(f"Validation batches per epoch: {len(valid_generator)}")



## === cell 6
fig, axes = plt.subplots(nrows=4, ncols=8, figsize=[32, 16], dpi=200)
axes = axes.ravel()
for i, arr in enumerate(next(train_generator)[0]):
    img = array_to_img(arr)
    axes[i].imshow(img)
plt.tight_layout()
plt.show()



## === cell 7
fig, axes = plt.subplots(nrows=4, ncols=8, figsize=[32, 16], dpi=200)
axes = axes.ravel()
for i, arr in enumerate(next(valid_generator)[0]):
    img = array_to_img(arr)
    axes[i].imshow(img)
plt.tight_layout()
plt.show()



## === cell 8
meta = pd.read_csv("../input/paddy-disease-classification/train.csv")
plt.figure(figsize=[24, 20], dpi=200)
sns.barplot(x="age", y="label", hue="variety", data=meta, palette="OrRd_r")
plt.show()



## === cell 9
plt.figure(figsize=[12, 6], dpi=200)
sns.barplot(
    x="age",
    y="label",
    hue="variety",
    data=meta.groupby(by=["age", "variety"])[["label"]].count().reset_index(),
    palette="OrRd_r",
)
plt.show()



## === cell 10
back_bone = tf.keras.applications.Xception(
    weights="imagenet", input_shape=(input_size, input_size, 3), include_top=False
)



## === cell 11
try:
    tf.keras.utils.plot_model(back_bone, to_file="xception.png")
except Exception as e:
    print("plot_model failed (non‑critical):", e)



## === cell 12
num_classes = len(train_generator.class_indices)

input_layer = Input(shape=(input_size, input_size, 3))
x = back_bone(input_layer)
x = GlobalAveragePooling2D()(x)
output_layer = Dense(num_classes, activation="softmax")(x)

model = Model(inputs=input_layer, outputs=output_layer)

model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])



## === cell 13
model.summary()



## === cell 14
history = model.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=epochs,
    callbacks=[early_stop, reduce_lr, checkpoint],
    verbose=1,
    workers=8,
    max_queue_size=32,
    use_multiprocessing=True,
)



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_55/831926951.py in <cell line: 0>()
----> 1 history = model.fit(
      2     train_generator,
      3     validation_data=valid_generator,
      4     epochs=epochs,
      5     callbacks=[early_stop, reduce_lr, checkpoint],

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    117             return fn(*args, **kwargs)
    118         except Exception as e:
--> 119             filtered_tb = _process_traceback_frames(e.__traceback__)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`

TypeError: TensorFlowTrainer.fit() got an unexpected keyword argument 'workers'

## === cell 15
eval_res = model.evaluate(valid_generator, verbose=1)
print(f"Validation loss / accuracy: {eval_res}")



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
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1993461226.py in <cell line: 0>()
      1 plt.figure(figsize=[12, 6], dpi=300)
      2 sns.lineplot(
----> 3     x=range(len(history.history["accuracy"])),
      4     y=history.history["accuracy"],
      5     label="train",

NameError: name 'history' is not defined

## === cell 17
plt.figure(figsize=[12, 6], dpi=300)
sns.lineplot(
    x=range(len(history.history["loss"])),
    y=history.history["loss"],
    label="train",
)
sns.lineplot(
    x=range(len(history.history["val_loss"])),
    y=history.history["val_loss"],
    label="validation",
)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4144340067.py in <cell line: 0>()
      1 plt.figure(figsize=[12, 6], dpi=300)
      2 sns.lineplot(
----> 3     x=range(len(history.history["loss"])),
      4     y=history.history["loss"],
      5     label="train",

NameError: name 'history' is not defined

## === cell 18
temp = pd.DataFrame(history.history)
temp.to_csv("model_xception_history.csv", index=False)



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2011983163.py in <cell line: 0>()
----> 1 temp = pd.DataFrame(history.history)
      2 temp.to_csv("model_xception_history.csv", index=False)
      3 

NameError: name 'history' is not defined

## === cell 19
model.save("model_xception.keras")
model.save_weights("model_xception_weights.weights.h5")



## === cell 20
test_dir = "../input/paddy-disease-classification/test_images"
test_files = sorted(glob.glob(os.path.join(test_dir, "*.jpg")))
test_df = pd.DataFrame({"filename": test_files})

test_datagen = ImageDataGenerator(rescale=1 / 255)

test_generator = test_datagen.flow_from_dataframe(
    test_df,
    x_col="filename",
    y_col=None,
    class_mode=None,
    target_size=(input_size, input_size),
    batch_size=batch_size,
    shuffle=False,
)



## === cell 21
print(f"Test batches: {len(test_generator)}")



## === cell 22
predict_max = np.argmax(model.predict(test_generator, verbose=1), axis=1)



## === cell 23
inverse_map = {v: k for k, v in train_generator.class_indices.items()}
predictions = [inverse_map[idx] for idx in predict_max]



## === cell 24
files = [os.path.basename(p) for p in test_df["filename"].values]

submission = pd.DataFrame({"image_id": files, "label": predictions})

submission.to_csv("model_submission_v5.csv", index=False)
print("Submission saved to model_submission_v5.csv")
print(submission.head())



## === cell 25
print(submission["label"].value_counts())
