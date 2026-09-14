# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

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

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print("Could not set memory growth:", e)




## === cell 1
train_meta_data = "../train.csv"
train_data_dir = "../input/paddy-disease-classification/train_images"
epochs = 30
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
def get_transforms_train(image):
    return image.astype(np.float32)


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




## === cell 4
print(f"Training batches per epoch: {len(train_generator)}")
print(f"Validation batches per epoch: {len(valid_generator)}")




## === cell 5
fig, axes = plt.subplots(nrows=4, ncols=8, figsize=[32, 16], dpi=200)
axes = axes.ravel()
for i, arr in enumerate(next(train_generator)[0]):
    img = array_to_img(arr)
    axes[i].imshow(img)
plt.tight_layout()
plt.show()




## === cell 6
fig, axes = plt.subplots(nrows=4, ncols=8, figsize=[32, 16], dpi=200)
axes = axes.ravel()
for i, arr in enumerate(next(valid_generator)[0]):
    img = array_to_img(arr)
    axes[i].imshow(img)
plt.tight_layout()
plt.show()




## === cell 7
meta = pd.read_csv("../input/paddy-disease-classification/train.csv")
plt.figure(figsize=[24, 20], dpi=200)
sns.barplot(x="age", y="label", hue="variety", data=meta, palette="OrRd_r")
plt.show()




## === cell 8
plt.figure(figsize=[12, 6], dpi=200)
sns.barplot(
    x="age",
    y="label",
    hue="variety",
    data=meta.groupby(by=["age", "variety"])[["label"]].count().reset_index(),
    palette="OrRd_r",
)
plt.show()




## === cell 9
back_bone = tf.keras.applications.Xception(
    weights="imagenet", input_shape=(input_size, input_size, 3), include_top=False
)




## === cell 10
try:
    tf.keras.utils.plot_model(back_bone, to_file="xception.png")
except Exception as e:
    print("plot_model failed (non‑critical):", e)




## === cell 11
num_classes = len(train_generator.class_indices)

input_layer = Input(shape=(input_size, input_size, 3))
x = back_bone(input_layer)
x = GlobalAveragePooling2D()(x)
output_layer = Dense(num_classes, activation="softmax")(x)

model = Model(inputs=input_layer, outputs=output_layer)

model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])




## === cell 12
model.summary()




## === cell 13
history = model.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=epochs,
    callbacks=[early_stop, reduce_lr, checkpoint],
    verbose=1,
)




## === cell 14
eval_res = model.evaluate(valid_generator, verbose=1)
print(f"Validation loss / accuracy: {eval_res}")




## === cell 15
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




## === cell 16
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




## === cell 17
temp = pd.DataFrame(history.history)
temp.to_csv("model_xception_history.csv", index=False)




## === cell 18
model.save("model_xception.keras")
model.save_weights("model_xception_weights.weights.h5")




## === cell 19
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




## === cell 20
print(f"Test batches: {len(test_generator)}")




## === cell 21
predict_max = np.argmax(model.predict(test_generator, verbose=1), axis=1)




## === cell 22
inverse_map = {v: k for k, v in train_generator.class_indices.items()}
predictions = [inverse_map[idx] for idx in predict_max]




## === cell 23
files = [os.path.basename(p) for p in test_df["filename"].values]

submission = pd.DataFrame({"image_id": files, "label": predictions})

submission.to_csv("model_submission.csv", index=False)
print("Submission saved to model_submission.csv")
print(submission.head())




## === cell 24
print(submission["label"].value_counts())
