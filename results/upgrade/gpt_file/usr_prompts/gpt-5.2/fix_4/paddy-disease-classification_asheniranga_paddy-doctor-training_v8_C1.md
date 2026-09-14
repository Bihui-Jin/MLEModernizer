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
import random
import numpy as np
import pandas as pd

import tensorflow as tf

import cv2
import albumentations as A
from albumentations import Compose

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

SEED = 42
tf.keras.utils.set_random_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

tf.config.optimizer.set_jit(True)

print("TF:", tf.__version__)




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
meta = pd.read_csv(train_meta_data)
class_names = sorted(meta["label"].unique().tolist())
print("Detected classes:", class_names, "count:", len(class_names))

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
    train_data_dir + "/",
    target_size=(input_size, input_size),
    batch_size=batch_size,
    subset="training",
    seed=SEED,
    classes=class_names,
    class_mode="categorical",
    shuffle=True,
)

valid_datagen = generator.flow_from_directory(
    train_data_dir + "/",
    target_size=(input_size, input_size),
    batch_size=batch_size,
    subset="validation",
    seed=SEED,
    classes=class_names,
    class_mode="categorical",
    shuffle=False,
)

train_ds = tf.data.Dataset.from_generator(
    lambda: train_datagen,
    output_signature=(
        tf.TensorSpec(shape=(None, input_size, input_size, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None, train_datagen.num_classes), dtype=tf.float32),
    ),
).prefetch(tf.data.AUTOTUNE)

valid_ds = tf.data.Dataset.from_generator(
    lambda: valid_datagen,
    output_signature=(
        tf.TensorSpec(shape=(None, input_size, input_size, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None, valid_datagen.num_classes), dtype=tf.float32),
    ),
).prefetch(tf.data.AUTOTUNE)

print("train_datagen.num_classes:", train_datagen.num_classes)
print("valid_datagen.num_classes:", valid_datagen.num_classes)




## === cell 6
xb, yb = next(iter(train_datagen))
xv, yv = next(iter(valid_datagen))
print("Sanity check batch shapes:", xb.shape, yb.shape, xv.shape, yv.shape)




## === cell 7
pass




## === cell 8
pass




## === cell 9
meta.head()




## === cell 10
pass




## === cell 11
pass




## === cell 12
back_bone = tf.keras.applications.Xception(
    weights="imagenet", input_shape=(input_size, input_size, 3), include_top=False
)
back_bone.summary()




## === cell 13
pass




## === cell 14
input_layer = Input(shape=(input_size, input_size, 3))
x = back_bone(input_layer)
x = GlobalAveragePooling2D()(x)
output_layer = Dense(train_datagen.num_classes, activation="softmax")(x)

model = Model(input_layer, output_layer)

model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"], jit_compile=True)




## === cell 15
model.summary()




## === cell 16
history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=epochs,
    callbacks=[early_stop, reduce_lr, checkpoint],
    verbose=1,
)




## === cell 17
model.evaluate(valid_ds, verbose=1)




## === cell 18
pass




## === cell 19
pass




## === cell 20
temp_hist = pd.DataFrame(history.history)
temp_hist.to_csv("model_xception_history.csv", index=False)
temp_hist.head()




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

test_ds = tf.data.Dataset.from_generator(
    lambda: test_data,
    output_signature=(
        tf.TensorSpec(shape=(None, input_size, input_size, 3), dtype=tf.float32),
        tf.TensorSpec(
            shape=(None, 1), dtype=tf.float32
        ),  # dummy labels from directory iterator
    ),
).prefetch(tf.data.AUTOTUNE)




## === cell 24
train_datagen.class_indices




## === cell 25
pred_probs = model.predict(test_ds, verbose=1)
predict_max = np.argmax(pred_probs, axis=1)




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
    most_common_label = meta["label"].mode().iloc[0]
    sub["label"] = sub["label"].fillna(most_common_label)

sub.to_csv("model_submission_v5.csv", index=False)
sub.head()




## === cell 28
sub.label.value_counts()
