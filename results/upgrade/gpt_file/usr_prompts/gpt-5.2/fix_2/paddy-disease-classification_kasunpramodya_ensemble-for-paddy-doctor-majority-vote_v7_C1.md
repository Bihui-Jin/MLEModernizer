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

No external packages required in the script and installed.

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

from tensorflow.keras.preprocessing.image import ImageDataGenerator

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

BASE_CANDIDATES = [
    "/kaggle/input/paddy-disease-classification",
    "/kaggle/input/paddy-disease-classification/paddy-disease-classification",
    "/kaggle/data/paddy-disease-classification",
    "/kaggle/data/paddy-disease-classification/paddy-disease-classification",
]
BASE = None
for c in BASE_CANDIDATES:
    if os.path.exists(c):
        BASE = c
        break
if BASE is None:
    raise FileNotFoundError(
        "Could not locate paddy-disease-classification dataset folder in known locations."
    )

TRAIN_CSV = os.path.join(BASE, "train.csv")
SAMPLE_SUB = os.path.join(BASE, "sample_submission.csv")

TRAIN_IMG_CANDIDATES = [
    os.path.join(BASE, "train_images"),
    os.path.join(BASE, "paddy-disease-classification", "train_images"),
]
TEST_IMG_CANDIDATES = [
    os.path.join(BASE, "test_images"),
    os.path.join(BASE, "paddy-disease-classification", "test_images"),
]

TRAIN_IMG_DIR = next((p for p in TRAIN_IMG_CANDIDATES if os.path.isdir(p)), None)
TEST_IMG_DIR = next((p for p in TEST_IMG_CANDIDATES if os.path.isdir(p)), None)
if TRAIN_IMG_DIR is None or TEST_IMG_DIR is None:
    raise FileNotFoundError(f"Could not find train_images/test_images under {BASE}")

print("BASE:", BASE)
print("TRAIN_CSV:", TRAIN_CSV)
print("TRAIN_IMG_DIR:", TRAIN_IMG_DIR)
print("TEST_IMG_DIR:", TEST_IMG_DIR)



## === cell 1
train_df = pd.read_csv(TRAIN_CSV)

required_cols = {"image_id", "label"}
missing = required_cols - set(train_df.columns)
if missing:
    raise ValueError(f"train.csv missing columns: {missing}")

labels_sorted = sorted(train_df["label"].unique().tolist())
print("Num classes:", len(labels_sorted))
print("Classes:", labels_sorted)

train_df = train_df.copy()
train_df["filename"] = train_df.apply(lambda r: f"{r['label']}/{r['image_id']}", axis=1)

for i in range(3):
    fp = os.path.join(TRAIN_IMG_DIR, train_df["filename"].iloc[i])
    if not os.path.exists(fp):
        raise FileNotFoundError(f"Example train image not found: {fp}")



## === cell 2
from sklearn.model_selection import train_test_split

train_split, valid_split = train_test_split(
    train_df,
    test_size=0.1,
    random_state=SEED,
    stratify=train_df["label"],
)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.1,
    horizontal_flip=True,
)

valid_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

train_gen = train_datagen.flow_from_dataframe(
    train_split,
    directory=TRAIN_IMG_DIR,
    x_col="filename",
    y_col="label",
    target_size=IMG_SIZE,
    class_mode="categorical",
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED,
)

valid_gen = valid_datagen.flow_from_dataframe(
    valid_split,
    directory=TRAIN_IMG_DIR,
    x_col="filename",
    y_col="label",
    target_size=IMG_SIZE,
    class_mode="categorical",
    batch_size=BATCH_SIZE,
    shuffle=False,
)

class_indices = train_gen.class_indices
inverse_map = {v: k for k, v in class_indices.items()}
print("class_indices:", class_indices)



## === cell 3
back_bone = tf.keras.applications.Xception(
    weights="imagenet",
    input_shape=(224, 224, 3),
    include_top=False,
)

input_layer = tf.keras.layers.Input(shape=(224, 224, 3))
x = back_bone(input_layer)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
output_layer = tf.keras.layers.Dense(len(class_indices), activation="softmax")(x)
model = tf.keras.models.Model(input_layer, output_layer)

back_bone.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.categorical_crossentropy,
    metrics=["accuracy"],
)

model.summary()



## === cell 4
EPOCHS_HEAD = 3
history_head = model.fit(
    train_gen,
    validation_data=valid_gen,
    epochs=EPOCHS_HEAD,
    verbose=1,
)

back_bone.trainable = True
for layer in back_bone.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=tf.keras.losses.categorical_crossentropy,
    metrics=["accuracy"],
)

EPOCHS_FT = 2
history_ft = model.fit(
    train_gen,
    validation_data=valid_gen,
    epochs=EPOCHS_FT,
    verbose=1,
)



## === cell 5
test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

test_gen = test_datagen.flow_from_directory(
    directory=TEST_IMG_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    classes=["."],
    class_mode=None,
    shuffle=False,
)

pred = model.predict(test_gen, verbose=1)
pred_idx = np.argmax(pred, axis=1)
pred_labels = [inverse_map[i] for i in pred_idx]

image_ids = [f.replace("./", "") for f in test_gen.filenames]

sub = pd.DataFrame({"image_id": image_ids, "label": pred_labels})

sample_sub = pd.read_csv(SAMPLE_SUB)
if "image_id" in sample_sub.columns and len(sample_sub) == len(sub):
    sub = sample_sub[["image_id"]].merge(sub, on="image_id", how="left")
    if sub["label"].isna().any():
        fill_label = train_df["label"].mode().iloc[0]
        sub["label"] = sub["label"].fillna(fill_label)

sub.to_csv("submission.csv", index=False)
print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
