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

3.12

# 3. Installed packages

geopandas==0.14.4
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
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
tf_keras==2.18.0

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

0.82373

# 6. Current score

0.72175

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.72175) has done: 'I remove notebook-only magics and the problematic unused imports that trigger the protobuf `MessageFactory` crash, so the script runs in Kaggle’s Python environment. I also fix the broken visualization cell paths (those images don’t exist) by making it robust/optional, without affecting training/inference. Then I correct two accuracy-killing logic bugs in the training pipeline: the validation dataset uses `(img_height, img_height)` instead of `(img_height, img_width)`, and the loss is incorrectly configured with `from_logits=True` despite a `softmax` output. Finally, I make test dataset loading work with the flat `test_images/` directory by creating a temporary “dummy class” folder with symlinks/copies so `image_dataset_from_directory()` can read all test images and produce a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from keras.callbacks import EarlyStopping

print("TensorFlow:", tf.__version__)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
data = pd.read_csv("/kaggle/input/paddy-disease-classification/train.csv")
data.head()



## === cell 2
data.shape



## === cell 3
data["label"].unique().tolist()



## === cell 4
data["variety"].unique().tolist()



## === cell 5
data.age.describe()



## === cell 6
fig, ax = plt.subplots(1, 1, figsize=(21, 7))
sns.histplot(x="variety", data=data, ax=ax)
plt.title("Variety distribution in the dataset")
plt.xticks(rotation=90)
plt.show()



## === cell 7
fig, ax = plt.subplots(1, 1, figsize=(21, 7))
sns.histplot(x="label", data=data, ax=ax)
plt.title("Disease distribution in the dataset")
plt.xticks(rotation=45)
plt.show()



## === cell 8
normal = data[data["label"] == "normal"]
normal = normal[normal["variety"] == "ADT45"]
five_normals = normal.image_id[:5].values
five_normals.tolist()



## === cell 9
dead = data[data["label"] == "dead_heart"]
dead = dead[dead["variety"] == "ADT45"]
five_deads = dead.image_id[:5].values
five_deads.tolist()



## === cell 10
plt.figure(figsize=(20, 10))
columns = 5
path = "/kaggle/input/paddy-disease-classification/train_images/"
for i, image_loc in enumerate(np.concatenate((five_normals, five_deads))):
    plt.subplot(10 // columns + 1, columns, i + 1)

    if i < 5:
        image_path = os.path.join(path, "normal", image_loc)
        title = "normal"
    else:
        image_path = os.path.join(path, "dead_heart", image_loc)
        title = "dead_heart"

    if os.path.exists(image_path):
        image = plt.imread(image_path)
        plt.title(title)
        plt.imshow(image)
        plt.axis("off")
plt.tight_layout()
plt.show()



## === cell 11
images = [
    "/kaggle/input/paddy-disease-classification/train_images/hispa/106590.jpg",
    "/kaggle/input/paddy-disease-classification/train_images/tungro/109629.jpg",
    "/kaggle/input/paddy-disease-classification/train_images/bacterial_leaf_blight/109372.jpg",
    "/kaggle/input/paddy-disease-classification/train_images/downy_mildew/102350.jpg",
    "/kaggle/input/paddy-disease-classification/train_images/blast/110243.jpg",
    "/kaggle/input/paddy-disease-classification/train_images/bacterial_leaf_streak/101104.jpg",
    "/kaggle/input/paddy-disease-classification/train_images/normal/109760.jpg",
    "/kaggle/input/paddy-disease-classification/train_images/brown_spot/104675.jpg",
    "/kaggle/input/paddy-disease-classification/train_images/dead_heart/105159.jpg",
    "/kaggle/input/paddy-disease-classification/train_images/bacterial_panicle_blight/101351.jpg",
]
diseases = [
    "hispa",
    "tungro",
    "bacterial_leaf_blight",
    "downy_mildew",
    "blast",
    "bacterial_leaf_streak",
    "normal",
    "brown_spot",
    "dead_heart",
    "bacterial_panicle_blight",
]
titles = [d + " image" for d in diseases]

existing = [(p, t) for p, t in zip(images, titles) if os.path.exists(p)]
if len(existing) > 0:
    plt.figure(figsize=(20, 10))
    columns = 5
    for i, (image_loc, title) in enumerate(existing):
        plt.subplot((len(existing) // columns) + 1, columns, i + 1)
        image = plt.imread(image_loc)
        plt.title(title)
        plt.imshow(image)
        plt.axis("off")
    plt.tight_layout()
    plt.show()
else:
    print("Example image paths not found; skipping example visualization.")



## === cell 12
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
data_enc = data.copy()
data_enc["label"] = encoder.fit_transform(data_enc["label"])
data_enc["variety"] = encoder.fit_transform(data_enc["variety"])
data_enc.head()



## === cell 13
batch_size = 32
img_height = 224
img_width = 224



## === cell 14
train_ds = tf.keras.utils.image_dataset_from_directory(
    directory=path,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
)



## === cell 15
val_ds = tf.keras.utils.image_dataset_from_directory(
    directory=path,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
)



## === cell 16
class_names = train_ds.class_names
print(class_names)



## === cell 17
for image_batch, label_batch in train_ds:
    print(image_batch.shape)
    print(label_batch.shape)
    break



## === cell 18
normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)



## === cell 19
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, label_batch = next(iter(normalized_ds))
first_image = image_batch[0]
print(np.min(first_image), np.max(first_image))



## === cell 20
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)



## === cell 21
num_classes = len(class_names)

model = tf.keras.Sequential(
    [
        tf.keras.layers.Rescaling(1.0 / 255),
        tf.keras.layers.Conv2D(32, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(128, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(256, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.15),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ]
)



## === cell 22
model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=["accuracy"],
)



## === cell 23
early_stopping = EarlyStopping(patience=20, restore_best_weights=True)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=100,
    callbacks=[early_stopping],
    verbose=1,
)

val_loss, val_acc = model.evaluate(val_ds, verbose=1)
print("Validation loss:", val_loss)
print("Validation accuracy:", val_acc)

plt.figure(figsize=(10, 4))
plt.plot(history.history["loss"], label="Train")
plt.plot(history.history["val_loss"], label="Validation")
plt.title("Model loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(loc="upper right")
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(history.history["accuracy"], label="Train")
plt.plot(history.history["val_accuracy"], label="Validation")
plt.title("Model accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(loc="lower left")
plt.show()



## === cell 24
model.summary()



## === cell 25
loss, accu = model.evaluate(val_ds, verbose=0)
print(f"the Testing loss is {loss:.4f}")
print(f"The testing accuracy is {accu*100:.2f}%")



## === cell 26
test_data_dir = "/kaggle/input/paddy-disease-classification/test_images/"



## === cell 27
tmp_test_root = "/kaggle/working/_tmp_test_images"
tmp_test_class_dir = os.path.join(tmp_test_root, "test")

if os.path.exists(tmp_test_root):
    shutil.rmtree(tmp_test_root)
os.makedirs(tmp_test_class_dir, exist_ok=True)

test_image_files = sorted(
    [f for f in os.listdir(test_data_dir) if f.lower().endswith(".jpg")]
)

for fname in test_image_files:
    src = os.path.join(test_data_dir, fname)
    dst = os.path.join(tmp_test_class_dir, fname)
    try:
        os.symlink(src, dst)
    except Exception:
        shutil.copy2(src, dst)

test_ds = tf.keras.utils.image_dataset_from_directory(
    tmp_test_root,
    labels=None,
    label_mode=None,
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    shuffle=False,
)

test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)



## === cell 28
y_pred = model.predict(test_ds, batch_size=batch_size, verbose=1)
y_pred.shape



## === cell 29
y_pred_classes = y_pred.argmax(axis=1)
y_pred_classes.shape



## === cell 30
y_classes_names = [class_names[x] for x in y_pred_classes]



## === cell 31
sub = pd.read_csv("/kaggle/input/paddy-disease-classification/sample_submission.csv")

pred_df = pd.DataFrame({"image_id": test_image_files, "label": y_classes_names})
pred_map = dict(zip(pred_df["image_id"], pred_df["label"]))

sub["label"] = sub["image_id"].map(pred_map)

sub["label"] = sub["label"].fillna(class_names[0])

sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())
