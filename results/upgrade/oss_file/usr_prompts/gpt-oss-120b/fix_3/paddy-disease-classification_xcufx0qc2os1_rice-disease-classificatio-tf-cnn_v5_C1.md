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

0.12644

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.12567) has done: 'I remove the IPython magic that caused the import error, protect the optional visualisation cell that fails on missing files, correct the validation image size, fix the loss‑function mismatch (softmax + `from_logits=True`), and keep the rest of the pipeline unchanged. These changes eliminate runtime failures and let the model train correctly, which should raise the validation accuracy toward the target score.'
- What this solution (achieved 0.12644) has done: 'I fix the protobuf import error by setting the required environment variable before importing TensorFlow, add a small data‑augmentation block to the model to boost validation accuracy, and give the early‑stopping callback a larger patience so the model can train longer. These changes resolve the runtime crash and modestly improve the score while preserving the original architecture and workflow.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import tensorflow as tf
from keras.callbacks import EarlyStopping
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd




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
plt.show()




## === cell 7
fig, ax = plt.subplots(1, 1, figsize=(21, 7))
sns.histplot(x="label", data=data, ax=ax)
plt.title("Disease distribution in the dataset")
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
try:
    plt.figure(figsize=(20, 10))
    columns = 5
    path = "/kaggle/input/paddy-disease-classification/train_images/"
    for i, image_loc in enumerate(np.concatenate((five_normals, five_deads))):
        plt.subplot(10 // columns + 1, columns, i + 1)
        if i < 5:
            image = plt.imread(path + "normal/" + image_loc)
            plt.title("normal")
        else:
            image = plt.imread(path + "dead_heart/" + image_loc)
            plt.title("dead_heart")
        plt.imshow(image)
except Exception as e:
    print(f"Visualization skipped due to: {e}")




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

diseases = [disease + " image" for disease in diseases]
plt.figure(figsize=(20, 10))
columns = 5
for i, image_loc in enumerate(images):
    plt.subplot(len(images) // columns + 1, columns, i + 1)
    try:
        image = plt.imread(image_loc)
        plt.title(diseases[i])
        plt.imshow(image)
    except FileNotFoundError:
        plt.title(f"Missing: {diseases[i]}")
        plt.axis("off")
plt.show()




## === cell 12
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
data["label"] = encoder.fit_transform(data["label"])
data["variety"] = encoder.fit_transform(data["variety"])
data.head()




## === cell 13
batch_size = 32
img_height = 224
img_width = 224




## === cell 14
train_ds = tf.keras.utils.image_dataset_from_directory(
    directory="/kaggle/input/paddy-disease-classification/train_images/",
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
)




## === cell 15
val_ds = tf.keras.utils.image_dataset_from_directory(
    directory="/kaggle/input/paddy-disease-classification/train_images/",
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
for image_batch, label_batch in train_ds.take(1):
    print(image_batch.shape)
    print(label_batch.shape)




## === cell 18
normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)




## === cell 19
normalized_train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
normalized_val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))




## === cell 20
AUTOTUNE = tf.data.AUTOTUNE
train_ds = normalized_train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = normalized_val_ds.cache().prefetch(buffer_size=AUTOTUNE)




## === cell 21
num_classes = len(class_names)

model = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.Conv2D(
            32, 3, activation="relu", input_shape=(img_height, img_width, 3)
        ),
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
    train_ds, validation_data=val_ds, epochs=50, callbacks=[early_stopping], verbose=2
)




## === cell 24
loss, accu = model.evaluate(val_ds, verbose=0)
print(f"The validation loss is {loss:.4f}")
print(f"The validation accuracy is {accu*100:.2f}%")




## === cell 25
test_data_dir = "/kaggle/input/paddy-disease-classification/test_images/"




## === cell 26
test_ds = tf.keras.utils.image_dataset_from_directory(
    test_data_dir,
    label_mode=None,
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    shuffle=False,
)

test_ds = test_ds.map(lambda x: normalization_layer(x))
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)




## === cell 27
y_pred = model.predict(test_ds, verbose=1)
y_pred_classes = y_pred.argmax(axis=1)




## === cell 28
y_class_names = [class_names[idx] for idx in y_pred_classes]




## === cell 29
submission = pd.read_csv(
    "/kaggle/input/paddy-disease-classification/sample_submission.csv"
)
submission["label"] = y_class_names
submission.to_csv("submission.csv", index=False)
