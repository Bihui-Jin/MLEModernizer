# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Classify plant seedlings into their respective species.

## Metric
Micro-averaged F1-score.

## Submission Format
For each `file` in the test set, you must predict a probability for the `species` variable. The file should contain a header and have the following format:

```
file,species
0021e90e4.png,Maize
003d61042.png,Sugar beet
007b3da8b.png,Common wheat
etc.
```

## Dataset
The list of species is as follows:

```
Black-grass
Charlock
Cleavers
Common Chickweed
Common wheat
Fat Hen
Loose Silky-bent
Maize
Scentless Mayweed
Shepherds Purse
Small-flowered Cranesbill
Sugar beet
```

- **train.csv** - the training set, with plant species organized by folder
- **test.csv** - the test set, you need to predict the species of each image
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.11

# 3. Installed packages

geopandas==0.14.4
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
scipy==1.15.3
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
            description.md (84 lines)
            sample_submission.csv (667 lines)
            sample_submission.csv.zip (4.6 kB)
            test.zip (259.0 MB)
            train.zip (1.5 GB)
            plant-seedlings-classification/
                description.md (84 lines)
                sample_submission.csv (667 lines)
                ... and 3 other files
                plant-seedlings-classification/
                test/
                    5db43df54.png (177.5 kB)
                    09d34fe5b.png (156.0 kB)
                    ... and 664 other files
                    test/
                train/
                    Black-grass/
                        2ed589264.png (44.6 kB)
                        840a7ed59.png (708.1 kB)
                        ... and 219 other files
                    Charlock/
                        ee4a02bf9.png (229.3 kB)
                        e795c53c9.png (354.4 kB)
                        ... and 322 other files
                    ... and 11 other folders
            test/
                5db43df54.png (177.5 kB)
                09d34fe5b.png (156.0 kB)
                ... and 664 other files
                test/
            train/
                Black-grass/
                    2ed589264.png (44.6 kB)
                    840a7ed59.png (708.1 kB)
                    ... and 219 other files
                Charlock/
                    ee4a02bf9.png (229.3 kB)
                    e795c53c9.png (354.4 kB)
                    ... and 322 other files
                ... and 11 other folders
        input/
            description.md (84 lines)
            sample_submission.csv (667 lines)
            sample_submission.csv.zip (4.6 kB)
            test.zip (259.0 MB)
            train.zip (1.5 GB)
            plant-seedlings-classification/
                description.md (84 lines)
                sample_submission.csv (667 lines)
                ... and 3 other files
                plant-seedlings-classification/
                test/
                    5db43df54.png (177.5 kB)
                    09d34fe5b.png (156.0 kB)
                    ... and 664 other files
                    test/
                train/
                    Black-grass/
                        2ed589264.png (44.6 kB)
                        840a7ed59.png (708.1 kB)
                        ... and 219 other files
                    Charlock/
                        ee4a02bf9.png (229.3 kB)
                        e795c53c9.png (354.4 kB)
                        ... and 322 other files
                    ... and 11 other folders
            test/
                5db43df54.png (177.5 kB)
                09d34fe5b.png (156.0 kB)
                ... and 664 other files
                test/
                    5db43df54.png (177.5 kB)
                    09d34fe5b.png (156.0 kB)
                    ... and 664 other files
                    test/
            train/
                Black-grass/
                    2ed589264.png (44.6 kB)
                    840a7ed59.png (708.1 kB)
                    ... and 219 other files
                Charlock/
                    ee4a02bf9.png (229.3 kB)
                    e795c53c9.png (354.4 kB)
                    ... and 322 other files
                ... and 11 other folders
        working/
            plant-seedlings-classification/
                description.md (84 lines)
                sample_submission.csv (667 lines)
                ... and 3 other files
                plant-seedlings-classification/
                test/
                    5db43df54.png (177.5 kB)
                    09d34fe5b.png (156.0 kB)
                    ... and 664 other files
                    test/
                train/
                    Black-grass/
                        2ed589264.png (44.6 kB)
                        840a7ed59.png (708.1 kB)
                        ... and 219 other files
                    Charlock/
                        ee4a02bf9.png (229.3 kB)
                        e795c53c9.png (354.4 kB)
                        ... and 322 other files
                    ... and 11 other folders
```

-> data/plant-seedlings-classification/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

-> data/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

-> input/plant-seedlings-classification/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

-> input/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

-> working/plant-seedlings-classification/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

# 5. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)




## === cell 1
species_list = [
    "Black-grass",
    "Charlock",
    "Cleavers",
    "Common Chickweed",
    "Common wheat",
    "Fat Hen",
    "Loose Silky-bent",
    "Maize",
    "Scentless Mayweed",
    "Shepherds Purse",
    "Small-flowered Cranesbill",
    "Sugar beet",
]

train_dir = "/kaggle/input/plant-seedlings-classification/train"
test_dir = "/kaggle/input/plant-seedlings-classification"
img_size = 224
batch_size = 32

datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1 / 255,
    rotation_range=30,
    brightness_range=[0.5, 1.2],
    horizontal_flip=True,
    validation_split=0.25,
    zoom_range=0.2,
)

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0 / 255, data_format="channels_last"
)

train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    shuffle=True,
    subset="training",
    class_mode="categorical",
    classes=species_list,
    seed=seed,
)

val_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    shuffle=False,
    subset="validation",
    class_mode="categorical",
    classes=species_list,
    seed=seed,
)

test_generator = test_datagen.flow_from_directory(
    directory=test_dir,
    classes=["test"],
    target_size=(img_size, img_size),
    batch_size=batch_size,
    shuffle=False,
    class_mode="categorical",
)

label = list(train_generator.class_indices.keys())




## === cell 2
samples = train_generator.__next__()
images, titles = samples[0], samples[1]
plt.figure(figsize=(20, 20))
for i in range(20):
    plt.subplot(5, 5, i + 1)
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    plt.imshow(images[i])
    plt.title(f"Class: {label[np.argmax(titles[i])]}")
    plt.axis("off")




## === cell 3
base_model_vgg16 = tf.keras.applications.VGG16(
    include_top=False, weights="imagenet", input_shape=(img_size, img_size, 3)
)




## === cell 4
for layer in base_model_vgg16.layers:
    print(f"Layer Name: {layer.name}, Trainable: {layer.trainable}")
len(base_model_vgg16.layers)




## === cell 5
for layer in base_model_vgg16.layers[:5]:
    layer.trainable = False




## === cell 6
for layer in base_model_vgg16.layers:
    print(f"Layer Name: {layer.name}, Trainable: {layer.trainable}")
len(base_model_vgg16.layers)




## === cell 7
model_vgg16 = tf.keras.models.Sequential(
    [
        base_model_vgg16,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(12, activation="softmax"),
    ]
)




## === cell 8
model_vgg16.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)




## === cell 9
model_name = "model_vgg16.h5"
Checkpoint = tf.keras.callbacks.ModelCheckpoint(
    model_name, monitor="val_loss", mode="min", save_best_only=True, verbose=1
)
es = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True
)
lrr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", patience=3, verbose=1, factor=0.3, min_lr=1e-8
)
cb_List = [Checkpoint, es, lrr]




## === cell 10
model_vgg16.summary()




## === cell 11
EPOCH = 30
history_vgg16 = model_vgg16.fit(
    train_generator,
    epochs=EPOCH,
    validation_data=val_generator,
    callbacks=cb_List,
    verbose=1,
)

base_model_vgg16.trainable = True
model_vgg16.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

fine_history = model_vgg16.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator,
    callbacks=cb_List,
    verbose=1,
)




## === cell 12
accuracy = history_vgg16.history["accuracy"] + fine_history.history["accuracy"]
val_accuracy = (
    history_vgg16.history["val_accuracy"] + fine_history.history["val_accuracy"]
)
loss = history_vgg16.history["loss"] + fine_history.history["loss"]
val_loss = history_vgg16.history["val_loss"] + fine_history.history["val_loss"]

num_epochs = len(accuracy)
epochs = list(range(1, num_epochs + 1))

plt.figure(figsize=(20, 8))
plt.plot(epochs, accuracy, label="Training Accuracy", color="blue")
plt.plot(epochs, val_accuracy, label="Validation Accuracy", color="green")
plt.xlabel("Epochs")
plt.ylabel("Value")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.show()




## === cell 13
plt.figure(figsize=(20, 8))
plt.plot(epochs, loss, label="Training Loss", color="red")
plt.plot(epochs, val_loss, label="Validation Loss", color="purple")
plt.xlabel("Epochs")
plt.ylabel("Value")
plt.title("Training and Validation Loss")
plt.legend()
plt.show()




## === cell 14
y_test = val_generator.classes
y_pred = model_vgg16.predict(val_generator, batch_size=batch_size)
y_pred = np.argmax(y_pred, axis=1)

from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_test, y_pred, target_names=label))




## === cell 15
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

import seaborn as sns

plt.figure(figsize=(10, 8))
sns.heatmap(
    conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=label, yticklabels=label
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()




## === cell 16
preds = model_vgg16.predict(
    test_generator, steps=test_generator.samples // batch_size + 1
)
class_list = [species_list[np.argmax(p)] for p in preds]

submission = pd.DataFrame({"file": test_generator.filenames, "species": class_list})
submission["file"] = submission["file"].str.replace(r"^test/", "", regex=True)




## === cell 17
submission.head(5)




## === cell 18
submission.to_csv("submission.csv", index=False)
