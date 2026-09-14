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

# 5. Target score

0.91813

# 6. Current score

0.13514

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.06156) has done: 'The changes introduce mixed‑precision training, increase the batch size, and enable parallel data loading with multiple workers during `model.fit`. These adjustments keep the architecture, loss, callbacks, and augmentation unchanged while greatly reducing GPU and CPU bottlenecks, allowing the full training to complete well within the 600‑second limit.'
- What this solution (achieved 0.0991) has done: 'Implemented fixes to resolve runtime errors and align evaluation handling, while keeping the original model logic intact.  
- Wrapped mixed‑precision setup in a safe try/except to avoid protobuf incompatibility.  
- Removed unsupported `workers` and `use_multiprocessing` arguments from `model.fit`.  
- Adjusted epoch count for faster execution within limits.  
- Fixed class‑name handling for the classification report and ensured the label list matches the number of classes.  
- Cleaned up plotting code to use the correct history object.  
- The script now runs end‑to‑end and writes a proper `submission.csv` file.'
- What this solution (achieved 0.0991) has done: 'The fix removes the mixed‑precision call that crashes on import, builds a reliable `class_names` list that exactly matches the number of classes for the validation report, and adds a small safety check when creating the submission. These changes eliminate the runtime errors and ensure the evaluation and submission formats are correct, allowing the model to train and produce a valid `submission.csv`. The core model and training logic remain unchanged.'
- What this solution (achieved 0.05405) has done: 'Implemented fixes to ensure the script runs end‑to‑end and produces a valid submission, while making modest adjustments to improve model performance:

* Re‑implemented `class_names` construction so its length matches the number of classes, removing the classification‑report error.
* Increased training epochs to 25 (still within the runtime limit) to allow the model to learn better and boost the F1‑score.
* Added a small comment clarifying the purpose of each change.'
- What this solution (achieved 0.06306) has done: 'I fixed the runtime errors by removing the failing mixed‑precision call, correctly captured the class names and test file paths before the datasets are transformed, and updated all later references to use these stored variables. This ensures the model can be built, trained, evaluated, and a proper `submission.csv` is written without attribute errors.'
- What this solution (achieved 0.06306) has done: 'I fixed the class‑name mismatch by explicitly enumerating the sub‑folders in the training directory (ignoring hidden files) so the list has exactly 12 species. I added a lightweight data‑augmentation block to the model to improve generalisation, and increased the number of training epochs (initial 12 + fine‑tune 6) to let the network learn better while staying within the runtime limit. These changes resolve the runtime error, produce a correct submission CSV and are expected to raise the micro‑F1 score toward the target.'
- What this solution (achieved 0.11261) has done: 'The update fixes the class‑name length mismatch by using the dataset‑provided `class_names`, freezes the ResNet‑50 base model for the first training phase (then fine‑tunes it), and slightly increases the epoch counts to let the network learn better while still respecting the runtime limit. All changes keep the original architecture and callbacks, and the script now writes a correct `submission.csv` file.'
- What this solution (achieved 0.0991) has done: 'I fix the protobuf import error, ensure the class list matches the actual 12 species (removing any hidden folders) and pass this list to the dataset loaders so the label ordering is consistent. I also slightly increase the training epochs to give the model more learning capacity, which should raise the micro‑F1 score toward the target while keeping the core architecture unchanged. The script now run end‑to‑end and write a proper `submission.csv` file.'
- What this solution (achieved 0.12012) has done: 'The fix removes the problematic protobuf attribute hack, correctly derives the list of class names directly from the TensorFlow dataset (ensuring the number of names matches the model’s output dimensions), and updates `num_classes` accordingly. This resolves the `classification_report` mismatch and lets the model’s predictions be mapped to the proper species, moving the micro‑F1 score toward the target while keeping the core architecture unchanged. The script now runs end‑to‑end and writes a valid `submission.csv`.'
- What this solution (achieved 0.06306) has done: 'The fix removes the fragile protobuf hack that caused an import‑time crash, builds a clean list of the 12 species folders (ignoring hidden files) and uses it for the TensorFlow datasets so the model’s output ordering matches the true class order. With the correct `class_names` length the `classification_report` runs without error, and the predictions are properly mapped, which dramatically improves the micro‑F1 score while keeping the original model architecture and training flow unchanged.'
- What this solution (achieved 0.13514) has done: 'Implemented a safe protobuf monkey‑patch to avoid the import crash and aligned the class list with TensorFlow’s generated `class_names` after building the dataset. This ensures the number of target names matches the model’s output dimension, fixing the `classification_report` error and resulting in a correct submission CSV. No core model logic was altered.'

# 9. Code solution

## === cell 0
import os

try:
    import google.protobuf.message_factory as _mf

    if not hasattr(_mf.MessageFactory, "GetPrototype"):
        _mf.MessageFactory.GetPrototype = _mf.MessageFactory.GetMessageClass
except Exception:
    pass

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_dir = "/kaggle/input/plant-seedlings-classification/train"
test_dir = "/kaggle/input/plant-seedlings-classification/test"

img_size = 224
batch_size = 64

class_names = sorted(
    [
        d
        for d in os.listdir(train_dir)
        if os.path.isdir(os.path.join(train_dir, d)) and not d.startswith(".")
    ]
)

train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    validation_split=0.25,
    subset="training",
    seed=42,
    image_size=(img_size, img_size),
    batch_size=batch_size,
    label_mode="categorical",
    class_names=class_names,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    validation_split=0.25,
    subset="validation",
    seed=42,
    image_size=(img_size, img_size),
    batch_size=batch_size,
    label_mode="categorical",
    class_names=class_names,
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    labels=None,
    image_size=(img_size, img_size),
    batch_size=1,
    shuffle=False,
)

test_file_paths = test_ds.file_paths

class_names = train_ds.class_names
num_classes = len(class_names)  # should be 12

normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)

train_ds = train_ds.map(
    lambda x, y: (normalization_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE
)
val_ds = val_ds.map(
    lambda x, y: (normalization_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE
)
test_ds = test_ds.map(
    lambda x: normalization_layer(x), num_parallel_calls=tf.data.AUTOTUNE
)

train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

train_generator = train_ds
val_generator = val_ds
test_generator = test_ds



## === cell 2
samples = next(iter(train_generator))
images, titles = samples[0], samples[1]
plt.figure(figsize=(20, 20))
for i in range(20):
    plt.subplot(5, 5, i + 1)
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    plt.imshow(images[i].numpy())
    plt.title(f"Class: {class_names[np.argmax(titles[i])]}")
    plt.axis("off")
plt.show()



## === cell 3
base_model_resnet50 = tf.keras.applications.ResNet50(
    include_top=False, weights="imagenet", input_shape=(img_size, img_size, 3)
)
base_model_resnet50.trainable = False



## === cell 4
augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
    ]
)

model_resnet50 = tf.keras.models.Sequential(
    [
        augmentation,
        base_model_resnet50,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ]
)



## === cell 5
model_resnet50.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)



## === cell 6
model_name = "model_resnet50.h5"
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



## === cell 7
EPOCH = 25  # longer training for better learning
history_resnet50 = model_resnet50.fit(
    train_generator,
    epochs=EPOCH,
    validation_data=val_generator,
    callbacks=cb_List,
)



## === cell 8
base_model_resnet50.trainable = True
model_resnet50.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

fine_epochs = 10
history_fine = model_resnet50.fit(
    train_generator,
    epochs=fine_epochs,
    validation_data=val_generator,
    callbacks=cb_List,
)



## === cell 9
acc = history_resnet50.history["accuracy"] + history_fine.history["accuracy"]
val_acc = (
    history_resnet50.history["val_accuracy"] + history_fine.history["val_accuracy"]
)
loss = history_resnet50.history["loss"] + history_fine.history["loss"]
val_loss = history_resnet50.history["val_loss"] + history_fine.history["val_loss"]

num_epochs = len(acc)
epochs = list(range(1, num_epochs + 1))

plt.figure(figsize=(20, 8))
plt.plot(epochs, acc, label="Training Accuracy", color="blue")
plt.plot(epochs, val_acc, label="Validation Accuracy", color="green")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.show()

plt.figure(figsize=(20, 8))
plt.plot(epochs, loss, label="Training Loss", color="red")
plt.plot(epochs, val_loss, label="Validation Loss", color="purple")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.show()



## === cell 10
y_test_onehot = np.concatenate([y.numpy() for _, y in val_generator], axis=0)
y_test = np.argmax(y_test_onehot, axis=1)

y_pred_prob = model_resnet50.predict(val_generator, batch_size=batch_size)
y_pred = np.argmax(y_pred_prob, axis=1)

print(classification_report(y_test, y_pred, target_names=class_names))



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/4088274066.py in <cell line: 0>()
      5 y_pred = np.argmax(y_pred_prob, axis=1)
      6 
----> 7 print(classification_report(y_test, y_pred, target_names=class_names))
      8 

/usr/local/lib/python3.11/dist-packages/sklearn/metrics/_classification.py in classification_report(y_true, y_pred, labels, target_names, sample_weight, digits, output_dict, zero_division)
   2330             )
   2331         else:
-> 2332             raise ValueError(
   2333                 "Number of classes, {0}, does not match size of "
   2334                 "target_names, {1}. Try specifying the labels "

ValueError: Number of classes, 12, does not match size of target_names, 13. Try specifying the labels parameter

## === cell 11
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

plt.figure(figsize=(10, 8))
sns.heatmap(
    conf_matrix,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names,
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()



## === cell 12
species_list = class_names
preds = model_resnet50.predict(
    test_generator, steps=test_generator.cardinality().numpy()
)
class_list = [species_list[np.argmax(p)] for p in preds]

submission = pd.DataFrame()
submission["file"] = test_file_paths
submission["file"] = submission["file"].str.replace(r"^.*/test/", "", regex=True)
submission["species"] = class_list



## === cell 13
print(submission.head())



## === cell 14
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
