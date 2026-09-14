# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.95717

# 6. Current score

0.05556

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0991) has done: 'Diagnosis: The crash happens in cell 1 during `import tensorflow as tf`, but the root cause is an incompatibility between `protobuf==6.33.0` and TensorFlow 2.18.0 that triggers an internal protobuf `MessageFactory.GetPrototype` AttributeError at import time. This is a known failure mode when TensorFlow’s generated protos expect older protobuf APIs. The minimal, deterministic workaround in-notebook is to force TensorFlow to use the pure-Python protobuf implementation (instead of the C++ one) before importing TensorFlow, which avoids the missing attribute path. This change is localized to cell 1 and preserves the rest of the notebook’s logic and interfaces.

Patch summary: In cell 1, set `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` (and its version) via `os.environ` before importing TensorFlow, then keep the rest of the imports unchanged. This prevents the TensorFlow import crash without altering any model/training logic.

Updated cells: (cell 1 only)

Compatibility notes for cell k+1: `tf`, `pd`, `np`, `scipy`, and `plt` are still imported with the same names, so cell 2 can run unchanged and see the same variables as expected.

Assumptions: The environment allows setting `os.environ` at runtime before the first TensorFlow import in the process (true for typical notebook execution), and no earlier cell imported TensorFlow successfully (cell 0 does not).'
- What this solution (achieved 0.07057) has done: 'Diagnosis: The crash happens during TensorFlow import in cell 1 due to an incompatibility between the installed `protobuf` (6.33.0) and TensorFlow’s generated protobuf bindings, producing `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'`. The two environment variables being set try to force the pure-Python protobuf runtime, but with protobuf 6 this still breaks because TensorFlow expects the older API. The minimal fix is to force protobuf to use the upb C++ implementation (not the pure-Python one) and to ensure this is set before importing TensorFlow.

Patch summary: In cell 1 only, change `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION` from `"python"` to `"upb"` (and keep the “set before TF import” ordering) to restore compatibility between TensorFlow 2.18 and protobuf 6 in this environment.

Updated cells: cell 1 only (below).

Compatibility notes for cell k+1: No variable names or interfaces change; `tf`, `pd`, `np`, `scipy`, and `plt` are still imported exactly as before, so cell 2 can run unchanged.

Assumptions: The runtime includes the protobuf upb backend (standard with protobuf>=4/5/6 wheels) and TensorFlow 2.18 is compatible with it in this environment.'
- What this solution (achieved 0.10811) has done: 'Diagnosis: The crash happens during TensorFlow import in cell 1 and is triggered by an incompatibility between TensorFlow (and/or its bundled/proto-dependent components) and the installed `protobuf==6.33.0`. Setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=upb` forces the C++/upb backend, which in this environment raises `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'` due to API changes in protobuf 6. The simplest deterministic fix is to force the pure-Python protobuf backend (`python`), which avoids the failing code path and lets TensorFlow import cleanly.

Patch summary: In cell 1 only, change the protobuf implementation env var from `"upb"` to `"python"` (set before importing TensorFlow). Keep everything else intact so downstream code (cell 2+) continues to work unchanged.

Updated cells: Only cell 1 is modified below.

Compatibility notes for cell k+1: Cell 2 expects `tf`, `pd`, `np`, `scipy`, and `plt` to be imported and available; this patch preserves those names and their interfaces, only changing the protobuf backend used during import.

Assumptions: This environment allows using the pure-Python protobuf implementation (slower but compatible) and does not require the upb backend for performance; no other protobuf-related configuration is required.'
- What this solution (achieved 0.14565) has done: 'I remove the in-notebook protobuf downgrade/restart logic that can prevent a successful end-to-end run and instead apply the minimal environment setting that avoids the TensorFlow/protobuf import crash in this environment. I also fix two runtime NameErrors (`base_model_inceptionv3` references) that currently stop execution before training and submission creation. Finally, I align the submission mapping to the generator’s actual class index order (instead of a hard-coded species list) so predicted indices always map to the correct species labels, which should materially increase the Kaggle micro-F1 toward your target while keeping the same model/training core logic.'
- What this solution (achieved 0.04204) has done: 'Diagnosis: The crash happens while importing TensorFlow in cell 1, and the traceback `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'` indicates an incompatibility between `tensorflow==2.18.0` and `protobuf==6.33.0` in this environment. The two environment variables set in cell 1 (`PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` and version `2`) do not resolve this specific API mismatch. TensorFlow provides a supported workaround: forcing the legacy Python protobuf implementation, which avoids the missing `GetPrototype` call in newer protobuf builds.

Patch summary: Modify only cell 1 to set `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` and `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION=3` (legacy python implementation behavior) before importing TensorFlow. Keep all imports and downstream variable names unchanged so later cells run identically.

Updated cells: Only cell 1 is changed below.

Compatibility notes for cell k+1: Cell 2 expects `tf`, `pd`, `np`, `scipy`, and `plt` to be imported and available; this patch preserves those names and does not alter TensorFlow/Keras APIs used by `ImageDataGenerator` and `flow_from_directory`.

Assumptions: This environment allows using the Python protobuf implementation (it is available since `protobuf` is installed), and no other code relies on the previously-set protobuf version "2" specifically.'
- What this solution (achieved 0.09309) has done: 'Diagnosis: The crash happens during `import tensorflow as tf` in cell 1, caused by an incompatibility between `protobuf==6.33.0` and TensorFlow 2.18 that triggers `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'`. The environment variables forcing the pure-Python protobuf implementation don’t resolve this incompatibility in this environment. The minimal deterministic fix is to pin protobuf to a TensorFlow-compatible 4.x release at runtime (before importing TensorFlow), then import TensorFlow normally.

Patch summary: In cell 1, install a compatible protobuf version (`<5`, i.e., 4.25.x) via pip before importing TensorFlow, and remove reliance on the `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION` workaround. This keeps the rest of the notebook unchanged and unblocks TensorFlow usage.

Updated cells: cell 1 only.

Compatibility notes for cell k+1: This preserves the `tf`, `pd`, `np`, `scipy`, and `plt` symbols exactly as expected by cell 2; no interface/shape changes.

Assumptions: pip installs are allowed in this runtime and the environment has access to the prebuilt wheels for `protobuf<5` (standard on Kaggle-like images). If pip cannot run, the only alternative would be changing the base environment, which is out of scope.'
- What this solution (achieved 0.06607) has done: 'Diagnosis: The training generator is producing one-hot labels with 13 classes (`target.shape=(None, 13)`), but the model’s final Dense layer is hard-coded to 12 outputs (`output.shape=(None, 12)`), causing `categorical_crossentropy` to fail due to shape mismatch. This happens when the dataset directory contains 13 class folders (as in Plant Seedlings). The fix is to make the final layer’s unit count match the number of classes inferred by `train_generator.num_classes`, without changing the model architecture otherwise.

Patch summary: In cell 8, replace the hard-coded `Dense(12, ...)` with `Dense(train_generator.num_classes, ...)` so the model output shape matches the generator labels. No other logic is altered.

Updated cells: Only cell 8 is changed.

Compatibility notes for cell k+1: The model object name (`model_vgg16`) and compilation/training code remain unchanged; only the output dimension is corrected, so cells 9–13 continue to work as written.

Assumptions: `train_generator` has already been created successfully in cell 2 and exposes `num_classes` consistent with `class_mode="categorical"`.'
- What this solution (achieved 0.07658) has done: 'Diagnosis: Cell 13 crashes because it references `history_vgg16`, but no prior cell creates it (there is no `model_vgg16.fit(...)` call captured in the provided cells). As a result, `history_vgg16` is undefined when trying to read `history_vgg16.history[...]`. To keep the notebook runnable without changing training/model logic elsewhere, cell 13 should defensively fall back to using an existing Keras `History` object if present, or skip plotting data creation when none exists.

Patch summary: Update cell 13 to (1) look for a `History` object among common variable names (`history_vgg16`, `history`, `hist`) or, as a last resort, scan globals for a `tf.keras.callbacks.History` instance; (2) only compute `accuracy/val_accuracy/loss/val_loss/epochs` if a history is found, otherwise set them to empty lists so downstream plotting cells (cell 14) do not crash.

Updated cells: Only cell 13 is changed.

Compatibility notes for cell k+1: Cell 14 expects `epochs`, `loss`, and `val_loss` to exist. This patch always defines them; when training history is missing they are empty lists, so cell 14 render an empty plot instead of raising `NameError`.

Assumptions: No earlier cell (outside the provided snippet) reliably defines `history_vgg16`; if it does, this patch use it unchanged. TensorFlow’s `tf.keras.callbacks.History` class is available (it is in TF 2.18).'
- What this solution (achieved 0.07057) has done: 'Diagnosis: Cell 13 iterates directly over `globals().values()` while also performing operations that may mutate the globals dictionary (directly or indirectly via TensorFlow objects), which can trigger `RuntimeError: dictionary changed size during iteration`. The failing loop is only used as a fallback to locate a `tf.keras.callbacks.History` instance, so it can safely iterate over a snapshot copy of the values.  

Patch summary: In cell 13, change the fallback loop to iterate over `list(globals().values())` instead of the live view `globals().values()`, preventing mutation during iteration. No other logic, outputs, or variable names are changed.  

Updated cells: Only cell 13 is updated.  

Compatibility notes for cell k+1: This preserves the creation of `accuracy`, `val_accuracy`, `loss`, `val_loss`, and `epochs`, which are used in cell 14 exactly as before.  

Assumptions: At most one `tf.keras.callbacks.History` object may exist in globals; if none exist, the code should keep returning empty metric lists as originally intended.'
- What this solution (achieved 0.05556) has done: 'Diagnosis: The crash happens because `target_names=label` has 13 entries while `y_test/y_pred` contain 12 class ids, so `sklearn.metrics.classification_report` rejects the mismatch. This mismatch is due to `label` being built from `train_generator.class_indices` order, which can include an extra entry relative to the classes present in `val_generator.classes`. The fix is to pass an explicit `labels` list matching the actual number of classes in `y_test/y_pred`, and to slice `target_names` to the same length so sklearn’s validation passes.

Patch summary: In cell 16 only, compute the label ids present (0..num_classes-1) and derive matching `target_names` of the same length, then call `classification_report` with both `labels` and `target_names`. This preserves evaluation semantics while preventing the error.

Updated cells: cell 16 only.

Compatibility notes for cell k+1: This change does not modify `label`, `y_test`, or `y_pred`, so cell 17 continues to run as before (it still uses `class_labels = label`). The confusion matrix visualization may still have a tick-label length mismatch if `label` truly has 13 while the matrix is 12x12, but that is outside the failing cell and not changed here.

Assumptions: Classes are encoded as contiguous integer ids starting at 0 in `val_generator.classes`, matching Keras DirectoryIterator behavior.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
import os

import sys
import subprocess

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "protobuf<5"])

import tensorflow as tf
import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt


## === cell 2
train_dir = "/kaggle/input/plant-seedlings-classification/train"
test_dir = "/kaggle/input/plant-seedlings-classification/"
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
)

val_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    shuffle=False,
    subset="validation",
    class_mode="categorical",
)

test_generator = test_datagen.flow_from_directory(
    directory=test_dir,
    classes=["test"],
    target_size=(img_size, img_size),
    batch_size=1,
    shuffle=False,
    class_mode="categorical",
)



## === cell 3
label = [k for k in train_generator.class_indices]
samples = train_generator.__next__()
images = samples[0]
titles = samples[1]
plt.figure(figsize=(20, 20))

for i in range(20):
    plt.subplot(5, 5, i + 1)
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    plt.imshow(images[i])
    plt.title(f"Class: {label[np.argmax(titles[i], axis=0)]}")
    plt.axis("off")



## === cell 4
base_model_vgg16 = tf.keras.applications.VGG16(
    include_top=False, weights="imagenet", input_shape=(img_size, img_size, 3)
)



## === cell 5
for layer in base_model_vgg16.layers:
    print(f"Layer Name: {layer.name}, Trainable: {layer.trainable}")

len(base_model_vgg16.layers)



## === cell 6
for layer in base_model_vgg16.layers[:5]:
    layer.trainable = False



## === cell 7
for layer in base_model_vgg16.layers:
    print(f"Layer Name: {layer.name}, Trainable: {layer.trainable}")

len(base_model_vgg16.layers)



## === cell 8
model_vgg16 = tf.keras.models.Sequential(
    layers=[
        base_model_vgg16,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(12, activation="softmax"),
    ]
)



## === cell 9
model_vgg16.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)



## === cell 10
model_name = "model_vgg16.h5"
Checkpoint = tf.keras.callbacks.ModelCheckpoint(
    model_name, monitor="val_loss", mode="min", save_best_only=True, verbose=1
)
es = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True
)
lrr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", patience=3, verbose=1, factor=0.3, min_lr=0.00000001
)
cb_List = [Checkpoint, es, lrr]



## === cell 11
model_vgg16.summary()



## === cell 12
model_vgg16 = tf.keras.models.Sequential(
    layers=[
        base_model_vgg16,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(train_generator.num_classes, activation="softmax"),
    ]
)


## === cell 13
history_obj = None

for _name in ("history_vgg16", "history", "hist"):
    if _name in globals():
        _val = globals().get(_name)
        if hasattr(_val, "history") and isinstance(getattr(_val, "history"), dict):
            history_obj = _val
            break

if history_obj is None:
    for _val in list(globals().values()):
        if isinstance(_val, tf.keras.callbacks.History):
            history_obj = _val
            break

if history_obj is None:
    accuracy, val_accuracy, loss, val_loss = [], [], [], []
    epochs = []
else:
    accuracy = history_obj.history.get("accuracy", [])
    val_accuracy = history_obj.history.get("val_accuracy", [])
    loss = history_obj.history.get("loss", [])
    val_loss = history_obj.history.get("val_loss", [])

    num_epochs = len(accuracy)
    epochs = list(range(1, num_epochs + 1))

    plt.figure(figsize=(20, 8))
    plt.plot(epochs, accuracy, label="Training Accuracy", color="blue")
    plt.plot(epochs, val_accuracy, label="Validation Accuracy", color="green")
    plt.xlabel("Epochs")
    plt.ylabel("Value")
    plt.title("Training and Validation Metrics")
    plt.legend()
    plt.show()


## === cell 14
plt.figure(figsize=(20, 8))
plt.plot(epochs, loss, label="Training Loss", color="red")
plt.plot(epochs, val_loss, label="Validation Loss", color="purple")
plt.xlabel("Epochs")
plt.ylabel("Value")
plt.title("Training and Validation Metrics")
plt.legend()
plt.show()



## === cell 15
y_test = val_generator.classes
y_pred = model_vgg16.predict(val_generator, batch_size=32)
y_pred = np.argmax(y_pred, axis=1)



## === cell 16
from sklearn.metrics import classification_report, confusion_matrix

n_classes = (
    int(max(np.max(y_test), np.max(y_pred)) + 1)
    if len(y_test) and len(y_pred)
    else len(label)
)
labels_ids = list(range(n_classes))
target_names = label[:n_classes]

print(
    classification_report(y_test, y_pred, labels=labels_ids, target_names=target_names)
)


## === cell 17
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

conf_matrix = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(conf_matrix)

import seaborn as sns
import matplotlib.pyplot as plt

class_labels = label

plt.figure(figsize=(10, 8))
sns.heatmap(
    conf_matrix,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_labels,
    yticklabels=class_labels,
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()



## === cell 18
idx_to_class = {v: k for k, v in train_generator.class_indices.items()}

preds = model_vgg16.predict(test_generator, steps=test_generator.samples)
pred_idx = np.argmax(preds, axis=1)
class_list = [idx_to_class[i] for i in pred_idx]

submission = pd.DataFrame()
submission["file"] = test_generator.filenames
submission["file"] = submission["file"].str.replace(r"^test/", "", regex=True)
submission["species"] = class_list



## === cell 19
submission.head(5)



## === cell 20
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
