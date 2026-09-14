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

3.7

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

0.92191

# 6. Current score

0.60661

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.37087) has done: 'I fix the breakages caused by using legacy Keras APIs (e.g., `np_utils`, `ImageDataGenerator`, `fit_generator`, and the old optimizer constructor) by switching to their current equivalents in the installed Keras/TensorFlow stack. I also correct the dataset paths to match the actual Kaggle directory layout you provided (`.../plant-seedlings-classification/train` and `.../test`) so images are found reliably. To keep the core logic intact, the model remains VGG16 (no top) + Flatten + Dense(256) + Dense(12) with the same loss; I only adjust the final activation to `softmax` to match `categorical_crossentropy` and proper multiclass probability semantics (score-improving and metric-aligned). Finally, I ensure the submission is written as `submission.csv` with exactly the required `file,species` columns and in the same file order as `sample_submission.csv`.'
- What this solution (achieved 0.27177) has done: 'I fix the runtime crash caused by an incompatible `protobuf` version (the `MessageFactory.GetPrototype` error) by forcing TensorFlow/Keras to use the pure-Python protobuf implementation before importing TensorFlow. Then I correct the train/test directory resolution so it always points to the actual `.../plant-seedlings-classification/train` and `.../test` folders (your current fallback can silently pick a wrong base). Finally, to move score toward the target without changing the core model, I apply the standard VGG16 `preprocess_input` consistently for both train and test (instead of simple `/255`), which is a minimal, metric-aligned calibration change that typically yields a large accuracy/F1 improvement for ImageNet backbones; everything else (VGG16 frozen + Flatten + Dense(256) + Dense(12), same loss, same training loop) stays intact and a valid `submission.csv` is written in sample order.'
- What this solution (achieved 0.26877) has done: 'You’re crashing before any training because TensorFlow import fails due to an incompatibility between TF 2.18 and protobuf 6.x; setting the env vars isn’t taking effect early enough in this notebook-style flow. I make the protobuf fix robust by forcing the pure-Python protobuf implementation *before* any protobuf/TensorFlow-related import and by clearing any preloaded protobuf modules, which avoids the `MessageFactory.GetPrototype` error. Then I keep your exact model/training logic intact (VGG16 frozen + Flatten + Dense(256) + Dense(12), same loss/optimizer/training loop) and only adjust data loading to ensure RGB channel order matches VGG16 preprocessing (OpenCV loads BGR), which is a minimal, metric-aligned fix that should move your score upward toward the target. Finally, I ensure the submission is written as `submission.csv` with `file,species` in the sample order.'
- What this solution (achieved 0.26877) has done: 'I fix the TensorFlow/protobuf import crash by forcing the pure-Python protobuf implementation in a way that reliably takes effect before TensorFlow loads (this is the root cause of the `MessageFactory.GetPrototype` error). Then I keep your exact model/training/inference logic unchanged, but add a small compatibility fallback for `ImageDataGenerator` imports across TF/Keras versions so the notebook runs end-to-end. Finally, I keep the submission writing exactly as required (`submission.csv` with `file,species` in sample order) and add a quick sanity check that the columns and row count match the sample.'
- What this solution (achieved 0.26877) has done: 'I fix the TensorFlow/protobuf crash by forcing a compatible protobuf runtime in a way that reliably takes effect before TensorFlow (and any protobuf-backed libs) are imported, including restarting the Python process once if needed. This is a correctness/stability change that unblocks training/inference and is score-neutral by itself. I also make the dataset path resolution slightly more robust (still using your existing candidate list) so train/test/sample_submission are always found in this Kaggle layout. The model, preprocessing, training loop, and submission formatting remain unchanged to preserve core logic while allowing your existing score-improving VGG16 preprocessing pipeline to run end-to-end and write `submission.csv`.'
- What this solution (achieved 0.26877) has done: 'I remove the failing self re-exec logic (it crashes in this notebook/kernel environment) and instead apply the protobuf compatibility fix in-process before importing TensorFlow/Keras, which directly addresses the `MessageFactory.GetPrototype` error and unblocks the pipeline. I also consolidate all critical imports and path resolution in the first cell so later cells don’t cascade into `NameError`s when an earlier cell fails. The model, preprocessing (VGG16 `preprocess_input`), training loop, and submission formatting remain the same core logic; changes are only to make it run end-to-end and reliably write `submission.csv` with `file,species` in sample order. Finally, I add a lightweight safety fallback to load images from the correct train/test directories and ensure the submission row count/columns match the sample submission.'
- What this solution (achieved 0.26877) has done: 'The crash happens before training because TensorFlow 2.18 is incompatible with protobuf 6.x in this environment; setting the env var alone isn’t sufficient because protobuf may already be imported. I fix this by forcibly downgrading protobuf to a TF-compatible 4.x version at runtime *before* importing TensorFlow, then proceed with your exact same VGG16 + Dense model and training loop unchanged. This is a stability fix (unblocks execution) and should also restore expected model behavior, moving the score up toward the target because you actually be training/predicting correctly. I also keep your existing path resolution and submission writing, ensuring `submission.csv` is produced with `file,species` in the sample order.'
- What this solution (achieved 0.51652) has done: 'Your score is far below the target, so we should improve performance with the smallest changes that preserve your core model and training approach. The biggest issue is you only train for 1 epoch with the VGG16 backbone frozen, which typically underfits badly; increasing epochs (keeping everything else the same) is the minimal, high-impact adjustment to move F1 upward toward the target. I also make the Dense(12) output size derive from the label encoder so it can’t silently mismatch if class discovery changes, and I add a deterministic setting for TensorFlow ops to reduce run-to-run variance (stability while improving). No architecture, loss, preprocessing, or training method is changed—just training longer and a couple of safety alignments.'
- What this solution (achieved 0.57357) has done: 'Your current score is far below the target, so we should cautiously improve generalization without changing the model architecture or preprocessing semantics. The smallest high-impact lever that preserves your core logic is to train longer (same frozen VGG16 + head, same optimizer/loss), because 8 epochs is still typically underfit for this task at 70×70 resolution. I also ensure the validation pipeline uses the exact same preprocessing function as training (via a small helper) to avoid any subtle mismatch, which can move F1 upward without changing the approach. Finally, I keep submission ordering strictly aligned to `sample_submission.csv` (already correct) and leave paths/I/O unchanged.'
- What this solution (achieved 0.60661) has done: 'Your current score (0.57357) is far below the target (0.92191), so we should improve performance with the smallest change that preserves the same model and training approach. The most impactful minimal adjustment is to **increase training epochs** while keeping the exact same frozen VGG16 + Dense head, optimizer, loss, preprocessing, and data pipeline. To avoid overfitting surprises while staying metric-aligned, I also add a micro-F1 check on the held-out split (does not affect training or submission) so you can see whether the longer training is moving in the right direction. Everything else (paths, preprocessing, submission order/format) remains unchanged and the script still writes a valid `submission.csv`.'
- What this solution (achieved 0.5976) has done: 'You’re still far below the target (0.60661 vs 0.92191), so the safest way to move the score upward without changing your core model is to (1) ensure VGG16 sees the exact input distribution it expects and (2) reduce generalization error with the smallest training-side change. I keep the same frozen VGG16 + Flatten + Dense(256) + Dense(12) head and the same optimizer/loss/training loop, but I add the standard VGG16-friendly geometric augment (small zoom) and include the usual `channel_shift_range`=0 while keeping semantics intact. I also make the validation preprocessing path identical to training by using the same preprocessing function wrapper everywhere (no functional change, but prevents subtle dtype/shape mismatches), and I bump epochs modestly (28→34) to improve fit while staying within the same approach. Finally, I keep submission ordering aligned to `sample_submission.csv` exactly as you already do.'
- What this solution (achieved 0.61411) has done: 'You’re still far below the target (0.5976 vs 0.92191), so we should move the score upward with the smallest changes that keep your VGG16-frozen + Flatten + Dense(256) + Dense(12) core intact. The most likely bottleneck now is underfitting in the classifier head: I keep the exact same model and training loop, but increase epochs modestly and add a small learning-rate reduction-on-plateau callback to help the existing head converge better without changing the architecture or loss. I also ensure we train on the full augmented pipeline for validation as well (only preprocessing, no augmentation) and keep submission ordering aligned to `sample_submission.csv` exactly as you already do. These are minimal, metric-aligned changes that typically lift micro-F1 meaningfully on this competition while staying within your constraints.'
- What this solution (achieved 0.60661) has done: 'You’re under the target (0.61411 vs 0.92191), so the smallest score-improving change that preserves your exact VGG16-frozen + Dense head core is to let the classifier head fit better by training a bit longer while keeping the same optimizer/loss and training loop. I also make the validation pipeline use the *same kind of generator preprocessing path* as training (preprocess only, no augmentation) to keep the data plumbing identical between train/val, which often lifts generalization a bit without changing semantics. Finally, I add a tiny class-weighting step (computed from your existing labels) to reduce bias toward frequent classes, which tends to improve micro-F1 for this dataset with minimal risk and no architecture changes. Everything else (paths, model, preprocessing function, submission order/format) stays the same and still writes `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import sys
import random
from glob import glob
import subprocess

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")


def _ensure_compatible_protobuf():
    try:
        import google.protobuf  # noqa: F401
        from google.protobuf import __version__ as pb_ver

        major = int(pb_ver.split(".")[0])
        if major >= 5:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", "protobuf>=4.21.0,<5"]
            )
            for m in list(sys.modules.keys()):
                if m.startswith("google.protobuf"):
                    del sys.modules[m]
    except Exception:
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", "protobuf>=4.21.0,<5"]
            )
            for m in list(sys.modules.keys()):
                if m.startswith("google.protobuf"):
                    del sys.modules[m]
        except Exception as e:
            print("WARNING: Could not enforce protobuf version:", repr(e))


_ensure_compatible_protobuf()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

import tensorflow as tf
import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

SEED = 7
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

CANDIDATE_ROOTS = [
    "/kaggle/input/plant-seedlings-classification/plant-seedlings-classification",
    "/kaggle/input/plant-seedlings-classification",
    "/kaggle/data/plant-seedlings-classification/plant-seedlings-classification",
    "/kaggle/data/plant-seedlings-classification",
    "/kaggle/working/plant-seedlings-classification/plant-seedlings-classification",
    "/kaggle/working/plant-seedlings-classification",
]

BASE = None
for p in CANDIDATE_ROOTS:
    if os.path.exists(os.path.join(p, "train")) and os.path.exists(
        os.path.join(p, "test")
    ):
        if os.path.exists(os.path.join(p, "sample_submission.csv")):
            BASE = p
            break

if BASE is None:
    for root in ["/kaggle/input", "/kaggle/data", "/kaggle/working"]:
        cand = os.path.join(
            root, "plant-seedlings-classification", "plant-seedlings-classification"
        )
        if os.path.exists(os.path.join(cand, "train")) and os.path.exists(
            os.path.join(cand, "test")
        ):
            BASE = cand
            break

if BASE is None:
    BASE = "/kaggle/input"

TRAIN_DIR = os.path.join(BASE, "train")
TEST_DIR = os.path.join(BASE, "test")
SAMPLE_SUB_PATH = os.path.join(BASE, "sample_submission.csv")

print("Resolved BASE:", BASE)
print("TRAIN_DIR:", TRAIN_DIR, "exists:", os.path.exists(TRAIN_DIR))
print("TEST_DIR :", TEST_DIR, "exists:", os.path.exists(TEST_DIR))
print("SAMPLE_SUB_PATH:", SAMPLE_SUB_PATH, "exists:", os.path.exists(SAMPLE_SUB_PATH))




## === cell 1
images_path = os.path.join(TRAIN_DIR, "*", "*.png")
images = sorted(glob(images_path))

train_images = []
train_labels = []

for img_path in images:
    img = cv2.imread(img_path)
    if img is None:
        continue
    img = cv2.resize(img, (70, 70))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    train_images.append(img)
    train_labels.append(os.path.basename(os.path.dirname(img_path)))

train_X = np.asarray(train_images, dtype=np.uint8)
train_Y = pd.Series(train_labels, name="species")

print(
    "Loaded train:",
    train_X.shape,
    "labels:",
    train_Y.shape,
    "unique:",
    train_Y.nunique(),
)




## === cell 2
idx = 100 if len(train_X) > 100 else 0
plt.title(train_Y.iloc[idx])
_ = plt.imshow(train_X[idx])  # already RGB
plt.axis("off")
plt.show()




## === cell 3
encoder = LabelEncoder()
encoder.fit(train_Y.values)

encoded_labels = encoder.transform(train_Y.values)
categorical_labels = keras.utils.to_categorical(
    encoded_labels, num_classes=len(encoder.classes_)
)

print("Classes:", list(encoder.classes_))
print("categorical_labels:", categorical_labels.shape)




## === cell 4
idx = 100 if len(train_X) > 100 else 0
plt.title(str(categorical_labels[idx]))
_ = plt.imshow(train_X[idx])  # already RGB
plt.axis("off")
plt.show()




## === cell 5
x_train, x_test, y_train, y_test, enc_train, enc_test = train_test_split(
    train_X,
    categorical_labels,
    encoded_labels,
    test_size=0.25,
    random_state=SEED,
    stratify=encoded_labels,
)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)




## === cell 6
from keras import layers
from keras.models import Sequential
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input

base_model = VGG16(include_top=False, weights="imagenet", input_shape=(70, 70, 3))
base_model.trainable = False  # preserve core transfer learning setup




## === cell 7
model = Sequential()
model.add(base_model)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation="relu"))
model.add(layers.Dense(len(encoder.classes_), activation="softmax"))




## === cell 8
opt = keras.optimizers.Adam(learning_rate=1e-4)
model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()




## === cell 9
try:
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
except Exception:
    from keras.preprocessing.image import ImageDataGenerator


def vgg16_preprocess_array(x):
    x = x.astype(np.float32, copy=False)
    return preprocess_input(x)


datagen = ImageDataGenerator(
    preprocessing_function=vgg16_preprocess_array,
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    rotation_range=0,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    vertical_flip=False,
)

val_datagen = ImageDataGenerator(preprocessing_function=vgg16_preprocess_array)

datagen.fit(x_train.astype(np.float32))




## === cell 10
BATCH_SIZE = 50
steps_per_epoch = int(np.ceil(x_train.shape[0] / BATCH_SIZE))
val_steps = int(np.ceil(x_test.shape[0] / BATCH_SIZE))

EPOCHS = 64

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1,
)

from sklearn.utils.class_weight import compute_class_weight

class_weights_arr = compute_class_weight(
    class_weight="balanced", classes=np.arange(len(encoder.classes_)), y=enc_train
)
class_weight = {i: float(w) for i, w in enumerate(class_weights_arr)}
print("Using class_weight:", class_weight)

history = model.fit(
    datagen.flow(x_train, y_train, batch_size=BATCH_SIZE, shuffle=True, seed=SEED),
    steps_per_epoch=steps_per_epoch,
    epochs=EPOCHS,
    validation_data=val_datagen.flow(
        x_test, y_test, batch_size=BATCH_SIZE, shuffle=False
    ),
    validation_steps=val_steps,
    callbacks=[reduce_lr],
    class_weight=class_weight,
    verbose=1,
)




## === cell 11
loss, accuracy = model.evaluate(vgg16_preprocess_array(x_test), y_test, verbose=0)
print("Test Set Accuracy: " + str(accuracy * 100) + "%")

from sklearn.metrics import f1_score

val_pred = model.predict(vgg16_preprocess_array(x_test), verbose=0)
y_true = np.argmax(y_test, axis=1)
y_hat = np.argmax(val_pred, axis=1)
print("Validation micro-F1:", f1_score(y_true, y_hat, average="micro"))




## === cell 12
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
test_files_order = sample_sub["file"].tolist()

test_images_arr = []
missing = 0
for fname in test_files_order:
    img_path = os.path.join(TEST_DIR, fname)
    img = cv2.imread(img_path)
    if img is None:
        missing += 1
        img = np.zeros((70, 70, 3), dtype=np.uint8)
    else:
        img = cv2.resize(img, (70, 70))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    test_images_arr.append(img)

test_X = np.asarray(test_images_arr, dtype=np.uint8)
print("Loaded test:", test_X.shape, "missing:", missing)

idx = 100 if len(test_X) > 100 else 0
_ = plt.imshow(test_X[idx])  # already RGB
plt.axis("off")
plt.show()




## === cell 13
predictions = model.predict(vgg16_preprocess_array(test_X), verbose=0)
preds = np.argmax(predictions, axis=1)
pred_str = encoder.classes_[preds]

print("Predictions:", predictions.shape, "Labels:", pred_str.shape)

final_predictions = pd.DataFrame({"file": test_files_order, "species": pred_str})

assert list(final_predictions.columns) == ["file", "species"]
assert final_predictions.shape[0] == sample_sub.shape[0]

final_predictions.to_csv("submission.csv", index=False)

print(final_predictions.head())
print("Wrote submission.csv with shape:", final_predictions.shape)
print("submission.csv exists:", os.path.exists("submission.csv"))
