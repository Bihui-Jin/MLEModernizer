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
scikit-image==0.25.2
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

0.11083

# 6. Current score

0.95195

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.07958) has done: 'I fix the broken dataset paths so the script actually finds the `train/` and `test/` folders, which currently makes your generated CSVs empty and later generators length-0. I also resolve the TensorFlow/Keras import crash by switching to `tf_keras` (already installed in your environment) while keeping the same ResNet50V2 architecture, preprocessing, and training loop semantics. Finally, I ensure test prediction order matches `sample_submission.csv` exactly and write a valid `/kaggle/working/submission.csv` with the required `file,species` columns. These are execution/unblocking fixes; they should also move the score upward from “no submission” to a real (and likely better-than-random) score.'
- What this solution (achieved 0.94595) has done: 'Most of the timeout comes from slow input pipelines (Python-generator → `tf.data.from_generator`) plus heavy per-batch augmentation running in the main thread, which leaves the GPU/CPU underutilized. I keep the exact same model, optimizer, callbacks, and training semantics, but switch to Keras’ built-in `Sequence` pipeline with multiprocessing workers and `max_queue_size`, and remove the redundant `tf.data` wrapper (which adds overhead and prevents Keras’ generator multiprocessing from helping). I also ensure `steps_per_epoch/validation_steps` cover the same number of batches as before and keep determinism/seeds intact. Prediction is similarly switched to direct generator prediction with workers to speed up test-time I/O/decoding.'
- What this solution (achieved 0.96096) has done: 'I fix the crash in the TensorFlow import cell caused by an incompatibility between TensorFlow 2.18 and protobuf 6.x by forcing protobuf to use the pure-Python backend before TensorFlow is imported, and by importing `tf_keras` first to keep your existing model/training logic unchanged. I also add a small safety fallback to disable XLA JIT if it triggers the same protobuf-related failure in this environment, without changing the model architecture or training loop semantics. Finally, I keep your submission-writing logic intact but ensure the output file is always produced at `/kaggle/working/submission.csv` with the correct columns and ordering.'
- What this solution (achieved 0.94595) has done: 'I fix the TensorFlow/protobuf crash in your import/config cell by forcing a compatible protobuf runtime (pure-Python) and disabling XLA JIT preemptively, since XLA commonly triggers the `MessageFactory.GetPrototype` failure in TF 2.18 + protobuf 6.x. This is an execution-only change: it doesn’t alter your model, data pipeline, training loop, or submission logic, so your score should remain essentially the same (and still far above the target, but correctness/stability is the priority here because the run currently cannot complete). I also make the environment variables take effect by setting them before importing TensorFlow/tf_keras and by preventing any later code from re-enabling JIT. Finally, I keep the same paths and ensure the script still writes `/kaggle/working/submission.csv` in the required `file,species` format.'
- What this solution (achieved 0.96697) has done: 'I fix the TensorFlow/protobuf crash by ensuring the protobuf runtime uses the pure-Python implementation *before* any TensorFlow/tf_keras-related import occurs (and by importing `google.protobuf` first to “lock in” the backend). I keep your exact model/data/training logic unchanged, only adjusting the import/config ordering so the notebook runs end-to-end again and reliably writes `/kaggle/working/submission.csv`. Because your current score (0.94595) is far above the target (0.11083), I won’t make any score-improving changes; this patch is intended to be score-neutral and focused on stability/execution. The rest of the pipeline (generators, ResNet50V2, callbacks, prediction ordering) remains as-is.'
- What this solution (achieved 0.94745) has done: 'We fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf backend and disabling XLA **before any protobuf/TensorFlow-related imports**, and by setting an additional env flag that prevents the C++ protobuf implementation from being used. This is an execution/stability fix only (no model/training/prediction logic changes), so it should keep your score essentially the same (still far above the target) while making the notebook run end-to-end reliably. We also add a tiny safety fallback: if `tf.config.experimental.enable_op_determinism()` triggers the protobuf issue in this environment, we skip it rather than crashing. Submission writing remains unchanged and still produces `/kaggle/working/submission.csv` with `file,species` aligned to `sample_submission.csv`.'
- What this solution (achieved 0.95195) has done: 'I fix the TensorFlow/protobuf runtime crash by forcing the pure-Python protobuf backend and the “fast C++ protos off” environment flag *before any* protobuf/TensorFlow imports, then importing TensorFlow first and `tf_keras` second (this ordering is the key stability fix for TF 2.18 + protobuf 6). I keep your model, generators, training loop, and submission-writing logic unchanged to avoid intentionally shifting the score (your current score is far above the target). I also add a small fallback to skip op determinism if it triggers the same protobuf issue, so the notebook can complete end-to-end reliably. Finally, I keep writing `/kaggle/working/submission.csv` with the required `file,species` columns aligned to `sample_submission.csv`.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_DISABLE_CPP_IMPLEMENTATION", "1")
os.environ.setdefault(
    "TF_XLA_FLAGS", "--tf_xla_auto_jit=0 --tf_xla_enable_xla_devices=false"
)

import google.protobuf  # noqa: F401

import numpy as np
import pandas as pd

BASE = "/kaggle/input/plant-seedlings-classification"
TRAIN_DIR = os.path.join(BASE, "train")
TEST_DIR = os.path.join(BASE, "test")

assert os.path.isdir(TRAIN_DIR), f"TRAIN_DIR not found: {TRAIN_DIR}"
assert os.path.isdir(TEST_DIR), f"TEST_DIR not found: {TEST_DIR}"

print("Using BASE:", BASE)
print("TRAIN_DIR:", TRAIN_DIR)
print("TEST_DIR:", TEST_DIR)



## === cell 1
class_names = sorted(
    [d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))]
)
print("Number of class folders:", len(class_names))
print("Class folders:", class_names)



## === cell 2
counts = {}
total = 0
img_ext = (".png", ".jpg", ".jpeg")

for cls in class_names:
    cls_dir = os.path.join(TRAIN_DIR, cls)
    n = 0
    with os.scandir(cls_dir) as it:
        for entry in it:
            if entry.is_file():
                name = entry.name.lower()
                if name.endswith(img_ext):
                    n += 1
    counts[cls] = n
    total += n

datos_classes = pd.DataFrame({"file": pd.Series(counts)}).sort_index()
print("Total train images:", total)
print(datos_classes)



## === cell 3
classes = np.array(class_names)
print(f"Number of classes: {len(classes)}")



## === cell 4
import matplotlib.pyplot as plt

try:
    plot = datos_classes.plot.pie(y="file", figsize=(5, 5), legend=False)
    plt.show()
except Exception as e:
    print("Skipping pie plot due to:", repr(e))



## === cell 5
print("Skipping sample image plotting to save time.")



## === cell 6
print("Skipping image-size histogram to save time.")



## === cell 7
import random

import tensorflow as tf
import tf_keras as keras
from tf_keras import layers
from tf_keras.layers import Dropout, BatchNormalization
from tf_keras.applications.resnet_v2 import ResNet50V2, preprocess_input
from tf_keras.models import Sequential
from tf_keras.optimizers import Adam
from tf_keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint
from tf_keras.preprocessing.image import ImageDataGenerator
from math import exp

print("TensorFlow:", tf.__version__)
print("tf_keras:", keras.__version__)

seed = 42
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)

try:
    tf.config.experimental.enable_op_determinism()
    print("Enabled op determinism")
except Exception as e:
    print("Op determinism not enabled due to:", repr(e))

try:
    tf.config.optimizer.set_jit(False)
    print("XLA JIT disabled")
except Exception as e:
    print("XLA JIT disable skipped due to:", repr(e))

try:
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print("GPUs:", gpus)
except Exception as e:
    print("GPU config skipped due to:", repr(e))



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 8
file = "/kaggle/working/pretrained_ResNet50V2_vFINAL"
batch_size = 32
val_split = 0.2
image_size = (256, 256)
PROYECT_FOLDER_TRAIN = TRAIN_DIR

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,  # Standardize for ResNetV2
    rotation_range=30,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=[0.7, 1.3],
    rescale=0.9,
    vertical_flip=True,
    horizontal_flip=True,
    validation_split=val_split,
)

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=val_split,
)

train_generator = train_datagen.flow_from_directory(
    PROYECT_FOLDER_TRAIN,
    target_size=image_size,
    color_mode="rgb",
    batch_size=batch_size,
    class_mode="categorical",
    subset="training",
    seed=seed,
    shuffle=True,
)

val_generator = val_datagen.flow_from_directory(
    PROYECT_FOLDER_TRAIN,
    target_size=image_size,
    color_mode="rgb",
    batch_size=batch_size,
    class_mode="categorical",
    subset="validation",
    seed=seed,
    shuffle=True,
)

idx_to_class = {v: k for k, v in train_generator.class_indices.items()}
print("Classes (generator order):", [idx_to_class[i] for i in range(len(idx_to_class))])
print("num_classes from generator:", len(train_generator.class_indices))

train_ds = train_generator
val_ds = val_generator



## === cell 9
input_shape_c = (image_size[0], image_size[1], 3)
print(input_shape_c)

base_model = ResNet50V2(
    weights="imagenet", include_top=False, input_shape=input_shape_c
)



## === cell 10
num_classes = len(train_generator.class_indices)

for layer in base_model.layers:
    if layer.name == "conv5_block1_1_conv":
        break
    layer.trainable = False

pre_trained_model = Sequential()
pre_trained_model.add(base_model)
pre_trained_model.add(layers.Flatten())
pre_trained_model.add(layers.Dense(512, activation="relu"))
pre_trained_model.add(Dropout(0.5))
pre_trained_model.add(BatchNormalization())
pre_trained_model.add(layers.Dense(num_classes, activation="softmax"))
pre_trained_model.summary()



## === cell 11
epochs = 200

print("[INFO]: Compiling the model...")
pre_trained_model.compile(
    loss="categorical_crossentropy",
    optimizer=Adam(learning_rate=1e-3),
    metrics=["accuracy"],
)


def scheduler(epoch, lr):
    if epoch < 5:
        return lr
    return lr * exp(-0.1)


annealer = LearningRateScheduler(scheduler)

earlystop = EarlyStopping(
    patience=5,
    monitor="val_loss",
)

modelsave = ModelCheckpoint(
    filepath=file + ".h5",
    save_best_only=True,
    verbose=1,
    save_weights_only=False,
)

steps_per_epoch = train_generator.n // train_generator.batch_size
validation_steps = val_generator.n // val_generator.batch_size

try:
    _cpu = os.cpu_count() or 2
    _workers = max(2, min(8, _cpu))  # cap to avoid overhead on Kaggle
except Exception:
    _workers = 4

print(f"[INFO]: Entrenando la red... (workers={_workers})")
H_pre = pre_trained_model.fit(
    train_ds,
    validation_data=val_ds,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    epochs=epochs,
    callbacks=[annealer, earlystop, modelsave],
    workers=_workers,
    use_multiprocessing=True,
    max_queue_size=32,
)



## === cell 12
print("[INFO]: Evaluating the model...")

num_epochs = len(H_pre.history["loss"])

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, num_epochs), H_pre.history["loss"], label="train_loss")
plt.plot(np.arange(0, num_epochs), H_pre.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, num_epochs), H_pre.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, num_epochs), H_pre.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()



## === cell 13
print("Skipping test.csv generation to save time.")



## === cell 14
test_files = []
with os.scandir(TEST_DIR) as it:
    for entry in it:
        if entry.is_file():
            n = entry.name.lower()
            if n.endswith((".png", ".jpg", ".jpeg")):
                test_files.append(entry.name)

print("Test images found (dir listing):", len(test_files))
if test_files:
    print(
        "Example test file:",
        test_files[0],
        "path:",
        os.path.join(TEST_DIR, test_files[0]),
    )



## === cell 15
batch_size = 32
image_size = (256, 256)
PROYECT_FOLDER_TEST = BASE  # flow_from_directory expects a base with subfolder "test"

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

test_generator = test_datagen.flow_from_directory(
    PROYECT_FOLDER_TEST,
    target_size=image_size,
    color_mode="rgb",
    batch_size=batch_size,
    classes=["test"],
    class_mode=None,  # no labels for test
    shuffle=False,  # preserve filename order for submission
)

print("Test samples found:", test_generator.n)

test_ds = test_generator



## === cell 16
best_path = file + ".h5"
if os.path.exists(best_path):
    pre_trained_model = keras.models.load_model(best_path, compile=False)
    print("Loaded best checkpoint:", best_path)
else:
    print("Checkpoint not found, using in-memory model weights:", best_path)

test_steps = int(np.ceil(test_generator.n / test_generator.batch_size))

try:
    _cpu = os.cpu_count() or 2
    _workers = max(2, min(8, _cpu))
except Exception:
    _workers = 4

predicted_class = pre_trained_model.predict(
    test_ds,
    steps=test_steps,
    verbose=1,
    workers=_workers,
    use_multiprocessing=True,
    max_queue_size=32,
)
predicted_class = predicted_class[: test_generator.n]
print("Pred shape:", predicted_class.shape)

predicted_class_number = np.argmax(predicted_class, axis=1)
print("Num predictions:", len(predicted_class_number))



## === cell 17
pred_files = [os.path.basename(f) for f in test_generator.filenames]
pred_species = [idx_to_class[i] for i in predicted_class_number]
pred_df = pd.DataFrame({"file": pred_files, "species": pred_species})

csv_resultsfile = "/kaggle/working/submission.csv"
sample_path = os.path.join(BASE, "sample_submission.csv")
sample = pd.read_csv(sample_path)

print("Sample shape:", sample.shape, "Pred df shape:", pred_df.shape)
missing = set(sample["file"]) - set(pred_df["file"])
extra = set(pred_df["file"]) - set(sample["file"])
print("Missing files:", len(missing), "Extra files:", len(extra))

submission = sample[["file"]].merge(pred_df, on="file", how="left")
assert (
    submission["species"].isna().sum() == 0
), "Some test files did not get predictions."

submission = submission[["file", "species"]]
submission.to_csv(csv_resultsfile, index=False)

print("Saved ordered submission to:", csv_resultsfile)
print(submission.head())
print("Submission columns:", submission.columns.tolist(), "rows:", len(submission))
