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
Classify each cassava image into four disease categories or a fifth category indicating a healthy leaf.

## Metric
Categorization accuracy.

## Submission Format
```
image_id,label
1000471002.jpg,4
1000840542.jpg,4
etc.
```

## Dataset
**[train/test]_images** the image files.

**train.csv**

- `image_id` the image file name.

- `label` the ID code for the disease.

**sample_submission.csv** A properly formatted sample submission, given the disclosed test set content.

- `image_id` the image file name.

- `label` the predicted ID code for the disease.

**[train/test]_tfrecords** the image files in tfrecord format.

**label_num_to_disease_map.json** The mapping between each disease code and the real disease name.

# 2. Python version

3.9

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        input/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        working/
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
```

-> data/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/cassava-leaf-disease-classification/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/cassava-leaf-disease-classification/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> data/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> input/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> (stopped after 10 files for performance)

# 5. Target score

0.6128739800543971

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.08819) has done: 'I fix the import-time crash by removing the incompatible protobuf-related callback import and relying on `tf.keras.callbacks` instead. I also make the EfficientNet weights loading robust: if the external `efficientnetb0_notop.h5` file is not present, it fall back to `weights="imagenet"` (same architecture/core logic) so the model is always defined and inference can run. Finally, I correct the test-time preprocessing and prediction logic (predict each image independently, apply the correct EfficientNet preprocessing, and avoid the incorrect running-majority vote), and ensure a valid `submission.csv` with the required columns is written.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
if "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION" in os.environ:
    os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

import tensorflow as tf

SEED = 2021
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass
try:
    tf.config.optimizer.set_experimental_options({"layout_optimizer": True})
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)  # let TF decide
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
DATA_PATH = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_IMG_DIR = os.path.join(DATA_PATH, "train_images")
TEST_IMG_DIR = os.path.join(DATA_PATH, "test_images")

assert os.path.exists(DATA_PATH), f"Missing DATA_PATH: {DATA_PATH}"
assert os.path.exists(TRAIN_IMG_DIR), f"Missing TRAIN_IMG_DIR: {TRAIN_IMG_DIR}"
assert os.path.exists(TEST_IMG_DIR), f"Missing TEST_IMG_DIR: {TEST_IMG_DIR}"




## === cell 2
import pandas as pd
import json

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau




## === cell 3
def first_existing_path(candidates):
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None




## === cell 4
path = DATA_PATH

train_df = pd.read_csv(os.path.join(path, "train.csv"))
print("Total images for Train (from train.csv): ", len(train_df))

with open(os.path.join(path, "label_num_to_disease_map.json")) as file:
    classes = json.loads(file.read())

print(json.dumps(classes, indent=4))

train_df["class"] = train_df["label"].map({int(i): c for i, c in classes.items()})
train_df.head()




## === cell 5
DO_PLOTS = False

if DO_PLOTS:
    import seaborn as sns
    import matplotlib.pyplot as plt

    plt.subplots(figsize=(12, 8))
    ax = sns.countplot(x="class", data=train_df)

    for a in ax.patches:
        ax.annotate("{:1}".format(a.get_height()), (a.get_x() + 0.3, a.get_height()))
    plt.xticks(rotation=90)
    ax.set_title("classes", fontdict={"fontsize": 15})
    plt.show()




## === cell 6
def plot_images(class_id, label, images_number, verbose=0):
    import cv2  # local import to avoid hard dependency when plotting is disabled
    import matplotlib.pyplot as plt

    plot_list = (
        train_df[train_df["label"] == class_id]
        .sample(images_number, random_state=SEED)["image_id"]
        .tolist()
    )

    if verbose:
        print(plot_list)

    labels = [label for _ in range(len(plot_list))]
    size = np.sqrt(images_number)
    if int(size) * int(size) < images_number:
        size = int(size) + 1

    plt.figure(figsize=(20, 20))

    for ind, (image_id, label) in enumerate(zip(plot_list, labels)):
        plt.subplot(size, size, ind + 1)
        image = cv2.imread(os.path.join(path, "train_images", image_id))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        plt.imshow(image)
        plt.title(label, fontsize=12)
        plt.axis("off")

    plt.show()


if DO_PLOTS:
    plot_images(class_id=4, label="Healthy", images_number=6, verbose=1)
    plot_images(
        class_id=3, label="Cassava Mosaic Disease (CMD)", images_number=6, verbose=1
    )
    plot_images(
        class_id=2, label="Cassava Green Mottle (CGM)", images_number=6, verbose=1
    )
    plot_images(
        class_id=1,
        label="Cassava Brown Streak Disease (CBSD)",
        images_number=6,
        verbose=1,
    )
    plot_images(
        class_id=0, label="Cassava Bacterial Blight (CBB)", images_number=6, verbose=1
    )




## === cell 7
train_df["label"] = train_df["label"].astype(str)




## === cell 8
TARGET_SIZE = (380, 380)
BATCH_SIZE = 16
STEPS_PER_EPOCH = int(len(train_df) * 0.8 // BATCH_SIZE)
VALIDATION_STEPS = int(len(train_df) * 0.2 // BATCH_SIZE)
EPOCHS = 5

print(
    "STEPS_PER_EPOCH:",
    STEPS_PER_EPOCH,
    "VALIDATION_STEPS:",
    VALIDATION_STEPS,
    "EPOCHS:",
    EPOCHS,
)




## === cell 9
train_datagen = ImageDataGenerator(
    validation_split=0.2,
    rotation_range=45,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode="nearest",
    height_shift_range=0.2,
    width_shift_range=0.2,
    preprocessing_function=preprocess_input,
)

train_generator = train_datagen.flow_from_dataframe(
    train_df,
    directory=TRAIN_IMG_DIR + "/",
    subset="training",
    x_col="image_id",
    y_col="label",
    target_size=TARGET_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="sparse",
    seed=SEED,
    shuffle=True,
)

validation_datagen = ImageDataGenerator(
    validation_split=0.2,
    preprocessing_function=preprocess_input,
)

validation_generator = validation_datagen.flow_from_dataframe(
    train_df,
    directory=TRAIN_IMG_DIR + "/",
    subset="validation",
    x_col="image_id",
    y_col="label",
    target_size=TARGET_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="sparse",
    seed=SEED,
    shuffle=True,
)

train_ds = tf.data.Dataset.from_generator(
    lambda: train_generator,
    output_signature=(
        tf.TensorSpec(shape=(None, *TARGET_SIZE, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.float32),
    ),
).prefetch(tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_generator(
    lambda: validation_generator,
    output_signature=(
        tf.TensorSpec(shape=(None, *TARGET_SIZE, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.float32),
    ),
).prefetch(tf.data.AUTOTUNE)




## === cell 10
candidate_notop_weights = [
    "/kaggle/input/efficientnetb0/efficientnetb0_notop.h5",
    "/kaggle/input/efficientnetb0-notop/efficientnetb0_notop.h5",
]
weights_path = first_existing_path(candidate_notop_weights)

if weights_path is not None:
    eff_weights = weights_path
    print("Using EfficientNetB0 notop weights from:", eff_weights)
else:
    eff_weights = "imagenet"
    print(
        "EfficientNetB0 notop weights file not found; falling back to weights='imagenet'."
    )

basemodel = EfficientNetB0(
    weights=eff_weights,
    include_top=False,
    input_shape=TARGET_SIZE + (3,),
)

headmodel = layers.GlobalAveragePooling2D()(basemodel.output)
headmodel = layers.Dense(5, activation="softmax")(headmodel)
model = keras.Model(inputs=basemodel.input, outputs=headmodel)




## === cell 11
candidate_finetuned = [
    "../input/best-weights-efficient/best.h5",
    "/kaggle/input/best-weights-efficient/best.h5",
    "/kaggle/input/best-weights-efficientnet/best.h5",
]
finetuned_path = first_existing_path(candidate_finetuned)

if finetuned_path is not None:
    try:
        model.load_weights(finetuned_path)
        print("Loaded fine-tuned weights from:", finetuned_path)
    except Exception as e:
        print(
            "Warning: found fine-tuned weights but failed to load; continuing without them."
        )
        print("Load error:", repr(e))
else:
    print("Fine-tuned weights not found; continuing with base weights.")

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)




## === cell 12
model_save = ModelCheckpoint(
    "./best_weights.h5",
    save_best_only=True,
    monitor="val_loss",
    mode="min",
    verbose=1,
)
reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.3,
    patience=2,
    min_lr=1e-6,
    mode="min",
    verbose=1,
)
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    mode="min",
    verbose=1,
    restore_best_weights=True,
)

history = model.fit(
    train_ds,
    steps_per_epoch=STEPS_PER_EPOCH,
    epochs=EPOCHS,
    validation_data=val_ds,
    validation_steps=VALIDATION_STEPS,
    callbacks=[model_save, early_stop, reduce_lr],
    workers=max(1, (os.cpu_count() or 2) // 2),
    use_multiprocessing=True,
    max_queue_size=32,
)

model.save("model.h5")

if os.path.exists("./best_weights.h5"):
    try:
        model.load_weights("./best_weights.h5")
        print("Loaded best checkpoint from ./best_weights.h5 for inference.")
    except Exception as e:
        print("Warning: failed to load ./best_weights.h5; using current model weights.")
        print("Load error:", repr(e))




## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4216094581.py in <cell line: 0>()
     24 # Speed (no logic change): enable parallel generator workers and larger queue to reduce input stalls.
     25 # Steps/epochs/augmentation/model remain identical.
---> 26 history = model.fit(
     27     train_ds,
     28     steps_per_epoch=STEPS_PER_EPOCH,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    117             return fn(*args, **kwargs)
    118         except Exception as e:
--> 119             filtered_tb = _process_traceback_frames(e.__traceback__)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`

TypeError: TensorFlowTrainer.fit() got an unexpected keyword argument 'workers'

## === cell 13
ss = pd.read_csv(os.path.join(DATA_PATH, "sample_submission.csv"))

test_df = ss[["image_id"]].copy()
test_df["label"] = "0"  # dummy, unused for predictions

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_generator = test_datagen.flow_from_dataframe(
    test_df,
    directory=TEST_IMG_DIR + "/",
    x_col="image_id",
    y_col=None,
    target_size=TARGET_SIZE,
    batch_size=64,
    class_mode=None,
    shuffle=False,
)

test_ds = tf.data.Dataset.from_generator(
    lambda: test_generator,
    output_signature=tf.TensorSpec(shape=(None, *TARGET_SIZE, 3), dtype=tf.float32),
).prefetch(tf.data.AUTOTUNE)

proba = model.predict(
    test_ds,
    steps=int(np.ceil(len(test_df) / test_generator.batch_size)),
    verbose=0,
    workers=max(1, (os.cpu_count() or 2) // 2),
    use_multiprocessing=True,
    max_queue_size=32,
)
test_preds = np.argmax(proba, axis=1).astype(int)

ss["label"] = test_preds[: len(ss)]
ss.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", ss.shape)
print(ss.head())
assert os.path.exists("submission.csv") and os.path.getsize("submission.csv") > 0

## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1609309741.py in <cell line: 0>()
     22 ).prefetch(tf.data.AUTOTUNE)
     23 
---> 24 proba = model.predict(
     25     test_ds,
     26     steps=int(np.ceil(len(test_df) / test_generator.batch_size)),

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    117             return fn(*args, **kwargs)
    118         except Exception as e:
--> 119             filtered_tb = _process_traceback_frames(e.__traceback__)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`

TypeError: TensorFlowTrainer.predict() got an unexpected keyword argument 'workers'
