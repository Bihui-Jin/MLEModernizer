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

# 5. Target score

0.9089861751152074

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.04304) has done: 'The changes enable mixed‑precision training (which runs faster on GPUs without changing model logic) and use multiple worker processes for data loading during `model.fit`. Both adjustments keep the architecture, loss, optimizer, epochs, and augmentation untouched, preserving the original training behaviour while reducing total runtime.'
- What this solution (achieved 0.07955) has done: 'The fix keeps the exact model architecture and training loop but speeds up data loading and reduces the number of gradient steps by increasing the batch size and using parallel workers for the generators. `flow_from_directory` now uses a larger batch (64) for both training and validation, and `model.fit` runs with multiple workers and multiprocessing enabled, which lowers per‑epoch overhead without changing the learning algorithm or final predictions.'
- What this solution (achieved 0.08109) has done: 'I fix the import error by removing the unnecessary protobuf environment setting, correct the model.fit call (the workers and use_multiprocessing arguments are not valid for a Keras generator), and unfreeze the MobileNetV2 backbone so the network can learn visual features, which should raise validation accuracy toward the target. The rest of the pipeline and submission format remain unchanged.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

if tf.config.list_physical_devices("GPU"):
    from tensorflow.keras import mixed_precision

    mixed_precision.set_global_policy("mixed_float16")

BASE_DIR = os.path.join("/kaggle", "input", "paddy-disease-classification")
print(f"Using base directory: {BASE_DIR}")




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
img_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    validation_split=0.2,
    rotation_range=5,
    shear_range=0.3,
    zoom_range=0.3,
    width_shift_range=0.05,
    height_shift_range=0.05,
    horizontal_flip=True,
    vertical_flip=True,
)




## === cell 2
train_dir = os.path.join(BASE_DIR, "train_images")
if not os.path.isdir(train_dir):
    raise FileNotFoundError(f"Training image directory not found: {train_dir}")

cpu_count = os.cpu_count() or 1

train_ds = img_datagen.flow_from_directory(
    train_dir,
    subset="training",
    seed=2021,
    class_mode="categorical",
    batch_size=64,
    target_size=(224, 224),
    shuffle=True,
    workers=cpu_count,
    use_multiprocessing=True,
)
val_ds = img_datagen.flow_from_directory(
    train_dir,
    subset="validation",
    seed=2021,
    class_mode="categorical",
    batch_size=64,
    target_size=(224, 224),
    shuffle=False,
    workers=cpu_count,
    use_multiprocessing=True,
)




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3521434778.py in <cell line: 0>()
      6 cpu_count = os.cpu_count() or 1
      7 
----> 8 train_ds = img_datagen.flow_from_directory(
      9     train_dir,
     10     subset="training",

TypeError: ImageDataGenerator.flow_from_directory() got an unexpected keyword argument 'workers'

## === cell 3
num_classes = train_ds.num_classes

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet",
    pooling="avg",
)

base_model.trainable = True

model = tf.keras.Sequential(
    [
        base_model,
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ]
)
model.summary()




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2975378397.py in <cell line: 0>()
----> 1 num_classes = train_ds.num_classes
      2 
      3 base_model = tf.keras.applications.MobileNetV2(
      4     input_shape=(224, 224, 3),
      5     include_top=False,

NameError: name 'train_ds' is not defined

## === cell 4
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4048970094.py in <cell line: 0>()
----> 1 model.compile(
      2     optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
      3     loss="categorical_crossentropy",
      4     metrics=["accuracy"],
      5 )

NameError: name 'model' is not defined

## === cell 5
lr_scheduler = ReduceLROnPlateau(
    monitor="val_accuracy", factor=0.8, patience=5, verbose=1
)
save_best = ModelCheckpoint(
    "Model.h5", monitor="val_accuracy", save_best_only=True, verbose=1
)




## === cell 6
steps_per_epoch = int(np.ceil(train_ds.samples / train_ds.batch_size))
validation_steps = int(np.ceil(val_ds.samples / val_ds.batch_size))

model.fit(
    train_ds,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_ds,
    validation_steps=validation_steps,
    epochs=50,  # increased epochs for better learning
    callbacks=[lr_scheduler, save_best],
    verbose=2,
    workers=cpu_count,
    use_multiprocessing=True,
)




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3885540711.py in <cell line: 0>()
----> 1 steps_per_epoch = int(np.ceil(train_ds.samples / train_ds.batch_size))
      2 validation_steps = int(np.ceil(val_ds.samples / val_ds.batch_size))
      3 
      4 model.fit(
      5     train_ds,

NameError: name 'train_ds' is not defined

## === cell 7
if os.path.exists("Model.h5"):
    model = tf.keras.models.load_model("Model.h5")
else:
    print("Checkpoint not found; using the model from the last epoch.")




## === cell 8
val_loss, val_acc = model.evaluate(val_ds, verbose=2)
print(f"Validation accuracy: {val_acc:.4f}")




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3014295008.py in <cell line: 0>()
----> 1 val_loss, val_acc = model.evaluate(val_ds, verbose=2)
      2 print(f"Validation accuracy: {val_acc:.4f}")
      3 
      4 

NameError: name 'model' is not defined

## === cell 9
test_dir = os.path.join(BASE_DIR, "test_images")
if not os.path.isdir(test_dir):
    raise FileNotFoundError(f"Test image directory not found: {test_dir}")

test_files = [f for f in os.listdir(test_dir) if f.lower().endswith(".jpg")]
test_df = pd.DataFrame({"filename": [os.path.join(test_dir, f) for f in test_files]})

test_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_gen = test_datagen.flow_from_dataframe(
    test_df,
    x_col="filename",
    y_col=None,
    target_size=(224, 224),
    batch_size=64,
    class_mode=None,
    shuffle=False,
    workers=cpu_count,
    use_multiprocessing=True,
)




## === cell 10
predict = model.predict(test_gen, verbose=1)




## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1806839393.py in <cell line: 0>()
----> 1 predict = model.predict(test_gen, verbose=1)
      2 
      3 

NameError: name 'model' is not defined

## === cell 11
predicted_class_indices = np.argmax(predict, axis=1)
inv_map = {v: k for k, v in train_ds.class_indices.items()}
predictions = [inv_map[idx] for idx in predicted_class_indices]

results = pd.DataFrame(
    {
        "image_id": test_df["filename"].apply(os.path.basename),
        "label": predictions,
    }
)
submission_path = "submission.csv"
results.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")
results.head()




## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/496956709.py in <cell line: 0>()
----> 1 predicted_class_indices = np.argmax(predict, axis=1)
      2 inv_map = {v: k for k, v in train_ds.class_indices.items()}
      3 predictions = [inv_map[idx] for idx in predicted_class_indices]
      4 
      5 results = pd.DataFrame(

NameError: name 'predict' is not defined

## === cell 12
print(results["label"].value_counts())

## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/283155104.py in <cell line: 0>()
----> 1 print(results["label"].value_counts())

NameError: name 'results' is not defined
