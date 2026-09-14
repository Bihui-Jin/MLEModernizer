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

0.9827188940092166

# 6. Current score

0.16833

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.16679) has done: 'The fix removes the broken external model loading and undefined objects, replaces them with a lightweight transfer‑learning pipeline using EfficientNetB0, and ensures the script writes a proper `model_submission_v23.csv` containing the required `image_id` and `label` columns. This resolves all import and name errors, creates functional data generators, trains a quick model, generates predictions for the test set, and produces a valid submission file.'
- What this solution (achieved 0.06341) has done: 'The changes increase data‑loading parallelism and use a larger batch size, which cuts the number of training steps and speeds up each epoch without altering the model architecture, loss, or training schedule. The same ImageDataGenerator logic is kept, and the only added arguments (`workers`, `use_multiprocessing`) only affect how data is read, preserving exact model semantics.'
- What this solution (achieved 0.16833) has done: 'The changes increase data‑loading parallelism and configure TensorFlow’s thread pools, which removes the main CPU bottleneck while keeping the model architecture, training schedule, and augmentation identical. We add explicit `workers` and `use_multiprocessing` arguments to the `flow_from_dataframe` calls and to `model.fit`, set a reproducible seed for the generators, and limit TensorFlow thread usage to the available CPU cores. These adjustments speed up image preprocessing and feeding without altering any core learning logic, preserving the exact model and results.'

# 9. Code solution

## === cell 0
import os
import multiprocessing

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split

num_threads = multiprocessing.cpu_count()
tf.config.threading.set_intra_op_parallelism_threads(num_threads)
tf.config.threading.set_inter_op_parallelism_threads(num_threads)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
base_path = "../input/paddy-disease-classification"
train_csv_path = os.path.join(base_path, "train.csv")
test_csv_path = os.path.join(base_path, "sample_submission.csv")
train_images_dir = os.path.join(base_path, "train_images")
test_images_dir = os.path.join(base_path, "test_images")

train_df = pd.read_csv(train_csv_path)
test_df = pd.read_csv(test_csv_path)



## === cell 2
train_df["filepath"] = train_df.apply(
    lambda row: os.path.join(train_images_dir, row["label"], row["image_id"]), axis=1
)
test_df["filepath"] = test_df["image_id"].apply(
    lambda x: os.path.join(test_images_dir, x)
)

train_df = train_df[train_df["filepath"].apply(os.path.exists)].reset_index(drop=True)
test_df = test_df[test_df["filepath"].apply(os.path.exists)].reset_index(drop=True)

label_list = sorted(train_df["label"].unique())
label_to_idx = {label: idx for idx, label in enumerate(label_list)}
idx_to_label = {idx: label for label, idx in label_to_idx.items()}



## === cell 3
train_df_split, val_df_split = train_test_split(
    train_df, test_size=0.1, stratify=train_df["label"], random_state=42
)



## === cell 4
img_size = 224
batch_sz = 32  # moderate batch size for compatibility

train_gen = ImageDataGenerator(
    rescale=1.0 / 255,
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=15,
    zoom_range=0.1,
)

val_gen = ImageDataGenerator(rescale=1.0 / 255)
test_gen = ImageDataGenerator(rescale=1.0 / 255)



## === cell 5
num_workers = min(8, multiprocessing.cpu_count())

train_flow = train_gen.flow_from_dataframe(
    dataframe=train_df_split,
    x_col="filepath",
    y_col="label",
    classes=label_list,
    target_size=(img_size, img_size),
    batch_size=batch_sz,
    class_mode="categorical",
    shuffle=True,
    seed=42,
    workers=num_workers,
    use_multiprocessing=True,
    max_queue_size=10,
)

val_flow = val_gen.flow_from_dataframe(
    dataframe=val_df_split,
    x_col="filepath",
    y_col="label",
    classes=label_list,
    target_size=(img_size, img_size),
    batch_size=batch_sz,
    class_mode="categorical",
    shuffle=False,
    seed=42,
    workers=num_workers,
    use_multiprocessing=True,
    max_queue_size=10,
)



## === cell 6
base_model = EfficientNetB0(
    include_top=False, weights="imagenet", input_shape=(img_size, img_size, 3)
)
base_model.trainable = False  # freeze initially

x = GlobalAveragePooling2D()(base_model.output)
output = Dense(len(label_list), activation="softmax")(x)
model = Model(inputs=base_model.input, outputs=output)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])



## === cell 7
model.fit(
    train_flow,
    epochs=5,
    validation_data=val_flow,
    verbose=1,
    workers=num_workers,
    use_multiprocessing=True,
)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3377116832.py in <cell line: 0>()
----> 1 model.fit(
      2     train_flow,
      3     epochs=5,
      4     validation_data=val_flow,
      5     verbose=1,

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

## === cell 8
base_model.trainable = True
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.fit(
    train_flow,
    epochs=10,
    validation_data=val_flow,
    verbose=1,
    workers=num_workers,
    use_multiprocessing=True,
)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1918488043.py in <cell line: 0>()
      6 )
      7 
----> 8 model.fit(
      9     train_flow,
     10     epochs=10,

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

## === cell 9
test_flow = test_gen.flow_from_dataframe(
    dataframe=test_df,
    x_col="filepath",
    y_col=None,
    target_size=(img_size, img_size),
    batch_size=batch_sz,
    class_mode=None,
    shuffle=False,
    seed=42,
    workers=num_workers,
    use_multiprocessing=True,
    max_queue_size=10,
)



## === cell 10
pred_probs = model.predict(test_flow, verbose=1)
pred_indices = np.argmax(pred_probs, axis=1)
pred_labels = [idx_to_label[idx] for idx in pred_indices]

submission = pd.DataFrame({"image_id": test_df["image_id"], "label": pred_labels})
submission_path = "model_submission_v23.csv"
submission.to_csv(submission_path, index=False)

print(f"Submission saved to {submission_path}")
