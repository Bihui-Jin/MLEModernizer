# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os
import random
import multiprocessing

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np
import pandas as pd

try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    TF_AVAILABLE = True
except Exception as e:
    print("TensorFlow import failed:", e)
    TF_AVAILABLE = False

if TF_AVAILABLE:
    tf.config.optimizer.set_jit(True)  # enable XLA
    cpu_count = multiprocessing.cpu_count()
    tf.config.threading.set_intra_op_parallelism_threads(cpu_count)
    tf.config.threading.set_inter_op_parallelism_threads(cpu_count)

seed = 42
if TF_AVAILABLE:
    tf.random.set_seed(seed)
np.random.seed(seed)
random.seed(seed)

BASE_DIR = "/kaggle/input/paddy-disease-classification"
TRAIN_IMG_DIR = os.path.join(BASE_DIR, "train_images")
TEST_IMG_DIR = os.path.join(BASE_DIR, "test_images")
TRAIN_CSV = os.path.join(BASE_DIR, "train.csv")



## === cell 1
train_df = pd.read_csv(TRAIN_CSV)
train_df["label"] = train_df["label"].astype(str)

labels = sorted(train_df["label"].unique())
class_indices = {label: idx for idx, label in enumerate(labels)}
inverse_map = {idx: label for label, idx in class_indices.items()}

train_df["filepath"] = (
    TRAIN_IMG_DIR + "/" + train_df["label"] + "/" + train_df["image_id"]
)



## === cell 2
if TF_AVAILABLE:
    train_datagen = ImageDataGenerator(
        validation_split=0.1,
        horizontal_flip=True,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
    )

    BATCH_SIZE = 512
    WORKERS = cpu_count

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col="filepath",
        y_col="label",
        target_size=(299, 299),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=labels,
        subset="training",
        shuffle=True,
        seed=seed,
        workers=WORKERS,
        use_multiprocessing=False,
        max_queue_size=20,
    )

    val_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col="filepath",
        y_col="label",
        target_size=(299, 299),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=labels,
        subset="validation",
        shuffle=False,
        seed=seed,
        workers=WORKERS,
        use_multiprocessing=False,
        max_queue_size=20,
    )
else:
    train_generator = None
    val_generator = None



## === cell 3
if TF_AVAILABLE:
    base_model = tf.keras.applications.Xception(
        weights="imagenet", include_top=False, input_shape=(299, 299, 3), pooling="avg"
    )

    inputs = tf.keras.Input(shape=(299, 299, 3))
    x = tf.keras.applications.xception.preprocess_input(inputs)
    x = base_model(x, training=False)
    outputs = tf.keras.layers.Dense(len(labels), activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(
        train_generator,
        epochs=5,
        validation_data=val_generator,
        verbose=1,
    )
else:
    most_common_label = train_df["label"].value_counts().idxmax()
    model = None



## === cell 4
test_files = sorted(
    [
        f
        for f in os.listdir(TEST_IMG_DIR)
        if os.path.isfile(os.path.join(TEST_IMG_DIR, f)) and f.lower().endswith(".jpg")
    ]
)
test_df = pd.DataFrame({"image_id": test_files})
test_df["filepath"] = TEST_IMG_DIR + "/" + test_df["image_id"]

if TF_AVAILABLE:
    test_datagen = ImageDataGenerator()
    test_generator = test_datagen.flow_from_dataframe(
        dataframe=test_df,
        x_col="filepath",
        y_col=None,
        target_size=(299, 299),
        batch_size=512,  # match training batch size for efficiency
        class_mode=None,
        shuffle=False,
        workers=cpu_count,
        use_multiprocessing=False,
        max_queue_size=20,
    )

    pred_probs = model.predict(
        test_generator,
        verbose=1,
    )
    pred_labels_idx = np.argmax(pred_probs, axis=1)
    pred_labels = [inverse_map[idx] for idx in pred_labels_idx]
else:
    pred_labels = [most_common_label] * len(test_df)



## === cell 5
submission = pd.DataFrame(
    {"image_id": test_df["image_id"].values, "label": pred_labels}
)
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
