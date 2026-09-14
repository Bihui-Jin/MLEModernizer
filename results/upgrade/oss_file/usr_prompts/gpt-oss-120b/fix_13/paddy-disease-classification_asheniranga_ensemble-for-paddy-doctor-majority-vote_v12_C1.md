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

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import mixed_precision

if tf.config.list_physical_devices("GPU"):
    mixed_precision.set_global_policy("mixed_float16")
else:
    mixed_precision.set_global_policy("float32")

tf.config.optimizer.set_jit(True)
tf.random.set_seed(42)
tf.config.threading.set_intra_op_parallelism_threads(8)
tf.config.threading.set_inter_op_parallelism_threads(8)



## === cell 1
train_dir = "../input/paddy-disease-classification/train_images"
test_dir = "../input/paddy-disease-classification/test_images"

full_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=(256, 256),
    batch_size=256,
    label_mode="int",
    shuffle=True,
    seed=42,
)

class_names = full_ds.class_names
num_classes = len(class_names)

AUTOTUNE = tf.data.AUTOTUNE
full_ds = full_ds.cache().prefetch(buffer_size=AUTOTUNE)

num_examples = tf.data.experimental.cardinality(full_ds).numpy()
train_size = int(0.8 * num_examples)

train_ds = full_ds.take(train_size)
val_ds = full_ds.skip(train_size)



## === cell 2
base_model = EfficientNetB0(
    weights="imagenet", include_top=False, input_shape=(256, 256, 3)
)
base_model.trainable = False  # freeze backbone for quick training

model = tf.keras.Sequential(
    [
        base_model,
        GlobalAveragePooling2D(),
        Dropout(0.2),
        Dense(num_classes, activation="softmax"),
    ]
)

model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.fit(train_ds, validation_data=val_ds, epochs=12, verbose=2)



## === cell 3
test_image_paths = []
for root, _, files in os.walk(test_dir):
    for f in files:
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            test_image_paths.append(os.path.join(root, f))
test_image_paths = sorted(test_image_paths)


def _load_and_preprocess(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [256, 256])
    img = tf.cast(img, tf.float32) / 255.0
    return img


test_ds = tf.data.Dataset.from_tensor_slices(test_image_paths)
test_ds = test_ds.map(_load_and_preprocess, num_parallel_calls=AUTOTUNE)
test_ds = test_ds.batch(256).prefetch(AUTOTUNE)

probs = model.predict(test_ds, verbose=0)
pred_indices = np.argmax(probs, axis=1)
pred_labels = [class_names[idx] for idx in pred_indices]

image_ids = [os.path.basename(p) for p in test_image_paths]
submission = pd.DataFrame({"image_id": image_ids, "label": pred_labels})

sample_sub_path = "../input/paddy-disease-classification/sample_submission.csv"
if os.path.exists(sample_sub_path):
    sample_sub = pd.read_csv(sample_sub_path)
    submission = (
        submission.set_index("image_id").reindex(sample_sub["image_id"]).reset_index()
    )



## === cell 4
submission_path = "model_submission_v12.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")
