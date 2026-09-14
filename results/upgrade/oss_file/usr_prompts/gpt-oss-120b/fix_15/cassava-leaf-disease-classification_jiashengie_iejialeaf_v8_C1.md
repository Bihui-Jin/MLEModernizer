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

3.12

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

0.8631006346328196

# 6. Current score

0.59454

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.08146) has done: 'I speed up the script by (1) replacing the slow per‑file copy with fast hard‑links (using `os.link` and a thread pool) to build the directory structure, and (2) batching the test‑image inference instead of predicting one image at a time. Both changes keep the exact same data that the model sees and produce identical predictions, while dramatically reducing I/O and Python‑loop overhead, allowing the whole pipeline to finish well under the 600 s limit.'
- What this solution (achieved 0.5867) has done: 'The main slowdown was the massive copying of all training images into a new folder hierarchy, which took many minutes. By replacing the copy operation with lightweight symbolic links (or a fallback copy when symlinks aren’t supported), we avoid the I/O bottleneck while keeping the directory structure required by `image_dataset_from_directory`. This change preserves all downstream logic and model training behavior, but dramatically reduces preparation time, allowing the whole script to finish well within the 600‑second limit.'
- What this solution (achieved 0.108) has done: 'I added a protobuf compatibility fix before importing TensorFlow, ensured the temporary dataset folder is freshly created to avoid stale or broken symlinks, and switched the loss to the standard categorical cross‑entropy which aligns with the softmax output and improves classification accuracy. These minimal changes resolve the import error, prevent the “file not found” issue during training, and help move the Kaggle score toward the target while keeping the original model architecture intact.'
- What this solution (achieved 0.5355) has done: 'We speed up the pipeline by fixing two costly spots while keeping the exact model type and data handling.  
1. Set a deterministic NumPy seed (no effect on logic but ensures reproducibility).  
2. Switch the logistic regression solver from the stochastic “saga” (slow on dense 12k‑feature data) to the dense‑optimised “lbfgs”, removing the unsupported `n_jobs` argument – this dramatically reduces training time without changing the model class or regularisation.  
All other code, including image loading, preprocessing, and prediction, remains unchanged.'
- What this solution (achieved 0.59454) has done: 'The update replaces the simple LogisticRegression with a lightweight convolutional neural network, normalizes pixel values, and uses TensorFlow/Keras for training—these changes are needed because the current gap (≈38 %) exceeds the 30 % threshold, allowing a model change. The new CNN is small enough to finish quickly while giving a much higher expected accuracy, moving the score toward the target. All other pipeline steps (image loading, validation split, and submission creation) remain unchanged.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import concurrent.futures
import tensorflow as tf

np.random.seed(42)
tf.random.set_seed(42)

WORK_DIR = "../input/cassava-leaf-disease-classification/"
TRAIN_IMG_DIR = os.path.join(WORK_DIR, "train_images")
TEST_IMG_DIR = os.path.join(WORK_DIR, "test_images")
OUTPUT_DIR = "./"
os.makedirs(OUTPUT_DIR, exist_ok=True)

train_df = pd.read_csv(os.path.join(WORK_DIR, "train.csv"))
print(f"Training samples: {len(train_df)}")




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def load_and_preprocess_image(path, size=(64, 64)):
    """Load an image, convert to RGB, resize, and return as a 1‑D float32 array."""
    with Image.open(path) as img:
        img = img.convert("RGB")
        img = img.resize(size, Image.BILINEAR)
        return np.asarray(img, dtype=np.float32).flatten()


img_size = 64 * 64 * 3

valid_paths = [
    os.path.join(TRAIN_IMG_DIR, img_id)
    for img_id in train_df["image_id"]
    if os.path.isfile(os.path.join(TRAIN_IMG_DIR, img_id))
]
valid_labels = [
    label
    for img_id, label in zip(train_df["image_id"], train_df["label"])
    if os.path.isfile(os.path.join(TRAIN_IMG_DIR, img_id))
]

missing = len(train_df) - len(valid_paths)
if missing:
    print(f"Warning: {missing} training images were not found and skipped.")

n_samples = len(valid_paths)
X = np.empty((n_samples, img_size), dtype=np.float32)

with concurrent.futures.ThreadPoolExecutor(
    max_workers=min(32, os.cpu_count() * 2)
) as executor:
    for idx, arr in enumerate(executor.map(load_and_preprocess_image, valid_paths)):
        X[idx] = arr

y = np.array(valid_labels)

le = LabelEncoder()
y_enc = le.fit_transform(y)

X_train, X_val, y_train, y_val = train_test_split(
    X, y_enc, test_size=0.1, random_state=123, stratify=y_enc
)




## === cell 2
def reshape_and_normalize(arr):
    return (arr.reshape(64, 64, 3) / 255.0).astype(np.float32)


X_train_img = np.stack([reshape_and_normalize(row) for row in X_train])
X_val_img = np.stack([reshape_and_normalize(row) for row in X_val])

num_classes = len(le.classes_)
y_train_oh = tf.keras.utils.to_categorical(y_train, num_classes)
y_val_oh = tf.keras.utils.to_categorical(y_val, num_classes)

model = tf.keras.Sequential(
    [
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(64, 64, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ]
)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

model.fit(
    X_train_img,
    y_train_oh,
    validation_data=(X_val_img, y_val_oh),
    epochs=5,
    batch_size=64,
    verbose=1,
)

val_loss, val_acc = model.evaluate(X_val_img, y_val_oh, verbose=0)
print(f"Validation accuracy (approx.): {val_acc:.4f}")



## === cell 3
test_files = sorted(
    [f for f in os.listdir(TEST_IMG_DIR) if f.lower().endswith((".jpg", ".png"))]
)

batch_size = 256  # process in batches to limit memory use
pred_labels = []

with concurrent.futures.ThreadPoolExecutor(
    max_workers=min(32, os.cpu_count() * 2)
) as executor:
    for i in range(0, len(test_files), batch_size):
        batch = test_files[i : i + batch_size]
        batch_paths = [os.path.join(TEST_IMG_DIR, f) for f in batch]

        batch_imgs = np.stack(
            [
                reshape_and_normalize(arr)
                for arr in executor.map(load_and_preprocess_image, batch_paths)
            ]
        )
        batch_probs = model.predict(batch_imgs, verbose=0)
        batch_preds = np.argmax(batch_probs, axis=1)
        pred_labels.extend(batch_preds.tolist())

pred_labels_original = le.inverse_transform(pred_labels)

submission = pd.DataFrame({"image_id": test_files, "label": pred_labels_original})
submission_path = os.path.join(OUTPUT_DIR, "submission.csv")
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")
