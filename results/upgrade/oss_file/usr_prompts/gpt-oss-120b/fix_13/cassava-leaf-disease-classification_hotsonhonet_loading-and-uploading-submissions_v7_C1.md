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

0.8245693563009973

# 6. Current score

0.61099

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.10762) has done: 'The fixes address the import error by setting the protobuf implementation before loading TensorFlow, replace the missing pretrained model with a lightweight EfficientNet‑B0 that is built and briefly trained on the provided data, correct path handling, and ensure all required libraries (e.g., pandas, cv2) are imported only when needed. The script now creates proper data generators, trains a model for a few epochs, evaluates accuracy, generates predictions for the test set, and writes a correctly‑formatted `submission.csv` file.'
- What this solution (achieved 0.56091) has done: 'Implemented a faster solver for the logistic regression model. Switching from the `saga` solver (optimized for sparse data) to the `lbfgs` solver (efficient for dense matrices) dramatically reduces training time while preserving the same model type and hyper‑parameters, keeping prediction accuracy unchanged.'
- What this solution (achieved 0.43797) has done: 'I add feature scaling, increase the image resolution to capture more detail, and allow the logistic regression more iterations to converge; these small preprocessing tweaks keep the same model while helping validation accuracy move nearer the target.'
- What this solution (achieved 0.31129) has done: 'I add a PCA dimensionality‑reduction step (200 components) after scaling, increase the logistic‑regression regularisation strength (C=4.0) and iterations (max_iter=500), and enable class‑weight balancing – these modest changes keep the overall linear‑model pipeline while giving the classifier more expressive power and better‑tuned regularisation, which should lift validation accuracy toward the target. The rest of the script and file handling remain unchanged.'
- What this solution (achieved 0.61099) has done: 'We replace the linear logistic‑regression pipeline with a small convolutional neural network that works directly on the resized RGB images. Using a deeper model (EfficientNet‑B0 base) and training for a few epochs typically lifts validation accuracy far above the current 0.31, moving it toward the 0.82 target while preserving the overall data‑loading and split logic.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf

tf.random.set_seed(42)
np.random.seed(42)

os.environ["OMP_NUM_THREADS"] = "1"

BASE_DIR = "/kaggle/input/cassava-leaf-disease-classification"

TRAIN_IMG_LOC = os.path.join(BASE_DIR, "train_images")
TEST_IMG_LOC = os.path.join(BASE_DIR, "test_images")
TRAIN_CSV = os.path.join(BASE_DIR, "train.csv")
SAMPLE_SUBMISSION_CSV = os.path.join(BASE_DIR, "sample_submission.csv")

print("Paths and libraries configured.")


## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_df = pd.read_csv(TRAIN_CSV)
train_df["filepath"] = train_df["image_id"].apply(
    lambda x: os.path.join(TRAIN_IMG_LOC, x)
)

train_split, val_split = train_test_split(
    train_df,
    test_size=0.1,
    stratify=train_df["label"],
    random_state=42,
)

print(f"Train samples: {len(train_split)}, Validation samples: {len(val_split)}")


## === cell 2
IMG_SIZE = (112, 112)  # modest size that keeps memory reasonable


def _load_and_process(path):
    """Open, resize, convert to RGB, and normalize a single image."""
    with Image.open(path) as img:
        img = img.convert("RGB").resize(IMG_SIZE)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return arr


_MAX_WORKERS = os.cpu_count() or 1


def load_images(df, img_col="filepath"):
    """Load images in parallel and return a 4‑D NumPy array."""
    from concurrent.futures import ThreadPoolExecutor

    paths = df[img_col].tolist()
    n = len(paths)

    sample_arr = _load_and_process(paths[0])
    dtype = sample_arr.dtype
    height, width, channels = sample_arr.shape

    data = np.empty((n, height, width, channels), dtype=dtype)

    chunksize = 128

    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as executor:
        for idx, arr in enumerate(
            executor.map(_load_and_process, paths, chunksize=chunksize)
        ):
            data[idx] = arr

    return data


X_train = load_images(train_split)
y_train = train_split["label"].values

X_val = load_images(val_split)
y_val = val_split["label"].values

print(f"Image arrays shaped – X_train: {X_train.shape}, X_val: {X_val.shape}")

base_model = tf.keras.applications.EfficientNetB0(
    include_top=False, weights="imagenet", input_shape=(*IMG_SIZE, 3), pooling="avg"
)
base_model.trainable = False  # freeze pretrained weights

model = tf.keras.Sequential(
    [base_model, tf.keras.layers.Dense(5, activation="softmax")]
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.fit(
    X_train, y_train, validation_data=(X_val, y_val), epochs=5, batch_size=32, verbose=2
)

val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
print(f"Validation accuracy: {val_acc:.4f}")


## === cell 3
test_submission = pd.read_csv(SAMPLE_SUBMISSION_CSV)
test_submission["filepath"] = test_submission["image_id"].apply(
    lambda x: os.path.join(TEST_IMG_LOC, x)
)

X_test = load_images(test_submission)

test_preds_prob = model.predict(X_test, batch_size=32, verbose=0)
test_preds = np.argmax(test_preds_prob, axis=1)

submission = pd.DataFrame(
    {"image_id": test_submission["image_id"], "label": test_preds}
)
submission.to_csv("submission.csv", index=False)
print("submission.csv written with", len(submission), "rows.")


## === cell 4
submission.head()
