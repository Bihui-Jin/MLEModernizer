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
Create a classifier to predict whether an image contains a cactus.

## Metric
Area under the ROC curve.

## Submission Format
For each ID in the test set, you must predict a probability for the `has_cactus` variable. The file should contain a header and have the following format:

```
id,has_cactus
000940378805c44108d287872b2f04ce.jpg,0.5
0017242f54ececa4512b4d7937d1e21e.jpg,0.5
001ee6d8564003107853118ab87df407.jpg,0.5
etc.
```

## Dataset
This dataset contains a large number of 32 x 32 thumbnail images containing aerial photos of a cactus. The file name of an image corresponds to its `id`.

- **train/** - the training set images
- **test/** - the test set images (you must predict the labels of these)
- **train.csv** - the training set labels, indicates whether the image has a cactus (`has_cactus = 1`)
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.7

# 3. Installed packages

geopandas==0.14.4
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
pillow==11.3.0
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
scipy==1.15.3
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
tqdm==4.67.1

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
        input/
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
        working/
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
```

-> data/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> data/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> (stopped after 10 files for performance)

# 5. Target score

0.7868

# 6. Current score

0.99134

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.9981) has done: 'The fix loads images from the correct directories, avoids the incompatible custom TensorFlow layer, sets an environment variable to work around the protobuf issue, builds a simple CNN, trains it, evaluates AUC on a validation split, and finally creates a proper `sample_submission.csv` with the required columns.'
- What this solution (achieved 0.99703) has done: 'I move the environment‑variable fix (`PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION`) to **before** TensorFlow is imported, which resolves the `MessageFactory` AttributeError that prevented the script from running. No other logic is changed, so the model, training, and validation remain identical, keeping the current high AUC score while producing a proper `sample_submission.csv`.'
- What this solution (achieved 0.99911) has done: 'I move the protobuf environment‑variable fix to the very top of the script (before any library imports) so TensorFlow loads without the `MessageFactory` error, and keep the rest of the logic unchanged. This guarantees the notebook runs end‑to‑end and still writes a correct `sample_submission.csv`, preserving the high AUC score that already exceeds the target.'
- What this solution (achieved 0.9979) has done: 'The fix moves the protobuf environment variable assignment to the very top of the notebook and forces the value, guaranteeing TensorFlow loads without the `MessageFactory` error. All subsequent imports and the original training‑inference pipeline remain unchanged, so the model’s high AUC is preserved while a proper `sample_submission.csv` is written.'
- What this solution (achieved 0.9926) has done: 'The fix moves the protobuf environment‑variable assignment to the very first lines of the script (before **any** library imports) so TensorFlow loads without the `MessageFactory` error, and then runs the original training, evaluation, and submission pipeline unchanged. This restores end‑to‑end execution and produces a valid `sample_submission.csv` while keeping the existing high AUC score.'
- What this solution (achieved 0.99853) has done: 'The fix moves the protobuf environment‑variable assignment to the very first lines of the script (before any imports) to guarantee TensorFlow loads without the `MessageFactory` error, and renumbers the cells starting at 1 while keeping the original logic unchanged. No model changes are made, preserving the high AUC, and the script now reliably writes a correct `sample_submission.csv`.'
- What this solution (achieved 0.99676) has done: 'The fix moves the protobuf environment‑variable assignment to the very top of the script, before any other imports, guaranteeing TensorFlow loads without the `MessageFactory` error. No other logic is changed, so the high AUC score is preserved and a correct `sample_submission.csv` is written.'
- What this solution (achieved 0.99771) has done: 'The fix moves the protobuf environment‑variable setting to the very top of the script (before any imports) and consolidates the initial import block into the first cell, then renumbers all cells starting at 1. This eliminates the `MessageFactory` error, ensures TensorFlow loads correctly, and retains the original model and training logic so the high AUC score is preserved while producing a proper `sample_submission.csv`.'
- What this solution (achieved 0.99134) has done: 'We move the protobuf‑environment fix to the very first cell, before any library (including TensorFlow) is imported, and then keep the original pipeline unchanged. This eliminates the `MessageFactory` import error while preserving the high AUC model, and we explicitly write the submission CSV to the current working directory so it is correctly saved.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"



## === cell 1
import warnings
import numpy as np, pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    BatchNormalization,
    Activation,
    Dropout,
)
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

warnings.filterwarnings("ignore")



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
possible_paths = [
    "../input/aerial-cactus-identification",
    "../input",
    "../../input",
    "/kaggle/input/aerial-cactus-identification",
    "/kaggle/input",
]
base_path = next((p for p in possible_paths if os.path.isdir(p)), None)
if base_path is None:
    raise FileNotFoundError("Base input directory not found.")

train_img_dir = os.path.join(base_path, "train")
test_img_dir = os.path.join(base_path, "test")
train_csv_path = os.path.join(base_path, "train.csv")
sample_sub_path = os.path.join(base_path, "sample_submission.csv")




## === cell 3
def load_image(filepath):
    img = tf.keras.preprocessing.image.load_img(filepath, target_size=(32, 32))
    arr = tf.keras.preprocessing.image.img_to_array(img)
    return arr / 255.0  # normalize to [0,1]




## === cell 4
train_df = pd.read_csv(train_csv_path)
X = []
y = train_df["has_cactus"].values.astype("float32")
for img_id in tqdm(train_df["id"], desc="Loading train images"):
    img_path = os.path.join(train_img_dir, img_id)
    X.append(load_image(img_path))
X = np.stack(X, axis=0)  # (n_samples, 32, 32, 3)



## === cell 5
sample_sub = pd.read_csv(sample_sub_path)
X_test = []
for img_id in tqdm(sample_sub["id"], desc="Loading test images"):
    img_path = os.path.join(test_img_dir, img_id)
    X_test.append(load_image(img_path))
X_test = np.stack(X_test, axis=0)



## === cell 6
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)




## === cell 7
def build_cnn(input_shape=(32, 32, 3)):
    model = Sequential(
        [
            Conv2D(32, (3, 3), padding="same", input_shape=input_shape),
            BatchNormalization(),
            Activation("relu"),
            MaxPooling2D(),
            Conv2D(64, (3, 3), padding="same"),
            BatchNormalization(),
            Activation("relu"),
            MaxPooling2D(),
            Conv2D(128, (3, 3), padding="same"),
            BatchNormalization(),
            Activation("relu"),
            MaxPooling2D(),
            Flatten(),
            Dropout(0.5),
            Dense(1, activation="sigmoid"),
        ]
    )
    return model


model = build_cnn()
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=[tf.keras.metrics.AUC(name="auc")],
)



## === cell 8
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=8,
    batch_size=64,
    verbose=2,
)



## === cell 9
val_preds = model.predict(X_val).ravel()
val_auc = roc_auc_score(y_val, val_preds)
print(f"Validation AUC: {val_auc:.5f}")



## === cell 10
test_preds = model.predict(X_test).ravel()
submission = pd.DataFrame({"id": sample_sub["id"], "has_cactus": test_preds})
submission_path = os.path.join(os.getcwd(), "sample_submission.csv")
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
