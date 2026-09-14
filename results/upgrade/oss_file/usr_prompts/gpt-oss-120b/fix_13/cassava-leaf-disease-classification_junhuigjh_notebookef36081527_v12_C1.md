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

3.13

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

0.804472650347537

# 6. Current score

0.30007

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.39836) has done: 'I wrap the TensorFlow model construction and training in a safe try/except block so that if any TensorFlow‑related error occurs (such as the protobuf incompatibility you saw), the code automatically fall back to the scikit‑learn LogisticRegression pipeline. This guarantees that a `model` object is always created and that the later prediction cell can run without a `NameError`, producing a valid `submission.csv` file.'
- What this solution (achieved 0.39836) has done: 'Implemented a protobuf compatibility fix by setting the appropriate environment variable before importing TensorFlow, allowing the ResNet‑based CNN to run instead of falling back to LogisticRegression. Added a small increase in training epochs for better accuracy while keeping the original model architecture unchanged.'
- What this solution (achieved 0.62257) has done: 'Implemented fixes to resolve runtime errors and improve model performance:

- Added missing import for `StandardScaler`.
- Adjusted PCA to retain more components (`n_components=200`) for richer feature representation.
- Increased logistic regression regularization strength (`C=10.0`) to better fit the data.
- Minor comment updates for clarity.

These changes ensure the script runs without errors, produces a valid `submission.csv`, and modestly boosts validation accuracy toward the target score.'
- What this solution (achieved 0.61622) has done: 'I keep the overall structure unchanged but adjust the sklearn fallback to use a richer feature representation and a more appropriately regularized classifier. By increasing the PCA component count to 500 we retain more image information, and setting the LogisticRegression `C` to 1.0 with a higher `max_iter` helps the model converge better without over‑fitting, which should raise validation accuracy toward the target while preserving the original pipeline logic.'
- What this solution (achieved 0.29073) has done: 'I keep the overall structure and fallback to the sklearn pipeline, but improve the classifier’s capacity and the amount of retained image information.  
- Increase the PCA dimensionality from 500 to 800 so more visual variance is kept.  
- Strengthen the logistic‑regression model by using a larger inverse‑regularisation C (10.0) and more optimisation steps (max_iter 2000).  
- Add `class_weight='balanced'` to help with any label imbalance.  
These minimal changes stay within the original pipeline while likely raising validation accuracy toward the target score.'
- What this solution (achieved 0.29073) has done: 'I enable the TensorFlow branch (which uses a pretrained ResNet‑50) by setting `tf_available = True` and give it a slightly longer training run (EPOCHS = 8). This minimal change keeps the overall pipeline unchanged, but lets the more powerful CNN model train when TensorFlow is present, which should raise validation accuracy toward the target score while still falling back to the sklearn pipeline if TensorFlow fails.'
- What this solution (achieved 0.29073) has done: 'I set the protobuf implementation environment variable before importing TensorFlow so the ResNet‑based CNN can be built instead of falling back to the weaker sklearn pipeline. This fixes the `'MessageFactory' object has no attribute 'GetPrototype'` error, allowing the more powerful model to train and improve validation accuracy toward the target while keeping the original workflow unchanged.'
- What this solution (achieved 0.30007) has done: 'I add a lightweight image‑augmentation step (horizontal flip) to boost the amount of training data, increase the PCA component count to retain more visual information, and keep the existing fallback pipeline. These changes stay within the original logic, prevent the TensorFlow error from breaking execution, and should improve validation accuracy toward the target score while still producing a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from PIL import Image

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

tf_available = True

BASE_PATH = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_IMG_DIR = os.path.join(BASE_PATH, "train_images")
TEST_IMG_DIR = os.path.join(BASE_PATH, "test_images")
TRAIN_CSV = os.path.join(BASE_PATH, "train.csv")
SAMPLE_SUB = os.path.join(BASE_PATH, "sample_submission.csv")

IMG_SIZE = (64, 64)  # small size for quick training
BATCH_SIZE = 32
EPOCHS = 8  # slightly longer training to improve performance
NUM_CLASSES = 5


def load_image(path):
    img = Image.open(path).convert("RGB")
    img = img.resize(IMG_SIZE)
    return np.asarray(img) / 255.0  # scale to [0,1]


train_df = pd.read_csv(TRAIN_CSV)
train_df["filepath"] = train_df["image_id"].apply(
    lambda x: os.path.join(TRAIN_IMG_DIR, x)
)

print("Loading training images …")
X_train = np.stack([load_image(p) for p in train_df["filepath"].values])
y_train = train_df["label"].values

print("Applying lightweight augmentation …")
X_flipped = np.flip(X_train, axis=2)  # flip width dimension (horizontal)
y_flipped = y_train.copy()

X_train = np.concatenate([X_train, X_flipped], axis=0)
y_train = np.concatenate([y_train, y_flipped], axis=0)

model = None  # placeholder to guarantee definition

if tf_available:
    try:
        import tensorflow as tf
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        from tensorflow.keras.applications import ResNet50
        from tensorflow.keras import layers, models
        from tensorflow.keras.optimizers import Adam

        datagen = ImageDataGenerator(
            rescale=1.0,
            validation_split=0.1,
            horizontal_flip=True,
            rotation_range=20,
            zoom_range=0.2,
        )

        train_gen = datagen.flow_from_dataframe(
            train_df,
            x_col="filepath",
            y_col="label",
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode="categorical",
            subset="training",
            shuffle=True,
            seed=42,
        )
        val_gen = datagen.flow_from_dataframe(
            train_df,
            x_col="filepath",
            y_col="label",
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode="categorical",
            subset="validation",
            shuffle=False,
        )

        base_model = ResNet50(
            weights="imagenet",
            include_top=False,
            input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        )
        base_model.trainable = False

        inputs = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
        model = models.Model(inputs, outputs)

        model.compile(
            optimizer=Adam(learning_rate=1e-3),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=EPOCHS,
            verbose=1,
        )
        print("TensorFlow model trained successfully.")
    except Exception as e:
        print("TensorFlow model failed – switching to sklearn fallback. Reason:", e)
        tf_available = False

if not tf_available:
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X_flat = X_train.reshape(len(X_train), -1)

    pca = PCA(
        n_components=1000,  # increased from 800
        svd_solver="randomized",
        random_state=42,
    )

    lr = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        max_iter=3000,
        C=10.0,
        n_jobs=-1,
        random_state=42,
        class_weight="balanced",
    )

    model = make_pipeline(
        StandardScaler(),
        pca,
        lr,
    )
    print("Training enhanced sklearn fallback model …")
    model.fit(X_flat, y_train)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
test_df = pd.read_csv(SAMPLE_SUB)
test_df["filepath"] = test_df["image_id"].apply(lambda x: os.path.join(TEST_IMG_DIR, x))

print("Loading test images …")
X_test = np.stack([load_image(p) for p in test_df["filepath"].values])

if tf_available:
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0)
    test_gen = test_datagen.flow(X_test, batch_size=BATCH_SIZE, shuffle=False)
    pred_probs = model.predict(test_gen, verbose=0)
    pred_labels = np.argmax(pred_probs, axis=1)
else:
    X_test_flat = X_test.reshape(len(X_test), -1)
    pred_labels = model.predict(X_test_flat)

submission = pd.DataFrame(
    {"image_id": test_df["image_id"], "label": pred_labels.astype(int)}
)
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission file written to {submission_path} – shape: {submission.shape}")
