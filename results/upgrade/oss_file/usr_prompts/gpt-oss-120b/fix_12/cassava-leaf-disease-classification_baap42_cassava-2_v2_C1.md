# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

geopandas==0.14.4
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
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
sklearn-pandas==2.2.0
tf_keras==2.18.0

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

0.8856149894227864

# 6. Current score

0.61099

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.61099) has done: 'The fix removes the failing model loading and replaces it with a simple baseline that predicts the most common training label for every test image. This eliminates the protobuf error, ensures `final_model` is defined (or safely skipped), and creates a submission CSV whose rows match the test set length, fixing the length‑mismatch issue.'
- What this solution (achieved 0.61099) has done: 'I add a lightweight transfer‑learning model (MobileNetV2) that trains for a few epochs on the training images and uses it to predict the test set. If any step fails, the code falls back to the original baseline of predicting the most common label, ensuring the script always writes a valid submission.csv. This small model boost is expected to raise the accuracy toward the target while keeping the original logic intact.'
- What this solution (achieved 0.18423) has done: 'The changes remove the TensorFlow‑based training (which caused the protobuf import error) and replace it with a lightweight centroid‑based classifier built from the resized training images. This avoids the failing dependency, ensures a valid `submission.csv` is always written, and generally improves accuracy over the constant‑label baseline, moving the score toward the target while keeping the original workflow intact.'
- What this solution (achieved 0.69021) has done: 'I replace the simple centroid classifier with a lightweight pretrained MobileNetV2 model that is fine‑tuned for a few epochs on the training images. The code keeps the original fallback to the most‑common label if training fails, preserving the overall workflow while significantly improving accuracy toward the target score. Only the model‑training and prediction sections are changed, and the final CSV is written exactly as required.'
- What this solution (achieved 0.61211) has done: 'I remove the TensorFlow import that triggers a protobuf error and replace the MobileNetV2 CNN with a lightweight scikit‑learn RandomForest classifier trained on the same resized images. This keeps the original data‑handling workflow, adds a functional model that can run in the given environment, and falls back to the most‑common label if anything fails. The changes ensure a valid `submission.csv` is written and should raise validation accuracy toward the target score.'
- What this solution (achieved 0.70142) has done: 'I replace the simple RandomForest with a lightweight transfer‑learning CNN (MobileNetV2) that fine‑tunes on the same 128×128 images. The training block now builds a Keras model, trains it for a few epochs, and uses it for predictions, while keeping the existing fallback to the most‑common label if anything fails. This change is expected to raise validation accuracy markedly toward the target while preserving the overall workflow and output format.'
- What this solution (achieved 0.60688) has done: 'Implemented fixes to eliminate the TensorFlow import error and replaced the failing CNN with a lightweight RandomForest model built on simple image statistics (mean and standard deviation of RGB channels). Added necessary scikit‑learn imports, updated feature extraction for training and inference, and ensured the fallback to the most‑common label still works. The script now runs end‑to‑end and writes a correct `submission.csv` while improving validation accuracy toward the target score.'
- What this solution (achieved 0.71114) has done: 'Implemented a lightweight transfer‑learning pipeline using a pretrained MobileNetV2 backbone instead of the simple statistical‑feature RandomForest. The new model reads and rescales the full images, fine‑tunes a few epochs, and predicts class indices for the test set. All original fallback logic is retained, and the script still writes a correctly‑formatted `submission.csv`. This change is expected to raise validation accuracy substantially toward the target score.'
- What this solution (achieved 0.70553) has done: 'Implemented fixes to resolve the protobuf import error and enable proper model training using the standalone Keras API. Replaced the TensorFlow import with Keras imports, updated all `tf.keras` references to `keras`, and adjusted the training loop to use the Keras components. Increased training epochs slightly for better convergence while preserving the original fallback logic and submission format.'
- What this solution (achieved 0.61099) has done: 'Implemented a fix by setting the Keras backend to NumPy before any Keras imports, which resolves the protobuf `MessageFactory` error and allows the MobileNetV2 model to be built and trained using the existing pipeline. No other logic changes were made, preserving the original workflow while enabling proper model training and prediction, thereby moving the score toward the target.'
- What this solution (achieved 0.61099) has done: 'I increase the model’s capacity to better match the target score by (1) resizing images to the MobileNetV2 native 224×224 resolution, (2) adding a lightweight data‑augmentation block, (3) fine‑tuning the last 20 layers of the pretrained backbone, and (4) training a few more epochs with a slightly lower learning rate and an additional ReduceLROnPlateau callback. These changes keep the original workflow intact while reasonably improving validation accuracy, thus moving the score closer to the target.'

# 9. Code solution

## === cell 0
import os
import json

os.environ["KERAS_BACKEND"] = "numpy"

BASE_DIR = "../input/cassava-leaf-disease-classification/"



## === cell 1
with open(os.path.join(BASE_DIR, "label_num_to_disease_map.json")) as file:
    map_classes = json.loads(file.read())
    map_classes = {int(k): v for k, v in map_classes.items()}

print(json.dumps(map_classes, indent=4))



## === cell 2
import pandas as pd
import cv2
import numpy as np
import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



## === cell 3
input_files = os.listdir(os.path.join(BASE_DIR, "train_images"))
print(f"Number of train images: {len(input_files)}")



## === cell 4
img_shapes = {}
for image_name in os.listdir(os.path.join(BASE_DIR, "train_images"))[:300]:
    image = cv2.imread(os.path.join(BASE_DIR, "train_images", image_name))
    if image is not None:
        img_shapes[image.shape] = img_shapes.get(image.shape, 0) + 1

print(img_shapes)



## === cell 5
df_train = pd.read_csv(os.path.join(BASE_DIR, "train.csv"))
df_train["class_name"] = df_train["label"].map(map_classes)
df_train.head()



## === cell 6
df_train["image_id"] = df_train["image_id"].astype("str")
df_train["label"] = df_train["label"].astype("str")



## === cell 7
df_train["label"].value_counts()



## === cell 8
df_train["label"] = df_train["label"].astype(str)
most_common_label_str = df_train["label"].value_counts().idxmax()
fallback_label = int(most_common_label_str)  # required integer label
print(f"Fallback label (most common in training): {fallback_label}")

df_train["filepath"] = df_train["image_id"].apply(
    lambda x: os.path.join(BASE_DIR, "train_images", x)
)

IMG_SIZE = (224, 224)  # resize dimensions

model_success = False
model = None

try:
    imgs = []
    labels = []
    for _, row in df_train.iterrows():
        img = cv2.imread(row["filepath"])
        if img is None:
            continue
        img = cv2.resize(img, IMG_SIZE)
        img = img.astype(np.float32) / 255.0  # normalize to [0,1]
        imgs.append(img)
        labels.append(int(row["label"]))
    X = np.stack(imgs)  # shape (N,224,224,3)
    y = np.array(labels)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.1, stratify=y, random_state=42
    )

    augmentation = keras.Sequential(
        [
            keras.layers.RandomFlip("horizontal"),
            keras.layers.RandomRotation(0.1),
        ]
    )

    base_model = keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        include_top=False,
        weights="imagenet",
        pooling="avg",
    )
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    inputs = keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = augmentation(inputs)
    x = base_model(x, training=False)
    outputs = keras.layers.Dense(5, activation="softmax")(x)
    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=5e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    es = keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=3, restore_best_weights=True
    )
    lr_red = keras.callbacks.ReduceLROnPlateau(
        monitor="val_accuracy", factor=0.5, patience=2, min_lr=1e-6, verbose=1
    )

    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=64,
        callbacks=[es, lr_red],
        verbose=2,
    )

    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"Validation accuracy: {val_acc:.4f}")

    model_success = True
    print("MobileNetV2 model trained successfully.")
except Exception as e:
    print(f"Model training failed ({e}); will fall back to constant label.")
    model_success = False
    model = None



## === cell 9
TEST_DIR = os.path.join(BASE_DIR, "test_images")
test_images = [
    f for f in os.listdir(TEST_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))
]
test_images.sort()  # deterministic order

if model_success:
    preds = []
    batch_size = 256
    for i in range(0, len(test_images), batch_size):
        batch_files = test_images[i : i + batch_size]
        batch_imgs = []
        for fname in batch_files:
            img_path = os.path.join(TEST_DIR, fname)
            img = cv2.imread(img_path)
            if img is None:
                img = np.zeros((*IMG_SIZE, 3), dtype=np.uint8)
            img = cv2.resize(img, IMG_SIZE)
            img = img.astype(np.float32) / 255.0
            batch_imgs.append(img)
        batch_imgs = np.stack(batch_imgs)  # (B,224,224,3)
        probs = model.predict(batch_imgs, verbose=0)
        batch_preds = np.argmax(probs, axis=1).astype(int).tolist()
        preds.extend(batch_preds)
    predictions = preds
    print("MobileNetV2 predictions generated for test set.")
else:
    print("Using fallback predictions for all test images.")
    predictions = [fallback_label] * len(test_images)

print(f"Generated predictions for {len(predictions)} test images.")



## === cell 10
sub = pd.DataFrame({"image_id": test_images, "label": predictions})
print(sub.head())
sub.to_csv("submission.csv", index=False)
print("Submission file written to submission.csv")
