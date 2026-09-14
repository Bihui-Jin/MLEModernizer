# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import re
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf

print("TF version:", tf.__version__)
print("Eager:", tf.executing_eagerly())

tf.keras.utils.set_random_seed(42)



## === cell 1
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from sklearn.preprocessing import LabelEncoder

label_to_disease = pd.read_json(
    "/kaggle/input/cassava-leaf-disease-classification/label_num_to_disease_map.json",
    typ="series",
)
train_csv = pd.read_csv("/kaggle/input/cassava-leaf-disease-classification/train.csv")

train_csv["disease"] = train_csv["label"].map(label_to_disease)
train_csv["path"] = (
    "/kaggle/input/cassava-leaf-disease-classification/train_images/"
    + train_csv["image_id"]
)

train_csv["label_encoded"] = LabelEncoder().fit_transform(train_csv["disease"])

train_csv["disease"] = train_csv["disease"].astype(str)
train_csv["label"] = train_csv["label"].astype(str)

train, valid = train_test_split(
    train_csv, test_size=0.2, stratify=train_csv["label"], random_state=42
)

datagen_aug = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=45,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode="nearest",
)

train_generator = datagen_aug.flow_from_dataframe(
    dataframe=train,
    x_col="path",
    y_col="disease",
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    shuffle=True,
    seed=42,
)

datagen_valid = ImageDataGenerator(preprocessing_function=preprocess_input)

valid_generator = datagen_valid.flow_from_dataframe(
    dataframe=valid,
    x_col="path",
    y_col="disease",
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    shuffle=False,
)




## === cell 2
def load_and_preprocess_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)  # Assuming the images are JPEGs
    image = tf.image.resize(image, [224, 224])  # Resize to the expected input size
    image = image / 255.0  # Normalize to [0, 1]
    return image, label


train_ds = tf.data.Dataset.from_tensor_slices(
    (train["path"].values, train["label_encoded"].values)
)
valid_ds = tf.data.Dataset.from_tensor_slices(
    (valid["path"].values, valid["label_encoded"].values)
)

train_ds = (
    train_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(32)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)
valid_ds = (
    valid_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(32)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)



## === cell 3
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
from tensorflow.keras.applications import DenseNet169, EfficientNetB4
from tensorflow.keras.layers import Dense, Input, Lambda
from tensorflow.keras.models import Model


class EarlyStoppingCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if self.model.stop_training:
            print(f"Early stopping triggered at epoch {epoch + 1}.")


early_stopping = EarlyStopping(
    monitor="val_loss", patience=3, restore_best_weights=True
)

learning_rate_reduction = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", patience=2, factor=0.5, min_lr=1e-6, verbose=1
)



## === cell 4
from pathlib import Path


def find_saved_model_dir(root_dir: str):
    root = Path(root_dir)
    if not root.exists():
        return None
    if (root / "saved_model.pb").exists() or (root / "saved_model.pbtxt").exists():
        return str(root)
    for p in root.rglob("saved_model.pb"):
        return str(p.parent)
    for p in root.rglob("saved_model.pbtxt"):
        return str(p.parent)
    return None


def build_tfsm_model_from_input_package(
    input_pkg_root: str,
    call_endpoint: str = "serving_default",
    input_shape=(224, 224, 3),
):
    from tensorflow.keras.layers import Input, TFSMLayer
    from tensorflow.keras.models import Model

    sm_dir = find_saved_model_dir(input_pkg_root)
    if sm_dir is None:
        raise OSError(f"Could not find a SavedModel under: {input_pkg_root}")
    layer = TFSMLayer(sm_dir, call_endpoint=call_endpoint)
    inp = Input(shape=input_shape)
    out = layer(inp)
    return Model(inputs=inp, outputs=out), sm_dir




## === cell 5
import numpy as np
import pandas as pd
import os
import tensorflow as tf

cropnet_input_root = "/kaggle/input/cropnet_model_unfrozen"
old_densenet_input_root = "/kaggle/input/old_densenet"
old_efficientnet_input_root = "/kaggle/input/old_efficient_net"

use_external_ensemble = True
try:
    cropnet_model, cropnet_sm_dir = build_tfsm_model_from_input_package(
        cropnet_input_root, call_endpoint="serving_default"
    )
    old_densenet_model, densenet_sm_dir = build_tfsm_model_from_input_package(
        old_densenet_input_root, call_endpoint="serving_default"
    )
    old_efficientnet_model, effnet_sm_dir = build_tfsm_model_from_input_package(
        old_efficientnet_input_root, call_endpoint="serving_default"
    )

    print("Loaded CropNet SavedModel from:", cropnet_sm_dir)
    print("Loaded old DenseNet SavedModel from:", densenet_sm_dir)
    print("Loaded old EfficientNet SavedModel from:", effnet_sm_dir)
except OSError as e:
    use_external_ensemble = False
    print(
        "External SavedModels not found; switching to fallback training. Details:",
        str(e),
    )

fallback_model = None
if not use_external_ensemble:
    base = EfficientNetB0(
        include_top=False, weights="imagenet", input_shape=(224, 224, 3)
    )
    x = GlobalAveragePooling2D()(base.output)
    out = Dense(5, activation="softmax")(x)
    fallback_model = Model(inputs=base.input, outputs=out)

    base.trainable = False

    fallback_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    fallback_model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=3,
        callbacks=[learning_rate_reduction],
        verbose=2,
    )



## === cell 6
import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import load_img, img_to_array

sample_sub_path = (
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)
sample_sub = pd.read_csv(sample_sub_path)

image_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images"
img_size = (224, 224)

predictions = []
image_names = []

for filename in sample_sub["image_id"].values:
    img_path = os.path.join(image_dir, filename)
    img = load_img(img_path, target_size=img_size)
    img_array = img_to_array(img) / 255.0  # Normalize image to [0, 1] range
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    if use_external_ensemble:
        cropnet_pred = cropnet_model(img_array)
        densenet_pred = old_densenet_model(img_array)
        efficientnet_pred = old_efficientnet_model(img_array)

        cropnet_pred = (
            list(cropnet_pred.values())[0]
            if isinstance(cropnet_pred, dict)
            else cropnet_pred
        )
        densenet_pred = (
            list(densenet_pred.values())[0]
            if isinstance(densenet_pred, dict)
            else densenet_pred
        )
        efficientnet_pred = (
            list(efficientnet_pred.values())[0]
            if isinstance(efficientnet_pred, dict)
            else efficientnet_pred
        )

        cropnet_pred = (
            cropnet_pred.numpy() if hasattr(cropnet_pred, "numpy") else cropnet_pred
        )
        densenet_pred = (
            densenet_pred.numpy() if hasattr(densenet_pred, "numpy") else densenet_pred
        )
        efficientnet_pred = (
            efficientnet_pred.numpy()
            if hasattr(efficientnet_pred, "numpy")
            else efficientnet_pred
        )

        cropnet_weight = 0.65
        densenet_weight = 0.25
        efficientnet_weight = 0.1

        avg_pred = (
            cropnet_weight * cropnet_pred
            + densenet_weight * densenet_pred
            + efficientnet_weight * efficientnet_pred
        )
        predicted_class = int(np.argmax(avg_pred, axis=-1)[0])
    else:
        pred = fallback_model.predict(img_array, verbose=0)
        predicted_class = int(np.argmax(pred, axis=-1)[0])

    predictions.append(predicted_class)
    image_names.append(filename)

submission_df = pd.DataFrame({"image_id": image_names, "label": predictions})

assert (
    submission_df.shape[0] == sample_sub.shape[0]
), "Row count mismatch vs sample_submission"
assert list(submission_df.columns) == [
    "image_id",
    "label",
], "Submission columns incorrect"

out_path = "/kaggle/working/submission.csv"
submission_df.to_csv(out_path, index=False)
print("Submission file created:", out_path)
print(submission_df.head())
