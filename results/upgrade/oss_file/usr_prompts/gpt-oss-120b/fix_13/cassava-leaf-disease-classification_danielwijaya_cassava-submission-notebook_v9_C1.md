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

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

0.8850105772136597

# 6. Current score

0.68685

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.61099) has done: 'I remove the failing TensorFlow and external‑library imports, replace the missing pretrained model with a simple fallback that predicts the most common class from the training data, and rewrite the pipeline to read the test image list directly from the provided `sample_submission.csv`. This eliminates the import errors and ensures the script runs end‑to‑end, producing a valid `submission.csv` file in the required format.'
- What this solution (achieved 0.06689) has done: 'I replace the naïve “most‑common‑class” baseline with a lightweight fine‑tuned MobileNetV2 model. The new cells load the training images, build a small transfer‑learning model (base frozen, only a final dense layer trained), train it for a few epochs, and then generate predictions for the test set. This modest upgrade is expected to raise accuracy from ~0.61 toward the target ~0.885 while keeping the overall pipeline and submission format unchanged.'
- What this solution (achieved 0.7059) has done: 'Implemented key fixes to get the pipeline running and improve model performance:
- Added a protobuf compatibility flag before importing TensorFlow.
- Corrected the `shuffle` call argument (`buffer_size` instead of the non‑existent `buffer_len`).
- Updated cell ordering to start from 1 while preserving all original logic.
- Minor clean‑ups ensure the validation dataset is created correctly and the final submission file is written after model inference.'
- What this solution (achieved 0.71749) has done: 'Implemented a protobuf compatibility patch before importing TensorFlow to resolve the `MessageFactory` attribute error, and increased training epochs to give the model more learning opportunity (EPOCHS = 5). These adjustments allow the full TensorFlow pipeline to run, produce model‑based predictions, and output a valid `submission.csv` that should improve accuracy toward the target score.'
- What this solution (achieved 0.71001) has done: 'Implemented two focused improvements to close the gap to the target accuracy:  
1. **Extended training** – increased epochs from 5 to 12 to give the model more learning iterations.  
2. **Light fine‑tuning** – unfroze the last 20 layers of the pretrained MobileNetV2 backbone and lowered the Adam learning rate to 5e‑4. These changes keep the original architecture intact while providing a modest boost in predictive performance.'
- What this solution (achieved 0.55904) has done: 'I improve the model’s generalisation by adding simple data‑augmentation layers and ensuring a truly random train/validation split. These small tweaks keep the original architecture and hyper‑parameters while providing the model with more varied inputs, which should raise validation accuracy and thus push the Kaggle score closer to the target. I also increase the training epochs slightly to let the model benefit from the richer data.'
- What this solution (achieved 0.61099) has done: 'I increased the training duration and softened the fine‑tuning learning rate so the model can learn more from the data without altering its architecture.  
- In **cell 4** the number of epochs is raised from 15 to 30.  
- In **cell 6** the Adam learning‑rate is changed from 5e‑4 to 1e‑4, which is better suited for fine‑tuning the unfrozen MobileNetV2 layers.  
These minimal adjustments keep the core pipeline intact while expected to raise validation accuracy and move the Kaggle score closer to the target.'
- What this solution (achieved 0.73916) has done: 'The changes focus on speeding up the data pipeline and leveraging mixed‑precision training, which keep the exact model architecture, loss, optimizer, and training loop unchanged. Caching the parsed images avoids repeated disk I/O each epoch, and `prefetch` with `AUTOTUNE` overlaps preprocessing with training. Enabling TensorFlow’s mixed‑precision policy accelerates computation on supported hardware without altering numerical results beyond negligible float‑16 differences. These adjustments dramatically reduce total runtime while preserving the original learning behavior.'
- What this solution (achieved 0.69581) has done: 'The fix removes the unnecessary protobuf monkey‑patch (which caused the AttributeError) and applies the correct MobileNetV2 preprocessing to the images, which improves model accuracy while keeping the original architecture and training pipeline unchanged. The script is renumbered to start at 1 and now runs end‑to‑end, producing a valid `submission.csv`.'
- What this solution (achieved 0.68685) has done: 'The fix switches the protobuf implementation to the pure‑Python version so TensorFlow can import, defines `SEED` before it’s used, and enables the backbone to train (by passing `training=True`). These changes resolve the import errors and let the original MobileNetV2 fine‑tuning run, producing a valid `submission.csv` while keeping the core pipeline unchanged.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))




## === cell 1
train_path = "../input/cassava-leaf-disease-classification/train.csv"
train_df = pd.read_csv(train_path)

most_common_label = train_df["label"].mode()[0]
print(f"Most common label in training set: {most_common_label}")




## === cell 2
test_sub_path = "../input/cassava-leaf-disease-classification/sample_submission.csv"
test_df = pd.read_csv(test_sub_path)

predictions = np.full(shape=len(test_df), fill_value=most_common_label, dtype=int)

print(f"Number of test images: {len(test_df)}")
print(f"Predictions shape: {predictions.shape}")




## === cell 3
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

tf.config.optimizer.set_jit(True)

mixed_precision.set_global_policy("mixed_float16")

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE
EPOCHS = 45  # keep original epoch count
SEED = 42
tf.random.set_seed(SEED)




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 4
train_images_dir = "../input/cassava-leaf-disease-classification/train_images"
train_df["image_path"] = train_df["image_id"].apply(
    lambda x: os.path.join(train_images_dir, x)
)

train_paths = train_df["image_path"].values
train_labels = train_df["label"].values.astype(np.int32)

np.random.seed(SEED)
perm = np.random.permutation(len(train_paths))
train_paths = train_paths[perm]
train_labels = train_labels[perm]

val_split = int(0.2 * len(train_paths))
train_paths_split, val_paths_split = train_paths[:-val_split], train_paths[-val_split:]
train_labels_split, val_labels_split = (
    train_labels[:-val_split],
    train_labels[-val_split:],
)


def _load_image(path, label=None):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = preprocess_input(image)
    if label is None:
        return image
    return image, label


train_ds = tf.data.Dataset.from_tensor_slices((train_paths_split, train_labels_split))
train_ds = train_ds.map(_load_image, num_parallel_calls=AUTOTUNE)
train_ds = train_ds.cache()
train_ds = train_ds.shuffle(buffer_size=len(train_paths_split), seed=SEED)
train_ds = train_ds.batch(BATCH_SIZE)
train_ds = train_ds.prefetch(AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((val_paths_split, val_labels_split))
val_ds = val_ds.map(_load_image, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.cache()
val_ds = val_ds.batch(BATCH_SIZE)
val_ds = val_ds.prefetch(AUTOTUNE)

options = tf.data.Options()
options.experimental_deterministic = False
train_ds = train_ds.with_options(options)
val_ds = val_ds.with_options(options)




## === cell 5
data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.2),
    ]
)

base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMAGE_SIZE + (3,), include_top=False, weights="imagenet"
)

base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

inputs = tf.keras.Input(shape=IMAGE_SIZE + (3,))
x = data_augmentation(inputs)
x = base_model(x, training=True)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
outputs = tf.keras.layers.Dense(5, activation="softmax")(x)

model = tf.keras.Model(inputs, outputs)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"],
)

model.summary()




## === cell 6
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=2)




## === cell 7
test_images_dir = "../input/cassava-leaf-disease-classification/test_images"
test_df["image_path"] = test_df["image_id"].apply(
    lambda x: os.path.join(test_images_dir, x)
)

test_paths = test_df["image_path"].values

test_ds = tf.data.Dataset.from_tensor_slices(test_paths)
test_ds = test_ds.map(lambda p: _load_image(p), num_parallel_calls=AUTOTUNE)
test_ds = test_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

pred_probs = model.predict(test_ds, verbose=0)
pred_labels = np.argmax(pred_probs, axis=1).astype(int)




## === cell 8
submission = pd.DataFrame({"image_id": test_df["image_id"], "label": pred_labels})

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Model‑based submission written to {submission_path}")
print("\nFirst few lines of the new submission:")
print(submission.head())
