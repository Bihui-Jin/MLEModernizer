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

0.8553943789664551

# 6. Current score

0.61622

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.11024) has done: 'The script now imports missing utilities, safely loads the pretrained model (falling back to a standard ImageNet ResNet50 if the custom file is absent), uses `tqdm` for progress, ensures all test images are processed, and writes a correctly‑sized `submission.csv` matching the required format.'
- What this solution (achieved 0.6151) has done: 'Implemented batch‑wise inference for the test set to eliminate the per‑image `model.predict` call, which removed a major source of overhead.  The training loop still uses the original `ImageSequence` (preserving core logic), while prediction now builds batches of size `batch_size`, preprocesses all images in the batch at once, runs a single `model.predict` call, and extracts class predictions for each image.  This change keeps results identical (same preprocessing and arg‑max selection) but dramatically reduces I/O and TensorFlow call overhead, ensuring the script finishes well within the 600‑second limit.'
- What this solution (achieved 0.61622) has done: 'The fix adds a small monkey‑patch for the protobuf MessageFactory before TensorFlow is imported, which resolves the `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'` that stopped the script from running. No core‑logic changes are made, so the model architecture, training loop, and inference remain unchanged, allowing the existing score‑improving workflow to run and produce a proper `submission.csv`.'

# 9. Code solution

## === cell 0
import os, glob, random
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

try:
    import google.protobuf.message_factory as _mf

    if not hasattr(_mf.MessageFactory, "GetPrototype"):
        _mf.MessageFactory.GetPrototype = _mf.MessageFactory.GetMessageClass
except Exception:
    pass  # If protobuf is unavailable, TensorFlow will raise its own error later

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
test_data_directory = "/kaggle/input/cassava-leaf-disease-classification/test_images"
train_data_directory = "/kaggle/input/cassava-leaf-disease-classification/train_images"
train_csv_path = "/kaggle/input/cassava-leaf-disease-classification/train.csv"
sample_submission_path = (
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)

num_classes = 5
batch_size = 32
epochs = 3
image_size = (224, 224)




## === cell 2
def resnet_preprocess(image):
    """Resize, scale to [0,1] and add batch dimension."""
    image = image.resize(image_size)
    img_array = np.array(image) / 255.0  # (H,W,3) in [0,1]
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)  # (1,H,W,3)
    return img_array




## === cell 3
try:
    resnet_model_path = (
        "/kaggle/input/resnet_cassava/keras/default/1/resnet_cassava.keras"
    )
    resnet_model = tf.keras.models.load_model(resnet_model_path)
    print("Custom model loaded.")
except Exception as e:
    print(f"Could not load custom model ({e}), building fallback model.")
    base = tf.keras.applications.ResNet50(
        weights="imagenet", include_top=False, input_shape=(224, 224, 3)
    )
    base.trainable = False
    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    output = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    resnet_model = tf.keras.Model(inputs=base.input, outputs=output)

resnet_model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"],
)




## === cell 4
train_df = pd.read_csv(train_csv_path)

train_df["image_path"] = train_df["image_id"].apply(
    lambda x: os.path.join(train_data_directory, x)
)

train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
val_split = int(0.1 * len(train_df))
val_df = train_df[:val_split]
train_df = train_df[val_split:]


class ImageSequence(tf.keras.utils.Sequence):
    def __init__(self, df, batch_size, augment=False):
        self.df = df
        self.batch_size = batch_size
        self.augment = augment

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def __getitem__(self, idx):
        batch_df = self.df.iloc[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_images = []
        batch_labels = []
        for _, row in batch_df.iterrows():
            img = Image.open(row["image_path"]).convert("RGB")
            img = img.resize(image_size)
            arr = np.array(img) / 255.0
            batch_images.append(arr)
            batch_labels.append(row["label"])
        X = np.stack(batch_images).astype(np.float32)
        y = np.array(batch_labels).astype(np.int32)
        return X, y


train_seq = ImageSequence(train_df, batch_size)
val_seq = ImageSequence(val_df, batch_size)




## === cell 5
resnet_model.fit(
    train_seq,
    validation_data=val_seq,
    epochs=epochs,
    verbose=1,
)




## === cell 6
image_paths = glob.glob(
    os.path.join(test_data_directory, "**", "*.jpg"), recursive=True
)
image_paths.sort()  # deterministic order

image_ids = []
predictions = []

for start_idx in range(0, len(image_paths), batch_size):
    batch_paths = image_paths[start_idx : start_idx + batch_size]
    batch_ids = [os.path.basename(p) for p in batch_paths]
    batch_imgs = []
    for p in batch_paths:
        img = Image.open(p).convert("RGB")
        img = img.resize(image_size)
        arr = np.array(img) / 255.0
        batch_imgs.append(arr.astype(np.float32))
    batch_tensor = np.stack(batch_imgs)  # shape (B, H, W, 3)

    batch_logits = resnet_model.predict(batch_tensor, verbose=0)
    batch_preds = np.argmax(batch_logits, axis=1).astype(int)

    image_ids.extend(batch_ids)
    predictions.extend(batch_preds.tolist())




## === cell 7
submission_df = pd.DataFrame({"image_id": image_ids, "label": predictions})

expected_len = pd.read_csv(sample_submission_path).shape[0]
if len(submission_df) != expected_len:
    if len(submission_df) < expected_len:
        pad_count = expected_len - len(submission_df)
        pad_df = pd.DataFrame(
            {
                "image_id": [f"pad_{i}.jpg" for i in range(pad_count)],
                "label": [0] * pad_count,
            }
        )
        submission_df = pd.concat([submission_df, pad_df], ignore_index=True)
    else:
        submission_df = submission_df.iloc[:expected_len]

submission_df.to_csv("submission.csv", index=False)
print("Submission file created: submission.csv")
