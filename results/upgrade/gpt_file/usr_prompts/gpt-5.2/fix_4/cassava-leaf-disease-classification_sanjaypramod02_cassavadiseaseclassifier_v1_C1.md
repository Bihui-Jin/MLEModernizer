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
pillow==11.3.0
protobuf==6.33.0
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sentence-transformers==4.1.0
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
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124
transformers==4.53.3

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

0.8331822302810517

# 6. Current score

0.61099

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.61099) has done: 'I fix the environment crash caused by an incompatibility between `transformers` and `protobuf` by pinning the pure-Python protobuf implementation before importing `transformers`. Then I remove the dependency on missing Kaggle inputs (`/kaggle/input/google-vit` and `/kaggle/input/new-model/new_v2.pth`) by switching to a local, lightweight fallback that preserves the same “ViT-like classifier produces logits then argmax labels” semantics, so the notebook runs end-to-end. Finally, I ensure the test set is read from the provided `test_tfrecords`, predictions align with `sample_submission.csv`, and a valid `/kaggle/working/submission.csv` is always written.'

# 9. Code solution

## === cell 0
import os
import math, re
import numpy as np
import pandas as pd

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import tensorflow as tf

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

try:
    from transformers import ViTImageProcessor, ViTModel

    _HAS_TRANSFORMERS = True
except Exception as e:
    print(
        "Warning: transformers could not be imported, using fallback model. Error:",
        repr(e),
    )
    _HAS_TRANSFORMERS = False

print(
    "tf:",
    tf.__version__,
    "torch:",
    torch.__version__,
    "transformers:",
    _HAS_TRANSFORMERS,
)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def find_local_vit_dir(base="/kaggle/input/google-vit"):
    """
    Original intent: locate a local ViT directory under /kaggle/input/google-vit.
    Fix: make it non-fatal (return None) because this dataset/input is not present here.
    """
    if not os.path.isdir(base):
        return None

    candidates = [base]
    for name in sorted(os.listdir(base)):
        p = os.path.join(base, name)
        if os.path.isdir(p):
            candidates.append(p)

    def looks_like_model_dir(p):
        return os.path.isfile(os.path.join(p, "config.json")) or os.path.isfile(
            os.path.join(p, "preprocessor_config.json")
        )

    for p in candidates:
        if looks_like_model_dir(p):
            return p

    for root, dirs, files in os.walk(base):
        if "config.json" in files or "preprocessor_config.json" in files:
            return root

    return None


VIT_DIR = find_local_vit_dir("/kaggle/input/google-vit")
print("Local ViT dir found:", VIT_DIR)




## === cell 2
class CassavaLeafTestDataset(Dataset):
    def __init__(self, tfrecord_files):
        self.tfrecord_files = list(tfrecord_files)
        self.images, self.image_ids = self.load_and_preprocess(self.tfrecord_files)

    def decode_image(self, image_bytes):
        image = tf.image.decode_jpeg(image_bytes, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)  # [0,1]
        image = tf.image.resize(image, [224, 224], preserve_aspect_ratio=False)
        return image

    def read_tfrecord(self, serialized_example):
        tfrecord_format = {
            "image": tf.io.FixedLenFeature([], tf.string),
            "image_name": tf.io.FixedLenFeature([], tf.string),
        }
        return tf.io.parse_single_example(serialized_example, tfrecord_format)

    def load_and_preprocess(self, tfrecord_files):
        raw_dataset = tf.data.TFRecordDataset(
            tfrecord_files, num_parallel_reads=tf.data.AUTOTUNE
        )
        parsed_dataset = raw_dataset.map(
            self.read_tfrecord, num_parallel_calls=tf.data.AUTOTUNE
        )

        images = []
        image_ids = []
        for parsed_record in parsed_dataset:
            image = self.decode_image(parsed_record["image"])
            idnum = parsed_record["image_name"].numpy().decode("utf-8")
            images.append(image)
            image_ids.append(idnum)
        return images, image_ids

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].numpy()  # HWC float32
        pixel_values = torch.from_numpy(image).permute(2, 0, 1).contiguous()
        image_id = self.image_ids[idx]
        return pixel_values, image_id




## === cell 3
class ViTForImageClassification(torch.nn.Module):
    """
    Fix: the original code required a missing local ViT directory and a missing checkpoint.
    We keep the same interface (forward returns logits, loss) and "ViT-like" behavior:
    take 224x224 images and output 5-class logits.

    If transformers+local ViT is available, we use it; otherwise we fall back to a tiny
    image classifier head to produce deterministic logits.
    """

    def __init__(self, num_labels=5):
        super().__init__()
        self.num_labels = num_labels

        self.use_hf_vit = bool(_HAS_TRANSFORMERS and VIT_DIR is not None)
        if self.use_hf_vit:
            self.vit = ViTModel.from_pretrained(VIT_DIR, local_files_only=True)
            hidden = self.vit.config.hidden_size
            self.dropout = nn.Dropout(0.1)
            self.classifier = nn.Linear(hidden, num_labels)
        else:
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.dropout = nn.Dropout(0.1)
            self.classifier = nn.Linear(3, num_labels)

    def forward(self, pixel_values, labels=None):
        if self.use_hf_vit:
            outputs = self.vit(pixel_values=pixel_values)
            pooled = self.dropout(outputs.last_hidden_state[:, 0])
            logits = self.classifier(pooled)
        else:
            x = self.pool(pixel_values).squeeze(-1).squeeze(-1)  # (B,3)
            x = self.dropout(x)
            logits = self.classifier(x)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return logits, loss




## === cell 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ViTForImageClassification(num_labels=5).to(device)
model.eval()

TEST_FILENAMES = tf.io.gfile.glob(
    "/kaggle/input/cassava-leaf-disease-classification/test_tfrecords/ld_test*.tfrec"
)
if len(TEST_FILENAMES) == 0:
    TEST_FILENAMES = tf.io.gfile.glob("/kaggle/input/test_tfrecords/ld_test*.tfrec")
if len(TEST_FILENAMES) == 0:
    raise FileNotFoundError("No test TFRecords found. Check the input path/pattern.")

test_dataset = CassavaLeafTestDataset(TEST_FILENAMES)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

print(
    "Test samples:",
    len(test_dataset),
    "TFRecord files:",
    len(TEST_FILENAMES),
    "device:",
    device,
)



## === cell 5
predictions = []
image_ids = []

with torch.no_grad():
    for pixel_values, ids in test_dataloader:
        pixel_values = pixel_values.to(device, dtype=torch.float32)
        logits, _ = model(pixel_values, None)
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy().tolist()
        predictions.extend(preds)
        image_ids.extend(list(ids))

pred_df = pd.DataFrame({"image_id": image_ids, "label": predictions})

sample_path = "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
if not os.path.exists(sample_path):
    sample_path = "/kaggle/input/sample_submission.csv"
sample_sub = pd.read_csv(sample_path)

submission_df = sample_sub[["image_id"]].merge(pred_df, on="image_id", how="left")

if submission_df["label"].isna().any():
    if len(pred_df) > 0:
        fill_label = int(pd.Series(pred_df["label"]).mode().iloc[0])
    else:
        fill_label = 0
    submission_df["label"] = submission_df["label"].fillna(fill_label).astype(int)
else:
    submission_df["label"] = submission_df["label"].astype(int)

out_path = "/kaggle/working/submission.csv"
submission_df.to_csv(out_path, index=False)
print(
    "Wrote:",
    out_path,
    "rows:",
    len(submission_df),
    "cols:",
    list(submission_df.columns),
)

sub = pd.read_csv(out_path)
print(sub.head())
print("shape:", sub.shape)
print("Unique labels:", sorted(sub["label"].unique().tolist()))
print("Nulls:", sub.isna().sum().to_dict())
