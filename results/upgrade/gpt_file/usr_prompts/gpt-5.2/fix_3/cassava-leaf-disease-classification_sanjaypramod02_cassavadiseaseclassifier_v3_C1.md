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

0.8268359020852222

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, sys, subprocess, math, re

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

os.environ["PYTHONHASHSEED"] = "0"



## === cell 1
import tensorflow as tf
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import ViTImageProcessor, ViTModel

np.random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
class CassavaLeafTestDataset(Dataset):
    def __init__(self, tfrecord_files, image_processor):
        self.tfrecord_files = tfrecord_files
        self.image_processor = image_processor
        self.images, self.image_ids = self.load_and_preprocess(tfrecord_files)

    def decode_image(self, image):
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.cast(image, tf.float32) / 255.0
        image = tf.image.resize(image, [224, 224])
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
        image = self.images[idx]
        inputs = self.image_processor(
            images=image, return_tensors="pt", do_resize=False, do_rescale=False
        )
        pixel_values = inputs["pixel_values"].squeeze(0)
        image_id = self.image_ids[idx]
        return pixel_values, image_id




## === cell 3
def _resolve_local_vit_dir(base_dir="/kaggle/input/google-vit"):
    """
    FIX: Transformers 4.53 + huggingface_hub validates repo IDs; we must pass a real local folder
    containing config.json / preprocessor_config.json etc. This helper finds the correct subdir.
    """
    candidates = [
        base_dir,
        os.path.join(base_dir, "google_vit"),
        os.path.join(base_dir, "google-vit"),
        os.path.join(base_dir, "vit"),
        os.path.join(base_dir, "model"),
    ]
    if os.path.isdir(base_dir):
        for name in sorted(os.listdir(base_dir)):
            p = os.path.join(base_dir, name)
            if os.path.isdir(p):
                candidates.append(p)

    def has_files(p):
        return os.path.isfile(os.path.join(p, "config.json"))

    def has_processor(p):
        return (
            os.path.isfile(os.path.join(p, "preprocessor_config.json"))
            or os.path.isfile(os.path.join(p, "image_processor.json"))
            or os.path.isfile(os.path.join(p, "feature_extractor_config.json"))
        )

    best = None
    for p in candidates:
        if os.path.isdir(p) and has_files(p):
            best = p
            if has_processor(p):
                return p
    if best is not None:
        return best
    raise FileNotFoundError(
        f"Could not find a local ViT model directory with config.json under: {base_dir}. "
        f"Checked: {candidates[:10]}"
    )


VIT_LOCAL_DIR = _resolve_local_vit_dir("/kaggle/input/google-vit")
VIT_LOCAL_DIR




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/74374525.py in <cell line: 0>()
     42 
     43 
---> 44 VIT_LOCAL_DIR = _resolve_local_vit_dir("/kaggle/input/google-vit")
     45 VIT_LOCAL_DIR
     46 

/tmp/ipykernel_55/74374525.py in _resolve_local_vit_dir(base_dir)
     36     if best is not None:
     37         return best
---> 38     raise FileNotFoundError(
     39         f"Could not find a local ViT model directory with config.json under: {base_dir}. "
     40         f"Checked: {candidates[:10]}"

FileNotFoundError: Could not find a local ViT model directory with config.json under: /kaggle/input/google-vit. Checked: ['/kaggle/input/google-vit', '/kaggle/input/google-vit/google_vit', '/kaggle/input/google-vit/google-vit', '/kaggle/input/google-vit/vit', '/kaggle/input/google-vit/model']

## === cell 4
class ViTForImageClassification(torch.nn.Module):
    def __init__(self, num_labels=5, vit_dir=VIT_LOCAL_DIR):
        super(ViTForImageClassification, self).__init__()
        self.num_labels = num_labels
        self.vit = ViTModel.from_pretrained(vit_dir, local_files_only=True)
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(self.vit.config.hidden_size, num_labels)

    def forward(self, pixel_values, labels=None):
        outputs = self.vit(pixel_values=pixel_values)
        pooled = self.dropout(outputs.last_hidden_state[:, 0])
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return logits, loss




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/750109801.py in <cell line: 0>()
----> 1 class ViTForImageClassification(torch.nn.Module):
      2     def __init__(self, num_labels=5, vit_dir=VIT_LOCAL_DIR):
      3         super(ViTForImageClassification, self).__init__()
      4         self.num_labels = num_labels
      5         # FIX: Use a resolved local folder path; enforce offline/local loading.

/tmp/ipykernel_55/750109801.py in ViTForImageClassification()
      1 class ViTForImageClassification(torch.nn.Module):
----> 2     def __init__(self, num_labels=5, vit_dir=VIT_LOCAL_DIR):
      3         super(ViTForImageClassification, self).__init__()
      4         self.num_labels = num_labels
      5         # FIX: Use a resolved local folder path; enforce offline/local loading.

NameError: name 'VIT_LOCAL_DIR' is not defined

## === cell 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = "/kaggle/input/new-pth/new_v2.pth"
model = ViTForImageClassification(num_labels=5, vit_dir=VIT_LOCAL_DIR).to(device)

state = torch.load(model_path, map_location=device)
if (
    isinstance(state, dict)
    and all(isinstance(k, str) for k in state.keys())
    and any(k.startswith("vit.") or k.startswith("classifier.") for k in state.keys())
):
    model.load_state_dict(state, strict=True)
elif (
    isinstance(state, dict)
    and "state_dict" in state
    and isinstance(state["state_dict"], dict)
):
    model.load_state_dict(state["state_dict"], strict=False)
else:
    model.load_state_dict(state, strict=False)

model.eval()



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/857503131.py in <cell line: 0>()
      2 
      3 model_path = "/kaggle/input/new-pth/new_v2.pth"
----> 4 model = ViTForImageClassification(num_labels=5, vit_dir=VIT_LOCAL_DIR).to(device)
      5 
      6 # FIX: robust load across common checkpoint formats; keep same semantics otherwise.

NameError: name 'ViTForImageClassification' is not defined

## === cell 6
TEST_FILENAMES = tf.io.gfile.glob(
    "/kaggle/input/cassava-leaf-disease-classification/test_tfrecords/ld_test*.tfrec"
)
TEST_FILENAMES = sorted(TEST_FILENAMES)

image_processor = ViTImageProcessor.from_pretrained(
    VIT_LOCAL_DIR, local_files_only=True
)

test_dataset = CassavaLeafTestDataset(TEST_FILENAMES, image_processor=image_processor)

test_dataloader = DataLoader(
    test_dataset,
    batch_size=12,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)

len(test_dataset), len(test_dataloader)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3935773462.py in <cell line: 0>()
      6 # FIX: Use the same resolved local dir for image processor to avoid HFValidationError/OSError.
      7 image_processor = ViTImageProcessor.from_pretrained(
----> 8     VIT_LOCAL_DIR, local_files_only=True
      9 )
     10 

NameError: name 'VIT_LOCAL_DIR' is not defined

## === cell 7
predictions = []
image_ids = []

with torch.no_grad():
    for pixel_values, ids in test_dataloader:
        pixel_values = pixel_values.to(device, non_blocking=torch.cuda.is_available())
        logits, _ = model(pixel_values, None)
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy().tolist()
        predictions.extend(preds)
        image_ids.extend(list(ids))

submission_df = pd.DataFrame({"image_id": image_ids, "label": predictions})
submission_df["image_id"] = submission_df["image_id"].astype(str)
submission_df["label"] = submission_df["label"].astype(int)

sample_path = "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
sample_sub = pd.read_csv(sample_path)
submission_df = sample_sub[["image_id"]].merge(submission_df, on="image_id", how="left")

if submission_df["label"].isna().any():
    fill_label = int(pd.Series(predictions).mode().iloc[0]) if len(predictions) else 0
    submission_df["label"] = submission_df["label"].fillna(fill_label).astype(int)

out_path = "/kaggle/working/submission.csv"
submission_df.to_csv(out_path, index=False)

out_path, submission_df.shape, submission_df.head()



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1189903265.py in <cell line: 0>()
      3 
      4 with torch.no_grad():
----> 5     for pixel_values, ids in test_dataloader:
      6         pixel_values = pixel_values.to(device, non_blocking=torch.cuda.is_available())
      7         logits, _ = model(pixel_values, None)

NameError: name 'test_dataloader' is not defined

## === cell 8
with open("/kaggle/working/submission.csv", "r") as f:
    for _ in range(5):
        print(f.readline().strip())

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/3600513885.py in <cell line: 0>()
----> 1 with open("/kaggle/working/submission.csv", "r") as f:
      2     for _ in range(5):
      3         print(f.readline().strip())

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/working/submission.csv'
