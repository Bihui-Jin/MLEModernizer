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
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
sklearn-pandas==2.2.0
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124

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

0.8404351767905711

# 6. Current score

0.61099

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.52205) has done: 'I first fix the runtime failure by removing the hard dependency on a missing external weight file and instead load torchvision’s built-in ResNeXt50 ImageNet weights (same architecture), which run in this environment and should substantially improve accuracy versus random init. I also fix the image preprocessing to always convert inputs to RGB (some images can be non-RGB) and move the transform definition outside the loop for correctness and speed. Finally, I make checkpoint loading robust (handles both full `state_dict` and wrapped dicts) while keeping the core inference logic (argmax over 5 logits) unchanged and still writing a valid `submission.csv`.'
- What this solution (achieved 0.61099) has done: 'Your current score is far below the target (0.52205 vs 0.8404), so we should improve accuracy with minimal, low-risk changes that keep the same core inference logic (single ResNeXt50, argmax over 5 logits). The biggest issue is that you replace the classifier head with a random 5-class layer but typically do not have a matching fine-tuned checkpoint available, so predictions collapse; we only replace `fc` when we actually load a compatible checkpoint, otherwise keep the pretrained ImageNet head and map its 1000-way outputs to 5 classes using fixed, human-defined class-name keyword matching from the provided disease map (no training, still pure inference + argmax semantics). We also switch preprocessing to the model’s official `weights.transforms()` to match ImageNet normalization/resize exactly (small but reliable gain) and implement a safe fallback to the sample-submission order to guarantee a valid CSV. This stays within Kaggle constraints, runs end-to-end, and should move the score substantially toward your target without changing the model family or adding training.'

# 9. Code solution

## === cell 0
import os
import json

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models



## === cell 1
model_path = "../input/model-class-weight/model(2).pth"
sample_sub_path = "../input/cassava-leaf-disease-classification/sample_submission.csv"
test_images_path = "../input/cassava-leaf-disease-classification/test_images"
label_map_path = (
    "../input/cassava-leaf-disease-classification/label_num_to_disease_map.json"
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



## === cell 2
try:
    weights = models.ResNeXt50_32X4D_Weights.DEFAULT
    model = models.resnext50_32x4d(weights=weights)
    preprocess = weights.transforms()
except Exception:
    model = models.resnext50_32x4d(pretrained=True)
    preprocess = models.ResNeXt50_32X4D_Weights.DEFAULT.transforms()

model = model.to(device)

ckpt_loaded = False
if os.path.exists(model_path):
    ckpt = torch.load(model_path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    if isinstance(ckpt, dict) and any(k.startswith("module.") for k in ckpt.keys()):
        ckpt = {k.replace("module.", "", 1): v for k, v in ckpt.items()}

    try:
        fc_w = ckpt.get("fc.weight", None) if isinstance(ckpt, dict) else None
        out_dim = int(fc_w.shape[0]) if isinstance(fc_w, torch.Tensor) else None
    except Exception:
        out_dim = None

    if out_dim == 5:
        model.fc = nn.Linear(2048, 5).to(device)
        model.load_state_dict(ckpt, strict=True)
        ckpt_loaded = True
    else:
        model.load_state_dict(ckpt, strict=False)
        ckpt_loaded = True

model.eval()



## === cell 3
with open(label_map_path, "r") as f:
    cassava_map = json.load(f)
cassava_names = {int(k): str(v).lower() for k, v in cassava_map.items()}

cassava_keywords = {
    0: ["bacterial blight", "blight", "bacterial"],
    1: ["brown streak", "streak"],
    2: ["green mottle", "mottle"],
    3: ["mosaic"],
    4: ["healthy"],
}

try:
    imagenet_categories = models.ResNeXt50_32X4D_Weights.DEFAULT.meta["categories"]
except Exception:
    imagenet_categories = None


def map_imagenet_to_cassava(imagenet_topk_indices):
    """
    Map ImageNet predicted classes (indices) to one of 5 cassava labels via keyword matching.
    If nothing matches, default to the most frequent cassava class in many solutions (CMD=3) as a stable fallback.
    """
    if imagenet_categories is None:
        return 3  # stable fallback if categories unavailable

    for idx in imagenet_topk_indices:
        name = imagenet_categories[int(idx)].lower()
        if "cassava" in name:
            return 4 if ("leaf" in name or "plant" in name) else 3

        for lbl, kws in cassava_keywords.items():
            for kw in kws:
                if kw in name:
                    return lbl

    return 3




## === cell 4
sample_sub = pd.read_csv(sample_sub_path)

predictions = []
with torch.inference_mode():
    for _, sample_row in sample_sub.iterrows():
        img_path = os.path.join(test_images_path, sample_row.image_id)
        image = Image.open(img_path).convert("RGB")

        x = preprocess(image).unsqueeze(0).to(device)
        outputs = model(x)

        if outputs.shape[1] == 5:
            pred_label = int(outputs.argmax(dim=1).item())
        else:
            topk = min(25, outputs.shape[1])
            topk_idx = outputs.topk(topk, dim=1).indices.squeeze(0).tolist()
            pred_label = int(map_imagenet_to_cassava(topk_idx))

        predictions.append([sample_row.image_id, pred_label])

sub_df = pd.DataFrame(predictions, columns=["image_id", "label"])

sub_df = sample_sub[["image_id"]].merge(sub_df, on="image_id", how="left")
sub_df["label"] = sub_df["label"].fillna(3).astype(int)

sub_df.to_csv("submission.csv", index=False)
print(sub_df.head())
print("Checkpoint loaded:", ckpt_loaded)
print("Wrote submission.csv with shape:", sub_df.shape)
