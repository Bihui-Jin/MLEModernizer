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

albumentations==2.0.8
geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
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

0.8762466001813236

# 6. Current score

0.09342

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.09342) has done: 'I remove the dependency on the missing `cassava_utils` module by inlining the minimal `Dataset` and k-fold inference logic it provided, keeping the same basic pipeline (resize/normalize, load models, average logits, argmax). I also fix the missing `../input/resnet50-fold` by falling back to a standard torchvision ResNet-50 inference model (ImageNet pretrained) so the notebook always runs end-to-end and writes `submission.csv`. To avoid further runtime issues, I make device selection robust (CPU fallback), set safe DataLoader defaults for Kaggle, and ensure the submission matches `sample_submission.csv` order and columns. This prioritizes correctness (valid CSV output) and produces a reasonable baseline score without changing the fundamental approach of “model(s) → inference on test → submission”.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

from PIL import Image
import torchvision



## === cell 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("DEVICE =", DEVICE)



## === cell 2
path_candidates = [
    "../input/cassava-leaf-disease-classification",
    "/kaggle/input/cassava-leaf-disease-classification",
    "/kaggle/data/cassava-leaf-disease-classification",
    "/kaggle/data/input/cassava-leaf-disease-classification",
]
path = next((p for p in path_candidates if os.path.exists(p)), None)
if path is None:
    raise FileNotFoundError(
        f"Could not find cassava dataset directory. Tried: {path_candidates}"
    )
print("Using dataset path:", path)

models_path = "../input/resnet50-fold"  # preserve original variable



## === cell 3
models_list = []
if os.path.exists(models_path):
    for model in os.listdir(models_path):
        model = os.path.join(models_path, model)
        if os.path.isfile(model):
            models_list.append(model)
    models_list.sort()
print("Found fold checkpoints:", len(models_list))



## === cell 4
data_albums = {
    "test": A.Compose(
        [
            A.Resize(height=400, width=400),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )
}




## === cell 5
class CassavaTestDataset(Dataset):
    def __init__(self, image_ids, root_dir, albums=None):
        self.image_ids = list(image_ids)
        self.root_dir = root_dir
        self.albums = albums

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_path = os.path.join(self.root_dir, image_id)

        img = Image.open(img_path).convert("RGB")
        img = np.array(img)

        if self.albums is not None:
            out = self.albums(image=img)
            img = out["image"]
        else:
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        return img, image_id


def create_test_dataloader(root_dir, data_albums):
    test_files = sorted([f for f in os.listdir(root_dir) if f.lower().endswith(".jpg")])
    test_dataset = CassavaTestDataset(test_files, root_dir, albums=data_albums["test"])
    num_workers = min(4, os.cpu_count() or 2)
    test_dataloader = {
        "test": DataLoader(
            test_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(DEVICE == "cuda"),
        )
    }
    return test_dataloader




## === cell 6
test_images_dir = os.path.join(path, "test_images")
if not os.path.exists(test_images_dir):
    alt = os.path.join(path, "cassava-leaf-disease-classification", "test_images")
    if os.path.exists(alt):
        test_images_dir = alt
    else:
        raise FileNotFoundError(f"Could not find test_images dir under {path}")
print("Using test_images_dir:", test_images_dir)

test_dataloader = create_test_dataloader(test_images_dir, data_albums)




## === cell 7
def build_fallback_model(num_classes=5):
    weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
    m = torchvision.models.resnet50(weights=weights)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


models = {}

if len(models_list) > 0:
    for fold, model_path in enumerate(models_list):
        ckpt = torch.load(model_path, map_location=DEVICE)
        if isinstance(ckpt, nn.Module):
            model = ckpt
        elif isinstance(ckpt, dict) and "state_dict" in ckpt:
            model = build_fallback_model(num_classes=5)
            sd = ckpt["state_dict"]
            sd = {k.replace("model.", ""): v for k, v in sd.items()}
            model.load_state_dict(sd, strict=False)
        elif isinstance(ckpt, dict):
            model = build_fallback_model(num_classes=5)
            model.load_state_dict(ckpt, strict=False)
        else:
            raise TypeError(
                f"Unsupported checkpoint type: {type(ckpt)} at {model_path}"
            )

        model.to(DEVICE)
        model.eval()
        models[fold] = model
else:
    model = build_fallback_model(num_classes=5).to(DEVICE)
    model.eval()
    models[0] = model

print("Number of models for inference:", len(models))




## === cell 8
@torch.inference_mode()
def predict_kfold(test_loader, models_dict, device=DEVICE):
    all_image_ids = []
    all_preds = []

    for batch in test_loader:
        images, image_ids = batch
        images = images.to(device, non_blocking=True)

        logits_sum = None
        for _, model in models_dict.items():
            logits = model(images)
            logits_sum = logits if logits_sum is None else logits_sum + logits

        logits_avg = logits_sum / float(len(models_dict))
        preds = torch.argmax(logits_avg, dim=1).detach().cpu().numpy()

        all_image_ids.extend(list(image_ids))
        all_preds.extend(list(preds))

    return pd.DataFrame({"image_id": all_image_ids, "label": all_preds})


def get_inference_df(dataloader, models):
    inference_df = predict_kfold(dataloader["test"], models, device=DEVICE)
    return inference_df




## === cell 9
inference_df = get_inference_df(test_dataloader, models)
print(inference_df.head())
print("Num predictions:", len(inference_df))



## === cell 10
sample_path = os.path.join(path, "sample_submission.csv")
if not os.path.exists(sample_path):
    alt = os.path.join(
        path, "cassava-leaf-disease-classification", "sample_submission.csv"
    )
    if os.path.exists(alt):
        sample_path = alt
    else:
        raise FileNotFoundError("sample_submission.csv not found.")
sample_sub = pd.read_csv(sample_path)

sub = sample_sub[["image_id"]].merge(inference_df, on="image_id", how="left")
sub["label"] = sub["label"].fillna(0).astype(int)

assert list(sub.columns) == ["image_id", "label"]
assert len(sub) == len(sample_sub)

sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())
