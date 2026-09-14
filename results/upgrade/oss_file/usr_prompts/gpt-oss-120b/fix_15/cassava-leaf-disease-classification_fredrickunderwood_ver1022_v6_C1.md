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

albumentations==2.0.8
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
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
sklearn-pandas==2.2.0
timm==1.0.19
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124
tqdm==4.67.1

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

0.8934723481414325

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.58445) has done: 'I fix the Albumentations `RandomResizedCrop` calls that expect a tuple for the `size` argument, updating them in both training and test augmentation pipelines. This resolves the validation errors, allowing the augmentation objects to be created and the inference loop to run, which produce a valid `submission.csv`. No other core logic is altered.'
- What this solution (achieved 0.48729) has done: 'The fix removes the nonexistent `albumentations.Cutout` transform (which caused the import error) and changes the inference post‑processing to combine model outputs as softmax probabilities instead of raw logits with L2‑normalisation. Using probabilities gives a more meaningful ensemble and moves the accuracy toward the target score while keeping the original architecture and training logic unchanged.'
- What this solution (achieved 0.53513) has done: 'I make three minimal adjustments that are expected to improve the validation‑style accuracy without altering the core model architecture or training logic.  
1. Set `TTA = 1` so the EfficientNet predictions are not affected by random test‑time augmentations.  
2. Replace the stochastic `test_augs` pipeline with a deterministic resize‑and‑normalize pipeline, removing random flips and crops that can hurt consistency.  
3. Balance the ensemble to an equal 0.5/0.5 weighting, which is a safer default when the individual model performance is unknown.  

These changes keep the original models and training code intact while making inference more stable, moving the score toward the target.'
- What this solution (achieved 0.53513) has done: 'I add a lightweight validation step that evaluates each pretrained model (and their simple average) on a small held‑out subset of the training data. Based on the validation accuracies the script choose the weighting (either the better single model or the equal‑weight ensemble) that gives the highest validation accuracy, and then use that weighting for the final test predictions. This small change keeps the original model architectures and inference pipeline unchanged while steering the ensemble toward a configuration that should raise the validation‑style accuracy and therefore move the public score closer to the target.'

# 9. Code solution

## === cell 0
import os
import random
import json
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

import timm
from sklearn.model_selection import train_test_split

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



## === cell 1
BASE_INPUT = "/kaggle/input"
TRAIN_CSV_PATH = os.path.join(
    BASE_INPUT, "cassava-leaf-disease-classification", "train.csv"
)
TRAIN_IMAGE_PATH = os.path.join(
    BASE_INPUT, "cassava-leaf-disease-classification", "train_images"
)
TEST_IMAGE_PATH = os.path.join(
    BASE_INPUT, "cassava-leaf-disease-classification", "test_images"
)
SUBMISSION_PATH = "submission.csv"

RESNEXT_PATH = "1022_res50.pth"
B4_PATH = "1022_b4ns.pth"

DEVICES = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())] or [
    torch.device("cpu")
]
DEVICE = DEVICES[0]

OUT_FEATURES = 5
NUM_EPOCHS = 17
BATCH_SIZE = 32
IMAGE_SIZE = 512
OPTIMIZER = torch.optim.AdamW
LR_START = 1e-5
LR_MAX = 2e-4
LR_FINAL = 1e-5
TTA = 5




## === cell 2
class CassavaDataset(Dataset):
    def __init__(self, df, img_dir, transforms=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["image_id"])
        image = np.array(Image.open(img_path).convert("RGB"))
        if self.transforms:
            image = self.transforms(image=image)["image"]
        label = row["label"] if "label" in row else -1
        return image, label


train_augs = A.Compose(
    [
        A.RandomResizedCrop(
            height=IMAGE_SIZE, width=IMAGE_SIZE, scale=(0.8, 1.0), p=1.0
        ),
        A.HorizontalFlip(p=0.5),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0,
        ),
        ToTensorV2(p=1.0),
    ]
)

test_augs = A.Compose(
    [
        A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE, p=1.0),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0,
        ),
        ToTensorV2(p=1.0),
    ]
)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
ValidationError                           Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/albumentations/core/validation.py in _validate_parameters(schema_cls, full_kwargs, param_names, strict)
     66             schema_kwargs["strict"] = strict
---> 67             config = schema_cls(**schema_kwargs)
     68             validated_kwargs = config.model_dump()

/usr/local/lib/python3.11/dist-packages/pydantic/main.py in __init__(self, **data)
    249         __tracebackhide__ = True
--> 250         validated_self = self.__pydantic_validator__.validate_python(data, self_instance=self)
    251         if self is not validated_self:

ValidationError: 1 validation error for InitSchema
size
  Field required [type=missing, input_value={'scale': (0.8, 1.0), 'p'...: None, 'strict': False}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.12/v/missing

The above exception was the direct cause of the following exception:

ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/1812744331.py in <cell line: 0>()
     22 train_augs = A.Compose(
     23     [
---> 24         A.RandomResizedCrop(
     25             height=IMAGE_SIZE, width=IMAGE_SIZE, scale=(0.8, 1.0), p=1.0
     26         ),

/usr/local/lib/python3.11/dist-packages/albumentations/core/validation.py in custom_init(self, *args, **kwargs)
    103                 full_kwargs, param_names, strict = cls._process_init_parameters(original_init, args, kwargs)
    104 
--> 105                 validated_kwargs = cls._validate_parameters(
    106                     dct["InitSchema"],
    107                     full_kwargs,

/usr/local/lib/python3.11/dist-packages/albumentations/core/validation.py in _validate_parameters(schema_cls, full_kwargs, param_names, strict)
     69             validated_kwargs.pop("strict", None)
     70         except ValidationError as e:
---> 71             raise ValueError(str(e)) from e
     72         except Exception as e:
     73             if strict:

ValueError: 1 validation error for InitSchema
size
  Field required [type=missing, input_value={'scale': (0.8, 1.0), 'p'...: None, 'strict': False}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.12/v/missing

## === cell 3
train_df = pd.read_csv(TRAIN_CSV_PATH)
train_df["label"] = train_df["label"].astype(int)

train_split, val_split = train_test_split(
    train_df, test_size=0.1, stratify=train_df["label"], random_state=SEED
)

train_dataset = CassavaDataset(train_split, TRAIN_IMAGE_PATH, transforms=train_augs)
val_dataset = CassavaDataset(val_split, TRAIN_IMAGE_PATH, transforms=test_augs)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3109758018.py in <cell line: 0>()
      7 )
      8 
----> 9 train_dataset = CassavaDataset(train_split, TRAIN_IMAGE_PATH, transforms=train_augs)
     10 val_dataset = CassavaDataset(val_split, TRAIN_IMAGE_PATH, transforms=test_augs)
     11 

NameError: name 'train_augs' is not defined

## === cell 4
model = timm.create_model(
    "tf_efficientnet_b4_ns", pretrained=True, num_classes=OUT_FEATURES
)
model = model.to(DEVICE)

if os.path.exists(B4_PATH):
    state = torch.load(B4_PATH, map_location=DEVICE)
    model.load_state_dict(state["model_state_dict"])

criterion = nn.CrossEntropyLoss()
optimizer = OPTIMIZER(model.parameters(), lr=LR_MAX)

scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=LR_MAX,
    total_steps=NUM_EPOCHS * len(train_loader),
    pct_start=0.1,
    anneal_strategy="cos",
    final_div_factor=LR_MAX / LR_FINAL,
)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/17437708.py in <cell line: 0>()
     16     optimizer,
     17     max_lr=LR_MAX,
---> 18     total_steps=NUM_EPOCHS * len(train_loader),
     19     pct_start=0.1,
     20     anneal_strategy="cos",

NameError: name 'train_loader' is not defined

## === cell 5
best_val_acc = 0.0
for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    epoch_loss = 0.0
    for images, targets in tqdm(
        train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [train]"
    ):
        images = images.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        scheduler.step()

        epoch_loss += loss.item() * images.size(0)

    epoch_loss /= len(train_loader.dataset)

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, targets in tqdm(
            val_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [val]"
        ):
            images = images.to(DEVICE, non_blocking=True)
            targets = targets.to(DEVICE, non_blocking=True)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

    val_acc = correct / total
    print(f"Epoch {epoch}: Train loss {epoch_loss:.4f} | Val Acc {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
            },
            B4_PATH,
        )




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1643518783.py in <cell line: 0>()
      5     epoch_loss = 0.0
      6     for images, targets in tqdm(
----> 7         train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [train]"
      8     ):
      9         images = images.to(DEVICE, non_blocking=True)

NameError: name 'train_loader' is not defined

## === cell 6
class TestDataset(Dataset):
    def __init__(self, df, img_dir, transforms):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_id = self.df.iloc[idx]["image_id"]
        img_path = os.path.join(self.img_dir, img_id)
        image = np.array(Image.open(img_path).convert("RGB"))
        image = self.transforms(image=image)["image"]
        return image, img_id


sample_sub = pd.read_csv(
    os.path.join(
        BASE_INPUT, "cassava-leaf-disease-classification", "sample_submission.csv"
    )
)
test_dataset = TestDataset(sample_sub, TEST_IMAGE_PATH, test_augs)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
)

model.eval()
all_preds = []
all_ids = []

with torch.no_grad():
    for images, ids in tqdm(test_loader, desc="Inference"):
        images = images.to(DEVICE)
        probs_sum = torch.zeros(images.size(0), OUT_FEATURES, device=DEVICE)
        for _ in range(TTA):
            aug_images = images  # base images already augmented by test_augs; additional randomness not needed
            outputs = model(aug_images)
            probs = F.softmax(outputs, dim=1)
            probs_sum += probs
        probs_avg = probs_sum / TTA
        preds = torch.argmax(probs_avg, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_ids.extend(ids)

submission = pd.DataFrame({"image_id": all_ids, "label": all_preds})
submission.to_csv(SUBMISSION_PATH, index=False)
print(f"Submission written to {SUBMISSION_PATH}")

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/622585826.py in <cell line: 0>()
     23     )
     24 )
---> 25 test_dataset = TestDataset(sample_sub, TEST_IMAGE_PATH, test_augs)
     26 test_loader = DataLoader(
     27     test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True

NameError: name 'test_augs' is not defined
