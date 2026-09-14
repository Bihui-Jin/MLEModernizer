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

0.8360531882744031

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.06278) has done: 'The failure was caused by iterating over a directory that contains a sub‑folder (`test_images`) and trying to open it as an image, which raised `IsADirectoryError`. The prediction loop is updated to filter only actual image files (jpg/jpeg/png) and to sort them for deterministic output. This ensures the lengths of `image_ids` and `predictions` match the test set, allowing a correct `submission.csv` to be created while keeping the original model logic unchanged.'
- What this solution (achieved 0.51308) has done: 'I fixed the model construction error by using the current enum name for the Vision‑Transformer weights (`DEFAULT` instead of the removed `IMAGENET1K_V1`). I also added a small safety fallback to keep the code robust if the enum changes again. With the model correctly instantiated, the inference loop runs, and the script now writes a proper `submission.csv` that matches the test set length.'
- What this solution (achieved 0.16218) has done: 'I add a lightweight fine‑tuning stage that trains the chosen pretrained model (EfficientNet‑V2‑L by default) on a small subset of the training images for a couple of epochs, then use the fine‑tuned model for inference. This keeps the original architecture and inference logic while giving the model task‑specific knowledge, which should raise the accuracy from 0.513 toward the target 0.836. I also switch the default `model_select` to the faster EfficientNet model and ensure deterministic ordering of test files.'

# 9. Code solution

## === cell 0
import os, random, json
import pandas as pd
import numpy as np
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")



## === cell 1
BASE_PATH = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_IMG_DIR = os.path.join(BASE_PATH, "train_images")
TEST_IMG_DIR = os.path.join(BASE_PATH, "test_images")
TRAIN_CSV = os.path.join(BASE_PATH, "train.csv")
SAMPLE_SUBMIT = os.path.join(BASE_PATH, "sample_submission.csv")


class CassavaDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.loc[idx, "image_id"]
        label = self.df.loc[idx, "label"]
        img_path = os.path.join(self.img_dir, img_name)
        image = torchvision.io.read_image(img_path).float() / 255.0  # C×H×W in [0,1]
        if self.transform:
            image = self.transform(image)
        return image, int(label)


class CassavaTestDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.filenames = sorted(
            [
                f
                for f in os.listdir(img_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
        )

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_name = self.filenames[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = torchvision.io.read_image(img_path).float() / 255.0
        if self.transform:
            image = self.transform(image)
        return image, img_name


transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

train_df = pd.read_csv(TRAIN_CSV)

val_size = int(0.1 * len(train_df))
train_size = len(train_df) - val_size
train_subset, val_subset = random_split(
    train_df, [train_size, val_size], generator=torch.Generator().manual_seed(42)
)

train_dataset = CassavaDataset(train_subset, TRAIN_IMG_DIR, transform=transform)
val_dataset = CassavaDataset(val_subset, TRAIN_IMG_DIR, transform=transform)

test_dataset = CassavaTestDataset(TEST_IMG_DIR, transform=transform)

model = torchvision.models.efficientnet_v2_l(weights="DEFAULT")
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 5)  # 5 classes
model = model.to(device)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_56/2024188867.py in <cell line: 0>()
     70 )
     71 
---> 72 train_dataset = CassavaDataset(train_subset, TRAIN_IMG_DIR, transform=transform)
     73 val_dataset = CassavaDataset(val_subset, TRAIN_IMG_DIR, transform=transform)
     74 

/tmp/ipykernel_56/2024188867.py in __init__(self, df, img_dir, transform)
     10 class CassavaDataset(Dataset):
     11     def __init__(self, df, img_dir, transform=None):
---> 12         self.df = df.reset_index(drop=True)
     13         self.img_dir = img_dir
     14         self.transform = transform

AttributeError: 'Subset' object has no attribute 'reset_index'

## === cell 2
train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
scaler = torch.cuda.amp.GradScaler()  # mixed‑precision scaler

model.train()
epochs = 3
for epoch in range(epochs):
    epoch_loss = 0.0
    for imgs, labels in tqdm(
        train_loader, desc=f"Fine‑tuning Epoch {epoch+1}/{epochs}"
    ):
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(imgs)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item() * imgs.size(0)

    avg_loss = epoch_loss / len(train_loader.dataset)
    print(f"Epoch {epoch+1} – Avg loss: {avg_loss:.4f}")

model.eval()
torch.cuda.empty_cache()



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/2083271083.py in <cell line: 0>()
      1 # Training utilities
      2 train_loader = DataLoader(
----> 3     train_dataset,
      4     batch_size=8,
      5     shuffle=True,

NameError: name 'train_dataset' is not defined

## === cell 3
test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
)

predictions = []
image_ids = []

with torch.no_grad():
    for imgs, names in tqdm(test_loader, desc="Test Inference"):
        imgs = imgs.to(device, non_blocking=True)
        with torch.cuda.amp.autocast():
            outputs = model(imgs)
        _, preds = torch.max(outputs, 1)
        predictions.extend(preds.cpu().tolist())
        image_ids.extend(names)

submission = pd.DataFrame({"image_id": image_ids, "label": predictions})
sample_sub = pd.read_csv(SAMPLE_SUBMIT)
submission = sample_sub[["image_id"]].merge(submission, on="image_id", how="left")
submission["label"] = submission["label"].fillna(0).astype(int)  # safety fallback

output_path = "/kaggle/working/submission.csv"
submission.to_csv(output_path, index=False)
print(f"Submission saved to {output_path}")

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/316954934.py in <cell line: 0>()
      1 # Inference on test set
      2 test_loader = DataLoader(
----> 3     test_dataset,
      4     batch_size=64,
      5     shuffle=False,

NameError: name 'test_dataset' is not defined
