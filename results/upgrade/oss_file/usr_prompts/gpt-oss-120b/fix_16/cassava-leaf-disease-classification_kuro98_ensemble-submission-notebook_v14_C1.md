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

3.12

# 3. Installed packages

geopandas==0.14.4
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

0.9008763977032336

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.12369) has done: 'I fixed the transformation pipeline by replacing the missing `v2.ToImage()` with `v2.ToTensor()`, which correctly converts PIL images to tensors and already returns float values, removing the unnecessary `ToDtype` step. This resolves the import error, allows the DataLoader to be created, and ensures the inference loop runs so predictions are collected and a proper CSV submission is written.'

# 9. Code solution

## === cell 0
import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import train_test_split

if "pin_mem" not in globals():
    pin_mem = True
if "use_cuda" not in globals():
    use_cuda = torch.cuda.is_available()
if "device" not in globals():
    device = torch.device("cuda" if use_cuda else "cpu")



## === cell 1
candidate_dirs = [
    "/kaggle/input/cassava-leaf-disease-classification",
    "/kaggle/working/cassava-leaf-disease-classification",
    "data",
    "input",
    "working",
]
base_dir = next((d for d in candidate_dirs if os.path.isdir(d)), None)
if base_dir is None:
    raise FileNotFoundError(
        "Unable to locate dataset directory. Checked: " + ", ".join(candidate_dirs)
    )

train_csv_path = os.path.join(base_dir, "train.csv")
sample_sub_path = os.path.join(base_dir, "sample_submission.csv")
train_img_dir = os.path.join(base_dir, "train_images")
test_img_dir = os.path.join(base_dir, "test_images")

batch_size = 64
num_epochs = 3
learning_rate = 1e-3
num_classes = 5

img_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class CassavaDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, img_dir, transform=None, is_test=False):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_id = self.df.iloc[idx]["image_id"]
        img_path = os.path.join(self.img_dir, img_id)
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        if self.is_test:
            return img, img_id
        label = self.df.iloc[idx]["label"]
        return img, label


train_df = pd.read_csv(train_csv_path)

train_idx, val_idx = train_test_split(
    train_df.index,
    test_size=0.2,
    stratify=train_df["label"],
    random_state=42,
)

split_dir = os.path.join("working")
os.makedirs(split_dir, exist_ok=True)

train_split_path = os.path.join(split_dir, "train_split.csv")
val_split_path = os.path.join(split_dir, "val_split.csv")
train_df.loc[train_idx].to_csv(train_split_path, index=False)
train_df.loc[val_idx].to_csv(val_split_path, index=False)

train_dataset = CassavaDataset(train_split_path, train_img_dir, transform=img_transform)
val_dataset = CassavaDataset(val_split_path, train_img_dir, transform=img_transform)
test_dataset = CassavaDataset(
    sample_sub_path, test_img_dir, transform=img_transform, is_test=True
)

num_workers = min(4, max(1, os.cpu_count() or 1))



## === cell 2
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=pin_mem,
    persistent_workers=True,
    prefetch_factor=2,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=pin_mem,
    persistent_workers=True,
    prefetch_factor=2,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=pin_mem,
    persistent_workers=True,
    prefetch_factor=2,
)



## === cell 3
if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")  # noqa: E402

eff_model = models.efficientnet_b0(weights="DEFAULT")
eff_model.classifier[1] = torch.nn.Linear(
    eff_model.classifier[1].in_features, num_classes
)
eff_model = eff_model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(eff_model.parameters(), lr=learning_rate)

scaler = torch.cuda.amp.GradScaler() if use_cuda else None

for epoch in range(1, num_epochs + 1):
    eff_model.train()
    running_loss = 0.0
    for imgs, labels in train_loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        if use_cuda:
            with torch.cuda.amp.autocast():
                outputs = eff_model(imgs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = eff_model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * imgs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch [{epoch}/{num_epochs}] - Train loss: {epoch_loss:.4f}")

    eff_model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if use_cuda:
                with torch.cuda.amp.autocast():
                    outputs = eff_model(imgs)
            else:
                outputs = eff_model(imgs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    val_acc = correct / total if total > 0 else 0.0
    print(f"Epoch [{epoch}/{num_epochs}] - Val accuracy: {val_acc:.4f}")



## === cell 4
eff_model.eval()
test_preds = []
image_ids = []

with torch.no_grad():
    for batch in test_loader:
        if isinstance(batch, (list, tuple)):
            imgs, ids = batch
            image_ids.extend(ids)
        elif isinstance(batch, dict):
            imgs = batch["image"]
            ids = batch.get("image_id")
            if ids is not None:
                image_ids.extend(ids)
        else:
            imgs = batch
        imgs = imgs.to(device, non_blocking=True)
        outputs = eff_model(imgs)
        _, preds = torch.max(outputs, 1)
        test_preds.extend(preds.cpu().numpy())

if len(image_ids) != len(test_preds):
    image_ids = [f"test_{i}.jpg" for i in range(len(test_preds))]

submission = pd.DataFrame({"image_id": image_ids, "label": test_preds})
submission_path = os.path.join("working", "submission.csv")
os.makedirs(os.path.dirname(submission_path), exist_ok=True)
submission.to_csv(submission_path, index=False)
print(f"Submission file written to {submission_path}")
