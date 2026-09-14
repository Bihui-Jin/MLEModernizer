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
imbalanced-learn==0.13.0
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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

0.6128739800543971

# 6. Current score

0.78774

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.78774) has done: 'The fix removes the failing `imblearn` import, replaces the missing checkpoint with a fresh ImageNet‑pretrained ResNet34, adds a small training loop (few epochs) so the model learns from the provided training data, corrects the dataset class and the inference loop (skip directories, use the proper paths), and finally writes a correctly‑shaped `submission.csv`. These changes unblock execution and give a realistic accuracy that moves the score toward the target while keeping the core modeling logic unchanged.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models

try:
    from imblearn.over_sampling import SMOTE
except Exception:
    SMOTE = None



## === cell 1
BASE = "../input/cassava-leaf-disease-classification"
train_csv_path = os.path.join(BASE, "train.csv")
train_img_dir = os.path.join(BASE, "train_images")

df_train = pd.read_csv(train_csv_path)
train_image_paths = [
    os.path.join(train_img_dir, fname) for fname in df_train["image_id"].values
]
train_targets = df_train["label"].values.astype(np.int64)



## === cell 2
from sklearn.model_selection import train_test_split

train_paths, val_paths, train_labels, val_labels = train_test_split(
    train_image_paths,
    train_targets,
    test_size=0.1,
    stratify=train_targets,
    random_state=42,
)




## === cell 3
class CassavaDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = int(self.labels[idx])
        return image, label


input_size = 384
imagenet_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
train_transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(*imagenet_stats),
    ]
)
val_transform = transforms.Compose(
    [
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(*imagenet_stats),
    ]
)



## === cell 4
batch_size = 32
train_dataset = CassavaDataset(train_paths, train_labels, transform=train_transform)
val_dataset = CassavaDataset(val_paths, val_labels, transform=val_transform)

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
)



## === cell 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet = models.resnet34(pretrained=True)
num_ftrs = resnet.fc.in_features
resnet.fc = nn.Linear(num_ftrs, 5)  # five classes
resnet = resnet.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(resnet.parameters(), lr=1e-3)




## === cell 6
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(device)
            lbls = lbls.to(device)
            outs = model(imgs)
            _, preds = torch.max(outs, 1)
            correct += (preds == lbls).sum().item()
            total += lbls.size(0)
    return correct / total if total > 0 else 0.0


best_val_acc = 0.0
num_epochs = 3  # small number to stay within time limits
for epoch in range(num_epochs):
    resnet.train()
    running_loss = 0.0
    for imgs, lbls in train_loader:
        imgs = imgs.to(device)
        lbls = lbls.to(device)

        optimizer.zero_grad()
        outputs = resnet(imgs)
        loss = criterion(outputs, lbls)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    val_acc = evaluate(resnet, val_loader)
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(resnet.state_dict(), "best_resnet.pth")
    print(
        f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f} - Val Acc: {val_acc:.4f}"
    )

print(f"Best validation accuracy: {best_val_acc:.4f}")



## === cell 7
resnet.load_state_dict(torch.load("best_resnet.pth"))
resnet.eval()



## === cell 8
test_dir = os.path.join(BASE, "test_images")
test_files = [f for f in os.listdir(test_dir) if f.lower().endswith(".jpg")]
test_paths = [os.path.join(test_dir, f) for f in test_files]

test_transform = transforms.Compose(
    [
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(*imagenet_stats),
    ]
)

preds = []
for img_path in test_paths:
    img = Image.open(img_path).convert("RGB")
    img = test_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = resnet(img)
        _, pred = torch.max(out, 1)
    preds.append(int(pred.item()))



## === cell 9
submission = pd.DataFrame({"image_id": test_files, "label": preds})
submission.to_csv("submission.csv", index=False)
print("Submission file written to submission.csv")
