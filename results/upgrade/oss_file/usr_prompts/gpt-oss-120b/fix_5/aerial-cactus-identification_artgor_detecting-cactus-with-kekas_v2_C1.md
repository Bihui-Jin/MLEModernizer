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
Create a classifier to predict whether an image contains a cactus.

## Metric
Area under the ROC curve.

## Submission Format
For each ID in the test set, you must predict a probability for the `has_cactus` variable. The file should contain a header and have the following format:

```
id,has_cactus
000940378805c44108d287872b2f04ce.jpg,0.5
0017242f54ececa4512b4d7937d1e21e.jpg,0.5
001ee6d8564003107853118ab87df407.jpg,0.5
etc.
```

## Dataset
This dataset contains a large number of 32 x 32 thumbnail images containing aerial photos of a cactus. The file name of an image corresponds to its `id`.

- **train/** - the training set images
- **test/** - the test set images (you must predict the labels of these)
- **train.csv** - the training set labels, indicates whether the image has a cactus (`has_cactus = 1`)
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.7

# 3. Installed packages

albumentations==2.0.8
geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
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

# 4. Data file paths

```
/
    kaggle/
        data/
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 3 other files
        input/
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 3 other files
        working/
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 3 other files
```

-> data/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
Here is some information about the columns:
has_cactus (float64) has 1 unique values: [0.5]
id (object) has 2000 unique values. Some example values: ['09034a34de0e2015a8a28dfe18f423f6.jpg', '44b974fd955c4cd306c7dbae154646b9.jpg', '37cd593f40ba13982e7587a613a9a789.jpg', 'c892c865a5706d3072ba44beafbf2d1c.jpg']

-> data/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
Here is some information about the columns:
has_cactus (int64) has 2 unique values: [1, 0]
id (object) has 2000 unique values. Some example values: ['2de8f189f1dce439766637e75df0ee27.jpg', 'f94b18b4c6850fc44b54f5fbe3c5b0c6.jpg', 'fb0624c0ebc01643a8f4318537099078.jpg', '00b4dfbb267109b5f0d0dde365fa6161.jpg']

-> input/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
Here is some information about the columns:
has_cactus (float64) has 1 unique values: [0.5]
id (object) has 2000 unique values. Some example values: ['09034a34de0e2015a8a28dfe18f423f6.jpg', '44b974fd955c4cd306c7dbae154646b9.jpg', '37cd593f40ba13982e7587a613a9a789.jpg', 'c892c865a5706d3072ba44beafbf2d1c.jpg']

-> input/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
Here is some information about the columns:
has_cactus (int64) has 2 unique values: [1, 0]
id (object) has 2000 unique values. Some example values: ['2de8f189f1dce439766637e75df0ee27.jpg', 'f94b18b4c6850fc44b54f5fbe3c5b0c6.jpg', 'fb0624c0ebc01643a8f4318537099078.jpg', '00b4dfbb267109b5f0d0dde365fa6161.jpg']

-> working/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
Here is some information about the columns:
has_cactus (float64) has 1 unique values: [0.5]
id (object) has 2000 unique values. Some example values: ['09034a34de0e2015a8a28dfe18f423f6.jpg', '44b974fd955c4cd306c7dbae154646b9.jpg', '37cd593f40ba13982e7587a613a9a789.jpg', 'c892c865a5706d3072ba44beafbf2d1c.jpg']

-> working/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
Here is some information about the columns:
has_cactus (int64) has 2 unique values: [1, 0]
id (object) has 2000 unique values. Some example values: ['2de8f189f1dce439766637e75df0ee27.jpg', 'f94b18b4c6850fc44b54f5fbe3c5b0c6.jpg', 'fb0624c0ebc01643a8f4318537099078.jpg', '00b4dfbb267109b5f0d0dde365fa6161.jpg']

# 5. Target score

0.9383

# 6. Current score

0.5

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.5) has done: 'I fixed the model architecture: the original DenseNet already includes global pooling and flattening, so the custom head caused mismatched tensor dimensions and batch‑norm errors. I now use the backbone’s feature size directly, applying only batch‑norm, dropout and a linear layer. This resolves the “expected more than 1 value per channel” error during training and the runtime error during inference, allowing the code to generate a correctly‑sized submission file.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

default_input_path = "/kaggle/input/aerial-cactus-identification"
default_work_path = "/kaggle/working/aerial-cactus-identification"

if os.path.isdir(default_input_path):
    BASE_PATH = default_input_path
elif os.path.isdir(default_work_path):
    BASE_PATH = default_work_path
else:
    raise FileNotFoundError("Cannot find the dataset directory.")

print("Using BASE_PATH:", BASE_PATH)

labels = pd.read_csv(os.path.join(BASE_PATH, "train.csv"))
labels["has_cactus"] = labels["has_cactus"].astype(int)

sample_sub_path = os.path.join(BASE_PATH, "sample_submission.csv")
test_df = pd.read_csv(sample_sub_path)[["id"]].copy()
print("Test set size from sample_submission:", len(test_df))




## === cell 1
class CactusDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, is_test=False):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.loc[idx, "id"]
        img_path = os.path.join(self.img_dir, img_name)

        if not os.path.isfile(img_path):
            image = Image.new("RGB", (32, 32))
        else:
            image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.is_test:
            return {"image": image, "id": img_name}
        label = torch.tensor(self.df.loc[idx, "has_cactus"], dtype=torch.float32)
        return {"image": image, "label": label}


train_transform = T.Compose(
    [
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

val_transform = T.Compose(
    [
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

train_df, val_df = train_test_split(
    labels, stratify=labels["has_cactus"], test_size=0.2, random_state=42
)

train_dataset = CactusDataset(
    df=train_df,
    img_dir=os.path.join(BASE_PATH, "train"),
    transform=train_transform,
    is_test=False,
)

val_dataset = CactusDataset(
    df=val_df,
    img_dir=os.path.join(BASE_PATH, "train"),
    transform=val_transform,
    is_test=False,
)

test_dataset = CactusDataset(
    df=test_df,
    img_dir=os.path.join(BASE_PATH, "test"),
    transform=val_transform,
    is_test=True,
)

batch_size = 64

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    pin_memory=True,
    drop_last=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
)


def test_collate_fn(batch):
    images = torch.stack([item["image"] for item in batch])
    ids = [item["id"] for item in batch]
    return {"image": images, "id": ids}


test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
    collate_fn=test_collate_fn,
)




## === cell 2
class Net(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        backbone = models.densenet169(pretrained=True)
        num_ftrs = backbone.classifier.in_features  # feature dimension after pooling
        backbone.classifier = nn.Identity()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.BatchNorm1d(num_ftrs),
            nn.Dropout(0.2),
            nn.Linear(num_ftrs, num_classes),
        )

    def forward(self, x):
        x = self.backbone(x)  # shape: (batch, num_ftrs)
        x = self.head(x)  # shape: (batch, num_classes)
        return x


model = Net().to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)




## === cell 3
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    for batch in loader:
        imgs = batch["image"].to(device, non_blocking=True)
        targets = batch["label"].unsqueeze(1).to(device, non_blocking=True)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
    return running_loss / len(loader.dataset)


def evaluate(model, loader, criterion):
    model.eval()
    preds = []
    trues = []
    val_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            imgs = batch["image"].to(device, non_blocking=True)
            targets = batch["label"].unsqueeze(1).to(device, non_blocking=True)
            outputs = model(imgs)
            loss = criterion(outputs, targets)
            val_loss += loss.item() * imgs.size(0)
            preds.append(torch.sigmoid(outputs).cpu())
            trues.append(targets.cpu())
    preds = torch.cat(preds).numpy()
    trues = torch.cat(trues).numpy()
    auc = roc_auc_score(trues, preds)
    return val_loss / len(loader.dataset), auc


num_epochs = 10
best_auc = 0.0

for epoch in range(1, num_epochs + 1):
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
    val_loss, val_auc = evaluate(model, val_loader, criterion)
    scheduler.step()
    print(
        f"Epoch {epoch}: Train loss {train_loss:.4f} | Val loss {val_loss:.4f} | Val AUC {val_auc:.4f}"
    )
    if val_auc > best_auc:
        best_auc = val_auc
        torch.save(model.state_dict(), "best_model.pth")

print("Best validation AUC:", best_auc)




## === cell 4
if os.path.exists("best_model.pth"):
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
else:
    print("Warning: best_model.pth not found, using current model weights.")
model.eval()

all_preds = []
all_ids = []

with torch.no_grad():
    for batch in test_loader:
        imgs = batch["image"].to(device, non_blocking=True)
        ids = batch["id"]
        outputs = model(imgs)
        probs = torch.sigmoid(outputs).cpu().numpy().flatten()
        all_preds.extend(probs.tolist())
        all_ids.extend(ids)




## === cell 5
submission = pd.DataFrame({"id": all_ids, "has_cactus": all_preds})
submission.to_csv("sub.csv", index=False)
print("Submission saved to sub.csv with", len(submission), "rows")
