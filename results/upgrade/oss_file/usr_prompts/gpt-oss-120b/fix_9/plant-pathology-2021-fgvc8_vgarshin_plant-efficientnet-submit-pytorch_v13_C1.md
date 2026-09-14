# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Detect apple diseases from images.

## Metric
Mean F1-Score

## Submission Format
labels should be a space-delimited list.

The file should contain a header and have the following format:

```
image, labels
85f8cb619c66b863.jpg,healthy
ad8770db05586b59.jpg,healthy
c7b03e718489f3ca.jpg,healthy
```

## Dataset
**train.csv** - the training set metadata.

- `image` - the image ID.
- `labels` - the target classes, a space delimited list of all diseases found in the image. Unhealthy leaves with too many diseases to classify visually will have the `complex` class, and may also have a subset of the diseases identified.

**sample_submission.csv** - A sample submission file in the correct format.

- `image`
- `labels`

**train_images** - The training set images.

**test_images** - The test set images. This competition has a hidden test set: only three images are provided here as samples while the remaining 5,000 images will be available to your notebook once it is submitted.

# 2. Python version

3.9

# 3. Installed packages

geopandas==0.14.4
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
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
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
        input/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
        working/
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
```

-> data/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> data/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os
import time
import numpy as np
import pandas as pd
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision import io
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
KAGGLE = True  # keep flag for compatibility
TEST = True  # we are generating predictions for the test set

torch.backends.cudnn.benchmark = True




## === cell 1
if KAGGLE:
    DATA_PATH = "../input/plant-pathology-2021-fgvc8"
else:
    DATA_PATH = "./data"

TRAIN_IMGS_PATH = os.path.join(DATA_PATH, "train_images")
TEST_IMGS_PATH = os.path.join(DATA_PATH, "test_images")
START_TIME = time.time()




## === cell 2
train_csv = os.path.join(DATA_PATH, "train.csv")
train_df = pd.read_csv(train_csv)

train_df["label_list"] = train_df["labels"].str.split()
all_labels = sorted({lbl for sublist in train_df["label_list"] for lbl in sublist})
label_to_idx = {lbl: i for i, lbl in enumerate(all_labels)}
idx_to_label = {i: lbl for lbl, i in label_to_idx.items()}
NUM_CLASSES = len(all_labels)

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def load_image_tensor(img_path: str) -> torch.Tensor:
    """
    Fast image loading with torchvision.io, resizing, conversion to float,
    and normalization. Returns a tensor ready for the model.
    """
    try:
        img = io.read_image(img_path)  # shape (3, H, W), RGB, uint8
    except Exception:
        img = torch.zeros((3, 224, 224), dtype=torch.uint8)

    if img.numel() == 0:
        img = torch.zeros((3, 224, 224), dtype=torch.uint8)

    img = img.float() / 255.0
    img = F.interpolate(
        img.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False
    ).squeeze(0)

    img = (img - _MEAN) / _STD
    return img  # shape (3, 224, 224)


class PlantDataset(torch.utils.data.Dataset):
    def __init__(self, df, img_dir, training=True):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.training = training

        image_paths = [
            os.path.join(self.img_dir, img_name) for img_name in self.df["image"]
        ]
        tensors = [load_image_tensor(p) for p in image_paths]
        self.image_tensors = torch.stack(tensors)  # shape (N, 3, 224, 224)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_tensor = self.image_tensors[idx]
        if self.training:
            lbls = self.df.iloc[idx]["label_list"]
            target = torch.zeros(NUM_CLASSES, dtype=torch.float32)
            for l in lbls:
                target[label_to_idx[l]] = 1.0
            return img_tensor, target
        else:
            return img_tensor, self.df.iloc[idx]["image"]  # image name for inference


full_dataset = PlantDataset(train_df, TRAIN_IMGS_PATH, training=True)

val_size = int(0.1 * len(full_dataset))
train_size = len(full_dataset) - val_size
train_set, val_set = torch.utils.data.random_split(
    full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
)

num_workers = min(8, os.cpu_count() or 1)

train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=64,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,
)
val_loader = torch.utils.data.DataLoader(
    val_set,
    batch_size=64,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,
)




## === cell 3
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(
    model.fc.in_features, NUM_CLASSES
)  # adapt to our number of classes
model = model.to(DEVICE)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)


def train_one_epoch(loader):
    model.train()
    running_loss = 0.0
    for imgs, targets in loader:
        imgs = imgs.to(DEVICE)
        targets = targets.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
    return running_loss / len(loader.dataset)


def evaluate(loader):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for imgs, targets in loader:
            imgs = imgs.to(DEVICE)
            targets = targets.to(DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * imgs.size(0)
    return running_loss / len(loader.dataset)


EPOCHS = 4  # a few more epochs for better learning while staying fast
for epoch in range(1, EPOCHS + 1):
    train_loss = train_one_epoch(train_loader)
    val_loss = evaluate(val_loader)
    print(
        f"Epoch {epoch}/{EPOCHS} – Train loss: {train_loss:.4f} – Val loss: {val_loss:.4f}"
    )




## === cell 4
test_files = [
    f
    for f in os.listdir(TEST_IMGS_PATH)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]
df_test = pd.DataFrame(test_files, columns=["image"])


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, df, img_dir):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir

        image_paths = [
            os.path.join(self.img_dir, img_name) for img_name in self.df["image"]
        ]
        tensors = [load_image_tensor(p) for p in image_paths]
        self.image_tensors = torch.stack(tensors)  # shape (N, 3, 224, 224)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_tensor = self.image_tensors[idx]
        img_name = self.df.iloc[idx]["image"]
        return img_tensor, img_name


test_dataset = TestDataset(df_test, TEST_IMGS_PATH)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,
)

model.eval()
predictions = []
threshold = 0.5  # probability threshold for label presence
with torch.no_grad():
    for imgs, names in test_loader:
        imgs = imgs.to(DEVICE)
        logits = model(imgs)
        probs = torch.sigmoid(logits).cpu().numpy()
        for prob_vec, img_name in zip(probs, names):
            idxs = np.where(prob_vec >= threshold)[0]
            if len(idxs) == 0:
                pred_str = "healthy"
            else:
                pred_str = " ".join([idx_to_label[i] for i in idxs])
            predictions.append({"image": img_name, "labels": pred_str})

df_sub = pd.DataFrame(predictions)




## === cell 5
submission_path = "submission.csv"
df_sub.to_csv(submission_path, index=False)
elapsed = time.time() - START_TIME
print(f"Submission written to {submission_path}")
print(f"Time elapsed: {int(elapsed // 60)} min {int(elapsed % 60)} sec")
