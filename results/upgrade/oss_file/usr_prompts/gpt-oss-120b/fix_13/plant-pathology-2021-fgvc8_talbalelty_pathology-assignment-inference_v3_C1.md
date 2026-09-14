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
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
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
import gc
import glob
import os
import warnings

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchmetrics
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import Dataset, DataLoader

torch.backends.cudnn.benchmark = True

warnings.filterwarnings("ignore")

DEBUG = False
DIMENTION = (256, 256)  # (height, width)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
THRESHOLD = 0.6  # adjusted threshold to bring F1 closer to target

train_data = pd.read_csv("/kaggle/input/plant-pathology-2021-fgvc8/train.csv")
test_images_path = "/kaggle/input/plant-pathology-2021-fgvc8/test_images/"
test_images_names = glob.glob(os.path.join(test_images_path, "*.jpg"))

train_data["labels"] = train_data["labels"].apply(lambda s: s.split(" "))
s = list(train_data["labels"])
mlb = MultiLabelBinarizer()
train_labels = pd.DataFrame(
    mlb.fit_transform(s), columns=mlb.classes_, index=train_data.index
)  # one‑hot encoding
labels_size = len(train_labels.columns)

train_df = train_data.copy()
for col in train_labels.columns:
    train_df[col] = train_labels[col].values




## === cell 1
class PlantDataSet(Dataset):
    def __init__(self, dataset, images_path, transform=None):
        super(PlantDataSet, self).__init__()
        self.dataset = dataset
        self.images_path = images_path
        self.transform = transform

    def __getitem__(self, idx):
        if self.images_path is not None:
            image_path = os.path.join(self.images_path, self.dataset.loc[idx, "image"])
            image = cv2.imread(image_path)
            if image is None:
                raise FileNotFoundError(f"Image not found: {image_path}")
            if image.ndim == 2:  # grayscale
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (DIMENTION[1], DIMENTION[0]))

            label_series = self.dataset.loc[
                idx, self.dataset.columns.difference(["image", "labels"])
            ]
            labels = torch.tensor(
                label_series.to_numpy(dtype=np.float32), dtype=torch.float32
            )
        else:
            image_path = self.dataset[idx]
            image = cv2.imread(image_path)
            if image is None:
                raise FileNotFoundError(f"Test image not found: {image_path}")
            if image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (DIMENTION[1], DIMENTION[0]))
            labels = torch.tensor([], dtype=torch.float32)  # no labels for test

        if self.transform:
            image = self.transform(image=image)["image"]

        return image, labels

    def __len__(self):
        return len(self.dataset)




## === cell 2
train_transform = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)

val_test_transform = A.Compose(
    [
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)

test_dataset = PlantDataSet(test_images_names, None, val_test_transform)

BS = 64
plants_test_data_loader = DataLoader(dataset=test_dataset, batch_size=BS, shuffle=False)




## === cell 3
def test(test_dataloader, model):
    predictions = None
    model.eval()
    model = model.to(device)
    with torch.no_grad():
        for i, (images, _) in enumerate(test_dataloader):
            images = images.float().to(device)
            output = model(images)
            probabilities = torch.sigmoid(output)
            if i == 0:
                predictions = probabilities.detach().cpu().numpy()
            else:
                predictions = np.concatenate(
                    (predictions, probabilities.detach().cpu().numpy()), axis=0
                )
    return np.array(predictions)




## === cell 4
def create_submission(test_images_paths, predictions):
    rows = []
    for img_path, prediction in zip(test_images_paths, predictions):
        name = os.path.basename(img_path)
        arr = [
            label
            for pred, label in zip(prediction, train_labels.columns)
            if pred > THRESHOLD
        ]
        if len(arr) == 0:
            arr = ["healthy"]
        prediction_labels = " ".join(arr)
        rows.append({"image": name, "labels": prediction_labels})
    submission_df = pd.DataFrame(rows, columns=["image", "labels"])
    submission_df.to_csv("submission.csv", index=False)




## === cell 5
train_split, val_split = train_test_split(train_df, test_size=0.2, random_state=42)
train_split = train_split.reset_index(drop=True)
val_split = val_split.reset_index(drop=True)

train_dataset = PlantDataSet(
    train_split,
    "/kaggle/input/plant-pathology-2021-fgvc8/train_images/",
    train_transform,
)
val_dataset = PlantDataSet(
    val_split,
    "/kaggle/input/plant-pathology-2021-fgvc8/train_images/",
    val_test_transform,
)

train_loader = DataLoader(
    train_dataset, batch_size=BS, shuffle=True, num_workers=4, pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=BS, shuffle=False, num_workers=4, pin_memory=True
)


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    for images, labels in loader:
        images = images.float().to(device)
        labels = labels.float().to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


def evaluate(model, loader, threshold=THRESHOLD):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.float().to(device)
            labels = labels.float()
            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu()
            all_preds.append(probs)
            all_labels.append(labels)
    preds = torch.cat(all_preds)
    targets = torch.cat(all_labels)
    preds_bin = (preds > threshold).int()
    f1_metric = torchmetrics.F1Score(
        task="multilabel", num_labels=labels_size, average="macro"
    )
    return f1_metric(preds_bin, targets.int()).item()




## === cell 6
try:
    resnet50 = models.resnet50(pretrained=False, num_classes=labels_size)
    resnet50.load_state_dict(
        torch.load("../input/resnet50-final/resnet50_final.pth", map_location=device)
    )
except FileNotFoundError:
    resnet50 = models.resnet50(pretrained=True)
    resnet50.fc = nn.Linear(resnet50.fc.in_features, labels_size)

resnet50 = resnet50.to(device)




## === cell 7
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(resnet50.parameters(), lr=5e-5)

EPOCHS = 5  # reduced for quicker execution while still learning
for epoch in range(1, EPOCHS + 1):
    train_loss = train_one_epoch(resnet50, train_loader, criterion, optimizer)
    val_f1 = evaluate(resnet50, val_loader)
    print(
        f"Epoch {epoch}/{EPOCHS} - Train loss: {train_loss:.4f} - Val F1: {val_f1:.4f}"
    )




## === cell 8
test_predictions = test(plants_test_data_loader, resnet50)
create_submission(test_images_names, test_predictions)
