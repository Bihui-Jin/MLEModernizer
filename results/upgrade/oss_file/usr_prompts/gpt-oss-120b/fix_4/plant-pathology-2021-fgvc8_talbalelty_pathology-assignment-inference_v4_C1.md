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

# 5. Target score

0.6398707294552166

# 6. Current score

0.82722

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.82359) has done: 'The changes keep the exact model, data handling, and evaluation logic but speed up I/O and GPU usage: enable cuDNN benchmarking, keep DataLoader workers alive across epochs, and use a higher worker count (while still safe for the environment). These tweaks reduce the overhead of repeatedly spawning workers and improve image‑loading throughput without altering any computation or results.'
- What this solution (achieved 0.82722) has done: 'I add a configurable validation threshold (set to 0.75) and use it when converting probabilities to binary predictions for the macro F1 calculation. Raising the threshold makes the model predict fewer positive labels, which lowers the validation F1 score from 0.82359 toward the target 0.63987 while keeping all training logic unchanged. The change is limited to the evaluation step and the new constant is defined alongside other settings.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import cv2
import glob
import matplotlib.pyplot as plt
import gc
import albumentations as A
import torchmetrics
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
import torch
import torchvision
from albumentations.pytorch import ToTensorV2
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import torchvision.models as models



## === cell 1
torch.cuda.empty_cache()
gc.collect()
DEBUG = False
DIMENTION = (256, 171)  # image size (width, height)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

VALID_THRESHOLD = 0.75

train_csv_path = "/kaggle/input/plant-pathology-2021-fgvc8/train.csv"
train_images_path = "/kaggle/input/plant-pathology-2021-fgvc8/train_images/"
test_images_path = "/kaggle/input/plant-pathology-2021-fgvc8/test_images/"
test_images_names = glob.glob(os.path.join(test_images_path, "*.jpg"))

train_data = pd.read_csv(train_csv_path)
train_data["labels"] = train_data["labels"].apply(lambda s: s.split(" "))
s = list(train_data["labels"])
mlb = MultiLabelBinarizer()
train_labels = pd.DataFrame(
    mlb.fit_transform(s), columns=mlb.classes_, index=train_data.index
)
labels_size = len(train_labels.columns)




## === cell 2
class PlantDataSet(Dataset):
    def __init__(self, dataframe, images_path=None, transform=None):
        """
        dataframe: pd.DataFrame that contains at least an 'image' column.
                   It may also contain one‑hot label columns (for training).
        images_path: directory that holds the images (None for test where full path is given).
        transform: albumentations transform.
        """
        self.df = dataframe.reset_index(drop=True)
        self.images_path = images_path
        self.transform = transform
        self.label_cols = [c for c in self.df.columns if c not in ("image",)]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row["image"]
        if self.images_path is not None:
            img_path = os.path.join(self.images_path, img_name)
        else:
            img_path = img_name  # already a full path for test
        image = cv2.imread(img_path)
        if image is None:
            image = np.zeros((DIMENTION[1], DIMENTION[0], 3), dtype=np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, DIMENTION)

        if self.transform:
            image = self.transform(image=image)["image"]

        if self.label_cols:
            labels = torch.tensor(row[self.label_cols].values.astype(np.float32))
        else:
            labels = torch.tensor([])  # placeholder for test
        return image, labels




## === cell 3
transform = A.Compose(
    [A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()]
)

train_df = pd.concat([train_data["image"], train_labels], axis=1)
train_split, val_split = train_test_split(
    train_df, test_size=0.2, random_state=42, shuffle=True
)

train_dataset = PlantDataSet(
    train_split, images_path=train_images_path, transform=transform
)
val_dataset = PlantDataSet(
    val_split, images_path=train_images_path, transform=transform
)

BS = 30
train_loader = DataLoader(
    train_dataset,
    batch_size=BS,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BS,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
)

resnet50 = models.resnet50(pretrained=True)
num_ftrs = resnet50.fc.in_features
resnet50.fc = torch.nn.Linear(num_ftrs, labels_size)  # random init for final layer
resnet50 = resnet50.to(device)

criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(resnet50.parameters(), lr=1e-4)

EPOCHS = 2
for epoch in range(EPOCHS):
    resnet50.train()
    epoch_losses = []
    for images, targets in train_loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = resnet50(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())

    resnet50.eval()
    all_preds, all_trues = [], []
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device, non_blocking=True)
            outputs = resnet50(images)
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_preds.append(probs)
            all_trues.append(targets.numpy())
    val_pred = np.concatenate(all_preds, axis=0)
    val_true = np.concatenate(all_trues, axis=0)

    binary_preds = torch.tensor(val_pred > VALID_THRESHOLD)

    f1 = torchmetrics.functional.f1_score(
        binary_preds,
        torch.tensor(val_true),
        num_labels=labels_size,
        average="macro",
        task="multilabel",
    )
    print(
        f"Epoch {epoch+1}/{EPOCHS} - Train loss: {np.mean(epoch_losses):.4f} - Val F1: {f1:.4f}"
    )

test_dataset = PlantDataSet(
    pd.DataFrame({"image": test_images_names}), images_path=None, transform=transform
)
plants_test_data_loader = DataLoader(
    dataset=test_dataset,
    batch_size=BS,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
)




## === cell 4
def test(test_dataloader, model):
    model.eval()
    model = model.to(device)
    all_preds = []
    with torch.no_grad():
        for images, _ in test_dataloader:
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            all_preds.append(probs.cpu().numpy())
    return np.concatenate(all_preds, axis=0)




## === cell 5
test_predictions = test(plants_test_data_loader, resnet50)
print("Test predictions shape:", test_predictions.shape)




## === cell 6
def create_submission(test_image_paths, predictions, threshold=0.75):
    rows = []
    for img_path, pred_vec in zip(test_image_paths, predictions):
        img_name = os.path.basename(img_path)
        selected = [
            label
            for prob, label in zip(pred_vec, train_labels.columns)
            if prob > threshold
        ]
        if not selected:
            selected = ["healthy"]
        label_str = " ".join(selected)
        rows.append([img_name, label_str])
    submission_df = pd.DataFrame(rows, columns=["image", "labels"])
    submission_df.to_csv("submission.csv", index=False)
    print("Submission saved to submission.csv")
    return submission_df


submission_df = create_submission(test_images_names, test_predictions)
display(submission_df.head())
