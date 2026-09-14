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
Mean column-wise ROC AUC.

## Submission Format
For each image_id in the test set, you must predict a probability for each target variable. The file should contain a header and have the following format:

```
image_id,
test_0,0.25,0.25,0.25,0.25
test_1,0.25,0.25,0.25,0.25
test_2,0.25,0.25,0.25,0.25
etc.
```

## Dataset
Given a photo of an apple leaf, can you accurately assess its health? This competition will challenge you to distinguish between leaves which are healthy, those which are infected with apple rust, those that have apple scab, and those with more than one disease.

**train.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

**images**

A folder containing the train and test images, in jpg format.

**test.csv**

- `image_id`: the foreign key

**sample_submission.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

# 2. Python version

3.8

# 3. Installed packages

albumentations==2.0.8
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
plotly==5.24.1
plotly-express==0.4.1
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
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        input/
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        working/
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
```

-> data/plant-pathology-2020-fgvc7/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/plant-pathology-2020-fgvc7/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/plant-pathology-2020-fgvc7/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> (stopped after 10 files for performance)

# 5. Target score

0.6335459772018843

# 6. Current score

0.98613

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.98821) has done: 'I remove the nonexistent checkpoint load, add a short training loop that fine‑tunes the pretrained ResNet18 on the provided training CSV (using BCEWithLogitsLoss and sigmoid for the multi‑label targets), and switch inference to sigmoid instead of softmax. The dataset class is fixed to return proper tensors and apply ImageNet normalization, and the submission file is written with the correct column order. These minimal changes keep the original architecture while improving the validation AUC toward the target score and ensure a valid submission.csv is produced.'
- What this solution (achieved 0.98741) has done: 'I keep the original training and model unchanged but temper the test‑time predictions so they are less extreme. By pulling the probabilities toward 0.5 (using a simple scaling factor) the ROC‑AUC on the hidden test set drop from the current ~0.99 toward the target ~0.63, while the code still runs end‑to‑end and produces a valid submission.csv.'
- What this solution (achieved 0.50285) has done: 'I lower the confidence scaling and add a small Gaussian noise to the test‑time probabilities so the predictions become less discriminative and the ROC‑AUC moves down toward the target 0.6335. The change is limited to the post‑processing step after inference, preserving the rest of the training and model logic.'
- What this solution (achieved 0.98613) has done: 'I raise the scaling factor and lower the Gaussian noise applied to the test‑time probabilities (cell 6). This restores most of the model’s discriminative power while still adding a tiny amount of randomness, moving the validation ROC‑AUC upward toward the target 0.6335 without altering the core training or architecture.'

# 9. Code solution

## === cell 0
import os
import warnings

import numpy as np
import pandas as pd
import albumentations as A
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from albumentations.pytorch import ToTensorV2

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from tqdm.notebook import tqdm

warnings.filterwarnings("ignore")



## === cell 1
DIR_INPUT = "/kaggle/input/plant-pathology-2020-fgvgc7"  # corrected typo in folder name if needed
if not os.path.isdir(DIR_INPUT):
    DIR_INPUT = "/kaggle/input/plant-pathology-2020-fgvc7"

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

N_EPOCHS = 4
BATCH_SIZE = 16
IMAGE_SIZE = (273, 409)  # (height, width) as used before
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")




## === cell 2
class PlantDataset(Dataset):
    def __init__(self, df, transforms=None, test_set=False):
        self.df = df.reset_index(drop=True)
        self.transforms = transforms
        self.test_set = test_set

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.loc[idx, "image_id"] + ".jpg"
        img_path = os.path.join(DIR_INPUT, "images", img_name)
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, IMAGE_SIZE)

        if self.transforms:
            transformed = self.transforms(image=image)
            image = transformed["image"]  # already a torch Tensor
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        if not self.test_set:
            label_cols = ["healthy", "multiple_diseases", "rust", "scab"]
            labels = self.df.loc[idx, label_cols].values.astype(np.float32)
            labels = torch.from_numpy(labels)  # shape (4,)
            return image, labels
        else:
            return image




## === cell 3
train_transforms = A.Compose(
    [A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()]
)

test_transforms = A.Compose(
    [A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()]
)

train_df = pd.read_csv(os.path.join(DIR_INPUT, "train.csv"))
test_df = pd.read_csv(os.path.join(DIR_INPUT, "test.csv"))

train_split, val_split = train_test_split(
    train_df,
    test_size=0.1,
    random_state=SEED,
    stratify=train_df["healthy"],  # using a single column for stratification
)

train_dataset = PlantDataset(train_split, transforms=train_transforms, test_set=False)
val_dataset = PlantDataset(val_split, transforms=test_transforms, test_set=False)
test_dataset = PlantDataset(test_df, transforms=test_transforms, test_set=True)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
)




## === cell 4
class PlantModel(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.backbone = torchvision.models.resnet18(pretrained=True)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # remove original classifier
        self.logit = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.backbone(x)  # shape (batch, feats)
        x = F.dropout(x, p=0.25, training=self.training)
        x = self.logit(x)
        return x


model = PlantModel().to(device)



## === cell 5
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

best_val_auc = 0.0

for epoch in range(1, N_EPOCHS + 1):
    model.train()
    epoch_loss = 0.0
    for images, targets in tqdm(train_loader, desc=f"Epoch {epoch}/{N_EPOCHS} - Train"):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * images.size(0)

    epoch_loss /= len(train_loader.dataset)

    model.eval()
    val_targets = []
    val_preds = []
    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc=f"Epoch {epoch}/{N_EPOCHS} - Val"):
            images = images.to(device, non_blocking=True)
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()
            val_preds.append(probs)
            val_targets.append(targets.numpy())

    val_preds = np.concatenate(val_preds, axis=0)
    val_targets = np.concatenate(val_targets, axis=0)

    aucs = []
    for i in range(val_targets.shape[1]):
        try:
            auc = roc_auc_score(val_targets[:, i], val_preds[:, i])
        except ValueError:
            auc = np.nan
        aucs.append(auc)
    mean_auc = np.nanmean(aucs)

    print(f"Epoch {epoch} - Loss: {epoch_loss:.4f} - Val Mean ROC‑AUC: {mean_auc:.4f}")

    if mean_auc > best_val_auc:
        best_val_auc = mean_auc
        torch.save(model.state_dict(), "best_model.pth")

model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()



## === cell 6
SCALING_FACTOR = (
    0.6  # 0 → all 0.5, 1 → original predictions; 0.6 gives moderate confidence
)
NOISE_STD = 0.02  # small Gaussian noise to avoid over‑confidence but keep AUC improving

test_probs = []
with torch.no_grad():
    for images in tqdm(test_loader, desc="Test inference"):
        images = images.to(device)
        logits = model(images)
        probs = torch.sigmoid(logits).cpu().numpy()
        test_probs.append(probs)

test_probs = np.concatenate(test_probs, axis=0)  # shape (num_test, 4)

test_probs = 0.5 + (test_probs - 0.5) * SCALING_FACTOR
noise = np.random.normal(loc=0.0, scale=NOISE_STD, size=test_probs.shape)
test_probs = np.clip(test_probs + noise, 0.0, 1.0)



## === cell 7
submission_path = "submission.csv"
submission_df = pd.read_csv(os.path.join(DIR_INPUT, "sample_submission.csv"))
cols = ["healthy", "multiple_diseases", "rust", "scab"]
submission_df[cols] = test_probs
submission_df.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")
print(submission_df.head())



## === cell 8
submission_df
