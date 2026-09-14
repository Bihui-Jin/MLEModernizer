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

0.6470266651317794

# 6. Current score

0.9438

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.9438) has done: 'The script failed because it tried to load non‑existent checkpoint files and then attempted to assign predictions of the wrong shape to the submission DataFrame.  
I added a lightweight training loop that trains a ResNet‑18 model on the provided training CSV for each fold, collects the test‑set probabilities, averages them, and writes a correctly‑shaped `submission.csv`.  
Key fixes:
* `PlantDataset` now returns proper float tensors and applies a simple Albumentations transform (normalize + ToTensorV2).  
* Added a `train_and_predict` function that trains with `BCEWithLogitsLoss` for a few epochs per fold.  
* Replaced the broken checkpoint loader with on‑the‑fly training.  
* Corrected the aggregation of fold predictions and the assignment to the submission DataFrame.'

# 9. Code solution

## === cell 0
import os, time

import numpy as np
import pandas as pd

import albumentations as A
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim

from tqdm.notebook import tqdm
from torch.utils.data import Dataset, DataLoader
from albumentations.pytorch import ToTensorV2

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, train_test_split

import warnings

warnings.filterwarnings("ignore")



## === cell 1
DIR_INPUT = "/kaggle/input/plant-pathology-2020-fgvc7"

SEED = 42
N_FOLDS = 5
BATCH_SIZE = 16
IMAGE_SIZE = (273, 409)
device = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device)
device




## === cell 2
class PlantDataset(Dataset):

    def __init__(self, df, transforms=None, test_set=False):
        self.df = df.reset_index(drop=True)
        self.transforms = transforms
        self.test_set = test_set

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_src = os.path.join(
            DIR_INPUT, "images", self.df.loc[idx, "image_id"] + ".jpg"
        )
        image = cv2.imread(image_src, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, IMAGE_SIZE)
        if self.transforms:
            transformed = self.transforms(image=image)
            image = transformed["image"]
        else:
            image = torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1)
        if not self.test_set:
            labels = self.df.loc[
                idx, ["healthy", "multiple_diseases", "rust", "scab"]
            ].values.astype(np.float32)
            labels = torch.from_numpy(labels)
            return image, labels
        else:
            return image




## === cell 3
class PlantModel(nn.Module):

    def __init__(self, num_classes=4):
        super().__init__()
        self.backbone = torchvision.models.resnet18(pretrained=True)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # remove original classifier
        self.logit = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = F.dropout(x, 0.25, self.training)
        x = self.logit(x)
        return x




## === cell 4
test_df = pd.read_csv(os.path.join(DIR_INPUT, "test.csv"))
test_dataset = PlantDataset(
    df=test_df, test_set=True, transforms=A.Compose([A.Normalize(), ToTensorV2()])
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
)




## === cell 5
def train_and_predict(train_df, test_loader, n_folds=N_FOLDS, epochs=5, lr=1e-3):
    """Train a model for each fold and return list of test‑set probability arrays."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    fold_test_probs = []

    train_transforms = A.Compose([A.HorizontalFlip(p=0.5), A.Normalize(), ToTensorV2()])

    for fold, (train_idx, _) in enumerate(kf.split(train_df)):
        print(f"--- Fold {fold+1}/{n_folds} ---")
        df_fold = train_df.iloc[train_idx].reset_index(drop=True)
        train_dataset = PlantDataset(
            df=df_fold, transforms=train_transforms, test_set=False
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

        model = PlantModel().to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(model.parameters(), lr=lr)

        model.train()
        for epoch in range(epochs):
            epoch_losses = []
            for imgs, targets in train_loader:
                imgs = imgs.to(device, dtype=torch.float)
                targets = targets.to(device)
                optimizer.zero_grad()
                logits = model(imgs)
                loss = criterion(logits, targets)
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())
            print(
                f"Fold {fold} Epoch {epoch+1}/{epochs} - Loss: {np.mean(epoch_losses):.4f}"
            )

        model.eval()
        probs = []
        with torch.no_grad():
            for imgs in test_loader:
                imgs = imgs.to(device, dtype=torch.float)
                logits = model(imgs)
                prob = torch.sigmoid(logits)  # multi‑label probabilities
                probs.append(prob.cpu().numpy())
        probs = np.concatenate(probs, axis=0)  # shape (num_test, 4)
        fold_test_probs.append(probs)

    return fold_test_probs




## === cell 6
train_df = pd.read_csv(os.path.join(DIR_INPUT, "train.csv"))
fold_predictions = train_and_predict(train_df, test_loader, epochs=5)
print("Finished training & inference for all folds.")



## === cell 7
test_probs_mean = np.mean(np.stack(fold_predictions, axis=0), axis=0)  # (num_test, 4)
test_probs_mean.shape


## === cell 8
submission_df = pd.read_csv(os.path.join(DIR_INPUT, "sample_submission.csv"))
submission_df = (
    submission_df.set_index("image_id").loc[test_df["image_id"]].reset_index()
)
submission_df[["healthy", "multiple_diseases", "rust", "scab"]] = test_probs_mean
submission_df.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")


## === cell 9
submission_df.head()
