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

0.62002333948532

# 6. Current score

0.97829

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.97829) has done: 'I replace the broken model‑loading code with a lightweight training loop that trains a fresh ResNet‑18 for one epoch on each K‑fold split, gathers test probabilities, averages them across folds, and writes them to a correctly‑shaped submission CSV. The dataset class is fixed to return float tensors normalized to [0,1] and proper label shapes, and all references to undefined paths are removed.'

# 9. Code solution

## === cell 0
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import KFold
from tqdm.notebook import tqdm



## === cell 1
DIR_INPUT = "/kaggle/input/plant-pathology-2020-fgvc7"
SEED = 42
N_FOLDS = 3
N_EPOCHS = 1
BATCH_SIZE = 16
IMAGE_SIZE = (273, 409)  # (height, width)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)
np.random.seed(SEED)




## === cell 2
class PlantDataset(Dataset):
    def __init__(self, df, test_set=False):
        self.df = df.reset_index(drop=True)
        self.test_set = test_set

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_id = self.df.loc[idx, "image_id"]
        img_path = os.path.join(DIR_INPUT, "images", f"{image_id}.jpg")
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, IMAGE_SIZE)
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0  # C,H,W normalized

        if not self.test_set:
            lbl = self.df.loc[
                idx, ["healthy", "multiple_diseases", "rust", "scab"]
            ].values
            lbl = torch.from_numpy(lbl.astype(np.float32))  # shape (4,)
            return img, lbl
        else:
            return img




## === cell 3
test_df = pd.read_csv(os.path.join(DIR_INPUT, "test.csv"))
test_dataset = PlantDataset(test_df, test_set=True)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
)




## === cell 4
class PlantModel(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.backbone = torchvision.models.resnet18(pretrained=True)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # remove original head
        self.logit = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = F.dropout(x, 0.25, self.training)
        return self.logit(x)




## === cell 5
def train_one_fold(train_idx, val_idx, df, fold):
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    train_set = PlantDataset(train_df, test_set=False)
    val_set = PlantDataset(val_df, test_set=False)

    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    model = PlantModel().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    model.train()
    for epoch in range(N_EPOCHS):
        epoch_loss = 0.0
        for imgs, labels in tqdm(
            train_loader, desc=f"Fold {fold} Epoch {epoch+1}", leave=False
        ):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * imgs.size(0)

    model.eval()
    test_probs = []
    with torch.no_grad():
        for imgs in test_loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = torch.sigmoid(logits).cpu().numpy()
            test_probs.append(probs)
    test_probs = np.concatenate(test_probs, axis=0)  # (num_test, 4)
    return test_probs




## === cell 6
train_df = pd.read_csv(os.path.join(DIR_INPUT, "train.csv"))

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
fold_predictions = []

for fold, (train_idx, val_idx) in enumerate(kf.split(train_df)):
    probs = train_one_fold(train_idx, val_idx, train_df, fold)
    fold_predictions.append(probs)



## === cell 7
test_probs_mean = np.mean(np.stack(fold_predictions, axis=0), axis=0)  # (num_test,4)



## === cell 8
submission_path = "submission.csv"
submission_df = pd.read_csv(os.path.join(DIR_INPUT, "sample_submission.csv"))
cols = ["healthy", "multiple_diseases", "rust", "scab"]
submission_df[cols] = test_probs_mean
submission_df.to_csv(submission_path, index=False)



## === cell 9
submission_df.head()
