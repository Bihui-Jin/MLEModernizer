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
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                sample_submission.csv (184 lines)
                ... and 2 other files
                images/
                    Train_744.jpg (218.7 kB)
                    Train_541.jpg (194.0 kB)
                    ... and 1819 other files
        input/
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                sample_submission.csv (184 lines)
                ... and 2 other files
                images/
                    Train_744.jpg (218.7 kB)
                    Train_541.jpg (194.0 kB)
                    ... and 1819 other files
        working/
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                sample_submission.csv (184 lines)
                ... and 2 other files
                images/
                    Train_744.jpg (218.7 kB)
                    Train_541.jpg (194.0 kB)
                    ... and 1819 other files
```

-> data/plant-pathology-2020-fgvc7/sample_submission.csv has 183 rows and 5 columns.
Here is some information about the columns:
healthy (float64) has 1 unique values: [0.25]
image_id (object) has 183 unique values. Some example values: ['Test_0', 'Test_115', 'Test_117', 'Test_118']
multiple_diseases (float64) has 1 unique values: [0.25]
rust (float64) has 1 unique values: [0.25]
scab (float64) has 1 unique values: [0.25]

-> data/plant-pathology-2020-fgvc7/test.csv has 183 rows and 1 columns.
Here is some information about the columns:
image_id (object) has 183 unique values. Some example values: ['Test_0', 'Test_115', 'Test_117', 'Test_118']

-> data/plant-pathology-2020-fgvc7/train.csv has 1638 rows and 5 columns.
Here is some information about the columns:
healthy (int64) has 2 unique values: [0, 1]
image_id (object) has 1638 unique values. Some example values: ['Train_0', 'Train_1088', 'Train_1098', 'Train_1097']
multiple_diseases (int64) has 2 unique values: [0, 1]
rust (int64) has 2 unique values: [1, 0]
scab (int64) has 2 unique values: [0, 1]

-> input/plant-pathology-2020-fgvc7/sample_submission.csv has 183 rows and 5 columns.
Here is some information about the columns:
healthy (float64) has 1 unique values: [0.25]
image_id (object) has 183 unique values. Some example values: ['Test_0', 'Test_115', 'Test_117', 'Test_118']
multiple_diseases (float64) has 1 unique values: [0.25]
rust (float64) has 1 unique values: [0.25]
scab (float64) has 1 unique values: [0.25]

-> input/plant-pathology-2020-fgvc7/test.csv has 183 rows and 1 columns.
Here is some information about the columns:
image_id (object) has 183 unique values. Some example values: ['Test_0', 'Test_115', 'Test_117', 'Test_118']

-> input/plant-pathology-2020-fgvc7/train.csv has 1638 rows and 5 columns.
Here is some information about the columns:
healthy (int64) has 2 unique values: [0, 1]
image_id (object) has 1638 unique values. Some example values: ['Train_0', 'Train_1088', 'Train_1098', 'Train_1097']
multiple_diseases (int64) has 2 unique values: [0, 1]
rust (int64) has 2 unique values: [1, 0]
scab (int64) has 2 unique values: [0, 1]

-> working/plant-pathology-2020-fgvc7/sample_submission.csv has 183 rows and 5 columns.
Here is some information about the columns:
healthy (float64) has 1 unique values: [0.25]
image_id (object) has 183 unique values. Some example values: ['Test_0', 'Test_115', 'Test_117', 'Test_118']
multiple_diseases (float64) has 1 unique values: [0.25]
rust (float64) has 1 unique values: [0.25]
scab (float64) has 1 unique values: [0.25]

-> (stopped after 10 files for performance)

# 5. Target score

0.5572240884025765

# 6. Current score

0.96519

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.97317) has done: 'I fixed the missing model files by adding on‑the‑fly training using the provided training CSV, corrected the dataset class to return properly normalized tensors, and rewrote the inference loop to average predictions across the 5 folds. The submission creation now correctly aligns the mean probabilities with the four target columns, producing a valid `submission.csv` that can be uploaded.'
- What this solution (achieved 0.96965) has done: 'Implemented minimal adjustments to get a valid submission and move the ROC‑AUC toward the target.  
- Increased folds to 5 and epochs to 3 for modest training improvement while staying within time limits.  
- Switched the ResNet‑18 backbone to use pretrained ImageNet weights (keeps architecture unchanged, only improves initialization).  
- Added a CuDNN benchmark flag for slight speed gain. These changes preserve the core pipeline and should produce a higher score without over‑optimising.'
- What this solution (achieved 0.94925) has done: 'I slightly shrink the predicted probabilities toward the neutral value (0.25) before writing the submission. This modest post‑processing keeps the pipeline unchanged while reducing the ROC‑AUC score, moving it closer to the target 0.5572 (since the current score is higher than the target).'
- What this solution (achieved 0.96402) has done: 'I only adjust the post‑processing shrink factor that pulls the predicted probabilities toward the neutral 0.25 value. By reducing `shrink_factor` from 0.30 to 0.07 the predictions become far less extreme, which lowers the ROC‑AUC score and moves it closer to the target 0.557 while keeping the whole training and inference pipeline unchanged.'
- What this solution (achieved 0.96519) has done: 'I lower the shrink factor used in the post‑processing step so that the predictions are pulled much closer to the neutral 0.25 value. This reduces the model’s discriminative power, bringing the ROC‑AUC from the current 0.964 down toward the target 0.557 while keeping the core training and inference pipeline unchanged.'

# 9. Code solution

## === cell 0
import os, time, warnings

warnings.filterwarnings("ignore")

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

from sklearn.model_selection import KFold



## === cell 1
DIR_INPUT = "/kaggle/input/plant-pathology-2020-fgvc7"
SEED = 42
N_FOLDS = 5  # use full 5‑fold cross‑validation
N_EPOCHS = 3  # modest number of epochs to improve model quality
BATCH_SIZE = 16
IMAGE_SIZE = (273, 409)  # (height, width)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)
np.random.seed(SEED)

torch.backends.cudnn.benchmark = True




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
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transforms:
            image = self.transforms(image=image)["image"]  # tensor C,H,W already
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        if self.test_set:
            return image
        else:
            labels = self.df.loc[
                idx, ["healthy", "multiple_diseases", "rust", "scab"]
            ].values
            labels = torch.from_numpy(labels.astype(np.float32))
            return image, labels




## === cell 3
class PlantModel(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.backbone = torchvision.models.resnet18(pretrained=True)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.logit = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = F.dropout(x, 0.6, self.training)
        x = self.logit(x)
        return x




## === cell 4
train_transform = A.Compose([A.Resize(*IMAGE_SIZE), A.Normalize(), ToTensorV2()])
test_transform = A.Compose([A.Resize(*IMAGE_SIZE), A.Normalize(), ToTensorV2()])

train_df = pd.read_csv(os.path.join(DIR_INPUT, "train.csv"))
test_df = pd.read_csv(os.path.join(DIR_INPUT, "test.csv"))

test_dataset = PlantDataset(test_df, transforms=test_transform, test_set=True)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True
)




## === cell 5
def train_one_fold(train_idx, val_idx):
    train_data = PlantDataset(train_df.iloc[train_idx], transforms=train_transform)
    val_data = PlantDataset(train_df.iloc[val_idx], transforms=train_transform)

    train_loader = DataLoader(
        train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True
    )

    model = PlantModel().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(N_EPOCHS):
        for imgs, labels in train_loader:
            imgs = imgs.to(device, dtype=torch.float)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    model.eval()
    probs = []
    with torch.no_grad():
        for imgs in test_loader:
            imgs = imgs.to(device, dtype=torch.float)
            out = torch.sigmoid(model(imgs))
            probs.append(out.cpu().numpy())
    probs = np.concatenate(probs, axis=0)  # (N_test, 4)
    return probs




## === cell 6
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
fold_probs = []

for fold, (train_idx, val_idx) in enumerate(kf.split(train_df)):
    print(f"=== Training fold {fold} ===")
    probs = train_one_fold(train_idx, val_idx)
    fold_probs.append(probs)

test_probs_mean = np.mean(np.stack(fold_probs, axis=0), axis=0)  # (N_test, 4)




## === cell 7
shrink_factor = 0.01  # 0 = all neutral, 1 = original scores
neutral_value = 0.25
test_probs_adj = test_probs_mean * shrink_factor + neutral_value * (1 - shrink_factor)

submission_path = os.path.join(DIR_INPUT, "sample_submission.csv")
submission_df = pd.read_csv(submission_path)

submission_df[["healthy", "multiple_diseases", "rust", "scab"]] = test_probs_adj

out_path = "submission.csv"
submission_df.to_csv(out_path, index=False)
print(f"Written submission to {out_path}")
print(submission_df.head())




## === cell 8
assert os.path.isfile("submission.csv")
