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

0.9478579761895304

# 6. Current score

0.5

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.5) has done: 'I replace the missing‑model loading with a simple baseline that uses the average label frequencies from the training set as constant predictions for every test image. This removes the FileNotFoundError, ensures the prediction array has the correct shape, and allows the script to write a valid `submission.csv` file. No other logic is altered.'

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

from torch.utils.data import Dataset, DataLoader
from albumentations.pytorch import ToTensorV2



## === cell 1
DIR_INPUT = "/kaggle/input/plant-pathology-2020-fgvc7"
SEED = 42
N_FOLDS = 5
N_EPOCHS = 3
BATCH_SIZE = 16
IMAGE_SIZE = (273, 409)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




## === cell 2
class PlantDataset(Dataset):
    def __init__(self, df, transforms=None, test_set=False):
        self.df = df
        self.transforms = transforms
        self.test_set = test_set
        if not self.transforms:
            self.transforms = A.Compose([ToTensorV2(p=1.0)])

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        image_src = os.path.join(
            DIR_INPUT, "images", self.df.loc[idx, "image_id"] + ".jpg"
        )
        image = cv2.imread(image_src, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, IMAGE_SIZE)
        transformed = self.transforms(image=image)
        image = transformed["image"]
        if not self.test_set:
            labels = self.df.loc[
                idx, ["healthy", "multiple_diseases", "rust", "scab"]
            ].values
            labels = torch.from_numpy(labels.astype(np.int8))
            labels = labels.unsqueeze(-1)
            return image, labels
        else:
            return image




## === cell 3
class PlantModel(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.backbone = torchvision.models.resnet50(pretrained=True)
        in_features = self.backbone.fc.in_features
        self.logit = nn.Linear(in_features, num_classes)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch_size, -1)
        x = F.dropout(x, 0.25, self.training)
        x = self.logit(x)
        return x




## === cell 4
transforms_valid = A.Compose([A.Normalize(p=1.0), ToTensorV2(p=1.0)])

test_df = pd.read_csv(os.path.join(DIR_INPUT, "test.csv"))
dataset_test = PlantDataset(df=test_df, test_set=True, transforms=transforms_valid)
testloader = DataLoader(
    dataset_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
)



## === cell 5
train_df = pd.read_csv(os.path.join(DIR_INPUT, "train.csv"))
class_means = (
    train_df[["healthy", "multiple_diseases", "rust", "scab"]].mean().values
)  # shape (4,)



## === cell 6
num_test = len(test_df)
test_probs = np.tile(class_means, (num_test, 1))  # shape (num_test, 4)



## === cell 7
test_probs_mean = test_probs



## === cell 8
submission_df = pd.read_csv(os.path.join(DIR_INPUT, "sample_submission.csv"))
submission_df[["healthy", "multiple_diseases", "rust", "scab"]] = test_probs_mean
submission_df.to_csv("submission.csv", index=False)



## === cell 9
submission_df.head()
