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

0.98531

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.98912) has done: 'I fix the immediate runtime blockers by removing the undefined `MODEL_PATH` load cell and replacing the missing external fold checkpoint dependency with an in-notebook training+inference flow that uses the same ResNet18 head and softmax probabilities. I also correct the test-time softmax call (must specify `dim=1`) and fix the averaging logic so `test_probs_mean` is a proper `(n_test, 4)` array that can be assigned into the submission columns. To keep changes minimal and stable, I add deterministic seeding, basic ImageNet normalization, and a simple stratified split for validation (not used for submission) while training on all train data for one epoch as originally configured. Finally, the script always write `submission.csv` with the required columns and row alignment.'
- What this solution (achieved 0.98984) has done: 'Your current score (0.98912) is far above the target (0.6200), so we should intentionally reduce performance toward the target band with the smallest, safest change while keeping the same model/training/prediction pipeline intact. The most controlled way is to apply a uniform probability-smoothing step at submission time (mixing your predicted distribution with a uniform prior), which preserves valid probabilities and doesn’t change the core logic. I add a single parameter `SMOOTH_ALPHA` and replace the current post-processing with a smoothed version; you can tune `SMOOTH_ALPHA` to land near 0.62. Everything else (architecture, loss, training loops, transforms, file paths, submission format) remains unchanged, and the script still writes `submission.csv`.'
- What this solution (achieved 0.98948) has done: 'Your current score (0.98984) is much higher than the target (0.6200), so the best way to move toward the target with minimal risk is to further dampen model skill via stronger, controlled probability smoothing at submission time (this keeps the same model/training/inference core logic intact). I only adjust the smoothing strength (increase `SMOOTH_ALPHA`) and make the smoothing numerically stable while preserving valid per-row probability distributions. This should reduce ROC AUC toward the target band without changing architecture, loss, transforms, or training loops. Everything else stays the same and the script still writes a valid `submission.csv`.'
- What this solution (achieved 0.9888) has done: 'Your current score (0.98948) is far above the target (0.62002), so the safest way to move toward the target (without changing the model/training core logic) is to further dampen predictive signal at submission time. I do this by increasing the uniform-mixing smoothing strength and making it explicit via a single constant, leaving the architecture, transforms, loss, and training loops unchanged. This keeps valid per-row probabilities (sum to 1, nonnegative) and should reduce ROC AUC toward the target band in a controlled, reversible way. The script still run end-to-end and write a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.98876) has done: 'Your current ROC AUC (0.9888) is far above the target (0.6200), so we should intentionally reduce performance in a controlled, minimal way rather than improve it. The smallest change that preserves your entire training/inference pipeline is to increase the uniform-mixing post-processing so predictions are closer to random while still being valid probabilities. I only adjust `SMOOTH_ALPHA` (stronger smoothing) and keep all architecture, loss, transforms, loaders, and loops identical. This should move the leaderboard score down toward the target band while still producing a correct `submission.csv`.'
- What this solution (achieved 0.98819) has done: 'Your current score (0.98876) is far above the target (0.62002), so we should intentionally reduce predictive signal in the smallest, safest way. We keep the entire training/inference pipeline unchanged and only adjust the final post-processing that mixes predictions with a uniform distribution. To move the score closer to the target, I increase `SMOOTH_ALPHA` so the submission becomes closer to random (AUC ~0.5) while staying valid probabilities and preserving submission schema. No changes to model architecture, loss, transforms, data loading, or training loops.'
- What this solution (achieved 0.5) has done: 'Your current score (0.98819) is far above the target (0.62002), so the correct direction is to intentionally reduce performance toward the target band with the smallest possible change. To do that while preserving the entire training/inference pipeline, I only adjust the final probability post-processing: increase the uniform-mixing smoothing so predictions become much closer to random (AUC approaches ~0.5). This keeps the submission valid (nonnegative probabilities summing to 1 per row) and is fully reversible by tuning one constant. I’m not changing the model, loss, transforms, data loading, or training loops—only the smoothing strength.'
- What this solution (achieved 0.99086) has done: 'Your current score (0.5) is below the target (0.6200), so we should *increase* predictive signal while keeping the same training/inference pipeline intact. The simplest, safest way is to reduce the submission-time uniform mixing (probability smoothing) that is currently set so high it nearly randomizes predictions. I only change `SMOOTH_ALPHA` to a moderate value so outputs remain valid probabilities but retain meaningful ranking information, which should move ROC AUC upward toward the target band. No changes to the model, loss, transforms, loaders, or training loops; the script still writes a valid `submission.csv`.'
- What this solution (achieved 0.98734) has done: 'Your current score (0.99086) is far above the target (0.62002), so the correct direction is to intentionally reduce predictive signal with the smallest possible change. We keep your entire training/inference pipeline intact and only adjust the submission-time uniform-mixing post-processing via `SMOOTH_ALPHA`, which is already the controlled “knob” for damping AUC toward random. To move much closer to ~0.62 (within the ±10% band), we increase `SMOOTH_ALPHA` substantially so probabilities are closer to uniform while remaining valid (nonnegative, row-sum=1). No changes are made to model architecture, loss, transforms, training loops, data paths, or submission schema—only the smoothing strength.'
- What this solution (achieved 0.9857) has done: 'Your current score (0.98734) is far above the target (0.62002), so we should intentionally reduce predictive signal in the smallest, most controlled way without touching the model/training/inference core logic. The safest “single knob” is the submission-time uniform mixing; we increase `SMOOTH_ALPHA` so predictions move closer to uniform (random AUC ≈ 0.5) and thereby bring ROC AUC down toward ~0.62. To make the adjustment predictable and avoid overshooting to exactly-uniform, we use a moderate-high value (0.78) rather than the extreme values that previously collapsed the score. No other changes are made; the script still runs end-to-end and writes a valid `submission.csv`.'
- What this solution (achieved 0.98531) has done: 'Your current score (0.9857) is far above the target (0.6200), so we should intentionally reduce predictive signal in the most controlled, minimal way while keeping the exact same training/inference pipeline. The smallest reliable “knob” is the existing submission-time uniform mixing; we increase `SMOOTH_ALPHA` moderately so probabilities move closer to uniform (AUC drifts toward ~0.5) without collapsing completely. Everything else (model, loss, transforms, loaders, epochs, and file paths) stays unchanged, and the script still write a valid `submission.csv`. This single-constant change is reversible and lets you tune if you undershoot/overshoot the target band.'

# 9. Code solution

## === cell 0
import os
import random

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

from sklearn.model_selection import train_test_split

import warnings

warnings.filterwarnings("ignore")




## === cell 1
DIR_INPUT = "/kaggle/input/plant-pathology-2020-fgvc7"

SEED = 42
N_FOLDS = 3  # kept (no longer used for loading external checkpoints)
N_EPOCHS = 1
BATCH_SIZE = 16
IMAGE_SIZE = (273, 409)

SMOOTH_ALPHA = 0.92  # in [0,1]; higher => more uniform => lower ROC AUC


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device)
device




## === cell 2
train_transforms = A.Compose(
    [
        A.Resize(IMAGE_SIZE[0], IMAGE_SIZE[1]),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)

test_transforms = A.Compose(
    [
        A.Resize(IMAGE_SIZE[0], IMAGE_SIZE[1]),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)




## === cell 3
class PlantDataset(Dataset):
    def __init__(self, df, transforms=None, test_set=False):
        self.df = df.reset_index(drop=True)
        self.transforms = transforms
        self.test_set = test_set

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        image_src = DIR_INPUT + "/images/" + self.df.loc[idx, "image_id"] + ".jpg"
        image = cv2.imread(image_src, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_src}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transforms:
            transformed = self.transforms(image=image)
            image = transformed["image"]  # torch tensor CHW float32
        else:
            image = cv2.resize(image, IMAGE_SIZE)
            image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0

        if not self.test_set:
            labels = self.df.loc[
                idx, ["healthy", "multiple_diseases", "rust", "scab"]
            ].values.astype(np.float32)
            labels = torch.from_numpy(labels)  # shape (4,)
            return image, labels
        else:
            return image




## === cell 4
test_df = pd.read_csv(DIR_INPUT + "/test.csv")
dataset_test = PlantDataset(df=test_df, transforms=test_transforms, test_set=True)
dataset_test[0].shape




## === cell 5
train_df = pd.read_csv(DIR_INPUT + "/train.csv")

y_strat = train_df[["healthy", "multiple_diseases", "rust", "scab"]].values.argmax(
    axis=1
)
train_idx, val_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=SEED, stratify=y_strat
)
train_df_tr = train_df.iloc[train_idx].reset_index(drop=True)
train_df_va = train_df.iloc[val_idx].reset_index(drop=True)

dataset_train = PlantDataset(
    df=train_df_tr, transforms=train_transforms, test_set=False
)
dataset_val = PlantDataset(df=train_df_va, transforms=test_transforms, test_set=False)

trainloader = DataLoader(
    dataset_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True
)
valloader = DataLoader(
    dataset_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
)




## === cell 6
class PlantModel(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        weights = torchvision.models.ResNet18_Weights.DEFAULT
        self.backbone = torchvision.models.resnet18(weights=weights)

        in_features = self.backbone.fc.in_features
        self.logit = nn.Linear(in_features, num_classes)

    def forward(self, x):
        batch_size, C, H, W = x.shape

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




## === cell 7
testloader = DataLoader(
    dataset_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
)




## === cell 8
model = PlantModel().to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)


def run_one_epoch_train(model, loader):
    model.train()
    running = 0.0
    for images, targets in tqdm(loader, total=len(loader)):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        running += loss.item()
    return running / max(1, len(loader))


@torch.no_grad()
def run_one_epoch_val(model, loader):
    model.eval()
    running = 0.0
    for images, targets in tqdm(loader, total=len(loader)):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, targets)
        running += loss.item()
    return running / max(1, len(loader))


for epoch in range(N_EPOCHS):
    tr_loss = run_one_epoch_train(model, trainloader)
    va_loss = run_one_epoch_val(model, valloader)
    print(
        f"Epoch {epoch+1}/{N_EPOCHS} - train_loss: {tr_loss:.4f} - val_loss: {va_loss:.4f}"
    )




## === cell 9
MODEL_PATH = None
print("MODEL_PATH not used; model trained in-notebook.")




## === cell 10
@torch.no_grad()
def test_model(model, testloader):
    model.eval()
    test_probs = []
    for images in tqdm(testloader, total=len(testloader)):
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probs = F.softmax(logits, dim=1)
        test_probs.append(probs.cpu().numpy())
    test_probs = np.concatenate(test_probs, axis=0)
    return test_probs




## === cell 11
dataset_full = PlantDataset(df=train_df, transforms=train_transforms, test_set=False)
fullloader = DataLoader(
    dataset_full, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True
)

model_full = PlantModel().to(device)
optimizer_full = optim.Adam(model_full.parameters(), lr=1e-4)

for epoch in range(N_EPOCHS):
    model_full.train()
    running = 0.0
    for images, targets in tqdm(fullloader, total=len(fullloader)):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer_full.zero_grad(set_to_none=True)
        logits = model_full(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer_full.step()
        running += loss.item()
    print(
        f"Full-fit Epoch {epoch+1}/{N_EPOCHS} - loss: {running/max(1,len(fullloader)):.4f}"
    )




## === cell 12
test_probs = test_model(model_full, testloader)
print("Finished Testing", test_probs.shape)




## === cell 13
test_probs_mean = test_probs




## === cell 14
submission_df = pd.read_csv(DIR_INPUT + "/sample_submission.csv")
submission_df[["healthy", "multiple_diseases", "rust", "scab"]] = (
    test_probs_mean.astype(np.float32)
)

pred = submission_df[["healthy", "multiple_diseases", "rust", "scab"]].values.astype(
    np.float64
)
pred = np.clip(pred, 1e-12, 1.0)
pred = pred / pred.sum(axis=1, keepdims=True)

uniform = np.full_like(pred, 1.0 / pred.shape[1], dtype=np.float64)

pred = (1.0 - SMOOTH_ALPHA) * pred + SMOOTH_ALPHA * uniform

pred = np.clip(pred, 1e-12, 1.0)
pred = pred / pred.sum(axis=1, keepdims=True)

submission_df[["healthy", "multiple_diseases", "rust", "scab"]] = pred.astype(
    np.float32
)

submission_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv", submission_df.shape, "SMOOTH_ALPHA=", SMOOTH_ALPHA)




## === cell 15
submission_df.head()




## === cell 16
submission_df
