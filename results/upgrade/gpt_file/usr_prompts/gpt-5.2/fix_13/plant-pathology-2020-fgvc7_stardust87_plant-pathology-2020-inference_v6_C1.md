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

0.97826

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.97518) has done: 'I remove the dependency on missing external `.pth` files by adding the minimal missing step: training the same `PlantModel` (ResNet18 + linear head) with the same multi-label setup and then running inference on the test set. I fix dataset bugs that would break both training and inference (incorrect tensor shaping, missing normalization, and indexing with `.loc[idx]` on non-0-based indices). I also fix the inference softmax bug (this is multi-label, so it must be sigmoid) and correct the fold-averaging logic so predictions are an `(n_test, 4)` array that can be written into the 4 submission columns. Finally, the script always write a valid `submission.csv` with the exact required column names.'
- What this solution (achieved 0.96475) has done: 'Your current score (0.97518) is far above the target (0.64703), so to move toward the target with minimal risk, we should *reduce* performance slightly rather than improve it. The smallest, safest way that preserves your core training/inference logic is to calibrate predictions after averaging folds by blending them toward the class priors from the training set (a legitimate post-processing step that keeps probabilities valid and does not change the model, loss, or training loop). This reduce AUC in a controlled way and is easy to tune via a single parameter `BLEND_ALPHA`. I also keep everything else the same and still write a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.97574) has done: 'Your current score (0.96475) is far above the target (0.64703), so the smallest safe move toward the target is to *intentionally reduce* performance without changing the model/training loop. I do this by increasing the existing, legitimate post-processing calibration: blend the fold-averaged probabilities more strongly toward the training-set class priors, which should lower ROC AUC in a controlled way while keeping valid probabilities. I keep everything else (dataset, ResNet18 head, BCEWithLogitsLoss, folds, epochs, inference) identical and still write a valid `submission.csv`. I also add a tiny safety check to ensure the blend alpha stays in [0,1].'
- What this solution (achieved 0.98025) has done: 'Your current score (0.97574) is far above the target (0.64703), so we should move *downward* in a controlled, minimal way while keeping your exact training/inference pipeline intact. The smallest change that reliably reduces ROC AUC without touching model/training is to strengthen the existing legitimate post-processing: blend predictions more toward the training-set class priors (i.e., reduce variance/ranking signal). I only change `BLEND_ALPHA` (and keep the same blending formula), leaving the dataset, ResNet18, loss, folds, epochs, and sigmoid inference unchanged. This should reduce the score toward the target without risking invalid submissions.'
- What this solution (achieved 0.97949) has done: 'Your current score (0.98025) is far above the target (0.64703), so to move closer we should deliberately reduce ranking signal while keeping your exact model/training/inference pipeline intact. The smallest, most controlled change is to strengthen your existing legitimate post-processing: blend the fold-averaged predictions much more toward the training-set class priors (this preserves valid probabilities and does not touch architecture, loss, folds, epochs, or sigmoid). I only change `BLEND_ALPHA` to a much smaller value so priors dominate, which should reduce mean ROC AUC toward the target band. Everything else, including submission schema and file writing, stays the same.'
- What this solution (achieved 0.97829) has done: 'Your current score (0.97949) is far above the target (0.64703), so we should deliberately reduce performance in the most controlled way while keeping your exact training/inference pipeline intact. The smallest change is to adjust only the existing post-processing blend so predictions are pulled more toward the training-set priors (reducing ranking signal and thus ROC AUC). To make this “move down” more reliably than an extreme near-zero alpha, I set `BLEND_ALPHA` to a moderate value so the model still contributes but is substantially damped. Everything else (dataset, ResNet18 head, loss, folds, epochs, sigmoid inference, submission writing) stays unchanged and still produces a valid `submission.csv`.'
- What this solution (achieved 0.98581) has done: 'Your current score (0.97829) is far above the target (0.64703), so we should deliberately reduce performance in a controlled way while preserving the exact training/inference pipeline. The smallest change that predictably lowers mean ROC AUC (without touching the model, loss, folds, epochs, or training loop) is to make the existing post-processing blending lean much more toward the training-set priors. I only adjust `BLEND_ALPHA` downward so the model’s ranking signal is damped more strongly, keeping probabilities valid and the submission format unchanged. Everything else remains identical and the script still writes a valid `submission.csv`.'
- What this solution (achieved 0.97132) has done: 'Your current score (0.98581) is far above the target (0.64703), so we should move *downward* with the smallest, safest change that preserves your model/training/inference exactly. The most controlled lever you already have is the legitimate post-processing blend toward training-set class priors; increasing the weight on priors reduces ranking signal and thus ROC AUC. I only change `BLEND_ALPHA` to a smaller value (more prior-dominant) and keep the blending formula, folds, epochs, architecture, loss, and sigmoid inference unchanged. The script still run end-to-end and write a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.97444) has done: 'Your current score (0.97132) is far above the target (0.64703), so we should move *downward* in a controlled way while keeping your training/inference pipeline unchanged. The smallest safe lever you already have is the post-processing blend toward training-set priors; increasing prior dominance reduce ranking signal and thus ROC AUC. I only adjust `BLEND_ALPHA` (keeping the same blending formula, folds, epochs, architecture, loss, and sigmoid inference) to pull predictions closer to priors and reduce the score toward the target band. Everything else remains identical and still writes a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.5) has done: 'Your current score (0.97444) is far above the target (0.64703), so we should move it downward with the smallest possible change while preserving your exact training/inference pipeline. The most controlled lever you already use is the post-processing blend toward training-set priors; setting `BLEND_ALPHA` to 0 makes predictions exactly equal to priors for every image, which should drive ROC AUC close to 0.5 and move substantially toward the target (likely undershooting, but much closer than 0.97). I only change `BLEND_ALPHA` (and keep the same blending formula) and keep everything else identical. The script still run end-to-end and write a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.97494) has done: 'Your current score (0.5) is below the target (0.6470), so we need to *increase* performance toward the target band with the smallest safe change. Since the core model/training/inference is already strong, the single lever you intentionally used to reduce AUC was `BLEND_ALPHA=0.0`, which collapses predictions to constant priors and yields ~0.5 AUC. I only raise `BLEND_ALPHA` to a modest value so predictions regain some ranking signal while still being damped toward priors (so we don’t jump back near ~0.97). Everything else (dataset, ResNet18 head, BCEWithLogitsLoss, folds, epochs, sigmoid inference, and submission writing) remains unchanged.'
- What this solution (achieved 0.97826) has done: 'Your current score (0.97494) is far above the target (0.64703), and the only lever in your current pipeline that intentionally controls score without changing the core model/training is the post-processing blend toward class priors. To move the score downward toward the target with minimal risk, I only reduce `BLEND_ALPHA` so the submission probabilities are more prior-dominant (less ranking signal), while keeping the model, folds, epochs, loss, and sigmoid inference unchanged. I also keep the existing clipping/sanity checks so the output remains valid probabilities and the submission schema stays correct. No other training/inference logic is modified.'

# 9. Code solution

## === cell 0
import os, time, random

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

from sklearn.model_selection import StratifiedKFold

import warnings

warnings.filterwarnings("ignore")



## === cell 1
DIR_INPUT = "/kaggle/input/plant-pathology-2020-fgvc7"

SEED = 42
N_FOLDS = 5
BATCH_SIZE = 16
IMAGE_SIZE = (273, 409)  # (H, W)
EPOCHS = 3
LR = 1e-3

BLEND_ALPHA = 0.02

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device




## === cell 2
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(SEED)




## === cell 3
class PlantDataset(Dataset):
    """
    Bugfixes vs original:
    - Use iloc instead of loc[idx] to avoid index misalignment after splits.
    - Return tensors in (C,H,W) directly using Albumentations ToTensorV2; remove invalid view().
    - Proper float labels shape (4,) for BCEWithLogitsLoss.
    """

    def __init__(self, df, transforms=None, test_set=False):
        self.df = df.reset_index(drop=True)
        self.transforms = transforms
        self.test_set = test_set

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_id = self.df.iloc[idx]["image_id"]
        image_src = os.path.join(DIR_INPUT, "images", f"{image_id}.jpg")
        image = cv2.imread(image_src, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_src}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (IMAGE_SIZE[1], IMAGE_SIZE[0]))  # cv2 uses (W,H)

        if self.transforms is not None:
            image = self.transforms(image=image)["image"]
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        if not self.test_set:
            labels = self.df.iloc[idx][
                ["healthy", "multiple_diseases", "rust", "scab"]
            ].values.astype(np.float32)
            labels = torch.from_numpy(labels)
            return image, labels
        return image




## === cell 4
train_tfms = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.03,
            scale_limit=0.10,
            rotate_limit=15,
            p=0.5,
            border_mode=cv2.BORDER_REFLECT_101,
        ),
        A.RandomBrightnessContrast(p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)

valid_tfms = A.Compose(
    [
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)




## === cell 5
class PlantModel(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        self.backbone = torchvision.models.resnet18(pretrained=True)
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




## === cell 6
train_df = pd.read_csv(os.path.join(DIR_INPUT, "train.csv"))
test_df = pd.read_csv(os.path.join(DIR_INPUT, "test.csv"))
target_cols = ["healthy", "multiple_diseases", "rust", "scab"]

stratify_label = train_df[target_cols].values.argmax(axis=1)

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

train_priors = train_df[target_cols].mean(axis=0).values.astype(np.float32)  # (4,)



## === cell 7
dataset_test = PlantDataset(df=test_df, transforms=valid_tfms, test_set=True)
testloader = DataLoader(
    dataset_test,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)




## === cell 8
def train_one_fold(fold, trn_idx, val_idx):
    trn_df = train_df.iloc[trn_idx].reset_index(drop=True)
    val_df = train_df.iloc[val_idx].reset_index(drop=True)

    dtrain = PlantDataset(trn_df, transforms=train_tfms, test_set=False)
    dvalid = PlantDataset(val_df, transforms=valid_tfms, test_set=False)

    trainloader = DataLoader(
        dtrain,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    validloader = DataLoader(
        dvalid,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    model = PlantModel(num_classes=4).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_state = None
    best_loss = float("inf")

    for epoch in range(EPOCHS):
        model.train()
        tr_losses = []

        for images, labels in trainloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            tr_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for images, labels in validloader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                logits = model(images)
                loss = criterion(logits, labels)
                val_losses.append(loss.item())

        mean_val_loss = float(np.mean(val_losses)) if len(val_losses) else float("inf")
        if mean_val_loss < best_loss:
            best_loss = mean_val_loss
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }

    model.load_state_dict(best_state)
    return model




## === cell 9
def predict_test(model, testloader):
    model.eval()
    probs_all = []
    with torch.no_grad():
        for images in testloader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            probs = torch.sigmoid(logits)
            probs_all.append(probs.detach().cpu().numpy())
    return np.concatenate(probs_all, axis=0)




## === cell 10
test_probs_folds = []
start = time.perf_counter()

for fold, (trn_idx, val_idx) in enumerate(skf.split(train_df, stratify_label)):
    model = train_one_fold(fold, trn_idx, val_idx)
    fold_probs = predict_test(model, testloader)
    test_probs_folds.append(fold_probs)

test_probs_folds = np.stack(test_probs_folds, axis=0)  # (n_folds, n_test, 4)
test_probs_mean = test_probs_folds.mean(axis=0)  # (n_test, 4)

BLEND_ALPHA = float(np.clip(BLEND_ALPHA, 0.0, 1.0))
test_probs_mean = (
    BLEND_ALPHA * test_probs_mean + (1.0 - BLEND_ALPHA) * train_priors[None, :]
).astype(np.float32)
test_probs_mean = np.clip(test_probs_mean, 0.0, 1.0)

print(f"Finished training+inference in {(time.perf_counter() - start):.2f} seconds")
print("test_probs_mean shape:", test_probs_mean.shape)
print("BLEND_ALPHA:", BLEND_ALPHA, "train_priors:", train_priors)



## === cell 11
submission_path = os.path.join(DIR_INPUT, "sample_submission.csv")
submission_df = pd.read_csv(submission_path)

submission_df = submission_df.merge(test_df[["image_id"]], on="image_id", how="right")

submission_df[target_cols] = test_probs_mean.astype(np.float32)
submission_df.to_csv("submission.csv", index=False)

submission_df.head()



## === cell 12
assert os.path.exists("submission.csv")
check = pd.read_csv("submission.csv")
assert list(check.columns) == ["image_id"] + target_cols
assert len(check) == len(test_df)
check.describe()
