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
timm==1.0.19
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

0.9648988014890484

# 6. Current score

0.5

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.5) has done: 'I fix the pipeline so it runs end-to-end without relying on missing external pretrained fold checkpoints (the `/kaggle/input/plant-pathology-2020-training/` path does not exist here). I also fix inference bugs (missing `dim` in `softmax`, incorrect averaging logic, and fragile dataset indexing) and point image loading to the actual competition image folder available in this environment. Finally, I generate a valid `submission.csv` with the required columns and row alignment to `test.csv`, using a deterministic fallback (uniform probabilities) if no trained weights are present, ensuring you always get a valid file.'

# 9. Code solution

## === cell 0
import os, time, glob

import numpy as np
import pandas as pd

import albumentations as A
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F

import timm

from tqdm.notebook import tqdm
from torch.utils.data import Dataset, DataLoader
from albumentations.pytorch import ToTensorV2

import warnings

warnings.filterwarnings("ignore")



## === cell 1
DIR_INPUT = "/kaggle/input/plant-pathology-2020-fgvc7"

IMAGE_DIR = os.path.join(DIR_INPUT, "images")

SEED = 42
N_FOLDS = 5
N_EPOCHS = 30
BATCH_SIZE = 8

device = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device)


def seed_everything(seed: int = 42):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(SEED)
device




## === cell 2
class PlantDataset(Dataset):
    def __init__(self, df, image_dir, transforms=None, test_set=False):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transforms = transforms
        self.test_set = test_set
        if not self.transforms:
            self.transforms = A.Compose([ToTensorV2(p=1.0)])

        self.targets = ["healthy", "multiple_diseases", "rust", "scab"]

    def __len__(self):
        return self.df.shape[0]

    def _read_image(self, image_id: str):
        path_jpg = os.path.join(self.image_dir, f"{image_id}.jpg")
        img = cv2.imread(path_jpg, cv2.IMREAD_COLOR)
        if img is None:
            matches = glob.glob(os.path.join(self.image_dir, f"{image_id}.*"))
            if matches:
                img = cv2.imread(matches[0], cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(
                f"Could not read image for image_id={image_id} in {self.image_dir}"
            )
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = row["image_id"]
        image = self._read_image(image_id)

        transformed = self.transforms(image=image)
        image = transformed["image"]

        if not self.test_set:
            labels = row[self.targets].values.astype(np.float32)
            labels = torch.from_numpy(labels)
            return image, labels
        else:
            return image




## === cell 3
def trim_network_at_index(network, index=-1):
    assert index < 0, f"Param index must be negative. Received {index}"
    return nn.Sequential(*list(network.children())[:index])


class PlantModel(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.backbone = timm.create_model("resnest269e", pretrained=True)
        in_features = self.backbone.fc.in_features
        self.backbone = trim_network_at_index(self.backbone, -1)
        self.logit = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        x = self.logit(x)
        return x




## === cell 4
transforms_valid = A.Compose([A.Normalize(p=1.0), ToTensorV2(p=1.0)])

test_df = pd.read_csv(os.path.join(DIR_INPUT, "test.csv"))
dataset_test = PlantDataset(
    df=test_df, image_dir=IMAGE_DIR, test_set=True, transforms=transforms_valid
)
testloader = DataLoader(
    dataset_test,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)

test_df.head()



## === cell 5
WEIGHTS_DIR = "/kaggle/input/plant-pathology-2020-training"  # may not exist
available_weight_files = []
if os.path.isdir(WEIGHTS_DIR):
    available_weight_files = sorted(glob.glob(os.path.join(WEIGHTS_DIR, "*.pth")))
available_weight_files[:5], len(available_weight_files)




## === cell 6
def predict_probs(model, loader):
    model.eval()
    all_probs = []
    with torch.no_grad():
        for batch in tqdm(loader, total=len(loader)):
            images = batch.to(device, non_blocking=True)
            logits = model(images)
            probs = F.softmax(logits, dim=1)
            all_probs.append(probs.detach().cpu().numpy())
    return np.concatenate(all_probs, axis=0)


def test_model(model_name, testloader):
    model_path = os.path.join(WEIGHTS_DIR, f"{model_name}.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(model_path)

    model = PlantModel()
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model = model.to(device)
    return predict_probs(model, testloader)




## === cell 7
start = time.perf_counter()

test_probs_folds = []

use_folds = True
missing = []
for i_fold in range(N_FOLDS):
    path = os.path.join(WEIGHTS_DIR, f"modelF{i_fold}.pth")
    if not os.path.exists(path):
        missing.append(path)

if missing:
    use_folds = False

if use_folds:
    for i_fold in range(N_FOLDS):
        probs_fold = test_model(f"modelF{i_fold}", testloader)
        test_probs_folds.append(probs_fold)
    test_probs_mean = np.mean(np.stack(test_probs_folds, axis=0), axis=0)
else:
    n = len(test_df)
    test_probs_mean = np.full((n, 4), 0.25, dtype=np.float32)

print(f"Finished Inference in {(time.perf_counter() - start):.2f} seconds")
test_probs_mean.shape, test_probs_mean[:2]



## === cell 8
submission_df = pd.read_csv(os.path.join(DIR_INPUT, "sample_submission.csv"))

submission_df = submission_df.merge(test_df[["image_id"]], on="image_id", how="right")
submission_df = submission_df[
    ["image_id", "healthy", "multiple_diseases", "rust", "scab"]
]

pred_cols = ["healthy", "multiple_diseases", "rust", "scab"]
submission_df.loc[:, pred_cols] = test_probs_mean.astype(np.float32)

submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)

submission_df.head(), submission_path



## === cell 9
assert os.path.exists("submission.csv")
assert submission_df.shape[0] == test_df.shape[0]
assert list(submission_df.columns) == [
    "image_id",
    "healthy",
    "multiple_diseases",
    "rust",
    "scab",
]
submission_df.describe()
