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
Create a classifier to predict the severity of diabetic retinopathy.

## Metric
Quadratic weighted kappa, which measures the agreement between two ratings. This metric typically varies from 0 (random agreement between raters) to 1 (complete agreement between raters). In the event that there is less agreement between the raters than expected by chance, this metric may go below 0. The quadratic weighted kappa is calculated between the scores assigned by the human rater and the predicted scores.

Images have five possible ratings, 0,1,2,3,4.  Each image is characterized by a tuple *(e*,*e)*, which corresponds to its scores by *Rater A* (human) and *Rater B* (predicted).  The quadratic weighted kappa is calculated as follows. First, an N x N histogram matrix *O* is constructed, such that *O* corresponds to the number of images that received a rating *i* by *A* and a rating *j* by *B*. An *N-by-N* matrix of weights, *w*, is calculated based on the difference between raters' scores:

An *N-by-N* histogram matrix of expected ratings, *E*, is calculated, assuming that there is no correlation between rating scores.  This is calculated as the outer product between each rater's histogram vector of ratings, normalized such that *E* and *O* have the same sum.

## Submission Format
```
id_code,diagnosis
0005cfc8afb6,0
003f0afdcd15,0
etc.
```

## Dataset
You are provided with a large set of retina images taken using [fundus photography](https://en.wikipedia.org/wiki/Fundus_photography) under a variety of imaging conditions.

Labels are on a scale of 0 to 4:

> 0 - No DR
> 1 - Mild
> 2 - Moderate
> 3 - Severe
> 4 - Proliferative DR

Images may contain artifacts, be out of focus, underexposed, or overexposed. The images were gathered from multiple clinics using a variety of cameras over an extended period of time, which will introduce further variation.

- **train.csv** - the training labels
- **test.csv** - the test set (you must predict the `diagnosis` value for these variables)
- **sample_submission.csv** - a sample submission file in the correct format
- **train.zip** - the training set images
- **test.zip** - the public test set images

# 2. Python version

3.10

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
            description.md (118 lines)
            sample_submission.csv (368 lines)
            sample_submission.csv.zip (3.2 kB)
            test.csv (368 lines)
            test.csv.zip (2.9 kB)
            test.zip (160 Bytes)
            test_images.zip (902.9 MB)
            train.csv (3296 lines)
            train.csv.zip (27.5 kB)
            train.zip (162 Bytes)
            train_images.zip (7.7 GB)
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
            test_images/
                218c822a3dd9.png (5.7 MB)
                0e82bcacc475.png (5.2 MB)
                ... and 365 other files
                test_images/
            train_images/
                184a185e7447.png (337.5 kB)
                c4aef0d88d1b.png (876.6 kB)
                ... and 3293 other files
                train_images/
        input/
            description.md (118 lines)
            sample_submission.csv (368 lines)
            sample_submission.csv.zip (3.2 kB)
            test.csv (368 lines)
            test.csv.zip (2.9 kB)
            test.zip (160 Bytes)
            test_images.zip (902.9 MB)
            train.csv (3296 lines)
            train.csv.zip (27.5 kB)
            train.zip (162 Bytes)
            train_images.zip (7.7 GB)
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
            test_images/
                218c822a3dd9.png (5.7 MB)
                0e82bcacc475.png (5.2 MB)
                ... and 365 other files
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
            train_images/
                184a185e7447.png (337.5 kB)
                c4aef0d88d1b.png (876.6 kB)
                ... and 3293 other files
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
        working/
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
```

-> data/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/aptos2019-blindness-detection/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/aptos2019-blindness-detection/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> input/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> (stopped after 10 files for performance)

# 5. Target score

0.4041534303444881

# 6. Current score

0.65392

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.65392) has done: 'I fix the immediate runtime blocker by removing the missing external weight file dependency and instead train the same EfficientNet+ordinal-logistic model on the provided `train.csv`/`train_images` within the notebook so inference can run end-to-end. I also fix the dataset to return correctly shaped float tensors in CHW order and (for train) return labels, because the current code reshapes incorrectly and can silently break model performance. Finally, I ensure the submission is created from `test.csv` in the correct row order with columns `id_code,diagnosis` and written to `submission.csv`. These changes preserve the core architecture (EfficientNet-B0 + ordinal link) and keep the approach as standard supervised training + inference, while ensuring a valid CSV is produced.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import cv2 as cv
import random
import warnings
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
from tqdm import tqdm
import albumentations as A
from torchvision import models
from copy import deepcopy



## === cell 1
SEED = 123
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

warnings.filterwarnings("ignore")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n Device : {device.upper()}")



## === cell 2
BASE_PATH = "../input/aptos2019-blindness-detection"
TRAIN_PATH = f"{BASE_PATH}/train.csv"
TRAIN_IMG = f"{BASE_PATH}/train_images"
TEST_PATH = f"{BASE_PATH}/test.csv"
TEST_IMG = f"{BASE_PATH}/test_images"
SAMPLE_SUB_PATH = f"{BASE_PATH}/sample_submission.csv"




## === cell 3
class AptosDataset(Dataset):
    def __init__(self, data_path, img_dir, name, transforms, resize=(512, 512)):
        self.data_path = data_path
        self.img_dir = img_dir
        self.resize = resize
        self.transforms = transforms
        self.df = pd.read_csv(self.data_path).reset_index(drop=True)
        self.name = name

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        img_id = self.df.loc[idx, "id_code"]
        img_name = img_id + ".png"
        img_path = os.path.join(self.img_dir, img_name)
        img = cv.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found/readable: {img_path}")
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        if self.resize:
            img = cv.resize(img, self.resize, interpolation=cv.INTER_AREA)

        if self.transforms:
            transformed = self.transforms(image=img)
            img = transformed["image"]

        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img_tensor = torch.from_numpy(img)

        if self.name != "test" and "diagnosis" in self.df.columns:
            y = int(self.df.loc[idx, "diagnosis"])
            return img_tensor, torch.tensor(y, dtype=torch.long)
        return img_tensor




## === cell 4
class LogisticCumulativeLink(nn.Module):
    def __init__(self, num_classes: int, init_cutpoints: str = "ordered") -> None:
        assert num_classes > 2, "Only use this model if you have 3 or more classes"
        super().__init__()
        self.num_classes = num_classes
        self.init_cutpoints = init_cutpoints
        if init_cutpoints == "ordered":
            num_cutpoints = self.num_classes - 1
            cutpoints = torch.arange(num_cutpoints).float() - num_cutpoints / 2
            self.cutpoints = nn.Parameter(cutpoints)
        elif init_cutpoints == "random":
            cutpoints = torch.rand(self.num_classes - 1).sort()[0]
            self.cutpoints = nn.Parameter(cutpoints)
        else:
            raise ValueError(f"{init_cutpoints} is not a valid init_cutpoints " f"type")

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        sigmoids = torch.sigmoid(self.cutpoints - X)
        link_mat = sigmoids[:, 1:] - sigmoids[:, :-1]
        link_mat = torch.cat(
            (sigmoids[:, [0]], link_mat, (1 - sigmoids[:, [-1]])), dim=1
        )
        return link_mat


class OrdinalLogisticModel(nn.Module):
    def __init__(
        self, predictor: nn.Module, num_classes: int, init_cutpoints: str = "ordered"
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.predictor = deepcopy(predictor)
        self.link = LogisticCumulativeLink(
            self.num_classes, init_cutpoints=init_cutpoints
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.link(self.predictor(X))




## === cell 5
BATCH_SIZE = 16
IMG_DIM = 512

transforms = A.Compose(
    [
        A.RandomBrightnessContrast(),
        A.HueSaturationValue(),
    ]
)

train_dataset = AptosDataset(
    TRAIN_PATH, TRAIN_IMG, "train", transforms, resize=(IMG_DIM, IMG_DIM)
)
test_dataset = AptosDataset(
    TEST_PATH, TEST_IMG, "test", transforms, resize=(IMG_DIM, IMG_DIM)
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)

model = models.efficientnet_b0(weights=None)
model.classifier = nn.Sequential(
    nn.Linear(in_features=1280, out_features=512, bias=True),
    nn.Linear(in_features=512, out_features=1, bias=True),
)
model = OrdinalLogisticModel(model, num_classes=5).to(device)

criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

EPOCHS = 1
model.train()
for epoch in range(EPOCHS):
    running_loss = 0.0
    pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{EPOCHS}", leave=False)
    for x, y in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        probs = model(x)  # (B, 5), sums to 1
        log_probs = torch.log(torch.clamp(probs, min=1e-8))
        loss = criterion(log_probs, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)
        pbar.set_postfix(loss=loss.item())
    print(f"Train loss: {running_loss / len(train_dataset):.5f}")



## === cell 6
model.eval()
y_pred = []
with torch.no_grad():
    for x in tqdm(test_loader, desc="inference", leave=False):
        x = x.to(device, non_blocking=True)
        probs = model(x)
        preds = torch.argmax(probs, dim=1).to("cpu").numpy().tolist()
        y_pred.extend(preds)

sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
if len(y_pred) != len(sample_sub):
    raise RuntimeError(
        f"Prediction length {len(y_pred)} != sample submission length {len(sample_sub)}"
    )
sample_sub["diagnosis"] = y_pred
sample_sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sample_sub.shape)
print(sample_sub.head())
