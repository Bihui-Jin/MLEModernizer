# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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
pillow==11.3.0
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

# 5. Code solution

## === cell 0
import cv2
import matplotlib.pyplot as plt
from os.path import isfile
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
from PIL import Image, ImageFilter
from sklearn.model_selection import train_test_split, StratifiedKFold
from torch.utils.data import Dataset
from torchvision import transforms
from torch.optim import Adam, SGD, RMSprop
import time
from torch.autograd import Variable
from tqdm import tqdm
from sklearn import metrics
import urllib
import pickle
from torchvision import models
import seaborn as sns
import random
import sys
import gc
import warnings



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
TEST_PATH = "../input/aptos2019-blindness-detection/test.csv"
TEST_IMG = "../input/aptos2019-blindness-detection/test_images"
SAMPLE_SUB_PATH = "../input/aptos2019-blindness-detection/sample_submission.csv"
TRAIN_PATH = "../input/aptos2019-blindness-detection/train.csv"
TRAIN_IMG = "../input/aptos2019-blindness-detection/train_images"

if not os.path.exists(TEST_PATH):
    TEST_PATH = "/kaggle/data/aptos2019-blindness-detection/test.csv"
if not os.path.exists(TEST_IMG):
    TEST_IMG = "/kaggle/data/aptos2019-blindness-detection/test_images"
if not os.path.exists(SAMPLE_SUB_PATH):
    SAMPLE_SUB_PATH = "/kaggle/data/aptos2019-blindness-detection/sample_submission.csv"
if not os.path.exists(TRAIN_PATH):
    TRAIN_PATH = "/kaggle/data/aptos2019-blindness-detection/train.csv"
if not os.path.exists(TRAIN_IMG):
    TRAIN_IMG = "/kaggle/data/aptos2019-blindness-detection/train_images"

test_csv = pd.read_csv(TEST_PATH)
train_csv = pd.read_csv(TRAIN_PATH)

print("train:", train_csv.shape, "test:", test_csv.shape)




## === cell 3
def expand_path(p):
    p = str(p)
    candidate = os.path.join(TEST_IMG, p + ".png")
    if isfile(candidate):
        return candidate
    return p




## === cell 4
def crop_image1(img, tol=7):
    mask = img > tol
    return img[np.ix_(mask.any(1), mask.any(0))]


def crop_image_from_gray(img, tol=7):
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol

        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if check_shape == 0:  # image is too dark so that we crop out everything
            return img
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)
        return img




## === cell 5
IMG_SIZE = 256


class MyDataset(Dataset):
    def __init__(self, dataframe, transform=None, img_dir=TEST_IMG, has_label=False):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
        self.img_dir = img_dir
        self.has_label = has_label

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        p = self.df.id_code.values[idx]
        p_path = os.path.join(self.img_dir, p + ".png")

        image = cv2.imread(p_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image at: {p_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = crop_image_from_gray(image)
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), 30), -4, 128)

        image = transforms.ToPILImage()(image)

        if self.transform:
            image = self.transform(image)

        if not torch.is_tensor(image):
            image = transforms.ToTensor()(image)

        if self.has_label:
            y = self.df.diagnosis.values[idx]
            y = torch.tensor(float(y), dtype=torch.float32)
            return image, y

        return image




## === cell 6
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

train_df, val_df = train_test_split(
    train_csv,
    test_size=0.15,
    random_state=SEED,
    stratify=train_csv["diagnosis"],
)

trainset = MyDataset(train_df, transform=transform, img_dir=TRAIN_IMG, has_label=True)
valset = MyDataset(val_df, transform=transform, img_dir=TRAIN_IMG, has_label=True)
testset = MyDataset(test_csv, transform=transform, img_dir=TEST_IMG, has_label=False)

train_loader = torch.utils.data.DataLoader(
    trainset,
    batch_size=16,
    shuffle=True,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)
val_loader = torch.utils.data.DataLoader(
    valset,
    batch_size=32,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)
test_loader = torch.utils.data.DataLoader(
    testset,
    batch_size=32,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)



## === cell 7
try:
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
except Exception:
    weights = None


def build_model():
    m = models.efficientnet_b0(weights=weights)
    m.classifier = nn.Linear(in_features=1280, out_features=1, bias=True)
    return m


model = build_model().to(device)

criterion = nn.MSELoss()
optimizer = Adam(model.parameters(), lr=1e-4)


def predict_loader(m, loader):
    m.eval()
    preds = []
    ys = []
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                x, y = batch
                ys.append(y.numpy())
            else:
                x = batch
            out = m(x.float().to(device)).detach().cpu().view(-1).numpy()
            preds.append(out)
    preds = np.concatenate(preds, axis=0)
    if ys:
        ys = np.concatenate(ys, axis=0)
        return preds, ys
    return preds


EPOCHS = 2  # keep minimal for runtime; enough to avoid 0.0 kappa from random head

for epoch in range(EPOCHS):
    model.train()
    running = 0.0
    n = 0
    for x, y in tqdm(
        train_loader, desc=f"train epoch {epoch+1}/{EPOCHS}", total=len(train_loader)
    ):
        x = x.float().to(device)
        y = y.float().to(device)

        optimizer.zero_grad(set_to_none=True)
        out = model(x).view(-1)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        running += loss.item() * x.size(0)
        n += x.size(0)

    val_pred, val_y = predict_loader(model, val_loader)
    val_pred_clip = np.clip(val_pred, 0.0, 4.0)
    naive_round = np.rint(val_pred_clip).astype(int)
    naive_round = np.clip(naive_round, 0, 4)
    kappa = metrics.cohen_kappa_score(
        val_y.astype(int), naive_round, weights="quadratic"
    )
    print(
        f"epoch {epoch+1}: train_loss={running/n:.5f}  val_kappa_naive_round={kappa:.5f}"
    )




## === cell 8
def apply_thresholds(preds_cont, coef):
    coef = list(coef)
    out = np.zeros_like(preds_cont, dtype=np.int64)
    out[preds_cont >= coef[0]] = 1
    out[preds_cont >= coef[1]] = 2
    out[preds_cont >= coef[2]] = 3
    out[preds_cont >= coef[3]] = 4
    return out


def qwk_from_coef(coef, preds_cont, y_true):
    preds_int = apply_thresholds(preds_cont, coef)
    return metrics.cohen_kappa_score(
        y_true.astype(int), preds_int.astype(int), weights="quadratic"
    )


val_pred, val_y = predict_loader(model, val_loader)
val_pred = np.clip(val_pred, 0.0, 4.0)

best_coef = np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float32)
best_score = qwk_from_coef(best_coef, val_pred, val_y)

steps = [0.2, 0.1, 0.05]  # small, deterministic search
for step in steps:
    improved = True
    while improved:
        improved = False
        for i in range(4):
            for delta in (-step, step):
                cand = best_coef.copy()
                cand[i] += delta
                cand = np.clip(cand, 0.05, 3.95)
                cand = np.sort(cand)
                score = qwk_from_coef(cand, val_pred, val_y)
                if score > best_score:
                    best_score = score
                    best_coef = cand
                    improved = True

print("Best val QWK:", float(best_score))
print("Best thresholds:", best_coef.tolist())



## === cell 9
full_trainset = MyDataset(
    train_csv, transform=transform, img_dir=TRAIN_IMG, has_label=True
)
full_train_loader = torch.utils.data.DataLoader(
    full_trainset,
    batch_size=16,
    shuffle=True,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)

final_model = build_model().to(device)
final_criterion = nn.MSELoss()
final_optimizer = Adam(final_model.parameters(), lr=1e-4)

for epoch in range(EPOCHS):
    final_model.train()
    running = 0.0
    n = 0
    for x, y in tqdm(
        full_train_loader,
        desc=f"final train epoch {epoch+1}/{EPOCHS}",
        total=len(full_train_loader),
    ):
        x = x.float().to(device)
        y = y.float().to(device)

        final_optimizer.zero_grad(set_to_none=True)
        out = final_model(x).view(-1)
        loss = final_criterion(out, y)
        loss.backward()
        final_optimizer.step()

        running += loss.item() * x.size(0)
        n += x.size(0)
    print(f"final epoch {epoch+1}: train_loss={running/n:.5f}")

final_model.eval()
y_pred = []
with torch.no_grad():
    for x in tqdm(test_loader, total=len(test_loader), desc="predict test"):
        output = final_model(x.float().to(device))
        y_pred.append(output.detach().cpu().view(-1))
y_pred = torch.cat(y_pred, dim=0).numpy().astype(np.float32)
y_pred = np.clip(y_pred, 0.0, 4.0)

assert len(y_pred) == len(
    test_csv
), f"Pred length {len(y_pred)} != test length {len(test_csv)}"

test_preds = apply_thresholds(y_pred, best_coef).astype(int).tolist()

sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

sub = test_csv[["id_code"]].copy()
sub["diagnosis"] = test_preds
sub["diagnosis"] = sub["diagnosis"].astype(int)

sub = sub[["id_code", "diagnosis"]]
sub.to_csv("submission.csv", index=False)

print(sub.head())
print(f"\nWrote submission.csv with shape: {sub.shape}")
print("diagnosis value counts:\n", sub["diagnosis"].value_counts().sort_index())
