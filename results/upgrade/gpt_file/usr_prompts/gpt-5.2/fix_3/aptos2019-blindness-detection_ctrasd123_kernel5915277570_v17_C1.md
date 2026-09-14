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

3.8

# 3. Installed packages

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
pillow==11.3.0
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

0.7715022529981524

# 6. Current score

0.59767

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.41373) has done: 'I remove the hard dependency on `tensorboardX` (it isn’t installed) by falling back to PyTorch’s built-in TensorBoard writer (or a no-op stub), which fixes the first crash without changing model behavior. Then I fix the missing checkpoint issue by training the existing DenseNet121-based model on `train.csv` inside the notebook and using it to generate test predictions, instead of trying to load a non-existent file. I also correct two logic/torch issues that would break inference/training: the DenseNet feature dimension is wrong and `torch.tensor(x)` in `forward()` detaches tensors; both be fixed minimally while preserving the same architecture and overall approach. Finally, I ensure a valid `/kaggle/working/submission.csv` is written with the required columns and aligned to `test.csv` order.'
- What this solution (achieved 0.59767) has done: 'I fix the TensorBoard import crash by avoiding the `torch.utils.tensorboard`/`tensorboardX` dependency entirely and using a safe no-op `SummaryWriter`, since logging isn’t required for training/inference. Then I make a minimal, score-oriented change by training with an 80/20 stratified split and selecting the best epoch by quadratic weighted kappa (the competition metric) without changing the model architecture or loss. Finally, I ensure inference uses that best checkpoint and that `/kaggle/working/submission.csv` is written in exactly the required `id_code,diagnosis` format aligned to `test.csv` order.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd

import os
import sys
import time
import datetime
import argparse
import os.path as osp
import random
from PIL import Image
import cv2
import csv

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm


class SummaryWriter:  # no-op stub
    def __init__(self, *args, **kwargs):
        pass

    def add_scalar(self, *args, **kwargs):
        pass

    def add_image(self, *args, **kwargs):
        pass

    def close(self):
        pass


from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
cudnn.deterministic = True
cudnn.benchmark = False

DATA_DIR = "/kaggle/input/aptos2019-blindness-detection"
TRAIN_CSV = osp.join(DATA_DIR, "train.csv")
TEST_CSV = osp.join(DATA_DIR, "test.csv")
TRAIN_IMG_DIR = osp.join(DATA_DIR, "train_images")
TEST_IMG_DIR = osp.join(DATA_DIR, "test_images")

print("train.csv exists:", osp.exists(TRAIN_CSV))
print("test.csv exists:", osp.exists(TEST_CSV))
print("train_images exists:", osp.exists(TRAIN_IMG_DIR))
print("test_images exists:", osp.exists(TEST_IMG_DIR))



## === cell 1
test_df = pd.read_csv(TEST_CSV)
content = (test_df["id_code"].astype(str) + ".png").tolist()
print("Num test images:", len(content))
print("First 3:", content[:3])




## === cell 2
class Baseline_single(nn.Module):
    def __init__(self, num_classes, loss_type="single BCE", **kwargs):
        super(Baseline_single, self).__init__()
        self.loss_type = loss_type

        densenet = torchvision.models.densenet121(weights=None)
        self.base = densenet.features

        self.feature_dim = 1024

        if self.loss_type == "single BCE":
            self.ap = nn.AdaptiveAvgPool2d(1)
            self.classifiers = nn.Linear(
                in_features=self.feature_dim, out_features=num_classes
            )
            self.sigmoid = nn.Sigmoid()
            self.dropout = nn.Dropout(0.5)

    def freeze_base(self):
        for p in self.base.parameters():
            p.requires_grad = False

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True

    def forward(self, x1):
        x = self.base(x1)
        x = F.relu(x, inplace=True)
        if self.loss_type == "single BCE":
            x = self.ap(x)
            x = self.dropout(x)
            x = x.view(x.size(0), -1)
            ys = self.classifiers(x)
        return ys




## === cell 3
def crop_image_from_gray(img, tol=7):
    if img is None:
        return img
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol
        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if check_shape == 0:
            return img
        img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
        img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
        img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
        return np.stack([img1, img2, img3], axis=-1)
    return img


def load_ben_yuan(image, sigmaX=10):
    if image is None:
        return None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_image_from_gray(image)
    image = cv2.resize(image, (512, 512))
    return image


class eye_dataset(Dataset):
    def __init__(self, img_names, img_dir, labels=None, transform=None):
        self.imgs = list(img_names)
        self.img_dir = img_dir
        self.labels = None if labels is None else list(labels)
        self.transform = transform

    def __getitem__(self, index):
        fn = self.imgs[index]
        path = osp.join(self.img_dir, fn)
        img = cv2.imread(path)
        img = load_ben_yuan(img)
        if img is None:
            img = np.zeros((512, 512, 3), dtype=np.uint8)
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        if self.labels is None:
            return img, fn[:-4]
        else:
            return img, int(self.labels[index])

    def __len__(self):
        return len(self.imgs)




## === cell 4
use_gpu = torch.cuda.is_available()
device = torch.device("cuda" if use_gpu else "cpu")
print("device:", device)

transform2 = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

train_df = pd.read_csv(TRAIN_CSV)
train_imgs_all = (train_df["id_code"].astype(str) + ".png").tolist()
train_labels_all = train_df["diagnosis"].astype(int).tolist()

tr_imgs, va_imgs, tr_y, va_y = train_test_split(
    train_imgs_all,
    train_labels_all,
    test_size=0.2,
    random_state=SEED,
    stratify=train_labels_all,
)

test_df = pd.read_csv(TEST_CSV)
test_imgs = (test_df["id_code"].astype(str) + ".png").tolist()

train_data = eye_dataset(tr_imgs, TRAIN_IMG_DIR, labels=tr_y, transform=transform2)
val_data = eye_dataset(va_imgs, TRAIN_IMG_DIR, labels=va_y, transform=transform2)
test_data = eye_dataset(test_imgs, TEST_IMG_DIR, labels=None, transform=transform2)

train_loader = DataLoader(
    train_data, batch_size=8, shuffle=True, num_workers=2, pin_memory=use_gpu
)
val_loader = DataLoader(
    val_data, batch_size=8, shuffle=False, num_workers=2, pin_memory=use_gpu
)
test_loader = DataLoader(
    test_data, batch_size=8, shuffle=False, num_workers=2, pin_memory=use_gpu
)

net = Baseline_single(num_classes=5).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=1e-4)

EPOCHS = 2  # keep core training budget unchanged

best_kappa = -1e9
best_state = None

for epoch in range(EPOCHS):
    net.train()
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(
        tqdm(train_loader, desc=f"train epoch {epoch+1}/{EPOCHS}")
    ):
        data = data.to(device, non_blocking=True)
        target = torch.as_tensor(target, device=device, dtype=torch.long)

        optimizer.zero_grad(set_to_none=True)
        out = net(data)
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print(f"epoch {epoch+1} train loss: {running_loss / max(1, len(train_loader)):.4f}")

    net.eval()
    val_preds = []
    val_true = []
    with torch.no_grad():
        for data, target in tqdm(val_loader, desc=f"val epoch {epoch+1}/{EPOCHS}"):
            data = data.to(device, non_blocking=True)
            out = net(data)
            pred = out.argmax(dim=1).detach().cpu().numpy()
            val_preds.append(pred)
            val_true.append(np.asarray(target, dtype=np.int64))
    val_preds = np.concatenate(val_preds)
    val_true = np.concatenate(val_true)
    kappa = cohen_kappa_score(val_true, val_preds, weights="quadratic")
    print(f"epoch {epoch+1} val QWK: {kappa:.6f}")

    if kappa > best_kappa:
        best_kappa = kappa
        best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}

print("best val QWK:", best_kappa)

if best_state is not None:
    net.load_state_dict(best_state)



## === cell 5
net.eval()
pred_rows = []
with torch.no_grad():
    for data, name in tqdm(test_loader, desc="infer"):
        data = data.to(device, non_blocking=True)
        out = net(data)
        predicted = out.argmax(dim=1).detach().cpu().numpy().tolist()
        for i, id_code in enumerate(name):
            pred_rows.append((id_code, int(predicted[i])))

sub = pd.DataFrame(pred_rows, columns=["id_code", "diagnosis"])

sub = test_df.merge(sub, on="id_code", how="left")
sub["diagnosis"] = sub["diagnosis"].fillna(0).astype(int)

out_path = "/kaggle/working/submission.csv"
sub.to_csv(out_path, index=False)
print("Wrote:", out_path)
print(sub.head())
print("submission shape:", sub.shape)
print("submission columns:", list(sub.columns))
print("diagnosis value counts:\n", sub["diagnosis"].value_counts().sort_index())
