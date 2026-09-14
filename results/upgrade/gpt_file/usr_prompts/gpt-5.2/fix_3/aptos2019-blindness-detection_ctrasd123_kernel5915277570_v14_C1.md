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

0.7032645054647115

# 6. Current score

0.88815

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.35993) has done: 'I remove the hard dependency on `tensorboardX` (it isn’t installed) by providing a safe fallback to PyTorch’s built-in `SummaryWriter`. Then I fix the missing pretrained weight path by training the same DenseNet121-based model inside the notebook using the provided `train.csv/train_images` so inference can run end-to-end without external files. Finally, I ensure the prediction-to-label conversion is consistent with the model output (logits) and write a properly formatted `/kaggle/working/submission.csv` with the required columns.'
- What this solution (achieved 0.88815) has done: 'I fix the import/runtime failure by removing the broken TensorBoard fallback that triggers a TensorFlow/tensorboard compatibility error in this environment, since logging isn’t used for training/inference here. Then I correct a logic bug in the model: the DenseNet121 feature dimension is 1024 (not `512*2`), which currently makes the classifier layer incompatible and/or degrades learning. Finally, to move score upward toward your target without changing the core approach, I enable ImageNet pretrained weights for DenseNet121 (same architecture/training loop) and keep everything else the same so the script runs end-to-end and writes `/kaggle/working/submission.csv`.'

# 9. Code solution

## === cell 0
import os
import sys
import time
import datetime
import argparse
import os.path as osp
import random
import csv

import numpy as np
import pandas as pd

from PIL import Image
import cv2

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

SummaryWriter = None

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
cudnn.deterministic = True
cudnn.benchmark = False

DATA_ROOT = "/kaggle/input/aptos2019-blindness-detection"
TRAIN_CSV = osp.join(DATA_ROOT, "train.csv")
TEST_CSV = osp.join(DATA_ROOT, "test.csv")
TRAIN_IMG_DIR = osp.join(DATA_ROOT, "train_images")
TEST_IMG_DIR = osp.join(DATA_ROOT, "test_images")



## === cell 1
test_df = pd.read_csv(TEST_CSV)
content = (test_df["id_code"].astype(str) + ".png").tolist()
print("Num test images:", len(content), "Example:", content[0])




## === cell 2
class Baseline_single(nn.Module):
    def __init__(self, num_classes, loss_type="single BCE", **kwargs):
        super(Baseline_single, self).__init__()
        self.loss_type = loss_type

        try:
            weights = torchvision.models.DenseNet121_Weights.IMAGENET1K_V1
        except Exception:
            weights = None

        densenet = torchvision.models.densenet121(weights=weights)
        self.base = nn.Sequential(*list(densenet.children())[:-1])

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
        if self.loss_type == "single BCE":
            x = self.ap(x)
            x = self.dropout(x)
            x = x.view(x.size(0), -1)
            ys = self.classifiers(x)  # logits for 5 classes
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
        img = np.stack([img1, img2, img3], axis=-1)
        return img


def load_ben_yuan(image, sigmaX=10):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_image_from_gray(image)
    image = cv2.resize(image, (512, 512))
    return image


class eye_dataset(Dataset):
    def __init__(self, img_list, img_dir, transform=None, labels=None):
        self.imgs = list(img_list)
        self.img_dir = img_dir
        self.transform = transform
        self.labels = labels  # dict id_code -> diagnosis (int) if provided

    def __getitem__(self, index):
        fn = self.imgs[index]  # e.g. xxxx.png
        path = osp.join(self.img_dir, fn)
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        img = load_ben_yuan(img)
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        id_code = fn[:-4]
        if self.labels is None:
            return img, id_code
        else:
            y = int(self.labels[id_code])
            return img, y

    def __len__(self):
        return len(self.imgs)




## === cell 4
def make_transforms():
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def train_one_epoch(model, loader, optimizer, device, criterion):
    model.train()
    running_loss = 0.0
    for xb, yb in tqdm(loader, desc="train", leave=False):
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * xb.size(0)
    return running_loss / len(loader.dataset)


def infer(model, loader, device):
    model.eval()
    preds = []
    ids = []
    with torch.no_grad():
        for xb, id_code in tqdm(loader, desc="infer", leave=False):
            xb = xb.to(device, non_blocking=True)
            logits = model(xb)
            pred = torch.argmax(logits, dim=1).detach().cpu().numpy().astype(int)
            preds.extend(pred.tolist())
            ids.extend(list(id_code))
    return ids, preds


if __name__ == "__main__":
    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")
    if not use_gpu:
        print("Currently using CPU (GPU is highly recommended)")

    train_df = pd.read_csv(TRAIN_CSV)
    train_df["filename"] = train_df["id_code"].astype(str) + ".png"

    perm = np.random.RandomState(SEED).permutation(len(train_df))
    train_idx = perm[:]  # use all data to maximize final model for test inference
    train_df_use = train_df.iloc[train_idx].reset_index(drop=True)

    labels_map = dict(
        zip(train_df_use["id_code"].astype(str), train_df_use["diagnosis"].astype(int))
    )

    transform = make_transforms()
    train_ds = eye_dataset(
        train_df_use["filename"].tolist(),
        TRAIN_IMG_DIR,
        transform=transform,
        labels=labels_map,
    )
    train_loader = DataLoader(
        train_ds, batch_size=8, shuffle=True, num_workers=2, pin_memory=use_gpu
    )

    net = Baseline_single(num_classes=5).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-4)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)

    EPOCHS = 2
    for epoch in range(EPOCHS):
        loss = train_one_epoch(net, train_loader, optimizer, device, criterion)
        scheduler.step()
        print(f"Epoch {epoch+1}/{EPOCHS} - loss: {loss:.4f}")

    test_ds = eye_dataset(content, TEST_IMG_DIR, transform=transform, labels=None)
    test_loader = DataLoader(
        test_ds, batch_size=8, shuffle=False, num_workers=2, pin_memory=use_gpu
    )

    ids, preds = infer(net, test_loader, device)

    sub = pd.DataFrame({"id_code": ids, "diagnosis": preds})
    sub = test_df.merge(sub, on="id_code", how="left")[["id_code", "diagnosis"]]
    sub_path = "/kaggle/working/submission.csv"
    sub.to_csv(sub_path, index=False)
    print("Wrote submission:", sub_path, "rows:", len(sub))
