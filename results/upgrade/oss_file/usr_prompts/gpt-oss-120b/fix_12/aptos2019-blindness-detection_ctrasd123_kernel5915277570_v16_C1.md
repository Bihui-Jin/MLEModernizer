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

0.7215394634641098

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
from __future__ import print_function, absolute_import
import os
import sys
import time
import datetime
import argparse
import os.path as osp
import numpy as np
import random
from PIL import Image
import tqdm
import cv2
import csv

import torchvision as tv
import torchvision
import torch.nn.functional as F
import torch.optim as optim
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.optim import lr_scheduler
from tqdm import tqdm
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from collections import Counter  # new import for class‑weight computation
from sklearn.model_selection import train_test_split
import torchmetrics




## === cell 1
class eye_dataset(Dataset):
    """
    Simple dataset for loading retina images.
    If label_dict is provided, returns (image, label).
    Otherwise returns (image, [id_code]) for test inference.
    Caches all transformed images in memory to avoid repeated disk I/O.
    """

    def __init__(self, file_list, transform=None, label_dict=None, root_dir=""):
        self.file_list = file_list
        self.transform = transform
        self.label_dict = label_dict
        self.root_dir = root_dir

        self.cached_images = []
        for img_name in self.file_list:
            img_path = os.path.join(self.root_dir, img_name)
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            self.cached_images.append(image)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        image = self.cached_images[idx]
        img_name = self.file_list[idx]

        if self.label_dict is not None:
            label = self.label_dict[img_name]
            return image, label
        else:
            id_code = img_name.replace(".png", "")
            return image, [id_code]




## === cell 2
class Baseline_single(nn.Module):
    def __init__(self, num_classes, loss_type="single BCE", **kwargs):
        super(Baseline_single, self).__init__()
        self.loss_type = loss_type
        densenet121 = torchvision.models.densenet121(pretrained=True)

        self.base = nn.Sequential(*list(densenet121.children())[:-1])
        self.feature_dim = 1024  # DenseNet121 outputs 1024‑dim features
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
        x = self.base(x1)  # shape: (B, 1024, 7, 7)
        if self.loss_type == "single BCE":
            x = self.ap(x)  # (B, 1024, 1, 1)
            x = self.dropout(x)
            x = x.view(x.size(0), -1)  # (B, 1024)
            ys = self.classifiers(x)  # (B, num_classes)
            return ys
        return x




## === cell 3
if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")
    if use_gpu:
        cudnn.benchmark = True
    else:
        print("Currently using CPU (GPU is highly recommended)")

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    base_path = "/kaggle/input/aptos2019-blindness-detection"
    train_csv_path = os.path.join(base_path, "train.csv")
    train_img_dir = os.path.join(base_path, "train_images")
    test_csv_path = os.path.join(base_path, "test.csv")
    test_img_dir = os.path.join(base_path, "test_images")

    label_dict = {}
    train_filenames = []
    with open(train_csv_path, "r") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            id_code, diag = row[0], int(row[1])
            filename = f"{id_code}.png"
            label_dict[filename] = diag
            train_filenames.append(filename)

    label_counts = Counter(label_dict.values())
    total_samples = sum(label_counts.values())
    num_classes = 5
    class_weights = []
    for i in range(num_classes):
        freq = label_counts.get(i, 0)
        weight = total_samples / (num_classes * (freq + 1e-6))
        class_weights.append(weight)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    strat_labels = [label_dict[f] for f in train_filenames]
    train_files, val_files = train_test_split(
        train_filenames,
        test_size=0.1,
        stratify=strat_labels,
        random_state=42,
    )

    num_workers = 0

    train_dataset = eye_dataset(
        train_files, transform=transform, label_dict=label_dict, root_dir=train_img_dir
    )
    val_dataset = eye_dataset(
        val_files, transform=transform, label_dict=label_dict, root_dir=train_img_dir
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_gpu,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_gpu,
    )

    net = Baseline_single(num_classes=num_classes)
    net = net.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)

    kappa_metric = torchmetrics.CohenKappa(
        num_classes=num_classes, weights="quadratic", task="multiclass"
    ).to(device)

    best_kappa = -1.0
    best_state = None

    net.train()
    for epoch in range(12):
        epoch_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/12"):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()
        print(f"Epoch {epoch+1} - Avg loss: {epoch_loss/len(train_loader):.4f}")

        net.eval()
        kappa_metric.reset()
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                outs = net(images)
                preds = torch.argmax(outs, dim=1)
                kappa_metric.update(preds, labels)
        val_kappa = kappa_metric.compute().item()
        print(f"Validation Quadratic Weighted Kappa: {val_kappa:.4f}")
        if val_kappa > best_kappa:
            best_kappa = val_kappa
            best_state = net.state_dict()
        net.train()

    if best_state is not None:
        net.load_state_dict(best_state)

    test_filenames = []
    with open(test_csv_path, "r") as f:
        csv_file = csv.reader(f)
        next(csv_file)  # skip header
        for row in csv_file:
            test_filenames.append(row[0] + ".png")

    test_dataset = eye_dataset(
        test_filenames, transform=transform, label_dict=None, root_dir=test_img_dir
    )
    dataloader_test = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_gpu,
    )

    submission_path = "./submission.csv"
    predictions = []

    net.eval()
    with torch.no_grad():
        for images, names in tqdm(dataloader_test, desc="Inference"):
            images = images.to(device, non_blocking=True)
            out = net(images)
            _, predicted = torch.max(out, 1)
            for name, pred in zip(names, predicted.cpu().numpy()):
                predictions.append([str(name[0]), str(int(pred))])

    with open(submission_path, "w", newline="") as f:
        f_csv = csv.writer(f)
        f_csv.writerow(["id_code", "diagnosis"])
        f_csv.writerows(predictions)

    print(f"Submission saved to {submission_path}")
