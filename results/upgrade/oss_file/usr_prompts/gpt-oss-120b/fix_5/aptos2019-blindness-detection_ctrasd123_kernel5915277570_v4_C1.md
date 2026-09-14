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

0.7955458937233086

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved -0.01599) has done: 'I fixed the import errors (removed the unused tensorboardX and the failing Inception imports), added a safe‑load for the model weights so the script runs even if the checkpoint is missing, and kept the original architecture and inference flow. The script now creates a proper `submission.csv` with the required columns and can be executed end‑to‑end.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os, csv, cv2, random, argparse, datetime, time, sys
from PIL import Image
import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
import torchvision as tv
import torchvision.transforms as transforms

from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)




## === cell 1
def cv_imread(file_path):
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
    return cv_img


def crop_image_from_gray(img, tol=7):
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


def load_ben_yuan(image, target_size=224):
    """
    Crop background and resize directly to the final model input size.
    This removes an unnecessary intermediate resize to 512×512.
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_image_from_gray(image)
    image = cv2.resize(image, (target_size, target_size))
    return image


class eye_dataset(Dataset):
    """Test‑time dataset – returns image tensor and filename (without extension)."""

    def __init__(self, fn_list, transform=None):
        self.imgs = fn_list
        self.transform = transform
        self.cached = []
        for fn in self.imgs:
            img = cv_imread(
                "/kaggle/input/aptos2019-blindness-detection/test_images/" + fn
            )
            img = load_ben_yuan(img)  # already 224×224
            self.cached.append(img)

    def __getitem__(self, index):
        fn = self.imgs[index]
        img = self.cached[index]
        if self.transform:
            img = self.transform(img)
        return img, fn[:-4]  # strip ".png"

    def __len__(self):
        return len(self.imgs)


class eye_dataset_train(Dataset):
    """Training / validation dataset – returns image tensor and integer label."""

    def __init__(self, csv_path, img_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform
        self.cached = []
        for idx in range(len(self.df)):
            fn = self.df.iloc[idx]["id_code"] + ".png"
            img_path = os.path.join(self.img_dir, fn)
            img = cv_imread(img_path)
            img = load_ben_yuan(img)  # already 224×224
            self.cached.append(img)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img = self.cached[idx]
        label = int(self.df.iloc[idx]["diagnosis"])
        if self.transform:
            img = self.transform(img)
        return img, label




## === cell 2
class Baseline_single(nn.Module):
    def __init__(self, num_classes, loss_type="single BCE", **kwargs):
        super(Baseline_single, self).__init__()
        self.loss_type = loss_type
        densenet = tv.models.densenet121(pretrained=True)
        self.base = nn.Sequential(*list(densenet.children())[:-1])
        self.feature_dim = 1024  # matches densenet121 output before classifier
        if self.loss_type == "single BCE":
            self.ap = nn.AdaptiveAvgPool2d(1)
            self.dropout = nn.Dropout(0.5)
            self.classifiers = nn.Linear(
                in_features=self.feature_dim, out_features=num_classes
            )

    def forward(self, x1):
        x = self.base(x1)
        if self.loss_type == "single BCE":
            x = self.ap(x)
            x = self.dropout(x)
            x = x.view(x.size(0), -1)
            ys = self.classifiers(x)
            return ys




## === cell 3
if __name__ == "__main__":
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(0)
    else:
        print("Currently using CPU (GPU is highly recommended)")

    train_csv_path = "../input/aptos2019-blindness-detection/train.csv"
    train_img_dir = "/kaggle/input/aptos2019-blindness-detection/train_images/"

    df_all = pd.read_csv(train_csv_path)
    train_idx, val_idx = train_test_split(
        df_all.index, test_size=0.10, stratify=df_all["diagnosis"], random_state=42
    )
    df_train = df_all.loc[train_idx].reset_index(drop=True)
    df_val = df_all.loc[val_idx].reset_index(drop=True)

    temp_train_csv = "/kaggle/working/temp_train.csv"
    temp_val_csv = "/kaggle/working/temp_val.csv"
    df_train.to_csv(temp_train_csv, index=False)
    df_val.to_csv(temp_val_csv, index=False)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    train_dataset = eye_dataset_train(
        temp_train_csv, train_img_dir, transform=train_transform
    )
    val_dataset = eye_dataset_train(
        temp_val_csv, train_img_dir, transform=val_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=2,
        pin_memory=use_gpu,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=2,
        pin_memory=use_gpu,
    )

    net = Baseline_single(num_classes=5)
    if use_gpu:
        net = net.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

    epochs = 6  # increased epochs for better learning
    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        for imgs, labels in tqdm.tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{epochs} [train]"
        ):
            if use_gpu:
                imgs, labels = imgs.cuda(non_blocking=True), labels.cuda(
                    non_blocking=True
                )
            optimizer.zero_grad()
            outputs = net(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
        epoch_loss = running_loss / len(train_dataset)
        print(f"Epoch {epoch+1} - Training loss: {epoch_loss:.4f}")

        net.eval()
        all_preds = []
        all_true = []
        with torch.no_grad():
            for imgs, labels in tqdm.tqdm(
                val_loader, desc=f"Epoch {epoch+1}/{epochs} [val]"
            ):
                if use_gpu:
                    imgs = imgs.cuda(non_blocking=True)
                outputs = net(imgs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_true.extend(labels.numpy())
        kappa = cohen_kappa_score(all_true, all_preds, weights="quadratic")
        print(f"Epoch {epoch+1} - Validation QWK: {kappa:.4f}")

        scheduler.step()  # update learning rate

    test_csv_path = "../input/aptos2019-blindness-detection/test.csv"
    with open(test_csv_path, "r") as f:
        reader = csv.reader(f)
        content = [line[0] + ".png" for line in list(reader)[1:]]

    test_transform = val_transform  # same as validation (no random ops)
    test_data = eye_dataset(content, transform=test_transform)

    net.eval()
    submission_path = "/kaggle/working/submission.csv"
    with open(submission_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id_code", "diagnosis"])
        with torch.no_grad():
            for data, name in tqdm.tqdm(
                DataLoader(
                    test_data,
                    batch_size=1,
                    shuffle=False,
                    num_workers=2,
                    pin_memory=use_gpu,
                )
            ):
                if use_gpu:
                    data = data.cuda(non_blocking=True)
                out = net(data)
                _, predicted = torch.max(out, 1)
                writer.writerow([name[0], predicted.item()])

    print(f"Submission file saved to {submission_path}")
