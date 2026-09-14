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

0.68411

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the runtime failure by making the weight loading robust: the current path points to a non-existent dataset (`/kaggle/input/temp-file/...`), so I search common Kaggle input locations for the `.pkl` and load it if found. If no weights are available, the script still run end-to-end by falling back to an untrained model (score be poor but a valid `submission.csv` be produced), rather than crashing. I also remove excessive per-row printing from CSV reading (which slows Kaggle) and make `DataLoader` worker count safer for Kaggle while keeping the model/inference logic identical. Finally, I keep the submission format exactly as required and write to `/kaggle/working/submission.csv`.'
- What this solution (achieved 0.68411) has done: 'Your 0.0 score is consistent with running an untrained model because the expected `.pkl` weights are not present in this dataset, so the key improvement is to train the existing DenseNet121 head+backbone on `train.csv` using the same CrossEntropy objective and then run the exact same argmax inference for the submission. I keep your model, preprocessing (512 resize + crop-from-gray), and inference semantics intact, and only add a minimal train/val split plus a short training loop that fits within the time limit. To better match quadratic weighted kappa without changing the core model, I select the best epoch by validation QWK (computed from argmax class predictions) and then predict test with that checkpoint. I also make the submission writing batched (still identical predictions) to avoid slow per-row file I/O, improving stability within Kaggle’s runtime limits.'

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

import torchvision as tv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from sklearn.metrics import f1_score, cohen_kappa_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
from torch.optim import lr_scheduler
from tqdm import tqdm
import torchvision.transforms as transforms



## === cell 1
name_file = "/kaggle/input/aptos2019-blindness-detection/test.csv"
with open(name_file, "r", newline="") as f:
    csv_file = csv.reader(f)
    content = []
    for line in csv_file:
        content.append(line[0] + ".png")
    content = content[1:]



## === cell 2
import torch
from torch import nn
import torchvision
import numpy as np


class Baseline_single(nn.Module):
    def __init__(self, num_classes, loss_type="single BCE", **kwargs):
        super(Baseline_single, self).__init__()
        self.loss_type = loss_type

        densenet121 = torchvision.models.densenet121(weights=None)
        self.base = densenet121.features  # outputs (B, 1024, H, W)
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
def cv_imread(file_path):
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
    return cv_img


def change_size(image):
    b = cv2.threshold(image, 15, 255, cv2.THRESH_BINARY)
    binary_image = b[1]
    binary_image = cv2.cvtColor(binary_image, cv2.COLOR_BGR2GRAY)
    print(binary_image.shape)

    x = binary_image.shape[0]
    print("高度x=", x)
    y = binary_image.shape[1]
    print("宽度y=", y)
    edges_x = []
    edges_y = []

    for i in range(x):
        for j in range(y):
            if binary_image[i][j] == 255:
                edges_x.append(i)
                edges_y.append(j)

    left = min(edges_x)
    right = max(edges_x)
    width = right - left

    bottom = min(edges_y)
    top = max(edges_y)
    height = top - bottom

    pre1_picture = image[left : left + width, bottom : bottom + height]
    return pre1_picture


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
        if check_shape == 0:
            return img
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)
        return img


def load_ben_color(image, sigmaX=10):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_image_from_gray(image)
    image = cv2.resize(image, (492, 492))
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)
    return image


def load_ben_yuan(image, sigmaX=10):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_image_from_gray(image)
    image = cv2.resize(image, (512, 512))
    return image


cnt_t = 0


class eye_dataset(Dataset):
    """docstring for data"""

    def __init__(self, txt_path, transform=None):
        imgs = []
        for img in txt_path:
            imgs.append(img)
        self.imgs = imgs
        self.transform = transform

    def __getitem__(self, index):
        fn = self.imgs[index]
        img = cv_imread("/kaggle/input/aptos2019-blindness-detection/test_images/" + fn)
        if img is None:
            raise FileNotFoundError(
                f"Failed to read image: /kaggle/input/aptos2019-blindness-detection/test_images/{fn}"
            )
        img = load_ben_yuan(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, fn[:-4]

    def __len__(self):
        return len(self.imgs)


class eye_dataset_train(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fn = row["id_code"] + ".png"
        y = int(row["diagnosis"])
        img = cv_imread(
            "/kaggle/input/aptos2019-blindness-detection/train_images/" + fn
        )
        if img is None:
            raise FileNotFoundError(
                f"Failed to read image: /kaggle/input/aptos2019-blindness-detection/train_images/{fn}"
            )
        img = load_ben_yuan(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, y

    def __len__(self):
        return len(self.df)




## === cell 4
def _find_first_existing_path(candidates):
    for p in candidates:
        if p and osp.exists(p):
            return p
    return None


def find_weights_path():
    """
    Bugfix retained: original code hard-coded a non-existent path /kaggle/input/temp-file/...
    We still search, but since aptos2019 dataset does not ship weights, we train below if none found.
    """
    expected_name = "model_yuan512_dense121_00001_adam_avg.pkl"
    candidates = [
        f"/kaggle/input/temp-file/{expected_name}",
        f"/kaggle/input/aptos2019-blindness-detection/{expected_name}",
        f"/kaggle/input/{expected_name}",
        f"/kaggle/working/{expected_name}",
    ]
    p = _find_first_existing_path(candidates)
    if p is not None:
        return p

    for root, _, files in os.walk("/kaggle/input"):
        if expected_name in files:
            return osp.join(root, expected_name)

    for root, _, files in os.walk("/kaggle/input"):
        for fn in files:
            if fn.lower().endswith(".pkl"):
                return osp.join(root, fn)

    return None




## === cell 5
if __name__ == "__main__":
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(seed)
    else:
        print("Currently using CPU (GPU is highly recommended)")

    transform2 = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    name_file = "/kaggle/input/aptos2019-blindness-detection/test.csv"
    with open(name_file, "r", newline="") as f:
        csv_file = csv.reader(f)
        content = []
        for line in csv_file:
            content.append(line[0] + ".png")
        content = content[1:]

    test_data = eye_dataset(content, transform2)

    net = Baseline_single(num_classes=5)
    if use_gpu:
        net = net.cuda()

    criterion = nn.CrossEntropyLoss()

    weights_path = find_weights_path()
    if weights_path is not None:
        print(f"Loading weights from: {weights_path}")
        state = torch.load(weights_path, map_location="cuda" if use_gpu else "cpu")
        if (
            isinstance(state, dict)
            and "state_dict" in state
            and isinstance(state["state_dict"], dict)
        ):
            state = state["state_dict"]
        net.load_state_dict(state, strict=True)
        best_state = net.state_dict()
    else:
        print(
            "No .pkl weights found under /kaggle/input; training from scratch to improve score."
        )

        train_df = pd.read_csv("/kaggle/input/aptos2019-blindness-detection/train.csv")
        tr_df, va_df = train_test_split(
            train_df,
            test_size=0.15,
            random_state=seed,
            stratify=train_df["diagnosis"],
        )

        train_ds = eye_dataset_train(tr_df, transform2)
        val_ds = eye_dataset_train(va_df, transform2)

        train_loader = DataLoader(
            train_ds,
            batch_size=8,
            shuffle=True,
            num_workers=2,
            pin_memory=use_gpu,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=8,
            shuffle=False,
            num_workers=2,
            pin_memory=use_gpu,
        )

        optimizer = optim.Adam(net.parameters(), lr=1e-4)

        def eval_qwk(model):
            model.eval()
            ys_true = []
            ys_pred = []
            with torch.no_grad():
                for x, y in val_loader:
                    if use_gpu:
                        x = x.cuda(non_blocking=True)
                    logits = model(x)
                    pred = torch.argmax(logits, dim=1).cpu().numpy().tolist()
                    ys_pred.extend(pred)
                    ys_true.extend(y.numpy().tolist())
            return cohen_kappa_score(ys_true, ys_pred, weights="quadratic")

        best_qwk = -1.0
        best_state = None

        epochs = 3
        for epoch in range(1, epochs + 1):
            net.train()
            running_loss = 0.0
            for x, y in tqdm(
                train_loader, desc=f"train epoch {epoch}/{epochs}", leave=False
            ):
                if use_gpu:
                    x = x.cuda(non_blocking=True)
                    y = y.cuda(non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                logits = net(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                running_loss += float(loss.item()) * x.size(0)

            qwk = eval_qwk(net)
            avg_loss = running_loss / max(1, len(train_ds))
            print(f"epoch {epoch}: train_loss={avg_loss:.5f} val_qwk={qwk:.5f}")

            if qwk > best_qwk:
                best_qwk = qwk
                best_state = {
                    k: v.detach().cpu().clone() for k, v in net.state_dict().items()
                }

        if best_state is None:
            best_state = net.state_dict()
        else:
            net.load_state_dict(best_state, strict=True)
        print(f"Best val QWK selected checkpoint: {best_qwk:.5f}")

    dataloader_test = DataLoader(
        test_data,
        batch_size=8,
        shuffle=False,
        num_workers=2,
        pin_memory=use_gpu,
    )

    sub_path = "/kaggle/working/submission.csv"
    ids_out = []
    preds_out = []

    with torch.no_grad():
        net.eval()
        for data, name in tqdm(dataloader_test, total=len(dataloader_test)):
            if use_gpu:
                data = data.cuda(non_blocking=True)
            out = net(data)
            predicted = torch.argmax(out, dim=1).cpu().numpy().astype(int).tolist()
            ids_out.extend([str(n) for n in name])
            preds_out.extend(predicted)

    sub = pd.DataFrame({"id_code": ids_out, "diagnosis": preds_out})
    sub.to_csv(sub_path, index=False)

    print(f"Wrote submission to: {sub_path}")
    print(pd.read_csv(sub_path).head())
    print("Submission shape:", sub.shape)
