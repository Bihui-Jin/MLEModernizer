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

0.8637257072282227

# 6. Current score

0.6859

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I remove the non-code text that’s currently inside code cells (it’s causing the `SyntaxError`) and keep everything as executable Python only. Then I fix the TensorBoard import crash by making the SummaryWriter optional (your code doesn’t actually use it for producing the submission), avoiding the broken `torch.utils.tensorboard` path in this environment. Finally, I keep your existing DenseNet+512px pipeline and fallback training logic, but make it deterministic and correctly formatted end-to-end so it always writes a valid `/kaggle/working/submission.csv` with the required columns.'
- What this solution (achieved 0.6859) has done: 'Your 0.0 score is almost certainly coming from untrained/random weights (missing checkpoint) plus a post-processing bug: `get_preds` expects a 6-column thresholded array but the model outputs 5 logits, which can generate degenerate predictions and tank QWK. I keep your DenseNet201 + BCE one-hot training exactly as-is, but fix prediction decoding to a valid 5-class argmax mapping (consistent with “single BCE” multi-label logits) so labels are always 0–4. I also make the fallback training split deterministic but shuffled (no label leakage, same logic) and add ImageNet normalization (no architecture change) to bring performance up toward your target without changing the training loop structure. Finally, I ensure the submission is written in the correct row order matching `test.csv`.'

# 9. Code solution

## === cell 0
import os
import sys
import time
import math
import csv
import random
import datetime
import argparse
import os.path as osp

import numpy as np
import pandas as pd

from PIL import Image

import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

try:
    from tensorboardX import SummaryWriter  # noqa: F401
except Exception:
    SummaryWriter = None

for dirname, _, filenames in os.walk("/kaggle/input/aptos2019-blindness-detection"):
    for filename in filenames[:5]:
        print(os.path.join(dirname, filename))
    break



## === cell 1
name_file = "/kaggle/input/aptos2019-blindness-detection/test.csv"
test_df = pd.read_csv(name_file)
content = (test_df["id_code"].astype(str) + ".png").tolist()
print("Loaded test images:", len(content), "example:", content[:3])




## === cell 2
class Baseline_single(nn.Module):
    def __init__(self, num_classes, loss_type="single BCE", **kwargs):
        super(Baseline_single, self).__init__()
        self.loss_type = loss_type

        backbone = torchvision.models.densenet201(weights=None)
        self.base = backbone.features  # outputs [B, 1920, H, W]
        self.feature_dim = 1920

        if self.loss_type == "single BCE":
            self.ap = nn.AdaptiveAvgPool2d(1)
            self.classifiers = nn.Linear(
                in_features=self.feature_dim, out_features=num_classes
            )
            self.sigmoid = nn.Sigmoid()
            self.dropout = nn.Dropout(0.5)
            self.cal_score = nn.Linear(in_features=num_classes, out_features=1)

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
        return x




## === cell 3
def crop_image_from_gray(img, tol=7):
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol

        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if check_shape == 0:  # image too dark
            return img
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)
        return img


def findCircle(image):
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    v_img = image[:, :, 2]
    height, width = v_img.shape
    mask_v_a = cv2.adaptiveThreshold(
        v_img,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        int(max(height, width) / 16) * 2 + 1,
        1,
    )

    ratio = 128 / min(height, width)
    msk = cv2.resize(
        mask_v_a,
        (int(width * ratio), int(height * ratio)),
        interpolation=cv2.INTER_CUBIC,
    )
    h, w = msk.shape
    msk_expand = np.zeros((3 * h, 3 * w), np.uint8)
    msk_expand[h : 2 * h, w : 2 * w] = msk
    long_edge = max(h, w)
    r0 = round(0.3 * long_edge)
    r1 = round(0.7 * long_edge)

    circles = cv2.HoughCircles(
        msk_expand,
        cv2.HOUGH_GRADIENT,
        1,
        90,
        param1=50,
        param2=5,
        minRadius=r0,
        maxRadius=r1,
    )

    if circles is None:
        c_x = width / 2
        c_y = height / 2
        radius = 0.55 * max(height, width)
    else:
        circles = np.uint16(np.around(circles))
        c_x = (circles[0, 0, 0] - w) / ratio
        c_y = (circles[0, 0, 1] - h) / ratio
        radius = circles[0, 0, 2] / ratio
    return c_x, c_y, radius


def circleCrop(c_x, c_y, radius, height, width):
    if math.floor(radius + c_y) > height:
        y0 = max(math.ceil(c_y - radius), 0)
        y1 = height
        x1 = width if math.floor(radius + c_x) > width else math.floor(radius + c_x)
        x0 = 0 if math.floor(c_x - radius < 0) else math.floor(c_x - radius)
    elif math.ceil(c_y - radius) < 0:
        y0 = 0
        y1 = min(math.floor(c_y + radius), height)
        x1 = width if math.floor(radius + c_x) > width else math.floor(radius + c_x)
        x0 = 0 if math.floor(c_x - radius < 0) else math.floor(c_x - radius)
    else:
        y0 = math.ceil(c_y - radius)
        y1 = math.floor(c_y + radius)
        x0 = math.ceil(c_x - radius)
        x1 = math.floor(c_x + radius)
    return x0, x1, y0, y1


def trimFundus(image):
    c_x, c_y, radius = findCircle(image)
    height = image.shape[0]
    width = image.shape[1]
    x0, x1, y0, y1 = circleCrop(c_x, c_y, radius, height, width)
    trimed = image[y0:y1, x0:x1, :]
    return trimed


def get_preds_from_logits(logits_5):
    return np.argmax(logits_5, axis=1).astype(np.int64)


class eye_dataset_orl(Dataset):
    def __init__(self, txt_path, transform=None, train=False, labels_df=None):
        self.imgs = list(txt_path)
        self.transform = transform
        self.train = train
        self.labels_df = labels_df

    def __getitem__(self, index):
        fn = self.imgs[index]
        if self.train:
            img_path = "/kaggle/input/aptos2019-blindness-detection/train_images/" + fn
        else:
            img_path = "/kaggle/input/aptos2019-blindness-detection/test_images/" + fn

        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = crop_image_from_gray(img)
        img = cv2.resize(img, (512, 512))
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.train:
            id_code = fn[:-4]
            y = int(self.labels_df.loc[id_code, "diagnosis"])
            return img, y
        else:
            return img, fn[:-4]

    def __len__(self):
        return len(self.imgs)




## === cell 4
if __name__ == "__main__":
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        cudnn.benchmark = True
    else:
        print("Currently using CPU (GPU is highly recommended)")

    transform2 = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    test_df = pd.read_csv("/kaggle/input/aptos2019-blindness-detection/test.csv")
    content = (test_df["id_code"].astype(str) + ".png").tolist()

    test_data = eye_dataset_orl(content, transform2, train=False)
    net = Baseline_single(num_classes=5)
    if use_gpu:
        net = net.cuda()

    ckpt_path = "/kaggle/input/temp-file/model_yuan512_dense201_00001_adam_combine_orl_bce_newest.pkl"
    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location="cuda" if use_gpu else "cpu")
        net.load_state_dict(state)
        print("Loaded checkpoint:", ckpt_path)
    else:
        print("Checkpoint not found:", ckpt_path)
        print(
            "Falling back to training for a few epochs to produce a valid submission."
        )

        train_df = pd.read_csv("/kaggle/input/aptos2019-blindness-detection/train.csv")
        labels_df = train_df.set_index("id_code")[["diagnosis"]]
        train_files = (train_df["id_code"].astype(str) + ".png").tolist()

        rng = np.random.RandomState(seed)
        perm = rng.permutation(len(train_files))
        train_files = [train_files[i] for i in perm]

        split = int(0.9 * len(train_files))
        train_list = train_files[:split]
        val_list = train_files[split:]

        train_data = eye_dataset_orl(
            train_list, transform2, train=True, labels_df=labels_df
        )
        val_data = eye_dataset_orl(
            val_list, transform2, train=True, labels_df=labels_df
        )

        train_loader = DataLoader(
            train_data, batch_size=8, shuffle=True, num_workers=2, pin_memory=use_gpu
        )
        val_loader = DataLoader(
            val_data, batch_size=8, shuffle=False, num_workers=2, pin_memory=use_gpu
        )

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(net.parameters(), lr=1e-4)

        net.train()
        for epoch in range(2):
            epoch_loss = 0.0
            for x, y in tqdm(train_loader, desc=f"train epoch {epoch+1}/2"):
                if use_gpu:
                    x = x.cuda(non_blocking=True)
                    y = torch.as_tensor(y, device="cuda", dtype=torch.long)
                else:
                    y = torch.as_tensor(y, dtype=torch.long)

                target = F.one_hot(y, num_classes=5).float()
                optimizer.zero_grad(set_to_none=True)
                out = net(x)
                loss = criterion(out, target)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item())
            print(f"epoch {epoch+1} loss {epoch_loss / max(1, len(train_loader)):.4f}")

    dataloader_test = DataLoader(
        test_data, batch_size=4, shuffle=False, num_workers=2, pin_memory=use_gpu
    )

    sub_path = "/kaggle/working/submission.csv"
    if os.path.exists(sub_path):
        os.remove(sub_path)

    with open(sub_path, "w", newline="") as f:
        f_csv = csv.writer(f)
        f_csv.writerow(["id_code", "diagnosis"])

    preds_map = {}

    with torch.no_grad():
        net.eval()
        for _, item in tqdm(
            enumerate(dataloader_test), total=len(dataloader_test), desc="infer"
        ):
            data, names = item
            if use_gpu:
                data = data.cuda(non_blocking=True)
            out = net(data)

            pred = get_preds_from_logits(out.detach().cpu().numpy())

            for n, p in zip(names, pred):
                preds_map[str(n)] = int(p)

    with open(sub_path, "a", newline="") as f:
        f_csv = csv.writer(f)
        for id_code in test_df["id_code"].astype(str).tolist():
            f_csv.writerow([id_code, preds_map[id_code]])

    print("Saved:", sub_path)
    sub_df = pd.read_csv(sub_path)
    print(sub_df.head())
    print(sub_df.diagnosis.value_counts(dropna=False).sort_index())
    assert list(sub_df.columns) == ["id_code", "diagnosis"]
    assert len(sub_df) == len(test_df)
    assert (
        sub_df["id_code"].astype(str).tolist()
        == test_df["id_code"].astype(str).tolist()
    )
