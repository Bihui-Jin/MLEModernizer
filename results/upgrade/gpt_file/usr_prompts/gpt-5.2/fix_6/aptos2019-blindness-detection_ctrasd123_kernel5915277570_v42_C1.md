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

0.8747400261442159

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I remove the non-Python prose and stray backticks that currently cause `SyntaxError`, keeping the same pipeline structure and core model/inference logic. I also eliminate the broken `tensorboard` fallback import path that triggers an internal `tensorboard` error, replacing it with a tiny no-op `SummaryWriter` stub (it isn’t used for submission). Next, I fix the `__future__` import placement (must be the first statement in the file) to avoid runtime syntax issues. Finally, I make file paths consistent with the provided Kaggle dataset layout and ensure we always write `/kaggle/working/submission.csv` with the required `id_code,diagnosis` columns and correct row alignment.'
- What this solution (achieved 0.00132) has done: 'Your 0.0 score is consistent with running inference from randomly initialized DenseNet weights (the checkpoint path points to a dataset that isn’t available), plus a brittle prediction rule that can output mostly a single class. To move the score upward toward the target while keeping the same model and inference semantics, I (1) load ImageNet pretrained weights for the DenseNet backbone when the competition checkpoint is missing, (2) use the standard ImageNet normalization at input (this matches the pretrained backbone expectations), and (3) make `get_preds` robust so it always maps the 5 sigmoid outputs to an ordinal class 0–4 without falling into the “all false → class 0” failure mode. These are minimal, local changes that preserve the architecture and overall pipeline while making predictions meaningfully non-random.'
- What this solution (achieved -0.02386) has done: 'Your current 0.00132 score suggests the model head is effectively random (checkpoint missing) and the fixed 0.5 thresholding is producing poorly calibrated ordinal labels. To move the score upward toward the 0.8747 target without changing the architecture or adding a training loop, I (1) keep ImageNet-pretrained DenseNet201 when the checkpoint is missing, but replace the brittle fixed thresholding with a tiny, validation-free calibration: choose thresholds so the predicted class distribution matches the training label distribution. This keeps the same evaluation semantics (predict 0–4) and uses only provided `train.csv` (no leakage from test labels). I also make inference deterministic and ensure the merge/output row order matches `test.csv` exactly.'
- What this solution (achieved 0.0) has done: 'Your score is far below the target, so we should increase performance with the smallest changes that keep your DenseNet201 + 5-logit ordinal-BCE-style head and “sum-of-sigmoids → scalar → thresholds → class” semantics intact. The biggest issue is that when the competition checkpoint is missing, the classification head is randomly initialized, making predictions essentially noise; we keep the same architecture but (1) initialize the head and `cal_score` weights to zeros for stable, monotonic outputs, and (2) calibrate thresholds on a tiny train-derived set of predictions (not labels) so the mapping from scalar score→class is aligned to the model’s output scale rather than using test-quantiles (which can be unstable for kappa). Finally, we sort `sub_pred` to match `test.csv` order directly (no merge ambiguity) and keep deterministic settings unchanged.'

# 9. Code solution

## === cell 0
from __future__ import print_function, absolute_import

import os
import os.path as osp
import sys
import time
import datetime
import argparse
import random
import math
import csv

import numpy as np
import pandas as pd
from PIL import Image
import cv2

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm


class SummaryWriter:  # minimal stub
    def __init__(self, *args, **kwargs):
        pass

    def add_scalar(self, *args, **kwargs):
        pass

    def add_image(self, *args, **kwargs):
        pass

    def close(self):
        pass


DATA_ROOT = "/kaggle/input/aptos2019-blindness-detection"
if not os.path.exists(DATA_ROOT):
    alt = "/kaggle/data/aptos2019-blindness-detection"
    if os.path.exists(alt):
        DATA_ROOT = alt

TEST_CSV = os.path.join(DATA_ROOT, "test.csv")
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test_images")
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train_images")




## === cell 1
class Baseline_single(nn.Module):
    def __init__(
        self, num_classes, loss_type="single BCE", use_imagenet_backbone=False, **kwargs
    ):
        super(Baseline_single, self).__init__()
        self.loss_type = loss_type

        if use_imagenet_backbone:
            try:
                backbone = torchvision.models.densenet201(
                    weights=torchvision.models.DenseNet201_Weights.IMAGENET1K_V1
                )
            except Exception:
                backbone = torchvision.models.densenet201(weights="IMAGENET1K_V1")
        else:
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
        x = nn.functional.relu(x, inplace=True)

        if self.loss_type == "single BCE":
            x = self.ap(x)
            x = self.dropout(x)
            x = x.view(x.size(0), -1)
            ys = self.classifiers(x)
            return ys

        return x




## === cell 2
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


def probs_to_score(p5):
    """
    p5: [B, 5] sigmoid outputs.
    Convert 5 independent probs to a single scalar severity score in [0,4] by summing
    ordinal threshold probabilities, then clip.
    """
    p5 = np.asarray(p5, dtype=np.float32)
    if p5.ndim == 1:
        p5 = p5[None, :]
    s = p5.sum(axis=1)
    return np.clip(s, 0.0, 4.0)


def make_quantile_thresholds_from_train_labels(train_csv_path):
    train_df = pd.read_csv(train_csv_path)
    y = train_df["diagnosis"].astype(int).values
    counts = np.bincount(y, minlength=5).astype(np.float64)
    priors = counts / counts.sum()
    cum = np.cumsum(priors)  # length 5
    qs = [cum[0], cum[1], cum[2], cum[3]]
    qs = [float(np.clip(q, 1e-6, 1 - 1e-6)) for q in qs]
    return qs


def fit_score_thresholds_from_train_predictions(train_scores, train_labels):
    """
    Minimal calibration to stabilize kappa: choose 4 score cutoffs so that the
    class frequencies on train predictions match the true train label frequencies.
    This uses only train labels and train *predicted scores* (no test leakage).
    """
    s = np.asarray(train_scores, dtype=np.float32).reshape(-1)
    y = np.asarray(train_labels, dtype=np.int64).reshape(-1)
    counts = np.bincount(y, minlength=5).astype(np.float64)
    priors = counts / counts.sum()
    cum = np.cumsum(priors)  # 5
    qs = [float(np.clip(cum[k], 1e-6, 1 - 1e-6)) for k in range(4)]
    t = np.quantile(s, qs)
    t = np.asarray(t, dtype=np.float32)
    for i in range(1, len(t)):
        if t[i] <= t[i - 1]:
            t[i] = np.nextafter(t[i - 1], np.float32(1e9))
    return t


def score_to_class(scores, thresholds):
    """
    Map scalar scores to classes: class = number of thresholds exceeded.
    thresholds: array-like length 4 in ascending order.
    """
    s = np.asarray(scores, dtype=np.float32).reshape(-1)
    t = np.asarray(thresholds, dtype=np.float32).reshape(4)
    cls = (s[:, None] > t[None, :]).sum(axis=1).astype(np.int64)
    return np.clip(cls, 0, 4)


class eye_dataset_orl(Dataset):
    def __init__(self, ids_with_ext, img_dir, transform=None):
        self.imgs = list(ids_with_ext)
        self.img_dir = img_dir
        self.transform = transform

    def __getitem__(self, index):
        fn = self.imgs[index]
        img_path = os.path.join(self.img_dir, fn)
        img = cv2.imread(img_path)
        if img is None:
            img = np.zeros((512, 512, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = crop_image_from_gray(img)
            img = cv2.resize(img, (512, 512))
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, fn[:-4]

    def __len__(self):
        return len(self.imgs)




## === cell 3
if __name__ == "__main__":
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.cuda.manual_seed_all(seed)
    else:
        print("Currently using CPU (GPU is highly recommended)")

    transform2 = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    if not os.path.exists(TEST_CSV):
        raise FileNotFoundError(f"Could not find test.csv at {TEST_CSV}")
    if not os.path.exists(TRAIN_CSV):
        raise FileNotFoundError(f"Could not find train.csv at {TRAIN_CSV}")

    test_df = pd.read_csv(TEST_CSV)
    train_df = pd.read_csv(TRAIN_CSV)

    test_content = (test_df["id_code"].astype(str) + ".png").tolist()
    test_data = eye_dataset_orl(test_content, TEST_IMG_DIR, transform2)

    ckpt_path = "/kaggle/input/temp-file/model_yuan512_dense201_00001_adam_combine_orl_bce_maxest.pkl"
    have_ckpt = os.path.exists(ckpt_path)

    net = Baseline_single(num_classes=5, use_imagenet_backbone=(not have_ckpt))

    if use_gpu:
        net = net.cuda()

    if have_ckpt:
        state = torch.load(ckpt_path, map_location="cuda" if use_gpu else "cpu")
        try:
            net.load_state_dict(state, strict=True)
            print(f"Loaded checkpoint: {ckpt_path}")
        except Exception as e:
            print(
                f"Checkpoint found but failed to load strictly ({e}); trying non-strict."
            )
            net.load_state_dict(state, strict=False)
    else:
        print(
            f"WARNING: checkpoint not found at {ckpt_path}. Using ImageNet-pretrained backbone + stable zero-initialized head."
        )

        with torch.no_grad():
            nn.init.zeros_(net.classifiers.weight)
            nn.init.zeros_(net.classifiers.bias)
            nn.init.zeros_(net.cal_score.weight)
            nn.init.zeros_(net.cal_score.bias)

    dataloader_test = DataLoader(
        test_data, batch_size=8, shuffle=False, num_workers=2, pin_memory=use_gpu
    )

    n_calib = min(512, len(train_df))
    calib_df = train_df.iloc[:n_calib].copy()
    calib_content = (calib_df["id_code"].astype(str) + ".png").tolist()
    calib_data = eye_dataset_orl(calib_content, TRAIN_IMG_DIR, transform2)
    dataloader_calib = DataLoader(
        calib_data, batch_size=8, shuffle=False, num_workers=2, pin_memory=use_gpu
    )

    calib_scores = []
    calib_labels = calib_df["diagnosis"].astype(int).values

    with torch.no_grad():
        net.eval()
        for _, item in tqdm(
            enumerate(dataloader_calib), total=len(dataloader_calib), desc="Calib"
        ):
            data, _name = item
            if use_gpu:
                data = data.cuda(non_blocking=True)
            out = net(data)
            p = torch.sigmoid(out).detach().cpu().numpy()
            s = probs_to_score(p)
            calib_scores.append(s)

    calib_scores = np.concatenate(calib_scores, axis=0)
    thresholds = fit_score_thresholds_from_train_predictions(calib_scores, calib_labels)

    names_all = []
    scores_all = []

    with torch.no_grad():
        net.eval()
        for _, item in tqdm(
            enumerate(dataloader_test), total=len(dataloader_test), desc="Test"
        ):
            data, name = item
            if use_gpu:
                data = data.cuda(non_blocking=True)
            out = net(data)
            p = torch.sigmoid(out).detach().cpu().numpy()  # [B,5]
            s = probs_to_score(p)  # [B]
            names_all.extend([str(n) for n in list(name)])
            scores_all.append(s)

    scores_all = np.concatenate(scores_all, axis=0)
    pred_cls = score_to_class(scores_all, thresholds)

    sub_pred = pd.DataFrame({"id_code": names_all, "diagnosis": pred_cls.astype(int)})

    sub = test_df[["id_code"]].merge(sub_pred, on="id_code", how="left", sort=False)
    sub["diagnosis"] = sub["diagnosis"].fillna(0).astype(int)

    sub_path = "/kaggle/working/submission.csv"
    sub.to_csv(sub_path, index=False)
    print("Wrote:", sub_path)
    print(sub["diagnosis"].value_counts().sort_index())
    print("Thresholds used:", thresholds)
