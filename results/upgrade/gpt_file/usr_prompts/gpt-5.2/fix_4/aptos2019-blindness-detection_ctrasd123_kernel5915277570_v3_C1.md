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

0.813172612933002

# 6. Current score

-0.35074

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -0.12777) has done: 'I fix the pipeline so it always produces a valid `submission.csv` even when the external pretrained weight file is not attached (your current crash). To keep the core model and preprocessing intact, I (1) make weight loading optional by falling back to torchvision’s built-in DenseNet121 ImageNet weights when the custom checkpoint is missing, and (2) fix the DenseNet feature-dimension mismatch by using `net.classifier.in_features` rather than a hardcoded value. I also make `load_state_dict` robust to common checkpoint formats (`state_dict` key, `module.` prefixes) without changing the architecture or inference logic. These changes unblock end-to-end execution and typically improve over random/uninitialized weights, moving score toward the target without altering evaluation semantics.'
- What this solution (achieved -0.35074) has done: 'Your current negative score is consistent with using ImageNet backbone features but a randomly initialized classifier head, so the smallest score-improving change (without altering your model or training loop) is to calibrate the 5-class outputs using the training label distribution. I keep your DenseNet121 + single-head inference exactly the same, but replace the raw `argmax` with a rank-based mapping that assigns predicted classes so the overall class counts match the empirical train priors (a common QWK post-processing that improves agreement for ordinal labels). This is purely post-processing on the same logits, doesn’t change architecture/loss/training, and typically moves QWK up substantially from near-random outputs. I also batch CSV writing to avoid per-row file opens (no semantic change) and ensure predictions are in {0..4}.'

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
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    from tensorboardX import SummaryWriter  # noqa: F401
except ModuleNotFoundError:
    SummaryWriter = None  # not used in this inference-only notebook/script

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)



## === cell 1
name_file = "/kaggle/input/aptos2019-blindness-detection/test.csv"
with open(name_file, "r", newline="") as f:
    csv_file = csv.reader(f)
    content = []
    for line in csv_file:
        content.append(line[0] + ".png")
content = content[1:]  # drop header



## === cell 2
import torch
from torch import nn
import torchvision


class Baseline_single(nn.Module):
    def __init__(self, num_classes, loss_type="single BCE", **kwargs):
        super(Baseline_single, self).__init__()
        self.loss_type = loss_type

        backbone = torchvision.models.densenet121(pretrained=False)

        self.base = backbone.features
        self.feature_dim = backbone.classifier.in_features  # 1024 for densenet121

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
            ys = self.classifiers(x)
        return ys




## === cell 3
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
        else:
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
    def __init__(self, txt_path, transform=None):
        self.imgs = list(txt_path)
        self.transform = transform

    def __getitem__(self, index):
        fn = self.imgs[index]
        img_path = "/kaggle/input/aptos2019-blindness-detection/test_images/" + fn
        img = cv_imread(img_path)
        img = load_ben_yuan(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, fn[:-4]

    def __len__(self):
        return len(self.imgs)




## === cell 4
def _find_model_path():
    candidates = [
        "/kaggle/input/temp-file/model_yuan512_dense121_00001_adam_max.pkl",
        "/kaggle/input/temp-file/model_yuan512_dense121_00001_adam_max.pth",
        "/kaggle/input/model_yuan512_dense121_00001_adam_max.pkl",
        "/kaggle/input/model_yuan512_dense121_00001_adam_max.pth",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p

    target_names = [
        "model_yuan512_dense121_00001_adam_max.pkl",
        "model_yuan512_dense121_00001_adam_max.pth",
    ]
    for root, _, files in os.walk("/kaggle/input"):
        for tn in target_names:
            if tn in files:
                return os.path.join(root, tn)

    return None


def _extract_state_dict(obj):
    if isinstance(obj, dict):
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return obj["state_dict"]
        if "model_state_dict" in obj and isinstance(obj["model_state_dict"], dict):
            return obj["model_state_dict"]
    return obj  # assume raw state_dict


def _strip_module_prefix(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict
    if not any(k.startswith("module.") for k in state_dict.keys()):
        return state_dict
    return {k.replace("module.", "", 1): v for k, v in state_dict.items()}


def _try_load_weights_or_fallback_imagenet(net, use_gpu):
    model_path = _find_model_path()
    if model_path is not None:
        state = torch.load(model_path, map_location="cuda" if use_gpu else "cpu")
        state = _extract_state_dict(state)
        state = _strip_module_prefix(state)
        missing, unexpected = net.load_state_dict(state, strict=False)
        print(f"Loaded checkpoint: {model_path}")
        if missing:
            print(f"Missing keys (showing up to 20): {missing[:20]}")
        if unexpected:
            print(f"Unexpected keys (showing up to 20): {unexpected[:20]}")
        return "custom_checkpoint"

    print(
        "WARNING: Custom model weights not found under /kaggle/input. "
        "Falling back to torchvision DenseNet121 ImageNet weights for the backbone."
    )
    try:
        weights = torchvision.models.DenseNet121_Weights.DEFAULT
        pretrained = torchvision.models.densenet121(weights=weights)
    except Exception:
        pretrained = torchvision.models.densenet121(pretrained=True)

    net.base.load_state_dict(pretrained.features.state_dict(), strict=True)
    return "imagenet_fallback"


def _rank_map_to_train_priors(scores_1d, train_labels, n_classes=5):
    scores_1d = np.asarray(scores_1d, dtype=np.float64)
    train_labels = np.asarray(train_labels, dtype=np.int64)
    n = scores_1d.shape[0]

    counts = np.bincount(train_labels, minlength=n_classes).astype(np.int64)
    if counts.sum() != len(train_labels):
        counts[-1] += len(train_labels) - counts.sum()

    proportions = counts / counts.sum()
    target = np.floor(proportions * n).astype(np.int64)
    remainder = n - target.sum()
    if remainder > 0:
        frac = (proportions * n) - np.floor(proportions * n)
        order = np.argsort(-frac)  # largest fractional parts first
        for i in range(remainder):
            target[order[i % n_classes]] += 1
    elif remainder < 0:
        frac = (proportions * n) - np.floor(proportions * n)
        order = np.argsort(frac)  # smallest fractional parts first
        for i in range(-remainder):
            for j in order:
                if target[j] > 0:
                    target[j] -= 1
                    break

    order = np.argsort(scores_1d)  # ascending
    y = np.empty(n, dtype=np.int64)
    start = 0
    for cls in range(n_classes):
        end = start + int(target[cls])
        idx = order[start:end]
        y[idx] = cls
        start = end

    if start != n:
        y[order[start:]] = min(2, n_classes - 1)

    return y


if __name__ == "__main__":
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(SEED)
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
    content = content[1:]  # drop header

    test_data = eye_dataset(content, transform2)
    net = Baseline_single(num_classes=5)
    if use_gpu:
        net = net.cuda()

    _ = _try_load_weights_or_fallback_imagenet(net, use_gpu)

    dataloader_test = DataLoader(
        test_data, batch_size=8, shuffle=False, num_workers=2, pin_memory=use_gpu
    )

    train_df = pd.read_csv("/kaggle/input/aptos2019-blindness-detection/train.csv")
    train_labels = train_df["diagnosis"].values

    ids = []
    severity_scores = []

    with torch.no_grad():
        net.eval()
        for _, item in tqdm(enumerate(dataloader_test), total=len(dataloader_test)):
            data, name = item
            if use_gpu:
                data = data.cuda(non_blocking=True)
            out = net(data)  # [B,5] logits
            prob = torch.softmax(out, dim=1)
            classes = torch.arange(5, device=prob.device, dtype=prob.dtype).view(1, -1)
            exp_sev = (prob * classes).sum(dim=1)  # [B]
            severity_scores.extend(exp_sev.detach().float().cpu().numpy().tolist())
            ids.extend([str(n) for n in name])

    mapped = _rank_map_to_train_priors(severity_scores, train_labels, n_classes=5)

    sub_path = "/kaggle/working/submission.csv"
    sub_df = pd.DataFrame({"id_code": ids, "diagnosis": mapped.astype(int)})
    sub_df.to_csv(sub_path, index=False)

    sub_df_check = pd.read_csv(sub_path)
    assert list(sub_df_check.columns) == ["id_code", "diagnosis"]
    assert len(sub_df_check) == len(
        test_data
    ), f"Submission rows {len(sub_df_check)} != test rows {len(test_data)}"
    assert sub_df_check["diagnosis"].between(0, 4).all()
    print(f"Wrote {sub_path} with shape {sub_df_check.shape}")
