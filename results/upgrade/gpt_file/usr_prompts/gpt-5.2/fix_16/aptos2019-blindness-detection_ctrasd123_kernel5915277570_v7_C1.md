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

0.8406780153978346

# 6. Current score

-0.01991

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -0.12777) has done: 'I remove the TensorBoard import path that is crashing due to an incompatibility in this Kaggle image, since it isn’t needed for inference. Then I fix the missing checkpoint failure by falling back to a torchvision DenseNet-121 ImageNet pretrained backbone when the provided `.pkl` is unavailable, keeping the same model class and inference flow so the script can still run end-to-end. I also correct the DenseNet feature dimension (1024, not 512*2) so the forward pass matches the actual backbone output, which is a runtime shape bug. Finally, I keep submission formatting identical (`id_code,diagnosis`) and ensure the output file is written to `/kaggle/working/submission.csv`.'
- What this solution (achieved -0.01991) has done: 'Your current negative kappa is very likely coming from a color-channel mismatch in preprocessing: the pipeline converts BGR→RGB but then uses `cv2.COLOR_RGB2GRAY` inside the crop, which is inconsistent and can badly distort inputs (especially since OpenCV reads as BGR). I fix this with a minimal change: keep everything in BGR until after cropping, and use `cv2.COLOR_BGR2GRAY` for the gray mask, preserving the exact model/inference flow and submission format. I also add standard ImageNet normalization (still just a transform change) because you’re often falling back to ImageNet-pretrained DenseNet121, and unnormalized inputs typically tank performance. These are small, directly relevant changes that should move the score upward toward your target without changing architecture or training logic.'
- What this solution (achieved 0.0) has done: 'Your score is far below the target (gap ≈ -0.8608), so we should improve performance, not tune for “best possible.” The biggest issue is that your model head is a sigmoid-style multi-label BCE head, but inference is using `argmax` on raw logits; for ordinal DR this often yields badly miscalibrated class picks and very low kappa. With minimal change and identical architecture/inference flow, we (1) apply the missing sigmoid at inference and (2) convert the 5 sigmoid outputs to a single ordinal class by summing “is grade ≥ k” probabilities (standard ordinal trick for this exact head style). Everything else (DenseNet121 backbone, checkpoint fallback, preprocessing, submission format/path) stays the same.'
- What this solution (achieved -0.01991) has done: 'Your current 0.0 kappa strongly suggests the ordinal post-processing is still mismatched to what this BCE-style head actually learned: summing sigmoids and flooring tends to collapse predictions and can be anti-correlated. To move the score upward toward your target with minimal risk and without changing the model/training logic, I keep the exact network and preprocessing, but change only the inference mapping from the 5 sigmoid outputs to an ordinal class using the standard cumulative-link rule: count how many “grade ≥ k” probabilities exceed 0.5. I also add a tiny safety fallback: if the checkpoint is missing (ImageNet DenseNet), use argmax on logits instead of the ordinal BCE mapping, since the pretrained head is random and the BCE mapping becomes meaningless. Submission format/path stays identical and the script still runs end-to-end.'
- What this solution (achieved -0.01991) has done: 'Your current score is far below the target (gap ≈ -0.8608), so we should improve performance with the smallest inference-only fix. The model is a 5-logit “single BCE” head, where each logit is meant to represent an ordinal threshold (“diagnosis ≥ k”); the current mapping incorrectly drops the first logit and uses a hard 0.5 cutoff, which often collapses predictions and hurts QWK. I keep the exact same model and preprocessing, but change only the post-processing to a standard ordinal mapping: apply sigmoid to all 5 outputs and sum all thresholds above a tuned cutoff (0.35 is a common better-than-0.5 default when no validation is available). The submission format/path and fallback behavior remain unchanged.'
- What this solution (achieved -0.01991) has done: 'Your current score is far below the target (gap ≈ -0.8608), so we should make the smallest inference-only changes that plausibly increase QWK without altering the model architecture or any training logic. The biggest likely remaining issue is ordinal decoding: your 5-logit “single BCE” head should be interpreted as cumulative thresholds, and a fixed cutoff (0.35) without any calibration can still yield poorly distributed predictions. I keep the exact model and preprocessing, but add a lightweight, deterministic test-time calibration that (a) uses per-class bias from the test-set predicted probability distribution to avoid collapsing to one class and (b) then applies the same cumulative-threshold counting rule. This stays within Kaggle constraints, preserves core semantics (same network, same forward pass), and should move the score upward toward your target.'
- What this solution (achieved -0.01991) has done: 'Your score is far below the target (gap ≈ -0.8608), so we should increase performance with the smallest inference-only adjustments. The current “test-time calibration” forces all five threshold logits toward the same mean probability, which breaks the cumulative/monotonic nature of ordinal thresholds and can easily yield anti-correlated grades (hurting QWK). I remove that calibration and replace it with a monotonic cumulative-link decoding: enforce non-increasing “P(grade ≥ k)” across k and then count how many exceed a single threshold (keeps the same model and sigmoid head, only changes post-processing). I also set `cudnn.deterministic=True` to stabilize outputs across runs without changing the modeling approach, and keep the submission format/path identical.'
- What this solution (achieved -0.01991) has done: 'Your score is far below the target (gap ≈ -0.8608), so we should make the smallest inference-only change that plausibly increases QWK without altering the model architecture or any training logic. The biggest remaining likely issue is the ordinal decoding: the current code enforces monotonicity in the wrong direction (`np.minimum.accumulate` makes later thresholds *smaller*, but for “grade ≥ k” probabilities they should be non-increasing with k), which can collapse the threshold counts and harm kappa. I switch that to a correct non-increasing enforcement using `np.maximum.accumulate` on the reversed axis, keeping the same sigmoid, same single cutoff, same checkpoint fallback behavior, and the same submission format/path. This is a tiny, directly relevant fix that should move predictions in the right direction toward the target score.'
- What this solution (achieved -0.01991) has done: 'Your negative QWK strongly suggests the current ordinal decoding is inconsistent with how a 5-logit “single BCE” ordinal head is typically trained (it usually has 4 thresholds for classes 0–4, not 5), so your `sum(prob > t)` mapping is likely shifted and collapsing predictions. I keep the exact same model, checkpoint loading, and preprocessing, but change only the inference post-processing when the checkpoint exists: use the standard “4-threshold” cumulative rule by ignoring the last logit (use logits 0..3), enforce proper monotonicity across thresholds, and then count thresholds exceeded. This is a minimal, inference-only fix that should move the score upward toward your target without altering architecture or training logic. Submission format/path stays identical.'
- What this solution (achieved -0.01991) has done: 'Your current score is far below the target, so we should improve performance with the smallest inference-only changes. The biggest likely issue is that the ordinal decoding is being applied on 4 thresholds but the model outputs 5 logits; for common ordinal “single BCE” setups on 5 classes, you typically decode using the first 4 logits as thresholds. I keep the same network and preprocessing, but adjust decoding to use only the first 4 logits, enforce proper monotonicity, and replace the fixed cutoff with a deterministic “match training prior” cutoff computed from the train label distribution (no label leakage from test). This keeps architecture/training untouched, changes only post-processing, and should move QWK upward toward your target while remaining stable and Kaggle-valid.'
- What this solution (achieved -0.01991) has done: 'Your current QWK is far below target, so we should make the smallest inference-only fixes that reduce the most likely source of anti-correlation. The biggest issue left is ordinal decoding: for “single BCE” ordinal heads, probabilities should be monotone **non-increasing** across thresholds, and your current enforcement makes them non-decreasing, which can invert grades and tank kappa. I fix monotonic enforcement (one-line logic change) and keep the same 4-threshold decoding, same checkpoint logic, same transforms, and the same submission writing. This keeps architecture/training untouched and should move the score upward toward the target.'
- What this solution (achieved -0.01991) has done: 'I make one minimal, score-relevant fix in the ordinal decoding: the monotonicity enforcement for “P(grade ≥ k)” thresholds is currently in the wrong direction, which can collapse/invert grades and drive QWK negative. We replace `np.minimum.accumulate(prob4, axis=1)` with the correct non-increasing enforcement across thresholds using a reverse-accumulate-reverse trick, keeping the same model, transforms, checkpoint logic, and threshold search against the train mean. This is an inference-only change that preserves core architecture/training semantics and should move the score upward toward your target. The submission writing and paths remain identical.'
- What this solution (achieved -0.01991) has done: 'Your score is far below the target (gap ≈ -0.8608), so we should increase performance with the smallest inference-only fix. The current “monotonic” enforcement uses `maximum.accumulate` on the reversed thresholds, which actually makes `P(grade ≥ k)` non-decreasing with k (the opposite of what ordinal thresholds require), and that can invert/collapse predictions and drive QWK negative. I change this to the correct non-increasing enforcement using a reverse-`minimum.accumulate`-reverse trick, keeping the same model, preprocessing, checkpoint logic, and threshold-to-match-train-mean search. Everything else (paths, submission format, no training, same architecture) remains unchanged.'
- What this solution (achieved -0.01991) has done: 'Your current kappa is far below the target, so we should make the smallest inference-only change that plausibly improves alignment with the QWK metric without touching architecture or training. The biggest remaining likely issue is that you’re decoding an ordinal-threshold head using a train-mean heuristic, which does not optimize QWK and can easily yield anti-correlated class assignments. I keep the same model, checkpoint loading, preprocessing, and monotonic ordinal probabilities, but tune the single threshold using a small train/validation split and directly maximize quadratic weighted kappa on that validation set (no test leakage). Then we use that chosen threshold for test inference and write the same `submission.csv` format/path.'

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
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import cohen_kappa_score


class SummaryWriter:  # no-op stub to preserve core logic if referenced
    def __init__(self, *args, **kwargs):
        pass

    def add_scalar(self, *args, **kwargs):
        pass

    def add_image(self, *args, **kwargs):
        pass

    def close(self):
        pass


DATA_ROOT = "/kaggle/input/aptos2019-blindness-detection"
TEST_CSV = osp.join(DATA_ROOT, "test.csv")
TEST_IMG_DIR = osp.join(DATA_ROOT, "test_images")
TRAIN_CSV = osp.join(DATA_ROOT, "train.csv")
TRAIN_IMG_DIR = osp.join(DATA_ROOT, "train_images")
OUT_SUB = "/kaggle/working/submission.csv"

CKPT_PATH = "/kaggle/input/temp-file/model_yuan512_dense121_00001_adam_avg_1.pkl"

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

cudnn.deterministic = True
cudnn.benchmark = False



## === cell 1
content = []
with open(TEST_CSV, "r", newline="") as f:
    reader = csv.reader(f)
    header = next(reader, None)
    for row in reader:
        if not row:
            continue
        content.append(row[0] + ".png")

print(f"Loaded {len(content)} test image names from {TEST_CSV}")




## === cell 2
class Baseline_single(nn.Module):
    def __init__(self, num_classes, loss_type="single BCE", pretrained=False, **kwargs):
        super(Baseline_single, self).__init__()
        self.loss_type = loss_type

        densenet121 = torchvision.models.densenet121(pretrained=pretrained)
        self.base = nn.Sequential(*list(densenet121.children())[:-1])

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
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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
    image = crop_image_from_gray(image)
    image = cv2.resize(image, (512, 512))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


class eye_dataset(Dataset):
    def __init__(self, txt_path, transform=None):
        self.imgs = list(txt_path)
        self.transform = transform

    def __getitem__(self, index):
        fn = self.imgs[index]
        img_path = osp.join(TEST_IMG_DIR, fn)
        img = cv_imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {img_path}")
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

    def __getitem__(self, index):
        row = self.df.iloc[index]
        fn = str(row["id_code"]) + ".png"
        y = int(row["diagnosis"])
        img_path = osp.join(TRAIN_IMG_DIR, fn)
        img = cv_imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {img_path}")
        img = load_ben_yuan(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, y

    def __len__(self):
        return len(self.df)




## === cell 4
def _ordinal_prob4_from_logits(out_tensor: torch.Tensor) -> np.ndarray:
    """
    Keeps existing core semantics:
    - Apply sigmoid to logits.
    - Use first 4 logits as thresholds for 5 classes.
    - Enforce monotone non-increasing P(y >= k) over k via reverse-min-accumulate-reverse.
    """
    prob = torch.sigmoid(out_tensor).detach().cpu().numpy()  # (B,5)
    prob4 = prob[:, :4]  # (B,4)
    prob4_mono = np.minimum.accumulate(prob4[:, ::-1], axis=1)[:, ::-1]
    return prob4_mono


def _tune_threshold_on_val(net, use_gpu: bool, transform2) -> float:
    """
    Score-relevant minimal change:
    Tune the single cutoff t by maximizing QWK on a small validation split from train.csv.
    This directly targets the competition metric and avoids test leakage.
    """
    df = pd.read_csv(TRAIN_CSV)
    y = df["diagnosis"].astype(int).values

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=0)
    _, val_idx = next(splitter.split(np.zeros(len(y)), y))
    df_val = df.iloc[val_idx].reset_index(drop=True)

    val_ds = eye_dataset_train(df_val, transform=transform2)
    val_loader = DataLoader(
        val_ds, batch_size=8, shuffle=False, num_workers=2, pin_memory=use_gpu
    )

    all_prob4 = []
    all_y = []
    with torch.no_grad():
        net.eval()
        for data, yb in tqdm(val_loader, total=len(val_loader), desc="Val infer"):
            if use_gpu:
                data = data.cuda(non_blocking=True)
            out = net(data)
            prob4 = _ordinal_prob4_from_logits(out)
            all_prob4.append(prob4)
            all_y.extend([int(v) for v in yb])

    all_prob4 = np.concatenate(all_prob4, axis=0)
    all_y = np.asarray(all_y, dtype=np.int64)

    t_grid = np.linspace(0.05, 0.95, 91)  # step 0.01
    best_t = 0.35
    best_kappa = -1e9
    for t in t_grid:
        pred = (all_prob4 > t).sum(axis=1).astype(np.int64)
        pred = np.clip(pred, 0, 4)
        kappa = cohen_kappa_score(all_y, pred, weights="quadratic")
        if kappa > best_kappa:
            best_kappa = float(kappa)
            best_t = float(t)

    print(
        f"Chosen threshold by val-QWK: t={best_t:.2f}, val_qwk={best_kappa:.5f}, n_val={len(all_y)}"
    )
    return best_t


def main():
    use_gpu = torch.cuda.is_available()
    if not use_gpu:
        print("Currently using CPU (GPU is highly recommended)")

    transform2 = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    test_data = eye_dataset(content, transform2)

    ckpt_exists = osp.exists(CKPT_PATH)
    net = Baseline_single(num_classes=5, pretrained=(not ckpt_exists))

    if use_gpu:
        net = net.cuda()

    if ckpt_exists:
        state = torch.load(CKPT_PATH, map_location=("cuda" if use_gpu else "cpu"))
        if isinstance(state, dict) and all(isinstance(k, str) for k in state.keys()):
            if any(k.startswith(("base.", "classifiers.")) for k in state.keys()):
                net.load_state_dict(state, strict=True)
            elif "state_dict" in state:
                net.load_state_dict(state["state_dict"], strict=True)
            elif "model" in state and isinstance(state["model"], dict):
                net.load_state_dict(state["model"], strict=True)
            else:
                net.load_state_dict(state, strict=False)
        else:
            net.load_state_dict(state, strict=False)
        print(f"Loaded checkpoint: {CKPT_PATH}")
    else:
        print(
            f"Checkpoint not found at {CKPT_PATH}; using torchvision ImageNet pretrained DenseNet121."
        )

    dataloader_test = DataLoader(
        test_data, batch_size=8, shuffle=False, num_workers=2, pin_memory=use_gpu
    )

    ids = []
    preds = []

    with torch.no_grad():
        net.eval()

        if ckpt_exists:
            best_t = _tune_threshold_on_val(net, use_gpu, transform2)

            all_prob4 = []
            all_names = []
            for _, item in tqdm(
                enumerate(dataloader_test),
                total=len(dataloader_test),
                desc="Test infer",
            ):
                data, name = item
                if use_gpu:
                    data = data.cuda(non_blocking=True)

                out = net(data)
                prob4_mono = _ordinal_prob4_from_logits(out)
                all_prob4.append(prob4_mono)
                all_names.extend([str(n) for n in name])

            all_prob4 = np.concatenate(all_prob4, axis=0)
            pred = (all_prob4 > best_t).sum(axis=1).astype(np.int64)
            pred = np.clip(pred, 0, 4)

            ids = all_names
            preds = [int(x) for x in pred]
            print(f"Test mean_pred={float(np.mean(preds)):.4f} using t={best_t:.2f}")

        else:
            for _, item in tqdm(
                enumerate(dataloader_test),
                total=len(dataloader_test),
                desc="Test infer",
            ):
                data, name = item
                if use_gpu:
                    data = data.cuda(non_blocking=True)
                out = net(data)
                predicted = out.argmax(dim=1).long().cpu().numpy()
                predicted = np.clip(predicted, 0, 4)
                ids.extend([str(n) for n in name])
                preds.extend([int(x) for x in predicted])

    with open(OUT_SUB, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id_code", "diagnosis"])
        for i, p in zip(ids, preds):
            writer.writerow([i, int(p)])

    print(f"Wrote submission to: {OUT_SUB}")
    sub = pd.read_csv(OUT_SUB)
    print(sub.head())
    print(sub.shape)


if __name__ == "__main__":
    main()
