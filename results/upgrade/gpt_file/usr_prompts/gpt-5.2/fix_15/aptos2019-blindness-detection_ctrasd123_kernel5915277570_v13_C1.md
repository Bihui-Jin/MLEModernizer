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

0.5402456897607164

# 6. Current score

0.78827

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -0.00728) has done: 'I remove the hard dependency on `tensorboardX` (it isn’t installed) by falling back to PyTorch’s built-in TensorBoard writer (or a no-op) so imports don’t crash. Then I fix the model definition to correctly use DenseNet121 features (your current `children()[:-1]` + `feature_dim` mismatch break loading/inference), and I make weight loading robust: if the referenced `.pkl` file isn’t present in `/kaggle/input`, the script still run end-to-end and write a valid `submission.csv` (with deterministic, reasonable default predictions). Finally, I remove excessive per-line printing and ensure paths use `/kaggle/input/aptos2019-blindness-detection/...` consistently so the pipeline finishes within the time limit and produces a correctly formatted submission.'
- What this solution (achieved -0.00351) has done: 'I fix the import crash coming from TensorBoard by preventing any TensorBoard import/use entirely (it’s not needed for inference) and keeping a safe no-op `SummaryWriter`. Then I make the test-time preprocessing match DenseNet’s expected input distribution by resizing to 224 and applying ImageNet normalization; this is a minimal change that typically moves kappa from near-random toward a reasonable baseline without altering the model architecture or inference loop. I also make the test image path resolution robust to the two common dataset directory layouts so missing images don’t silently zero out the input. Finally, I keep the submission writing logic but ensure deterministic execution and correct row alignment/length checks.'
- What this solution (achieved -0.00853) has done: 'Your current score is far below the target (higher-is-better), so we should make a minimal change that legitimately boosts kappa without changing the model or training loop. The biggest issue is that you’re using a randomly initialized DenseNet because the external weight file isn’t present; that produces near-random predictions and negative kappa. I keep your exact architecture and inference logic, but load standard ImageNet-pretrained DenseNet121 weights (available inside torchvision) into the DenseNet feature extractor when the competition-specific `.pkl` isn’t found. This should move predictions from random toward a reasonable baseline and thus increase kappa substantially while preserving your pipeline and submission format.'
- What this solution (achieved 0.27916) has done: 'Your current kappa is far below the target, so we should make a minimal, metric-aligned change that improves predictions without altering the model/training logic. The biggest issue is that the model outputs 5 logits but you’re taking `argmax` directly, which is not consistent with your declared “single BCE” multi-label style head; a small, standard fix is to apply `sigmoid` at inference and convert the 5 binary outputs into a single ordinal class using a count-based rule. This preserves the architecture and loss semantics, but makes post-processing consistent and usually yields a large kappa jump from near-random toward a reasonable baseline. I also make the output layer initialization consistent with the original intent by using the already-defined `self.sigmoid` in forward (still same layers), and I keep submission alignment checks unchanged.'
- What this solution (achieved 0.23679) has done: 'Your current score (0.27916) is far below the target (0.5402), so we should make a small, metric-aligned inference change that tends to increase QWK without changing your model/training core. The main weakness is the fixed `0.5` threshold for converting sigmoid outputs into an ordinal class; for this competition, calibrating thresholds on the training set (using out-of-fold-style validation) usually improves kappa substantially while preserving the same architecture and “single BCE” semantics. I keep your network exactly the same, but add a minimal train/val split to learn 4 thresholds (0–4) that maximize QWK on the validation set, then apply them to test predictions. If the competition checkpoint is missing and we fall back to ImageNet features, threshold calibration still helps; if calibration fails for any reason, it falls back to your existing `>0.5` counting rule so you still get a valid submission.'
- What this solution (achieved 0.2424) has done: 'Your current score is far below the target (higher-is-better), so the smallest legitimate move toward the target is to improve inference calibration without changing the model or training loop. I keep your architecture, weights logic, and ordinal-from-sigmoid setup, but tune the 4 thresholds in a slightly more reliable way: use stratified K-fold out-of-fold predictions (instead of a single split) to fit thresholds, which reduces variance and usually increases QWK. To avoid altering “core logic”, the network is never trained; we only evaluate it on multiple validation folds and optimize thresholds on those out-of-fold scores, then apply the tuned thresholds to the test set. I also make the tuned thresholds the default only when tuning succeeds; otherwise your original 0.5-count rule remains unchanged.'
- What this solution (achieved 0.34973) has done: 'Your score is far below the target (higher-is-better), so we should make a small, metric-aligned inference improvement without changing the model architecture or training loop. The main weakness is that we collapse the 5 sigmoid outputs by summing them and then applying thresholds; a minimal, consistent alternative for “single BCE” ordinal heads is to use the probability of each “is at least k” output directly with four per-output thresholds. I keep the same OOF threshold-tuning approach, but tune four sigmoid thresholds (one per ordinal output) to maximize OOF QWK, then convert to class by counting how many outputs exceed their own thresholds. Everything else (paths, preprocessing, weight-loading fallback, batching, CSV writing) stays the same.'
- What this solution (achieved 0.25564) has done: 'Your current score (0.34973) is well below the target (0.54025), so we should make a small, metric-aligned inference change that usually improves QWK without changing your model, loss semantics, or any training loop. The biggest remaining gap is that you’re using a multi-label ordinal head (5 sigmoid outputs) but only tuning 4 per-output thresholds and ignoring the 5th logit; instead, we use all 5 outputs as “is at least k” (k=0..4), enforce the always-true first threshold implicitly, and tune only thresholds for k=1..4, which better matches the intended ordinal encoding. Concretely: convert probabilities to a continuous “severity score” as the sum of the 5 sigmoid outputs, tune 4 cutpoints on that score via OOF QWK, and then apply those cutpoints to test; this keeps the same network and post-processing style (thresholding) but aligns it more consistently. Everything else (paths, preprocessing, weight-loading fallback, batching, CSV writing) stays the same and still produces a valid `submission.csv`.'
- What this solution (achieved 0.25564) has done: 'To move your QWK up toward the 0.54 target with minimal core-logic changes, I keep the exact model and inference flow but fix two calibration issues that commonly suppress kappa: (1) make the ordinal “sum of sigmoids” score monotonic by sorting the 5 probabilities before summing, and (2) tune cutpoints on an ordinal-appropriate score using out-of-fold predictions, then apply those cutpoints at test time. This preserves your architecture, loss semantics (sigmoid+BCE-style head), and “thresholding” post-processing, but makes the threshold tuning more stable/metric-aligned. I also ensure the tuned cutpoints are constrained to a sensible range for a 0–4 label to avoid pathological thresholds that can hurt QWK.'
- What this solution (achieved 0.25247) has done: 'We need to move your QWK up toward the 0.540 target with minimal changes, and your current 0.255 suggests the main issue is calibration/post-processing rather than submission validity. I keep the exact model and inference flow, but improve the cutpoint tuning so it directly optimizes the *same* prediction rule used at test time (i.e., optimize cutpoints on the continuous score and then apply them), using a small, robust 1D coordinate descent with progressively finer grids. I also remove the “sort probs then sum” step (sorting doesn’t change the sum and can introduce unnecessary quirks), and I tune cutpoints on the unsorted sum-of-sigmoids score with safe constraints and deterministic behavior. Everything else (paths, transforms, weights fallback, dataloading, CSV writing) stays the same.'
- What this solution (achieved 0.21861) has done: 'Your current score (0.25247) is far below the target (0.54025), so we should make small, metric-aligned inference fixes rather than change the model/training. The biggest improvement with minimal disruption is to (1) use the DenseNet121 ImageNet weights as a fallback for the *full* backbone (not just features), and (2) fix the ordinal post-processing to use only the 4 meaningful “is grade ≥ k” outputs (k=1..4) instead of summing all 5, which often miscalibrates the continuous score. Then we tune 4 cutpoints on the resulting continuous ordinal score using OOF predictions and apply them at test time; if tuning fails, we keep the existing safe fallback to produce a valid submission. These changes keep the same architecture and inference loop, but make the score and thresholding consistent with the competition’s ordinal nature, which should move QWK upward toward your target.'
- What this solution (achieved 0.14805) has done: 'Your current score is far below the target (higher-is-better), so we make a minimal, metric-aligned inference improvement without changing the model architecture or any training. The main issue is that the fallback backbone is ImageNet-only and your ordinal head is untrained, so threshold tuning on those raw outputs is unstable; instead we calibrate the *continuous severity score* using the training labels via a simple, deterministic class-conditional mapping (fit 4 cutpoints from the score distributions), which is consistent with QWK’s ordinal nature. We keep your existing OOF tuning as the first choice when the competition checkpoint exists, but when it’s missing we switch to this more stable calibration that typically boosts kappa from the low 0.2s toward a reasonable baseline. All paths, preprocessing, dataloading, and submission writing remain the same and the script still produces a valid `submission.csv`.'
- What this solution (achieved 0.78827) has done: 'The timeout is dominated by the OOF cutpoint tuning path: it runs 5 full validation passes over the entire training set, repeatedly decoding/cropping/resizing images in Python, which is far too slow for a 600s budget. The biggest safe win is to avoid recomputing image preprocessing during OOF by caching the preprocessed 224×224 RGB arrays to disk once (losslessly), then having all datasets read from that cache; this preserves identical model inputs and thus accuracy. In addition, we increase DataLoader throughput (more workers, persistent workers, prefetching, pinned memory) and remove per-batch file open/close in submission writing (buffer rows and write once), all of which are correctness-preserving.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os
import os.path as osp
import sys
import time
import datetime
import argparse
import random
from PIL import Image
import cv2
import csv

import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score


class SummaryWriter:  # minimal no-op fallback
    def __init__(self, *args, **kwargs):
        pass

    def add_scalar(self, *args, **kwargs):
        pass

    def add_histogram(self, *args, **kwargs):
        pass

    def close(self):
        pass


def seed_everything(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_everything(0)

if torch.cuda.is_available():
    cudnn.benchmark = True


## === cell 1
name_file = "/kaggle/input/aptos2019-blindness-detection/test.csv"
test_df = pd.read_csv(name_file)
content = (test_df["id_code"].astype(str) + ".png").tolist()
print(f"Loaded test ids: {len(content)} images")




## === cell 2
class Baseline_single(nn.Module):
    def __init__(self, num_classes, loss_type="single BCE", **kwargs):
        super(Baseline_single, self).__init__()
        self.loss_type = loss_type

        densenet = torchvision.models.densenet121(weights=None)
        self.base = densenet.features  # (B, 1024, H, W)
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
        x = nn.functional.relu(x, inplace=True)
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
        img = np.stack([img1, img2, img3], axis=-1)
        return img
    return img


def load_ben_yuan(image, sigmaX=10):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_image_from_gray(image)
    image = cv2.resize(image, (224, 224))
    return image


def _cached_preprocessed_rgb_u8(img_path, cache_path):
    if osp.exists(cache_path):
        arr = np.load(cache_path, allow_pickle=False)
        if (
            isinstance(arr, np.ndarray)
            and arr.shape == (224, 224, 3)
            and arr.dtype == np.uint8
        ):
            return arr
    img = cv2.imread(img_path)
    if img is None:
        arr = np.zeros((224, 224, 3), dtype=np.uint8)
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    else:
        arr = load_ben_yuan(img)
        if arr is None or arr.size == 0:
            arr = np.zeros((224, 224, 3), dtype=np.uint8)
    os.makedirs(osp.dirname(cache_path), exist_ok=True)
    np.save(cache_path, arr, allow_pickle=False)
    return arr


def _build_cache_for_ids(ids_png, img_dir, cache_dir, desc, max_workers=0):
    os.makedirs(cache_dir, exist_ok=True)
    t0 = time.time()
    n = 0
    for fn in tqdm(ids_png, desc=desc, total=len(ids_png)):
        img_path = osp.join(img_dir, fn)
        cache_path = osp.join(cache_dir, fn[:-4] + ".npy")
        if not osp.exists(cache_path):
            _cached_preprocessed_rgb_u8(img_path, cache_path)
        n += 1
    print(
        f"Cache ready: {desc}: {n} files, elapsed {time.time()-t0:.1f}s at {cache_dir}"
    )


class eye_dataset(Dataset):
    def __init__(self, txt_path, transform=None, cache_dir=None):
        self.imgs = list(txt_path)
        self.transform = transform

        cand_dirs = [
            "/kaggle/input/aptos2019-blindness-detection/test_images",
            "/kaggle/input/test_images",
            "/kaggle/input/aptos2019-blindness-detection/aptos2019-blindness-detection/test_images",
        ]
        self.test_img_dir = next((d for d in cand_dirs if osp.isdir(d)), cand_dirs[0])
        self.cache_dir = cache_dir

    def __getitem__(self, index):
        fn = self.imgs[index]
        if self.cache_dir is not None:
            cache_path = osp.join(self.cache_dir, fn[:-4] + ".npy")
            img = _cached_preprocessed_rgb_u8(
                osp.join(self.test_img_dir, fn), cache_path
            )
        else:
            img_path = osp.join(self.test_img_dir, fn)
            img = cv2.imread(img_path)
            if img is None:
                img = np.zeros((224, 224, 3), dtype=np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img = load_ben_yuan(img)

        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, fn[:-4]

    def __len__(self):
        return len(self.imgs)


class eye_train_dataset(Dataset):
    def __init__(self, df, img_dir, transform=None, cache_dir=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.cache_dir = cache_dir

    def __getitem__(self, index):
        row = self.df.iloc[index]
        fn = str(row["id_code"]) + ".png"
        label = int(row["diagnosis"])

        if self.cache_dir is not None:
            cache_path = osp.join(self.cache_dir, str(row["id_code"]) + ".npy")
            img = _cached_preprocessed_rgb_u8(osp.join(self.img_dir, fn), cache_path)
        else:
            img_path = osp.join(self.img_dir, fn)
            img = cv2.imread(img_path)
            if img is None:
                img = np.zeros((224, 224, 3), dtype=np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img = load_ben_yuan(img)

        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.df)


def _predict_sigmoid_prob(net, loader, use_gpu):
    net.eval()
    all_prob = []
    all_y = []
    with torch.no_grad():
        for data, y in loader:
            if use_gpu:
                data = data.cuda(non_blocking=True)
            out = net(data)
            prob = torch.sigmoid(out).detach().cpu().numpy()
            all_prob.append(prob)
            all_y.append(np.asarray(y, dtype=np.int64))
    return np.concatenate(all_prob, axis=0), np.concatenate(all_y, axis=0)


def _score_from_prob(prob5):
    prob5 = np.asarray(prob5, dtype=np.float32)
    if prob5.ndim != 2 or prob5.shape[1] != 5:
        raise ValueError("Expected prob with shape (N, 5)")
    return prob5[:, 1:].sum(axis=1).astype(np.float32)


def _apply_cutpoints(score, cuts4):
    cuts = np.asarray(cuts4, dtype=np.float32).reshape(-1)
    if cuts.shape[0] != 4:
        raise ValueError("cuts4 must have length 4")
    return (
        (score > cuts[0]).astype(np.int64)
        + (score > cuts[1]).astype(np.int64)
        + (score > cuts[2]).astype(np.int64)
        + (score > cuts[3]).astype(np.int64)
    )


def _tune_cutpoints_coorddesc(y_true, score, init_cuts=None):
    y_true = np.asarray(y_true, dtype=np.int64)
    score = np.asarray(score, dtype=np.float32)

    if init_cuts is None:
        init_cuts = [0.5, 1.5, 2.5, 3.5]
    cuts = np.array(init_cuts, dtype=np.float32)
    cuts.sort()

    def kappa_for(cuts_vec):
        pred = _apply_cutpoints(score, cuts_vec)
        return cohen_kappa_score(y_true, pred, weights="quadratic")

    best = float(kappa_for(cuts))

    stages = [
        (np.linspace(0.05, 3.95, 100, dtype=np.float32), 2),
        (None, 2),
        (None, 2),
    ]

    for stage_idx, (grid, iters) in enumerate(stages):
        for _ in range(iters):
            improved_any = False
            for i in range(4):
                if grid is None:
                    lo = max(0.05, float(cuts[i] - 0.35))
                    hi = min(3.95, float(cuts[i] + 0.35))
                    grid_i = np.linspace(lo, hi, 81, dtype=np.float32)
                else:
                    grid_i = grid

                eps = 1e-3
                left = 0.05 if i == 0 else float(cuts[i - 1] + eps)
                right = 3.95 if i == 3 else float(cuts[i + 1] - eps)
                grid_i = grid_i[(grid_i >= left) & (grid_i <= right)]
                if grid_i.size == 0:
                    continue

                local_best = best
                local_best_cut = float(cuts[i])

                for v in grid_i:
                    cuts_try = cuts.copy()
                    cuts_try[i] = float(v)
                    k = float(kappa_for(cuts_try))
                    if k > local_best:
                        local_best = k
                        local_best_cut = float(v)

                if local_best > best:
                    cuts[i] = local_best_cut
                    best = local_best
                    improved_any = True

            if not improved_any:
                break

    return cuts.tolist(), float(best)


def _tune_cutpoints_oof(
    net, train_df, train_img_dir, transform2, use_gpu, n_splits=5, cache_dir=None
):
    y = train_df["diagnosis"].astype(int).values
    oof_prob = np.empty((len(train_df), 5), dtype=np.float32)
    oof_prob[:] = np.nan

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    for fold, (_, va_idx) in enumerate(skf.split(np.zeros(len(train_df)), y), start=1):
        val_df = train_df.iloc[va_idx].reset_index(drop=True)
        val_ds = eye_train_dataset(
            val_df, img_dir=train_img_dir, transform=transform2, cache_dir=cache_dir
        )

        nworkers = min(8, (os.cpu_count() or 2))
        val_loader = DataLoader(
            val_ds,
            batch_size=16,
            shuffle=False,
            num_workers=nworkers,
            pin_memory=use_gpu,
            persistent_workers=(nworkers > 0),
            prefetch_factor=4 if nworkers > 0 else None,
        )
        val_prob, _ = _predict_sigmoid_prob(net, val_loader, use_gpu=use_gpu)
        oof_prob[va_idx, :] = val_prob.astype(np.float32)

    ok = np.isfinite(oof_prob).all(axis=1)
    if ok.sum() != len(train_df):
        raise RuntimeError(
            "OOF scoring failed: some folds did not produce predictions."
        )

    oof_score = _score_from_prob(oof_prob[ok])

    cuts4, best_k = _tune_cutpoints_coorddesc(
        y_true=y[ok], score=oof_score, init_cuts=[0.5, 1.5, 2.5, 3.5]
    )
    return cuts4, best_k


def _fit_cuts_by_class_quantiles(y_true, score):
    y_true = np.asarray(y_true, dtype=np.int64)
    score = np.asarray(score, dtype=np.float32)
    cuts = []
    for c in [0, 1, 2, 3]:
        a = score[y_true == c]
        b = score[y_true == (c + 1)]
        if a.size < 5 or b.size < 5:
            qa = float(np.quantile(score, (c + 1) / 5.0))
            qb = float(np.quantile(score, (c + 2) / 5.0))
            cuts.append((qa + qb) / 2.0)
        else:
            lo = float(np.quantile(a, 0.80))
            hi = float(np.quantile(b, 0.20))
            cuts.append((lo + hi) / 2.0)
    cuts = np.array(cuts, dtype=np.float32)
    cuts = np.clip(cuts, 0.05, 3.95)
    cuts.sort()
    eps = 1e-3
    for i in range(1, 4):
        if cuts[i] <= cuts[i - 1] + eps:
            cuts[i] = min(3.95, float(cuts[i - 1] + eps))
    return cuts.tolist()


def _make_ordinal_targets(labels, num_classes=5):
    labels = np.asarray(labels, dtype=np.int64)
    t = np.zeros((labels.shape[0], num_classes), dtype=np.float32)
    for i, y in enumerate(labels):
        y = int(y)
        if y < 0:
            y = 0
        if y > num_classes - 1:
            y = num_classes - 1
        t[i, : y + 1] = 1.0
    return t


def _fit_linear_head_bce(
    net, train_df, train_img_dir, transform2, use_gpu, cache_dir=None
):
    net.train()
    net.freeze_base()

    ds = eye_train_dataset(
        train_df, img_dir=train_img_dir, transform=transform2, cache_dir=cache_dir
    )

    nworkers = min(8, (os.cpu_count() or 2))
    loader = DataLoader(
        ds,
        batch_size=16,
        shuffle=True,
        num_workers=nworkers,
        pin_memory=use_gpu,
        persistent_workers=(nworkers > 0),
        prefetch_factor=4 if nworkers > 0 else None,
    )

    crit = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(net.classifiers.parameters(), lr=1e-3, weight_decay=1e-4)

    epochs = 3

    for ep in range(1, epochs + 1):
        running = 0.0
        n = 0
        for x, y in loader:
            y_np = np.asarray(y, dtype=np.int64)
            t_np = _make_ordinal_targets(y_np, num_classes=5)
            t = torch.from_numpy(t_np)

            if use_gpu:
                x = x.cuda(non_blocking=True)
                t = t.cuda(non_blocking=True)

            opt.zero_grad(set_to_none=True)
            out = net(x)
            loss = crit(out, t)
            loss.backward()
            opt.step()

            bs = int(x.size(0))
            running += float(loss.detach().cpu().item()) * bs
            n += bs

        print(
            f"Fitted linear head epoch {ep}/{epochs}, BCE loss={running/max(1,n):.5f}"
        )

    net.eval()
    return net




## === cell 4
if __name__ == "__main__":
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        torch.cuda.manual_seed_all(0)
    else:
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

    name_file = "/kaggle/input/aptos2019-blindness-detection/test.csv"
    test_df = pd.read_csv(name_file)
    content = (test_df["id_code"].astype(str) + ".png").tolist()

    cache_root = "/kaggle/working/aptos_cache_224"
    test_cache_dir = osp.join(cache_root, "test")
    train_cache_dir = osp.join(cache_root, "train")

    train_img_cand = [
        "/kaggle/input/aptos2019-blindness-detection/train_images",
        "/kaggle/input/train_images",
        "/kaggle/input/aptos2019-blindness-detection/aptos2019-blindness-detection/train_images",
    ]
    train_img_dir = next((d for d in train_img_cand if osp.isdir(d)), train_img_cand[0])

    _build_cache_for_ids(
        content,
        img_dir=next(
            (
                d
                for d in [
                    "/kaggle/input/aptos2019-blindness-detection/test_images",
                    "/kaggle/input/test_images",
                    "/kaggle/input/aptos2019-blindness-detection/aptos2019-blindness-detection/test_images",
                ]
                if osp.isdir(d)
            ),
            "/kaggle/input/aptos2019-blindness-detection/test_images",
        ),
        cache_dir=test_cache_dir,
        desc="Caching test preprocessed",
    )

    train_csv = "/kaggle/input/aptos2019-blindness-detection/train.csv"
    train_df = pd.read_csv(train_csv)
    train_ids_png = (train_df["id_code"].astype(str) + ".png").tolist()
    _build_cache_for_ids(
        train_ids_png,
        img_dir=train_img_dir,
        cache_dir=train_cache_dir,
        desc="Caching train preprocessed",
    )

    test_data = eye_dataset(content, transform2, cache_dir=test_cache_dir)
    net = Baseline_single(num_classes=5)

    if use_gpu:
        net = net.cuda()

    candidate_weight_paths = [
        "/kaggle/input/temp-file/model_yuan492_dense_00001_adam_pre_z_ji.pkl",
        "/kaggle/input/aptos2019-blindness-detection/model_yuan492_dense_00001_adam_pre_z_ji.pkl",
        "/kaggle/input/model_yuan492_dense_00001_adam_pre_z_ji.pkl",
    ]
    weight_path = next((p for p in candidate_weight_paths if os.path.exists(p)), None)

    have_comp_checkpoint = False
    if weight_path is not None:
        state = torch.load(weight_path, map_location="cuda" if use_gpu else "cpu")
        missing, unexpected = net.load_state_dict(state, strict=False)
        print(f"Loaded weights: {weight_path}")
        have_comp_checkpoint = True
        if missing:
            print(f"Missing keys (ignored): {len(missing)}")
        if unexpected:
            print(f"Unexpected keys (ignored): {len(unexpected)}")
    else:
        try:
            imagenet_weights = torchvision.models.DenseNet121_Weights.IMAGENET1K_V1
            d_im = torchvision.models.densenet121(weights=imagenet_weights)
            net.base.load_state_dict(d_im.features.state_dict(), strict=True)
            print(
                "WARNING: Competition checkpoint not found; loaded ImageNet-pretrained DenseNet121 "
                "features as a minimal performance-improving fallback."
            )
        except Exception as e:
            print(
                "WARNING: Competition checkpoint not found and ImageNet fallback failed "
                f"({type(e).__name__}: {e}). Proceeding with untrained model."
            )

    tuned_cuts4 = None
    try:
        if have_comp_checkpoint:
            tuned_cuts4, best_k = _tune_cutpoints_oof(
                net=net,
                train_df=train_df,
                train_img_dir=train_img_dir,
                transform2=transform2,
                use_gpu=use_gpu,
                n_splits=5,
                cache_dir=train_cache_dir,
            )
            print(
                f"OOF cutpoint tuning complete. OOF QWK={best_k:.5f}, cuts4={tuned_cuts4}"
            )
        else:
            net = _fit_linear_head_bce(
                net=net,
                train_df=train_df,
                train_img_dir=train_img_dir,
                transform2=transform2,
                use_gpu=use_gpu,
                cache_dir=train_cache_dir,
            )

            train_ds = eye_train_dataset(
                train_df,
                img_dir=train_img_dir,
                transform=transform2,
                cache_dir=train_cache_dir,
            )
            nworkers = min(8, (os.cpu_count() or 2))
            train_loader = DataLoader(
                train_ds,
                batch_size=16,
                shuffle=False,
                num_workers=nworkers,
                pin_memory=use_gpu,
                persistent_workers=(nworkers > 0),
                prefetch_factor=4 if nworkers > 0 else None,
            )
            prob_tr, y_tr = _predict_sigmoid_prob(net, train_loader, use_gpu=use_gpu)
            score_tr = _score_from_prob(prob_tr)
            tuned_cuts4 = _fit_cuts_by_class_quantiles(y_tr, score_tr)
            pred_tr = _apply_cutpoints(score_tr, tuned_cuts4)
            best_k = cohen_kappa_score(y_tr, pred_tr, weights="quadratic")
            print(
                f"Quantile cut calibration complete. Train QWK={best_k:.5f}, cuts4={tuned_cuts4}"
            )

    except Exception as e:
        tuned_cuts4 = None
        print(
            f"WARNING: Cutpoint tuning skipped due to {type(e).__name__}: {e}. "
            "Using default 0.5-count rule."
        )

    nworkers = min(8, (os.cpu_count() or 2))
    dataloader_test = DataLoader(
        test_data,
        batch_size=8,
        shuffle=False,
        num_workers=nworkers,
        pin_memory=use_gpu,
        persistent_workers=(nworkers > 0),
        prefetch_factor=4 if nworkers > 0 else None,
    )

    sub_path = "/kaggle/working/submission.csv"

    net.eval()
    rows = [("id_code", "diagnosis")]

    with torch.no_grad():
        for data, names in tqdm(dataloader_test, total=len(dataloader_test)):
            if use_gpu:
                data = data.cuda(non_blocking=True)

            out = net(data)
            prob = torch.sigmoid(out).detach().cpu().numpy()

            if tuned_cuts4 is None:
                predicted = (prob > 0.5).sum(axis=1)
                predicted = np.clip(predicted, 0, 4).astype(int)
            else:
                score = _score_from_prob(prob)
                predicted = _apply_cutpoints(score, tuned_cuts4).astype(int)

            rows.extend((str(n), int(p)) for n, p in zip(list(names), list(predicted)))

    with open(sub_path, "w", newline="") as f:
        f_csv = csv.writer(f)
        f_csv.writerows(rows)

    sub_df = pd.read_csv(sub_path)
    assert list(sub_df.columns) == ["id_code", "diagnosis"]
    assert len(sub_df) == len(test_df)
    print(f"Wrote submission: {sub_path} with shape {sub_df.shape}")
