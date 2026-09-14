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

0.46701

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I remove the non-Python prose and stray backticks that currently cause `SyntaxError`, keeping the same pipeline structure and core model/inference logic. I also eliminate the broken `tensorboard` fallback import path that triggers an internal `tensorboard` error, replacing it with a tiny no-op `SummaryWriter` stub (it isn’t used for submission). Next, I fix the `__future__` import placement (must be the first statement in the file) to avoid runtime syntax issues. Finally, I make file paths consistent with the provided Kaggle dataset layout and ensure we always write `/kaggle/working/submission.csv` with the required `id_code,diagnosis` columns and correct row alignment.'
- What this solution (achieved 0.00132) has done: 'Your 0.0 score is consistent with running inference from randomly initialized DenseNet weights (the checkpoint path points to a dataset that isn’t available), plus a brittle prediction rule that can output mostly a single class. To move the score upward toward the target while keeping the same model and inference semantics, I (1) load ImageNet pretrained weights for the DenseNet backbone when the competition checkpoint is missing, (2) use the standard ImageNet normalization at input (this matches the pretrained backbone expectations), and (3) make `get_preds` robust so it always maps the 5 sigmoid outputs to an ordinal class 0–4 without falling into the “all false → class 0” failure mode. These are minimal, local changes that preserve the architecture and overall pipeline while making predictions meaningfully non-random.'
- What this solution (achieved -0.02386) has done: 'Your current 0.00132 score suggests the model head is effectively random (checkpoint missing) and the fixed 0.5 thresholding is producing poorly calibrated ordinal labels. To move the score upward toward the 0.8747 target without changing the architecture or adding a training loop, I (1) keep ImageNet-pretrained DenseNet201 when the checkpoint is missing, but replace the brittle fixed thresholding with a tiny, validation-free calibration: choose thresholds so the predicted class distribution matches the training label distribution. This keeps the same evaluation semantics (predict 0–4) and uses only provided `train.csv` (no leakage from test labels). I also make inference deterministic and ensure the merge/output row order matches `test.csv` exactly.'
- What this solution (achieved 0.0) has done: 'Your score is far below the target, so we should increase performance with the smallest changes that keep your DenseNet201 + 5-logit ordinal-BCE-style head and “sum-of-sigmoids → scalar → thresholds → class” semantics intact. The biggest issue is that when the competition checkpoint is missing, the classification head is randomly initialized, making predictions essentially noise; we keep the same architecture but (1) initialize the head and `cal_score` weights to zeros for stable, monotonic outputs, and (2) calibrate thresholds on a tiny train-derived set of predictions (not labels) so the mapping from scalar score→class is aligned to the model’s output scale rather than using test-quantiles (which can be unstable for kappa). Finally, we sort `sub_pred` to match `test.csv` order directly (no merge ambiguity) and keep deterministic settings unchanged.'
- What this solution (achieved -0.02363) has done: 'Your 0.0 score indicates the submission is effectively uninformative; the biggest cause in your current code is that when the competition checkpoint is missing you zero-initialize the classification head, which forces all sigmoid outputs to 0.5 and yields constant predictions after thresholding. To move the score upward toward the 0.8747 target with minimal change and identical architecture/inference semantics, I keep the ImageNet-pretrained DenseNet201 backbone fallback but remove the head zero-initialization so the head uses its default random init (at least produces non-constant, image-dependent outputs). I also compute calibration thresholds using *all* training IDs (still no test-label leakage) rather than only the first 512, which reduces instability in the distribution-matching calibration you already use. Submission writing/ordering stays the same, and the code remains deterministic.'
- What this solution (achieved 0.0972) has done: 'Your score is far below the target, so we should increase it with the smallest changes that fix the biggest accuracy killers while keeping your DenseNet201 + 5-logit ordinal head and “sigmoid→sum→thresholds→class” semantics intact. The main issue is that without the competition checkpoint, the head is random and the backbone features don’t match APTOS well; a minimal, non-architectural improvement is to use a standard Retina competition preprocessing (Ben Graham-style circular crop + CLAHE-ish weighted blur) during both calibration and test inference. Next, your calibration currently runs the whole training set at 512×512 (slow) and may hit time/memory instability; we keep calibration but compute thresholds from train predictions more robustly with a deterministic subset size cap that still covers label distribution well. Finally, we keep submission alignment strict to `test.csv` order and add a safety assert for row count and nulls.'
- What this solution (achieved -0.19493) has done: 'To move your QWK score upward toward 0.8747 with minimal disruption, I’m keeping the exact DenseNet201 + 5-logit ordinal head and the same “sigmoid → sum-of-probs → thresholds → class” semantics. The main score killer is that, when the competition checkpoint is missing, the head is random and your distribution-matching calibration on a subset can be unstable; I make calibration use the full training set but at a smaller image size (256) to stay within the 600s budget while improving threshold stability. I also (minimally) upgrade the scalar score computation to use your existing `cal_score` layer (already in the model) to learn a consistent linear severity score from the 5 logits, which is more aligned than a raw sum while preserving the ordinal mapping pipeline. Finally, I keep strict submission alignment to `test.csv` order and keep preprocessing consistent between calibration and test.'
- What this solution (achieved 0.07252) has done: 'Your current score is far below the target, so we should increase it with the smallest changes that keep your DenseNet201 + 5-logit head and “logits → linear score → thresholds → class 0–4” semantics intact. The biggest likely score killer is that `cal_score` is untrained/random when the competition checkpoint is missing, so the scalar score is essentially noise; we keep the same mapping pipeline but replace the scalar severity score with a fixed, monotonic ordinal score `sum(sigmoid(logits))`, which matches your original ordinal-BCE intent and is stable without training. To stay within the 600s budget while improving stability, we also make calibration and test use the same image size (256) so thresholds are fit on the same score distribution used at inference. Finally, we keep deterministic settings and strict submission alignment to `test.csv` order.'
- What this solution (achieved 0.07252) has done: 'We keep your DenseNet201 + 5-logit ordinal head and the same “sigmoid→sum score→4 thresholds→class 0–4” semantics, but fix two calibration/inference mismatches that severely hurt QWK. First, we compute thresholds using **out-of-fold (OOF)** scores (stratified 5-fold) instead of in-sample train scores, which reduces overfitting and usually improves generalization without changing the model or adding training. Second, we use `torch.inference_mode()` and slightly larger batches/workers to stay within the 600s budget while running the extra OOF passes. Submission alignment remains strict to `test.csv` order and we still write `/kaggle/working/submission.csv`.'
- What this solution (achieved 0.46701) has done: 'Your current score (0.07252) is far below the target (0.87474), so we should increase it with the smallest changes that keep your exact DenseNet201 + 5-logit ordinal head and the same “sigmoid→scalar score→4 thresholds→class 0–4” semantics. The biggest avoidable weakness right now is that, without the competition checkpoint, your classifier head is randomly initialized; we keep the same architecture but load ImageNet-pretrained weights for the DenseNet201 **features** and also initialize the linear head from the pretrained DenseNet201 classifier weights (same tensor shape: 1000→5 slice) to make predictions non-random yet still consistent with your pipeline. Next, we fit thresholds using the OOF scores you already compute, but we optimize thresholds directly for quadratic weighted kappa on OOF (coordinate descent), which is a minimal post-processing change and typically yields a large jump for this metric. Finally, we keep deterministic behavior, identical preprocessing, strict test order alignment, and still write `/kaggle/working/submission.csv`.'

# 9. Code solution

## === cell 0
from __future__ import print_function, absolute_import

import os
import os.path as osp
import random
import math

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

from sklearn.model_selection import StratifiedKFold


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


def ben_graham_preprocess(rgb, out_size=512, sigma=10):
    rgb = crop_image_from_gray(rgb)
    if rgb.size == 0:
        return np.zeros((out_size, out_size, 3), dtype=np.uint8)
    rgb = cv2.resize(rgb, (out_size, out_size))
    blur = cv2.GaussianBlur(rgb, (0, 0), sigma)
    out = cv2.addWeighted(rgb, 4, blur, -4, 128)
    return out


def logits_to_score_ordinal_sum(logits5):
    """
    logits5: torch.Tensor [B,5] (pre-sigmoid logits from classifiers)
    Returns numpy float32 scores [B] in [0,5] approx (sum of sigmoid probs).
    """
    s = torch.sigmoid(logits5).sum(dim=1)
    return s.detach().cpu().numpy().astype(np.float32)


def score_to_class(scores, thresholds):
    s = np.asarray(scores, dtype=np.float32).reshape(-1)
    t = np.asarray(thresholds, dtype=np.float32).reshape(4)
    cls = (s[:, None] > t[None, :]).sum(axis=1).astype(np.int64)
    return np.clip(cls, 0, 4)


def qwk_quadratic(y_true, y_pred, n_classes=5):
    """
    Minimal, dependency-free quadratic weighted kappa for labels 0..n_classes-1.
    """
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.int64).reshape(-1)
    assert y_true.shape == y_pred.shape

    O = np.zeros((n_classes, n_classes), dtype=np.float64)
    for a, b in zip(y_true, y_pred):
        if 0 <= a < n_classes and 0 <= b < n_classes:
            O[a, b] += 1.0

    act_hist = O.sum(axis=1)
    pred_hist = O.sum(axis=0)
    E = np.outer(act_hist, pred_hist)
    if E.sum() > 0:
        E = E * (O.sum() / E.sum())

    W = np.zeros((n_classes, n_classes), dtype=np.float64)
    denom = float((n_classes - 1) ** 2)
    for i in range(n_classes):
        for j in range(n_classes):
            W[i, j] = ((i - j) ** 2) / denom

    num = (W * O).sum()
    den = (W * E).sum()
    if den == 0:
        return 0.0
    return 1.0 - (num / den)


def optimize_thresholds_for_qwk(scores, labels, init_thresholds, n_iter=3):
    """
    Why: directly optimizing thresholds for QWK on OOF reduces the gap to the target
    without changing model architecture or training. We keep the same mapping:
    score -> 4 thresholds -> class 0..4, but choose thresholds that maximize QWK.
    """
    s = np.asarray(scores, dtype=np.float32).reshape(-1)
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    t = np.asarray(init_thresholds, dtype=np.float32).reshape(4).copy()

    qs = np.linspace(0.05, 0.95, 91)
    cand = np.quantile(s, qs).astype(np.float32)
    cand = np.unique(cand)

    best_t = t.copy()
    best_k = qwk_quadratic(y, score_to_class(s, best_t), n_classes=5)

    for _ in range(int(n_iter)):
        for k in range(4):
            cur_best_tk = best_t[k]
            cur_best_kappa = best_k
            lo = -np.inf if k == 0 else best_t[k - 1] + np.float32(1e-6)
            hi = np.inf if k == 3 else best_t[k + 1] - np.float32(1e-6)

            for v in cand:
                if not (lo < v < hi):
                    continue
                trial_t = best_t.copy()
                trial_t[k] = np.float32(v)
                kappa = qwk_quadratic(y, score_to_class(s, trial_t), n_classes=5)
                if kappa > cur_best_kappa:
                    cur_best_kappa = kappa
                    cur_best_tk = np.float32(v)

            best_t[k] = cur_best_tk
            best_k = cur_best_kappa

    return best_t, float(best_k)


def fit_score_thresholds_from_train_predictions(train_scores, train_labels):
    """
    Baseline init: match class priors via quantiles (your existing semantics).
    """
    s = np.asarray(train_scores, dtype=np.float32).reshape(-1)
    y = np.asarray(train_labels, dtype=np.int64).reshape(-1)
    counts = np.bincount(y, minlength=5).astype(np.float64)
    priors = counts / counts.sum()
    cum = np.cumsum(priors)  # 5
    qs = [float(np.clip(cum[k], 1e-6, 1 - 1e-6)) for k in range(4)]
    t = np.quantile(s, qs).astype(np.float32)
    for i in range(1, len(t)):
        if t[i] <= t[i - 1]:
            t[i] = np.nextafter(t[i - 1], np.float32(1e9))
    return t


class eye_dataset_orl(Dataset):
    def __init__(
        self, ids_with_ext, img_dir, transform=None, use_ben=True, out_size=512
    ):
        self.imgs = list(ids_with_ext)
        self.img_dir = img_dir
        self.transform = transform
        self.use_ben = use_ben
        self.out_size = int(out_size)

    def __getitem__(self, index):
        fn = self.imgs[index]
        img_path = os.path.join(self.img_dir, fn)
        img = cv2.imread(img_path)
        if img is None:
            img = np.zeros((self.out_size, self.out_size, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if self.use_ben:
                img = ben_graham_preprocess(img, out_size=self.out_size, sigma=10)
            else:
                img = crop_image_from_gray(img)
                if img.size == 0:
                    img = np.zeros((self.out_size, self.out_size, 3), dtype=np.uint8)
                else:
                    img = cv2.resize(img, (self.out_size, self.out_size))
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

    out_size = 256

    test_content = (test_df["id_code"].astype(str) + ".png").tolist()
    test_data = eye_dataset_orl(
        test_content, TEST_IMG_DIR, transform2, use_ben=True, out_size=out_size
    )

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
            f"WARNING: checkpoint not found at {ckpt_path}. Using ImageNet-pretrained backbone + ImageNet-initialized head slice."
        )
        try:
            try:
                ref = torchvision.models.densenet201(
                    weights=torchvision.models.DenseNet201_Weights.IMAGENET1K_V1
                )
            except Exception:
                ref = torchvision.models.densenet201(weights="IMAGENET1K_V1")

            w = ref.classifier.weight.detach().cpu()  # [1000,1920]
            b = ref.classifier.bias.detach().cpu()  # [1000]
            net.classifiers.weight.data.copy_(
                w[:5].to(net.classifiers.weight.data.device)
            )
            net.classifiers.bias.data.copy_(b[:5].to(net.classifiers.bias.data.device))
        except Exception as e:
            print(
                "Head init from pretrained classifier failed; keeping default init. Error:",
                repr(e),
            )

    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    calib_labels = train_df["diagnosis"].astype(int).values

    oof_scores = np.zeros(len(train_df), dtype=np.float32)

    num_workers = 2
    if use_gpu:
        try:
            num_workers = min(4, max(2, os.cpu_count() // 2))
        except Exception:
            num_workers = 2

    net.eval()
    with torch.inference_mode():
        for fold, (_, val_idx) in enumerate(
            skf.split(np.zeros(len(train_df)), calib_labels), 1
        ):
            val_ids = (train_df.iloc[val_idx]["id_code"].astype(str) + ".png").tolist()
            val_data = eye_dataset_orl(
                val_ids, TRAIN_IMG_DIR, transform2, use_ben=True, out_size=out_size
            )
            val_loader = DataLoader(
                val_data,
                batch_size=32 if use_gpu else 8,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=use_gpu,
            )

            fold_scores = []
            for data, _name in tqdm(
                val_loader, total=len(val_loader), desc=f"OOF fold {fold}/{n_splits}"
            ):
                if use_gpu:
                    data = data.cuda(non_blocking=True)
                out = net(data)  # logits [B,5]
                fold_scores.append(logits_to_score_ordinal_sum(out))
            fold_scores = np.concatenate(fold_scores, axis=0)
            oof_scores[val_idx] = fold_scores

    thresholds0 = fit_score_thresholds_from_train_predictions(oof_scores, calib_labels)
    thresholds, oof_kappa = optimize_thresholds_for_qwk(
        oof_scores, calib_labels, thresholds0, n_iter=3
    )
    print("OOF QWK (threshold-optimized):", oof_kappa)
    print("Thresholds used:", thresholds)

    dataloader_test = DataLoader(
        test_data,
        batch_size=32 if use_gpu else 8,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_gpu,
    )

    names_all = []
    scores_all = []

    net.eval()
    with torch.inference_mode():
        for data, name in tqdm(
            dataloader_test, total=len(dataloader_test), desc="Test"
        ):
            if use_gpu:
                data = data.cuda(non_blocking=True)
            out = net(data)  # logits [B,5]
            s = logits_to_score_ordinal_sum(out)
            names_all.extend([str(n) for n in list(name)])
            scores_all.append(s)

    scores_all = np.concatenate(scores_all, axis=0)
    pred_cls = score_to_class(scores_all, thresholds)

    sub_pred = pd.DataFrame({"id_code": names_all, "diagnosis": pred_cls.astype(int)})

    sub = test_df[["id_code"]].merge(sub_pred, on="id_code", how="left", sort=False)
    sub["diagnosis"] = sub["diagnosis"].fillna(0).astype(int)

    assert len(sub) == len(test_df), "Submission row count mismatch."
    assert sub["diagnosis"].isna().sum() == 0, "Found NaNs in diagnosis."

    sub_path = "/kaggle/working/submission.csv"
    sub.to_csv(sub_path, index=False)
    print("Wrote:", sub_path)
    print(sub["diagnosis"].value_counts().sort_index())
