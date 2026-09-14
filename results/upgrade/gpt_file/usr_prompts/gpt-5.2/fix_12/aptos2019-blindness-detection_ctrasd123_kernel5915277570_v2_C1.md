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

0.8127123486953338

# 6. Current score

0.00369

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -0.18876) has done: 'I remove the TensorBoard import that is crashing due to an incompatible tensorboard/protobuf setup, since it isn’t used for inference here. Then I fix the missing checkpoint issue by falling back to a deterministic, zero-training inference path (using ImageNet-pretrained DenseNet121 weights) so the notebook always runs end-to-end and writes a valid `submission.csv`. Finally, I make prediction post-processing slightly more metric-aligned (ordinal rounding of expected class from softmax probabilities) while keeping the same model family and inference-only approach, to improve QWK vs raw argmax.'
- What this solution (achieved -0.11354) has done: 'Your current score is far below the target, so we should improve it with the smallest changes that keep your “inference-only ImageNet DenseNet121” core logic intact. The biggest issue is that your model head is randomly initialized (not ImageNet-pretrained), so predictions are essentially noise; we can fix this by converting the DenseNet to a proper 5-class classifier by replacing `classifier` and using the model’s native forward, while preserving your checkpoint-loading fallback behavior. Next, to better match the QWK ordinal nature without changing training, we add simple test-time augmentation (horizontal flip) and average probabilities before ordinal rounding; this is a minimal inference-only change that typically improves kappa. Finally, we ensure the submission row order exactly matches `test.csv` by writing predictions into a dict and then emitting rows in `test_df` order.'
- What this solution (achieved -0.06357) has done: 'Your score is far below the target (gap ≈ -0.926), so we should improve QWK with minimal, inference-only changes while keeping your DenseNet121 + preprocessing pipeline intact. The biggest lift comes from aligning preprocessing with ImageNet expectations: ensure inputs are resized to 224 and use bicubic interpolation, since DenseNet121 was pretrained on 224px images; this is a small change but usually improves predictions significantly. Next, add a very small TTA set (original + hflip + vflip) and average probabilities, which tends to improve ordinal consistency for kappa without changing training. Finally, use the same expected-value rounding but ensure tensors are contiguous after flips for safe/consistent inference.'
- What this solution (achieved 0.05854) has done: 'Your current score is far below the target, so we should improve QWK with the smallest inference-only changes while keeping your DenseNet121 core intact. The biggest issue is that an ImageNet-pretrained 1000-class DenseNet with a random 5-class head produce near-random predictions; we keep the same DenseNet121, but use the pretrained 1000-way logits and map them to 5 ordinal classes via a fixed, deterministic “severity score” projection (no training, no extra data). Then we keep your existing expected-value rounding + small TTA averaging, but apply it to these calibrated ordinal probabilities rather than the random 5-class head outputs. This preserves your end-to-end pipeline and submission format while making predictions meaningfully ordinal, which should move QWK upward toward the target.'
- What this solution (achieved 0.0496) has done: 'Your current score (0.05854) is far below the target (0.8127), so we should improve it with the smallest change that makes predictions meaningfully related to DR severity. The main issue is that you’re using an ImageNet 1000-class DenseNet output and a heuristic mapping to 5 classes, which is essentially uncalibrated for this task. Without changing the model family or adding training, a minimal but high-impact fix is to replace that heuristic with a deterministic retina-specific signal extracted from the input (mean green-channel intensity after your existing crop/resize), then convert that continuous score into the 5 labels using fixed thresholds chosen from the training-label distribution (computed from train.csv only). This preserves your end-to-end inference pipeline and submission semantics, but should move QWK substantially upward toward the target compared to the current near-random mapping.'
- What this solution (achieved 0.02511) has done: 'Your current score (0.0496) is far below the target (0.8127), so we should improve it with minimal, legitimate changes that better align predictions with DR severity without changing the overall “no-training inference” nature of your pipeline. The biggest issue is that the DenseNet model is currently unused; we can keep your existing preprocessing and DataLoader, but generate an ordinal severity score from the pretrained DenseNet’s penultimate features (still deterministic inference-only) instead of raw green-channel mean, which is too weak. We then keep your existing “match train label distribution via quantiles” conversion to 0–4 (same post-processing concept), just applied to the stronger feature-based score. Finally, we add a tiny, safe TTA (original + horizontal flip) and average the extracted scores to improve stability and typically increase QWK without altering training or architecture.'
- What this solution (achieved -0.11569) has done: 'Your current score (0.02511) is far below the target (0.8127), so we should improve QWK with the smallest legitimate inference-only changes without changing your overall “DenseNet121 feature → scalar score → quantile binning to 0–4” core logic. The biggest issue is that your scalar score (L2 norm of pooled ReLU features) is not very DR-specific; we can keep the exact same feature extraction stage but change the scalar to a more discriminative, still-deterministic statistic: the mean of the top-k channel activations (per image) from the pooled feature vector. Next, we make the test-time augmentation slightly stronger but still minimal (add vertical flip alongside your horizontal flip) and average the three scores to stabilize the ranking, which usually helps kappa when you later quantile-bin. Finally, we keep the exact same quantile matching to the train label distribution and the same submission-writing logic/order to ensure validity.'
- What this solution (achieved 0.02511) has done: 'Your current score is far below the target, so we need a small but meaningful improvement without changing the overall “DenseNet feature → scalar score → quantile binning” approach. The main issue is that the current scalar (top‑k mean of pooled ReLU activations) is a weak, often non-ordinal proxy for DR severity, which can invert rankings and hurt QWK; we keep the same features but switch to a more stable scalar: the log of the mean pooled activation energy (mean of squared pooled features), which tends to correlate better with “amount of abnormal structure” while remaining deterministic and inference-only. We also make flip-TTA consistent with retina geometry by averaging only original + horizontal flip (vertical flip can distort anatomical orientation and can degrade ranking), which is a minimal change expected to improve ordinal consistency. Everything else (paths, DenseNet backbone, preprocessing, quantile-matching to train label distribution, submission writing/order) stays the same so it still runs end-to-end and writes a valid `submission.csv`.'
- What this solution (achieved -0.01315) has done: 'Your score is far below the target, so we should improve QWK with the smallest change that keeps the same “DenseNet features → scalar severity score → quantile binning” core logic. The biggest issue is that your current scalar (log mean squared pooled activations) is a weak proxy; we keep the same extracted pooled feature vector but compute a more discriminative, still-deterministic scalar: the log of the mean of the top‑k pooled feature magnitudes (a common way to capture “abnormality strength” while staying ordinal-friendly). To stabilize the ranking for quantile binning, we also add one minimal extra TTA (90° rotation) alongside your existing horizontal flip and average the three scores. Everything else (paths, backbone, preprocessing, quantile-matching, submission writing/order) stays unchanged and still writes a valid `submission.csv`.'
- What this solution (achieved 0.00369) has done: 'Your current score (-0.01315) is far below the target (0.8127), so we need a real signal improvement while keeping your core “DenseNet features → scalar severity score → quantile binning” pipeline unchanged. The biggest issue is that the scalar you extract from DenseNet activations is essentially unrelated to DR; a minimal but high-impact fix is to keep the same DenseNet feature extraction, but compute a retina-relevant scalar from the image itself (after your existing crop/resize): the fraction of bright lesions in the green channel via a simple top-hat + threshold measure. We keep the same flip/rotation TTA averaging concept (still deterministic inference-only), but apply it to this lesion score instead of DenseNet features, then keep the same quantile-matching to the train label distribution and the same submission writing/order. This should move QWK substantially upward versus near-random predictions, without introducing training or changing evaluation semantics.'

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
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

SummaryWriter = None

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)



## === cell 1
TEST_CSV_PATH = "/kaggle/input/aptos2019-blindness-detection/test.csv"
test_df = pd.read_csv(TEST_CSV_PATH)
content = (test_df["id_code"].astype(str) + ".png").tolist()

print("Num test images:", len(content))
print("First 3:", content[:3])



## === cell 2
import torch
from torch import nn
import torchvision


class Baseline_single(nn.Module):
    def __init__(self, num_classes, loss_type="single BCE", **kwargs):
        super(Baseline_single, self).__init__()
        self.loss_type = loss_type

        try:
            self.model = torchvision.models.densenet121(
                weights=torchvision.models.DenseNet121_Weights.IMAGENET1K_V1
            )
        except Exception:
            self.model = torchvision.models.densenet121(pretrained=True)

        self.num_classes = num_classes

    def freeze_base(self):
        for name, p in self.model.named_parameters():
            if "classifier" not in name:
                p.requires_grad = False

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True

    def forward(self, x1):
        return self.model(x1)




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
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_CUBIC)
    return image


class eye_dataset(Dataset):
    def __init__(self, txt_path, transform=None):
        self.imgs = list(txt_path)
        self.transform = transform

    def __getitem__(self, index):
        fn = self.imgs[index]
        img = cv_imread("/kaggle/input/aptos2019-blindness-detection/test_images/" + fn)
        img = load_ben_yuan(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, fn[:-4]

    def __len__(self):
        return len(self.imgs)




## === cell 4
use_gpu = torch.cuda.is_available()
device = torch.device("cuda" if use_gpu else "cpu")

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

test_data = eye_dataset(content, transform2)

net = Baseline_single(num_classes=5).to(device)

CKPT_PATH = "/kaggle/input/temp-file/model_dense_m_adam.pkl"
loaded_ckpt = False
if osp.exists(CKPT_PATH):
    ckpt = torch.load(CKPT_PATH, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt

    if isinstance(state_dict, dict) and any(
        k.startswith("module.") for k in state_dict.keys()
    ):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    missing, unexpected = net.load_state_dict(state_dict, strict=False)
    print(
        "Checkpoint load (strict=False). Missing keys:",
        len(missing),
        "Unexpected keys:",
        len(unexpected),
    )
    loaded_ckpt = True

print(f"Checkpoint loaded: {loaded_ckpt}")

dataloader_test = DataLoader(
    test_data, batch_size=16, shuffle=False, num_workers=2, pin_memory=use_gpu
)

out_path = "/kaggle/working/submission.csv"

TRAIN_CSV_PATH = "/kaggle/input/aptos2019-blindness-detection/train.csv"
train_df = pd.read_csv(TRAIN_CSV_PATH)
train_counts = train_df["diagnosis"].value_counts().sort_index()
train_props = (
    (train_counts / train_counts.sum()).reindex([0, 1, 2, 3, 4]).fillna(0.0).values
)
cum_props = np.cumsum(train_props)

q_levels = [cum_props[0], cum_props[1], cum_props[2], cum_props[3]]


def _lesion_fraction_from_rgb_uint8(rgb224: np.ndarray) -> float:
    """
    rgb224: uint8 RGB image [224,224,3] after crop/resize.
    Returns a continuous score where larger means more bright-lesion-like signal.
    """
    g = rgb224[:, :, 1].astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    tophat = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, kernel)

    thr = int(np.clip(np.percentile(tophat, 99.0), 10, 255))
    mask = (tophat >= thr).astype(np.uint8)

    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    )

    frac = float(mask.mean())
    return float(np.log1p(1000.0 * frac))


def _tensor_to_rgb_uint8(x_normed: torch.Tensor) -> np.ndarray:
    """
    x_normed: [3,224,224] normalized by ImageNet mean/std.
    Returns uint8 RGB [224,224,3].
    """
    mean = torch.tensor([0.485, 0.456, 0.406], device=x_normed.device)[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=x_normed.device)[:, None, None]
    x = (x_normed * std + mean).clamp(0.0, 1.0)
    x = (x * 255.0).round().to(torch.uint8)
    rgb = x.permute(1, 2, 0).contiguous().cpu().numpy()
    return rgb


def extract_severity_score(x_normed: torch.Tensor) -> torch.Tensor:
    """
    x_normed: normalized tensor [B,3,224,224] with ImageNet mean/std.
    Returns: score [B], deterministic.
    """
    scores_local = []
    for i in range(x_normed.shape[0]):
        rgb = _tensor_to_rgb_uint8(x_normed[i])
        scores_local.append(_lesion_fraction_from_rgb_uint8(rgb))
    return torch.tensor(scores_local, device=x_normed.device, dtype=torch.float32)


scores = []
names_all = []

with torch.no_grad():
    for _, item in tqdm(enumerate(dataloader_test), total=len(dataloader_test)):
        data, name = item
        data = data.to(device, non_blocking=True)

        s1 = extract_severity_score(data)

        data_hf = torch.flip(data, dims=[3]).contiguous()
        s2 = extract_severity_score(data_hf)

        data_r90 = torch.rot90(data, k=1, dims=[2, 3]).contiguous()
        s3 = extract_severity_score(data_r90)

        s = (s1 + s2 + s3) / 3.0

        scores.extend(s.detach().cpu().numpy().tolist())
        names_all.extend([str(n) for n in name])

scores = np.asarray(scores, dtype=np.float32)

q_levels_safe = [float(np.clip(q, 0.0, 1.0)) for q in q_levels]
t1, t2, t3, t4 = np.quantile(scores, q_levels_safe).tolist()


def score_to_label(s):
    if s <= t1:
        return 0
    elif s <= t2:
        return 1
    elif s <= t3:
        return 2
    elif s <= t4:
        return 3
    else:
        return 4


pred_dict = {n: score_to_label(s) for n, s in zip(names_all, scores)}

with open(out_path, "w", newline="") as f:
    f_csv = csv.writer(f)
    f_csv.writerow(["id_code", "diagnosis"])
    for id_code in test_df["id_code"].astype(str).tolist():
        f_csv.writerow([id_code, int(pred_dict[id_code])])

print("Wrote:", out_path)
sub = pd.read_csv(out_path)
print(sub.head())
print("Num rows:", len(sub))
print("Unique diagnosis counts:\n", sub["diagnosis"].value_counts().sort_index())
print("Thresholds used:", (t1, t2, t3, t4))
