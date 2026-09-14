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

3.10

# 3. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
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
seaborn==0.12.2
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

0.8401562603955726

# 6. Current score

-0.08445

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the immediate runtime blockers by (1) correcting broken/unused path code, (2) making the dataset robust to missing/failed image reads, and (3) handling the missing external weight file by falling back to the base EfficientNet weights so inference still runs end-to-end. I also remove stochastic test-time augmentation (random flip/rotation) to make predictions deterministic and more stable. Finally, I ensure the prediction tensor-to-float conversion is correct and that the submission length matches `sample_submission.csv`, writing a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.31821) has done: 'Your current 0.0 score is most consistent with a submission/schema mismatch (wrong `id_code` ↔ `diagnosis` alignment) and/or producing nearly-constant predictions because the model head is random when the external trained weights are missing. I make the submission explicitly align predictions to `test.csv` order (not whatever order `sample_submission.csv` happens to be in) to eliminate accidental misalignment. Then, to move the score upward with minimal semantic change, I change only the final layer assignment to match EfficientNet-B0’s classifier structure (a Sequential with Dropout+Linear) so we correctly replace the last Linear while keeping pretrained features intact. Finally, I convert the single-regression output into 5 classes using simple rank-based binning (quintiles) instead of fixed thresholds, which avoids collapsing to mostly one class when the output scale is unknown, typically improving QWK from 0.0 toward a reasonable baseline.'
- What this solution (achieved 0.0) has done: 'Your current score is far below the target, so we need a small, legitimate uplift without changing the model or training loop (there is no training here). The biggest issue is that, when custom weights are missing, the classifier head is random because you replace it with a new Linear layer; this destroys the usefulness of pretrained features and leads to weak/near-random predictions. I keep the same EfficientNet-B0 backbone and single-output head, but initialize the replacement head to zeros so outputs are stable (and not random), and then replace the brittle quintile-binning with a tiny bit of label-distribution calibration: fit 4 thresholds on a stratified validation split of the training set using the model’s own outputs to directly maximize QWK. This preserves your inference pipeline and semantics (still a single regression output mapped to 0–4) while making the discretization match the metric more closely, typically moving QWK substantially upward toward your target.'
- What this solution (achieved 0.02982) has done: 'Your 0.0 score is most consistent with producing (nearly) constant predictions, which in your current code happens when the custom weights file is missing and you zero-initialize the new 1-output head—threshold calibration can’t recover from identical logits. To move the score upward with minimal semantic change, I keep the exact same EfficientNet-B0 backbone and the same “single regression output + thresholding to 0–4” approach, but I stop zeroing the head and instead keep ImageNet-pretrained features by *not* replacing the classifier when custom weights are unavailable (fallback to a 1000-class argmax mapped to 0–4). When custom weights are available, your current regression+threshold calibration path is preserved. I also make the submission explicitly follow `test.csv` order (it already does) and keep everything deterministic and within time.'
- What this solution (achieved 0.40038) has done: 'Your current score is far below the target, so we should make a small, legitimate uplift without changing the model/training approach (there is no training here). The main weakness is the fallback path: using ImageNet argmax bucketed into 5 bins is a very weak proxy for DR severity, so it tends to score near-random; instead we can leverage the *same EfficientNet-B0 backbone* and *same single-output regression + thresholding logic* by building a 1D “severity score” from the 1000-way logits (a fixed weighted sum over class indices) and then calibrating 4 thresholds on the validation split to maximize QWK. This keeps evaluation semantics identical (continuous score → 0–4 via thresholds), is deterministic, and should move the score substantially toward your target. The custom-weights path is preserved; only the fallback produces a better continuous score and uses the same threshold calibration routine.'
- What this solution (achieved -0.08445) has done: 'Your current score (0.40038) is far below the target (0.84016), so we need a legitimate uplift while keeping the same model/inference core (EfficientNet-B0 forward pass + continuous score + thresholding). The smallest high-impact fix is to stop deriving a “severity score” from ImageNet logits and instead extract a stable 1D severity embedding from the pretrained backbone (same model, no training), then calibrate thresholds on the validation split as you already do. Concretely, when custom regression weights are missing, we compute `raw = mean(features)` from EfficientNet’s pooled features (1280-d), which is much more image-dependent than ImageNet class-index expectation and remains deterministic. Everything else (data pipeline, transforms, threshold calibration routine, submission format) stays the same.'

# 9. Code solution

## === cell 0
import cv2
import matplotlib.pyplot as plt
from os.path import isfile
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
from PIL import Image, ImageFilter
from sklearn.model_selection import train_test_split, StratifiedKFold
from torch.utils.data import Dataset
from torchvision import transforms
from torch.optim import Adam, SGD, RMSprop
import time
from torch.autograd import Variable
from tqdm import tqdm
from sklearn import metrics
import urllib
import pickle
from torchvision import models
import seaborn as sns
import random
import sys
import gc
import warnings



## === cell 1
SEED = 123
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

warnings.filterwarnings("ignore")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n Device : {device.upper()}")



## === cell 2
BASE_INPUT = "../input/aptos2019-blindness-detection"
if not os.path.exists(BASE_INPUT):
    BASE_INPUT = "/kaggle/data/aptos2019-blindness-detection"
if not os.path.exists(BASE_INPUT):
    BASE_INPUT = "/kaggle/input/aptos2019-blindness-detection"

TEST_PATH = os.path.join(BASE_INPUT, "test.csv")
TEST_IMG = os.path.join(BASE_INPUT, "test_images")
SAMPLE_SUB_PATH = os.path.join(BASE_INPUT, "sample_submission.csv")

TRAIN_PATH = os.path.join(BASE_INPUT, "train.csv")
TRAIN_IMG = os.path.join(BASE_INPUT, "train_images")

test_csv = pd.read_csv(TEST_PATH)
train_csv = pd.read_csv(TRAIN_PATH)

print("test_csv shape:", test_csv.shape)
print("train_csv shape:", train_csv.shape)
print("TEST_IMG exists:", os.path.exists(TEST_IMG))
print("TRAIN_IMG exists:", os.path.exists(TRAIN_IMG))




## === cell 3
def expand_path(p, img_dir=TEST_IMG):
    p = str(p)
    candidate = os.path.join(img_dir, p + ".png")
    if isfile(candidate):
        return candidate
    return p




## === cell 4
def crop_image1(img, tol=7):
    mask = img > tol
    return img[np.ix_(mask.any(1), mask.any(0))]


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
        if check_shape == 0:  # too dark, cropping would remove everything
            return img
        img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
        img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
        img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
        img = np.stack([img1, img2, img3], axis=-1)
        return img
    return img




## === cell 5
IMG_SIZE = 256


class MyDataset(Dataset):
    def __init__(self, dataframe, transform=None, img_dir=TEST_IMG):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
        self.img_dir = img_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        p = self.df.id_code.values[idx]
        p_path = os.path.join(self.img_dir, p + ".png")

        image = cv2.imread(p_path)
        if image is None:
            image = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = crop_image_from_gray(image)
            if image is None or image.size == 0:
                image = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
            image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
            image = cv2.addWeighted(
                image, 4, cv2.GaussianBlur(image, (0, 0), 30), -4, 128
            )

        image = transforms.ToPILImage()(image)
        if self.transform:
            image = self.transform(image)

        if isinstance(image, torch.Tensor):
            x = image
        else:
            x = transforms.ToTensor()(image)
        return x




## === cell 6
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

testset = MyDataset(test_csv, transform=transform, img_dir=TEST_IMG)
test_loader = torch.utils.data.DataLoader(
    testset,
    batch_size=32,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)

tr_idx, va_idx = train_test_split(
    np.arange(len(train_csv)),
    test_size=0.25,
    random_state=SEED,
    stratify=train_csv["diagnosis"].values,
)
val_df = train_csv.iloc[va_idx].reset_index(drop=True)
valset = MyDataset(val_df[["id_code"]], transform=transform, img_dir=TRAIN_IMG)
val_loader = torch.utils.data.DataLoader(
    valset,
    batch_size=32,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)

val_y_true = val_df["diagnosis"].astype(int).values



## === cell 7
weights_path = (
    "../input/pretrained-and-trained-3-epochs/pretrainedandtrained3epochs.bin"
)

use_custom_regression = os.path.exists(weights_path)

try:
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
except Exception:
    model = models.efficientnet_b0(pretrained=True)

if use_custom_regression:
    if (
        isinstance(model.classifier, nn.Sequential)
        and len(model.classifier) >= 2
        and isinstance(model.classifier[-1], nn.Linear)
    ):
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(
            in_features=in_features, out_features=1, bias=True
        )
    else:
        model.classifier = nn.Linear(in_features=1280, out_features=1, bias=True)

    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state)
    print("Loaded custom weights:", weights_path)
else:
    print(
        "Custom weights not found; using base EfficientNet_B0 pretrained ImageNet model.",
        weights_path,
    )

model = model.to(device)
model.eval()




## === cell 8
def predict_loader(m, loader):
    preds = []
    with torch.no_grad():
        for x in tqdm(loader, total=len(loader)):
            x = x.float().to(device, non_blocking=True)
            out = m(x)
            preds.append(out.detach().cpu())
    return torch.cat(preds, dim=0)


val_out = predict_loader(model, val_loader)
test_out = predict_loader(model, test_loader)

print("Val out:", tuple(val_out.shape), "Test out:", tuple(test_out.shape))




## === cell 9
def apply_thresholds(preds, thr):
    thr = np.asarray(thr, dtype=np.float32)
    return np.digitize(preds, bins=thr, right=False).astype(int)


def qwk(y_true, y_pred):
    return metrics.cohen_kappa_score(y_true, y_pred, weights="quadratic")


def calibrate_thresholds(val_raw, val_y_true, iters=2):
    val_raw = np.asarray(val_raw, dtype=np.float32).reshape(-1)

    init_thr = np.quantile(val_raw, [0.2, 0.4, 0.6, 0.8]).astype(np.float32)
    if np.allclose(val_raw.min(), val_raw.max()):
        init_thr = np.array([-1.0, -0.5, 0.5, 1.0], dtype=np.float32)

    best_thr = init_thr.copy()
    best_score = qwk(val_y_true, apply_thresholds(val_raw, best_thr))

    span = float(np.std(val_raw) + 1e-6)
    deltas = np.array([-0.6, -0.3, -0.15, 0.0, 0.15, 0.3, 0.6], dtype=np.float32) * span

    for _ in range(iters):
        for k in range(4):
            cand_best = best_thr.copy()
            cand_score_best = best_score
            for d in deltas:
                cand = best_thr.copy()
                cand[k] = cand[k] + float(d)
                cand = np.sort(cand)
                score = qwk(val_y_true, apply_thresholds(val_raw, cand))
                if score > cand_score_best:
                    cand_score_best = score
                    cand_best = cand
            best_thr = cand_best
            best_score = cand_score_best

    return best_thr, float(best_score)


def pooled_feature_mean_raw(m, loader):
    raw = []
    with torch.no_grad():
        for x in tqdm(loader, total=len(loader)):
            x = x.float().to(device, non_blocking=True)
            feats = m.features(x)
            feats = m.avgpool(feats)
            feats = torch.flatten(feats, 1)  # (B, 1280)
            raw.append(feats.mean(dim=1).detach().cpu())
    return torch.cat(raw, dim=0).numpy().astype(np.float32)


if use_custom_regression:
    val_raw = val_out.view(-1).numpy().astype(np.float32)
    test_raw = test_out.view(-1).numpy().astype(np.float32)

    best_thr, best_score = calibrate_thresholds(val_raw, val_y_true, iters=2)
    print("Validation QWK after threshold calibration:", best_score)
    print("Chosen thresholds:", best_thr)

    test_preds = apply_thresholds(test_raw, best_thr)
    test_preds = np.clip(test_preds, 0, 4).astype(int)
else:
    val_raw = pooled_feature_mean_raw(model, val_loader)
    test_raw = pooled_feature_mean_raw(model, test_loader)

    best_thr, best_score = calibrate_thresholds(val_raw, val_y_true, iters=2)
    print("Validation QWK (pooled-feature-mean + calibrated thresholds):", best_score)
    print("Chosen thresholds:", best_thr)

    test_preds = apply_thresholds(test_raw, best_thr)
    test_preds = np.clip(test_preds, 0, 4).astype(int)

submission = test_csv.copy()
submission["diagnosis"] = test_preds
submission["diagnosis"] = submission["diagnosis"].clip(0, 4).astype(int)

if submission.shape[0] != pd.read_csv(SAMPLE_SUB_PATH).shape[0]:
    raise ValueError("Row count mismatch vs sample_submission.csv")

submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
