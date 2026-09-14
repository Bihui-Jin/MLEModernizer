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

3.9

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
timm==1.0.19
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124

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

0.9143358388025438

# 6. Current score

-0.07332

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the missing-weights crash by loading the model weights only if the file exists; if not, the code still run end-to-end using the untrained model (score be poor but a valid submission be produced). I also fix the hardcoded CUDA device that fails on CPU-only Kaggle runtimes by selecting `cuda` only when available and ensuring tensors are created on the correct device. Finally, I prevent the transform pipeline from returning `None` (your `trim()` sometimes returns nothing) and vectorize/batch inference via a `Dataset/DataLoader` so the submission is guaranteed non-empty and generated reliably within time.'
- What this solution (achieved -0.03395) has done: 'Your current 0.0 score is consistent with the script silently running with randomly initialized weights (because `../input/weights/...` does not exist in this dataset layout). The smallest change that legitimately improves score toward your target is to load pretrained backbone weights (ImageNet) so inference is meaningful without needing external weight files, while keeping the exact same model heads, forward logic, transforms, and thresholding. I also keep submission ordering and format identical, and keep runtime safe by using a modest batch size and workers. This should move the score upward substantially (toward the 0.91 target band) without changing the core approach.'
- What this solution (achieved -0.0774) has done: 'Your current score is far below the target, and the main reason is that you are not using the model’s intended prediction head: you’re discarding the strong classifier/ordinal outputs and only thresholding the regressor, which is poorly calibrated without the missing competition-specific weights. To move the score upward with minimal change (no training, same architecture), I keep your model and transforms but switch inference to use the existing classifier logits (argmax) as the primary diagnosis prediction, which is meaningful with ImageNet-pretrained features even without the saved weights. I also keep everything deterministic and preserve the exact submission format and ordering. This should substantially increase QWK versus the current near-random performance while staying within your constraints.'
- What this solution (achieved -0.06511) has done: 'Your current score is far below the target, so we should improve legitimately with minimal semantic changes. The biggest issue is that without the missing competition-specific weights, your classifier head is essentially random even if the backbone is ImageNet-pretrained; the regressor/ordinal outputs also won’t be calibrated. A small, metric-aligned improvement that keeps your exact model and transforms is to (1) add test-time augmentation (horizontal flip) and average the *probabilities* from the classifier head, and (2) apply a tiny, safe amount of label-smoothing via probability averaging (not changing training) to reduce overconfident random spikes. This should move QWK upward versus the current argmax-on-random-head behavior while keeping runtime under 600s and producing the same submission schema.'
- What this solution (achieved 0.02904) has done: 'Your score is far below the target, and the main cause is that without the intended competition weights your randomly initialized heads (classifier/regressor/ordinal/final_regressor) dominate the prediction. To improve legitimately while preserving your architecture and inference flow, I keep using the classifier head but calibrate its output by blending in the regressor-derived class probabilities (which at least has a meaningful ImageNet-pretrained backbone + simple head) and apply a monotonic temperature scaling to reduce overconfident noise. I keep your existing TTA (hflip) and smoothing, but make the smoothing smaller and add deterministic settings so the submission is stable across runs. These changes are minimal, metric-aligned (better ordinal consistency), and should move QWK upward toward your target band.'
- What this solution (achieved 0.02904) has done: 'Your current score (0.02904) is far below the target (0.9143), so we should improve legitimately with minimal semantic changes; the biggest issue is that your heads are still random without the missing competition weights, so blending/temperature/TTA can’t recover much. The smallest, high-impact fix that keeps your model architecture and inference flow is to *fit the 4 regression thresholds on the training labels distribution* (quantile-based) so your regressor-to-class mapping matches the dataset’s ordinal class balance, then use those learned thresholds consistently at inference. This keeps the same model, same forward pass, and same “regress then discretize” semantics, but makes the discretization far less arbitrary than the hardcoded `[0.75, 1.5, 2.5, 3.5]`. I keep your existing classifier/regressor probability blending and TTA, only swapping in the learned thresholds to make the regressor probabilities meaningful and more ordinal-consistent for QWK.'
- What this solution (achieved 0.02904) has done: 'Your current score is far below the target, so we should legitimately improve the signal with minimal semantic change. The main problem is that without the competition-trained weights, the classifier/regressor/ordinal heads are essentially random, and blending them cannot recover meaningful predictions. Keeping your exact model architecture and inference flow, the smallest high-impact change is to load a widely-available pretrained checkpoint for the same backbone (EfficientNet-B4 NoisyStudent) from timm and apply it **only** to the backbone with `strict=False`, leaving your heads untouched. This gives much stronger features so the existing heads’ outputs become less noisy, and we keep your thresholds, TTA, blending, and submission formatting unchanged.'
- What this solution (achieved -0.04042) has done: 'Your current score is far below the target, so we should make a small, metric-aligned change that improves ordinal consistency without changing your model or adding training. I keep your exact architecture, transforms, TTA, and probability blending, but replace the final `argmax` with an expected-value regression from the blended class probabilities and then discretize using your already-learned thresholds (same “regress then discretize” semantics). This tends to improve QWK because it respects class order and reduces noisy class flips. I also calibrate the blend weight automatically based on how peaked the classifier probabilities are (still deterministic, no training), which can help when the random head is overconfident.'
- What this solution (achieved 0.0) has done: 'Your current score is far below the target, and the main reason is that with missing competition-specific weights the randomly initialized heads dominate, so predictions are near-random. With minimal change and preserving your model/loops, I keep your inference pipeline but switch the discretization to a simple, deterministic “dataset-prior” mapping: output the train-set class distribution for every test image (majority-class sampling would add noise, so we use the mode for stability). This is a legitimate baseline that often yields a non-negative QWK and should move you substantially upward from -0.04 without changing architecture, training, transforms, or I/O paths. The rest of the code (dataset, dataloader, model creation) is left intact to satisfy your constraints and ensure a valid submission is always produced.'
- What this solution (achieved -0.07332) has done: 'Your current 0.0 score comes from predicting the same majority class for every test image, which is a very weak baseline for QWK. To move the score upward toward your target while keeping your model, transforms, and no-training approach intact, I switch inference back to using the model’s outputs (classifier + regressor + ordinal) with your existing learned thresholds and an ordinal-consistent expected-value discretization. I also add a minimal, deterministic horizontal-flip TTA and average predictions, which typically stabilizes ordinal predictions without changing the core modeling logic. The submission format, ordering, and file path remain identical.'

# 9. Code solution

## === cell 0
import os
import random
import time
import math
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms
from torchvision.transforms import functional as FT

from PIL import Image, ImageChops

import timm

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)



## === cell 1
threshold = [0.75, 1.5, 2.5, 3.5]


def regress2class(out):
    prediction = torch.zeros(out.size(0), device=out.device, dtype=torch.long)
    for i in range(4):
        prediction += (out >= threshold[i]).long()
    return prediction


def ordinal2class_prob(out):
    pred_prob = torch.zeros(out.size(0), 5, device=out.device, dtype=out.dtype)
    pred_prob[:, 0] = (1 - out[:, 0]).squeeze()
    pred_prob[:, 1] = (out[:, 0] * (1 - out[:, 1])).squeeze()
    pred_prob[:, 2] = (out[:, 1] * (1 - out[:, 2])).squeeze()
    pred_prob[:, 3] = (out[:, 2] * (1 - out[:, 3])).squeeze()
    pred_prob[:, 4] = out[:, 3].squeeze()
    return F.softmax(pred_prob, dim=1)


def regress2class_prob(out):
    """
    Convert continuous regression output in [0, 4.5] into a soft 5-class distribution
    by linear interpolation between neighboring integer classes.
    """
    out = out.clamp(0.0, 4.5)
    out_floor = torch.floor(out).to(torch.long)
    out_ceil = torch.ceil(out).to(torch.long)
    out_floor = out_floor.clamp(0, 4)
    out_ceil = out_ceil.clamp(0, 4)

    pred_prob = torch.zeros((out.size(0), 5), device=out.device, dtype=out.dtype)

    same = out_floor == out_ceil
    if same.any():
        pred_prob[same, out_floor[same]] = 1.0

    diff = ~same
    if diff.any():
        of = out_floor[diff]
        oc = out_ceil[diff]
        w2 = out[diff] - of.to(out.dtype)  # weight for ceil class
        w1 = 1.0 - w2  # weight for floor class
        pred_prob[diff, of] = w1
        pred_prob[diff, oc] = w2

    return pred_prob


def prob2expected_reg(p):
    """
    Convert 5-class probabilities into an ordinal expected value on the same 0..4.5
    regression scale used by thresholds.
    """
    class_values = torch.tensor(
        [0.0, 1.0, 2.0, 3.0, 4.0], device=p.device, dtype=p.dtype
    )
    exp_class = (p * class_values[None, :]).sum(dim=1)  # 0..4
    exp_reg = exp_class * (4.5 / 4.0)  # 0..4.5
    return exp_reg




## === cell 2
def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, flatten=False):
        super(GeM, self).__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps
        self.flatten = flatten

    def forward(self, x):
        x = gem(x, p=self.p, eps=self.eps)
        if self.flatten:
            x = x.flatten(1)
        return x

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "p="
            + "{:.4f}".format(self.p.data.tolist()[0])
            + ", "
            + "eps="
            + str(self.eps)
            + ")"
        )


class Regressor(nn.Module):
    def __init__(self):
        super(Regressor, self).__init__()
        self.backbone = timm.models.tf_efficientnet_b5_ns(pretrained=True)
        self.backbone.global_pool = GeM(flatten=True)
        self.regressor = nn.Linear(1000, 1)

    def forward(self, x):
        x = self.backbone(x)
        out = self.regressor(x)
        out = torch.sigmoid(out) * 4.5
        return out


class ThreeStage_Model(nn.Module):
    def __init__(self, backbone=None):
        super(ThreeStage_Model, self).__init__()

        self.backbone = timm.models.tf_efficientnet_b4_ns(pretrained=True)
        self.backbone.global_pool = GeM(flatten=True)

        self.classifier = nn.Sequential(
            nn.SiLU(),
            nn.Linear(1000, 500),
            nn.SiLU(),
            nn.Linear(500, 5),
        )

        self.regressor = nn.Sequential(
            nn.SiLU(),
            nn.Linear(1000, 500),
            nn.SiLU(),
            nn.Linear(500, 1),
        )

        self.ordinal = nn.Sequential(
            nn.SiLU(),
            nn.Linear(1000, 500),
            nn.SiLU(),
            nn.Linear(500, 4),
        )

        self.final_regressor = nn.Sequential(
            nn.SiLU(),
            nn.Linear(10, 1),
        )

    def forward(self, x, final=False):
        x = self.backbone(x)

        c_out = self.classifier(x)
        r_out = self.regressor(x)
        o_out = self.ordinal(x)

        if final:
            out = torch.cat((c_out, r_out, o_out), 1)
            out = self.final_regressor(out)
            out = torch.sigmoid(out) * 4.5
            return out
        else:
            r_out = torch.sigmoid(r_out) * 4.5
            o_out = torch.sigmoid(o_out)
            return c_out, r_out, o_out




## === cell 3
class photometric_distort(object):
    def __call__(self, image):
        distortions = [
            FT.adjust_brightness,
            FT.adjust_contrast,
            FT.adjust_saturation,
            FT.adjust_hue,
        ]
        random.shuffle(distortions)

        for d in distortions:
            if random.random() < 0.5:
                if d.__name__ == "adjust_hue":
                    adjust_factor = random.uniform(-16 / 255.0, 16 / 255.0)
                else:
                    adjust_factor = random.uniform(0.7, 1.3)
                image = d(image, adjust_factor)
        return image


class cropTo4_3(object):
    def __call__(self, image):
        w, h = image.size
        if (w / h) >= (4 / 3):
            new_h = h
            new_w = int(h * 4 / 3)
        else:
            new_h = int(w * 3 / 4)
            new_w = w

        left = (w - new_w) / 2
        top = (h - new_h) / 2
        right = left + new_w
        bottom = top + new_h

        return image.crop((left, top, right, bottom))


class trim(object):
    def __call__(self, image):
        bg = Image.new(image.mode, image.size, image.getpixel((0, 0)))
        diff = ImageChops.difference(image, bg)
        diff = ImageChops.add(diff, diff, 2.0, -10)
        bbox = diff.getbbox()
        if bbox:
            return image.crop(bbox)
        return image




## === cell 4
BASE_PATH = "../input/aptos2019-blindness-detection"
TEST_CSV = os.path.join(BASE_PATH, "test.csv")
TEST_IMG_DIR = os.path.join(BASE_PATH, "test_images")
TRAIN_CSV = os.path.join(BASE_PATH, "train.csv")

test_df = pd.read_csv(TEST_CSV)
test_ids = test_df["id_code"].values.tolist()

train_df = pd.read_csv(TRAIN_CSV)
y = train_df["diagnosis"].astype(int).values
counts = np.bincount(y, minlength=5).astype(np.float64)
cum = np.cumsum(counts) / counts.sum()
cum_boundaries = [cum[0], cum[1], cum[2], cum[3]]  # P(y<=k)
class_axis_thresholds = np.array([1.0, 2.0, 3.0, 4.0])  # boundaries on 0..4 axis
scale = 4.5 / 4.0
prevalence_scaled = np.array(cum_boundaries) * 4.5
fixed_scaled = class_axis_thresholds * scale
learned = 0.65 * fixed_scaled + 0.35 * prevalence_scaled
threshold = learned.tolist()
print("Learned thresholds (reg scale):", threshold)

input_size = 380
transform = transforms.Compose(
    [
        trim(),
        cropTo4_3(),
        transforms.Resize((input_size * 3 // 4, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.384, 0.258, 0.174], std=[0.124, 0.089, 0.094]),
    ]
)

net = ThreeStage_Model()

WEIGHTS_PATH = "../input/weights/B4_3stage_51epoch_CLAHE.pkl"
if os.path.exists(WEIGHTS_PATH):
    state = torch.load(WEIGHTS_PATH, map_location="cpu")
    net.load_state_dict(state)
    print("Loaded weights:", WEIGHTS_PATH)
else:
    print("WARNING: Weights not found at:", WEIGHTS_PATH)
    print("Proceeding without competition weights.")

net = net.to(device)
net.eval()




## === cell 5
class APTOSImageDataset(Dataset):
    def __init__(self, ids, img_dir, transform=None):
        self.ids = ids
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id_code = self.ids[idx]
        img_path = os.path.join(self.img_dir, f"{id_code}.png")
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return id_code, img


ds = APTOSImageDataset(test_ids, TEST_IMG_DIR, transform=transform)

loader = DataLoader(
    ds,
    batch_size=8 if device.type == "cuda" else 4,
    shuffle=False,
    num_workers=2,
    pin_memory=(device.type == "cuda"),
)

print("Train label distribution:", counts.astype(int).tolist())

all_ids = []
all_preds = []

tta_hflip = True
p_blend = 0.70  # emphasize classifier logits (more robust than random regressor head); keep fixed/deterministic
temperature = 1.25  # soften overconfident logits a bit; deterministic

with torch.no_grad():
    for batch_ids, batch_imgs in loader:
        batch_imgs = batch_imgs.to(device, non_blocking=True)
        all_ids.extend(list(batch_ids))

        probs_accum = None
        n_views = 0

        for do_flip in [False, True] if tta_hflip else [False]:
            imgs = torch.flip(batch_imgs, dims=[3]) if do_flip else batch_imgs

            c_out, r_out, o_out = net(imgs, final=False)

            c_prob = F.softmax(c_out / temperature, dim=1)
            r_prob = regress2class_prob(r_out.squeeze(1))
            o_prob = ordinal2class_prob(o_out)

            p = p_blend * c_prob + (1.0 - p_blend) * (0.65 * o_prob + 0.35 * r_prob)

            probs_accum = p if probs_accum is None else (probs_accum + p)
            n_views += 1

        probs = probs_accum / float(n_views)

        exp_reg = prob2expected_reg(probs)
        pred = regress2class(exp_reg)

        all_preds.extend(pred.detach().cpu().numpy().astype(int).tolist())

submission_df = pd.DataFrame({"id_code": all_ids, "diagnosis": all_preds})
submission_df = submission_df.set_index("id_code").loc[test_ids].reset_index()

print("Submission shape:", submission_df.shape)
print(submission_df.head())



## === cell 6
out_path = "submission.csv"
submission_df.to_csv(out_path, index=False)

assert os.path.exists(out_path) and out_path.endswith(".csv")
assert submission_df.shape[0] == len(test_ids)
assert list(submission_df.columns) == ["id_code", "diagnosis"]
print("Wrote:", out_path)
