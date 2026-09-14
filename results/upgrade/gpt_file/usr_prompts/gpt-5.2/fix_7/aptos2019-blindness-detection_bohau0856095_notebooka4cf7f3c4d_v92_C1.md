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

0.9210502495724562

# 6. Current score

0.11904

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -0.02083) has done: 'I (1) remove the dependency on a missing `../input/weights/...pkl` by loading weights only if they exist and otherwise falling back to a simple, deterministic heuristic so a non-empty submission is always produced. I (2) make device selection robust by using CUDA only when available, fixing the “no NVIDIA driver” crash. I (3) fix a couple of transform bugs (`trim` sometimes returning `None`, and `is` vs `==` in the hue check) that can silently break preprocessing. These changes are execution/stability focused (since no valid score was yielded yet) and generate a valid `submission.csv` in the correct format.'
- What this solution (achieved 0.01787) has done: 'Your current score is extremely low because the pipeline usually falls back to the heuristic (no weights found), which produces near-random labels for QWK. To move the score toward your target with minimal core-logic change, I (1) ensure the script actually uses the provided EfficientNet model by enabling `pretrained=True` when no finetuned weights are found, and (2) switch inference to use the model’s `final=True` head (the architecture already defines it) so predictions leverage classifier+regressor+ordinal jointly instead of only the regressor. I keep the same preprocessing and the same `regress2class` thresholding to preserve evaluation semantics. The result remains a single `submission.csv` with the correct columns and ordering.'
- What this solution (achieved -0.04581) has done: 'Your current score is far below the target because inference is effectively using a randomly-initialized head when finetuned weights are missing: you replace only `net.backbone` with an ImageNet-pretrained model but keep the (random) classifier/regressor/ordinal and then call `final=True`, which depends heavily on those random heads. To move the score upward with minimal change and preserved architecture, I (1) build the whole `ThreeStage_Model` backbone as pretrained when no finetuned weights exist, and (2) when weights are missing, avoid the random `final_regressor` by deriving the final continuous score from the backbone outputs (`c_out`, `r_out`, `o_out`) using only fixed post-processing (expected value of class probs and ordinal probs plus the regressor), then convert with your existing `regress2class` thresholds. I also ensure the same resizing is used for EfficientNet-B4 (384) to better match its pretraining, which is a small, metric-aligned preprocessing correction rather than a logic rewrite. The script still run end-to-end and always write a valid `submission.csv` with the required columns and ordering.'
- What this solution (achieved 0.01221) has done: 'Your score is very low because when finetuned weights are missing you’re effectively using a randomly-initialized classifier/regressor/ordinal head (only the backbone is pretrained), so the produced labels are close to noise for QWK. To move the score upward with minimal change and without altering the model/training logic, I (1) make the entire `ThreeStage_Model` ImageNet-pretrained when no finetuned weights are found (not just the backbone), and (2) use the more stable regressor-only path for predictions in that case (since it attaches to pretrained features and avoids random multi-head mixing). I also ensure inference uses the EfficientNet-B4 native input size (380) instead of 384 to better match pretraining, which is a small preprocessing alignment. Everything still runs end-to-end and writes a valid `submission.csv` with the required columns/order.'
- What this solution (achieved -0.01313) has done: 'Your current score is far below the target because in the common “no finetuned weights” case you only swap in an ImageNet-pretrained backbone while keeping randomly-initialized heads, then you use the regressor head alone; that produces near-noise ordinal labels for QWK. To move the score upward with minimal semantic change, I (1) load the full `ThreeStage_Model` backbone from timm with `pretrained=True` directly (same architecture, just correct weight init across the model’s feature extractor), and (2) when finetuned weights are missing, compute a more stable continuous score by averaging the regressor output with the expected value from the classifier logits (while keeping your existing `regress2class` thresholds and no retraining). I also batch inference via a `DataLoader` (same transforms/prediction logic) to reduce per-image overhead and reduce the chance of sporadic PIL/CUDA hiccups that can trigger the heuristic fallback. The script still run end-to-end and write a valid `submission.csv` with the required columns and row order.'
- What this solution (achieved 0.11904) has done: 'Your score is far below the target, and the main reason is that when finetuned weights are missing you’re effectively using a randomly-initialized classifier/regressor head on top of an ImageNet-pretrained backbone, which yields near-random ordinal labels under QWK. To improve toward the target with minimal semantic change, I (1) switch the model to use a timm EfficientNet created with `num_classes=0` so we extract stable pretrained features (instead of a random 1000-way head output), and (2) adjust the linear head input dimensions from 1000 to the true backbone feature dimension. I keep your same transforms, thresholds, and overall inference flow, but in the “no finetuned weights” case I use regressor-only output (more stable than mixing with random classifier logits). This preserves your core architecture/training approach while making the “no weights available” path produce meaningful predictions rather than noise, which should move QWK upward substantially.'

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

import torchvision.transforms as transforms
from torchvision.transforms import functional as FT
from torch.utils.data import Dataset, DataLoader

from PIL import Image, ImageChops
import cv2

import timm

device = "cuda" if torch.cuda.is_available() else "cpu"

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)



## === cell 1
threshold = [0.75, 1.5, 2.5, 3.5]


def regress2class(out):
    prediction = torch.zeros(out.size(0), device="cpu")
    out_cpu = out.detach().view(-1).cpu()
    for i in range(4):
        prediction += (out_cpu >= threshold[i]).to(torch.float32)
    return prediction


def ordinal2class_prob(out):
    pred_prob = torch.zeros(out.size(0), 5, device=out.device)
    pred_prob[:, 0] = (1 - out[:, 0]).squeeze()
    pred_prob[:, 1] = (out[:, 0] * (1 - out[:, 1])).squeeze()
    pred_prob[:, 2] = (out[:, 1] * (1 - out[:, 2])).squeeze()
    pred_prob[:, 3] = (out[:, 2] * (1 - out[:, 3])).squeeze()
    pred_prob[:, 4] = out[:, 3].squeeze()
    return F.softmax(pred_prob, dim=1)


def regress2class_prob(out):
    pred_prob = torch.zeros((out.size(0), 5), device=out.device)
    out = out.view(-1)
    for i in range(out.size(0)):
        if out[i] < 4.0:
            l1 = int(math.floor(float(out[i])))
            l2 = int(math.ceil(float(out[i])))
            pred_prob[i][l1] = 1 - (out[i] - l1)
            pred_prob[i][l2] = 1 - (l2 - out[i])
        else:
            pred_prob[i][4] = 1.0
    return pred_prob




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
        self.backbone = timm.create_model(
            "tf_efficientnet_b5_ns", pretrained=False, num_classes=0, global_pool=""
        )
        self.backbone.global_pool = GeM(flatten=True)

        feat_dim = self.backbone.num_features
        self.regressor = nn.Linear(feat_dim, 1)

    def forward(self, x):
        x = self.backbone(x)
        out = self.regressor(x)
        out = torch.sigmoid(out) * 4.5
        return out


class ThreeStage_Model(nn.Module):
    def __init__(self, backbone=None):
        super(ThreeStage_Model, self).__init__()

        self.backbone = timm.create_model(
            "tf_efficientnet_b4_ns", pretrained=False, num_classes=0, global_pool=""
        )
        self.backbone.global_pool = GeM(flatten=True)

        feat_dim = self.backbone.num_features

        self.classifier = nn.Sequential(
            nn.SiLU(),
            nn.Linear(feat_dim, 500),
            nn.SiLU(),
            nn.Linear(500, 5),
        )

        self.regressor = nn.Sequential(
            nn.SiLU(),
            nn.Linear(feat_dim, 500),
            nn.SiLU(),
            nn.Linear(500, 1),
        )

        self.ordinal = nn.Sequential(
            nn.SiLU(),
            nn.Linear(feat_dim, 500),
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
BASE = "../input/aptos2019-blindness-detection"
TEST_CSV = os.path.join(BASE, "test.csv")
TEST_IMG_DIR = os.path.join(BASE, "test_images")

test_ids = pd.read_csv(TEST_CSV)["id_code"].values

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

WEIGHT_CANDIDATES = [
    "../input/weights/B4_3stage_5epoch_finetune3.pkl",
    "../input/weights/B4_3stage_5epoch_finetune3.pth",
    "../input/weights/B4_3stage_5epoch_finetune3.pt",
]

net = ThreeStage_Model().to(device)
net.eval()

weights_path = None
for p in WEIGHT_CANDIDATES:
    if os.path.exists(p):
        weights_path = p
        break

use_model = False
has_finetuned_weights = False
if weights_path is not None:
    try:
        state = torch.load(weights_path, map_location=device)
        if (
            isinstance(state, dict)
            and all(isinstance(k, str) for k in state.keys())
            and any(k.startswith("backbone.") for k in state.keys())
        ):
            net.load_state_dict(state, strict=True)
        elif isinstance(state, dict) and "state_dict" in state:
            net.load_state_dict(state["state_dict"], strict=True)
        else:
            net.load_state_dict(state, strict=True)
        use_model = True
        has_finetuned_weights = True
        print(f"Loaded weights from: {weights_path}")
    except Exception as e:
        print(
            f"WARNING: found weights at {weights_path} but failed to load ({type(e).__name__}: {e}). Will use ImageNet-pretrained weights instead."
        )
        has_finetuned_weights = False
        use_model = True
else:
    print(
        "WARNING: no finetuned weights found in ../input/weights/. Using ImageNet-pretrained weights."
    )
    use_model = True

if use_model and (not has_finetuned_weights):
    net.backbone = timm.create_model(
        "tf_efficientnet_b4_ns", pretrained=True, num_classes=0, global_pool=""
    )
    net.backbone.global_pool = GeM(flatten=True)
    net = net.to(device)
    net.eval()




## === cell 5
def heuristic_predict_label_from_image(pil_img):
    """
    Fallback predictor: uses simple brightness/contrast statistics to assign 0-4.
    This is only used if model inference fails for some reason, to ensure a valid submission.
    """
    img = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    m = float(gray.mean())
    s = float(gray.std())

    score = 0
    if m < 55:
        score += 2
    elif m < 80:
        score += 1

    if s < 25:
        score += 1
    if s < 15:
        score += 1

    return int(np.clip(score, 0, 4))


def continuous_from_classifier_and_regressor(c_out, r_out):
    c_prob = F.softmax(c_out, dim=1)
    cls_expect = (
        c_prob * torch.arange(5, device=c_out.device, dtype=c_prob.dtype).view(1, -1)
    ).sum(dim=1, keepdim=True)
    return (cls_expect + r_out) / 2.0


class TestDataset(Dataset):
    def __init__(self, ids, img_dir, transform):
        self.ids = list(ids)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        idx = self.ids[i]
        image_name = os.path.join(self.img_dir, f"{idx}.png")
        img_pil = Image.open(image_name).convert("RGB")
        x = self.transform(img_pil)
        return idx, x


dataset = TestDataset(test_ids, TEST_IMG_DIR, transform)
loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)

submission = []
net.eval()
with torch.no_grad():
    if use_model:
        for batch_ids, batch_x in loader:
            try:
                batch_x = batch_x.to(device, non_blocking=True)
                if has_finetuned_weights:
                    out = net(batch_x, final=True)
                else:
                    _, r_out, _ = net(batch_x, final=False)
                    out = r_out
                preds = regress2class(out.data.squeeze(1)).numpy().astype(int).tolist()
                submission.extend([[bid, int(p)] for bid, p in zip(batch_ids, preds)])
            except Exception:
                for bid, x_cpu in zip(batch_ids, batch_x.cpu()):
                    try:
                        x1 = x_cpu.unsqueeze(0).to(device)
                        if has_finetuned_weights:
                            out1 = net(x1, final=True)
                        else:
                            _, r1, _ = net(x1, final=False)
                            out1 = r1
                        p1 = regress2class(out1.data.squeeze(1))[0].item()
                        submission.append([bid, int(p1)])
                    except Exception:
                        img_pil = Image.open(
                            os.path.join(TEST_IMG_DIR, f"{bid}.png")
                        ).convert("RGB")
                        submission.append(
                            [bid, heuristic_predict_label_from_image(img_pil)]
                        )
    else:
        for idx in test_ids:
            img_pil = Image.open(os.path.join(TEST_IMG_DIR, f"{idx}.png")).convert(
                "RGB"
            )
            submission.append([idx, heuristic_predict_label_from_image(img_pil)])

submission = np.array(submission, dtype=object)



## === cell 6
df = pd.DataFrame(submission, columns=["id_code", "diagnosis"])
df["id_code"] = df["id_code"].astype(str)
df["diagnosis"] = df["diagnosis"].astype(int)

sample_path = os.path.join(BASE, "sample_submission.csv")
if os.path.exists(sample_path):
    sample = pd.read_csv(sample_path)
    df = sample[["id_code"]].merge(df, on="id_code", how="left")
    df["diagnosis"] = df["diagnosis"].fillna(0).astype(int)

out_path = "submission.csv"
df.to_csv(out_path, index=False)
print(f"Wrote {out_path} with shape {df.shape} and columns {list(df.columns)}")
print(df.head())
