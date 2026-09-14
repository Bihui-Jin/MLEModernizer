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

0.9134323743353088

# 6. Current score

-0.07145

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.08817) has done: 'I make the code robust to missing weight files and to environments without a GPU. The model fall back to using pretrained ImageNet weights (instead of the unavailable fine‑tuned checkpoint) and run on CPU when CUDA isn’t available. I also guard the weight‑loading step with a try/except so the script continues even if the file is absent, and I ensure the submission DataFrame is written correctly.'
- What this solution (achieved -0.0607) has done: 'I keep the overall model and data pipeline unchanged but replace the weak regression‑based class conversion with the model’s built‑in 5‑class classifier logits. Using `argmax` on the classifier output provides a more sensible prediction than the hard‑coded thresholds, which should raise the quadratic weighted kappa toward the target while preserving the core architecture.'
- What this solution (achieved 0.12675) has done: 'I keep the model architecture unchanged but improve the prediction step by (1) applying a simple test‑time augmentation (horizontal flip) and averaging the classifier logits, (2) converting logits to class probabilities with softmax and using the expected value rounded to the nearest integer instead of a raw argmax. This small change usually yields more calibrated predictions and should raise the quadratic weighted kappa toward the target while preserving the core logic.'
- What this solution (achieved 0.08677) has done: 'I add a lightweight vertical‑flip test‑time augmentation and combine the classifier’s argmax prediction with the regression output (both averaged over the augmentations) to produce a more calibrated integer diagnosis. This keeps the original model unchanged, only expands the inference step, and should raise the quadratic weighted‑kappa toward the target while still writing a valid submission.csv.'
- What this solution (achieved 0.60072) has done: 'The fix adds a lightweight training step that runs only when the fine‑tuned checkpoint is missing.  
It loads the training images, fine‑tunes the same `ThreeStage_Model` (keeping the backbone unchanged) for a few epochs using cross‑entropy loss, then uses the now‑trained classifier for predictions (dropping the noisy regression‑based averaging).  
A flag tracks whether pretrained weights were loaded, and the script still writes a valid `submission.csv` at the end.'
- What this solution (achieved 0.0) has done: 'I increase the quick‑fine‑tuning from 2 to 5 epochs (still short enough) and, during inference, combine the classifier logits with the regression‑based class probabilities. The combined probabilities are used to compute an expected value that is rounded to the nearest integer class, which usually yields a better calibrated prediction and pushes the quadratic weighted kappa toward the target.'
- What this solution (achieved 0.0) has done: 'We skip the optional fine‑tuning step when the pretrained weight file is missing (the original code only uses it as a fallback) and replace the three‑pass image transforms in the test dataset with a single deterministic transform followed by cheap tensor flips. This removes the costly extra passes through Pillow and the 5‑epoch training loop, while keeping the model architecture and inference logic unchanged, so the predictions remain equivalent.'
- What this solution (achieved -0.07145) has done: 'The error arose because the code tried to squeeze a non‑existent third dimension of `r_out` after averaging, causing an `IndexError`. The fix replaces the erroneous `squeeze(2)` with a simple `squeeze()` (or `squeeze(-1)`) so the tensor correctly collapses to shape `(B,)`. This resolves the runtime crash and allows the pipeline to generate a valid `submission.csv`. No other logic is altered, preserving the original model and inference approach.'

# 9. Code solution

## === cell 0
import random
import time
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.transforms import functional as FT
from PIL import Image, ImageChops
import cv2
from sklearn.metrics import cohen_kappa_score
import timm

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True  # enable cudnn auto‑tuner for speed




## === cell 1
threshold = [0.75, 1.5, 2.5, 3.5]


def regress2class(out):
    prediction = torch.zeros(out.size(0))
    for i in range(4):
        prediction += (out.data >= threshold[i]).squeeze().cpu()
    return prediction


def ordinal2class_prob(out):
    pred_prob = torch.zeros(out.size(0), 5).to(device)
    pred_prob[:, 0] = (1 - out[:, 0]).squeeze()
    pred_prob[:, 1] = (out[:, 0] * (1 - out[:, 1])).squeeze()
    pred_prob[:, 2] = (out[:, 1] * (1 - out[:, 2])).squeeze()
    pred_prob[:, 3] = (out[:, 2] * (1 - out[:, 3])).squeeze()
    pred_prob[:, 4] = out[:, 3].squeeze()
    return F.softmax(pred_prob, dim=1)


def regress2class_prob(out):
    out = out.squeeze()
    batch = out.size(0)
    floor = torch.floor(out).long()
    ceil = torch.ceil(out).long()
    prob = torch.zeros((batch, 5), device=out.device)

    mask_last = out >= 4.0
    not_last = ~mask_last

    f = floor[not_last]
    c = ceil[not_last]
    o = out[not_last]
    prob[not_last, f] = 1 - (o - f.float())
    prob[not_last, c] = 1 - (c.float() - o)

    prob[mask_last, 4] = 1.0

    return F.softmax(prob, dim=1)




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
        self.backbone = timm.create_model("tf_efficientnet_b5_ns", pretrained=True)
        self.backbone.global_pool = GeM(flatten=True)
        self.regressor = nn.Linear(1000, 1)

    def forward(self, x):
        x = self.backbone(x)
        out = self.regressor(x)
        out = torch.sigmoid(out) * 4.5
        return out


class ThreeStage_Model(nn.Module):
    def __init__(self):
        super(ThreeStage_Model, self).__init__()
        self.backbone = timm.create_model("tf_efficientnet_b4_ns", pretrained=True)
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
class trim(object):
    def __call__(self, image):
        bg = Image.new(image.mode, image.size, image.getpixel((0, 0)))
        diff = ImageChops.difference(image, bg)
        diff = ImageChops.add(diff, diff, 2.0, -10)
        bbox = diff.getbbox()
        if bbox:
            return image.crop(bbox)
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


input_size = 380
train_transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(
            (input_size * 3 // 4, input_size), scale=(0.8, 1.0)
        ),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.384, 0.258, 0.174], std=[0.124, 0.089, 0.094]),
    ]
)

test_transform = transforms.Compose(
    [
        trim(),
        cropTo4_3(),
        transforms.Resize((input_size * 3 // 4, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.384, 0.258, 0.174], std=[0.124, 0.089, 0.094]),
    ]
)


class AptosDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = f"{self.img_dir}/{row['id_code']}.png"
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        label = int(row["diagnosis"])
        return image, label


net = ThreeStage_Model()
weight_path = "../input/weights/B4_3stage_4epoch_finetune.pkl"
weights_loaded = False
try:
    net.load_state_dict(torch.load(weight_path, map_location=device))
    print("Loaded fine‑tuned weights.")
    weights_loaded = True
except Exception as e:
    print(
        f"Fine‑tuned weights not found or failed to load ({e}); will use pretrained model."
    )

net = net.to(device)
net.eval()

if not weights_loaded:
    print("Proceeding with pretrained model only (no additional fine‑tuning).")




## === cell 4
class TestDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform):
        self.ids = pd.read_csv(csv_path)["id_code"].tolist()
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id_code = self.ids[idx]
        img_path = f"{self.img_dir}/{id_code}.png"
        img = Image.open(img_path).convert("RGB")
        base = self.transform(img)  # (C, H, W)
        hflip = torch.flip(base, dims=[2])  # horizontal flip
        vflip = torch.flip(base, dims=[1])  # vertical flip
        return base, hflip, vflip, id_code


test_csv = "../input/aptos2019-blindness-detection/test.csv"
test_img_dir = "../input/aptos2019-blindness-detection/test_images"
test_dataset = TestDataset(test_csv, test_img_dir, test_transform)
test_loader = DataLoader(
    test_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True
)

submission = []
class_indices = torch.arange(5, dtype=torch.float32, device=device)

for batch_idx, (orig, hflip, vflip, id_codes) in enumerate(test_loader):
    batch = torch.stack([orig, hflip, vflip], dim=1).to(device)  # (B,3,C,H,W)
    B, N, C, H, W = batch.shape
    batch_flat = batch.view(B * N, C, H, W)

    with torch.no_grad():
        c_out, r_out, _ = net(batch_flat)  # (B*3, 5) and (B*3, 1)
        c_out = c_out.view(B, N, -1)  # (B,3,5)
        r_out = r_out.view(B, N, -1)  # (B,3,1)

    c_out_avg = c_out.mean(dim=1)  # (B,5)
    probs_clf = torch.softmax(c_out_avg, dim=1)

    r_out_avg = r_out.mean(dim=1).squeeze()  # (B,)
    probs_reg = regress2class_prob(r_out_avg)

    combined_probs = (probs_clf + probs_reg) / 2.0
    expected = torch.sum(combined_probs * class_indices, dim=1)
    pred_class = torch.clamp(torch.round(expected), 0, 4).int().cpu().numpy()

    for id_code, pred in zip(id_codes, pred_class):
        submission.append([id_code, int(pred)])

    if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(test_loader):
        print(
            f"Processed { (batch_idx + 1) * test_loader.batch_size } / {len(test_dataset)} images"
        )

submission = np.array(submission)




## === cell 5
df = pd.DataFrame(submission, columns=["id_code", "diagnosis"])
if df.empty:
    print("Warning: submission is empty. Writing a fallback submission with class 0.")
    fallback = pd.read_csv(
        "../input/aptos2019-blindness-detection/sample_submission.csv"
    )
    fallback["diagnosis"] = 0
    df = fallback

df.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")
