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

0.8985926606060796

# 6. Current score

-0.04804

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -0.02066) has done: 'Implemented device‑auto‑selection and safe weight loading (fallback to pretrained backbone when the custom checkpoint is missing). Added required imports and handled missing weight file gracefully, ensuring the model runs on CPU if no GPU is available. The rest of the pipeline remains unchanged, now producing a non‑empty `submission.csv` that follows the required format.'
- What this solution (achieved -0.05666) has done: 'I replace the custom threshold‑based conversion with a simple rounding of the regression output (clipped to the 0‑4 range). This aligns the post‑processing more directly with the intended label scale and should raise the quadratic weighted kappa toward the target without altering the model architecture or training logic.'
- What this solution (achieved 0.05432) has done: 'I adjust the conversion from the regression output to class labels to use the predefined threshold list instead of simple rounding. This aligns the post‑processing with the intended rating boundaries and should move the quadratic weighted kappa toward the target score while keeping the core model unchanged.'
- What this solution (achieved -0.09706) has done: 'I modify the inference step to use the model’s built‑in classifier head (the 5‑dimensional logits) instead of the regression output that was previously bucketized. Taking the `argmax` of the classifier gives a direct class prediction (0‑4) and requires only a one‑line change, preserving the existing architecture and training logic while likely boosting the quadratic weighted kappa toward the target. The rest of the pipeline remains unchanged.'
- What this solution (achieved -0.01482) has done: 'I switch the inference to use the model’s final regression head (which combines classifier, regressor and ordinal outputs) and then convert that regression value to a class with the existing `regress2class` thresholds. Additionally, I apply a simple horizontal‑flip test‑time augmentation and average the two regression outputs before bucketizing – a minimal change that keeps the core architecture unchanged while aiming to raise the quadratic weighted kappa toward the target.'
- What this solution (achieved 0.01886) has done: 'I replace the regression‑based inference with a direct use of the model’s classifier logits, averaging the softmax probabilities from the original and horizontally‑flipped image. This keeps the architecture unchanged but aligns predictions with the intended 0‑4 classes, which should raise the quadratic weighted kappa toward the target. The only modification is in the inference loop (cell 6) where we now obtain `c_out` from the model (with `final=False`), compute softmax, average the two augmentations, and take `argmax` as the final class.'
- What this solution (achieved 0.06517) has done: 'I replace the classifier‑based inference with the regression output that is bucketized using the existing `regress2class` thresholds, averaging the original and horizontally‑flipped predictions. This keeps the model architecture unchanged while aligning predictions more directly with the ordinal label scale, which should raise the quadratic weighted‑kappa toward the target.'
- What this solution (achieved 0.22585) has done: 'I switch the inference to use the model’s 5‑class classifier logits (with soft‑max and horizontal‑flip TTA) instead of the regression output. This aligns predictions directly with the label space and is expected to raise the quadratic weighted kappa toward the target while keeping the rest of the pipeline unchanged.'
- What this solution (achieved 0.05376) has done: 'I change the inference step (cell 6) to use the model’s final regression output instead of the classifier logits. The regression prediction is averaged over the original and horizontally‑flipped image and then converted to a class label with the existing `regress2class` function. This keeps the model architecture unchanged while aligning the predictions with the ordinal label scale, which is expected to raise the quadratic weighted kappa toward the target.'
- What this solution (achieved 0.17178) has done: 'I modify the inference step (cell 6) to use the model’s 5‑class classifier logits instead of the regression head, averaging the original and horizontally‑flipped logits before taking the arg‑max. This keeps the architecture unchanged while aligning predictions directly with the label space, which should move the quadratic weighted‑kappa score closer to the target.'
- What this solution (achieved -0.04804) has done: 'I replace the inference logic to use the model’s final regression head (which combines classifier, regressor and ordinal outputs) and convert its averaged output to a class via the existing `regress2class` function. This keeps the architecture unchanged while aligning predictions directly with the ordinal label scale, which should move the quadratic weighted‑kappa score closer to the target.'

# 9. Code solution

## === cell 0
import subprocess, os, sys

wheel_path = "../input/weights/timm-0.3.1-py3-none-any.whl"
if os.path.exists(wheel_path):
    subprocess.check_call([sys.executable, "-m", "pip", "install", wheel_path])



## === cell 1
import random
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image, ImageChops
import os

from sklearn.metrics import cohen_kappa_score
import timm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



## === cell 2
threshold = [0.7, 1.5, 2.5, 3.5]


def regress2class(out):
    """
    Convert regression output to integer class using the predefined thresholds.
    This replaces the previous rounding‑based conversion to better match the
    label scale (0‑4) and is expected to improve the kappa score.
    """
    out = out.squeeze()
    th = torch.tensor(threshold, device=out.device, dtype=out.dtype)
    pred = torch.bucketize(out, th, right=False)
    pred = torch.clamp(pred, 0, 4).long()
    return pred




## === cell 3
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

        self.backbone = timm.models.tf_efficientnet_b5_ns(pretrained=False)
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

        self.backbone = timm.models.tf_efficientnet_b4_ns(pretrained=False)
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




## === cell 4
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




## === cell 5
test_ids = pd.read_csv("../input/aptos2019-blindness-detection/test.csv")
test_ids = np.squeeze(test_ids.values)

input_size = 320

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
weights_path = "../input/weights/B4_3stage_60epoch_AdamW.pkl"
if os.path.exists(weights_path):
    net.load_state_dict(torch.load(weights_path, map_location=device))
else:
    net.backbone = timm.create_model("tf_efficientnet_b4_ns", pretrained=True)
    net.backbone.global_pool = GeM(flatten=True)

net = net.to(device)
net.eval()



## === cell 6
submission = []

with torch.no_grad():
    for i, idx in enumerate(test_ids):
        print(i)
        image_path = f"../input/aptos2019-blindness-detection/test_images/{idx}.png"
        img = Image.open(image_path).convert("RGB")
        img = transform(img).unsqueeze(0).to(device)

        out_orig = net(img, final=True)
        img_flipped = torch.flip(img, dims=[3])
        out_flip = net(img_flipped, final=True)

        avg_out = (out_orig + out_flip) / 2.0

        pred = regress2class(avg_out).cpu().item()

        submission.append([idx, int(pred)])

submission = np.array(submission)



## === cell 7
df = pd.DataFrame(submission, columns=["id_code", "diagnosis"])
df.to_csv("submission.csv", index=False)
