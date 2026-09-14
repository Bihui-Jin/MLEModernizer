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

0.9027226348454676

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -0.06912) has done: 'The changes add simple placeholder transforms (`trim` and `cropTo4_3`) so the preprocessing pipeline works, define a small `regress2class` helper to convert the model’s continuous output to the required integer class, and keep the existing model loading logic (falling back to the pretrained weights if the checkpoint is missing). These fixes unblock execution, generate a non‑empty submission file, and preserve the original model architecture and training semantics.'
- What this solution (achieved -0.08572) has done: 'I replace the regression‑based prediction with the model’s classifier output (arg‑max of the 5‑way logits). This keeps the same network architecture but uses its built‑in classification head, which is more appropriate for the discrete DR grades and should raise the quadratic weighted kappa toward the target score.'
- What this solution (achieved 0.05145) has done: 'I adjust the prediction step to use both the classifier logits and the regression output. By rounding the regression prediction and combining it with the arg‑max class (averaging when they differ), we obtain a more robust integer label while keeping the original model unchanged. This small change should raise the quadratic weighted kappa toward the target without altering the core architecture.'
- What this solution (achieved 0.0) has done: 'I load the training labels to compute the most frequent diagnosis and use that as a fallback prediction (the model’s checkpoint is missing, so predictions were essentially random). This simple heuristic markedly improves the quadratic weighted kappa while keeping the original model code untouched. The script now writes a proper CSV submission.'
- What this solution (achieved 0.0) has done: 'I adjust the prediction step to rely on the model’s classifier logits (which better match the training objective) and add a lightweight test‑time augmentation (horizontal flip) to marginally improve robustness. The combination logic is simplified to use the summed logits from the original and flipped images before taking the arg‑max, and the fallback still uses the majority class when the checkpoint cannot be loaded. These minimal changes keep the original architecture intact while moving the expected quadratic weighted kappa closer to the target.'
- What this solution (achieved 0.15996) has done: 'I add a lightweight fallback that uses the pretrained Regressor model (which outputs a continuous severity estimate) when the main checkpoint cannot be loaded. Its prediction is rounded to the nearest integer 0‑4, giving non‑constant, image‑dependent labels and thus moving the quadratic weighted kappa away from 0.0 toward the target. The core architecture and training logic remain unchanged.'
- What this solution (achieved 0.20603) has done: 'I keep the existing model and data pipeline but change the prediction logic to combine classifier logits and regression outputs more effectively. For the loaded checkpoint case, I compute soft‑max probabilities from the summed logits, take the expected class value, average it with the regression output, and then round to the nearest integer (clipped to 0‑4). This modest calibration should raise the quadratic weighted kappa toward the target while preserving the original architecture and training semantics. The fallback regressor path remains unchanged.'
- What this solution (achieved -0.0052) has done: 'I adjust the prediction logic to use the ordinal head, which provides a more appropriate estimate for the ordered DR grades. By converting the ordinal logits into cumulative probabilities and summing them we obtain an expected class value; this is then blended with the regression output (as before) to give a calibrated integer prediction. This change keeps the overall architecture unchanged while providing a more meaningful use of the model’s outputs, aiming to raise the quadratic weighted kappa toward the target.'
- What this solution (achieved 0.0) has done: 'I replace the TTA prediction function so it now uses the model’s classifier logits (with soft‑max averaging over the original and horizontally‑flipped image) and blends this discrete class with the regression output for a calibrated integer prediction. I also simplify the fallback when the checkpoint cannot be loaded by using the majority‑class label, which is a more sensible constant guess than the previous regression‑based fallback. These minimal changes keep the architecture untouched while providing a much more appropriate prediction strategy, moving the score toward the target.'

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
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import functional as FT
from PIL import Image, ImageChops
import cv2

from sklearn.metrics import cohen_kappa_score
import timm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def trim():
    """Identity transform placeholder for trimming black borders."""
    return transforms.Lambda(lambda img: img)


def cropTo4_3():
    """Identity transform placeholder for cropping to 4:3 aspect ratio."""
    return transforms.Lambda(lambda img: img)


test_ids = pd.read_csv("../input/aptos2019-blindness-detection/test.csv")
test_ids = np.squeeze(test_ids.values)

train_df = pd.read_csv("../input/aptos2019-blindness-detection/train.csv")
fallback_class = int(train_df["diagnosis"].mode()[0])  # majority class

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
loaded = False
try:
    net.load_state_dict(
        torch.load("../input/weights/B4_3stage_43epoch_CLAHE.pkl", map_location=device)
    )
    loaded = True
except Exception as e:
    pass

net = net.to(device)
net.eval()

fallback_regressor = Regressor().to(device)
fallback_regressor.eval()




## === cell 1
def predict_with_tta_combined(img_tensor):
    """
    Perform horizontal‑flip TTA and produce an integer prediction.
    Uses the classifier logits (soft‑max averaged over original & flipped)
    blended with the regression output for a calibrated result.
    """
    with torch.no_grad():
        c_out, r_out, _ = net(img_tensor)  # c_out: (1,5), r_out: (1,1)
        c_out_f, r_out_f, _ = net(torch.flip(img_tensor, dims=[3]))

    prob = torch.softmax(c_out, dim=1)
    prob_f = torch.softmax(c_out_f, dim=1)
    avg_prob = (prob + prob_f) / 2.0
    pred_class = torch.argmax(avg_prob, dim=1).item()  # 0‑4 integer

    reg_pred = ((r_out + r_out_f) / 2.0).squeeze().item()  # float in [0,4.5]

    blended = (pred_class + round(reg_pred)) / 2.0
    final_pred = int(round(blended))
    final_pred = max(0, min(4, final_pred))
    return final_pred


submission = []
for i, idx in enumerate(test_ids):
    print(f"Processing {i+1}/{len(test_ids)}: {idx}")
    image_path = f"../input/aptos2019-blindness-detection/test_images/{idx}.png"
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    if loaded:
        pred = predict_with_tta_combined(img)
    else:
        pred = fallback_class

    submission.append([idx, pred])

submission = np.array(submission)




## === cell 2
df = pd.DataFrame(submission, columns=["id_code", "diagnosis"])
df.to_csv("submission.csv", index=False)
