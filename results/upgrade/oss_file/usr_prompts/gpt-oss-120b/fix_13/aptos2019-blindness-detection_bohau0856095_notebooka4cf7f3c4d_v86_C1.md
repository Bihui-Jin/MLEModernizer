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

0.9224480930331804

# 6. Current score

0.08715

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -0.0516) has done: 'I fix the shape mismatch between the EfficientNet backbone and the downstream linear layers by dynamically using the backbone’s feature dimension, and adjust the checkpoint loading to ignore mismatched keys. This resolves the runtime error, ensures the model runs, and produces a non‑empty submission CSV.'
- What this solution (achieved -0.11213) has done: 'I switch the prediction from the regression‐based output to the classifier logits that the model already produces. Using `argmax` on the classifier scores gives a more direct class prediction and is expected to raise the Quadratic Weighted Kappa substantially, moving the current –0.0516 score toward the target 0.9224. The change is limited to the inference loop, preserving the original model architecture and training pipeline.'
- What this solution (achieved 0.07348) has done: 'I adjust the inference step to use the model’s combined final regressor output (which aggregates classifier, regression and ordinal information) instead of only the raw classifier logits. By calling `net(img, final=True)` we obtain a scalar prediction in the 0‑4.5 range; rounding and clamping it to the valid class indices (0‑4) should give predictions that better reflect the trained ensemble and therefore move the Quadratic Weighted Kappa score toward the target.'
- What this solution (achieved 0.03765) has done: 'I switch the inference to use the classifier logits produced by the network instead of the combined regression output. By taking the argmax of the classifier’s raw scores (after a softmax) we obtain a direct class prediction, which aligns better with the Quadratic Weighted Kappa metric and should raise the score toward the target while keeping the rest of the pipeline unchanged.'
- What this solution (achieved 0.0117) has done: 'I keep the model and preprocessing unchanged but modify the inference step to use the network’s final regression output (a continuous severity score) instead of the raw classifier logits. Rounding this output to the nearest integer and clamping it to the valid range 0‑4 usually aligns better with the quadratic weighted kappa metric, so the predictions should move noticeably toward the target score. The rest of the pipeline (loading, transforms, CSV creation) stays identical.'
- What this solution (achieved -0.02633) has done: 'I adjust the inference step to use the model’s classifier logits instead of the final regression output. By taking the argmax of the classifier scores we obtain a direct class prediction, which aligns better with the quadratic weighted kappa metric and should move the score significantly toward the target while keeping the rest of the pipeline unchanged.'
- What this solution (achieved -0.06489) has done: 'I keep the model and preprocessing unchanged and modify the inference loop to combine the classifier’s expected value with the model’s final regression output, then round and clamp the result to the valid class range 0‑4. This simple calibration step is inexpensive, preserves the core architecture, and is expected to raise the quadratic weighted kappa toward the target score.'
- What this solution (achieved -0.03803) has done: 'I replace the inference logic that averages the classifier expectation with the regression output by a direct class prediction using the classifier logits (argmax). This small change aligns predictions with the discrete label space and is expected to raise the quadratic weighted kappa toward the target without altering the model architecture or training process.'
- What this solution (achieved -0.08771) has done: 'I replace the simple arg‑max class prediction with an expectation‑based prediction: compute soft‑max probabilities from the classifier logits, take the weighted average of class indices, round to the nearest integer and clamp to 0‑4. This uses the same model output but aligns the prediction more closely with the continuous nature of the trained regressor, which should raise the Quadratic Weighted Kappa toward the target score while keeping the core architecture unchanged.'
- What this solution (achieved -0.11499) has done: 'I replace the inference logic to use the model’s aggregated final regression output (`net(img, final=True)`) instead of the classifier‑based expectation. This output combines classifier, regression and ordinal branches, then is scaled to the 0‑4.5 range; rounding and clamping it to 0‑4 yields a discrete prediction that aligns better with the Quadratic Weighted Kappa metric, moving the score toward the target while keeping the rest of the pipeline unchanged.'
- What this solution (achieved 0.08715) has done: 'I modify the inference step to use the model’s classifier logits (argmax) together with the regression output, averaging them to produce a more calibrated class prediction. This small change keeps the core architecture unchanged while aligning predictions better with the discrete label space, which should move the Quadratic Weighted Kappa score upward toward the target.'

# 9. Code solution

## === cell 0
import os, subprocess, sys

wheel_path = "../input/weights/timm-0.3.1-py3-none-any.whl"
if os.path.exists(wheel_path):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", wheel_path])
    except Exception as e:
        print(f"Optional wheel install failed: {e}")



## === cell 1
import random
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import functional as FT
from PIL import Image, ImageChops
import cv2
from sklearn.metrics import cohen_kappa_score
import timm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")



## === cell 2
threshold = [0.75, 1.5, 2.5, 3.5]


def regress2class(out):
    """
    Convert regression output to integer class label using the thresholds.
    """
    prediction = torch.zeros(out.size(0), device=out.device)
    for i in range(4):
        prediction += (out >= threshold[i]).float()
    return prediction


def ordinal2class_prob(out):
    pred_prob = torch.zeros(out.size(0), 5, device=out.device)
    pred_prob[:, 0] = 1 - out[:, 0]
    pred_prob[:, 1] = out[:, 0] * (1 - out[:, 1])
    pred_prob[:, 2] = out[:, 1] * (1 - out[:, 2])
    pred_prob[:, 3] = out[:, 2] * (1 - out[:, 3])
    pred_prob[:, 4] = out[:, 3]
    return F.softmax(pred_prob, dim=1)


def regress2class_prob(out):
    pred_prob = torch.zeros((out.size(0), 5), device=out.device)
    for i in range(out.size(0)):
        val = out[i].item()
        if val < 4.0:
            l1 = int(math.floor(val))
            l2 = int(math.ceil(val))
            pred_prob[i, l1] = 1 - (val - l1)
            pred_prob[i, l2] = 1 - (l2 - val)
        else:
            pred_prob[i, 4] = 1.0
    return pred_prob




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
        return f"{self.__class__.__name__}(p={self.p.item():.4f}, eps={self.eps})"


class ThreeStage_Model(nn.Module):
    def __init__(self, backbone=None):
        super(ThreeStage_Model, self).__init__()

        self.backbone = timm.create_model(
            "tf_efficientnet_b4_ns", pretrained=True, num_classes=0
        )
        self.backbone.global_pool = GeM(flatten=True)

        feat_dim = self.backbone.num_features  # should be 1792 for EfficientNet-B4

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




## === cell 4
test_ids_df = pd.read_csv("../input/aptos2019-blindness-detection/test.csv")
test_ids = test_ids_df["id_code"].values.squeeze()

input_size = 512

transform = transforms.Compose(
    [
        lambda img: ImageChops.difference(
            img, Image.new(img.mode, img.size, img.getpixel((0, 0)))
        ).getbbox()
        and img.crop(
            ImageChops.difference(
                img, Image.new(img.mode, img.size, img.getpixel((0, 0)))
            ).getbbox()
        ),
        lambda img: (
            lambda w, h: (
                img.crop(((w - int(h * 4 / 3)) // 2, 0, (w + int(h * 4 / 3)) // 2, h))
                if w / h >= 4 / 3
                else img.crop(
                    (0, (h - int(w * 3 / 4)) // 2, w, (h + int(w * 3 / 4)) // 2)
                )
            )
        )(*img.size),
        transforms.Resize((input_size * 3 // 4, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.384, 0.258, 0.174], std=[0.124, 0.089, 0.094]),
    ]
)



## === cell 5
net = ThreeStage_Model()
ckpt_path = "../input/weights/B4_3stage_14epoch_finetune2_512.pkl"
if os.path.exists(ckpt_path):
    try:
        net.load_state_dict(torch.load(ckpt_path, map_location=device), strict=False)
        print("Loaded checkpoint (partial).")
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
else:
    print("Checkpoint not found; using pretrained backbone only.")

net = net.to(device)
net.eval()



## === cell 6
submission = []

with torch.no_grad():
    for i, idx in enumerate(test_ids):
        if i % 50 == 0:
            print(f"Processing {i}/{len(test_ids)}")
        image_path = f"../input/aptos2019-blindness-detection/test_images/{idx}.png"
        img = Image.open(image_path).convert("RGB")
        img = transform(img).unsqueeze(0).to(device)

        c_out, r_out, _ = net(img, final=False)  # c_out: [1,5], r_out: [1,1]

        class_pred = torch.argmax(c_out, dim=1).item()

        reg_val = r_out.squeeze().item()
        reg_pred = int(round(reg_val))
        reg_pred = max(0, min(4, reg_pred))

        avg_pred = int(round((class_pred + reg_pred) / 2.0))

        final_pred = max(0, min(4, avg_pred))

        submission.append([idx, final_pred])



## === cell 7
df = pd.DataFrame(submission, columns=["id_code", "diagnosis"])
output_path = "submission.csv"
df.to_csv(output_path, index=False)
print(f"Submission saved to {output_path}, rows: {len(df)}")
