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

0.8976533744850824

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the runtime errors by (1) switching the model to run on CPU, (2) correcting the weight‑file path and adding a safe fallback when the file is missing, and (3) ensuring the inference loop works without CUDA. These changes let the script finish end‑to‑end and produce a non‑empty `submission.csv` while preserving the original model logic.'
- What this solution (achieved 0.09795) has done: 'Implemented fixes:
- Rewrote `backboneNet_efficient` to use the available EfficientNet API safely, removing invalid attribute references and adding optional pretrained weights.
- Adjusted forward pass to use `forward_features` and proper global pooling.
- Kept the original heads and dropout logic intact.
- Updated cells numbering to start from 1 and ensured the inference loop uses the corrected model.
- Fixed submission handling to correctly detect an empty result and write a proper CSV.'
- What this solution (achieved -0.14288) has done: 'Implemented missing imports, device handling, and a concrete ThreeStage_Model using a pretrained EfficientNet‑B4 backbone with three heads (classification, regression, ordinal). Added utility cell for imports and device setup, reorganized cells to start at 1, and ensured the inference loop and CSV writing work without errors. The script now runs end‑to‑end and produces a valid submission.csv while preserving the original prediction logic.'
- What this solution (achieved 0.0) has done: 'The fix corrects a typo in the model’s forward method where the regression head was applied to an undefined variable `fets`. Changing it to the proper `feats` allows the network to run, enabling inference and producing a non‑empty submission.csv file. No other logic is altered, preserving the original architecture and scoring approach.'

# 9. Code solution

## === cell 0
import os
import math
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import timm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class trim:
    def __call__(self, img):
        return img  # no trimming performed


class cropTo4_3:
    def __call__(self, img):
        w, h = img.size
        target_ratio = 4 / 3
        if w / h > target_ratio:
            new_w = int(h * target_ratio)
            left = (w - new_w) // 2
            right = left + new_w
            top = 0
            bottom = h
        else:
            new_h = int(w / target_ratio)
            top = (h - new_h) // 2
            bottom = top + new_h
            left = 0
            right = w
        return img.crop((left, top, right, bottom))




## === cell 1
threshold = [0.75, 1.5, 2.5, 3.5]


def regress2class(out):
    prediction = 0
    for i in range(4):
        prediction += (out.data >= threshold[i]).squeeze().cpu().item()
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
    out = out.squeeze(-1)  # keep batch dim
    out = torch.clamp(out, 0.0, 4.0)
    pred_prob = torch.zeros((out.size(0), 5), device=out.device)
    for i in range(out.size(0)):
        val = out[i].item()
        l1 = int(math.floor(val))
        l2 = int(math.ceil(val))
        if l1 == l2:
            pred_prob[i][l1] = 1.0
        else:
            pred_prob[i][l1] = 1 - (val - l1)
            pred_prob[i][l2] = 1 - (l2 - val)
    return pred_prob


def combine3output(r_out, c_out, o_out):
    """
    Convert each head's raw output to a probability distribution over the 5 classes,
    average the three distributions, and return the class with the highest average probability.
    """
    prob_r = regress2class_prob(r_out)  # (B,5)
    prob_c = F.softmax(c_out, dim=1)  # (B,5)
    prob_o = ordinal2class_prob(o_out)  # (B,5)
    avg_prob = (prob_r + prob_c + prob_o) / 3.0
    pred = torch.argmax(avg_prob, dim=1)
    return int(pred.item())




## === cell 2
class ThreeStage_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            "tf_efficientnet_b4", pretrained=True, num_classes=0
        )
        feat_dim = self.backbone.num_features  # typically 1792 for B4

        self.cls_head = nn.Linear(feat_dim, 5)  # classification logits
        self.reg_head = nn.Linear(feat_dim, 1)  # regression scalar
        self.ord_head = nn.Linear(feat_dim, 4)  # ordinal logits

    def forward(self, x):
        feats = self.backbone.forward_features(x)  # (B, C, H, W)
        feats = self.backbone.global_pool(feats)  # (B, C)
        c_out = self.cls_head(feats)  # (B,5)
        r_out = self.reg_head(feats)  # (B,1)  <-- fixed typo
        o_out = self.ord_head(feats)  # (B,4)
        return c_out, r_out, o_out




## === cell 3
test_csv_paths = [
    "../input/aptos2019-blindness-detection/test.csv",
    "./input/aptos2019-blindness-detection/test.csv",
    "../input/test.csv",
    "./input/test.csv",
]
test_ids = None
for p in test_csv_paths:
    if os.path.isfile(p):
        test_ids = pd.read_csv(p)
        break
if test_ids is None:
    raise FileNotFoundError("Test CSV file not found in expected locations.")
test_ids = test_ids["id_code"].astype(str).tolist()

transform1 = transforms.Compose(
    [
        trim(),
        cropTo4_3(),
        transforms.Resize((288, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.384, 0.258, 0.174], std=[0.124, 0.089, 0.094]),
    ]
)

net1 = ThreeStage_Model()

weight_path_candidates = [
    "../input/weights/B4_3stage_22epoch_320.pkl",
    "/kaggle/input/weights/B4_3stage_22epoch_320.pkl",
    "./input/aptos2019-blindness-detection/weights/B4_3stage_22epoch_320.pkl",
    "./input/weights/B4_3stage_22epoch_320.pkl",
    "./weights/B4_3stage_22epoch_320.pkl",
    "./working/weights/B4_3stage_22epoch_320.pkl",
    "./aptos2019-blindness-detection/weights/B4_3stage_22epoch_320.pkl",
]

loaded = False
for wp in weight_path_candidates:
    if os.path.isfile(wp):
        try:
            state = torch.load(wp, map_location=device)
            net1.load_state_dict(state)
            print(f"Loaded weights from {wp}")
            loaded = True
            break
        except Exception as e:
            print(f"Failed to load weights from {wp}: {e}")

fallback_class = None
if not loaded:
    train_csv_paths = [
        "../input/aptos2019-blindness-detection/train.csv",
        "./input/aptos2019-blindness-detection/train.csv",
        "../input/train.csv",
        "./input/train.csv",
    ]
    train_df = None
    for p in train_csv_paths:
        if os.path.isfile(p):
            train_df = pd.read_csv(p)
            break
    if train_df is not None:
        fallback_class = int(train_df["diagnosis"].mode()[0])
        print(f"Fallback class (most common in training): {fallback_class}")
    else:
        fallback_class = 0
        print("Training file not found; defaulting fallback class to 0.")

net1 = net1.to(device)
net1.eval()




## === cell 4
submission = []
with torch.no_grad():
    for i, idx in enumerate(test_ids):
        if i % 100 == 0:
            print(f"Processing {i}/{len(test_ids)}")
        image_path = f"../input/aptos2019-blindness-detection/test_images/{idx}.png"
        if not os.path.isfile(image_path):
            image_path = (
                f"/kaggle/input/aptos2019-blindness-detection/test_images/{idx}.png"
            )
        if not os.path.isfile(image_path):
            pred_class = fallback_class
        else:
            img = Image.open(image_path).convert("RGB")
            img = transform1(img).unsqueeze(0).to(device)
            c_out, r_out, o_out = net1(img)  # (cls, rg, ord)
            if loaded:
                pred_class = combine3output(r_out, c_out, o_out)
            else:
                pred_class = fallback_class
        submission.append([idx, pred_class])

submission = np.array(submission)




## === cell 5
if len(submission) == 0:
    raise RuntimeError("Submission array is empty; inference may have failed.")
df = pd.DataFrame(submission, columns=["id_code", "diagnosis"])
output_path = "submission.csv"
df.to_csv(output_path, index=False)
print(f"Submission saved to {output_path}, rows: {len(df)}")
