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

0.8891994668302018

# 6. Current score

-0.01254

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -0.11541) has done: 'I fix the missing weights issue by loading the provided model weights only if they exist and otherwise falling back to a standard timm pretrained EfficientNet backbone so inference can still run end-to-end. I also fix device handling (CPU fallback) and ensure `torch.load` uses `map_location` to avoid GPU/CPU mismatch errors. Finally, I make the submission generation robust (correctly reading `id_code`, clamping predictions to 0–4, and always writing `submission.csv` with the required columns). These changes preserve the model’s core forward logic while unblocking execution and producing a valid CSV.'
- What this solution (achieved -0.1102) has done: 'Your score is negative largely because the inference post-processing is misaligned with the model’s intended output: you’re rounding the *regression head* (`r_out`) even though this architecture defines a kappa-oriented discretization helper (`regress2class`) based on tuned thresholds. I keep the model and inference flow the same, but switch prediction from naive `round()` to `regress2class(r_out)` and ensure the output tensor stays on the right device/dtype. I also make the test image directory selection robust to the two common Kaggle paths present in your environment, avoiding silent file-not-found issues that can corrupt predictions. These are minimal changes intended to move QWK substantially upward toward your 0.889 target without changing the core model.'
- What this solution (achieved -0.01834) has done: 'Your current negative QWK strongly suggests the submission row order is misaligned with `test.csv` (Kaggle expects predictions in exactly the same order as `test.csv`), and your code also uses only the regression head while the model defines a `final=True` head that was likely used to produce competition-grade predictions. To move the score upward toward the target with minimal semantic change, I (1) generate predictions using `net(img, final=True)` and discretize with the same `regress2class` helper, and (2) build the submission by merging predictions back onto `test_df` to guarantee correct ordering and completeness. I also fix `regress2class` to allocate tensors on the correct device/dtype (no `.data` usage) to avoid subtle inconsistencies. These are small inference/post-processing fixes that keep the architecture and weights usage unchanged but should substantially improve QWK from the current negative score.'
- What this solution (achieved -0.01254) has done: 'Your current negative QWK is most consistent with a “random-ish” model output because the provided trained weights aren’t being loaded (the path `../input/weights/B7_ns_50epoch.pkl` doesn’t exist in your data tree), so you’re effectively submitting predictions from an ImageNet-pretrained backbone plus random heads. To move the score sharply upward toward your 0.889 target while preserving the exact architecture and inference semantics, I (1) automatically locate the competition weight file (by searching under `../input` for `B7_ns_50epoch.pkl`) and load it when present, and (2) add a deterministic test-time augmentation (horizontal flip) averaged in the same continuous space before applying your existing `regress2class` thresholds. These are minimal inference-only changes; they don’t alter training, losses, or the model’s forward structure, and they should materially improve QWK if the intended weights are available.'
- What this solution (achieved -0.01254) has done: 'Your current score is far below the target (gap ≈ -0.902), and the most likely cause is that the intended trained weights are still not being loaded (so the model behaves close to random even with sensible post-processing). I make a minimal, inference-only change to robustly discover and load the weight file by searching for common filenames/extensions (not just `B7_ns_50epoch.pkl`) under `../input`, and I fall back safely if nothing is found. I also add a very small sanity check that prints which weights were loaded so you can confirm you’re not accidentally running untrained heads. This keeps the same model, thresholds, and submission semantics, but should move QWK sharply upward toward your target if the weights exist in the dataset.'
- What this solution (achieved -0.01254) has done: 'Your score is still far from the target (gap ≈ -0.902), so we need a small change that can materially improve QWK without changing the model or training loop. The main likely issue now is that `regress2class()` uses fixed thresholds intended for the older `r_out` scaling, but you are feeding it `final=True` outputs scaled to `[0, 4.5]`, which makes those thresholds mismatched and can collapse predictions. I keep the same inference (final head + simple TTA) but change discretization to use calibrated thresholds in the same scale as `final=True` by mapping the 5 classes to midpoints and thresholding at `0.5, 1.5, 2.5, 3.5` (the standard cutpoints for 0–4). This is minimal post-processing, preserves evaluation semantics, and is very likely to move QWK upward toward your target.'

# 9. Code solution

## === cell 0
import os
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
from PIL import Image

from sklearn.metrics import cohen_kappa_score
import timm

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

device = "cuda:0" if torch.cuda.is_available() else "cpu"



## === cell 1
threshold = [0.5, 1.5, 2.5, 3.5]


def regress2class(out: torch.Tensor) -> torch.Tensor:
    out = out.view(-1)
    prediction = torch.zeros(out.size(0), device=out.device, dtype=torch.float32)
    for i in range(4):
        prediction += (out >= threshold[i]).to(torch.float32)
    return prediction




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


class Pretrain_Model(nn.Module):
    def __init__(self, backbone=None, pretrain=False):
        super(Pretrain_Model, self).__init__()

        if backbone is None:
            self.backbone = timm.models.tf_efficientnet_b4_ns(pretrained=pretrain)
            self.backbone.global_pool = GeM(flatten=True)
        else:
            self.backbone = backbone

        self.classifier1 = nn.Linear(1000, 500)
        self.classifier2 = nn.Linear(500, 5)

        self.regressor1 = nn.Linear(1000, 500)
        self.regressor2 = nn.Linear(500, 1)

        self.ordinal1 = nn.Linear(1000, 500)
        self.ordinal2 = nn.Linear(500, 4)

    def forward(self, x):
        x = self.backbone(x)

        c_out = self.classifier1(x)
        c_out = self.classifier2(c_out)

        r_out = self.regressor1(x)
        r_out = self.regressor2(r_out)
        r_out = torch.sigmoid(r_out) * 5 - 0.5

        o_out = self.ordinal1(x)
        o_out = self.ordinal2(o_out)
        o_out = torch.sigmoid(o_out)

        return c_out, r_out, o_out


class Maintrain_Model(Pretrain_Model):
    def __init__(self, weight_path):
        model = Pretrain_Model()
        model.load_state_dict(torch.load(weight_path, map_location="cpu"))
        super(Maintrain_Model, self).__init__(model.backbone)


class Posttrain_Model(nn.Module):
    def __init__(self, weight_path=None):
        super(Posttrain_Model, self).__init__()

        self.model = Pretrain_Model()
        if weight_path is not None:
            self.model.load_state_dict(torch.load(weight_path, map_location="cpu"))

        self.regressor = nn.Linear(10, 1)

    def forward(self, x):
        c_out, r_out, o_out = self.model(x)

        out = torch.cat((c_out, r_out, o_out), 1)
        out = self.regressor(out)
        out = torch.sigmoid(out) * 5 - 0.5

        return out




## === cell 3
test_df = pd.read_csv("../input/aptos2019-blindness-detection/test.csv")
test_df["id_code"] = test_df["id_code"].astype(str)
test_ids = test_df["id_code"].tolist()

input_size = 256

tranforms = transforms.Compose(
    [
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.384, 0.258, 0.174], std=[0.124, 0.089, 0.094]),
    ]
)


class ThreeStage_Model(nn.Module):
    def __init__(self, backbone=None, pretrained=False):
        super(ThreeStage_Model, self).__init__()

        self.backbone = timm.models.tf_efficientnet_b7_ns(pretrained=pretrained)
        self.backbone.global_pool = GeM(flatten=True)

        self.classifier1 = nn.Linear(1000, 500)
        self.classifier2 = nn.Linear(500, 5)

        self.regressor1 = nn.Linear(1000, 500)
        self.regressor2 = nn.Linear(500, 1)

        self.ordinal1 = nn.Linear(1000, 500)
        self.ordinal2 = nn.Linear(500, 4)

        self.final_regressor = nn.Linear(10, 1)

    def forward(self, x, final=False):
        x = self.backbone(x)

        c_out = self.classifier1(x)
        c_out = self.classifier2(c_out)

        r_out = self.regressor1(x)
        r_out = self.regressor2(r_out)
        r_out = torch.sigmoid(r_out) * 5 - 0.5

        o_out = self.ordinal1(x)
        o_out = self.ordinal2(o_out)
        o_out = torch.sigmoid(o_out)

        if final:
            out = torch.cat((c_out, r_out, o_out), 1)
            out = self.final_regressor(out)
            out = torch.sigmoid(out) * 4.5
            return out
        else:
            return c_out, r_out, o_out


def find_weight_file_any(candidates, root: str = "../input") -> str:
    for dirpath, _, filenames in os.walk(root):
        fn_set = set(filenames)
        for cand in candidates:
            if cand in fn_set:
                return os.path.join(dirpath, cand)
    return ""


weight_candidates = [
    "B7_ns_50epoch.pkl",
    "B7_ns_50epoch.pth",
    "B7_ns_50epoch.pt",
    "b7_ns_50epoch.pkl",
    "b7_ns_50epoch.pth",
    "b7_ns_50epoch.pt",
    "efficientnet_b7_ns_50epoch.pkl",
    "efficientnet_b7_ns_50epoch.pth",
    "efficientnet_b7_ns_50epoch.pt",
]

default_weights_path = "../input/weights/B7_ns_50epoch.pkl"
if os.path.exists(default_weights_path):
    weights_path = default_weights_path
else:
    weights_path = find_weight_file_any(weight_candidates, "../input")

use_pretrained_backbone = not (weights_path and os.path.exists(weights_path))

net = ThreeStage_Model(pretrained=use_pretrained_backbone)
loaded_msg = "none"
if weights_path and os.path.exists(weights_path):
    state = torch.load(weights_path, map_location="cpu")
    net.load_state_dict(state, strict=True)
    loaded_msg = weights_path
else:
    print(
        "WARNING: Could not find trained weights under ../input; using timm pretrained backbone only (likely low score)."
    )

print("Weights loaded:", loaded_msg)

net = net.to(device)
net.eval()

candidate_test_dirs = [
    "../input/aptos2019-blindness-detection/test_images",
    "../input/test_images",
]
test_img_dir = None
for d in candidate_test_dirs:
    if os.path.isdir(d):
        test_img_dir = d
        break
if test_img_dir is None:
    raise FileNotFoundError(
        f"Could not find test_images directory. Tried: {candidate_test_dirs}"
    )

pred_map = {}
with torch.no_grad():
    for idx in test_ids:
        image_name = os.path.join(test_img_dir, f"{idx}.png")
        img_pil = Image.open(image_name).convert("RGB")

        img1 = tranforms(img_pil).unsqueeze(0).to(device)
        img2_pil = img_pil.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        img2 = tranforms(img2_pil).unsqueeze(0).to(device)

        out1 = net(img1, final=True)  # [1,1] in ~[0, 4.5]
        out2 = net(img2, final=True)  # [1,1]
        out = 0.5 * (out1 + out2)

        pred_cls = regress2class(out.view(-1)).to("cpu")
        pred_int = int(np.clip(pred_cls.item(), 0, 4))
        pred_map[idx] = pred_int

pred_df = pd.DataFrame(
    {"id_code": list(pred_map.keys()), "diagnosis": list(pred_map.values())}
)
submission_df = test_df.merge(pred_df, on="id_code", how="left")
if submission_df["diagnosis"].isna().any():
    missing = (
        submission_df.loc[submission_df["diagnosis"].isna(), "id_code"].head(5).tolist()
    )
    raise RuntimeError(
        f"Missing predictions for some test ids (showing up to 5): {missing}"
    )

submission_df["diagnosis"] = submission_df["diagnosis"].astype(int)



## === cell 4
submission_df.to_csv("submission.csv", index=False)
print(submission_df.head())
print("Wrote submission.csv with shape:", submission_df.shape)
