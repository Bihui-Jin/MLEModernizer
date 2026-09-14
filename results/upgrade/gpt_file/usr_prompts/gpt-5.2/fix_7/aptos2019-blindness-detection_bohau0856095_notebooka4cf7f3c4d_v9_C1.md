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

0.8923503909317514

# 6. Current score

-0.01427

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -0.2545) has done: 'I fix the immediate runtime blockers so the notebook runs end-to-end: the missing `Parameter` import, the misspelled `tranforms` variable, the undefined `classes_num`, and the bad weight path (it currently contains an extra space before `.pth`). To keep core logic intact, I still load the same saved model if it exists; if it doesn’t (common in this environment), I fall back to constructing the same architecture and running it with pretrained backbone weights so a valid submission is always produced. I also make device selection robust (CPU fallback) and ensure the test IDs are read as strings and the submission is written with the exact required columns and `.csv` suffix.'
- What this solution (achieved 0.10673) has done: 'Your score is far below the target, so we should improve predictions without changing the model architecture or training (you’re doing pure inference). The main issue is that you’re ignoring the model’s regression head and directly argmax’ing the classifier logits, which often performs poorly for quadratic weighted kappa on this task. I keep the exact same network and weights-loading logic, but change only the prediction post-processing to use the regressor output (rounded and clipped to 0–4), with a simple average ensemble between classifier-expected-class and regressed value to stabilize results. I also fix the `tranforms` typo safely by using a correctly named `transforms_fn` variable (no functional change).'
- What this solution (achieved -0.06528) has done: 'Your current score is far below the target, so we should improve predictions while keeping the same model and pure-inference workflow. The largest likely issue is a mismatch between the training-time pooling/feature dimension and your inference-time head definition: `tf_efficientnet_b4_ns`’s classifier feature dim is not 1000, so loading weights with `strict=False` can silently leave your heads randomly initialized, crushing performance. I keep the exact architecture pieces (EfficientNet-B4 + GeM + linear classifier/regressor) but set the head input dim from the backbone (`num_features`) and replace the backbone’s classifier with identity so GeM actually feeds the heads. Finally, I keep your blended prediction idea but slightly bias toward the regressor (commonly better aligned with QWK) as a minimal, low-risk calibration tweak.'
- What this solution (achieved -0.00664) has done: 'We need to move the score upward toward the target, and your current score suggests the model weights likely aren’t being applied correctly at inference (heads left randomly initialized or backbone feature shape mismatch). I keep the same EfficientNet-B4 + GeM + (classifier, regressor) core, but make weight loading deterministic and strict about matching keys/shapes by constructing `Model()` first and loading a cleaned state_dict into it (instead of `torch.load()` potentially returning a different object). I also ensure we use `forward_features()` so GeM is applied to the final conv features (not to an already-pooled vector), which fixes a common mismatch with EfficientNet in timm while preserving the same architecture intent. Finally, I keep your blended post-processing but make it conditional: if regressor outputs look out-of-range (a sign of bad weights), fall back to classifier-expected-class to avoid catastrophic negative kappa.'
- What this solution (achieved -0.31853) has done: 'Your current score is still far below the target, so the most likely cause is that the competition weights are not being loaded correctly (because `strict=False` silently leaves mismatched layers randomly initialized). I keep your exact EfficientNet-B4 + GeM + (classifier, regressor) design and pure-inference approach, but make weight loading “strict when possible” and explicitly verify that a meaningful fraction of parameters actually matched; if not, we fall back to loading the entire saved `nn.Module` object when available. I also ensure the backbone head is set to identity so `forward_features()` and GeM always feed your heads consistently, which avoids common timm EfficientNet head-shape traps without changing the intended architecture. Finally, I keep your blended post-processing but make it slightly more robust by computing everything on CPU-safe tensors and keeping deterministic settings.'
- What this solution (achieved -0.01427) has done: 'The negative kappa strongly suggests the loaded weights are not actually being used as intended, so the smallest change with the biggest expected gain is to make weight loading strict and fail fast into a known-good fallback. I keep your exact EfficientNet-B4 + GeM + (classifier, regressor) core and the same inference-only workflow, but I (1) search for weights in common Kaggle locations (including the dataset folder itself), (2) only accept a state_dict load if it matches the model shapes with a high ratio, and otherwise (3) fall back to the pretrained backbone and use the classifier-expected-class only (more stable than blending with a likely-untrained regressor). This should move the score upward toward the target by avoiding catastrophic random-head predictions while preserving your overall approach and producing the same submission format.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

import torchvision.transforms as transforms
from PIL import Image
import timm

torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = "cuda:0" if torch.cuda.is_available() else "cpu"

test_df = pd.read_csv(
    "../input/aptos2019-blindness-detection/test.csv", dtype={"id_code": str}
)
test_ids = test_df["id_code"].tolist()

transforms_fn = transforms.Compose(
    [
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.384, 0.258, 0.174], std=[0.124, 0.089, 0.094]),
    ]
)

classes_num = 5


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


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.backbone = timm.models.tf_efficientnet_b4_ns(pretrained=True)
        feat_dim = getattr(self.backbone, "num_features", 1000)

        if hasattr(self.backbone, "classifier"):
            self.backbone.classifier = nn.Identity()
        if hasattr(self.backbone, "global_pool"):
            self.backbone.global_pool = nn.Identity()

        self.pool = GeM(flatten=True)
        self.classifier = nn.Linear(feat_dim, classes_num)
        self.regressor = nn.Linear(feat_dim, 1)

    def forward(self, x):
        feats = self.backbone.forward_features(x)  # (B, C, H, W)
        pooled = self.pool(feats)  # (B, C)
        out1 = self.classifier(pooled)
        out2 = self.regressor(pooled)
        return out1, out2


def _clean_state_dict(state):
    if (
        isinstance(state, dict)
        and "state_dict" in state
        and isinstance(state["state_dict"], dict)
    ):
        state = state["state_dict"]
    if not isinstance(state, dict):
        return None
    cleaned = {}
    for k, v in state.items():
        nk = k.replace("module.", "")
        cleaned[nk] = v
    return cleaned


def _try_load_state_dict_strictish(
    model: nn.Module, weight_path: str, min_match_ratio: float = 0.95
):
    """
    Change rationale (score-up, minimal): negative QWK is typical when heads are random due to partial/mismatched loads.
    We only accept a weights file if it matches model tensor shapes at a high ratio; otherwise we treat it as unusable.
    """
    state = torch.load(weight_path, map_location="cpu")

    if isinstance(state, nn.Module):
        return state, True

    cleaned = _clean_state_dict(state)
    if cleaned is None:
        return model, False

    model_sd = model.state_dict()
    loadable = {}
    matched = 0
    for k, v in cleaned.items():
        if k in model_sd and hasattr(v, "shape") and model_sd[k].shape == v.shape:
            loadable[k] = v
            matched += 1

    match_ratio = matched / max(1, len(model_sd))
    if match_ratio < min_match_ratio:
        return model, False

    model.load_state_dict(loadable, strict=False)
    return model, True


weight_candidates = [
    "../input/weights/tf_efficientnet_b4_ns_regress.pth",
    "../input/weights/tf_efficientnet_b4_ns_regress .pth",
    "../input/aptos2019-blindness-detection/tf_efficientnet_b4_ns_regress.pth",
    "../input/aptos2019-blindness-detection/tf_efficientnet_b4_ns_regress .pth",
    "/kaggle/input/weights/tf_efficientnet_b4_ns_regress.pth",
]

weight_path = None
for p in weight_candidates:
    if os.path.exists(p):
        weight_path = p
        break

net = Model()
weights_ok = False
if weight_path is not None:
    net, weights_ok = _try_load_state_dict_strictish(
        net, weight_path, min_match_ratio=0.95
    )

net = net.to(device)
net.eval()

submission = []
with torch.inference_mode():
    arng = torch.arange(classes_num, device=device).float()
    for idx in test_ids:
        image_name = f"../input/aptos2019-blindness-detection/test_images/{idx}.png"
        img = Image.open(image_name).convert("RGB")
        img = transforms_fn(img).unsqueeze(0).to(device)

        logits, reg = net(img)

        probs = torch.softmax(logits, dim=1)
        cls_exp = (probs * arng).sum(dim=1)

        if weights_ok:
            reg_val = reg.view(-1)
            if torch.isfinite(reg_val).all() and (reg_val.abs().max().item() <= 20.0):
                blended = 0.35 * cls_exp + 0.65 * reg_val
            else:
                blended = cls_exp
        else:
            blended = cls_exp

        pred = torch.round(blended).clamp(0, classes_num - 1).long().item()
        submission.append([idx, int(pred)])

submission = np.array(submission, dtype=object)



## === cell 1
df = pd.DataFrame(submission, columns=["id_code", "diagnosis"])
df["id_code"] = df["id_code"].astype(str)
df["diagnosis"] = df["diagnosis"].astype(int)

df.to_csv("submission.csv", index=False)

print(df.head())
print("Wrote submission.csv with shape:", df.shape)
