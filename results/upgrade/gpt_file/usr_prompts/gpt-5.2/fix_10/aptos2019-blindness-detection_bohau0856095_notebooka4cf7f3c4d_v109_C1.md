# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.92318552899077

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I (1) remove the hard failure on a missing external checkpoint by falling back to the model’s randomly initialized weights so the notebook always runs end-to-end and writes `submission.csv`. Then I fix the CUDA/CPU dtype mismatch by ensuring the loaded state dict tensors (if any) are moved into the model on the correct device via `net.to(device)` after loading and by not mixing CPU weights with CUDA inputs. Finally, I make the prediction loop robust (batching with a DataLoader, consistent ordering, and safe image loading) so it produces exactly 367 predictions aligned to `test.csv`, which is required for a valid submission file.'
- What this solution (achieved 0.04339) has done: 'Your current 0.0 score is consistent with producing essentially random predictions because the external checkpoint is missing; with randomly initialized weights, quadratic weighted kappa collapses. To move the score toward the 0.923 target with minimal changes, the safest fix is to ensure the model uses ImageNet-pretrained weights from `timm` (no new files needed) while keeping the exact same architecture and inference logic. I also make the `regress2class` thresholding device/shape-safe (no CPU/GPU mixing) without changing its semantics. Everything else (transforms, model heads, prediction pipeline, submission schema) stays the same, and it still write `submission.csv`.'
- What this solution (achieved -0.07689) has done: 'Your current score is far below the target, so we should cautiously improve signal without changing the model/loop/heads: the biggest likely issue is a train/test preprocessing mismatch with the ImageNet-pretrained EfficientNet backbone, which can make predictions near-random. I keep the exact same model and inference flow, but switch the normalization to the backbone’s recommended `timm` preprocessing (mean/std) while keeping your custom crop/trim/resize intact. I also ensure the input tensor dtype matches the model parameters (float32) for consistent inference and reduce accidental variability. This is a minimal change expected to move QWK upward toward the target without altering architecture or adding training.'
- What this solution (achieved 0.02272) has done: 'Your score is far below the target (gap ≈ -1.000), so we should make the smallest change that restores meaningful signal without changing the model/loop/heads. Right now you’re using the regression head (`r_out`) from an ImageNet-pretrained backbone with random (untrained) heads, which tends to be near-random for this task and can produce negative QWK. A minimal improvement is to switch inference to use the classifier head (`c_out`) via `argmax`, which at least leverages a trained (ImageNet) linear head and typically yields more stable, non-degenerate class outputs than the random regression head. Everything else (architecture, transforms, DataLoader, submission schema) stays the same and it still writes a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your current score is far below the target, so we need a small change that adds real signal without changing the model architecture or training loop. Right now you’re using an ImageNet-pretrained backbone but completely random classification/regression/ordinal heads, so `argmax(c_out)` is effectively random. The minimal, metric-aligned improvement is to run inference with `final=True` and convert the single regression output into classes via your existing `regress2class` thresholds (keeping semantics intact), which at least uses the designed fusion head path. I also load the checkpoint strictly if it exists (same behavior when missing), and keep preprocessing and submission alignment unchanged.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score indicates the model is still producing effectively uninformative predictions on the test set; with no usable external checkpoint, the randomly initialized heads dominate and collapse QWK. To move the score upward toward the 0.923 target with minimal semantic change, I keep the exact same model and inference path (`final=True` + `regress2class`) but calibrate the fixed regression-to-class thresholds using out-of-fold predictions on `train.csv` (no training, just optimizing thresholds for QWK). This is a small, metric-aligned change that typically provides a large lift for APTOS-like pipelines while preserving architecture and loops. I also ensure the threshold search is deterministic and runs quickly by using a small coordinate descent over the 4 cutpoints.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score strongly suggests the submission is effectively uninformative; since we don’t have a real trained checkpoint, the untrained heads dominate and threshold tuning on those random-ish outputs won’t transfer to the test set. To move the score upward with minimal semantic change, I keep your exact model and inference path (`final=True` + `regress2class`) but (1) enable test-time augmentation via simple deterministic flips averaged in regression space, and (2) add a tiny, bounded “prior-matching” calibration that shifts/scales test regression outputs to match the train OOF mean/std before applying the already-tuned thresholds. These are small post-processing steps aligned with QWK that often improve stability without changing architecture, loss, or training. Everything still runs end-to-end and writes a valid `submission.csv` with 367 rows.'

# 9. Code solution

## === cell 0
import os
import glob
import random
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torchvision.transforms as transforms
from torchvision.transforms import functional as FT
from PIL import Image, ImageChops

from sklearn.metrics import (
    cohen_kappa_score,
)  # kept to preserve original imports/semantics
from sklearn.model_selection import StratifiedKFold
import timm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)



## === cell 1
threshold = [0.75, 1.5, 2.5, 3.5]


def regress2class(out):
    out = out.view(-1)
    thr = torch.tensor(threshold, device=out.device, dtype=out.dtype).view(1, -1)
    prediction = (out.view(-1, 1) >= thr).sum(dim=1).to(torch.int64)
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
BASE_CANDIDATES = [
    "/kaggle/input/aptos2019-blindness-detection",
    "../input/aptos2019-blindness-detection",
]
base = None
for b in BASE_CANDIDATES:
    if os.path.exists(b):
        base = b
        break
if base is None:
    base = "../input/aptos2019-blindness-detection"  # fallback

train_csv_path = os.path.join(base, "train.csv")
test_csv_path = os.path.join(base, "test.csv")
train_img_dir = os.path.join(base, "train_images")
test_img_dir = os.path.join(base, "test_images")

train_df = pd.read_csv(train_csv_path)
test_df = pd.read_csv(test_csv_path)

train_df["id_code"] = train_df["id_code"].astype(str)
test_df["id_code"] = test_df["id_code"].astype(str)

train_ids = train_df["id_code"].values
train_y = train_df["diagnosis"].astype(int).values
test_ids = test_df["id_code"].values

input_size = 384

data_cfg = timm.data.resolve_model_data_config(
    timm.create_model("tf_efficientnet_b4_ns", pretrained=True)
)
mean = data_cfg.get("mean", (0.485, 0.456, 0.406))
std = data_cfg.get("std", (0.229, 0.224, 0.225))
print("Using timm data config mean/std:", mean, std)

transform = transforms.Compose(
    [
        trim(),
        cropTo4_3(),
        transforms.Resize((input_size * 3 // 4, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
)

net = ThreeStage_Model()

ckpt_name = "B4_3stage_17epoch_320finetune.pkl"
candidate_paths = [
    "../input/weights/" + ckpt_name,
    "/kaggle/input/weights/" + ckpt_name,
]
candidate_paths += glob.glob("/kaggle/input/**/" + ckpt_name, recursive=True)

ckpt_path = None
for p in candidate_paths:
    if os.path.exists(p):
        ckpt_path = p
        break

if ckpt_path is not None:
    state = torch.load(ckpt_path, map_location="cpu")
    if (
        isinstance(state, dict)
        and "state_dict" in state
        and isinstance(state["state_dict"], dict)
    ):
        state = state["state_dict"]
    try:
        net.load_state_dict(state, strict=True)
        print("Loaded checkpoint (strict=True):", ckpt_path)
    except Exception as e:
        missing, unexpected = net.load_state_dict(state, strict=False)
        print("Loaded checkpoint (strict=False):", ckpt_path)
        print("Strict load failed with:", repr(e))
        if missing:
            print("Missing keys (loaded with strict=False):", len(missing))
        if unexpected:
            print("Unexpected keys (loaded with strict=False):", len(unexpected))
else:
    print(
        f"WARNING: Checkpoint '{ckpt_name}' not found; using ImageNet-pretrained backbone weights."
    )
    head_src = timm.create_model(
        "tf_efficientnet_b4_ns", pretrained=True, num_classes=5
    )
    sd = head_src.state_dict()
    with torch.no_grad():
        if "classifier.weight" in sd and "classifier.bias" in sd:
            net.classifier[3].weight.copy_(sd["classifier.weight"])
            net.classifier[3].bias.copy_(sd["classifier.bias"])
            print("Injected timm pretrained classifier head into net.classifier[3].")
        else:
            print(
                "WARNING: Could not find timm classifier.* in state_dict; skipped head injection."
            )

net = net.to(device)
net.eval()

print("Num train images:", len(train_ids))
print("Train images dir exists:", os.path.exists(train_img_dir))
print("Num test images:", len(test_ids))
print("Test images dir exists:", os.path.exists(test_img_dir))



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/2548388496.py in <cell line: 0>()
     94     with torch.no_grad():
     95         if "classifier.weight" in sd and "classifier.bias" in sd:
---> 96             net.classifier[3].weight.copy_(sd["classifier.weight"])
     97             net.classifier[3].bias.copy_(sd["classifier.bias"])
     98             # keep net.classifier[1] random (shape mismatch with timm head), preserving architecture

RuntimeError: The size of tensor a (500) must match the size of tensor b (1792) at non-singleton dimension 1

## === cell 5
from torch.utils.data import Dataset, DataLoader


class ImgDataset(Dataset):
    def __init__(self, ids, img_dir, transform=None, labels=None):
        self.ids = list(ids)
        self.img_dir = img_dir
        self.transform = transform
        self.labels = None if labels is None else np.asarray(labels).astype(int)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        idx = self.ids[i]
        image_name = os.path.join(self.img_dir, f"{idx}.png")
        try:
            img = Image.open(image_name).convert("RGB")
        except Exception:
            img = Image.new("RGB", (input_size, input_size), (0, 0, 0))
        if self.transform is not None:
            img = self.transform(img)
        if self.labels is None:
            return idx, img
        return idx, img, int(self.labels[i])


def infer_regression(ids, img_dir, bs=8, nw=2):
    ds = ImgDataset(ids, img_dir, transform=transform, labels=None)
    dl = DataLoader(
        ds,
        batch_size=bs,
        shuffle=False,
        num_workers=nw,
        pin_memory=torch.cuda.is_available(),
    )
    out_all = []
    ids_all = []
    with torch.no_grad():
        for ids_batch, x in dl:
            x = x.to(device, non_blocking=True).float()
            out = net(x, final=True).view(-1).detach().cpu().numpy().astype(np.float32)
            out_all.append(out)
            ids_all.extend([str(z) for z in ids_batch])
    out_all = np.concatenate(out_all, axis=0)
    return np.array(ids_all), out_all


def apply_thresholds(pred_cont, thr_list):
    thr = np.array(thr_list, dtype=np.float32).reshape(1, -1)
    pred_cont = np.asarray(pred_cont, dtype=np.float32).reshape(-1, 1)
    return (pred_cont >= thr).sum(axis=1).astype(np.int64)


def qwk(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


def tune_thresholds_oof(pred_cont, y_true, init_thr, n_iter=2):
    thr = np.array(init_thr, dtype=np.float32)

    def project(t):
        t = np.clip(t, 0.0, 4.5)
        t = np.sort(t)
        min_gap = 1e-3
        for i in range(1, 4):
            if t[i] <= t[i - 1] + min_gap:
                t[i] = min(t[i - 1] + min_gap, 4.5)
        return t

    thr = project(thr)
    best_thr = thr.copy()
    best_score = qwk(y_true, apply_thresholds(pred_cont, best_thr))

    steps = [0.25, 0.10, 0.05]
    for _ in range(n_iter):
        for step in steps:
            for j in range(4):
                candidates = []
                for delta in (-step, 0.0, step):
                    t = best_thr.copy()
                    t[j] = t[j] + delta
                    t = project(t)
                    candidates.append(t)
                local_best_thr = best_thr
                local_best_score = best_score
                for t in candidates:
                    s = qwk(y_true, apply_thresholds(pred_cont, t))
                    if s > local_best_score:
                        local_best_score = s
                        local_best_thr = t
                best_thr, best_score = local_best_thr, local_best_score

    return best_thr.tolist(), float(best_score)


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_pred = np.zeros(len(train_df), dtype=np.float32)

for fold, (tr_idx, va_idx) in enumerate(skf.split(train_ids, train_y), 1):
    va_ids = train_ids[va_idx]
    _, va_pred = infer_regression(va_ids, train_img_dir, bs=8, nw=2)
    oof_pred[va_idx] = va_pred
    print(f"Fold {fold}: inferred {len(va_idx)} train images")

best_thr, best_oof_qwk = tune_thresholds_oof(oof_pred, train_y, threshold, n_iter=2)
print("OOF QWK (with tuned thresholds):", best_oof_qwk)
print("Tuned thresholds:", best_thr)

threshold = best_thr




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/3302151483.py in <cell line: 0>()
    100 for fold, (tr_idx, va_idx) in enumerate(skf.split(train_ids, train_y), 1):
    101     va_ids = train_ids[va_idx]
--> 102     _, va_pred = infer_regression(va_ids, train_img_dir, bs=8, nw=2)
    103     oof_pred[va_idx] = va_pred
    104     print(f"Fold {fold}: inferred {len(va_idx)} train images")

/tmp/ipykernel_55/3302151483.py in infer_regression(ids, img_dir, bs, nw)
     40         for ids_batch, x in dl:
     41             x = x.to(device, non_blocking=True).float()
---> 42             out = net(x, final=True).view(-1).detach().cpu().numpy().astype(np.float32)
     43             out_all.append(out)
     44             ids_all.extend([str(z) for z in ids_batch])

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/tmp/ipykernel_55/2281285121.py in forward(self, x, final)
     77 
     78     def forward(self, x, final=False):
---> 79         x = self.backbone(x)
     80 
     81         c_out = self.classifier(x)

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/usr/local/lib/python3.11/dist-packages/timm/models/efficientnet.py in forward(self, x)
    337     def forward(self, x: torch.Tensor) -> torch.Tensor:
    338         """Forward pass."""
--> 339         x = self.forward_features(x)
    340         x = self.forward_head(x)
    341         return x

/usr/local/lib/python3.11/dist-packages/timm/models/efficientnet.py in forward_features(self, x)
    310     def forward_features(self, x: torch.Tensor) -> torch.Tensor:
    311         """Forward pass through feature extraction layers."""
--> 312         x = self.conv_stem(x)
    313         x = self.bn1(x)
    314         if self.grad_checkpointing and not torch.jit.is_scripting():

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/usr/local/lib/python3.11/dist-packages/timm/layers/conv2d_same.py in forward(self, x)
     51 
     52     def forward(self, x):
---> 53         return conv2d_same(
     54             x, self.weight, self.bias,
     55             self.stride, self.padding, self.dilation, self.groups,

/usr/local/lib/python3.11/dist-packages/timm/layers/conv2d_same.py in conv2d_same(x, weight, bias, stride, padding, dilation, groups)
     26 ):
     27     x = pad_same(x, weight.shape[-2:], stride, dilation)
---> 28     return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
     29 
     30 

RuntimeError: Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor) should be the same

## === cell 6
def infer_regression_tta(ids, img_dir, bs=8, nw=2):
    ds = ImgDataset(ids, img_dir, transform=transform, labels=None)
    dl = DataLoader(
        ds,
        batch_size=bs,
        shuffle=False,
        num_workers=nw,
        pin_memory=torch.cuda.is_available(),
    )
    out_all = []
    ids_all = []
    with torch.no_grad():
        for ids_batch, x in dl:
            x = x.to(device, non_blocking=True).float()

            out0 = net(x, final=True).view(-1)

            x_h = torch.flip(x, dims=[3])
            out1 = net(x_h, final=True).view(-1)

            x_v = torch.flip(x, dims=[2])
            out2 = net(x_v, final=True).view(-1)

            out = (out0 + out1 + out2) / 3.0
            out_all.append(out.detach().cpu().numpy().astype(np.float32))
            ids_all.extend([str(z) for z in ids_batch])
    out_all = np.concatenate(out_all, axis=0)
    return np.array(ids_all), out_all


def calibrate_to_oof(test_pred, oof_pred, clip_min=0.0, clip_max=4.5):
    test_pred = np.asarray(test_pred, dtype=np.float32)
    oof_pred = np.asarray(oof_pred, dtype=np.float32)

    m_test, s_test = float(test_pred.mean()), float(test_pred.std() + 1e-6)
    m_oof, s_oof = float(oof_pred.mean()), float(oof_pred.std() + 1e-6)

    scale = s_oof / s_test
    shift = m_oof - scale * m_test

    scale = float(np.clip(scale, 0.8, 1.25))
    shift = float(np.clip(shift, -0.5, 0.5))

    cal = test_pred * scale + shift
    cal = np.clip(cal, clip_min, clip_max).astype(np.float32)
    return cal, {
        "scale": scale,
        "shift": shift,
        "m_test": m_test,
        "s_test": s_test,
        "m_oof": m_oof,
        "s_oof": s_oof,
    }


test_ids_out, test_pred_cont = infer_regression_tta(test_ids, test_img_dir, bs=8, nw=2)

pred_map = {k: v for k, v in zip(test_ids_out.tolist(), test_pred_cont.tolist())}
test_pred_cont_aligned = np.array(
    [pred_map.get(str(i), 0.0) for i in test_ids], dtype=np.float32
)

test_pred_cont_cal, cal_stats = calibrate_to_oof(test_pred_cont_aligned, oof_pred)
print("Calibration stats:", cal_stats)

test_pred_cls = apply_thresholds(test_pred_cont_cal, threshold)

pred_rows = [[str(i), int(p)] for i, p in zip(test_ids, test_pred_cls.tolist())]
print("Predicted rows:", len(pred_rows))



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/1659440547.py in <cell line: 0>()
     54 
     55 
---> 56 test_ids_out, test_pred_cont = infer_regression_tta(test_ids, test_img_dir, bs=8, nw=2)
     57 
     58 pred_map = {k: v for k, v in zip(test_ids_out.tolist(), test_pred_cont.tolist())}

/tmp/ipykernel_55/1659440547.py in infer_regression_tta(ids, img_dir, bs, nw)
     14             x = x.to(device, non_blocking=True).float()
     15 
---> 16             out0 = net(x, final=True).view(-1)
     17 
     18             x_h = torch.flip(x, dims=[3])

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/tmp/ipykernel_55/2281285121.py in forward(self, x, final)
     77 
     78     def forward(self, x, final=False):
---> 79         x = self.backbone(x)
     80 
     81         c_out = self.classifier(x)

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/usr/local/lib/python3.11/dist-packages/timm/models/efficientnet.py in forward(self, x)
    337     def forward(self, x: torch.Tensor) -> torch.Tensor:
    338         """Forward pass."""
--> 339         x = self.forward_features(x)
    340         x = self.forward_head(x)
    341         return x

/usr/local/lib/python3.11/dist-packages/timm/models/efficientnet.py in forward_features(self, x)
    310     def forward_features(self, x: torch.Tensor) -> torch.Tensor:
    311         """Forward pass through feature extraction layers."""
--> 312         x = self.conv_stem(x)
    313         x = self.bn1(x)
    314         if self.grad_checkpointing and not torch.jit.is_scripting():

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/usr/local/lib/python3.11/dist-packages/timm/layers/conv2d_same.py in forward(self, x)
     51 
     52     def forward(self, x):
---> 53         return conv2d_same(
     54             x, self.weight, self.bias,
     55             self.stride, self.padding, self.dilation, self.groups,

/usr/local/lib/python3.11/dist-packages/timm/layers/conv2d_same.py in conv2d_same(x, weight, bias, stride, padding, dilation, groups)
     26 ):
     27     x = pad_same(x, weight.shape[-2:], stride, dilation)
---> 28     return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
     29 
     30 

RuntimeError: Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor) should be the same

## === cell 7
if len(pred_rows) == 0:
    raise RuntimeError("No predictions were generated; submission would be empty.")

pred_df = pd.DataFrame(pred_rows, columns=["id_code", "diagnosis"])
pred_df["id_code"] = pred_df["id_code"].astype(str)
pred_df["diagnosis"] = pred_df["diagnosis"].astype(int)

sub_df = test_df.copy()
sub_df["id_code"] = sub_df["id_code"].astype(str)
sub_df = sub_df.merge(pred_df, on="id_code", how="left")

if sub_df["diagnosis"].isna().any():
    sub_df["diagnosis"] = sub_df["diagnosis"].fillna(0).astype(int)

sub_df.to_csv("submission.csv", index=False)
print(sub_df.head())
print("Wrote submission.csv with shape:", sub_df.shape)
print("Unique diagnosis values:", sorted(sub_df["diagnosis"].unique().tolist()))
print("Final thresholds used:", threshold)

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/97397764.py in <cell line: 0>()
----> 1 if len(pred_rows) == 0:
      2     raise RuntimeError("No predictions were generated; submission would be empty.")
      3 
      4 pred_df = pd.DataFrame(pred_rows, columns=["id_code", "diagnosis"])
      5 pred_df["id_code"] = pred_df["id_code"].astype(str)

NameError: name 'pred_rows' is not defined
