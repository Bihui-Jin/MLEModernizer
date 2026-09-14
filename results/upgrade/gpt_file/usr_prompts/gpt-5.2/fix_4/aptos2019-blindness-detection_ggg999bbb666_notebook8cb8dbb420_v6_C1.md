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

0.9030739204737794

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.0) has done: 'I make minimal changes to ensure you always generate a valid submission and improve the expected kappa by fixing two common inference-time issues: checkpoint loading compatibility (handling `fc.*` shape mismatches and other non-critical key mismatches safely) and applying test-time augmentation (horizontal flip) while keeping the same model and thresholds. I also switch preprocessing to use the same normalization but run inference in batches to reduce overhead and help complete within the time limit (without changing model logic). Finally, I keep your exact thresholding scheme (evaluation semantics) so any score changes come from more reliable/robust predictions rather than a different mapping.'

# 9. Code solution

## === cell 0
import os
from glob import glob

import numpy as np
import pandas as pd
from PIL import Image, ImageFile

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torchvision import transforms
from torchvision.models import resnet50
from torchvision.models.resnet import ResNet, Bottleneck

ImageFile.LOAD_TRUNCATED_IMAGES = True

DATA_ROOT = "../input/aptos2019-blindness-detection"
TEST_CSV = os.path.join(DATA_ROOT, "test.csv")
TEST_IMAGE_PATH = os.path.join(DATA_ROOT, "test_images")

MODEL_PATH_list = [
    "../input/224best/224best.pth",
    "../input/128best/128best.pth",
    "../input/seresnet384/model_epoch33.pth",
]
resize_list = [224, 128, 384]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def _resolve_ckpt_path(p: str) -> str:
    """
    Bugfix: Kaggle datasets are mounted under /kaggle/input, but this notebook uses ../input.
    Keep original paths, and if missing, search /kaggle/input for the same basename.
    """
    if os.path.isfile(p):
        return p
    base = os.path.basename(p)
    candidates = glob(f"/kaggle/input/**/{base}", recursive=True)
    if len(candidates) > 0:
        candidates = sorted(candidates, key=lambda x: (len(x), x))
        return candidates[0]
    return p


MODEL_PATH_list = [_resolve_ckpt_path(p) for p in MODEL_PATH_list]
print("Resolved model paths:")
for p in MODEL_PATH_list:
    print(" -", p, "| exists:", os.path.isfile(p))

if not os.path.isfile(TEST_CSV):
    alt_root = "/kaggle/input/aptos2019-blindness-detection"
    alt_csv = os.path.join(alt_root, "test.csv")
    alt_img = os.path.join(alt_root, "test_images")
    if os.path.isfile(alt_csv):
        DATA_ROOT = alt_root
        TEST_CSV = alt_csv
        TEST_IMAGE_PATH = alt_img

print("DATA_ROOT:", DATA_ROOT)
print("TEST_CSV exists:", os.path.isfile(TEST_CSV))
print("TEST_IMAGE_PATH exists:", os.path.isdir(TEST_IMAGE_PATH))




## === cell 1
class SEModule(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        s = self.avg_pool(x)
        s = self.fc1(s)
        s = self.relu(s)
        s = self.fc2(s)
        s = self.sigmoid(s)
        return x * s


class SEBottleneck(Bottleneck):
    def __init__(self, *args, reduction=16, **kwargs):
        super().__init__(*args, **kwargs)
        self.se_module = SEModule(self.conv3.out_channels, reduction=reduction)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = self.se_module(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        x = F.avg_pool2d(x, (x.size(-2), x.size(-1)))
        return x.pow(1.0 / self.p)

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p.data.tolist()[0]:.4f}, eps={self.eps})"


def get_se_resnet50_gem(pretrain=None):
    model = ResNet(block=SEBottleneck, layers=[3, 4, 6, 3], num_classes=1000)

    if pretrain == "imagenet":
        base = resnet50(weights="DEFAULT")
        model.load_state_dict(base.state_dict(), strict=False)

    model.avgpool = GeM()
    model.fc = nn.Linear(2048, 1)
    return model


def _clean_state_dict_for_model(state, model):
    """
    Score-relevant robustness: handle common checkpoint wrappers and safely drop
    incompatible keys (e.g., different fc head naming/shape) instead of crashing
    or partially loading wrong tensors.
    """
    if (
        isinstance(state, dict)
        and "state_dict" in state
        and isinstance(state["state_dict"], dict)
    ):
        state = state["state_dict"]

    if not isinstance(state, dict):
        raise ValueError("Loaded checkpoint is not a state_dict-like dict")

    if len(state) > 0:
        k0 = next(iter(state.keys()))
        if isinstance(k0, str) and k0.startswith("module."):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}

    model_sd = model.state_dict()
    filtered = {}
    dropped = 0
    for k, v in state.items():
        if k in model_sd and hasattr(v, "shape") and model_sd[k].shape == v.shape:
            filtered[k] = v
        else:
            dropped += 1

    if dropped > 0:
        print(f"Checkpoint keys dropped due to mismatch/unexpected: {dropped}")

    return filtered




## === cell 2
test_df = pd.read_csv(TEST_CSV)
test_ids = test_df["id_code"].astype(str).tolist()
test_images = [os.path.join(TEST_IMAGE_PATH, f"{iid}.png") for iid in test_ids]

if len(test_images) > 0 and (not os.path.isfile(test_images[0])):
    nested = os.path.join(TEST_IMAGE_PATH, "test_images")
    if os.path.isdir(nested):
        TEST_IMAGE_PATH = nested
        test_images = [os.path.join(TEST_IMAGE_PATH, f"{iid}.png") for iid in test_ids]

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def load_image_tensor(path, size, hflip=False):
    try:
        img = Image.open(path).convert("RGB")
    except Exception:
        img = Image.new("RGB", (size, size), (0, 0, 0))
    img = img.resize((size, size), resample=Image.BILINEAR)
    if hflip:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    t = transforms.ToTensor()(img)
    t = normalize(t)
    return t


def predict_model(model, image_paths, size, batch_size=16):
    model.eval()
    out = np.zeros(len(image_paths), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(image_paths), batch_size):
            end = min(len(image_paths), start + batch_size)
            batch_paths = image_paths[start:end]

            x1 = torch.stack(
                [load_image_tensor(p, size, hflip=False) for p in batch_paths], dim=0
            )
            x2 = torch.stack(
                [load_image_tensor(p, size, hflip=True) for p in batch_paths], dim=0
            )

            x1 = x1.to(device, non_blocking=True)
            x2 = x2.to(device, non_blocking=True)

            y1 = model(x1).view(-1).float()
            y2 = model(x2).view(-1).float()
            y = 0.5 * (y1 + y2)

            out[start:end] = y.detach().cpu().numpy()

            if start % (batch_size * 10) == 0:
                print(f"  infer {start}/{len(image_paths)}")
    return out


preds = np.zeros(len(test_images), dtype=np.float32)

available = [os.path.isfile(p) for p in MODEL_PATH_list]
if not any(available):
    print(
        "WARNING: No checkpoints found; using ImageNet weights for a single model to produce a valid submission."
    )
    MODEL_PATH_list = [None]
    resize_list = [224]
    available = [False]

for idx, (ckpt_path, sz) in enumerate(zip(MODEL_PATH_list, resize_list)):
    use_imagenet = ckpt_path is None
    model = get_se_resnet50_gem(pretrain="imagenet" if use_imagenet else None).to(
        device
    )

    if ckpt_path is not None:
        state = torch.load(ckpt_path, map_location="cpu")
        state = _clean_state_dict_for_model(state, model)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if len(missing) > 0:
            print(f"Missing keys (after filtering): {len(missing)}")
        if len(unexpected) > 0:
            print(f"Unexpected keys (after filtering): {len(unexpected)}")

    print(f"Predicting with model {idx+1}/{len(MODEL_PATH_list)} at size {sz} ...")
    model_pred = predict_model(model, test_images, sz, batch_size=16)
    preds += model_pred / len(MODEL_PATH_list)

raw_submission = pd.DataFrame({"id_code": test_ids, "diagnosis": preds})
raw_submission.to_csv("submission_raw_value.csv", index=False)

submission = raw_submission.copy()
submission.loc[submission.diagnosis < 0.7, "diagnosis"] = 0
submission.loc[
    (0.7 <= submission.diagnosis) & (submission.diagnosis < 1.5), "diagnosis"
] = 1
submission.loc[
    (1.5 <= submission.diagnosis) & (submission.diagnosis < 2.5), "diagnosis"
] = 2
submission.loc[
    (2.5 <= submission.diagnosis) & (submission.diagnosis < 3.5), "diagnosis"
] = 3
submission.loc[3.5 <= submission.diagnosis, "diagnosis"] = 4

submission["diagnosis"] = submission["diagnosis"].astype(np.int64).clip(0, 4)

submission = submission.set_index("id_code").loc[test_ids].reset_index()

submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote: submission.csv")
print("submission.csv shape:", submission.shape)
print("Unique diagnosis counts:\n", submission["diagnosis"].value_counts().sort_index())
