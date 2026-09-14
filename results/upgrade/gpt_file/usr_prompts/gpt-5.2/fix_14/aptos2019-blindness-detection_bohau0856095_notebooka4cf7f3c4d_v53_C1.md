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

0.9118061233462

# 6. Current score

0.33572

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.01221) has done: 'I fix the pipeline so it always produces a valid `submission.csv` end-to-end by (1) removing the hard dependency on a missing weight file and falling back safely to an available pretrained backbone when the `.pkl` isn’t present, and (2) fixing the CUDA/CPU dtype mismatch by ensuring the loaded state dict (if found) is moved to the same device as the model. I also correct the prediction loop to avoid creating an empty submission if any single image load fails (robust per-image exception handling) while keeping the core model and thresholding logic unchanged. These changes are necessary for correctness (no runtime errors, non-empty CSV) and should also improve score versus random/untrained inference by using ImageNet pretrained weights when competition weights are unavailable.'
- What this solution (achieved 0.00484) has done: 'Your current score is far below the target, and the biggest issue is that the inference pipeline is effectively using an untrained head (and likely mismatched normalization) when the competition weights are missing, which makes predictions close to random. To move the score upward with minimal semantic change, I keep the same model and regression-to-class thresholding, but (1) ensure the EfficientNet backbone is always ImageNet-pretrained (even in the “no .pkl found” path) and (2) switch normalization to the standard ImageNet stats that match the pretrained backbone. These are small, directly relevant changes that should substantially increase QWK compared to the current near-random output, while preserving the overall approach and output format. The script still run end-to-end and write a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your current score (0.00484) is far below the target (0.9118), and the main reason is that the inference path is using the 5-way classifier head (ImageNet) rather than the DR-specific “final” regressor head that your thresholding expects, so predictions are effectively random. With minimal change to core logic, I (1) keep the exact same model/thresholding but call `net(img, final=True)` so we use the intended DR regression output, and (2) ensure weight loading (when present) doesn’t silently miss keys due to `module.` prefixes. These are directly relevant to metric alignment and should substantially increase QWK toward the target while keeping everything else (transforms, thresholds, architecture) intact. The script still run end-to-end and write a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is consistent with producing an invalid or degenerate submission (often all-one-class) rather than a model-quality issue, so the smallest high-impact fix is to ensure predictions are made in a numerically safe way and that the regressor-to-class conversion doesn’t silently break due to shape/device edge cases. I (1) make `regress2class` operate fully on-device with correct shapes (no `.data`, no CPU toggling inside), and (2) explicitly clamp the regressor output to `[0, 4.0]` before thresholding to avoid out-of-range effects that can collapse predictions. These changes keep the exact same model, weights logic, transforms, and thresholding semantics, but remove fragile tensor handling that can yield all-zeros/all-4s submissions and thus a near-zero QWK. The script still run end-to-end and write a valid `submission.csv` with the required columns and ordering.'
- What this solution (achieved 0.20594) has done: 'Your current 0.0 score is most consistent with the submission containing a single class (or near-single-class) due to missing competition weights, which makes the randomly-initialized heads dominate and collapse predictions. To move the score upward with minimal semantic change, I keep the exact same model and regress-to-class thresholding, but when the `.pkl` weights are not found I fall back to a safer inference path that uses the pretrained backbone + the classifier head only (argmax) rather than the untrained final regressor. When weights are found, the behavior remains exactly the same as your intended pipeline (`final=True` + `regress2class`). This should avoid degenerate all-one-class outputs and move QWK upward toward your target while still producing a valid `submission.csv`.'
- What this solution (achieved 0.00606) has done: 'Your current score (0.20594) is far below the target (0.9118), and the main bottleneck is that when competition weights are missing you fall back to the ImageNet-pretrained 5-way head, which is not aligned to this DR label space and typically yields near-random QWK. With minimal change to your existing architecture and inference semantics, I keep the same ThreeStage model and `regress2class` thresholds but switch the fallback to a “no-competition-weights calibration” path: use the model’s own regressor/ordinal outputs (already defined) and combine them into a single continuous severity estimate, then threshold exactly as before. This uses meaningful pretrained backbone features without introducing new training, and should move QWK upward toward the target while preserving your overall pipeline and producing the same submission format. I also keep your robust image-load exception handling and submission row ordering unchanged.'
- What this solution (achieved 0.08643) has done: 'Your current score is far below the target, and the most likely reason (given the code) is that you are almost always in the “no competition weights” fallback, where the regressor/ordinal heads are randomly initialized and thus produce near-random predictions. To move the score upward with minimal semantic change, I keep the same ThreeStage model and the same `regress2class` thresholds, but I add a tiny “quick-fit calibration” step: train only the already-defined `final_regressor` (and nothing else) on the provided `train.csv` using frozen backbone+heads outputs as features. This preserves your architecture, loss family (still regression to severity), and inference pathway (`net(..., final=True)`), while turning the random fusion into a DR-aligned mapping and should substantially improve QWK toward your target within the time budget. If the competition weights file is found, the script behaves as before (no training) to avoid unnecessary changes.'
- What this solution (achieved 0.50511) has done: 'We need to stop the CuBLAS deterministic error by either setting `CUBLAS_WORKSPACE_CONFIG` before importing torch or by disabling `torch.use_deterministic_algorithms(True)`; here we do both safely (set env early and fall back to non-deterministic if needed) so Linear layers on GPU run. Next, your submission-building failed because `batch_ids` are not plain strings (they come from a DataLoader collation), so the produced `id_code` values didn’t match `test_df`; we fix this by using a custom `collate_fn` that keeps `id_code` as a list of strings. Finally, we make the final submission robust by merging predictions onto `test_df` (so ordering is guaranteed and missing predictions become 0), while keeping your model/inference logic unchanged.'
- What this solution (achieved 0.46778) has done: 'Your current score (0.50511) is far below the target (0.9118), so we should push performance up with the smallest changes that keep your architecture/inference semantics intact. The biggest likely mismatch is that your backbone is ImageNet-pretrained but the input size/resize scheme doesn’t match the backbone’s native preprocessing, which can noticeably hurt predictions; we switch to timm’s built-in `create_transform` for the exact model to align resize/crop/interpolation/normalization while keeping your trim/cropTo4_3 steps. Next, your fallback “quick-fit” trains `final_regressor` against raw labels (0–4) but the model output is scaled to 0–4.5; we map labels into the same 0–4.5 scale to remove a systematic bias without changing the loss or training loop. These two minimal, metric-relevant fixes should move QWK upward toward the target while still producing the same `submission.csv` format.'
- What this solution (achieved 0.33572) has done: 'Your current gap to the target is large (0.4678 → 0.9118), so we need a meaningful uplift while keeping your architecture and overall inference semantics intact. The biggest low-risk win is to align your “quick-fit” stage with what the metric actually cares about: quadratic weighted kappa on *discrete* classes 0–4, not MSE on a 0–4.5 continuous target. We keep the exact same feature extraction, same final_regressor head, and same training loop structure, but swap the quick-fit loss to a differentiable surrogate that matches QWK better (weighted cross-entropy on class bins induced by your existing thresholds), and then keep the same `regress2class` at inference. Additionally, we fix a small training-mode bug: feature extraction for the cache should run with the backbone in `eval()` and with gradients off (already mostly true), but the head training should not reuse the extra sigmoid*4.5 scaling inside the loss; instead we train on logits mapped to class probabilities via fixed bin edges to stabilize calibration toward your thresholds.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import glob
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

from PIL import Image, ImageChops
import timm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

_DETERMINISTIC_ENABLED = False
try:
    torch.use_deterministic_algorithms(True)
    _DETERMINISTIC_ENABLED = True
except Exception as e:
    print(f"WARNING: could not enable deterministic algorithms: {repr(e)}")
    _DETERMINISTIC_ENABLED = False



## === cell 1
threshold = [0.75, 1.5, 2.5, 3.5]


def regress2class(out: torch.Tensor) -> torch.Tensor:
    """
    out: shape [B] or [B,1], float tensor on any device
    returns: shape [B] int-like tensor on CPU (for submission)
    """
    if out.ndim == 2 and out.size(1) == 1:
        out = out.squeeze(1)
    out = out.to(dtype=torch.float32)

    out = out.clamp(0.0, 4.0)

    pred = torch.zeros(out.size(0), device=out.device, dtype=torch.int64)
    for t in threshold:
        pred += (out >= t).to(torch.int64)
    return pred.cpu()


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
DATA_DIR = "../input/aptos2019-blindness-detection"
TEST_CSV = os.path.join(DATA_DIR, "test.csv")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")

test_df = pd.read_csv(TEST_CSV)
test_ids = test_df["id_code"].astype(str).values

input_size = 380

timm_cfg = timm.data.resolve_model_data_config(net_name := "tf_efficientnet_b4_ns")
timm_tf = timm.data.create_transform(**timm_cfg, is_training=False)

transform = transforms.Compose(
    [
        trim(),
        cropTo4_3(),
        timm_tf,
    ]
)

net = ThreeStage_Model()

expected_name = "B4_3stage_52epoch_CLAHE.pkl"
candidates = glob.glob(os.path.join("../input", "**", expected_name), recursive=True)

has_competition_weights = False

if len(candidates) > 0:
    weights_path = candidates[0]
    print("Loading weights from:", weights_path)
    state = torch.load(weights_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    if isinstance(state, dict):
        has_module_prefix = any(k.startswith("module.") for k in state.keys())
        if has_module_prefix:
            state = {k.replace("module.", "", 1): v for k, v in state.items()}

    missing, unexpected = net.load_state_dict(state, strict=False)
    has_competition_weights = True
    if missing or unexpected:
        print(
            f"State dict loaded with strict=False. Missing keys: {len(missing)}, unexpected keys: {len(unexpected)}"
        )
else:
    print(
        f"WARNING: Could not find {expected_name} under ../input. Will quick-fit only the existing final_regressor on train.csv using frozen backbone/heads to move QWK upward."
    )

net = net.to(device)
net.eval()

print("Model ready on device:", next(net.parameters()).device)
print("Competition weights loaded:", has_competition_weights)
print("Using timm data config:", timm_cfg)



## === cell 5
from torch.utils.data import Dataset, DataLoader


def _extract_features_for_final_regressor(
    model: ThreeStage_Model, img_t: torch.Tensor
) -> torch.Tensor:
    x = model.backbone(img_t)
    c_out = model.classifier(x)  # [B,5]
    r_out = model.regressor(x)  # [B,1] (RAW, not sigmoid-scaled here)
    o_out = model.ordinal(x)  # [B,4] (RAW)
    return torch.cat((c_out, r_out, o_out), dim=1)


class _AptosTrainDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_dir: str, transform):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img_path = os.path.join(self.img_dir, f"{row['id_code']}.png")
        try:
            im = Image.open(img_path).convert("RGB")
            x = self.transform(im)
            y_cls = int(row["diagnosis"])
            ok = 1
        except Exception:
            x = torch.zeros(3, input_size, input_size, dtype=torch.float32)
            y_cls = 0
            ok = 0
        return x, y_cls, ok


def _build_train_feature_cache_dataloader(
    model: ThreeStage_Model,
    train_csv: str,
    train_img_dir: str,
    max_items: int = 1400,
    batch_size: int = 32,
    num_workers: int = None,
):
    train_df = pd.read_csv(train_csv)
    train_df = train_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    if max_items is not None:
        train_df = train_df.iloc[:max_items].reset_index(drop=True)

    ds = _AptosTrainDataset(train_df, train_img_dir, transform)

    if num_workers is None:
        cpu = os.cpu_count() or 2
        num_workers = min(4, max(1, cpu // 2))

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )

    feats_cpu = []
    ys_cpu = []

    model.eval()
    with torch.inference_mode():
        for xb, yb_cls, okb in loader:
            if okb.sum().item() == 0:
                continue
            xb = xb.to(device, non_blocking=True)
            feat_b = _extract_features_for_final_regressor(model, xb).detach().cpu()
            feats_cpu.append(feat_b)
            ys_cpu.append(yb_cls.to(dtype=torch.int64).cpu())

    if len(feats_cpu) == 0:
        return None, None

    X = torch.cat(feats_cpu, dim=0)
    y_cls = torch.cat(ys_cpu, dim=0)
    return X, y_cls


def _severity_to_soft_bins(sev: torch.Tensor, edges: torch.Tensor, tau: float = 0.18):
    """
    Differentiable soft assignment of a scalar severity to 5 class bins defined by edges.
    edges: tensor of shape [4] (same semantics as `threshold`)
    Returns probs: [B,5]
    """
    s = (sev.unsqueeze(1) - edges.view(1, -1)) / tau
    ge = torch.sigmoid(s)  # [B,4]
    p0 = 1.0 - ge[:, 0]
    p1 = ge[:, 0] * (1.0 - ge[:, 1])
    p2 = ge[:, 1] * (1.0 - ge[:, 2])
    p3 = ge[:, 2] * (1.0 - ge[:, 3])
    p4 = ge[:, 3]
    probs = torch.stack([p0, p1, p2, p3, p4], dim=1).clamp_min(1e-8)
    probs = probs / probs.sum(dim=1, keepdim=True)
    return probs


def quick_fit_final_regressor_if_needed(
    model: ThreeStage_Model,
    train_csv: str,
    train_img_dir: str,
    max_seconds: float = 420.0,
    batch_size: int = 64,
    lr: float = 3e-3,
):
    if has_competition_weights:
        return

    if not os.path.exists(train_csv):
        print("WARNING: train.csv not found; skipping quick-fit.")
        return

    for p in model.parameters():
        p.requires_grad = False
    for p in model.final_regressor.parameters():
        p.requires_grad = True

    cache_path = os.path.join(
        "../working", f"final_regressor_cache_b4_3stage_in{input_size}_n1400.pt"
    )

    X = y_cls = None
    if os.path.exists(cache_path):
        try:
            cache = torch.load(cache_path, map_location="cpu")
            X, y_cls = cache["X"], cache["y_cls"]
            print(
                f"Loaded feature cache: {cache_path} | X={tuple(X.shape)} y_cls={tuple(y_cls.shape)}"
            )
        except Exception:
            X = y_cls = None

    if X is None:
        print("Building frozen feature cache for quick-fit (DataLoader)...")
        X, y_cls = _build_train_feature_cache_dataloader(
            model,
            train_csv=train_csv,
            train_img_dir=train_img_dir,
            max_items=1400,
            batch_size=32,
            num_workers=None,
        )
        if X is None:
            print("WARNING: could not build feature cache; skipping quick-fit.")
            return
        try:
            torch.save({"X": X, "y_cls": y_cls}, cache_path)
            print(f"Saved feature cache: {cache_path}")
        except Exception as e:
            print(f"WARNING: could not save cache: {repr(e)}")

    X = X.to(device)
    y_cls = y_cls.to(device)

    model.train()
    opt = torch.optim.AdamW(
        model.final_regressor.parameters(), lr=lr, weight_decay=1e-4
    )

    edges = torch.tensor(threshold, device=device, dtype=torch.float32)
    cls_weights = torch.tensor(
        [1.0, 2.0, 4.0, 2.0, 1.0], device=device
    )  # mild emphasis mid classes
    cls_weights = cls_weights / cls_weights.mean()
    nll = nn.NLLLoss(weight=cls_weights)

    start = time.time()
    step = 0

    n = X.size(0)
    g = torch.Generator(device="cpu")
    g.manual_seed(42)

    while time.time() - start < max_seconds:
        perm = torch.randperm(n, generator=g, device="cpu")
        for s in range(0, n, batch_size):
            if time.time() - start >= max_seconds:
                break
            idx = perm[s : s + batch_size].to(device, non_blocking=True)
            feat = X[idx]
            y_b = y_cls[idx]

            raw = model.final_regressor(feat)  # [B,1]
            sev = (torch.sigmoid(raw).squeeze(1)) * 4.5  # [B] severity in [0,4.5]

            probs = _severity_to_soft_bins(sev, edges=edges, tau=0.18)  # [B,5]
            loss = nll(torch.log(probs), y_b)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            step += 1
            if step % 200 == 0:
                elapsed = time.time() - start
                print(
                    f"quick-fit step {step} | nll {loss.item():.4f} | elapsed {elapsed:.1f}s"
                )

    model.eval()
    print(
        f"Quick-fit done. Steps: {step}, elapsed: {time.time()-start:.1f}s, cache N={n}"
    )


try:
    quick_fit_final_regressor_if_needed(net, TRAIN_CSV, TRAIN_IMG_DIR)
except RuntimeError as e:
    msg = str(e)
    if (
        "CUBLAS_WORKSPACE_CONFIG" in msg
        or "not deterministic because it uses CuBLAS" in msg
    ):
        print(
            "WARNING: Determinism+CuBLAS error encountered; disabling deterministic algorithms and retrying."
        )
        try:
            torch.use_deterministic_algorithms(False)
        except Exception:
            pass
        quick_fit_final_regressor_if_needed(net, TRAIN_CSV, TRAIN_IMG_DIR)
    else:
        raise




## === cell 6
class _AptosTestDataset(Dataset):
    def __init__(self, ids, img_dir: str, transform):
        self.ids = list(map(str, ids))
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        idx = self.ids[i]
        image_name = os.path.join(self.img_dir, f"{idx}.png")
        try:
            img = Image.open(image_name).convert("RGB")
            x = self.transform(img)
            ok = 1
        except Exception:
            x = torch.zeros(3, input_size, input_size, dtype=torch.float32)
            ok = 0
        return idx, x, ok


def _test_collate_fn(batch):
    ids = [b[0] for b in batch]
    xb = torch.stack([b[1] for b in batch], dim=0)
    okb = torch.tensor([b[2] for b in batch], dtype=torch.int64)
    return ids, xb, okb


submission = []
num_failed = 0

cpu = os.cpu_count() or 2
num_workers = min(4, max(1, cpu // 2))
test_loader = DataLoader(
    _AptosTestDataset(test_ids, TEST_IMG_DIR, transform),
    batch_size=16 if torch.cuda.is_available() else 4,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(num_workers > 0),
    prefetch_factor=2 if num_workers > 0 else None,
    collate_fn=_test_collate_fn,
)


def _predict_batch(xb):
    try:
        return net(xb, final=True)
    except RuntimeError as e:
        msg = str(e)
        if (
            "CUBLAS_WORKSPACE_CONFIG" in msg
            or "not deterministic because it uses CuBLAS" in msg
        ):
            print(
                "WARNING: Determinism+CuBLAS error during inference; disabling deterministic algorithms and retrying."
            )
            try:
                torch.use_deterministic_algorithms(False)
            except Exception:
                pass
            return net(xb, final=True)
        raise


with torch.inference_mode():
    seen = 0
    for batch_ids, xb, okb in test_loader:
        if seen % 50 == 0:
            print(seen, "/", len(test_ids))
        seen += len(batch_ids)

        xb = xb.to(device, non_blocking=True)
        r_out = _predict_batch(xb)  # [B,1] in [0,4.5]
        pred = regress2class(r_out)  # CPU tensor [B]

        ok_np = okb.cpu().numpy().astype(np.int32)
        pred_np = pred.numpy().astype(np.int64)
        for i in range(len(batch_ids)):
            idx = str(batch_ids[i])
            if ok_np[i] == 0:
                num_failed += 1
                submission.append([idx, 0])
            else:
                submission.append([idx, int(pred_np[i])])

print("Predictions:", len(submission), "failed:", num_failed)



## === cell 7
pred_df = pd.DataFrame(submission, columns=["id_code", "diagnosis"])
pred_df["id_code"] = pred_df["id_code"].astype(str)
pred_df["diagnosis"] = pred_df["diagnosis"].astype("int64")

out_df = test_df.copy()
out_df["id_code"] = out_df["id_code"].astype(str)
out_df = out_df.merge(pred_df, on="id_code", how="left")
out_df["diagnosis"] = out_df["diagnosis"].fillna(0).astype(int)

out_path = "submission.csv"
out_df.to_csv(out_path, index=False)
print("Wrote:", out_path, "rows:", len(out_df))
print(out_df.head())
print(out_df["diagnosis"].value_counts().sort_index())
