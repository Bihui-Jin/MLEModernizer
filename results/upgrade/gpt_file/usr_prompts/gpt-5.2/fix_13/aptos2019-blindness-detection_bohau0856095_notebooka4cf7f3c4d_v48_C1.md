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

0.9062655211787224

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.01221) has done: 'I make the code robust to missing external weight files by falling back to the same model architecture loaded with timm’s pretrained EfficientNet weights (so it still produces a reasonable submission instead of crashing). I also fix the CUDA/CPU dtype/device mismatch by ensuring the loaded state_dict is moved onto the same device as the model and inputs. Finally, I prevent the downstream “submission length mismatch” by only asserting after successful inference, and I keep the submission formatting aligned to `test.csv` order with the required `id_code,diagnosis` columns.'
- What this solution (achieved 0.02381) has done: 'Your current 0.01221 score is consistent with an “untrained head” problem: when the external weights aren’t found, the code only swaps in a pretrained backbone but leaves the classifier/regressor/ordinal heads randomly initialized, so predictions collapse and QWK is near-random. To move the score sharply toward the 0.906 target while keeping core logic identical, I change the fallback to load full-image-net pretrained weights for the exact same tf_efficientnet_b4_ns backbone (instead of replacing the backbone module), and then use the model’s intended `final=True` path for inference (the final regressor is already part of the architecture). I also run inference as a batch DataLoader (same transforms, same model) to reduce overhead and stay within the time budget, without changing semantics. These are minimal, directly score-relevant changes: they make the fallback produce meaningful features and use the intended output head.'
- What this solution (achieved 0.02381) has done: 'The crash comes from the custom GeM layer: its learnable parameter `p` stays on CPU after you replace the backbone’s `global_pool` (or after moving the model), so `x` is on CUDA while `p` is on CPU. I fix this in a minimal, score-neutral way by making `gem()` always use `p.to(x.device)` (and matching dtype), which prevents device mismatches regardless of how the module was moved/assigned. I also ensure that after setting `net.backbone.global_pool = GeM(...)` we move the whole model back to the target device once, so all parameters land correctly. No changes to model architecture, thresholds, transforms, or inference semantics; it run end-to-end and write `submission.csv`.'
- What this solution (achieved 0.07391) has done: 'Your score is far below the target (0.02381 vs 0.9063), and the main reason is that when the external competition weights are missing you only preload the backbone, leaving all heads (classifier/regressor/ordinal/final_regressor) randomly initialized—so predictions are near-random. To move sharply toward the target while preserving the exact same architecture and inference semantics, I change the fallback to load ImageNet pretrained weights into the *entire* `ThreeStage_Model` where possible (i.e., backbone + any matching layers), and I also fix the backbone output dimension bug by using `num_classes=0` so the backbone emits feature vectors (not 1000-class logits), matching your heads. Finally, I keep the same `final=True` inference path, but add deterministic settings and a small, safe DataLoader speed tweak so inference reliably completes within the time budget without changing predictions.'
- What this solution (achieved 0.0) has done: 'Your current score is far below the target because the fallback path only loads ImageNet weights into the backbone, leaving *all heads* (classifier/regressor/ordinal/final_regressor) randomly initialized, so predictions are essentially random. To move the score sharply upward while preserving the exact same architecture and inference semantics, I (1) try harder to locate and load the intended competition weight file (including common `.pth/.pt` alternatives), and (2) if not found, load a pretrained EfficientNet-B4 feature extractor **and freeze the random heads into a stable, non-random “neutral” state** so outputs aren’t dominated by randomness. This keeps your exact forward path (`final=True`) and thresholding unchanged, but makes the fallback deterministic and much less noisy (typically improving QWK substantially versus random). I also keep submission alignment exactly matching `test.csv`.'
- What this solution (achieved -0.01725) has done: 'Your current 0.0 score is caused by the “neutralized heads” fallback: it forces the model to output a constant value (~2 after sigmoid), which collapses predictions and yields QWK≈0. To move the score sharply upward toward the 0.906 target while keeping the same architecture and inference semantics, I remove head-neutralization and instead initialize the heads in a deterministic, non-degenerate way (biasing the classifier to the majority class and making the final regressor approximate the regressor output). I also fix a hidden backbone/feature-dim mismatch in your `Regressor` helper (use `num_classes=0` + correct linear in_features) to avoid accidentally training/inferencing on 1000-logit outputs if it’s ever used. These are minimal, directly score-relevant changes; no training loop, transforms, thresholds, or metric semantics are altered, and the script still writes a valid `submission.csv`.'
- What this solution (achieved 0.25821) has done: 'Your current negative QWK is consistent with a “constant-ish/poorly calibrated regressor” fallback: even with an ImageNet-pretrained backbone, your randomly initialized heads (and especially the final regressor) produce nearly uninformative, monotone outputs that don’t map well to the 0–4 ordinal labels. To move the score sharply upward toward the 0.906 target without changing your model architecture or training loop (you have none), the smallest score-relevant change is to calibrate the 4 thresholds on the *training set* using out-of-fold predictions from the same forward path (`final=True`), then apply those optimized thresholds at test time. This preserves evaluation semantics (still regression→threshold→class for QWK) and keeps all layers/forward logic intact, but aligns the discretization step to maximize QWK, which is typically the dominant lever for this competition. I also keep your weight-loading logic unchanged, only adding a robust threshold-fitting routine and using the fitted thresholds in `regress2class`.'
- What this solution (achieved 0.25821) has done: 'Your current gap to the target is large (0.258 → 0.906, higher-is-better), and the biggest issue is that your thresholds are being fit on **in-sample** predictions from a model whose heads are effectively untrained (fallback path), so the fitted thresholds overfit and don’t generalize to test—QWK collapses. I keep your exact model/forward (`final=True`) and transforms, but change threshold fitting to use **out-of-fold (OOF) predictions** via a simple 5-fold split, which is the minimal calibration change that usually boosts public QWK substantially for this competition. I also make the threshold optimizer a bit more stable by using a safer initialization from the OOF prediction quantiles (still 4 monotone thresholds, same discretization semantics). Everything still runs end-to-end and writes `submission.csv` with the required columns/order.'
- What this solution (achieved 0.0) has done: 'Your current score (0.258) is far below the target (0.906), so we should improve legitimately without changing the model architecture or training approach. The biggest issue is that you fit thresholds using OOF predictions from a *fixed* (non-trained) model; that’s fine, but you’re currently recomputing inference on every fold (wasteful) and—more importantly—your threshold fitting is unconstrained to the actual label distribution, often producing class-collapse on test. I (1) compute OOF predictions in one pass (still OOF, same semantics) by predicting the full train set once and then indexing by folds (since the model is fixed), and (2) fit thresholds to match the training label priors via quantile mapping (still 4 monotone thresholds, same regress→threshold→class semantics), then do a small local search around them for QWK. This keeps your forward path (`final=True`), transforms, and discretization logic intact, but makes thresholds much more likely to generalize, moving QWK toward the target.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is almost certainly coming from a submission-format/order issue or a prediction collapse caused by fitting thresholds on in-sample train predictions from a fixed (untrained-head) fallback model. To move the score upward toward 0.906 while keeping your model and inference path unchanged, I (1) fix DataLoader stability (avoid `persistent_workers=True` when `num_workers=0` and default Kaggle constraints), (2) replace the in-sample threshold fit with true out-of-fold (StratifiedKFold) threshold fitting using the *same* `net(imgs, final=True)` outputs, and (3) guarantee `submission.csv` exactly matches `test.csv` order with integer labels 0–4. These are minimal, metric-aligned changes that typically improve QWK a lot without altering architecture/loss/training loops.'

# 9. Code solution

## === cell 0
import os
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
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.transforms import functional as FT
from PIL import Image, ImageChops

from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold
import timm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



## === cell 1
threshold = [0.75, 1.5, 2.5, 3.5]


def regress2class(out: torch.Tensor) -> torch.Tensor:
    prediction = torch.zeros(out.size(0), device=out.device, dtype=torch.float32)
    for i in range(4):
        prediction += (out.data >= threshold[i]).squeeze()
    return prediction


def ordinal2class_prob(out: torch.Tensor) -> torch.Tensor:
    pred_prob = torch.zeros(out.size(0), 5, device=out.device, dtype=out.dtype)
    pred_prob[:, 0] = (1 - out[:, 0]).squeeze()
    pred_prob[:, 1] = (out[:, 0] * (1 - out[:, 1])).squeeze()
    pred_prob[:, 2] = (out[:, 1] * (1 - out[:, 2])).squeeze()
    pred_prob[:, 3] = (out[:, 2] * (1 - out[:, 3])).squeeze()
    pred_prob[:, 4] = out[:, 3].squeeze()
    return F.softmax(pred_prob, dim=1)


def regress2class_prob(out: torch.Tensor) -> torch.Tensor:
    pred_prob = torch.zeros((out.size(0), 5), device=out.device, dtype=out.dtype)
    for i in range(out.size(0)):
        v = float(out[i].detach().cpu().item())
        if v < 4.0:
            l1 = int(math.floor(v))
            l2 = int(math.ceil(v))
            pred_prob[i, l1] = 1 - (v - l1)
            pred_prob[i, l2] = 1 - (l2 - v)
        else:
            pred_prob[i, 4] = 1.0
    return pred_prob




## === cell 2
def gem(x, p=3, eps=1e-6):
    if torch.is_tensor(p):
        p = p.to(device=x.device, dtype=x.dtype)
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
        self.backbone = timm.create_model(
            "tf_efficientnet_b5_ns", pretrained=False, num_classes=0
        )
        self.backbone.global_pool = GeM(flatten=True)
        feat_dim = getattr(self.backbone, "num_features", 1000)
        self.regressor = nn.Linear(feat_dim, 1)

    def forward(self, x):
        x = self.backbone(x)
        out = self.regressor(x)
        out = torch.sigmoid(out) * 4.5
        return out


class ThreeStage_Model(nn.Module):
    def __init__(self, backbone=None):
        super(ThreeStage_Model, self).__init__()

        self.backbone = timm.create_model(
            "tf_efficientnet_b4_ns", pretrained=False, num_classes=0
        )
        self.backbone.global_pool = GeM(flatten=True)

        feat_dim = getattr(self.backbone, "num_features", 1000)

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
DATA_ROOT = "../input/aptos2019-blindness-detection"
TEST_CSV = f"{DATA_ROOT}/test.csv"
TEST_IMG_DIR = f"{DATA_ROOT}/test_images"
TRAIN_CSV = f"{DATA_ROOT}/train.csv"
TRAIN_IMG_DIR = f"{DATA_ROOT}/train_images"

test_df = pd.read_csv(TEST_CSV)
test_ids = test_df["id_code"].values

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


def find_weight_file(preferred_rel_path: str) -> str:
    candidates = []
    if preferred_rel_path and os.path.exists(preferred_rel_path):
        candidates.append(preferred_rel_path)

    if preferred_rel_path:
        base, ext = os.path.splitext(preferred_rel_path)
        for e in [".pkl", ".pth", ".pt", ".bin"]:
            p = base + e
            if os.path.exists(p):
                candidates.append(p)

    fnames = []
    if preferred_rel_path:
        base = os.path.splitext(os.path.basename(preferred_rel_path))[0]
        fnames = [base + e for e in [".pkl", ".pth", ".pt", ".bin"]]
    for fname in fnames:
        candidates.extend(glob.glob(f"/kaggle/input/**/{fname}", recursive=True))

    candidates = [c for c in candidates if os.path.isfile(c)]
    if not candidates:
        return ""
    candidates = sorted(set(candidates))
    chosen = candidates[0]
    print("Using weight file:", chosen)
    return chosen


def init_fallback_heads_stably(net: ThreeStage_Model, train_csv_path: str):
    train_df_ = pd.read_csv(train_csv_path)
    counts = (
        train_df_["diagnosis"]
        .value_counts()
        .reindex([0, 1, 2, 3, 4], fill_value=0)
        .astype(float)
        .values
    )
    priors = counts / max(counts.sum(), 1.0)
    priors = np.clip(priors, 1e-6, 1.0)

    last = net.classifier[-1]
    if isinstance(last, nn.Linear) and last.out_features == 5:
        nn.init.zeros_(last.weight)
        with torch.no_grad():
            last.bias.copy_(torch.tensor(np.log(priors), dtype=last.bias.dtype))

    final_lin = net.final_regressor[-1]
    if isinstance(final_lin, nn.Linear) and final_lin.in_features == 10:
        nn.init.zeros_(final_lin.weight)
        nn.init.zeros_(final_lin.bias)
        with torch.no_grad():
            final_lin.weight[0, 5] = 1.0  # pass-through raw regressor logit


net = ThreeStage_Model().to(device)

weight_path = find_weight_file("../input/weights/B4_3stage_36epoch_CLAHE.pkl")
if weight_path:
    state = torch.load(weight_path, map_location="cpu")
    if (
        isinstance(state, dict)
        and "state_dict" in state
        and isinstance(state["state_dict"], dict)
    ):
        state = state["state_dict"]
    missing, unexpected = net.load_state_dict(state, strict=False)
    print(
        f"Loaded weights with strict=False. Missing keys: {len(missing)} Unexpected keys: {len(unexpected)}"
    )
else:
    print(
        "WARNING: External weights not found. Falling back to timm pretrained backbone + stable head init (non-degenerate)."
    )
    pretrained_net = timm.create_model(
        "tf_efficientnet_b4_ns", pretrained=True, num_classes=0
    )
    pretrained_net.global_pool = GeM(flatten=True)
    missing, unexpected = net.backbone.load_state_dict(
        pretrained_net.state_dict(), strict=False
    )
    print(
        f"Backbone pretrained load strict=False. Missing keys: {len(missing)} Unexpected keys: {len(unexpected)}"
    )

    init_fallback_heads_stably(net, TRAIN_CSV)
    net = net.to(device)

net.eval()




## === cell 5
class ImageIdDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_dir: str, transform):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        idx = row["id_code"]
        image_name = f"{self.img_dir}/{idx}.png"
        img = Image.open(image_name).convert("RGB")
        img = self.transform(img)
        y = row["diagnosis"] if "diagnosis" in self.df.columns else -1
        return idx, img, int(y)


def _make_loader(ds, batch_size: int):
    num_workers = 2 if os.cpu_count() and os.cpu_count() > 2 else 0
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
    )


def get_continuous_preds_for_df(
    df_subset: pd.DataFrame, img_dir: str, batch_size: int = 8
):
    ds = ImageIdDataset(df_subset, img_dir, transform)
    loader = _make_loader(ds, batch_size=batch_size)

    ids_all, y_true_all, y_pred_all = [], [], []
    with torch.no_grad():
        for ids_batch, imgs, y in loader:
            imgs = imgs.to(device, non_blocking=True)
            out = net(imgs, final=True).squeeze(1)  # (B,)
            ids_all.extend(list(ids_batch))
            y_true_all.append(np.asarray(y, dtype=np.int64))
            y_pred_all.append(out.detach().float().cpu().numpy())
    return np.array(ids_all), np.concatenate(y_true_all), np.concatenate(y_pred_all)


def apply_thresholds_np(pred_cont: np.ndarray, thr_list):
    thr = np.array(thr_list, dtype=np.float64)
    thr = np.sort(thr)
    thr = np.maximum.accumulate(thr + np.array([0.0, 1e-4, 2e-4, 3e-4]))
    cls = np.zeros(pred_cont.shape[0], dtype=np.int64)
    for t in thr:
        cls += (pred_cont >= t).astype(np.int64)
    return np.clip(cls, 0, 4)


def fit_thresholds_for_qwk_with_priors(
    y_true: np.ndarray, y_pred_cont: np.ndarray, init_thr=None
):
    y_true = y_true.astype(int)
    y_pred_cont = y_pred_cont.astype(np.float64)

    if init_thr is None:
        counts = np.bincount(y_true, minlength=5).astype(np.float64)
        priors = counts / max(counts.sum(), 1.0)
        cum = np.cumsum(priors)
        qs = np.clip(cum[:4], 1e-6, 1 - 1e-6)
        init_thr = np.quantile(y_pred_cont, qs).astype(np.float64)
        init_thr = np.clip(init_thr, 0.0, 4.5)
    else:
        init_thr = np.array(init_thr, dtype=np.float64)

    def score(thr):
        cls = apply_thresholds_np(y_pred_cont, thr)
        return cohen_kappa_score(y_true, cls, weights="quadratic")

    best_thr = init_thr.copy()
    best_score = score(best_thr)

    step_sizes = [0.2, 0.1, 0.05, 0.02]
    for step in step_sizes:
        improved = True
        it = 0
        while improved and it < 30:
            improved = False
            it += 1
            for i in range(4):
                for direction in (-1.0, 1.0):
                    cand = best_thr.copy()
                    cand[i] += direction * step
                    cand = np.clip(cand, 0.0, 4.5)
                    cand = np.sort(cand)
                    cand = np.maximum.accumulate(
                        cand + np.array([0.0, 1e-4, 2e-4, 3e-4])
                    )
                    s = score(cand)
                    if s > best_score + 1e-7:
                        best_score = s
                        best_thr = cand
                        improved = True

    return best_thr.tolist(), best_score


train_df = pd.read_csv(TRAIN_CSV).reset_index(drop=True)
y_true = train_df["diagnosis"].values.astype(np.int64)

t0 = time.time()
_, _, full_train_pred = get_continuous_preds_for_df(
    train_df, TRAIN_IMG_DIR, batch_size=8
)
print("Full-train inference seconds:", round(time.time() - t0, 2))

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_pred = np.zeros(len(train_df), dtype=np.float64)
for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(train_df)), y_true), 1):
    oof_pred[va_idx] = full_train_pred[va_idx]
    print(f"Prepared OOF fold {fold}/5 size={len(va_idx)}")

threshold, oof_qwk = fit_thresholds_for_qwk_with_priors(
    y_true, oof_pred, init_thr=threshold
)
print("OOF QWK (for threshold fit):", oof_qwk)
print("Using thresholds for test:", threshold)




## === cell 6
class TestDataset(Dataset):
    def __init__(self, ids, img_dir, transform):
        self.ids = list(ids)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        idx = self.ids[i]
        image_name = f"{self.img_dir}/{idx}.png"
        img = Image.open(image_name).convert("RGB")
        img = self.transform(img)
        return idx, img


test_ds = TestDataset(test_ids, TEST_IMG_DIR, transform)
test_loader = _make_loader(test_ds, batch_size=8)

id_to_pred = {}
with torch.no_grad():
    seen = 0
    for ids_batch, imgs in test_loader:
        imgs = imgs.to(device, non_blocking=True)
        out = net(imgs, final=True)  # (B, 1)
        pred = regress2class(out.squeeze(1))
        for k in range(len(ids_batch)):
            id_to_pred[str(ids_batch[k])] = int(pred[k].item())

        seen += len(ids_batch)
        if seen % 50 == 0 or seen == len(test_ids):
            print(seen, "/", len(test_ids))

sub_df = pd.DataFrame({"id_code": test_ids})
sub_df["diagnosis"] = (
    sub_df["id_code"].astype(str).map(id_to_pred).fillna(0).astype(int).clip(0, 4)
)

assert len(sub_df) == len(
    test_ids
), f"Submission length mismatch: got {len(sub_df)} expected {len(test_ids)}."
sub_df.to_csv("submission.csv", index=False)
print(sub_df.head())
print("Wrote submission.csv with shape:", sub_df.shape)
print("submission.csv path:", os.path.abspath("submission.csv"))
