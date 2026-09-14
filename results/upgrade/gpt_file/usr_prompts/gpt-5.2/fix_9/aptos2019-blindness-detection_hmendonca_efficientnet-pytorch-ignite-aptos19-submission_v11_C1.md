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

3.7

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
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124
tqdm==4.67.1

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

0.7959466149511489

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I remove the hard dependency on a missing external `.pth` checkpoint by training the same EfficientNet architecture on the provided `train.csv`/`train_images` so the notebook can run end-to-end in this environment. I also fix the CUDA/CPU dtype mismatch by ensuring the model weights and inputs live on the same device, and I add a small validation split to pick the best epoch without changing the core model/loss. Finally, I keep the existing test-time augmentation (flip + softmax averaging) and write a correctly formatted `submission.csv` with `id_code,diagnosis`.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is most consistent with a submission alignment bug (predictions not matched to the correct `id_code` order), not just weak modeling. I make the smallest change that preserves your model/training/inference logic: build the submission by merging predictions with `test.csv` and then reindexing to the exact `sample_submission.csv` order by `id_code`. This avoids accidental row-order mismatches and guarantees the correct pairing of each prediction with its image id, which should move the score sharply upward toward the target without changing the underlying model. I also add a quick assertion to ensure `id_code` uniqueness and full coverage so the notebook fails fast if something is off.'
- What this solution (achieved 0.0) has done: 'Your 0.0 public score is most consistent with a label-format mismatch for this competition: QWK expects ordinal class predictions 0–4, but the current pipeline outputs raw argmax classes from a cross-entropy model, which is usually poorly calibrated for QWK and can collapse toward a single class. To move the score upward toward your ~0.796 target while preserving the same model/loss/training loop, I keep training exactly as-is but add a minimal, metric-aligned post-processing step: learn 4 optimal thresholds on the validation split to convert the model’s expected class value (softmax-weighted) into discrete 0–4 labels. I then apply those thresholds at test time (still with the same flip-TTA averaging) and keep the same robust `id_code` alignment against `sample_submission.csv`. This is a small semantic change in prediction discretization only (not architecture/training), and it typically yields a large QWK jump versus plain argmax.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with the external checkpoint path being found but incompatible (loading succeeds or silently mismatches distribution), combined with skipping threshold fitting when a checkpoint exists; this often collapses predictions and tanks QWK. I keep the same EfficientNet architecture/training/inference/TTA, but make two minimal score-relevant fixes: always fit thresholds on a small validation split (even when a checkpoint is present) and ensure we fall back to local training if checkpoint loading fails. This preserves core semantics (same model/loss/loop), but aligns discretization to QWK and avoids “bad checkpoint” failure modes. I also add a strict sanity check that predicted labels are within {0..4} and that submission ordering matches `sample_submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is still most consistent with a submission/prediction mismatch rather than pure model quality, so I make the smallest changes that directly prevent “wrong label space” outputs. Specifically, I enforce that the discretization thresholds are strictly increasing and within sensible bounds, and I clamp the continuous predictions to the valid [0, 4] range before digitization so `np.digitize` can’t accidentally spill into invalid classes. I also remove the duplicate `ImageDataset` definition to avoid any subtle behavioral divergence between train/valid and test loading, keeping the exact same transforms, model, loss, and TTA logic. These changes preserve the core approach while making the predictions reliably ordinal 0–4 and aligned to the QWK evaluation.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is almost certainly not a “model quality” issue but an evaluation semantic issue: APTOS2019 expects **integer class labels 0–4** in `diagnosis`, while your current pipeline’s thresholding can still output `5` when the last threshold is pushed slightly below 4.0 during sanitization (then `np.digitize(4.0, bins<than 4)` becomes 5). I make the smallest change that directly prevents invalid labels by (1) ensuring the last threshold is always strictly below 4.0 and (2) clipping the digitized output to `[0, 4]` as a final safety net, without changing your model, training loop, TTA, or loss. This should move your score sharply upward toward the target by making the submission valid under the metric rather than being penalized/treated as garbage. I also add one explicit assertion to fail fast if anything outside 0–4 still appears.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with “evaluation-semantic” issues rather than model quality, so the smallest high-impact fix is to ensure the submission labels are exactly valid integers in {0,1,2,3,4} and that the mapping from predictions to `id_code` is guaranteed correct. I keep your model/training/TTA exactly the same, but make threshold fitting more robust by fitting thresholds on out-of-fold (OOF) validation predictions (same single split, just collected explicitly) and then applying strictly-sanitized thresholds with a final hard clip to [0,4]. I also add one extra alignment assertion that the test dataloader order matches `test.csv` order (since any accidental reordering would destroy QWK). These changes are minimal, do not alter architecture/loss/training loop, and are directly targeted at moving the score upward toward your ~0.796 target.'

# 9. Code solution

## === cell 0
import os, glob, shutil, pathlib, sys, subprocess, textwrap, random

subprocess.run("ls -la ../input/ | head -200", shell=True, check=False)
subprocess.run("ls -la ../input/* 2>/dev/null | head -200", shell=True, check=False)


def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    import numpy as np

    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


seed_everything(42)

DATA_ROOT = "../input/aptos2019-blindness-detection"
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
TEST_CSV = os.path.join(DATA_ROOT, "test.csv")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train_images")
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test_images")

model_path = "efficientNet_best.pth"
search_patterns = [
    "../input/efficientnet*/efficientNet_*.pth",
    "../input/**/efficientNet_*.pth",
    "../input/**/efficientnet*.pth",
    "../input/**/efficientnet_*.pth",
    "../input/**/efficientNet_best.pth",
    "../input/**/model*.pth",
]
candidates = []
for pat in search_patterns:
    candidates.extend(glob.glob(pat, recursive=True))

preferred = [p for p in candidates if os.path.basename(p) == "efficientNet_best.pth"]
ckpt_src = preferred[0] if preferred else (candidates[0] if candidates else None)

if ckpt_src is not None and os.path.abspath(ckpt_src) != os.path.abspath(model_path):
    shutil.copy2(ckpt_src, model_path)

print("Found external checkpoint:" if ckpt_src else "No external checkpoint found.")
print("Will use checkpoint path:", model_path)
print("Checkpoint exists already:", os.path.exists(model_path))



## === cell 1
import torch
import torch.nn as nn


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Flatten(nn.Module):
    def forward(self, x):
        return x.reshape(x.shape[0], -1)


class SqueezeExcitation(nn.Module):

    def __init__(self, inplanes, se_planes):
        super(SqueezeExcitation, self).__init__()
        self.reduce_expand = nn.Sequential(
            nn.Conv2d(
                inplanes, se_planes, kernel_size=1, stride=1, padding=0, bias=True
            ),
            Swish(),
            nn.Conv2d(
                se_planes, inplanes, kernel_size=1, stride=1, padding=0, bias=True
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x_se = torch.mean(x, dim=(-2, -1), keepdim=True)
        x_se = self.reduce_expand(x_se)
        return x_se * x


from torch.nn import functional as F


class MBConv(nn.Module):
    def __init__(
        self,
        inplanes,
        planes,
        kernel_size,
        stride,
        expand_rate=1.0,
        se_rate=0.25,
        drop_connect_rate=0.2,
    ):
        super(MBConv, self).__init__()

        expand_planes = int(inplanes * expand_rate)
        se_planes = max(1, int(inplanes * se_rate))

        self.expansion_conv = None
        if expand_rate > 1.0:
            self.expansion_conv = nn.Sequential(
                nn.Conv2d(
                    inplanes,
                    expand_planes,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
                nn.BatchNorm2d(expand_planes, momentum=0.01, eps=1e-3),
                Swish(),
            )
            inplanes = expand_planes

        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(
                inplanes,
                expand_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                groups=expand_planes,
                bias=False,
            ),
            nn.BatchNorm2d(expand_planes, momentum=0.01, eps=1e-3),
            Swish(),
        )

        self.squeeze_excitation = SqueezeExcitation(expand_planes, se_planes)

        self.project_conv = nn.Sequential(
            nn.Conv2d(
                expand_planes, planes, kernel_size=1, stride=1, padding=0, bias=False
            ),
            nn.BatchNorm2d(planes, momentum=0.01, eps=1e-3),
        )

        self.with_skip = stride == 1
        self.drop_connect_rate = float(drop_connect_rate)

    def _drop_connect(self, x):
        keep_prob = 1.0 - self.drop_connect_rate
        drop_mask = torch.rand(x.shape[0], 1, 1, 1, device=x.device) + keep_prob
        drop_mask = drop_mask.type_as(x)
        drop_mask.floor_()
        return drop_mask * x / keep_prob

    def forward(self, x):
        z = x
        if self.expansion_conv is not None:
            x = self.expansion_conv(x)

        x = self.depthwise_conv(x)
        x = self.squeeze_excitation(x)
        x = self.project_conv(x)

        if x.shape == z.shape and self.with_skip:
            if (
                self.training
                and self.drop_connect_rate is not None
                and self.drop_connect_rate > 0
            ):
                x = self._drop_connect(x)
            x += z
        return x


from collections import OrderedDict
import math


def init_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, a=0, mode="fan_out")
    elif isinstance(module, nn.Linear):
        init_range = 1.0 / math.sqrt(module.weight.shape[1])
        nn.init.uniform_(module.weight, a=-init_range, b=init_range)


class EfficientNet(nn.Module):

    def _setup_repeats(self, num_repeats):
        return int(math.ceil(self.depth_coefficient * num_repeats))

    def _setup_channels(self, num_channels):
        num_channels *= self.width_coefficient
        new_num_channels = math.floor(num_channels / self.divisor + 0.5) * self.divisor
        new_num_channels = max(self.divisor, new_num_channels)
        if new_num_channels < 0.9 * num_channels:
            new_num_channels += self.divisor
        return new_num_channels

    def __init__(
        self,
        num_classes,
        width_coefficient=1.0,
        depth_coefficient=1.0,
        se_rate=0.25,
        dropout_rate=0.2,
        drop_connect_rate=0.2,
    ):
        super(EfficientNet, self).__init__()

        self.width_coefficient = width_coefficient
        self.depth_coefficient = depth_coefficient
        self.divisor = 8

        list_channels = [32, 16, 24, 40, 80, 112, 192, 320, 1280]
        list_channels = [self._setup_channels(c) for c in list_channels]

        list_num_repeats = [1, 2, 2, 3, 3, 4, 1]
        list_num_repeats = [self._setup_repeats(r) for r in list_num_repeats]

        expand_rates = [1, 6, 6, 6, 6, 6, 6]
        strides = [1, 2, 2, 2, 1, 2, 1]
        kernel_sizes = [3, 3, 5, 3, 5, 5, 3]

        self.stem = nn.Sequential(
            nn.Conv2d(
                3, list_channels[0], kernel_size=3, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(list_channels[0], momentum=0.01, eps=1e-3),
            Swish(),
        )

        blocks = []
        counter = 0
        num_blocks = sum(list_num_repeats)
        for idx in range(7):

            num_channels = list_channels[idx]
            next_num_channels = list_channels[idx + 1]
            num_repeats = list_num_repeats[idx]
            expand_rate = expand_rates[idx]
            kernel_size = kernel_sizes[idx]
            stride = strides[idx]
            drop_rate = drop_connect_rate * counter / num_blocks

            name = "MBConv{}_{}".format(expand_rate, counter)
            blocks.append(
                (
                    name,
                    MBConv(
                        num_channels,
                        next_num_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        expand_rate=expand_rate,
                        se_rate=se_rate,
                        drop_connect_rate=drop_rate,
                    ),
                )
            )
            counter += 1
            for i in range(1, num_repeats):
                name = "MBConv{}_{}".format(expand_rate, counter)
                drop_rate = drop_connect_rate * counter / num_blocks
                blocks.append(
                    (
                        name,
                        MBConv(
                            next_num_channels,
                            next_num_channels,
                            kernel_size=kernel_size,
                            stride=1,
                            expand_rate=expand_rate,
                            se_rate=se_rate,
                            drop_connect_rate=drop_rate,
                        ),
                    )
                )
                counter += 1

        self.blocks = nn.Sequential(OrderedDict(blocks))

        self.head = nn.Sequential(
            nn.Conv2d(list_channels[-2], list_channels[-1], kernel_size=1, bias=False),
            nn.BatchNorm2d(list_channels[-1], momentum=0.01, eps=1e-3),
            Swish(),
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(list_channels[-1], num_classes),
        )

        self.apply(init_weights)

    def forward(self, x):
        f = self.stem(x)
        f = self.blocks(f)
        y = self.head(f)
        return y




## === cell 2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

best_model = EfficientNet(num_classes=5).to(device)


def _load_state_into_model(model, ckpt_path: str):
    state = torch.load(ckpt_path, map_location="cpu")
    if (
        isinstance(state, dict)
        and "state_dict" in state
        and isinstance(state["state_dict"], dict)
    ):
        state = state["state_dict"]
    if isinstance(state, dict):
        new_state = {}
        for k, v in state.items():
            nk = k[7:] if k.startswith("module.") else k
            new_state[nk] = v
        state = new_state
    model.load_state_dict(state, strict=True)
    return model


def quadratic_weighted_kappa(y_true, y_pred, n_classes=5):
    return cohen_kappa_score(
        y_true, y_pred, weights="quadratic", labels=list(range(n_classes))
    )


def _expected_class_from_logits(logits: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits, dim=-1)
    classes = torch.arange(0, probs.shape[-1], device=probs.device, dtype=probs.dtype)
    return (probs * classes).sum(dim=-1)


def apply_thresholds(preds_cont: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    preds_cont = np.asarray(preds_cont, dtype=np.float32)
    preds_cont = np.clip(preds_cont, 0.0, 4.0)
    thresholds = np.asarray(thresholds, dtype=np.float32)
    y = np.digitize(preds_cont, bins=thresholds).astype(np.int64)
    return np.clip(y, 0, 4).astype(np.int64)


def _sanitize_thresholds(th: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    th = np.asarray(th, dtype=np.float32)
    th = np.clip(th, 0.0, 4.0 - eps)
    th = np.sort(th)
    for i in range(1, len(th)):
        if th[i] <= th[i - 1] + eps:
            th[i] = th[i - 1] + eps
    th = np.clip(th, 0.0, 4.0 - eps)
    return th.astype(np.float32)


def _fit_thresholds_bruteforce(
    y_true: np.ndarray, preds_cont: np.ndarray
) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=np.int64)
    preds_cont = np.asarray(preds_cont, dtype=np.float32)

    th = np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float32)
    th = _sanitize_thresholds(th)

    def score(thr):
        y_pred = apply_thresholds(preds_cont, thr)
        return quadratic_weighted_kappa(y_true, y_pred, n_classes=5)

    best = score(th)
    for _ in range(12):
        improved = False
        for i in range(4):
            base = th[i]
            candidates = base + np.array(
                [-0.2, -0.1, -0.05, 0.05, 0.1, 0.2], dtype=np.float32
            )
            for c in candidates:
                thr = th.copy()
                thr[i] = c
                thr = _sanitize_thresholds(thr)
                s = score(thr)
                if s > best:
                    best = s
                    th = thr
                    improved = True
        if not improved:
            break
    return _sanitize_thresholds(th).astype(np.float32)


from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL.Image import BICUBIC
from PIL import Image
from torch.utils.data import DataLoader


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, root, path_list, targets=None, transform=None, extension=".png"):
        super().__init__()
        self.root = root
        self.path_list = list(path_list)
        self.targets = targets
        self.transform = transform
        self.extension = extension
        if targets is not None:
            assert len(self.path_list) == len(self.targets)
            self.targets = torch.LongTensor(np.asarray(targets, dtype=np.int64))

    def __getitem__(self, index):
        path = self.path_list[index]
        sample = Image.open(os.path.join(self.root, path + self.extension)).convert(
            "RGB"
        )
        if self.transform is not None:
            sample = self.transform(sample)
        if self.targets is not None:
            return sample, self.targets[index]
        else:
            return sample, torch.LongTensor([])

    def __len__(self):
        return len(self.path_list)


image_size = 224
train_transform = Compose(
    [
        Resize((image_size, image_size), BICUBIC),
        ToTensor(),
        Normalize(mean=[0.42, 0.22, 0.075], std=[0.27, 0.15, 0.081]),
    ]
)

df_train = pd.read_csv(TRAIN_CSV)
tr_ids = df_train["id_code"].values
tr_y = df_train["diagnosis"].values

tr_ids_a, va_ids, tr_y_a, va_y = train_test_split(
    tr_ids, tr_y, test_size=0.15, random_state=42, stratify=tr_y
)

valid_ds = ImageDataset(TRAIN_IMG_DIR, va_ids, va_y, transform=train_transform)
num_workers = min(4, (os.cpu_count() or 2))
valid_loader = DataLoader(
    valid_ds,
    batch_size=64,
    shuffle=False,
    num_workers=num_workers,
    drop_last=False,
    pin_memory=(device.type == "cuda"),
)

loaded_ok = False
if os.path.exists(model_path):
    try:
        best_model = _load_state_into_model(best_model, model_path).to(device)
        loaded_ok = True
        print("Loaded checkpoint:", model_path)
    except Exception as e:
        print("Checkpoint load failed; will train locally. Error:", repr(e))
        loaded_ok = False

if not loaded_ok:
    print("Training a model locally to create:", model_path)

    train_ds = ImageDataset(TRAIN_IMG_DIR, tr_ids_a, tr_y_a, transform=train_transform)

    batch_size = 32
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=(device.type == "cuda"),
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(best_model.parameters(), lr=1e-3)

    best_kappa = -1.0
    best_state = None

    epochs = 3
    for epoch in range(1, epochs + 1):
        best_model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = best_model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * xb.size(0)

        train_loss = running_loss / len(train_ds)

        best_model.eval()
        va_true = []
        va_pred = []
        with torch.no_grad():
            for xb, yb in valid_loader:
                xb = xb.to(device, non_blocking=True)
                logits = best_model(xb)
                preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
                va_pred.extend(preds.tolist())
                va_true.extend(yb.numpy().tolist())

        kappa = quadratic_weighted_kappa(va_true, va_pred, n_classes=5)
        print(
            f"epoch {epoch}/{epochs} - train_loss: {train_loss:.4f} - val_qwk(argmax): {kappa:.5f}"
        )

        if kappa > best_kappa:
            best_kappa = kappa
            best_state = {
                k: v.detach().cpu().clone() for k, v in best_model.state_dict().items()
            }

    if best_state is not None:
        best_model.load_state_dict(best_state, strict=True)
        torch.save(best_model.state_dict(), model_path)
        print(
            "Saved trained checkpoint to:",
            model_path,
            "best_val_qwk(argmax):",
            best_kappa,
        )
    else:
        torch.save(best_model.state_dict(), model_path)
        print("Saved checkpoint (no val improvement tracked) to:", model_path)

best_model = best_model.to(device)

best_model.eval()
va_true = []
va_cont = []
with torch.no_grad():
    for xb, yb in valid_loader:
        xb = xb.to(device, non_blocking=True)
        logits = best_model(xb)
        cont = _expected_class_from_logits(logits).detach().cpu().numpy()
        va_cont.extend(cont.tolist())
        va_true.extend(yb.numpy().tolist())

learned_thresholds = _fit_thresholds_bruteforce(np.array(va_true), np.array(va_cont))
learned_thresholds = _sanitize_thresholds(learned_thresholds)
va_pred_thr = apply_thresholds(np.array(va_cont), learned_thresholds)
kappa_thr = quadratic_weighted_kappa(np.array(va_true), va_pred_thr, n_classes=5)
print(
    "Learned thresholds:", learned_thresholds, "val_qwk(thresholded):", float(kappa_thr)
)

assert np.all(np.isfinite(learned_thresholds)), "Thresholds must be finite"
assert np.all(np.diff(learned_thresholds) > 0), "Thresholds must be strictly increasing"
assert learned_thresholds[-1] < 4.0, "Last threshold must be < 4.0 to avoid class 5"



## === cell 3
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import pandas as pd
from PIL.Image import BICUBIC

image_size = 224

test_transform = Compose(
    [
        Resize((image_size, image_size), BICUBIC),
        ToTensor(),
        Normalize(mean=[0.42, 0.22, 0.075], std=[0.27, 0.15, 0.081]),
    ]
)

df_test = pd.read_csv(TEST_CSV)
test_dataset = ImageDataset(
    root=TEST_IMG_DIR,
    path_list=df_test.id_code.values,
    transform=test_transform,
)



## === cell 4
from torch.utils.data import DataLoader

batch_size = 64
num_workers = min(4, (os.cpu_count() or 2))
print("num_workers:", num_workers)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    shuffle=False,
    drop_last=False,
    pin_memory=(device.type == "cuda"),
)



## === cell 5
from tqdm.auto import tqdm
import numpy as np

best_model = best_model.to(device)
all_cont = []
best_model.eval()

assert len(test_dataset) == len(df_test)
assert list(test_dataset.path_list[:10]) == list(df_test.id_code.values[:10])

with torch.no_grad():
    for x, _ in tqdm(test_loader, total=len(test_loader)):
        x = x.to(device, non_blocking=True)
        y_pred1 = best_model(x)
        y_pred2 = best_model(x.flip(dims=(-1,)))
        curr_pred = 0.5 * (F.softmax(y_pred1, dim=-1) + F.softmax(y_pred2, dim=-1))

        classes = torch.arange(0, 5, device=curr_pred.device, dtype=curr_pred.dtype)
        cont = (curr_pred * classes).sum(dim=-1).detach().cpu().numpy()
        all_cont.extend(cont.tolist())

all_cont = np.asarray(all_cont, dtype=np.float32)
all_pred = apply_thresholds(all_cont, learned_thresholds).astype(int).tolist()

print("num predictions:", len(all_pred))
print("thresholds used:", learned_thresholds)
print(
    "pred distribution:",
    dict(zip(*np.unique(np.asarray(all_pred), return_counts=True))),
)

assert set(np.unique(np.asarray(all_pred))).issubset(
    set(range(5))
), "Pred labels must be in {0,1,2,3,4}"



## === cell 6
import pandas as pd

sub = pd.read_csv(SAMPLE_SUB)
df_test = pd.read_csv(TEST_CSV)

assert df_test["id_code"].is_unique, "test.csv id_code must be unique"
assert sub["id_code"].is_unique, "sample_submission id_code must be unique"
assert len(all_pred) == len(
    df_test
), f"Predictions ({len(all_pred)}) must match test.csv rows ({len(df_test)})"

pred_df = pd.DataFrame(
    {"id_code": df_test["id_code"].values, "diagnosis": np.array(all_pred, dtype=int)}
)

sub = sub[["id_code"]].merge(pred_df, on="id_code", how="left")
if sub["diagnosis"].isna().any():
    missing = sub.loc[sub["diagnosis"].isna(), "id_code"].head(5).tolist()
    raise ValueError(f"Missing predictions for some ids (examples): {missing}")

sub["diagnosis"] = sub["diagnosis"].astype(int).clip(0, 4)
assert set(sub["diagnosis"].unique()).issubset(set(range(5)))

sub.to_csv("submission.csv", index=False)
print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)



## === cell 7
_ = sub.hist()



## === cell 8
tr = pd.read_csv(TRAIN_CSV)
_ = tr.hist()
