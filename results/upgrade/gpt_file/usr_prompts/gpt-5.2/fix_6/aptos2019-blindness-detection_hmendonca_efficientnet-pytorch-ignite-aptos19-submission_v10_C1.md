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

0.7583415713181103

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the missing model weights by making the weights path resolution robust (searching common Kaggle input locations and avoiding shell-glob pitfalls) and load the checkpoint correctly even if it was saved with a wrapper key like `state_dict`. I also make the code run on CPU-only environments by selecting `cuda` only when available, and ensure inference uses the correct test-time augmentation average (it currently averages `y_pred1` twice). Finally, I ensure the prediction list length matches `sample_submission.csv` by iterating deterministically and writing `submission.csv` with the required columns.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score strongly suggests the model is effectively untrained at inference time (random weights or mostly-mismatched checkpoint load), even though a submission is produced. I make the weight selection stricter (prefer an actual EfficientNet “best” checkpoint and avoid accidentally copying some unrelated `.pth`), and I load the checkpoint more robustly by handling nested prefixes like `model.`/`net.` in addition to `module.` while keeping the same model and inference logic. To avoid silently “half-loaded” weights, I also report the fraction of parameters loaded and only fall back to random weights if nothing sensible loads. These are minimal changes aimed at moving QWK upward toward your 0.758 target without changing architecture or inference semantics.'
- What this solution (achieved 0.0) has done: 'Your 0.0 QWK is consistent with either (a) weights not actually loading (despite a file being copied) or (b) a mismatched EfficientNet variant where only a small fraction of parameters overlap, leaving the model effectively random. I make the weight selection less error-prone by prioritizing checkpoints that look like full model weights (and de-prioritizing optimizer/epoch-only artifacts), and I add a “best-effort” fallback: if overlap is low, try additional candidate checkpoints until one loads with high key overlap. This keeps the same model architecture and inference logic, but greatly increases the chance you’re using the intended trained weights, which should move the score up toward your target. I also ensure deterministic id alignment by building the submission directly from `df_test` order (still with the required columns), avoiding any subtle merge/order issues.'
- What this solution (achieved 0.0) has done: 'Your 0.0 QWK is most consistent with “random/untrained” inference due to weights not actually being found/loaded in this environment, so I make the weight discovery deterministic and point it at the provided Kaggle dataset paths (under `/kaggle/input/...`) instead of only `../input/**`, while keeping your model and inference unchanged. I also strengthen the checkpoint selection to prefer large full-model checkpoints and explicitly stop after the best-overlap load succeeds, which should move the score upward toward your 0.758 target. Finally, I keep the submission alignment strictly in `test.csv` order and add a hard assertion so we never silently pad/trim predictions (which can otherwise ruin QWK without crashing).'
- What this solution (achieved 0.0) has done: 'Your 0.0 QWK is most consistent with the model running with random (or mostly-unloaded) weights, so the smallest meaningful improvement is to make checkpoint discovery include this environment’s actual dataset roots (not just `/kaggle/input`) and then select the checkpoint that *actually* loads best into your exact EfficientNet (highest key overlap and large file size), without changing the model or inference logic. I also fix a current runtime hazard where `shutil` may be undefined in the weight-loading cell (it’s only imported in your first cell), which can prevent copying the best checkpoint into place. Finally, I keep submission ordering strictly in `test.csv` order and keep your same TTA/argmax prediction semantics to preserve evaluation meaning while moving score upward toward the target band.'

# 9. Code solution

## === cell 0
import os, glob, shutil
from pathlib import Path

model_path = "efficientNet_best.pth"

search_roots = [
    "../input",  # classic notebook relative path
    "/kaggle/input",  # Kaggle standard absolute path
    "/kaggle/data/input",  # provided in this environment listing
    "/kaggle/data",  # additional root present in this environment listing
]

candidate_patterns = []
for r in search_roots:
    candidate_patterns.extend(
        [
            f"{r}/**/efficientnet*_best*.pth",
            f"{r}/**/efficientnet*best*.pth",
            f"{r}/**/efficientnet*.pth",
            f"{r}/**/efficientNet*_best*.pth",
            f"{r}/**/efficientNet*best*.pth",
            f"{r}/**/efficientNet*.pth",
            f"{r}/**/*.pth",
        ]
    )

candidates = []
for pat in candidate_patterns:
    candidates.extend(glob.glob(pat, recursive=True))


def _score_name(p):
    name = os.path.basename(p).lower()
    score = 0
    if "efficientnet" in name or "efficient" in name:
        score += 60
    if "best" in name:
        score += 30
    if "fold" in name:
        score += 3
    if "final" in name:
        score += 5
    if "optim" in name or "optimizer" in name:
        score -= 80
    if "sched" in name or "scheduler" in name:
        score -= 40
    if "ema" in name:
        score -= 5
    if "epoch" in name and "best" not in name:
        score -= 3
    if name.endswith(".pth"):
        score += 1
    try:
        sz = os.path.getsize(p)
    except OSError:
        sz = 0
    if sz > 80_000_000:
        score += 35
    elif sz > 50_000_000:
        score += 25
    elif sz > 10_000_000:
        score += 10
    elif sz > 1_000_000:
        score += 2
    else:
        score -= 10
    return score


def _sort_key(p):
    try:
        sz = os.path.getsize(p)
    except OSError:
        sz = 0
    return (-_score_name(p), -sz, p)


candidates = sorted(set(candidates), key=_sort_key)
print("Found .pth candidates (top 25):")
for p in candidates[:25]:
    try:
        sz = os.path.getsize(p)
    except OSError:
        sz = -1
    print(" -", p, "score=", _score_name(p), "size=", sz)

if len(candidates) == 0:
    print(
        "WARNING: No .pth weights found in search roots. The model will run with random weights (submission will be poor but valid)."
    )
else:
    src = candidates[0]
    shutil.copy(src, model_path)
    print(f"Copied weights: {src} -> {model_path}")

print(
    "model_path exists?",
    os.path.exists(model_path),
    "size:",
    os.path.getsize(model_path) if os.path.exists(model_path) else None,
)



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
        self.drop_connect_rate = torch.tensor(drop_connect_rate, requires_grad=False)

    def _drop_connect(self, x):
        keep_prob = 1.0 - self.drop_connect_rate
        drop_mask = torch.rand(x.shape[0], 1, 1, 1) + keep_prob
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
            if self.training and self.drop_connect_rate is not None:
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
import os
import shutil  # Change (score/stability): ensure available for potential best_path -> model_path copy.
import torch
from collections import OrderedDict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

best_model = EfficientNet(num_classes=5)


def _extract_state_dict(ckpt_obj):
    if isinstance(ckpt_obj, dict):
        for k in ["state_dict", "model_state_dict", "model", "net", "weights"]:
            if k in ckpt_obj and isinstance(ckpt_obj[k], dict):
                return ckpt_obj[k]
    return ckpt_obj


def _strip_known_prefixes(state):
    prefixes = ("module.", "model.", "net.", "encoder.", "backbone.")
    new_state = OrderedDict()
    for k, v in state.items():
        nk = k
        changed = True
        while changed:
            changed = False
            for pref in prefixes:
                if nk.startswith(pref):
                    nk = nk[len(pref) :]
                    changed = True
        new_state[nk] = v
    return new_state


def _try_load_checkpoint(path, model):
    try:
        ckpt = torch.load(path, map_location="cpu")
    except Exception as e:
        print(f"Could not load checkpoint {path}: {type(e).__name__}: {e}")
        return 0.0, False

    state = _extract_state_dict(ckpt)
    if not isinstance(state, dict):
        print(f"Checkpoint {path} is not state_dict-like; skipping.")
        return 0.0, False

    state = _strip_known_prefixes(state)

    model_keys = set(model.state_dict().keys())
    state_keys = set(state.keys())
    overlap = len(model_keys & state_keys)
    loaded_frac = overlap / max(1, len(model_keys))

    missing, unexpected = model.load_state_dict(state, strict=False)
    print(
        f"Try load: {path} | overlap={overlap}/{len(model_keys)} ({loaded_frac:.3f}) | missing={len(missing)} unexpected={len(unexpected)}"
    )
    return loaded_frac, True


loaded_ok = False
best_loaded_frac = 0.0
best_path = None

cand_list = []
if "candidates" in globals():
    cand_list = list(candidates)
if os.path.exists(model_path):
    cand_list = [model_path] + [
        p for p in cand_list if os.path.abspath(p) != os.path.abspath(model_path)
    ]
else:
    print("WARNING: model weights not found at model_path; will search candidates.")

seen = set()
for p in cand_list:
    ap = os.path.abspath(p)
    if ap in seen:
        continue
    seen.add(ap)
    frac, ok = _try_load_checkpoint(p, best_model)
    if ok and frac > best_loaded_frac:
        best_loaded_frac = frac
        best_path = p
    if ok and frac >= 0.80:
        loaded_ok = True
        best_path = p
        print(f"Selected checkpoint with good overlap: {p}")
        break

if (not loaded_ok) and (best_path is not None) and (best_path != model_path):
    try:
        shutil.copy(best_path, model_path)
    except Exception:
        pass
    print(
        f"WARNING: No checkpoint reached overlap>=0.80; using best-overlap={best_loaded_frac:.3f} from {best_path}"
    )
elif not loaded_ok:
    print(
        f"WARNING: No checkpoint loaded; model likely untrained/random => score likely poor."
    )

best_model = best_model.to(device)



## === cell 3
from torchvision.transforms import Compose, Resize
from torchvision.transforms import ToTensor, Normalize

import os
import pandas as pd
from PIL import Image
from PIL.Image import BICUBIC


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
            self.targets = torch.LongTensor(targets)

    def __getitem__(self, index):
        path = self.path_list[index]
        img_path = os.path.join(self.root, path + self.extension)
        sample = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            sample = self.transform(sample)

        if self.targets is not None:
            return sample, self.targets[index]
        else:
            return sample, torch.LongTensor([])

    def __len__(self):
        return len(self.path_list)


image_size = 224

test_transform = Compose(
    [
        Resize((image_size, image_size), BICUBIC),
        ToTensor(),
        Normalize(mean=[0.42, 0.22, 0.075], std=[0.27, 0.15, 0.081]),
    ]
)

test_csv_path = "../input/aptos2019-blindness-detection/test.csv"
test_img_root = "../input/aptos2019-blindness-detection/test_images"
if not os.path.exists(test_csv_path):
    for alt_root in ["/kaggle/input", "/kaggle/data/input", "/kaggle/data"]:
        alt = f"{alt_root}/aptos2019-blindness-detection/test.csv"
        alt_img = f"{alt_root}/aptos2019-blindness-detection/test_images"
        if os.path.exists(alt):
            test_csv_path = alt
            test_img_root = alt_img
            break

df_test = pd.read_csv(test_csv_path)
test_dataset = ImageDataset(
    root=test_img_root,
    path_list=df_test.id_code.values,
    transform=test_transform,
)

print("Test samples:", len(test_dataset))



## === cell 4
from torch.utils.data import DataLoader
import os

batch_size = 64
num_workers = min(4, os.cpu_count() or 0)
print("num_workers:", num_workers)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    shuffle=False,
    drop_last=False,
    pin_memory=torch.cuda.is_available(),
)



## === cell 5
from tqdm import tqdm
from torch.nn import functional as F
import torch

all_pred = []
best_model.eval()
with torch.no_grad():
    for x, _ in tqdm(test_loader, total=len(test_loader)):
        x = x.to(device, non_blocking=True)

        y_pred1 = best_model(x)
        y_pred2 = best_model(x.flip(dims=(-1,)))
        curr_pred = 0.5 * (F.softmax(y_pred1, dim=-1) + F.softmax(y_pred2, dim=-1))

        all_pred += torch.argmax(curr_pred, dim=1).cpu().numpy().astype(int).tolist()

print("Predictions:", len(all_pred))



## === cell 6
import numpy as np
import pandas as pd

assert len(all_pred) == len(
    df_test
), f"Prediction length mismatch: {len(all_pred)} vs {len(df_test)}"

sub = pd.DataFrame({"id_code": df_test["id_code"].values})
sub["diagnosis"] = np.array(all_pred, dtype=int).clip(0, 4)

sub.to_csv("submission.csv", index=False)
print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)



## === cell 7
import matplotlib

matplotlib.use("Agg")
_ = sub.hist()



## === cell 8
import matplotlib
import pandas as pd
import os

matplotlib.use("Agg")

train_csv_path = "../input/aptos2019-blindness-detection/train.csv"
if not os.path.exists(train_csv_path):
    for alt_root in ["/kaggle/input", "/kaggle/data/input", "/kaggle/data"]:
        alt = f"{alt_root}/aptos2019-blindness-detection/train.csv"
        if os.path.exists(alt):
            train_csv_path = alt
            break

tr = pd.read_csv(train_csv_path)
_ = tr.hist()
print("Train label distribution:\n", tr["diagnosis"].value_counts().sort_index())
