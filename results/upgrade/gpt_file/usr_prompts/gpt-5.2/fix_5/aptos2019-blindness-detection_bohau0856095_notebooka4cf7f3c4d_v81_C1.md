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

0.9232816474245888

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I (1) remove the failing `pip install` and make `timm` use the already-installed version, (2) make the code run on CPU when no GPU is available, and (3) fix the transform bug where `trim()` can return `None`, which was breaking inference and leading to an empty submission. I also fix a small logic bug in `ordinal2class_prob` (CPU/GPU mix) and make checkpoint loading robust by automatically picking an available weights file if the referenced one is missing; if no weights exist, it still run end-to-end with random weights and produce a valid `submission.csv`. Finally, I ensure test IDs are read correctly and inference iterates without printing every row (to stay within runtime limits) while preserving the core model and prediction logic.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is most consistent with a format/alignment or label-range issue rather than model quality, so the smallest score-improving change is to ensure predictions are in the valid class set {0,1,2,3,4} and that the submission order matches `test.csv` exactly. I keep your model and transforms identical, but make `regress2class` run purely on the same device and explicitly clamp outputs before thresholding to avoid any out-of-range/NaN edge cases. I also force the final `df` to be reindexed to `test.csv` order (even if anything ever gets appended out of order) and assert the class range before writing the CSV. These are minimal, semantics-preserving changes that prevent “silent invalid submission” patterns and should move the score upward toward the target if the weights are being loaded.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with the checkpoint not actually loading (so you’re submitting near-random predictions), rather than a modeling issue, since your submission format/alignment code is already robust. I make checkpoint discovery/load stricter and more compatible with common training wrappers by (1) searching recursively under `WEIGHTS_DIR`, (2) auto-stripping `module.` / `model.` / `net.` prefixes, and (3) selecting the candidate checkpoint that yields the highest key-overlap with your model (so we’re far more likely to load the intended weights). These are minimal changes that preserve your model and inference logic while making it much more likely you’re using real trained weights, which should move your score up toward the target. I also add a hard failure if `WEIGHTS_DIR` exists but no weights are found, to avoid silently producing a bad submission.'

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
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import functional as FT
from PIL import Image, ImageChops

from sklearn.metrics import cohen_kappa_score
import timm

device = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

DATA_DIR = "../input/aptos2019-blindness-detection"
WEIGHTS_DIR = "../input/weights"



## === cell 1
threshold = [0.75, 1.5, 2.5, 3.5]


def regress2class(out):
    out = out.detach()
    out = torch.nan_to_num(out, nan=0.0, posinf=4.5, neginf=0.0).clamp(0.0, 4.5)
    prediction = torch.zeros(out.size(0), device=out.device, dtype=torch.long)
    for i in range(4):
        prediction += (out >= threshold[i]).to(torch.long).squeeze()
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
            l1 = int(math.floor(out[i].item()))
            l2 = int(math.ceil(out[i].item()))
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

        self.backbone = timm.models.tf_efficientnet_b4_ns(pretrained=False)
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
def _extract_state_dict(obj):
    if isinstance(obj, nn.Module):
        return obj.state_dict()

    if isinstance(obj, dict):
        for k in ["state_dict", "model_state_dict", "model", "net", "weights"]:
            if k in obj:
                v = obj[k]
                if isinstance(v, nn.Module):
                    return v.state_dict()
                if isinstance(v, dict):
                    return v
        if len(obj) > 0 and all(isinstance(k, str) for k in obj.keys()):
            return obj
    return obj


def _strip_prefix_if_present(state_dict, prefix):
    if not isinstance(state_dict, dict):
        return state_dict
    if not any(k.startswith(prefix) for k in state_dict.keys()):
        return state_dict
    return {k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)}


def _normalize_state_keys(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict
    for pref in ["module.", "model.", "net."]:
        state_dict = _strip_prefix_if_present(state_dict, pref)
    return state_dict


def _score_state_dict_match(model, state_dict):
    if not isinstance(state_dict, dict):
        return -1
    model_keys = set(model.state_dict().keys())
    sd_keys = set(state_dict.keys())
    return len(model_keys & sd_keys)


def _collect_checkpoint_candidates(search_dirs):
    patterns = ["*.pkl", "*.pth", "*.pt", "*.bin", "*.ckpt"]
    candidates = []
    for d in search_dirs:
        if not d or not os.path.exists(d):
            continue
        for pat in patterns:
            candidates.extend(glob.glob(os.path.join(d, "**", pat), recursive=True))
    seen = set()
    out = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _find_best_checkpoint(
    model, weights_dir, preferred_path, extra_search_dirs=None, min_overlap_ratio=0.70
):
    candidates = []
    if preferred_path and os.path.exists(preferred_path):
        candidates.append(preferred_path)

    search_dirs = [weights_dir]
    if extra_search_dirs:
        search_dirs.extend(extra_search_dirs)

    candidates.extend(_collect_checkpoint_candidates(search_dirs))

    seen = set()
    candidates = [c for c in candidates if not (c in seen or seen.add(c))]

    best = None
    best_score = -1
    best_info = None

    model_nkeys = len(model.state_dict().keys())
    min_overlap = int(model_nkeys * float(min_overlap_ratio))

    for path in candidates:
        try:
            raw = torch.load(path, map_location="cpu")
            sd = _normalize_state_keys(_extract_state_dict(raw))
            s = _score_state_dict_match(model, sd)
            if s >= min_overlap and s > best_score:
                best_score = s
                best = path
                best_info = (s, len(sd) if isinstance(sd, dict) else -1)
        except Exception:
            continue

    return best, best_score, best_info




## === cell 5
test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
test_ids = test_df["id_code"].astype(str).tolist()

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

net = ThreeStage_Model()

preferred_ckpt = os.path.join(WEIGHTS_DIR, "B4_3stage_9epoch_finetune2.pkl")

ckpt_path, overlap, info = _find_best_checkpoint(
    net,
    WEIGHTS_DIR,
    preferred_ckpt,
    extra_search_dirs=[DATA_DIR, os.path.join("..", "input")],
    min_overlap_ratio=0.70,
)

if ckpt_path is None:
    if os.path.exists(WEIGHTS_DIR) or os.path.exists(DATA_DIR):
        raise RuntimeError(
            f"No suitable checkpoint found (min key-overlap threshold not met). "
            f"Searched WEIGHTS_DIR={WEIGHTS_DIR} and DATA_DIR={DATA_DIR}. "
            "This would likely produce near-random predictions and a very low score."
        )
else:
    state = torch.load(ckpt_path, map_location="cpu")
    state = _normalize_state_keys(_extract_state_dict(state))
    missing, unexpected = net.load_state_dict(state, strict=False)
    print(f"Loaded checkpoint: {ckpt_path}")
    print(
        f"State_dict key overlap with model: {overlap} / {len(net.state_dict().keys())}"
    )
    if missing:
        print(f"Missing keys (first 20): {missing[:20]}")
    if unexpected:
        print(f"Unexpected keys (first 20): {unexpected[:20]}")

net = net.to(device)
net.eval()



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/417580752.py in <cell line: 0>()
     29 if ckpt_path is None:
     30     if os.path.exists(WEIGHTS_DIR) or os.path.exists(DATA_DIR):
---> 31         raise RuntimeError(
     32             f"No suitable checkpoint found (min key-overlap threshold not met). "
     33             f"Searched WEIGHTS_DIR={WEIGHTS_DIR} and DATA_DIR={DATA_DIR}. "

RuntimeError: No suitable checkpoint found (min key-overlap threshold not met). Searched WEIGHTS_DIR=../input/weights and DATA_DIR=../input/aptos2019-blindness-detection. This would likely produce near-random predictions and a very low score.

## === cell 6
submission = []
missing_images = 0

with torch.no_grad():
    for i, idx in enumerate(test_ids):
        image_name = os.path.join(DATA_DIR, "test_images", f"{idx}.png")
        if not os.path.exists(image_name):
            missing_images += 1
            submission.append([idx, 0])
            continue

        img = Image.open(image_name).convert("RGB")
        img = transform(img).unsqueeze(0).to(device)

        _, r_out, _ = net(img)
        pred = regress2class(r_out.squeeze(1)).clamp(0, 4)
        submission.append([idx, int(pred.item())])

submission = np.array(submission, dtype=object)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/1459782614.py in <cell line: 0>()
     13         img = transform(img).unsqueeze(0).to(device)
     14 
---> 15         _, r_out, _ = net(img)
     16         pred = regress2class(r_out.squeeze(1)).clamp(0, 4)
     17         submission.append([idx, int(pred.item())])

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

/tmp/ipykernel_55/1773810286.py in forward(self, x, final)
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
df = pd.DataFrame(submission, columns=["id_code", "diagnosis"])

df["id_code"] = df["id_code"].astype(str)
df["diagnosis"] = df["diagnosis"].astype(int)

if len(df) == 0:
    raise RuntimeError(
        "Submission DataFrame is empty; inference did not produce any rows."
    )
if len(df) != len(test_ids):
    raise RuntimeError(
        f"Submission row count {len(df)} != test row count {len(test_ids)}"
    )

df = test_df[["id_code"]].merge(df, on="id_code", how="left", validate="one_to_one")
if df["diagnosis"].isna().any():
    raise RuntimeError(
        "Some test ids did not receive predictions; submission would be invalid/misaligned."
    )
df["diagnosis"] = df["diagnosis"].astype(int).clip(0, 4)

u = sorted(df["diagnosis"].unique().tolist())
if (df["diagnosis"].min() < 0) or (df["diagnosis"].max() > 4):
    raise RuntimeError("Predicted diagnosis out of valid range 0..4.")
if len(u) == 0:
    raise RuntimeError("No predictions found.")

df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", df.shape)
if missing_images:
    print("Warning: missing images encountered:", missing_images)
print("Diagnosis unique values:", u)

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/1210102141.py in <cell line: 0>()
      5 
      6 if len(df) == 0:
----> 7     raise RuntimeError(
      8         "Submission DataFrame is empty; inference did not produce any rows."
      9     )

RuntimeError: Submission DataFrame is empty; inference did not produce any rows.
