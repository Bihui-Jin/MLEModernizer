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

0.9244430251754334

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the missing-weights crash by loading the model weights only if they exist and otherwise falling back to a no-training baseline that still produces a valid submission. I also fix the CUDA-only device selection so the notebook runs on CPU-only Kaggle sessions without crashing. Finally, I correct a couple of small logic bugs in the transforms/utilities (`is` vs `==`, and a `.cpu()` that breaks tensors) and ensure the submission is always written with the required columns and non-empty rows.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is coming from the “no weights found → all-zero predictions” fallback, so the smallest score-moving change is to make the script actually load weights that exist in this Kaggle dataset (there is no `../input/weights/...`). I keep your exact model and inference logic, but change `WEIGHTS_PATH` to a valid file path inside the competition input directory if present, and add a safe auto-discovery of a `.pkl/.pth/.pt` weights file under `../input` so you don’t silently fall back to zeros. I also harden the state_dict loading to handle common checkpoint formats (`{"state_dict": ...}`) without changing model behavior. This should move the score upward toward your target while keeping everything else the same.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score comes from the “no weights found → all-zero predictions” fallback, so the smallest score-improving change is to make weight loading actually succeed reliably. I keep your exact model/inference logic, but harden checkpoint loading to handle common `DataParallel` prefixes (`module.`) and key mismatches without silently failing, and I ensure we always pick the intended weights file if it exists under `../input`. I also remove the unnecessary `.data` usages (no semantic change under `no_grad`) and make `regress2class` stay on CPU deterministically to avoid device edge-cases. This should move the score up toward your target by enabling real predictions rather than zeros, without changing the model architecture or training approach.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is consistent with the current “no weights found → all-zero predictions” path still being taken, so the smallest score-moving change is to make checkpoint discovery/loading reliable and verifiably succeed (otherwise nothing else matters). I keep your exact model and inference logic, but (1) constrain the weight auto-discovery to filenames that match your expected model (avoid accidentally loading unrelated `.pth/.pt` files), (2) robustly load common checkpoint formats (plain `state_dict`, wrapped dict, `DataParallel` prefixes), and (3) add an explicit sanity check that flips `has_weights=True` only after a successful `load_state_dict`. This should move the score upward toward your target by enabling real predictions instead of zeros, without changing the architecture, transforms, or post-processing.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is still explained by `has_weights=False` (all-zero fallback), so the smallest score-moving change is to make weight loading succeed deterministically or fail loudly before writing a useless submission. I keep your exact model, transforms, and inference (same `ThreeStage_Model` forward path and `regress2class` thresholds), but improve weight discovery to search inside the competition dataset directory and to prefer an exact filename match before “near” matches. I also harden checkpoint parsing for common formats (e.g., `{'model_state_dict': ...}`) and set `has_weights=True` only after a successful `load_state_dict` call, while printing a clear summary of what was loaded. If no weights exist anywhere under `../input`, the code still produce a valid CSV, but now you immediately see that the run cannot possibly score above 0.0 without adding weights.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is because the code is still falling back to all-zero predictions when it can’t find/load weights; without real weights, this model can’t reach the 0.924 target. The minimal score-moving change is therefore to (1) search the full `../input` tree for likely checkpoints (not just B4-3stage-named ones), (2) validate compatibility by checking key overlap before loading, and (3) only set `has_weights=True` after a successful load with meaningful overlap—otherwise fail loudly so you don’t submit another guaranteed-0.0 file by accident. This keeps your exact model architecture and inference logic (same forward path and `regress2class` thresholds), but makes weight loading actually succeed when a compatible checkpoint exists in the environment.'

# 9. Code solution

## === cell 0
import os
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

from sklearn.metrics import cohen_kappa_score
import timm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)



## === cell 1
threshold = [0.75, 1.5, 2.5, 3.5]


def regress2class(out):
    out_cpu = out.detach().view(-1).to("cpu")
    prediction = torch.zeros(out_cpu.size(0), device="cpu")
    for i in range(4):
        prediction += (out_cpu >= threshold[i]).float()
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
        super().__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps
        self.flatten = flatten

    def forward(self, x):
        x = gem(x, p=self.p, eps=self.eps)
        if self.flatten:
            x = x.flatten(1)
        return x

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p.data.tolist()[0]:.4f}, eps={self.eps})"


class Regressor(nn.Module):
    def __init__(self):
        super().__init__()
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
        super().__init__()
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
DATA_DIR = "../input/aptos2019-blindness-detection"
TEST_CSV = os.path.join(DATA_DIR, "test.csv")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")

WEIGHTS_PATH = os.path.join(DATA_DIR, "B4_3stage_13epoch_finetune3.pkl")


def _extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for key in ("state_dict", "model_state_dict", "model", "net", "weights"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
        return ckpt
    return ckpt


def _strip_module_prefix(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict
    if not any(k.startswith("module.") for k in state_dict.keys()):
        return state_dict
    return {k.replace("module.", "", 1): v for k, v in state_dict.items()}


def _state_dict_overlap_ratio(model, state_dict):
    """Why: avoid 'successful' strict=False loads that actually load nothing (leading to junk/zeros)."""
    if not isinstance(state_dict, dict):
        return 0.0, 0, 0
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(state_dict.keys())
    inter = len(model_keys & ckpt_keys)
    denom = max(1, len(model_keys))
    return inter / denom, inter, len(model_keys)


def _find_compatible_weights_file(model, search_roots, max_candidates=400):
    """
    Why: current 0.0 comes from not finding/loading any compatible checkpoint.
    This searches broadly under ../input for .pth/.pt/.pkl, ranks by key overlap, and picks the best match.
    """
    exts = (".pkl", ".pth", ".pt")
    candidates = []
    for search_root in search_roots:
        if not os.path.isdir(search_root):
            continue
        for root, _, files in os.walk(search_root):
            for fn in files:
                fn_l = fn.lower()
                if not fn_l.endswith(exts):
                    continue
                full = os.path.join(root, fn)
                candidates.append(full)
                if len(candidates) >= max_candidates:
                    break
            if len(candidates) >= max_candidates:
                break

    if not candidates:
        return None, None

    scored = []
    for path in candidates:
        try:
            ckpt = torch.load(path, map_location="cpu")
            state = _strip_module_prefix(_extract_state_dict(ckpt))
            ratio, inter, total = _state_dict_overlap_ratio(model, state)
            if ratio < 0.20:
                continue
            scored.append((ratio, inter, total, path))
        except Exception:
            continue

    if not scored:
        return None, None

    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    best = scored[0]
    return best[3], best


test_df = pd.read_csv(TEST_CSV)
test_ids = test_df["id_code"].astype(str).values

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

net = ThreeStage_Model().to(device)
net.eval()

has_weights = False

candidate_path = WEIGHTS_PATH if os.path.exists(WEIGHTS_PATH) else None
candidate_info = None

if candidate_path is None:
    discovered_path, discovered_info = _find_compatible_weights_file(
        net, [DATA_DIR, "../input"]
    )
    if discovered_path is not None:
        candidate_path = discovered_path
        candidate_info = discovered_info
        print("Discovered compatible weights file:", candidate_path)
        if candidate_info is not None:
            ratio, inter, total, _ = candidate_info
            print(f"  Key overlap: {inter}/{total} ({ratio:.1%})")

if candidate_path is not None and os.path.exists(candidate_path):
    try:
        ckpt = torch.load(candidate_path, map_location="cpu")
        state = _strip_module_prefix(_extract_state_dict(ckpt))

        ratio, inter, total = _state_dict_overlap_ratio(net, state)
        if ratio < 0.20:
            raise RuntimeError(
                f"Checkpoint appears incompatible (key overlap {inter}/{total} = {ratio:.1%})."
            )

        load_res = net.load_state_dict(state, strict=False)
        net = net.to(device)
        net.eval()
        has_weights = True

        missing, unexpected = load_res.missing_keys, load_res.unexpected_keys
        print("Loaded weights:", candidate_path)
        print(f"  Loaded with key overlap: {inter}/{total} ({ratio:.1%})")
        if len(missing) or len(unexpected):
            print("WARNING: checkpoint key mismatch (strict=False).")
            print("  Missing keys:", len(missing))
            print("  Unexpected keys:", len(unexpected))
    except Exception as e:
        print("WARNING: failed to load weights from:", candidate_path)
        print("  Error:", repr(e))
        has_weights = False

if not has_weights:
    raise RuntimeError(
        "No compatible weights were found/loaded under ../input, so predictions would be all zeros (0.0 score). "
        "Add the intended checkpoint to the notebook/dataset, or ensure the expected weights file exists."
    )



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/1715470731.py in <cell line: 0>()
    145 if not has_weights:
    146     # Why: a 0.0 submission is guaranteed useless; fail loudly so you don't unknowingly submit zeros again.
--> 147     raise RuntimeError(
    148         "No compatible weights were found/loaded under ../input, so predictions would be all zeros (0.0 score). "
    149         "Add the intended checkpoint to the notebook/dataset, or ensure the expected weights file exists."

RuntimeError: No compatible weights were found/loaded under ../input, so predictions would be all zeros (0.0 score). Add the intended checkpoint to the notebook/dataset, or ensure the expected weights file exists.

## === cell 5
submission = []
with torch.no_grad():
    for i, idx in enumerate(test_ids):
        if i % 50 == 0:
            print("Predicting", i, "/", len(test_ids))

        image_name = os.path.join(TEST_IMG_DIR, f"{idx}.png")
        img = Image.open(image_name).convert("RGB")
        img = transform(img).unsqueeze(0).to(device)

        _, r_out, _ = net(img)
        pred = regress2class(r_out.squeeze(1))
        pred_int = int(pred.item())
        submission.append([idx, pred_int])

submission = np.array(submission, dtype=object)
print("Submission rows:", submission.shape[0])



## === cell 6
df = pd.DataFrame(submission, columns=["id_code", "diagnosis"])
df["id_code"] = df["id_code"].astype(str)
df["diagnosis"] = df["diagnosis"].astype(int)
df.to_csv("submission.csv", index=False)
print(df.head())
print("Wrote submission.csv with shape:", df.shape)
