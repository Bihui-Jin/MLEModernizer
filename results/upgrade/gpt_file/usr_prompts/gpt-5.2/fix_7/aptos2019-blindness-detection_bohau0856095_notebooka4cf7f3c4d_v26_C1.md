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

0.8971545795095285

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I make the script robust to the Kaggle environment by removing the failing offline `pip install`, auto-selecting CPU when no GPU is available, and fixing the custom `trim()` transform to always return an image (it currently can return `None`, which can break transforms). I also fix the missing weights issue by searching for the checkpoint under common Kaggle input locations and loading it with `map_location=device` if found; if not found, the code still run end-to-end using randomly initialized weights and produce a valid `submission.csv` (score be low, but it no longer crash). Finally, I vectorize inference with a `Dataset/DataLoader` to keep runtime under control and ensure `submission.csv` is always non-empty and correctly formatted.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with a label/row-order mismatch between `submission.csv` and Kaggle’s expected `test.csv` order (QWK can collapse when predictions are paired with the wrong ground-truth rows). I keep your model and thresholds unchanged, but enforce exact alignment by merging predictions back onto `test.csv` and writing in that exact order. I also make `regress2class()` device-safe and deterministic by computing thresholds on the same device and avoiding `.data`/CPU roundtrips during batching, which can subtly scramble types and ordering. These are minimal changes aimed specifically at turning a likely “invalidly aligned” 0.0 into a meaningful score, moving toward your 0.897 target.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with the model running with random weights because the checkpoint isn’t actually being found/loaded, which yields near-random predictions and kappa near 0. I keep your model/inference/thresholding intact, but make checkpoint discovery robust to common Kaggle dataset layouts and to different checkpoint formats (raw state_dict vs wrapped dict). I also add a strict check that we really loaded non-default weights (by comparing one tensor before/after) so you don’t silently submit random predictions again. These changes are minimal, do not alter architecture or inference semantics, and primarily aim to move the score upward toward your 0.897 target.'
- What this solution (achieved 0.0) has done: 'I make two minimal fixes so the notebook runs end-to-end and writes a valid `submission.csv`: (1) remove the hard fail when a checkpoint can’t be found, and instead proceed with a warning (so you at least get a score rather than “Not yielded”); and (2) fix the device mismatch by moving the model to `device` before loading weights (or re-moving after load) so CUDA inputs match CUDA weights. I also robustly load checkpoints saved as full dicts / state_dicts and strip common prefixes (`module.`, `_orig_mod.`) without changing your model/inference logic. Finally, submission rows be forced to match `test.csv` order via merge (as you intended), preventing accidental id/order mismatches.'

# 9. Code solution

## === cell 0
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image, ImageChops

from sklearn.metrics import cohen_kappa_score  # kept for parity (not used)
import timm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)



## === cell 1
threshold = [0.7, 1.5, 2.5, 3.5]


def regress2class(out: torch.Tensor) -> torch.Tensor:
    """
    Keep exact thresholding logic; run on-device for determinism and avoid CPU roundtrips.
    """
    thr = torch.tensor(threshold, device=out.device, dtype=out.dtype).view(1, -1)
    pred = (out.view(-1, 1) >= thr).sum(dim=1).to(torch.int64)
    return pred.cpu()




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
    Path("../input/aptos2019-blindness-detection"),
    Path("/kaggle/input/aptos2019-blindness-detection"),
    Path("/kaggle/data/aptos2019-blindness-detection"),
    Path("../kaggle/data/aptos2019-blindness-detection"),
]


def resolve_base_dir():
    for p in BASE_CANDIDATES:
        if (p / "test.csv").exists():
            return p
    for root in [
        Path("/kaggle/input"),
        Path("../input"),
        Path("/kaggle/data"),
        Path("../kaggle/data"),
    ]:
        if root.exists():
            hits = list(root.glob("**/test.csv"))
            for h in hits:
                if h.name == "test.csv" and (h.parent / "test_images").exists():
                    return h.parent
    return BASE_CANDIDATES[0]


BASE_DIR = resolve_base_dir()
TEST_CSV = BASE_DIR / "test.csv"
TEST_IMG_DIR = BASE_DIR / "test_images"

test_df = pd.read_csv(TEST_CSV)
test_ids = test_df["id_code"].astype(str).values

input_size = 300
transform = transforms.Compose(
    [
        trim(),
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.384, 0.258, 0.174], std=[0.124, 0.089, 0.094]),
    ]
)

net = ThreeStage_Model()


def _unwrap_state_dict(state_obj):
    """
    Many checkpoints are saved as {'state_dict': ...} or {'model': ...}.
    Unwrap to get the actual state_dict when possible.
    """
    if isinstance(state_obj, dict):
        for k in ["state_dict", "model_state_dict", "model", "net", "weights"]:
            if k in state_obj and isinstance(state_obj[k], dict):
                return state_obj[k]
    return state_obj


def _strip_state_dict_prefixes(state_dict):
    """
    Fix rationale (bugfix): checkpoints may be saved from DataParallel ('module.')
    or torch.compile ('_orig_mod.'). Stripping lets weights load without changing model.
    """
    if not isinstance(state_dict, dict):
        return state_dict
    new_sd = {}
    for k, v in state_dict.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module.") :]
        if nk.startswith("_orig_mod."):
            nk = nk[len("_orig_mod.") :]
        new_sd[nk] = v
    return new_sd


def find_checkpoint():
    """
    Fix rationale (score-impacting): 0.0 is consistent with random weights due to missing checkpoint.
    Search more exhaustively inside the provided Kaggle dataset folders, while keeping load semantics identical.
    """
    roots = [
        Path("."),  # current working dir
        Path("/kaggle/working"),
        Path("../input"),
        Path("/kaggle/input"),
        Path("/kaggle/data"),
        Path("../kaggle/data"),
        BASE_DIR,  # include resolved competition folder explicitly
    ]
    for r in [Path("/kaggle/input"), Path("../input")]:
        if r.exists():
            for child in r.iterdir():
                if child.is_dir():
                    roots.append(child)

    preferred_names = [
        "efficient_b4_ns_3stage.pkl",
        "efficient_b4_ns_3stage.pth",
        "efficient_b4_ns_3stage.pt",
        "three_stage.pth",
        "three_stage.pkl",
        "model.pth",
        "weights.pth",
        "best.pth",
        "best_model.pth",
        "checkpoint.pth",
    ]
    for root in roots:
        if not root.exists():
            continue
        for name in preferred_names:
            hits = list(root.glob(f"**/{name}"))
            if hits:
                return hits[0]

    patterns = [
        "**/*b4*3stage*.pth",
        "**/*b4*3stage*.pkl",
        "**/*b4*3stage*.pt",
        "**/*3stage*.pth",
        "**/*3stage*.pkl",
        "**/*three*stage*.pth",
        "**/*three*stage*.pkl",
        "**/*efficient*3stage*.pth",
        "**/*efficient*3stage*.pkl",
        "**/*efficientnet*b4*.pth",
    ]
    for root in roots:
        if not root.exists():
            continue
        for pat in patterns:
            hits = list(root.glob(pat))
            if hits:
                return hits[0]
    return None


def _get_probe_tensor(model: nn.Module):
    for n, p in model.named_parameters():
        if p is not None and p.numel() > 0:
            return n, p.detach().float().view(-1)[:64].cpu().clone()
    return None, None


probe_name, probe_before = _get_probe_tensor(net)

ckpt_path = find_checkpoint()
loaded_ok = False
if ckpt_path is not None and ckpt_path.exists():
    state = torch.load(ckpt_path, map_location="cpu")
    state = _unwrap_state_dict(state)
    state = _strip_state_dict_prefixes(state)
    try:
        missing, unexpected = net.load_state_dict(state, strict=False)
        loaded_ok = True
        print(f"Loaded checkpoint: {ckpt_path}")
        print(f"Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")
    except Exception as e:
        loaded_ok = False
        raise RuntimeError(f"Failed to load checkpoint at {ckpt_path}: {e}") from e
else:
    loaded_ok = False

probe_name2, probe_after = _get_probe_tensor(net)
if (
    (not loaded_ok)
    or (probe_before is None)
    or (probe_after is None)
    or torch.allclose(probe_before, probe_after)
):
    raise RuntimeError(
        "Checkpoint was not reliably loaded (or weights unchanged). "
        "Stopping to avoid producing a random/0.0-score submission. "
        f"Found ckpt: {ckpt_path}"
    )

net = net.to(device)
net.eval()




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/3825917828.py in <cell line: 0>()
    173     or torch.allclose(probe_before, probe_after)
    174 ):
--> 175     raise RuntimeError(
    176         "Checkpoint was not reliably loaded (or weights unchanged). "
    177         "Stopping to avoid producing a random/0.0-score submission. "

RuntimeError: Checkpoint was not reliably loaded (or weights unchanged). Stopping to avoid producing a random/0.0-score submission. Found ckpt: None

## === cell 5
class TestDataset(Dataset):
    def __init__(self, ids, img_dir: Path, tfm):
        self.ids = list(ids)
        self.img_dir = img_dir
        self.tfm = tfm

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        idx = self.ids[i]
        image_name = self.img_dir / f"{idx}.png"
        img = Image.open(image_name).convert("RGB")
        img = self.tfm(img)
        return idx, img


ds = TestDataset(test_ids, TEST_IMG_DIR, transform)
dl = DataLoader(
    ds, batch_size=8, shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available()
)

pred_rows = []
with torch.no_grad():
    for batch_ids, batch_imgs in dl:
        batch_imgs = batch_imgs.to(device, non_blocking=True)
        r_out = net(batch_imgs, final=True).squeeze(1)  # (B,)
        pred = regress2class(r_out).numpy()
        for _id, _p in zip(batch_ids, pred):
            pred_rows.append((_id, int(_p)))

pred_df = pd.DataFrame(pred_rows, columns=["id_code", "diagnosis"])
pred_df["id_code"] = pred_df["id_code"].astype(str)
pred_df["diagnosis"] = pred_df["diagnosis"].astype(int)



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/1035969096.py in <cell line: 0>()
     25     for batch_ids, batch_imgs in dl:
     26         batch_imgs = batch_imgs.to(device, non_blocking=True)
---> 27         r_out = net(batch_imgs, final=True).squeeze(1)  # (B,)
     28         pred = regress2class(r_out).numpy()
     29         for _id, _p in zip(batch_ids, pred):

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

## === cell 6
out_df = test_df.copy()
out_df["id_code"] = out_df["id_code"].astype(str)

out_df = out_df.merge(pred_df, on="id_code", how="left", validate="one_to_one")

if out_df["diagnosis"].isna().any():
    missing = out_df.loc[out_df["diagnosis"].isna(), "id_code"].head(10).tolist()
    raise RuntimeError(f"Missing predictions for some test ids, e.g.: {missing}")

out_df["diagnosis"] = out_df["diagnosis"].astype(int)

if len(out_df) != len(test_ids):
    raise RuntimeError(
        f"Submission length mismatch: got {len(out_df)} rows, expected {len(test_ids)}."
    )

out_df.to_csv("submission.csv", index=False)
print(out_df.head())
print("Wrote submission.csv with", len(out_df), "rows.")
print("Checkpoint loaded:", loaded_ok)

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/263872040.py in <cell line: 0>()
      2 out_df["id_code"] = out_df["id_code"].astype(str)
      3 
----> 4 out_df = out_df.merge(pred_df, on="id_code", how="left", validate="one_to_one")
      5 
      6 if out_df["diagnosis"].isna().any():

NameError: name 'pred_df' is not defined
