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

0.9155783313307148

# 6. Current score

-0.48329

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.03194) has done: 'The pipeline fails because it hard-requires an external weight file that is not present in your Kaggle environment, which prevents the dataloader and submission from ever being created. I make the smallest change to allow the notebook to run end-to-end by (1) using `pretrained=True` for the EfficientNet backbone only when the custom `.pkl` weights are missing, and (2) keeping the same inference path and regression→class thresholding so evaluation semantics stay consistent. I also harden the weight-loading logic to handle common checkpoint key prefixes (`module.`, `model.`) without changing the model. Finally, I ensure the submission is always written as `submission.csv` with the required columns and row count.'
- What this solution (achieved -0.48329) has done: 'The timeout is overwhelmingly dominated by heavy per-image PIL preprocessing (trim + crop + resize) executed in Python for every sample, plus slow DataLoader settings, plus extra overhead from searching for weights recursively. To keep the exact model logic and transforms while cutting wall time, the main speedup is to cache transformed tensors to disk on first use (so repeated runs don’t redo expensive PIL work) and to speed up DataLoader throughput (more workers, persistent workers, prefetching). We also make `regress2class` vectorized and keep predictions on-device until the end of the batch to reduce CPU/GPU sync overhead, without changing thresholds or semantics. Finally, we avoid an expensive recursive glob over all of `/kaggle/input` by checking a small set of likely locations first (same behavior if the file exists elsewhere, but much faster in Kaggle).'

# 9. Code solution

## === cell 0
import os
import glob
import random
import time
import hashlib
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader, Dataset

import torchvision.transforms as transforms
from PIL import Image, ImageChops

import timm

device = "cuda" if torch.cuda.is_available() else "cpu"

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True




## === cell 1
threshold = [0.75, 1.5, 2.5, 3.5]


def regress2class(out: torch.Tensor) -> torch.Tensor:
    """
    out: shape [B] or [B,1] regression output on 0..4.5 scale.
    returns: shape [B] class in {0,1,2,3,4} (float32, like original)
    """
    if out.ndim == 2 and out.size(1) == 1:
        out = out.squeeze(1)
    thr = torch.tensor(threshold, device=out.device, dtype=out.dtype)  # [4]
    pred = (out.unsqueeze(1) >= thr.unsqueeze(0)).sum(dim=1).to(dtype=torch.float32)
    return pred




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
    def __init__(self, backbone=None, pretrained_backbone: bool = False):
        super(ThreeStage_Model, self).__init__()

        self.backbone = timm.models.tf_efficientnet_b4_ns(
            pretrained=pretrained_backbone
        )
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
BASE_INPUT = "../input/aptos2019-blindness-detection"
if not os.path.exists(BASE_INPUT):
    BASE_INPUT = "/kaggle/data/aptos2019-blindness-detection"
if not os.path.exists(BASE_INPUT):
    BASE_INPUT = "/kaggle/input/aptos2019-blindness-detection"

train_csv_path = os.path.join(BASE_INPUT, "train.csv")
train_img_dir = os.path.join(BASE_INPUT, "train_images")

test_csv_path = os.path.join(BASE_INPUT, "test.csv")
test_img_dir = os.path.join(BASE_INPUT, "test_images")

train_df = pd.read_csv(train_csv_path)
test_df = pd.read_csv(test_csv_path)

test_ids = test_df["id_code"].values

input_size = 512
transform = transforms.Compose(
    [
        trim(),
        cropTo4_3(),
        transforms.Resize((input_size * 3 // 4, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.384, 0.258, 0.174], std=[0.124, 0.089, 0.094]),
    ]
)


def find_weight_file(filename: str) -> str:
    direct_candidates = [
        os.path.join(BASE_INPUT, filename),
        os.path.join("../input", filename),
        os.path.join("/kaggle/input", filename),
        os.path.join("/kaggle/data", filename),
        os.path.join("/kaggle/working", filename),
    ]
    for p in direct_candidates:
        if os.path.exists(p):
            return p

    candidates = []
    for root in ["../input", "/kaggle/data", "/kaggle/input"]:
        if os.path.exists(root):
            candidates.extend(
                glob.glob(os.path.join(root, "**", filename), recursive=True)
            )
            if candidates:
                break
    return candidates[0] if candidates else ""


def _clean_state_dict_keys(sd: dict) -> dict:
    if not isinstance(sd, dict):
        return sd
    out = {}
    for k, v in sd.items():
        nk = k
        for prefix in ("module.", "model.", "net."):
            if nk.startswith(prefix):
                nk = nk[len(prefix) :]
        out[nk] = v
    return out


weight_filename = "B4_3stage_60epoch_AdamW_crop.pkl"
weight_path = find_weight_file(weight_filename)

use_pretrained_backbone = not bool(weight_path)
net = ThreeStage_Model(pretrained_backbone=use_pretrained_backbone)

if weight_path:
    state = torch.load(weight_path, map_location="cpu")
    if (
        isinstance(state, dict)
        and "state_dict" in state
        and isinstance(state["state_dict"], dict)
    ):
        sd = _clean_state_dict_keys(state["state_dict"])
        net.load_state_dict(sd, strict=True)
    elif isinstance(state, dict) and all(isinstance(k, str) for k in state.keys()):
        sd = _clean_state_dict_keys(state)
        net.load_state_dict(sd, strict=True)
    else:
        if hasattr(state, "state_dict"):
            net = state
        else:
            raise RuntimeError(f"Unrecognized weight format at {weight_path}")
else:
    print(
        f"WARNING: Weight file '{weight_filename}' not found. "
        f"Using ImageNet-pretrained EfficientNet-B4 backbone and will fine-tune heads briefly."
    )

net = net.to(device)

CACHE_DIR = "/kaggle/working/aptos_cache_tensors_v1"
os.makedirs(CACHE_DIR, exist_ok=True)

_transform_sig = f"trim_cropTo4_3_resize_{input_size*3//4}x{input_size}_norm_0.384_0.258_0.174_0.124_0.089_0.094"
_transform_hash = hashlib.md5(_transform_sig.encode("utf-8")).hexdigest()[:10]


def _cache_path(img_path: str) -> str:
    base = os.path.splitext(os.path.basename(img_path))[0]
    return os.path.join(CACHE_DIR, f"{base}_{_transform_hash}.pt")


class TrainDataset(Dataset):
    def __init__(self, df, img_dir, transform):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        idx = row["id_code"]
        y = int(row["diagnosis"])
        image_name = os.path.join(self.img_dir, f"{idx}.png")
        cp = _cache_path(image_name)
        if os.path.exists(cp):
            img = torch.load(cp, map_location="cpu")
        else:
            img_pil = Image.open(image_name).convert("RGB")
            img = self.transform(img_pil)
            torch.save(img, cp)
        return img, torch.tensor(y, dtype=torch.long)


class TestDataset(Dataset):
    def __init__(self, ids, img_dir, transform):
        self.ids = ids
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        idx = self.ids[i]
        image_name = os.path.join(self.img_dir, f"{idx}.png")
        cp = _cache_path(image_name)
        if os.path.exists(cp):
            img = torch.load(cp, map_location="cpu")
        else:
            img_pil = Image.open(image_name).convert("RGB")
            img = self.transform(img_pil)
            torch.save(img, cp)
        return idx, img




## === cell 5
def _set_requires_grad(module: nn.Module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag


if not weight_path:
    from sklearn.model_selection import StratifiedShuffleSplit

    y_all = train_df["diagnosis"].values
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    tr_idx, va_idx = next(splitter.split(np.zeros(len(y_all)), y_all))
    tr_df = train_df.iloc[tr_idx].reset_index(drop=True)
    va_df = train_df.iloc[va_idx].reset_index(drop=True)

    train_ds = TrainDataset(tr_df, train_img_dir, transform)
    val_ds = TrainDataset(va_df, train_img_dir, transform)

    num_workers = min(8, (os.cpu_count() or 4))
    train_loader = DataLoader(
        train_ds,
        batch_size=4 if device == "cuda" else 2,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=8 if device == "cuda" else 4,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
    )

    _set_requires_grad(net.backbone, False)
    if hasattr(net.backbone, "blocks"):
        _set_requires_grad(net.backbone.blocks[-1], True)
    _set_requires_grad(net.classifier, True)
    _set_requires_grad(net.regressor, True)
    _set_requires_grad(net.ordinal, True)
    _set_requires_grad(net.final_regressor, True)

    opt = torch.optim.AdamW(
        [p for p in net.parameters() if p.requires_grad], lr=2e-4, weight_decay=1e-4
    )
    ce = nn.CrossEntropyLoss()

    net.train()
    start = time.time()
    epochs = 3  # unchanged
    for ep in range(epochs):
        for imgs, y in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            c_out, _, _ = net(imgs, final=False)
            loss = ce(c_out, y)
            loss.backward()
            opt.step()

        net.eval()
        val_loss = 0.0
        n = 0
        with torch.no_grad():
            for imgs, y in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                c_out, _, _ = net(imgs, final=False)
                l = ce(c_out, y)
                bs = imgs.size(0)
                val_loss += l.item() * bs
                n += bs
        val_loss /= max(n, 1)
        net.train()
        print(
            f"epoch {ep+1}/{epochs} val_ce={val_loss:.4f} elapsed={time.time()-start:.1f}s"
        )

    net.eval()
else:
    net.eval()




## === cell 6
test_ds = TestDataset(test_ids, test_img_dir, transform)

num_workers = min(8, (os.cpu_count() or 4))
test_loader = DataLoader(
    test_ds,
    batch_size=(
        16 if device == "cuda" else 4
    ),  # larger batch reduces overhead; inference-only so safe
    shuffle=False,
    num_workers=num_workers,
    pin_memory=(device == "cuda"),
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else None,
)

submission_rows = []

with torch.no_grad():
    for batch_ids, batch_imgs in test_loader:
        batch_imgs = batch_imgs.to(device, non_blocking=True)

        r_out = net(batch_imgs, final=True)

        preds = regress2class(r_out).to(torch.int64).cpu().numpy()
        submission_rows.extend(zip(list(batch_ids), preds.tolist()))

pred_map = {k: int(v) for k, v in submission_rows}
ordered_preds = [pred_map[_id] for _id in test_ids]
submission = pd.DataFrame({"id_code": test_ids, "diagnosis": ordered_preds})

if (
    submission.empty
    or submission.shape[0] != len(test_ids)
    or list(submission.columns) != ["id_code", "diagnosis"]
):
    raise RuntimeError(
        f"Invalid submission shape/columns: got {submission.shape} {list(submission.columns)}, expected ({len(test_ids)}, 2) with ['id_code','diagnosis']"
    )




## === cell 7
submission.to_csv("submission.csv", index=False)
print(submission.head())
print(f"Wrote submission.csv with {len(submission)} rows")
