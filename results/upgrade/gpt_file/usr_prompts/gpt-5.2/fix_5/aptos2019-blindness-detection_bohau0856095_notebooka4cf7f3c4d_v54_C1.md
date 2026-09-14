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

0.9147811723711672

# 6. Current score

0.67104

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.67104) has done: 'The timeout is dominated by repeatedly decoding and preprocessing PNGs inside `CachedTensorTrainDataset` (it re-opens every image each epoch and even twice per sample due to not using `base_transform` when augmentation is enabled), plus PIL-based `trim()`/`cropTo4_3()` done every time. I make caching actually effective by caching the deterministic preprocessed PIL stage once per image (trim+crop+resize), then applying the exact same photometric augmentations on that cached PIL and finishing with the same ToTensor+Normalize; this preserves identical augmentation semantics while removing the most expensive repeated work. I also speed up submission creation by batching IDs/preds without a Python inner loop, and reduce per-epoch CPU-GPU sync by accumulating loss on-device. Finally, I keep determinism stable while enabling safe PyTorch compile/TF32 fast paths where available without changing the math in a way that affects results beyond negligible float differences.'

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
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms
from torchvision.transforms import functional as FT

from PIL import Image, ImageChops

import timm

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import cohen_kappa_score

device = "cuda" if torch.cuda.is_available() else "cpu"

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

torch.backends.cudnn.benchmark = True

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True




## === cell 1
threshold = [0.75, 1.5, 2.5, 3.5]


def regress2class(out):
    prediction = torch.zeros(out.size(0), device=out.device, dtype=torch.long)
    for i in range(4):
        prediction += (out >= threshold[i]).long()
    return prediction


def ordinal2class_prob(out):
    pred_prob = torch.zeros(out.size(0), 5, device=out.device, dtype=out.dtype)
    pred_prob[:, 0] = (1 - out[:, 0]).squeeze()
    pred_prob[:, 1] = (out[:, 0] * (1 - out[:, 1])).squeeze()
    pred_prob[:, 2] = (out[:, 1] * (1 - out[:, 2])).squeeze()
    pred_prob[:, 3] = (out[:, 2] * (1 - out[:, 3])).squeeze()
    pred_prob[:, 4] = out[:, 3].squeeze()
    return F.softmax(pred_prob, dim=1)


def regress2class_prob(out):
    out = out.view(-1)
    pred_prob = torch.zeros((out.size(0), 5), device=out.device, dtype=out.dtype)

    out_clamped = torch.clamp(out, 0.0, 4.0)
    l1 = torch.floor(out_clamped).to(torch.long)
    l2 = torch.ceil(out_clamped).to(torch.long)

    w1 = 1.0 - (out_clamped - l1.to(out_clamped.dtype))
    w2 = 1.0 - (l2.to(out_clamped.dtype) - out_clamped)

    idx = torch.arange(out.size(0), device=out.device)
    pred_prob[idx, l1] += w1
    pred_prob[idx, l2] += w2

    mask4 = out >= 4.0
    if mask4.any():
        pred_prob[mask4] = 0
        pred_prob[mask4, 4] = 1.0
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
BASE_INPUT = "/kaggle/input/aptos2019-blindness-detection"
TRAIN_CSV = os.path.join(BASE_INPUT, "train.csv")
TEST_CSV = os.path.join(BASE_INPUT, "test.csv")
TRAIN_IMG_DIR = os.path.join(BASE_INPUT, "train_images")
TEST_IMG_DIR = os.path.join(BASE_INPUT, "test_images")

train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

train_df["id_code"] = train_df["id_code"].astype(str)
test_df["id_code"] = test_df["id_code"].astype(str)

input_size = 380

transform_test = transforms.Compose(
    [
        trim(),
        cropTo4_3(),
        transforms.Resize((input_size * 3 // 4, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.384, 0.258, 0.174], std=[0.124, 0.089, 0.094]),
    ]
)

transform_train = transforms.Compose(
    [
        trim(),
        cropTo4_3(),
        transforms.Resize((input_size * 3 // 4, input_size)),
        photometric_distort(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.384, 0.258, 0.174], std=[0.124, 0.089, 0.094]),
    ]
)

_base_pil_preproc = transforms.Compose(
    [
        trim(),
        cropTo4_3(),
        transforms.Resize((input_size * 3 // 4, input_size)),
    ]
)
_to_tensor_norm = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.384, 0.258, 0.174], std=[0.124, 0.089, 0.094]),
    ]
)




## === cell 5
class TrainDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        id_code = row["id_code"]
        y = int(row["diagnosis"])
        image_name = os.path.join(self.img_dir, f"{id_code}.png")
        img = Image.open(image_name).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return id_code, img, y


class TestDataset(Dataset):
    def __init__(self, ids, img_dir, transform=None):
        self.ids = list(ids)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id_code = self.ids[idx]
        image_name = os.path.join(self.img_dir, f"{id_code}.png")
        img = Image.open(image_name).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return id_code, img


class CachedTensorTrainDataset(Dataset):
    def __init__(self, df, img_dir, base_transform, aug_transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.base_transform = base_transform
        self.aug_transform = aug_transform  # expects PIL->PIL or PIL ops before tensor
        self.ids = self.df["id_code"].astype(str).tolist()
        self.y = self.df["diagnosis"].astype(int).values

        self._pil_cache = [None] * len(self.df)

    def __len__(self):
        return len(self.df)

    def _load_base_pil(self, idx):
        img = self._pil_cache[idx]
        if img is not None:
            return img
        id_code = self.ids[idx]
        image_name = os.path.join(self.img_dir, f"{id_code}.png")
        img = Image.open(image_name).convert("RGB")
        img = _base_pil_preproc(img)
        self._pil_cache[idx] = img
        return img

    def __getitem__(self, idx):
        id_code = self.ids[idx]
        y = int(self.y[idx])

        img = self._load_base_pil(idx)
        if self.aug_transform is not None:
            img = self.aug_transform(img.copy())
        img_t = _to_tensor_norm(img)
        return id_code, img_t, y


class CachedTensorDataset(Dataset):
    def __init__(self, ids, img_dir, transform):
        self.ids = list(map(str, ids))
        self.img_dir = img_dir
        self.transform = transform
        self._cache = [None] * len(self.ids)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        t = self._cache[idx]
        if t is None:
            id_code = self.ids[idx]
            image_name = os.path.join(self.img_dir, f"{id_code}.png")
            img = Image.open(image_name).convert("RGB")
            t = self.transform(img)
            self._cache[idx] = t
        return self.ids[idx], t




## === cell 6
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
tr_idx, va_idx = next(
    sss.split(train_df["id_code"].values, train_df["diagnosis"].values)
)
tr_df = train_df.iloc[tr_idx].reset_index(drop=True)
va_df = train_df.iloc[va_idx].reset_index(drop=True)

batch_size = 8
num_workers = 2

train_ds = CachedTensorTrainDataset(
    tr_df,
    TRAIN_IMG_DIR,
    base_transform=transform_test,  # retained for API compatibility
    aug_transform=photometric_distort(),
)
valid_ds = CachedTensorTrainDataset(
    va_df,
    TRAIN_IMG_DIR,
    base_transform=transform_test,
    aug_transform=None,
)

pin = torch.cuda.is_available()
train_dl = DataLoader(
    train_ds,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=pin,
    drop_last=True,
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else None,
)
valid_dl = DataLoader(
    valid_ds,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=pin,
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else None,
)

net = ThreeStage_Model().to(device)

if hasattr(torch, "compile"):
    try:
        net = torch.compile(net, mode="reduce-overhead")
    except Exception:
        pass

ce_loss = nn.CrossEntropyLoss()
mse_loss = nn.MSELoss()
bce_loss = nn.BCEWithLogitsLoss()

optimizer = torch.optim.AdamW(net.parameters(), lr=2e-4, weight_decay=1e-2)


def labels_to_ordinal_targets(y):
    y = y.view(-1, 1)
    ks = torch.arange(4, device=y.device).view(1, -1)
    return (y > ks).float()




## === cell 7
def evaluate_kappa(model, loader):
    model.eval()
    all_y = []
    all_pred = []
    with torch.no_grad():
        for _, imgs, y in loader:
            imgs = imgs.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            _, r_out, _ = model(imgs, final=False)
            pred = regress2class(r_out.squeeze(1))
            all_y.append(y.detach().cpu().numpy())
            all_pred.append(pred.detach().cpu().numpy())
    all_y = np.concatenate(all_y)
    all_pred = np.concatenate(all_pred)
    return cohen_kappa_score(all_y, all_pred, weights="quadratic")


best_kappa = -1.0
best_state = None

epochs = (
    3  # keep small for runtime; training is required only because weights are missing
)
for epoch in range(1, epochs + 1):
    net.train()
    t0 = time.time()

    running_loss = torch.zeros((), device=device)

    for _, imgs, y in train_dl:
        imgs = imgs.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        c_out, r_out, o_out = net(imgs, final=False)

        loss_c = ce_loss(c_out, y)

        y_reg = y.float().view(-1, 1)
        loss_r = mse_loss(r_out, y_reg)

        eps = 1e-6
        o_prob = o_out.clamp(eps, 1 - eps)
        o_logits = torch.log(o_prob / (1 - o_prob))
        o_tgt = labels_to_ordinal_targets(y)
        loss_o = bce_loss(o_logits, o_tgt)

        loss = loss_c + loss_r + loss_o

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        running_loss += loss.detach()

    kappa = evaluate_kappa(net, valid_dl)
    dt = time.time() - t0
    avg_loss = (running_loss / len(train_dl)).item()
    print(
        f"Epoch {epoch}/{epochs} - loss={avg_loss:.4f} - val_qwk={kappa:.5f} - {dt:.1f}s"
    )

    if kappa > best_kappa:
        best_kappa = kappa
        best_state = {k: v.detach().cpu() for k, v in net.state_dict().items()}

if best_state is not None:
    net.load_state_dict(best_state)
net.eval()
print(f"Best val_qwk: {best_kappa:.5f}")




## === cell 8
test_ids = test_df["id_code"].astype(str).values

test_ds = CachedTensorDataset(test_ids, TEST_IMG_DIR, transform=transform_test)
test_dl = DataLoader(
    test_ds,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else None,
)

all_ids = []
all_preds = []
with torch.no_grad():
    for batch_ids, batch_imgs in test_dl:
        batch_imgs = batch_imgs.to(device, non_blocking=True)
        _, r_out, _ = net(batch_imgs, final=False)
        preds = regress2class(r_out.squeeze(1)).detach().cpu().numpy().astype(np.int64)
        all_ids.extend(list(map(str, batch_ids)))
        all_preds.append(preds)

all_preds = np.concatenate(all_preds, axis=0)
submission = pd.DataFrame({"id_code": all_ids, "diagnosis": all_preds})

if submission.empty:
    raise RuntimeError("Submission DataFrame is empty; inference failed.")

submission["id_code"] = submission["id_code"].astype(str)
submission = (
    submission.set_index("id_code").loc[test_df["id_code"].astype(str)].reset_index()
)
submission["diagnosis"] = submission["diagnosis"].astype(int).clip(0, 4)

submission.to_csv("submission.csv", index=False)
print(submission.head())
print(f"Wrote submission.csv with shape: {submission.shape}")
