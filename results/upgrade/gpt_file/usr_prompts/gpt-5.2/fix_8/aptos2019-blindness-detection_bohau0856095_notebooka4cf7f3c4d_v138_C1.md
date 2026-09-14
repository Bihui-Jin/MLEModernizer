# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os
import math
import time
import glob
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.utils.data import Dataset, DataLoader, random_split

import torchvision.transforms as transforms
from torchvision.transforms import functional as FT
from torchvision.io import read_image, ImageReadMode

from PIL import Image, ImageChops
import cv2

from sklearn.metrics import cohen_kappa_score
import timm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)



## === cell 1
threshold = [0.75, 1.5, 2.5, 3.5]


def regress2class(out):
    prediction = 0
    for i in range(4):
        prediction += (out.data >= threshold[i]).squeeze().cpu().item()
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
            l1 = int(math.floor(out[i]))
            l2 = int(math.ceil(out[i]))
            pred_prob[i][l1] = 1 - (out[i] - l1)
            pred_prob[i][l2] = 1 - (l2 - out[i])
        else:
            pred_prob[i][4] = 1.0
    return pred_prob


def combine3output(r_out, c_out, o_out):
    R = regress2class(r_out.data)
    _, C = torch.max(c_out.data, 1)
    C = C.squeeze().item()
    _, O = torch.max(o_out.data, 1)
    O = O.squeeze().item()

    P = (R + C + O) / 3.0
    P = int(round(P))
    return P




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


def crop_image_from_gray(img, tol=7):
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol

        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if check_shape == 0:
            return img
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)
        return img




## === cell 4
def resolve_base_input():
    candidates = [
        "../input/aptos2019-blindness-detection",
        "/kaggle/input/aptos2019-blindness-detection",
        "../kaggle/data/aptos2019-blindness-detection",
        "/kaggle/data/aptos2019-blindness-detection",
        "../input",
        "/kaggle/input",
        "../kaggle/data",
        "/kaggle/data",
    ]
    for base in candidates:
        if os.path.exists(os.path.join(base, "train.csv")) and os.path.exists(
            os.path.join(base, "test.csv")
        ):
            if os.path.basename(base) in ["input", "data"] and os.path.exists(
                os.path.join(base, "aptos2019-blindness-detection", "train.csv")
            ):
                return os.path.join(base, "aptos2019-blindness-detection")
            return base
    return "../input/aptos2019-blindness-detection"


BASE_INPUT = resolve_base_input()
train_csv_path = os.path.join(BASE_INPUT, "train.csv")
test_csv_path = os.path.join(BASE_INPUT, "test.csv")
train_img_dir = os.path.join(BASE_INPUT, "train_images")
test_img_dir = os.path.join(BASE_INPUT, "test_images")

print("Using BASE_INPUT:", BASE_INPUT)
print("train_csv_path exists:", os.path.exists(train_csv_path))
print("test_csv_path exists:", os.path.exists(test_csv_path))
print("train_img_dir exists:", os.path.exists(train_img_dir))
print("test_img_dir exists:", os.path.exists(test_img_dir))

train_df = pd.read_csv(train_csv_path)
test_df = pd.read_csv(test_csv_path)
test_ids = test_df["id_code"].values

transform_infer = transforms.Compose(
    [
        trim(),
        cropTo4_3(),
        transforms.Resize((288, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.384, 0.258, 0.174], std=[0.124, 0.089, 0.094]),
    ]
)

transform_train = transforms.Compose(
    [
        trim(),
        cropTo4_3(),
        transforms.Resize((288, 384)),
        photometric_distort(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.384, 0.258, 0.174], std=[0.124, 0.089, 0.094]),
    ]
)


def _safe_makedirs(p):
    try:
        os.makedirs(p, exist_ok=True)
    except Exception:
        pass


class APTOSDataset(Dataset):
    def __init__(
        self,
        df,
        img_dir,
        transform,
        with_labels=True,
        cache_transformed=False,
        cache_dir=None,
        use_fast_png_decode=True,
    ):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.with_labels = with_labels
        self.cache_transformed = cache_transformed
        self.cache_dir = cache_dir
        self.use_fast_png_decode = use_fast_png_decode

        if self.cache_transformed:
            if self.cache_dir is None:
                self.cache_dir = os.path.join("/kaggle/working", "aptos_cache")
            _safe_makedirs(self.cache_dir)

    def __len__(self):
        return len(self.df)

    def _img_path(self, id_code: str) -> str:
        return os.path.join(self.img_dir, f"{id_code}.png")

    def _cache_path(self, id_code: str) -> str:
        tag = "infer" if self.transform is transform_infer else "train"
        return os.path.join(self.cache_dir, f"{id_code}__{tag}.pt")

    def _load_pil_rgb(self, img_path: str) -> Image.Image:
        if self.use_fast_png_decode:
            t = read_image(img_path, mode=ImageReadMode.RGB)  # uint8 CHW
            np_img = t.permute(1, 2, 0).contiguous().numpy()
            return Image.fromarray(np_img, mode="RGB")
        return Image.open(img_path).convert("RGB")

    def _load_and_transform(self, img_path: str):
        img = self._load_pil_rgb(img_path)
        return self.transform(img)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        id_code = str(row["id_code"])
        img_path = self._img_path(id_code)

        if self.cache_transformed:
            cpath = self._cache_path(id_code)
            try:
                img = torch.load(cpath, map_location="cpu", weights_only=False)
            except Exception:
                img = self._load_and_transform(img_path)
                tmp = cpath + f".tmp.{os.getpid()}"
                torch.save(img, tmp)
                os.replace(tmp, cpath)
        else:
            img = self._load_and_transform(img_path)

        if self.with_labels:
            y = int(row["diagnosis"])
            return img, y
        return img, id_code




## === cell 5
net1 = ThreeStage_Model().to(device)

ce_loss = nn.CrossEntropyLoss()
mse_loss = nn.MSELoss()
bce_loss = nn.BCELoss()


def label_to_ordinal_targets(y, num_levels=4):
    y = y.view(-1, 1)
    levels = torch.arange(num_levels, device=y.device).view(1, -1)
    return (y > levels).float()


full_train_ds = APTOSDataset(
    train_df, train_img_dir, transform_train, with_labels=True, cache_transformed=False
)

val_size = max(1, int(0.1 * len(full_train_ds)))
train_size = len(full_train_ds) - val_size
train_ds, val_ds = random_split(
    full_train_ds, [train_size, val_size], generator=torch.Generator().manual_seed(42)
)

val_df = train_df.iloc[val_ds.indices].reset_index(drop=True)
val_cached_ds = APTOSDataset(
    val_df,
    train_img_dir,
    transform_infer,
    with_labels=True,
    cache_transformed=True,
    cache_dir=os.path.join("/kaggle/working", "aptos_cache_val"),
)

_num_workers = min(8, (os.cpu_count() or 2))
train_loader = DataLoader(
    train_ds,
    batch_size=8,
    shuffle=True,
    num_workers=_num_workers,
    pin_memory=(device.type == "cuda"),
    persistent_workers=(_num_workers > 0),
    prefetch_factor=4 if _num_workers > 0 else None,
)
val_loader = DataLoader(
    val_cached_ds,
    batch_size=8,
    shuffle=False,
    num_workers=_num_workers,
    pin_memory=(device.type == "cuda"),
    persistent_workers=(_num_workers > 0),
    prefetch_factor=4 if _num_workers > 0 else None,
)

optimizer = optim.Adam(net1.parameters(), lr=1e-4)

_thresholds_t = torch.tensor(threshold, device=device).view(1, -1)


def predict_batch(c_out, r_out, o_out_sigmoid):
    R = (r_out >= _thresholds_t).sum(dim=1)  # (B,)
    C = torch.argmax(c_out, dim=1)
    O = torch.argmax(o_out_sigmoid, dim=1)
    P = torch.round(
        (R.to(torch.float32) + C.to(torch.float32) + O.to(torch.float32)) / 3.0
    ).to(torch.int64)
    return P


def evaluate_kappa(model, loader, max_batches=None):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for bi, (x, y) in enumerate(loader):
            if max_batches is not None and bi >= max_batches:
                break
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            c_out, r_out, o_out = model(x)
            o_out = torch.sigmoid(o_out)
            p = predict_batch(c_out, r_out, o_out)
            ps.extend(p.detach().cpu().numpy().tolist())
            ys.extend(y.detach().cpu().numpy().tolist())
    return cohen_kappa_score(ys, ps, weights="quadratic")


EPOCHS = 8

net1.train()
for epoch in range(EPOCHS):
    t0 = time.time()
    running_loss = 0.0
    for x, y in train_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        c_out, r_out, o_out = net1(x)

        loss_c = ce_loss(c_out, y)

        y_reg = y.float().view(-1, 1)
        loss_r = mse_loss(r_out, y_reg)

        y_ord = label_to_ordinal_targets(y, num_levels=4)
        loss_o = bce_loss(torch.sigmoid(o_out), y_ord)

        loss = loss_c + 0.5 * loss_r + 0.5 * loss_o

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    val_kappa = evaluate_kappa(net1, val_loader, max_batches=50)
    print(
        f"Epoch {epoch+1}/{EPOCHS} - loss: {epoch_loss:.4f} - val_qwk(sampled): {val_kappa:.4f} - time: {time.time()-t0:.1f}s"
    )

net1.eval()



## === cell 6
test_infer_ds = APTOSDataset(
    test_df,
    test_img_dir,
    transform_infer,
    with_labels=False,
    cache_transformed=True,
    cache_dir=os.path.join("/kaggle/working", "aptos_cache_test"),
)
test_loader = DataLoader(
    test_infer_ds,
    batch_size=8,
    shuffle=False,
    num_workers=_num_workers,
    pin_memory=(device.type == "cuda"),
    persistent_workers=(_num_workers > 0),
    prefetch_factor=4 if _num_workers > 0 else None,
)

submission = []
seen = 0
with torch.no_grad():
    for xb, idb in test_loader:
        if seen % 50 == 0:
            print(f"Infer {seen}/{len(test_infer_ds)}")
        xb = xb.to(device, non_blocking=True)
        c_out, r_out, o_out = net1(xb)
        o_out = torch.sigmoid(o_out)
        preds = (
            predict_batch(c_out, r_out, o_out)
            .detach()
            .cpu()
            .numpy()
            .astype(int)
            .tolist()
        )
        for _id, p in zip(idb, preds):
            submission.append([_id, int(p)])
        seen += len(idb)

submission = np.array(submission, dtype=object)



## === cell 7
df = pd.DataFrame(submission, columns=["id_code", "diagnosis"])
df["id_code"] = df["id_code"].astype(str)
df["diagnosis"] = df["diagnosis"].astype(int).clip(0, 4)

df = test_df[["id_code"]].merge(df, on="id_code", how="left")
df["diagnosis"] = df["diagnosis"].fillna(0).astype(int).clip(0, 4)

out_path = "submission.csv"
df.to_csv(out_path, index=False)
print("Wrote", out_path, "with shape", df.shape)
print(df.head())
