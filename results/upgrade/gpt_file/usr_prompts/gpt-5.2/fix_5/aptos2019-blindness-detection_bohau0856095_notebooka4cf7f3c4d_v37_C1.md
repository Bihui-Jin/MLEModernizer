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

from sklearn.model_selection import train_test_split
import timm

device = "cuda" if torch.cuda.is_available() else "cpu"

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def _seed_worker(worker_id: int):
    seed = 42 + worker_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


_g = torch.Generator()
_g.manual_seed(42)




## === cell 1
threshold = [0.75, 1.5, 2.5, 3.5]


def regress2class(out):
    thr = torch.tensor(threshold, device=out.device, dtype=out.dtype).view(1, -1)
    return (out.view(-1, 1) >= thr).sum(dim=1).to(dtype=torch.float32).cpu()


def ordinal2class_prob(out):
    dev = out.device
    pred_prob = torch.zeros(out.size(0), 5, device=dev)
    pred_prob[:, 0] = (1 - out[:, 0]).squeeze()
    pred_prob[:, 1] = (out[:, 0] * (1 - out[:, 1])).squeeze()
    pred_prob[:, 2] = (out[:, 1] * (1 - out[:, 2])).squeeze()
    pred_prob[:, 3] = (out[:, 2] * (1 - out[:, 3])).squeeze()
    pred_prob[:, 4] = out[:, 3].squeeze()
    return F.softmax(pred_prob, dim=1)


def regress2class_prob(out):
    dev = out.device
    pred_prob = torch.zeros((out.size(0), 5), device=dev)
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
BASE = "../input/aptos2019-blindness-detection"
TRAIN_CSV = f"{BASE}/train.csv"
TEST_CSV = f"{BASE}/test.csv"
TRAIN_IMG_DIR = f"{BASE}/train_images"
TEST_IMG_DIR = f"{BASE}/test_images"
WEIGHTS_PATH = "../input/weights/B4_3stage_44epoch_CLAHE.pkl"

train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

train_ids = train_df["id_code"].astype(str).values
train_labels = train_df["diagnosis"].astype(int).values
test_ids = test_df["id_code"].astype(str).values

input_size = 380

transform_eval = transforms.Compose(
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
        photometric_distort(),
        transforms.Resize((input_size * 3 // 4, input_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.384, 0.258, 0.174], std=[0.124, 0.089, 0.094]),
    ]
)


class AptosDataset(Dataset):
    def __init__(self, df, img_dir, transform):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img_path = os.path.join(self.img_dir, f"{row['id_code']}.png")
        img = Image.open(img_path).convert("RGB")
        x = self.transform(img)
        y = int(row["diagnosis"]) if "diagnosis" in self.df.columns else -1
        return x, y, str(row["id_code"])


class AptosTestTTADataset(Dataset):
    def __init__(self, ids, img_dir, transform, use_tta: bool = True):
        self.ids = list(map(str, ids))
        self.img_dir = img_dir
        self.transform = transform
        self.use_tta = use_tta

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        idx = self.ids[i]
        img_path = os.path.join(self.img_dir, f"{idx}.png")
        if not os.path.exists(img_path):
            return None, idx

        img0 = Image.open(img_path).convert("RGB")
        x0 = self.transform(img0)
        if self.use_tta:
            x1 = self.transform(FT.hflip(img0))
            x = torch.stack([x0, x1], dim=0)  # [2,C,H,W]
        else:
            x = x0.unsqueeze(0)  # [1,C,H,W]
        return x, idx


def _collate_test_tta(batch):
    xs = []
    ids = []
    missing = []
    for item, idx in batch:
        if item is None:
            missing.append(idx)
        else:
            xs.append(item)
            ids.append(idx)
    if len(xs) == 0:
        x = None
    else:
        x = torch.stack(xs, dim=0)  # [B,T,C,H,W]
    return x, ids, missing




## === cell 5
net = ThreeStage_Model()
using_competition_weights = False

if os.path.exists(WEIGHTS_PATH):
    state = torch.load(WEIGHTS_PATH, map_location="cpu")
    net.load_state_dict(state)
    using_competition_weights = True
else:
    net.backbone = timm.models.tf_efficientnet_b4_ns(pretrained=True)
    net.backbone.global_pool = GeM(flatten=True)

net = net.to(device)
print(f"using_competition_weights={using_competition_weights} device={device}")




## === cell 6
if not using_competition_weights:
    tr_df, va_df = train_test_split(
        train_df, test_size=0.15, random_state=42, stratify=train_df["diagnosis"]
    )

    train_ds = AptosDataset(tr_df, TRAIN_IMG_DIR, transform_train)
    val_ds = AptosDataset(va_df, TRAIN_IMG_DIR, transform_eval)

    bs = 8 if device == "cuda" else 2
    train_loader = DataLoader(
        train_ds,
        batch_size=bs,
        shuffle=True,
        num_workers=2,
        pin_memory=(device == "cuda"),
        worker_init_fn=_seed_worker,
        generator=_g,
        persistent_workers=True if 2 > 0 else False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=bs,
        shuffle=False,
        num_workers=2,
        pin_memory=(device == "cuda"),
        worker_init_fn=_seed_worker,
        generator=_g,
        persistent_workers=True if 2 > 0 else False,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(net.parameters(), lr=2e-5, weight_decay=1e-4)

    epochs = 2  # unchanged
    net.train()
    start = time.time()

    for ep in range(1, epochs + 1):
        ep_loss = 0.0
        n = 0
        for xb, yb, _ in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            c_out, r_out, o_out = net(xb)
            loss = criterion(c_out, yb)
            loss.backward()
            optimizer.step()

            ep_loss += float(loss.item()) * xb.size(0)
            n += xb.size(0)

        net.eval()
        correct = 0
        tot = 0
        with torch.no_grad():
            for xb, yb, _ in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                c_out, _, _ = net(xb)
                pred = torch.argmax(c_out, dim=1)
                correct += int((pred == yb).sum().item())
                tot += xb.size(0)
        net.train()

        dt = time.time() - start
        print(
            f"epoch {ep}/{epochs} loss={ep_loss/max(n,1):.4f} val_acc={correct/max(tot,1):.4f} time={dt:.1f}s"
        )

    net.eval()
    torch.set_grad_enabled(False)




## === cell 7
net.eval()
torch.set_grad_enabled(False)

use_tta = True
test_ds = AptosTestTTADataset(test_ids, TEST_IMG_DIR, transform_eval, use_tta=use_tta)

num_workers_test = 4 if device == "cuda" else 2
bs_test = 8 if device == "cuda" else 2

test_loader = DataLoader(
    test_ds,
    batch_size=bs_test,
    shuffle=False,
    num_workers=num_workers_test,
    pin_memory=(device == "cuda"),
    collate_fn=_collate_test_tta,
    worker_init_fn=_seed_worker,
    generator=_g,
    persistent_workers=True if num_workers_test > 0 else False,
)

pred_map = {}
missing_images = 0

with torch.no_grad():
    for x, ids_ok, missing in test_loader:
        for mid in missing:
            pred_map[mid] = 0
        missing_images += len(missing)

        if x is None:
            continue

        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w).to(device, non_blocking=True)

        c_out, r_out, o_out = net(x)

        if using_competition_weights:
            cls_each = regress2class(r_out.data.squeeze(1)).to(
                torch.int64
            )  # CPU tensor length b*t
            cls_each = cls_each.view(b, t)
            pred_class = (
                torch.round(cls_each.float().mean(dim=1)).to(torch.int64).tolist()
            )
        else:
            probs = F.softmax(c_out, dim=1).view(b, t, 5).mean(dim=1)
            pred_class = torch.argmax(probs, dim=1).to(torch.int64).tolist()

        for idx, p in zip(ids_ok, pred_class):
            p = int(p)
            if p < 0:
                p = 0
            elif p > 4:
                p = 4
            pred_map[idx] = p

submission = np.empty((len(test_ids), 2), dtype=object)
for i, idx in enumerate(test_ids):
    submission[i, 0] = idx
    submission[i, 1] = int(pred_map.get(str(idx), 0))

assert submission.shape[0] == len(test_ids), "Submission row count mismatch."




## === cell 8
df = pd.DataFrame(submission, columns=["id_code", "diagnosis"])
df["id_code"] = df["id_code"].astype(str)
df["diagnosis"] = df["diagnosis"].astype(int)

df = df.set_index("id_code").loc[test_ids].reset_index()

out_path = "submission.csv"
df.to_csv(out_path, index=False)

print(f"Wrote {out_path} with shape={df.shape}, missing_images={missing_images}")
print(df.head())
print("diagnosis value counts:\n", df["diagnosis"].value_counts().sort_index())
