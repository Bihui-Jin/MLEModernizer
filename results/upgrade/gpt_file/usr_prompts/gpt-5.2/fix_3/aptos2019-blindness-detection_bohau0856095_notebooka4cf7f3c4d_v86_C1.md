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

0.9224480930331804

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5993) has done: 'I fix the missing weight-file issue by removing the dependency on `../input/weights/...` and instead training the exact same `ThreeStage_Model` defined in your code on `train.csv`, then running inference on `test.csv`. I also fix the CUDA crash by selecting `cuda` only if available and otherwise running on CPU. To make the solution finish within the time limit while preserving the same architecture and loss family (classification/regression/ordinal), I train on a single resized resolution with a standard multi-head loss and then generate a non-empty `submission.csv` with the required columns and row order. Finally, I make the transforms robust (your `trim()` sometimes returns `None`) so the pipeline never produces an empty submission.'
- What this solution (achieved 0.0) has done: 'Your current score is far below the target, so the smallest safe way to move toward 0.922 is to (1) use ImageNet-pretrained weights for the same EfficientNet backbones (no architecture change, just initialization), and (2) actually use your model’s intended “final” head at inference (you trained with multi-head supervision but predicted only from the regressor). To better align predictions with quadratic weighted kappa without changing losses/training loops, I also pick thresholds on the validation fold (instead of fixed [0.75,1.5,2.5,3.5]) and then apply those thresholds to test predictions. These are minimal changes that keep the same dataset, transforms, losses, and training structure, while typically yielding a large kappa jump versus random-from-scratch + fixed thresholds. The script still runs end-to-end and writes a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import random
import math
import time
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms
from torchvision.transforms import functional as FT
from PIL import Image, ImageChops

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score

import timm


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


seed_everything(42)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)




## === cell 1
threshold = [0.75, 1.5, 2.5, 3.5]


def regress2class(out):
    prediction = torch.zeros(out.size(0), device="cpu")
    out_cpu = out.detach().view(-1).cpu()
    for i in range(4):
        prediction += (out_cpu >= threshold[i]).to(torch.float32)
    return prediction


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
    outv = out.detach().view(-1)
    for i in range(outv.size(0)):
        if outv[i] < 4.0:
            l1 = int(math.floor(float(outv[i].item())))
            l2 = int(math.ceil(float(outv[i].item())))
            pred_prob[i][l1] = 1 - (outv[i] - l1)
            pred_prob[i][l2] = 1 - (l2 - outv[i])
        else:
            pred_prob[i][4] = 1.0
    return pred_prob


def regress2class_with_thresholds(out, thr):
    out_cpu = out.detach().view(-1).cpu().numpy()
    thr = np.asarray(thr, dtype=np.float32)
    pred = np.zeros(out_cpu.shape[0], dtype=np.int64)
    for t in thr:
        pred += (out_cpu >= t).astype(np.int64)
    return pred


def tune_thresholds_by_val_kappa(
    y_true, y_pred_cont, init_thr=(0.75, 1.5, 2.5, 3.5), n_iter=60
):
    """
    Coordinate-descent on thresholds to maximize quadratic weighted kappa on a single validation fold.
    Minimal-risk post-processing that keeps core model unchanged but better matches the competition metric.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred_cont = np.asarray(y_pred_cont, dtype=np.float32)

    thr = np.array(init_thr, dtype=np.float32)

    def score(thr_vec):
        thr_vec = np.sort(thr_vec)
        pred = regress2class_with_thresholds(torch.from_numpy(y_pred_cont), thr_vec)
        return cohen_kappa_score(y_true, pred, weights="quadratic")

    best = score(thr)

    steps = [0.30, 0.15, 0.08, 0.04, 0.02]
    for step in steps:
        improved = True
        inner = 0
        while improved and inner < n_iter:
            improved = False
            inner += 1
            for i in range(4):
                for delta in (-step, step):
                    cand = thr.copy()
                    cand[i] = cand[i] + delta
                    cand = np.clip(cand, -0.5, 4.5)
                    cand = np.sort(cand)
                    s = score(cand)
                    if s > best + 1e-8:
                        thr = cand
                        best = s
                        improved = True
    return thr.tolist(), float(best)




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
        self.backbone = timm.models.tf_efficientnet_b5_ns(pretrained=True)
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
TRAIN_CSV = os.path.join(BASE, "train.csv")
TEST_CSV = os.path.join(BASE, "test.csv")
TRAIN_IMG_DIR = os.path.join(BASE, "train_images")
TEST_IMG_DIR = os.path.join(BASE, "test_images")

train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

print("train:", train_df.shape, "test:", test_df.shape)

input_size = 384

mean = [0.384, 0.258, 0.174]
std = [0.124, 0.089, 0.094]

train_transform = transforms.Compose(
    [
        trim(),
        cropTo4_3(),
        transforms.Resize((input_size * 3 // 4, input_size)),
        photometric_distort(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
)

test_transform = transforms.Compose(
    [
        trim(),
        cropTo4_3(),
        transforms.Resize((input_size * 3 // 4, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
)


class APTOSDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, has_label=True):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.has_label = has_label

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, f"{row['id_code']}.png")
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        if self.has_label:
            y = int(row["diagnosis"])
            return img, y
        return img, row["id_code"]




## === cell 5
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, val_idx = next(skf.split(train_df["id_code"], train_df["diagnosis"]))

tr_df = train_df.iloc[train_idx].reset_index(drop=True)
va_df = train_df.iloc[val_idx].reset_index(drop=True)

train_ds = APTOSDataset(tr_df, TRAIN_IMG_DIR, transform=train_transform, has_label=True)
val_ds = APTOSDataset(va_df, TRAIN_IMG_DIR, transform=test_transform, has_label=True)

batch_size = 8 if device == "cuda" else 4
num_workers = 2

train_loader = DataLoader(
    train_ds,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=(device == "cuda"),
)
val_loader = DataLoader(
    val_ds,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=(device == "cuda"),
)

net = ThreeStage_Model().to(device)

ce_loss = nn.CrossEntropyLoss()
l1_loss = nn.SmoothL1Loss()


def make_ordinal_targets(y):
    y = y.view(-1, 1)
    ks = torch.arange(4, device=y.device).view(1, -1)
    return (y > ks).to(torch.float32)


optimizer = optim.Adam(net.parameters(), lr=1e-4)


def evaluate_kappa(model, loader):
    model.eval()
    preds = []
    gts = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            r_final = model(x, final=True)
            pred_cls = regress2class(r_final.squeeze(1)).numpy().astype(int)
            preds.extend(pred_cls.tolist())
            gts.extend(y.detach().cpu().numpy().astype(int).tolist())
    return cohen_kappa_score(gts, preds, weights="quadratic")


epochs = 2 if device == "cuda" else 1
print("epochs:", epochs)

best_kappa = -1.0
best_state = None

for epoch in range(epochs):
    net.train()
    t0 = time.time()
    running = 0.0
    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)

        c_out, r_out, o_out = net(x)

        loss_c = ce_loss(c_out, y)

        y_reg = y.to(torch.float32)
        loss_r = l1_loss(r_out.squeeze(1), y_reg)

        y_ord = make_ordinal_targets(y)
        loss_o = F.binary_cross_entropy(o_out, y_ord)

        loss = loss_c + loss_r + loss_o

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        running += float(loss.item())

    kappa = evaluate_kappa(net, val_loader)
    dt = time.time() - t0
    print(
        f"epoch {epoch+1}/{epochs} loss={running/max(1,len(train_loader)):.4f} val_kappa={kappa:.4f} time={dt:.1f}s"
    )

    if kappa > best_kappa:
        best_kappa = kappa
        best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}

if best_state is not None:
    net.load_state_dict(best_state)
net.eval()

val_y_true = []
val_y_pred_cont = []
with torch.no_grad():
    for x, y in val_loader:
        x = x.to(device)
        r_final = net(x, final=True).squeeze(1).detach().cpu().numpy()
        val_y_pred_cont.extend(r_final.tolist())
        val_y_true.extend(y.numpy().astype(int).tolist())

new_thr, tuned_kappa = tune_thresholds_by_val_kappa(
    val_y_true, val_y_pred_cont, init_thr=threshold
)
print("old thresholds:", threshold)
print("tuned thresholds:", new_thr, "tuned_val_kappa:", f"{tuned_kappa:.4f}")
threshold = new_thr




## === cell 6
test_ds = APTOSDataset(test_df, TEST_IMG_DIR, transform=test_transform, has_label=False)
test_loader = DataLoader(
    test_ds,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=(device == "cuda"),
)

ids = []
preds = []

with torch.no_grad():
    for x, id_code in test_loader:
        x = x.to(device)
        r_final = net(x, final=True)
        pred = regress2class(r_final.squeeze(1)).numpy().astype(int)
        ids.extend(list(id_code))
        preds.extend(pred.tolist())

sub = pd.DataFrame({"id_code": ids, "diagnosis": preds})
sub = test_df.merge(sub, on="id_code", how="left")
assert sub["diagnosis"].notna().all(), "Some predictions are missing"
sub["diagnosis"] = sub["diagnosis"].astype(int)
sub.to_csv("submission.csv", index=False)

print(sub.head())
print("wrote submission.csv with shape:", sub.shape)
