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

0.9231115838857415

# 6. Current score

0.08416

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.02381) has done: 'I remove the hard dependency on external weight files (which aren’t present in your environment) by loading ImageNet pretrained weights from `timm` instead, so the notebook can run end-to-end and produce a valid `submission.csv`. I also fix the device/type mismatch by moving the model to the target device *after* loading weights, and ensuring inputs and model weights share the same dtype/device. Finally, I make submission generation robust by reading `test.csv` as a DataFrame (keeping `id_code` strings) and using a DataLoader so inference is stable and non-empty.'
- What this solution (achieved 0.08416) has done: 'Your low score is consistent with using an essentially untrained head (random classifier/regressor/ordinal/final layers) while only loading an ImageNet backbone, so predictions collapse and kappa stays near zero. To move the score toward your target with minimal logic change, I (1) load the official `train.csv` and run a short, deterministic fine-tuning pass on the existing `final_regressor` pathway using the model’s own `final=True` output, and (2) compute thresholds on a held-out validation split to maximize quadratic weighted kappa (same evaluation semantics, just calibrated cutpoints). This keeps your architecture and overall inference approach intact, but replaces fixed thresholds with data-driven ones and ensures the final head is trained instead of random. The script still runs end-to-end within the time limit and writes a valid `submission.csv` with the required columns.'

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
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.transforms import functional as FT
from PIL import Image, ImageChops
import cv2

from sklearn.metrics import cohen_kappa_score
import timm

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False




## === cell 1
threshold = [0.75, 1.5, 2.5, 3.5]


def regress2class(out):
    prediction = torch.zeros(out.size(0))
    for i in range(4):
        prediction += (out.data >= threshold[i]).squeeze().cpu()
    return prediction


def ordinal2class_prob(out):
    pred_prob = torch.zeros(out.size(0), 5).to(out.device)
    pred_prob[:, 0] = (1 - out[:, 0]).squeeze()
    pred_prob[:, 1] = (out[:, 0] * (1 - out[:, 1])).squeeze()
    pred_prob[:, 2] = (out[:, 1] * (1 - out[:, 2])).squeeze()
    pred_prob[:, 3] = (out[:, 2] * (1 - out[:, 3])).squeeze()
    pred_prob[:, 4] = out[:, 3].squeeze()
    return F.softmax(pred_prob, dim=1)


def regress2class_prob(out):
    pred_prob = torch.zeros((out.size(0), 5)).to(out.device)
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
DATA_ROOT = "../input/aptos2019-blindness-detection"

TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train_images")

TEST_CSV = os.path.join(DATA_ROOT, "test.csv")
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test_images")

assert os.path.isfile(TEST_CSV), f"Missing test.csv at {TEST_CSV}"
assert os.path.isdir(TEST_IMG_DIR), f"Missing test_images directory at {TEST_IMG_DIR}"
assert os.path.isfile(TRAIN_CSV), f"Missing train.csv at {TRAIN_CSV}"
assert os.path.isdir(
    TRAIN_IMG_DIR
), f"Missing train_images directory at {TRAIN_IMG_DIR}"

train_df = pd.read_csv(TRAIN_CSV)
assert {"id_code", "diagnosis"}.issubset(train_df.columns)
train_df["id_code"] = train_df["id_code"].astype(str)
train_df["diagnosis"] = train_df["diagnosis"].astype(int)

test_df = pd.read_csv(TEST_CSV)
assert "id_code" in test_df.columns
test_df["id_code"] = test_df["id_code"].astype(str)

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

preferred_weight_path = "../input/weights/B4_3stage_19epoch_finetune2.pkl"
candidate_paths = [
    preferred_weight_path,
    os.path.join(DATA_ROOT, "B4_3stage_19epoch_finetune2.pkl"),
]
candidate_paths += glob.glob(
    "../input/**/B4_3stage_19epoch_finetune2.pkl", recursive=True
)

weight_path = None
for p in candidate_paths:
    if os.path.isfile(p):
        weight_path = p
        break

if weight_path is not None:
    print("Loading weights from:", weight_path)
    state = torch.load(weight_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict):
        new_state = {}
        for k, v in state.items():
            nk = k
            if nk.startswith("module."):
                nk = nk[len("module.") :]
            new_state[nk] = v
        state = new_state
    missing, unexpected = net.load_state_dict(state, strict=False)
    print("Missing keys:", len(missing), "Unexpected keys:", len(unexpected))
else:
    print(
        "No provided weights found; using timm ImageNet pretrained backbone weights for tf_efficientnet_b4_ns."
    )
    pretrained_backbone = timm.create_model("tf_efficientnet_b4_ns", pretrained=True)
    backbone_sd = pretrained_backbone.state_dict()
    net.backbone.load_state_dict(backbone_sd, strict=False)

net = net.to(device)




## === cell 5
class TrainDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        idx = self.df.loc[i, "id_code"]
        y = int(self.df.loc[i, "diagnosis"])
        image_name = os.path.join(self.img_dir, f"{idx}.png")
        img = Image.open(image_name).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, y


def _split_train_valid(df, valid_frac=0.15, seed=42):
    rng = np.random.RandomState(seed)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    n_valid = int(round(len(df) * valid_frac))
    valid_idx = idx[:n_valid]
    train_idx = idx[n_valid:]
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[valid_idx].reset_index(
        drop=True
    )


train_tr_df, valid_df = _split_train_valid(train_df, valid_frac=0.15, seed=42)
print("Train/valid sizes:", len(train_tr_df), len(valid_df))

train_ds = TrainDataset(train_tr_df, TRAIN_IMG_DIR, transform=transform)
valid_ds = TrainDataset(valid_df, TRAIN_IMG_DIR, transform=transform)

train_loader = DataLoader(
    train_ds,
    batch_size=8,
    shuffle=True,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)
valid_loader = DataLoader(
    valid_ds,
    batch_size=8,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)

for p in net.parameters():
    p.requires_grad = False
for p in net.final_regressor.parameters():
    p.requires_grad = True

net.train()
optimizer = optim.Adam(net.final_regressor.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

EPOCHS = 2
start = time.time()
for epoch in range(EPOCHS):
    running = 0.0
    n = 0
    for imgs, y in train_loader:
        imgs = imgs.to(device, non_blocking=True)
        y = y.to(device, dtype=torch.float32).view(-1, 1)

        optimizer.zero_grad(set_to_none=True)
        out = net(imgs, final=True)  # [B,1], already sigmoid*4.5
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()

        running += float(loss.item()) * imgs.size(0)
        n += imgs.size(0)

    print(f"Epoch {epoch+1}/{EPOCHS} train MSE:", running / max(1, n))

print("Fine-tuning seconds:", round(time.time() - start, 1))

net.eval()




## === cell 6
@torch.no_grad()
def predict_continuous(model, loader):
    ys = []
    preds = []
    for imgs, y in loader:
        imgs = imgs.to(device, non_blocking=True)
        out = model(imgs, final=True).squeeze(1).detach().float().cpu().numpy()
        preds.append(out)
        ys.append(np.asarray(y, dtype=np.int64))
    return np.concatenate(ys, axis=0), np.concatenate(preds, axis=0)


def apply_thresholds(cont, thr):
    thr = list(thr)
    pred = np.zeros_like(cont, dtype=np.int64)
    for t in thr:
        pred += (cont >= t).astype(np.int64)
    pred = np.clip(pred, 0, 4)
    return pred


def tune_thresholds(y_true, cont_pred, init_thr=(0.75, 1.5, 2.5, 3.5), n_iter=25):
    thr = np.array(init_thr, dtype=np.float64)
    best_thr = thr.copy()
    best = cohen_kappa_score(
        y_true, apply_thresholds(cont_pred, best_thr), weights="quadratic"
    )

    step = 0.25
    for _ in range(n_iter):
        improved = False
        for i in range(4):
            for delta in (-step, step):
                cand = best_thr.copy()
                cand[i] = cand[i] + delta
                cand = np.maximum.accumulate(cand)
                score = cohen_kappa_score(
                    y_true, apply_thresholds(cont_pred, cand), weights="quadratic"
                )
                if score > best:
                    best = score
                    best_thr = cand
                    improved = True
        if not improved:
            step *= 0.5
            if step < 1e-3:
                break
    return best_thr.tolist(), float(best)


y_val, cont_val = predict_continuous(net, valid_loader)
tuned_thr, val_kappa = tune_thresholds(y_val, cont_val, init_thr=threshold, n_iter=30)

print("Old thresholds:", threshold)
print("Tuned thresholds:", tuned_thr)
print("Validation QWK (tuned):", val_kappa)

threshold = tuned_thr




## === cell 7
class TestDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        idx = self.df.loc[i, "id_code"]
        image_name = os.path.join(self.img_dir, f"{idx}.png")
        img = Image.open(image_name).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return idx, img


test_ds = TestDataset(test_df, TEST_IMG_DIR, transform=transform)
test_loader = DataLoader(
    test_ds,
    batch_size=8,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)

submission = []
with torch.no_grad():
    seen = 0
    for batch_i, (ids, imgs) in enumerate(test_loader):
        if batch_i % 10 == 0:
            print("Predicting batch", batch_i, "/", len(test_loader))
        imgs = imgs.to(device, non_blocking=True)

        try:
            r_out = net(imgs, final=True)  # [B,1]
            preds = regress2class(r_out.data.squeeze(1))
        except TypeError:
            _, r_out, _ = net(imgs)
            preds = regress2class(r_out.data.squeeze(1))

        preds = preds.to(torch.int64).cpu().numpy().tolist()
        ids = [str(x) for x in ids]
        submission.extend(list(zip(ids, preds)))
        seen += len(ids)

assert seen == len(test_df), f"Predicted {seen} rows, expected {len(test_df)}"




## === cell 8
df = pd.DataFrame(submission, columns=["id_code", "diagnosis"])

df["id_code"] = df["id_code"].astype(str)
df["diagnosis"] = df["diagnosis"].astype(int)

assert len(df) > 0, "Submission DataFrame is empty"
assert list(df.columns) == ["id_code", "diagnosis"], "Wrong submission columns"
assert len(df) == len(test_df), "Submission row count does not match test.csv"
assert df["diagnosis"].between(0, 4).all(), "Found predictions outside [0,4]"

df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", df.shape)
print(df.head())
