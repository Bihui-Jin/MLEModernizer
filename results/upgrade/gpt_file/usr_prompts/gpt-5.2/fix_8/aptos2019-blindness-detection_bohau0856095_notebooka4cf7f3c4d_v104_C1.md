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

import timm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score

device = "cuda" if torch.cuda.is_available() else "cpu"

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if device == "cuda":
    torch.cuda.manual_seed_all(42)



## === cell 1
threshold = [0.75, 1.5, 2.5, 3.5]


def regress2class(out):
    prediction = torch.zeros(out.size(0), dtype=torch.int64)
    for i in range(4):
        prediction += (out.data >= threshold[i]).squeeze().cpu().to(torch.int64)
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
    pred_prob = torch.zeros((out.size(0), 5), device=out.device, dtype=out.dtype)
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
    def __init__(self, backbone_pretrained=False):
        super(Regressor, self).__init__()
        self.backbone = timm.models.tf_efficientnet_b5_ns(
            pretrained=backbone_pretrained
        )
        self.backbone.global_pool = GeM(flatten=True)
        self.regressor = nn.Linear(1000, 1)

    def forward(self, x):
        x = self.backbone(x)
        out = self.regressor(x)
        out = torch.sigmoid(out) * 4.5
        return out


class ThreeStage_Model(nn.Module):
    def __init__(self, backbone=None, backbone_pretrained=False):
        super(ThreeStage_Model, self).__init__()

        self.backbone = timm.models.tf_efficientnet_b4_ns(
            pretrained=backbone_pretrained
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
BASE_INPUT = "../input/aptos2019-blindness-detection"
TRAIN_CSV = os.path.join(BASE_INPUT, "train.csv")
TEST_CSV = os.path.join(BASE_INPUT, "test.csv")
TRAIN_IMG_DIR = os.path.join(BASE_INPUT, "train_images")
TEST_IMG_DIR = os.path.join(BASE_INPUT, "test_images")

train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)
test_ids = test_df["id_code"].values

MODEL_NAME = "tf_efficientnet_b4_ns"
data_cfg = timm.data.resolve_model_data_config(
    timm.create_model(MODEL_NAME, pretrained=True)
)
input_size = int(data_cfg["input_size"][-1])  # typically 380 for b4
mean = data_cfg["mean"]
std = data_cfg["std"]

transform = transforms.Compose(
    [
        trim(),
        cropTo4_3(),
        transforms.Resize(
            (input_size * 3 // 4, input_size),
            interpolation=transforms.InterpolationMode.BILINEAR,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
)

EXPECTED_WEIGHT_NAME = "B4_3stage_27epoch_320.pkl"

candidate_weight_paths = [
    "../input/weights/B4_3stage_27epoch_320.pkl",
    os.path.join(BASE_INPUT, "weights", EXPECTED_WEIGHT_NAME),
    "/kaggle/input/weights/B4_3stage_27epoch_320.pkl",
]

for root, dirs, files in os.walk("../input"):
    if EXPECTED_WEIGHT_NAME in files:
        candidate_weight_paths.append(os.path.join(root, EXPECTED_WEIGHT_NAME))

seen = set()
candidate_weight_paths = [
    p for p in candidate_weight_paths if not (p in seen or seen.add(p))
]
weight_path = next((p for p in candidate_weight_paths if os.path.exists(p)), None)
use_checkpoint = weight_path is not None

net = ThreeStage_Model(backbone_pretrained=True)

if use_checkpoint:
    print("Loading weights from:", weight_path)
    state = torch.load(weight_path, map_location="cpu")
    if (
        isinstance(state, dict)
        and "state_dict" in state
        and isinstance(state["state_dict"], dict)
    ):
        state = state["state_dict"]
    if isinstance(state, dict):
        new_state = {}
        for k, v in state.items():
            nk = k.replace("module.", "")
            new_state[nk] = v
        state = new_state

    missing, unexpected = net.load_state_dict(state, strict=False)
    if len(unexpected) > 0:
        raise RuntimeError(
            "Checkpoint has unexpected keys for this model: "
            f"{unexpected[:10]}{'...' if len(unexpected) > 10 else ''}"
        )
    if len(missing) > 0:
        print(
            "WARNING: Checkpoint missing keys; proceeding with available weights. "
            f"Missing keys (first 10): {missing[:10]}{'...' if len(missing) > 10 else ''}"
        )
else:
    print(
        "WARNING: No external checkpoint found; using timm pretrained EfficientNet-B4 backbone "
        "with randomly initialized heads. Will do minimal head-only training of final_regressor to move score upward."
    )

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
        row = self.df.iloc[i]
        idx = str(row["id_code"])
        y = int(row["diagnosis"])
        image_name = os.path.join(self.img_dir, f"{idx}.png")
        img = Image.open(image_name).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, y


class TestDataset(Dataset):
    def __init__(self, ids, img_dir, transform=None):
        self.ids = list(ids)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        idx = self.ids[i]
        image_name = os.path.join(self.img_dir, f"{idx}.png")
        img = Image.open(image_name).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return str(idx), img




## === cell 6
def head_only_finetune_final(net, train_df, transform, epochs=2, batch_size=8):
    for p in net.backbone.parameters():
        p.requires_grad = False
    for head in [net.classifier, net.regressor, net.ordinal]:
        for p in head.parameters():
            p.requires_grad = False
    for p in net.final_regressor.parameters():
        p.requires_grad = True

    ds = TrainDataset(train_df, TRAIN_IMG_DIR, transform=transform)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=(device == "cuda"),
        drop_last=False,
    )

    opt = torch.optim.AdamW(
        [p for p in net.parameters() if p.requires_grad], lr=3e-4, weight_decay=1e-4
    )
    crit = nn.MSELoss()

    net.train()
    for ep in range(epochs):
        total_loss = 0.0
        n = 0
        for imgs, y in dl:
            imgs = imgs.to(device, non_blocking=True)
            y = torch.as_tensor(y, device=device, dtype=torch.float32)

            opt.zero_grad(set_to_none=True)
            s = net(imgs, final=True).squeeze(1)
            loss = crit(s, y)
            loss.backward()
            opt.step()

            bs = imgs.size(0)
            total_loss += float(loss.detach().cpu()) * bs
            n += bs
        print(f"finetune-final epoch {ep+1}/{epochs} - mse: {total_loss/max(n,1):.4f}")
    net.eval()
    return net


def predict_continuous_final(net, dl):
    all_ids = []
    all_scores = []
    with torch.no_grad():
        for batch_ids, batch_imgs in dl:
            batch_imgs = batch_imgs.to(device, non_blocking=True)

            s0 = net(batch_imgs, final=True).squeeze(1)
            s1 = net(torch.flip(batch_imgs, dims=[3]), final=True).squeeze(
                1
            )  # horizontal
            s2 = net(torch.flip(batch_imgs, dims=[2]), final=True).squeeze(
                1
            )  # vertical
            s3 = net(torch.flip(batch_imgs, dims=[2, 3]), final=True).squeeze(1)  # hv
            s_avg = (s0 + s1 + s2 + s3) / 4.0

            all_ids.extend([str(x) for x in batch_ids])
            all_scores.extend(s_avg.detach().cpu().numpy().astype(np.float32).tolist())
    return np.array(all_ids), np.array(all_scores, dtype=np.float32)


def apply_thresholds(scores, thr):
    thr = list(thr)
    preds = np.zeros_like(scores, dtype=np.int64)
    for t in thr:
        preds += (scores >= t).astype(np.int64)
    return preds


def fit_thresholds_by_grid(y_true, scores, initial=(0.75, 1.5, 2.5, 3.5)):
    best_thr = list(initial)
    best_k = -1e9

    y_true = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float32)

    thr = list(initial)
    for step in [0.5, 0.2, 0.1]:
        for idx in range(4):
            base = thr[idx]
            candidates = np.arange(base - 1.0, base + 1.0 + 1e-9, step)
            local_best = thr[idx]
            local_best_k = best_k
            for c in candidates:
                cand = thr.copy()
                cand[idx] = float(c)
                cand = sorted(cand)
                preds = apply_thresholds(scores, cand)
                k = cohen_kappa_score(y_true, preds, weights="quadratic")
                if k > local_best_k:
                    local_best_k = k
                    local_best = float(c)
                    best_thr = cand
                    best_k = k
            thr[idx] = local_best
            thr = sorted(thr)

    return best_thr, float(best_k)




## === cell 7
test_ds = TestDataset(test_ids, TEST_IMG_DIR, transform=transform)
test_dl = DataLoader(
    test_ds,
    batch_size=8 if device == "cuda" else 4,
    shuffle=False,
    num_workers=2,
    pin_memory=(device == "cuda"),
)

if not use_checkpoint:
    net = head_only_finetune_final(
        net,
        train_df,
        transform=transform,
        epochs=2,
        batch_size=(8 if device == "cuda" else 4),
    )

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_scores = np.zeros(len(train_df), dtype=np.float32)


class IdWrap(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        idx = str(row["id_code"])
        image_name = os.path.join(self.img_dir, f"{idx}.png")
        img = Image.open(image_name).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return idx, img


for fold, (_, val_idx) in enumerate(
    skf.split(train_df["id_code"].values, train_df["diagnosis"].values), start=1
):
    val_df = train_df.iloc[val_idx].reset_index(drop=True)
    val_wrap = IdWrap(val_df, TRAIN_IMG_DIR, transform=transform)
    val_wrap_dl = DataLoader(
        val_wrap,
        batch_size=8 if device == "cuda" else 4,
        shuffle=False,
        num_workers=2,
        pin_memory=(device == "cuda"),
    )
    net.eval()
    val_ids, val_s = predict_continuous_final(net, val_wrap_dl)

    id2pos = {str(train_df.iloc[i]["id_code"]): i for i in val_idx}
    for vid, vs in zip(val_ids.tolist(), val_s.tolist()):
        oof_scores[id2pos[str(vid)]] = float(vs)

y_all = train_df["diagnosis"].astype(int).values
thr_fit, k_oof = fit_thresholds_by_grid(
    y_all, oof_scores, initial=(0.75, 1.5, 2.5, 3.5)
)
print("Fitted thresholds (OOF):", thr_fit, "oof_kappa:", k_oof)



## === cell 8
test_ids_out, test_scores = predict_continuous_final(net, test_dl)
test_preds = apply_thresholds(test_scores, thr_fit).astype(int)

pred_map = {str(i): int(p) for i, p in zip(test_ids_out.tolist(), test_preds.tolist())}

test_df_out = test_df.copy()
test_df_out["diagnosis"] = [
    pred_map.get(i, 0) for i in test_df_out["id_code"].astype(str).tolist()
]
test_df_out["diagnosis"] = test_df_out["diagnosis"].astype(int)

test_df_out.to_csv("submission.csv", index=False)
print(test_df_out.head())
print("Wrote submission.csv with shape:", test_df_out.shape)
print("diagnosis value counts:\n", test_df_out["diagnosis"].value_counts().sort_index())
