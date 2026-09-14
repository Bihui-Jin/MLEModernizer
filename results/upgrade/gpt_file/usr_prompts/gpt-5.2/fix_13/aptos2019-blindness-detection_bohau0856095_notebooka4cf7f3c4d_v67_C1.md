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

0.916944508715546

# 6. Current score

None

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

N/A

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
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.transforms import functional as FT
from PIL import Image, ImageChops

from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold
import timm

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True




## === cell 1
threshold = [0.75, 1.5, 2.5, 3.5]


def regress2class(out, thr=None):
    if thr is None:
        thr = threshold
    prediction = torch.zeros(out.size(0))
    for i in range(4):
        prediction += (out.data >= thr[i]).squeeze().cpu()
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
def _first_existing_path(candidates):
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None


def _find_any_weights(search_roots, exts=(".pkl", ".pth", ".pt")):
    found = []
    for root in search_roots:
        if not root or not os.path.exists(root):
            continue
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                if fn.lower().endswith(exts):
                    found.append(os.path.join(dirpath, fn))
    found.sort()
    return found[0] if found else None


def _find_preferred_weights(search_roots, exts=(".pkl", ".pth", ".pt")):
    all_ckpts = []
    for root in search_roots:
        if not root or not os.path.exists(root):
            continue
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                if fn.lower().endswith(exts):
                    all_ckpts.append(os.path.join(dirpath, fn))

    if not all_ckpts:
        return None

    def score_path(p):
        name = os.path.basename(p).lower()
        s = 0
        if "3stage" in name or "three" in name:
            s += 80
        if "b4" in name:
            s += 30
        if "b5" in name:
            s += 10
        if "aptos" in name or "blind" in name or "retina" in name:
            s += 15
        if "final" in name:
            s += 10
        if "finetune" in name or "epoch" in name:
            s += 5
        if "optimizer" in name or "sched" in name or "ema" in name:
            s -= 10
        return (-s, p)

    all_ckpts.sort(key=score_path)
    return all_ckpts[0]


DATASET_DIR = _first_existing_path(
    [
        "/kaggle/data/aptos2019-blindness-detection",
        "/kaggle/input/aptos2019-blindness-detection",
        "../input/aptos2019-blindness-detection",
        "../data/aptos2019-blindness-detection",
    ]
)
if DATASET_DIR is None:
    raise FileNotFoundError(
        "Could not locate aptos2019-blindness-detection dataset directory."
    )

TRAIN_CSV_PATH = os.path.join(DATASET_DIR, "train.csv")
TRAIN_IMG_DIR = os.path.join(DATASET_DIR, "train_images")
TEST_CSV_PATH = os.path.join(DATASET_DIR, "test.csv")
TEST_IMG_DIR = os.path.join(DATASET_DIR, "test_images")

WEIGHTS_PATH = "../input/weights/B4_3stage_5epoch_finetune.pkl"

if not os.path.exists(TEST_CSV_PATH):
    raise FileNotFoundError(f"Missing test.csv at: {TEST_CSV_PATH}")
if not os.path.isdir(TEST_IMG_DIR):
    raise FileNotFoundError(f"Missing test image directory at: {TEST_IMG_DIR}")

test_ids = pd.read_csv(TEST_CSV_PATH)["id_code"].astype(str).values

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


def _strip_module_prefix(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict
    keys = list(state_dict.keys())
    if len(keys) > 0 and all(k.startswith("module.") for k in keys):
        return {k[len("module.") :]: v for k, v in state_dict.items()}
    return state_dict


def _unwrap_checkpoint(obj):
    if isinstance(obj, dict):
        for k in ("state_dict", "model", "model_state_dict", "net", "weights"):
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
    return obj


def _checkpoint_to_state_dict(obj):
    if isinstance(obj, nn.Module):
        return obj.state_dict()
    return obj


net = ThreeStage_Model()
loaded_custom_weights = False

weights_to_try = []
if os.path.exists(WEIGHTS_PATH):
    weights_to_try.append(WEIGHTS_PATH)
else:
    auto_weight = _find_preferred_weights(
        search_roots=[
            "/kaggle/input",
            "/kaggle/data",
            "/kaggle/working",
            "../input",
            "../working",
            "../data",
            DATASET_DIR,
            ".",
        ]
    )
    if auto_weight is None:
        auto_weight = _find_any_weights(
            search_roots=[
                "/kaggle/input",
                "/kaggle/data",
                "/kaggle/working",
                "../input",
                "../working",
                "../data",
                DATASET_DIR,
                ".",
            ]
        )
    if auto_weight is not None:
        weights_to_try.append(auto_weight)

if weights_to_try:
    wpath = weights_to_try[0]
    state = torch.load(wpath, map_location="cpu")
    state = _unwrap_checkpoint(state)
    state = _checkpoint_to_state_dict(state)
    state = _strip_module_prefix(state)
    if not isinstance(state, dict):
        raise TypeError(
            f"Loaded checkpoint is not a state_dict-like object: {type(state)}"
        )
    missing, unexpected = net.load_state_dict(state, strict=False)
    print(f"Loaded weights: {wpath}")
    print(f"Missing keys: {len(missing)}; Unexpected keys: {len(unexpected)}")
    loaded_custom_weights = True
else:
    print(
        "WARNING: No custom checkpoint found. Using timm pretrained EfficientNet-B4 backbone weights."
    )
    net.backbone = timm.models.tf_efficientnet_b4_ns(pretrained=True)
    net.backbone.global_pool = GeM(flatten=True)
    loaded_custom_weights = False

net = net.to(device)
net.eval()




## === cell 5
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
        if not os.path.exists(image_name):
            raise FileNotFoundError(f"Missing image: {image_name}")
        img = Image.open(image_name).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return id_code, img


class TrainDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True).copy()
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        id_code = str(self.df.loc[idx, "id_code"])
        y = int(self.df.loc[idx, "diagnosis"])
        image_name = os.path.join(self.img_dir, f"{id_code}.png")
        if not os.path.exists(image_name):
            raise FileNotFoundError(f"Missing train image: {image_name}")
        img = Image.open(image_name).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, y


def _collate_test(batch):
    ids = [b[0] for b in batch]
    imgs = torch.stack([b[1] for b in batch], dim=0)
    return ids, imgs


def _predict_regression(loader):
    preds = []
    with torch.inference_mode():
        for batch in loader:
            if (
                isinstance(batch, (list, tuple))
                and len(batch) == 2
                and isinstance(batch[0], torch.Tensor)
            ):
                imgs = batch[0]
            else:
                imgs = batch[1]
            imgs = imgs.to(device, non_blocking=True)
            r_out = net(imgs, final=True).view(-1)
            preds.append(r_out.detach().float().cpu().numpy())
    return np.concatenate(preds, axis=0)


def _apply_thresholds(preds_continuous, thr):
    thr = list(map(float, thr))
    out = torch.tensor(preds_continuous, dtype=torch.float32)
    cls = regress2class(out, thr=thr).numpy().astype(int)
    return cls


def _tune_thresholds_qwk(y_true, y_pred_cont, init_thr=None, max_iter=12):
    if init_thr is None:
        init_thr = [0.75, 1.5, 2.5, 3.5]
    thr = np.array(init_thr, dtype=np.float64)

    def qwk_for(thr_vec):
        y_pred = _apply_thresholds(y_pred_cont, thr_vec)
        return cohen_kappa_score(y_true, y_pred, weights="quadratic")

    best = qwk_for(thr)
    step = 0.25
    for _ in range(max_iter):
        improved = False
        for i in range(4):
            candidates = []
            for delta in (-step, 0.0, step):
                cand = thr.copy()
                cand[i] += delta
                if not (0.0 <= cand[0] < cand[1] < cand[2] < cand[3] <= 4.5):
                    continue
                candidates.append(cand)
            for cand in candidates:
                score = qwk_for(cand)
                if score > best + 1e-7:
                    thr, best = cand, score
                    improved = True
        if not improved:
            step *= 0.5
            if step < 0.02:
                break
    return thr.tolist(), float(best)




## === cell 6
batch_size = 8 if torch.cuda.is_available() else 4
num_workers = 2

tuned_thresholds = None

if (
    loaded_custom_weights
    and os.path.exists(TRAIN_CSV_PATH)
    and os.path.isdir(TRAIN_IMG_DIR)
):
    try:
        train_df = pd.read_csv(TRAIN_CSV_PATH)[["id_code", "diagnosis"]].copy()
        train_df["id_code"] = train_df["id_code"].astype(str)
        y_all = train_df["diagnosis"].values.astype(int)

        n_splits = 5
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        oof_pred_cont = np.zeros(len(train_df), dtype=np.float32)

        for fold, (_, val_idx) in enumerate(
            skf.split(train_df["id_code"].values, y_all), start=1
        ):
            val_df = train_df.iloc[val_idx].reset_index(drop=True)
            val_ds = TrainDataset(val_df, TRAIN_IMG_DIR, transform=transform)
            val_loader = DataLoader(
                val_ds,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available(),
                drop_last=False,
            )
            oof_pred_cont[val_idx] = _predict_regression(val_loader)
            print(f"OOF fold {fold}/{n_splits}: predicted {len(val_idx)} images")

        tuned_thresholds, oof_qwk = _tune_thresholds_qwk(
            y_all, oof_pred_cont, init_thr=threshold, max_iter=12
        )
        print("Tuned thresholds (OOF):", tuned_thresholds)
        print("OOF QWK with tuned thresholds:", oof_qwk)
    except Exception as e:
        tuned_thresholds = None
        print("WARNING: OOF threshold tuning skipped due to error:", repr(e))
else:
    print("Skipping threshold tuning (requires loaded custom weights + train data).")

test_ds = TestDataset(test_ids, TEST_IMG_DIR, transform=transform)
test_loader = DataLoader(
    test_ds,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    drop_last=False,
    collate_fn=_collate_test,
)

thr_to_use = tuned_thresholds if tuned_thresholds is not None else threshold

submission_rows = []
with torch.inference_mode():
    for id_codes, imgs in test_loader:
        imgs = imgs.to(device, non_blocking=True)
        r_out = net(imgs, final=True).view(-1)
        preds = regress2class(r_out, thr=thr_to_use)
        preds = preds.numpy().astype(int)
        preds = np.clip(preds, 0, 4).tolist()
        for i, idc in enumerate(id_codes):
            submission_rows.append([str(idc), int(preds[i])])

df = pd.DataFrame(submission_rows, columns=["id_code", "diagnosis"])
if df.empty:
    raise RuntimeError(
        "Submission DataFrame is empty; inference did not produce any rows."
    )

df["id_code"] = df["id_code"].astype(str)
df = df.set_index("id_code").loc[pd.Index(test_ids, dtype=str)].reset_index()

df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", df.shape)
print(df.head())
print("Custom weights loaded:", loaded_custom_weights)
print("Used thresholds:", thr_to_use)
