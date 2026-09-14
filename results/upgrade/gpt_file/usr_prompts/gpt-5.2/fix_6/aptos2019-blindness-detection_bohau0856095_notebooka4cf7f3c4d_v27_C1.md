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

0.8985926606060796

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I (1) remove the notebook-only `pip install` cell and instead rely on the already-installed `timm` package to avoid a missing `../input/weights/` dependency. Then I (2) fix device selection so the code runs on Kaggle CPU-only environments by using `cuda` only if available, and (3) make inference robust by batching with a `Dataset/DataLoader` and ensuring `trim()` always returns an image (it currently returns `None` sometimes). Finally, because the provided pretrained weight file path is missing, I keep the exact model architecture but fall back to a safe, deterministic baseline prediction (all zeros) that produces a valid non-empty `submission.csv` in the required format.'
- What this solution (achieved 0.03856) has done: 'Your current 0.0 score is coming from the “no weights found → predict all zeros” fallback, so the smallest legitimate move toward the target is to load a real pretrained backbone (same architecture) from timm and then calibrate the 4 thresholds on a held-out validation split using quadratic weighted kappa. This preserves your model definition and inference semantics (regression output → thresholding to classes), but replaces the missing external `.pkl` dependency with an available pretrained initialization and uses data-driven thresholds instead of fixed constants. I’m also making prediction deterministic and ensuring the submission ordering exactly matches `test.csv`. These changes should substantially increase QWK versus the all-zeros submission while keeping the core logic intact.'
- What this solution (achieved 0.04786) has done: 'Your score gap to the target is large, so the most likely issue is that you’re doing inference with essentially untrained heads (classifier/regressor/ordinal), even though the backbone is pretrained; this collapses predictions and keeps QWK low. I keep your exact architecture and regression→thresholding logic, but (1) load ImageNet weights directly into your `ThreeStage_Model` backbone via `timm.create_model(..., pretrained=True, num_classes=0)` to avoid any classifier-shape mismatches, and (2) use the model’s `final=True` regressor at inference time (it exists in your code and uses more information than `r_out`) while still tuning the same 4 thresholds on a held-out split with QWK. I also make the validation labels alignment explicit by indexing `train_va` with the returned `va_ids` to avoid any accidental order mismatch. These are minimal, metric-aligned changes that should move QWK substantially upward toward the target without changing the training approach (there is none) or the model definition.'
- What this solution (achieved 0.0) has done: 'Your current score is far below the target, so we need a legitimate performance lift while keeping your same model and regression→thresholding approach. The biggest issue is that you never actually use the `backbone` argument (so your pretrained timm backbone isn’t guaranteed to match your modified pooling), and your EfficientNet is producing features in a way that can be inconsistent with your GeM + “1000-dim” heads. I minimally fix `ThreeStage_Model` to build the backbone via `timm.create_model(..., num_classes=0)` and feed extracted features into your existing heads unchanged, then load ImageNet weights directly through timm (no external pkl). I also use timm’s recommended normalization/config for the chosen backbone (instead of hard-coded mean/std) to better match pretrained weights, which should improve validation QWK and thus public score without changing the core inference semantics.'
- What this solution (achieved 0.0) has done: 'Your 0.0 public score strongly suggests the submission is effectively predicting a single class (or nearly so), which can happen because your “final_regressor” head is randomly initialized (never trained) and you’re currently using it (`use_final=True`) for both validation threshold-tuning and test inference. To move toward the target with minimal core-logic change, I switch inference back to the regressor branch (`r_out`), which is still untrained but is simpler/less noisy than a random “final” fusion head, and I keep the same regression→thresholding pipeline. I also tune thresholds using out-of-fold predictions over the full training set (still no training loop added) to make threshold calibration less brittle than a single split, which should improve QWK without changing your architecture or loss/training approach. All I/O paths and the submission schema remain unchanged, and it still write a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms
from PIL import Image, ImageChops

import timm
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import cohen_kappa_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)




## === cell 1
threshold = [0.7, 1.5, 2.5, 3.5]


def regress2class(out, thr=None):
    thr = threshold if thr is None else thr
    prediction = torch.zeros(out.size(0), dtype=torch.long)
    for i in range(4):
        prediction += (out >= thr[i]).to(torch.long)
    return prediction


def apply_thresholds_np(y_pred_cont, thr):
    y_pred_cont = np.asarray(y_pred_cont, dtype=np.float32)
    pred = np.zeros_like(y_pred_cont, dtype=np.int64)
    for t in thr:
        pred += (y_pred_cont >= t).astype(np.int64)
    return pred


def tune_thresholds_qwk(y_true, y_pred_cont, init_thr=None, n_iter=3):
    """
    Coordinate descent over 4 thresholds to maximize quadratic weighted kappa on a validation set.
    Minimal post-processing change that aligns predictions to the evaluation metric.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred_cont = np.asarray(y_pred_cont, dtype=np.float32)

    thr = np.array(
        init_thr if init_thr is not None else [0.7, 1.5, 2.5, 3.5], dtype=np.float32
    )

    for _ in range(n_iter):
        for i in range(4):
            lo = 0.0 if i == 0 else float(thr[i - 1] + 0.05)
            hi = 4.5 if i == 3 else float(thr[i + 1] - 0.05)
            if hi <= lo:
                continue

            grid = np.linspace(lo, hi, 41, dtype=np.float32)
            best_t = float(thr[i])
            best_k = -1e9
            for t in grid:
                cand = thr.copy()
                cand[i] = t
                pred = apply_thresholds_np(y_pred_cont, cand)
                k = cohen_kappa_score(y_true, pred, weights="quadratic")
                if k > best_k:
                    best_k = k
                    best_t = float(t)
            thr[i] = best_t

    thr = np.maximum.accumulate(thr)
    for i in range(1, 4):
        if thr[i] <= thr[i - 1]:
            thr[i] = min(4.5, thr[i - 1] + 0.05)

    return [float(x) for x in thr]




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

        if backbone is None:
            self.backbone = timm.create_model(
                "tf_efficientnet_b4_ns",
                pretrained=False,
                num_classes=0,  # return features
                global_pool="",  # keep spatial, we apply GeM ourselves
            )
        else:
            self.backbone = backbone

        self.global_pool = GeM(flatten=True)

        feat_dim = getattr(self.backbone, "num_features", 1000)

        self.classifier = nn.Sequential(
            nn.SiLU(),
            nn.Linear(feat_dim, 500),
            nn.SiLU(),
            nn.Linear(500, 5),
        )

        self.regressor = nn.Sequential(
            nn.SiLU(),
            nn.Linear(feat_dim, 500),
            nn.SiLU(),
            nn.Linear(500, 1),
        )

        self.ordinal = nn.Sequential(
            nn.SiLU(),
            nn.Linear(feat_dim, 500),
            nn.SiLU(),
            nn.Linear(500, 4),
        )

        self.final_regressor = nn.Sequential(
            nn.SiLU(),
            nn.Linear(10, 1),
        )

    def forward(self, x, final=False):
        x = self.backbone(x)  # (B,C,H,W) because global_pool=""
        x = self.global_pool(x)  # (B,C)

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
DATA_ROOT = "../input/aptos2019-blindness-detection"
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train_images")
TEST_CSV = os.path.join(DATA_ROOT, "test.csv")
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test_images")

train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

test_ids = test_df["id_code"].astype(str).values

input_size = 320

tmp = timm.create_model(
    "tf_efficientnet_b4_ns", pretrained=True, num_classes=0, global_pool=""
)
cfg = timm.data.resolve_model_data_config(tmp)
mean = cfg["mean"]
std = cfg["std"]
del tmp

transform = transforms.Compose(
    [
        trim(),
        cropTo4_3(),
        transforms.Resize((input_size * 3 // 4, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
)

WEIGHTS_PATH = "../input/weights/B4_3stage_60epoch_AdamW.pkl"

net = ThreeStage_Model().to(device)
net.eval()

weights_loaded = False
if os.path.exists(WEIGHTS_PATH):
    state = torch.load(WEIGHTS_PATH, map_location="cpu")
    net.load_state_dict(state)
    net.eval()
    weights_loaded = True
else:
    pretrained_backbone = timm.create_model(
        "tf_efficientnet_b4_ns", pretrained=True, num_classes=0, global_pool=""
    )
    net.backbone.load_state_dict(pretrained_backbone.state_dict(), strict=True)
    net.eval()
    weights_loaded = True

print(
    f"Device: {device}, weights_loaded: {weights_loaded}, used_external_pkl: {os.path.exists(WEIGHTS_PATH)}"
)




## === cell 5
class ImageDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, with_label=False):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.with_label = with_label

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        idx = str(row["id_code"])
        image_name = os.path.join(self.img_dir, f"{idx}.png")
        img = Image.open(image_name).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        if self.with_label:
            y = int(row["diagnosis"])
            return idx, img, y
        return idx, img


def predict_regression(model, loader, use_final=False):
    model.eval()
    ids_all = []
    pred_cont = []
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                ids_batch, x, _ = batch
            else:
                ids_batch, x = batch
            x = x.to(device)

            if use_final:
                out = model(x, final=True).squeeze(1)
            else:
                _, r_out, _ = model(x)
                out = r_out.squeeze(1)

            out = out.detach().cpu().numpy().astype(np.float32)
            ids_all.extend(list(ids_batch))
            pred_cont.extend(list(out))
    return np.array(ids_all), np.array(pred_cont, dtype=np.float32)


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_pred = np.zeros(len(train_df), dtype=np.float32)

for fold, (_, va_idx) in enumerate(
    skf.split(train_df, train_df["diagnosis"].values), 1
):
    va_df = train_df.iloc[va_idx].reset_index(drop=True)
    va_ds = ImageDataset(va_df, TRAIN_IMG_DIR, transform=transform, with_label=True)
    va_dl = DataLoader(
        va_ds,
        batch_size=8,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    _, va_pred_cont = predict_regression(net, va_dl, use_final=False)
    oof_pred[va_idx] = va_pred_cont
    print(
        f"Fold {fold}/5 done. va_pred_cont mean={va_pred_cont.mean():.4f}, std={va_pred_cont.std():.4f}"
    )

y_true_full = train_df["diagnosis"].astype(int).to_numpy(dtype=np.int64)

init_thr = threshold
tuned_thr = tune_thresholds_qwk(y_true_full, oof_pred, init_thr=init_thr, n_iter=3)
oof_pred_cls = apply_thresholds_np(oof_pred, tuned_thr)
oof_kappa = cohen_kappa_score(y_true_full, oof_pred_cls, weights="quadratic")
print(f"Initial thr: {init_thr} -> Tuned thr: {tuned_thr} | OOF QWK: {oof_kappa:.5f}")

test_ds = ImageDataset(test_df, TEST_IMG_DIR, transform=transform, with_label=False)
test_dl = DataLoader(
    test_ds,
    batch_size=8,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)

pred_ids, test_pred_cont = predict_regression(net, test_dl, use_final=False)
test_pred_cls = apply_thresholds_np(test_pred_cont, tuned_thr).astype(int)

submission = pd.DataFrame({"id_code": pred_ids, "diagnosis": test_pred_cls})
submission = test_df.merge(submission, on="id_code", how="left")[
    ["id_code", "diagnosis"]
]




## === cell 6
assert submission.shape[0] == len(test_df), "Submission row count mismatch."
assert list(submission.columns) == [
    "id_code",
    "diagnosis",
], "Submission columns mismatch."
assert submission["diagnosis"].notna().all(), "Found missing predictions."

out_path = "submission.csv"
submission["diagnosis"] = submission["diagnosis"].astype(int)
submission.to_csv(out_path, index=False)
print(submission.head())
print(f"Wrote {out_path} with shape {submission.shape}")
