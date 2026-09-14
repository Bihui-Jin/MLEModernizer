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

0.9028777099794276

# 6. Current score

0.1002

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -0.03395) has done: 'I fix the missing weights issue by loading the model checkpoint only if it exists and otherwise fall back to the model’s native pretrained weights so inference can still run end-to-end. I also fix the GPU crash by selecting `cuda` only when available and ensuring all tensors are created on the correct device (there are a couple of hard-coded `.cuda()` calls and a CPU/GPU mix bug). Finally, I make the image trimming transform always return an image (it currently returns `None` sometimes) and switch inference to `torch.no_grad()` with batched DataLoader inference to finish within the time limit and reliably write a non-empty `submission.csv` with the correct columns.'
- What this solution (achieved 0.02381) has done: 'Your score is extremely low because the inference code is using the *regression head alone* (and fixed thresholds) even though the model is a multi-head “three-stage” design; without the intended trained checkpoint, those regression outputs are essentially uncalibrated. To move the score upward toward the target while keeping the same model and inference semantics, I (1) make checkpoint loading robust to common key-prefix differences so the correct weights load when present, and (2) switch inference to use the model’s `final=True` path (the intended aggregated head) and then discretize the scalar output the same way as before. These are minimal changes that keep the architecture and dataset processing intact, but they should substantially improve agreement and thus QWK. The submission writing and row alignment logic remains unchanged.'
- What this solution (achieved 0.02381) has done: 'Your gap to the target is very large (0.02381 vs 0.90288), so the most likely issue is that you’re not actually running the intended trained weights and/or your final scalar-to-class discretization is badly miscalibrated for QWK. I keep your model and inference flow intact, but (1) make checkpoint discovery robust (so you actually load the provided competition weights if they exist anywhere under `../input/`), (2) make `load_state_dict` tolerant to minor key mismatches while still verifying that most weights loaded, and (3) calibrate only the *four thresholds* on a small validation split using the same `final=True` output (this preserves your regression→thresholding semantics but aligns it to QWK). These changes are minimal, keep architecture/loss/training untouched, and are directly aimed at lifting QWK substantially toward your target while still producing a valid `submission.csv`.'
- What this solution (achieved 0.1249) has done: 'Your current score (0.02381) is far below the target (0.90288), so we should increase performance with the smallest safe changes that keep your model/inference semantics intact. The biggest likely issue is that threshold calibration is only executed when a custom checkpoint is found; if the checkpoint isn’t found/loaded, predictions are essentially uncalibrated and QWK collapses. I (1) make checkpoint discovery more robust (search multiple likely filenames/locations) while still verifying that most weights loaded, and (2) run the same QWK threshold fitting whenever we have train data available (even if using ImageNet backbone), because this preserves your “regression→thresholding” core logic but aligns discretization to QWK. I also make `regress2class` device-agnostic (no hard-coded CPU tensor creation) to avoid subtle device/copy issues without changing outputs.'
- What this solution (achieved 0.1002) has done: 'Your score is far below target, so we should increase QWK while keeping your model and “regression→thresholding” semantics intact. The biggest low-risk win is to calibrate thresholds on a more reliable validation prediction stream: ensure deterministic inference, use more stable DataLoader settings, and (critically) run TTA with a horizontal flip during both validation-threshold fitting and test inference (this doesn’t change the model, loss, or training loop; it only averages two forward passes). I also make sure the threshold fitter sees exactly the same type of continuous predictions you use at test time (with TTA), so the learned thresholds transfer correctly. These changes are minimal and directly aimed at boosting agreement/QWK without altering architecture or training.'

# 9. Code solution

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
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import functional as FT
from PIL import Image, ImageChops

from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedShuffleSplit
import timm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    prediction = torch.zeros(out.size(0), device=out.device, dtype=torch.float32)
    for i in range(4):
        prediction += (out.data >= threshold[i]).squeeze().detach().to(out.device)
    return prediction.to("cpu")


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
        v = float(out[i].detach().cpu())
        if v < 4.0:
            l1 = int(math.floor(v))
            l2 = int(math.ceil(v))
            pred_prob[i][l1] = 1 - (v - l1)
            pred_prob[i][l2] = 1 - (l2 - v)
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
TEST_CSV = os.path.join(BASE, "test.csv")
TEST_IMG_DIR = os.path.join(BASE, "test_images")
TRAIN_CSV = os.path.join(BASE, "train.csv")
TRAIN_IMG_DIR = os.path.join(BASE, "train_images")

test_df = pd.read_csv(TEST_CSV)
test_ids = test_df["id_code"].astype(str).tolist()

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


def _find_ckpt_candidates(search_root="../input", filenames=()):
    hits = []
    for fn in filenames:
        for root, _, files in os.walk(search_root):
            if fn in files:
                hits.append(os.path.join(root, fn))
    seen = set()
    out = []
    for p in hits:
        if p not in seen:
            out.append(p)
            seen.add(p)
    return out


ckpt_candidates = _find_ckpt_candidates(
    search_root="../input",
    filenames=(
        "B4_3stage_42epoch_CLAHE.pkl",
        "B4_3stage_42epoch_CLAHE.pth",
        "B4_3stage_42epoch_CLAHE.pt",
    ),
)

loaded_custom = False
loaded_from = None

if len(ckpt_candidates) > 0:
    ckpt_path = ckpt_candidates[0]
    state = torch.load(ckpt_path, map_location="cpu")
    if (
        isinstance(state, dict)
        and "state_dict" in state
        and isinstance(state["state_dict"], dict)
    ):
        state = state["state_dict"]

    if isinstance(state, dict) and len(state) > 0:
        example_key = next(iter(state.keys()))
        if example_key.startswith("module."):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}
        example_key = next(iter(state.keys()))
        if example_key.startswith("model."):
            state = {k.replace("model.", "", 1): v for k, v in state.items()}

    missing, unexpected = net.load_state_dict(state, strict=False)
    total_keys = len(net.state_dict())
    loaded_keys = total_keys - len(missing)
    loaded_ratio = loaded_keys / max(1, total_keys)
    print(f"Checkpoint found: {ckpt_path}")
    print(
        f"load_state_dict strict=False -> missing={len(missing)} unexpected={len(unexpected)} loaded_ratio={loaded_ratio:.3f}"
    )

    if loaded_ratio < 0.90:
        raise RuntimeError(
            f"Checkpoint appears incompatible: loaded_ratio={loaded_ratio:.3f} (<0.90). "
            f"Missing sample keys: {missing[:5]}"
        )
    loaded_custom = True
    loaded_from = ckpt_path
else:
    net.backbone = timm.models.tf_efficientnet_b4_ns(pretrained=True)
    net.backbone.global_pool = GeM(flatten=True)
    print(
        "Checkpoint not found; using ImageNet-pretrained backbone only (expected lower QWK)."
    )

net = net.to(device)
net.eval()
print("Loaded custom checkpoint:", loaded_custom, "| path:", loaded_from)




## === cell 5
class TestDataset(Dataset):
    def __init__(self, ids, img_dir, tfm):
        self.ids = ids
        self.img_dir = img_dir
        self.tfm = tfm

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        idx = self.ids[i]
        path = os.path.join(self.img_dir, f"{idx}.png")
        img = Image.open(path).convert("RGB")
        img = self.tfm(img)
        return idx, img


test_ds = TestDataset(test_ids, TEST_IMG_DIR, transform)

num_workers = 4 if device.type == "cuda" else 2
test_loader = DataLoader(
    test_ds,
    batch_size=8 if device.type == "cuda" else 4,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=(device.type == "cuda"),
    persistent_workers=(num_workers > 0),
    drop_last=False,
)




## === cell 6
class TrainDataset(Dataset):
    def __init__(self, df, img_dir, tfm):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.tfm = tfm

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        idx = str(self.df.loc[i, "id_code"])
        y = int(self.df.loc[i, "diagnosis"])
        path = os.path.join(self.img_dir, f"{idx}.png")
        img = Image.open(path).convert("RGB")
        img = self.tfm(img)
        return idx, img, y


def _predict_continuous(loader, use_tta=True):
    ids_all, y_all, out_all = [], [], []
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                ids, imgs, y = batch
                y_all.append(np.asarray(y, dtype=np.int64))
            else:
                ids, imgs = batch
                y = None

            imgs = imgs.to(device, non_blocking=True)

            out1 = net(imgs, final=True).squeeze(1)  # (B,) in 0..4.5
            if use_tta:
                imgs_flip = torch.flip(imgs, dims=[3])  # horizontal flip (W dimension)
                out2 = net(imgs_flip, final=True).squeeze(1)
                out = 0.5 * (out1 + out2)
            else:
                out = out1

            out_all.append(out.detach().cpu().numpy())
            ids_all.extend(list(ids))

    x = np.concatenate(out_all) if len(out_all) else np.array([], dtype=np.float32)
    y_cat = np.concatenate(y_all) if len(y_all) else None
    return x, y_cat, ids_all


def _apply_thresholds(x, thr):
    thr = list(thr)
    pred = np.zeros_like(x, dtype=np.int64)
    for t in thr:
        pred += (x >= t).astype(np.int64)
    return pred


def _fit_thresholds_qwk(x, y, init_thr, max_iter=3):
    thr = np.array(init_thr, dtype=np.float32)

    lo, hi = 0.0, 4.5
    step_schedule = [0.25, 0.10, 0.05]

    best_thr = thr.copy()
    best_score = cohen_kappa_score(
        y, _apply_thresholds(x, best_thr), weights="quadratic"
    )

    for step in step_schedule[:max_iter]:
        improved = True
        while improved:
            improved = False
            for i in range(4):
                left = lo if i == 0 else (best_thr[i - 1] + 1e-3)
                right = hi if i == 3 else (best_thr[i + 1] - 1e-3)
                center = float(best_thr[i])
                candidates = np.array(
                    [
                        center - 2 * step,
                        center - step,
                        center,
                        center + step,
                        center + 2 * step,
                    ],
                    dtype=np.float32,
                )
                candidates = np.clip(candidates, left, right)

                local_best_t = best_thr[i]
                local_best_score = best_score
                for cand in candidates:
                    trial = best_thr.copy()
                    trial[i] = cand
                    score = cohen_kappa_score(
                        y, _apply_thresholds(x, trial), weights="quadratic"
                    )
                    if score > local_best_score + 1e-8:
                        local_best_score = score
                        local_best_t = cand

                if local_best_score > best_score + 1e-8:
                    best_thr[i] = local_best_t
                    best_score = local_best_score
                    improved = True

    return best_thr.tolist(), float(best_score)


if os.path.exists(TRAIN_CSV) and os.path.isdir(TRAIN_IMG_DIR):
    train_df = pd.read_csv(TRAIN_CSV)
    y = train_df["diagnosis"].astype(int).values

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    tr_idx, va_idx = next(splitter.split(train_df, y))
    va_df = train_df.iloc[va_idx].copy()

    va_ds = TrainDataset(va_df, TRAIN_IMG_DIR, transform)
    va_loader = DataLoader(
        va_ds,
        batch_size=8 if device.type == "cuda" else 4,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
        drop_last=False,
    )

    x_va, y_va, _ = _predict_continuous(va_loader, use_tta=True)
    fitted_thr, va_qwk = _fit_thresholds_qwk(x_va, y_va, threshold, max_iter=3)
    print("Validation QWK (TTA + threshold-calibrated):", va_qwk)
    print("Old thresholds:", threshold)
    print("New thresholds:", fitted_thr)

    threshold = fitted_thr
else:
    print(
        "Skipping threshold calibration (train data unavailable). Using default thresholds:",
        threshold,
    )



## === cell 7
submission = []

with torch.no_grad():
    for b, (ids, imgs) in enumerate(test_loader):
        imgs = imgs.to(device, non_blocking=True)

        out1 = net(imgs, final=True).squeeze(1)  # (B,) scaled to 0..4.5
        imgs_flip = torch.flip(imgs, dims=[3])
        out2 = net(imgs_flip, final=True).squeeze(1)
        out = 0.5 * (out1 + out2)

        preds = regress2class(out)  # returns cpu tensor
        preds = preds.numpy().astype(int)

        for idx, p in zip(ids, preds):
            submission.append([idx, int(p)])

submission = np.array(submission, dtype=object)
print("Predicted rows:", submission.shape[0])



## === cell 8
df = pd.DataFrame(submission, columns=["id_code", "diagnosis"])
df["id_code"] = df["id_code"].astype(str)
df["diagnosis"] = df["diagnosis"].astype(int)

sample_path = os.path.join(BASE, "sample_submission.csv")
if os.path.exists(sample_path):
    samp = pd.read_csv(sample_path)
    df = samp[["id_code"]].merge(df, on="id_code", how="left")
    df["diagnosis"] = df["diagnosis"].fillna(0).astype(int)

out_path = "submission.csv"
df.to_csv(out_path, index=False)
print("Wrote:", out_path, "shape:", df.shape)
print(df.head())
