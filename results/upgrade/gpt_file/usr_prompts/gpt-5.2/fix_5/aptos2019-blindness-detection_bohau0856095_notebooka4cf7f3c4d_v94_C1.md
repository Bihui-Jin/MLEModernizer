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

0.9231639730214162

# 6. Current score

0.19588

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I (1) remove the failing `pip install` of a missing wheel and rely on the already-installed `timm`, (2) make device selection robust by falling back to CPU when CUDA isn’t available, and (3) fix the missing weights issue by loading the model checkpoint only if it exists and otherwise running untrained inference (so the notebook still completes and writes a valid `submission.csv`). I also fix small logic bugs that can crash transforms (a `trim()` that sometimes returns `None`, and a string `is` comparison) and correct tensor device handling in the probability helpers. Finally, I ensure the submission is always created with the required columns and non-empty rows aligned to `test.csv`.'
- What this solution (achieved -0.00596) has done: 'Your current 0.0 score is because the model is running with random weights (the checkpoint path `/kaggle/input/weights/...` doesn’t exist in this environment), so predictions are essentially random. The smallest legitimate way to move toward your target is to keep the exact same model/transform/inference pipeline, but add a lightweight on-the-fly training step using the provided `train.csv` + `train_images/` (no architecture or loss changes) and then run the same regressor-to-class thresholding for test inference. To keep runtime under 600s, the patch trains only the existing regressor head for a short fixed number of iterations while freezing the backbone, which typically gives a large jump from 0.0 without rewriting core logic. The submission format and ID alignment remain identical, and the code still writes `submission.csv` end-to-end.'
- What this solution (achieved 0.16911) has done: 'Your score is far below the target, so we should improve (not degrade) predictions with the smallest change that keeps your model and inference semantics intact. Right now you train only the regressor head using raw float labels but then discretize with fixed thresholds; this usually gives poorly calibrated cutoffs for quadratic weighted kappa. The minimal, competition-relevant fix is to tune the 4 thresholds on a held-out validation split after the quick head-only training, then use those tuned thresholds at test-time (same regressor output, same `regress2class` logic). I also switch test inference to a DataLoader (same transforms/model) to reduce overhead and keep runtime within 600s reliably.'
- What this solution (achieved 0.19588) has done: 'Your current gap to the target is large, so we should improve score with minimal, metric-aligned changes while keeping your model and regression-to-threshold logic intact. The main low-risk win is to (1) train the same regressor head on more than a tiny 512-image subset (still freezing the backbone and other heads, same loss/optimizer family), and (2) tune thresholds on a stratified validation split so all classes are represented, which usually improves quadratic weighted kappa versus a purely random split. I also make inference/training deterministic and slightly more stable by using consistent seeding per worker and disabling heavy augmentation during the quick head-only training (augmentation noise can hurt when training is very short), without changing the architecture or objective. The script still runs end-to-end and always writes a valid `submission.csv` aligned to `test.csv`.'

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
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import functional as FT
from PIL import Image, ImageChops
import cv2

from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedShuffleSplit
import timm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def seed_worker(worker_id: int):
    worker_seed = SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


g = torch.Generator()
g.manual_seed(SEED)



## === cell 1
threshold = [0.75, 1.5, 2.5, 3.5]


def regress2class(out):
    prediction = torch.zeros(out.size(0), device=out.device)
    for i in range(4):
        prediction += (out >= threshold[i]).to(prediction.dtype)
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
            l1 = int(math.floor(float(out[i].item())))
            l2 = int(math.ceil(float(out[i].item())))
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
BASE1 = "../input/aptos2019-blindness-detection"
BASE2 = "/kaggle/input/aptos2019-blindness-detection"
BASE3 = "/kaggle/data/aptos2019-blindness-detection"
BASE = BASE1 if os.path.exists(BASE1) else (BASE2 if os.path.exists(BASE2) else BASE3)

train_csv_path = os.path.join(BASE, "train.csv")
test_csv_path = os.path.join(BASE, "test.csv")
train_img_dir = os.path.join(BASE, "train_images")
test_img_dir = os.path.join(BASE, "test_images")

train_df = pd.read_csv(train_csv_path)
test_df = pd.read_csv(test_csv_path)
test_ids = test_df["id_code"].astype(str).values

print("Num train:", len(train_df), "Num test:", len(test_df))
print("Train images dir exists:", os.path.isdir(train_img_dir))
print("Test images dir exists:", os.path.isdir(test_img_dir))

input_size = 380

transform_infer = transforms.Compose(
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
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.384, 0.258, 0.174], std=[0.124, 0.089, 0.094]),
    ]
)

transform_quick_train = transform_infer

net = ThreeStage_Model()

WEIGHTS1 = "../input/weights/B4_3stage_14epoch_finetune3.pkl"
WEIGHTS2 = "/kaggle/input/weights/B4_3stage_14epoch_finetune3.pkl"
WEIGHTS3 = "/kaggle/data/weights/B4_3stage_14epoch_finetune3.pkl"
weights_path = (
    WEIGHTS1
    if os.path.exists(WEIGHTS1)
    else (WEIGHTS2 if os.path.exists(WEIGHTS2) else WEIGHTS3)
)

if os.path.exists(weights_path):
    state = torch.load(weights_path, map_location="cpu")
    net.load_state_dict(state)
    print("Loaded weights:", weights_path)
else:
    print(
        "WARNING: weights not found at",
        WEIGHTS1,
        "or",
        WEIGHTS2,
        "or",
        WEIGHTS3,
        "- will do quick head-only training to avoid random predictions.",
    )

net = net.to(device)




## === cell 5
class AptosDataset(Dataset):
    def __init__(self, df, img_dir, transform, is_test=False):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img_path = os.path.join(self.img_dir, f"{row['id_code']}.png")
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        if self.is_test:
            return row["id_code"], img
        y = float(row["diagnosis"])
        return img, torch.tensor(y, dtype=torch.float32)


def quick_train_regressor_head_if_needed(net, train_df, train_img_dir):
    if os.path.exists(weights_path):
        return

    for p in net.backbone.parameters():
        p.requires_grad = False
    for p in net.classifier.parameters():
        p.requires_grad = False
    for p in net.ordinal.parameters():
        p.requires_grad = False
    for p in net.final_regressor.parameters():
        p.requires_grad = False
    for p in net.regressor.parameters():
        p.requires_grad = True

    net.train()

    df = train_df.sample(n=min(2048, len(train_df)), random_state=SEED).reset_index(
        drop=True
    )

    ds = AptosDataset(df, train_img_dir, transform_quick_train)
    loader = DataLoader(
        ds,
        batch_size=12,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        [p for p in net.parameters() if p.requires_grad], lr=2e-3, weight_decay=1e-4
    )

    max_steps = 240  # fixed; no early stopping
    step = 0
    t0 = time.time()
    for epoch in range(50):
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).view(-1, 1)

            optimizer.zero_grad(set_to_none=True)
            _, r_out, _ = net(x)  # r_out already sigmoid*4.5
            loss = criterion(r_out, y)
            loss.backward()
            optimizer.step()

            step += 1
            if step % 40 == 0:
                print(f"train step {step}/{max_steps} loss {loss.item():.4f}")
            if step >= max_steps:
                break
        if step >= max_steps:
            break

    dt = time.time() - t0
    print(f"Quick head-only training finished: steps={step}, time={dt:.1f}s")
    net.eval()


def tune_thresholds_on_val(net, train_df, train_img_dir, max_val=512):
    global threshold

    y = train_df["diagnosis"].astype(int).values
    sss = StratifiedShuffleSplit(
        n_splits=1,
        test_size=min(max_val, max(256, int(0.15 * len(train_df)))) / len(train_df),
        random_state=SEED,
    )
    train_idx, val_idx = next(sss.split(np.zeros(len(train_df)), y))
    val_df = train_df.iloc[val_idx].reset_index(drop=True)

    val_ds = AptosDataset(val_df, train_img_dir, transform_infer)
    val_loader = DataLoader(
        val_ds,
        batch_size=16,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=g,
    )

    net.eval()
    preds = []
    ys = []
    with torch.no_grad():
        for x, yb in val_loader:
            x = x.to(device, non_blocking=True)
            _, r_out, _ = net(x)
            preds.append(r_out.squeeze(1).detach().cpu().numpy())
            ys.append(yb.numpy())
    preds = np.concatenate(preds).astype(np.float32)
    ys = np.concatenate(ys).astype(np.int64)

    def apply_thr(p, thr):
        thr = list(thr)
        out = np.zeros_like(p, dtype=np.int64)
        for t in thr:
            out += (p >= t).astype(np.int64)
        return out

    def kappa_for(thr):
        pcls = apply_thr(preds, thr)
        return cohen_kappa_score(ys, pcls, weights="quadratic")

    best_thr = list(map(float, threshold))
    best_k = kappa_for(best_thr)

    grids = [
        np.arange(0.25, 1.26, 0.05),
        np.arange(1.00, 2.26, 0.05),
        np.arange(1.75, 3.26, 0.05),
        np.arange(2.75, 4.26, 0.05),
    ]

    for _round in range(2):
        for i in range(4):
            local_best_thr = best_thr[:]
            local_best_k = best_k
            for cand in grids[i]:
                cand_thr = best_thr[:]
                cand_thr[i] = float(cand)
                if not (cand_thr[0] < cand_thr[1] < cand_thr[2] < cand_thr[3]):
                    continue
                k = kappa_for(cand_thr)
                if k > local_best_k:
                    local_best_k = k
                    local_best_thr = cand_thr
            best_thr, best_k = local_best_thr, local_best_k

    threshold = best_thr
    print(
        "Tuned thresholds:",
        threshold,
        "val_kappa:",
        float(best_k),
        "val_size:",
        len(val_df),
    )


quick_train_regressor_head_if_needed(net, train_df, train_img_dir)
tune_thresholds_on_val(net, train_df, train_img_dir)
net.eval()



## === cell 6
test_ds = AptosDataset(test_df, test_img_dir, transform_infer, is_test=True)
test_loader = DataLoader(
    test_ds,
    batch_size=16,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
    drop_last=False,
    worker_init_fn=seed_worker,
    generator=g,
)

pred_map = {}
missing_images = 0

net.eval()
with torch.no_grad():
    seen = 0
    for ids, x in test_loader:
        ids = [str(i) for i in ids]
        x = x.to(device, non_blocking=True)
        _, r_out, _ = net(x)
        pred = regress2class(r_out.squeeze(1)).detach().cpu().numpy().astype(np.int64)
        for id_code, p in zip(ids, pred):
            pred_map[id_code] = int(p)
        seen += len(ids)
        if seen % 64 == 0:
            print("Predicting", seen, "/", len(test_df))

for idx in test_ids:
    if idx not in pred_map:
        missing_images += 1
        pred_map[str(idx)] = 0

print("Missing images:", missing_images)



## === cell 7
df = pd.DataFrame({"id_code": test_ids.astype(str)})
df["diagnosis"] = [int(pred_map[str(i)]) for i in df["id_code"].values]

out_path = "submission.csv"
df[["id_code", "diagnosis"]].to_csv(out_path, index=False)
print("Wrote", out_path, "with shape", df.shape)
print(df.head())
