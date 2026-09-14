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

import torchvision.transforms as transforms
from torchvision.transforms import functional as FT
from PIL import Image, ImageChops

from sklearn.metrics import cohen_kappa_score
import timm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)



## === cell 1
threshold = [0.75, 1.5, 2.5, 3.5]


def regress2class(out: torch.Tensor):
    out = out.view(-1)
    prediction = torch.zeros(out.size(0), device=out.device, dtype=torch.float32)
    for i in range(4):
        prediction += (out >= threshold[i]).to(torch.float32)
    return prediction


def ordinal2class_prob(out: torch.Tensor):
    pred_prob = torch.zeros(out.size(0), 5, device=out.device, dtype=out.dtype)
    pred_prob[:, 0] = (1 - out[:, 0]).squeeze()
    pred_prob[:, 1] = (out[:, 0] * (1 - out[:, 1])).squeeze()
    pred_prob[:, 2] = (out[:, 1] * (1 - out[:, 2])).squeeze()
    pred_prob[:, 3] = (out[:, 2] * (1 - out[:, 3])).squeeze()
    pred_prob[:, 4] = out[:, 3].squeeze()
    return F.softmax(pred_prob, dim=1)


def regress2class_prob(out: torch.Tensor):
    out = out.view(-1)
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
    def __init__(self, pretrained_backbone: bool = False):
        super(Regressor, self).__init__()
        self.backbone = timm.models.tf_efficientnet_b5_ns(
            pretrained=pretrained_backbone
        )
        self.backbone.global_pool = GeM(flatten=True)
        self.regressor = nn.Linear(1000, 1)

    def forward(self, x):
        x = self.backbone(x)
        out = self.regressor(x)
        out = torch.sigmoid(out) * 4.5
        return out


class ThreeStage_Model(nn.Module):
    def __init__(self, backbone=None, pretrained_backbone: bool = False):
        super(ThreeStage_Model, self).__init__()

        self.backbone = timm.models.tf_efficientnet_b4_ns(
            pretrained=pretrained_backbone
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
DATA_ROOT = "../input/aptos2019-blindness-detection"
test_csv = os.path.join(DATA_ROOT, "test.csv")
test_img_dir = os.path.join(DATA_ROOT, "test_images")
train_csv = os.path.join(DATA_ROOT, "train.csv")
train_img_dir = os.path.join(DATA_ROOT, "train_images")

if not os.path.exists(test_csv):
    alt_root = "../input"
    if os.path.exists(os.path.join(alt_root, "test.csv")):
        DATA_ROOT = alt_root
        test_csv = os.path.join(DATA_ROOT, "test.csv")
        test_img_dir = os.path.join(DATA_ROOT, "test_images")
        train_csv = os.path.join(DATA_ROOT, "train.csv")
        train_img_dir = os.path.join(DATA_ROOT, "train_images")

test_ids_df = pd.read_csv(test_csv)
test_ids = np.squeeze(test_ids_df["id_code"].values)

train_df = pd.read_csv(train_csv)

input_size = 256

transform = transforms.Compose(
    [
        trim(),
        cropTo4_3(),
        transforms.Resize((input_size * 3 // 4, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.384, 0.258, 0.174], std=[0.124, 0.089, 0.094]),
    ]
)

net = ThreeStage_Model(pretrained_backbone=True)
fallback_regressor = None


def _find_checkpoint():
    preferred = [
        "../input/weights/B4_3stage_25epoch_320.pkl",
        "../input/weights/B4_3stage_25epoch_320.pth",
        "../input/weights/B4_3stage_25epoch_320.pt",
        "../input/weights/B4_3stage_25epoch_320.bin",
        "../input/weights/B4_3stage_25epoch_320.ckpt",
    ]
    for p in preferred:
        if os.path.exists(p):
            return p

    search_roots = ["../input", "/kaggle/input"]
    exts = (".pth", ".pt", ".pkl", ".bin", ".ckpt")

    good_substrings = [
        "3stage",
        "three",
        "three_stage",
        "threestage",
        "final_regressor",
        "efficientnet",
        "tf_efficientnet",
        "b4",
        "aptos",
        "blindness",
        "retina",
        "retinopathy",
        "dr",
        "kappa",
        "checkpoint",
        "weights",
        "model",
        "state_dict",
    ]

    best = None
    best_score = -1

    for root in search_roots:
        if not os.path.exists(root):
            continue
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                fn_l = fn.lower()
                if not fn_l.endswith(exts):
                    continue

                score = sum(1 for s in good_substrings if s in fn_l)

                if "b4_3stage_25epoch_320" in fn_l:
                    score += 200

                if any(
                    bad in fn_l
                    for bad in [
                        "optimizer",
                        "sched",
                        "scheduler",
                        "adam",
                        "ema",
                        "fold",
                    ]
                ):
                    score -= 3

                dir_l = dirpath.lower()
                if "weights" in dir_l or "checkpoint" in dir_l:
                    score += 2
                if "aptos" in dir_l or "blind" in dir_l or "retina" in dir_l:
                    score += 2

                if score > best_score:
                    best_score = score
                    best = os.path.join(dirpath, fn)

    if best_score < 3:
        return None
    return best


def _unwrap_state_dict(state):
    if isinstance(state, dict):
        for key in [
            "state_dict",
            "model_state_dict",
            "model",
            "net",
            "network",
            "weights",
        ]:
            if key in state and isinstance(state[key], dict):
                return state[key]
    return state


def _normalize_state_keys(sd):
    if not isinstance(sd, dict):
        return sd
    new_sd = {}
    for k, v in sd.items():
        nk = k
        for prefix in ("module.", "model.", "net.", "network."):
            if nk.startswith(prefix):
                nk = nk[len(prefix) :]
        new_sd[nk] = v
    return new_sd


def _select_best_tensor_dict(obj, model_state_keys):
    candidates = []

    def add_candidate(d, name):
        if isinstance(d, dict) and any(torch.is_tensor(v) for v in d.values()):
            sd = _normalize_state_keys(d)
            overlap = len(set(sd.keys()) & model_state_keys)
            candidates.append((overlap, name, sd))

    if isinstance(obj, dict):
        add_candidate(obj, "root")
        for k, v in obj.items():
            if isinstance(v, dict):
                add_candidate(v, f"root[{k}]")
                for k2, v2 in v.items():
                    if isinstance(v2, dict):
                        add_candidate(v2, f"root[{k}][{k2}]")

    if not candidates:
        return None, None

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][2], candidates[0][1]


weights_path = _find_checkpoint()
loaded_weights = False
if weights_path is not None and os.path.exists(weights_path):
    try:
        raw = torch.load(weights_path, map_location="cpu")
        model_keys = set(net.state_dict().keys())

        state = _unwrap_state_dict(raw)
        state = _normalize_state_keys(state)

        if not (
            isinstance(state, dict) and any(torch.is_tensor(v) for v in state.values())
        ):
            state, picked_from = _select_best_tensor_dict(raw, model_keys)
        else:
            picked_from = "unwrap_state_dict"

        if state is None:
            raise ValueError(
                "No compatible tensor state_dict found inside checkpoint object"
            )

        missing, unexpected = net.load_state_dict(state, strict=False)
        loaded_weights = True
        print(f"Loaded checkpoint from: {weights_path} (picked: {picked_from})")
        if len(missing) or len(unexpected):
            print(
                f"Note: load_state_dict strict=False, missing={len(missing)}, unexpected={len(unexpected)}"
            )
    except Exception as e:
        print(f"Failed to load checkpoint at {weights_path}: {repr(e)}")
        loaded_weights = False
        weights_path = None
else:
    print("No checkpoint found for ThreeStage model.")

net = net.to(device)
net.eval()


def _compute_three_stage_features(
    model: ThreeStage_Model, x: torch.Tensor
) -> torch.Tensor:
    c_out, r_out, o_out = model(x, final=False)
    feat = torch.cat((c_out, r_out, o_out), dim=1)
    return feat


def _quick_train_fallback_final_regressor(
    train_df: pd.DataFrame, model: ThreeStage_Model
):
    start_t = time.time()

    model.train()
    for p in model.parameters():
        p.requires_grad = False
    for p in model.final_regressor.parameters():
        p.requires_grad = True

    model.backbone.eval()
    model.classifier.eval()
    model.regressor.eval()
    model.ordinal.eval()
    model.final_regressor.train()

    df = train_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    max_train = min(
        len(df), 1600
    )  # bounded for runtime; slightly higher for better signal
    df = df.iloc[:max_train].copy()

    opt = torch.optim.Adam(model.final_regressor.parameters(), lr=2e-3)
    loss_fn = nn.SmoothL1Loss()

    epochs = 4
    bs = 12

    img_paths = [os.path.join(train_img_dir, f"{i}.png") for i in df["id_code"].values]
    labels = df["diagnosis"].astype(np.float32).values

    for ep in range(epochs):
        perm = np.random.RandomState(42 + ep).permutation(len(df))
        running = 0.0
        n = 0
        for s in range(0, len(df), bs):
            idxs = perm[s : s + bs]
            xs = []
            ys = []
            for j in idxs:
                pth = img_paths[j]
                y = labels[j]
                try:
                    img = Image.open(pth).convert("RGB")
                    x = transform(img)
                    xs.append(x)
                    ys.append(y)
                except Exception:
                    continue
            if not xs:
                continue

            xb = torch.stack(xs, dim=0).to(device)
            yb = torch.tensor(ys, device=device, dtype=torch.float32).view(-1, 1)

            with torch.no_grad():
                feat = _compute_three_stage_features(model, xb)

            opt.zero_grad(set_to_none=True)
            out = model.final_regressor(feat)
            out = torch.sigmoid(out) * 4.5
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()

            running += float(loss.item()) * xb.size(0)
            n += xb.size(0)

        if n > 0:
            print(
                f"[fallback_final_regressor] epoch {ep+1}/{epochs} loss={running/n:.4f}"
            )
        else:
            print(
                f"[fallback_final_regressor] epoch {ep+1}/{epochs} had no valid batches"
            )

        if time.time() - start_t > 480:
            break

    model.eval()
    return model


if not loaded_weights:
    net = _quick_train_fallback_final_regressor(train_df, net)

print(f"Device: {device}")
print(f"Test images dir: {test_img_dir}")
print(f"Loaded weights: {loaded_weights} ({weights_path})")
print(f"Fallback trained final_regressor: {not loaded_weights}")



## === cell 5
submission_rows = []

with torch.no_grad():
    for i, idx in enumerate(test_ids):
        if i % 50 == 0:
            print(f"Inferencing {i}/{len(test_ids)}")

        image_name = os.path.join(test_img_dir, f"{idx}.png")

        try:
            img = Image.open(image_name).convert("RGB")
            x = transform(img).unsqueeze(0).to(device)

            r_out = net(x, final=True)  # [1,1] in 0..4.5
            pred = regress2class(r_out.squeeze(1)).item()
            pred_int = int(pred)

        except Exception:
            pred_int = 0

        submission_rows.append([idx, pred_int])

submission = pd.DataFrame(submission_rows, columns=["id_code", "diagnosis"])
submission["diagnosis"] = submission["diagnosis"].astype(int)
assert len(submission) == len(test_ids_df), "Submission row count must match test.csv"



## === cell 6
out_path = "submission.csv"
submission.to_csv(out_path, index=False)
print(
    f"Wrote {out_path} with shape {submission.shape} and columns {list(submission.columns)}"
)
print(submission.head())
