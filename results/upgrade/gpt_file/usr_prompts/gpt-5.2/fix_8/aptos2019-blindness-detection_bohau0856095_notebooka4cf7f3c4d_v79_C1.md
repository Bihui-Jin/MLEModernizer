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

0.922081949172175

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I make the code robust to the Kaggle environment by (1) removing the unnecessary pip install of a missing local timm wheel, (2) selecting CPU when no GPU is available, and (3) fixing the image transform bug where `trim()` can return `None`. Since the pretrained weights file is missing, I load weights only if present; otherwise the model still run end-to-end and write a non-empty `submission.csv`. I also fix a small logic issue in `ordinal2class_prob` (mixing CPU/GPU tensors) to prevent potential runtime errors if you later enable that path. Finally, I ensure the submission is aligned with `sample_submission.csv`’s ids and always writes the correct columns.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is consistent with the model running on **randomly initialized weights** (the weights path points to a non-existent dataset folder), so the smallest meaningful improvement is to correctly locate and load the pretrained `.pkl` if it exists somewhere under `../input/`. I keep the same model, transforms, and prediction logic, and only add a lightweight weights auto-discovery step plus strict `state_dict` handling to avoid silent key mismatches. This should move the score upward toward your target without changing the modeling approach. The submission writing/alignment logic stays the same, ensuring a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is consistent with running the EfficientNet model with random weights because the checkpoint isn’t actually present under `../input/` in this environment. The smallest change that should move the score up toward your target is to (1) automatically search for the intended `.pkl` under both `../input` and `/kaggle/data` (your filesystem shows the dataset there), and (2) load the checkpoint more robustly by unwrapping common key formats (`state_dict`, `model`, `net`) and stripping `module.` prefixes. This keeps the exact same model, transforms, and prediction logic; it only increases the chance that the intended pretrained weights get applied. The script still writes a valid `submission.csv` aligned to `sample_submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 is most consistent with the model still running with random weights (or weights loading with major key mismatches), which yields near-random class predictions under fixed thresholds. I keep the exact same model, transforms, and inference path, but make the checkpoint loading more robust by (1) searching for any `.pkl/.pth/.pt` file containing the expected stem, and (2) selecting the best-matching state_dict by overlap with the model’s keys before loading. This is a minimal change focused purely on ensuring the intended pretrained weights actually get applied, which should increase QWK toward your target without changing evaluation semantics. The submission writing and id alignment remain unchanged.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score strongly suggests the intended checkpoint still isn’t being loaded (random-init predictions are essentially uncorrelated with labels). I keep the exact same model and inference logic, but make the checkpoint discovery more permissive (don’t require the filename stem) and make loading more robust by auto-selecting the checkpoint whose `state_dict` has the highest key-overlap with the model. To avoid accidentally picking the wrong model entirely, I only accept checkpoints with a meaningful overlap ratio and prefer ones whose filenames contain `b4`/`3stage`/`aptos`. This is the smallest change that should move QWK upward toward your target without altering architecture or prediction semantics, and it still always writes a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is still most consistent with the checkpoint not being loaded (or loading the wrong file with low key overlap), so the smallest change to move toward the target is to make checkpoint discovery less restrictive and select the best matching state_dict more reliably. I keep the exact same model, transforms, and inference (still using `r_out` + `regress2class` thresholds), but (1) broaden the search to include common checkpoint filenames (not only those containing specific tokens), and (2) prefer checkpoints with both high key-overlap and a filename hint, with a stricter minimum overlap ratio to avoid loading unrelated models. This should materially increase the chance that the intended pretrained weights are applied, which should increase QWK toward your target. Submission writing and id alignment remain unchanged and still produce a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is still most consistent with the checkpoint not being loaded (so the model predicts essentially random classes). I keep the exact same model and inference path, but make the checkpoint selection/loading stricter and more targeted: prefer checkpoints whose tensor shapes match the model parameters (not just key names), and if multiple are viable, pick the one with the best combined (shape-match + key-overlap + filename hint) score. This is a minimal change that increases the probability the intended EfficientNet-B4 3-stage weights actually get applied, which should raise QWK toward your target. Submission writing stays identical and still always produce a valid `submission.csv`.'

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
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import functional as FT
from PIL import Image, ImageChops

from sklearn.metrics import cohen_kappa_score
import timm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)



## === cell 1
threshold = [0.75, 1.5, 2.5, 3.5]


def regress2class(out):
    prediction = torch.zeros(out.size(0))
    for i in range(4):
        prediction += (out.data >= threshold[i]).squeeze().cpu()
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
DATA_ROOT = "../input/aptos2019-blindness-detection"
TEST_CSV = os.path.join(DATA_ROOT, "test.csv")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test_images")

if not os.path.exists(TEST_CSV):
    alt_root = "/kaggle/data/aptos2019-blindness-detection"
    alt_test = os.path.join(alt_root, "test.csv")
    alt_sample = os.path.join(alt_root, "sample_submission.csv")
    alt_imgdir = os.path.join(alt_root, "test_images")
    if (
        os.path.exists(alt_test)
        and os.path.exists(alt_sample)
        and os.path.isdir(alt_imgdir)
    ):
        DATA_ROOT = alt_root
        TEST_CSV = alt_test
        SAMPLE_SUB = alt_sample
        TEST_IMG_DIR = alt_imgdir

test_df = pd.read_csv(TEST_CSV)
test_ids = test_df["id_code"].values

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


def _unwrap_state_dict(obj):
    if isinstance(obj, dict):
        for k in ["state_dict", "model", "net", "model_state_dict"]:
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
    return obj


def _strip_module_prefix(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict
    if not any(
        isinstance(k, str) and k.startswith("module.") for k in state_dict.keys()
    ):
        return state_dict
    return {k.replace("module.", "", 1): v for k, v in state_dict.items()}


def _score_state_dict_key_overlap(state_dict, model_state_keys):
    if not isinstance(state_dict, dict):
        return -1
    overlap = 0
    for k in state_dict.keys():
        if isinstance(k, str) and k in model_state_keys:
            overlap += 1
    return overlap


def _score_state_dict_shape_match(state_dict, model_state_dict):
    if not isinstance(state_dict, dict):
        return -1, -1
    matched = 0
    considered = 0
    for k, v in state_dict.items():
        if not isinstance(k, str):
            continue
        if k in model_state_dict:
            considered += 1
            mv = model_state_dict[k]
            try:
                if (
                    hasattr(v, "shape")
                    and hasattr(mv, "shape")
                    and tuple(v.shape) == tuple(mv.shape)
                ):
                    matched += 1
            except Exception:
                pass
    return matched, considered


WEIGHTS_STEM = "B4_3stage_3epoch_finetune2"
SEARCH_ROOTS = ["../input", "/kaggle/data"]
CKPT_EXTS = (".pkl", ".pth", ".pt")


def _filename_preference_score(path):
    p = os.path.basename(path).lower()
    score = 0
    for tok, w in [
        ("b4", 5),
        ("efficientnet", 2),
        ("3stage", 6),
        ("three", 2),
        ("aptos", 2),
        ("blind", 1),
        ("kappa", 1),
        ("finetune", 2),
        ("epoch", 1),
        ("fold", 1),
    ]:
        if tok in p:
            score += w
    if WEIGHTS_STEM.lower() in p:
        score += 20
    return score


preferred = [
    os.path.join("../input/weights", WEIGHTS_STEM + ".pkl"),
    os.path.join(DATA_ROOT, WEIGHTS_STEM + ".pkl"),
    os.path.join("../input", WEIGHTS_STEM + ".pkl"),
    os.path.join("/kaggle/data", WEIGHTS_STEM + ".pkl"),
]

candidate_ckpts = [p for p in preferred if os.path.exists(p)]

for root in SEARCH_ROOTS:
    if not os.path.isdir(root):
        continue
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            lfn = fn.lower()
            if not lfn.endswith(CKPT_EXTS):
                continue
            candidate_ckpts.append(os.path.join(dirpath, fn))

seen = set()
candidate_ckpts = [p for p in candidate_ckpts if not (p in seen or seen.add(p))]

loaded = False
if len(candidate_ckpts) > 0:
    model_sd = net.state_dict()
    model_keys = set(model_sd.keys())
    model_nkeys = len(model_keys)

    best_path, best_state = None, None
    best_shape_ratio = -1.0
    best_overlap = -1
    best_pref = -1

    for p in candidate_ckpts:
        try:
            obj = torch.load(p, map_location="cpu")
            sd = _strip_module_prefix(_unwrap_state_dict(obj))
            if not isinstance(sd, dict):
                continue

            overlap = _score_state_dict_key_overlap(sd, model_keys)
            shape_matched, shape_considered = _score_state_dict_shape_match(
                sd, model_sd
            )
            shape_ratio = shape_matched / max(1, shape_considered)

            pref = _filename_preference_score(p)

            if (
                (shape_ratio > best_shape_ratio)
                or (shape_ratio == best_shape_ratio and overlap > best_overlap)
                or (
                    shape_ratio == best_shape_ratio
                    and overlap == best_overlap
                    and pref > best_pref
                )
            ):
                best_shape_ratio = shape_ratio
                best_overlap = overlap
                best_pref = pref
                best_path = p
                best_state = sd
        except Exception:
            continue

    if best_path is not None and isinstance(best_state, dict):
        overlap_ratio = best_overlap / max(1, model_nkeys)

        if best_shape_ratio >= 0.95 and overlap_ratio >= 0.80:
            missing, unexpected = net.load_state_dict(best_state, strict=False)
            print(f"Loaded weights from: {best_path}")
            print(
                f"Shape-match ratio (among overlapping keys): {best_shape_ratio:.3f} | "
                f"Key overlap with model: {best_overlap} / {model_nkeys} (ratio={overlap_ratio:.3f})"
            )
            if len(missing) or len(unexpected):
                print(
                    f"Note: load_state_dict strict=False, missing={len(missing)}, unexpected={len(unexpected)}"
                )
            loaded = True
        else:
            print(
                f"Found ckpt candidates but best match too low: {best_path} "
                f"shape_ratio={best_shape_ratio:.3f} overlap={best_overlap}/{model_nkeys} ratio={overlap_ratio:.3f}"
            )

if not loaded:
    print(
        f"WARNING: No usable weights found under {SEARCH_ROOTS}. Running random-init (likely ~0 QWK)."
    )

net = net.to(device)
net.eval()



## === cell 5
submission = []
with torch.no_grad():
    for i, idx in enumerate(test_ids):
        image_name = os.path.join(TEST_IMG_DIR, f"{idx}.png")
        img = Image.open(image_name).convert("RGB")
        img = transform(img).unsqueeze(0).to(device)

        _, r_out, _ = net(img)
        pred = regress2class(r_out.data.squeeze(1))
        submission.append([idx, int(pred.item())])

submission = np.array(submission, dtype=object)



## === cell 6
sub_df = pd.read_csv(SAMPLE_SUB)
pred_df = pd.DataFrame(submission, columns=["id_code", "diagnosis"])
pred_df["diagnosis"] = pred_df["diagnosis"].astype(int)

sub_df = sub_df[["id_code"]].merge(pred_df, on="id_code", how="left")
sub_df["diagnosis"] = sub_df["diagnosis"].fillna(0).astype(int)

sub_df.to_csv("submission.csv", index=False)
print(sub_df.head())
print("Wrote submission.csv with shape:", sub_df.shape)
