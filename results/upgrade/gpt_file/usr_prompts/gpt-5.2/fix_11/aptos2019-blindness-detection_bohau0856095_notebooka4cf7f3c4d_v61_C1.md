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

0.9075007834368836

# 6. Current score

0.01438

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the missing weights path by discovering the available weight file(s) under `../input/weights` (and gracefully falling back to an untrained model if none exist, so a CSV is still produced). I also fix the CUDA crash by selecting `cuda` only when available and ensuring all tensors are created on the correct device (removing hard-coded `.cuda()` usage). To prevent an empty submission, I correct the test id loading to a 1D list of strings and add robust image-path resolution for both possible dataset folder layouts. Finally, I keep your model and prediction logic unchanged and only adjust I/O/device handling so it runs end-to-end and writes `submission.csv`.'
- What this solution (achieved 0.01221) has done: 'Your current 0.0 score is consistent with running inference using randomly initialized weights (your own log already warns that no weights are found), so the smallest score-improving change is to load a valid pretrained checkpoint from the competition dataset itself (via `timm`’s pretrained weights) instead of relying on a missing `../input/weights` folder. This keeps your exact model architecture, transforms, and prediction logic intact, but makes the backbone non-random so predictions become meaningful and the kappa should move sharply upward toward the target band. I also add a deterministic path selection for the dataset root and a strict-but-safe state_dict loader that won’t silently fall back to random when key prefixes differ. The script still writes a valid `submission.csv` with the required columns and row alignment.'
- What this solution (achieved -0.01524) has done: 'Your current score is far below the target, and the main reason is that you’re effectively running a model with randomly initialized heads (only the backbone is pretrained), which produces near-random ordinal outputs. To move the score up sharply without changing your model/training logic, I (1) add a tiny “self-calibration” step that fits the 4 regression-to-class thresholds on the training set predictions (using the same forward path you already use for test), and then (2) apply those calibrated thresholds at inference. This preserves your architecture, transforms, and prediction semantics (still regressor→thresholds→class), but aligns the discretization to the model’s output distribution, which is directly relevant for quadratic weighted kappa. I also make dataset root resolution include `/kaggle/data/input/...` to match your provided filesystem so the code reliably finds train/test images and CSVs.'
- What this solution (achieved -0.21311) has done: 'Your current score is extremely far below the target (higher-is-better), so we need a legitimate, minimal change that meaningfully improves kappa without changing the model or training loop. The biggest issue is that you are predicting classes from the regressor output using fixed/weak thresholds, while you already have an ordinal head that is better aligned with the 5-level ordered labels. I keep the same network and inference flow, but calibrate the 4 class thresholds on the training set using the model’s own ordinal cumulative probabilities (convert ordinal→expected score in [0,4]) and then apply those calibrated thresholds to test predictions. This preserves evaluation semantics (regression-like output → thresholds → class) while making the discretization consistent with the model’s ordinal output distribution, which should move kappa sharply upward toward the target band.'
- What this solution (achieved -0.21311) has done: 'Your score is far below the target, so we should improve predictive signal without changing the model/training loop. The biggest low-risk issue is that `ordinal2class_prob()` is mathematically wrong (it applies a softmax to values that are already probabilities and don’t sum to 1), which makes any ordinal-based score/threshold calibration unreliable; fixing it keeps the same semantics but makes outputs meaningful. Then, instead of using a per-image Python `sum(...).item()` (which is slow and forces scalar extraction), we compute the ordinal expected score in a vectorized, correct way and use it consistently for calibration and test inference. Finally, we keep your threshold calibration approach but make it deterministic and aligned (use the same forward outputs, no sampling/early stopping changes), which should move QWK substantially upward toward the target.'
- What this solution (achieved -0.21355) has done: 'Your negative kappa is consistent with ordinal probabilities being converted to an expected score incorrectly: summing the 4 sigmoid outputs gives a value in \[0,4\] but it is not the expected class score unless you first convert the ordinal “cumulative” outputs into a proper 5-class probability distribution. I keep your model and inference flow the same (use the ordinal head, then discretize via thresholds), but fix `_ordinal_expected_score_batch()` to compute the expected score from `ordinal2class_prob()` (which you already have) in a vectorized, mathematically consistent way. I also remove the training-image subsampling in calibration (it was actively harming threshold estimation, not a core modeling change) and calibrate using all available train images that exist, which should move the QWK sharply upward toward your target. All paths, transforms, and the submission format remain unchanged, and the script still writes `submission.csv`.'
- What this solution (achieved -0.21506) has done: 'Your current negative QWK is most consistent with using a randomly-initialized model head (no external checkpoint), plus a calibration step that can’t reliably “create” signal from an untrained head. To move the score sharply upward toward the target without changing your model/training approach, the smallest legitimate fix is to load a real pretrained checkpoint (when present) from the competition dataset itself (common in Kaggle: `../input/aptos2019-blindness-detection/...pth`), and only fall back to “timm-pretrained backbone + random heads” if none exists. I also make the checkpoint loading stricter in a safe way (try `strict=True` first, then fall back to `strict=False` with clear logging) to avoid silently loading incompatible weights that hurt performance. Finally, I keep your ordinal→expected score→thresholding pipeline intact, but speed up/standardize calibration inference via a DataLoader (no logic change) so it can use all train images within the 600s budget and produce a stable `submission.csv`.'
- What this solution (achieved 0.00095) has done: 'Your score is far below the target, so we need a legitimate improvement with minimal disruption. The biggest issue is that even after your ordinal→expected-score conversion, you still discretize using a fixed threshold scheme that’s not optimized for quadratic weighted kappa; we can keep the same prediction semantics but fit thresholds by directly maximizing QWK on out-of-fold train predictions. To avoid leakage while staying within time, I compute train predictions once, do a lightweight 5-fold threshold optimization (only 4 parameters), then apply those thresholds to test. I also ensure thresholds are actually used consistently by `regress2class` (move tensor/device handling to avoid subtle CPU/GPU mismatches), without changing your model or transforms.'
- What this solution (achieved 0.01438) has done: 'Your current score is extremely far below the target (higher-is-better), and the main minimal issue is that you are “optimizing thresholds” on the same predictions you fit them on (and not actually doing out-of-fold prediction), which tends to produce unstable/poor generalization and can tank QWK on Kaggle. I keep your exact model, transforms, and ordinal→expected-score→threshold→class semantics, but change threshold tuning to use true out-of-fold (OOF) predictions from 5-fold inference so the thresholds are fitted on unbiased predictions. Then I fit thresholds once on the full OOF set (instead of per-fold medians), and apply them to test; this is a small, directly metric-aligned change that should move the score sharply upward toward the target band. I also switch test inference to use a DataLoader (same predictions, just batched) to keep runtime within 600s without changing evaluation semantics.'
- What this solution (achieved 0.01438) has done: 'Your current score (0.01438) is far below the target (0.9075), so we should increase performance with the smallest change that directly affects QWK without changing your model or training loop. The biggest issue is that your “OOF” threshold tuning is not truly OOF: you run the same fixed model on each fold’s validation set, so folding adds noise and the coordinate-search thresholds are unstable; instead, we can tune thresholds on *all* train predictions from the same fixed model (no leakage because there is no training happening). I keep your exact architecture and ordinal→expected-score→threshold→class semantics, but replace the fold-based threshold tuning with a single full-train inference pass (batched) and then run the same QWK threshold search once. This is minimal, deterministic, and should move QWK substantially upward toward the target band while keeping runtime under the 600s budget.'

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
from sklearn.model_selection import StratifiedKFold
import timm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)



## === cell 1
threshold = [0.75, 1.5, 2.5, 3.5]


def regress2class(out):
    thr = [float(x) for x in threshold]
    out_f = out.detach().float()
    pred = torch.zeros(out_f.size(0), device=out_f.device, dtype=torch.int64)
    for t in thr:
        pred += (out_f >= t).to(torch.int64)
    return pred.cpu()


def ordinal2class_prob(out):
    pred_prob = torch.zeros(out.size(0), 5, device=out.device)
    pred_prob[:, 0] = (1 - out[:, 0]).squeeze()
    pred_prob[:, 1] = (out[:, 0] * (1 - out[:, 1])).squeeze()
    pred_prob[:, 2] = (out[:, 1] * (1 - out[:, 2])).squeeze()
    pred_prob[:, 3] = (out[:, 2] * (1 - out[:, 3])).squeeze()
    pred_prob[:, 4] = out[:, 3].squeeze()
    pred_prob = pred_prob.clamp_min(0.0)
    pred_prob = pred_prob / pred_prob.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return pred_prob


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
DATA_CANDIDATES = [
    "/kaggle/input/aptos2019-blindness-detection",
    "../input/aptos2019-blindness-detection",
    "/kaggle/input",
    "../input",
    "/kaggle/data/aptos2019-blindness-detection",
    "/kaggle/data/input/aptos2019-blindness-detection",
    "/kaggle/data/input",
    "/kaggle/data",
]
data_root = None
for p in DATA_CANDIDATES:
    if os.path.exists(os.path.join(p, "test.csv")) and (
        os.path.exists(os.path.join(p, "test_images"))
        or os.path.exists(os.path.join(p, "test_images.zip"))
    ):
        data_root = p
        break
if data_root is None:
    raise FileNotFoundError(
        "Could not locate dataset folder containing test.csv and test_images."
    )

test_csv_path = os.path.join(data_root, "test.csv")
test_img_dir = os.path.join(data_root, "test_images")
train_csv_path = os.path.join(data_root, "train.csv")
train_img_dir = os.path.join(data_root, "train_images")

test_df = pd.read_csv(test_csv_path)
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

weights_candidates = [
    "../input/weights/B4_3stage_60epoch_CLAHE.pkl",
    "/kaggle/input/weights/B4_3stage_60epoch_CLAHE.pkl",
]
weights_candidates += glob.glob("../input/weights/*.*") + glob.glob(
    "/kaggle/input/weights/*.*"
)

weights_candidates += glob.glob(os.path.join(data_root, "**", "*.pth"), recursive=True)
weights_candidates += glob.glob(os.path.join(data_root, "**", "*.pt"), recursive=True)
weights_candidates += glob.glob(os.path.join(data_root, "**", "*.pkl"), recursive=True)

weights_path = None
preferred_names = {
    "B4_3stage_60epoch_CLAHE.pkl",
    "B4_3stage_60epoch_CLAHE.pth",
    "B4_3stage_60epoch_CLAHE.pt",
}
for wp in weights_candidates:
    if os.path.isfile(wp) and (
        wp.endswith(".pkl") or wp.endswith(".pth") or wp.endswith(".pt")
    ):
        if os.path.basename(wp) in preferred_names:
            weights_path = wp
            break
        if weights_path is None:
            weights_path = wp


def _clean_state_dict(state):
    if not isinstance(state, dict):
        return None
    if "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    state = {k.replace("module.", ""): v for k, v in state.items()}
    return state


if weights_path is None:
    print(
        "INFO: No external weights found. Using timm pretrained backbone + random heads."
    )
else:
    print("Loading external weights from:", weights_path)
    state = torch.load(weights_path, map_location="cpu")
    state = _clean_state_dict(state)
    if state is None:
        raise ValueError(f"Unrecognized checkpoint format in: {weights_path}")
    try:
        net.load_state_dict(state, strict=True)
        print("Loaded checkpoint with strict=True")
    except Exception as e:
        print("WARN: strict=True failed:", repr(e))
        missing, unexpected = net.load_state_dict(state, strict=False)
        print(
            f"Loaded checkpoint with strict=False; missing keys: {len(missing)}; unexpected keys: {len(unexpected)}"
        )

net = net.to(device)
net.eval()




## === cell 5
def _resolve_image_path(img_dir, idx):
    p = os.path.join(img_dir, f"{idx}.png")
    if os.path.exists(p):
        return p
    alt = os.path.join(
        "../input/aptos2019-blindness-detection",
        os.path.basename(img_dir),
        f"{idx}.png",
    )
    if os.path.exists(alt):
        return alt
    alt2 = os.path.join(
        "/kaggle/data/input/aptos2019-blindness-detection",
        os.path.basename(img_dir),
        f"{idx}.png",
    )
    if os.path.exists(alt2):
        return alt2
    return None


def _apply_thresholds_float_to_class(x_float, thr):
    thr = list(map(float, thr))
    y = np.zeros_like(x_float, dtype=np.int64)
    for t in thr:
        y += (x_float >= t).astype(np.int64)
    return y


def _ordinal_expected_score_batch(o_out_sigmoid_4):
    prob5 = ordinal2class_prob(o_out_sigmoid_4)  # (B,5), sums to 1
    classes = torch.arange(5, device=o_out_sigmoid_4.device, dtype=prob5.dtype).view(
        1, 5
    )
    exp_score = (prob5 * classes).sum(dim=1)  # (B,)
    return exp_score


class _ImageIdDataset(Dataset):
    def __init__(self, ids, labels, img_dir, transform):
        self.ids = list(ids)
        self.labels = None if labels is None else np.asarray(labels, dtype=np.int64)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        idx = self.ids[i]
        pth = _resolve_image_path(self.img_dir, idx)
        if pth is None:
            return idx, None, (-1 if self.labels is None else int(self.labels[i]))
        img = Image.open(pth).convert("RGB")
        img = self.transform(img)
        y = -1 if self.labels is None else int(self.labels[i])
        return idx, img, y


def _collate_drop_missing(batch):
    ids, imgs, ys = [], [], []
    missing = 0
    for idx, img, y in batch:
        if img is None:
            missing += 1
            continue
        ids.append(idx)
        imgs.append(img)
        ys.append(y)
    if len(imgs) == 0:
        return [], None, np.array([], dtype=np.int64), missing
    return ids, torch.stack(imgs, 0), np.array(ys, dtype=np.int64), missing


def _fit_thresholds_by_qwk_search(y_true, p_float, init_thr=None, n_iter=3):
    y_true = np.asarray(y_true, dtype=np.int64)
    p_float = np.asarray(p_float, dtype=np.float32)

    if init_thr is None:
        thr = np.array([0.75, 1.5, 2.5, 3.5], dtype=np.float32)
    else:
        thr = np.array(init_thr, dtype=np.float32)

    def score(thr_):
        thr_ = np.sort(np.clip(np.asarray(thr_, dtype=np.float32), 0.0, 4.0))
        if not (thr_[0] < thr_[1] < thr_[2] < thr_[3]):
            return -1.0
        y_hat = _apply_thresholds_float_to_class(p_float, thr_)
        return float(cohen_kappa_score(y_true, y_hat, weights="quadratic"))

    best_thr = thr.copy()
    best_s = score(best_thr)

    steps = [0.25, 0.10, 0.05][: max(1, int(n_iter))]
    for step in steps:
        improved = True
        while improved:
            improved = False
            for j in range(4):
                for delta in (-step, step):
                    cand = best_thr.copy()
                    cand[j] += delta
                    cand = np.sort(np.clip(cand, 0.0, 4.0))
                    if not (cand[0] < cand[1] < cand[2] < cand[3]):
                        continue
                    s = score(cand)
                    if s > best_s + 1e-8:
                        best_s = s
                        best_thr = cand
                        improved = True
    return best_thr.tolist(), best_s


train_df = pd.read_csv(train_csv_path)
train_ids = train_df["id_code"].astype(str).tolist()
train_y = train_df["diagnosis"].astype(int).to_numpy()

train_ds_full = _ImageIdDataset(train_ids, train_y, train_img_dir, transform)
train_loader_full = DataLoader(
    train_ds_full,
    batch_size=16,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
    collate_fn=_collate_drop_missing,
)

train_pred = np.full(len(train_ids), np.nan, dtype=np.float32)
missing_train_images = 0

t0 = time.time()
ptr = 0
with torch.no_grad():
    for ids_b, x_b, y_b, miss in train_loader_full:
        missing_train_images += int(miss)
        if x_b is None:
            continue
        x_b = x_b.to(device)
        _, _, o_out = net(x_b)
        exp_score = (
            _ordinal_expected_score_batch(o_out)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        for k, idx in enumerate(ids_b):
            train_pred[ptr] = exp_score[k]
            ptr += 1

if missing_train_images > 0:
    ptr = 0
    with torch.no_grad():
        for i, idx in enumerate(train_ids):
            pth = _resolve_image_path(train_img_dir, idx)
            if pth is None:
                continue
            img = Image.open(pth).convert("RGB")
            x = transform(img).unsqueeze(0).to(device)
            _, _, o_out = net(x)
            train_pred[i] = float(
                _ordinal_expected_score_batch(o_out)[0].detach().cpu()
            )

print(
    "Train prediction done: total",
    np.isfinite(train_pred).sum(),
    "images; missing",
    missing_train_images,
    "time(s)",
    round(time.time() - t0, 2),
)

valid_mask = np.isfinite(train_pred)
train_pred_valid = train_pred[valid_mask]
train_y_valid = train_y[valid_mask]

if len(train_pred_valid) >= 500 and len(np.unique(train_y_valid)) == 5:
    new_thr, best_s = _fit_thresholds_by_qwk_search(
        train_y_valid, train_pred_valid, init_thr=threshold, n_iter=3
    )
    yhat_all = _apply_thresholds_float_to_class(train_pred_valid, new_thr)
    kappa_all = cohen_kappa_score(train_y_valid, yhat_all, weights="quadratic")
    print("Old thresholds:", threshold)
    print("New train-QWK-tuned thresholds:", new_thr, "best_search_qwk:", float(best_s))
    print("Train quadratic kappa (sanity, not Kaggle):", float(kappa_all))
    threshold = new_thr
else:
    print(
        "Threshold tuning skipped (insufficient train data / missing classes). Using thresholds:",
        threshold,
    )




## === cell 6
class _TestDataset(Dataset):
    def __init__(self, ids, img_dir, transform):
        self.ids = list(ids)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        idx = self.ids[i]
        pth = _resolve_image_path(self.img_dir, idx)
        if pth is None:
            return idx, None
        img = Image.open(pth).convert("RGB")
        img = self.transform(img)
        return idx, img


def _collate_test(batch):
    ids, imgs = [], []
    missing = 0
    for idx, img in batch:
        if img is None:
            missing += 1
            continue
        ids.append(idx)
        imgs.append(img)
    if len(imgs) == 0:
        return [], None, missing
    return ids, torch.stack(imgs, 0), missing


test_ds = _TestDataset(test_ids, test_img_dir, transform)
test_loader = DataLoader(
    test_ds,
    batch_size=16,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
    collate_fn=_collate_test,
)

pred_map = {}
missing_images = 0

t0 = time.time()
with torch.no_grad():
    seen = 0
    for ids_b, x_b, miss in test_loader:
        missing_images += int(miss)
        if x_b is None:
            continue
        x_b = x_b.to(device)
        _, _, o_out = net(x_b)
        scores = _ordinal_expected_score_batch(o_out)  # (B,) tensor
        preds = regress2class(scores).numpy().astype(int).tolist()
        for idx, p in zip(ids_b, preds):
            pred_map[idx] = int(p)
        seen += len(ids_b)
        if seen % 50 < len(ids_b):
            print("Predicting", min(seen, len(test_ids)), "/", len(test_ids))

print("Test inference time(s):", round(time.time() - t0, 2))
print("Missing images:", missing_images)

submission = [[idx, pred_map.get(idx, 0)] for idx in test_ids]
submission = np.array(submission, dtype=object)



## === cell 7
df = pd.DataFrame(submission, columns=["id_code", "diagnosis"])
df["id_code"] = df["id_code"].astype(str)
df["diagnosis"] = df["diagnosis"].astype(int)

assert len(df) == len(
    test_ids
), f"Submission rows ({len(df)}) != test rows ({len(test_ids)})"
assert list(df.columns) == ["id_code", "diagnosis"]

df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", df.shape)
print(df.head())
