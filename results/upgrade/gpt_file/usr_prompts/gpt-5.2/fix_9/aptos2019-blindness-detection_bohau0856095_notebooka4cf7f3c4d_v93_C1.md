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

0.9228320167757752

# 6. Current score

-0.02152

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I make the weights loading robust so the notebook doesn’t crash when no external weight file is present, and still produces a valid `submission.csv` end-to-end (fallback to random-init model if needed). I also fix the CUDA/CPU dtype mismatch by ensuring the model parameters are moved to the same device as the input before inference. Finally, I harden image path discovery (some Kaggle copies have nested folders) and make the inference loop resilient so it always writes a non-empty submission with the exact required columns.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is consistent with running a randomly initialized model (no weights found), which produce near-random predictions. The smallest change that should move the score toward your target is to use the exact same architecture/inference logic but load an actual pretrained checkpoint if it exists in common Kaggle locations, and fail loudly (instead of silently) if none is available so you don’t accidentally submit random predictions again. I also align prediction post-processing with the model’s intended “final” head when (and only when) the loaded checkpoint contains `final_regressor` weights; otherwise it preserves the original `r_out` path. Finally, I keep your submission alignment safeguards and ensure `submission.csv` is always generated.'
- What this solution (achieved 0.01221) has done: 'I make the notebook run end-to-end again by removing the hard failure when no external checkpoint is attached, while keeping your model/inference logic unchanged. To move the score up from 0.0 toward the target, I add an in-notebook fallback that loads ImageNet pretrained weights for the EfficientNet backbone (same architecture) when the competition checkpoint isn’t found, so predictions are no longer random. I also ensure `use_final_head` is always defined (to fix the `NameError`) and keep submission alignment/format exactly as required. The changes are limited to robust weight loading and deterministic, valid inference/submission writing.'
- What this solution (achieved -0.03395) has done: 'Your current score (0.01221) is far below the target (0.9228), and the most likely reason is that you are still effectively using untrained/random heads (even if the backbone is ImageNet-pretrained). To move the score up toward the target without changing the model/training core logic, I (1) search more broadly and intelligently for a real APTOS-trained checkpoint (including common Kaggle notebook output/working locations) and load it robustly, and (2) ensure we use the correct inference head automatically by probing the loaded state dict (final head vs. regressor path). Additionally, I do a minimal but high-impact inference fix: batch inference with a DataLoader to avoid per-image overhead/timeouts and ensure the full test set is processed deterministically within the time limit. The submission writing/format and the model architecture/loss semantics remain unchanged.'
- What this solution (achieved -0.02576) has done: 'Your score is far below the target, and the main cause is that the notebook is effectively doing inference with randomly initialized heads (and often no APTOS-trained checkpoint), which yields near-random class predictions and very low/negative QWK. The smallest legitimate change that can move you sharply toward the target (without changing architecture/training/loss) is to (1) reliably find and load an APTOS-trained checkpoint from the dataset itself (many Kaggle datasets include .pth/.pkl files under the competition folder), and (2) run a minimal “test-time augmentation” as an inference-only averaging step (original + horizontal flip) to improve stability without altering the model. I also fix a subtle but important bug in `regress2class` (it was producing float tensors via implicit addition and could behave oddly); making it return integer class indices deterministically improves the mapping to the competition’s discrete labels. All I/O paths and the submission schema remain unchanged, and the script still writes `submission.csv` end-to-end.'
- What this solution (achieved -0.00925) has done: 'Your very low/negative QWK strongly suggests the inference-time discretization (the fixed thresholds in `regress2class`) is badly miscalibrated for the checkpoint you’re actually using (often ImageNet-backbone + random heads), so small label shifts create large kappa penalties. To move the score sharply upward toward your target without changing the model architecture or training, I add a minimal, inference-only calibration step: fit the 4 regression thresholds on the training set using the same model predictions and choose the thresholds that maximize QWK on a held-out validation split. Then I apply those fitted thresholds to convert test regression outputs to the required discrete labels (still 0–4), keeping the rest of your pipeline intact (same model, same transforms, same TTA, same submission format). This is a small change in post-processing aligned to the metric and should reduce the large gap toward the target while staying within Kaggle constraints.'
- What this solution (achieved -0.02152) has done: 'The negative QWK suggests your current discretization is still unstable/miscalibrated; to move score up toward the target without changing the model itself, I keep your exact model/inference logic but make the threshold calibration stronger and consistent with how you generate the submission. Specifically, I (1) calibrate thresholds on a larger, more stable set of training predictions using out-of-fold inference (no training, just inference), and (2) replace the coarse coordinate search with a deterministic Nelder–Mead optimization over thresholds (still inference-only post-processing) to better maximize QWK. Finally, I apply the fitted thresholds in `regress2class` exactly as used for test, keeping submission format/paths identical.'

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
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import functional as FT
from PIL import Image, ImageChops

from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedShuffleSplit
import timm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



## === cell 1
threshold = [0.75, 1.5, 2.5, 3.5]


def regress2class(out: torch.Tensor) -> torch.Tensor:
    out = out.view(-1)
    pred = torch.zeros(out.size(0), dtype=torch.long, device=out.device)
    for t in threshold:
        pred += (out >= t).long()
    return pred


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
CANDIDATE_ROOTS = [
    "/kaggle/input/aptos2019-blindness-detection",
    "../input/aptos2019-blindness-detection",
    "/kaggle/data/aptos2019-blindness-detection",
    "/kaggle/data/input/aptos2019-blindness-detection",
]
DATA_ROOT = None
for p in CANDIDATE_ROOTS:
    if os.path.exists(os.path.join(p, "test.csv")):
        if os.path.isdir(os.path.join(p, "test_images")) or os.path.isdir(
            os.path.join(p, "aptos2019-blindness-detection", "test_images")
        ):
            DATA_ROOT = p
            break

if DATA_ROOT is None:
    hits = glob.glob(
        "/kaggle/**/aptos2019-blindness-detection/test.csv", recursive=True
    )
    if hits:
        DATA_ROOT = os.path.dirname(hits[0])

if DATA_ROOT is None:
    raise FileNotFoundError(
        "Could not locate aptos2019-blindness-detection dataset directory."
    )


def resolve_img_dir(root, split):
    d1 = os.path.join(root, f"{split}_images")
    d2 = os.path.join(root, "aptos2019-blindness-detection", f"{split}_images")
    if os.path.isdir(d1):
        return d1
    if os.path.isdir(d2):
        return d2
    raise FileNotFoundError(f"Could not find {split}_images directory under: {root}")


TRAIN_IMG_DIR = resolve_img_dir(DATA_ROOT, "train")
TEST_IMG_DIR = resolve_img_dir(DATA_ROOT, "test")

train_df = pd.read_csv(os.path.join(DATA_ROOT, "train.csv"))
test_df = pd.read_csv(os.path.join(DATA_ROOT, "test.csv"))
test_ids = np.squeeze(test_df["id_code"].values)

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

transform_hflip = transforms.Compose(
    [
        trim(),
        cropTo4_3(),
        transforms.Resize((input_size * 3 // 4, input_size)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.384, 0.258, 0.174], std=[0.124, 0.089, 0.094]),
    ]
)


class TestDataset(Dataset):
    def __init__(self, ids, img_dir, transform, transform2=None):
        self.ids = list(ids)
        self.img_dir = img_dir
        self.transform = transform
        self.transform2 = transform2

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        idx = self.ids[i]
        image_name = os.path.join(self.img_dir, f"{idx}.png")
        try:
            img = Image.open(image_name).convert("RGB")
            x1 = self.transform(img)
            if self.transform2 is not None:
                x2 = self.transform2(img)
            else:
                x2 = None
        except Exception:
            x1, x2 = None, None
        return idx, x1, x2


class TrainInferDataset(Dataset):
    def __init__(self, ids, labels, img_dir, transform, transform2=None):
        self.ids = list(ids)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.img_dir = img_dir
        self.transform = transform
        self.transform2 = transform2

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        idx = self.ids[i]
        y = int(self.labels[i])
        image_name = os.path.join(self.img_dir, f"{idx}.png")
        try:
            img = Image.open(image_name).convert("RGB")
            x1 = self.transform(img)
            if self.transform2 is not None:
                x2 = self.transform2(img)
            else:
                x2 = None
        except Exception:
            x1, x2 = None, None
        return idx, x1, x2, y


net = ThreeStage_Model()
use_final_head = False  # ensure always defined

WEIGHT_CANDIDATES = [
    "../input/weights/B4_3stage_7epoch_finetune3.pkl",
    "/kaggle/input/weights/B4_3stage_7epoch_finetune3.pkl",
]

WEIGHT_CANDIDATES += glob.glob(
    os.path.join(DATA_ROOT, "**", "B4_3stage_7epoch_finetune3.pkl"), recursive=True
)
WEIGHT_CANDIDATES += glob.glob(
    os.path.join(DATA_ROOT, "**", "B4_3stage_*.pkl"), recursive=True
)
WEIGHT_CANDIDATES += glob.glob(
    os.path.join(DATA_ROOT, "**", "B4_3stage_*.pth"), recursive=True
)
WEIGHT_CANDIDATES += glob.glob(
    os.path.join(DATA_ROOT, "**", "B4_3stage_*.pt"), recursive=True
)

WEIGHT_CANDIDATES += glob.glob(
    "/kaggle/input/**/B4_3stage_7epoch_finetune3.pkl", recursive=True
)
WEIGHT_CANDIDATES += glob.glob("/kaggle/input/**/B4_3stage_*.pkl", recursive=True)
WEIGHT_CANDIDATES += glob.glob("/kaggle/input/**/B4_3stage_*.pth", recursive=True)

WEIGHT_CANDIDATES += glob.glob("/kaggle/working/**/*.pth", recursive=True)
WEIGHT_CANDIDATES += glob.glob("/kaggle/working/**/*.pkl", recursive=True)
WEIGHT_CANDIDATES += glob.glob("/kaggle/working/**/*.pt", recursive=True)


def _load_state_dict_safely(path):
    state = torch.load(path, map_location="cpu")
    if (
        isinstance(state, dict)
        and "state_dict" in state
        and isinstance(state["state_dict"], dict)
    ):
        state = state["state_dict"]
    if (
        isinstance(state, dict)
        and "model" in state
        and isinstance(state["model"], dict)
    ):
        state = state["model"]
    if isinstance(state, dict):
        new_state = {}
        for k, v in state.items():
            nk = (
                k[len("module.") :]
                if isinstance(k, str) and k.startswith("module.")
                else k
            )
            new_state[nk] = v
        state = new_state
    return state


weight_path = None
for wp in WEIGHT_CANDIDATES:
    if os.path.isfile(wp):
        weight_path = wp
        break

if weight_path is not None:
    state = _load_state_dict_safely(weight_path)
    missing, unexpected = net.load_state_dict(state, strict=False)
    print(f"Loaded weights from: {weight_path}")
    print(f"Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")

    if isinstance(state, dict):
        use_final_head = any(
            str(k).startswith("final_regressor.") for k in state.keys()
        )
    else:
        use_final_head = False
    print("use_final_head:", use_final_head)
else:
    print(
        "WARNING: No external trained checkpoint found under DATA_ROOT,/kaggle/input or /kaggle/working. "
        "Falling back to ImageNet-pretrained EfficientNet-B4 backbone weights "
        "(heads remain randomly initialized, score will be low)."
    )
    net.backbone = timm.models.tf_efficientnet_b4_ns(pretrained=True)
    net.backbone.global_pool = GeM(flatten=True)
    use_final_head = False

net = net.to(device)
net.eval()




## === cell 5
def apply_thresholds_np(preds_cont: np.ndarray, thr: np.ndarray) -> np.ndarray:
    preds_cont = np.asarray(preds_cont, dtype=np.float32).reshape(-1)
    thr = np.asarray(thr, dtype=np.float32).reshape(-1)
    out = np.zeros_like(preds_cont, dtype=np.int64)
    for t in thr:
        out += (preds_cont >= t).astype(np.int64)
    return out


def qwk(y_true, y_pred) -> float:
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


def infer_regression_outputs(
    ids, img_dir, batch_size=8, transform1=None, transform2=None
):
    ds = TestDataset(ids, img_dir, transform1, transform2=transform2)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    preds = []
    valid_ids = []

    with torch.no_grad():
        for batch in loader:
            b_ids, x1s, x2s = batch

            if isinstance(x1s, (list, tuple)):
                for idx, x1, x2 in zip(b_ids, x1s, x2s):
                    if x1 is None:
                        continue
                    x1 = x1.unsqueeze(0).to(device, non_blocking=True)
                    if x2 is not None:
                        x2 = x2.unsqueeze(0).to(device, non_blocking=True)

                    if use_final_head:
                        out1 = net(x1, final=True)
                        if x2 is not None:
                            out2 = net(x2, final=True)
                            out = 0.5 * (out1 + out2)
                        else:
                            out = out1
                        p = float(out.squeeze(1).detach().cpu().numpy()[0])
                    else:
                        _, r_out1, _ = net(x1)
                        if x2 is not None:
                            _, r_out2, _ = net(x2)
                            r_out = 0.5 * (r_out1 + r_out2)
                        else:
                            r_out = r_out1
                        p = float(r_out.squeeze(1).detach().cpu().numpy()[0])

                    preds.append(p)
                    valid_ids.append(str(idx))
                continue

            x1 = x1s.to(device, non_blocking=True)
            x2 = x2s.to(device, non_blocking=True)

            if use_final_head:
                out1 = net(x1, final=True)
                out2 = net(x2, final=True)
                out = 0.5 * (out1 + out2)
                p = out.squeeze(1).detach().cpu().numpy().astype(np.float32)
            else:
                _, r_out1, _ = net(x1)
                _, r_out2, _ = net(x2)
                r_out = 0.5 * (r_out1 + r_out2)
                p = r_out.squeeze(1).detach().cpu().numpy().astype(np.float32)

            preds.extend(list(p.reshape(-1)))
            valid_ids.extend([str(i) for i in b_ids])

    return np.asarray(preds, dtype=np.float32), np.asarray(valid_ids, dtype=object)


def fit_thresholds_qwk_nelder_mead(
    preds_cont: np.ndarray,
    y_true: np.ndarray,
    init_thr=None,
    max_iter=120,
) -> np.ndarray:
    preds_cont = np.asarray(preds_cont, dtype=np.float32).reshape(-1)
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)

    if init_thr is None:
        x0 = np.array([0.75, 1.5, 2.5, 3.5], dtype=np.float32)
    else:
        x0 = np.array(init_thr, dtype=np.float32).reshape(-1)

    def project(x):
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        x = np.clip(x, 0.0, 4.5)
        x = np.sort(x)
        for i in range(1, 4):
            if x[i] <= x[i - 1] + 1e-4:
                x[i] = min(4.5, x[i - 1] + 1e-4)
        return x

    def score(x):
        x = project(x)
        return qwk(y_true, apply_thresholds_np(preds_cont, x))

    n = 4
    x0 = project(x0)
    f0 = score(x0)

    simplex = [x0]
    for i in range(n):
        xi = x0.copy()
        xi[
            i
        ] += 0.2  # modest initial perturbation (keeps changes minimal but effective)
        simplex.append(project(xi))
    simplex = np.stack(simplex, axis=0)
    fvals = np.array([score(s) for s in simplex], dtype=np.float32)

    alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5

    for _ in range(max_iter):
        order = np.argsort(-fvals)
        simplex = simplex[order]
        fvals = fvals[order]

        best = simplex[0]
        worst = simplex[-1]
        second_worst = simplex[-2]

        centroid = np.mean(simplex[:-1], axis=0)

        xr = project(centroid + alpha * (centroid - worst))
        fr = score(xr)

        if fr > fvals[0]:
            xe = project(centroid + gamma * (xr - centroid))
            fe = score(xe)
            if fe > fr:
                simplex[-1], fvals[-1] = xe, fe
            else:
                simplex[-1], fvals[-1] = xr, fr
        elif fr > fvals[-2]:
            simplex[-1], fvals[-1] = xr, fr
        else:
            if fr > fvals[-1]:
                xc = project(centroid + rho * (xr - centroid))
            else:
                xc = project(centroid + rho * (worst - centroid))
            fc = score(xc)
            if fc > fvals[-1]:
                simplex[-1], fvals[-1] = xc, fc
            else:
                for i in range(1, n + 1):
                    simplex[i] = project(best + sigma * (simplex[i] - best))
                    fvals[i] = score(simplex[i])

        if np.max(np.abs(simplex[0] - simplex[-1])) < 1e-3:
            break

    order = np.argsort(-fvals)
    return simplex[order][0]


train_ids_all = train_df["id_code"].astype(str).values
train_y_all = train_df["diagnosis"].astype(int).values

bs = 8 if torch.cuda.is_available() else 4

oof_preds = []
oof_y = []
oof_ids = []

seed_splitter = StratifiedShuffleSplit(n_splits=3, test_size=0.34, random_state=42)
for split_i, (tr_idx, va_idx) in enumerate(
    seed_splitter.split(train_ids_all, train_y_all), start=1
):
    va_ids = train_ids_all[va_idx]
    va_y = train_y_all[va_idx]
    va_preds_cont, va_ids_valid = infer_regression_outputs(
        va_ids,
        TRAIN_IMG_DIR,
        batch_size=bs,
        transform1=transform,
        transform2=transform_hflip,
    )
    va_y_map = {str(i): int(y) for i, y in zip(va_ids, va_y)}
    va_y_valid = np.asarray([va_y_map[i] for i in va_ids_valid], dtype=np.int64)

    oof_preds.append(va_preds_cont)
    oof_y.append(va_y_valid)
    oof_ids.append(va_ids_valid)

    base_qwk = qwk(
        va_y_valid, apply_thresholds_np(va_preds_cont, np.asarray(threshold))
    )
    print(
        f"Split {split_i}: valid={len(va_preds_cont)} | base QWK={float(base_qwk):.6f}"
    )

oof_preds = (
    np.concatenate(oof_preds, axis=0)
    if len(oof_preds)
    else np.zeros((0,), dtype=np.float32)
)
oof_y = np.concatenate(oof_y, axis=0) if len(oof_y) else np.zeros((0,), dtype=np.int64)

if len(oof_preds) >= 200:
    fitted_thr = fit_thresholds_qwk_nelder_mead(
        oof_preds, oof_y, init_thr=threshold, max_iter=140
    )
    old_score = qwk(oof_y, apply_thresholds_np(oof_preds, np.asarray(threshold)))
    new_score = qwk(oof_y, apply_thresholds_np(oof_preds, fitted_thr))
    print("Threshold calibration on OOF preds:")
    print("  default thr:", threshold, "QWK:", float(old_score))
    print("  fitted  thr:", [float(x) for x in fitted_thr], "QWK:", float(new_score))
    threshold = [float(x) for x in fitted_thr]
else:
    print(
        "Not enough valid OOF predictions for threshold calibration; using default thresholds:",
        threshold,
    )



## === cell 6
ds = TestDataset(test_ids, TEST_IMG_DIR, transform, transform2=transform_hflip)
loader = DataLoader(
    ds,
    batch_size=8 if torch.cuda.is_available() else 4,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)

submission_rows = []
with torch.no_grad():
    for batch in loader:
        ids, x1s, x2s = batch

        if isinstance(x1s, (list, tuple)):
            for idx, x1, x2 in zip(ids, x1s, x2s):
                if x1 is None:
                    submission_rows.append([str(idx), 0])
                    continue
                x1 = x1.unsqueeze(0).to(device, non_blocking=True)
                if x2 is not None:
                    x2 = x2.unsqueeze(0).to(device, non_blocking=True)

                if use_final_head:
                    out1 = net(x1, final=True)
                    if x2 is not None:
                        out2 = net(x2, final=True)
                        out = 0.5 * (out1 + out2)
                    else:
                        out = out1
                    pred = regress2class(out.squeeze(1))
                else:
                    _, r_out1, _ = net(x1)
                    if x2 is not None:
                        _, r_out2, _ = net(x2)
                        r_out = 0.5 * (r_out1 + r_out2)
                    else:
                        r_out = r_out1
                    pred = regress2class(r_out.squeeze(1))
                submission_rows.append([str(idx), int(pred.item())])
            continue

        x1 = x1s.to(device, non_blocking=True)
        x2 = x2s.to(device, non_blocking=True)

        if use_final_head:
            out1 = net(x1, final=True)
            out2 = net(x2, final=True)
            out = 0.5 * (out1 + out2)
            preds = regress2class(out.squeeze(1)).detach().cpu().numpy().astype(int)
        else:
            _, r_out1, _ = net(x1)
            _, r_out2, _ = net(x2)
            r_out = 0.5 * (r_out1 + r_out2)
            preds = regress2class(r_out.squeeze(1)).detach().cpu().numpy().astype(int)

        for idx, p in zip(ids, preds):
            submission_rows.append([str(idx), int(p)])

submission = np.array(submission_rows, dtype=object)



## === cell 7
df = pd.DataFrame(submission, columns=["id_code", "diagnosis"])
df["id_code"] = df["id_code"].astype(str)
df["diagnosis"] = df["diagnosis"].astype(int)

df = test_df.merge(df, on="id_code", how="left")
df["diagnosis"] = df["diagnosis"].fillna(0).astype(int)

df.to_csv("submission.csv", index=False)
print(df.head())
print("Wrote submission.csv with shape:", df.shape)
print("submission.csv path:", os.path.abspath("submission.csv"))
print("Final thresholds used:", threshold)
