# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

3.10

# 3. Installed packages

albumentations==2.0.8
geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124
tqdm==4.67.1

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

0.5198717250023988

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the pipeline so it always produces a valid `submission.csv` end-to-end by (1) correcting the model head definition (Softmax needs an explicit `dim`) and image tensor formatting (proper HWC→CHW, normalization to [0,1]). I also handle the missing `../input/pretrained/model-2.bin` robustly by searching common Kaggle input locations; if no weights are found, the code still run and create a submission (score be poor, but you at least get a valid file instead of a runtime error). Finally, I ensure inference uses `num_workers=0` for Kaggle stability and that the predictions align exactly to `sample_submission.csv` row order.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is consistent with running inference using random weights (or mismatched weights), so the smallest legitimate improvement is to (1) reliably use ImageNet pretrained ResNet50 weights when your competition checkpoint isn’t found, and (2) match the standard ResNet normalization at inference time. This keeps your architecture and inference loop the same, but makes predictions meaningfully better than random, moving the kappa score up toward your 0.52 target. I also keep output alignment to `sample_submission.csv` and still write a valid `submission.csv` end-to-end.'
- What this solution (achieved 0.07186) has done: 'Your 0.0 score is consistent with systematically bad predictions caused by a preprocessing mismatch: you apply `A.Normalize(...)` (which expects uint8 0..255) and then you also divide by 255 again, effectively shrinking inputs to ~0..0.004 and breaking the ImageNet-pretrained backbone. To move the score upward toward your 0.52 target with minimal change and identical core inference logic, I fix the image pipeline so it does exactly one correct ImageNet normalization (convert to float [0,1] then normalize), and I also convert OpenCV BGR to RGB to match torchvision ResNet expectations. Everything else (model, head, weights fallback behavior, dataloader, argmax predictions, submission writing) stays the same.'
- What this solution (achieved 0.06884) has done: 'Your current 0.07186 is far below the 0.5199 target, so we should improve predictions with the smallest change that keeps your inference-only core logic intact. The biggest issue is the model head: applying a Softmax and then argmax is redundant, and more importantly it make any real competition checkpoint incompatible (most checkpoints are trained with logits, not post-softmax outputs), hurting both weight loading and output quality. I remove the final Softmax (keeping the same three Linear layers) so the model outputs logits, which preserves evaluation semantics (argmax class) but aligns with typical training/checkpointing and improves calibration. I also move resizing before albumentations normalization (so statistics match the final input scale) and add a safe, minimal fallback to resize if transforms are None.'
- What this solution (achieved 0.07116) has done: 'Your current score (0.06884) is far below the 0.5199 target, so we should make a small, legitimate improvement that keeps the same architecture and inference loop. The biggest low-risk gain here is to match the input preprocessing to what ResNet50 ImageNet weights expect: center-crop style geometry (resize shorter side then center crop) instead of directly warping every image to 512×512, which can distort lesions and reduce signal. This preserves your core logic (same model, same head, same argmax) while making inputs more consistent with the pretrained backbone and typically improves kappa. I also add a minimal fallback for weight loading (`strict=False` with reporting) so that if the checkpoint keys slightly differ, you still load what matches rather than crashing or silently using random head weights.'
- What this solution (achieved 0.0) has done: 'Your current score is far below the target (0.07116 vs 0.51987), so we should make a small change that legitimately improves predictions without changing the model/loop: calibrate the final discrete classes for quadratic weighted kappa. The simplest way is to keep your argmax-based class prediction as the starting point, then apply a lightweight, deterministic post-processing step that searches for the best set of class boundaries on a held-out validation split using out-of-fold logits. This preserves your ResNet50 + 3-linear-head architecture and inference semantics (still producing 0–4 labels), but typically moves QWK substantially upward toward your target by fixing class imbalance/calibration. The submission file format and row alignment remain unchanged, and runtime stays within limits by using a small subset of images for calibration.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is consistent with a broken preprocessing path: after `A.Normalize(...)` the tensor is already float32 normalized, but your dataset then transposes without scaling to `[0,1]` when `transforms` is `None`, and more importantly, when `transforms` is used you never enforce CHW float32 contiguous output (Albumentations returns HWC). I make the smallest fix that keeps your model and inference logic identical: ensure Albumentations always produces an HWC float32 image and then convert to a contiguous CHW torch tensor in one consistent place. I also fix the calibration step so it uses a deterministic, stratified subset rather than a pure random sample (still 900 images) to stabilize threshold fitting and improve QWK toward your 0.52 target without changing architecture/training. Finally, I keep submission alignment to `sample_submission.csv` unchanged and still write `submission.csv` end-to-end.'
- What this solution (achieved 0.0) has done: 'Your pipeline already runs end-to-end and writes a valid `submission.csv`, so the smallest score-moving change is to improve the threshold calibration signal (which directly optimizes QWK) without changing the model or inference loop. I replace the current “expected class” continuous score with a cumulative-probability (ordinal) score that typically aligns better with the ordered nature of DR grades and makes threshold fitting more effective. I also make the 2-fold split stratified (still 2 folds, same size) so each fold has all classes, stabilizing the fitted thresholds and reducing the chance of degenerate QWK during calibration. Everything else (ResNet50 backbone choice, 3-Linear head, dataloaders, logits inference, and submission formatting) stays the same.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2 as cv
import random
import warnings
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
from tqdm import tqdm
import albumentations as A
from torchvision.models import resnet50, ResNet50_Weights



## === cell 1
SEED = 123
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

warnings.filterwarnings("ignore")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n Device : {device.upper()}")




## === cell 2
def resolve_path(path: str) -> str:
    if path is None:
        return path
    if os.path.exists(path):
        return path

    p = path.replace("\\", "/")

    for anchor in ("/input/", "/data/"):
        if anchor in p:
            suffix = p.split(anchor, 1)[1]
            for base in ("/kaggle/input/", "/kaggle/data/"):
                cand = os.path.join(base, suffix)
                if os.path.exists(cand):
                    return cand

    for prefix in ("../input/", "../data/"):
        if p.startswith(prefix):
            suffix = p[len(prefix) :]
            for base in ("/kaggle/input/", "/kaggle/data/"):
                cand = os.path.join(base, suffix)
                if os.path.exists(cand):
                    return cand

    return path  # fall back; will error later with a clear message


TEST_PATH = resolve_path("../input/aptos2019-blindness-detection/test.csv")
TEST_IMG = resolve_path("../input/aptos2019-blindness-detection/test_images")
SAMPLE_SUB_PATH = resolve_path(
    "../input/aptos2019-blindness-detection/sample_submission.csv"
)

TRAIN_PATH = resolve_path("../input/aptos2019-blindness-detection/train.csv")
TRAIN_IMG = resolve_path("../input/aptos2019-blindness-detection/train_images")

print("Resolved paths:")
print(" TEST_PATH:", TEST_PATH)
print(" TEST_IMG :", TEST_IMG)
print(" SAMPLE   :", SAMPLE_SUB_PATH)
print(" TRAIN    :", TRAIN_PATH)
print(" TRAIN_IMG:", TRAIN_IMG)

WORKDIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
os.makedirs(WORKDIR, exist_ok=True)




## === cell 3
class AptosDataset(Dataset):
    def __init__(
        self,
        data_path,
        img_dir,
        name,
        transforms,
        resize=(512, 512),
        return_label=False,
    ):
        self.data_path = data_path
        self.img_dir = img_dir
        self.resize = resize
        self.transforms = transforms
        self.df = pd.read_csv(self.data_path)
        self.name = name
        self.return_label = return_label

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = row["id_code"]
        img_name = img_id + ".png"
        img_path = os.path.join(self.img_dir, img_name)

        img = cv.imread(img_path, cv.IMREAD_COLOR)  # BGR uint8
        if img is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")

        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        if self.transforms:
            transformed = self.transforms(image=img)
            img = transformed["image"]
            if self.resize is not None:
                if img.ndim != 3:
                    raise RuntimeError(
                        f"Unexpected image shape after transforms: {img.shape}"
                    )
                h, w = img.shape[:2]
                if (h, w) != (self.resize[1], self.resize[0]):
                    img = cv.resize(img, self.resize, interpolation=cv.INTER_AREA)
        else:
            if self.resize:
                img = cv.resize(img, self.resize, interpolation=cv.INTER_AREA)
            img = img.astype(np.float32) / 255.0

        if img.dtype != np.float32:
            img = img.astype(np.float32)

        img = np.ascontiguousarray(img.transpose(2, 0, 1))
        img = torch.from_numpy(img)

        if self.return_label:
            y = int(row["diagnosis"])
            return img, y
        return img




## === cell 4
def find_first_existing(paths):
    for p in paths:
        if p and os.path.isfile(p):
            return p
    return None


def _unwrap_checkpoint_to_state_dict(obj):
    if not isinstance(obj, dict):
        return None
    if len(obj) > 0 and any(isinstance(k, str) and "." in k for k in obj.keys()):
        return obj
    for key in ("state_dict", "model_state_dict", "model", "net", "weights"):
        v = obj.get(key, None)
        if isinstance(v, dict) and len(v) > 0:
            return v
    return obj


def _strip_prefix(state, prefixes=("module.", "model.", "net.")):
    if not isinstance(state, dict) or len(state) == 0:
        return state
    out = {}
    for k, v in state.items():
        nk = k
        for p in prefixes:
            if nk.startswith(p):
                nk = nk[len(p) :]
        out[nk] = v
    return out


candidate_weight_paths = [
    "../input/pretrained/model-2.bin",
    "/kaggle/input/pretrained/model-2.bin",
    "../input/pretrained/model-2.pth",
    "/kaggle/input/pretrained/model-2.pth",
    "../input/pretrained/model-2.pt",
    "/kaggle/input/pretrained/model-2.pt",
]

for root, dirs, files in os.walk("/kaggle/input"):
    for fn in files:
        if fn in ("model-2.bin", "model-2.pth", "model-2.pt"):
            candidate_weight_paths.append(os.path.join(root, fn))

WEIGHTS_PATH = find_first_existing(candidate_weight_paths)
print(
    "Competition weights found at:",
    (
        WEIGHTS_PATH
        if WEIGHTS_PATH
        else "NOT FOUND (will fall back to ImageNet pretrained backbone)"
    ),
)



## === cell 5
BATCH_SIZE = 16
IMG_DIM = 512

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

test_tfms = A.Compose(
    [
        A.SmallestMaxSize(max_size=IMG_DIM, interpolation=cv.INTER_AREA),
        A.CenterCrop(height=IMG_DIM, width=IMG_DIM),
        A.ToFloat(max_value=255.0),
        A.Normalize(mean=imagenet_mean, std=imagenet_std, max_pixel_value=1.0),
    ]
)

train_tfms = test_tfms

test_dataset = AptosDataset(
    TEST_PATH,
    TEST_IMG,
    "test",
    transforms=test_tfms,
    resize=(IMG_DIM, IMG_DIM),
    return_label=False,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)

if WEIGHTS_PATH is None:
    base = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
else:
    base = None
    model = resnet50(weights=None)

model.fc = nn.Sequential(
    nn.Linear(in_features=2048, out_features=1024, bias=True),
    nn.Linear(in_features=1024, out_features=512, bias=True),
    nn.Linear(in_features=512, out_features=5, bias=True),
)

if WEIGHTS_PATH is None:
    with torch.no_grad():
        if isinstance(base.fc, nn.Linear) and base.fc.weight.shape[1] == 2048:
            model.fc[0].weight.copy_(base.fc.weight[:1024, :])
            model.fc[0].bias.copy_(base.fc.bias[:1024])
        nn.init.kaiming_normal_(
            model.fc[1].weight, mode="fan_out", nonlinearity="linear"
        )
        nn.init.zeros_(model.fc[1].bias)
        nn.init.kaiming_normal_(
            model.fc[2].weight, mode="fan_out", nonlinearity="linear"
        )
        nn.init.zeros_(model.fc[2].bias)

if WEIGHTS_PATH is not None:
    raw = torch.load(WEIGHTS_PATH, map_location="cpu")
    state = _unwrap_checkpoint_to_state_dict(raw)
    state = _strip_prefix(state)

    try:
        model.load_state_dict(state, strict=True)
        print("Loaded competition weights with strict=True")
    except RuntimeError as e:
        print(
            "Strict weight load failed; retrying with strict=False. Error was:\n",
            str(e)[:800],
        )
        incompat = model.load_state_dict(state, strict=False)
        missing = list(getattr(incompat, "missing_keys", []))
        unexpected = list(getattr(incompat, "unexpected_keys", []))
        print(
            "Loaded with strict=False. Missing keys:",
            len(missing),
            "Unexpected keys:",
            len(unexpected),
        )

model = model.to(device)
model.eval()




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/2559402395.py in <cell line: 0>()
     53         # Copy the first 1024 rows/biases: preserves a meaningful projection instead of random init.
     54         if isinstance(base.fc, nn.Linear) and base.fc.weight.shape[1] == 2048:
---> 55             model.fc[0].weight.copy_(base.fc.weight[:1024, :])
     56             model.fc[0].bias.copy_(base.fc.bias[:1024])
     57         # Initialize second and third layers in a stable, small way.

RuntimeError: The size of tensor a (1024) must match the size of tensor b (1000) at non-singleton dimension 0

## === cell 6
def quadratic_weighted_kappa(y_true, y_pred, n_classes=5):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    assert y_true.shape == y_pred.shape

    O = np.zeros((n_classes, n_classes), dtype=np.float64)
    for a, b in zip(y_true, y_pred):
        if 0 <= a < n_classes and 0 <= b < n_classes:
            O[a, b] += 1.0

    act_hist = O.sum(axis=1)
    pred_hist = O.sum(axis=0)
    E = np.outer(act_hist, pred_hist)
    if E.sum() > 0:
        E = E * (O.sum() / E.sum())

    W = np.zeros((n_classes, n_classes), dtype=np.float64)
    for i in range(n_classes):
        for j in range(n_classes):
            W[i, j] = ((i - j) ** 2) / ((n_classes - 1) ** 2)

    num = (W * O).sum()
    den = (W * E).sum()
    return 1.0 - (num / den if den > 0 else 0.0)


def logits_to_continuous_score(logits_5):
    logits_5 = np.asarray(logits_5, dtype=np.float32)
    p = (
        torch.softmax(torch.from_numpy(logits_5), dim=1)
        .cpu()
        .numpy()
        .astype(np.float32)
    )
    cdf = np.cumsum(p, axis=1)  # P(y <= k)
    score = 4.0 - cdf[:, :4].sum(axis=1)
    return score.astype(np.float32)


def apply_thresholds(scores, thresholds):
    scores = np.asarray(scores, dtype=np.float32)
    t = np.asarray(thresholds, dtype=np.float32)
    pred = np.zeros(scores.shape[0], dtype=int)
    pred[scores > t[0]] = 1
    pred[scores > t[1]] = 2
    pred[scores > t[2]] = 3
    pred[scores > t[3]] = 4
    return pred


def fit_thresholds_by_grid(
    scores, y_true, base_thresholds=(0.5, 1.5, 2.5, 3.5), step=0.05, span=0.8
):
    scores = np.asarray(scores, dtype=np.float32)
    y_true = np.asarray(y_true, dtype=int)

    base = np.asarray(base_thresholds, dtype=np.float32)
    grids = []
    for b in base:
        lo = b - span
        hi = b + span
        grids.append(np.arange(lo, hi + 1e-9, step, dtype=np.float32))

    best_kappa = -1e9
    best_t = base.copy()

    for t0 in grids[0]:
        for t1 in grids[1]:
            if t1 <= t0:
                continue
            for t2 in grids[2]:
                if t2 <= t1:
                    continue
                for t3 in grids[3]:
                    if t3 <= t2:
                        continue
                    pred = apply_thresholds(scores, (t0, t1, t2, t3))
                    k = quadratic_weighted_kappa(y_true, pred, n_classes=5)
                    if k > best_kappa:
                        best_kappa = k
                        best_t = np.array([t0, t1, t2, t3], dtype=np.float32)
    return best_t, float(best_kappa)


train_df = pd.read_csv(TRAIN_PATH)
train_df = train_df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

max_calib = min(1200, len(train_df))
per_class = max(1, max_calib // 5)
parts = []
for c in range(5):
    df_c = train_df[train_df["diagnosis"] == c]
    take = min(per_class, len(df_c))
    if take > 0:
        parts.append(df_c.iloc[:take])
calib_df = pd.concat(parts, axis=0).reset_index(drop=True)

if len(calib_df) < max_calib:
    remaining = train_df[~train_df["id_code"].isin(calib_df["id_code"])].reset_index(
        drop=True
    )
    need = max_calib - len(calib_df)
    calib_df = pd.concat([calib_df, remaining.iloc[:need]], axis=0).reset_index(
        drop=True
    )

fold = np.full(len(calib_df), -1, dtype=int)
for c in range(5):
    idx_c = np.where(calib_df["diagnosis"].values.astype(int) == c)[0]
    fold[idx_c] = (np.arange(len(idx_c)) % 2).astype(int)


def infer_logits_for_df(df_subset, csv_name):
    csv_path = os.path.join(WORKDIR, csv_name)
    df_subset.to_csv(csv_path, index=False)

    ds = AptosDataset(
        csv_path,
        TRAIN_IMG,
        "calib_part",
        transforms=train_tfms,
        resize=(IMG_DIM, IMG_DIM),
        return_label=True,
    )
    dl = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    logits_list = []
    y_list = []
    with torch.no_grad():
        for x, y in tqdm(dl, desc=f"Infer {csv_name}", leave=False):
            out = model(x.to(device))
            logits_list.append(out.detach().cpu())
            y_list.extend(y.numpy().tolist())
    logits = torch.cat(logits_list, dim=0).numpy()
    return logits, np.asarray(y_list, dtype=int)


oof_scores = np.zeros(len(calib_df), dtype=np.float32)
oof_y = calib_df["diagnosis"].astype(int).values

for f in (0, 1):
    va_idx = np.where(fold == f)[0]
    va_df = calib_df.iloc[va_idx].reset_index(drop=True)
    va_logits, va_y = infer_logits_for_df(va_df, f"calib_fold{f}.csv")
    va_scores = logits_to_continuous_score(va_logits)

    if len(va_scores) != len(va_idx) or len(va_y) != len(va_idx):
        raise RuntimeError("OOF inference size mismatch.")
    oof_scores[va_idx] = va_scores.astype(np.float32)

thresholds, best_k = fit_thresholds_by_grid(
    oof_scores, oof_y, base_thresholds=(0.5, 1.5, 2.5, 3.5), step=0.10, span=0.6
)
print("Fitted thresholds (OOF):", thresholds.tolist(), "OOF QWK:", best_k)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/523816759.py in <cell line: 0>()
    147     va_idx = np.where(fold == f)[0]
    148     va_df = calib_df.iloc[va_idx].reset_index(drop=True)
--> 149     va_logits, va_y = infer_logits_for_df(va_df, f"calib_fold{f}.csv")
    150     va_scores = logits_to_continuous_score(va_logits)
    151 

/tmp/ipykernel_55/523816759.py in infer_logits_for_df(df_subset, csv_name)
    134     with torch.no_grad():
    135         for x, y in tqdm(dl, desc=f"Infer {csv_name}", leave=False):
--> 136             out = model(x.to(device))
    137             logits_list.append(out.detach().cpu())
    138             y_list.extend(y.numpy().tolist())

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/usr/local/lib/python3.11/dist-packages/torchvision/models/resnet.py in forward(self, x)
    283 
    284     def forward(self, x: Tensor) -> Tensor:
--> 285         return self._forward_impl(x)
    286 
    287 

/usr/local/lib/python3.11/dist-packages/torchvision/models/resnet.py in _forward_impl(self, x)
    266     def _forward_impl(self, x: Tensor) -> Tensor:
    267         # See note [TorchScript super()]
--> 268         x = self.conv1(x)
    269         x = self.bn1(x)
    270         x = self.relu(x)

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/conv.py in forward(self, input)
    552 
    553     def forward(self, input: Tensor) -> Tensor:
--> 554         return self._conv_forward(input, self.weight, self.bias)
    555 
    556 

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/conv.py in _conv_forward(self, input, weight, bias)
    547                 self.groups,
    548             )
--> 549         return F.conv2d(
    550             input, weight, bias, self.stride, self.padding, self.dilation, self.groups
    551         )

RuntimeError: Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor) should be the same

## === cell 7
test_logits = []
with torch.no_grad():
    for x in tqdm(test_loader, desc="Inference", leave=False):
        logits = model(x.to(device))
        test_logits.append(logits.detach().cpu())
test_logits = torch.cat(test_logits, dim=0).numpy()

test_scores = logits_to_continuous_score(test_logits)
labels = apply_thresholds(test_scores, thresholds).tolist()

print("Preds:", len(labels), "Expected:", len(test_dataset))



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/3583926036.py in <cell line: 0>()
      2 with torch.no_grad():
      3     for x in tqdm(test_loader, desc="Inference", leave=False):
----> 4         logits = model(x.to(device))
      5         test_logits.append(logits.detach().cpu())
      6 test_logits = torch.cat(test_logits, dim=0).numpy()

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/usr/local/lib/python3.11/dist-packages/torchvision/models/resnet.py in forward(self, x)
    283 
    284     def forward(self, x: Tensor) -> Tensor:
--> 285         return self._forward_impl(x)
    286 
    287 

/usr/local/lib/python3.11/dist-packages/torchvision/models/resnet.py in _forward_impl(self, x)
    266     def _forward_impl(self, x: Tensor) -> Tensor:
    267         # See note [TorchScript super()]
--> 268         x = self.conv1(x)
    269         x = self.bn1(x)
    270         x = self.relu(x)

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/conv.py in forward(self, input)
    552 
    553     def forward(self, input: Tensor) -> Tensor:
--> 554         return self._conv_forward(input, self.weight, self.bias)
    555 
    556 

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/conv.py in _conv_forward(self, input, weight, bias)
    547                 self.groups,
    548             )
--> 549         return F.conv2d(
    550             input, weight, bias, self.stride, self.padding, self.dilation, self.groups
    551         )

RuntimeError: Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor) should be the same

## === cell 8
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
if len(labels) != len(sample_sub):
    raise RuntimeError(
        f"Prediction length mismatch: got {len(labels)} preds, submission expects {len(sample_sub)} rows"
    )

sample_sub["diagnosis"] = labels
sample_sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sample_sub.shape)
print("submission.csv path:", os.path.abspath("submission.csv"))

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1335215481.py in <cell line: 0>()
      1 sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
----> 2 if len(labels) != len(sample_sub):
      3     raise RuntimeError(
      4         f"Prediction length mismatch: got {len(labels)} preds, submission expects {len(sample_sub)} rows"
      5     )

NameError: name 'labels' is not defined

## --- ERROR in outputing the csv:
Invalid submission: Submission must have the same id_codes as answers
