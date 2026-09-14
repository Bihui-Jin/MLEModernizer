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
Classify each cassava image into four disease categories or a fifth category indicating a healthy leaf.

## Metric
Categorization accuracy.

## Submission Format
```
image_id,label
1000471002.jpg,4
1000840542.jpg,4
etc.
```

## Dataset
**[train/test]_images** the image files.

**train.csv**

- `image_id` the image file name.

- `label` the ID code for the disease.

**sample_submission.csv** A properly formatted sample submission, given the disclosed test set content.

- `image_id` the image file name.

- `label` the predicted ID code for the disease.

**[train/test]_tfrecords** the image files in tfrecord format.

**label_num_to_disease_map.json** The mapping between each disease code and the real disease name.

# 2. Python version

3.12

# 3. Installed packages

albumentations==2.0.8
geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
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
tqdm==4.67.1

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        input/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        working/
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
```

-> data/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/cassava-leaf-disease-classification/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/cassava-leaf-disease-classification/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> data/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> input/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> (stopped after 10 files for performance)

# 5. Target score

0.8977032336053188

# 6. Current score

0.61211

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.25299) has done: 'I fix the Albumentations runtime errors by updating `RandomResizedCrop` to the v2 API signature (using `size=(H, W)`), keeping all augmentation logic the same. I also fix the missing-weights crash by switching `INPUT_PATH` to the actual competition dataset directory and adding a small fallback search so the code can locate the provided `.pth` files if they exist. To ensure a valid submission is always produced, I add a safe fallback to run the models with `pretrained=True` if the checkpoint files are not found (core model architectures and inference approach remain unchanged), and I make test image ordering deterministic so predictions align consistently with `image_id`. Lastly, I make the submission columns exactly `image_id,label` and write `submission.csv` in the working directory.'
- What this solution (achieved 0.24253) has done: 'I fix the immediate runtime crash by replacing the removed `A.Cutout` augmentation with the Albumentations v2 equivalent `A.CoarseDropout` configured to behave like cutout, keeping the augmentation intent the same. I also correct the focal-loss helper (it currently computes `p_t` incorrectly) to avoid silent logic bugs if you later train with it, without changing your current inference-only path. Finally, I make model loading order safe (wrap with `DataParallel` before/after loading consistently) and ensure test images are read from the correct directory, so the script reliably runs end-to-end and writes a valid `submission.csv`.'
- What this solution (achieved 0.11809) has done: 'Your low score is most consistent with the fallback path being used (missing checkpoints) while still keeping randomly-initialized final classification layers, which makes predictions nearly random. I keep the exact same two-model + TTA + normalized-logit ensembling core logic, but change the fallback behavior so that when checkpoints are missing we do **not** replace the classifier layers (so ImageNet-pretrained heads are used instead of random heads). Since this creates 1000-class logits, I add a minimal, deterministic mapping from ImageNet logits to 5 cassava labels via a fixed random projection (seeded), preserving the same inference semantics while making predictions far more informative than random. I also ensure checkpoints (if present) still load exactly as before and keep the submission formatting and deterministic test ordering unchanged.'
- What this solution (achieved 0.60575) has done: 'The timeout is dominated by extremely inefficient inference: each test image is opened from disk inside the inner TTA loop (so every file is decoded 8–16× per model), and inference is done one image at a time, preventing GPU batching and wasting DataLoader-style parallelism. I keep the exact same models, weights, TTA count, and augmentation pipeline, but restructure inference to (1) load/decode each image only once, (2) apply TTA on the already-decoded array, and (3) batch multiple images per forward pass. I also replace the slow prototype-building loop’s repeated pandas filtering with a one-time groupby cache and avoid repeated PIL->numpy conversions. These changes are provably equivalent in semantics (same augmentations and averages), but drastically reduce Python overhead and disk I/O so the run fits under 600 seconds.'
- What this solution (achieved 0.61883) has done: 'Your current score suggests at least one of the two models is still running in the ImageNet-fallback path, and that path is intrinsically much weaker than using the intended 5-class cassava checkpoints. I make the smallest change that most directly improves accuracy: aggressively search for the `.pth` files inside the provided dataset directory (including `/kaggle/data/**`) and load them when found, while keeping your exact model definitions, TTA, and ensembling logic unchanged. I also make checkpoint loading tolerant to common key mismatches (`model.`, `module.`, etc.) so valid weights don’t get silently skipped due to naming differences. If checkpoints truly aren’t present, behavior remains exactly as before (ImageNet fallback + prototype mapping), ensuring the script still runs end-to-end and writes `submission.csv`.'
- What this solution (achieved 0.61211) has done: 'Your score gap suggests the intended cassava checkpoints still aren’t being found/loaded reliably, so the smallest change that should boost accuracy is to (1) resolve the competition root directory robustly from the known filesystem layout and (2) search specifically for the checkpoint basenames inside that root (and common Kaggle roots) before falling back to ImageNet. I also make checkpoint loading more tolerant by trying `strict=True` first and then `strict=False` with a clear warning if keys don’t match exactly, which prevents silently skipping usable weights. Finally, I keep your exact model definitions, TTA/ensemble logic, and submission semantics unchanged; the only behavior change is “use real 5-class weights if they exist,” which should move accuracy toward the target.'

# 9. Code solution

## === cell 0
import os
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch import nn
import torch.nn.functional as F

import albumentations as A
from albumentations.pytorch import ToTensorV2

import timm
import matplotlib.pyplot as plt



## === cell 1
INPUT_PATH = "../input/cassava-leaf-disease-classification/"
TRAIN_CSV_PATH = "../input/cassava-leaf-disease-classification/train.csv"
TRAIN_IMAGE_PATH = "../input/cassava-leaf-disease-classification/train_images/"
TEST_IMAGE_PATH = "../input/cassava-leaf-disease-classification/test_images/"
SUBMISSION_PATH = "submission.csv"

RESNEXT_PATH = "1022_res50.pth"
B4_PATH = "1022_b4ns.pth"

DEVICES = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
OUT_FEATURES = 5
NUM_EPOCHS = 17
BATCH_SIZE = 32
IMAGE_SIZE = 512
OPTIMIZER = torch.optim.AdamW
SEED = 42
LR_START = 1e-5
LR_MAX = 2e-4
LR_FINAL = 1e-5
TTA = 8

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_comp_root() -> str:
    """
    Improvement for score: checkpoint discovery often fails because INPUT_PATH points to a non-existent
    Kaggle mount in this environment. We resolve the actual competition root by checking the provided
    filesystem layout, which increases the chance we load the real 5-class .pth weights (biggest driver
    of accuracy toward the target).
    """
    candidates = [
        Path(INPUT_PATH),
        Path("/kaggle/input/cassava-leaf-disease-classification"),
        Path("/kaggle/data/cassava-leaf-disease-classification"),
        Path("../input/cassava-leaf-disease-classification"),
        Path("../data/cassava-leaf-disease-classification"),
        Path("..") / "input" / "cassava-leaf-disease-classification",
        Path("..") / "data" / "cassava-leaf-disease-classification",
    ]
    for c in candidates:
        if (c / "train.csv").is_file() and (c / "test_images").is_dir():
            return str(c)
    for root in [
        Path("/kaggle/data"),
        Path("/kaggle/input"),
        Path("../input"),
        Path("../data"),
        Path("."),
    ]:
        if root.exists():
            for p in root.rglob("train.csv"):
                if (
                    p.is_file()
                    and p.parent.name == "cassava-leaf-disease-classification"
                ):
                    return str(p.parent)
    return str(Path(INPUT_PATH))


COMP_ROOT = _resolve_comp_root()
INPUT_PATH = COMP_ROOT  # keep variable name, but point it to the resolved root

TRAIN_CSV_PATH = str(Path(COMP_ROOT) / "train.csv")
TRAIN_IMAGE_PATH = str(Path(COMP_ROOT) / "train_images")
TEST_IMAGE_PATH = str(Path(COMP_ROOT) / "test_images")


def _find_file_in_input(filename: str):
    """
    Improvement for score: aggressively find the .pth checkpoints by scanning the resolved competition
    root first, then common Kaggle roots; this directly increases odds we use real cassava weights
    rather than ImageNet fallback.
    """
    direct_candidates = [
        Path(COMP_ROOT) / filename,
        Path(COMP_ROOT) / "models" / filename,
        Path(COMP_ROOT) / "weights" / filename,
        Path("/kaggle/working") / filename,
        Path("/kaggle/data") / filename,
        Path("/kaggle/input") / filename,
        Path("../input") / filename,
    ]
    for c in direct_candidates:
        if c.is_file():
            return str(c)

    roots = [
        Path(COMP_ROOT),
        Path("/kaggle/data"),
        Path("/kaggle/input"),
        Path("/kaggle/working"),
        Path("../input"),
        Path("../data"),
    ]
    seen = set()
    for root in roots:
        if root.exists() and str(root) not in seen:
            seen.add(str(root))
            for p in root.rglob(filename):
                if p.is_file():
                    return str(p)
    return None


def _resolve_dir(path: str) -> str:
    """Make directory resolution robust to Kaggle dataset nesting."""
    p = Path(path)
    if p.is_dir():
        return str(p)

    p2 = Path(COMP_ROOT) / Path(path).name
    if p2.is_dir():
        return str(p2)

    roots = [
        Path(COMP_ROOT),
        Path("/kaggle/data"),
        Path("/kaggle/input"),
        Path("/kaggle/working"),
        Path("../input"),
        Path("../data"),
    ]
    for root in roots:
        if root.exists():
            for cand in root.rglob(Path(path).name):
                if cand.is_dir():
                    return str(cand)
    return str(p)  # let it fail loudly later if truly missing


TRAIN_IMAGE_PATH = _resolve_dir(TRAIN_IMAGE_PATH)
TEST_IMAGE_PATH = _resolve_dir(TEST_IMAGE_PATH)




## === cell 2
def sigmoid_focal_cross_entropy(y_hat, y_true, alpha=0.25, gamma=2.0):
    def smooth(y, smooth_factor):
        assert len(y.shape) == 2
        y = y * (1 - smooth_factor) + smooth_factor / y.shape[1]
        return y

    smooth_factor = 0.1

    if not isinstance(y_true, torch.Tensor):
        y_true = torch.tensor(y_true)
    if not isinstance(y_hat, torch.Tensor):
        y_hat = torch.tensor(y_hat)

    y_true = y_true.to(dtype=y_hat.dtype, device=y_hat.device)
    y_true = smooth(y_true, smooth_factor)

    bce = F.binary_cross_entropy_with_logits(y_hat, y_true, reduction="none")
    p = torch.sigmoid(y_hat)
    p_t = y_true * p + (1 - y_true) * (1 - p)

    alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
    modulating_factor = (1.0 - p_t).pow(gamma)

    return torch.sum(alpha_t * modulating_factor * bce, dim=-1)




## === cell 3
def lr_tune(epoch, num_epochs=NUM_EPOCHS):
    lr_start = LR_START
    lr_max = LR_MAX
    lr_final = LR_FINAL
    lr_warmup_epoch = 4
    lr_sustain_epoch = 0
    lr_decay_epoch = num_epochs - lr_warmup_epoch - lr_sustain_epoch - 1

    if epoch <= lr_warmup_epoch:
        lr = lr_start + (lr_max - lr_start) * (epoch / lr_warmup_epoch) ** 2.5
    elif epoch < lr_warmup_epoch + lr_sustain_epoch:
        lr = lr_max
    else:
        epoch_diff = epoch - lr_warmup_epoch - lr_sustain_epoch
        decay_factor = (epoch_diff / lr_decay_epoch) * math.pi
        decay_factor = (torch.cos(torch.tensor(decay_factor)).numpy() + 1) / 2
        lr = lr_final + (lr_max - lr_final) * decay_factor
    return lr


x = [i for i in range(NUM_EPOCHS)]
y = [lr_tune(i) for i in x]
plt.plot(x, y)



## === cell 4
train_augs = A.Compose(
    [
        A.RandomResizedCrop(size=(IMAGE_SIZE, IMAGE_SIZE)),
        A.Transpose(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(p=0.5),
        A.HueSaturationValue(
            hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5
        ),
        A.RandomBrightnessContrast(
            brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5
        ),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0,
        ),
        A.CoarseDropout(p=0.5),
        A.CoarseDropout(
            num_holes_range=(1, 1),
            hole_height_range=(0.1, 0.5),
            hole_width_range=(0.1, 0.5),
            fill=0,
            p=0.5,
        ),
        ToTensorV2(p=1.0),
    ],
    p=1.0,
)

valid_augs = A.Compose(
    [
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.CenterCrop(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]
)



## === cell 5
test_augs = A.Compose(
    [
        A.OneOf(
            [
                A.Resize(IMAGE_SIZE, IMAGE_SIZE, p=1.0),
                A.CenterCrop(IMAGE_SIZE, IMAGE_SIZE, p=1.0),
                A.RandomResizedCrop(size=(IMAGE_SIZE, IMAGE_SIZE), p=1.0),
            ],
            p=1.0,
        ),
        A.Transpose(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0,
        ),
        ToTensorV2(p=1.0),
    ],
    p=1.0,
)




## === cell 6
def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(SEED)



## === cell 7
model_name1 = "resnext50_32x4d"
my_model_1 = timm.create_model(model_name1, pretrained=False)
my_model_1.fc = nn.Linear(my_model_1.fc.in_features, OUT_FEATURES)
nn.init.xavier_uniform_(my_model_1.fc.weight)
if my_model_1.fc.bias is not None:
    nn.init.zeros_(my_model_1.fc.bias)
my_model_1



## === cell 8
model_name2 = "tf_efficientnet_b4_ns"
my_model_2 = timm.create_model(model_name2, pretrained=False)
my_model_2.classifier = nn.Linear(my_model_2.classifier.in_features, OUT_FEATURES)
nn.init.xavier_uniform_(my_model_2.classifier.weight)
if my_model_2.classifier.bias is not None:
    nn.init.zeros_(my_model_2.classifier.bias)
my_model_2



## === cell 9
torch.cuda.empty_cache()




## === cell 10
def _load_checkpoint_into_model(model, ckpt_path, strict=True):
    """
    Improvement for score: tolerate common checkpoint key prefixes and avoid skipping usable weights.
    We still try strict=True first (preferred), then strict=False only if needed.
    """
    model_param = torch.load(ckpt_path, map_location="cpu")
    if isinstance(model_param, dict) and "state_dict" in model_param:
        model_param = model_param["state_dict"]

    if not isinstance(model_param, dict):
        raise ValueError("Unsupported checkpoint format")

    sd = model_param
    cleaned = {}
    for k, v in sd.items():
        if not torch.is_tensor(v):
            continue
        nk = k
        for pref in ("module.", "model.", "net.", "encoder."):
            if nk.startswith(pref):
                nk = nk[len(pref) :]
        cleaned[nk] = v

    try:
        model.load_state_dict(cleaned, strict=strict)
    except RuntimeError as e:
        if strict:
            print(f"[WARN] strict=True load failed for {ckpt_path}: {e}")
            print(
                "[WARN] Retrying with strict=False to load matching keys only (keeps core logic)."
            )
            model.load_state_dict(cleaned, strict=False)
        else:
            raise


resnext_ckpt = _find_file_in_input(RESNEXT_PATH)
b4_ckpt = _find_file_in_input(B4_PATH)

using_imagenet_fallback_1 = resnext_ckpt is None
using_imagenet_fallback_2 = b4_ckpt is None

if using_imagenet_fallback_1:
    my_model_1 = timm.create_model(model_name1, pretrained=True)  # keep original head
if using_imagenet_fallback_2:
    my_model_2 = timm.create_model(model_name2, pretrained=True)  # keep original head




## === cell 11
def _build_imagenet_prototypes(
    model: nn.Module,
    model_is_dataparallel: bool,
    train_csv_path: str,
    train_image_dir: str,
    augs: A.Compose,
    num_classes: int = 5,
    max_per_class: int = 64,
    tta: int = 4,
    device: torch.device = DEVICE,
    seed: int = 42,
):
    df = pd.read_csv(train_csv_path)
    grouped = df.groupby("label")["image_id"].apply(np.array).to_dict()

    rng = np.random.default_rng(seed)
    proto = torch.zeros(num_classes, 1000, dtype=torch.float32)
    counts = torch.zeros(num_classes, dtype=torch.int64)

    model.eval()
    for c in range(num_classes):
        sub = grouped.get(c, None)
        if sub is None or len(sub) == 0:
            continue
        take = min(max_per_class, len(sub))
        idx = rng.choice(len(sub), size=take, replace=False)
        chosen = sub[idx]

        for image_id in chosen:
            img_path = os.path.join(train_image_dir, image_id)
            if not os.path.isfile(img_path):
                continue
            try:
                image_np = np.array(Image.open(img_path).convert("RGB"))
            except Exception:
                continue

            with torch.no_grad():
                acc = torch.zeros(1000, device=device, dtype=torch.float32)
                for _ in range(tta):
                    aug_image = augs(image=image_np)["image"]
                    x = aug_image.unsqueeze(0).to(device=device, dtype=torch.float32)
                    out = model(x)
                    out = out.view(-1)
                    if out.numel() != 1000:
                        raise ValueError(
                            f"Expected 1000 ImageNet logits in fallback, got {out.numel()}"
                        )
                    acc += out.float()
                acc /= float(tta)
                proto[c] += acc.detach().cpu()
                counts[c] += 1

    for c in range(num_classes):
        if counts[c] > 0:
            proto[c] /= counts[c].item()

    proto = F.normalize(proto, p=2, dim=1)
    return proto  # [5,1000] on CPU float32


def _to_5_logits_with_proto(
    logits: torch.Tensor, proto: torch.Tensor | None
) -> torch.Tensor:
    """If proto is provided, map [B,1000] -> [B,5] via cosine-similarity logits."""
    if proto is None:
        return logits
    x = F.normalize(logits, p=2, dim=1)
    W = proto.to(device=logits.device, dtype=logits.dtype)  # [5,1000]
    return x @ W.T  # [B,5]


proto1 = None
proto2 = None

my_model_1 = nn.DataParallel(my_model_1).to(DEVICE)
my_model_2 = nn.DataParallel(my_model_2).to(DEVICE)

if (resnext_ckpt is None) and using_imagenet_fallback_1:
    proto1 = _build_imagenet_prototypes(
        model=my_model_1,
        model_is_dataparallel=True,
        train_csv_path=TRAIN_CSV_PATH,
        train_image_dir=TRAIN_IMAGE_PATH,
        augs=valid_augs,
        num_classes=OUT_FEATURES,
        max_per_class=64,
        tta=4,
        device=DEVICE,
        seed=SEED + 101,
    )

if (b4_ckpt is None) and using_imagenet_fallback_2:
    proto2 = _build_imagenet_prototypes(
        model=my_model_2,
        model_is_dataparallel=True,
        train_csv_path=TRAIN_CSV_PATH,
        train_image_dir=TRAIN_IMAGE_PATH,
        augs=valid_augs,
        num_classes=OUT_FEATURES,
        max_per_class=64,
        tta=4,
        device=DEVICE,
        seed=SEED + 202,
    )




## === cell 12
def _iter_test_batches(image_names, image_dir, batch_size):
    for i in range(0, len(image_names), batch_size):
        batch_names = image_names[i : i + batch_size]
        batch_np = []
        for n in batch_names:
            img_path = os.path.join(image_dir, n)
            batch_np.append(np.array(Image.open(img_path).convert("RGB")))
        yield batch_names, batch_np


def _predict_with_tta_batched(
    model: nn.Module,
    image_names,
    image_dir: str,
    augs: A.Compose,
    out_features: int,
    tta: int,
    proto: torch.Tensor | None,
    device: torch.device,
    batch_size: int,
):
    model.eval()
    all_preds = []

    with torch.no_grad():
        for _, batch_np in _iter_test_batches(image_names, image_dir, batch_size):
            bs = len(batch_np)
            ans = torch.zeros(bs, out_features, device=device, dtype=torch.float32)

            for _ in range(tta):
                x = torch.stack([augs(image=im)["image"] for im in batch_np], dim=0).to(
                    device=device, dtype=torch.float32
                )
                raw = model(x)
                raw = _to_5_logits_with_proto(raw, proto)
                ans += raw.view(bs, out_features)

            ans /= float(tta)
            all_preds.append(ans.detach().cpu())

    return torch.cat(all_preds, dim=0)


test_image_list = sorted(
    [
        image_name
        for image_name in os.listdir(TEST_IMAGE_PATH)
        if image_name.lower().endswith(".jpg")
    ]
)

my_model_1.eval()
if resnext_ckpt is not None:
    _load_checkpoint_into_model(my_model_1.module, resnext_ckpt, strict=True)

tta1 = TTA if using_imagenet_fallback_1 else 1
predictions_1 = _predict_with_tta_batched(
    model=my_model_1,
    image_names=test_image_list,
    image_dir=TEST_IMAGE_PATH,
    augs=test_augs,
    out_features=OUT_FEATURES,
    tta=tta1,
    proto=proto1,
    device=DEVICE,
    batch_size=BATCH_SIZE,
)
normalize_pred_1 = F.normalize(predictions_1.T, p=2, dim=0).T
torch.cuda.empty_cache()

my_model_2.eval()
if b4_ckpt is not None:
    _load_checkpoint_into_model(my_model_2.module, b4_ckpt, strict=True)

predictions_2 = _predict_with_tta_batched(
    model=my_model_2,
    image_names=test_image_list,
    image_dir=TEST_IMAGE_PATH,
    augs=test_augs,
    out_features=OUT_FEATURES,
    tta=TTA,
    proto=proto2,
    device=DEVICE,
    batch_size=BATCH_SIZE,
)
normalize_pred_2 = F.normalize(predictions_2.T, p=2, dim=0).T

final_pred = (normalize_pred_1 * 0.475) + (normalize_pred_2 * 0.525)
label = final_pred.argmax(dim=-1).numpy().astype(int)

df_submission = pd.DataFrame({"image_id": test_image_list, "label": label})
df_submission.to_csv(SUBMISSION_PATH, index=False)

print(f"Wrote submission to: {SUBMISSION_PATH}")
print(df_submission.head())
print("Num rows:", len(df_submission))
print("Resolved COMP_ROOT:", COMP_ROOT)
print(
    "Fallbacks used:",
    {
        "resnext_imagenet_fallback": using_imagenet_fallback_1,
        "b4_imagenet_fallback": using_imagenet_fallback_2,
    },
)
print(
    "Prototype mapping used:",
    {
        "resnext_proto": proto1 is not None,
        "b4_proto": proto2 is not None,
    },
)
print(
    "Checkpoints resolved:",
    {
        "resnext_ckpt": resnext_ckpt,
        "b4_ckpt": b4_ckpt,
    },
)
