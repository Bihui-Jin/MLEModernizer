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

0.8964944091870656

# 6. Current score

0.12257

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.10987) has done: 'I fix the Albumentations v2 API break by updating `RandomResizedCrop` calls to use the new `size=(h, w)` signature, which resolves the current runtime exceptions. Then I make the checkpoint loading robust to missing files by automatically searching common Kaggle input locations for the `.pth` weights and, if they truly don’t exist, falling back to running with randomly initialized weights (still producing a valid submission). I also fix device handling so the code runs on either GPU or CPU without crashing, and ensure the submission uses the exact `sample_submission.csv` ordering to avoid any image_id misalignment. These changes are execution-unblocking and score-neutral when weights are found (and are the only way to yield a valid CSV when they are not).'
- What this solution (achieved 0.11099) has done: 'Your low score strongly suggests the intended pretrained checkpoints are not being found/loaded, so the ensemble is effectively running with random weights. I make the checkpoint discovery stricter and more compatible by (1) auto-detecting the real dataset root under `/kaggle/input` or your provided `/kaggle/data/input` mirror, and (2) searching for `.pth` files by basename across those roots (including the common “dataset subfolder inside dataset” nesting). I also fix a subtle but important issue: `eval()` is currently called before wrapping in `DataParallel`, so the actually-used model can remain in train mode; moving `eval()` after `.to(device)` ensures deterministic inference behavior consistent with the saved weights. These are minimal, execution-safe changes that keep the same inference logic but should raise accuracy substantially when the correct weights are present.'
- What this solution (achieved 0.11136) has done: 'Your score (~0.11) strongly indicates the `.pth` checkpoints still aren’t being found/loaded, so inference is effectively random. I make checkpoint discovery deterministic and much more likely to succeed by (1) searching explicitly for any `.pth` under all Kaggle input roots and picking the best match for each expected checkpoint name, and (2) verifying the loaded state_dict has meaningful overlap with the model (otherwise treat as not-loaded). I also fix the (currently incorrect) focal-loss helper `p_t` computation even though it isn’t used, to prevent accidental future use from silently breaking training semantics. Core model architectures, inference loop, TTA, and ensembling logic remain unchanged; this only aims to ensure the intended trained weights are actually used.'
- What this solution (achieved 0.59828) has done: 'I fix the execution-blocking error by making the code robust when the expected `.pth` ensemble checkpoints are not present in this Kaggle environment (your file tree shows only the competition dataset, not the extra `ensemble-1023` dataset). Specifically, I (1) add a deterministic fallback to use `timm` pretrained ImageNet weights when the custom checkpoints can’t be found, (2) keep the same architectures and inference/ensemble logic, and (3) ensure the submission is written in exactly the `sample_submission.csv` order with the required `.csv` suffix. This should yield a valid submission end-to-end and materially increase accuracy from the “random weights” behavior, while keeping changes minimal and score-directed.'
- What this solution (achieved 0.12257) has done: 'Your current score suggests the custom fine-tuned checkpoints still aren’t being used, so the random re-initialized classifier heads dominate and accuracy stays low. The smallest score-improving change that preserves your core inference/ensemble logic is: when a checkpoint is missing, don’t random-init the head; keep the model’s pretrained classifier weights (ImageNet-1k) and map its 1000-class logits to your 5 cassava classes via a fixed random projection matrix (no training, same argmax submission semantics). This keeps everything inference-only, avoids changing your architectures/loops/augmentations, and should materially improve accuracy versus a random 5-class head while staying within the 600s limit. I also ensure the projection is deterministic via the existing global seed so results are stable.'

# 9. Code solution

## === cell 0
import os
import math
import random
import warnings

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch import nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import timm

warnings.filterwarnings("ignore")




## === cell 1
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


def get_device():
    return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def resolve_first_existing(paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


def detect_kaggle_input_roots():
    roots = [
        "/kaggle/input",
        "/kaggle/data/input",
        "/kaggle/data",
        "/kaggle/working",
        "../input",
        ".",
        "../kaggle/data/input",
        "../kaggle/data",
    ]
    out = []
    for r in roots:
        if os.path.isdir(r):
            out.append(os.path.abspath(r))
    seen = set()
    uniq = []
    for r in out:
        if r not in seen:
            uniq.append(r)
            seen.add(r)
    return uniq


def list_all_weight_files(search_roots, max_files=60000):
    out = []
    seen = set()
    for root in search_roots:
        if not os.path.isdir(root):
            continue
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                lfn = fn.lower()
                if lfn.endswith(".pth") or lfn.endswith(".pt"):
                    p = os.path.join(dirpath, fn)
                    if p not in seen:
                        out.append(p)
                        seen.add(p)
                        if len(out) >= max_files:
                            return out
    return out


def find_file(filename, search_roots):
    cand_names = [filename]
    base, ext = os.path.splitext(filename)
    if ext.lower() == ".pth":
        cand_names.append(base + ".pt")
    elif ext.lower() == ".pt":
        cand_names.append(base + ".pth")

    for root in search_roots:
        for name in cand_names:
            cand = os.path.join(root, name)
            if os.path.isfile(cand):
                return cand

    for root in search_roots:
        if os.path.isdir(root):
            for dirpath, _, filenames in os.walk(root):
                for name in cand_names:
                    if name in filenames:
                        return os.path.join(dirpath, name)
    return None


def detect_dataset_root():
    candidates = [
        "../input/cassava-leaf-disease-classification",
        "/kaggle/input/cassava-leaf-disease-classification",
        "/kaggle/data/input/cassava-leaf-disease-classification",
        "/kaggle/data/cassava-leaf-disease-classification",
        "/kaggle/data/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
        "/kaggle/data/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
        "../kaggle/data/input/cassava-leaf-disease-classification",
        "../kaggle/data/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
    ]
    root = resolve_first_existing(candidates)
    if root is not None:
        return os.path.abspath(root)

    for base in detect_kaggle_input_roots():
        try:
            for dirpath, _, filenames in os.walk(base):
                if "train.csv" in filenames and "sample_submission.csv" in filenames:
                    return os.path.abspath(dirpath)
        except Exception:
            continue
    return None


def detect_ensemble_root(preferred_subdir="ensemble-1023"):
    candidates = [
        f"../input/{preferred_subdir}/",
        f"/kaggle/input/{preferred_subdir}/",
        f"/kaggle/data/input/{preferred_subdir}/",
        f"../kaggle/data/input/{preferred_subdir}/",
        f"/kaggle/data/{preferred_subdir}/",
    ]
    for c in candidates:
        if os.path.isdir(c):
            return os.path.abspath(c) + "/"
    return None




## === cell 2
INPUT_PATH = "../input/ensemble-1023/"
DATASET_ROOT = detect_dataset_root()

TRAIN_CSV_PATH = "../input/cassava-leaf-disease-classification/train.csv"
TRAIN_IMAGE_PATH = "../input/cassava-leaf-disease-classification/train_images/"
TEST_IMAGE_PATH = "../input/cassava-leaf-disease-classification/test_images/"
SAMPLE_SUB_PATH = "../input/cassava-leaf-disease-classification/sample_submission.csv"

SUBMISSION_PATH = "submission.csv"
RESNEXT_PATH = "1022_res50.pth"
B4_PATH = "1022_b4ns.pth"

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

device = get_device()
DEVICES = [device]

_auto_ensemble_root = detect_ensemble_root("ensemble-1023")
if _auto_ensemble_root is not None:
    INPUT_PATH = _auto_ensemble_root

if DATASET_ROOT is not None:
    TRAIN_CSV_PATH = os.path.join(DATASET_ROOT, "train.csv")
    TRAIN_IMAGE_PATH = os.path.join(DATASET_ROOT, "train_images") + "/"
    TEST_IMAGE_PATH = os.path.join(DATASET_ROOT, "test_images") + "/"
    SAMPLE_SUB_PATH = os.path.join(DATASET_ROOT, "sample_submission.csv")

print(f"[INFO] device: {device}")
print(f"[INFO] DATASET_ROOT: {DATASET_ROOT}")
print(f"[INFO] Using INPUT_PATH for checkpoints (if exists): {INPUT_PATH}")
print(f"[INFO] TRAIN_CSV_PATH: {TRAIN_CSV_PATH}")
print(f"[INFO] TEST_IMAGE_PATH: {TEST_IMAGE_PATH}")
print(f"[INFO] SAMPLE_SUB_PATH: {SAMPLE_SUB_PATH}")
print(f"[INFO] Kaggle input roots detected: {detect_kaggle_input_roots()}")




## === cell 3
seed_everything(SEED)




## === cell 4
def sigmoid_focal_cross_entropy(y_hat, y_true, alpha=0.25, gamma=2.0):
    def smooth(y, smooth_factor):
        assert len(y.shape) == 2
        y *= 1 - smooth_factor
        y += smooth_factor / y.shape[1]
        return y

    smooth_factor = 0.1

    if not isinstance(y_true, torch.Tensor):
        y_true = torch.tensor(y_true)
    if not isinstance(y_hat, torch.Tensor):
        y_hat = torch.tensor(y_hat)

    y_true = smooth(y_true, smooth_factor)

    cross_entropy = F.binary_cross_entropy_with_logits(y_hat, y_true, reduction="none")

    p = torch.sigmoid(y_hat)
    p_t = y_true * p + (1 - y_true) * (1 - p)

    alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
    modulating_factor = (1.0 - p_t).pow(gamma)

    return torch.sum(alpha_t * modulating_factor * cross_entropy, dim=-1)




## === cell 5
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




## === cell 6
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




## === cell 7
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




## === cell 8
def make_fixed_projection(in_dim, out_dim, seed=42, device="cpu"):
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    W = torch.randn(in_dim, out_dim, generator=g, dtype=torch.float32)
    W = W / (W.norm(dim=0, keepdim=True) + 1e-6)
    return W.to(device)


class LogitProjector(nn.Module):
    def __init__(self, base_model: nn.Module, proj: torch.Tensor):
        super().__init__()
        self.base_model = base_model
        self.register_buffer("proj", proj)  # [in_dim, out_dim]

    def forward(self, x):
        logits = self.base_model(x)  # [B, in_dim]
        return logits @ self.proj  # [B, out_dim]




## === cell 9
model_name1 = "resnext50_32x4d"
my_model_1 = timm.create_model(model_name1, pretrained=True)
_model1_imagenet_dim = my_model_1.fc.out_features if hasattr(my_model_1, "fc") else 1000
my_model_1.fc = nn.Linear(my_model_1.fc.in_features, OUT_FEATURES)
nn.init.xavier_uniform_(my_model_1.fc.weight)
if my_model_1.fc.bias is not None:
    nn.init.zeros_(my_model_1.fc.bias)
my_model_1




## === cell 10
model_name2 = "tf_efficientnet_b4_ns"
my_model_2 = timm.create_model(model_name2, pretrained=True)
_model2_imagenet_dim = (
    my_model_2.classifier.out_features if hasattr(my_model_2, "classifier") else 1000
)
my_model_2.classifier = nn.Linear(my_model_2.classifier.in_features, OUT_FEATURES)
nn.init.xavier_uniform_(my_model_2.classifier.weight)
if my_model_2.classifier.bias is not None:
    nn.init.zeros_(my_model_2.classifier.bias)
my_model_2




## === cell 11
def load_checkpoint_into_model(model, ckpt_filename):
    search_roots = []

    if INPUT_PATH and os.path.isdir(INPUT_PATH):
        search_roots.append(INPUT_PATH)

    if DATASET_ROOT and os.path.isdir(DATASET_ROOT):
        search_roots.append(DATASET_ROOT)

    search_roots.extend(detect_kaggle_input_roots())
    search_roots.append(os.getcwd())

    nested = []
    for r in list(search_roots):
        if os.path.isdir(r):
            try:
                for name in os.listdir(r):
                    p = os.path.join(r, name)
                    if os.path.isdir(p):
                        nested.append(p)
            except Exception:
                pass
    search_roots = search_roots + nested

    seen = set()
    uniq = []
    for r in search_roots:
        ar = os.path.abspath(r)
        if ar not in seen and os.path.isdir(ar):
            uniq.append(ar)
            seen.add(ar)
    search_roots = uniq

    all_wts = list_all_weight_files(search_roots)
    ckpt_path = find_file(ckpt_filename, search_roots)

    if ckpt_path is None and len(all_wts) > 0:
        target_base = (
            os.path.splitext(os.path.basename(ckpt_filename))[0]
            .lower()
            .replace("-", "_")
        )
        tokens = [t for t in target_base.split("_") if t]
        best = None
        best_score = -1
        for p in all_wts:
            fn_base = os.path.splitext(os.path.basename(p))[0].lower().replace("-", "_")
            score = sum(1 for t in tokens if t in fn_base)
            if score > best_score:
                best_score = score
                best = p
        ckpt_path = best if best_score > 0 else None

    if ckpt_path is None:
        print(f"[WARN] Checkpoint not found anywhere: {ckpt_filename}")
        if len(all_wts) > 0:
            print(
                f"[WARN] Discovered weight files (.pth/.pt) (showing up to 30 of {len(all_wts)}):"
            )
            for p in all_wts[:30]:
                print("  ", p)
        return False

    state = torch.load(ckpt_path, map_location="cpu")
    if (
        isinstance(state, dict)
        and "state_dict" in state
        and isinstance(state["state_dict"], dict)
    ):
        state = state["state_dict"]

    if not isinstance(state, dict):
        print(
            f"[WARN] Unrecognized checkpoint format for {ckpt_filename} (loaded {type(state)})."
        )
        return False

    new_state = {}
    for k, v in state.items():
        if isinstance(k, str) and k.startswith("module."):
            new_state[k[7:]] = v
        else:
            new_state[k] = v

    missing, unexpected = model.load_state_dict(new_state, strict=False)

    total_params = len(model.state_dict())
    loaded_params = total_params - len(missing)
    if loaded_params <= max(5, int(0.05 * total_params)):
        print(
            f"[WARN] Checkpoint '{ckpt_path}' has too little overlap with the model "
            f"(loaded {loaded_params}/{total_params}); treating as not-loaded."
        )
        return False

    print(f"[INFO] Loaded checkpoint: {ckpt_path}")
    if len(missing) > 0 or len(unexpected) > 0:
        print(
            f"[INFO] Loaded with strict=False. Missing: {len(missing)}, Unexpected: {len(unexpected)}"
        )
    return True




## === cell 12
torch.cuda.empty_cache()

sample_df = pd.read_csv(SAMPLE_SUB_PATH)
test_image_list = sample_df["image_id"].astype(str).values

preds_1 = []
loaded1 = load_checkpoint_into_model(my_model_1, RESNEXT_PATH)
if not loaded1:
    print(
        f"[WARN] Proceeding without '{RESNEXT_PATH}' custom weights; using ImageNet logits -> fixed projection to 5 classes."
    )
    base1 = timm.create_model(model_name1, pretrained=True)
    proj1 = make_fixed_projection(
        in_dim=base1.fc.out_features,
        out_dim=OUT_FEATURES,
        seed=SEED + 101,
        device=device,
    )
    my_model_1 = LogitProjector(base1, proj1)

if torch.cuda.device_count() > 1 and device.type == "cuda":
    my_model_1 = nn.DataParallel(my_model_1).to(device)
else:
    my_model_1 = my_model_1.to(device)
my_model_1.eval()

for single_image_name in tqdm(
    test_image_list, desc="Predict model_1", total=len(test_image_list)
):
    with torch.no_grad():
        ans = torch.zeros(OUT_FEATURES, device=device)
        for _ in range(3):
            image = Image.open(
                os.path.join(TEST_IMAGE_PATH, single_image_name)
            ).convert("RGB")
            aug_image = test_augs(image=np.array(image))["image"]
            test_image = aug_image.unsqueeze(0).to(device)
            ans += my_model_1(test_image).view(ans.shape)
        ans /= 3
        preds_1.append(ans.detach().cpu())

predictions_1 = torch.stack(preds_1, dim=0)
normalize_pred_1 = F.normalize(predictions_1.T, p=2, dim=0).T
torch.cuda.empty_cache()

preds_2 = []
loaded2 = load_checkpoint_into_model(my_model_2, B4_PATH)
if not loaded2:
    print(
        f"[WARN] Proceeding without '{B4_PATH}' custom weights; using ImageNet logits -> fixed projection to 5 classes."
    )
    base2 = timm.create_model(model_name2, pretrained=True)
    proj2 = make_fixed_projection(
        in_dim=base2.classifier.out_features,
        out_dim=OUT_FEATURES,
        seed=SEED + 202,
        device=device,
    )
    my_model_2 = LogitProjector(base2, proj2)

if torch.cuda.device_count() > 1 and device.type == "cuda":
    my_model_2 = nn.DataParallel(my_model_2).to(device)
else:
    my_model_2 = my_model_2.to(device)
my_model_2.eval()

for single_image_name in tqdm(
    test_image_list, desc="Predict model_2", total=len(test_image_list)
):
    with torch.no_grad():
        ans = torch.zeros(OUT_FEATURES, device=device)
        for _ in range(TTA):
            image = Image.open(
                os.path.join(TEST_IMAGE_PATH, single_image_name)
            ).convert("RGB")
            aug_image = test_augs(image=np.array(image))["image"]
            test_image = aug_image.unsqueeze(0).to(device)
            ans += my_model_2(test_image).view(ans.shape)
        ans /= TTA
        preds_2.append(ans.detach().cpu())

predictions_2 = torch.stack(preds_2, dim=0)
normalize_pred_2 = F.normalize(predictions_2.T, p=2, dim=0).T

final_pred = (normalize_pred_1 * 0.45) + (normalize_pred_2 * 0.55)
label = final_pred.argmax(dim=-1).numpy().astype(int)

unique, counts = np.unique(label, return_counts=True)
dist = {int(k): int(v) for k, v in zip(unique, counts)}
print(f"[INFO] Predicted label distribution: {dist}")

df_submission = pd.DataFrame({"image_id": test_image_list, "label": label})
df_submission.to_csv(SUBMISSION_PATH, index=False)
print(
    f"Saved submission to: {SUBMISSION_PATH} with shape {df_submission.shape} and columns {list(df_submission.columns)}"
)
print(df_submission.head())
