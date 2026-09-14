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

0.8955877908733756

# 6. Current score

0.13266

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.13528) has done: 'I fix the Albumentations v2 API breakage by updating `RandomResizedCrop` usage so the augmentations can be constructed without validation errors. I also fix the missing weights path issue by detecting whether the `.pth` files exist; if they do, the code use them exactly as intended, and if not, it fall back to running the same inference pipeline with randomly initialized models so a valid `submission.csv` is still produced. Additionally, I make the submission creation use `sample_submission.csv` (not `train.csv`) to guarantee correct columns and row order, and I sort/test-align predictions by `image_id` to prevent mismatches. These changes are minimal, unblock end-to-end execution, and should allow you to get a valid Kaggle submission; if weights are present, score behavior should match the original logic.'
- What this solution (achieved 0.13677) has done: 'Your current score (0.13528) is far below the target (0.8956), and the biggest likely reason is that your code is running with randomly initialized models because the weight files aren’t being found/loaded. I make a minimal, score-relevant change to correctly discover the real dataset root in this environment (`/kaggle/data/...` as shown in your paths) and to search a few plausible locations for the two `.pth` files, so your intended pretrained ensemble actually loads. I also fix the weight-loading logic so it robustly handles both “plain state_dict” and “checkpoint dict with state_dict” formats, without changing the model architecture or inference semantics. With weights correctly loaded, your existing inference + TTA + ensembling should move accuracy sharply upward toward your target.'
- What this solution (achieved 0.13378) has done: 'I fix the crash by making weight discovery robust to this environment’s actual dataset root and by no longer hard-failing before a submission is written. The core inference logic (two timm models, TTA, normalization, weighted ensemble, argmax) is kept identical; the only score-relevant change is ensuring the intended `.pth` weights are actually found and loaded when present. If weights still cannot be located, the code produce a valid `submission.csv` using the same pipeline (but warn you that the score be low). I also guard against missing/empty test image lists and ensure `test_image_list`/`label` exist before building the submission.'
- What this solution (achieved 0.13117) has done: 'Your score is far below the target, which strongly suggests your intended pretrained weights still aren’t being loaded correctly (so you’re effectively predicting with random heads). I make minimal, score-relevant fixes to (1) resolve the dataset/weights roots for this environment (your files are under `/kaggle/data/...`, not `../input/...`), and (2) make weight loading robust to common checkpoint formats and key prefixes so the full model (including the classifier head) actually loads. I not change the model architectures, augmentations, TTA, ensembling, or post-processing; only path/weight-loading correctness and strict verification are adjusted. If weights truly don’t exist anywhere, you still get a valid `submission.csv`, but the code clearly report that it’s running unweighted.'
- What this solution (achieved 0.13154) has done: 'Your gap to the target is very large, and with this exact pipeline the most likely cause is still that the intended `.pth` weights aren’t being found/loaded, leaving you with near-random predictions. I make a minimal, score-relevant change to (1) search for the weight files more robustly by also glob-searching common Kaggle locations (including under `/kaggle/data/**` and `/kaggle/input/**`), and (2) load checkpoints more faithfully by preferring an exact key-match load (and only then falling back to filtered loading) while still verifying the classifier head is loaded. I keep your models, augmentations, TTA, ensembling, and submission formatting the same, and I also print a short diagnostic about whether head weights actually changed from initialization (so you can confirm you’re not silently running unweighted). This should move the accuracy sharply upward toward your target if the weights exist anywhere in the environment.'
- What this solution (achieved 0.13266) has done: 'Your current score is extremely low relative to the target, which strongly indicates the pretrained `.pth` weights still aren’t being loaded (so predictions are near-random). I make the smallest score-relevant changes to (1) resolve the correct dataset root (`/kaggle/data/...` in your environment) and (2) search for the weight files more effectively, including inside `/kaggle/working` (where uploaded datasets/models often land) and common “input dataset” locations. I also fix the weight loading to correctly handle checkpoints saved from `DataParallel` vs non-`DataParallel` by loading into the *base* model before wrapping, avoiding key-mismatch pitfalls while keeping the same model architecture and inference logic. Everything else (TTA, normalization, ensembling, argmax, submission formatting) stays the same.'

# 9. Code solution

## === cell 0
import os
import math
import random
import warnings
import glob

import numpy as np
import pandas as pd

import torch
from torch import nn
import torch.nn.functional as F

import albumentations as A
from albumentations.pytorch import ToTensorV2

from PIL import Image
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




## === cell 2
INPUT_PATH = "../input/ensemble-1023/"
TRAIN_CSV_PATH = "../input/cassava-leaf-disease-classification/train.csv"
TRAIN_IMAGE_PATH = "../input/cassava-leaf-disease-classification/train_images/"
TEST_IMAGE_PATH = "../input/cassava-leaf-disease-classification/test_images/"
SAMPLE_SUB_PATH = "../input/cassava-leaf-disease-classification/sample_submission.csv"

SUBMISSION_PATH = "submission.csv"
RESNEXT_PATH = "1022_res50.pth"
B4_PATH = "1022_b4ns.pth"

DEVICES = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
DEVICE = DEVICES[0] if len(DEVICES) else torch.device("cpu")

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

seed_everything(SEED)




## === cell 3
def _pick_existing_path(candidates, kind="file"):
    for p in candidates:
        if kind == "file" and os.path.isfile(p):
            return p
        if kind == "dir" and os.path.isdir(p):
            return p
    return None


TEST_IMAGE_PATH = (
    _pick_existing_path(
        [
            "/kaggle/data/cassava-leaf-disease-classification/test_images",
            "/kaggle/data/cassava-leaf-disease-classification/cassava-leaf-disease-classification/test_images",
            "/kaggle/data/input/cassava-leaf-disease-classification/test_images",
            "/kaggle/data/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification/test_images",
            "/kaggle/input/cassava-leaf-disease-classification/test_images",
            "/kaggle/input/cassava-leaf-disease-classification/test_images/test_images",
            TEST_IMAGE_PATH,
        ],
        kind="dir",
    )
    or TEST_IMAGE_PATH
)

SAMPLE_SUB_PATH = (
    _pick_existing_path(
        [
            "/kaggle/data/cassava-leaf-disease-classification/sample_submission.csv",
            "/kaggle/data/cassava-leaf-disease-classification/cassava-leaf-disease-classification/sample_submission.csv",
            "/kaggle/data/input/cassava-leaf-disease-classification/sample_submission.csv",
            "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv",
            SAMPLE_SUB_PATH,
        ],
        kind="file",
    )
    or SAMPLE_SUB_PATH
)

print("Resolved TEST_IMAGE_PATH:", TEST_IMAGE_PATH)
print("Resolved SAMPLE_SUB_PATH:", SAMPLE_SUB_PATH)
print("DEVICE:", DEVICE)




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
    p_t = y_true * y_hat + (1 - y_true) * (1 - y_hat)
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



## === cell 7
model_name1 = "resnext50_32x4d"
my_model_1 = timm.create_model(model_name1, pretrained=False)
my_model_1.fc = nn.Linear(my_model_1.fc.in_features, OUT_FEATURES)
nn.init.xavier_uniform_(my_model_1.fc.weight)
if my_model_1.fc.bias is not None:
    nn.init.zeros_(my_model_1.fc.bias)

model_name2 = "tf_efficientnet_b4_ns"
my_model_2 = timm.create_model(model_name2, pretrained=False)
my_model_2.classifier = nn.Linear(my_model_2.classifier.in_features, OUT_FEATURES)
nn.init.xavier_uniform_(my_model_2.classifier.weight)
if my_model_2.classifier.bias is not None:
    nn.init.zeros_(my_model_2.classifier.bias)




## === cell 8
def _find_weight_file(filename, search_roots):
    for root in search_roots:
        if root is None:
            continue
        cand = os.path.join(root, filename)
        if os.path.isfile(cand):
            return cand

    glob_patterns = []
    for root in search_roots:
        if root is None:
            continue
        glob_patterns.append(os.path.join(root, "**", filename))
    glob_patterns += [
        os.path.join("/kaggle/input", "**", filename),
        os.path.join("/kaggle/data", "**", filename),
        os.path.join("/kaggle/working", "**", filename),
    ]

    for pat in glob_patterns:
        matches = glob.glob(pat, recursive=True)
        if matches:
            matches = sorted(matches, key=lambda p: (len(p), p))
            for m in matches:
                if os.path.isfile(m):
                    return m

    for root in search_roots:
        if root is None or (not os.path.isdir(root)):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            if filename in filenames:
                return os.path.join(dirpath, filename)
            rel = os.path.relpath(dirpath, root)
            if rel.count(os.sep) >= 6:
                dirnames[:] = []
    return None


def _extract_state_dict(obj):
    if isinstance(obj, dict):
        for k in [
            "state_dict",
            "model_state_dict",
            "model",
            "net",
            "ema",
            "student",
            "teacher",
        ]:
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
        return obj
    return None


def _strip_prefix_if_present(state, prefix):
    if not isinstance(state, dict):
        return state
    if not any(k.startswith(prefix) for k in state.keys()):
        return state
    return {k[len(prefix) :]: v for k, v in state.items()}


def _add_prefix_if_missing(state, prefix):
    if not isinstance(state, dict):
        return state
    if any(k.startswith(prefix) for k in state.keys()):
        return state
    return {prefix + k: v for k, v in state.items()}


def try_load_weights_strict_head(model, weight_path, head_key_candidates):
    if (weight_path is None) or (not os.path.isfile(weight_path)):
        return False, "missing weight file"

    obj = torch.load(weight_path, map_location="cpu")
    state = _extract_state_dict(obj)
    if not isinstance(state, dict):
        return False, "invalid checkpoint format"

    model_state = model.state_dict()

    has_module_in_model = any(k.startswith("module.") for k in model_state.keys())
    has_module_in_ckpt = any(k.startswith("module.") for k in state.keys())
    if has_module_in_ckpt and (not has_module_in_model):
        state = _strip_prefix_if_present(state, "module.")
    elif (not has_module_in_ckpt) and has_module_in_model:
        state = _add_prefix_if_missing(state, "module.")

    missing, unexpected = model.load_state_dict(state, strict=False)

    ok = False
    for hk in head_key_candidates:
        if hk in state and hk in model_state:
            v = state[hk]
            if hasattr(v, "shape") and tuple(v.shape) == tuple(model_state[hk].shape):
                ok = True
                break

    if ok:
        return (
            True,
            f"loaded direct; missing={len(missing)} unexpected={len(unexpected)}",
        )

    new_state = {}
    for k, v in state.items():
        if k in model_state and hasattr(v, "shape") and v.shape == model_state[k].shape:
            new_state[k] = v

    missing2, unexpected2 = model.load_state_dict(new_state, strict=False)

    ok2 = False
    for hk in head_key_candidates:
        if (
            hk in new_state
            and hk in model_state
            and tuple(new_state[hk].shape) == tuple(model_state[hk].shape)
        ):
            ok2 = True
            break
    if not ok2:
        return False, f"head not loaded; missing keys example: {missing2[:5]}"

    return (
        True,
        f"loaded filtered={len(new_state)}; missing={len(missing2)} unexpected={len(unexpected2)}",
    )


def _head_param_snapshot(model, head_param_names):
    sd = model.state_dict()
    snap = {}
    for n in head_param_names:
        if n in sd:
            t = sd[n].detach().float().cpu()
            snap[n] = (t.mean().item(), t.std().item())
    return snap




## === cell 9
torch.cuda.empty_cache()

weight_search_roots = [
    INPUT_PATH,
    "../input",
    "../input/ensemble-1023",
    "/kaggle/input",
    "/kaggle/data",
    "/kaggle/working",
    "/kaggle/data/kaggle/data",
    "/kaggle/data/kaggle/working",
    "/kaggle/data/input",
    "/kaggle/data/working",
    "/kaggle/data/cassava-leaf-disease-classification",
    "/kaggle/data/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
    "/kaggle/data/input/cassava-leaf-disease-classification",
    "/kaggle/data/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
]

resnext_weight = _find_weight_file(RESNEXT_PATH, weight_search_roots)
b4_weight = _find_weight_file(B4_PATH, weight_search_roots)

pre_head_1 = _head_param_snapshot(my_model_1, ["fc.weight", "fc.bias"])
pre_head_2 = _head_param_snapshot(my_model_2, ["classifier.weight", "classifier.bias"])

loaded_1, msg_1 = try_load_weights_strict_head(
    my_model_1,
    resnext_weight,
    head_key_candidates=["fc.weight", "fc.bias"],
)
loaded_2, msg_2 = try_load_weights_strict_head(
    my_model_2,
    b4_weight,
    head_key_candidates=["classifier.weight", "classifier.bias"],
)

post_head_1 = _head_param_snapshot(my_model_1, ["fc.weight", "fc.bias"])
post_head_2 = _head_param_snapshot(my_model_2, ["classifier.weight", "classifier.bias"])

my_model_1 = nn.DataParallel(my_model_1).to(DEVICE)
my_model_2 = nn.DataParallel(my_model_2).to(DEVICE)

print(f"Resolved RESNEXT weight: {resnext_weight} (loaded={loaded_1}) [{msg_1}]")
print(f"Resolved B4 weight: {b4_weight} (loaded={loaded_2}) [{msg_2}]")
print("ResNext head stats pre->post:", pre_head_1, "->", post_head_1)
print("EffNet head stats pre->post:", pre_head_2, "->", post_head_2)

if (not loaded_1) or (not loaded_2):
    warnings.warn(
        "Pretrained weights were not successfully loaded. A submission will still be produced, "
        "but expected accuracy will be very low. Ensure 1022_res50.pth and 1022_b4ns.pth exist in the environment."
    )

my_model_1.eval()
my_model_2.eval()

if not os.path.isdir(TEST_IMAGE_PATH):
    raise FileNotFoundError(f"TEST_IMAGE_PATH does not exist: {TEST_IMAGE_PATH}")

test_image_list = sorted(
    [fn for fn in os.listdir(TEST_IMAGE_PATH) if fn.lower().endswith(".jpg")]
)
if len(test_image_list) == 0:
    raise FileNotFoundError(
        f"No .jpg files found under TEST_IMAGE_PATH={TEST_IMAGE_PATH}"
    )

preds_1 = []
preds_2 = []

for single_image_name in tqdm(test_image_list, desc="Predicting"):
    image = Image.open(os.path.join(TEST_IMAGE_PATH, single_image_name)).convert("RGB")
    img_np = np.array(image)

    with torch.no_grad():
        ans1 = torch.zeros(OUT_FEATURES, device=DEVICE)
        for _ in range(1):
            aug_image = test_augs(image=img_np)["image"]
            test_image = aug_image.float().unsqueeze(0).to(DEVICE)
            ans1 += my_model_1(test_image).view(ans1.shape)
        preds_1.append(ans1.detach().cpu())

        ans2 = torch.zeros(OUT_FEATURES, device=DEVICE)
        for _ in range(TTA):
            aug_image = test_augs(image=img_np)["image"]
            test_image = aug_image.float().unsqueeze(0).to(DEVICE)
            ans2 += my_model_2(test_image).view(ans2.shape)
        ans2 /= TTA
        preds_2.append(ans2.detach().cpu())

predictions_1 = torch.stack(preds_1, dim=0)
predictions_2 = torch.stack(preds_2, dim=0)

normalize_pred_1 = F.normalize(predictions_1.T, p=2, dim=0).T
normalize_pred_2 = F.normalize(predictions_2.T, p=2, dim=0).T

final_pred = (normalize_pred_1 * 0.3) + (normalize_pred_2 * 0.7)
label = final_pred.argmax(dim=-1).numpy().astype(int)



## === cell 10
df_sub = pd.read_csv(SAMPLE_SUB_PATH)
df_sub = df_sub.sort_values("image_id").reset_index(drop=True)

pred_df = pd.DataFrame({"image_id": test_image_list, "label": label})
pred_df = pred_df.sort_values("image_id").reset_index(drop=True)

df_sub = df_sub.merge(pred_df, on="image_id", how="left", suffixes=("", "_pred"))
if "label_pred" in df_sub.columns:
    df_sub["label_pred"] = df_sub["label_pred"].fillna(0).astype(int)
    df_sub["label"] = df_sub["label_pred"].astype(int)
    df_sub = df_sub[["image_id", "label"]]

assert list(df_sub.columns) == [
    "image_id",
    "label",
], f"Bad submission columns: {df_sub.columns.tolist()}"
assert df_sub["label"].notna().all(), "Some labels are missing after merge."
assert len(df_sub) == len(
    pd.read_csv(SAMPLE_SUB_PATH)
), "Submission row count mismatch vs sample_submission."

df_sub.to_csv(SUBMISSION_PATH, index=False)
print(f"Saved submission to: {SUBMISSION_PATH}")
print(df_sub.head())
print("Submission shape:", df_sub.shape)
