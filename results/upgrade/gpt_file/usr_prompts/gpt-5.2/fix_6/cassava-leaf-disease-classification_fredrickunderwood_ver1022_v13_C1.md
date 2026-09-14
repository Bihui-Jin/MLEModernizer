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

0.8957388939256573

# 6. Current score

0.05531

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.05531) has done: 'I fix the Albumentations API breakage by updating `RandomResizedCrop` calls to the v2 signature so augmentations build correctly under albumentations==2.0.8. Then I fix the missing checkpoint path by switching `INPUT_PATH` to the provided competition dataset folder and adding a safe fallback that runs even if those `.pth` files are not present (so you always get a valid `submission.csv`). I also make inference robust and aligned with the required submission format by using `sample_submission.csv` ordering, sorting test filenames, using `map_location`, and not double-wrapping tensors with `torch.tensor()` after `ToTensorV2`. These changes are minimal and unblock end-to-end execution; if checkpoints exist, it use them as intended, otherwise it still output a valid submission file.'
- What this solution (achieved 0.05531) has done: 'I fix the immediate runtime error by replacing the removed `A.Cutout` transform with the albumentations v2 equivalent (`A.CoarseDropout`) while keeping the augmentation intent the same. I also remove the duplicate dropout transform so the pipeline remains stable and close to the original logic. This should let the notebook run end-to-end and actually load checkpoints (if present) and perform inference instead of falling back to all-zero labels, which is the main reason the current score is extremely low. Finally, I keep submission ordering aligned to `sample_submission.csv` and ensure a valid `submission.csv` is always written.'
- What this solution (achieved 0.05531) has done: 'Your current score is extremely low mainly because inference falls back to predicting all-zero labels when checkpoint files are missing, which is almost certainly happening in your environment. To move the accuracy toward your target with minimal disruption, I keep your exact model definitions and inference ensemble logic, but ensure checkpoints are actually found by searching in the real competition dataset folders (including the common `/kaggle/working` location) instead of assuming they live next to the dataset. If checkpoints still aren’t present anywhere, the code still produce a valid `submission.csv` (as before), but this change should eliminate the all-zero fallback on runs where the `.pth` files exist, which is the smallest, most direct fix to improve score.'
- What this solution (achieved 0.05531) has done: 'Your current score (0.05531) is far below the target, and the main cause is that inference is almost certainly falling back to all-zero labels because the required `.pth` checkpoints are not being found/loaded. I keep your exact model architectures and ensemble logic, but make checkpoint discovery robust by recursively searching the available Kaggle filesystem (including `/kaggle/input`, `/kaggle/working`, and `/kaggle/data`) for the two checkpoint filenames before giving up. I also allow safe loading when checkpoints were saved from a slightly different wrapper (e.g., `model.` prefix) without changing weights or inference semantics, so a valid checkpoint actually load instead of silently failing. If checkpoints still truly don’t exist, the code behave as before and still write a valid `submission.csv`.'
- What this solution (achieved 0.05531) has done: 'Your score is extremely far below the target, and the most direct reason in this code is that it outputs all-zero labels whenever either checkpoint is missing or fails to load, which guarantees very low accuracy. I keep your exact architectures and ensemble logic, but make checkpoint loading work correctly with `DataParallel` by wrapping the models before loading and falling back to `strict=False` only when needed (so real checkpoints actually load instead of silently failing). I also remove the “all-zero unless both checkpoints exist” gate: if only one checkpoint is available, we still run inference with the available model(s) using the same normalization+weighted blending semantics (single-model becomes weight 1.0). These are minimal changes that should move accuracy sharply upward toward your target while preserving the core approach.'

# 9. Code solution

## === cell 0
import os
import math
import random
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




## === cell 1
def _find_existing_path(candidates):
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[0]


BASE_INPUT = _find_existing_path(
    [
        "../input/cassava-leaf-disease-classification",
        "/kaggle/input/cassava-leaf-disease-classification",
        "/kaggle/data/cassava-leaf-disease-classification",
    ]
)



## === cell 2
INPUT_PATH = BASE_INPUT  # dataset root (images/csv)
TRAIN_CSV_PATH = os.path.join(BASE_INPUT, "train.csv")
TRAIN_IMAGE_PATH = os.path.join(BASE_INPUT, "train_images")
TEST_IMAGE_PATH = os.path.join(BASE_INPUT, "test_images")
SAMPLE_SUB_PATH = os.path.join(BASE_INPUT, "sample_submission.csv")

SUBMISSION_PATH = "submission.csv"
RESNEXT_PATH = "1022_res50.pth"
B4_PATH = "1022_b4ns.pth"

CKPT_SEARCH_ROOTS = [
    ".",  # current working dir
    "/kaggle/working",
    os.path.join("/kaggle/working", "cassava-leaf-disease-classification"),
    BASE_INPUT,  # dataset dir (if user attached as dataset with checkpoints)
    "/kaggle/input",
    "/kaggle/data",
]

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
TTA = 6



## === cell 3
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")




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


x = [i for i in range(NUM_EPOCHS)]
y = [lr_tune(i) for i in x]
plt.plot(x, y)



## === cell 6
train_augs = A.Compose(
    [
        A.RandomResizedCrop(
            size=(IMAGE_SIZE, IMAGE_SIZE),
            scale=(0.8, 1.0),
            ratio=(0.75, 1.3333333333333333),
            p=1.0,
        ),
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
                A.RandomResizedCrop(
                    size=(IMAGE_SIZE, IMAGE_SIZE),
                    scale=(0.9, 1.0),
                    ratio=(0.9, 1.1),
                    p=1.0,
                ),
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



## === cell 9
model_name1 = "resnext50_32x4d"
my_model_1 = timm.create_model(model_name1, pretrained=False)
my_model_1.fc = nn.Linear(my_model_1.fc.in_features, OUT_FEATURES)
nn.init.xavier_uniform_(my_model_1.fc.weight)
if my_model_1.fc.bias is not None:
    nn.init.zeros_(my_model_1.fc.bias)
my_model_1



## === cell 10
model_name2 = "tf_efficientnet_b4_ns"
my_model_2 = timm.create_model(model_name2, pretrained=False)
my_model_2.classifier = nn.Linear(my_model_2.classifier.in_features, OUT_FEATURES)
nn.init.xavier_uniform_(my_model_2.classifier.weight)
if my_model_2.classifier.bias is not None:
    nn.init.zeros_(my_model_2.classifier.bias)
my_model_2



## === cell 11
torch.cuda.empty_cache()




## === cell 12
def _load_checkpoint_if_exists(model, ckpt_path, device):
    if not os.path.exists(ckpt_path):
        return False

    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        return False

    def _strip_prefix(d, prefix):
        if any(k.startswith(prefix) for k in d.keys()):
            return {k.replace(prefix, "", 1): v for k, v in d.items()}
        return d

    state_a = _strip_prefix(state, "module.")
    state_a = _strip_prefix(state_a, "model.")
    state_b = state.copy()
    if not any(k.startswith("module.") for k in state_b.keys()):
        state_b = {("module." + k): v for k, v in state_b.items()}

    target = model.module if isinstance(model, nn.DataParallel) else model

    try:
        target.load_state_dict(state_a, strict=True)
        return True
    except Exception:
        pass

    try:
        target.load_state_dict(state_a, strict=False)
        return True
    except Exception:
        pass

    try:
        model.load_state_dict(state_b, strict=False)
        return True
    except Exception:
        return False


def _find_ckpt_file(filename, roots):
    checked = []
    for r in roots:
        cand = os.path.join(r, filename)
        checked.append(cand)
        if os.path.exists(cand):
            return cand

    max_depth = 4
    for r in roots:
        if not os.path.isdir(r):
            continue
        r = os.path.abspath(r)
        for dirpath, dirnames, filenames in os.walk(r):
            rel = os.path.relpath(dirpath, r)
            depth = 0 if rel == "." else rel.count(os.sep) + 1
            if depth > max_depth:
                dirnames[:] = []
                continue
            if filename in filenames:
                return os.path.join(dirpath, filename)

    return checked[0] if checked else os.path.join(roots[0], filename)




## === cell 13
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
test_image_list = sample_sub["image_id"].astype(str).tolist()

if len(test_image_list) == 0:
    test_image_list = sorted(
        [f for f in os.listdir(TEST_IMAGE_PATH) if f.lower().endswith(".jpg")]
    )




## === cell 14
def _predict_logits_single(model, image_path, tta, augs, device):
    model.eval()
    with torch.no_grad():
        acc = torch.zeros(OUT_FEATURES, device=device)
        for _ in range(tta):
            img = Image.open(image_path).convert("RGB")
            aug = augs(image=np.array(img))["image"]  # torch.Tensor CHW float32
            x = aug.unsqueeze(0).to(device, non_blocking=True)
            acc += model(x).view(-1)
        acc /= float(tta)
    return acc




## === cell 15
ckpt1 = _find_ckpt_file(RESNEXT_PATH, CKPT_SEARCH_ROOTS)
ckpt2 = _find_ckpt_file(B4_PATH, CKPT_SEARCH_ROOTS)

print("Checkpoint candidates:")
print(" - resnext:", ckpt1, "exists:", os.path.exists(ckpt1))
print(" - b4ns  :", ckpt2, "exists:", os.path.exists(ckpt2))

my_model_1 = (
    nn.DataParallel(my_model_1).to(DEVICE)
    if torch.cuda.device_count() > 1
    else my_model_1.to(DEVICE)
)
my_model_2 = (
    nn.DataParallel(my_model_2).to(DEVICE)
    if torch.cuda.device_count() > 1
    else my_model_2.to(DEVICE)
)

has_ckpt1 = _load_checkpoint_if_exists(my_model_1, ckpt1, DEVICE)
has_ckpt2 = _load_checkpoint_if_exists(my_model_2, ckpt2, DEVICE)
print("Loaded checkpoints:", {"resnext": has_ckpt1, "b4ns": has_ckpt2})

preds_1 = []
preds_2 = []

if not (has_ckpt1 or has_ckpt2):
    labels = np.zeros(len(test_image_list), dtype=np.int64)
else:
    for single_image_name in tqdm(test_image_list, desc="Predicting"):
        img_path = os.path.join(TEST_IMAGE_PATH, single_image_name)

        if has_ckpt1:
            logit1 = _predict_logits_single(
                my_model_1, img_path, tta=1, augs=test_augs, device=DEVICE
            )
            preds_1.append(logit1.detach().cpu())

        if has_ckpt2:
            logit2 = _predict_logits_single(
                my_model_2, img_path, tta=TTA, augs=test_augs, device=DEVICE
            )
            preds_2.append(logit2.detach().cpu())

    if has_ckpt1:
        predictions_1 = torch.stack(preds_1, dim=0)
        normalize_pred_1 = F.normalize(predictions_1.T, p=2, dim=0).T
    if has_ckpt2:
        predictions_2 = torch.stack(preds_2, dim=0)
        normalize_pred_2 = F.normalize(predictions_2.T, p=2, dim=0).T

    if has_ckpt1 and has_ckpt2:
        final_pred = (normalize_pred_1 * 0.43) + (normalize_pred_2 * 0.57)
    elif has_ckpt1:
        final_pred = normalize_pred_1
    else:
        final_pred = normalize_pred_2

    labels = final_pred.argmax(dim=-1).numpy().astype(np.int64)



## === cell 16
df_submission = sample_sub.copy()
df_submission["label"] = labels
df_submission.to_csv(SUBMISSION_PATH, index=False)

print("Wrote:", SUBMISSION_PATH)
print(df_submission.head())
print(
    "Unique predicted labels:", pd.Series(labels).value_counts().sort_index().to_dict()
)
