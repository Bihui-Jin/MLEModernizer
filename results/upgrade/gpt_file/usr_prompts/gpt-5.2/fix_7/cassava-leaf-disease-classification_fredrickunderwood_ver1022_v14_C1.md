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

0.8986098519190088

# 6. Current score

0.26943

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.27205) has done: 'I fix the Albumentations `RandomResizedCrop` API breakage by switching to the v2 signature that expects a `size=(H, W)` tuple, which unblocks augmentation construction without changing intent. Then I fix the missing checkpoint path by auto-detecting the correct Kaggle input root (`/kaggle/input/...`) and falling back to a safe, built-in pretrained TIMM initialization if the ensemble weight files aren’t present, ensuring the notebook always runs end-to-end and produces `submission.csv`. I also make inference deterministic/stable (proper device handling, `map_location`, RGB conversion) and speed-safe (batched DataLoader inference instead of per-file loops) without changing the ensemble logic (two models + normalized logits + weighted sum + argmax). Finally, I ensure the submission uses exactly the sample submission `image_id` order and required columns.'
- What this solution (achieved 0.26457) has done: 'I fix the immediate Albumentations v2 runtime error by replacing the removed `A.Cutout` with the v2 equivalent (`A.CoarseDropout`) while keeping the same augmentation intent (random rectangular holes). I also remove the accidental double-application of dropout (you already have `A.CoarseDropout` once) so training/inference semantics remain consistent and not overly destructive. To move score toward the target, I ensure the intended learned ensemble is actually used by loading the provided weights from the correct Kaggle input directory when available (and only fall back to pretrained backbone initialization if the weights truly aren’t present). Finally, I keep submission ordering exactly as `sample_submission.csv` and write `submission.csv` end-to-end.'
- What this solution (achieved 0.27354) has done: 'Your current score (0.26457) is far below the target (0.8986), and the biggest likely cause is that inference is effectively random because the competition weight files are not being found/loaded, so the fallback path initializes new random classification heads. I make the smallest changes that (1) reliably locate the weights under any Kaggle input folder (without changing your ensemble logic), and (2) fix the TTA loop so it actually applies different stochastic augmentations per pass (right now it re-runs the exact same tensor, which can’t help). I also ensure the weight-loader can handle common checkpoint formats (`state_dict`, `model`, or raw state dict) and strip `module.` safely, while keeping the same models, heads, normalization, weighting, and argmax submission semantics. These changes should move accuracy sharply upward toward the target if the intended weights exist; if they don’t, it still run end-to-end and write a valid `submission.csv`.'
- What this solution (achieved 0.26868) has done: 'Your score is far below the target, so the priority is to make sure you’re actually using trained weights and not accidentally running with randomly initialized heads (which would cap accuracy near chance). I make the smallest change to the weight-loading fallback: if the `.pth` files are not found, I *not* reinitialize a new random classification head; instead I keep the pretrained model’s own classifier head so predictions are at least meaningful. I also fix a subtle focal-loss bug (currently `p_t` is computed from logits, not probabilities), which doesn’t change your current inference-only path but prevents training from being broken if you run it. Everything else (models, ensemble math, normalization, TTA loop, submission formatting) stays the same.'
- What this solution (achieved 0.26943) has done: 'I fix the DataLoader batching error by removing the uncollatable PIL Image object from the dataset output and instead returning the raw NumPy image (which can be stacked) alongside the transformed tensor. Then I update the TTA loop to use those returned NumPy images so augmentation is still applied per-pass without changing your ensemble math or model architecture. Finally, I add a small robustness tweak to the DataLoader (persistent_workers only when num_workers>0) so it runs reliably in Kaggle, and ensure `labels` is produced so `submission.csv` is written end-to-end.'

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
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

import timm




## === cell 1
def resolve_kaggle_input_path(rel_path: str) -> str:
    """
    Try common Kaggle roots for a given relative '../input/...' style path.
    Returns the first existing candidate, else returns the original path.
    """
    rel_path = rel_path.replace("\\", "/")
    candidates = [
        rel_path,
        rel_path.replace("../input/", "/kaggle/input/"),
        rel_path.replace("../input/", "/kaggle/data/input/"),
        rel_path.replace("../input/", "/kaggle/data/"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return rel_path




## === cell 2
INPUT_PATH = "../input/ensemble-1023/"
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
LR_FINAL = 1e-5
TTA = 8

INPUT_PATH = resolve_kaggle_input_path(INPUT_PATH)
TRAIN_CSV_PATH = resolve_kaggle_input_path(TRAIN_CSV_PATH)
TRAIN_IMAGE_PATH = resolve_kaggle_input_path(TRAIN_IMAGE_PATH)
TEST_IMAGE_PATH = resolve_kaggle_input_path(TEST_IMAGE_PATH)


def _auto_find_weights_dir(expected_dir: str, weight_filenames: list[str]) -> str:
    expected_dir = expected_dir.replace("\\", "/")
    if os.path.isdir(expected_dir) and all(
        os.path.exists(os.path.join(expected_dir, w)) for w in weight_filenames
    ):
        return expected_dir

    roots = ["/kaggle/input", "/kaggle/data/input", "/kaggle/data"]
    for root in roots:
        if not os.path.isdir(root):
            continue
        for dirpath, _, filenames in os.walk(root):
            if all(w in filenames for w in weight_filenames):
                return dirpath
    return expected_dir


INPUT_PATH = _auto_find_weights_dir(INPUT_PATH, [RESNEXT_PATH, B4_PATH])



## === cell 3
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




## === cell 4
def sigmoid_focal_cross_entropy(y_hat, y_true, alpha=0.25, gamma=2.0):
    def smooth(y, smooth_factor):
        assert len(y.shape) == 2
        y = y * (1 - smooth_factor)
        y = y + smooth_factor / y.shape[1]
        return y

    smooth_factor = 0.1

    if not isinstance(y_true, torch.Tensor):
        y_true = torch.tensor(y_true)
    if not isinstance(y_hat, torch.Tensor):
        y_hat = torch.tensor(y_hat)

    y_true = smooth(y_true, smooth_factor).to(y_hat.device).type_as(y_hat)

    bce = F.binary_cross_entropy_with_logits(y_hat, y_true, reduction="none")
    p = torch.sigmoid(y_hat)
    p_t = y_true * p + (1 - y_true) * (1 - p)

    alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
    modulating_factor = (1.0 - p_t).pow(gamma)

    return torch.sum(alpha_t * modulating_factor * bce, dim=-1)




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
class CassavaTestDataset(Dataset):
    def __init__(self, image_ids, image_dir, transform):
        self.image_ids = list(image_ids)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_name = self.image_ids[idx]
        img_path = os.path.join(self.image_dir, img_name)

        img_pil = Image.open(img_path).convert("RGB")
        img_np = np.array(img_pil)  # uint8 HWC, collatable
        img_t = self.transform(image=img_np)["image"]
        return img_name, img_np, img_t  # (name, raw_np, tensor)




## === cell 10
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




## === cell 11
def _extract_state_dict(obj):
    if isinstance(obj, dict):
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return obj["state_dict"]
        if "model" in obj and isinstance(obj["model"], dict):
            return obj["model"]
    return obj


def _strip_module_prefix(state_dict: dict) -> dict:
    if not isinstance(state_dict, dict):
        return state_dict
    if any(str(k).startswith("module.") for k in state_dict.keys()):
        return {k[7:]: v for k, v in state_dict.items() if str(k).startswith("module.")}
    return state_dict


def _try_load_state_dict_flexible(model: nn.Module, state: dict) -> bool:
    """
    Change (moves score toward target): robust weight loading.
    - First try strict=True (ideal, preserves intended ensemble).
    - If it fails due to minor key differences, retry strict=False so we still load the backbone.
      This is far better than leaving random weights (observed ~0.27).
    """
    try:
        model.load_state_dict(state, strict=True)
        return True
    except Exception:
        model.load_state_dict(state, strict=False)
        return True


def try_load_weights_or_pretrain(model, model_name, weight_path, head_attr_name):
    """
    Try to load state_dict from a .pth.
    If not found, recreate model with pretrained=True.

    Note: we keep your core logic (two specific models + OUT_FEATURES heads).
    """
    weight_full_path = os.path.join(INPUT_PATH, weight_path)
    if os.path.exists(weight_full_path):
        state = torch.load(weight_full_path, map_location="cpu")
        state = _extract_state_dict(state)
        state = _strip_module_prefix(state)

        _try_load_state_dict_flexible(model, state)
        return model, True, weight_full_path

    m = timm.create_model(model_name, pretrained=True)

    if head_attr_name == "fc":
        out_dim = getattr(m.fc, "out_features", None)
        if out_dim != OUT_FEATURES:
            m.fc = nn.Linear(m.fc.in_features, OUT_FEATURES)
            nn.init.xavier_uniform_(m.fc.weight)
            if m.fc.bias is not None:
                nn.init.zeros_(m.fc.bias)
    elif head_attr_name == "classifier":
        out_dim = getattr(m.classifier, "out_features", None)
        if out_dim != OUT_FEATURES:
            m.classifier = nn.Linear(m.classifier.in_features, OUT_FEATURES)
            nn.init.xavier_uniform_(m.classifier.weight)
            if m.classifier.bias is not None:
                nn.init.zeros_(m.classifier.bias)
    else:
        raise ValueError(f"Unknown head_attr_name={head_attr_name}")
    return m, False, weight_full_path


my_model_1, loaded1, wpath1 = try_load_weights_or_pretrain(
    my_model_1, model_name1, RESNEXT_PATH, "fc"
)
my_model_2, loaded2, wpath2 = try_load_weights_or_pretrain(
    my_model_2, model_name2, B4_PATH, "classifier"
)



## === cell 12
sample_sub_path = resolve_kaggle_input_path(
    "../input/cassava-leaf-disease-classification/sample_submission.csv"
)
sample_df = pd.read_csv(sample_sub_path)
test_image_list = sample_df["image_id"].tolist()

test_ds = CassavaTestDataset(test_image_list, TEST_IMAGE_PATH, test_augs)
num_workers = min(4, os.cpu_count() or 1)
test_loader = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    drop_last=False,
    persistent_workers=(num_workers > 0),
)



## === cell 13
torch.cuda.empty_cache()


def predict_logits(model, loader, tta: int, transform_for_tta):
    model.eval()
    model = model.to(device)
    if torch.cuda.device_count() > 1 and device.type == "cuda":
        model = nn.DataParallel(model)

    all_logits = []
    with torch.no_grad():
        for _, raw_nps, images in loader:
            if tta <= 1:
                batch = images.to(device, non_blocking=True).float()
                logits = model(batch)
            else:
                logits = torch.zeros((images.size(0), OUT_FEATURES), device=device)

                batch0 = images.to(device, non_blocking=True).float()
                logits += model(batch0)

                for _ in range(tta - 1):
                    aug_tensors = []
                    for img_np in raw_nps:
                        if isinstance(img_np, torch.Tensor):
                            img_np = img_np.cpu().numpy()
                        img_np = np.asarray(img_np)
                        t = transform_for_tta(image=img_np)["image"]
                        aug_tensors.append(t)
                    batch_aug = (
                        torch.stack(aug_tensors, dim=0)
                        .to(device, non_blocking=True)
                        .float()
                    )
                    logits += model(batch_aug)

                logits /= tta

            all_logits.append(logits.detach().cpu())
    return torch.cat(all_logits, dim=0)


logits_1 = predict_logits(my_model_1, test_loader, tta=1, transform_for_tta=test_augs)
normalize_pred_1 = F.normalize(logits_1.T, p=2, dim=0).T

torch.cuda.empty_cache()

logits_2 = predict_logits(my_model_2, test_loader, tta=TTA, transform_for_tta=test_augs)
normalize_pred_2 = F.normalize(logits_2.T, p=2, dim=0).T

final_pred = (normalize_pred_1 * 0.45) + (normalize_pred_2 * 0.55)
labels = final_pred.argmax(dim=-1).numpy().astype(int).tolist()



## === cell 14
df_submission = pd.DataFrame({"image_id": test_image_list, "label": labels})
df_submission.to_csv(SUBMISSION_PATH, index=False)

print(f"Wrote submission: {SUBMISSION_PATH} rows={len(df_submission)}")
print(f"Ensemble weights dir: {INPUT_PATH}")
print(f"Weight1 loaded={loaded1} from {wpath1}")
print(f"Weight2 loaded={loaded2} from {wpath2}")
print(df_submission.head())
