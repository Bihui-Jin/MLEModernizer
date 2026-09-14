# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

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
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import timm

warnings.filterwarnings("ignore")


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




## === cell 1
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

FINETUNE_IF_NO_CKPT = True
FINETUNE_EPOCHS = 3
FINETUNE_VAL_SPLIT = 0.1
FINETUNE_NUM_WORKERS = 2

PREDICT_NUM_WORKERS = 4
FINETUNE_PREFETCH_FACTOR = 4
PREDICT_PREFETCH_FACTOR = 4

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

if device.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

print(f"[INFO] device: {device}")
print(f"[INFO] DATASET_ROOT: {DATASET_ROOT}")
print(f"[INFO] Using INPUT_PATH for checkpoints (if exists): {INPUT_PATH}")
print(f"[INFO] TRAIN_CSV_PATH: {TRAIN_CSV_PATH}")
print(f"[INFO] TRAIN_IMAGE_PATH: {TRAIN_IMAGE_PATH}")
print(f"[INFO] TEST_IMAGE_PATH: {TEST_IMAGE_PATH}")
print(f"[INFO] SAMPLE_SUB_PATH: {SAMPLE_SUB_PATH}")
print(f"[INFO] Kaggle input roots detected: {detect_kaggle_input_roots()}")



## === cell 2
seed_everything(SEED)




## === cell 3
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




## === cell 4
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



## === cell 5
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




## === cell 6
def build_test_augs_from_timm(model, img_size=IMAGE_SIZE):
    cfg = timm.data.resolve_model_data_config(model)
    mean = cfg.get("mean", (0.485, 0.456, 0.406))
    std = cfg.get("std", (0.229, 0.224, 0.225))
    return A.Compose(
        [
            A.OneOf(
                [
                    A.Resize(img_size, img_size, p=1.0),
                    A.CenterCrop(img_size, img_size, p=1.0),
                    A.RandomResizedCrop(size=(img_size, img_size), p=1.0),
                ],
                p=1.0,
            ),
            A.Transpose(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Resize(img_size, img_size),
            A.Normalize(
                mean=list(mean),
                std=list(std),
                max_pixel_value=255.0,
                p=1.0,
            ),
            ToTensorV2(p=1.0),
        ],
        p=1.0,
    )




## === cell 7
model_name1 = "resnext50_32x4d"
my_model_1 = timm.create_model(model_name1, pretrained=True)
my_model_1.fc = nn.Linear(my_model_1.fc.in_features, OUT_FEATURES)
nn.init.xavier_uniform_(my_model_1.fc.weight)
if my_model_1.fc.bias is not None:
    nn.init.zeros_(my_model_1.fc.bias)
my_model_1



## === cell 8
model_name2 = "tf_efficientnet_b4_ns"
my_model_2 = timm.create_model(model_name2, pretrained=True)
my_model_2.classifier = nn.Linear(my_model_2.classifier.in_features, OUT_FEATURES)
nn.init.xavier_uniform_(my_model_2.classifier.weight)
if my_model_2.classifier.bias is not None:
    nn.init.zeros_(my_model_2.classifier.bias)
my_model_2



## === cell 9
_CKPT_SEARCH_ROOTS = None
_CKPT_FOUND_CACHE = {}


def _build_ckpt_search_roots():
    search_roots = []
    if INPUT_PATH and os.path.isdir(INPUT_PATH):
        search_roots.append(os.path.abspath(INPUT_PATH))
    if DATASET_ROOT and os.path.isdir(DATASET_ROOT):
        search_roots.append(os.path.abspath(DATASET_ROOT))
    search_roots.append(os.path.abspath(os.getcwd()))
    nested = []
    for r in list(search_roots):
        try:
            for name in os.listdir(r):
                p = os.path.join(r, name)
                if os.path.isdir(p):
                    nested.append(os.path.abspath(p))
        except Exception:
            pass
    seen = set()
    uniq = []
    for r in search_roots + nested:
        if r not in seen and os.path.isdir(r):
            uniq.append(r)
            seen.add(r)
    return uniq


def load_checkpoint_into_model(model, ckpt_filename):
    global _CKPT_SEARCH_ROOTS, _CKPT_FOUND_CACHE

    if ckpt_filename in _CKPT_FOUND_CACHE:
        ckpt_path = _CKPT_FOUND_CACHE[ckpt_filename]
        if ckpt_path is None:
            print(f"[WARN] Checkpoint not found anywhere (cached): {ckpt_filename}")
            return False
    else:
        if _CKPT_SEARCH_ROOTS is None:
            _CKPT_SEARCH_ROOTS = _build_ckpt_search_roots()

        ckpt_path = find_file(ckpt_filename, _CKPT_SEARCH_ROOTS)

        _CKPT_FOUND_CACHE[ckpt_filename] = ckpt_path
        if ckpt_path is None:
            print(f"[WARN] Checkpoint not found anywhere: {ckpt_filename}")
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


def build_imagenet_fallback_resnext50(num_classes=OUT_FEATURES):
    m = timm.create_model("resnext50_32x4d", pretrained=True, num_classes=num_classes)
    return m


def build_imagenet_fallback_effb4(num_classes=OUT_FEATURES):
    m = timm.create_model(
        "tf_efficientnet_b4_ns", pretrained=True, num_classes=num_classes
    )
    return m


class CassavaTrainDataset(Dataset):
    def __init__(self, df, image_dir, augs):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.augs = augs

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        image_id = str(r["image_id"])
        y = int(r["label"])
        img = Image.open(os.path.join(self.image_dir, image_id)).convert("RGB")
        x = self.augs(image=np.array(img))["image"]
        return x, torch.tensor(y, dtype=torch.long)


def finetune_model_if_needed(
    model,
    model_kind: str,
    train_csv_path: str,
    train_image_dir: str,
    epochs: int,
    batch_size: int,
    num_workers: int,
    seed: int,
    val_split: float,
):
    if epochs <= 0:
        return model

    df = pd.read_csv(train_csv_path)
    df["image_id"] = df["image_id"].astype(str)
    df["label"] = df["label"].astype(int)

    rng = np.random.RandomState(seed)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    n_val = max(1, int(len(df) * val_split))
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]

    df_tr = df.iloc[tr_idx].reset_index(drop=True)
    df_va = df.iloc[val_idx].reset_index(drop=True)

    ds_tr = CassavaTrainDataset(df_tr, train_image_dir, train_augs)
    ds_va = CassavaTrainDataset(df_va, train_image_dir, valid_augs)

    dl_tr = DataLoader(
        ds_tr,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=(FINETUNE_PREFETCH_FACTOR if num_workers > 0 else None),
    )
    dl_va = DataLoader(
        ds_va,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        persistent_workers=(num_workers > 0),
        prefetch_factor=(FINETUNE_PREFETCH_FACTOR if num_workers > 0 else None),
    )

    model = model.to(device)
    if torch.cuda.device_count() > 1 and device.type == "cuda":
        model = nn.DataParallel(model).to(device)

    opt = OPTIMIZER(model.parameters(), lr=LR_START)
    criterion = nn.CrossEntropyLoss()

    for ep in range(epochs):
        lr = float(lr_tune(ep, num_epochs=max(NUM_EPOCHS, epochs)))
        for g in opt.param_groups:
            g["lr"] = lr

        model.train()
        tr_loss = 0.0
        tr_n = 0
        for xb, yb in tqdm(
            dl_tr,
            desc=f"[FINETUNE {model_kind}] epoch {ep+1}/{epochs} train",
            leave=False,
        ):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            opt.step()
            tr_loss += float(loss.item()) * xb.size(0)
            tr_n += xb.size(0)

        model.eval()
        va_correct = 0
        va_n = 0
        with torch.no_grad():
            for xb, yb in tqdm(
                dl_va,
                desc=f"[FINETUNE {model_kind}] epoch {ep+1}/{epochs} valid",
                leave=False,
            ):
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                out = model(xb)
                pred = out.argmax(dim=1)
                va_correct += int((pred == yb).sum().item())
                va_n += xb.size(0)

        print(
            f"[INFO] Finetune {model_kind}: epoch {ep+1}/{epochs} lr={lr:.2e} "
            f"train_loss={(tr_loss/max(1,tr_n)):.4f} val_acc={(va_correct/max(1,va_n)):.4f}"
        )

    return model


class CassavaTestDatasetTTAStream(Dataset):
    def __init__(self, image_ids, image_dir, augs, tta=8, image_cache=None):
        self.image_ids = list(map(str, image_ids))
        self.image_dir = image_dir
        self.augs = augs
        self.tta = int(tta)
        self.image_cache = (
            image_cache  # dict image_id -> np.uint8 HWC RGB (decoded once)
        )

        self._n = len(self.image_ids)

    def __len__(self):
        return int(self._n * self.tta)

    def __getitem__(self, idx):
        img_i = int(idx // self.tta)
        image_id = self.image_ids[img_i]
        if self.image_cache is not None:
            arr = self.image_cache[image_id]
        else:
            img = Image.open(os.path.join(self.image_dir, image_id)).convert("RGB")
            arr = np.array(img)
        x = self.augs(image=arr)["image"]  # torch tensor CHW float32
        return x, img_i


def _build_image_cache(image_ids, image_dir):
    cache = {}
    for image_id in tqdm(
        image_ids, desc="[INFO] Caching test images (decode once)", leave=False
    ):
        img = Image.open(os.path.join(image_dir, str(image_id))).convert("RGB")
        cache[str(image_id)] = np.array(img)
    return cache


def predict_tta(
    model,
    image_ids,
    image_dir,
    augs,
    tta=8,
    batch_size=32,
    num_workers=2,
    image_cache=None,
):
    image_ids = list(map(str, image_ids))

    ds = CassavaTestDatasetTTAStream(
        image_ids, image_dir, augs, tta=tta, image_cache=image_cache
    )

    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        persistent_workers=(num_workers > 0),
        prefetch_factor=(PREDICT_PREFETCH_FACTOR if num_workers > 0 else None),
    )

    n = len(image_ids)
    logits_sum = torch.zeros((n, OUT_FEATURES), dtype=torch.float32)
    counts = torch.zeros((n,), dtype=torch.int32)

    model.eval()
    with torch.inference_mode():
        for xb, img_idx in dl:
            xb = xb.to(device, non_blocking=True)
            out = model(xb).detach().cpu()  # [B, C]
            img_idx = torch.as_tensor(img_idx, dtype=torch.long)
            logits_sum.index_add_(0, img_idx, out)
            counts += torch.bincount(img_idx, minlength=n).to(torch.int32)

    logits_mean = logits_sum / counts.to(logits_sum.dtype).unsqueeze(1).clamp_min(1)
    return logits_mean  # [N, C]


torch.cuda.empty_cache()

sample_df = pd.read_csv(SAMPLE_SUB_PATH)
test_image_list = sample_df["image_id"].astype(str).values

test_image_cache = _build_image_cache(test_image_list, TEST_IMAGE_PATH)

loaded1 = load_checkpoint_into_model(my_model_1, RESNEXT_PATH)
if not loaded1:
    print(f"[WARN] Proceeding without '{RESNEXT_PATH}' custom weights.")
    if FINETUNE_IF_NO_CKPT:
        my_model_1 = timm.create_model(
            "resnext50_32x4d", pretrained=True, num_classes=OUT_FEATURES
        )
        my_model_1 = finetune_model_if_needed(
            my_model_1,
            model_kind="resnext50",
            train_csv_path=TRAIN_CSV_PATH,
            train_image_dir=TRAIN_IMAGE_PATH,
            epochs=FINETUNE_EPOCHS,
            batch_size=BATCH_SIZE,
            num_workers=FINETUNE_NUM_WORKERS,
            seed=SEED,
            val_split=FINETUNE_VAL_SPLIT,
        )
    else:
        my_model_1 = build_imagenet_fallback_resnext50(OUT_FEATURES)

my_model_1 = my_model_1.to(device)
if (
    torch.cuda.device_count() > 1
    and device.type == "cuda"
    and not isinstance(my_model_1, nn.DataParallel)
):
    my_model_1 = nn.DataParallel(my_model_1).to(device)

test_augs_1 = build_test_augs_from_timm(
    my_model_1.module if isinstance(my_model_1, nn.DataParallel) else my_model_1,
    IMAGE_SIZE,
)

loaded2 = load_checkpoint_into_model(my_model_2, B4_PATH)
if not loaded2:
    print(f"[WARN] Proceeding without '{B4_PATH}' custom weights.")
    if FINETUNE_IF_NO_CKPT:
        my_model_2 = timm.create_model(
            "tf_efficientnet_b4_ns", pretrained=True, num_classes=OUT_FEATURES
        )
        my_model_2 = finetune_model_if_needed(
            my_model_2,
            model_kind="effb4",
            train_csv_path=TRAIN_CSV_PATH,
            train_image_dir=TRAIN_IMAGE_PATH,
            epochs=FINETUNE_EPOCHS,
            batch_size=BATCH_SIZE,
            num_workers=FINETUNE_NUM_WORKERS,
            seed=SEED + 1,
            val_split=FINETUNE_VAL_SPLIT,
        )
    else:
        my_model_2 = build_imagenet_fallback_effb4(OUT_FEATURES)

my_model_2 = my_model_2.to(device)
if (
    torch.cuda.device_count() > 1
    and device.type == "cuda"
    and not isinstance(my_model_2, nn.DataParallel)
):
    my_model_2 = nn.DataParallel(my_model_2).to(device)

test_augs_2 = build_test_augs_from_timm(
    my_model_2.module if isinstance(my_model_2, nn.DataParallel) else my_model_2,
    IMAGE_SIZE,
)

predictions_1 = predict_tta(
    my_model_1,
    test_image_list,
    TEST_IMAGE_PATH,
    test_augs_1,
    tta=TTA,
    batch_size=BATCH_SIZE,
    num_workers=PREDICT_NUM_WORKERS,
    image_cache=test_image_cache,
)
normalize_pred_1 = F.normalize(predictions_1.T, p=2, dim=0).T
torch.cuda.empty_cache()

predictions_2 = predict_tta(
    my_model_2,
    test_image_list,
    TEST_IMAGE_PATH,
    test_augs_2,
    tta=TTA,
    batch_size=BATCH_SIZE,
    num_workers=PREDICT_NUM_WORKERS,
    image_cache=test_image_cache,
)
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
