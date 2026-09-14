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

3.9

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
pillow==11.3.0
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
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
import os, sys, subprocess, textwrap, json, warnings

warnings.filterwarnings("ignore")


def run_cmd(cmd):
    try:
        subprocess.check_call(cmd, shell=True)
    except Exception as e:
        print(f"[WARN] Command failed (continuing): {cmd}\n  -> {e}")




## === cell 1
run_cmd(
    "ls -la ../input/efficientnetpytorch-install/dist/efficientnet_pytorch-0.7.0.tar 2>/dev/null || true"
)
if os.path.exists(
    "../input/efficientnetpytorch-install/dist/efficientnet_pytorch-0.7.0.tar"
):
    run_cmd(
        "pip -q install ../input/efficientnetpytorch-install/dist/efficientnet_pytorch-0.7.0.tar"
    )


## === cell 2
run_cmd("ls -lrt ../input/ramki-cassava-weights/weight.pt 2>/dev/null || true")


## === cell 3
run_cmd("ls ../input/cassava-leaf-disease-classification 2>/dev/null || true")
run_cmd("ls /kaggle/input/cassava-leaf-disease-classification 2>/dev/null || true")


## === cell 4
CANDIDATE_BASES = [
    "../input/cassava-leaf-disease-classification/",
    "/kaggle/input/cassava-leaf-disease-classification/",
    "/kaggle/data/cassava-leaf-disease-classification/",
    "/kaggle/data/input/cassava-leaf-disease-classification/",
]
base_path = None
for p in CANDIDATE_BASES:
    if os.path.exists(os.path.join(p, "train.csv")) and os.path.exists(
        os.path.join(p, "sample_submission.csv")
    ):
        base_path = p
        break
if base_path is None:
    raise FileNotFoundError(
        "Could not find cassava-leaf-disease-classification dataset folder in expected locations."
    )
print("Using base_path:", base_path)


## === cell 5
Training = False


## === cell 6
import numpy as np
import pandas as pd
from PIL import Image
import cv2

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import seaborn as sns



## === cell 7
import torchvision
from torchvision import models
from torchvision.models import EfficientNet_B7_Weights

import albumentations as A
from albumentations.pytorch import ToTensorV2



## === cell 8
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device


## === cell 9
if torch.cuda.is_available():
    torch.cuda.device_count()


## === cell 10
writer = None


## === cell 11
SEED = 42
N_FOLDS = 10
N_EPOCHS = 20
BATCH_SIZE = 16
IMG_SIZE = 224
LR = 5e-4
NUM_CLASSES = 5




## === cell 12
def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False  # preserve determinism


seed_everything(SEED)


## === cell 13
base_path = base_path


## === cell 14
train_path = os.path.join(base_path, "train_images/")
test_path = os.path.join(base_path, "test_images/")

train_csv = pd.read_csv(os.path.join(base_path, "train.csv"))
sample = pd.read_csv(os.path.join(base_path, "sample_submission.csv"))


## === cell 15
train_csv.head()


## === cell 16
train_csv_disease = train_csv.label.map(
    {
        0: "Cassava Bacterial Blight (CBB)",
        1: "Cassava Brown Streak Disease (CBSD)",
        2: "Cassava Green Mottle (CGM)",
        3: "Cassava Mosaic Disease (CMD)",
        4: "Healthy",
    }
)
diseases = train_csv_disease.value_counts()


## === cell 17
diseases


## === cell 18
try:
    diseases.plot.pie()
    plt.show()
except Exception as e:
    print("[WARN] plot failed:", e)


## === cell 19
assert (
    "image_id" in sample.columns and "label" in sample.columns
), "sample_submission.csv must have image_id and label columns"




## === cell 20
class MyDataset(Dataset):
    def __init__(self, dataframe, transforms=None, test=False, cache_images=False):
        self.df = dataframe.reset_index(drop=True)
        self.transforms = transforms
        self.test = test
        self.cache_images = cache_images
        self._cache = {} if cache_images else None

    def __len__(self):
        return len(self.df)

    def _read_image_rgb_uint8(self, p_path: str):
        img = cv2.imread(p_path, cv2.IMREAD_COLOR)
        if img is None:
            img = np.array(Image.open(p_path).convert("RGB"))
            return img
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def __getitem__(self, idx):
        label = int(self.df.iloc[idx].label) if "label" in self.df.columns else 0
        p = self.df.iloc[idx].image_id
        p_path = (train_path if not self.test else test_path) + p

        if self.cache_images:
            img = self._cache.get(p_path)
            if img is None:
                img = self._read_image_rgb_uint8(p_path)
                self._cache[p_path] = img
        else:
            img = self._read_image_rgb_uint8(p_path)

        if self.transforms:
            transformed = self.transforms(image=img)
            image = transformed["image"]
        else:
            image = img

        return image, label




## === cell 21
transforms_train = A.Compose(
    [
        A.RandomResizedCrop(
            size=(IMG_SIZE, IMG_SIZE), scale=(0.8, 1.0), ratio=(0.75, 1.333), p=1.0
        ),
        A.Transpose(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.HueSaturationValue(
            hue_shift_limit=20, sat_shift_limit=20, val_shift_limit=20, p=0.5
        ),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
        A.CoarseDropout(
            num_holes_range=(1, 8),
            hole_height_range=(8, 32),
            hole_width_range=(8, 32),
            fill=0,
            p=0.5,
        ),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            max_pixel_value=255.0,
            p=1.0,
        ),
        ToTensorV2(p=1.0),
    ],
    p=1.0,
)


def make_valid_transform(norm_name: str, img_size: int = 224):
    if norm_name == "imagenet":
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    elif norm_name == "half":
        mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    elif norm_name == "none01":
        mean, std = (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)
    else:
        raise ValueError(f"Unknown norm_name: {norm_name}")

    resize_side = 256 if img_size <= 224 else 600
    return A.Compose(
        [
            A.Resize(height=resize_side, width=resize_side, p=1.0),
            A.CenterCrop(height=img_size, width=img_size, p=1.0),
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ],
        p=1.0,
    )


transforms_valid = make_valid_transform("imagenet", img_size=IMG_SIZE)


## === cell 22
folds = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)


## === cell 23
train_csv.shape


## === cell 24
model_name = "efficientnet-b7"


def build_model(num_classes=5, imagenet_fallback=False):
    weights = EfficientNet_B7_Weights.DEFAULT if imagenet_fallback else None
    m = models.efficientnet_b7(weights=weights)
    in_features = m.classifier[1].in_features
    m.classifier[1] = nn.Linear(in_features, num_classes)
    return m


def _extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for k in ["state_dict", "model_state_dict", "model", "net", "weights"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
        if any(isinstance(v, torch.Tensor) for v in ckpt.values()):
            return ckpt
    return ckpt


def _strip_prefix(state_dict, prefixes=("module.", "model.", "net.")):
    if not isinstance(state_dict, dict):
        return state_dict
    out = {}
    for k, v in state_dict.items():
        nk = k
        for p in prefixes:
            if nk.startswith(p):
                nk = nk[len(p) :]
        out[nk] = v
    return out


def _prefer_efficientnet_pytorch(state_dict):
    if not isinstance(state_dict, dict):
        return False
    keys = list(state_dict.keys())
    return any(
        k.startswith("_conv_stem.") or k.startswith("_fc.") or k.startswith("_blocks.")
        for k in keys
    )


def _build_effnet_pytorch_b7(num_classes=5, imagenet_init=True):
    try:
        from efficientnet_pytorch import EfficientNet
    except Exception as e:
        raise ImportError(
            "efficientnet_pytorch not available but checkpoint looks like efficientnet_pytorch."
        ) from e

    if imagenet_init:
        m = EfficientNet.from_pretrained("efficientnet-b7")
    else:
        m = EfficientNet.from_name("efficientnet-b7")

    in_features = m._fc.in_features
    m._fc = nn.Linear(in_features, num_classes)
    return m


def _remap_head_keys_for_effnet_pytorch(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict
    sd = dict(state_dict)

    aliases = [
        ("classifier.1.weight", "_fc.weight"),
        ("classifier.1.bias", "_fc.bias"),
        ("fc.weight", "_fc.weight"),
        ("fc.bias", "_fc.bias"),
        ("classifier.weight", "_fc.weight"),
        ("classifier.bias", "_fc.bias"),
        ("_fc.weight", "_fc.weight"),
        ("_fc.bias", "_fc.bias"),
    ]
    for src, dst in aliases:
        if src in sd and dst not in sd:
            sd[dst] = sd[src]
    return sd


def _remap_head_keys_for_torchvision_efficientnet(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict
    sd = dict(state_dict)

    if "head.weight" in sd and "classifier.1.weight" not in sd:
        sd["classifier.1.weight"] = sd["head.weight"]
    if "head.bias" in sd and "classifier.1.bias" not in sd:
        sd["classifier.1.bias"] = sd["head.bias"]

    if "fc.weight" in sd and "classifier.1.weight" not in sd:
        sd["classifier.1.weight"] = sd["fc.weight"]
    if "fc.bias" in sd and "classifier.1.bias" not in sd:
        sd["classifier.1.bias"] = sd["fc.bias"]

    if "classifier.weight" in sd and "classifier.1.weight" not in sd:
        sd["classifier.1.weight"] = sd["classifier.weight"]
    if "classifier.bias" in sd and "classifier.1.bias" not in sd:
        sd["classifier.1.bias"] = sd["classifier.bias"]

    if "_fc.weight" in sd and "classifier.1.weight" not in sd:
        sd["classifier.1.weight"] = sd["_fc.weight"]
    if "_fc.bias" in sd and "classifier.1.bias" not in sd:
        sd["classifier.1.bias"] = sd["_fc.bias"]

    return sd


def _load_state_with_report(model, state):
    missing, unexpected = model.load_state_dict(state, strict=False)

    changed = 0
    total = 0
    msd = model.state_dict()
    if isinstance(state, dict):
        for k, v in state.items():
            if (
                k in msd
                and isinstance(v, torch.Tensor)
                and isinstance(msd[k], torch.Tensor)
            ):
                total += 1
                if not torch.equal(msd[k].detach().cpu(), v.detach().cpu()):
                    pass
        changed = sum(1 for k in state.keys() if k in msd)
        total = len(msd)

    head_changed = 0
    head_keys = []
    if "classifier.1.weight" in msd:
        head_keys = ["classifier.1.weight", "classifier.1.bias"]
    elif "_fc.weight" in msd:
        head_keys = ["_fc.weight", "_fc.bias"]

    if isinstance(state, dict):
        for hk in head_keys:
            if hk in state:
                head_changed += 1

    print(
        f"[INFO] load_state_dict changed_tensors={changed}/{total} head_changed={head_changed}/{len(head_keys)} missing={len(missing)} unexpected={len(unexpected)}"
    )
    return changed, head_changed, missing, unexpected


weights_file = "../input/ramki-cassava-weights/weight.pt"
if not os.path.exists(weights_file):
    alt = "/kaggle/input/ramki-cassava-weights/weight.pt"
    if os.path.exists(alt):
        weights_file = alt

ckpt_state = None
ckpt_is_effnet_pytorch = False
if (not Training) and os.path.exists(weights_file):
    try:
        ckpt = torch.load(weights_file, map_location="cpu")
        ckpt_state = _strip_prefix(_extract_state_dict(ckpt))
        ckpt_is_effnet_pytorch = _prefer_efficientnet_pytorch(ckpt_state)
        print(
            f"[INFO] Detected checkpoint type: {'efficientnet_pytorch' if ckpt_is_effnet_pytorch else 'torchvision/other'}"
        )
    except Exception as e:
        print("[WARN] Could not read checkpoint to detect type:", repr(e))
        ckpt_state = None
        ckpt_is_effnet_pytorch = False

if not Training and ckpt_state is not None:
    if ckpt_is_effnet_pytorch:
        model = _build_effnet_pytorch_b7(num_classes=NUM_CLASSES, imagenet_init=True)
    else:
        model = build_model(num_classes=NUM_CLASSES, imagenet_fallback=False)
else:
    model = build_model(num_classes=NUM_CLASSES, imagenet_fallback=(not Training))

loaded_ok = False
if not Training and ckpt_state is not None:
    try:
        state = ckpt_state

        if ckpt_is_effnet_pytorch:
            state = _remap_head_keys_for_effnet_pytorch(state)
        else:
            state = _remap_head_keys_for_torchvision_efficientnet(state)

        changed, head_changed, missing, unexpected = _load_state_with_report(
            model, state
        )

        loaded_ok = (head_changed >= 2) and (changed >= 50)
        print("Loaded weights (non-strict):", weights_file, "| loaded_ok:", loaded_ok)
    except Exception as e:
        print(
            "[WARN] Could not load weights into selected EfficientNet implementation. Reason:",
            repr(e),
        )
        loaded_ok = False

if (not Training) and (not loaded_ok):
    print(
        "[WARN] Proceeding without external cassava weights; falling back to ImageNet backbone + new 5-class head."
    )
    model = build_model(num_classes=NUM_CLASSES, imagenet_fallback=True)

model.to(device)
model.eval()




## === cell 25
def _num_workers():
    try:
        return min(4, max(2, (os.cpu_count() or 4) // 2))
    except Exception:
        return 2


DL_NUM_WORKERS = _num_workers()
DL_PREFETCH = 4
DL_PERSISTENT = True

trainset = MyDataset(
    train_csv, transforms=transforms_train, test=False, cache_images=False
)
train_loader = DataLoader(
    trainset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=DL_NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(DL_PERSISTENT and DL_NUM_WORKERS > 0),
    prefetch_factor=DL_PREFETCH if DL_NUM_WORKERS > 0 else None,
)

testset = MyDataset(sample, transforms=transforms_valid, test=True, cache_images=False)
test_loader = DataLoader(
    testset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=DL_NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(DL_PERSISTENT and DL_NUM_WORKERS > 0),
    prefetch_factor=DL_PREFETCH if DL_NUM_WORKERS > 0 else None,
)


## === cell 26
len(train_loader)


## === cell 27
BATCH_SIZE




## === cell 28
class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count




## === cell 29
def train_model(model, epoch, dataloader_train, criterion, optimizer):
    model.train()
    losses = AverageMeter()
    accs = AverageMeter()
    tk = tqdm(dataloader_train, total=len(dataloader_train), position=0, leave=True)
    for idx, (imgs, labels) in enumerate(tk):
        imgs_train, labels_train = (
            imgs.to(device, non_blocking=True),
            labels.to(device, non_blocking=True).long(),
        )
        output_train = model(imgs_train)

        loss = criterion(output_train, labels_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        accs.update(
            (output_train.argmax(1) == labels_train).sum().item() / imgs_train.size(0),
            imgs_train.size(0),
        )
        losses.update(loss.item(), imgs_train.size(0))
        tk.set_postfix(loss=losses.avg, acc=accs.avg)
    return losses.avg


def test_model(model, dataloader_valid, criterion):
    model.eval()
    losses = AverageMeter()
    accs = AverageMeter()

    with torch.no_grad():
        tk = tqdm(dataloader_valid, total=len(dataloader_valid), position=0, leave=True)
        for idx, (imgs, labels) in enumerate(tk):
            imgs_valid, labels_valid = (
                imgs.to(device, non_blocking=True),
                labels.to(device, non_blocking=True).long(),
            )
            output_valid = model(imgs_valid)

            loss = criterion(output_valid, labels_valid)
            losses.update(loss.item(), imgs_valid.size(0))
            accs.update(
                (output_valid.argmax(1) == labels_valid).sum().item()
                / imgs_valid.size(0),
                imgs_valid.size(0),
            )
            tk.set_postfix(loss=losses.avg, acc=accs.avg)

    return losses.avg, accs.avg




## === cell 30
X = train_csv.iloc[:, :-1]
y = train_csv["label"].values




## === cell 31
def calibrate_head_if_needed(
    model, transforms_for_calib, max_samples: int = 4096, epochs: int = 2
):
    head_params = []
    backbone_params = []

    for n, p in model.named_parameters():
        if (
            "classifier.1" in n
            or n.endswith("classifier.1.weight")
            or n.endswith("classifier.1.bias")
        ):
            head_params.append(p)
        else:
            backbone_params.append(p)

    if len(head_params) == 0:
        for n, p in model.named_parameters():
            if n.startswith("_fc."):
                head_params.append(p)
            else:
                backbone_params.append(p)

    if len(head_params) == 0:
        print(
            "[WARN] Could not identify classifier head parameters; skipping calibration."
        )
        return model

    for p in backbone_params:
        p.requires_grad = False
    for p in head_params:
        p.requires_grad = True

    sss = StratifiedShuffleSplit(
        n_splits=1,
        test_size=min(max_samples, len(train_csv)) / len(train_csv),
        random_state=SEED,
    )
    _, va_idx = next(sss.split(train_csv["image_id"].values, train_csv["label"].values))
    calib_df = train_csv.iloc[va_idx].reset_index(drop=True)

    ds = MyDataset(
        calib_df, transforms=transforms_for_calib, test=False, cache_images=True
    )
    dl = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=DL_NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(DL_PERSISTENT and DL_NUM_WORKERS > 0),
        prefetch_factor=DL_PREFETCH if DL_NUM_WORKERS > 0 else None,
    )

    optimizer = torch.optim.AdamW(head_params, lr=LR, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for ep in range(epochs):
        losses = AverageMeter()
        accs = AverageMeter()
        tk = tqdm(dl, total=len(dl), position=0, leave=True)
        for imgs, labels in tk:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).long()
            logits = model(imgs)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            accs.update(
                (logits.argmax(1) == labels).float().mean().item(), imgs.size(0)
            )
            losses.update(loss.item(), imgs.size(0))
            tk.set_postfix(calib_ep=ep + 1, loss=losses.avg, acc=accs.avg)

    model.eval()
    return model


def evaluate_preproc_choice(norm_name: str, img_size: int, n_samples: int = 1024):
    sss = StratifiedShuffleSplit(
        n_splits=1,
        test_size=min(n_samples, len(train_csv)) / len(train_csv),
        random_state=SEED,
    )
    _, va_idx = next(sss.split(train_csv["image_id"].values, train_csv["label"].values))
    va_df = train_csv.iloc[va_idx].reset_index(drop=True)

    ds = MyDataset(
        va_df,
        transforms=make_valid_transform(norm_name, img_size),
        test=False,
        cache_images=True,
    )
    dl = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=DL_NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(DL_PERSISTENT and DL_NUM_WORKERS > 0),
        prefetch_factor=DL_PREFETCH if DL_NUM_WORKERS > 0 else None,
    )

    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for imgs, labels in dl:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).long()
            logits = model(imgs)
            pred = logits.argmax(1)
            correct += (pred == labels).sum().item()
            total += labels.numel()
    acc = correct / max(1, total)
    return acc, len(va_df)


def select_best_preproc_with_optional_calibration(
    norm_candidates, size_candidates, n_samples_eval=1024, calib_epochs=2
):
    best_choice = None
    best_acc = -1.0

    for sz in size_candidates:
        for nm in norm_candidates:
            try:
                acc0, used = evaluate_preproc_choice(nm, sz, n_samples=n_samples_eval)
                print(
                    f"[INFO] preproc=norm:{nm} size:{sz} pre_calib_acc={acc0:.4f} (n={used})"
                )

                acc_final = acc0

                if not loaded_ok:
                    tmp_model = build_model(
                        num_classes=NUM_CLASSES, imagenet_fallback=True
                    ).to(device)
                    tmp_model.eval()
                    tmp_model = calibrate_head_if_needed(
                        tmp_model,
                        make_valid_transform(nm, sz),
                        max_samples=4096,
                        epochs=calib_epochs,
                    )

                    global model
                    old_model = model
                    model = tmp_model
                    acc1, used1 = evaluate_preproc_choice(
                        nm, sz, n_samples=n_samples_eval
                    )
                    model = old_model

                    acc_final = acc1
                    print(
                        f"[INFO] preproc=norm:{nm} size:{sz} post_calib_acc={acc1:.4f} (n={used1})"
                    )

                if acc_final > best_acc:
                    best_acc = acc_final
                    best_choice = (nm, sz)
            except Exception as e:
                print(f"[WARN] preproc norm={nm} size={sz} evaluation failed:", repr(e))

    if best_choice is None:
        best_choice = ("imagenet", IMG_SIZE)
    return best_choice, best_acc


if not Training:
    norm_candidates = ["imagenet", "half", "none01"]
    size_candidates = [224, 512]

    best_choice, best_acc = select_best_preproc_with_optional_calibration(
        norm_candidates, size_candidates, n_samples_eval=1024, calib_epochs=2
    )
    best_norm, best_size = best_choice
    print(
        f"[INFO] Selected preprocessing for inference: norm={best_norm} size={best_size} (selected_acc={best_acc:.4f})"
    )

    if not loaded_ok:
        model = calibrate_head_if_needed(
            model,
            make_valid_transform(best_norm, best_size),
            max_samples=4096,
            epochs=2,
        )
        try:
            acc2, used2 = evaluate_preproc_choice(best_norm, best_size, n_samples=1024)
            print(
                f"[INFO] post-calibration holdout_acc={acc2:.4f} (n={used2}) using norm={best_norm} size={best_size}"
            )
        except Exception as e:
            print("[WARN] post-calibration evaluation failed:", repr(e))

    transforms_valid = make_valid_transform(best_norm, best_size)

    testset = MyDataset(
        sample, transforms=transforms_valid, test=True, cache_images=False
    )
    test_loader = DataLoader(
        testset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=DL_NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(DL_PERSISTENT and DL_NUM_WORKERS > 0),
        prefetch_factor=DL_PREFETCH if DL_NUM_WORKERS > 0 else None,
    )


## === cell 32
if Training:
    for i_fold, (train_idx, valid_idx) in enumerate(folds.split(X, y)):
        print("Fold {}/{}".format(i_fold + 1, N_FOLDS))

        valid = train_csv.iloc[valid_idx].reset_index(drop=True)
        train = train_csv.iloc[train_idx].reset_index(drop=True)

        dataset_train = MyDataset(train, transforms=transforms_train)
        dataset_valid = MyDataset(valid, transforms=transforms_valid)

        dataloader_train = DataLoader(
            dataset_train,
            batch_size=BATCH_SIZE,
            num_workers=DL_NUM_WORKERS,
            shuffle=True,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=(DL_PERSISTENT and DL_NUM_WORKERS > 0),
            prefetch_factor=DL_PREFETCH if DL_NUM_WORKERS > 0 else None,
        )
        dataloader_valid = DataLoader(
            dataset_valid,
            batch_size=BATCH_SIZE,
            num_workers=DL_NUM_WORKERS,
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=(DL_PERSISTENT and DL_NUM_WORKERS > 0),
            prefetch_factor=DL_PREFETCH if DL_NUM_WORKERS > 0 else None,
        )

        model = build_model(num_classes=NUM_CLASSES).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0)
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=1, verbose=True, min_lr=1e-5
        )

        best_acc = 0.0
        for epoch in range(N_EPOCHS):
            train_loss = train_model(
                model, epoch, dataloader_train, criterion, optimizer
            )
            val_loss, acc = test_model(model, dataloader_valid, criterion)
            scheduler.step(acc)

            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), "weight.pt")

            print("current_val_acc:", acc, "best_val_acc:", best_acc)


## === cell 33
test_pred = []

model.eval()
with torch.no_grad():
    for i, (images, _) in enumerate(tqdm(test_loader, position=0, leave=True)):
        images = images.to(device, non_blocking=True)

        logits1 = model(images)
        logits2 = model(torch.flip(images, dims=[3]))  # horizontal flip (W dimension)
        logits = (logits1 + logits2) / 2.0

        pred = logits.argmax(1).cpu().numpy().astype(int)
        test_pred.extend(pred.tolist())

sample["label"] = test_pred
sample = sample[["image_id", "label"]]
sample.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sample.shape)


## === cell 34
sample.head()


## === cell 35
with open("submission.csv", "r") as f:
    for _ in range(5):
        print(f.readline().strip())
