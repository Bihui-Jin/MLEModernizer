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
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch import nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm
from tqdm import tqdm



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
TTA = 10

device = DEVICES[0] if len(DEVICES) else torch.device("cpu")




## === cell 2
def sigmoid_focal_cross_entropy(y_hat, y_true, alpha=0.25, gamma=2.0):
    """
    Note: This function is not used in the current inference-only script,
    but keep it correct to avoid silent bugs if training is added back.
    """

    def smooth(y, smooth_factor):
        assert len(y.shape) == 2
        y = y * (1 - smooth_factor) + smooth_factor / y.shape[1]
        return y

    smooth_factor = 0.1

    if not isinstance(y_true, torch.Tensor):
        y_true = torch.tensor(y_true)
    if not isinstance(y_hat, torch.Tensor):
        y_hat = torch.tensor(y_hat)

    y_true = smooth(y_true, smooth_factor).to(dtype=y_hat.dtype, device=y_hat.device)

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
plt.title("LR schedule")
plt.show()



## === cell 4
train_augs = A.Compose(
    [
        A.RandomResizedCrop(size=(IMAGE_SIZE, IMAGE_SIZE), scale=(0.08, 1.0)),
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
        A.CoarseDropout(p=0.5),  # replacement for removed Cutout
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
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.CenterCrop(IMAGE_SIZE, IMAGE_SIZE),
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
my_model_1 = timm.create_model(model_name1, pretrained=False, num_classes=OUT_FEATURES)
my_model_1



## === cell 8
model_name2 = "tf_efficientnet_b4_ns"
my_model_2 = timm.create_model(model_name2, pretrained=False, num_classes=OUT_FEATURES)
my_model_2



## === cell 9
torch.cuda.empty_cache()




## === cell 10
def _extract_state_dict(ckpt_obj):
    """
    Score-relevant robustness:
    support common wrappers: state_dict/model/net/ema/... and nested dicts.
    """
    if isinstance(ckpt_obj, dict) and len(ckpt_obj) > 0:
        for k in (
            "state_dict",
            "model_state_dict",
            "model",
            "net",
            "network",
            "module",
            "ema",
            "ema_state_dict",
            "model_ema",
            "student",
            "student_state_dict",
            "teacher",
            "teacher_state_dict",
        ):
            v = ckpt_obj.get(k, None)
            if isinstance(v, dict) and len(v) > 0:
                nested = v.get("state_dict", None) if isinstance(v, dict) else None
                if isinstance(nested, dict) and len(nested) > 0:
                    return nested
                return v

        if all(isinstance(k, str) for k in ckpt_obj.keys()) and any(
            (".weight" in k) or (".bias" in k) for k in ckpt_obj.keys()
        ):
            return ckpt_obj

    return None


def _strip_common_prefixes(state: dict) -> dict:
    """
    Score-relevant fix: handle DataParallel/torch.compile prefixes so keys match.
    """
    if not isinstance(state, dict) or len(state) == 0:
        return state

    def strip_prefix(d, prefix):
        if any(k.startswith(prefix) for k in d.keys()):
            return {
                (k[len(prefix) :] if k.startswith(prefix) else k): v
                for k, v in d.items()
            }
        return d

    state = strip_prefix(state, "module._orig_mod.")
    state = strip_prefix(state, "_orig_mod.")
    state = strip_prefix(state, "module.")
    state = strip_prefix(state, "model.")
    return state


def _maybe_cast_fp16_to_fp32(state: dict) -> dict:
    """
    Score-relevant: some checkpoints save fp16 weights; loading into fp32 model is fine,
    but ensure tensors are float32 on CPU to avoid dtype surprises.
    """
    if not isinstance(state, dict) or len(state) == 0:
        return state
    out = {}
    for k, v in state.items():
        if torch.is_tensor(v) and v.dtype in (torch.float16, torch.bfloat16):
            out[k] = v.float()
        else:
            out[k] = v
    return out


def _head_key_present(state_keys: set, possible_prefixes: list[str]) -> bool:
    for p in possible_prefixes:
        if any(k.startswith(p) for k in state_keys):
            return True
    return False


def _load_state_dict_if_exists(model: nn.Module, ckpt_path: str) -> bool:
    """
    Tries to load a checkpoint if present. Returns True if loaded and *effective*.
    """
    if not os.path.exists(ckpt_path):
        return False

    ckpt_obj = torch.load(ckpt_path, map_location="cpu")
    state = _extract_state_dict(ckpt_obj)
    if state is None:
        return False

    state = _strip_common_prefixes(state)
    state = _maybe_cast_fp16_to_fp32(state)

    target = model.module if isinstance(model, nn.DataParallel) else model
    model_keys = set(target.state_dict().keys())
    state_keys = set(state.keys())

    matched = len(model_keys & state_keys)

    model_head_prefixes = ["fc.", "classifier.", "head.", "head.fc.", "classifier.fc."]
    model_has_head = any(
        any(k.startswith(p) for k in model_keys) for p in model_head_prefixes
    )
    head_ok = (
        _head_key_present(state_keys, model_head_prefixes) if model_has_head else True
    )

    incompatible = target.load_state_dict(state, strict=False)
    try:
        n_missing = len(incompatible.missing_keys)
        n_unexpected = len(incompatible.unexpected_keys)
    except Exception:
        n_missing, n_unexpected = -1, -1

    print(
        f"Loaded ckpt={ckpt_path} with strict=False: matched_keys={matched}/{len(model_keys)}, "
        f"missing={n_missing}, unexpected={n_unexpected}, head_present={head_ok}"
    )

    if (matched < max(50, int(0.20 * len(model_keys)))) or (not head_ok):
        return False
    return True


def _resolve_ckpt_path(input_path: str, ckpt_name: str) -> str:
    candidates = [
        os.path.join(input_path, ckpt_name),
        os.path.join(input_path, "cassava-leaf-disease-classification", ckpt_name),
        os.path.join("../input", ckpt_name),
        os.path.join("../input", "cassava-leaf-disease-classification", ckpt_name),
        os.path.join("/kaggle/input", ckpt_name),
        os.path.join("/kaggle/input/cassava-leaf-disease-classification", ckpt_name),
        os.path.join(
            "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
            ckpt_name,
        ),
        os.path.join("../working", ckpt_name),
        os.path.join("/kaggle/working", ckpt_name),
        os.path.join("./", ckpt_name),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c

    search_roots = [
        Path("/kaggle/input"),
        Path("/kaggle/working"),
        Path("../input"),
        Path("../working"),
        Path(input_path),
        Path("."),
    ]

    for root in search_roots:
        if not root.exists() or not root.is_dir():
            continue
        try:
            hits = list(root.rglob(ckpt_name))
            if hits:
                hits = sorted(hits, key=lambda p: len(str(p)))
                return str(hits[0])
        except Exception:
            pass

    return candidates[0]


class CassavaDataset(torch.utils.data.Dataset):
    def __init__(self, df, image_dir, transforms=None):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row["image_id"])
        image = Image.open(img_path).convert("RGB")
        if self.transforms is not None:
            image = self.transforms(image=np.array(image))["image"]
        label = int(row["label"])
        return image, label


def _get_head_attr(target: nn.Module) -> str:
    """
    Minimal helper: find the attribute name of the classifier head module used by timm models.
    """
    for attr in ("fc", "classifier", "head"):
        if hasattr(target, attr) and isinstance(getattr(target, attr), nn.Module):
            return attr
    raise RuntimeError(
        "Could not locate a classifier head attribute among: fc/classifier/head"
    )


def _infer_head_in_features(head_module: nn.Module) -> int:
    """
    Determine the input feature dim expected by the classifier head.
    Supports Linear and Sequential with last Linear.
    """
    if isinstance(head_module, nn.Linear):
        return head_module.in_features
    if isinstance(head_module, nn.Sequential):
        for m in reversed(list(head_module.modules())):
            if isinstance(m, nn.Linear):
                return m.in_features
    for m in reversed(list(head_module.modules())):
        if isinstance(m, nn.Linear):
            return m.in_features
    raise RuntimeError("Could not infer head in_features from head module")


def _linear_probe_fit_head_from_train_csv(model: nn.Module, max_epochs: int = 2):
    """
    Bug fix (runtime) + score-relevant:
    Previously used a separate features_only extractor whose channel dim may not match the model head,
    causing size mismatch when copying weights (e.g., EfficientNet B4).
    Now we extract pooled features using the SAME model with its head temporarily replaced by Identity,
    ensuring the feature dim exactly matches the head's expected input dim.
    """
    train_df = pd.read_csv(TRAIN_CSV_PATH)
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(len(train_df))
    n_val = int(0.05 * len(train_df))
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]
    tr_df = train_df.iloc[tr_idx].reset_index(drop=True)
    va_df = train_df.iloc[val_idx].reset_index(drop=True)

    ds_tr = CassavaDataset(tr_df, TRAIN_IMAGE_PATH, transforms=train_augs)
    ds_va = CassavaDataset(va_df, TRAIN_IMAGE_PATH, transforms=valid_augs)

    dl_tr = torch.utils.data.DataLoader(
        ds_tr,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    dl_va = torch.utils.data.DataLoader(
        ds_va,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    target = model.module if isinstance(model, nn.DataParallel) else model
    target.to(device)
    target.eval()

    head_attr = _get_head_attr(target)
    head_module = getattr(target, head_attr)
    feat_dim = _infer_head_in_features(head_module)

    original_head = head_module
    setattr(target, head_attr, nn.Identity())

    def forward_features_pooled(x: torch.Tensor) -> torch.Tensor:
        out = target(x)
        if out.ndim == 4:
            out = out.mean(dim=(2, 3))
        if out.ndim != 2:
            raise RuntimeError(
                f"Unexpected feature tensor shape from model forward: {tuple(out.shape)}"
            )
        return out

    xb0, _ = next(iter(dl_tr))
    xb0 = xb0.to(device=device, dtype=torch.float32)
    with torch.no_grad():
        feats0 = forward_features_pooled(xb0)
    if feats0.shape[1] != feat_dim:
        setattr(target, head_attr, original_head)
        raise RuntimeError(
            f"Feature dim mismatch: inferred head in_features={feat_dim}, "
            f"but model-with-Identity produced {feats0.shape[1]}"
        )

    head = nn.Linear(feat_dim, OUT_FEATURES).to(device)
    nn.init.xavier_uniform_(head.weight)
    if head.bias is not None:
        nn.init.zeros_(head.bias)

    opt = OPTIMIZER(head.parameters(), lr=1e-3, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()

    for epoch in range(max_epochs):
        head.train()
        tr_loss = 0.0
        n_seen = 0
        for xb, yb in dl_tr:
            xb = xb.to(device=device, dtype=torch.float32)
            yb = yb.to(device=device, dtype=torch.long)

            with torch.no_grad():
                feats = forward_features_pooled(xb)

            logits = head(feats)
            loss = ce(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            tr_loss += loss.item() * xb.size(0)
            n_seen += xb.size(0)

        head.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in dl_va:
                xb = xb.to(device=device, dtype=torch.float32)
                yb = yb.to(device=device, dtype=torch.long)
                feats = forward_features_pooled(xb)
                logits = head(feats)
                pred = logits.argmax(dim=1)
                correct += (pred == yb).sum().item()
                total += yb.numel()

        print(
            f"[Linear-probe fallback] epoch={epoch+1}/{max_epochs} "
            f"train_loss={tr_loss/max(n_seen,1):.4f} val_acc={correct/max(total,1):.4f}"
        )

    setattr(target, head_attr, original_head)

    if isinstance(original_head, nn.Linear):
        with torch.no_grad():
            original_head.weight.copy_(head.weight)
            original_head.bias.copy_(head.bias)
    elif isinstance(original_head, nn.Sequential):
        last_linear = None
        for m in reversed(list(original_head.modules())):
            if isinstance(m, nn.Linear):
                last_linear = m
                break
        if last_linear is None:
            raise RuntimeError(
                "Sequential head found but no Linear layer inside to copy weights into."
            )
        if last_linear.weight.shape != head.weight.shape:
            raise RuntimeError(
                f"Head copy shape mismatch: target {tuple(last_linear.weight.shape)} vs trained {tuple(head.weight.shape)}"
            )
        with torch.no_grad():
            last_linear.weight.copy_(head.weight)
            if last_linear.bias is not None and head.bias is not None:
                last_linear.bias.copy_(head.bias)
    else:
        raise RuntimeError(
            f"Unsupported head module type for copying: {type(original_head)}"
        )

    return True


test_image_list = np.asarray(
    sorted(
        [
            image_name
            for image_name in os.listdir(TEST_IMAGE_PATH)
            if image_name.lower().endswith(".jpg")
        ]
    )
)

if len(test_image_list) == 0:
    raise RuntimeError(f"No .jpg files found under TEST_IMAGE_PATH={TEST_IMAGE_PATH}")

my_model_1 = (
    nn.DataParallel(my_model_1).to(device)
    if torch.cuda.is_available() and len(DEVICES) > 1
    else my_model_1.to(device)
)
my_model_2 = (
    nn.DataParallel(my_model_2).to(device)
    if torch.cuda.is_available() and len(DEVICES) > 1
    else my_model_2.to(device)
)

ckpt1 = _resolve_ckpt_path(INPUT_PATH, RESNEXT_PATH)
ckpt2 = _resolve_ckpt_path(INPUT_PATH, B4_PATH)

loaded1 = _load_state_dict_if_exists(my_model_1, ckpt1)
loaded2 = _load_state_dict_if_exists(my_model_2, ckpt2)

need_fallback = not (loaded1 and loaded2)
if need_fallback:
    print(
        "At least one checkpoint not loaded effectively -> using fast linear-probe head training for BOTH models."
    )
    _linear_probe_fit_head_from_train_csv(my_model_1, max_epochs=2)
    _linear_probe_fit_head_from_train_csv(my_model_2, max_epochs=2)
else:
    print("Both checkpoints loaded effectively.")
    print(f"  model1: {ckpt1}")
    print(f"  model2: {ckpt2}")

preds_1 = []
my_model_1.eval()

for single_image_name in tqdm(
    test_image_list,
    desc=f"Infer model1 ({'ckpt' if (not need_fallback) else 'linear-probe-fallback'})",
):
    with torch.no_grad():
        ans = torch.zeros(OUT_FEATURES, device=device)
        for _ in range(1):
            image = Image.open(
                os.path.join(TEST_IMAGE_PATH, single_image_name)
            ).convert("RGB")
            aug_image = test_augs(image=np.array(image))["image"]  # torch.Tensor CxHxW
            test_image = aug_image.unsqueeze(0).to(device=device, dtype=torch.float32)
            ans += my_model_1(test_image).view(ans.shape)
        preds_1.append(ans.detach().cpu())

predictions_1 = torch.stack(preds_1, dim=0)
normalize_pred_1 = F.normalize(predictions_1.T, p=2, dim=0).T
torch.cuda.empty_cache()

preds_2 = []
my_model_2.eval()

for single_image_name in tqdm(
    test_image_list,
    desc=f"Infer model2 ({'ckpt' if (not need_fallback) else 'linear-probe-fallback'})",
):
    with torch.no_grad():
        ans = torch.zeros(OUT_FEATURES, device=device)
        for _ in range(TTA):
            image = Image.open(
                os.path.join(TEST_IMAGE_PATH, single_image_name)
            ).convert("RGB")
            aug_image = test_augs(image=np.array(image))["image"]
            test_image = aug_image.unsqueeze(0).to(device=device, dtype=torch.float32)
            ans += my_model_2(test_image).view(ans.shape)
        ans /= TTA
        preds_2.append(ans.detach().cpu())

predictions_2 = torch.stack(preds_2, dim=0)
normalize_pred_2 = F.normalize(predictions_2.T, p=2, dim=0).T

final_pred = (normalize_pred_1 * 0.45) + (normalize_pred_2 * 0.55)
label = final_pred.argmax(dim=-1).detach().cpu().numpy().astype(int).tolist()

df_submission = pd.DataFrame({"image_id": test_image_list.tolist(), "label": label})
df_submission.to_csv(SUBMISSION_PATH, index=False)

print(f"Wrote submission to: {SUBMISSION_PATH}")
print(df_submission.head())
print(
    f"Checkpoints effective: model1={loaded1} (tried {ckpt1}), model2={loaded2} (tried {ckpt2})"
)
print(f"Submission rows: {len(df_submission)} (expected {len(test_image_list)})")
if len(df_submission) != len(test_image_list):
    raise RuntimeError("Submission row count does not match number of test images.")
