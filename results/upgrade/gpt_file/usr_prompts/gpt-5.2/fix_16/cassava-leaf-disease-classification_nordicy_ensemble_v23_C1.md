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

3.13

# 3. Installed packages

albumentations==2.0.8
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

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from tqdm import tqdm

SEED = 1337
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

try:
    torch.use_deterministic_algorithms(True)
except Exception as e:
    print("Warning: could not enable full deterministic algorithms:", repr(e))
    try:
        torch.use_deterministic_algorithms(False)
    except Exception:
        pass

if torch.cuda.is_available():
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass

try:
    cv2.setNumThreads(0)
except Exception:
    pass

torch.set_grad_enabled(True)

try:
    cv2.setUseOptimized(True)
except Exception:
    pass




## === cell 1
num_tta = 5




## === cell 2
CANDIDATE_COMP_DIRS = [
    "/kaggle/input/cassava-leaf-disease-classification",
    "/kaggle/data/cassava-leaf-disease-classification",
    "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
    "/kaggle/data/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
]
COMP_DIR = None
for d in CANDIDATE_COMP_DIRS:
    if os.path.exists(d):
        COMP_DIR = d
        break
if COMP_DIR is None:
    raise FileNotFoundError(
        "Could not locate cassava-leaf-disease-classification directory under expected paths."
    )

test_image_dir = os.path.join(COMP_DIR, "test_images")
train_image_dir = os.path.join(COMP_DIR, "train_images")
sample_sub_path = os.path.join(COMP_DIR, "sample_submission.csv")
train_csv_path = os.path.join(COMP_DIR, "train.csv")

print("Using COMP_DIR:", COMP_DIR)
print("test_image_dir exists:", os.path.exists(test_image_dir))
print("train_image_dir exists:", os.path.exists(train_image_dir))
print("sample_sub_path exists:", os.path.exists(sample_sub_path))
print("train_csv_path exists:", os.path.exists(train_csv_path))




## === cell 3
test_df = pd.read_csv(sample_sub_path)
test_df.head()




## === cell 4
efficientnet_transforms = A.Compose(
    [
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
        A.Resize(384, 384),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)




## === cell 5
class CassavaTestDataset(Dataset):
    def __init__(self, dataframe, image_dir):
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_dir = image_dir

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = self.dataframe.iloc[idx, 0]  # image_id
        img_path = os.path.join(self.image_dir, img_name)

        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image, img_name




## === cell 6
tta_transform = A.ReplayCompose(
    [
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=30, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
        A.Resize(384, 384),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)




## === cell 7
def collate_images_and_names(batch):
    images = [b[0] for b in batch]
    names = [str(b[1]) for b in batch]
    return images, names


test_dataset = CassavaTestDataset(test_df, test_image_dir)

_cpu = os.cpu_count() or 2
_num_workers = min(12, _cpu)  # a bit higher helps keep GPU busy on Kaggle CPUs
TEST_BATCH_SIZE = (
    48 if torch.cuda.is_available() else 8
)  # larger batch reduces per-batch CPU TTA overhead

test_loader = DataLoader(
    test_dataset,
    batch_size=TEST_BATCH_SIZE,
    shuffle=False,
    num_workers=_num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(_num_workers > 0),
    prefetch_factor=4 if _num_workers > 0 else None,
    collate_fn=collate_images_and_names,
)




## === cell 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)




## === cell 9
def _extract_state_dict(loaded_obj):
    if isinstance(loaded_obj, dict):
        for k in ["state_dict", "model_state_dict", "model", "net", "weights"]:
            if k in loaded_obj and isinstance(loaded_obj[k], dict):
                return loaded_obj[k]
    return loaded_obj


def _strip_prefix_if_present(state_dict, prefix="module."):
    if not isinstance(state_dict, dict):
        return state_dict
    if any(k.startswith(prefix) for k in state_dict.keys()):
        return {
            k[len(prefix) :] if k.startswith(prefix) else k: v
            for k, v in state_dict.items()
        }
    return state_dict


def safe_load_state_dict(model, ckpt_path, device):
    if ckpt_path is None or (not os.path.exists(ckpt_path)):
        return False
    obj = torch.load(ckpt_path, map_location=device)
    state = _extract_state_dict(obj)
    state = _strip_prefix_if_present(state, prefix="module.")
    try:
        model.load_state_dict(state, strict=True)
        return True
    except RuntimeError:
        model.load_state_dict(state, strict=False)
        return True


def find_first_existing(paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


RESNET_CKPT_CANDIDATES = [
    "/kaggle/input/casava-aug/pytorch/default/1/cassava_leaf_best_model_fine_aug.pth",
]
EFF1_CKPT_CANDIDATES = [
    "/kaggle/input/eff-5/pytorch/default/1/Eff_best5.pth",
]
EFF7_CKPT_CANDIDATES = [
    "/kaggle/input/effbest10temp/pytorch/default/1/Eff_best10.pth",
]
EFF8_CKPT_CANDIDATES = [
    "/kaggle/input/eff-6/pytorch/default/1/Eff_best6.pth",
]

resnet_ckpt = find_first_existing(RESNET_CKPT_CANDIDATES)
eff1_ckpt = find_first_existing(EFF1_CKPT_CANDIDATES)
eff7_ckpt = find_first_existing(EFF7_CKPT_CANDIDATES)
eff8_ckpt = find_first_existing(EFF8_CKPT_CANDIDATES)

print(
    "Found ckpts:",
    {"resnet": resnet_ckpt, "eff1": eff1_ckpt, "eff7": eff7_ckpt, "eff8": eff8_ckpt},
)

NO_CUSTOM_CKPTS = all(p is None for p in [eff1_ckpt, eff7_ckpt, eff8_ckpt, resnet_ckpt])
print("NO_CUSTOM_CKPTS:", NO_CUSTOM_CKPTS)




## === cell 10
def _to_channels_last(m: torch.nn.Module) -> torch.nn.Module:
    try:
        return m.to(memory_format=torch.channels_last)
    except Exception:
        return m


resnet_model = models.resnet50(
    weights=models.ResNet50_Weights.IMAGENET1K_V2 if NO_CUSTOM_CKPTS else None
)
num_ftrs = resnet_model.fc.in_features
resnet_model.fc = nn.Linear(num_ftrs, 5)
_loaded_resnet = (
    safe_load_state_dict(resnet_model, resnet_ckpt, device) if resnet_ckpt else False
)
print("resnet loaded custom:", _loaded_resnet)
resnet_model = _to_channels_last(resnet_model.to(device))
resnet_model.eval()




## === cell 11
def build_efficientnet_v2_s(num_classes=5, dropout_p=0.8, use_imagenet_weights=False):
    if use_imagenet_weights:
        m = models.efficientnet_v2_s(
            weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
        )
    else:
        m = models.efficientnet_v2_s(weights=None)
    num_features = m.classifier[1].in_features
    m.classifier = nn.Sequential(
        nn.Dropout(p=dropout_p), nn.Linear(num_features, num_classes)
    )
    return m


efficientnet_model_1 = build_efficientnet_v2_s(use_imagenet_weights=(eff1_ckpt is None))
loaded1 = (
    safe_load_state_dict(efficientnet_model_1, eff1_ckpt, device)
    if eff1_ckpt
    else False
)
efficientnet_model_1 = _to_channels_last(efficientnet_model_1.to(device).eval())
print("eff1 loaded custom:", loaded1)

efficientnet_model_7 = build_efficientnet_v2_s(use_imagenet_weights=(eff7_ckpt is None))
loaded7 = (
    safe_load_state_dict(efficientnet_model_7, eff7_ckpt, device)
    if eff7_ckpt
    else False
)
efficientnet_model_7 = _to_channels_last(efficientnet_model_7.to(device).eval())
print("eff7 loaded custom:", loaded7)

efficientnet_model_8 = build_efficientnet_v2_s(use_imagenet_weights=(eff8_ckpt is None))
loaded8 = (
    safe_load_state_dict(efficientnet_model_8, eff8_ckpt, device)
    if eff8_ckpt
    else False
)
efficientnet_model_8 = _to_channels_last(efficientnet_model_8.to(device).eval())
print("eff8 loaded custom:", loaded8)


def _maybe_compile(m: torch.nn.Module) -> torch.nn.Module:
    return m


efficientnet_model_1 = _maybe_compile(efficientnet_model_1)
efficientnet_model_7 = _maybe_compile(efficientnet_model_7)
efficientnet_model_8 = _maybe_compile(efficientnet_model_8)




## === cell 12
class CassavaTrainDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform):
        self.df = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]["image_id"]
        y = int(self.df.iloc[idx]["label"])
        img_path = os.path.join(self.image_dir, img_name)
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        x = self.transform(image=image)["image"]
        return x, y


def train_one_model_head_only(model, train_loader, device, epochs=1, lr=3e-4):
    model.train()
    for p in model.parameters():
        p.requires_grad = False

    trainable = []
    if hasattr(model, "classifier"):
        for p in model.classifier.parameters():
            p.requires_grad = True
        trainable += list(model.classifier.parameters())
    if hasattr(model, "fc"):
        for p in model.fc.parameters():
            p.requires_grad = True
        trainable += list(model.fc.parameters())

    opt = torch.optim.AdamW(trainable, lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(epochs):
        for xb, yb in tqdm(
            train_loader, desc=f"Train head ({model.__class__.__name__})", leave=False
        ):
            xb = xb.to(device, non_blocking=True)
            yb = torch.as_tensor(yb, dtype=torch.long, device=device)
            opt.zero_grad(set_to_none=True)
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()

    model.eval()
    return model


if NO_CUSTOM_CKPTS:
    train_df = pd.read_csv(train_csv_path)
    idx = np.arange(len(train_df))
    rng = np.random.RandomState(SEED)
    rng.shuffle(idx)
    train_idx = idx[: int(0.98 * len(idx))]
    train_small = train_df.iloc[train_idx].reset_index(drop=True)

    train_ds = CassavaTrainDataset(
        train_small, train_image_dir, efficientnet_transforms
    )

    g = torch.Generator()
    g.manual_seed(SEED)

    train_loader = DataLoader(
        train_ds,
        batch_size=32,
        shuffle=True,
        generator=g,
        num_workers=_num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(_num_workers > 0),
        prefetch_factor=(4 if _num_workers > 0 else None),
    )

    efficientnet_model_1 = train_one_model_head_only(
        efficientnet_model_1, train_loader, device, epochs=2, lr=3e-4
    )
    efficientnet_model_7 = train_one_model_head_only(
        efficientnet_model_7, train_loader, device, epochs=2, lr=3e-4
    )
    efficientnet_model_8 = train_one_model_head_only(
        efficientnet_model_8, train_loader, device, epochs=2, lr=3e-4
    )




## === cell 13
def _apply_albu_replay_with_seed(
    replay_compose_tfm: A.ReplayCompose, image_np, seed: int
):
    py_state = random.getstate()
    np_state = np.random.get_state()
    try:
        random.seed(int(seed))
        np.random.seed(int(seed) % (2**32 - 1))
        out = replay_compose_tfm(image=image_np)
        return out["image"], out["replay"]
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)


def _make_big_tta_batch_stack_in_memory(images_np, n_tta, seed_base: int):
    bsz = len(images_np)
    out_tensors = [None] * (bsz * n_tta)
    k = 0
    for i in range(bsz):
        img = images_np[i]
        for t in range(n_tta):
            base_t = int(seed_base + t * 10000)
            s = base_t + i
            img_t, _replay = _apply_albu_replay_with_seed(tta_transform, img, seed=s)
            out_tensors[k] = img_t
            k += 1
    big = torch.stack(out_tensors, dim=0)
    return big


ensemble_predictions = []
image_names = []

models_tuple = (efficientnet_model_1, efficientnet_model_7, efficientnet_model_8)

with torch.inference_mode():
    for batch_i, (images_np, names) in enumerate(
        tqdm(test_loader, total=len(test_loader), desc="Infer")
    ):
        seed_base = SEED + 100000 * batch_i

        big_batch_cpu = _make_big_tta_batch_stack_in_memory(
            images_np, num_tta, seed_base
        )
        try:
            big_batch_cpu = big_batch_cpu.contiguous(memory_format=torch.channels_last)
        except Exception:
            pass

        if torch.cuda.is_available():
            big_batch = big_batch_cpu.to(device, non_blocking=True)
        else:
            big_batch = big_batch_cpu.to(device)

        bsz = len(images_np)
        probs_sum = None

        for m in models_tuple:
            logits = m(big_batch)
            probs = F.softmax(logits, dim=1).view(bsz, num_tta, -1).mean(dim=1)
            probs_sum = probs if probs_sum is None else (probs_sum + probs)

        combined = probs_sum / 3.0
        preds = combined.argmax(dim=1).detach().cpu().numpy().astype(int).tolist()

        ensemble_predictions.extend(preds)
        image_names.extend([str(n) for n in names])




## === cell 14
submission_df = pd.DataFrame({"image_id": image_names, "label": ensemble_predictions})

submission_df = test_df[["image_id"]].merge(submission_df, on="image_id", how="left")
if submission_df["label"].isna().any():
    submission_df["label"] = submission_df["label"].fillna(0).astype(int)
else:
    submission_df["label"] = submission_df["label"].astype(int)

submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)

print(f"Saved: {submission_path}")
print(submission_df.head())
print("Rows:", len(submission_df), "Cols:", submission_df.columns.tolist())
print("Missing labels after merge:", int(submission_df["label"].isna().sum()))
