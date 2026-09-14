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
import glob
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torchvision import models
from torchvision.models import EfficientNet_V2_S_Weights, MobileNet_V3_Large_Weights

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False




## === cell 1
num_tta = 5




## === cell 2
test_image_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images"
train_image_dir = "/kaggle/input/cassava-leaf-disease-classification/train_images"




## === cell 3
test_df = pd.read_csv(
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)
train_df = pd.read_csv("/kaggle/input/cassava-leaf-disease-classification/train.csv")
test_df.head()




## === cell 4
EFF_SIZE = 384

eff_normalize = A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
eff_to_tensor = ToTensorV2()


def preprocess_efficientnet_np_rgb_from_albu(augmented_np_rgb):
    """augmented_np_rgb: HWC uint8/float RGB -> CHW float tensor normalized as EfficientNet expects."""
    out = A.Compose([eff_normalize, eff_to_tensor])(image=augmented_np_rgb)["image"]
    return out


mobilenet_base_transform = A.Compose(
    [
        A.Resize(224, 224),
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
        img_name = str(self.dataframe.iloc[idx, 0])  # image_id
        img_path = os.path.join(self.image_dir, img_name)

        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image, img_name




## === cell 6
class CassavaTrainDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform):
        self.df = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = str(self.df.iloc[idx]["image_id"])
        y = int(self.df.iloc[idx]["label"])
        img_path = os.path.join(self.image_dir, img_name)

        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        x = self.transform(image=image)["image"]
        return x, y




## === cell 7
common_transforms_rgb_only = [
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=20, p=0.7),
    A.HueSaturationValue(
        hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.5
    ),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.HorizontalFlip(p=0.5),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.4),
]

tta_transform_rgb_eff = A.Compose(
    common_transforms_rgb_only
    + [
        A.RandomResizedCrop(size=(EFF_SIZE, EFF_SIZE), scale=(0.8, 1.0), p=1.0),
    ]
)

tta_transform_mobilenet = A.Compose(
    common_transforms_rgb_only
    + [
        A.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0), p=1.0),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)




## === cell 8
def identity_collate(batch):
    return batch


test_dataset = CassavaTestDataset(test_df, test_image_dir)

_num_workers = 0
test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=_num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=False,
    prefetch_factor=None,
    collate_fn=identity_collate,
)




## === cell 9
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device




## === cell 10
def _try_load_state_dict(model: torch.nn.Module, ckpt_path: str) -> bool:
    try:
        sd = torch.load(ckpt_path, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        if isinstance(sd, dict):
            new_sd = {}
            for k, v in sd.items():
                nk = k
                if nk.startswith("module."):
                    nk = nk[len("module.") :]
                if nk.startswith("model."):
                    nk = nk[len("model.") :]
                new_sd[nk] = v
            sd = new_sd
        missing, unexpected = model.load_state_dict(sd, strict=False)
        loaded_any = (len(missing) < len(model.state_dict())) and (
            len(unexpected) <= len(sd)
        )
        return bool(loaded_any)
    except Exception:
        return False


def find_candidate_checkpoints(patterns):
    roots = ["/kaggle/input", "/kaggle/working", "/kaggle/data", "/kaggle"]
    found = []
    for r in roots:
        for pat in patterns:
            found.extend(glob.glob(os.path.join(r, "**", pat), recursive=True))
    found = sorted(set(found), key=lambda x: (len(x), x))
    return found


ALL_CKPTS = find_candidate_checkpoints(["*.pth", "*.pt", "*.ckpt"])




## === cell 11
resnet_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
num_ftrs = resnet_model.fc.in_features
resnet_model.fc = nn.Linear(num_ftrs, 5)
resnet_model = resnet_model.to(device)




## === cell 12
from sklearn.model_selection import train_test_split

resnet_ckpt_path = "/kaggle/working/resnet50_cassava_fc.pth"

resnet_train_tf = A.Compose(
    [
        A.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)
resnet_val_tf = A.Compose(
    [
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)

train_idx, val_idx = train_test_split(
    np.arange(len(train_df)),
    test_size=0.08,
    random_state=SEED,
    stratify=train_df["label"].values,
)
tr_df = train_df.iloc[train_idx].reset_index(drop=True)
va_df = train_df.iloc[val_idx].reset_index(drop=True)

tr_ds = CassavaTrainDataset(tr_df, train_image_dir, resnet_train_tf)
va_ds = CassavaTrainDataset(va_df, train_image_dir, resnet_val_tf)

train_loader = DataLoader(
    tr_ds,
    batch_size=64,
    shuffle=True,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=True if 2 > 0 else False,
)
val_loader = DataLoader(
    va_ds,
    batch_size=128,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=True if 2 > 0 else False,
)

for p in resnet_model.parameters():
    p.requires_grad = False
for p in resnet_model.fc.parameters():
    p.requires_grad = True

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(resnet_model.fc.parameters(), lr=2e-3, weight_decay=1e-4)


def _evaluate_resnet(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.inference_mode():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = model(xb)
            pred = logits.argmax(dim=1)
            correct += (pred == yb).sum().item()
            total += yb.numel()
    return correct / max(1, total)


if os.path.exists(resnet_ckpt_path):
    sd = torch.load(resnet_ckpt_path, map_location="cpu")
    resnet_model.load_state_dict(sd, strict=True)
else:
    resnet_model.train()
    epochs = 3  # small, to stay within runtime; enough to move score meaningfully from ~0.10 upward
    for ep in range(epochs):
        resnet_model.train()
        running = 0.0
        n = 0
        for xb, yb in tqdm(
            train_loader, desc=f"Train ResNet fc ep{ep+1}/{epochs}", leave=False
        ):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = resnet_model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            running += float(loss.item()) * yb.size(0)
            n += yb.size(0)

        val_acc = _evaluate_resnet(resnet_model, val_loader)
        print(
            f"[ResNet fc] ep {ep+1}/{epochs} train_loss={running/max(1,n):.4f} val_acc={val_acc:.4f}"
        )

    torch.save(resnet_model.state_dict(), resnet_ckpt_path)

resnet_model.eval()
print(
    "ResNet checkpoint:", resnet_ckpt_path, "exists:", os.path.exists(resnet_ckpt_path)
)




## === cell 13
def build_efficientnet_cassava_or_fallback(dropout_p: float, tag: str, ckpts):
    try:
        m = torch.hub.load(
            "pytorch/vision",
            "efficientnet_v2_s",
            weights="EfficientNet_V2_S_Weights.IMAGENET1K_V1",
        )
        in_features = m.classifier[1].in_features
        m.classifier = nn.Sequential(nn.Dropout(p=dropout_p), nn.Linear(in_features, 5))

        from torchvision.models import EfficientNet_V2_S_Weights as _EV2W

        candidate_weight_names = [
            "CASSAVA_LEAF_DISEASE",
            "CASSAVA_LEAF_DISEASES",
            "CASSAVA",
            "CASSAVA_LEAF_DISEASE_CLASSIFICATION",
        ]
        loaded = False
        loaded_path = None

        for wn in candidate_weight_names:
            w = getattr(_EV2W, wn, None)
            if w is None:
                continue
            try:
                sd = w.get_state_dict(progress=False)
                m.load_state_dict(sd, strict=False)
                loaded = True
                loaded_path = f"torchvision:{wn}"
                break
            except Exception:
                pass

        if loaded:
            return m.to(device).eval(), True, loaded_path

    except Exception:
        pass

    m = models.efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
    in_features = m.classifier[1].in_features
    m.classifier = nn.Sequential(nn.Dropout(p=dropout_p), nn.Linear(in_features, 5))

    preferred = []
    for p in ckpts:
        lp = p.lower()
        if any(
            s in lp
            for s in ["cassava", "efficientnet", "effnet", "eff", "v2", "ev2", "fold"]
        ):
            preferred.append(p)
    preferred = preferred if preferred else ckpts

    loaded_path = None
    for p in preferred:
        if _try_load_state_dict(m, p):
            loaded_path = p
            break

    if loaded_path is not None:
        m = m.to(device).eval()
        return m, True, loaded_path

    m_fallback = (
        models.efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
        .to(device)
        .eval()
    )
    return m_fallback, False, None


efficientnet_model_7, eff7_is_5class, eff7_ckpt = (
    build_efficientnet_cassava_or_fallback(0.2, "eff7", ALL_CKPTS)
)
efficientnet_model_1, eff1_is_5class, eff1_ckpt = (
    build_efficientnet_cassava_or_fallback(0.8, "eff1", ALL_CKPTS)
)
efficientnet_model_8, eff8_is_5class, eff8_ckpt = (
    build_efficientnet_cassava_or_fallback(0.8, "eff8", ALL_CKPTS)
)

print("EfficientNet model status:")
print(
    " - eff7:",
    "5-class ckpt" if eff7_is_5class else "ImageNet-1000 fallback",
    eff7_ckpt,
)
print(
    " - eff1:",
    "5-class ckpt" if eff1_is_5class else "ImageNet-1000 fallback",
    eff1_ckpt,
)
print(
    " - eff8:",
    "5-class ckpt" if eff8_is_5class else "ImageNet-1000 fallback",
    eff8_ckpt,
)




## === cell 14
mobile_model = models.mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
mobile_model = mobile_model.to(device)
mobile_model.eval()


def mobilenet_logits_to_5class_probs(logits_1000):
    probs = F.softmax(logits_1000, dim=1)
    chunks = torch.chunk(probs, chunks=5, dim=1)
    probs5 = torch.stack([c.mean(dim=1) for c in chunks], dim=1)
    probs5 = probs5 / probs5.sum(dim=1, keepdim=True)
    return probs5




## === cell 15
def eff_logits_to_5class_probs_if_needed(logits, is_5class: bool):
    if is_5class:
        return F.softmax(logits, dim=1)
    probs = F.softmax(logits, dim=1)
    chunks = torch.chunk(probs, chunks=5, dim=1)
    probs5 = torch.stack([c.mean(dim=1) for c in chunks], dim=1)
    probs5 = probs5 / probs5.sum(dim=1, keepdim=True)
    return probs5


def make_eff_tta_batch(image_np_rgb, tta_transform, n_tta, post_tensor_fn):
    aug_tensors = []
    for _ in range(n_tta):
        augmented = tta_transform(image=image_np_rgb)["image"]  # HWC RGB
        augmented = post_tensor_fn(augmented)  # CHW tensor normalized
        aug_tensors.append(augmented)
    return torch.stack(aug_tensors, dim=0)  # [T,C,H,W]


def make_mobilenet_tta_batch(image_np_rgb, tta_transform, n_tta):
    aug_tensors = []
    for _ in range(n_tta):
        aug = tta_transform(image=image_np_rgb)["image"]  # tensor CHW already
        aug_tensors.append(aug)
    return torch.stack(aug_tensors, dim=0)  # [T,C,H,W]


def _to_device_infer_batch(
    batch_cpu: torch.Tensor, device: torch.device
) -> torch.Tensor:
    if batch_cpu.device.type != "cpu":
        batch_cpu = batch_cpu.cpu()
    if device.type == "cuda":
        batch_cpu = batch_cpu.contiguous(memory_format=torch.channels_last)
        return batch_cpu.to(device, non_blocking=True)
    return batch_cpu


resnet_tta_transform = A.Compose(
    common_transforms_rgb_only
    + [
        A.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0), p=1.0),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)


def make_resnet_tta_batch(image_np_rgb, tta_transform, n_tta):
    aug_tensors = []
    for _ in range(n_tta):
        aug = tta_transform(image=image_np_rgb)["image"]  # tensor CHW
        aug_tensors.append(aug)
    return torch.stack(aug_tensors, dim=0)


ensemble_predictions = []
image_names = []

with torch.inference_mode():
    for batch in tqdm(test_loader, total=len(test_loader), desc="Predict"):
        image, img_name = batch[0]

        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        elif not isinstance(image, np.ndarray):
            image = np.array(image)

        img_name = str(img_name)

        eff_batch_cpu = make_eff_tta_batch(
            image,
            tta_transform_rgb_eff,
            n_tta=num_tta,
            post_tensor_fn=preprocess_efficientnet_np_rgb_from_albu,
        )
        mob_batch_cpu = make_mobilenet_tta_batch(
            image, tta_transform_mobilenet, n_tta=num_tta
        )
        res_batch_cpu = make_resnet_tta_batch(
            image, resnet_tta_transform, n_tta=num_tta
        )

        eff_batch = _to_device_infer_batch(eff_batch_cpu, device)
        mob_batch = _to_device_infer_batch(mob_batch_cpu, device)
        res_batch = _to_device_infer_batch(res_batch_cpu, device)

        out_eff7 = efficientnet_model_7(eff_batch)
        probs_eff7 = eff_logits_to_5class_probs_if_needed(
            out_eff7, is_5class=eff7_is_5class
        ).mean(dim=0, keepdim=True)

        out_eff1 = efficientnet_model_1(eff_batch)
        probs_eff1 = eff_logits_to_5class_probs_if_needed(
            out_eff1, is_5class=eff1_is_5class
        ).mean(dim=0, keepdim=True)

        out_eff8 = efficientnet_model_8(eff_batch)
        probs_eff8 = eff_logits_to_5class_probs_if_needed(
            out_eff8, is_5class=eff8_is_5class
        ).mean(dim=0, keepdim=True)

        out_m = mobile_model(mob_batch)
        probs_m = mobilenet_logits_to_5class_probs(out_m).mean(dim=0, keepdim=True)

        out_r = resnet_model(res_batch)
        probs_r = F.softmax(out_r, dim=1).mean(dim=0, keepdim=True)

        combined_probs = (
            probs_eff7 + probs_eff1 + probs_eff8 + probs_m + probs_r
        ) / 5.0
        final_pred = int(combined_probs.argmax(dim=1).cpu().item())

        ensemble_predictions.append(final_pred)
        image_names.append(img_name)

len(image_names), len(ensemble_predictions), len(test_df)




## === cell 16
submission_df = pd.DataFrame({"image_id": image_names, "label": ensemble_predictions})
submission_df = (
    submission_df.set_index("image_id").reindex(test_df["image_id"]).reset_index()
)

assert len(submission_df) == len(
    test_df
), "Submission must have the same length as sample_submission."
assert list(submission_df.columns) == [
    "image_id",
    "label",
], "Submission must have columns: image_id,label"
assert submission_df["label"].notna().all(), "Found NaN labels after reindexing."

submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)
print(f"Submission file saved as '{submission_path}' with shape {submission_df.shape}")
submission_df.head()
