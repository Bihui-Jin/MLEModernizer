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
import numpy as np
import pandas as pd
import os

shown = 0
for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        if shown < 10:
            print(os.path.join(dirname, filename))
            shown += 1
        else:
            break
    if shown >= 10:
        break



## === cell 1
import json
import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import Dataset, DataLoader
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2



## === cell 2
num_tta = 5



## === cell 3
test_image_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images"
train_image_dir = "/kaggle/input/cassava-leaf-disease-classification/train_images"



## === cell 4
test_df = pd.read_csv(
    "../input/cassava-leaf-disease-classification/sample_submission.csv"
)
test_df.head()



## === cell 5
train_df = pd.read_csv("../input/cassava-leaf-disease-classification/train.csv")
train_df.head()



## === cell 6
try:
    cv2.setNumThreads(max(0, (os.cpu_count() or 4)))
except Exception:
    pass

clahe_only = A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0)

train_eff_post = A.Compose(
    [
        A.Resize(384, 384),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.ShiftScaleRotate(
            shift_limit=0.02,
            scale_limit=0.05,
            rotate_limit=10,
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.5,
        ),
        A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02, p=0.4),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)

train_res_post = A.Compose(
    [
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.ShiftScaleRotate(
            shift_limit=0.02,
            scale_limit=0.05,
            rotate_limit=10,
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.5,
        ),
        A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02, p=0.4),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)

eff_post = A.Compose(
    [
        A.Resize(384, 384),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)

res_post = A.Compose(
    [
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)

efficientnet_transforms = eff_post
resnet_transforms = res_post

tta_transforms = [
    A.Compose([A.HorizontalFlip(p=1.0)]),
    A.Compose([A.VerticalFlip(p=1.0)]),
    A.Compose([A.Rotate(limit=10, border_mode=cv2.BORDER_REFLECT_101, p=1.0)]),
    A.Compose(
        [
            A.ShiftScaleRotate(
                shift_limit=0.02,
                scale_limit=0.02,
                rotate_limit=5,
                border_mode=cv2.BORDER_REFLECT_101,
                p=1.0,
            )
        ]
    ),
]




## === cell 7
class CassavaTestDataset(Dataset):
    def __init__(
        self, dataframe, image_dir, eff_transform, res_transform, tta_list=None
    ):
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.eff_transform = eff_transform
        self.res_transform = res_transform
        self.tta_list = tta_list or []

        self.image_ids = self.dataframe.iloc[:, 0].astype(str).to_numpy()
        self.image_paths = [os.path.join(self.image_dir, n) for n in self.image_ids]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_name = self.image_ids[idx]
        img_path = self.image_paths[idx]

        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image at path: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        base = clahe_only(image=image)["image"]

        eff_img = self.eff_transform(image=base)["image"]  # torch.FloatTensor (C,H,W)
        res_img = self.res_transform(image=base)["image"]  # torch.FloatTensor (C,H,W)

        T = len(self.tta_list)
        if T:
            eff_ttas = torch.empty((T,) + tuple(eff_img.shape), dtype=eff_img.dtype)
            res_ttas = torch.empty((T,) + tuple(res_img.shape), dtype=res_img.dtype)
            for i, t in enumerate(self.tta_list):
                aug = t(image=base)["image"]
                eff_ttas[i] = self.eff_transform(image=aug)["image"]
                res_ttas[i] = self.res_transform(image=aug)["image"]
        else:
            eff_ttas = torch.empty((0,) + tuple(eff_img.shape), dtype=eff_img.dtype)
            res_ttas = torch.empty((0,) + tuple(res_img.shape), dtype=res_img.dtype)

        return eff_img, res_img, eff_ttas, res_ttas, img_name




## === cell 8
class CassavaTrainDataset(Dataset):
    def __init__(self, dataframe, image_dir, eff_transform, res_transform):
        self.df = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.eff_transform = eff_transform
        self.res_transform = res_transform

        self.image_ids = self.df["image_id"].astype(str).to_numpy()
        self.labels = self.df["label"].astype(np.int64).to_numpy()
        self.image_paths = [os.path.join(self.image_dir, n) for n in self.image_ids]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        y = int(self.labels[idx])
        img_path = self.image_paths[idx]

        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image at path: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        base = clahe_only(image=image)["image"]
        eff_img = self.eff_transform(image=base)["image"]
        res_img = self.res_transform(image=base)["image"]
        return eff_img, res_img, torch.tensor(y, dtype=torch.long)




## === cell 9
active_tta = (
    tta_transforms[: max(0, min(num_tta - 1, len(tta_transforms)))]
    if num_tta and num_tta > 1
    else []
)

test_dataset = CassavaTestDataset(
    test_df,
    test_image_dir,
    efficientnet_transforms,
    resnet_transforms,
    tta_list=active_tta,
)



## === cell 10
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def seed_worker(worker_id):
    worker_seed = (seed + worker_id) % 2**32
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


g = torch.Generator()
g.manual_seed(seed)

_NUM_WORKERS = max(2, min(8, (os.cpu_count() or 4) // 2))


def cassava_test_collate(batch):
    eff_imgs, res_imgs, eff_ttas, res_ttas, names = zip(*batch)
    eff_batch = torch.stack(eff_imgs, dim=0)
    res_batch = torch.stack(res_imgs, dim=0)
    eff_ttas_batch = torch.stack(eff_ttas, dim=0)  # [B,T,C,H,W]
    res_ttas_batch = torch.stack(res_ttas, dim=0)  # [B,T,C,H,W]
    return eff_batch, res_batch, eff_ttas_batch, res_ttas_batch, list(names)


test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=_NUM_WORKERS,
    pin_memory=True,
    persistent_workers=(_NUM_WORKERS > 0),
    prefetch_factor=4 if _NUM_WORKERS > 0 else None,
    worker_init_fn=seed_worker,
    generator=g,
    collate_fn=cassava_test_collate,
)



## === cell 11
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device




## === cell 12
def resolve_checkpoint_path(candidates):
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None




## === cell 13
def try_load_state_dict(model, ckpt_path):
    if ckpt_path is None:
        return False
    try:
        state = torch.load(ckpt_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]

        cleaned = {}
        for k, v in state.items():
            nk = k
            if nk.startswith("module."):
                nk = nk[len("module.") :]
            if nk.startswith("model."):
                nk = nk[len("model.") :]
            cleaned[nk] = v

        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        print(f"Loaded checkpoint: {ckpt_path}")
        if missing:
            print(f"  Missing keys (truncated): {missing[:10]}")
        if unexpected:
            print(f"  Unexpected keys (truncated): {unexpected[:10]}")
        return True
    except Exception as e:
        print(f"Failed to load checkpoint {ckpt_path}: {e}")
        return False




## === cell 14
CKPT_ROOTS = [
    "/kaggle/input/cassava-leaf-disease-classification",
    "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
]

_COMMON_CKPT_NAMES = [
    "resnet50.pth",
    "resnet_50.pth",
    "resnet50.pt",
    "resnet50.bin",
    "efficientnet_v2_s.pth",
    "efficientnetv2s.pth",
    "efficientnet_v2_s.pt",
    "efficientnet_v2_s.bin",
    "best.pth",
    "final.pth",
    "model.pth",
]

_CKPT_FILE_CACHE = None


def _build_ckpt_file_cache():
    files = []
    for root in CKPT_ROOTS:
        if not os.path.isdir(root):
            continue
        candidates = [root]
        try:
            for d in os.listdir(root):
                p = os.path.join(root, d)
                if os.path.isdir(p):
                    candidates.append(p)
        except Exception:
            pass

        for d in candidates:
            try:
                for fn in os.listdir(d):
                    lfn = fn.lower()
                    if (
                        lfn.endswith(".pth")
                        or lfn.endswith(".pt")
                        or lfn.endswith(".bin")
                    ):
                        files.append((d, fn, lfn))
            except Exception:
                continue
    return files


def find_ckpt_by_keywords(keywords):
    global _CKPT_FILE_CACHE
    if _CKPT_FILE_CACHE is None:
        _CKPT_FILE_CACHE = _build_ckpt_file_cache()
    keywords = [k.lower() for k in keywords]
    candidates = []
    for dirpath, fn, lfn in _CKPT_FILE_CACHE:
        if all(k in lfn for k in keywords):
            candidates.append(os.path.join(dirpath, fn))
    for root in CKPT_ROOTS:
        for nm in _COMMON_CKPT_NAMES:
            p = os.path.join(root, nm)
            if os.path.exists(p):
                lfn = os.path.basename(p).lower()
                if all(k in lfn for k in keywords):
                    candidates.append(p)
    seen = set()
    out = []
    for p in candidates:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return sorted(out)


def rank_ckpts(paths):
    def score(p):
        l = os.path.basename(p).lower()
        s = 0
        for kw, w in [
            ("best", 50),
            ("final", 20),
            ("fold", 10),
            ("epoch", 5),
            ("acc", 10),
            ("metric", 10),
        ]:
            if kw in l:
                s += w
        try:
            s += min(int(os.path.getsize(p) / (1024 * 1024)), 200) / 10.0
        except Exception:
            pass
        return s

    return sorted(paths, key=score, reverse=True)


def try_load_any(model, ckpt_paths, max_tries=5):
    ckpt_paths = rank_ckpts(ckpt_paths)
    tried = 0
    for p in ckpt_paths:
        if tried >= max_tries:
            break
        tried += 1
        if try_load_state_dict(model, p):
            return p
    return None




## === cell 15
loaded_any_cassava_ckpt = False

resnet_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
num_ftrs = resnet_model.fc.in_features
resnet_model.fc = nn.Linear(num_ftrs, 5)

resnet_ckpts = find_ckpt_by_keywords(["resnet", "50"])
_loaded_resnet = (
    try_load_any(resnet_model, resnet_ckpts, max_tries=5) if len(resnet_ckpts) else None
)
if _loaded_resnet is None:
    print(
        "No usable ResNet50 cassava checkpoint found; will fine-tune from ImageNet initialization."
    )
else:
    loaded_any_cassava_ckpt = True

resnet_model = resnet_model.to(device)



## === cell 16
efficientnet_model_1 = models.efficientnet_v2_s(
    weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
)
num_features_efficientnet = efficientnet_model_1.classifier[1].in_features
efficientnet_model_1.classifier[1] = nn.Linear(num_features_efficientnet, 5)

eff_ckpts = find_ckpt_by_keywords(["efficientnet", "v2", "s"])
_loaded_eff1 = (
    try_load_any(efficientnet_model_1, eff_ckpts, max_tries=5)
    if len(eff_ckpts)
    else None
)
if _loaded_eff1 is None:
    print(
        "No usable EfficientNetV2-S cassava checkpoint found; will fine-tune from ImageNet initialization."
    )
else:
    loaded_any_cassava_ckpt = True

efficientnet_model_1 = efficientnet_model_1.to(device)



## === cell 17
efficientnet_model_7 = models.efficientnet_v2_s(
    weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
)
num_features_efficientnet = efficientnet_model_7.classifier[1].in_features
efficientnet_model_7.classifier = nn.Sequential(
    nn.Dropout(p=0.8), nn.Linear(num_features_efficientnet, 5)
)

eff_ckpts_ranked = rank_ckpts(eff_ckpts) if len(eff_ckpts) else []
cand7 = eff_ckpts_ranked[1:] if len(eff_ckpts_ranked) > 1 else eff_ckpts_ranked
_loaded_eff7 = (
    try_load_any(efficientnet_model_7, cand7, max_tries=5) if len(cand7) else None
)
if _loaded_eff7 is None and len(eff_ckpts_ranked) > 0:
    ok = try_load_state_dict(efficientnet_model_7, eff_ckpts_ranked[0])
    if ok:
        _loaded_eff7 = eff_ckpts_ranked[0]
if _loaded_eff7 is not None:
    loaded_any_cassava_ckpt = True

efficientnet_model_7 = efficientnet_model_7.to(device)



## === cell 18
efficientnet_model_8 = models.efficientnet_v2_s(
    weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
)
num_features_efficientnet = efficientnet_model_8.classifier[1].in_features
efficientnet_model_8.classifier = nn.Sequential(
    nn.Dropout(p=0.8), nn.Linear(num_features_efficientnet, 5)
)

cand8 = eff_ckpts_ranked[2:] if len(eff_ckpts_ranked) > 2 else eff_ckpts_ranked
_loaded_eff8 = (
    try_load_any(efficientnet_model_8, cand8, max_tries=5) if len(cand8) else None
)
if _loaded_eff8 is None and len(eff_ckpts_ranked) > 0:
    ok = try_load_state_dict(efficientnet_model_8, eff_ckpts_ranked[0])
    if ok:
        _loaded_eff8 = eff_ckpts_ranked[0]
if _loaded_eff8 is not None:
    loaded_any_cassava_ckpt = True

efficientnet_model_8 = efficientnet_model_8.to(device)



## === cell 19
label_map_path = (
    "/kaggle/input/cassava-leaf-disease-classification/label_num_to_disease_map.json"
)
with open(label_map_path, "r") as f:
    cassava_label_to_name = json.load(f)  # keys are "0".."4" as strings

imagenet_categories = models.ResNet50_Weights.IMAGENET1K_V2.meta.get("categories", None)
if imagenet_categories is None:
    imagenet_categories = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1.meta[
        "categories"
    ]

cassava_keywords = {
    0: ["blight", "bacterial", "leaf spot", "leaf blotch"],
    1: ["streak", "brown", "necrosis"],
    2: ["mottle", "mottled", "green"],
    3: ["mosaic"],
    4: ["leaf", "plant", "tree", "herb", "vegetable", "flower"],
}

cat_lower = [c.lower() for c in imagenet_categories]
scores = np.zeros((len(cat_lower), 5), dtype=np.float32)
for cass_label, kws in cassava_keywords.items():
    for kw in kws:
        hit = np.array([1.0 if kw in c else 0.0 for c in cat_lower], dtype=np.float32)
        scores[:, cass_label] += hit

imagenet_to_cassava = scores.argmax(axis=1).astype(np.int64)
no_match = scores.max(axis=1) == 0
imagenet_to_cassava[no_match] = 4

fallback_imagenet_model = None
use_fallback_imagenet_mapping = not loaded_any_cassava_ckpt
if use_fallback_imagenet_mapping:
    print(
        "No cassava-trained checkpoints loaded -> fallback mapping exists, but we will fine-tune instead for higher accuracy."
    )
    use_fallback_imagenet_mapping = False  # prefer supervised fine-tuning



## === cell 20
from sklearn.model_selection import GroupShuffleSplit

_groups = train_df["image_id"].astype(str).str.split(".").str[0].str[:6].values
gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=seed)
train_idx, val_idx = next(gss.split(train_df, train_df["label"].values, groups=_groups))

tr_df = train_df.iloc[train_idx].reset_index(drop=True)
va_df = train_df.iloc[val_idx].reset_index(drop=True)

train_ds = CassavaTrainDataset(tr_df, train_image_dir, train_eff_post, train_res_post)
val_ds = CassavaTrainDataset(
    va_df, train_image_dir, efficientnet_transforms, resnet_transforms
)

train_loader = DataLoader(
    train_ds,
    batch_size=24,
    shuffle=True,
    num_workers=_NUM_WORKERS,
    pin_memory=True,
    persistent_workers=(_NUM_WORKERS > 0),
    prefetch_factor=4 if _NUM_WORKERS > 0 else None,
    worker_init_fn=seed_worker,
    generator=g,
    drop_last=True,
)
val_loader = DataLoader(
    val_ds,
    batch_size=48,
    shuffle=False,
    num_workers=_NUM_WORKERS,
    pin_memory=True,
    persistent_workers=(_NUM_WORKERS > 0),
    prefetch_factor=4 if _NUM_WORKERS > 0 else None,
    worker_init_fn=seed_worker,
    generator=g,
)



## === cell 21
criterion = nn.CrossEntropyLoss(label_smoothing=0.05)


def set_trainable_for_speed(model, train_last_n_children=2):
    children = list(model.children())
    for p in model.parameters():
        p.requires_grad = False
    if hasattr(model, "fc"):
        for p in model.fc.parameters():
            p.requires_grad = True
    if hasattr(model, "classifier"):
        for p in model.classifier.parameters():
            p.requires_grad = True
    for ch in children[-train_last_n_children:]:
        for p in ch.parameters():
            p.requires_grad = True


set_trainable_for_speed(resnet_model, train_last_n_children=2)
set_trainable_for_speed(efficientnet_model_1, train_last_n_children=2)
set_trainable_for_speed(efficientnet_model_7, train_last_n_children=2)
set_trainable_for_speed(efficientnet_model_8, train_last_n_children=2)

params = []
for m in [
    resnet_model,
    efficientnet_model_1,
    efficientnet_model_7,
    efficientnet_model_8,
]:
    params += [p for p in m.parameters() if p.requires_grad]

optimizer = torch.optim.AdamW(params, lr=2e-4, weight_decay=1e-4)

epochs = 3
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)


def forward_ensemble_logits(eff_batch, res_batch, w_eff=0.7, w_res=0.3):
    out_eff1 = efficientnet_model_1(eff_batch)
    out_eff7 = efficientnet_model_7(eff_batch)
    out_eff8 = efficientnet_model_8(eff_batch)
    out_res = resnet_model(res_batch)
    out_eff = (out_eff1 + out_eff7 + out_eff8) / 3.0
    out = w_eff * out_eff + w_res * out_res
    return out


class _CudaPrefetcher:
    def __init__(self, loader, device):
        self.loader = loader
        self.device = device
        self.stream = torch.cuda.Stream()
        self.iter = None
        self.next_batch = None

    def __iter__(self):
        self.iter = iter(self.loader)
        self._preload()
        return self

    def __next__(self):
        if self.next_batch is None:
            raise StopIteration
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.next_batch
        self._preload()
        return batch

    def _preload(self):
        try:
            batch = next(self.iter)
        except StopIteration:
            self.next_batch = None
            return
        with torch.cuda.stream(self.stream):
            eff_x, res_x, y = batch
            eff_x = eff_x.to(self.device, non_blocking=True)
            res_x = res_x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            self.next_batch = (eff_x, res_x, y)


def _maybe_prefetch(loader):
    if device.type == "cuda":
        return _CudaPrefetcher(loader, device)
    return loader


def eval_acc(w_eff=0.7, w_res=0.3):
    for m in [
        resnet_model,
        efficientnet_model_1,
        efficientnet_model_7,
        efficientnet_model_8,
    ]:
        m.eval()
    correct = 0
    total = 0
    with torch.inference_mode():
        for eff_x, res_x, y in _maybe_prefetch(val_loader):
            if device.type != "cuda":
                eff_x = eff_x.to(device, non_blocking=True)
                res_x = res_x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
            logits = forward_ensemble_logits(eff_x, res_x, w_eff=w_eff, w_res=w_res)
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total += y.numel()
    return correct / max(1, total)


for epoch in range(epochs):
    for m in [
        resnet_model,
        efficientnet_model_1,
        efficientnet_model_7,
        efficientnet_model_8,
    ]:
        m.train()
    running_loss = 0.0
    n = 0
    for eff_x, res_x, y in _maybe_prefetch(train_loader):
        if device.type != "cuda":
            eff_x = eff_x.to(device, non_blocking=True)
            res_x = res_x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = forward_ensemble_logits(eff_x, res_x, w_eff=0.7, w_res=0.3)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * y.size(0)
        n += y.size(0)

    scheduler.step()
    va = eval_acc(w_eff=0.7, w_res=0.3)
    print(
        f"epoch {epoch+1}/{epochs} - train_loss={running_loss/max(1,n):.4f} - val_acc={va:.4f} - lr={optimizer.param_groups[0]['lr']:.2e}"
    )

for m in [
    resnet_model,
    efficientnet_model_1,
    efficientnet_model_7,
    efficientnet_model_8,
]:
    m.eval()

best_w_eff = 0.7
best_va = -1.0
for w_eff in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]:
    w_res = 1.0 - w_eff
    va = eval_acc(w_eff=w_eff, w_res=w_res)
    if va > best_va:
        best_va = va
        best_w_eff = w_eff
weight_efficientnet = float(best_w_eff)
weight_resnet = float(1.0 - best_w_eff)
print(
    f"Chosen ensemble weights from val: w_eff={weight_efficientnet:.2f}, w_res={weight_resnet:.2f} (val_acc={best_va:.4f})"
)



## === cell 22
ensemble_predictions = []
image_names = []


def forward_ensemble(eff_batch, res_batch):
    out_eff1 = efficientnet_model_1(eff_batch)
    out_eff7 = efficientnet_model_7(eff_batch)
    out_eff8 = efficientnet_model_8(eff_batch)
    out_res = resnet_model(res_batch)
    out_eff = (out_eff1 + out_eff7 + out_eff8) / 3.0
    out = weight_efficientnet * out_eff + weight_resnet * out_res
    return out


def forward_fallback(res_batch):
    logits_1000 = fallback_imagenet_model(res_batch)  # [B,1000]
    pred_1000 = logits_1000.argmax(dim=1).detach().cpu().numpy()
    pred_5 = imagenet_to_cassava[pred_1000]
    return pred_5.tolist()


class _CudaPrefetcherTest:
    def __init__(self, loader, device):
        self.loader = loader
        self.device = device
        self.stream = torch.cuda.Stream()
        self.iter = None
        self.next_batch = None

    def __iter__(self):
        self.iter = iter(self.loader)
        self._preload()
        return self

    def __next__(self):
        if self.next_batch is None:
            raise StopIteration
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.next_batch
        self._preload()
        return batch

    def _preload(self):
        try:
            batch = next(self.iter)
        except StopIteration:
            self.next_batch = None
            return
        with torch.cuda.stream(self.stream):
            eff_batch, res_batch, eff_ttas, res_ttas, img_names = batch
            eff_batch = eff_batch.to(self.device, non_blocking=True)
            res_batch = res_batch.to(self.device, non_blocking=True)
            eff_ttas = eff_ttas.to(self.device, non_blocking=True)
            res_ttas = res_ttas.to(self.device, non_blocking=True)
            self.next_batch = (eff_batch, res_batch, eff_ttas, res_ttas, img_names)


def _maybe_prefetch_test(loader):
    if device.type == "cuda":
        return _CudaPrefetcherTest(loader, device)
    return loader


with torch.inference_mode():
    for eff_batch, res_batch, eff_ttas, res_ttas, img_names in _maybe_prefetch_test(
        test_loader
    ):
        if device.type != "cuda":
            eff_batch = eff_batch.to(device, non_blocking=True)
            res_batch = res_batch.to(device, non_blocking=True)
            eff_ttas = eff_ttas.to(device, non_blocking=True)
            res_ttas = res_ttas.to(device, non_blocking=True)

        if use_fallback_imagenet_mapping:
            preds = forward_fallback(res_batch)
            ensemble_predictions.extend(preds)
            image_names.extend(list(img_names))
            continue

        logits = forward_ensemble(eff_batch, res_batch)  # [B,5]

        T = eff_ttas.shape[1]
        if T > 0:
            B = eff_ttas.shape[0]
            eff_flat = eff_ttas.reshape(B * T, *eff_ttas.shape[2:])
            res_flat = res_ttas.reshape(B * T, *res_ttas.shape[2:])
            logits_tta = (
                forward_ensemble(eff_flat, res_flat).reshape(B, T, -1).sum(dim=1)
            )
            logits = (logits + logits_tta) / (1.0 + float(T))

        preds = logits.argmax(dim=1).detach().cpu().numpy().tolist()
        ensemble_predictions.extend(preds)
        image_names.extend(list(img_names))

if len(image_names) != len(test_df) or len(ensemble_predictions) != len(test_df):
    print(
        f"WARNING: prediction length mismatch (names={len(image_names)}, preds={len(ensemble_predictions)}, expected={len(test_df)}). "
        "Rerunning no-TTA inference to ensure a complete submission."
    )
    simple_ds = CassavaTestDataset(
        test_df,
        test_image_dir,
        efficientnet_transforms,
        resnet_transforms,
        tta_list=[],
    )
    simple_loader = DataLoader(
        simple_ds,
        batch_size=32,
        shuffle=False,
        num_workers=_NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(_NUM_WORKERS > 0),
        prefetch_factor=4 if _NUM_WORKERS > 0 else None,
        worker_init_fn=seed_worker,
        generator=g,
        collate_fn=cassava_test_collate,
    )
    ensemble_predictions = []
    image_names = []
    with torch.inference_mode():
        for eff_batch, res_batch, eff_ttas, res_ttas, img_names in _maybe_prefetch_test(
            simple_loader
        ):
            if device.type != "cuda":
                eff_batch = eff_batch.to(device, non_blocking=True)
                res_batch = res_batch.to(device, non_blocking=True)
            logits = forward_ensemble(eff_batch, res_batch)
            preds = logits.argmax(dim=1).detach().cpu().numpy().tolist()
            ensemble_predictions.extend(preds)
            image_names.extend(list(img_names))



## === cell 23
submission_df = pd.DataFrame({"image_id": image_names, "label": ensemble_predictions})

submission_df = test_df[["image_id"]].merge(submission_df, on="image_id", how="left")

if submission_df["label"].isna().any():
    submission_df["label"] = submission_df["label"].fillna(0)

submission_df["label"] = submission_df["label"].astype(int)

assert len(submission_df) == len(
    test_df
), f"Row count mismatch: {len(submission_df)} vs {len(test_df)}"
assert list(submission_df.columns) == [
    "image_id",
    "label",
], f"Bad columns: {submission_df.columns.tolist()}"

submission_df.to_csv("submission.csv", index=False)
print("Submission file saved as 'submission.csv'")
print(submission_df.head())
