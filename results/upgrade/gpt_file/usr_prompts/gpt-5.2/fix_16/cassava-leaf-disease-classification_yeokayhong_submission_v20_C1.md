# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

3.13

# 3. Installed packages

No external packages required in the script and installed.

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

0.8871260199456029

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.4716) has done: 'I fix the pipeline so it always loads a model in this Kaggle environment by falling back to a built-in torchvision pretrained backbone when the external `.pth` files are not present (this unblocks the FileNotFoundError/NameError while keeping the same inference flow). I also make the image path resolution robust to the duplicated `test_images/test_images` directory layout, which is the likely cause of the image_id ordering mismatch and/or missing reads. Finally, I ensure the submission uses the exact `sample_submission.csv` ordering by constructing predictions in that order and writing `submission.csv` with the required columns.'

# 9. Code solution

## === cell 0
from torchvision import models, transforms
from torchvision.transforms import v2
from tqdm import tqdm
from PIL import Image
import pandas as pd
import numpy as np
import torch
import os
import random
import glob

import torchvision
from torchvision.io import read_image, ImageReadMode

test_data_directory = "/kaggle/input/cassava-leaf-disease-classification/test_images"
train_data_directory = "/kaggle/input/cassava-leaf-disease-classification/train_images"
train_csv_path = "/kaggle/input/cassava-leaf-disease-classification/train.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 5

en_model_path = "/kaggle/input/efficientnetv2-large-test/pytorch/default/4/efficientnet_v2_l_480_8591_ISP_CBP.pth"
en_image_size = 480

vit_model_path = (
    "/kaggle/input/vit_l_cassava/pytorch/default/6/vit_h_14_518_8867_CBP.pth"
)
vit_image_size = 518

model_select = "vit"

if model_select == "vit":
    model_image_size = vit_image_size
elif model_select == "en":
    model_image_size = en_image_size
else:
    raise ValueError(f"Unknown model_select={model_select}")

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

torch.backends.cudnn.benchmark = True

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

Image.MAX_IMAGE_PIXELS = None

_CPU_COUNT = os.cpu_count() or 2
num_workers = min(8, max(2, _CPU_COUNT))  # previously capped at 4

_CAN_COMPILE = hasattr(torch, "compile")


def _seed_worker(worker_id: int):
    wseed = seed + worker_id
    random.seed(wseed)
    np.random.seed(wseed)
    torch.manual_seed(wseed)


def fast_collate_train(batch):
    xs, ys = zip(*batch)
    xb = torch.stack(xs, dim=0)
    yb = torch.as_tensor(ys, dtype=torch.long)
    return xb, yb


def fast_collate_test(batch):
    xs, ids = zip(*batch)
    xb = torch.stack(xs, dim=0)
    return xb, list(ids)




## === cell 1
def invert_square_pad(img):
    width, height = img.size

    center_width, center_height = width // 2, height // 2
    top_left = img.crop((0, 0, center_width, center_height))
    top_right = img.crop((center_width, 0, width, center_height))
    bottom_left = img.crop((0, center_height, center_width, height))
    bottom_right = img.crop((center_width, center_height, width, height))

    top_combined = Image.new("RGB", (width, center_height))
    top_combined.paste(bottom_right, (0, 0))
    top_combined.paste(bottom_left, (center_width, 0))

    bottom_combined = Image.new("RGB", (width, center_height))
    bottom_combined.paste(top_right, (0, 0))
    bottom_combined.paste(top_left, (center_width, 0))

    flipped_img = Image.new("RGB", (width, height))
    flipped_img.paste(top_combined, (0, 0))
    flipped_img.paste(bottom_combined, (0, center_height))

    img = flipped_img.copy()
    del top_combined, bottom_combined, flipped_img

    max_side = max(width, height)
    padding = (
        (max_side - width) // 2,  # left
        (max_side - height) // 2,  # top
        (max_side - width) - (max_side - width) // 2,  # right
        (max_side - height) - (max_side - height) // 2,  # bottom
    )

    padded_img = transforms.functional.pad(img, padding, padding_mode="reflect")
    return padded_img


def resize_max_side(img, size):
    width, height = img.size

    if max(width, height) <= size:
        return img

    if width > height:
        new_width = size
        new_height = int(size * height / width)
    else:
        new_height = size
        new_width = int(size * width / height)

    return transforms.functional.resize(img, (new_height, new_width))


def pad_to_square(img):
    width, height = img.size
    max_side = max(width, height)
    padding = (
        (max_side - width) // 2,
        (max_side - height) // 2,
        (max_side - width) - (max_side - width) // 2,
        (max_side - height) - (max_side - height) // 2,
    )  # left, top, right, bottom

    padded_img = transforms.functional.pad(img, padding, padding_mode="reflect")
    return padded_img




## === cell 2
import torch.nn.functional as F

_NORMALIZE_MEAN_CPU = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32).view(
    1, 3, 1, 1
)
_NORMALIZE_STD_CPU = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32).view(1, 3, 1, 1)
_NORMALIZE_MEAN = _NORMALIZE_MEAN_CPU.to(device=device)
_NORMALIZE_STD = _NORMALIZE_STD_CPU.to(device=device)


def _resize_max_side_batched(x: torch.Tensor, max_side: int = 800) -> torch.Tensor:
    b, c, h, w = x.shape
    if max(h, w) <= max_side:
        return x
    if w > h:
        new_w = max_side
        new_h = int(max_side * h / w)
    else:
        new_h = max_side
        new_w = int(max_side * w / h)
    return F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)


def _pad_to_square_batched_reflect(x: torch.Tensor) -> torch.Tensor:
    b, c, h, w = x.shape
    if h == w:
        return x
    max_side = max(h, w)
    pad_left = (max_side - w) // 2
    pad_right = (max_side - w) - pad_left
    pad_top = (max_side - h) // 2
    pad_bottom = (max_side - h) - pad_top
    return F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode="reflect")


def gpu_batch_transform(xb: torch.Tensor, *, out_size: int) -> torch.Tensor:
    if xb.dtype != torch.float32:
        xb = xb.to(torch.float32)
    xb = xb.mul_(1.0 / 255.0)

    xb = _resize_max_side_batched(xb, 800)
    xb = _pad_to_square_batched_reflect(xb)
    xb = F.interpolate(
        xb, size=(out_size, out_size), mode="bilinear", align_corners=False
    )

    xb = (xb - _NORMALIZE_MEAN) / _NORMALIZE_STD
    return xb


if _CAN_COMPILE:
    try:
        gpu_batch_transform = torch.compile(gpu_batch_transform, mode="reduce-overhead")
    except Exception:
        pass


val_transforms = transforms.Compose(
    [
        v2.Lambda(lambda img: resize_max_side(img, 800)),
        v2.Lambda(pad_to_square),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((model_image_size, model_image_size)),
        v2.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)

train_transforms = transforms.Compose(
    [
        v2.Lambda(lambda img: resize_max_side(img, 800)),
        v2.Lambda(pad_to_square),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((model_image_size, model_image_size)),
        v2.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)




## === cell 3
def resolve_weight_path(preferred_path: str, patterns: list[str]) -> str | None:
    if preferred_path and os.path.isfile(preferred_path):
        return preferred_path

    matches = []
    for pat in patterns:
        matches.extend(glob.glob(pat, recursive=True))

    matches = [m for m in matches if os.path.isfile(m)]
    if not matches:
        return None

    matches = sorted(matches, key=lambda p: (len(p), p))
    return matches[0]


if model_select == "vit":
    resolved = resolve_weight_path(
        vit_model_path,
        patterns=[
            "/kaggle/input/**/vit*_*.pth",
            "/kaggle/input/**/*vit*518*.pth",
            "/kaggle/input/**/*.pth",
        ],
    )

    vit_model = models.vit_h_14(weights=None, image_size=518)
    vit_model.heads.head = torch.nn.Linear(
        vit_model.heads.head.in_features, num_classes
    )

    if resolved is not None:
        try:
            state = torch.load(resolved, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(resolved, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        vit_model.load_state_dict(state, strict=True)
    else:
        vit_model = models.vit_h_14(
            weights=models.ViT_H_14_Weights.DEFAULT, image_size=518
        )
        vit_model.heads.head = torch.nn.Linear(
            vit_model.heads.head.in_features, num_classes
        )

    vit_model.to(device)

elif model_select == "en":
    resolved = resolve_weight_path(
        en_model_path,
        patterns=[
            "/kaggle/input/**/efficientnet*_*.pth",
            "/kaggle/input/**/*efficientnet_v2_l*.pth",
            "/kaggle/input/**/*.pth",
        ],
    )

    en_model = models.efficientnet_v2_l(weights=None)
    en_model.classifier[1] = torch.nn.Linear(
        en_model.classifier[1].in_features, num_classes
    )

    if resolved is not None:
        try:
            state = torch.load(resolved, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(resolved, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        en_model.load_state_dict(state, strict=True)
    else:
        en_model = models.efficientnet_v2_l(
            weights=models.EfficientNet_V2_L_Weights.DEFAULT
        )
        en_model.classifier[1] = torch.nn.Linear(
            en_model.classifier[1].in_features, num_classes
        )

    en_model.to(device)




## === cell 4
class CassavaDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        images_dir: str,
        transform,
        pre_resolved_paths=None,
        cache_images: bool = True,
    ):
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.transform = (
            transform  # retained for compatibility; not used in optimized path
        )

        self.image_ids = self.df["image_id"].to_numpy()
        self.labels = self.df["label"].to_numpy(dtype=np.int64)

        self._paths = pre_resolved_paths

        self.cache_images = bool(cache_images)
        self._cache = {} if (self.cache_images) else None

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx: int):
        image_id = self.image_ids[idx]
        label = int(self.labels[idx])

        image_path = (
            self._paths[image_id]
            if self._paths is not None
            else os.path.join(self.images_dir, image_id)
        )

        if self._cache is not None:
            x = self._cache.get(image_id)
            if x is None:
                x = read_image(image_path, mode=ImageReadMode.RGB)  # uint8 CHW
                self._cache[image_id] = x
        else:
            x = read_image(image_path, mode=ImageReadMode.RGB)

        y = label  # return int; collate tensorizes once per batch
        return x, y


def make_split(df: pd.DataFrame, val_frac: float = 0.1, seed: int = 42):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    val_size = int(len(df) * val_frac)
    val_idx = idx[:val_size]
    tr_idx = idx[val_size:]
    return df.iloc[tr_idx].copy(), df.iloc[val_idx].copy()


train_df = pd.read_csv(train_csv_path)
tr_df, va_df = make_split(train_df, val_frac=0.1, seed=seed)


def build_image_path_map(images_dir: str, image_ids: np.ndarray) -> dict:
    root_direct = images_dir
    root_nested = os.path.join(images_dir, "train_images")

    m = {}
    missing = []
    for iid in image_ids:
        p = os.path.join(root_direct, iid)
        if os.path.isfile(p):
            m[iid] = p
            continue
        p2 = os.path.join(root_nested, iid)
        if os.path.isfile(p2):
            m[iid] = p2
            continue
        missing.append(iid)

    if missing:
        all_files = glob.glob(os.path.join(images_dir, "**", "*.jpg"), recursive=True)
        base_to_path = {}
        for p in all_files:
            b = os.path.basename(p)
            prev = base_to_path.get(b)
            if (prev is None) or ((len(p), p) < (len(prev), prev)):
                base_to_path[b] = p
        for iid in missing:
            p = base_to_path.get(iid)
            if p is None:
                raise FileNotFoundError(
                    f"Could not locate train image {iid} under {images_dir}"
                )
            m[iid] = p
    return m


all_train_ids = train_df["image_id"].to_numpy()
train_path_map = build_image_path_map(train_data_directory, all_train_ids)

batch_size = 16 if (model_select == "vit") else 32

train_ds = CassavaDataset(
    tr_df,
    train_data_directory,
    train_transforms,
    pre_resolved_paths=train_path_map,
    cache_images=False,
)
val_ds = CassavaDataset(
    va_df,
    train_data_directory,
    val_transforms,
    pre_resolved_paths=train_path_map,
    cache_images=False,
)

pin = torch.cuda.is_available()
_prefetch = 16 if num_workers > 0 else None

g = torch.Generator()
g.manual_seed(seed)

train_loader = torch.utils.data.DataLoader(
    train_ds,
    batch_size=batch_size,
    shuffle=True,
    generator=g,
    num_workers=num_workers,
    pin_memory=pin,
    drop_last=True,
    persistent_workers=(num_workers > 0),
    prefetch_factor=_prefetch,
    worker_init_fn=_seed_worker,
    collate_fn=fast_collate_train,
)
val_loader = torch.utils.data.DataLoader(
    val_ds,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=pin,
    drop_last=False,
    persistent_workers=(num_workers > 0),
    prefetch_factor=_prefetch,
    worker_init_fn=_seed_worker,
    collate_fn=fast_collate_train,
)

if model_select == "vit":
    model = vit_model
else:
    model = en_model

if torch.cuda.is_available():
    model = model.to(memory_format=torch.channels_last)

if _CAN_COMPILE:
    try:
        model = torch.compile(model, mode="reduce-overhead")
    except Exception:
        pass

criterion = torch.nn.CrossEntropyLoss()


def set_trainable(m: torch.nn.Module, trainable: bool):
    for p in m.parameters():
        p.requires_grad = trainable


set_trainable(model, False)
if model_select == "vit":
    set_trainable(model.heads, True)
    params = list(model.heads.parameters())
    lr = 3e-3
else:
    set_trainable(model.classifier, True)
    params = list(model.classifier.parameters())
    lr = 3e-3

optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)


def accuracy_from_logits(logits, y):
    return (logits.argmax(dim=1) == y).float().mean()


best_val_acc = -1.0
best_state = None


def _copy_state_to_cpu(sd: dict) -> dict:
    out = {}
    for k, v in sd.items():
        if torch.is_tensor(v):
            out[k] = v.detach().to("cpu", copy=True)
        else:
            out[k] = v
    return out


def run_epoch(train: bool):
    model.train(mode=train)

    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0
    loader = train_loader if train else val_loader

    if train:
        context = torch.enable_grad()
    else:
        context = torch.inference_mode()

    with context:
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            if torch.cuda.is_available():
                xb = xb.contiguous(memory_format=torch.channels_last)
            xb = gpu_batch_transform(xb, out_size=model_image_size)

            yb = yb.to(device, non_blocking=True)

            logits = model(xb)
            loss = criterion(logits, yb)
            acc = accuracy_from_logits(logits, yb)

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            total_loss += float(loss.detach())
            total_acc += float(acc.detach())
            n_batches += 1

    inv = 1.0 / max(n_batches, 1)
    return total_loss * inv, total_acc * inv


epochs_head = 2
for ep in range(epochs_head):
    tr_loss, tr_acc = run_epoch(train=True)
    va_loss, va_acc = run_epoch(train=False)
    if va_acc > best_val_acc:
        best_val_acc = va_acc
        best_state = _copy_state_to_cpu(model.state_dict())
    print(
        f"[Head] epoch={ep+1}/{epochs_head} train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} val_loss={va_loss:.4f} val_acc={va_acc:.4f}"
    )

set_trainable(model, True)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=1e-4)

epochs_full = 1
for ep in range(epochs_full):
    tr_loss, tr_acc = run_epoch(train=True)
    va_loss, va_acc = run_epoch(train=False)
    if va_acc > best_val_acc:
        best_val_acc = va_acc
        best_state = _copy_state_to_cpu(model.state_dict())
    print(
        f"[Full] epoch={ep+1}/{epochs_full} train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} val_loss={va_loss:.4f} val_acc={va_acc:.4f}"
    )

if best_state is not None:
    model.load_state_dict(best_state, strict=True)

model.eval()
print(f"Best val acc (for sanity): {best_val_acc:.4f}")




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/2978801200.py in <cell line: 0>()
    244 epochs_head = 2
    245 for ep in range(epochs_head):
--> 246     tr_loss, tr_acc = run_epoch(train=True)
    247     va_loss, va_acc = run_epoch(train=False)
    248     if va_acc > best_val_acc:

/tmp/ipykernel_55/2978801200.py in run_epoch(train)
    225             yb = yb.to(device, non_blocking=True)
    226 
--> 227             logits = model(xb)
    228             loss = criterion(logits, yb)
    229             acc = accuracy_from_logits(logits, yb)

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/eval_frame.py in _fn(*args, **kwargs)
    572 
    573             try:
--> 574                 return fn(*args, **kwargs)
    575             finally:
    576                 # Restore the dynamic layer stack depth if necessary.

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/usr/local/lib/python3.11/dist-packages/torchvision/models/vision_transformer.py in forward(self, x)
    287         return x
    288 
--> 289     def forward(self, x: torch.Tensor):
    290         # Reshape and permute the input tensor
    291         x = self._process_input(x)

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/eval_frame.py in _fn(*args, **kwargs)
    743             )
    744             try:
--> 745                 return fn(*args, **kwargs)
    746             finally:
    747                 _maybe_set_eval_frame(prior)

/usr/local/lib/python3.11/dist-packages/torch/_functorch/aot_autograd.py in forward(*runtime_args)
   1182         full_args.extend(params_flat)
   1183         full_args.extend(runtime_args)
-> 1184         return compiled_fn(full_args)
   1185 
   1186     # Just for convenience

/usr/local/lib/python3.11/dist-packages/torch/_functorch/_aot_autograd/runtime_wrappers.py in runtime_wrapper(args)
    308                 True
    309             ), torch.enable_grad():
--> 310                 all_outs = call_func_at_runtime_with_args(
    311                     compiled_fn, args_, disable_amp=disable_amp, steal_args=True
    312                 )

/usr/local/lib/python3.11/dist-packages/torch/_functorch/_aot_autograd/utils.py in call_func_at_runtime_with_args(f, args, steal_args, disable_amp)
    124     with context():
    125         if hasattr(f, "_boxed_call"):
--> 126             out = normalize_as_list(f(args))
    127         else:
    128             # TODO: Please remove soon

/usr/local/lib/python3.11/dist-packages/torch/_functorch/_aot_autograd/utils.py in g(args)
     98 def make_boxed_func(f):
     99     def g(args):
--> 100         return f(*args)
    101 
    102     g._boxed_call = True  # type: ignore[attr-defined]

/usr/local/lib/python3.11/dist-packages/torch/autograd/function.py in apply(cls, *args, **kwargs)
    573             # See NOTE: [functorch vjp and autograd interaction]
    574             args = _functorch.utils.unwrap_dead_wrappers(args)
--> 575             return super().apply(*args, **kwargs)  # type: ignore[misc]
    576 
    577         if not is_setup_ctx_defined:

/usr/local/lib/python3.11/dist-packages/torch/_functorch/_aot_autograd/runtime_wrappers.py in forward(ctx, *deduped_flat_tensor_args)
   1583                 # - Note that donated buffer logic requires (*saved_tensors, *saved_symints) showing up last
   1584                 #   in the fw output order.
-> 1585                 fw_outs = call_func_at_runtime_with_args(
   1586                     CompiledFunction.compiled_fw,
   1587                     args,

/usr/local/lib/python3.11/dist-packages/torch/_functorch/_aot_autograd/utils.py in call_func_at_runtime_with_args(f, args, steal_args, disable_amp)
    124     with context():
    125         if hasattr(f, "_boxed_call"):
--> 126             out = normalize_as_list(f(args))
    127         else:
    128             # TODO: Please remove soon

/usr/local/lib/python3.11/dist-packages/torch/_functorch/_aot_autograd/runtime_wrappers.py in wrapper(runtime_args)
    488                 )
    489                 return out
--> 490             return compiled_fn(runtime_args)
    491 
    492         return wrapper

/usr/local/lib/python3.11/dist-packages/torch/_functorch/_aot_autograd/runtime_wrappers.py in inner_fn(args)
    670                 old_args.clear()
    671 
--> 672             outs = compiled_fn(args)
    673 
    674             # Inductor cache DummyModule can return None

/usr/local/lib/python3.11/dist-packages/torch/_inductor/output_code.py in __call__(self, inputs)
    464         assert self.current_callable is not None
    465         try:
--> 466             return self.current_callable(inputs)
    467         finally:
    468             AutotuneCacheBundler.end_compile()

/usr/local/lib/python3.11/dist-packages/torch/_inductor/compile_fx.py in run(new_inputs)
   1206             ), dynamo_utils.preserve_rng_state():
   1207                 compiled_fn = cudagraphify_fn(model, new_inputs, static_input_idxs)
-> 1208         return compiled_fn(new_inputs)
   1209 
   1210     return run

/usr/local/lib/python3.11/dist-packages/torch/_inductor/cudagraph_trees.py in deferred_cudagraphify(inputs)
    396         copy_misaligned_inputs(inputs, check_input_idxs)
    397 
--> 398         fn, out = cudagraphify(model, inputs, new_static_input_idxs, *args, **kwargs)
    399         fn = align_inputs_from_check_idxs(fn, inputs_to_check=check_input_idxs)
    400         fn_cache[int_key] = fn

/usr/local/lib/python3.11/dist-packages/torch/_inductor/cudagraph_trees.py in cudagraphify(model, inputs, static_input_idxs, device_index, is_backward, is_inference, stack_traces, constants, placeholders, mutated_input_idxs)
    426     )
    427 
--> 428     return manager.add_function(
    429         model,
    430         inputs,

/usr/local/lib/python3.11/dist-packages/torch/_inductor/cudagraph_trees.py in add_function(self, model, inputs, static_input_idxs, stack_traces, mode, constants, placeholders, mutated_input_idxs)
   2251         # container needs to set clean up when fn dies
   2252         get_container(self.device_index).add_strong_reference(fn)
-> 2253         return fn, fn(inputs)
   2254 
   2255     @property

/usr/local/lib/python3.11/dist-packages/torch/_inductor/cudagraph_trees.py in run(self, new_inputs, function_id)
   1945         assert self.graph is not None, "Running CUDAGraph after shutdown"
   1946         self.mode = self.id_to_mode[function_id]
-> 1947         out = self._run(new_inputs, function_id)
   1948 
   1949         # The forwards are only pending following invocation, not before

/usr/local/lib/python3.11/dist-packages/torch/_inductor/cudagraph_trees.py in _run(self, new_inputs, function_id)
   2053                 log_pt2_compile_event=True,
   2054             ):
-> 2055                 out = self.run_eager(new_inputs, function_id)
   2056 
   2057             return out

/usr/local/lib/python3.11/dist-packages/torch/_inductor/cudagraph_trees.py in run_eager(self, new_inputs, function_id)
   2217         self.path_state = ExecutionState.WARMUP
   2218         self.update_generation()
-> 2219         return node.run(new_inputs)
   2220 
   2221     def new_graph_id(self) -> GraphID:

/usr/local/lib/python3.11/dist-packages/torch/_inductor/cudagraph_trees.py in run(self, new_inputs)
    630             return non_cudagraph_inps
    631 
--> 632         non_cudagraph_inps_storages = get_non_cudagraph_inps()
    633 
    634         if config.triton.slow_path_cudagraph_asserts and not self.already_warm:

/usr/local/lib/python3.11/dist-packages/torch/_inductor/cudagraph_trees.py in get_non_cudagraph_inps()
    622 
    623         def get_non_cudagraph_inps() -> List[weakref.ReferenceType[UntypedStorage]]:
--> 624             non_cudagraph_inps = [
    625                 weakref.ref(t.untyped_storage())
    626                 for t in itertools.chain(new_inputs, self.wrapped_function.constants)

/usr/local/lib/python3.11/dist-packages/torch/_inductor/cudagraph_trees.py in <listcomp>(.0)
    626                 for t in itertools.chain(new_inputs, self.wrapped_function.constants)
    627                 if isinstance(t, torch.Tensor)
--> 628                 and t.untyped_storage().data_ptr() not in existing_path_data_ptrs
    629             ]
    630             return non_cudagraph_inps

RuntimeError: Error: accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run. Stack trace: File "/tmp/ipykernel_55/3155300467.py", line 47, in gpu_batch_transform
    xb = (xb - _NORMALIZE_MEAN) / _NORMALIZE_STD. To prevent overwriting, clone the tensor outside of torch.compile() or call torch.compiler.cudagraph_mark_step_begin() before each model invocation.

## === cell 5
def resolve_test_image_path(base_dir: str, image_name: str) -> str:
    p1 = os.path.join(base_dir, image_name)
    if os.path.isfile(p1):
        return p1
    p2 = os.path.join(base_dir, "test_images", image_name)
    if os.path.isfile(p2):
        return p2
    alt = glob.glob(os.path.join(base_dir, "**", image_name), recursive=True)
    alt = [p for p in alt if os.path.isfile(p)]
    if alt:
        return sorted(alt, key=lambda p: (len(p), p))[0]
    raise FileNotFoundError(
        f"Could not locate test image {image_name} under {base_dir}"
    )


sample_path = "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
sample_df = pd.read_csv(sample_path)
ordered_image_ids = sample_df["image_id"].tolist()

test_paths = {}
missing = []
base_dir = test_data_directory
nested_dir = os.path.join(test_data_directory, "test_images")
for iid in ordered_image_ids:
    p = os.path.join(base_dir, iid)
    if os.path.isfile(p):
        test_paths[iid] = p
        continue
    p2 = os.path.join(nested_dir, iid)
    if os.path.isfile(p2):
        test_paths[iid] = p2
        continue
    missing.append(iid)

if missing:
    all_files = glob.glob(
        os.path.join(test_data_directory, "**", "*.jpg"), recursive=True
    )
    base_to_path = {}
    for p in all_files:
        b = os.path.basename(p)
        prev = base_to_path.get(b)
        if (prev is None) or ((len(p), p) < (len(prev), prev)):
            base_to_path[b] = p
    for iid in missing:
        p = base_to_path.get(iid)
        if p is not None:
            test_paths[iid] = p
        else:
            test_paths[iid] = resolve_test_image_path(test_data_directory, iid)


class CassavaTestDataset(torch.utils.data.Dataset):
    def __init__(self, image_ids, path_map, transform, cache_images: bool = True):
        self.image_ids = list(image_ids)
        self.path_map = path_map
        self.transform = transform  # retained; not used in optimized path

        self.cache_images = bool(cache_images)
        self._cache = {} if (self.cache_images) else None

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx: int):
        image_id = self.image_ids[idx]
        image_path = self.path_map[image_id]
        if self._cache is not None:
            x = self._cache.get(image_id)
            if x is None:
                x = read_image(image_path, mode=ImageReadMode.RGB)  # uint8 CHW
                self._cache[image_id] = x
        else:
            x = read_image(image_path, mode=ImageReadMode.RGB)  # uint8 CHW
        return x, image_id


test_ds = CassavaTestDataset(
    ordered_image_ids, test_paths, val_transforms, cache_images=False
)

test_loader = torch.utils.data.DataLoader(
    test_ds,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(num_workers > 0),
    prefetch_factor=_prefetch,
    worker_init_fn=_seed_worker,
    collate_fn=fast_collate_test,
)

predictions_arr = np.empty(len(ordered_image_ids), dtype=np.int64)
image_ids = ordered_image_ids  # preserve exact sample_submission ordering
offset = 0

with torch.inference_mode():
    for xb, ids in tqdm(test_loader, desc="Test", total=len(test_loader)):
        xb = xb.to(device, non_blocking=True)
        if torch.cuda.is_available():
            xb = xb.contiguous(memory_format=torch.channels_last)
        xb = gpu_batch_transform(xb, out_size=model_image_size)
        out = model(xb)
        preds = out.argmax(dim=1).to("cpu").numpy().astype(np.int64)
        bs = preds.shape[0]
        predictions_arr[offset : offset + bs] = preds
        offset += bs

predictions = predictions_arr.tolist()



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
AssertionError                            Traceback (most recent call last)
/tmp/ipykernel_55/1224613900.py in <cell line: 0>()
    102         if torch.cuda.is_available():
    103             xb = xb.contiguous(memory_format=torch.channels_last)
--> 104         xb = gpu_batch_transform(xb, out_size=model_image_size)
    105         out = model(xb)
    106         preds = out.argmax(dim=1).to("cpu").numpy().astype(np.int64)

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/eval_frame.py in _fn(*args, **kwargs)
    572 
    573             try:
--> 574                 return fn(*args, **kwargs)
    575             finally:
    576                 # Restore the dynamic layer stack depth if necessary.

/tmp/ipykernel_55/3155300467.py in gpu_batch_transform(xb, out_size)
     34 
     35 
---> 36 def gpu_batch_transform(xb: torch.Tensor, *, out_size: int) -> torch.Tensor:
     37     if xb.dtype != torch.float32:
     38         xb = xb.to(torch.float32)

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/eval_frame.py in _fn(*args, **kwargs)
    743             )
    744             try:
--> 745                 return fn(*args, **kwargs)
    746             finally:
    747                 _maybe_set_eval_frame(prior)

/usr/local/lib/python3.11/dist-packages/torch/_functorch/aot_autograd.py in forward(*runtime_args)
   1182         full_args.extend(params_flat)
   1183         full_args.extend(runtime_args)
-> 1184         return compiled_fn(full_args)
   1185 
   1186     # Just for convenience

/usr/local/lib/python3.11/dist-packages/torch/_functorch/_aot_autograd/runtime_wrappers.py in runtime_wrapper(args)
    321                 if grad_enabled:
    322                     torch._C._set_grad_enabled(False)
--> 323                 all_outs = call_func_at_runtime_with_args(
    324                     compiled_fn, args, disable_amp=disable_amp, steal_args=True
    325                 )

/usr/local/lib/python3.11/dist-packages/torch/_functorch/_aot_autograd/utils.py in call_func_at_runtime_with_args(f, args, steal_args, disable_amp)
    124     with context():
    125         if hasattr(f, "_boxed_call"):
--> 126             out = normalize_as_list(f(args))
    127         else:
    128             # TODO: Please remove soon

/usr/local/lib/python3.11/dist-packages/torch/_functorch/_aot_autograd/runtime_wrappers.py in inner_fn(args)
    670                 old_args.clear()
    671 
--> 672             outs = compiled_fn(args)
    673 
    674             # Inductor cache DummyModule can return None

/usr/local/lib/python3.11/dist-packages/torch/_functorch/_aot_autograd/runtime_wrappers.py in wrapper(runtime_args)
    488                 )
    489                 return out
--> 490             return compiled_fn(runtime_args)
    491 
    492         return wrapper

/usr/local/lib/python3.11/dist-packages/torch/_inductor/output_code.py in __call__(self, inputs)
    464         assert self.current_callable is not None
    465         try:
--> 466             return self.current_callable(inputs)
    467         finally:
    468             AutotuneCacheBundler.end_compile()

/usr/local/lib/python3.11/dist-packages/torch/_inductor/compile_fx.py in run(new_inputs)
   1206             ), dynamo_utils.preserve_rng_state():
   1207                 compiled_fn = cudagraphify_fn(model, new_inputs, static_input_idxs)
-> 1208         return compiled_fn(new_inputs)
   1209 
   1210     return run

/usr/local/lib/python3.11/dist-packages/torch/_inductor/cudagraph_trees.py in deferred_cudagraphify(inputs)
    396         copy_misaligned_inputs(inputs, check_input_idxs)
    397 
--> 398         fn, out = cudagraphify(model, inputs, new_static_input_idxs, *args, **kwargs)
    399         fn = align_inputs_from_check_idxs(fn, inputs_to_check=check_input_idxs)
    400         fn_cache[int_key] = fn

/usr/local/lib/python3.11/dist-packages/torch/_inductor/cudagraph_trees.py in cudagraphify(model, inputs, static_input_idxs, device_index, is_backward, is_inference, stack_traces, constants, placeholders, mutated_input_idxs)
    426     )
    427 
--> 428     return manager.add_function(
    429         model,
    430         inputs,

/usr/local/lib/python3.11/dist-packages/torch/_inductor/cudagraph_trees.py in add_function(self, model, inputs, static_input_idxs, stack_traces, mode, constants, placeholders, mutated_input_idxs)
   2251         # container needs to set clean up when fn dies
   2252         get_container(self.device_index).add_strong_reference(fn)
-> 2253         return fn, fn(inputs)
   2254 
   2255     @property

/usr/local/lib/python3.11/dist-packages/torch/_inductor/cudagraph_trees.py in run(self, new_inputs, function_id)
   1945         assert self.graph is not None, "Running CUDAGraph after shutdown"
   1946         self.mode = self.id_to_mode[function_id]
-> 1947         out = self._run(new_inputs, function_id)
   1948 
   1949         # The forwards are only pending following invocation, not before

/usr/local/lib/python3.11/dist-packages/torch/_inductor/cudagraph_trees.py in _run(self, new_inputs, function_id)
   2015 
   2016         if self.in_warmup:
-> 2017             self.try_end_curr_warmup(function_id)
   2018 
   2019         node_id = self._get_node_id()

/usr/local/lib/python3.11/dist-packages/torch/_inductor/cudagraph_trees.py in try_end_curr_warmup(self, function_id)
   2344     def try_end_curr_warmup(self, function_id: FunctionID) -> None:
   2345         if self.can_start_new_generation():
-> 2346             self.dealloc_current_path_weakrefs()
   2347             self.current_node = None
   2348             return

/usr/local/lib/python3.11/dist-packages/torch/_inductor/cudagraph_trees.py in dealloc_current_path_weakrefs(self)
   2409         for node in self.current_node._path_from_root:
   2410             assert node.stack_traces is not None
-> 2411             assert len(node.tensor_weakrefs) == len(node.stack_traces)
   2412             for t, stack_trace in zip(node.tensor_weakrefs, node.stack_traces):
   2413                 ten = None if t is None else t()

AssertionError: 

## === cell 6
submission_df = pd.DataFrame({"image_id": image_ids, "label": predictions})

assert (
    submission_df["image_id"].tolist() == sample_df["image_id"].tolist()
), "image_id ordering mismatch vs sample_submission; this would invalidate the submission."
assert len(submission_df) == len(
    sample_df
), "Submission row count does not match sample_submission."
assert set(submission_df.columns) == {"image_id", "label"}

submission_df.to_csv("submission.csv", index=False)
print("Submission file created: submission.csv")
print(submission_df.head())

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/144377194.py in <cell line: 0>()
----> 1 submission_df = pd.DataFrame({"image_id": image_ids, "label": predictions})
      2 
      3 assert (
      4     submission_df["image_id"].tolist() == sample_df["image_id"].tolist()
      5 ), "image_id ordering mismatch vs sample_submission; this would invalidate the submission."

NameError: name 'predictions' is not defined
