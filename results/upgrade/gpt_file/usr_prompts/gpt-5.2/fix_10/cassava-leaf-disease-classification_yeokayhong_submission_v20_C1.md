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

# 5. Code solution

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
torch.backends.cudnn.benchmark = False

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

Image.MAX_IMAGE_PIXELS = None

_CPU_COUNT = os.cpu_count() or 2
num_workers = min(8, max(2, _CPU_COUNT // 2))

_CAN_COMPILE = hasattr(torch, "compile")


def _seed_worker(worker_id: int):
    wseed = seed + worker_id
    random.seed(wseed)
    np.random.seed(wseed)
    torch.manual_seed(wseed)




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
        self, df: pd.DataFrame, images_dir: str, transform, pre_resolved_paths=None
    ):
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.transform = transform

        self.image_ids = self.df["image_id"].to_numpy()
        self.labels = self.df["label"].to_numpy(dtype=np.int64)

        self._paths = pre_resolved_paths

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx: int):
        image_id = self.image_ids[idx]
        label = int(self.labels[idx])

        if self._paths is not None:
            image_path = self._paths[image_id]
        else:
            image_path = os.path.join(self.images_dir, image_id)
            if not os.path.isfile(image_path):
                alt = glob.glob(
                    os.path.join(self.images_dir, "**", image_id), recursive=True
                )
                alt = [p for p in alt if os.path.isfile(p)]
                if not alt:
                    raise FileNotFoundError(
                        f"Could not locate train image {image_id} under {self.images_dir}"
                    )
                image_path = sorted(alt, key=lambda p: (len(p), p))[0]

        im = Image.open(image_path)
        try:
            image = im.convert("RGB")
        finally:
            im.close()

        x = self.transform(image)
        y = torch.tensor(label, dtype=torch.long)
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
    tr_df, train_data_directory, train_transforms, pre_resolved_paths=train_path_map
)
val_ds = CassavaDataset(
    va_df, train_data_directory, val_transforms, pre_resolved_paths=train_path_map
)

pin = torch.cuda.is_available()
_prefetch = 4 if num_workers > 0 else None

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
)

if model_select == "vit":
    model = vit_model
else:
    model = en_model

if _CAN_COMPILE:
    try:
        model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
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


_val_cache_x = {}
_val_cache_lock = (
    None  # placeholder (single-process safe; DataLoader workers won't share)
)


def run_epoch(train: bool):
    if train:
        model.train()
        ctx = torch.enable_grad()
        loader = train_loader
        use_cache = False
    else:
        model.eval()
        ctx = torch.no_grad()
        loader = val_loader
        use_cache = True

    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0

    with ctx:
        if not use_cache:
            for xb, yb in loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)

                logits = model(xb)
                loss = criterion(logits, yb)
                acc = accuracy_from_logits(logits, yb)

                if train:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()

                total_loss += float(loss.detach().cpu())
                total_acc += float(acc.detach().cpu())
                n_batches += 1
        else:
            for xb, yb in loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)

                logits = model(xb)
                loss = criterion(logits, yb)
                acc = accuracy_from_logits(logits, yb)

                total_loss += float(loss.detach().cpu())
                total_acc += float(acc.detach().cpu())
                n_batches += 1

    inv = 1.0 / max(n_batches, 1)
    return total_loss * inv, total_acc * inv


epochs_head = 2
for ep in range(epochs_head):
    tr_loss, tr_acc = run_epoch(train=True)
    va_loss, va_acc = run_epoch(train=False)
    if va_acc > best_val_acc:
        best_val_acc = va_acc
        best_state = {
            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
        }
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
        best_state = {
            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
        }
    print(
        f"[Full] epoch={ep+1}/{epochs_full} train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} val_loss={va_loss:.4f} val_acc={va_acc:.4f}"
    )

if best_state is not None:
    model.load_state_dict(best_state, strict=True)

model.eval()
print(f"Best val acc (for sanity): {best_val_acc:.4f}")




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
    def __init__(self, image_ids, path_map, transform):
        self.image_ids = list(image_ids)
        self.path_map = path_map
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx: int):
        image_id = self.image_ids[idx]
        image_path = self.path_map[image_id]
        im = Image.open(image_path)
        try:
            image = im.convert("RGB")
        finally:
            im.close()
        x = self.transform(image)
        return x, image_id


test_ds = CassavaTestDataset(ordered_image_ids, test_paths, val_transforms)

test_loader = torch.utils.data.DataLoader(
    test_ds,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(num_workers > 0),
    prefetch_factor=_prefetch,
    worker_init_fn=_seed_worker,
)

predictions = []
image_ids = []

with torch.inference_mode():
    for xb, ids in tqdm(test_loader, desc="Test", total=len(test_loader)):
        xb = xb.to(device, non_blocking=True)
        out = model(xb)
        preds = out.argmax(dim=1).to("cpu").numpy().astype(np.int64)
        predictions.extend(preds.tolist())
        image_ids.extend(list(ids))



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
