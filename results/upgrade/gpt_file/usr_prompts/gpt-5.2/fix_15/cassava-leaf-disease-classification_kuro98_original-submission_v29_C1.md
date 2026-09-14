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

geopandas==0.14.4
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
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124

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
import time
import random

import pandas as pd
import torch
from torch.backends import cudnn
import torchvision

torch.manual_seed(3407)
random.seed(3407)
if torch.cuda.is_available():
    torch.cuda.manual_seed(3407)

cudnn.deterministic = True
cudnn.benchmark = False

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

try:
    torchvision.set_image_backend("accimage")
except Exception:
    pass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

base_dir = "/kaggle/input/cassava-leaf-disease-classification"
train_csv_path = f"{base_dir}/train.csv"
train_dir = f"{base_dir}/train_images/"
test_dir = f"{base_dir}/test_images/"
sample_sub_path = f"{base_dir}/sample_submission.csv"

img_size = 224

batch_size = 16

_cpu = os.cpu_count() or 4
num_workers = min(8, max(2, _cpu - 2))

num_classes = 5
tta = True

epochs = 6

lr = 3e-4
weight_decay = 1e-4
train_print_every = 50

use_pretrained = True


def _find_checkpoint():
    local_candidates = [
        "best_vit_b16.pth",
        "vit_b16_trained.pth",
        "submission_model.pth",
    ]
    for p in local_candidates:
        if os.path.exists(p):
            return p

    preferred_names = [
        "best_vit_b16.pth",
        "vit_b16_trained.pth",
        "submission_model.pth",
        "vit_v6.pt",
        "vit-v6.pt",
        "vitv6.pt",
        "model.pth",
        "checkpoint.pth",
        "best.pth",
    ]

    for name in preferred_names:
        p = os.path.join("/kaggle/working", name)
        if os.path.exists(p):
            return p

    for name in preferred_names:
        p = os.path.join(base_dir, name)
        if os.path.exists(p):
            return p

    roots = ["/kaggle/input", "/kaggle/working"]
    for root in roots:
        try:
            for name in preferred_names:
                for p in glob.glob(os.path.join(root, "*", name)):
                    if os.path.exists(p):
                        return p
                for p in glob.glob(os.path.join(root, "*", "*", name)):
                    if os.path.exists(p):
                        return p
        except Exception:
            pass

    return None


def _build_model(num_classes=5, use_pretrained=True):
    if use_pretrained:
        weights = torchvision.models.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1
        m = torchvision.models.vit_b_16(weights=weights)
    else:
        m = torchvision.models.vit_b_16(weights=None)

    in_features = m.heads.head.in_features
    m.heads.head = torch.nn.Linear(in_features, num_classes)
    return m


def _load_model_from_checkpoint(ckpt_path, num_classes=5, use_pretrained=True):
    m = _build_model(num_classes=num_classes, use_pretrained=use_pretrained)

    obj = None
    try:
        obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        obj = torch.load(ckpt_path, map_location="cpu")
    except Exception as e:
        print("torch.load failed:", repr(e))
        return None

    if isinstance(obj, torch.nn.Module):
        return obj

    if isinstance(obj, dict):
        state = None
        for k in ("state_dict", "model_state_dict", "model", "net", "weights"):
            if k in obj and isinstance(obj[k], dict):
                state = obj[k]
                break
        if state is None and all(isinstance(k, str) for k in obj.keys()):
            state = obj

        if state is None:
            print("Checkpoint dict did not contain a usable state_dict-like object.")
            return None

        cleaned = {}
        for k, v in state.items():
            nk = k
            if nk.startswith("module."):
                nk = nk[len("module.") :]
            if nk.startswith("model."):
                nk = nk[len("model.") :]
            cleaned[nk] = v

        missing, unexpected = m.load_state_dict(cleaned, strict=False)
        print(
            "Loaded state_dict with strict=False. Missing keys:",
            len(missing),
            "Unexpected keys:",
            len(unexpected),
        )
        return m

    print("Unsupported checkpoint object type:", type(obj))
    return None


ckpt_path = _find_checkpoint()
print("Checkpoint found:", ckpt_path)

model = None
if ckpt_path is not None and os.path.exists(ckpt_path):
    model = _load_model_from_checkpoint(
        ckpt_path, num_classes=num_classes, use_pretrained=use_pretrained
    )
    if model is not None:
        print("Loaded checkpoint:", ckpt_path)
    else:
        print("Failed to interpret checkpoint contents; will train a new model.")

if model is None:
    model = _build_model(num_classes=num_classes, use_pretrained=use_pretrained)
    print("Initialized torchvision vit_b_16 (will train). Pretrained:", use_pretrained)

if hasattr(model, "image_size"):
    try:
        img_size = int(model.image_size)
        print("Using model.image_size:", img_size)
    except Exception:
        pass

model = model.to(device)




## === cell 1
import torchvision.io as tvio
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset

from functools import lru_cache


@lru_cache(maxsize=4096)
def _read_image_rgb_cached(path: str):
    return tvio.read_image(path, mode=tvio.ImageReadMode.RGB)


def _read_image_rgb(path: str):
    return _read_image_rgb_cached(path)


class CassavaTrainDataset(VisionDataset):
    """Train dataset (image_id + label) using the same image loading path as test."""

    def __init__(self, data_dir, df, transform=None):
        super().__init__(root=data_dir)
        df = df.reset_index(drop=True)
        self.image_ids = df["image_id"].to_numpy()
        self.labels = df["label"].to_numpy(dtype="int64")
        self.transform = transform

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        label = int(self.labels[idx])
        img = _read_image_rgb(os.path.join(self.root, image_id))
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.image_ids)


class CassavaDataset(VisionDataset):
    """Custom dataset for Cassava test data (image_id only)."""

    def __init__(self, data_dir, image_ids=None, transform=None, ttas=None):
        super().__init__(root=data_dir)
        self.transform = transform
        self.ttas = ttas

        if image_ids is None:
            self.images = sorted(os.listdir(data_dir))
        else:
            self.images = list(image_ids)

    def __getitem__(self, idx):
        filename = self.images[idx]
        img = _read_image_rgb(os.path.join(self.root, filename))

        if self.transform:
            img = self.transform(img)

        return img, filename

    def __len__(self):
        return len(self.images)




## === cell 2
from torchvision.transforms import InterpolationMode, v2

vit_weights = torchvision.models.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1
vit_mean = list(vit_weights.transforms().mean)
vit_std = list(vit_weights.transforms().std)

train_transforms = v2.Compose(
    [
        v2.Resize((img_size, img_size), interpolation=InterpolationMode.BILINEAR),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomRotation(degrees=15, interpolation=InterpolationMode.BILINEAR),
        v2.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=vit_mean, std=vit_std),
    ]
)

test_transforms_uint8 = v2.Compose(
    [
        v2.CenterCrop((600, 600)),
        v2.ToImage(),  # keep uint8 CHW
    ]
)

if tta:

    def test_transforms(img):
        base = test_transforms_uint8(img)
        return [
            base,
            v2.functional.horizontal_flip(base),
            v2.functional.vertical_flip(base),
            v2.functional.vertical_flip(v2.functional.horizontal_flip(base)),
        ]

    ttas = ["id", "hflip", "vflip", "hvflip"]
else:

    def test_transforms(img):
        return test_transforms_uint8(img)

    ttas = None


train_df = pd.read_csv(train_csv_path)
sample_sub = pd.read_csv(sample_sub_path)
test_image_ids = sample_sub["image_id"].tolist()

val_frac = 0.1

train_df = train_df.sample(frac=1.0, random_state=3407).reset_index(drop=True)
grp_idx = train_df.groupby("label", sort=False).cumcount()
grp_size = train_df.groupby("label", sort=False)["label"].transform("size")
n_val_per_row = (grp_size * val_frac).round().astype("int64").clip(lower=1)
is_val = grp_idx < n_val_per_row
val_df_split = train_df[is_val].reset_index(drop=True)
train_df_split = train_df[~is_val].reset_index(drop=True)

train_dataset = CassavaTrainDataset(
    train_dir, train_df_split, transform=train_transforms
)

test_transforms_base = v2.Compose(
    [
        v2.CenterCrop((600, 600)),
        v2.Resize((img_size, img_size), interpolation=InterpolationMode.BILINEAR),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=vit_mean, std=vit_std),
    ]
)

val_dataset = CassavaTrainDataset(
    train_dir, val_df_split, transform=test_transforms_base
)


def _seed_worker(worker_id: int):
    seed = 3407 + worker_id
    random.seed(seed)
    torch.manual_seed(seed)


loader_kwargs = dict(
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else None,
    worker_init_fn=_seed_worker if num_workers > 0 else None,
)

loader_kwargs = {k: v for k, v in loader_kwargs.items() if v is not None}


def _collate_test_tta(batch):
    imgs0, _ = batch[0]
    t = len(imgs0)
    b = len(batch)
    c, h, w = imgs0[0].shape
    out = torch.empty((t * b, c, h, w), dtype=imgs0[0].dtype)
    filenames = [None] * b
    k = 0
    for i, (imgs, fn) in enumerate(batch):
        filenames[i] = fn
        for j in range(t):
            out[k].copy_(imgs[j])
            k += 1
    return out, t, filenames


def _collate_test_no_tta(batch):
    imgs = torch.stack([x for x, _ in batch], dim=0)
    fns = [fn for _, fn in batch]
    return imgs, fns


train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    **loader_kwargs,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    **loader_kwargs,
)

test_dataset = CassavaDataset(
    test_dir, image_ids=test_image_ids, transform=test_transforms, ttas=ttas
)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=_collate_test_tta if tta else _collate_test_no_tta,
    **loader_kwargs,
)

normalizer = torch.nn.Softmax(dim=1)

_counts = (
    train_df_split["label"]
    .value_counts()
    .reindex(range(num_classes), fill_value=0)
    .values
)
_counts_t = torch.tensor(_counts, dtype=torch.float32)
_class_weights = _counts_t.sum() / (_counts_t + 1e-6)  # inverse frequency
_class_weights = (_class_weights / _class_weights.mean()).to(device)  # normalized

print("Train split class counts:", _counts.tolist())
print("Using class weights:", [float(x) for x in _class_weights.detach().cpu()])

_mean = torch.tensor(vit_mean, device=device).view(1, 3, 1, 1)
_std = torch.tensor(vit_std, device=device).view(1, 3, 1, 1)




## === cell 3
use_amp = torch.cuda.is_available()
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

criterion_train = torch.nn.CrossEntropyLoss(weight=_class_weights)
criterion_val = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

best_acc = -1.0
best_path = "best_vit_b16.pth"

trained_or_loaded = False
if os.path.exists(best_path):
    loaded = _load_model_from_checkpoint(
        best_path, num_classes=num_classes, use_pretrained=use_pretrained
    )
    if loaded is not None:
        model = loaded.to(device)
        trained_or_loaded = True
        print("Found existing best checkpoint; skipping training and using:", best_path)
    else:
        print("Existing best checkpoint found but failed to load; proceeding to train.")

if torch.cuda.is_available():
    model = model.to(memory_format=torch.channels_last)

if torch.cuda.is_available():
    try:
        model = torch.compile(model, mode="reduce-overhead")
        print("torch.compile enabled (reduce-overhead)")
    except Exception as e:
        print("torch.compile unavailable, continuing without it:", repr(e))

if not trained_or_loaded:
    start_time = time.time()
    n_train_steps = len(train_loader)
    n_val_steps = len(val_loader)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n_seen = 0

        for step, (x, y) in enumerate(train_loader, start=1):
            if torch.cuda.is_available():
                x = x.to(device, non_blocking=True, memory_format=torch.channels_last)
            else:
                x = x.to(device)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(x)
                loss = criterion_train(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            bsz = x.size(0)
            running_loss += loss.item() * bsz
            n_seen += bsz

            if step % train_print_every == 0:
                print(
                    f"epoch {epoch}/{epochs} step {step}/{n_train_steps} "
                    f"loss {running_loss/n_seen:.4f} elapsed {time.time()-start_time:.1f}s"
                )

        epoch_loss = running_loss / max(1, n_seen)
        print(f"epoch {epoch}/{epochs} train_loss {epoch_loss:.4f}")

        model.eval()
        correct = 0
        total = 0
        val_loss_sum = 0.0
        with torch.inference_mode():
            for x, y in val_loader:
                if torch.cuda.is_available():
                    x = x.to(
                        device, non_blocking=True, memory_format=torch.channels_last
                    )
                else:
                    x = x.to(device)
                y = y.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    logits = model(x)
                    loss = criterion_val(logits, y)
                val_loss_sum += loss.item() * x.size(0)

                preds = torch.argmax(logits, dim=1)
                correct += (preds == y).sum().item()
                total += y.numel()

        val_acc = correct / max(1, total)
        val_loss = val_loss_sum / max(1, total)
        print(f"epoch {epoch}/{epochs} val_loss {val_loss:.4f} val_acc {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"state_dict": model.state_dict()}, best_path)
            print("Saved best checkpoint to:", best_path, "val_acc:", best_acc)

    loaded = _load_model_from_checkpoint(
        best_path, num_classes=num_classes, use_pretrained=use_pretrained
    )
    if loaded is not None:
        model = loaded.to(device)
        if torch.cuda.is_available():
            model = model.to(memory_format=torch.channels_last)
            try:
                model = torch.compile(model, mode="reduce-overhead")
            except Exception:
                pass
        print("Reloaded best checkpoint for inference:", best_path)
    else:
        print(
            "Warning: failed to reload best checkpoint; using current in-memory model."
        )




## === cell 4
all_names = []
all_preds = []

model.eval()

normalizer = torch.nn.Softmax(dim=1)


def _preprocess_test_batch_uint8(x_uint8_bchw: torch.Tensor) -> torch.Tensor:
    if x_uint8_bchw.device.type != device.type:
        if torch.cuda.is_available():
            x_uint8_bchw = x_uint8_bchw.to(
                device, non_blocking=True, memory_format=torch.channels_last
            )
        else:
            x_uint8_bchw = x_uint8_bchw.to(device)
    x = x_uint8_bchw.to(torch.float32).div_(255.0)
    x = torch.nn.functional.interpolate(
        x, size=(img_size, img_size), mode="bilinear", align_corners=False
    )
    x = (x - _mean) / _std
    if torch.cuda.is_available():
        x = x.to(memory_format=torch.channels_last)
    return x


with torch.inference_mode():
    if tta:
        for batch_idx, (inputs_cat_uint8, t, filenames) in enumerate(test_loader):
            bsz = len(filenames)

            inputs_cat = _preprocess_test_batch_uint8(inputs_cat_uint8)

            with torch.cuda.amp.autocast(enabled=use_amp):
                probs = normalizer(model(inputs_cat))
            probs = probs.view(t, bsz, -1).mean(dim=0)
            pred_labels = torch.argmax(probs, dim=1).tolist()

            all_names.extend(list(filenames))
            all_preds.extend(pred_labels)
    else:
        for batch_idx, (inputs_uint8, filenames) in enumerate(test_loader):
            inputs = _preprocess_test_batch_uint8(inputs_uint8)
            with torch.cuda.amp.autocast(enabled=use_amp):
                probs = normalizer(model(inputs))
            pred_labels = torch.argmax(probs, dim=1).tolist()

            all_names.extend(list(filenames))
            all_preds.extend(pred_labels)

assert len(all_names) == len(
    test_image_ids
), f"Pred count {len(all_names)} != expected {len(test_image_ids)}"
assert len(all_preds) == len(
    test_image_ids
), f"Label count {len(all_preds)} != expected {len(test_image_ids)}"

my_submission = pd.DataFrame({"image_id": all_names, "label": all_preds})




## === cell 5
my_submission = (
    my_submission.set_index("image_id").loc[sample_sub["image_id"]].reset_index()
)
my_submission["label"] = my_submission["label"].astype(int)

out_path = "submission.csv"
my_submission.to_csv(out_path, index=False)
print("Wrote:", out_path, "rows:", len(my_submission))
print(my_submission.head())
