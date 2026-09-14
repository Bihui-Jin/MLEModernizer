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
import hashlib
import pathlib
import random

import pandas as pd
import torch
from PIL import Image, ImageFile
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
from torchvision.transforms import InterpolationMode, v2

torch.set_num_threads(1)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

torch.manual_seed(3407)
random.seed(3407)
if torch.cuda.is_available():
    torch.cuda.manual_seed(3407)

cudnn.deterministic = True
cudnn.benchmark = False

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification"

train_csv_path = f"{DATA_DIR}/train.csv"
train_dir = f"{DATA_DIR}/train_images/"

test_dir = f"{DATA_DIR}/test_images/"
sample_sub_path = f"{DATA_DIR}/sample_submission.csv"

img_size = 224

batch_size = 16
num_workers = min(12, (os.cpu_count() or 4))
num_classes = 5
tta = True

train_epochs = 6
train_lr = 3e-4
finetune_epochs = 2
finetune_lr = 1e-5

from torchvision.models import vit_b_16, ViT_B_16_Weights

weights = ViT_B_16_Weights.IMAGENET1K_V1
model = vit_b_16(weights=weights)  # image_size fixed by weights (=224)
in_features = model.heads.head.in_features
model.heads.head = torch.nn.Linear(in_features, num_classes)

model.to(device)




## === cell 1
vit_preprocess = weights.transforms()

train_transforms = vit_preprocess
test_transforms = vit_preprocess

if tta:
    ttas = [
        v2.Identity(),  # include the original view
        v2.RandomHorizontalFlip(p=1.0),  # deterministic flip
        v2.RandomVerticalFlip(p=1.0),  # deterministic flip
        v2.RandomRotation(degrees=(90, 90), interpolation=InterpolationMode.BILINEAR),
    ]
else:
    ttas = None




## === cell 2
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

CACHE_DIR = "/kaggle/working/cassava_cc_cache_png"
pathlib.Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)

from collections import OrderedDict

_TENSOR_CACHE = OrderedDict()
_TENSOR_CACHE_MAX = 4096  # bounded in-RAM cache per worker


def _cache_key(img_path: str, key_suffix: str) -> str:
    st = os.stat(img_path)
    sig = f"{img_path}|{st.st_size}|{int(st.st_mtime)}|{key_suffix}"
    h = hashlib.sha1(sig.encode("utf-8")).hexdigest()
    return os.path.join(CACHE_DIR, f"{h}.pt")


def _load_rgb_pil(img_path: str) -> Image.Image:
    with Image.open(img_path) as im:
        im = im.convert("RGB")
        try:
            im.draft("RGB", (512, 512))
        except Exception:
            pass
        im.load()
    return im


def _atomic_save_tensor(t: torch.Tensor, cache_path: str) -> None:
    tmp_path = cache_path + f".tmp_{os.getpid()}"
    try:
        torch.save(t, tmp_path)
        try:
            os.replace(tmp_path, cache_path)
        except FileExistsError:
            pass
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def _cached_preprocessed_tensor(
    img_path: str, cc_transform, preprocess_transform, key_suffix: str
) -> torch.Tensor:
    """Returns a CPU float tensor [C,H,W] equivalent to preprocess_transform(cc_transform(PIL_RGB))."""
    k = (img_path, key_suffix)
    im_t = _TENSOR_CACHE.get(k)
    if im_t is not None:
        _TENSOR_CACHE.move_to_end(k)
        return im_t

    cache_path = _cache_key(img_path, key_suffix)
    if os.path.exists(cache_path):
        t = torch.load(cache_path, map_location="cpu", weights_only=True)
    else:
        im = _load_rgb_pil(img_path)
        im = cc_transform(im)
        t = preprocess_transform(im)  # tensor, normalized, etc.
        _atomic_save_tensor(t, cache_path)

    _TENSOR_CACHE[k] = t
    if len(_TENSOR_CACHE) > _TENSOR_CACHE_MAX:
        _TENSOR_CACHE.popitem(last=False)
    return t


class CassavaTrainDataset(VisionDataset):
    """Minimal train dataset: (image, label) from train.csv"""

    def __init__(self, data_dir, df, transform=None):
        super().__init__(root=data_dir)
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.cc = v2.CenterCrop((400, 400))
        self._key_suffix = f"cc400|{type(transform).__name__}|vit_preprocess_v1"

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = row["image_id"]
        label = int(row["label"])
        img_path = os.path.join(self.root, filename)

        if self.transform is not None:
            img = _cached_preprocessed_tensor(
                img_path, self.cc, self.transform, self._key_suffix
            )
        else:
            img = self.cc(_load_rgb_pil(img_path))
        return img, label

    def __len__(self):
        return len(self.df)


class CassavaTestDataset(VisionDataset):
    """Custom dataset for the Cassava test data (ordered by sample submission)."""

    def __init__(self, data_dir, image_ids, transform=None, ttas=None):
        super().__init__(root=data_dir)
        self.transform = transform
        self.image_ids = list(image_ids)
        self.ttas = ttas
        self.cc = v2.CenterCrop((400, 400))
        self._key_suffix_base = f"cc400|{type(transform).__name__}|vit_preprocess_v1"

    def __getitem__(self, idx):
        filename = self.image_ids[idx]
        img_path = os.path.join(self.root, filename)

        if self.ttas is not None and self.transform is not None:
            im = self.cc(_load_rgb_pil(img_path))
            imgs = [self.transform(t(im)) for t in self.ttas]
            return imgs, filename
        elif self.transform is not None:
            img = _cached_preprocessed_tensor(
                img_path, self.cc, self.transform, self._key_suffix_base
            )
            return img, filename
        else:
            return self.cc(_load_rgb_pil(img_path)), filename

    def __len__(self):
        return len(self.image_ids)


def _warm_cache_for_dataset(ds: VisionDataset, max_items: int | None = None) -> None:
    if not hasattr(ds, "df"):
        return
    n = len(ds) if max_items is None else min(len(ds), max_items)
    for i in range(n):
        _ = ds[i][0]




## === cell 3
train_df = pd.read_csv(train_csv_path)

g = torch.Generator().manual_seed(3407)
labels = torch.from_numpy(train_df["label"].to_numpy())
idx_all = torch.arange(len(train_df))

val_mask = torch.zeros(len(train_df), dtype=torch.bool)
for lbl in torch.unique(labels).tolist():
    idxs = idx_all[labels == lbl]
    perm = idxs[torch.randperm(len(idxs), generator=g)]
    n_val = max(1, int(round(0.1 * len(idxs))))
    val_mask[perm[:n_val]] = True

val_size = int(0.1 * len(train_df))
val_idx = torch.nonzero(val_mask, as_tuple=False).squeeze(1)[:val_size].tolist()

train_df_val = train_df.iloc[val_idx].reset_index(drop=True)
train_df_tr = train_df.drop(index=val_idx).reset_index(drop=True)

train_dataset = CassavaTrainDataset(
    train_dir, df=train_df_tr, transform=train_transforms
)
val_dataset = CassavaTrainDataset(train_dir, df=train_df_val, transform=test_transforms)

_warm_cache_for_dataset(train_dataset)
_warm_cache_for_dataset(val_dataset)

loader_kwargs = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else None,
)

train_loader = DataLoader(
    train_dataset,
    shuffle=True,
    generator=g,  # keep shuffle deterministic across runs
    **{k: v for k, v in loader_kwargs.items() if v is not None},
)

val_loader = DataLoader(
    val_dataset,
    shuffle=False,
    **{k: v for k, v in loader_kwargs.items() if v is not None},
)

for p in model.parameters():
    p.requires_grad = False
for p in model.heads.head.parameters():
    p.requires_grad = True

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.heads.head.parameters(), lr=train_lr)


def _eval_acc(loader):
    model.eval()
    correct = 0
    total = 0
    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            preds = torch.argmax(logits, dim=1)
            correct += int((preds == labels).sum().item())
            total += int(labels.size(0))
    return correct / max(total, 1)


model.train()
for epoch in range(train_epochs):
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in train_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += float(loss.item()) * images.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += int((preds == labels).sum().item())
        total += int(labels.size(0))

    train_acc = correct / max(total, 1)
    val_acc = _eval_acc(val_loader)
    print(
        f"stage1 epoch {epoch+1}/{train_epochs} - loss: {running_loss/max(total,1):.4f} - train_acc: {train_acc:.4f} - val_acc: {val_acc:.4f}"
    )
    model.train()

for p in model.parameters():
    p.requires_grad = False
for p in model.heads.head.parameters():
    p.requires_grad = True

for p in model.encoder.layers[-1].parameters():
    p.requires_grad = True
for p in model.encoder.ln.parameters():
    p.requires_grad = True

finetune_params = [p for p in model.parameters() if p.requires_grad]
optimizer_ft = torch.optim.AdamW(finetune_params, lr=finetune_lr)

model.train()
for epoch in range(finetune_epochs):
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in train_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer_ft.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer_ft.step()

        running_loss += float(loss.item()) * images.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += int((preds == labels).sum().item())
        total += int(labels.size(0))

    train_acc = correct / max(total, 1)
    val_acc = _eval_acc(val_loader)
    print(
        f"stage2 epoch {epoch+1}/{finetune_epochs} - loss: {running_loss/max(total,1):.4f} - train_acc: {train_acc:.4f} - val_acc: {val_acc:.4f}"
    )
    model.train()

model.eval()




## === cell 4
sample_sub = pd.read_csv(sample_sub_path)
test_image_ids = sample_sub["image_id"].tolist()

test_dataset = CassavaTestDataset(
    test_dir, image_ids=test_image_ids, transform=test_transforms, ttas=ttas
)

test_loader_kwargs = {
    k: v for k, v in loader_kwargs.items() if v is not None and k != "batch_size"
}


def _tta_collate(batch):
    imgs_list, names = zip(*batch)  # each element: list length T of tensors [C,H,W]
    B = len(imgs_list)
    T = len(imgs_list[0])
    per_t = [torch.stack([imgs_list[b][t] for b in range(B)], dim=0) for t in range(T)]
    return per_t, list(names)


test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=_tta_collate if tta else None,
    **test_loader_kwargs,
)

normalizer = torch.nn.Softmax(dim=1)




## === cell 5
all_names = []
all_preds = []

model.eval()
with torch.inference_mode():
    for batch_idx, (inputs, filenames) in enumerate(test_loader):
        filenames = list(filenames)

        if tta:
            tta_tensor = torch.stack(inputs, dim=0)  # [T,B,C,H,W]
            T, B = tta_tensor.shape[0], tta_tensor.shape[1]
            flat = tta_tensor.reshape(T * B, *tta_tensor.shape[2:]).to(
                device, non_blocking=True
            )

            preds = normalizer(model(flat))  # [T*B, num_classes]
            preds = preds.reshape(T, B, -1).mean(dim=0)  # [B, num_classes]
            pred_labels = torch.argmax(preds, dim=1).tolist()
        else:
            inputs = inputs.to(device, non_blocking=True)
            preds = normalizer(model(inputs))
            pred_labels = torch.argmax(preds, dim=1).tolist()

        all_names.extend(filenames)
        all_preds.extend(pred_labels)




## === cell 6
pred_df = pd.DataFrame({"image_id": all_names, "label": all_preds})

pred_df = pred_df.drop_duplicates(subset=["image_id"], keep="first")

submission = sample_sub[["image_id"]].merge(pred_df, on="image_id", how="left")

submission["label"] = submission["label"].fillna(0).astype(int)

assert submission.shape[0] == sample_sub.shape[0], (submission.shape, sample_sub.shape)
assert submission["label"].between(0, num_classes - 1).all()

submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)




## === cell 7
submission.head()
