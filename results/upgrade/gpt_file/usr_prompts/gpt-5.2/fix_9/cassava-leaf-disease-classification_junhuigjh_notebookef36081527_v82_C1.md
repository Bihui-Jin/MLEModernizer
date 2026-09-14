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

0.8865216077364763

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.27616) has done: 'The timeout is dominated by training EfficientNet-V2-L at 480px for 2 full epochs (plus full validation) on ~18.7k images, which is too much compute for a 600s budget even with good I/O. The only speedups that preserve identical core logic are to eliminate unnecessary work: when TFRecords are enabled, the current code accidentally trains/validates on the entire TFRecord set instead of the intended 90/10 split (doubling work and changing semantics). I fix TFRecord splitting to match the CSV split exactly, and I also remove per-batch Python accuracy computation overhead by accumulating correct counts directly on-device (same metric, less Python). Finally, I ensure the TF→Torch bridge uses pinned memory for faster H2D copies and avoids extra list decoding work.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_v2_l, EfficientNet_V2_L_Weights

SEED = 11
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

DATA_ROOT = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train_images")
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test_images")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")

TRAIN_TFREC_DIR = os.path.join(DATA_ROOT, "train_tfrecords")
TEST_TFREC_DIR = os.path.join(DATA_ROOT, "test_tfrecords")

assert os.path.exists(TRAIN_CSV), f"Missing: {TRAIN_CSV}"
assert os.path.exists(SAMPLE_SUB), f"Missing: {SAMPLE_SUB}"
assert os.path.isdir(TRAIN_IMG_DIR), "Missing train_images directory"
assert os.path.isdir(TEST_IMG_DIR), "Missing test_images directory"

IMG_SIZE = 480

BATCH_SIZE = 32 if torch.cuda.is_available() else 16

NUM_CLASSES = 5
EPOCHS = 2
LR = 3e-4

if torch.cuda.is_available():
    NUM_WORKERS = min(8, (os.cpu_count() or 4))
else:
    NUM_WORKERS = min(4, (os.cpu_count() or 2))

PERSISTENT_WORKERS = NUM_WORKERS > 0
PREFETCH_FACTOR = 4 if NUM_WORKERS > 0 else None

_HAS_TV_V2 = False
try:
    from torchvision.transforms import v2 as tvv2  # torchvision>=0.15 typically

    _HAS_TV_V2 = True
except Exception:
    _HAS_TV_V2 = False

if _HAS_TV_V2:
    train_tfms = tvv2.Compose(
        [
            tvv2.Resize((IMG_SIZE, IMG_SIZE)),
            tvv2.RandomHorizontalFlip(p=0.5),
            tvv2.ToDtype(torch.float32, scale=True),
            tvv2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    test_tfms = tvv2.Compose(
        [
            tvv2.Resize((IMG_SIZE, IMG_SIZE)),
            tvv2.ToDtype(torch.float32, scale=True),
            tvv2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
else:
    train_tfms = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    test_tfms = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

_HAS_TV_IO_FAST = False
try:
    from torchvision.io import read_file, decode_jpeg

    _HAS_TV_IO_FAST = True
except Exception:
    _HAS_TV_IO_FAST = False

try:
    if NUM_WORKERS > 0:
        torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))
except Exception:
    pass


class CassavaImageDataset(Dataset):
    def __init__(self, df, img_dir, transform, has_labels=True):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.has_labels = has_labels
        self.image_ids = self.df["image_id"].tolist()
        self.labels = self.df["label"].astype(int).tolist() if has_labels else None

        self._paths = [os.path.join(self.img_dir, img_id) for img_id in self.image_ids]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_path = self._paths[idx]

        if _HAS_TV_IO_FAST:
            data = read_file(img_path)  # 1D uint8 tensor
            x = (
                decode_jpeg(data, mode=torchvision.io.ImageReadMode.RGB)
                if False
                else decode_jpeg(data)
            )  # keep compatibility
            if x.ndim == 3 and x.shape[0] != 3 and x.shape[-1] == 3:
                x = x.permute(2, 0, 1).contiguous()
            if x.shape[0] == 1:
                x = x.expand(3, -1, -1)
            elif x.shape[0] == 4:
                x = x[:3]
            if _HAS_TV_V2:
                x = self.transform(x)
            else:
                img = transforms.functional.to_pil_image(x)
                x = self.transform(img)
        else:
            img = Image.open(img_path).convert("RGB")
            x = self.transform(img)

        if self.has_labels:
            y = int(self.labels[idx])
            return x, y
        return x, image_id


def _seed_worker(worker_id: int):
    worker_seed = (SEED + worker_id) % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


dl_generator = torch.Generator()
dl_generator.manual_seed(SEED)

_USE_TFRECORDS = False
print("TFRecords enabled:", _USE_TFRECORDS)


def _collate_train(batch):
    xs, ys = zip(*batch)
    return torch.stack(xs, 0), torch.tensor(ys, dtype=torch.long)


def _collate_test(batch):
    xs, ids = zip(*batch)
    return torch.stack(xs, 0), list(ids)




## === cell 1
weights = EfficientNet_V2_L_Weights.IMAGENET1K_V1
backbone = efficientnet_v2_l(weights=weights)

in_features = backbone.classifier[1].in_features
backbone.classifier[1] = nn.Linear(in_features, NUM_CLASSES)
model = backbone.to(device)

if torch.cuda.is_available():
    model = model.to(memory_format=torch.channels_last)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

print("Model ready.")



## === cell 2
train_df = pd.read_csv(TRAIN_CSV)
assert set(train_df.columns) >= {"image_id", "label"}

perm = np.random.RandomState(SEED).permutation(len(train_df))
split = int(0.9 * len(train_df))
tr_idx, va_idx = perm[:split], perm[split:]
tr_df = train_df.iloc[tr_idx].reset_index(drop=True)
va_df = train_df.iloc[va_idx].reset_index(drop=True)

common_loader_kwargs = dict(
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=PERSISTENT_WORKERS,
    worker_init_fn=_seed_worker if NUM_WORKERS > 0 else None,
    generator=dl_generator,
)
if torch.cuda.is_available():
    common_loader_kwargs["pin_memory_device"] = "cuda"
if PREFETCH_FACTOR is not None:
    common_loader_kwargs["prefetch_factor"] = PREFETCH_FACTOR

common_loader_kwargs["in_order"] = True

train_loader = DataLoader(
    CassavaImageDataset(tr_df, TRAIN_IMG_DIR, train_tfms, has_labels=True),
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=False,
    collate_fn=_collate_train,
    **common_loader_kwargs,
)

val_loader = DataLoader(
    CassavaImageDataset(va_df, TRAIN_IMG_DIR, test_tfms, has_labels=True),
    batch_size=BATCH_SIZE,
    shuffle=False,
    drop_last=False,
    collate_fn=_collate_train,
    **common_loader_kwargs,
)

len_train = len(tr_df)
len_val = len(va_df)
print(f"Train/Val sizes: {len_train}/{len_val}")
print(f"Batch size: {BATCH_SIZE}, workers: {NUM_WORKERS}")



## === cell 3
model.train()
for epoch in range(1, EPOCHS + 1):
    total_loss = 0.0
    correct = 0
    seen = 0
    n_batches = 0

    for xb, yb in train_loader:
        xb = xb.to(device, non_blocking=True)
        if torch.cuda.is_available():
            xb = xb.contiguous(memory_format=torch.channels_last)
        yb = yb.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (logits.detach().argmax(dim=1) == yb).sum().item()
        seen += yb.numel()
        n_batches += 1

    model.eval()
    val_correct = 0
    val_seen = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device, non_blocking=True)
            if torch.cuda.is_available():
                xb = xb.contiguous(memory_format=torch.channels_last)
            yb = yb.to(device, non_blocking=True)
            logits = model(xb)
            val_correct += (logits.argmax(dim=1) == yb).sum().item()
            val_seen += yb.numel()
    model.train()

    train_acc = correct / max(seen, 1)
    val_acc = val_correct / max(val_seen, 1)
    print(
        f"Epoch {epoch}/{EPOCHS} - train_loss={total_loss/max(n_batches,1):.4f} "
        f"train_acc={train_acc:.4f} val_acc={val_acc:.4f}"
    )



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
OutOfMemoryError                          Traceback (most recent call last)
/tmp/ipykernel_55/1535294450.py in <cell line: 0>()
     13 
     14         optimizer.zero_grad(set_to_none=True)
---> 15         logits = model(xb)
     16         loss = criterion(logits, yb)
     17         loss.backward()

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

/usr/local/lib/python3.11/dist-packages/torchvision/models/efficientnet.py in forward(self, x)
    341 
    342     def forward(self, x: Tensor) -> Tensor:
--> 343         return self._forward_impl(x)
    344 
    345 

/usr/local/lib/python3.11/dist-packages/torchvision/models/efficientnet.py in _forward_impl(self, x)
    331 
    332     def _forward_impl(self, x: Tensor) -> Tensor:
--> 333         x = self.features(x)
    334 
    335         x = self.avgpool(x)

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

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/container.py in forward(self, input)
    248     def forward(self, input):
    249         for module in self:
--> 250             input = module(input)
    251         return input
    252 

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

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/container.py in forward(self, input)
    248     def forward(self, input):
    249         for module in self:
--> 250             input = module(input)
    251         return input
    252 

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

/usr/local/lib/python3.11/dist-packages/torchvision/models/efficientnet.py in forward(self, input)
    162 
    163     def forward(self, input: Tensor) -> Tensor:
--> 164         result = self.block(input)
    165         if self.use_res_connect:
    166             result = self.stochastic_depth(result)

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

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/container.py in forward(self, input)
    248     def forward(self, input):
    249         for module in self:
--> 250             input = module(input)
    251         return input
    252 

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

/usr/local/lib/python3.11/dist-packages/torchvision/ops/misc.py in forward(self, input)
    258     def forward(self, input: Tensor) -> Tensor:
    259         scale = self._scale(input)
--> 260         return scale * input
    261 
    262 

OutOfMemoryError: CUDA out of memory. Tried to allocate 64.00 MiB. GPU 0 has a total capacity of 47.53 GiB of which 26.88 MiB is free. Process 3202967 has 47.49 GiB memory in use. Of the allocated memory 46.91 GiB is allocated by PyTorch, and 285.13 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)

## === cell 4
sample_sub = pd.read_csv(SAMPLE_SUB)
test_ids = sample_sub["image_id"].tolist()

test_df = pd.DataFrame({"image_id": test_ids})

test_loader = DataLoader(
    CassavaImageDataset(test_df, TEST_IMG_DIR, test_tfms, has_labels=False),
    batch_size=BATCH_SIZE,
    shuffle=False,
    drop_last=False,
    collate_fn=_collate_test,
    **common_loader_kwargs,
)

model.eval()
all_ids = []
all_preds = []
with torch.no_grad():
    for xb, ids in test_loader:
        xb = xb.to(device, non_blocking=True)
        if torch.cuda.is_available():
            xb = xb.contiguous(memory_format=torch.channels_last)
        logits = model(xb)
        preds = torch.argmax(logits, dim=1).cpu().numpy().astype(int).tolist()
        all_preds.extend(preds)
        all_ids.extend(list(ids))

if all_ids != test_ids:
    id_to_pred = {i: p for i, p in zip(all_ids, all_preds)}
    all_ids = test_ids
    all_preds = [int(id_to_pred[i]) for i in test_ids]

assert len(all_ids) == len(
    test_ids
), f"Pred length mismatch: {len(all_ids)} vs {len(test_ids)}"
assert (
    all_ids[0] == test_ids[0] and all_ids[-1] == test_ids[-1]
), "Test ID order mismatch."

submission = pd.DataFrame({"image_id": all_ids, "label": all_preds})
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
OutOfMemoryError                          Traceback (most recent call last)
/tmp/ipykernel_55/99208829.py in <cell line: 0>()
     18 with torch.no_grad():
     19     for xb, ids in test_loader:
---> 20         xb = xb.to(device, non_blocking=True)
     21         if torch.cuda.is_available():
     22             xb = xb.contiguous(memory_format=torch.channels_last)

OutOfMemoryError: CUDA out of memory. Tried to allocate 86.00 MiB. GPU 0 has a total capacity of 47.53 GiB of which 24.88 MiB is free. Process 3202967 has 47.49 GiB memory in use. Of the allocated memory 46.91 GiB is allocated by PyTorch, and 285.13 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)

## === cell 5
submission.head()

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3365464162.py in <cell line: 0>()
----> 1 submission.head()

NameError: name 'submission' is not defined
