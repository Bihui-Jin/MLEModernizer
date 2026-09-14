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

import pandas as pd
import torch
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision.transforms import v2

torch.set_num_threads(max(1, min(4, (os.cpu_count() or 4))))

torch.manual_seed(3407)
torch.cuda.manual_seed(3407)

cudnn.deterministic = True
cudnn.benchmark = True  # faster deterministic kernel selection

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

try:
    torch.multiprocessing.set_start_method("fork", force=True)
except RuntimeError:
    pass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

test_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images/"
train_dir = "/kaggle/input/cassava-leaf-disease-classification/train_images/"
train_csv_path = "/kaggle/input/cassava-leaf-disease-classification/train.csv"

train_tfrec_dir = "/kaggle/input/cassava-leaf-disease-classification/train_tfrecords/"
test_tfrec_dir = "/kaggle/input/cassava-leaf-disease-classification/test_tfrecords/"

img_size = 384
batch_size = 16
num_workers = min(8, (os.cpu_count() or 4))
num_classes = 5
tta = True

finetune_head = True
finetune_epochs = 3
finetune_lr = 3e-4

from torchvision.models import vit_b_16, ViT_B_16_Weights

weights = ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1  # image_size=384
model = vit_b_16(weights=weights)
model.heads.head = torch.nn.Linear(model.heads.head.in_features, num_classes)

ckpt_candidates = [
    "./model.pth",
    "./best.pth",
    "./checkpoint.pth",
    "/kaggle/working/model.pth",
    "/kaggle/working/best.pth",
    "/kaggle/working/checkpoint.pth",
]
loaded_any_ckpt = False
for p in ckpt_candidates:
    if os.path.isfile(p):
        state = torch.load(p, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if isinstance(state, dict):
            new_state = {}
            for k, v in state.items():
                nk = k.replace("module.", "")
                new_state[nk] = v
            missing, unexpected = model.load_state_dict(new_state, strict=False)
            print(f"Loaded checkpoint: {p}")
            print(f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
            loaded_any_ckpt = True
        break

model.to(device)

if device.type == "cuda":
    model = model.to(memory_format=torch.channels_last)

try:
    if hasattr(torch, "compile"):
        model = torch.compile(
            model, mode="reduce-overhead", fullgraph=False, dynamic=False
        )
        print("torch.compile enabled")
except Exception as e:
    print("torch.compile not available, continuing eager. Reason:", repr(e))




## === cell 1
import glob
from typing import List, Optional

from PIL import Image
from torch.utils.data import Dataset

weights_tfms = weights.transforms()


def _seed_worker(worker_id: int):
    base_seed = 3407
    seed = (base_seed + worker_id) % (2**32 - 1)
    torch.manual_seed(seed)


def _list_image_files(root_dir: str) -> List[str]:
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    files = []
    for r, _, fnames in os.walk(root_dir):
        for f in fnames:
            if f.lower().endswith(exts):
                files.append(os.path.join(r, f))
    files = sorted(files)
    if len(files) == 0:
        raise RuntimeError(f"No image files found under: {root_dir}")
    return files


class CassavaImageDataset(Dataset):
    """
    Map-style dataset over image files.
    If labeled=True yields (img_tensor, label_int), else yields (img_tensor, image_id_str).
    """

    def __init__(
        self,
        image_paths: List[str],
        labels: Optional[List[int]] = None,
        transform=None,
    ):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.labeled = labels is not None
        if self.labeled:
            if len(self.image_paths) != len(self.labels):
                raise ValueError("image_paths and labels must have same length")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        p = self.image_paths[idx]
        img = Image.open(p).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        image_id = os.path.basename(p)
        if self.labeled:
            return img, int(self.labels[idx])
        return img, image_id




## === cell 2
if finetune_head and (not loaded_any_ckpt):
    train_df = pd.read_csv(train_csv_path)
    train_paths = [os.path.join(train_dir, x) for x in train_df["image_id"].tolist()]
    train_labels = train_df["label"].astype(int).tolist()

    train_dataset = CassavaImageDataset(
        train_paths,
        labels=train_labels,
        transform=weights_tfms,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # map-style dataset; OK to shuffle here
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=(4 if num_workers > 0 else None),
        drop_last=False,
        worker_init_fn=_seed_worker if num_workers > 0 else None,
    )

    for p in model.parameters():
        p.requires_grad = False
    for p in model.heads.head.parameters():
        p.requires_grad = True

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.heads.head.parameters(), lr=finetune_lr)

    model.train()
    for epoch in range(finetune_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            if device.type == "cuda":
                xb = xb.contiguous(memory_format=torch.channels_last)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            bs = xb.size(0)
            running_loss += float(loss.item()) * bs
            preds = torch.argmax(logits, dim=1)
            correct += int((preds == yb).sum().item())
            total += int(bs)

        print(
            f"[finetune head] epoch {epoch+1}/{finetune_epochs} "
            f"loss={running_loss/max(total,1):.4f} acc={correct/max(total,1):.4f}"
        )

    torch.save(model.state_dict(), "/kaggle/working/model.pth")
    model.eval()




## === cell 3
if tta:
    ttas = [
        v2.RandomRotation(180),
        v2.RandomVerticalFlip(1),
        v2.RandomPerspective(p=1),
    ]
else:
    ttas = None

test_paths = _list_image_files(test_dir)
test_dataset = CassavaImageDataset(
    test_paths,
    labels=None,
    transform=weights_tfms,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else None,
    worker_init_fn=_seed_worker if num_workers > 0 else None,
)

_softmax_dim1 = lambda x: torch.softmax(x, dim=1)




## === cell 4
all_names = []
all_preds = []

model.eval()


def _seed_for_batch_view(batch_idx: int, view_j: int) -> int:
    return int((3407 + batch_idx * 1000003 + view_j * 9176) % (2**31 - 1))


_cpu_g = torch.Generator(device="cpu")
_cuda_g = torch.Generator(device="cuda") if device.type == "cuda" else None

with torch.inference_mode():
    for batch_idx, (inputs, filenames) in enumerate(test_loader):
        xb = inputs.to(device, non_blocking=True)  # (B,C,H,W)
        if device.type == "cuda":
            xb = xb.contiguous(memory_format=torch.channels_last)

        if tta:
            bsz, c, h, w = xb.shape
            n_tta = len(ttas)
            x_all = xb.new_empty((bsz * (n_tta + 1), c, h, w))
            x_all[0:bsz].copy_(xb)

            for j, t in enumerate(ttas):
                seed = _seed_for_batch_view(batch_idx, j)
                _cpu_g.manual_seed(seed)
                if _cuda_g is not None:
                    _cuda_g.manual_seed(seed)

                try:
                    x_aug = t(
                        xb, generator=_cuda_g if device.type == "cuda" else _cpu_g
                    )
                except TypeError:
                    torch.manual_seed(seed)
                    if device.type == "cuda":
                        torch.cuda.manual_seed(seed)
                    x_aug = t(xb)

                if device.type == "cuda":
                    x_aug = x_aug.contiguous(memory_format=torch.channels_last)
                start = (j + 1) * bsz
                x_all[start : start + bsz].copy_(x_aug)

            preds_flat = _softmax_dim1(model(x_all))  # (B*(T+1), num_classes)
            preds_bt = preds_flat.view(n_tta + 1, bsz, -1).transpose(0, 1)
            mean_preds = preds_bt.mean(dim=1)  # (B, num_classes)
            pred_labels = torch.argmax(mean_preds, dim=1).tolist()
        else:
            preds = _softmax_dim1(model(xb))
            pred_labels = torch.argmax(preds, dim=1).tolist()

        all_names.extend(list(filenames))
        all_preds.extend(pred_labels)

print("Preds generated:", len(all_preds), "Unique files:", len(set(all_names)))




## === cell 5
sample_path = "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
sample = pd.read_csv(sample_path)

pred_ser = pd.Series(all_preds, index=pd.Index(all_names, name="image_id"))
pred_ser = pred_ser[~pred_ser.index.duplicated(keep="first")]

mapped = sample["image_id"].map(pred_ser)
sample["label"] = mapped.fillna(0).astype(int)

my_submission = sample[["image_id", "label"]]
my_submission.to_csv("submission.csv", index=False)

print("Saved submission.csv with shape:", my_submission.shape)
print("Any missing predictions filled with 0:", int(mapped.isna().sum()))




## === cell 6
my_submission.head()
