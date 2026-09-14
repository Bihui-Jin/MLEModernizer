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
from PIL import Image
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
from torchvision.transforms import InterpolationMode, v2



## === cell 1
torch.manual_seed(3407)
torch.cuda.manual_seed(3407)

cudnn.deterministic = True
cudnn.benchmark = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

train_dir = "/kaggle/input/cassava-leaf-disease-classification/train_images/"
test_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images/"

img_size = 384
batch_size = 16
num_workers = 4
num_classes = 5
tta = True

ckpt_path = "/kaggle/input/vit-v3/vit_v3.pt"
model = None
if os.path.exists(ckpt_path):
    model = torch.load(ckpt_path, map_location=device)
else:
    from torchvision.models import vit_b_16

    model = vit_b_16(weights=None, image_size=img_size)
    if hasattr(model, "heads") and hasattr(model.heads, "head"):
        in_features = model.heads.head.in_features
        model.heads.head = torch.nn.Linear(in_features, num_classes)

model = model.to(device)



## === cell 2
import hashlib
import torchvision
from torchvision.io import ImageReadMode


class CassavaDataset(VisionDataset):
    """Custom dataset for the Cassava data.

    Speed fix (correctness-preserving):
    - Replace per-Dataset in-memory cache (which doesn't help with multi-worker DataLoader because each
      worker has its own process/memory) with an on-disk cache of the deterministic "decode + center-crop"
      tensor. This avoids repeatedly decoding JPEGs across epochs/validation/test and across workers.
    - Cached tensors are stored as uint8 CHW, exactly matching torchvision.io.read_image output.
    """

    def __init__(
        self,
        data_dir,
        transform=None,
        ttas=None,
        img_size=384,
        filenames=None,
        labels_df=None,
        cache_dir="/kaggle/working/_cc_cache",
        cc_size=(600, 600),
    ):
        super().__init__(root=data_dir)
        self.transform = transform
        self.ttas = ttas

        if filenames is None:
            candidates = os.listdir(data_dir)
            self.images = sorted(
                [
                    f
                    for f in candidates
                    if os.path.isfile(os.path.join(data_dir, f))
                    and f.lower().endswith((".jpg", ".jpeg", ".png"))
                ]
            )
        else:
            self.images = list(filenames)

        self.labels_df = labels_df
        if self.labels_df is not None:
            self.label_map = dict(
                zip(
                    self.labels_df["image_id"].astype(str).tolist(),
                    self.labels_df["label"].astype(int).tolist(),
                )
            )
        else:
            self.label_map = None

        self._cc_size = tuple(cc_size)
        self._cache_dir = cache_dir
        os.makedirs(self._cache_dir, exist_ok=True)

    def _cache_path(self, filename: str) -> str:
        h = hashlib.md5(filename.encode("utf-8")).hexdigest()
        th, tw = self._cc_size
        return os.path.join(self._cache_dir, f"{h}_{th}x{tw}.pt")

    def _decode_and_center_crop_tensor(self, path: str):
        img = torchvision.io.read_image(path, mode=ImageReadMode.RGB)  # uint8 CHW
        h, w = int(img.shape[1]), int(img.shape[2])
        th, tw = self._cc_size
        i = max(0, int((h - th) // 2))
        j = max(0, int((w - tw) // 2))
        img = img[:, i : i + th, j : j + tw]
        return img

    def _get_center_cropped(self, filename: str, path: str):
        cpath = self._cache_path(filename)
        if os.path.exists(cpath):
            return torch.load(cpath, map_location="cpu")
        img = self._decode_and_center_crop_tensor(path)
        tmp = cpath + f".tmp_{os.getpid()}"
        torch.save(img, tmp)
        os.replace(tmp, cpath)
        return img

    def __getitem__(self, idx):
        filename = self.images[idx]
        path = os.path.join(self.root, filename)

        img = self._get_center_cropped(filename, path)

        if self.ttas is not None and self.transform is not None:
            img_out = [self.transform(t(img)) for t in self.ttas]
        elif self.transform:
            img_out = self.transform(img)
        else:
            img_out = img

        if self.label_map is None:
            return img_out, filename

        label = self.label_map[filename]
        return img_out, label

    def __len__(self):
        return len(self.images)




## === cell 3
train_transforms = v2.Compose(
    [
        v2.Resize((img_size, img_size), interpolation=InterpolationMode.BICUBIC),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

test_transforms = v2.Compose(
    [
        v2.Resize((img_size, img_size), interpolation=InterpolationMode.BICUBIC),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

if tta:
    ttas = [
        v2.RandomRotation(180),
        v2.RandomVerticalFlip(1),
        v2.RandomPerspective(p=1),
    ]
else:
    ttas = None



## === cell 4
train_csv_path = "/kaggle/input/cassava-leaf-disease-classification/train.csv"
train_df = pd.read_csv(train_csv_path)

paths = (train_dir.rstrip("/") + "/" + train_df["image_id"].astype(str)).tolist()
exists_mask = [os.path.isfile(p) for p in paths]
train_df = train_df.loc[exists_mask].reset_index(drop=True)

g = torch.Generator()
g.manual_seed(3407)
perm = torch.randperm(len(train_df), generator=g).tolist()
val_frac = 0.1
val_size = int(len(train_df) * val_frac)
val_idx = set(perm[:val_size])

image_ids = train_df["image_id"].astype(str).tolist()
train_filenames = [image_ids[i] for i in range(len(image_ids)) if i not in val_idx]
val_filenames = [image_ids[i] for i in range(len(image_ids)) if i in val_idx]

train_dataset = CassavaDataset(
    train_dir,
    transform=train_transforms,
    ttas=None,
    filenames=train_filenames,
    labels_df=train_df,
    cache_dir="/kaggle/working/_cc_cache_train",
)
val_dataset = CassavaDataset(
    train_dir,
    transform=test_transforms,
    ttas=None,
    filenames=val_filenames,
    labels_df=train_df,
    cache_dir="/kaggle/working/_cc_cache_train",
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else None,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else None,
)

need_train = not os.path.exists(ckpt_path)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)


def evaluate_acc(m, loader):
    m.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = m(x)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.numel()
    return correct / max(1, total)


if need_train:
    epochs = 2
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * y.size(0)

        val_acc = evaluate_acc(model, val_loader)
        print(
            f"epoch {epoch}/{epochs} - train_loss: {running_loss/len(train_dataset):.4f} - val_acc: {val_acc:.4f}"
        )
else:
    print("Loaded external checkpoint; skipping training.")



## === cell 5
sample_path = "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
sample_sub = pd.read_csv(sample_path)
test_filenames = sample_sub["image_id"].astype(str).tolist()

test_dataset = CassavaDataset(
    test_dir,
    transform=test_transforms,
    ttas=ttas,
    filenames=test_filenames,
    cache_dir="/kaggle/working/_cc_cache_test",
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else None,
)

normalizer = torch.nn.Softmax(dim=1)



## === cell 6
all_names = []
all_preds = []

model.eval()

with torch.no_grad():
    for batch_idx, (inputs, filenames) in enumerate(test_loader):
        filenames = list(filenames)

        if tta:
            inputs_cat = torch.cat(inputs, dim=0).to(device, non_blocking=True)
            B = len(filenames)
            probs = normalizer(model(inputs_cat))  # [T*B, C]
            T = probs.shape[0] // B
            mean_probs = probs.view(T, B, -1).mean(dim=0)  # [B, C]
            pred_labels = torch.argmax(mean_probs, 1).tolist()
        else:
            inputs = inputs.to(device, non_blocking=True)
            probs = normalizer(model(inputs))
            pred_labels = torch.argmax(probs, 1).tolist()

        all_names.extend(filenames)
        all_preds.extend(pred_labels)



## === cell 7
pred_df = pd.DataFrame({"image_id": all_names, "label": all_preds})

pred_df = pred_df.drop_duplicates(subset=["image_id"], keep="first")

my_submission = sample_sub[["image_id"]].merge(pred_df, on="image_id", how="left")

if my_submission["label"].isna().any():
    fill_value = int(pred_df["label"].mode().iloc[0]) if len(pred_df) else 0
    my_submission["label"] = my_submission["label"].fillna(fill_value).astype(int)
else:
    my_submission["label"] = my_submission["label"].astype(int)

my_submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", my_submission.shape)



## === cell 8
my_submission.head()
