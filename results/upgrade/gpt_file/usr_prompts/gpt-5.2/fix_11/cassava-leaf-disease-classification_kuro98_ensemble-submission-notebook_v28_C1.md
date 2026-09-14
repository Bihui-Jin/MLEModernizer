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
from torchvision.datasets import VisionDataset
from torchvision.transforms import InterpolationMode, v2

torch.manual_seed(3407)
torch.cuda.manual_seed(3407)

cudnn.deterministic = False
cudnn.benchmark = True

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

torch.set_float32_matmul_precision("high")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

data_root = "/kaggle/input/cassava-leaf-disease-classification/"
train_csv_path = os.path.join(data_root, "train.csv")
train_dir = os.path.join(data_root, "train_images/")
test_dir = os.path.join(data_root, "test_images/")
sample_path = os.path.join(data_root, "sample_submission.csv")

model_b_img_size = 384
model_a_img_size = 384
batch_size = 16

_cpu = os.cpu_count() or 4
num_workers = min(4, _cpu)  # was min(8, ...)
num_classes = 5
tta = False

train_if_missing = True
train_epochs = 3
train_lr = 3e-4

from torchvision.models import vit_b_16, ViT_B_16_Weights


def _build_vit(num_classes: int):
    m = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1)
    if hasattr(m, "heads") and hasattr(m.heads, "head"):
        in_f = m.heads.head.in_features
        m.heads.head = torch.nn.Linear(in_f, num_classes)
    return m


def _load_or_build_vit(path: str, num_classes: int):
    if os.path.exists(path):
        obj = torch.load(path, map_location=device)
        if isinstance(obj, dict) and all(isinstance(k, str) for k in obj.keys()):
            m = _build_vit(num_classes)
            m.load_state_dict(obj, strict=False)
            return m.to(device)
        return obj.to(device)
    m = _build_vit(num_classes)
    return m.to(device)


model_a_ckpt = "/kaggle/input/vit-v1/vit_v1.pt"
model_b_ckpt = "/kaggle/input/vit-boosted/vit_boosted.pt"

model_a = _load_or_build_vit(model_a_ckpt, num_classes)
model_b = _load_or_build_vit(model_b_ckpt, num_classes)

linear_head_path = "/kaggle/input/linear-head/linear_cls.pt"
linear_head = (
    torch.load(linear_head_path, map_location=device)
    if os.path.exists(linear_head_path)
    else torch.nn.Identity()
)


def _maybe_compile(m: torch.nn.Module):
    if hasattr(torch, "compile"):
        try:
            return torch.compile(m, mode="reduce-overhead", fullgraph=False)
        except Exception:
            return m
    return m




## === cell 1
import torchvision

try:
    torchvision.set_image_backend("accimage")
except Exception:
    pass


class CassavaDataset(VisionDataset):
    """Custom dataset for Cassava images.

    For test-time: pass explicit image_ids to match sample_submission.csv order.
    For train-time: pass image_ids and labels from train.csv.
    """

    def __init__(
        self,
        data_dir,
        model_a_size,
        model_b_size,
        transform=None,
        ttas=None,
        image_ids=None,
        labels=None,
    ):
        super().__init__(root=data_dir)

        self.transform = transform
        self.images = (
            list(image_ids) if image_ids is not None else sorted(os.listdir(data_dir))
        )
        self.labels = labels  # list/array aligned with self.images or None for test
        self.ttas = ttas
        self.model_a_size = int(model_a_size)
        self.model_b_size = int(model_b_size)
        self.same_size = self.model_a_size == self.model_b_size

    def __getitem__(self, idx):
        filename = self.images[idx]
        path = os.path.join(self.root, filename)

        img = torchvision.io.read_image(path, mode=torchvision.io.ImageReadMode.RGB)

        if self.labels is None:
            return img, filename
        else:
            return img, int(self.labels[idx])

    def __len__(self):
        return len(self.images)




## === cell 2
from torchvision.transforms.v2 import functional as F

MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)


def _fast_norm_transform(x: torch.Tensor) -> torch.Tensor:
    x = x.to(dtype=torch.float32).mul_(1.0 / 255.0)
    x = x.sub_(MEAN.view(3, 1, 1)).div_(STD.view(3, 1, 1))
    return x


def _preprocess_batch(imgs_u8: torch.Tensor, out_size: int) -> torch.Tensor:
    x = imgs_u8
    x = F.center_crop(x, output_size=[600, 600])
    x = F.resize(
        x,
        size=[out_size, out_size],
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    )
    x = x.to(dtype=torch.float32).mul_(1.0 / 255.0)
    x.sub_(MEAN).div_(STD)
    return x


def _collate_test_fast(batch):
    imgs, names = zip(*batch)
    imgs = torch.stack(imgs, 0)  # uint8
    a = _preprocess_batch(imgs, model_a_img_size)
    if model_a_img_size == model_b_img_size:
        b = a
    else:
        b = _preprocess_batch(imgs, model_b_img_size)
    return a, b, list(names)


def _collate_train_fast(batch):
    imgs, y = zip(*batch)
    imgs = torch.stack(imgs, 0)  # uint8
    a = _preprocess_batch(imgs, model_a_img_size)
    if model_a_img_size == model_b_img_size:
        b = a
    else:
        b = _preprocess_batch(imgs, model_b_img_size)
    return a, b, torch.as_tensor(y, dtype=torch.long)


test_transforms = None if not tta else _fast_norm_transform
train_transforms = None

if tta:
    ttas = [
        v2.RandomRotation(180),
        v2.RandomVerticalFlip(1),
        v2.RandomPerspective(p=1),
    ]
else:
    ttas = None

sample_sub = pd.read_csv(sample_path)
test_image_ids = sample_sub["image_id"].tolist()

test_dataset = CassavaDataset(
    test_dir,
    model_a_img_size,
    model_b_img_size,
    transform=test_transforms,
    ttas=ttas,
    image_ids=test_image_ids,
)

_pin = torch.cuda.is_available()


def _seed_worker(worker_id: int):
    seed = 3407 + worker_id
    torch.manual_seed(seed)


_loader_gen = torch.Generator()
_loader_gen.manual_seed(3407)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,  # critical for deterministic, aligned ordering
    num_workers=num_workers,
    pin_memory=_pin,
    pin_memory_device="cuda" if _pin else "",
    persistent_workers=(num_workers > 0),
    prefetch_factor=(4 if num_workers > 0 else None),
    collate_fn=None if tta else _collate_test_fast,
    worker_init_fn=_seed_worker if num_workers > 0 else None,
    generator=_loader_gen,
)

normalizer = torch.nn.Softmax(dim=1)




## === cell 3
def _looks_like_untrained_head(m: torch.nn.Module) -> bool:
    return True


need_train = train_if_missing and (
    not os.path.exists(model_a_ckpt) or not os.path.exists(model_b_ckpt)
)
if need_train:
    train_df = pd.read_csv(train_csv_path)
    train_image_ids = train_df["image_id"].tolist()
    train_labels = train_df["label"].tolist()

    train_dataset = CassavaDataset(
        train_dir,
        model_a_img_size,
        model_b_img_size,
        transform=train_transforms,
        ttas=None,
        image_ids=train_image_ids,
        labels=train_labels,
    )

    _pin = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=_pin,
        pin_memory_device="cuda" if _pin else "",
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
        collate_fn=_collate_train_fast,
        worker_init_fn=_seed_worker if num_workers > 0 else None,
        generator=_loader_gen,
    )

    def _train_one_model(model: torch.nn.Module, which: str):
        model.train()
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=train_lr)

        model_local = _maybe_compile(model)

        for epoch in range(train_epochs):
            running_loss = 0.0
            n = 0
            for model_a_x, model_b_x, y in train_loader:
                x = model_a_x if which == "a" else model_b_x
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                if device.type == "cuda":
                    x = x.contiguous(memory_format=torch.channels_last)

                optimizer.zero_grad(set_to_none=True)
                logits = model_local(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()

                running_loss += float(loss.item()) * y.size(0)
                n += y.size(0)

            print(
                f"train {which}: epoch {epoch+1}/{train_epochs} loss={running_loss/max(n,1):.4f}"
            )

        model.eval()

    _train_one_model(model_a, which="a")
    _train_one_model(model_b, which="b")




## === cell 4
all_names = []
all_preds = []

model_a.eval()
model_b.eval()

if device.type == "cuda":
    model_a = model_a.to(memory_format=torch.channels_last)
    model_b = model_b.to(memory_format=torch.channels_last)

model_a_infer = _maybe_compile(model_a)
model_b_infer = _maybe_compile(model_b)

with torch.inference_mode():
    for model_a_inputs, model_b_inputs, filenames in test_loader:
        this_bs = len(filenames)

        if tta:
            model_a_inputs = torch.cat(model_a_inputs, dim=0).to(
                device, non_blocking=True
            )
            model_b_inputs = torch.cat(model_b_inputs, dim=0).to(
                device, non_blocking=True
            )

            if device.type == "cuda":
                model_a_inputs = model_a_inputs.contiguous(
                    memory_format=torch.channels_last
                )
                model_b_inputs = model_b_inputs.contiguous(
                    memory_format=torch.channels_last
                )

            logits_a = model_a_infer(model_a_inputs)
            logits_b = model_b_infer(model_b_inputs)

            model_a_batch_logits = torch.stack(torch.split(logits_a, this_bs), dim=0)
            model_a_mean_logits = torch.mean(model_a_batch_logits, dim=0)

            model_b_batch_logits = torch.stack(torch.split(logits_b, this_bs), dim=0)
            model_b_mean_logits = torch.mean(model_b_batch_logits, dim=0)

            outputs = (0.9 * model_a_mean_logits + 0.1 * model_b_mean_logits) / 2
            mean_preds = normalizer(outputs)
            pred_labels = torch.argmax(mean_preds, 1).tolist()
        else:
            model_a_inputs = model_a_inputs.to(device, non_blocking=True)
            model_b_inputs = model_b_inputs.to(device, non_blocking=True)

            if device.type == "cuda":
                model_a_inputs = model_a_inputs.contiguous(
                    memory_format=torch.channels_last
                )
                model_b_inputs = model_b_inputs.contiguous(
                    memory_format=torch.channels_last
                )

            logits_a = model_a_infer(model_a_inputs)
            logits_b = model_b_infer(model_b_inputs)

            outputs = (0.93 * logits_a + 0.07 * logits_b) / 2
            preds = normalizer(outputs)
            pred_labels = torch.argmax(preds, 1).tolist()

        all_names.extend(filenames)
        all_preds.extend(pred_labels)




## === cell 5
pred_df = pd.DataFrame({"image_id": all_names, "label": all_preds})

pred_df = sample_sub[["image_id"]].merge(pred_df, on="image_id", how="left")
pred_df["label"] = pred_df["label"].fillna(0).astype(int)

pred_df.to_csv("submission.csv", index=False)
print(pred_df.shape)
print(pred_df.head())
