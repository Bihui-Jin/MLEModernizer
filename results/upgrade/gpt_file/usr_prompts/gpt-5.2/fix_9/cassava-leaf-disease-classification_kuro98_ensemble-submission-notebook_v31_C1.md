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

# 5. Target score

0.8957388939256573

# 6. Current score

0.05531

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.05531) has done: 'I first fix the runtime blocker by removing the hard dependency on missing external model files and replacing it with a locally available torchvision ViT that matches the same inference semantics (logits → softmax → argmax). Then I fix the submission-length/index issue by making test iteration deterministic (no shuffle) and by aligning predictions to `sample_submission.csv` ordering (and filling any missing predictions safely). I also fix a small ensembling bug where weights were unintentionally halved by `/ 2`, preserving the intended 0.95/0.05 mix. Finally, I keep the rest of the pipeline (dataset, TTA averaging, normalization, output formatting) intact and ensure a valid `submission.csv` is written.'
- What this solution (achieved 0.13752) has done: 'I fix the runtime crash by making the ViT models’ expected `image_size` match the 384×384 inputs you’re already generating, instead of the default 224×224. This is a minimal change that preserves the existing inference semantics (logits → softmax → argmax), TTA path, and ensembling weights. I also keep everything else intact so the pipeline runs end-to-end and writes a valid `submission.csv` in the required format. Note: your current score is extremely low because you’re using randomly initialized ViTs (no trained weights), but the requested scope/constraints don’t include adding a training stage.'
- What this solution (achieved 0.13752) has done: 'The crash comes from the test images folder containing a nested `test_images/` directory, and the dataset currently tries to open that directory as if it were an image file. I fix this by filtering `os.listdir()` to include only real image files (and optionally checking common extensions), keeping ordering deterministic. This change is score-neutral but unblocks inference so the pipeline completes and writes `submission.csv`. No model/inference semantics are changed beyond preventing non-file paths from entering the dataloader.'
- What this solution (achieved 0.05531) has done: 'The timeout is dominated by (1) training a full ViT-B/16 from scratch over the entire 18.7k-image training set for 2 epochs, and (2) extremely expensive per-sample CPU TTA generation in `collate_fn` (creating new `torch.Generator` objects and running PIL/transform ops multiple times per image). Without changing the model, loss, epochs, or TTA semantics, the biggest safe wins are to: cache and reuse constant transform objects, eliminate redundant `ToImage()` conversions, move deterministic TTA seeding to a cheap per-batch precomputed seed tensor, and make the training loader cheaper by avoiding repeated pandas `.iloc` overhead and repeated path joins. Additionally, enabling `torch.set_float32_matmul_precision("high")` and using `non_blocking` + pinned memory consistently improves GPU throughput without altering the computation graph. The code below keeps the same architecture, training loop, augmentations, and TTA types, but removes avoidable Python/PIL overhead and repeated work.'

# 9. Code solution

## === cell 0
import os
import random

import pandas as pd
import torch
from PIL import Image, ImageFile
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
from torchvision.transforms import InterpolationMode, v2

ImageFile.LOAD_TRUNCATED_IMAGES = True

torch.manual_seed(3407)
random.seed(3407)
if torch.cuda.is_available():
    torch.cuda.manual_seed(3407)

cudnn.deterministic = False
cudnn.benchmark = True
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

test_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images/"
train_csv_path = "/kaggle/input/cassava-leaf-disease-classification/train.csv"
train_dir = "/kaggle/input/cassava-leaf-disease-classification/train_images/"

model_b_img_size = 384
model_a_img_size = 384
batch_size = 16
num_workers = 4
num_classes = 5
tta = True

train_epochs = 2
train_lr = 3e-4
weight_decay = 0.05

import torchvision


def _build_vit_5cls(img_size: int):
    m = torchvision.models.vit_b_16(weights=None, image_size=img_size)
    if hasattr(m, "heads") and hasattr(m.heads, "head"):
        in_features = m.heads.head.in_features
        m.heads.head = torch.nn.Linear(in_features, num_classes)
    else:
        in_features = m.classifier.in_features
        m.classifier = torch.nn.Linear(in_features, num_classes)
    return m


model_a = _build_vit_5cls(model_a_img_size).to(device)
model_b = _build_vit_5cls(model_b_img_size).to(device)

linear_head = torch.nn.Identity()


def _maybe_compile(m):
    try:
        return torch.compile(m, mode="reduce-overhead")
    except Exception as e:
        print("torch.compile unavailable/fallback:", repr(e))
        return m


if torch.cuda.is_available():
    model_a = _maybe_compile(model_a)
    model_b = _maybe_compile(model_b)

if torch.cuda.is_available():
    model_a = model_a.to(memory_format=torch.channels_last)
    model_b = model_b.to(memory_format=torch.channels_last)




## === cell 1
class CassavaDataset(VisionDataset):
    """Custom dataset for the Cassava test data."""

    def __init__(
        self,
        data_dir,
        model_a_size,
        model_b_size,
        transform=None,
        ttas=None,
        base_seed: int = 3407,
    ):
        super().__init__(root=data_dir)

        self.transform = transform

        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
        entries = sorted(os.listdir(data_dir))
        self.images = []
        for name in entries:
            full = os.path.join(data_dir, name)
            if os.path.isfile(full) and (os.path.splitext(name)[1].lower() in exts):
                self.images.append(name)

        self.ttas = ttas
        self.base_seed = int(base_seed)

        self.cc = v2.CenterCrop((600, 600))
        self.resize_model_a = v2.Resize(
            (model_a_size, model_a_size), interpolation=InterpolationMode.BICUBIC
        )
        self.resize_model_b = v2.Resize(
            (model_b_size, model_b_size), interpolation=InterpolationMode.BICUBIC
        )

        self._to_image = v2.ToImage()

    def _get_base_pair(self, idx: int):
        filename = self.images[idx]
        img = Image.open(os.path.join(self.root, filename)).convert("RGB")
        img = self.cc(img)
        a_img = self.resize_model_a(img)
        b_img = self.resize_model_b(img)
        a = self._to_image(a_img)
        b = self._to_image(b_img)
        return a, b

    def __getitem__(self, idx):
        filename = self.images[idx]
        model_a_img, model_b_img = self._get_base_pair(idx)
        return model_a_img, model_b_img, filename, idx

    def __len__(self):
        return len(self.images)


class CassavaTrainDataset(VisionDataset):
    def __init__(self, csv_path, images_dir, model_a_size, transform=None):
        super().__init__(root=images_dir)
        df = pd.read_csv(csv_path)

        self.image_ids = df["image_id"].tolist()
        self.labels = df["label"].astype("int64").tolist()

        self.transform = transform
        self.cc = v2.CenterCrop((600, 600))
        self.resize = v2.Resize(
            (model_a_size, model_a_size), interpolation=InterpolationMode.BICUBIC
        )
        self._to_image = v2.ToImage()

        self.paths = [os.path.join(self.root, fn) for fn in self.image_ids]

    def _get_base(self, idx: int):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = self.cc(img)
        img = self.resize(img)
        x = self._to_image(img)  # uint8 tensor
        return x

    def __getitem__(self, idx):
        label = int(self.labels[idx])
        img = self._get_base(idx)
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.labels)




## === cell 2
test_transforms = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

if tta:
    ttas = [
        v2.RandomRotation(180, interpolation=InterpolationMode.BILINEAR),
        v2.RandomVerticalFlip(1),
        v2.RandomPerspective(p=1),
    ]
else:
    ttas = None

test_dataset = CassavaDataset(
    test_dir, model_a_img_size, model_b_img_size, transform=test_transforms, ttas=ttas
)


def _seed_worker(worker_id: int):
    seed = 3407 + worker_id
    random.seed(seed)
    torch.manual_seed(seed)


def _make_test_collate_fn(dataset: CassavaDataset):
    base_seed = dataset.base_seed
    transform = dataset.transform
    ttas_local = dataset.ttas

    if ttas_local is not None:
        gen_pool_a = [torch.Generator() for _ in range(len(ttas_local))]
        gen_pool_b = [torch.Generator() for _ in range(len(ttas_local))]
    else:
        gen_pool_a = None
        gen_pool_b = None

    def _collate(batch):
        a_imgs, b_imgs, fnames, idxs = zip(*batch)

        if ttas_local is not None and transform is not None:
            T = len(ttas_local)
            a_out = []
            b_out = []

            for a_img, b_img, idx in zip(a_imgs, b_imgs, idxs):
                idx_i = int(idx)
                a_list = []
                b_list = []
                seed_base = base_seed + idx_i * 1000

                for j, t in enumerate(ttas_local):
                    g = gen_pool_a[j]
                    g.manual_seed(seed_base + j)
                    a_list.append(transform(t(a_img, generator=g)))

                    g2 = gen_pool_b[j]
                    g2.manual_seed(seed_base + j)
                    b_list.append(transform(t(b_img, generator=g2)))

                a_out.append(torch.stack(a_list, dim=0))  # [T,C,H,W]
                b_out.append(torch.stack(b_list, dim=0))  # [T,C,H,W]

            return torch.stack(a_out, dim=0), torch.stack(b_out, dim=0), list(fnames)

        elif transform is not None:
            a_out = torch.stack([transform(x) for x in a_imgs], dim=0)
            b_out = torch.stack([transform(x) for x in b_imgs], dim=0)
            return a_out, b_out, list(fnames)
        else:
            return list(a_imgs), list(b_imgs), list(fnames)

    return _collate


_eff_workers = min(8, (os.cpu_count() or 4))
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=_eff_workers,
    pin_memory=True,
    persistent_workers=(_eff_workers > 0),
    prefetch_factor=(4 if _eff_workers > 0 else None),
    worker_init_fn=_seed_worker if _eff_workers > 0 else None,
    collate_fn=_make_test_collate_fn(test_dataset),
)

train_transforms = v2.Compose(
    [
        v2.ToImage(),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),
        v2.RandomRotation(20, interpolation=InterpolationMode.BILINEAR),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

train_dataset = CassavaTrainDataset(
    csv_path=train_csv_path,
    images_dir=train_dir,
    model_a_size=model_a_img_size,
    transform=train_transforms,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=_eff_workers,
    pin_memory=True,
    persistent_workers=(_eff_workers > 0),
    prefetch_factor=4 if _eff_workers > 0 else None,
    worker_init_fn=_seed_worker if _eff_workers > 0 else None,
)

normalizer = torch.nn.Softmax(dim=1)




## === cell 3
model_a.train()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    model_a.parameters(), lr=train_lr, weight_decay=weight_decay
)

for epoch in range(train_epochs):
    running_loss = 0.0
    correct = 0
    total = 0
    for imgs, labels in train_loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if torch.cuda.is_available():
            imgs = imgs.contiguous(memory_format=torch.channels_last)

        optimizer.zero_grad(set_to_none=True)
        logits = model_a(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += float(loss.item()) * imgs.size(0)
        preds = torch.argmax(logits.detach(), dim=1)
        correct += int((preds == labels).sum().item())
        total += int(labels.numel())

    print(
        f"epoch {epoch+1}/{train_epochs} "
        f"loss={running_loss/max(total,1):.4f} "
        f"train_acc={correct/max(total,1):.4f}"
    )

model_a.eval()
model_b.eval()
if hasattr(linear_head, "eval"):
    linear_head.eval()




## === cell 4
all_names = []
all_preds = []

with torch.inference_mode():
    for batch_idx, (model_a_inputs, model_b_inputs, filenames) in enumerate(
        test_loader
    ):
        bs = len(filenames)

        if tta:
            model_a_inputs = model_a_inputs.view(-1, *model_a_inputs.shape[2:]).to(
                device, non_blocking=True
            )
            model_b_inputs = model_b_inputs.view(-1, *model_b_inputs.shape[2:]).to(
                device, non_blocking=True
            )
            if torch.cuda.is_available():
                model_a_inputs = model_a_inputs.contiguous(
                    memory_format=torch.channels_last
                )
                model_b_inputs = model_b_inputs.contiguous(
                    memory_format=torch.channels_last
                )

            model_a_outputs = model_a(model_a_inputs)
            model_b_outputs = model_b(model_b_inputs)

            model_a_batch_logits = model_a_outputs.view(
                -1, bs, model_a_outputs.shape[-1]
            )
            model_a_mean_logits = torch.mean(model_a_batch_logits, dim=0)

            model_b_batch_logits = model_b_outputs.view(
                -1, bs, model_b_outputs.shape[-1]
            )
            model_b_mean_logits = torch.mean(model_b_batch_logits, dim=0)

            outputs = 0.95 * model_a_mean_logits + 0.05 * model_b_mean_logits

            mean_preds = normalizer(outputs)
            pred_labels = torch.argmax(mean_preds, 1).tolist()
        else:
            model_a_inputs = model_a_inputs.to(device, non_blocking=True)
            model_b_inputs = model_b_inputs.to(device, non_blocking=True)
            if torch.cuda.is_available():
                model_a_inputs = model_a_inputs.contiguous(
                    memory_format=torch.channels_last
                )
                model_b_inputs = model_b_inputs.contiguous(
                    memory_format=torch.channels_last
                )

            model_a_outputs = model_a(model_a_inputs)
            model_b_outputs = model_b(model_b_inputs)

            outputs = 0.95 * model_a_outputs + 0.05 * model_b_outputs

            preds = normalizer(outputs)
            pred_labels = torch.argmax(preds, 1).tolist()

        all_names.extend(filenames)
        all_preds.extend(pred_labels)




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3273737553.py in <cell line: 0>()
      3 
      4 with torch.inference_mode():
----> 5     for batch_idx, (model_a_inputs, model_b_inputs, filenames) in enumerate(
      6         test_loader
      7     ):

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in __next__(self)
    706                 # TODO(https://github.com/pytorch/pytorch/issues/76750)
    707                 self._reset()  # type: ignore[call-arg]
--> 708             data = self._next_data()
    709             self._num_yielded += 1
    710             if (

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _next_data(self)
   1478                 del self._task_info[idx]
   1479                 self._rcvd_idx += 1
-> 1480                 return self._process_data(data)
   1481 
   1482     def _try_put_index(self):

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _process_data(self, data)
   1503         self._try_put_index()
   1504         if isinstance(data, ExceptionWrapper):
-> 1505             data.reraise()
   1506         return data
   1507 

/usr/local/lib/python3.11/dist-packages/torch/_utils.py in reraise(self)
    731             # instantiate since we don't know how to
    732             raise RuntimeError(msg) from None
--> 733         raise exception
    734 
    735 

TypeError: Caught TypeError in DataLoader worker process 0.
Original Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/worker.py", line 349, in _worker_loop
    data = fetcher.fetch(index)  # type: ignore[possibly-undefined]
           ^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py", line 55, in fetch
    return self.collate_fn(data)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/ipykernel_55/3727614912.py", line 62, in _collate
    a_list.append(transform(t(a_img, generator=g)))
                            ^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: Transform.forward() got an unexpected keyword argument 'generator'


## === cell 5
sample_path = "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
sample_sub = pd.read_csv(sample_path)

pred_map = dict(zip(all_names, all_preds))
sub = sample_sub.copy()
sub["label"] = sub["image_id"].map(pred_map)

if sub["label"].isna().any():
    if len(all_preds) > 0:
        fallback = int(pd.Series(all_preds).mode().iloc[0])
    else:
        fallback = 0
    sub["label"] = sub["label"].fillna(fallback).astype(int)
else:
    sub["label"] = sub["label"].astype(int)

sub.to_csv("submission.csv", index=False)
print(sub.shape)
print(sub.head())
