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

0.900725294650952

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.05531) has done: 'I fix the immediate runtime blocker by removing the dependency on missing external model files and replacing it with a lightweight, deterministic fallback that still produces valid class predictions. I also fix the submission-length issue by ensuring we iterate test images in the exact order of `sample_submission.csv` (no shuffling) and by enforcing that every `image_id` in the sample gets exactly one prediction. Finally, I make TTA deterministic and compatible with this fallback so the pipeline runs end-to-end and writes a correct `submission.csv` with the required columns and row count.'
- What this solution (achieved 0.12294) has done: 'I fix the ViT input-size mismatch by aligning `vit_img_size` with the model’s expected `image_size` (224 for `vit_b_16`), which is the immediate runtime blocker. I also make the TTA batching robust by stacking TTA views per sample instead of relying on DataLoader’s default list collation, preventing shape/ordering issues and ensuring one prediction per test image. Finally, I keep the ensemble and submission ordering exactly as-is (using `sample_submission.csv` order) so the script runs end-to-end and writes a valid `submission.csv`.'
- What this solution (achieved 0.76756) has done: 'Your score is far below the target, and the main reason is that your current pipeline loads no trained cassava weights and falls back to ImageNet-pretrained models with randomly initialized 5-class heads, which yields near-random predictions. The smallest score-improving change that preserves your core inference logic is to add a lightweight training step on `train.csv` using the exact same two backbones you already use, then reuse the same ensemble + (optional) linear head at test time. To stay minimal and stable, I freeze both backbones and train only their classification heads (and keep the same transforms/TTA/inference code paths), which should move accuracy substantially toward the target without changing the architecture or loss. I also keep submission ordering locked to `sample_submission.csv` and write `submission.csv` exactly as required.'

# 9. Code solution

## === cell 0
import os
import math
import multiprocessing as mp

import pandas as pd
import torch
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
from torchvision.transforms import InterpolationMode, v2

torch.manual_seed(3407)
if torch.cuda.is_available():
    torch.cuda.manual_seed(3407)

cudnn.deterministic = False
cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification"
train_csv_path = f"{DATA_DIR}/train.csv"
train_dir = f"{DATA_DIR}/train_images/"
test_dir = f"{DATA_DIR}/test_images/"
sample_path = f"{DATA_DIR}/sample_submission.csv"

eff_img_size = 528
vit_img_size = 224

batch_size = 16
num_workers = min(8, max(2, (os.cpu_count() or 2)))
num_classes = 5
tta = True

train_epochs = 6
train_lr = 1e-3


def _safe_torch_load(path: str, map_location):
    try:
        if os.path.exists(path):
            return torch.load(path, map_location=map_location)
    except Exception:
        pass
    return None


vit_model = _safe_torch_load(
    "/kaggle/input/vit-v1-update/vit_v1_1.pt", map_location=device
)
eff_model = _safe_torch_load(
    "/kaggle/input/efficient-net/vit_cont_3.pt", map_location=device
)
linear_head = _safe_torch_load(
    "/kaggle/input/linear-head/linear_cls.pt", map_location=device
)

if vit_model is None or eff_model is None:
    import torchvision

    vit_backbone = torchvision.models.vit_b_16(
        weights=torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1
    )
    in_features_vit = vit_backbone.heads.head.in_features
    vit_backbone.heads.head = torch.nn.Linear(in_features_vit, num_classes)

    eff_backbone = torchvision.models.efficientnet_b0(
        weights=torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1
    )
    in_features_eff = eff_backbone.classifier[-1].in_features
    eff_backbone.classifier[-1] = torch.nn.Linear(in_features_eff, num_classes)

    vit_model = vit_backbone
    eff_model = eff_backbone

if linear_head is None:
    linear_head = torch.nn.Identity()

vit_model = vit_model.to(device).to(memory_format=torch.channels_last)
eff_model = eff_model.to(device).to(memory_format=torch.channels_last)
linear_head = linear_head.to(device)



## === cell 1
import torchvision
from torchvision.io import read_file, decode_jpeg
from torchvision.transforms.v2 import functional as F

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

test_transforms = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=list(_IMAGENET_MEAN), std=list(_IMAGENET_STD)),
    ]
)

if tta:
    ttas = [
        v2.Identity(),
        v2.RandomHorizontalFlip(p=1.0),
        v2.RandomVerticalFlip(p=1.0),
        v2.RandomRotation(degrees=(90, 90)),
        v2.RandomRotation(degrees=(270, 270)),
    ]
else:
    ttas = None

sample_df = pd.read_csv(sample_path)
test_image_ids = sample_df["image_id"].tolist()

normalizer = torch.nn.Softmax(dim=1)




## === cell 2
class CassavaDataset(VisionDataset):
    """Custom dataset for Cassava data (supports train with labels and test without)."""

    def __init__(
        self,
        data_dir,
        vit_size,
        efficient_size,
        image_ids,
        labels=None,
        transform=None,
        ttas=None,
        cache_preprocessed=False,
    ):
        super().__init__(root=data_dir)
        self.transform = (
            transform  # kept for backward compatibility; not used in fast path
        )
        self.images = list(image_ids)
        self.labels = None if labels is None else list(labels)
        self.ttas = ttas
        self.cache_preprocessed = cache_preprocessed

        self.vit_size = int(vit_size)
        self.efficient_size = int(efficient_size)

        self.max_size = int(max(self.vit_size, self.efficient_size))
        self._cache = {} if cache_preprocessed else None

        self._mean = torch.tensor(_IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
        self._std = torch.tensor(_IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)

    @staticmethod
    def _center_crop_600(img_chw: torch.Tensor) -> torch.Tensor:
        _, h, w = img_chw.shape
        th = tw = 600
        i = int(round((h - th) / 2.0))
        j = int(round((w - tw) / 2.0))
        return img_chw[:, i : i + th, j : j + tw]

    def _load_preprocessed_max_tensor(self, filename: str) -> torch.Tensor:
        if self._cache is not None:
            cached = self._cache.get(filename, None)
            if cached is not None:
                return cached

        img_path = os.path.join(self.root, filename)

        data = read_file(img_path)
        img = decode_jpeg(data, mode=torchvision.io.ImageReadMode.RGB)  # HWC uint8
        img = img.permute(2, 0, 1).contiguous()  # CHW uint8

        img = self._center_crop_600(img)

        img = F.resize(
            img,
            [self.max_size, self.max_size],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )

        img = img.to(dtype=torch.float32).div_(255.0)
        img = (img - self._mean) / self._std

        if self._cache is not None:
            self._cache[filename] = img
        return img

    def __getitem__(self, idx):
        filename = self.images[idx]
        img_max = self._load_preprocessed_max_tensor(filename)

        if self.ttas is not None:
            vit_list = []
            eff_list = []
            for t in self.ttas:
                img_t = t(img_max)

                vit_img = (
                    img_t
                    if self.vit_size == self.max_size
                    else F.resize(
                        img_t,
                        [self.vit_size, self.vit_size],
                        interpolation=InterpolationMode.BICUBIC,
                        antialias=True,
                    )
                )
                eff_img = (
                    img_t
                    if self.efficient_size == self.max_size
                    else F.resize(
                        img_t,
                        [self.efficient_size, self.efficient_size],
                        interpolation=InterpolationMode.BICUBIC,
                        antialias=True,
                    )
                )

                vit_list.append(vit_img)
                eff_list.append(eff_img)

            vit_stack = torch.stack(vit_list, dim=0)
            eff_stack = torch.stack(eff_list, dim=0)

            if self.labels is None:
                return vit_stack, eff_stack, filename
            return vit_stack, eff_stack, int(self.labels[idx])

        vit_img = (
            img_max
            if self.vit_size == self.max_size
            else F.resize(
                img_max,
                [self.vit_size, self.vit_size],
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            )
        )
        eff_img = (
            img_max
            if self.efficient_size == self.max_size
            else F.resize(
                img_max,
                [self.efficient_size, self.efficient_size],
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            )
        )

        if self.labels is None:
            return vit_img, eff_img, filename
        return vit_img, eff_img, int(self.labels[idx])

    def __len__(self):
        return len(self.images)




## === cell 3
vit_loaded = os.path.exists("/kaggle/input/vit-v1-update/vit_v1_1.pt")
eff_loaded = os.path.exists("/kaggle/input/efficient-net/vit_cont_3.pt")
need_train = not (vit_loaded and eff_loaded)

_can_mark_step = hasattr(torch, "compiler") and hasattr(
    torch.compiler, "cudagraph_mark_step_begin"
)

try:
    vit_model = torch.compile(
        vit_model, mode="reduce-overhead", fullgraph=False, disable_cudagraphs=True
    )
    eff_model = torch.compile(
        eff_model, mode="reduce-overhead", fullgraph=False, disable_cudagraphs=True
    )
    if not isinstance(linear_head, torch.nn.Identity):
        linear_head = torch.compile(
            linear_head,
            mode="reduce-overhead",
            fullgraph=False,
            disable_cudagraphs=True,
        )
except Exception:
    pass

if need_train:
    train_df = pd.read_csv(train_csv_path)

    g = train_df.groupby("label", group_keys=False)

    def _split_group(df, frac=0.1, seed=3407):
        df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        n_val = max(1, int(len(df) * frac))
        return df.iloc[n_val:], df.iloc[:n_val]

    train_parts = []
    val_parts = []
    for _, df_g in g:
        tr_g, va_g = _split_group(df_g, frac=0.1, seed=3407)
        train_parts.append(tr_g)
        val_parts.append(va_g)

    tr_df = (
        pd.concat(train_parts, axis=0)
        .sample(frac=1.0, random_state=3407)
        .reset_index(drop=True)
    )
    va_df = (
        pd.concat(val_parts, axis=0)
        .sample(frac=1.0, random_state=3407)
        .reset_index(drop=True)
    )

    train_dataset = CassavaDataset(
        train_dir,
        vit_img_size,
        eff_img_size,
        image_ids=tr_df["image_id"].tolist(),
        labels=tr_df["label"].tolist(),
        transform=test_transforms,
        ttas=None,
        cache_preprocessed=False,
    )
    val_dataset = CassavaDataset(
        train_dir,
        vit_img_size,
        eff_img_size,
        image_ids=va_df["image_id"].tolist(),
        labels=va_df["label"].tolist(),
        transform=test_transforms,
        ttas=None,
        cache_preprocessed=False,
    )

    _dl_kw = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    if num_workers > 0:
        _dl_kw["prefetch_factor"] = 4

    train_loader = DataLoader(train_dataset, shuffle=True, **_dl_kw)
    val_loader = DataLoader(val_dataset, shuffle=False, **_dl_kw)

    for p in vit_model.parameters():
        p.requires_grad = False
    for p in eff_model.parameters():
        p.requires_grad = False

    vit_head_params = []
    eff_head_params = []

    if hasattr(vit_model, "heads") and hasattr(vit_model.heads, "head"):
        for p in vit_model.heads.head.parameters():
            p.requires_grad = True
        vit_head_params = list(vit_model.heads.head.parameters())

    if hasattr(eff_model, "classifier") and isinstance(
        eff_model.classifier, torch.nn.Sequential
    ):
        for p in eff_model.classifier[-1].parameters():
            p.requires_grad = True
        eff_head_params = list(eff_model.classifier[-1].parameters())

    params = vit_head_params + eff_head_params
    if len(params) == 0:
        raise RuntimeError("Could not find classification head parameters to train.")

    optimizer = torch.optim.Adam(params, lr=train_lr)
    criterion = torch.nn.CrossEntropyLoss()

    best_acc = -1.0
    best_state = None

    for epoch in range(train_epochs):
        vit_model.eval()
        eff_model.eval()
        if hasattr(vit_model, "heads") and hasattr(vit_model.heads, "head"):
            vit_model.heads.head.train()
        if hasattr(eff_model, "classifier") and isinstance(
            eff_model.classifier, torch.nn.Sequential
        ):
            eff_model.classifier[-1].train()

        running_loss = 0.0
        seen = 0

        for vit_inputs, eff_inputs, y in train_loader:
            vit_inputs = vit_inputs.to(device, non_blocking=True).to(
                memory_format=torch.channels_last
            )
            eff_inputs = eff_inputs.to(device, non_blocking=True).to(
                memory_format=torch.channels_last
            )
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            if _can_mark_step:
                torch.compiler.cudagraph_mark_step_begin()
            vit_logits = vit_model(vit_inputs)

            if _can_mark_step:
                torch.compiler.cudagraph_mark_step_begin()
            eff_logits = eff_model(eff_inputs)

            logits = (vit_logits + eff_logits) / 2.0

            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            bs = y.size(0)
            running_loss += loss.item() * bs
            seen += bs

        vit_model.eval()
        eff_model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for vit_inputs, eff_inputs, y in val_loader:
                vit_inputs = vit_inputs.to(device, non_blocking=True).to(
                    memory_format=torch.channels_last
                )
                eff_inputs = eff_inputs.to(device, non_blocking=True).to(
                    memory_format=torch.channels_last
                )
                y = y.to(device, non_blocking=True)

                if _can_mark_step:
                    torch.compiler.cudagraph_mark_step_begin()
                vit_logits = vit_model(vit_inputs)

                if _can_mark_step:
                    torch.compiler.cudagraph_mark_step_begin()
                eff_logits = eff_model(eff_inputs)

                logits = (vit_logits + eff_logits) / 2.0
                pred = torch.argmax(logits, dim=1)
                correct += (pred == y).sum().item()
                total += y.numel()

        val_acc = correct / max(1, total)
        print(
            f"epoch {epoch+1}/{train_epochs} loss={running_loss/max(seen,1):.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {
                "vit_head": (
                    vit_model.heads.head.state_dict()
                    if hasattr(vit_model, "heads") and hasattr(vit_model.heads, "head")
                    else None
                ),
                "eff_head": (
                    eff_model.classifier[-1].state_dict()
                    if hasattr(eff_model, "classifier")
                    and isinstance(eff_model.classifier, torch.nn.Sequential)
                    else None
                ),
            }

    if best_state is not None:
        if best_state["vit_head"] is not None:
            vit_model.heads.head.load_state_dict(best_state["vit_head"])
        if best_state["eff_head"] is not None:
            eff_model.classifier[-1].load_state_dict(best_state["eff_head"])

    vit_model.eval()
    eff_model.eval()
    if not isinstance(linear_head, torch.nn.Identity):
        linear_head.eval()




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/2287018221.py in <cell line: 0>()
    129         seen = 0
    130 
--> 131         for vit_inputs, eff_inputs, y in train_loader:
    132             vit_inputs = vit_inputs.to(device, non_blocking=True).to(
    133                 memory_format=torch.channels_last

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

RuntimeError: Caught RuntimeError in DataLoader worker process 0.
Original Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/worker.py", line 349, in _worker_loop
    data = fetcher.fetch(index)  # type: ignore[possibly-undefined]
           ^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py", line 52, in fetch
    data = [self.dataset[idx] for idx in possibly_batched_index]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py", line 52, in <listcomp>
    data = [self.dataset[idx] for idx in possibly_batched_index]
            ~~~~~~~~~~~~^^^^^
  File "/tmp/ipykernel_55/3629388164.py", line 78, in __getitem__
    img_max = self._load_preprocessed_max_tensor(filename)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/ipykernel_55/3629388164.py", line 70, in _load_preprocessed_max_tensor
    img = (img - self._mean) / self._std
           ~~~~^~~~~~~~~~~~
RuntimeError: The size of tensor a (800) must match the size of tensor b (3) at non-singleton dimension 0


## === cell 4
def _collate_with_filenames(batch):
    vit, eff, names = zip(*batch)
    return torch.stack(vit, 0), torch.stack(eff, 0), names


test_dataset = CassavaDataset(
    test_dir,
    vit_img_size,
    eff_img_size,
    image_ids=test_image_ids,
    labels=None,
    transform=test_transforms,
    ttas=ttas,
    cache_preprocessed=True,
)

_dl_kw = dict(
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=(num_workers > 0),
    collate_fn=_collate_with_filenames,
)
if num_workers > 0:
    _dl_kw["prefetch_factor"] = 4

test_loader = DataLoader(test_dataset, **_dl_kw)

all_names = []
all_preds = []

vit_model.eval()
eff_model.eval()
linear_head.eval()

with torch.inference_mode():
    for batch_idx, (vit_inputs, eff_inputs, filenames) in enumerate(test_loader):
        cur_bs = len(filenames)

        if tta:
            vit_inputs = (
                vit_inputs.flatten(0, 1)
                .to(device, non_blocking=True)
                .to(memory_format=torch.channels_last)
            )
            eff_inputs = (
                eff_inputs.flatten(0, 1)
                .to(device, non_blocking=True)
                .to(memory_format=torch.channels_last)
            )

            if _can_mark_step:
                torch.compiler.cudagraph_mark_step_begin()
            vit_outputs = vit_model(vit_inputs)

            if _can_mark_step:
                torch.compiler.cudagraph_mark_step_begin()
            eff_outputs = eff_model(eff_inputs)

            n_tta = vit_outputs.shape[0] // cur_bs
            vit_mean_logits = vit_outputs.view(cur_bs, n_tta, -1).mean(dim=1)
            eff_mean_logits = eff_outputs.view(cur_bs, n_tta, -1).mean(dim=1)

            outputs = (vit_mean_logits + eff_mean_logits) / 2.0
            outputs = (
                linear_head(outputs)
                if not isinstance(linear_head, torch.nn.Identity)
                else outputs
            )

            mean_preds = normalizer(outputs)
            pred_labels = torch.argmax(mean_preds, dim=1).tolist()
        else:
            vit_inputs = vit_inputs.to(device, non_blocking=True).to(
                memory_format=torch.channels_last
            )
            eff_inputs = eff_inputs.to(device, non_blocking=True).to(
                memory_format=torch.channels_last
            )

            if _can_mark_step:
                torch.compiler.cudagraph_mark_step_begin()
            vit_outputs = vit_model(vit_inputs)

            if _can_mark_step:
                torch.compiler.cudagraph_mark_step_begin()
            eff_outputs = eff_model(eff_inputs)

            outputs = (vit_outputs + eff_outputs) / 2.0
            outputs = (
                linear_head(outputs)
                if not isinstance(linear_head, torch.nn.Identity)
                else outputs
            )

            preds = normalizer(outputs)
            pred_labels = torch.argmax(preds, dim=1).tolist()

        all_names.extend(filenames)
        all_preds.extend(pred_labels)

pred_map = dict(zip(all_names, all_preds))
final_preds = [int(pred_map[iid]) for iid in test_image_ids]

assert len(final_preds) == len(
    sample_df
), "Prediction length mismatch with sample_submission."

my_submission = pd.DataFrame({"image_id": test_image_ids, "label": final_preds})
my_submission.to_csv("submission.csv", index=False)
print(my_submission.head())
print("Wrote submission.csv with rows:", len(my_submission))

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/3408944662.py in <cell line: 0>()
     38 
     39 with torch.inference_mode():
---> 40     for batch_idx, (vit_inputs, eff_inputs, filenames) in enumerate(test_loader):
     41         cur_bs = len(filenames)
     42 

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

RuntimeError: Caught RuntimeError in DataLoader worker process 0.
Original Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/worker.py", line 349, in _worker_loop
    data = fetcher.fetch(index)  # type: ignore[possibly-undefined]
           ^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py", line 52, in fetch
    data = [self.dataset[idx] for idx in possibly_batched_index]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py", line 52, in <listcomp>
    data = [self.dataset[idx] for idx in possibly_batched_index]
            ~~~~~~~~~~~~^^^^^
  File "/tmp/ipykernel_55/3629388164.py", line 78, in __getitem__
    img_max = self._load_preprocessed_max_tensor(filename)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/ipykernel_55/3629388164.py", line 70, in _load_preprocessed_max_tensor
    img = (img - self._mean) / self._std
           ~~~~^~~~~~~~~~~~
RuntimeError: The size of tensor a (800) must match the size of tensor b (3) at non-singleton dimension 0
