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

0.8966455122393472

# 6. Current score

0.05531

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.05531) has done: 'I first fix the checkpoint loading so it no longer hard-depends on missing `/kaggle/input/...` model files; if those files aren’t present, the script fall back to standard torchvision backbones with a compatible linear head, so inference can run end-to-end. I also fix a resizing bug in the dataset where the “efficient” image path mistakenly used the ViT size, and make the test DataLoader deterministic and non-shuffled to ensure stable ordering. Finally, I enforce that the submission rows exactly match `sample_submission.csv` (same length and same image_id order), which fixes the “same length as the answers” error and guarantees a valid `submission.csv`.'
- What this solution (achieved 0.12369) has done: 'I fix the immediate runtime failures by (1) matching the ViT dummy input size to what the torchvision ViT actually expects (224), and (2) ensuring `linear_head` is always a valid `nn.Module` even if checkpoint loading returns `None` or a state_dict. I also make checkpoint loading more robust by accepting both full modules and state_dicts, so the code won’t crash or silently create a `None` head. These changes preserve the original inference logic (ViT + EfficientNet feature concat + linear head, with the same TTA structure) while making the pipeline run end-to-end and write a valid `submission.csv`.'
- What this solution (achieved 0.05531) has done: 'I fix the dataset file discovery bug that’s causing the `IsADirectoryError` by filtering `os.listdir()` to include only actual image files (and sorting them for determinism). I also make the test directory resolution more robust to the common nested `test_images/test_images` structure without changing any model/inference logic. These changes unblock end-to-end inference and ensure the prediction ordering is stable, while keeping the core ViT+EfficientNet feature concat + linear head + TTA pipeline intact. The submission generation remains aligned to `sample_submission.csv` and always write a valid `submission.csv`.'
- What this solution (achieved 0.05531) has done: 'Your current score indicates the pipeline is effectively using random/untrained weights for the linear head (and likely the backbones too), so predictions collapse toward near-chance accuracy; the smallest way to move toward the target is to ensure we actually load the intended checkpoints when they exist, and otherwise train only the existing linear head on top of frozen pretrained backbones using `train.csv` (keeping the same ViT+EfficientNet feature concat and same head/loss semantics). I also fix the TTA to be deterministic at inference time (your current `Random*` TTA is stochastic across runs and across worker processes), while preserving the same “3 TTA transforms then mean” logic. Finally, I keep the submission alignment to `sample_submission.csv` exactly as you already do.'

# 9. Code solution

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
if torch.cuda.is_available():
    torch.cuda.manual_seed(3407)

cudnn.deterministic = True
cudnn.benchmark = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


def _resolve_path(p: str) -> str:
    if os.path.exists(p):
        return p
    if p.startswith("/kaggle/input/"):
        alt = p.replace("/kaggle/input/", "/kaggle/data/")
        if os.path.exists(alt):
            return alt
    return p


_base_test_dir = _resolve_path(
    "/kaggle/input/cassava-leaf-disease-classification/test_images/"
)
_nested_test_dir = os.path.join(_base_test_dir, "test_images")
test_dir = _nested_test_dir if os.path.isdir(_nested_test_dir) else _base_test_dir

_base_train_dir = _resolve_path(
    "/kaggle/input/cassava-leaf-disease-classification/train_images/"
)
_nested_train_dir = os.path.join(_base_train_dir, "train_images")
train_dir = _nested_train_dir if os.path.isdir(_nested_train_dir) else _base_train_dir

train_csv_path = _resolve_path(
    "/kaggle/input/cassava-leaf-disease-classification/train.csv"
)
sample_sub_path = _resolve_path(
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)

eff_img_size = 528
vit_img_size = 224

batch_size = 16
num_workers = 4
num_classes = 5
tta = True

do_fallback_head_train = True
fallback_epochs = 2
fallback_lr = 3e-3


def _safe_torch_load(path, map_location):
    path = _resolve_path(path)
    try:
        if path and os.path.exists(path):
            return torch.load(path, map_location=map_location)
    except Exception:
        return None
    return None


def _as_module_or_none(obj):
    import torch.nn as nn

    if obj is None:
        return None, None
    if isinstance(obj, nn.Module):
        return obj, None
    if isinstance(obj, dict):
        return None, obj
    return None, None


vit_obj = _safe_torch_load("/kaggle/input/vit-v1-update/vit_v1_1.pt", device)
eff_obj = _safe_torch_load("/kaggle/input/efficient-net/vit_cont_3.pt", device)
head_obj = _safe_torch_load("/kaggle/input/linear-cls/linear_cls.pt", device)

vit_model, vit_sd = _as_module_or_none(vit_obj)
eff_model, eff_sd = _as_module_or_none(eff_obj)
linear_head, head_sd = _as_module_or_none(head_obj)

using_fallback_backbones = False
if vit_model is None or eff_model is None:
    using_fallback_backbones = True
    import torch.nn as nn
    import torchvision

    vit_model = torchvision.models.vit_b_16(
        weights=torchvision.models.ViT_B_16_Weights.DEFAULT
    )
    vit_model.heads = nn.Identity()

    eff_model = torchvision.models.efficientnet_b0(
        weights=torchvision.models.EfficientNet_B0_Weights.DEFAULT
    )
    eff_model.classifier = nn.Identity()

vit_model.eval()
eff_model.eval()
with torch.no_grad():
    dummy_vit = torch.zeros(1, 3, vit_img_size, vit_img_size)
    dummy_eff = torch.zeros(1, 3, eff_img_size, eff_img_size)
    vit_dim = vit_model(dummy_vit).shape[1]
    eff_dim = eff_model(dummy_eff).shape[1]

if linear_head is None:
    import torch.nn as nn

    linear_head = nn.Linear(vit_dim + eff_dim, num_classes)

if head_sd is not None:
    try:
        linear_head.load_state_dict(head_sd, strict=True)
    except Exception:
        pass

if vit_sd is not None:
    try:
        vit_model.load_state_dict(vit_sd, strict=False)
    except Exception:
        pass
if eff_sd is not None:
    try:
        eff_model.load_state_dict(eff_sd, strict=False)
    except Exception:
        pass

vit_model = vit_model.to(device).eval()
eff_model = eff_model.to(device).eval()
linear_head = linear_head.to(device).eval()




## === cell 2
class CassavaDataset(VisionDataset):
    """Custom dataset for the Cassava data.

    Args:
        data_dir: base directory to the images.
        transforms: set of transforms to be used.
    """

    def __init__(
        self,
        data_dir,
        vit_size,
        efficient_size,
        transform=None,
        ttas=None,
        img_size=384,
        labels_df=None,  # optional (for training head fallback)
    ):
        super().__init__(root=data_dir)

        self.transform = transform
        self.labels_df = labels_df

        exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
        names = os.listdir(data_dir)
        self.images = sorted(
            [
                n
                for n in names
                if os.path.isfile(os.path.join(data_dir, n))
                and n.lower().endswith(exts)
            ]
        )

        self.ttas = ttas
        self.cc = v2.CenterCrop((600, 600))
        self.resize_vit = v2.Resize(
            (vit_size, vit_size), interpolation=InterpolationMode.BICUBIC
        )

        self.resize_efficient = v2.Resize(
            (efficient_size, efficient_size), interpolation=InterpolationMode.BICUBIC
        )

        if self.labels_df is not None:
            self._label_map = dict(
                zip(self.labels_df["image_id"].values, self.labels_df["label"].values)
            )
        else:
            self._label_map = None

    def __getitem__(self, idx):
        filename = self.images[idx]
        img = Image.open(os.path.join(self.root, filename)).convert("RGB")
        img = self.cc(img)
        vit_img = self.resize_vit(img)
        eff_img = self.resize_efficient(img)

        if self.ttas is not None and self.transform is not None:
            vit_img = [self.transform(t(vit_img)) for t in self.ttas]
            eff_img = [self.transform(t(eff_img)) for t in self.ttas]
        elif self.transform:
            vit_img = self.transform(vit_img)
            eff_img = self.transform(eff_img)

        if self._label_map is None:
            return vit_img, eff_img, filename
        else:
            y = int(self._label_map[filename])
            return vit_img, eff_img, y, filename

    def __len__(self):
        return len(self.images)




## === cell 3
test_transforms = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

if tta:
    ttas = [
        v2.Lambda(lambda x: x.rotate(180, resample=Image.BICUBIC)),
        v2.Lambda(lambda x: x.transpose(Image.FLIP_TOP_BOTTOM)),
        v2.Lambda(
            lambda x: x
        ),  # identity as a mild third view; preserves TTA count/structure deterministically
    ]
else:
    ttas = None

test_dataset = CassavaDataset(
    test_dir, vit_img_size, eff_img_size, transform=test_transforms, ttas=ttas
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
)

normalizer = torch.nn.Softmax(dim=1)



## === cell 4
need_head_training = (head_sd is None) and do_fallback_head_train

if need_head_training:
    import torch.nn as nn

    train_df = pd.read_csv(train_csv_path)

    train_dataset = CassavaDataset(
        train_dir,
        vit_img_size,
        eff_img_size,
        transform=test_transforms,
        ttas=None,
        labels_df=train_df,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    for p in vit_model.parameters():
        p.requires_grad = False
    for p in eff_model.parameters():
        p.requires_grad = False
    for p in linear_head.parameters():
        p.requires_grad = True

    vit_model.eval()
    eff_model.eval()
    linear_head.train()

    opt = torch.optim.AdamW(linear_head.parameters(), lr=fallback_lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(fallback_epochs):
        for vit_inputs, eff_inputs, y, _fn in train_loader:
            vit_inputs = vit_inputs.to(device, non_blocking=True)
            eff_inputs = eff_inputs.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.no_grad():
                vit_feats = vit_model(vit_inputs)
                eff_feats = eff_model(eff_inputs)
                feats = torch.cat([vit_feats, eff_feats], dim=1)

            logits = linear_head(feats)
            loss = criterion(logits, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    linear_head.eval()



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/2698680261.py in <cell line: 0>()
     17         labels_df=train_df,
     18     )
---> 19     train_loader = DataLoader(
     20         train_dataset,
     21         batch_size=batch_size,

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in __init__(self, dataset, batch_size, shuffle, sampler, batch_sampler, num_workers, collate_fn, pin_memory, drop_last, timeout, worker_init_fn, multiprocessing_context, generator, prefetch_factor, persistent_workers, pin_memory_device, in_order)
    381             else:  # map-style
    382                 if shuffle:
--> 383                     sampler = RandomSampler(dataset, generator=generator)  # type: ignore[arg-type]
    384                 else:
    385                     sampler = SequentialSampler(dataset)  # type: ignore[arg-type]

/usr/local/lib/python3.11/dist-packages/torch/utils/data/sampler.py in __init__(self, data_source, replacement, num_samples, generator)
    163 
    164         if not isinstance(self.num_samples, int) or self.num_samples <= 0:
--> 165             raise ValueError(
    166                 f"num_samples should be a positive integer value, but got num_samples={self.num_samples}"
    167             )

ValueError: num_samples should be a positive integer value, but got num_samples=0

## === cell 5
all_names = []
all_preds = []

vit_model.eval()
eff_model.eval()
linear_head.eval()

with torch.no_grad():
    for batch_idx, (vit_inputs, eff_inputs, filenames) in enumerate(test_loader):
        bs = len(filenames)

        if tta:
            vit_inputs = torch.cat(vit_inputs, dim=0).to(device, non_blocking=True)
            eff_inputs = torch.cat(eff_inputs, dim=0).to(device, non_blocking=True)
            filenames = list(filenames)

            vit_outputs = vit_model(vit_inputs)
            eff_outputs = eff_model(eff_inputs)

            vit_batch_logits = torch.stack(torch.split(vit_outputs, bs), dim=0)
            vit_mean_logits = torch.mean(vit_batch_logits, dim=0)

            eff_batch_logits = torch.stack(torch.split(eff_outputs, bs), dim=0)
            eff_mean_logits = torch.mean(eff_batch_logits, dim=0)

            logit_inputs = torch.cat([vit_mean_logits, eff_mean_logits], dim=1)
            outputs = linear_head(logit_inputs)

            mean_preds = normalizer(outputs)
            pred_labels = torch.argmax(mean_preds, 1).tolist()
        else:
            vit_inputs = vit_inputs.to(device, non_blocking=True)
            eff_inputs = eff_inputs.to(device, non_blocking=True)
            filenames = list(filenames)

            vit_outputs = vit_model(vit_inputs)
            eff_outputs = eff_model(eff_inputs)

            logit_inputs = torch.cat([vit_outputs, eff_outputs], dim=1)
            outputs = linear_head(logit_inputs)

            preds = normalizer(outputs)
            pred_labels = torch.argmax(preds, 1).tolist()

        all_names.extend(filenames)
        all_preds.extend(pred_labels)



## === cell 6
sample_sub = pd.read_csv(sample_sub_path)
pred_map = dict(zip(all_names, all_preds))

sample_sub["label"] = sample_sub["image_id"].map(pred_map)
sample_sub["label"] = sample_sub["label"].fillna(0).astype(int)

sample_sub.to_csv("submission.csv", index=False)
sample_sub
