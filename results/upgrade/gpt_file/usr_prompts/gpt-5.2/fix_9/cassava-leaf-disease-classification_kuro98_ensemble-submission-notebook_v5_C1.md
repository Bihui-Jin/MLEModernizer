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


def _resolve_images_dir(images_path: str) -> str:
    images_path = _resolve_path(images_path)

    if os.path.isfile(images_path):
        images_path = os.path.dirname(images_path)

    candidates = [
        images_path,
        os.path.join(images_path, "train_images"),
        os.path.join(images_path, "test_images"),
        os.path.join(images_path, "train_images", "train_images"),
        os.path.join(images_path, "test_images", "test_images"),
    ]

    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")

    def has_images(d: str) -> bool:
        if not os.path.isdir(d):
            return False
        try:
            for n in os.listdir(d):
                fp = os.path.join(d, n)
                if os.path.isfile(fp) and n.lower().endswith(exts):
                    return True
        except Exception:
            return False
        return False

    for c in candidates:
        if has_images(c):
            return c

    for c in candidates:
        if os.path.isdir(c):
            return c

    return images_path


test_dir = _resolve_images_dir(
    "/kaggle/input/cassava-leaf-disease-classification/test_images/"
)
train_dir = _resolve_images_dir(
    "/kaggle/input/cassava-leaf-disease-classification/train_images/"
)

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
fallback_epochs = 6
fallback_lr = 1e-3


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




## === cell 1
class CassavaDataset(VisionDataset):
    """Custom dataset for the Cassava data.

    Args:
        data_dir: base directory to the images.
        transforms: set of transforms to be used.
        image_ids: optional explicit list of image filenames to use (e.g., from CSV).
    """

    def __init__(
        self,
        data_dir,
        vit_size,
        efficient_size,
        vit_transform=None,  # allow per-backbone normalization
        eff_transform=None,  # allow per-backbone normalization
        ttas=None,
        img_size=384,
        labels_df=None,  # optional (for training head fallback)
        image_ids=None,  # optional explicit list of filenames
    ):
        data_dir = _resolve_path(data_dir)
        exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")

        def list_images(d):
            if not os.path.isdir(d):
                return []
            names = os.listdir(d)
            return sorted(
                [
                    n
                    for n in names
                    if os.path.isfile(os.path.join(d, n)) and n.lower().endswith(exts)
                ]
            )

        imgs_here = list_images(data_dir)
        if len(imgs_here) == 0:
            base = data_dir
            candidates = [
                os.path.join(base, "train_images"),
                os.path.join(base, "test_images"),
                os.path.join(base, os.path.basename(base)),
            ]
            for c in candidates:
                imgs_c = list_images(c)
                if len(imgs_c) > 0:
                    data_dir = c
                    imgs_here = imgs_c
                    break

        super().__init__(root=data_dir)

        self.vit_transform = vit_transform
        self.eff_transform = eff_transform
        self.labels_df = labels_df

        if image_ids is not None:
            imgs = []
            for n in image_ids:
                fp = os.path.join(self.root, n)
                if os.path.isfile(fp) and n.lower().endswith(exts):
                    imgs.append(n)
            self.images = imgs
        else:
            self.images = imgs_here

        self.ttas = ttas

        self.cc = v2.CenterCrop((600, 600))
        self.resize_vit = v2.Resize(
            (vit_size, vit_size),
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )
        self.resize_efficient = v2.Resize(
            (efficient_size, efficient_size),
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
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

        if self.ttas is not None:
            vit_imgs = [t(vit_img) for t in self.ttas]
            eff_imgs = [t(eff_img) for t in self.ttas]
            if self.vit_transform is not None:
                vit_imgs = [self.vit_transform(x) for x in vit_imgs]
            if self.eff_transform is not None:
                eff_imgs = [self.eff_transform(x) for x in eff_imgs]
            vit_img, eff_img = vit_imgs, eff_imgs
        else:
            if self.vit_transform is not None:
                vit_img = self.vit_transform(vit_img)
            if self.eff_transform is not None:
                eff_img = self.eff_transform(eff_img)

        if self._label_map is None:
            return vit_img, eff_img, filename
        else:
            y = int(self._label_map[filename])
            return vit_img, eff_img, y, filename

    def __len__(self):
        return len(self.images)




## === cell 2
vit_transforms = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # ViT_B_16 default
    ]
)

eff_transforms = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),  # EfficientNet default
    ]
)

if tta:
    ttas = [
        v2.RandomHorizontalFlip(p=1.0),
        v2.RandomVerticalFlip(p=1.0),
        v2.Identity(),
    ]
else:
    ttas = None

test_dataset = CassavaDataset(
    test_dir,
    vit_img_size,
    eff_img_size,
    vit_transform=vit_transforms,
    eff_transform=eff_transforms,
    ttas=ttas,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
)

normalizer = torch.nn.Softmax(dim=1)



## === cell 3
need_head_training = (head_sd is None) and do_fallback_head_train

if need_head_training:
    import torch.nn as nn

    train_df = pd.read_csv(train_csv_path)
    train_image_ids = train_df["image_id"].astype(str).tolist()

    train_dataset = CassavaDataset(
        train_dir,
        vit_img_size,
        eff_img_size,
        vit_transform=vit_transforms,
        eff_transform=eff_transforms,
        ttas=None,
        labels_df=train_df,
        image_ids=train_image_ids,
    )

    if len(train_dataset) == 0:
        raise RuntimeError(
            f"Training dataset is empty after filtering. train_dir={train_dir} (resolved root={train_dataset.root})"
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



## === cell 4
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



## === cell 5
sample_sub = pd.read_csv(sample_sub_path)
pred_map = dict(zip(all_names, all_preds))

sample_sub["label"] = sample_sub["image_id"].map(pred_map)
sample_sub["label"] = sample_sub["label"].fillna(0).astype(int)

sample_sub.to_csv("submission.csv", index=False)
sample_sub
