# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.8990631610758537

# 6. Current score

0.1136

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.05531) has done: 'I fix the immediate runtime blocker by removing the hard dependency on missing `/kaggle/input/...` weight files and instead instantiate the same kind of inference pipeline with a torchvision backbone that is available in the Kaggle environment. I also fix submission correctness issues: `shuffle=True` causes misalignment/duplicates and the current TTA dataset/collation returns nested lists that don’t batch safely, which can lead to wrong row counts. To preserve the core “two-model ensemble + optional TTA + softmax + argmax” semantics, I keep the same structure but implement TTA as a fixed, deterministic set of transforms applied per-image and aggregate logits across TTA samples. Finally, I ensure the submission exactly matches `sample_submission.csv` ordering/length and always writes `submission.csv`.'
- What this solution (achieved 0.1136) has done: 'I fix the ViT input-size mismatch causing the runtime AssertionError by resizing ViT inputs to the model’s required `image_size` (224 for `vit_b_16`) instead of 384, while keeping the same two-model ensemble + optional TTA + softmax + argmax semantics. I also correct a small ensemble weighting bug where logits are divided by 2.0 after applying weights (this unintentionally shrinks logits and can hurt accuracy); the weighted sum be used directly. Finally, I keep `shuffle=False` and ensure the submission is aligned to `sample_submission.csv` and always written as `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
from torchvision.transforms import InterpolationMode, v2
import torchvision



## === cell 1
SEED = 3407
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

cudnn.deterministic = False
cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

DATA_ROOT = Path("/kaggle/input/cassava-leaf-disease-classification")
test_dir = str(DATA_ROOT / "test_images")
sample_sub_path = str(DATA_ROOT / "sample_submission.csv")

eff_img_size = 528
vit_img_size = 224

batch_size = 16
num_workers = 4
num_classes = 5
tta = True


def build_backbone(model_name: str):
    if model_name == "vit_b_16":
        weights = torchvision.models.ViT_B_16_Weights.DEFAULT
        model = torchvision.models.vit_b_16(weights=weights)
        in_features = model.heads.head.in_features
        model.heads.head = torch.nn.Linear(in_features, num_classes)
        input_size = model.image_size  # should be 224
    elif model_name == "efficientnet_b0":
        weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
        model = torchvision.models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = torch.nn.Linear(in_features, num_classes)
        input_size = eff_img_size
    else:
        raise ValueError(f"Unknown model_name: {model_name}")
    return model.to(device), input_size


vit_model, vit_expected_size = build_backbone("vit_b_16")
eff_model, _ = build_backbone("efficientnet_b0")

assert (
    vit_expected_size == vit_img_size
), f"vit_img_size must match model.image_size ({vit_expected_size})"

vit_model.eval()
eff_model.eval()

normalizer = torch.nn.Softmax(dim=1)



## === cell 2
base_transform = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def tta_variants(img):
    variants = [img]
    variants.append(v2.functional.hflip(img))
    variants.append(v2.functional.vflip(img))
    variants.append(
        v2.functional.rotate(
            img, 90, interpolation=InterpolationMode.BICUBIC, expand=False
        )
    )
    variants.append(
        v2.functional.rotate(
            img, 180, interpolation=InterpolationMode.BICUBIC, expand=False
        )
    )
    variants.append(
        v2.functional.rotate(
            img, 270, interpolation=InterpolationMode.BICUBIC, expand=False
        )
    )
    return variants


class CassavaDataset(VisionDataset):
    """Inference dataset returning tensors shaped for easy batching.

    Returns:
      vit_imgs: Tensor [T, 3, vit, vit] if TTA else [3, vit, vit]
      eff_imgs: Tensor [T, 3, eff, eff] if TTA else [3, eff, eff]
      filename: str
    """

    def __init__(
        self, data_dir, vit_size, efficient_size, transform=None, use_tta=False
    ):
        super().__init__(root=data_dir)
        self.transform = transform
        self.use_tta = use_tta

        self.images = sorted(
            [f for f in os.listdir(data_dir) if f.lower().endswith(".jpg")]
        )

        self.cc = v2.CenterCrop((600, 600))
        self.resize_vit = v2.Resize(
            (vit_size, vit_size), interpolation=InterpolationMode.BICUBIC
        )
        self.resize_eff = v2.Resize(
            (efficient_size, efficient_size), interpolation=InterpolationMode.BICUBIC
        )

    def __getitem__(self, idx):
        filename = self.images[idx]
        path = os.path.join(self.root, filename)

        img = Image.open(path).convert("RGB")
        img = self.cc(img)

        vit_img = self.resize_vit(img)
        eff_img = self.resize_eff(img)

        if self.use_tta:
            vit_list = tta_variants(vit_img)
            eff_list = tta_variants(eff_img)

            vit_t = torch.stack([self.transform(x) for x in vit_list], dim=0)
            eff_t = torch.stack([self.transform(x) for x in eff_list], dim=0)
            return vit_t, eff_t, filename

        vit_t = self.transform(vit_img)
        eff_t = self.transform(eff_img)
        return vit_t, eff_t, filename

    def __len__(self):
        return len(self.images)


test_dataset = CassavaDataset(
    test_dir, vit_img_size, eff_img_size, transform=base_transform, use_tta=tta
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
)



## === cell 3
all_names = []
all_preds = []

vit_w = 0.45
eff_w = 0.55

with torch.no_grad():
    for vit_inputs, eff_inputs, filenames in test_loader:
        if tta:
            B, T = vit_inputs.shape[0], vit_inputs.shape[1]

            vit_flat = vit_inputs.view(B * T, *vit_inputs.shape[2:]).to(
                device, non_blocking=True
            )
            eff_flat = eff_inputs.view(B * T, *eff_inputs.shape[2:]).to(
                device, non_blocking=True
            )

            vit_logits = vit_model(vit_flat).view(B, T, num_classes).mean(dim=1)
            eff_logits = eff_model(eff_flat).view(B, T, num_classes).mean(dim=1)

            outputs = vit_w * vit_logits + eff_w * eff_logits
            probs = normalizer(outputs)
            pred_labels = torch.argmax(probs, dim=1).tolist()
        else:
            vit_inputs = vit_inputs.to(device, non_blocking=True)
            eff_inputs = eff_inputs.to(device, non_blocking=True)

            vit_logits = vit_model(vit_inputs)
            eff_logits = eff_model(eff_inputs)
            outputs = (vit_logits + eff_logits) / 2.0

            probs = normalizer(outputs)
            pred_labels = torch.argmax(probs, dim=1).tolist()

        all_names.extend(list(filenames))
        all_preds.extend(pred_labels)

print("predictions:", len(all_preds), "filenames:", len(all_names))
print("unique filenames:", len(set(all_names)))



## === cell 4
sample_sub = pd.read_csv(sample_sub_path)
pred_map = dict(zip(all_names, all_preds))

if len(pred_map) != len(sample_sub):
    if len(all_preds) > 0:
        default_label = int(pd.Series(all_preds).value_counts().idxmax())
    else:
        default_label = 0
    sample_sub["label"] = (
        sample_sub["image_id"].map(pred_map).fillna(default_label).astype(int)
    )
else:
    sample_sub["label"] = sample_sub["image_id"].map(pred_map).astype(int)

assert len(sample_sub) == 2676, f"Unexpected submission length: {len(sample_sub)}"
assert list(sample_sub.columns) == ["image_id", "label"]

out_path = "submission.csv"
sample_sub.to_csv(out_path, index=False)
print("Wrote:", out_path)
print(sample_sub.head())
