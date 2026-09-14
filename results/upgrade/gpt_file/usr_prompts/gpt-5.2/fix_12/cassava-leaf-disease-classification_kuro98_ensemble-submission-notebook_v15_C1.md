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

0.900725294650952

# 6. Current score

0.73019

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.08221) has done: 'The main runtime blocker is that the notebook tries to `torch.load()` three pretrained models from `/kaggle/input/...` paths that are not available in your environment, which then cascades into `vit_model` being undefined. To keep the core inference logic intact while making the pipeline runnable end-to-end, I replace those missing models with lightweight “compatibility wrapper” modules that produce 5-class logits from the input images, so the rest of your TTA/ensemble code can run unchanged. I also fix the submission-length error by (1) disabling `shuffle=True` in the test DataLoader and (2) reindexing predictions to exactly match `sample_submission.csv` order, ensuring the produced `submission.csv` has the correct length and alignment.'
- What this solution (achieved 0.11584) has done: 'I fix the dataset bug causing `IsADirectoryError` by ensuring we only load real image files (and not nested `test_images/` directories) and by sorting filenames for stable, reproducible ordering. I also make TTA deterministic per-image (so it doesn’t act like random noise at inference) by replacing the random TTA transforms with fixed geometric transforms; this keeps the same “TTA averaging” core logic but should increase accuracy substantially toward your target. Finally, I keep `shuffle=False` and preserve the reindexing to `sample_submission.csv` so the produced `submission.csv` is correctly aligned and valid.'
- What this solution (achieved 0.19619) has done: 'Your current score is far below the target, so the smallest safe way to move accuracy upward is to make the “fallback” models less random and more aligned with ImageNet-style normalization you already use. I keep your exact inference/TTA/ensemble structure, but load torchvision ImageNet-pretrained backbones as the fallback (instead of training-from-scratch tiny convnets), and ensure they output 5 logits. I also fix a small bug where `linear_head` is never applied (without changing the overall ensemble logic, just applying it to the combined logits when it exists). These changes should substantially increase accuracy while keeping the pipeline end-to-end and submission format unchanged.'
- What this solution (achieved 0.19619) has done: 'Your current score is far below the target, so we should make the smallest legitimate accuracy-improving change without altering your inference/TTA/ensemble structure. The main issue is that the “fallback” models are ImageNet-pretrained but not adapted to cassava at all, so predictions are near-random; to move toward the target, we minimally add a quick fine-tuning step on `train.csv` using the same transforms and the same two backbones (ConvNeXt-Tiny + EfficientNet-B0) and the same CE loss (standard for accuracy). We keep your existing TTA averaging, model weighting, and submission alignment unchanged; we only train the classification heads (fast, stable) and then run the same inference code. This should substantially increase accuracy while staying within the 600s budget and preserving core logic.'
- What this solution (achieved 0.79372) has done: 'I fix the finetuning crash by correctly detecting which classifier layer is the trainable head for each torchvision backbone (ConvNeXt vs EfficientNet) instead of always indexing `classifier[2]`. I also make sure `linear_head` is only applied when it’s compatible with 5-class logits (otherwise safely skip it) to avoid silent shape/logic issues. These are minimal changes that unblock end-to-end execution and allow the intended “train heads then do the same TTA ensemble inference” logic to run, which should improve accuracy substantially vs the current broken finetune. The submission writing and alignment to `sample_submission.csv` are kept intact.'
- What this solution (achieved 0.79111) has done: 'Your current score (0.79372) is below the target (0.9007), so we need a modest accuracy lift without changing the overall “two-backbone + TTA + linear_head + submission alignment” structure. The biggest low-risk gain is to fix the TTA transforms: using `Random*` at inference makes the pipeline stochastic and can degrade accuracy; replacing them with deterministic geometric equivalents preserves the exact TTA-averaging logic but improves signal. I also correct a small scaling bug in your ensemble where you divide by 2 after already using weighted averaging (this unintentionally shrinks logits and can hurt argmax stability); keeping the same weights but removing the extra `/2` is a minimal semantic fix. Finally, I add a lightweight, competition-standard class-weighted CE during the existing 1-epoch head fine-tune to better handle class imbalance (no new loop/architecture, just a better loss weighting), which typically moves accuracy upward.'
- What this solution (achieved 0.78251) has done: 'You’re still materially below the target (0.79111 vs 0.90073), so we should make a small, legitimate accuracy lift while keeping the same “two-backbone + 1-epoch head finetune + TTA averaging + linear_head + aligned submission” structure. The biggest low-risk improvement is to use a simple, standard training-time augmentation pipeline (random resized crop + horizontal flip) only for finetuning, while keeping your deterministic center-crop pipeline for test-time inference unchanged; this typically boosts generalization noticeably for cassava. I also make the finetune DataLoader deterministic/reproducible by adding a fixed generator and seeding workers (this stabilizes results without changing semantics). Finally, I switch the inference normalization to use `softmax` directly on logits (no change in predictions vs your `Softmax` module) and leave ensemble weights/TTA logic intact.'
- What this solution (achieved 0.78176) has done: 'Your current score (0.78251) is well below the target (0.90073), so we need a modest but real accuracy lift while keeping your two-backbone + 1-epoch head finetune + TTA ensemble core logic unchanged. The biggest low-risk issue is a transform mismatch: you normalize before resizing in finetune, which means `Resize` is operating on a normalized tensor rather than image-like values; fixing the order (augment/crop -> resize -> normalize) usually improves head adaptation and thus test accuracy. I also make the train-time pipeline match the test-time “center-crop then resize” geometry more closely by doing the random crop at 600 then resize, but only changing ordering (no new augment types, no loop changes). Finally, I keep submission alignment logic intact and add a small safety to ensure no NaNs in logits during inference (without changing predicted argmax unless NaNs occur).'
- What this solution (achieved 0.77541) has done: 'Your current score (0.78176) is well below the target (0.90073), so we need a real accuracy lift but with minimal changes and the same overall “two-backbone + 1-epoch head finetune + TTA ensemble + aligned submission” structure. The biggest low-risk gain is to make finetuning and inference use the *same input normalization pipeline that matches each torchvision backbone’s own pretrained weights* (ConvNeXt vs EfficientNet), instead of forcing both through generic ImageNet mean/std; this usually improves transfer accuracy substantially. To keep core logic intact, I only (1) split transforms into vit-specific and eff-specific using `weights.transforms()` and (2) apply those transforms consistently in both finetune and test dataset, keeping your exact loaders, epochs, loss, weights, and TTA averaging. Submission writing/alignment stays unchanged.'
- What this solution (achieved 0.75224) has done: 'Your current score (0.77541) is well below the target (0.9007), so we need a modest accuracy lift while keeping your same two-backbone + 1-epoch head finetune + TTA-averaging ensemble intact. The biggest low-risk issue is that your finetune uses random augmentations but does not control all RNGs per-worker, which can make the single-epoch head adaptation unstable; I make augmentation deterministic/reproducible across runs by seeding Python/NumPy and using a proper worker seeding function. Next, your finetune currently averages logits from two models without applying the same 0.55/0.45 weighting you use at inference; I make finetune match inference weighting (same semantics, just consistent), which typically improves transfer to the test-time ensemble. Finally, I add a standard “best of both worlds” tweak that doesn’t change architecture/loops: use label-smoothing CrossEntropy for the head finetune (still CE, same objective family) to reduce overconfidence and usually improve accuracy for noisy labels like cassava.'
- What this solution (achieved 0.73019) has done: 'Your current score (0.75224) is materially below the target (0.9007), so we need a real but still minimal lift without changing the overall “two-backbone + 1-epoch head finetune + TTA averaging + weighted ensemble + aligned submission” structure. The biggest low-risk accuracy gain here is to make the finetune step update BatchNorm statistics (keep BN layers trainable) while still training only the classifier heads, because freezing BN often harms transfer on cassava. I also add a minimal validation split and report accuracy during finetune, and then reload the best head weights from that split (same training loop/epochs/CE objective, just choosing the better state within the same run). Finally, I keep inference/TTA/submission logic unchanged, only ensuring determinism/reproducibility stays intact.'

# 9. Code solution

## === cell 0
import os
import random

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
from torchvision.transforms import InterpolationMode, v2

torch.manual_seed(3407)
torch.cuda.manual_seed(3407)
random.seed(3407)
np.random.seed(3407)

cudnn.deterministic = False
cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

test_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images/"
train_csv_path = "/kaggle/input/cassava-leaf-disease-classification/train.csv"
train_dir = "/kaggle/input/cassava-leaf-disease-classification/train_images/"

eff_img_size = 528
vit_img_size = 384
batch_size = 16
num_workers = 4
num_classes = 5
tta = True

finetune = True
finetune_epochs = 1
finetune_lr = 2e-3
finetune_batch_size = 16

import torch.nn as nn
import torchvision


class _FallbackCassavaModel(nn.Module):
    def __init__(self, num_classes: int = 5, arch: str = "convnext_tiny"):
        super().__init__()
        arch = (arch or "").lower()

        if arch == "convnext_tiny":
            weights = torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
            m = torchvision.models.convnext_tiny(weights=weights)
            in_features = m.classifier[2].in_features
            m.classifier[2] = nn.Linear(in_features, num_classes)
            self.model = m
            self.arch = "convnext_tiny"
        elif arch == "efficientnet_b0":
            weights = torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1
            m = torchvision.models.efficientnet_b0(weights=weights)
            in_features = m.classifier[1].in_features
            m.classifier[1] = nn.Linear(in_features, num_classes)
            self.model = m
            self.arch = "efficientnet_b0"
        else:
            self.model = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(16),
                nn.SiLU(inplace=True),
                nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.SiLU(inplace=True),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.SiLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(64, num_classes),
            )
            self.arch = "tiny_cnn"

    def forward(self, x):
        return self.model(x)


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

if vit_model is None:
    vit_model = _FallbackCassavaModel(num_classes=num_classes, arch="convnext_tiny")
if eff_model is None:
    eff_model = _FallbackCassavaModel(num_classes=num_classes, arch="efficientnet_b0")
if linear_head is None:
    linear_head = nn.Identity()

vit_model = vit_model.to(device)
eff_model = eff_model.to(device)
linear_head = linear_head.to(device)

_vit_weights = torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
_eff_weights = torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1
vit_weights_tfms = _vit_weights.transforms()
eff_weights_tfms = _eff_weights.transforms()

VIT_W = 0.55
EFF_W = 0.45




## === cell 1
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
        vit_transform=None,
        eff_transform=None,
        ttas=None,
    ):
        super().__init__(root=data_dir)

        self.vit_transform = vit_transform
        self.eff_transform = eff_transform

        valid_ext = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
        self.images = sorted(
            [
                f
                for f in os.listdir(data_dir)
                if os.path.isfile(os.path.join(data_dir, f))
                and f.lower().endswith(valid_ext)
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

    def __getitem__(self, idx):
        filename = self.images[idx]
        img = Image.open(os.path.join(self.root, filename)).convert("RGB")
        img = self.cc(img)
        vit_img = self.resize_vit(img)
        eff_img = self.resize_efficient(img)

        if (
            self.ttas is not None
            and (self.vit_transform is not None)
            and (self.eff_transform is not None)
        ):
            vit_img = [self.vit_transform(t(vit_img)) for t in self.ttas]
            eff_img = [self.eff_transform(t(eff_img)) for t in self.ttas]
        else:
            if self.vit_transform is not None:
                vit_img = self.vit_transform(vit_img)
            if self.eff_transform is not None:
                eff_img = self.eff_transform(eff_img)

        return vit_img, eff_img, filename

    def __len__(self):
        return len(self.images)




## === cell 2
if tta:
    ttas = [
        v2.Identity(),
        v2.RandomHorizontalFlip(p=1.0),  # deterministic because p=1
        v2.RandomVerticalFlip(p=1.0),  # deterministic because p=1
        v2.RandomRotation(
            degrees=(90, 90), interpolation=InterpolationMode.BILINEAR
        ),  # deterministic
    ]
else:
    ttas = None

test_dataset = CassavaDataset(
    test_dir,
    vit_img_size,
    eff_img_size,
    vit_transform=vit_weights_tfms,
    eff_transform=eff_weights_tfms,
    ttas=ttas,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
)



## === cell 3
if finetune:
    train_df = pd.read_csv(train_csv_path)

    finetune_geom = v2.Compose(
        [
            v2.ToImage(),
            v2.RandomResizedCrop(
                size=(600, 600),
                scale=(0.7, 1.0),
                ratio=(0.9, 1.1),
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ),
            v2.RandomHorizontalFlip(p=0.5),
        ]
    )

    class CassavaTrainDataset(torch.utils.data.Dataset):
        def __init__(
            self,
            df,
            data_dir,
            vit_size,
            eff_size,
            geom_transform,
            vit_transform,
            eff_transform,
        ):
            self.df = df.reset_index(drop=True)
            self.data_dir = data_dir
            self.geom_transform = geom_transform
            self.vit_transform = vit_transform
            self.eff_transform = eff_transform
            self.resize_vit = v2.Resize(
                (vit_size, vit_size), interpolation=InterpolationMode.BICUBIC
            )
            self.resize_eff = v2.Resize(
                (eff_size, eff_size), interpolation=InterpolationMode.BICUBIC
            )

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            fname = row["image_id"]
            y = int(row["label"])
            img = Image.open(os.path.join(self.data_dir, fname)).convert("RGB")

            img = self.geom_transform(img)

            vit_img = self.resize_vit(img)
            eff_img = self.resize_eff(img)

            vit_img = self.vit_transform(vit_img)
            eff_img = self.eff_transform(eff_img)

            return vit_img, eff_img, torch.tensor(y, dtype=torch.long)

    def _seed_worker(worker_id: int):
        seed = 3407 + worker_id
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    g = torch.Generator()
    g.manual_seed(3407)

    idx_all = np.arange(len(train_df))
    y_all = train_df["label"].to_numpy()
    val_frac = 0.08
    per_class = {c: np.where(y_all == c)[0] for c in range(num_classes)}
    val_idx = []
    for c, idxs in per_class.items():
        if len(idxs) == 0:
            continue
        rng = np.random.RandomState(3407 + c)
        rng.shuffle(idxs)
        k = max(1, int(round(len(idxs) * val_frac)))
        val_idx.extend(idxs[:k].tolist())
    val_idx = np.array(sorted(set(val_idx)), dtype=int)
    train_idx = np.array(
        sorted(set(idx_all.tolist()) - set(val_idx.tolist())), dtype=int
    )

    train_split_df = train_df.iloc[train_idx].reset_index(drop=True)
    val_split_df = train_df.iloc[val_idx].reset_index(drop=True)

    train_dataset = CassavaTrainDataset(
        train_split_df,
        train_dir,
        vit_img_size,
        eff_img_size,
        geom_transform=finetune_geom,
        vit_transform=vit_weights_tfms,
        eff_transform=eff_weights_tfms,
    )
    val_dataset = CassavaTrainDataset(
        val_split_df,
        train_dir,
        vit_img_size,
        eff_img_size,
        geom_transform=v2.Compose([v2.ToImage(), v2.CenterCrop((600, 600))]),
        vit_transform=vit_weights_tfms,
        eff_transform=eff_weights_tfms,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=finetune_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=_seed_worker,
        generator=g,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=finetune_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    def _set_head_only_trainable(m: nn.Module):
        for p in m.parameters():
            p.requires_grad = False

        base = getattr(m, "model", m)

        if hasattr(base, "classifier") and isinstance(base.classifier, nn.Sequential):
            head = None
            for idx in range(len(base.classifier) - 1, -1, -1):
                if isinstance(base.classifier[idx], nn.Linear):
                    head = base.classifier[idx]
                    break
            if head is not None:
                for p in head.parameters():
                    p.requires_grad = True
                return list(head.parameters())

        if (
            isinstance(base, nn.Sequential)
            and len(base) > 0
            and isinstance(base[-1], nn.Linear)
        ):
            for p in base[-1].parameters():
                p.requires_grad = True
            return list(base[-1].parameters())

        return []

    def _ensure_compatible_linear_head(head: nn.Module, num_classes: int, device):
        if head is None:
            return nn.Identity().to(device)
        head = head.to(device)
        if isinstance(head, nn.Identity):
            return head
        try:
            head.eval()
            with torch.no_grad():
                dummy = torch.zeros(2, num_classes, device=device)
                out = head(dummy)
            if isinstance(out, torch.Tensor) and out.shape == dummy.shape:
                return head
        except Exception:
            pass
        return nn.Identity().to(device)

    linear_head = _ensure_compatible_linear_head(
        linear_head, num_classes=num_classes, device=device
    )

    vit_params = _set_head_only_trainable(vit_model)
    eff_params = _set_head_only_trainable(eff_model)

    def _set_bn_train(m: nn.Module):
        for mod in m.modules():
            if isinstance(mod, nn.BatchNorm2d):
                mod.train()

    trainable = vit_params + eff_params
    if len(trainable) == 0:
        finetune = False
        print("No trainable head parameters detected; skipping finetune.")
    else:
        optimizer = torch.optim.AdamW(trainable, lr=finetune_lr, weight_decay=1e-2)

        label_counts = train_df["label"].value_counts().sort_index()
        counts = torch.tensor(
            [label_counts.get(i, 0) for i in range(num_classes)], dtype=torch.float32
        )
        counts = torch.clamp(counts, min=1.0)
        class_weights = counts.sum() / counts
        class_weights = class_weights / class_weights.mean()
        class_weights = class_weights.to(device)

        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)

        best_acc = -1.0
        best_state = None

        for epoch in range(finetune_epochs):
            vit_model.train()
            eff_model.train()
            linear_head.train()
            _set_bn_train(vit_model)
            _set_bn_train(eff_model)

            running_loss = 0.0
            seen = 0
            for vit_x, eff_x, y in train_loader:
                vit_x = vit_x.to(device, non_blocking=True)
                eff_x = eff_x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                vit_logits = vit_model(vit_x)
                eff_logits = eff_model(eff_x)

                logits = (VIT_W * vit_logits) + (EFF_W * eff_logits)
                logits = linear_head(logits)

                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()

                bs = y.size(0)
                running_loss += loss.item() * bs
                seen += bs

            vit_model.eval()
            eff_model.eval()
            linear_head.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for vit_x, eff_x, y in val_loader:
                    vit_x = vit_x.to(device, non_blocking=True)
                    eff_x = eff_x.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)
                    vit_logits = vit_model(vit_x)
                    eff_logits = eff_model(eff_x)
                    logits = (VIT_W * vit_logits) + (EFF_W * eff_logits)
                    logits = linear_head(logits)
                    pred = torch.argmax(logits, dim=1)
                    correct += (pred == y).sum().item()
                    total += y.numel()
            val_acc = correct / max(1, total)

            print(
                f"finetune epoch {epoch+1}/{finetune_epochs} - loss: {running_loss/max(seen,1):.4f} - val_acc: {val_acc:.4f}"
            )

            if val_acc > best_acc:
                best_acc = val_acc
                best_state = {
                    "vit": {
                        k: v.detach().cpu().clone()
                        for k, v in vit_model.state_dict().items()
                    },
                    "eff": {
                        k: v.detach().cpu().clone()
                        for k, v in eff_model.state_dict().items()
                    },
                    "lin": {
                        k: v.detach().cpu().clone()
                        for k, v in linear_head.state_dict().items()
                    },
                }

        if best_state is not None:
            vit_model.load_state_dict(best_state["vit"], strict=True)
            eff_model.load_state_dict(best_state["eff"], strict=True)
            linear_head.load_state_dict(best_state["lin"], strict=True)

        vit_model.eval()
        eff_model.eval()
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
            vit_inputs = torch.cat(vit_inputs, dim=0).to(device)
            eff_inputs = torch.cat(eff_inputs, dim=0).to(device)
            filenames = list(filenames)

            vit_outputs = vit_model(vit_inputs)
            eff_outputs = eff_model(eff_inputs)

            vit_batch_logits = torch.stack(torch.split(vit_outputs, bs), dim=0)
            vit_mean_logits = torch.mean(vit_batch_logits, dim=0)

            eff_batch_logits = torch.stack(torch.split(eff_outputs, bs), dim=0)
            eff_mean_logits = torch.mean(eff_batch_logits, dim=0)

            outputs = VIT_W * vit_mean_logits + EFF_W * eff_mean_logits
            outputs = linear_head(outputs)

            outputs = torch.nan_to_num(outputs, nan=0.0, posinf=1e4, neginf=-1e4)

            pred_labels = torch.argmax(outputs.softmax(dim=1), 1).tolist()
        else:
            vit_inputs = vit_inputs.to(device)
            eff_inputs = eff_inputs.to(device)
            filenames = list(filenames)

            vit_outputs = vit_model(vit_inputs)
            eff_outputs = eff_model(eff_inputs)

            outputs = (VIT_W * vit_outputs) + (EFF_W * eff_outputs)
            outputs = linear_head(outputs)
            outputs = torch.nan_to_num(outputs, nan=0.0, posinf=1e4, neginf=-1e4)

            pred_labels = torch.argmax(outputs.softmax(dim=1), 1).tolist()

        all_names.extend(filenames)
        all_preds.extend(pred_labels)



## === cell 5
sample_path = "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
sample_sub = pd.read_csv(sample_path)

pred_df = pd.DataFrame({"image_id": all_names, "label": all_preds})
pred_df = pred_df.drop_duplicates(subset=["image_id"], keep="first").set_index(
    "image_id"
)

aligned = pred_df.reindex(sample_sub["image_id"])
aligned["label"] = aligned["label"].fillna(0).astype(int)

my_submission = pd.DataFrame(
    {"image_id": sample_sub["image_id"], "label": aligned["label"].values}
)
my_submission.to_csv("submission.csv", index=False)

print("Saved submission.csv with shape:", my_submission.shape)
print(my_submission.head())
