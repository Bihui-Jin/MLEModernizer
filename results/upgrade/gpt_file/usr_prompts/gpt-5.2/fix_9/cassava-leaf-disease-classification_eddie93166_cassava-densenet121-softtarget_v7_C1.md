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

3.9

# 3. Installed packages

geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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
tqdm==4.67.1

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

0.880930794802055

# 6. Current score

0.15209

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.29111) has done: 'I fix the inference crash by removing the hard-coded missing pretrained-weight path and instead using torchvision’s built-in ResNet-50 pretrained weights when no local checkpoint is available. I also fix a latent CutMix bug (undefined `lam`/deprecated `np.int`) so training mode wouldn’t error if enabled, without changing the default behavior (your current settings keep CutMix off). Finally, I make the dataset file ordering deterministic and ensure the submission is aligned to `sample_submission.csv` ordering, producing a valid `/kaggle/working/submission.csv` end-to-end.'
- What this solution (achieved 0.15172) has done: 'Main bottlenecks are (1) training 5 folds × 10 epochs on 18k images at 448×448 with small batch size (far beyond 600s), (2) slow CPU image decoding/Resize in the dataloader, and (3) extremely slow inference due to batch_size=1 plus Python-side softmax/aggregation loops and repeated CPU↔GPU syncs. To finish within 600s without changing the model/loss/training semantics, the script below (a) automatically disables training unless explicitly requested via an environment flag (so the kernel doesn’t waste >10 minutes retraining), (b) accelerates dataloading with persistent workers, pinned memory, prefetching, and faster PIL decoding settings, and (c) speeds inference by batching, using `torch.inference_mode()`, and doing softmax + fold-ensemble aggregation fully in Torch (vectorized) before a single CPU transfer. All changes preserve evaluation semantics (same transforms, same model forward, same checkpoint averaging logic) with only negligible FP differences.'
- What this solution (achieved 0.15209) has done: 'Your current low score is mainly because inference usually falls back to an ImageNet-pretrained ResNet50 with a randomly initialized 5-class head, which makes predictions near-random (~0.15–0.20 accuracy). To move the score toward the 0.88 target with minimal semantic changes, I (1) load checkpoints robustly by matching common key prefixes and handling both full-model and state_dict formats, and (2) use all available checkpoints from both the original folder and any locally trained folds (instead of choosing only one source). These changes keep your model/transform/loss/training logic intact, but greatly increase the chance you actually use trained cassava weights during inference. I also ensure strict alignment to `sample_submission.csv` ordering (no merge-induced misalignment) and keep the submission writing path unchanged.'
- What this solution (achieved 0.19694) has done: 'Your score is far below the target, and the most likely reason is that you are not actually loading cassava-trained weights (your checkpoint glob points to a dataset that often doesn’t exist, and even when it exists the files may be named differently). I make a minimal, inference-only change: broaden checkpoint discovery to include common locations under `/kaggle/input/` and accept common extensions (`.pth/.pt/.pkl/.bin`), then require that a checkpoint loads a meaningful number of tensors before using it (otherwise skip it instead of silently ensembling random heads). This keeps your model, transforms, and prediction logic the same, but greatly increases the chance you’re using real trained weights, moving accuracy toward the 0.88 target. I also fix the checkpoint path to use absolute Kaggle paths (not `../input/...`) to avoid missing files due to working-directory differences.'
- What this solution (achieved 0.15172) has done: 'Your current score (0.19694) is far below the target (0.88093), and the biggest likely cause is that you still aren’t loading any real cassava-trained checkpoints, so inference is effectively “ImageNet backbone + random 5-class head.” I make a minimal, inference-only change to checkpoint discovery so it specifically finds likely cassava fold checkpoints (common dataset names/filenames under `/kaggle/input/`) instead of the overly-broad glob that mostly returns irrelevant `.pth/.pt` files. I also tighten the “usable checkpoint” filter to require that the classifier head (`fc.weight`/`fc.bias`) is loaded with the correct shape, which prevents ensembling models whose head stayed random (the main failure mode). These changes preserve your model architecture, transforms, loss, and prediction logic, but should move accuracy substantially toward the target by actually using cassava-trained weights when present.'
- What this solution (achieved 0.15209) has done: 'Your score is far below the 0.88 target, and the most likely cause is that you’re still not loading any cassava-trained weights because your “usable checkpoint” filter is too strict and incorrectly assumes a ResNet50 head of shape (5, 2048) even though your saved training checkpoints come from `torchvision.models.resnet50` whose `fc.in_features` is 2048 but your saved filenames indicate `resnext50_32x4d` and, more importantly, many Kaggle cassava checkpoints use different key names (e.g., `classifier.*`) or are full-model checkpoints. I make a minimal inference-only change: robustly infer the classifier head keys/shapes from the current model (instead of hardcoding `fc.weight` shape), and accept checkpoints that load a meaningful fraction of tensors and successfully load the head for 5 classes. This preserves your model architecture/training code and only increases the chance that inference uses real trained cassava weights (which should move accuracy sharply toward the target). I also ensure we don’t accidentally ensemble lots of irrelevant checkpoints by capping the number of checkpoints used (deterministically) to keep runtime within 600s.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import torch

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = (
    True  # fixed input size -> safe speedup without semantic change
)

os.environ.setdefault("PILLOW_JPEG_FAST", "1")
os.environ.setdefault("PILLOW_JPEG_PROGRESSIVE", "1")

if "TRAINING" not in globals():
    TRAINING = True
if TRAINING and os.environ.get("RUN_TRAINING", "0") != "1":
    print(
        "Auto-disabling TRAINING to meet the 600s timeout. Set env RUN_TRAINING=1 to enable training."
    )
    TRAINING = False

if "BATCH_SIZE" not in globals():
    BATCH_SIZE = 8
if "EPOCH" not in globals():
    EPOCH = 10
if "WD" not in globals():
    WD = 1e-4
if "LR" not in globals():
    LR = 0.0001
if "VAL_RATIO" not in globals():
    VAL_RATIO = 0.2
if "PHASE" not in globals():
    PHASE = ["train", "val"]
if "BETA" not in globals():
    BETA = 0.0
if "CUTMIX_PROB" not in globals():
    CUTMIX_PROB = 0.0
if "K_FOLD" not in globals():
    K_FOLD = 5



## === cell 1
from torch.utils.data.dataset import Dataset
import glob
import pandas as pd
from PIL import Image


class CLD_Dataset(Dataset):
    def __init__(self, image_root, label_path=None, transform=None, return_name=False):
        super(CLD_Dataset, self).__init__()
        self.transform = transform
        self.image_paths = sorted(glob.glob(os.path.join(image_root, "*.jpg")))
        if not return_name:
            if label_path is None:
                raise ValueError("label_path must be provided when return_name=False")
            self.label = pd.read_csv(label_path, index_col="image_id")
        self.return_name = return_name

    def set_transform(self, transform):
        self.transform = transform

    def __getitem__(self, x):
        img_path = self.image_paths[x]
        with Image.open(img_path) as im:
            img = im.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        if self.return_name:
            return img, os.path.basename(img_path)
        else:
            image_id = os.path.basename(img_path)
            label = self.label.loc[image_id].label
            return img, int(label)

    def __len__(self):
        return len(self.image_paths)




## === cell 2
import torchvision.transforms as transform
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold

train_transform = transform.Compose(
    [
        transform.Resize((448, 448)),
        transform.RandomHorizontalFlip(),
        transform.ToTensor(),
        transform.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

val_transform = transform.Compose(
    [
        transform.Resize((448, 448)),
        transform.ToTensor(),
        transform.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

all_train_dataset = CLD_Dataset(
    "/kaggle/input/cassava-leaf-disease-classification/train_images",
    "/kaggle/input/cassava-leaf-disease-classification/train.csv",
    train_transform,
)
dataset_size = len(all_train_dataset)


def _make_loader(ds, batch_size, shuffle):
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True,
        prefetch_factor=4,
    )


fold_dataloader = []
if K_FOLD != 1:
    kf = KFold(K_FOLD, shuffle=True, random_state=42)

    index = 0
    for train_idx, val_idx in kf.split(all_train_dataset):
        train_dataset = torch.utils.data.Subset(all_train_dataset, train_idx)
        val_dataset = torch.utils.data.Subset(all_train_dataset, val_idx)
        fold_dataloader.append(
            {
                "train": _make_loader(train_dataset, BATCH_SIZE, True),
                "val": _make_loader(val_dataset, BATCH_SIZE, False),
            }
        )
        index += 1
    print(f"Prepared {len(fold_dataloader)} folds.")
else:
    fold_dataloader.append(
        {
            "train": _make_loader(all_train_dataset, BATCH_SIZE, True),
            "val": None,
        }
    )



## === cell 3
import torch.nn as nn



## === cell 4
__all__ = [
    "ResNet",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "resnext50_32x4d",
    "resnext101_32x8d",
    "wide_resnet50_2",
    "wide_resnet101_2",
]


model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
    "resnext50_32x4d": "https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth",
    "resnext101_32x8d": "https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth",
    "wide_resnet50_2": "https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth",
    "wide_resnet101_2": "https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth",
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ["downsample"]

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ["downsample"]

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(
        self,
        block,
        layers,
        num_classes=1000,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
    ):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2]
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        print(
            "Load pretrain : requested=True, but custom local weights are not bundled here."
        )
        print("Load : 0 layers")
        print("Miss : 0 layers")
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    return _resnet("resnet18", BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)


def resnet34(pretrained=False, progress=True, **kwargs):
    return _resnet("resnet34", BasicBlock, [3, 4, 6, 3], pretrained, progress, **kwargs)


def resnet50(pretrained=False, progress=True, **kwargs):
    return _resnet("resnet50", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)


def resnet101(pretrained=False, progress=True, **kwargs):
    return _resnet(
        "resnet101", Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs
    )


def resnet152(pretrained=False, progress=True, **kwargs):
    return _resnet(
        "resnet152", Bottleneck, [3, 8, 36, 3], pretrained, progress, **kwargs
    )


def resnext50_32x4d(pretrained=False, progress=True, **kwargs):
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 4
    return _resnet(
        "resnext_50_32x4d", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs
    )


def resnext101_32x8d(pretrained=False, progress=True, **kwargs):
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 8
    return _resnet(
        "resnext101_32x8d", Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs
    )


def wide_resnet50_2(pretrained=False, progress=True, **kwargs):
    kwargs["width_per_group"] = 64 * 2
    return _resnet(
        "wide_resnet50_2", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs
    )


def wide_resnet101_2(pretrained=False, progress=True, **kwargs):
    kwargs["width_per_group"] = 64 * 2
    return _resnet(
        "wide_resnet101_2", Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs
    )




## === cell 5
import torchvision

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"
print(device)


def create_new_model(pretrained=True):
    if pretrained:
        m = torchvision.models.resnet50(
            weights=torchvision.models.ResNet50_Weights.DEFAULT
        )
        m.fc = nn.Linear(m.fc.in_features, 5)
        return m.to(device)
    return resnet50(num_classes=5, pretrained=False).to(device)




## === cell 6
def create_loss_opti():
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=1, eta_min=1e-6, last_epoch=-1
    )
    return criterion, optimizer, lr_scheduler




## === cell 7
def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2




## === cell 8
class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, acc):
        self.reset()
        self.acc = acc

    def reset(self):
        self.value = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, value, batch):
        self.value = value
        if self.acc:
            self.sum += value
        else:
            self.sum += value * batch
        self.count += batch
        self.avg = self.sum / self.count




## === cell 9
def train_step(model, criterion, optimizer, image, label, phase):
    b_image = image.to(device, non_blocking=True)
    b_label = label.to(device, non_blocking=True)

    r = np.random.rand(1)
    if BETA > 0 and r < CUTMIX_PROB:
        lam = np.random.beta(BETA, BETA)
        rand_index = torch.randperm(b_image.size()[0]).to(device)
        target_a = b_label
        target_b = b_label[rand_index]
        bbx1, bby1, bbx2, bby2 = rand_bbox(b_image.size(), lam)
        b_image[:, :, bbx1:bbx2, bby1:bby2] = b_image[
            rand_index, :, bbx1:bbx2, bby1:bby2
        ]
        lam = 1 - (
            (bbx2 - bbx1) * (bby2 - bby1) / (b_image.size()[-1] * b_image.size()[-2])
        )

        output = model(b_image)
        loss = criterion(output, target_a) * lam + criterion(output, target_b) * (
            1.0 - lam
        )
    else:
        output = model(b_image)
        loss = criterion(output, b_label)

    _, predicted = torch.max(output.data, dim=1)
    correct = (predicted.cpu() == label).sum().item()
    if phase == "train":
        optimizer.zero_grad(set_to_none=True)  # faster, identical grads semantics
        loss.backward()
        optimizer.step()

    return correct, loss.item()




## === cell 10
from tqdm import tqdm

max_acc = 0.0
ACCMeter = []
LOSSMeter = []
for i in range(K_FOLD):
    ACCMeter.append(AverageMeter(True))
    LOSSMeter.append(AverageMeter(False))

if TRAINING:
    for index, dataloader in enumerate(fold_dataloader):
        model = create_new_model()
        criterion, optimizer, lr_scheduler = create_loss_opti()
        Best_ACC = 0.0
        tmp_ACCMeter = AverageMeter(True)
        tmp_LOSSMeter = AverageMeter(False)
        for epoch in range(1, EPOCH + 1):
            correct_t = 0
            total = 0
            loss_t = 0.0
            for phase in PHASE:
                if phase == "train":
                    model.train(True)
                    all_train_dataset.set_transform(train_transform)
                else:
                    model.train(False)
                    all_train_dataset.set_transform(val_transform)

                for image, label in tqdm(
                    dataloader[phase],
                    total=len(dataloader[phase]),
                    position=0,
                    leave=True,
                ):
                    correct, loss = train_step(
                        model, criterion, optimizer, image, label, phase
                    )

                    if phase == "val":
                        tmp_ACCMeter.update(correct, label.size(0))
                        tmp_LOSSMeter.update(loss, label.size(0))
                        total += label.size(0)
                        loss_t += loss * label.size(0)
                        correct_t += correct

                if phase == "val" and Best_ACC < tmp_ACCMeter.avg:
                    Best_ACC = tmp_ACCMeter.avg
                    ACCMeter[index] = tmp_ACCMeter
                    LOSSMeter[index] = tmp_LOSSMeter
                    torch.save(
                        model.state_dict(),
                        "./resnext50_32x4d_kfold_{}_{}_{:.2f}.pkl".format(
                            index + 1, epoch, tmp_ACCMeter.avg
                        ),
                    )

            lr_scheduler.step()
            print(
                "Fold : {}/ {} Epoch : {} / {} loss : {:.6f} ACC : {:.6f}".format(
                    index + 1, K_FOLD, epoch, EPOCH, loss_t / total, correct_t / total
                )
            )



## === cell 11
acc_sum = 0
loss_sum = 0
for i in range(K_FOLD):
    acc_sum += ACCMeter[i].avg
    loss_sum += LOSSMeter[i].avg

print("K-fold {} ACC : {:.6f}".format(K_FOLD, acc_sum / K_FOLD))
print("K-fold {} ACC : {:.6f}".format(K_FOLD, loss_sum / K_FOLD))



## === cell 12
import pandas as pd
import glob
from torch.utils.data import DataLoader


def _extract_state_dict(obj):
    if isinstance(obj, dict):
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return obj["state_dict"]
        if "model" in obj and isinstance(obj["model"], dict):
            return obj["model"]
        return obj
    return None


def _strip_prefix_if_present(sd, prefix):
    if not isinstance(sd, dict):
        return sd
    if any(k.startswith(prefix) for k in sd.keys()):
        return {k[len(prefix) :]: v for k, v in sd.items() if k.startswith(prefix)}
    return sd


def _infer_head_keys(model_sd):
    for wkey, bkey in [
        ("fc.weight", "fc.bias"),
        ("classifier.weight", "classifier.bias"),
        ("head.weight", "head.bias"),
        ("model.fc.weight", "model.fc.bias"),
    ]:
        if wkey in model_sd and bkey in model_sd:
            return wkey, bkey
    return None, None


def load_checkpoint_fuzzy(model, ckpt_path):
    obj = torch.load(ckpt_path, map_location="cpu")
    sd = _extract_state_dict(obj)
    if sd is None:
        print(f"WARNING: Could not parse checkpoint at {ckpt_path}; skipping.")
        return 0, 0

    for pfx in ("module.", "model.", "net."):
        sd = _strip_prefix_if_present(sd, pfx)

    model_sd = model.state_dict()
    loaded = 0
    missed = 0

    for k, v in sd.items():
        if (
            k in model_sd
            and isinstance(v, torch.Tensor)
            and model_sd[k].shape == v.shape
        ):
            model_sd[k].copy_(v)
            loaded += 1
        else:
            missed += 1

    model.load_state_dict(model_sd, strict=False)
    return loaded, missed


test_dataset = CLD_Dataset(
    "/kaggle/input/cassava-leaf-disease-classification/test_images",
    transform=val_transform,
    return_name=True,
)

TEST_BATCH_SIZE = 64 if torch.cuda.is_available() else 16
test_dataloader = DataLoader(
    test_dataset,
    batch_size=TEST_BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=True,
    prefetch_factor=4,
)


def find_ckpts():
    exts = ("*.pkl", "*.pth", "*.pt", "*.bin")
    patterns = []

    cassava_roots = [
        "/kaggle/input/cassava-leaf-disease-classification",
        "/kaggle/input/cassava-leaf-disease",
        "/kaggle/input/cassava",
        "/kaggle/input/cassava-resnet50",
        "/kaggle/input/cassava-model",
        "/kaggle/input/cassava-models",
        "/kaggle/input/cassava-leaf-disease-model",
        "/kaggle/input/cassava-leaf-disease-classification-model",
        "/kaggle/input/cassava-leaf-disease-classification-models",
    ]

    name_hints = [
        "cassava",
        "cld",
        "leaf",
        "disease",
        "resnet",
        "resnext",
        "efficientnet",
        "fold",
        "kfold",
        "best",
    ]

    for root in cassava_roots:
        for e in exts:
            for hint in name_hints:
                patterns.append(f"{root}/**/*{hint}*{e}")

    patterns.append("./resnext50_32x4d_kfold_*.pkl")
    patterns.append("./resnext50_32x4d_kfold_*.pth")
    patterns.append("./resnext50_32x4d_kfold_*.pt")

    for e in exts:
        patterns.append(f"/kaggle/input/**/cassava*/*{e}")
        patterns.append(f"/kaggle/input/**/cassava*/*/*{e}")
        patterns.append(f"/kaggle/input/**/cassava*/*/*/*{e}")

    ckpts = []
    for pat in patterns:
        ckpts.extend(glob.glob(pat, recursive=True))
    ckpts = sorted(set(ckpts))
    return ckpts


ckpt_paths_all = find_ckpts()

usable_ckpts = []
for p in ckpt_paths_all:
    try:
        m = create_new_model(pretrained=True)  # same architecture each time
        l, miss = load_checkpoint_fuzzy(m, p)

        msd = m.state_dict()
        wkey, bkey = _infer_head_keys(msd)
        head_ok = False
        if wkey is not None and bkey is not None:
            head_ok = (
                isinstance(msd[wkey], torch.Tensor)
                and isinstance(msd[bkey], torch.Tensor)
                and msd[wkey].shape[0] == 5
                and msd[bkey].shape[0] == 5
            )

        if l >= 50 and head_ok:
            usable_ckpts.append(p)
        else:
            print(f"Skipping checkpoint (matched={l}, head_ok={head_ok}): {p}")
    except Exception as e:
        print(f"Skipping checkpoint (load error): {p} -> {repr(e)}")

MAX_CKPTS = 8
usable_ckpts = sorted(usable_ckpts)[:MAX_CKPTS]

if len(usable_ckpts) == 0:
    print(
        "WARNING: No usable cassava checkpoints found. Falling back to a single torchvision ImageNet-pretrained model for inference."
    )
    ckpt_paths = [None]
else:
    ckpt_paths = usable_ckpts
    print(f"Using {len(ckpt_paths)} usable checkpoints for ensembling.")
    for p in ckpt_paths:
        print("  ckpt:", p)

all_image_names = []
for _, img_names in test_dataloader:
    all_image_names.extend(list(img_names))
num_images = len(all_image_names)

probs_sum = torch.zeros((num_images, 5), dtype=torch.float32, device=device)

for params_path in ckpt_paths:
    model = create_new_model(pretrained=(params_path is None))

    if params_path is not None:
        l, m = load_checkpoint_fuzzy(model, params_path)
        print(f"Loaded checkpoint: {params_path}")
        print(f"Trained weight load : {l}")
        print(f"Trained weight not load : {m}")

    model.train(False)
    offset = 0
    with torch.inference_mode():
        for img, _img_name in test_dataloader:
            b_img = img.to(device, non_blocking=True)
            logits = model(b_img)
            p = torch.softmax(logits, dim=1)
            p = p / (p.sum(dim=1, keepdim=True) + 1e-4)
            bs = p.shape[0]
            probs_sum[offset : offset + bs] += p
            offset += bs

image_labels = torch.argmax(probs_sum, dim=1).to("cpu").numpy().astype(int)

sub = pd.read_csv(
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)
pred_df = pd.DataFrame(
    {"image_id": np.array(all_image_names), "label": image_labels}
).set_index("image_id")
sub["label"] = sub["image_id"].map(pred_df["label"]).fillna(0).astype(int)

print(sub.head())
sub.to_csv("/kaggle/working/submission.csv", index=False)
print("Wrote /kaggle/working/submission.csv with shape:", sub.shape)
