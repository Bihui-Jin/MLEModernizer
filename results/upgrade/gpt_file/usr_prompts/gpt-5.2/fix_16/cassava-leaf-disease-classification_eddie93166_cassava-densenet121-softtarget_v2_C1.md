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

0.8627984285282563

# 6. Current score

0.1207

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.60575) has done: 'I fix the missing pretrained weight dependency by switching the ResNeXt50 initialization to use torchvision’s built-in pretrained weights (available offline in the Kaggle image) instead of loading `../input/resnext50-32x4d/...`. I also fix a CutMix helper bug (`rand_bbox` used an undefined `lam` and deprecated `np.int`) to prevent runtime issues if TRAINING is enabled later, without changing the training logic. For inference, I ensure the model is put in `eval()` and wrap prediction in `torch.no_grad()` for correctness and speed, while keeping the same transforms and argmax labeling. Finally, the script always write a valid `/kaggle/working/submission.csv` with the required columns.'
- What this solution (achieved 0.25037) has done: 'Your current score (0.60575) is far below the target (0.8628), so we should safely improve inference quality without changing the core model/training logic. The biggest issue is that you are using the *training* transform (random horizontal flip) during test-time inference, which injects randomness and degrades accuracy; switching to a deterministic eval transform typically yields a large score gain. I also make test ordering deterministic (sort image paths) and increase inference batch size (no semantic change, just faster) while keeping the same architecture, weights loading, and argmax labeling. These are minimal, evaluation-aligned changes that should move the score substantially toward the target.'
- What this solution (achieved 0.10463) has done: 'Your current score (0.25037) is far below the target (0.8628), so we should make the smallest fixes that restore the intended inference behavior of your fine-tuned checkpoint. The biggest likely issue is that `WEIGHT` points to a path that doesn’t exist in this environment, so the model is effectively running with ImageNet-pretrained/random head weights, which collapses accuracy; I switch to loading the checkpoint from `/kaggle/input/` (and auto-discover a `.pth/.pt/.bin` file if the exact name differs) without changing the model. I also ensure we load the checkpoint strictly when possible (to avoid silently missing tensors) and keep eval-time transforms deterministic as you already set. These are minimal, execution-safe changes that should move the score substantially toward the target while preserving the architecture and inference semantics (argmax over 5 logits).'
- What this solution (achieved 0.59529) has done: 'Your score (0.10463) is far below the target (0.8628), and the most likely cause is that the fine-tuned checkpoint is still not being loaded (or is being loaded but with mismatched keys/shapes), so the model predicts near-random. I make checkpoint loading strict-but-informative: auto-detect common wrappers (`state_dict`, `model`, `net`), strip prefixes (`module.`, `model.`), and only load tensors whose shapes match the current model; then I print how many keys actually loaded so you can confirm it’s not silently failing. If no checkpoint is found or nothing matches, I fail fast with a clear error (instead of producing a bad submission), because generating random predictions guarantees a very low score. These changes do not alter the model architecture, transforms, or argmax inference semantics—they only ensure the intended trained weights are actually used.'
- What this solution (achieved 0.11622) has done: 'I fix the immediate runtime failure by making checkpoint discovery robust: instead of hard-failing when `/kaggle/input/cutmix/...` doesn’t exist, the code search under `/kaggle/input/` (including the nested competition directory) for a plausible `.pth/.pt/.bin/.ckpt` file and load it if found. If no fine-tuned checkpoint exists in the environment, the script still complete and write a valid `submission.csv` (using the ImageNet-pretrained backbone + random head), because producing a submission is required; this change is score-neutral when a checkpoint is present and prevents “no submission generated” failures. I also fix a small but real dataset init bug (uninitialized `self.label` when `return_name=True` and `label_path` is None) without changing any semantics. These changes preserve the model architecture, transforms, and argmax inference logic.'
- What this solution (achieved 0.07848) has done: 'Most of the timeout comes from training a heavy ResNeXt50 on 18.7k images at 448×448 for 5 epochs using PIL-based decoding and CPU transforms; that’s far beyond 10 minutes on typical Kaggle GPU/CPU budgets. To finish under 600s without changing the algorithm, the key is to avoid training during submission runs (use the provided checkpoint path when available) and to remove extra overhead that doesn’t affect predictions (e.g., unnecessary k-fold/val plumbing). I also speed up input by switching to `torchvision.io` image decoding (much faster than PIL) while keeping identical resize/normalize semantics, and I enable safe GPU-side optimizations (`cudnn.benchmark` already on) plus inference-only compilation to reduce overhead. These changes preserve the same model, loss, and evaluation semantics; they only eliminate redundant training work and accelerate equivalent preprocessing/inference.'
- What this solution (achieved 0.07848) has done: 'I fix the immediate runtime error by moving the hyperparameter/config cell (where `K_FOLD`, `BATCH_SIZE`, etc. are defined) before the dataloader construction so `cell 2` can run. I also fix a logic issue that was dragging your score down badly: your model is being `torch.compile()`’d before checkpoint loading, which can prevent correct weight loading and effectively leaves you with near-random predictions; I ensure the checkpoint is loaded into the uncompiled model first, then compile afterward (inference-only) without changing the model architecture or inference semantics. Finally, I make `_find_checkpoint` search more robust under `/kaggle/input/` for common checkpoint filenames, so the intended fine-tuned weights are much more likely to be found and loaded, moving accuracy toward the target while still always producing a valid `submission.csv`.'
- What this solution (achieved 0.07848) has done: 'Your current score (0.07848) is far below the target (0.8628), and the most likely reason is still that the fine-tuned cassava checkpoint is not actually being loaded, so the classifier head remains random and predictions are near-uniform. I make checkpoint loading deterministic and stricter in the specific way that matters here: explicitly require that the final `fc.weight` and `fc.bias` for 5 classes are loaded (otherwise we stop instead of producing a low-score submission), and we prioritize finding checkpoints that look like cassava-trained ResNeXt weights under `/kaggle/input/`. These are minimal changes that preserve your exact model/transform/inference logic (still argmax over 5 logits), but ensure you’re not accidentally submitting an untrained head. The script still write a valid `submission.csv` when a valid checkpoint is found and loaded.'
- What this solution (achieved 0.1207) has done: 'I remove the hard failure when no fine-tuned checkpoint is found, because that currently stops execution before a submission can be written. Instead, the script (1) try to find and load a compatible checkpoint, (2) if none exists, continue with the ImageNet-pretrained backbone and a deterministic fallback head initialization so results are stable. This keeps your model architecture, transforms, and argmax inference semantics unchanged, while guaranteeing an end-to-end run that always produces `/kaggle/working/submission.csv`. This likely improve your current 0.07848 somewhat (at least by ensuring consistent inference), but reaching ~0.86 still requires an actual cassava-trained checkpoint to be present in `/kaggle/input/`.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import torch

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if torch.cuda.is_available():
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def seed_worker(worker_id: int):
    worker_seed = SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


g = torch.Generator()
g.manual_seed(SEED)



## === cell 1
BATCH_SIZE = 16
EPOCH = 5
WD = 1e-4
LR = 0.0001
VAL_RATIO = 0.2
PHASE = ["train", "val"]
BETA = 1.0
CUTMIX_PROB = 1.0

TRAINING = False

WEIGHT = "/kaggle/input/cutmix/resnext_kfold0_17_0.839"
K_FOLD = 1



## === cell 2
from torch.utils.data.dataset import Dataset
import glob
import pandas as pd
import torchvision.transforms as T
from torchvision.io import read_image, ImageReadMode


class CLD_Dataset(Dataset):
    def __init__(self, image_root, label_path=None, transform=None, return_name=False):
        super(CLD_Dataset, self).__init__()
        self.transform = transform
        self.image_paths = sorted(glob.glob(os.path.join(image_root, "*.jpg")))
        self.return_name = return_name

        self.label_map = None
        if not self.return_name:
            if label_path is None:
                raise ValueError("label_path must be provided when return_name=False")
            df = pd.read_csv(label_path)
            self.label_map = dict(
                zip(df["image_id"].values, df["label"].astype(int).values)
            )

    def __getitem__(self, x):
        img_path = self.image_paths[x]
        im = read_image(img_path, mode=ImageReadMode.RGB)

        if self.transform is not None:
            img = self.transform(im)
        else:
            img = im.float().div_(255.0)

        file_name = os.path.basename(img_path)
        if self.return_name:
            return img, file_name
        else:
            label = self.label_map[file_name]
            return img, label

    def __len__(self):
        return len(self.image_paths)




## === cell 3
import os
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold

train_transform = T.Compose(
    [
        T.Resize((448, 448), antialias=True),
        T.RandomHorizontalFlip(),
        T.ConvertImageDtype(torch.float32),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

eval_transform = T.Compose(
    [
        T.Resize((448, 448), antialias=True),
        T.ConvertImageDtype(torch.float32),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

all_train_dataset = CLD_Dataset(
    "/kaggle/input/cassava-leaf-disease-classification/train_images",
    "/kaggle/input/cassava-leaf-disease-classification/train.csv",
    train_transform,
)
dataset_size = len(all_train_dataset)

NUM_WORKERS = min(4, (os.cpu_count() or 4))
PIN_MEMORY = torch.cuda.is_available()
PREFETCH = 4 if NUM_WORKERS > 0 else None
PERSIST = NUM_WORKERS > 0

fold_dataloader = []
if K_FOLD != 1:
    kf = KFold(K_FOLD, shuffle=True, random_state=SEED)

    index = 0
    for train_idx, val_idx in kf.split(all_train_dataset):
        train_dataset = torch.utils.data.Subset(all_train_dataset, train_idx)
        val_dataset = torch.utils.data.Subset(all_train_dataset, val_idx)
        fold_dataloader.append(
            {
                "train": DataLoader(
                    train_dataset,
                    batch_size=BATCH_SIZE,
                    shuffle=True,
                    num_workers=NUM_WORKERS,
                    pin_memory=PIN_MEMORY,
                    persistent_workers=PERSIST,
                    prefetch_factor=PREFETCH,
                    worker_init_fn=seed_worker,
                    generator=g,
                ),
                "val": DataLoader(
                    val_dataset,
                    batch_size=BATCH_SIZE,
                    shuffle=False,
                    num_workers=NUM_WORKERS,
                    pin_memory=PIN_MEMORY,
                    persistent_workers=PERSIST,
                    prefetch_factor=PREFETCH,
                    worker_init_fn=seed_worker,
                    generator=g,
                ),
            }
        )
        index += 1
else:
    fold_dataloader.append(
        {
            "train": DataLoader(
                all_train_dataset,
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=NUM_WORKERS,
                pin_memory=PIN_MEMORY,
                persistent_workers=PERSIST,
                prefetch_factor=PREFETCH,
                worker_init_fn=seed_worker,
                generator=g,
            ),
            "val": None,
        }
    )



## === cell 4
import torch.nn as nn



## === cell 5
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
        try:
            import torchvision

            if arch == "resnext50_32x4d":
                tv = torchvision.models.resnext50_32x4d(
                    weights=torchvision.models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1
                )
                state_dict = tv.state_dict()
            elif arch == "resnext101_32x8d":
                tv = torchvision.models.resnext101_32x8d(
                    weights=torchvision.models.ResNeXt101_32X8D_Weights.IMAGENET1K_V1
                )
                state_dict = tv.state_dict()
            else:
                state_dict = None

            if state_dict is not None:
                load = []
                not_load = []
                model_dict = model.state_dict()
                for name, param in state_dict.items():
                    if name in model_dict and model_dict[name].shape == param.shape:
                        model_dict[name].copy_(param)
                        load.append(name)
                    else:
                        not_load.append(name)
                print("Load : {} layers".format(len(load)))
                print("Miss : {} layers".format(len(not_load)))
            else:
                print("Pretrained requested but not available for arch:", arch)
        except Exception as e:
            print(
                "Pretrained loading failed; continuing with random init. Error:",
                repr(e),
            )

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
        "resnext50_32x4d", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs
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




## === cell 6
if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"
print(device)

model = resnext50_32x4d(num_classes=5, pretrained=True).to(device)

if device.startswith("cuda"):
    model = model.to(memory_format=torch.channels_last)



## === cell 7
pass



## === cell 8
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 2, gamma=0.5, last_epoch=-1)



## === cell 9
pass




## === cell 10
def rand_bbox_torch(size, lam, device):
    W = size[2]
    H = size[3]
    cut_rat = torch.sqrt(torch.as_tensor(1.0 - lam, device=device))
    cut_w = (W * cut_rat).to(torch.int64).clamp_(0, W)
    cut_h = (H * cut_rat).to(torch.int64).clamp_(0, H)

    cx = torch.randint(0, W, (1,), device=device, dtype=torch.int64)
    cy = torch.randint(0, H, (1,), device=device, dtype=torch.int64)

    cw = cut_w // 2
    ch = cut_h // 2

    bbx1 = int(torch.clamp(cx - cw, 0, W).item())
    bby1 = int(torch.clamp(cy - ch, 0, H).item())
    bbx2 = int(torch.clamp(cx + cw, 0, W).item())
    bby2 = int(torch.clamp(cy + ch, 0, H).item())
    return bbx1, bby1, bbx2, bby2




## === cell 11
pass




## === cell 12
class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.value = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, value, batch):
        self.value = float(value)
        self.sum += float(value) * int(batch)
        self.count += int(batch)
        self.avg = self.sum / self.count




## === cell 13
pass



## === cell 14
val_max_acc = 0
from tqdm import tqdm

val_max_acc = 0.0
ACCMeter = [
    AverageMeter(),
    AverageMeter(),
    AverageMeter(),
    AverageMeter(),
    AverageMeter(),
]
LOSSMeter = [
    AverageMeter(),
    AverageMeter(),
    AverageMeter(),
    AverageMeter(),
    AverageMeter(),
]

_ce = criterion
_model = model
_opt = optimizer

_torch_max = torch.max
_torch_randperm = torch.randperm

if TRAINING:
    for epoch in range(EPOCH):
        for index, dataloader in enumerate(fold_dataloader):
            for phase in PHASE:
                correct = 0
                total = 0
                loss_t = 0.0

                if phase == "train":
                    _model.train(True)
                else:
                    if dataloader[phase] is None:
                        continue
                    _model.train(False)

                grad_ctx = torch.enable_grad() if phase == "train" else torch.no_grad()
                dl = dataloader[phase]

                it = tqdm(
                    dl,
                    total=len(dl) if phase == "train" else None,
                    position=0,
                    leave=False,
                    disable=(phase != "train"),
                    miniters=50,
                )

                with grad_ctx:
                    for image, label in it:
                        b_image = image.to(device, non_blocking=True)
                        b_label = label.to(device, non_blocking=True)

                        if device.startswith("cuda"):
                            b_image = b_image.contiguous(
                                memory_format=torch.channels_last
                            )

                        do_cutmix = (BETA > 0) and (
                            torch.rand(1, device=b_image.device).item() < CUTMIX_PROB
                        )
                        if do_cutmix:
                            lam = float(np.random.beta(BETA, BETA))
                            rand_index = _torch_randperm(
                                b_image.size(0), device=b_image.device
                            )
                            target_a = b_label
                            target_b = b_label[rand_index]
                            bbx1, bby1, bbx2, bby2 = rand_bbox_torch(
                                b_image.size(), lam, device=b_image.device
                            )
                            b_image[:, :, bbx1:bbx2, bby1:bby2] = b_image[
                                rand_index, :, bbx1:bbx2, bby1:bby2
                            ]
                            denom = b_image.size(-1) * b_image.size(-2)
                            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / denom)

                            output = _model(b_image)
                            loss = _ce(output, target_a) * lam + _ce(
                                output, target_b
                            ) * (1.0 - lam)
                        else:
                            output = _model(b_image)
                            loss = _ce(output, b_label)

                        _, predicted = _torch_max(output, dim=1)
                        correct += (predicted == b_label).sum().item()
                        bs = b_label.size(0)
                        total += bs
                        loss_t += loss.item() * bs

                        if phase == "train":
                            _opt.zero_grad(set_to_none=True)
                            loss.backward()
                            _opt.step()

                if epoch == EPOCH - 1:
                    ACCMeter[index].update(correct, total)
                    LOSSMeter[index].update(loss_t / max(total, 1), 1)

                if phase == "val":
                    print(
                        "Fold : {}/ {} Epoch : {} / {} loss : {:.6f} ACC : {:.6f}".format(
                            index + 1,
                            K_FOLD,
                            epoch,
                            EPOCH,
                            loss_t / max(total, 1),
                            correct / max(total, 1),
                        )
                    )

        lr_scheduler.step()



## === cell 15
pass



## === cell 16
acc_sum = 0
loss_sum = 0
for i in range(K_FOLD):
    acc_sum += ACCMeter[i].avg
    loss_sum += LOSSMeter[i].avg

print("K-fold {} ACC : {:.6f}".format(K_FOLD, acc_sum / K_FOLD))
print("K-fold {} ACC : {:.6f}".format(K_FOLD, loss_sum / K_FOLD))



## === cell 17
pass



## === cell 18
import glob
import os
import torch

model.train(False)


def _find_checkpoint(path_like: str) -> str:
    if path_like and os.path.exists(path_like) and os.path.isfile(path_like):
        return path_like

    if path_like and os.path.exists(path_like) and os.path.isdir(path_like):
        patterns = ["*.pth", "*.pt", "*.bin", "*.ckpt"]
        found = []
        for pat in patterns:
            found.extend(glob.glob(os.path.join(path_like, pat)))
        found = sorted(found)
        if found:
            return found[-1]

    roots = []
    if path_like:
        parent = os.path.dirname(path_like.rstrip("/"))
        if parent and os.path.isdir(parent):
            roots.append(parent)

    roots.extend(
        [
            "/kaggle/input",
            "/kaggle/input/cassava-leaf-disease-classification",
            "/kaggle/input/cutmix",
        ]
    )

    patterns = ["*.pth", "*.pt", "*.bin", "*.ckpt"]
    found = []
    for root in roots:
        if os.path.isdir(root):
            for pat in patterns:
                found.extend(glob.glob(os.path.join(root, pat)))
                found.extend(glob.glob(os.path.join(root, "**", pat), recursive=True))

    found = sorted(set(found))
    if not found:
        return ""

    def _score(p: str) -> tuple:
        name = os.path.basename(p).lower()
        bonus = 0
        for tok in [
            "cassava",
            "cld",
            "leaf",
            "resnext",
            "cutmix",
            "kfold",
            "fold",
            "best",
            "acc",
        ]:
            if tok in name:
                bonus += 1
        return (bonus, os.path.getmtime(p))

    found = sorted(found, key=_score)
    return found[-1]


def _extract_state_dict(obj):
    if isinstance(obj, dict):
        for k in ["state_dict", "model", "net", "model_state_dict", "params"]:
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
        if len(obj) > 0 and all(hasattr(v, "shape") for v in obj.values()):
            return obj
    return obj


def _clean_key(k: str) -> str:
    for pref in ("module.", "model.", "net."):
        if k.startswith(pref):
            k = k[len(pref) :]
    return k


def load_checkpoint_safely(model: torch.nn.Module, ckpt_path: str) -> int:
    params = torch.load(ckpt_path, map_location="cpu")
    sd = _extract_state_dict(params)
    if not isinstance(sd, dict):
        raise RuntimeError(f"Checkpoint at {ckpt_path} is not a state_dict-like dict.")

    model_sd = model.state_dict()
    filtered = {}
    loaded = 0
    for k, v in sd.items():
        k2 = _clean_key(k)
        if k2 in model_sd and hasattr(v, "shape") and model_sd[k2].shape == v.shape:
            filtered[k2] = v
            loaded += 1

    missing, unexpected = model.load_state_dict(filtered, strict=False)
    print(f"Checkpoint tensors matched+loaded: {loaded}/{len(model_sd)}")
    print(
        f"Missing after load (count): {len(missing)}; Unexpected (count): {len(unexpected)}"
    )
    return loaded


ckpt_loaded = False
if not TRAINING:
    ckpt_path = _find_checkpoint(WEIGHT)
    if ckpt_path:
        print("Loading checkpoint:", ckpt_path)
        try:
            _ = load_checkpoint_safely(model, ckpt_path)
            ckpt_loaded = True
        except Exception as e:
            print(
                "WARNING: checkpoint load failed; continuing without it. Error:",
                repr(e),
            )
    else:
        print(
            f"WARNING: Could not locate any checkpoint for WEIGHT={WEIGHT}. "
            "Continuing with ImageNet-pretrained backbone + randomly initialized 5-class head."
        )

    if not ckpt_loaded:
        torch.manual_seed(SEED)
        if (
            hasattr(model, "fc")
            and isinstance(model.fc, torch.nn.Linear)
            and model.fc.out_features == 5
        ):
            torch.nn.init.normal_(model.fc.weight, mean=0.0, std=0.01)
            torch.nn.init.constant_(model.fc.bias, 0.0)

if (not TRAINING) and hasattr(torch, "compile"):
    try:
        model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
    except Exception:
        pass



## === cell 19
import pandas as pd
from torch.utils.data import DataLoader
import torch

test_dataset = CLD_Dataset(
    "/kaggle/input/cassava-leaf-disease-classification/test_images",
    transform=eval_transform,
    return_name=True,
)

test_dataloader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    persistent_workers=(NUM_WORKERS > 0),
    prefetch_factor=PREFETCH,
    worker_init_fn=seed_worker,
    generator=g,
)

image_name = []
image_label = []

model.eval()
with torch.no_grad():
    for step, (img, img_name) in enumerate(test_dataloader):
        b_img = img.to(device, non_blocking=True)
        if device.startswith("cuda"):
            b_img = b_img.contiguous(memory_format=torch.channels_last)

        output = model(b_img)
        predicted = torch.argmax(output, dim=1).detach().cpu().numpy().tolist()

        image_name.extend(list(img_name))
        image_label.extend([int(x) for x in predicted])

df = pd.DataFrame({"image_id": image_name, "label": image_label})

try:
    sample = pd.read_csv(
        "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
    )
    df = sample[["image_id"]].merge(df, on="image_id", how="left")
except Exception:
    pass

df.to_csv("/kaggle/working/submission.csv", index=False)
print("Wrote submission to /kaggle/working/submission.csv with shape:", df.shape)
