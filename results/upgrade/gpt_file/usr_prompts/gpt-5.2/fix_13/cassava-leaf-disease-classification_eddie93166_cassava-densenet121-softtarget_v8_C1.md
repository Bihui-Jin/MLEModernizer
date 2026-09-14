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

0.8821396192203083

# 6. Current score

0.67601

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.10762) has done: 'I fix inference so it always produces predictions even when the external pretrained `.pkl` files are not found (which currently makes `image_probs` empty and triggers the `IndexError`). I also correct a softmax-normalization bug (`sum(prob_e + 1e-4)` was summing a scalar, not the vector) and make the probability aggregation numerically stable by using `torch.softmax` (same evaluation semantics: argmax of class probabilities). Finally, I ensure image ordering is deterministic (sorted file list) and that the generated `submission.csv` exactly matches `sample_submission.csv` ordering to avoid any accidental misalignment.'
- What this solution (achieved 0.10613) has done: 'Your low score is consistent with running inference using an effectively untrained/random model because `create_new_model(pretrained=False)` is used and the code can’t download weights, while the external `.pkl` ensemble folder is usually absent. To move accuracy toward the target with minimal changes, I keep your exact model/training/inference structure but switch the fallback to use torchvision’s built-in ImageNet pretrained ResNeXt50_32x4d weights (available offline in torchvision) and copy them into your custom ResNet implementation. I also ensure the classifier head is initialized safely (since ImageNet has 1000 classes) while preserving the rest of your logic and submission alignment. This should substantially improve predictions without changing the training loop or adding new modeling steps.'
- What this solution (achieved 0.11323) has done: 'Your current score is low because inference is still effectively using a random 5-class head (the ImageNet weights load only the backbone and you then re-init the `fc`, and `pretrained=False` is used in cell 8). To move accuracy toward the target with minimal changes and without changing the model/training/inference structure, I (1) enable ImageNet pretrained loading for the default model, (2) map the ImageNet `fc` weights into your 5 cassava classes using a simple weight-averaging “prototype” mapping (keeps the same ResNeXt backbone and same argmax-over-softmax evaluation), and (3) keep the submission aligned to `sample_submission.csv` exactly as you already do. This should substantially increase accuracy versus a random head while preserving your core logic and finishing within the time limit.'
- What this solution (achieved 0.11323) has done: 'Your score is far below the target, and the main reason is still that the 5-class classifier head is essentially not trained for cassava (the “ImageNet prototype” mapping is too weak), so predictions are near-random. To move accuracy sharply toward the target without changing your core model/training/inference logic, I keep the same custom ResNeXt50 architecture and inference loop, but load an actual cassava-trained checkpoint from the competition dataset (commonly available in `/kaggle/input/...`) when present. If that checkpoint is not found, the code fall back exactly to your current ImageNet-backbone + prototype head behavior, so it remains robust and always produces a valid `submission.csv`. I also switch test inference to batch size 16 (no semantic change) to ensure runtime stays well within limits while scoring improvement comes from better weights, not new modeling.'
- What this solution (achieved 0.11323) has done: 'Your score is still near-random because the inference path almost never finds cassava-finetuned weights, so you’re effectively predicting with an ImageNet backbone + an arbitrary 5-class head. I make a minimal, execution-safe change to load cassava-trained weights from the competition dataset itself (searching `/kaggle/input/cassava-leaf-disease-classification/**` first, then the rest of `/kaggle/input/**`), which preserves your exact architecture and inference semantics but should move accuracy strongly toward the 0.88 target. I also ensure any discovered checkpoint with `num_classes=1000` is handled by loading backbone-only when needed (so it never crashes), and keep your submission alignment to `sample_submission.csv` unchanged.'
- What this solution (achieved 0.76457) has done: 'Your current score (0.113) is far below the target (0.882), which is consistent with running a cassava-untrained classifier head (near-random predictions). With minimal changes and identical overall training/inference flow, I (1) compute a cheap 5-class linear head by extracting features with your existing ImageNet-pretrained ResNeXt backbone on a subset of train images and fitting a multinomial logistic regression (no deep training loop changes), then (2) copy those learned weights into `model.fc` for test inference. This keeps the same architecture, same softmax→argmax evaluation semantics, and should move accuracy sharply toward the target while staying within the time limit. If anything fails (e.g., sklearn fit), it safely fall back to your current prototype head to still produce a valid `submission.csv`.'
- What this solution (achieved 0.65546) has done: 'We keep your model, transforms, and inference pipeline intact, but make the LogisticRegression head fit stronger and more stable so accuracy moves closer to the 0.882 target. Specifically, we (1) remove the random subsampling and instead use a deterministic, class-balanced subset up to the same cap, and (2) add simple feature standardization (mean/std) before LogisticRegression, which typically improves linear head quality without changing evaluation semantics (still softmax→argmax). We also slightly increase LogisticRegression iterations to ensure convergence (no approximations/early stopping) while keeping runtime under the limit by keeping the same max sample cap. Submission writing and ordering remain exactly aligned to `sample_submission.csv`.'
- What this solution (achieved 0.65433) has done: 'Your current score (0.655) is well below the 0.882 target, so we should improve the linear-head fitting step without changing your backbone/architecture or inference semantics. I keep your ResNeXt feature extractor and LogisticRegression approach, but fix a subtle bug in the “extra fill” logic (it re-reads train.csv multiple times and uses mismatched indices) and make the balanced subset deterministic and truly class-balanced. Then I use class-balanced weights in LogisticRegression (still multinomial lbfgs, same argmax-over-softmax semantics) to reduce bias toward majority classes, which typically gives a noticeable accuracy lift toward your target. Everything else (transforms, model, inference loop, submission alignment) stays the same and it still writes `/kaggle/working/submission.csv`.'
- What this solution (achieved 0.64723) has done: 'Your current score (0.654) is far below the target (0.882), so we should improve the linear-head fitting step while keeping your ResNeXt backbone + “extract features → multinomial LogisticRegression → copy into fc → softmax/argmax” inference semantics unchanged. The biggest gain with minimal risk is to fit the LogReg head on more (and better balanced) training data and to use the full multiclass probability output rather than relying on potentially weaker calibration. Concretely, I increase the LogReg fitting sample cap (still deterministic and class-balanced) and slightly increase `max_iter`/`C` to ensure good convergence/fit quality without changing the model architecture or adding new training loops. Everything else (dataset reading, transforms, model, submission alignment) is preserved and the script still writes `/kaggle/working/submission.csv`.'
- What this solution (achieved 0.6775) has done: 'Your current score (0.647) is far below the target (0.882), so we should improve the fitted linear head while keeping your exact “fixed ResNeXt backbone → extract features → multinomial LogisticRegression → copy into fc → softmax/argmax” semantics unchanged. The minimal, high-impact change is to fit the LogReg head on more deterministic, class-balanced training samples (still no deep training loop), which usually gives a sizable jump in accuracy for this competition. To avoid degrading performance, we also ensure we do not accidentally skip the LogReg fit due to unrelated checkpoints (and we still never override a true cassava 5-class checkpoint if found). Finally, we keep the submission alignment to `sample_submission.csv` exactly the same and still write `/kaggle/working/submission.csv`.'
- What this solution (achieved 0.68012) has done: 'Your current score (0.6775) is well below the 0.882 target, so we should improve the linear head quality without changing your backbone, transforms, or inference semantics (still softmax→argmax). The smallest high-impact change is to fit the LogisticRegression head on *all* training images (you already extract features once; this just removes the 18k cap), which typically moves accuracy materially closer to your target for Cassava. To keep this stable and deterministic, we keep the same feature standardization and solver, but bump `max_iter` to ensure convergence on the larger dataset (no early stopping). Everything else—including checkpoint fallback behavior and submission alignment to `sample_submission.csv`—stays the same.'
- What this solution (achieved 0.67601) has done: 'Your score (0.680) is far below the target (0.882), so we should improve the linear head fit while keeping your backbone, transforms, and inference semantics unchanged. The biggest low-risk gain is to make the LogisticRegression fit more stable and better calibrated by (1) using `torch.cuda.amp.autocast` for faster feature extraction (same features in practice, just more throughput), (2) fitting on full train set but with a stronger solver setup (slightly higher `max_iter`, `C`, and explicit `tol`) to ensure convergence, and (3) using consistent, deterministic preprocessing and feature normalization. Everything else—including model architecture, softmax→argmax prediction, and submission alignment to `sample_submission.csv`—stays the same and it still writes `/kaggle/working/submission.csv`.'

# 9. Code solution

## === cell 0
BATCH_SIZE = 8
EPOCH = 10
WD = 1e-4
LR = 0.0001
VAL_RATIO = 0.2
PHASE = ["train", "val"]
BETA = 0.0
CUTMIX_PROB = 0.0
TRAINING = False
K_FOLD = 5



## === cell 1
import os
import glob
import numpy as np
import pandas as pd
import torch



## === cell 2
from torch.utils.data.dataset import Dataset
from PIL import Image


class CLD_Dataset(Dataset):
    def __init__(self, image_root, label_path=None, transform=None, return_name=False):
        super(CLD_Dataset, self).__init__()
        self.transform = transform
        self.image_paths = sorted(glob.glob(os.path.join(image_root, "*.jpg")))
        if not return_name:
            self.label = pd.read_csv(label_path, index_col="image_id")
        self.return_name = return_name

    def set_transform(self, transform):
        self.transform = transform

    def __getitem__(self, x):
        img = Image.open(self.image_paths[x]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        if self.return_name:
            return img, os.path.basename(self.image_paths[x])
        else:
            label = self.label.loc[os.path.basename(self.image_paths[x])].label
            return img, int(label)

    def __len__(self):
        return len(self.image_paths)




## === cell 3
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

fold_dataloader = []
if K_FOLD != 1:
    kf = KFold(K_FOLD, shuffle=True, random_state=42)

    index = 0
    for train_idx, val_idx in kf.split(all_train_dataset):
        train_dataset = torch.utils.data.Subset(all_train_dataset, train_idx)
        val_dataset = torch.utils.data.Subset(all_train_dataset, val_idx)
        fold_dataloader.append(
            {
                "train": DataLoader(
                    train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
                ),
                "val": DataLoader(
                    val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
                ),
            }
        )
        index += 1
    print(f"Built {len(fold_dataloader)} fold dataloaders.")
else:
    fold_dataloader.append(
        {
            "train": DataLoader(
                all_train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
            ),
            "val": None,
        }
    )



## === cell 4
import random


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)



## === cell 5
import torch.nn as nn



## === cell 6
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
    load = []
    not_load = []
    if pretrained:
        weight_path = "../input/resnext50-32x4d/resnext50_32x4d.pth"
        if os.path.exists(weight_path):
            state_dict = torch.load(weight_path, map_location="cpu")
            for name, param in state_dict.items():
                if name in model.state_dict():
                    try:
                        load.append(name)
                        model.state_dict()[name].copy_(param)
                    except Exception:
                        not_load.append(name)
        else:
            print(
                f"Pretrained weight file not found at {weight_path}. Using random init."
            )
    print("Load pretrain : ")
    print("Load : {} layers".format(len(load)))
    print("Miss : {} layers".format(len(not_load)))
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




## === cell 7
if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"
print(device)

import torchvision


def _load_torchvision_resnext50_32x4d_imagenet_into_custom(model: nn.Module) -> int:
    try:
        from torchvision.models import resnext50_32x4d, ResNeXt50_32X4D_Weights

        tv = resnext50_32x4d(weights=ResNeXt50_32X4D_Weights.DEFAULT)
        tv_sd = tv.state_dict()
    except Exception as e:
        print(
            "Could not load torchvision pretrained weights, using current init. Err:",
            repr(e),
        )
        return 0

    msd = model.state_dict()
    loaded = 0

    for k, v in tv_sd.items():
        if k.startswith("fc."):
            continue
        if k in msd and msd[k].shape == v.shape:
            msd[k].copy_(v)
            loaded += 1

    model.load_state_dict(msd, strict=False)
    return loaded


def _init_fc_from_imagenet_fc_prototypes(model: nn.Module) -> bool:
    try:
        from torchvision.models import resnext50_32x4d, ResNeXt50_32X4D_Weights

        tv = resnext50_32x4d(weights=ResNeXt50_32X4D_Weights.DEFAULT)
        w = tv.fc.weight.detach().cpu()  # [1000, 2048]
        b = tv.fc.bias.detach().cpu()  # [1000]
    except Exception as e:
        print("Could not load torchvision fc weights for prototype init. Err:", repr(e))
        return False

    g = 5
    idx = torch.arange(w.shape[0])
    groups = [idx[i::g] for i in range(g)]

    w5 = torch.stack([w[gi].mean(dim=0) for gi in groups], dim=0)  # [5, 2048]
    b5 = torch.stack([b[gi].mean(dim=0) for gi in groups], dim=0)  # [5]

    with torch.no_grad():
        model.fc.weight.copy_(
            w5.to(model.fc.weight.device, dtype=model.fc.weight.dtype)
        )
        model.fc.bias.copy_(b5.to(model.fc.bias.device, dtype=model.fc.bias.dtype))
    return True


def _find_best_local_cassava_checkpoint() -> str | None:
    patterns_primary = [
        "/kaggle/input/cassava-leaf-disease-classification/**/*.pth",
        "/kaggle/input/cassava-leaf-disease-classification/**/*.pt",
        "/kaggle/input/cassava-leaf-disease-classification/**/*.pkl",
    ]
    patterns_fallback = [
        "/kaggle/input/**/best.pth",
        "/kaggle/input/**/best.pt",
        "/kaggle/input/**/model.pth",
        "/kaggle/input/**/model.pt",
        "/kaggle/input/**/checkpoint.pth",
        "/kaggle/input/**/checkpoint.pt",
        "/kaggle/input/**/resnext50*.pth",
        "/kaggle/input/**/resnext50*.pt",
        "/kaggle/input/**/resnext50*.pkl",
        "/kaggle/input/**/cassava*.pth",
        "/kaggle/input/**/cassava*.pt",
        "/kaggle/input/**/cassava*.pkl",
    ]

    def _collect(patterns):
        cands = []
        for p in patterns:
            cands.extend(glob.glob(p, recursive=True))
        return [c for c in cands if os.path.isfile(c)]

    cands = _collect(patterns_primary)
    if not cands:
        cands = _collect(patterns_fallback)
    if not cands:
        return None

    priority = []
    for c in cands:
        lc = c.lower()
        score = 0
        if "cassava" in lc:
            score += 5
        if "leaf" in lc:
            score += 3
        if "disease" in lc:
            score += 2
        if "resnext" in lc:
            score += 2
        if "fold" in lc:
            score += 1
        if "best" in lc:
            score += 1
        if "5" in os.path.basename(lc):
            score += 1
        priority.append((score, c))
    priority.sort(key=lambda x: (-x[0], x[1]))
    return priority[0][1]


def _load_checkpoint_flex(model: nn.Module, ckpt_path: str) -> tuple[int, int]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict):
        for key in ["state_dict", "model_state_dict", "model", "net", "weights"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
                break

    if isinstance(ckpt, dict):
        sd = {}
        for k, v in ckpt.items():
            nk = k
            if nk.startswith("model."):
                nk = nk[len("model.") :]
            if nk.startswith("net."):
                nk = nk[len("net.") :]
            if nk.startswith("module."):
                nk = nk[len("module.") :]
            sd[nk] = v
        ckpt = sd

    msd = model.state_dict()
    loaded, missed = 0, 0
    with torch.no_grad():
        for k, v in ckpt.items():
            if k.startswith("fc."):
                missed += 1
                continue
            if k in msd and msd[k].shape == v.shape:
                msd[k].copy_(v)
                loaded += 1
            else:
                missed += 1
    model.load_state_dict(msd, strict=False)
    return loaded, missed


def create_new_model(pretrained=True):
    model = resnext50_32x4d(num_classes=5, pretrained=False).to(device)

    ckpt_path = _find_best_local_cassava_checkpoint()
    if ckpt_path is not None:
        loaded, missed = _load_checkpoint_flex(model, ckpt_path)
        print(f"Loaded local checkpoint: {ckpt_path}")
        print(
            f"Checkpoint tensors loaded (non-fc): {loaded}, missed/non-matching: {missed}"
        )
        return model

    if pretrained:
        n_loaded = _load_torchvision_resnext50_32x4d_imagenet_into_custom(model)
        print(
            f"Loaded {n_loaded} matching tensors from torchvision ImageNet weights into custom model."
        )

        ok = _init_fc_from_imagenet_fc_prototypes(model)
        if ok:
            print("Initialized 5-class fc from ImageNet fc prototypes (deterministic).")
        else:
            nn.init.normal_(model.fc.weight, 0, 0.01)
            nn.init.constant_(model.fc.bias, 0)

    return model




## === cell 8
model = create_new_model(pretrained=True)




## === cell 9
def create_loss_opti():
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=1, eta_min=1e-6, last_epoch=-1
    )
    return criterion, optimizer, lr_scheduler




## === cell 10
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




## === cell 11
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




## === cell 12
def train_step(model, criterion, optimizer, image, label, phase):
    b_image = image.to(device)
    b_label = label.to(device)

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
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return correct, loss.item()




## === cell 13
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



## === cell 14
acc_sum = 0
loss_sum = 0
for i in range(K_FOLD):
    acc_sum += ACCMeter[i].avg
    loss_sum += LOSSMeter[i].avg

print("K-fold {} ACC : {:.6f}".format(K_FOLD, acc_sum / K_FOLD))
print("K-fold {} LOSS : {:.6f}".format(K_FOLD, loss_sum / K_FOLD))



## === cell 15
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression


def _fit_cassava_fc_via_logreg(
    model: nn.Module,
    train_csv: str,
    train_img_root: str,
    transform_for_features,
    max_samples: int | None = None,
    batch_size: int = 32,
) -> bool:
    """
    Change rationale (score): keep the exact same "extract fixed backbone features -> multinomial LogisticRegression -> copy into fc"
    logic, but make the fit more reliable on full-data by ensuring strong convergence settings, and make feature extraction faster
    (amp autocast) so we can comfortably use the full train set within the time limit.
    """
    full = pd.read_csv(train_csv).sort_values("image_id").reset_index(drop=True)

    df = full
    if max_samples is not None and len(full) > max_samples:
        n_classes = int(full["label"].nunique())
        per_class = max_samples // n_classes

        parts = []
        for c in sorted(full["label"].unique()):
            parts.append(full[full["label"] == c].head(per_class))
        df = pd.concat(parts, axis=0).reset_index(drop=True)

        if len(df) < max_samples:
            need = max_samples - len(df)
            taken = set(df["image_id"].tolist())
            extra = full[~full["image_id"].isin(taken)].head(need)
            df = pd.concat([df, extra], axis=0).reset_index(drop=True)

    img_ids = df["image_id"].tolist()
    labels = df["label"].astype(int).tolist()
    paths = [os.path.join(train_img_root, x) for x in img_ids]

    class _ListDataset(torch.utils.data.Dataset):
        def __init__(self, paths, labels, tfm):
            self.paths = paths
            self.labels = labels
            self.tfm = tfm

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, i):
            img = Image.open(self.paths[i]).convert("RGB")
            if self.tfm is not None:
                img = self.tfm(img)
            return img, int(self.labels[i])

    ds = _ListDataset(paths, labels, transform_for_features)
    dl = DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    model.eval()
    feats = []
    ys = []

    use_amp = device.startswith("cuda")
    with torch.inference_mode():
        for xb, yb in tqdm(
            dl, total=len(dl), position=0, leave=True, desc="Extract feats"
        ):
            xb = xb.to(device, non_blocking=True)

            if use_amp:
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    x = model.conv1(xb)
                    x = model.bn1(x)
                    x = model.relu(x)
                    x = model.maxpool(x)
                    x = model.layer1(x)
                    x = model.layer2(x)
                    x = model.layer3(x)
                    x = model.layer4(x)
                    x = model.avgpool(x)
                    x = torch.flatten(x, 1)  # [B,2048]
            else:
                x = model.conv1(xb)
                x = model.bn1(x)
                x = model.relu(x)
                x = model.maxpool(x)
                x = model.layer1(x)
                x = model.layer2(x)
                x = model.layer3(x)
                x = model.layer4(x)
                x = model.avgpool(x)
                x = torch.flatten(x, 1)  # [B,2048]

            feats.append(x.detach().float().cpu().numpy())
            ys.append(np.array(yb, dtype=np.int64))

    X = np.concatenate(feats, axis=0)
    y = np.concatenate(ys, axis=0)

    mu = X.mean(axis=0, keepdims=True)
    sigma = X.std(axis=0, keepdims=True) + 1e-6
    Xs = (X - mu) / sigma

    clf = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        max_iter=2000,
        tol=1e-4,
        n_jobs=None,
        random_state=42,
        C=5.0,
        class_weight="balanced",
    )
    clf.fit(Xs, y)

    Wz = clf.coef_  # [5,2048]
    bz = clf.intercept_  # [5]
    W_raw = Wz / sigma  # broadcast over classes
    b_raw = bz - (Wz * (mu / sigma)).sum(axis=1)

    W = torch.from_numpy(W_raw).to(model.fc.weight.device, dtype=model.fc.weight.dtype)
    b = torch.from_numpy(b_raw).to(model.fc.bias.device, dtype=model.fc.bias.dtype)
    with torch.no_grad():
        model.fc.weight.copy_(W)
        model.fc.bias.copy_(b)
    return True


def _looks_like_cassava_5class_checkpoint(ckpt_path: str) -> bool:
    """
    Change rationale (score): avoid skipping the LogReg head fit due to unrelated .pth/.pt files
    in /kaggle/input that are not actually cassava 5-class checkpoints.
    """
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        if isinstance(ckpt, dict):
            for key in ["state_dict", "model_state_dict", "model", "net", "weights"]:
                if key in ckpt and isinstance(ckpt[key], dict):
                    ckpt = ckpt[key]
                    break
        if not isinstance(ckpt, dict):
            return False

        for k in ["fc.weight", "model.fc.weight", "net.fc.weight", "module.fc.weight"]:
            if k in ckpt and hasattr(ckpt[k], "shape"):
                return tuple(ckpt[k].shape)[0] == 5
        return False
    except Exception:
        return False


try:
    maybe_ckpt = _find_best_local_cassava_checkpoint()
    if maybe_ckpt is None or (
        maybe_ckpt is not None
        and (not _looks_like_cassava_5class_checkpoint(maybe_ckpt))
    ):
        all_train_dataset.set_transform(val_transform)
        ok = _fit_cassava_fc_via_logreg(
            model=model,
            train_csv="/kaggle/input/cassava-leaf-disease-classification/train.csv",
            train_img_root="/kaggle/input/cassava-leaf-disease-classification/train_images",
            transform_for_features=val_transform,
            max_samples=None,
            batch_size=32,
        )
        print("LogReg head fit:", ok)
    else:
        print(
            "Cassava-like 5-class checkpoint exists; not fitting LogReg head to avoid overriding."
        )
except Exception as e:
    print("Warning: failed to fit LogReg head; keeping existing head. Err:", repr(e))



## === cell 16
test_dataset = CLD_Dataset(
    "/kaggle/input/cassava-leaf-disease-classification/test_images",
    transform=val_transform,
    return_name=True,
)

test_dataloader = DataLoader(
    test_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True
)

weight_paths = sorted(glob.glob("../input/cassave-resnext50-30x4d/*.pkl"))
if len(weight_paths) == 0:
    print(
        "No external fold .pkl weights found in ../input/cassave-resnext50-30x4d/. "
        "Falling back to a single best-local-checkpoint (if any) else torchvision-ImageNet-pretrained model (optionally with fitted LogReg head) for inference."
    )
    weight_paths = [None]
else:
    print(f"Found {len(weight_paths)} weight files for ensembling.")

image_names_folds = []
image_probs_folds = []

for params_path in weight_paths:
    if params_path is None:
        infer_model = model
        print("Using prepared single model for inference.")
    else:
        infer_model = create_new_model(pretrained=False)
        params = torch.load(params_path, map_location="cpu")
        load = []
        not_load = []
        for name, param in params.items():
            if name in infer_model.state_dict():
                try:
                    infer_model.state_dict()[name].copy_(param)
                    load.append(name)
                except Exception:
                    not_load.append(name)
        print("Trained weight load : {}".format(len(load)))
        print("Trained weight not load : {}".format(len(not_load)))

    infer_model.eval()

    fold_names = []
    fold_probs = []

    use_amp = device.startswith("cuda")
    with torch.inference_mode():
        for img, img_name in tqdm(
            test_dataloader, total=len(test_dataloader), position=0, leave=True
        ):
            b_img = img.to(device, non_blocking=True)
            if use_amp:
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    logits = infer_model(b_img)  # [B,5]
            else:
                logits = infer_model(b_img)
            probs = F.softmax(logits, dim=1).detach().cpu().numpy()  # [B,5]
            fold_names.extend(list(img_name))
            fold_probs.extend(list(probs))

    image_names_folds.append(fold_names)
    image_probs_folds.append(fold_probs)

image_names_folds = np.array(image_names_folds)  # [F, N]
image_probs_folds = np.array(image_probs_folds)  # [F, N, 5]

avg_probs = image_probs_folds.mean(axis=0)  # [N, 5]
image_labels = np.argmax(avg_probs, axis=1).astype(int)  # [N]

sample_sub = pd.read_csv(
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)
pred_df = pd.DataFrame({"image_id": image_names_folds[0], "label": image_labels})
pred_df = sample_sub[["image_id"]].merge(pred_df, on="image_id", how="left")

if pred_df["label"].isna().any():
    missing = pred_df["label"].isna().sum()
    print(
        f"Warning: {missing} test images missing predictions; filling with 0 to keep submission valid."
    )
    pred_df["label"] = pred_df["label"].fillna(0).astype(int)

print(pred_df.head())
pred_df.to_csv("/kaggle/working/submission.csv", index=False)
print("Wrote /kaggle/working/submission.csv with shape:", pred_df.shape)
