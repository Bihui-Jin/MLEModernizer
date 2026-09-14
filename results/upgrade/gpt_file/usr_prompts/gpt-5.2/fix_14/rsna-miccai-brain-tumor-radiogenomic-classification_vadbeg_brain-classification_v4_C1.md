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
Predict the genetic subtype of glioblastoma using MRI (magnetic resonance imaging) scans to detect for the presence of MGMT promoter methylation.

## Metric
Area under the ROC curve between the predicted probability and the observed target.

## Submission Format
For each `BraTS21ID` in the test set, you must predict a probability for the target `MGMT_value`. The file should contain a header and have the following format:

```
BraTS21ID,MGMT_value
00001,0.5
00013,0.5
00015,0.5
etc.
```

## Dataset
- **train/** - folder containing the training files, with each top-level folder representing a subject. **NOTE:** There are some unexpected issues with the following three cases in the training dataset, participants can exclude the cases during training: `[00109, 00123, 00709]`. We have checked and confirmed that the testing dataset is free from such issues.
- **train_labels.csv** - file containing the target `MGMT_value` for each subject in the training data (e.g. the presence of MGMT promoter methylation)
- **test/** - the test files, which use the same structure as `train/`; your task is to predict the `MGMT_value` for each subject in the test data. **NOTE**: the total size of the rerun test set (Public and Private) is ~5x the size of the Public test set
- **sample_submission.csv** - a sample submission file in the correct format

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
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
simpleitk==2.5.2
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
            description.md (202 lines)
            sample_submission.csv (60 lines)
            sample_submission.csv.zip (382 Bytes)
            test.zip (1.3 GB)
            train.zip (10.2 GB)
            train_labels.csv (527 lines)
            train_labels.csv.zip (1.4 kB)
            rsna-miccai-brain-tumor-radiogenomic-classification/
                description.md (202 lines)
                sample_submission.csv (60 lines)
                ... and 5 other files
                rsna-miccai-brain-tumor-radiogenomic-classification/
                test/
                    00002/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00019/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 58 other folders
                train/
                    00000/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00003/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 525 other folders
            test/
                00002/
                    FLAIR/
                        Image-387.dcm (525.4 kB)
                        Image-388.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 29 other files
                    T1wCE/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 382 other files
                00019/
                    FLAIR/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 30 other files
                    T1wCE/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T2w/
                        Image-258.dcm (525.4 kB)
                        Image-259.dcm (525.4 kB)
                        ... and 127 other files
                ... and 58 other folders
            train/
                00000/
                    FLAIR/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 398 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 31 other files
                    T1wCE/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 406 other files
                00003/
                    FLAIR/
                        Image-387.dcm (525.4 kB)
                        Image-388.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 31 other files
                    T1wCE/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 406 other files
                ... and 525 other folders
        input/
            description.md (202 lines)
            sample_submission.csv (60 lines)
            sample_submission.csv.zip (382 Bytes)
            test.zip (1.3 GB)
            train.zip (10.2 GB)
            train_labels.csv (527 lines)
            train_labels.csv.zip (1.4 kB)
            rsna-miccai-brain-tumor-radiogenomic-classification/
                description.md (202 lines)
                sample_submission.csv (60 lines)
                ... and 5 other files
                rsna-miccai-brain-tumor-radiogenomic-classification/
                test/
                    00002/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00019/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 58 other folders
                train/
                    00000/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00003/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 525 other folders
            test/
                00002/
                    FLAIR/
                        Image-387.dcm (525.4 kB)
                        Image-388.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 29 other files
                    T1wCE/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 382 other files
                00019/
                    FLAIR/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 30 other files
                    T1wCE/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T2w/
                        Image-258.dcm (525.4 kB)
                        Image-259.dcm (525.4 kB)
                        ... and 127 other files
                ... and 58 other folders
            train/
                00000/
                    FLAIR/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 398 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 31 other files
                    T1wCE/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 406 other files
                00003/
                    FLAIR/
                        Image-387.dcm (525.4 kB)
                        Image-388.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 31 other files
                    T1wCE/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 406 other files
                ... and 525 other folders
        working/
            rsna-miccai-brain-tumor-radiogenomic-classification/
                description.md (202 lines)
                sample_submission.csv (60 lines)
                ... and 5 other files
                rsna-miccai-brain-tumor-radiogenomic-classification/
                test/
                    00002/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00019/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 58 other folders
                train/
                    00000/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00003/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 525 other folders
```

-> data/rsna-miccai-brain-tumor-radiogenomic-classification/sample_submission.csv has 59 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> data/rsna-miccai-brain-tumor-radiogenomic-classification/train_labels.csv has 526 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> data/sample_submission.csv has 59 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> data/train_labels.csv has 526 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> input/rsna-miccai-brain-tumor-radiogenomic-classification/sample_submission.csv has 59 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> input/rsna-miccai-brain-tumor-radiogenomic-classification/train_labels.csv has 526 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> (stopped after 10 files for performance)

# 5. Target score

-1.0

# 6. Current score

0.42471

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.44941) has done: 'I fix the runtime failure by making the checkpoint loading robust to the Kaggle environment: if the referenced `/kaggle/input/brainclassificationmodel/...` file is missing, the script automatically search for any `.ckpt`/`.pth`/`.pt` under `/kaggle/input` and load the first compatible checkpoint (or fall back to random weights but still produce a valid submission). I also fix a path-parsing bug that can return the wrong `BraTS21ID` (it currently takes the parent folder name `FLAIR` instead of the case folder), ensuring IDs align with `sample_submission.csv`. Finally, I ensure the output `submission.csv` strictly matches the sample submission ordering and format, guaranteeing a valid file is written end-to-end.'
- What this solution (achieved 0.56) has done: 'Your current score (0.44941 AUC) is already vastly better than the provided target score (-1.0), and for ROC-AUC the valid score range is [0, 1], so it’s impossible to move “toward -1.0” in any legitimate way. Given that, the safest approach is to keep the core inference logic intact and only make minimal stability fixes that avoid accidental score drops (e.g., ensure the intended checkpoint is actually found first, and make DICOM series selection deterministic). These changes should preserve or slightly improve your score while maintaining identical evaluation semantics and producing a valid `submission.csv`. I not change the model architecture, preprocessing, or prediction logic.'
- What this solution (achieved 0.47176) has done: 'Because ROC-AUC is bounded to \[0, 1\], the provided target score (-1.0) is unattainable; the safest “toward target” behavior is therefore to avoid changes that could accidentally degrade your working 0.56 submission and instead apply only stability fixes that tend to preserve or slightly improve score. I make checkpoint selection deterministic and more likely to pick the intended RSNA/MGMT model by preferring exact filename matches and larger (more complete) checkpoints, rather than “first compatible” by loose heuristics. I also harden DICOM series selection by choosing the series with the most slices (instead of lexicographically first series ID), which reduces the risk of reading an incomplete/auxiliary series and should improve prediction consistency. Core model, preprocessing, and inference semantics remain unchanged; the script still writes a valid `submission.csv` aligned to `sample_submission.csv`.'
- What this solution (achieved 0.50353) has done: 'The target score (-1.0) is unattainable for ROC-AUC (valid range is 0–1), so the only safe way to “move toward target” without illegitimate manipulation is to avoid changes that could accidentally degrade your working submission and instead make small correctness/stability fixes that tend to preserve (or slightly improve) score. I (1) fix the BraTS21ID extraction to be robust to getting either the modality folder or the case folder path (preventing silent ID mismatches), (2) make DICOM series selection deterministic even under ties (same number of slices), and (3) make checkpoint selection slightly stricter by preferring checkpoints that actually match a large fraction of model tensor shapes (reducing the chance of loading an incompatible but “ranked” file). These are minimal, inference-only changes that keep your model architecture and prediction semantics the same while reducing brittle failure modes that can hurt AUC.'
- What this solution (achieved 0.46353) has done: 'The target score (-1.0) is impossible for ROC-AUC (valid range is 0–1), so the safest way to reduce the chance of accidental score regressions is to make only correctness/stability fixes that keep your inference semantics the same. I keep the same model, same preprocessing, same softmax probability for class 1, and the same “use FLAIR folder” input pattern. The main change is to make checkpoint selection deterministic and *more likely to load the intended weights* by preferring checkpoints that match the model tensor shapes and have a high fraction of matched keys (not just raw count), which reduces silent partial loads that can hurt AUC. I also make BraTS21ID extraction strict for the known `.../<case>/<modality>` structure and ensure the saved submission strictly follows `sample_submission.csv` order (already mostly done) to avoid any hidden alignment issues.'
- What this solution (achieved 0.42471) has done: 'Your target score of `-1.0` is unattainable for ROC-AUC (valid range is `[0, 1]`), so the only legitimate “toward target” action is to avoid accidental improvements and keep behavior stable; however you asked to increase score, so I apply the smallest correctness fixes that typically improve AUC without changing the model or preprocessing logic. I (1) force deterministic inference settings and proper `torch.inference_mode()` to remove nondeterministic noise, and (2) make checkpoint selection prefer *strict/full* matches when available rather than best fractional match (to reduce partial-load silent degradation). Everything else (ResNet3D architecture, FLAIR-only input, preprocessing, softmax class-1 probability, and submission formatting) stays the same, and the script still writes a valid `submission.csv`.'
- What this solution (achieved 0.42471) has done: 'Your target score of `-1.0` is impossible for ROC-AUC (valid range is 0–1), so the only legitimate way to follow your request to “increase score” is to reduce accidental degradation and make tiny correctness tweaks that typically improve AUC without changing the model architecture or preprocessing. I keep the exact same ResNet3D, FLAIR-only input, resizing/normalization, and softmax(class=1) probability, but (1) load checkpoints more correctly by handling common Lightning key prefixes (avoids silent partial loads), and (2) enable CUDA inference speedups (`cudnn.benchmark=True`) since input size is fixed, reducing timeouts/instability while preserving evaluation semantics. Everything else remains the same and it still write a valid `submission.csv` aligned to `sample_submission.csv`.'
- What this solution (achieved 0.42471) has done: 'I keep your model, preprocessing, and inference logic exactly the same, and only make two minimal changes that typically *increase* AUC without changing semantics: (1) ensure checkpoint loading matches your model weights more completely by also stripping the common Lightning prefix `model._orig_mod.` (a frequent cause of silent partial loads in torch2 compile/export pipelines), and (2) make CUDA math deterministic/stable by disabling TF32 (this is not an approximation; it uses higher-precision matmul/conv paths and can improve probability ranking consistency for AUC). These are small, inference-only stability fixes that shouldn’t change the architecture or data pipeline and still write a valid `submission.csv`. Everything else, including FLAIR-only input, resizing/normalization, softmax class-1 probability, and submission alignment to `sample_submission.csv`, remains unchanged.'
- What this solution (achieved 0.42471) has done: 'I make two minimal, inference-only correctness changes that tend to improve ROC-AUC without altering your model architecture or preprocessing: (1) robustly standardize the loaded checkpoint tensors to `float32` before `load_state_dict` (prevents silent dtype mismatches/partial loads that can hurt ranking quality), and (2) ensure the model is moved to device *before* loading weights so any checkpoint tensors (after casting) align cleanly with module buffers/BN stats handling. Everything else (ResNet3D-10, FLAIR-only input, resizing/normalization, softmax class-1 probability, and submission alignment to `sample_submission.csv`) stays the same and still writes a valid `submission.csv`. These are small, low-risk fixes aimed at nudging score upward from 0.42471.'
- What this solution (achieved 0.42471) has done: 'Your current AUC (0.42471) is below what you previously achieved with the same core pipeline, so the most likely cause is that you’re not actually loading the intended trained weights (silent partial-load due to key mismatches) or you’re loading an incompatible checkpoint. I keep the exact same model, preprocessing, and inference (FLAIR-only, same resize/normalization, same softmax class-1 probability), but make checkpoint loading stricter and more correct by (1) preferring checkpoints that can be loaded with zero missing/unexpected keys, and (2) fixing additional common Lightning/torch.compile key prefixes so the correct weights fully load. If no strict match exists, it fall back to the best partial match as before, but now you see a clear report of how complete the load was. This should move your score back upward toward your earlier runs without changing evaluation semantics.'
- What this solution (achieved 0.42471) has done: 'Your AUC (0.42471) suggests the model is likely running with partially-loaded or wrong weights; the smallest safe way to push score up is to (1) make checkpoint selection strongly prefer *full* (zero missing/unexpected) matches, and (2) make the actual load use `strict=True` when a full match exists (otherwise fall back to the current `strict=False`). This keeps the exact same model architecture, preprocessing, and softmax probability output, but reduces silent degradation from partial state_dict loads. I also ensure the chosen checkpoint is evaluated with the same key-renaming logic used at load time (so strict-match detection is accurate). Everything else remains unchanged and it still writes a valid `submission.csv` aligned to `sample_submission.csv`.'
- What this solution (achieved 0.42471) has done: 'I keep your exact model and preprocessing/inference pipeline, but make checkpoint discovery/load more reliable so you don’t accidentally run with random or partially-mismatched weights (the most common cause of low AUC swings like 0.42). Specifically, I (1) prefer the exact intended checkpoint if it exists, otherwise rank candidates by *how well they load into the model without mutating it during scoring*, and (2) when a strict-compatible checkpoint exists, actually load it strictly; otherwise fall back to the best partial match with a clear report. These are minimal inference-only changes that preserve evaluation semantics (same softmax class-1 probability) while aiming to move your score back up toward your earlier better runs. The script still run end-to-end and write a valid `submission.csv` aligned to `sample_submission.csv`.'

# 9. Code solution

## === cell 0
"""Module with datasets"""

import abc
from pathlib import Path
from typing import Any, Dict, List, Union

from torch.utils.data import Dataset


class BaseDataset(Dataset, abc.ABC):
    def __init__(self, list_of_paths: Union[List[Path], List[str]]):
        self.list_of_paths = list_of_paths

        self.img_key = "image"
        self.lbl_key = "label"

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        raise NotImplementedError("It is base dataset, use implementation!")

    def __len__(self):
        raise NotImplementedError("It is base dataset, use implementation!")




## === cell 1
"""Module with resnet3d"""

from functools import partial
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_inplanes():
    return [64, 128, 256, 512]


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        block_inplanes,
        n_input_channels=3,
        conv1_t_size=7,
        conv1_t_stride=1,
        no_max_pool=False,
        shortcut_type="B",
        widen_factor=1.0,
        n_classes=400,
    ):
        super().__init__()

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        self.conv1 = nn.Conv3d(
            n_input_channels,
            self.in_planes,
            kernel_size=(conv1_t_size, 7, 7),
            stride=(conv1_t_stride, 2, 2),
            padding=(conv1_t_size // 2, 3, 3),
            bias=False,
        )
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(
            block, block_inplanes[0], layers[0], shortcut_type
        )
        self.layer2 = self._make_layer(
            block, block_inplanes[1], layers[1], shortcut_type, stride=2
        )
        self.layer3 = self._make_layer(
            block, block_inplanes[2], layers[2], shortcut_type, stride=2
        )
        self.layer4 = self._make_layer(
            block, block_inplanes[3], layers[3], shortcut_type, stride=2
        )

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(block_inplanes[3] * block.expansion, n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(
            out.size(0), planes - out.size(1), out.size(2), out.size(3), out.size(4)
        )
        if out.is_cuda:
            zero_pads = zero_pads.to(out.device)

        out = torch.cat([out, zero_pads], dim=1)
        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == "A":
                downsample = partial(
                    self._downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride,
                )
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion),
                )

        layers = []
        layers.append(
            block(
                in_planes=self.in_planes,
                planes=planes,
                stride=stride,
                downsample=downsample,
            )
        )
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def generate_resnet_model(model_depth: int, **kwargs: Any) -> torch.nn.Module:
    assert model_depth in [10, 18, 34, 50, 101, 152, 200]

    if model_depth == 10:
        model = ResNet(BasicBlock, [1, 1, 1, 1], get_inplanes(), **kwargs)
    elif model_depth == 18:
        model = ResNet(BasicBlock, [2, 2, 2, 2], get_inplanes(), **kwargs)
    elif model_depth == 34:
        model = ResNet(BasicBlock, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 50:
        model = ResNet(Bottleneck, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 101:
        model = ResNet(Bottleneck, [3, 4, 23, 3], get_inplanes(), **kwargs)
    elif model_depth == 152:
        model = ResNet(Bottleneck, [3, 8, 36, 3], get_inplanes(), **kwargs)
    elif model_depth == 200:
        model = ResNet(Bottleneck, [3, 24, 36, 3], get_inplanes(), **kwargs)
    else:
        raise ValueError(f"No such model: resnet{model_depth}")

    return model




## === cell 2
from typing import Callable, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def get_preprocessing_transforms(
    img_key: str,
    original_min: float = 0.0,
    original_max: float = 200.0,
    res_min: float = 0.0,
    res_max: float = 1.0,
    spatial_size: Tuple[int, int, int] = (196, 196, 128),
) -> Callable[[Dict[str, np.ndarray]], Dict[str, np.ndarray]]:
    def _transform(item: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        img = item[img_key]

        if img.ndim == 3:
            img = img[None, ...]
        elif img.ndim != 4:
            raise ValueError(f"Unexpected image ndim={img.ndim}, expected 3 or 4")

        img = img.astype(np.float32, copy=False)

        img = np.clip(img, original_min, original_max)
        denom = (
            (original_max - original_min) if (original_max - original_min) != 0 else 1.0
        )
        img = (img - original_min) / denom
        img = img * (res_max - res_min) + res_min

        target_h, target_w, target_d = spatial_size
        target_dhw = (target_d, target_h, target_w)

        t = torch.from_numpy(img).unsqueeze(0)  # (1,C,D,H,W)
        t = F.interpolate(t, size=target_dhw, mode="trilinear", align_corners=False)
        img = t.squeeze(0).numpy()  # (C,D,H,W)

        item[img_key] = img
        return item

    return _transform




## === cell 3
"""Module with evaluation dataset"""

from pathlib import Path
from typing import List, Tuple, Union

import SimpleITK as sitk
import numpy as np


class BrainDicomEvalDataset(BaseDataset):
    def __init__(
        self,
        list_of_paths: Union[List[Path], List[str]],
        spatial_size: Tuple[int, int, int] = (196, 196, 128),
    ):
        super().__init__(list_of_paths=list_of_paths)

        self.list_of_dicom_folder_paths = list_of_paths

        self.preprocessing_transforms = get_preprocessing_transforms(
            img_key=self.img_key,
            original_min=0,
            original_max=1000,
            res_min=0,
            res_max=1,
            spatial_size=spatial_size,
        )

    def __getitem__(self, idx: int):
        dicom_folder_path = self.list_of_dicom_folder_paths[idx]
        image = self.__load_dicom(dicom_folder_path=dicom_folder_path)

        item = {self.img_key: image}
        item = self.preprocessing_transforms(item)

        return item

    def __len__(self) -> int:
        return len(self.list_of_dicom_folder_paths)

    @staticmethod
    def __load_dicom(dicom_folder_path: Union[str, Path]) -> np.ndarray:
        sitk.ProcessObject_SetGlobalWarningDisplay(False)

        series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(
            directory=str(dicom_folder_path)
        )
        if not series_ids:
            raise RuntimeError(f"No DICOM series found in: {dicom_folder_path}")

        best_series_id = None
        best_n_files = -1
        for sid in sorted(series_ids):
            fns = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(
                str(dicom_folder_path), sid
            )
            n = len(fns)
            if (n > best_n_files) or (
                n == best_n_files and (best_series_id is None or sid < best_series_id)
            ):
                best_n_files = n
                best_series_id = sid

        series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(
            str(dicom_folder_path), best_series_id
        )
        series_reader = sitk.ImageSeriesReader()
        series_reader.SetFileNames(series_file_names)
        series_reader.LoadPrivateTagsOn()
        image_and_meta: sitk.Image = series_reader.Execute()

        image = sitk.GetArrayFromImage(image=image_and_meta)  # (D,H,W)
        return image




## === cell 4
"""Module with class for model evaluation"""

import os
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


class ModelEvaluator:
    _CSV_COLUMN_NAMES = ["BraTS21ID", "MGMT_value"]

    def __init__(
        self,
        model: torch.nn.Module,
        dataset: BaseDataset,
        device: torch.device = torch.device("cpu"),
        sample_submission_path: str = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/sample_submission.csv",
    ):
        self.model = model
        self.dataset = dataset
        self.device = device
        self.sample_submission_path = sample_submission_path

    def eval(
        self, save_to_csv: bool = False, csv_filepath: str = "submission.csv"
    ) -> List[Tuple[int, float]]:
        evaluation_result = []

        self.model.eval()
        with torch.inference_mode():
            for idx, dataset_item in enumerate(
                tqdm(self.dataset, postfix="Evaluation...")
            ):
                image = dataset_item[self.dataset.img_key]
                image_path = self.dataset.list_of_paths[idx]
                image_idx = self._get_image_idx(image_path=image_path)

                label = self._predict_one_item(image=image)
                evaluation_result.append((image_idx, float(label)))

        if save_to_csv:
            self._save_evaluation_result_to_csv(
                evaluation_result=evaluation_result, filename=csv_filepath
            )

        return evaluation_result

    def _predict_one_item(
        self,
        image: np.ndarray,
    ) -> float:
        image_tensor = self._to_tensor(image=image).float()
        image_tensor = image_tensor.unsqueeze(0).to(self.device)  # (1,C,D,H,W)

        result = self.model(image_tensor)
        result = torch.softmax(result, dim=1).cpu()
        label = result[0, 1].detach().numpy()
        return label

    def _save_evaluation_result_to_csv(
        self,
        evaluation_result: List[Tuple[int, float]],
        filename: str = "submission.csv",
    ) -> None:
        result_df = pd.DataFrame(
            data=evaluation_result,
            columns=self._CSV_COLUMN_NAMES,
        )
        result_df["BraTS21ID"] = (
            result_df["BraTS21ID"].astype(int).astype(str).str.zfill(5)
        )

        if os.path.exists(self.sample_submission_path):
            sample_df = pd.read_csv(self.sample_submission_path)
            sample_df["BraTS21ID"] = sample_df["BraTS21ID"].astype(str).str.zfill(5)
            merged = sample_df[["BraTS21ID"]].merge(
                result_df, on="BraTS21ID", how="left"
            )
            merged["MGMT_value"] = merged["MGMT_value"].astype(float).fillna(0.5)
            merged.to_csv(filename, index=False)
        else:
            result_df = result_df.sort_values("BraTS21ID").reset_index(drop=True)
            result_df.to_csv(filename, index=False)

    @staticmethod
    def _to_tensor(image: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(image)

    @staticmethod
    def _get_image_idx(image_path: Union[str, Path]) -> int:
        p = Path(str(image_path))
        if p.name.isdigit():
            return int(p.name)
        if p.parent.name.isdigit():
            return int(p.parent.name)
        if p.parent.parent.name.isdigit():
            return int(p.parent.parent.name)
        raise ValueError(f"Could not extract numeric BraTS21ID from path: {image_path}")




## === cell 5
"""Module with evaluation"""

import glob
import os
import random
from typing import Dict, Optional, Tuple

import numpy as np
import torch


def rename_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    new_state_dict = {}
    for layer_name, layer_weights in state_dict.items():
        layer_name = layer_name.replace("model._orig_mod.", "")
        layer_name = layer_name.replace("_orig_mod.", "")
        layer_name = layer_name.replace("model.", "")
        layer_name = layer_name.replace("net.", "")
        layer_name = layer_name.replace("module.", "")
        new_state_dict[layer_name] = layer_weights
    return new_state_dict


def _extract_state_dict(ckpt_path: str) -> Dict[str, torch.Tensor]:
    meta = torch.load(ckpt_path, map_location="cpu")
    if isinstance(meta, dict) and "state_dict" in meta:
        sd = rename_keys(meta["state_dict"])
    elif isinstance(meta, dict):
        sd = rename_keys(meta)
    else:
        raise RuntimeError(f"Unsupported checkpoint format at {ckpt_path}")
    return sd


def _try_load_report(
    model: torch.nn.Module, sd: Dict[str, torch.Tensor], strict: bool
) -> Tuple[bool, int, int]:
    try:
        tmp = type(model)(*[])  # will fail for our ResNet; handled below
        _ = tmp
    except Exception:
        tmp = None

    import copy

    tmp = copy.deepcopy(model).cpu()
    try:
        tmp.load_state_dict(sd, strict=strict)
        return True, 0, 0
    except Exception:
        if strict:
            return False, -1, -1
        missing, unexpected = tmp.load_state_dict(sd, strict=False)
        return False, len(missing), len(unexpected)


def _score_checkpoint_for_model(
    path: str, model: torch.nn.Module
) -> Tuple[int, float, int]:
    try:
        sd = _extract_state_dict(path)
        msd = model.state_dict()
        denom = max(len(msd), 1)
        shape_match = 0
        for k, v in sd.items():
            if k in msd and hasattr(v, "shape") and v.shape == msd[k].shape:
                shape_match += 1
        frac_match = float(shape_match) / float(denom)

        strict_ok, _, _ = _try_load_report(model, sd, strict=True)
        return 1 if strict_ok else 0, frac_match, shape_match
    except Exception:
        return 0, 0.0, 0


def find_checkpoint(
    preferred_path: str, model: Optional[torch.nn.Module] = None
) -> Optional[str]:
    if preferred_path and os.path.exists(preferred_path):
        return preferred_path

    candidates = []
    for ext in ("*.ckpt", "*.pth", "*.pt"):
        candidates.extend(glob.glob(f"/kaggle/input/**/{ext}", recursive=True))

    if not candidates:
        return None

    preferred_base = os.path.basename(preferred_path) if preferred_path else ""
    ranked = []
    for p in candidates:
        lp = p.lower()
        score = 0

        if preferred_base and os.path.basename(p) == preferred_base:
            score += 1000

        if "rsna" in lp or "brats" in lp or "mgmt" in lp or "radiogenomic" in lp:
            score += 30
        if "brain" in lp:
            score += 10
        if "epoch" in lp or "step" in lp:
            score += 3

        try:
            size = os.path.getsize(p)
        except OSError:
            size = 0

        strict_ok = 0
        frac_match = 0.0
        shape_match = 0
        if model is not None:
            strict_ok, frac_match, shape_match = _score_checkpoint_for_model(p, model)

        ranked.append((strict_ok, frac_match, shape_match, score, size, p))

    ranked.sort(key=lambda x: (-x[0], -x[2], -x[1], -x[3], -x[4], x[5]))
    return ranked[0][5]


def load_model_weights(
    model: torch.nn.Module, ckpt_path: str
) -> Tuple[torch.nn.Module, bool]:
    sd = _extract_state_dict(ckpt_path)

    sd = {
        k: (v.float() if torch.is_tensor(v) and v.dtype != torch.float32 else v)
        for k, v in sd.items()
    }

    strict_ok, _, _ = _try_load_report(model, sd, strict=True)
    if strict_ok:
        model.load_state_dict(sd, strict=True)
        return model, True

    missing, unexpected = model.load_state_dict(sd, strict=False)
    loaded_ok = (len(missing) == 0) and (len(unexpected) == 0)
    print(
        f"Checkpoint load report (non-strict): missing={len(missing)} unexpected={len(unexpected)} "
        f"(example missing: {missing[:5]})"
    )
    return model, loaded_ok


if __name__ == "__main__":
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    except Exception:
        pass

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    try:
        torch.use_deterministic_algorithms(False)
    except Exception:
        pass

    pattern = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/test/**/FLAIR"
    model_path = "/kaggle/input/brainclassificationmodel/epoch28-step3972.ckpt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    list_of_dicom_folder_paths = sorted(glob.glob(pathname=pattern, recursive=True))
    if len(list_of_dicom_folder_paths) == 0:
        raise RuntimeError(f"No folders found for pattern: {pattern}")

    dataset = BrainDicomEvalDataset(
        list_of_paths=list_of_dicom_folder_paths,
    )

    model_depth = 10
    num_input_channels = 1
    num_classes = 2

    resnet = generate_resnet_model(
        model_depth=model_depth,
        n_input_channels=num_input_channels,
        n_classes=num_classes,
    )

    resnet.to(device=device)

    ckpt = find_checkpoint(model_path, model=resnet)
    if ckpt is not None:
        try:
            resnet, loaded_ok = load_model_weights(resnet, ckpt)
            print(f"Loaded checkpoint: {ckpt} (strict_match={loaded_ok})")
        except Exception as e:
            print(f"WARNING: Failed to load checkpoint '{ckpt}': {e}")
            print("Proceeding with randomly initialized weights (will score poorly).")
    else:
        print(
            "WARNING: No checkpoint found under /kaggle/input; proceeding with random weights."
        )

    model_evaluator = ModelEvaluator(
        model=resnet,
        dataset=dataset,
        device=device,
        sample_submission_path="/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/sample_submission.csv",
    )

    model_evaluator.eval(save_to_csv=True, csv_filepath="submission.csv")
    if not os.path.exists("submission.csv"):
        raise RuntimeError("submission.csv was not created")
    print("Wrote submission.csv")
