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

No external packages required in the script and installed.

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

0.5795937446880843

# 6. Current score

0.5

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'I fix the checkpoint loading failure by removing the hard dependency on an external Kaggle Dataset (`brainclassificationeffnet`) and instead initialize the same EfficientNet3D model without pretrained weights when no checkpoint is found. To keep the pipeline producing a valid submission, inference still run end-to-end and write `submission.csv` with the required columns and ordering. I also make DICOM→NIfTI conversion use a unique temp filename per case to avoid collisions and intermittent read/write issues during iteration. These changes are execution-stability focused and preserve the existing model architecture and prediction semantics (softmax probability for class 1).'
- What this solution (achieved 0.5) has done: 'Your current 0.5 AUC is consistent with essentially random (uninformative) predictions because the intended checkpoint is missing; the smallest legitimate way to move toward the 0.5796 target is to actually load a model that learned something. I keep the exact same EfficientNet3D architecture and inference semantics, but add a lightweight “fallback checkpoint search” that looks for any `.ckpt` shipped inside the competition dataset folder (or working directory) and loads it if present. If no checkpoint exists, I keep your deterministic untrained behavior (still yields ~0.5) rather than changing the modeling approach. I also make the temp NIfTI filename include the case id to avoid rare collisions if evaluation order changes, without altering preprocessing math.'
- What this solution (achieved 0.5) has done: 'The current 0.5 AUC is consistent with effectively constant/random predictions, and the smallest legitimate nudge toward the 0.5796 target is to improve inference determinism and ensure we actually use the trained weights if a checkpoint is present. I keep your exact model, preprocessing, and softmax probability output, but (1) make sure the model is seeded *before* initialization so the “no checkpoint” fallback is stable (not unintentionally varying), and (2) improve checkpoint loading robustness by accepting common Lightning/torch checkpoint formats and handling `module.` prefixes—this can turn “checkpoint found but not loaded correctly” into “checkpoint used”, which is the only realistic way to move above 0.5 without changing core logic. I also align the submission rows to `sample_submission.csv` ordering (same predictions, safer ID alignment) to avoid any rare formatting/index mismatches that can silently hurt score. All changes are minimal and keep evaluation semantics identical.'
- What this solution (achieved 0.5) has done: 'Your 0.5 AUC strongly suggests the model is still running with effectively untrained/random weights, so the smallest legitimate move toward the 0.5796 target is to (a) reliably find and correctly load any available checkpoint in common Torch/Lightning formats and (b) ensure we don’t silently fail strict loading due to key-prefix or shape mismatches. I keep the exact same EfficientNet3D architecture, preprocessing, and “softmax probability of class 1” output, but make checkpoint resolution prefer this competition’s input tree and add robust state_dict extraction + safe prefix stripping. If no checkpoint exists, behavior remains the same (still produces a valid submission), but if a checkpoint is present this should nudge the score upward toward your target without changing evaluation semantics. I also keep submission row alignment to `sample_submission.csv` to prevent any ID-order mismatches.'
- What this solution (achieved 0.47588) has done: 'We need to move AUC up from 0.5 toward 0.5796; with your current pipeline the only realistic minimal improvement (without changing model/preprocess/inference semantics) is to ensure any available trained weights are actually discovered and loaded correctly. I (1) broaden and prioritize checkpoint discovery to include common Kaggle “output” artifacts (e.g., `/kaggle/input/**/output/**`) and prefer larger/more-likely “best” files, (2) make checkpoint loading more robust by handling nested keys like `ema_state_dict` and stripping additional common prefixes, while still keeping the same architecture and softmax probability output, and (3) add a sanity warning and a deterministic fallback probability based on the model’s own output mean if loading fails (still legitimate inference; no label use) to avoid near-constant 0.5 when the network outputs are degenerate. These are minimal, score-relevant changes and keep the core logic identical: same EfficientNet3D, same preprocessing, same forward+softmax for class-1 probability, same submission format and ordering.'
- What this solution (achieved 0.47588) has done: 'Your score gap is ~0.1037 AUC below target, and with this fixed architecture/inference pipeline the only realistic minimal way to move upward is to ensure the model is actually running with meaningful weights and in true inference mode. I make checkpoint loading robust for common “Lightning” and “torch.save(model)” formats by also accepting full `nn.Module` objects and by filtering state_dict keys to those that match the current model’s shapes (prevents silent bad/partial loads that behave like random). I also ensure BatchNorm statistics aren’t accidentally overwritten by loading (keep `model.eval()` and load on CPU before `.to(device)`), and add a tiny safety clamp on probabilities (AUC-safe, avoids exact 0/1). These changes preserve your core model, preprocessing, and softmax probability semantics, but should nudge you above your current near-random behavior when any usable weights exist.'
- What this solution (achieved 0.5) has done: 'Your current AUC (0.47588) is below the target (0.5796), so we should cautiously increase performance with minimal, score-relevant changes. The most likely issue is that a checkpoint is either not being found or not being loaded well enough (too few matching keys), leaving the model effectively random; I make checkpoint discovery prefer “non-input” artifacts (working/output) and explicitly pick the checkpoint that yields the highest fraction of loadable parameters. I also fix the running-mean initialization bug (it currently biases the mean toward 1.0 and can trigger the “degenerate output” nudge incorrectly), and remove the post-hoc probability nudge so we don’t distort model scores in an AUC-sensitive way. These changes keep the same EfficientNet3D architecture, preprocessing, and “softmax probability for class 1” semantics, but should move you upward toward the target when any usable weights exist.'
- What this solution (achieved 0.5) has done: 'We need to move AUC up from 0.5 toward 0.5796 (higher-is-better), and with your fixed model/preprocessing/inference the only minimal legitimate lever is to ensure you actually load meaningful trained weights when they exist. I keep the exact EfficientNet3D architecture and softmax(class=1) output, but improve checkpoint discovery to prioritize likely RSNA artifacts (EfficientNet-B5, 3D, MGMT) and to avoid picking unrelated `.pth/.pt` from other notebooks that can partially-load yet behave random. I also make checkpoint selection stricter by requiring a minimum state_dict shape-coverage before accepting it, so we either use a highly-compatible checkpoint (better than 0.5) or cleanly fall back (still valid submission) rather than silently loading a poor mismatch. Finally, I keep submission alignment to `sample_submission.csv` while ensuring the evaluated path list is sorted for stable ID→prediction mapping.'

# 9. Code solution

## === cell 0
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
import collections
import math
import re
from functools import partial

import torch
from torch import nn
from torch.nn import functional as F


class MBConvBlock3D(nn.Module):
    def __init__(self, block_args, global_params):
        super().__init__()
        self._block_args = block_args
        self._bn_mom = 1 - global_params.batch_norm_momentum
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = (self._block_args.se_ratio is not None) and (
            0 < self._block_args.se_ratio <= 1
        )
        self.id_skip = block_args.id_skip

        Conv3d = get_same_padding_conv3d(image_size=global_params.image_size)

        inp = self._block_args.input_filters
        oup = self._block_args.input_filters * self._block_args.expand_ratio
        if self._block_args.expand_ratio != 1:
            self._expand_conv = Conv3d(
                in_channels=inp, out_channels=oup, kernel_size=1, bias=False
            )
            self._bn0 = nn.BatchNorm3d(
                num_features=oup, momentum=self._bn_mom, eps=self._bn_eps
            )

        k = self._block_args.kernel_size
        s = self._block_args.stride
        self._depthwise_conv = Conv3d(
            in_channels=oup,
            out_channels=oup,
            groups=oup,
            kernel_size=k,
            stride=s,
            bias=False,
        )
        self._bn1 = nn.BatchNorm3d(
            num_features=oup, momentum=self._bn_mom, eps=self._bn_eps
        )

        if self.has_se:
            num_squeezed_channels = max(
                1, int(self._block_args.input_filters * self._block_args.se_ratio)
            )
            self._se_reduce = Conv3d(
                in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1
            )
            self._se_expand = Conv3d(
                in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1
            )

        final_oup = self._block_args.output_filters
        self._project_conv = Conv3d(
            in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False
        )
        self._bn2 = nn.BatchNorm3d(
            num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps
        )
        self._swish = MemoryEfficientSwish()

    def forward(self, inputs, drop_connect_rate=None):
        x = inputs
        if self._block_args.expand_ratio != 1:
            x = self._swish(self._bn0(self._expand_conv(inputs)))
        x = self._swish(self._bn1(self._depthwise_conv(x)))

        if self.has_se:
            x_squeezed = F.adaptive_avg_pool3d(x, 1)
            x_squeezed = self._se_expand(self._swish(self._se_reduce(x_squeezed)))
            x = torch.sigmoid(x_squeezed) * x

        x = self._bn2(self._project_conv(x))

        input_filters, output_filters = (
            self._block_args.input_filters,
            self._block_args.output_filters,
        )
        if (
            self.id_skip
            and self._block_args.stride == 1
            and input_filters == output_filters
        ):
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs
        return x

    def set_swish(self, memory_efficient=True):
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()


class EfficientNet3D(nn.Module):
    def __init__(self, blocks_args=None, global_params=None, in_channels=3):
        super().__init__()
        assert isinstance(blocks_args, list), "blocks_args should be a list"
        assert len(blocks_args) > 0, "block args must be greater than 0"
        self._global_params = global_params
        self._blocks_args = blocks_args

        Conv3d = get_same_padding_conv3d(image_size=global_params.image_size)

        bn_mom = 1 - self._global_params.batch_norm_momentum
        bn_eps = self._global_params.batch_norm_epsilon

        out_channels = round_filters(32, self._global_params)
        self._conv_stem = Conv3d(
            in_channels, out_channels, kernel_size=3, stride=2, bias=False
        )
        self._bn0 = nn.BatchNorm3d(
            num_features=out_channels, momentum=bn_mom, eps=bn_eps
        )

        self._blocks = nn.ModuleList([])
        for block_args in self._blocks_args:
            block_args = block_args._replace(
                input_filters=round_filters(
                    block_args.input_filters, self._global_params
                ),
                output_filters=round_filters(
                    block_args.output_filters, self._global_params
                ),
                num_repeat=round_repeats(block_args.num_repeat, self._global_params),
            )

            self._blocks.append(MBConvBlock3D(block_args, self._global_params))
            if block_args.num_repeat > 1:
                block_args = block_args._replace(
                    input_filters=block_args.output_filters, stride=1
                )
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(MBConvBlock3D(block_args, self._global_params))

        in_channels = block_args.output_filters
        out_channels = round_filters(1280, self._global_params)
        self._conv_head = Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        self._bn1 = nn.BatchNorm3d(
            num_features=out_channels, momentum=bn_mom, eps=bn_eps
        )

        self._avg_pooling = nn.AdaptiveAvgPool3d(1)
        self._dropout = nn.Dropout(self._global_params.dropout_rate)
        self._fc = nn.Linear(out_channels, self._global_params.num_classes)
        self._swish = MemoryEfficientSwish()

    def set_swish(self, memory_efficient=True):
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()
        for block in self._blocks:
            block.set_swish(memory_efficient)

    def extract_features(self, inputs):
        x = self._swish(self._bn0(self._conv_stem(inputs)))
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
        x = self._swish(self._bn1(self._conv_head(x)))
        return x

    def forward(self, inputs):
        bs = inputs.size(0)
        x = self.extract_features(inputs)
        if self._global_params.include_top:
            x = self._avg_pooling(x)
            x = x.view(bs, -1)
            x = self._dropout(x)
            x = self._fc(x)
        return x

    @classmethod
    def from_name(cls, model_name, override_params=None, in_channels=3):
        cls._check_model_name_is_valid(model_name)
        blocks_args, global_params = get_model_params(model_name, override_params)
        return cls(blocks_args, global_params, in_channels)

    @classmethod
    def get_image_size(cls, model_name):
        cls._check_model_name_is_valid(model_name)
        _, _, res, _ = efficientnet_params(model_name)
        return res

    @classmethod
    def _check_model_name_is_valid(cls, model_name):
        valid_models = ["efficientnet-b" + str(i) for i in range(9)]
        if model_name not in valid_models:
            raise ValueError("model_name should be one of: " + ", ".join(valid_models))


GlobalParams = collections.namedtuple(
    "GlobalParams",
    [
        "batch_norm_momentum",
        "batch_norm_epsilon",
        "dropout_rate",
        "num_classes",
        "width_coefficient",
        "depth_coefficient",
        "depth_divisor",
        "min_depth",
        "drop_connect_rate",
        "image_size",
        "include_top",
    ],
)

BlockArgs = collections.namedtuple(
    "BlockArgs",
    [
        "kernel_size",
        "num_repeat",
        "input_filters",
        "output_filters",
        "expand_ratio",
        "id_skip",
        "stride",
        "se_ratio",
    ],
)

GlobalParams.__new__.__defaults__ = (None,) * len(GlobalParams._fields)  # type: ignore
BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)  # type: ignore


class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


def round_filters(filters, global_params):
    multiplier = global_params.width_coefficient
    if not multiplier:
        return filters
    divisor = global_params.depth_divisor
    min_depth = global_params.min_depth
    filters *= multiplier
    min_depth = min_depth or divisor
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    if new_filters < 0.9 * filters:
        new_filters += divisor
    return int(new_filters)


def round_repeats(repeats, global_params):
    multiplier = global_params.depth_coefficient
    if not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))


def drop_connect(inputs, p, training):
    if not training:
        return inputs
    batch_size = inputs.shape[0]
    keep_prob = 1 - p
    random_tensor = keep_prob
    random_tensor += torch.rand(
        [batch_size, 1, 1, 1, 1], dtype=inputs.dtype, device=inputs.device
    )
    binary_tensor = torch.floor(random_tensor)
    output = inputs / keep_prob * binary_tensor
    return output


def get_same_padding_conv3d(image_size=None):
    if image_size is None:
        return Conv3dDynamicSamePadding
    else:
        return partial(Conv3dStaticSamePadding, image_size=image_size)


class Conv3dDynamicSamePadding(nn.Conv3d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super().__init__(
            in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias
        )
        self.stride = self.stride if len(self.stride) == 3 else [self.stride[0]] * 3

    def forward(self, x):
        ih, iw, iz = x.size()[-3:]
        kh, kw, kz = self.weight.size()[-3:]
        sh, sw, sz = self.stride
        oh, ow, oz = math.ceil(ih / sh), math.ceil(iw / sw), math.ceil(iz / oz)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        pad_z = max((oz - 1) * self.stride[2] + (kz - 1) * self.dilation[2] + 1 - iz, 0)
        if pad_h > 0 or pad_w > 0 or pad_z > 0:
            x = F.pad(
                x,
                [
                    pad_w // 2,
                    pad_w - pad_w // 2,
                    pad_h // 2,
                    pad_h - pad_h // 2,
                    pad_z // 2,
                    pad_z - pad_z // 2,
                ],
            )
        return F.conv3d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input


class Conv3dStaticSamePadding(nn.Conv3d):
    def __init__(
        self, in_channels, out_channels, kernel_size, image_size=None, **kwargs
    ):
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)
        self.stride = self.stride if len(self.stride) == 3 else [self.stride[0]] * 3
        assert image_size is not None
        ih, iw, iz = (
            image_size
            if type(image_size) == list
            else [image_size, image_size, image_size]
        )
        kh, kw, kz = self.weight.size()[-3:]
        sh, sw, sz = self.stride
        oh, ow, oz = math.ceil(ih / sh), math.ceil(iw / sw), math.ceil(iz / sz)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        pad_z = max((oz - 1) * self.stride[2] + (kz - 1) * self.dilation[2] + 1 - iz, 0)
        if pad_h > 0 or pad_w > 0 or pad_z > 0:
            self._pad = (
                pad_w // 2,
                pad_w - pad_w // 2,
                pad_h // 2,
                pad_h - pad_h // 2,
                pad_z // 2,
                pad_z - pad_z // 2,
            )
        else:
            self._pad = None

    def forward(self, x):
        if self._pad is not None:
            x = F.pad(x, list(self._pad))
        return F.conv3d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


def efficientnet_params(model_name):
    params_dict = {
        "efficientnet-b0": (1.0, 1.0, 224, 0.2),
        "efficientnet-b1": (1.0, 1.1, 240, 0.2),
        "efficientnet-b2": (1.1, 1.2, 260, 0.3),
        "efficientnet-b3": (1.2, 1.4, 300, 0.3),
        "efficientnet-b4": (1.4, 1.8, 380, 0.4),
        "efficientnet-b5": (1.6, 2.2, 456, 0.4),
        "efficientnet-b6": (1.8, 2.6, 528, 0.5),
        "efficientnet-b7": (2.0, 3.1, 600, 0.5),
        "efficientnet-b8": (2.2, 3.6, 672, 0.5),
        "efficientnet-l2": (4.3, 5.3, 800, 0.5),
    }
    return params_dict[model_name]


class BlockDecoder(object):
    @staticmethod
    def _decode_block_string(block_string):
        assert isinstance(block_string, str)
        ops = block_string.split("_")
        options = {}
        for op in ops:
            splits = re.split(r"(\d.*)", op)
            if len(splits) >= 2:
                key, value = splits[:2]
                options[key] = value

        assert ("s" in options and len(options["s"]) == 1) or (
            len(options["s"]) == 3
            and options["s"][0] == options["s"][1] == options["s"][2]
        )

        return BlockArgs(
            kernel_size=int(options["k"]),
            num_repeat=int(options["r"]),
            input_filters=int(options["i"]),
            output_filters=int(options["o"]),
            expand_ratio=int(options["e"]),
            id_skip=("noskip" not in block_string),
            se_ratio=float(options["se"]) if "se" in options else None,
            stride=[int(options["s"][0])],
        )

    @staticmethod
    def decode(string_list):
        assert isinstance(string_list, list)
        blocks_args = []
        for block_string in string_list:
            blocks_args.append(BlockDecoder._decode_block_string(block_string))
        return blocks_args


def efficientnet3d(
    width_coefficient=None,
    depth_coefficient=None,
    dropout_rate=0.2,
    drop_connect_rate=0.2,
    image_size=None,
    num_classes=1000,
    include_top=True,
):
    blocks_args = [
        "r1_k3_s222_e1_i32_o16_se0.25",
        "r2_k3_s222_e6_i16_o24_se0.25",
        "r2_k5_s222_e6_i24_o40_se0.25",
        "r3_k3_s222_e6_i40_o80_se0.25",
        "r3_k5_s111_e6_i80_o112_se0.25",
        "r4_k5_s222_e6_i112_o192_se0.25",
        "r1_k3_s111_e6_i192_o320_se0.25",
    ]
    blocks_args = BlockDecoder.decode(blocks_args)

    global_params = GlobalParams(
        batch_norm_momentum=0.99,
        batch_norm_epsilon=1e-3,
        dropout_rate=dropout_rate,
        drop_connect_rate=drop_connect_rate,
        num_classes=num_classes,
        width_coefficient=width_coefficient,
        depth_coefficient=depth_coefficient,
        depth_divisor=8,
        min_depth=None,
        image_size=image_size,
        include_top=include_top,
    )

    return blocks_args, global_params


def get_model_params(model_name, override_params):
    if model_name.startswith("efficientnet"):
        w, d, s, p = efficientnet_params(model_name)
        blocks_args, global_params = efficientnet3d(
            width_coefficient=w, depth_coefficient=d, dropout_rate=p, image_size=s
        )
    else:
        raise NotImplementedError("model name is not pre-defined: %s" % model_name)
    if override_params:
        global_params = global_params._replace(**override_params)
    return blocks_args, global_params




## === cell 2
from typing import Callable, Tuple

import numpy as np


def _resize3d_nearest(x: np.ndarray, out_shape: Tuple[int, int, int]) -> np.ndarray:
    in_d, in_h, in_w = x.shape
    out_d, out_h, out_w = out_shape
    zd = (np.linspace(0, in_d - 1, out_d)).round().astype(np.int64)
    yh = (np.linspace(0, in_h - 1, out_h)).round().astype(np.int64)
    xw = (np.linspace(0, in_w - 1, out_w)).round().astype(np.int64)
    return x[zd[:, None, None], yh[None, :, None], xw[None, None, :]]


def get_preprocessing_transforms(
    img_key: str,
    original_min: float = 0.0,
    original_max: float = 200.0,
    res_min: float = 0.0,
    res_max: float = 1.0,
    spatial_size: Tuple[int, int, int] = (196, 196, 128),
) -> Callable[[dict], dict]:
    def _transform(item: dict) -> dict:
        img = item[img_key]
        img = img.astype(np.float32, copy=False)

        if img.ndim == 3:
            img = img[None, ...]
        elif img.ndim == 4 and img.shape[0] != 1:
            pass

        img = np.clip(img, original_min, original_max)
        if original_max > original_min:
            img = (img - original_min) / (original_max - original_min)
        img = img * (res_max - res_min) + res_min

        c, d, h, w = img.shape
        out_d, out_h, out_w = spatial_size
        resized = np.empty((c, out_d, out_h, out_w), dtype=np.float32)
        for ci in range(c):
            resized[ci] = _resize3d_nearest(img[ci], (out_d, out_h, out_w))
        item[img_key] = resized
        return item

    return _transform




## === cell 3
from pathlib import Path
from typing import Dict, List, Tuple, Union

import nibabel as nib
import SimpleITK as sitk


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
            original_min=-200,
            original_max=2500,
            res_min=0,
            res_max=1,
            spatial_size=spatial_size,
        )

    def __getitem__(self, idx: int) -> Dict:
        dicom_folder_path = self.list_of_dicom_folder_paths[idx]

        case_id = str(Path(dicom_folder_path).parents[0].name)
        temp_nii = Path(f"temp_{case_id}_{idx:05d}.nii")

        self.__save_dicom(dicom_folder_path=dicom_folder_path, out_nii_path=temp_nii)
        image = self._load_ct(ct_path=temp_nii)

        item = {self.img_key: image}
        item = self.preprocessing_transforms(item)

        try:
            temp_nii.unlink(missing_ok=True)
        except Exception:
            pass

        return item

    def __len__(self) -> int:
        return len(self.list_of_dicom_folder_paths)

    @staticmethod
    def __save_dicom(
        dicom_folder_path: Union[str, Path], out_nii_path: Union[str, Path]
    ) -> None:
        sitk.ProcessObject_SetGlobalWarningDisplay(False)

        series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(
            directory=str(dicom_folder_path)
        )
        if not series_ids:
            raise FileNotFoundError(f"No DICOM series found in: {dicom_folder_path}")

        series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(
            str(dicom_folder_path), series_ids[0]
        )
        series_reader = sitk.ImageSeriesReader()
        series_reader.SetFileNames(series_file_names)
        series_reader.LoadPrivateTagsOn()
        image = series_reader.Execute()

        sitk.WriteImage(image=image, fileName=str(out_nii_path), useCompression=False)

    @staticmethod
    def _load_ct(ct_path: Union[str, Path]) -> np.ndarray:
        ct_path = str(ct_path)

        ct_all_info: nib.Nifti1Image = nib.load(filename=ct_path)
        orig_ornt = nib.io_orientation(ct_all_info.affine)
        targ_ornt = nib.orientations.axcodes2ornt(axcodes="LPS")
        transform = nib.orientations.ornt_transform(
            start_ornt=orig_ornt, end_ornt=targ_ornt
        )
        img_ornt = ct_all_info.as_reoriented(ornt=transform)

        return img_ornt.get_fdata(dtype=np.float32)




## === cell 4
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

        self._running_prob_mean = 0.0
        self._running_n = 0

    def eval(
        self, save_to_csv: bool = False, csv_filepath: str = "submission.csv"
    ) -> List[Tuple[int, float]]:
        evaluation_result: List[Tuple[int, float]] = []

        for idx, dataset_item in enumerate(tqdm(self.dataset, postfix="Evaluation...")):
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

    def _predict_one_item(self, image: np.ndarray) -> float:
        image_tensor = self._to_tensor(image=image).float()
        image_tensor = image_tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            result = self.model(image_tensor)
            result = torch.softmax(result, dim=1).cpu()
        prob = float(result[0, 1].item())

        prob = float(min(0.999999, max(1e-6, prob)))

        self._running_n += 1
        self._running_prob_mean += (prob - self._running_prob_mean) / self._running_n

        return prob

    def _save_evaluation_result_to_csv(
        self,
        evaluation_result: List[Tuple[int, float]],
        filename: str = "submission.csv",
    ) -> None:
        result_df = pd.DataFrame(data=evaluation_result, columns=self._CSV_COLUMN_NAMES)
        result_df["BraTS21ID"] = (
            result_df["BraTS21ID"].astype(int).map(lambda x: f"{x:05d}")
        )
        result_df = (
            result_df.sort_values("BraTS21ID")
            .drop_duplicates("BraTS21ID", keep="first")
            .reset_index(drop=True)
        )

        if os.path.exists(self.sample_submission_path):
            sub = pd.read_csv(self.sample_submission_path)
            sub["BraTS21ID"] = sub["BraTS21ID"].astype(str).str.zfill(5)
            result_df = sub[["BraTS21ID"]].merge(result_df, on="BraTS21ID", how="left")
            result_df["MGMT_value"] = result_df["MGMT_value"].fillna(0.5)

        result_df.to_csv(filename, index=False)

    @staticmethod
    def _to_tensor(image: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(image)

    @staticmethod
    def _get_image_idx(image_path: Union[str, Path]) -> int:
        image_path = str(image_path)
        image_case_name = image_path.split(os.sep)[-2]
        return int(image_case_name)




## === cell 5
import glob
import os
from typing import Dict, Optional, Tuple

import torch


def rename_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    new_state_dict = {}
    for layer_name, layer_weights in state_dict.items():
        for prefix in ("model.", "module.", "net.", "backbone.", "encoder."):
            if layer_name.startswith(prefix):
                layer_name = layer_name[len(prefix) :]
        new_state_dict[layer_name] = layer_weights
    return new_state_dict


def _resolve_first_existing(patterns):
    for p in patterns:
        matches = glob.glob(p, recursive=True)
        if matches:
            return matches
    return []


def _checkpoint_score(path: str) -> Tuple[int, int]:
    base = os.path.basename(path).lower()
    name_bonus = 0
    for token, bonus in [
        ("mgmt", 10),
        ("radiogenomic", 6),
        ("brats", 6),
        ("3d", 5),
        ("efficientnet", 5),
        ("b5", 5),
        ("best", 5),
        ("epoch", 3),
        ("last", 2),
        ("final", 2),
        ("model", 1),
    ]:
        if token in base:
            name_bonus += bonus
    try:
        size = os.path.getsize(path)
    except Exception:
        size = 0
    return (name_bonus, size)


def _find_checkpoint_fallback() -> Optional[str]:
    ckpt_globs = [
        "/kaggle/working/**/*.ckpt",
        "/kaggle/working/**/*.pth",
        "/kaggle/working/**/*.pt",
        "/kaggle/output/**/*.ckpt",
        "/kaggle/output/**/*.pth",
        "/kaggle/output/**/*.pt",
        "/kaggle/input/**/output/**/*.ckpt",
        "/kaggle/input/**/output/**/*.pth",
        "/kaggle/input/**/output/**/*.pt",
        "./**/*.ckpt",
        "./**/*.pth",
        "./**/*.pt",
    ]

    hits = []
    for g in ckpt_globs:
        hits.extend(glob.glob(g, recursive=True))

    hits = [h for h in hits if os.path.isfile(h)]
    if not hits:
        return None

    hits = sorted(hits, key=lambda p: _checkpoint_score(p), reverse=True)
    return hits[0]


def _extract_state_dict(ckpt_obj: object) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt_obj, torch.nn.Module):
        return ckpt_obj.state_dict()

    if isinstance(ckpt_obj, dict):
        for k in (
            "state_dict",
            "model_state_dict",
            "model",
            "ema_state_dict",
            "weights",
        ):
            if k in ckpt_obj and isinstance(ckpt_obj[k], dict):
                return ckpt_obj[k]  # type: ignore[return-value]
        if all(isinstance(k, str) for k in ckpt_obj.keys()):
            tensor_like = [v for v in ckpt_obj.values() if torch.is_tensor(v)]
            if len(tensor_like) > 0:
                return ckpt_obj  # type: ignore[return-value]
    raise ValueError(
        "Unsupported checkpoint format; expected dict with state_dict/model_state_dict/model/ema_state_dict/weights, raw state_dict, or nn.Module."
    )


def _filter_state_dict_by_shape(
    model: torch.nn.Module, sd: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    msd = model.state_dict()
    out: Dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        if k in msd and torch.is_tensor(v) and tuple(v.shape) == tuple(msd[k].shape):
            out[k] = v
    return out


def _state_dict_match_ratio(
    model: torch.nn.Module, sd: Dict[str, torch.Tensor]
) -> float:
    msd = model.state_dict()
    if len(msd) == 0:
        return 0.0
    matched = 0
    for k, v in sd.items():
        if k in msd and torch.is_tensor(v) and tuple(v.shape) == tuple(msd[k].shape):
            matched += 1
    return matched / float(len(msd))


def _load_checkpoint_into_model(model: torch.nn.Module, model_path: str) -> bool:
    try:
        ckpt_obj = torch.load(model_path, map_location="cpu")
        sd = _extract_state_dict(ckpt_obj)
        sd = rename_keys(sd)
        sd = _filter_state_dict_by_shape(model=model, sd=sd)

        if len(sd) == 0:
            return False

        model.load_state_dict(sd, strict=False)
        return True
    except Exception:
        return False


def _select_best_checkpoint_by_coverage(model: torch.nn.Module, paths) -> Optional[str]:
    best_path = None
    best_key = (-1.0, -1, -1)  # (match_ratio, name_bonus, size)
    for p in paths:
        try:
            obj = torch.load(p, map_location="cpu")
            sd = rename_keys(_extract_state_dict(obj))
            ratio = _state_dict_match_ratio(model=model, sd=sd)
            bonus, size = _checkpoint_score(p)
            key = (ratio, bonus, size)
            if key > best_key:
                best_key = key
                best_path = p
        except Exception:
            continue
    return best_path


if __name__ == "__main__":
    flair_patterns = [
        "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/test/**/FLAIR",
        "../input/rsna-miccai-brain-tumor-radiogenomic-classification/test/**/FLAIR",
        "/kaggle/data/rsna-miccai-brain-tumor-radiogenomic-classification/test/**/FLAIR",
        "../data/rsna-miccai-brain-tumor-radiogenomic-classification/test/**/FLAIR",
    ]
    ckpt_candidates = [
        "/kaggle/input/brainclassificationeffnet/epoch25-step2677.ckpt",
        "../input/brainclassificationeffnet/epoch25-step2677.ckpt",
    ]

    list_of_dicom_folder_paths = _resolve_first_existing(flair_patterns)
    if not list_of_dicom_folder_paths:
        raise FileNotFoundError(
            f"Could not find test FLAIR folders with patterns: {flair_patterns}"
        )

    list_of_dicom_folder_paths = sorted(list_of_dicom_folder_paths)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    dataset = BrainDicomEvalDataset(list_of_paths=list_of_dicom_folder_paths)

    model = EfficientNet3D.from_name(
        "efficientnet-b5", override_params={"num_classes": 2}, in_channels=1
    )

    all_ckpt_hits = []
    for c in ckpt_candidates:
        if os.path.exists(c) and os.path.isfile(c):
            all_ckpt_hits.append(c)

    fallback = _find_checkpoint_fallback()
    if fallback is not None and os.path.exists(fallback):
        all_ckpt_hits.append(fallback)

    seen = set()
    all_ckpt_hits = [p for p in all_ckpt_hits if not (p in seen or seen.add(p))]

    model_path = None
    if len(all_ckpt_hits) == 1:
        model_path = all_ckpt_hits[0]
    elif len(all_ckpt_hits) > 1:
        model_path = _select_best_checkpoint_by_coverage(
            model=model, paths=all_ckpt_hits
        )

    loaded = False
    if model_path is not None and os.path.exists(model_path):
        try:
            obj = torch.load(model_path, map_location="cpu")
            sd0 = rename_keys(_extract_state_dict(obj))
            ratio0 = _state_dict_match_ratio(model=model, sd=sd0)
        except Exception:
            ratio0 = 0.0

        if ratio0 >= 0.80:
            loaded = _load_checkpoint_into_model(model=model, model_path=model_path)
        else:
            loaded = False

    if not loaded:
        print(
            "WARNING: No sufficiently-matching checkpoint loaded; predictions may be near-random (~0.5 AUC)."
        )
    else:
        print(f"Loaded checkpoint: {model_path}")

    model.to(device=device)
    model.eval()

    model_evaluator = ModelEvaluator(model=model, dataset=dataset, device=device)
    model_evaluator.eval(save_to_csv=True, csv_filepath="submission.csv")
