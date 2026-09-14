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
Create a classifier to predict the severity of diabetic retinopathy.

## Metric
Quadratic weighted kappa, which measures the agreement between two ratings. This metric typically varies from 0 (random agreement between raters) to 1 (complete agreement between raters). In the event that there is less agreement between the raters than expected by chance, this metric may go below 0. The quadratic weighted kappa is calculated between the scores assigned by the human rater and the predicted scores.

Images have five possible ratings, 0,1,2,3,4.  Each image is characterized by a tuple *(e*,*e)*, which corresponds to its scores by *Rater A* (human) and *Rater B* (predicted).  The quadratic weighted kappa is calculated as follows. First, an N x N histogram matrix *O* is constructed, such that *O* corresponds to the number of images that received a rating *i* by *A* and a rating *j* by *B*. An *N-by-N* matrix of weights, *w*, is calculated based on the difference between raters' scores:

An *N-by-N* histogram matrix of expected ratings, *E*, is calculated, assuming that there is no correlation between rating scores.  This is calculated as the outer product between each rater's histogram vector of ratings, normalized such that *E* and *O* have the same sum.

## Submission Format
```
id_code,diagnosis
0005cfc8afb6,0
003f0afdcd15,0
etc.
```

## Dataset
You are provided with a large set of retina images taken using [fundus photography](https://en.wikipedia.org/wiki/Fundus_photography) under a variety of imaging conditions.

Labels are on a scale of 0 to 4:

> 0 - No DR
> 1 - Mild
> 2 - Moderate
> 3 - Severe
> 4 - Proliferative DR

Images may contain artifacts, be out of focus, underexposed, or overexposed. The images were gathered from multiple clinics using a variety of cameras over an extended period of time, which will introduce further variation.

- **train.csv** - the training labels
- **test.csv** - the test set (you must predict the `diagnosis` value for these variables)
- **sample_submission.csv** - a sample submission file in the correct format
- **train.zip** - the training set images
- **test.zip** - the public test set images

# 2. Python version

3.7

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
            description.md (118 lines)
            sample_submission.csv (368 lines)
            sample_submission.csv.zip (3.2 kB)
            test.csv (368 lines)
            test.csv.zip (2.9 kB)
            test.zip (160 Bytes)
            test_images.zip (902.9 MB)
            train.csv (3296 lines)
            train.csv.zip (27.5 kB)
            train.zip (162 Bytes)
            train_images.zip (7.7 GB)
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
            test_images/
                218c822a3dd9.png (5.7 MB)
                0e82bcacc475.png (5.2 MB)
                ... and 365 other files
                test_images/
            train_images/
                184a185e7447.png (337.5 kB)
                c4aef0d88d1b.png (876.6 kB)
                ... and 3293 other files
                train_images/
        input/
            description.md (118 lines)
            sample_submission.csv (368 lines)
            sample_submission.csv.zip (3.2 kB)
            test.csv (368 lines)
            test.csv.zip (2.9 kB)
            test.zip (160 Bytes)
            test_images.zip (902.9 MB)
            train.csv (3296 lines)
            train.csv.zip (27.5 kB)
            train.zip (162 Bytes)
            train_images.zip (7.7 GB)
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
            test_images/
                218c822a3dd9.png (5.7 MB)
                0e82bcacc475.png (5.2 MB)
                ... and 365 other files
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
            train_images/
                184a185e7447.png (337.5 kB)
                c4aef0d88d1b.png (876.6 kB)
                ... and 3293 other files
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
        working/
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
```

-> data/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/aptos2019-blindness-detection/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/aptos2019-blindness-detection/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> input/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> (stopped after 10 files for performance)

# 5. Target score

0.5027069774014674

# 6. Current score

0.0514

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.09896) has done: 'I fix the immediate runtime failure by correcting the dataset/weights paths to match the provided Kaggle filesystem and by making the model weight-loading robust when the external weights file is not present (so inference can still run end-to-end). To preserve the existing core model and preprocessing logic, I keep the same ResNet101 architecture and transforms, only adding a safe fallback to torchvision’s built-in ResNet101 ImageNet weights when the custom checkpoint can’t be found. I also ensure `Transforms` is always defined before it is used and that the submission is written as `submission.csv` with the required columns and row alignment. These changes are necessary to produce a valid `.csv` submission and should yield a reasonable kappa score compared to random output, moving toward the target.'
- What this solution (achieved 0.22398) has done: 'Your current score is far below the target, so we should cautiously improve it without changing the core model or training (there is no training here) logic. The biggest gain with minimal risk for QWK on this competition is to calibrate the final 0–4 predictions rather than taking a raw argmax, because QWK is sensitive to class-threshold placement and the test class distribution differs from ImageNet. I keep the exact same ResNet101 and transforms, but add a tiny “post-processing” step: compute class probabilities, then choose a single temperature (on a held-out split of train.csv) that slightly sharpens/softens the distribution before argmax, and (optionally) apply a very small global class-prior adjustment estimated from train labels. This stays within the same evaluation semantics (still outputs integer classes 0–4) and typically moves a weak baseline toward a mid-range kappa without heavy changes.'
- What this solution (achieved 0.20825) has done: 'Your current score (0.22398) is well below the target (0.5027), so we should improve QWK with minimal, low-risk changes that don’t alter the model/training core. The biggest likely issue is that the custom checkpoint isn’t being found, so you’re effectively using ImageNet weights with a randomly-initialized 5-class head, which severely limits performance; we robustly search the provided filesystem for the checkpoint and load it when available. Second, QWK usually benefits from mapping logits to an ordinal prediction via thresholds on an expected severity score (rather than argmax); we fit 4 thresholds on a held-out split to maximize QWK and apply them to test predictions. These are pure inference/post-processing changes, keep the same model forward pass, and should move the score substantially toward the target while staying stable and within runtime.'
- What this solution (achieved 0.17497) has done: 'Your current score (0.20825) is far below the target (0.5027), so the smallest likely win is to fix an inference/training mismatch: the model was built with `sample_duration=16` but you always pass a single image (T=1), so almost all channels are effectively missing. I keep the exact same model architecture and transforms, but at inference time I repeat the single image 16 times so the input tensor matches what the network expects. I also make the checkpoint loading tolerant to `DataParallel` key prefixes so custom weights (if present) actually load instead of silently failing due to `module.` mismatches. These two changes preserve the core logic (same network, same preprocessing, same post-processing) while addressing the most probable cause of the low kappa.'
- What this solution (achieved 0.17497) has done: 'The timeout is dominated by repeatedly opening/decoding PNGs and running transforms for 5-fold OOF inference (3295 images) plus test inference, all in small Python batches. To keep the same model and inference semantics, the main speedups come from (1) using a multi-worker `DataLoader` to parallelize image decode/resize/crop on CPU with pinned-memory transfers, (2) avoiding per-image tensor concatenations by letting the loader batch tensors directly, and (3) enabling inference-time CUDA optimizations (`cudnn.benchmark`, `inference_mode`) while keeping determinism settings stable. The calibration and threshold search logic is unchanged; only the data feeding and inference plumbing is optimized and made more parallel. Paths and output format remain identical.'
- What this solution (achieved 0.17497) has done: 'Your score is far below the target (0.17497 vs 0.5027), so we should improve QWK with the smallest, safest change that doesn’t touch the model architecture or training loop. The biggest likely issue is that your OOF threshold fitting is accidentally using *different* softmax behavior than test-time because `expected_severity_from_logits()` re-softmaxes already-calibrated logits without applying the same temperature/prior logic consistently in torch; we make OOF and test severity computation use the same calibrated logits path. Second, we reduce OOF noise by computing OOF logits in a single pass over the full train set (still using the same model and transforms) instead of 5 repeated passes, which preserves evaluation semantics but removes fold-induced randomness and speeds up, allowing more stable threshold selection. Finally, we harden DataLoader settings to avoid `persistent_workers` issues on some Kaggle CPU setups while keeping parallelism, so the run reliably finishes and writes `submission.csv`.'
- What this solution (achieved 0.0514) has done: 'Your current score (0.17497) is far below the target (0.5027), so we should improve QWK with the smallest changes that keep the same model and inference pipeline. The biggest likely issue is a train–test domain mismatch: you’re using training-set mean/std (not ImageNet) while also sometimes loading ImageNet weights, which can heavily hurt predictions; we switch normalization to the standard ImageNet mean/std (a minimal transform change) to better match either torchvision weights or common pretrained DR checkpoints. Next, your “prior adjustment” is currently using the train label prior, which can over-bias predictions and reduce kappa when the model is weak; we keep the same calibration logic but shrink the alpha search range and prefer smaller alphas via a tiny tie-break to reduce overcorrection. Finally, we make threshold fitting a bit more robust by fitting thresholds on a held-out stratified split (still same thresholding approach) to reduce overfitting to train, which often improves public LB QWK.'

# 9. Code solution

## === cell 0
import os
import time
import math
import random
import numbers
from functools import partial

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import cohen_kappa_score

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class GroupRandomCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img_group):
        w, h = img_group[0].size
        th, tw = self.size
        out_images = list()

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)

        for img in img_group:
            assert img.size[0] == w and img.size[1] == h
            if w == tw and h == th:
                out_images.append(img)
            else:
                out_images.append(img.crop((x1, y1, x1 + tw, y1 + th)))
        return out_images


class GroupCenterCrop(object):
    def __init__(self, size):
        self.worker = transforms.CenterCrop(size)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]


class GroupRandomHorizontalFlip(object):
    def __init__(self, is_flow=False):
        self.is_flow = is_flow

    def __call__(self, img_group, is_flow=False):
        v = random.random()
        if v < 0.5:
            ret = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in img_group]
            if self.is_flow:
                for i in range(0, len(ret), 2):
                    ret[i] = ImageOps.invert(ret[i])
            return ret
        else:
            return img_group


class GroupRandomVerticalFlip(object):
    def __init__(self, is_flow=False):
        self.is_flow = is_flow

    def __call__(self, img_group):
        v = random.random()
        if v < 0.5:
            ret = [img.transpose(Image.FLIP_TOP_BOTTOM) for img in img_group]
            if self.is_flow:
                for i in range(1, len(ret), 2):
                    ret[i] = ImageOps.invert(ret[i])
            return ret
        else:
            return img_group


class GroupNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        rep_mean = self.mean * (tensor.size()[0] // len(self.mean))
        rep_std = self.std * (tensor.size()[0] // len(self.std))
        for t, m, s in zip(tensor, rep_mean, rep_std):
            t.sub_(m).div_(s)
        return tensor


class GroupScale(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.worker = transforms.Resize(size, interpolation)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]


class GroupOverSample(object):
    def __init__(self, crop_size, scale_size=None):
        self.crop_size = (
            crop_size if not isinstance(crop_size, int) else (crop_size, crop_size)
        )
        if scale_size is not None:
            self.scale_worker = GroupScale(scale_size)
        else:
            self.scale_worker = None

    def __call__(self, img_group):
        if self.scale_worker is not None:
            img_group = self.scale_worker(img_group)

        image_w, image_h = img_group[0].size
        crop_w, crop_h = self.crop_size

        offsets = GroupMultiScaleCrop.fill_fix_offset(
            False, image_w, image_h, crop_w, crop_h
        )
        oversample_group = list()
        for o_w, o_h in offsets:
            normal_group = list()
            flip_group = list()
            for i, img in enumerate(img_group):
                crop = img.crop((o_w, o_h, o_w + crop_w, o_h + crop_h))
                normal_group.append(crop)
                flip_crop = crop.copy().transpose(Image.FLIP_LEFT_RIGHT)

                if img.mode == "L" and i % 2 == 0:
                    flip_group.append(ImageOps.invert(flip_crop))
                else:
                    flip_group.append(flip_crop)

            oversample_group.extend(normal_group)
            oversample_group.extend(flip_group)
        return oversample_group


class GroupMultiScaleCrop(object):
    def __init__(
        self, input_size, scales=None, max_distort=1, fix_crop=True, more_fix_crop=True
    ):
        self.scales = scales if scales is not None else [1, 0.875, 0.75, 0.66]
        self.max_distort = max_distort
        self.fix_crop = fix_crop
        self.more_fix_crop = more_fix_crop
        self.input_size = (
            input_size if not isinstance(input_size, int) else [input_size, input_size]
        )
        self.interpolation = Image.BILINEAR

    def __call__(self, img_group):
        im_size = img_group[0].size
        crop_w, crop_h, offset_w, offset_h = self._sample_crop_size(im_size)
        crop_img_group = [
            img.crop((offset_w, offset_h, offset_w + crop_w, offset_h + crop_h))
            for img in img_group
        ]
        ret_img_group = [
            img.resize((self.input_size[0], self.input_size[1]), self.interpolation)
            for img in crop_img_group
        ]
        return ret_img_group

    def _sample_crop_size(self, im_size):
        image_w, image_h = im_size[0], im_size[1]
        base_size = min(image_w, image_h)
        crop_sizes = [int(base_size * x) for x in self.scales]
        crop_h = [
            self.input_size[1] if abs(x - self.input_size[1]) < 3 else x
            for x in crop_sizes
        ]
        crop_w = [
            self.input_size[0] if abs(x - self.input_size[0]) < 3 else x
            for x in crop_sizes
        ]

        pairs = []
        for i, h in enumerate(crop_h):
            for j, w in enumerate(crop_w):
                if abs(i - j) <= self.max_distort:
                    pairs.append((w, h))

        crop_pair = random.choice(pairs)
        if not self.fix_crop:
            w_offset = random.randint(0, image_w - crop_pair[0])
            h_offset = random.randint(0, image_h - crop_pair[1])
        else:
            w_offset, h_offset = self._sample_fix_offset(
                image_w, image_h, crop_pair[0], crop_pair[1]
            )

        return crop_pair[0], crop_pair[1], w_offset, h_offset

    def _sample_fix_offset(self, image_w, image_h, crop_w, crop_h):
        offsets = self.fill_fix_offset(
            self.more_fix_crop, image_w, image_h, crop_w, crop_h
        )
        return random.choice(offsets)

    @staticmethod
    def fill_fix_offset(more_fix_crop, image_w, image_h, crop_w, crop_h):
        w_step = (image_w - crop_w) / 4
        h_step = (image_h - crop_h) / 4

        ret = list()
        ret.append((0, 0))
        ret.append((4 * w_step, 0))
        ret.append((0, 4 * h_step))
        ret.append((4 * w_step, 4 * h_step))
        ret.append((2 * w_step, 2 * h_step))

        if more_fix_crop:
            ret.append((0, 2 * h_step))
            ret.append((4 * w_step, 2 * h_step))
            ret.append((2 * w_step, 4 * h_step))
            ret.append((2 * w_step, 0 * h_step))

            ret.append((1 * w_step, 1 * h_step))
            ret.append((3 * w_step, 1 * h_step))
            ret.append((1 * w_step, 3 * h_step))
            ret.append((3 * w_step, 3 * h_step))
        return ret


class GroupRandomResizedCrop(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img_group):
        for attempt in range(10):
            area = img_group[0].size[0] * img_group[0].size[1]
            target_area = random.uniform(0.08, 1.0) * area
            aspect_ratio = random.uniform(3.0 / 4, 4.0 / 3)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img_group[0].size[0] and h <= img_group[0].size[1]:
                x1 = random.randint(0, img_group[0].size[0] - w)
                y1 = random.randint(0, img_group[0].size[1] - h)
                found = True
                break
        else:
            found = False
            x1 = 0
            y1 = 0

        if found:
            out_group = list()
            for img in img_group:
                img = img.crop((x1, y1, x1 + w, y1 + h))
                assert img.size == (w, h)
                out_group.append(img.resize((self.size, self.size), self.interpolation))
            return out_group
        else:
            scale = GroupScale(self.size, interpolation=self.interpolation)
            crop = GroupRandomCrop(self.size)
            return crop(scale(img_group))


class GroupColorJitter(object):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.worker = transforms.ColorJitter(
            brightness=brightness, contrast=contrast, saturation=saturation, hue=hue
        )

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]


class GroupRandomRotate90(object):
    def __init__(self, is_flow=False):
        self.is_flow = is_flow

    def __call__(self, img_group):
        v = random.random()
        if v < 0.5:
            ret = [img.rotate(90) for img in img_group]
            if self.is_flow:
                ret2 = []
                for i in range(0, len(ret), 2):
                    ret2.append(ret[i + 1])
                    ret2.append(ret[i])
                ret = ret2
            return ret
        else:
            return img_group


class Stack(object):
    def __init__(self, is_flow=False):
        self.is_flow = is_flow

    def __call__(self, img_group):
        if self.is_flow is False:
            return np.stack(img_group, axis=0)
        else:
            cat_img = []
            for i in range(0, len(img_group), 2):
                cat_img.append(np.stack(img_group[i : i + 2], axis=2))
            return np.stack(cat_img, axis=0)


class ToTorchFormatTensor(object):
    def __init__(self, div=True):
        self.div = div

    def __call__(self, pic):
        img = torch.from_numpy(pic).permute(3, 0, 1, 2).contiguous()
        return img.float().div(255) if self.div else img.float()




## === cell 1
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool2d(x, kernel_size=1, stride=stride)
    zero_pads = torch.zeros(
        out.size(0),
        planes - out.size(1),
        out.size(2),
        out.size(3),
        device=out.device,
        dtype=out.dtype,
    )
    out = torch.cat([out, zero_pads], dim=1)
    return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
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

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
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
        sample_size,
        sample_duration,
        shortcut_type,
        num_classes,
        dropout,
    ):
        self.inplanes = 64
        super(ResNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], shortcut_type, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], shortcut_type, stride=2)

        last_size = int(math.ceil(sample_size / 32.0))
        self.avgpool = nn.AvgPool2d((last_size, last_size), stride=1)
        self.drop = nn.Dropout(p=dropout)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode="fan_out")
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == "A":
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride,
                )
            else:
                downsample = nn.Sequential(
                    nn.Conv2d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm2d(planes * block.expansion),
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.drop(x)
        x = self.fc(x)
        return x


def resnet34(**kwargs):
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet50(**kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101(**kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet152(**kwargs):
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model




## === cell 2
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_dict = {
    "34": [resnet34, "A"],
    "50": [resnet50, "B"],
    "101": [resnet101, "B"],
    "152": [resnet152, "B"],
}

Transforms = {
    "val": transforms.Compose(
        [
            GroupScale(128),
            GroupCenterCrop(112),
            Stack(is_flow=False),
            ToTorchFormatTensor(div=True),
            GroupNormalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
}


def _strip_module_prefix(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict
    keys = list(state_dict.keys())
    if len(keys) == 0:
        return state_dict
    if all(isinstance(k, str) and k.startswith("module.") for k in keys):
        return {k[len("module.") :]: v for k, v in state_dict.items()}
    return state_dict


def find_checkpoint():
    candidates = [
        "../input/weights/blind_resnet101_best_73.30567080549184.pth",
        "/kaggle/input/weights/blind_resnet101_best_73.30567080549184.pth",
        "/kaggle/data/weights/blind_resnet101_best_73.30567080549184.pth",
        "../input/blind_resnet101_best_73.30567080549184.pth",
        "/kaggle/input/blind_resnet101_best_73.30567080549184.pth",
        "/kaggle/data/blind_resnet101_best_73.30567080549184.pth",
    ]
    for p in candidates:
        if isinstance(p, str) and os.path.exists(p):
            return p
    roots = ["../input", "/kaggle/input", "/kaggle/data"]
    target_name = "blind_resnet101_best_73.30567080549184.pth"
    for r in roots:
        if os.path.isdir(r):
            for dirpath, _, filenames in os.walk(r):
                if target_name in filenames:
                    return os.path.join(dirpath, target_name)
    return None


weights = find_checkpoint()

model = model_dict["101"][0](
    sample_size=112,
    sample_duration=16,
    shortcut_type=model_dict["101"][1],
    num_classes=5,
    dropout=0,
)

if device.type == "cuda" and torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)

model = model.to(device)

loaded_custom = False
if isinstance(weights, str) and os.path.exists(weights):
    ckpt = torch.load(weights, map_location=device)
    state_dict = (
        ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    )
    state_dict = _strip_module_prefix(state_dict)

    model_state = model.state_dict()
    model_is_dp = any(k.startswith("module.") for k in model_state.keys())
    ckpt_is_dp = any(
        isinstance(k, str) and k.startswith("module.") for k in state_dict.keys()
    )
    if model_is_dp and (not ckpt_is_dp):
        state_dict = {("module." + k): v for k, v in state_dict.items()}
    if (not model_is_dp) and ckpt_is_dp:
        state_dict = _strip_module_prefix(state_dict)

    model.load_state_dict(state_dict, strict=True)
    loaded_custom = True
    print(f"Loaded custom weights from: {weights}")
else:
    tv = torchvision.models.resnet101(
        weights=torchvision.models.ResNet101_Weights.IMAGENET1K_V2
    )
    tv.fc = nn.Linear(tv.fc.in_features, 5)
    missing, unexpected = model.load_state_dict(tv.state_dict(), strict=False)
    print(
        "WARNING: custom weights not found. Using torchvision ImageNet weights as fallback "
        "(expected lower QWK than target)."
    )
    print(
        f"State dict load (fallback) missing={len(missing)} unexpected={len(unexpected)}"
    )

model.eval()
print(f"Model ready on device={device}. loaded_custom_weights={loaded_custom}")

SAMPLE_DURATION = 16



## === cell 3
base1 = "../input/aptos2019-blindness-detection"
base2 = "/kaggle/input/aptos2019-blindness-detection"
base3 = "/kaggle/data/aptos2019-blindness-detection"
base = base1 if os.path.exists(base1) else (base2 if os.path.exists(base2) else base3)

train_csv = os.path.join(base, "train.csv")
train_img_dir = os.path.join(base, "train_images")

if not os.path.exists(train_csv):
    raise FileNotFoundError(f"train.csv not found at {train_csv}")
if not os.path.isdir(train_img_dir):
    raise FileNotFoundError(f"train_images dir not found at {train_img_dir}")

train_df = pd.read_csv(train_csv)
y = train_df["diagnosis"].astype(int).values

transform_val = Transforms["val"]


class RetinaDurationDataset(torch.utils.data.Dataset):
    def __init__(self, id_codes, img_dir, transform_fn, duration=16):
        self.id_codes = list(id_codes)
        self.img_dir = img_dir
        self.transform_fn = transform_fn
        self.duration = int(duration)

    def __len__(self):
        return len(self.id_codes)

    def __getitem__(self, idx):
        id_code = self.id_codes[idx]
        img_path = os.path.join(self.img_dir, f"{id_code}.png")
        with Image.open(img_path) as im:
            img = im.convert("RGB")
        img_group = [img] * self.duration
        x = self.transform_fn(img_group)  # (T, C, H, W)
        x = x.view(-1, 3, 112, 112)  # (T, C, H, W)
        return x


def _num_workers_for_loader():
    cpu = os.cpu_count() or 4
    return max(2, min(8, cpu // 2))


def _dl_kwargs():
    nw = _num_workers_for_loader()
    kwargs = dict(
        num_workers=nw,
        pin_memory=(device.type == "cuda"),
    )
    if nw > 0:
        kwargs["persistent_workers"] = False
        kwargs["prefetch_factor"] = 2
    return kwargs


def infer_logits_for_ids(id_list, img_dir, batch_size=16, verbose_prefix=""):
    ds = RetinaDurationDataset(
        id_list, img_dir, transform_val, duration=SAMPLE_DURATION
    )
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        **_dl_kwargs(),
    )

    logits_all = []
    t0 = time.time()
    with torch.inference_mode():
        for bi, xb in enumerate(loader, start=1):
            xb = xb.view(-1, 3, 112, 112).to(device, non_blocking=True)
            out = model(xb)  # (B*T, 5)
            out = out.view(-1, SAMPLE_DURATION, 5).mean(dim=1)  # (B, 5)
            logits_all.append(out.cpu())
            if verbose_prefix and (bi % 20 == 0):
                seen = min(bi * batch_size, len(ds))
                print(
                    f"{verbose_prefix} {seen}/{len(ds)} elapsed={time.time()-t0:.1f}s"
                )
    return torch.cat(logits_all, dim=0).numpy()


all_train_ids = train_df["id_code"].tolist()
oof_logits = infer_logits_for_ids(
    all_train_ids,
    train_img_dir,
    batch_size=16,
    verbose_prefix="OOF(all-train) infer",
).astype(np.float32)

prior_smooth = 0.05
train_counts = np.bincount(y, minlength=5).astype(np.float64)
train_prior = (train_counts + prior_smooth) / (train_counts.sum() + 5.0 * prior_smooth)
log_prior = np.log(np.clip(train_prior, 1e-12, 1.0))

temps = np.linspace(0.7, 2.0, 14)
best_t = 1.0
best_a = 0.0
best_kappa = -1.0
alphas = [0.0, 0.05, 0.10, 0.15, 0.20]

for t in temps:
    scaled = oof_logits / float(t)
    for a in alphas:
        adj = scaled + float(a) * log_prior[None, :]
        pred = adj.argmax(axis=1).astype(int)
        k = cohen_kappa_score(y, pred, weights="quadratic")
        if (k > best_kappa) or (abs(k - best_kappa) < 1e-12 and a < best_a):
            best_kappa = k
            best_t = float(t)
            best_a = float(a)

print(
    f"Logit calibration chosen (train-fit): temperature={best_t:.3f} prior_alpha={best_a:.2f} QWK(argmax)={best_kappa:.4f}"
)


def expected_severity_from_calibrated_logits_np(cal_logits_np):
    lt = torch.from_numpy(cal_logits_np.astype(np.float32))
    p = torch.softmax(lt, dim=1).numpy()
    classes = np.arange(5, dtype=np.float32)[None, :]
    return (p * classes).sum(axis=1)


def apply_thresholds(x, thr):
    return np.digitize(x, bins=np.array(thr, dtype=np.float32), right=False).astype(int)


def fit_thresholds_bruteforce(x, y_true):
    best_thr = [0.5, 1.5, 2.5, 3.5]
    best_k = -1.0

    g0 = np.arange(0.2, 1.6, 0.2)
    g1 = np.arange(0.8, 2.6, 0.2)
    g2 = np.arange(1.6, 3.4, 0.2)
    g3 = np.arange(2.4, 4.2, 0.2)
    for t0 in g0:
        for t1 in g1:
            if t1 <= t0:
                continue
            for t2 in g2:
                if t2 <= t1:
                    continue
                for t3 in g3:
                    if t3 <= t2:
                        continue
                    pred = apply_thresholds(x, [t0, t1, t2, t3])
                    k = cohen_kappa_score(y_true, pred, weights="quadratic")
                    if k > best_k:
                        best_k = k
                        best_thr = [float(t0), float(t1), float(t2), float(t3)]

    def fine_range(center, step=0.05, width=0.25, lo=0.0, hi=4.5):
        a = max(lo, center - width)
        b = min(hi, center + width + 1e-9)
        return np.arange(a, b, step)

    f0 = fine_range(best_thr[0], lo=0.0, hi=best_thr[1] - 0.05)
    f1 = fine_range(best_thr[1], lo=best_thr[0] + 0.05, hi=best_thr[2] - 0.05)
    f2 = fine_range(best_thr[2], lo=best_thr[1] + 0.05, hi=best_thr[3] - 0.05)
    f3 = fine_range(best_thr[3], lo=best_thr[2] + 0.05, hi=4.5)

    for t0 in f0:
        for t1 in f1:
            if t1 <= t0:
                continue
            for t2 in f2:
                if t2 <= t1:
                    continue
                for t3 in f3:
                    if t3 <= t2:
                        continue
                    pred = apply_thresholds(x, [t0, t1, t2, t3])
                    k = cohen_kappa_score(y_true, pred, weights="quadratic")
                    if k > best_k:
                        best_k = k
                        best_thr = [float(t0), float(t1), float(t2), float(t3)]
    return best_thr, best_k


oof_adj = (oof_logits / best_t) + best_a * log_prior[None, :]
oof_sev = expected_severity_from_calibrated_logits_np(oof_adj.astype(np.float32))

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
tr_idx, va_idx = next(sss.split(np.zeros(len(y)), y))
best_thr, best_thr_kappa = fit_thresholds_bruteforce(oof_sev[va_idx], y[va_idx])

print(
    f"Thresholds chosen (val-fit): {best_thr} QWK(val, thresholded)={best_thr_kappa:.4f}"
)



## === cell 4
test_csv = os.path.join(base, "test.csv")
test_img_dir = os.path.join(base, "test_images")
sample_sub_path = os.path.join(base, "sample_submission.csv")

if not os.path.exists(test_csv):
    raise FileNotFoundError(f"test.csv not found at {test_csv}")
if not os.path.isdir(test_img_dir):
    raise FileNotFoundError(f"test_images dir not found at {test_img_dir}")
if not os.path.exists(sample_sub_path):
    raise FileNotFoundError(f"sample_submission.csv not found at {sample_sub_path}")

imgs = pd.read_csv(test_csv, usecols=[0])
ids = imgs["id_code"].tolist()

t0 = time.time()


def logits_to_expected_severity_torch(logits_t):
    p = torch.softmax(logits_t, dim=1)
    classes = torch.arange(5, device=logits_t.device, dtype=p.dtype).view(1, -1)
    return (p * classes).sum(dim=1)


test_ds = RetinaDurationDataset(
    ids, test_img_dir, transform_val, duration=SAMPLE_DURATION
)
batch_size = 16
test_loader = torch.utils.data.DataLoader(
    test_ds,
    batch_size=batch_size,
    shuffle=False,
    **_dl_kwargs(),
)

log_prior_t = torch.from_numpy(log_prior.astype(np.float32)).to(device).view(1, -1)

preds = []
with torch.inference_mode():
    for bi, xb in enumerate(test_loader, start=1):
        xb = xb.view(-1, 3, 112, 112).to(device, non_blocking=True)
        output = model(xb)  # (B*T, 5)
        output = output.view(-1, SAMPLE_DURATION, 5).mean(dim=1)  # (B, 5)

        output = output / best_t
        output = output + best_a * log_prior_t

        sev = logits_to_expected_severity_torch(output).cpu().numpy()
        batch_pred = (
            np.digitize(sev, bins=np.array(best_thr, dtype=np.float32), right=False)
            .astype(int)
            .tolist()
        )
        preds.extend(batch_pred)

        if (bi % 5) == 0:
            seen = min(bi * batch_size, len(test_ds))
            print(
                f"Inference {seen}/{len(test_ds)} done. Elapsed={time.time()-t0:.1f}s"
            )

sub = pd.read_csv(sample_sub_path)
if len(sub) != len(preds):
    raise RuntimeError(
        f"Prediction length mismatch: len(sub)={len(sub)} vs len(preds)={len(preds)}"
    )

sub["diagnosis"] = preds
sub.to_csv("submission.csv", index=False)

print("Wrote submission.csv")
print(sub.head())
