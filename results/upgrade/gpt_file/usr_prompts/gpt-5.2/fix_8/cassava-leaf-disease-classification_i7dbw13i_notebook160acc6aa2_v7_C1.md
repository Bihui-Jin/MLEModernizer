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

3.11

# 3. Installed packages

albumentations==2.0.8
geopandas==0.14.4
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
protobuf==6.33.0
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sentence-transformers==4.1.0
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1
timm==1.0.19
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
transformers==4.53.3

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

0.7997884557268057

# 6. Current score

0.61099

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.05531) has done: 'I fix the runtime error by changing the missing MobileNet weights path to a safe, existing location (falling back across common Kaggle input/working paths) and, if no weights are found, fall back to deterministic zero-probabilities so the pipeline still produces a valid CSV. I also fix the “Invalid submission length” issue by building the submission strictly from `sample_submission.csv` (the authoritative test image list/order) instead of `os.listdir()`, which can include nested folders or mismatched counts. Finally, I guard the ensembling step so it only uses model columns that were actually produced, preventing the `KeyError: 'mobilenet'` and ensuring end-to-end execution with a valid `submission.csv`.'
- What this solution (achieved 0.61099) has done: 'Your very low accuracy (0.05531) strongly suggests the MobileNet checkpoint is not being found/loaded, so you’re effectively submitting constant/garbage predictions. I make the smallest change that increases score toward the target by (1) resolving the checkpoint more robustly (including the common Kaggle “dataset is mounted at /kaggle/input/<dataset>/<file>” case) and (2) if the checkpoint is still missing, falling back to a sane prior (predict the most frequent training class) instead of all-zeros. This preserves your core model/inference logic and submission semantics, but avoids the catastrophic failure mode that tanks accuracy. The rest of the pipeline (using sample_submission ordering and producing a valid submission.csv) is kept intact.'
- What this solution (achieved 0.61099) has done: 'Your current score (0.61099) is well below the target (0.79979), and the biggest likely gap driver is a preprocessing mismatch: the MobileNetV3 checkpoint was almost certainly trained with ImageNet normalization (mean/std), while your inference uses (0.5, 0.5, 0.5). I make the smallest safe change to align inference normalization to ImageNet mean/std (keeping the same model, checkpoint, and inference loop), which typically yields a large accuracy jump for pretrained CNN checkpoints. I also ensure the test image loading order remains exactly the sample submission order (already correct) and keep everything else unchanged to preserve core logic and semantics.'
- What this solution (achieved 0.61099) has done: 'Your current score (0.61099) is below the target (0.79979), so we should cautiously improve accuracy without changing the model or inference logic. The biggest minimal win here is fixing an input-size mismatch: your MobileNetV3 was trained with a specific resolution (very likely 512), but your inference pipeline never resizes, so you’re feeding variable native sizes which can significantly hurt performance. I add a deterministic resize to `IMAGE_SIZE` inside the existing `transforms_val` (keeping the same normalization and model), and keep submission ordering strictly from `sample_submission.csv`. Everything else (checkpoint loading, architecture, softmax, ensembling logic, CSV writing) stays the same.'
- What this solution (achieved 0.61099) has done: 'Your current score (0.61099) is below the target (0.79979), so we should make a small, low-risk improvement that preserves your exact model/inference core. The biggest likely remaining issue is a mismatch between the model’s expected input preprocessing at training time and your current inference transforms: many Cassava MobileNet checkpoints were trained with center-crop (or similar) after resizing, not a direct warp-resize to (512,512). I keep your same model, checkpoint loading, softmax, and submission logic, but change the validation transform to `Resize -> CenterCrop -> ToTensor -> Normalize` (still deterministic) to better match typical training pipelines and improve accuracy. I also add a safe RGB conversion inside the transform path (without changing dataset logic) by ensuring PIL is consistently used as RGB (already) and keep everything else identical.'
- What this solution (achieved 0.61099) has done: 'Your score gap to the target is large (0.61099 → 0.79979), so the most likely minimal “real” improvement (without changing your model or inference loop) is fixing a train/infer mismatch in the *classifier head*: your checkpoint almost certainly was trained for 5 classes, but the current code instantiates `mobilenetv3_large()` with the default `num_classes=1000` and then slices `output[:, :5]`, which makes the learned 5-way head unused/misaligned. I change only the model instantiation to `num_classes=5` (same architecture family, same checkpoint, same prediction code), and make the softmax apply to the full logits (no slicing) so it matches the 5-class head. Everything else (checkpoint resolution, transforms, dataloader, merge, submission ordering/format) stays the same to preserve your core logic and semantics and keep runtime under the limit.'

# 9. Code solution

## === cell 0
import os
import gc
import math
import random

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import torchvision.transforms as transforms



## === cell 1
path = "/kaggle/input/cassava-leaf-disease-classification/"
sample_path = os.path.join(path, "sample_submission.csv")
submission_df = pd.read_csv(sample_path)

IMAGE_SIZE = (512, 512)

submission_df["label"] = 0
test_image_ids = submission_df["image_id"].tolist()



## === cell 2
onlykeras = False

used_models_pytorch = {"mobilenet": "../input/model-mobilenet/mn3_bt20_ep5_lr1.pth"}

stacked_mean = False



## === cell 3
pass



## === cell 4
pass



## === cell 5
pass



## === cell 6
pass



## === cell 7
if "mobilenet" in used_models_pytorch:
    __all__ = ["mobilenetv3_large", "mobilenetv3_small"]

    def _make_divisible(v, divisor, min_value=None):
        if min_value is None:
            min_value = divisor
        new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
        if new_v < 0.9 * v:
            new_v += divisor
        return new_v

    class h_sigmoid(nn.Module):
        def __init__(self, inplace=True):
            super(h_sigmoid, self).__init__()
            self.relu = nn.ReLU6(inplace=inplace)

        def forward(self, x):
            return self.relu(x + 3) / 6

    class h_swish(nn.Module):
        def __init__(self, inplace=True):
            super(h_swish, self).__init__()
            self.sigmoid = h_sigmoid(inplace=inplace)

        def forward(self, x):
            return x * self.sigmoid(x)

    class SELayer(nn.Module):
        def __init__(self, channel, reduction=4):
            super(SELayer, self).__init__()
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Sequential(
                nn.Linear(channel, _make_divisible(channel // reduction, 8)),
                nn.ReLU(inplace=True),
                nn.Linear(_make_divisible(channel // reduction, 8), channel),
                h_sigmoid(),
            )

        def forward(self, x):
            b, c, _, _ = x.size()
            y = self.avg_pool(x).view(b, c)
            y = self.fc(y).view(b, c, 1, 1)
            return x * y

    def conv_3x3_bn(inp, oup, stride):
        return nn.Sequential(
            nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
            nn.BatchNorm2d(oup),
            h_swish(),
        )

    def conv_1x1_bn(inp, oup):
        return nn.Sequential(
            nn.Conv2d(inp, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup), h_swish()
        )

    class InvertedResidual(nn.Module):
        def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs):
            super(InvertedResidual, self).__init__()
            assert stride in [1, 2]

            self.identity = stride == 1 and inp == oup

            if inp == hidden_dim:
                self.conv = nn.Sequential(
                    nn.Conv2d(
                        hidden_dim,
                        hidden_dim,
                        kernel_size,
                        stride,
                        (kernel_size - 1) // 2,
                        groups=hidden_dim,
                        bias=False,
                    ),
                    nn.BatchNorm2d(hidden_dim),
                    h_swish() if use_hs else nn.ReLU(inplace=True),
                    SELayer(hidden_dim) if use_se else nn.Identity(),
                    nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(oup),
                )
            else:
                self.conv = nn.Sequential(
                    nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    h_swish() if use_hs else nn.ReLU(inplace=True),
                    nn.Conv2d(
                        hidden_dim,
                        hidden_dim,
                        kernel_size,
                        stride,
                        (kernel_size - 1) // 2,
                        groups=hidden_dim,
                        bias=False,
                    ),
                    nn.BatchNorm2d(hidden_dim),
                    SELayer(hidden_dim) if use_se else nn.Identity(),
                    h_swish() if use_hs else nn.ReLU(inplace=True),
                    nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(oup),
                )

        def forward(self, x):
            if self.identity:
                return x + self.conv(x)
            else:
                return self.conv(x)

    class MobileNetV3(nn.Module):
        def __init__(self, cfgs, mode, num_classes=1000, width_mult=1.0):
            super(MobileNetV3, self).__init__()
            self.cfgs = cfgs
            assert mode in ["large", "small"]

            input_channel = _make_divisible(16 * width_mult, 8)
            layers = [conv_3x3_bn(3, input_channel, 2)]
            block = InvertedResidual
            for k, t, c, use_se, use_hs, s in self.cfgs:
                output_channel = _make_divisible(c * width_mult, 8)
                exp_size = _make_divisible(input_channel * t, 8)
                layers.append(
                    block(input_channel, exp_size, output_channel, k, s, use_se, use_hs)
                )
                input_channel = output_channel
            self.features = nn.Sequential(*layers)
            self.conv = conv_1x1_bn(input_channel, exp_size)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            output_channel = {"large": 1280, "small": 1024}
            output_channel = (
                _make_divisible(output_channel[mode] * width_mult, 8)
                if width_mult > 1.0
                else output_channel[mode]
            )
            self.classifier = nn.Sequential(
                nn.Linear(exp_size, output_channel),
                h_swish(),
                nn.Dropout(0.2),
                nn.Linear(output_channel, num_classes),
            )

            self._initialize_weights()

        def forward(self, x):
            x = self.features(x)
            x = self.conv(x)
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x

        def _initialize_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2.0 / n))
                    if m.bias is not None:
                        m.bias.data.zero_()
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.Linear):
                    m.weight.data.normal_(0, 0.01)
                    m.bias.data.zero_()

    def mobilenetv3_large(**kwargs):
        cfgs = [
            [3, 1, 16, 0, 0, 1],
            [3, 4, 24, 0, 0, 2],
            [3, 3, 24, 0, 0, 1],
            [5, 3, 40, 1, 0, 2],
            [5, 3, 40, 1, 0, 1],
            [5, 3, 40, 1, 0, 1],
            [3, 6, 80, 0, 1, 2],
            [3, 2.5, 80, 0, 1, 1],
            [3, 2.3, 80, 0, 1, 1],
            [3, 2.3, 80, 0, 1, 1],
            [3, 6, 112, 1, 1, 1],
            [3, 6, 112, 1, 1, 1],
            [5, 6, 160, 1, 1, 2],
            [5, 6, 160, 1, 1, 1],
            [5, 6, 160, 1, 1, 1],
        ]
        return MobileNetV3(cfgs, mode="large", **kwargs)

    def mobilenetv3_small(**kwargs):
        cfgs = [
            [3, 1, 16, 1, 0, 2],
            [3, 4.5, 24, 0, 0, 2],
            [3, 3.67, 24, 0, 0, 1],
            [5, 4, 40, 1, 1, 2],
            [5, 6, 40, 1, 1, 1],
            [5, 6, 40, 1, 1, 1],
            [5, 3, 48, 1, 1, 1],
            [5, 3, 48, 1, 1, 1],
            [5, 6, 96, 1, 1, 2],
            [5, 6, 96, 1, 1, 1],
            [5, 6, 96, 1, 1, 1],
        ]
        return MobileNetV3(cfgs, mode="small", **kwargs)

    class LeafDataset(torch.utils.data.Dataset):
        def __init__(self, df, data_path, mode="train", transforms=None):
            super().__init__()
            self.df_data = df.values
            self.data_path = data_path
            self.transforms = transforms
            self.mode = mode
            self.data_dir = "train_images" if mode == "train" else "test_images"

        def __len__(self):
            return len(self.df_data)

        def __getitem__(self, index):
            img_name = self.df_data[index][0]
            img_path = os.path.join(self.data_path, self.data_dir, img_name)
            img = Image.open(img_path).convert("RGB")
            if self.transforms is not None:
                img = self.transforms(img)
            return img

    def predict(model, test_dataset):
        preds = []
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=20,
            shuffle=False,
            num_workers=2,
            pin_memory=torch.cuda.is_available(),
        )
        model.eval()
        for test_images in tqdm(test_dataloader, desc="Predict mobilenet"):
            test_images = test_images.to(device)
            with torch.no_grad():
                output = model(test_images)
            preds.extend(output.softmax(1).cpu().numpy())
        return preds

    def _resolve_checkpoint_path(p):
        candidates = []
        if p:
            candidates.append(p)

            if p.startswith("../input/"):
                candidates.append(os.path.join("/kaggle", p.lstrip("../")))

            if p.startswith("/kaggle/input/"):
                candidates.append(p)

            candidates.extend(
                [
                    os.path.join(
                        "/kaggle/input",
                        os.path.basename(os.path.dirname(p)),
                        os.path.basename(p),
                    ),
                    os.path.join("/kaggle/input/model-mobilenet", os.path.basename(p)),
                    os.path.join("/kaggle/working", os.path.basename(p)),
                ]
            )

        for c in candidates:
            if c and os.path.exists(c):
                return c
        try:
            base = os.path.basename(p) if p else None
            if base:
                for root, _, files in os.walk("/kaggle/input"):
                    if base in files:
                        return os.path.join(root, base)
        except Exception:
            pass
        return None

    transforms_val = transforms.Compose(
        [
            transforms.Resize(
                IMAGE_SIZE[0] + 32, interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    predictions_mobilenet = pd.DataFrame({"image_id": submission_df["image_id"].values})

    model = mobilenetv3_large(num_classes=5)
    model.to(device)

    ckpt_path = _resolve_checkpoint_path(used_models_pytorch["mobilenet"])

    if ckpt_path is None:
        train_csv_path = os.path.join(path, "train.csv")
        if os.path.exists(train_csv_path):
            train_df = pd.read_csv(train_csv_path)
            counts = train_df["label"].value_counts().sort_index()
            prior = np.zeros(5, dtype=np.float32)
            for k, v in counts.items():
                if 0 <= int(k) < 5:
                    prior[int(k)] = float(v)
            s = prior.sum()
            if s > 0:
                prior = prior / s
            else:
                prior[:] = 1.0 / 5.0
        else:
            prior = np.ones(5, dtype=np.float32) / 5.0

        predictions_raw_mobilenet = [
            prior.copy() for _ in range(len(predictions_mobilenet))
        ]
    else:
        state = torch.load(ckpt_path, map_location=device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if isinstance(state, dict):
            new_state = {}
            for k, v in state.items():
                nk = k[7:] if k.startswith("module.") else k
                new_state[nk] = v
            state = new_state
        model.load_state_dict(state, strict=True)

        test_dataset = LeafDataset(
            df=predictions_mobilenet,
            data_path=path,
            mode="test",
            transforms=transforms_val,
        )
        predictions_raw_mobilenet = predict(model, test_dataset)

    predictions_mobilenet["mobilenet"] = [
        np.squeeze(p) for p in predictions_raw_mobilenet
    ]

    torch.cuda.empty_cache()
    try:
        del model
    except Exception:
        pass
    gc.collect()



## === cell 8
pass



## === cell 9
submission_df["label"] = 0

if "mobilenet" in used_models_pytorch and "predictions_mobilenet" in globals():
    submission_df = submission_df.merge(
        predictions_mobilenet, on="image_id", how="left"
    )



## === cell 10
if stacked_mean:
    needed = {"vit2020", "resnext", "mobilenet", "efficientnetb4"}
    if needed.issubset(set(submission_df.columns)):
        submission_df["stage_1"] = submission_df.apply(
            lambda row: [np.mean(e) for e in zip(row["vit2020"], row["resnext"])],
            axis=1,
        )
        submission_df["label"] = submission_df.apply(
            lambda row: np.argmax(
                [
                    np.sum(e)
                    for e in zip(
                        row["mobilenet"], row["stage_1"], row["efficientnetb4"]
                    )
                ]
            ),
            axis=1,
        )
    else:
        submission_df["label"] = 0
else:
    model_keys = [k for k in used_models_pytorch.keys() if k in submission_df.columns]
    if len(model_keys) == 0:
        submission_df["label"] = 0
    else:
        submission_df["label"] = submission_df.apply(
            lambda row: int(
                np.argmax([np.sum(e) for e in zip(*[row[m] for m in model_keys])])
            ),
            axis=1,
        )



## === cell 11
submission_df.head(3)



## === cell 12
submission_df["label"] = submission_df["label"].astype(int)

out_path = "submission.csv"
submission_df[["image_id", "label"]].to_csv(out_path, index=False)

check = pd.read_csv(out_path)
print(check.head())
print("Wrote submission.csv with rows:", len(check))
print("Expected rows (sample_submission):", len(pd.read_csv(sample_path)))
assert len(check) == len(pd.read_csv(sample_path))
assert list(check.columns) == ["image_id", "label"]
