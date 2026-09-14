# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import sys
import os
import random
import json
import gc
import cv2
import pandas as pd
import numpy as np

from tqdm import tqdm
from PIL import Image
from sklearn.metrics import accuracy_score
from functools import partial

from albumentations import Compose, Normalize, Resize
from albumentations.pytorch import ToTensorV2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

try:
    from transformers import ViTForImageClassification  # may raise protobuf error

    HAVE_VIT = True
except Exception as e:
    print("ViT import failed, proceeding without Vision Transformer:", e)
    HAVE_VIT = False

import timm
import math



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
path = "/kaggle/input/cassava-leaf-disease-classification/"
image_path = os.path.join(path, "test_images/")

submission_df = pd.DataFrame(columns=["image_id", "label"])
submission_df["image_id"] = sorted(os.listdir(image_path))
submission_df["label"] = 0  # default prediction



## === cell 2
used_models_pytorch = {"mobilenet": "../input/model-mobilenet/mn3_bt20_ep5_lr1.pth"}

verified_models = {}
for name, chk_path in used_models_pytorch.items():
    if os.path.exists(chk_path):
        verified_models[name] = chk_path
    else:
        print(f"Model file for {name} not found at {chk_path}; skipping this model.")
used_models_pytorch = verified_models



## === cell 3
if "resnext" in used_models_pytorch:

    class CustomResNext(nn.Module):
        def __init__(self, model_name="resnext50_32x4d", pretrained=False):
            super().__init__()
            self.model = timm.create_model(model_name, pretrained=pretrained)
            n_features = self.model.fc.in_features
            self.model.fc = nn.Linear(n_features, 5)

        def forward(self, x):
            return self.model(x)

    class TestDataset(Dataset):
        def __init__(self, df, transform=None):
            self.df = df
            self.file_names = df["image_path_id"].values
            self.transform = transform

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            file_name = self.file_names[idx]
            image = cv2.imread(file_name)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if self.transform:
                augmented = self.transform(image=image)
                image = augmented["image"]
            return image

    def get_transforms():
        return Compose(
            [
                Resize(512, 512),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )

    def inference_resnext(model, states, loader, device):
        model.to(device)
        probs = []
        for imgs in loader:
            imgs = imgs.to(device)
            batch_preds = []
            for st in states:
                model.load_state_dict(st["model"])
                model.eval()
                with torch.no_grad():
                    out = model(imgs)
                batch_preds.append(out.softmax(1).cpu().numpy())
            probs.append(np.mean(batch_preds, axis=0))
        return np.concatenate(probs)

    predictions_resnext = pd.DataFrame({"image_id": submission_df["image_id"]})
    predictions_resnext["image_path_id"] = image_path + predictions_resnext[
        "image_id"
    ].astype(str)

    model_resnext = CustomResNext(pretrained=False)
    states_resnext = [
        torch.load(p, map_location="cpu") for p in used_models_pytorch["resnext"]
    ]

    test_dataset = TestDataset(predictions_resnext, transform=get_transforms())
    test_loader = DataLoader(
        test_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True
    )

    resnext_probs = inference_resnext(
        model_resnext,
        states_resnext,
        test_loader,
        torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    predictions_resnext["resnext"] = list(resnext_probs)
    del model_resnext, states_resnext, test_dataset, test_loader
    gc.collect()
else:
    predictions_resnext = None



## === cell 4
if HAVE_VIT and "vit" in used_models_pytorch:
    IMG_SIZE = 224
    BATCH_SIZE = 16
    NUM_CLASSES = 5
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    class LeafDatasetVT(Dataset):
        def __init__(self, df, data_path, mode="test", transforms=None):
            self.entries = df.values
            self.base = data_path
            self.mode = mode
            self.transforms = transforms
            self.dir = "train_images" if mode == "train" else "test_images"

        def __len__(self):
            return len(self.entries)

        def __getitem__(self, idx):
            img_name = self.entries[idx][0]
            img_path = os.path.join(self.base, self.dir, img_name)
            img = Image.open(img_path).convert("RGB")
            if self.transforms:
                img = self.transforms(img)
            return img

    transforms_val = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    def predict_vit(model, dataset):
        loader = DataLoader(dataset, batch_size=BATCH_SIZE)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        all_preds = []
        for batch in tqdm(loader, desc="ViT inference"):
            batch = batch.to(device)
            model.eval()
            with torch.no_grad():
                out = model(batch)
            all_preds.extend(out.logits.softmax(1).cpu().numpy())
        return all_preds

    predictions_vit = pd.DataFrame({"image_id": submission_df["image_id"]})
    vit_model = ViTForImageClassification.from_pretrained(
        used_models_pytorch["vit"], num_labels=NUM_CLASSES
    )
    vit_dataset = LeafDatasetVT(
        predictions_vit, data_path=path, mode="test", transforms=transforms_val
    )
    vit_probs = predict_vit(vit_model, vit_dataset)
    predictions_vit["vit"] = list(vit_probs)
    del vit_model, vit_dataset
    gc.collect()
else:
    predictions_vit = None



## === cell 5
if "mobilenet" in used_models_pytorch:
    def _make_divisible(v, divisor, min_value=None):
        if min_value is None:
            min_value = divisor
        new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
        if new_v < 0.9 * v:
            new_v += divisor
        return new_v

    class h_sigmoid(nn.Module):
        def __init__(self, inplace=True):
            super().__init__()
            self.relu = nn.ReLU6(inplace=inplace)

        def forward(self, x):
            return self.relu(x + 3) / 6

    class h_swish(nn.Module):
        def __init__(self, inplace=True):
            super().__init__()
            self.sigmoid = h_sigmoid(inplace=inplace)

        def forward(self, x):
            return x * self.sigmoid(x)

    class SELayer(nn.Module):
        def __init__(self, channel, reduction=4):
            super().__init__()
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
            nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
            h_swish(),
        )

    class InvertedResidual(nn.Module):
        def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs):
            super().__init__()
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
            return x + self.conv(x) if self.identity else self.conv(x)

    class MobileNetV3(nn.Module):
        def __init__(self, cfgs, mode, num_classes=1000, width_mult=1.0):
            super().__init__()
            assert mode in ["large", "small"]
            input_channel = _make_divisible(16 * width_mult, 8)
            layers = [conv_3x3_bn(3, input_channel, 2)]
            block = InvertedResidual
            for k, t, c, use_se, use_hs, s in cfgs:
                output_channel = _make_divisible(c * width_mult, 8)
                exp_size = _make_divisible(input_channel * t, 8)
                layers.append(
                    block(input_channel, exp_size, output_channel, k, s, use_se, use_hs)
                )
                input_channel = output_channel
            self.features = nn.Sequential(*layers)
            self.conv = conv_1x1_bn(input_channel, exp_size)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            out_ch = {"large": 1280, "small": 1024}[mode]
            out_ch = (
                _make_divisible(out_ch * width_mult, 8) if width_mult > 1.0 else out_ch
            )
            self.classifier = nn.Sequential(
                nn.Linear(exp_size, out_ch),
                h_swish(),
                nn.Dropout(0.2),
                nn.Linear(out_ch, num_classes),
            )
            self._initialize_weights()

        def forward(self, x):
            x = self.features(x)
            x = self.conv(x)
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            return self.classifier(x)

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

    class LeafDatasetMob(Dataset):
        def __init__(self, df, data_path, mode="test", transforms=None):
            self.entries = df.values
            self.base = data_path
            self.mode = mode
            self.transforms = transforms
            self.dir = "train_images" if mode == "train" else "test_images"

        def __len__(self):
            return len(self.entries)

        def __getitem__(self, idx):
            img_name = self.entries[idx][0]
            img_path = os.path.join(self.base, self.dir, img_name)
            img = Image.open(img_path).convert("RGB")
            if self.transforms:
                img = self.transforms(img)
            return img

    def predict_mobilenet(model, dataset):
        loader = DataLoader(dataset, batch_size=20)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        all_preds = []
        for batch in tqdm(loader, desc="MobileNet inference"):
            batch = batch.to(device)
            model.eval()
            with torch.no_grad():
                out = model(batch)
            all_preds.extend(out[:, :5].softmax(1).cpu().numpy())
        return all_preds

    transforms_val = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    predictions_mobilenet = pd.DataFrame({"image_id": submission_df["image_id"]})

    model_mob = mobilenetv3_large()
    model_mob.load_state_dict(
        torch.load(used_models_pytorch["mobilenet"], map_location=device)
    )
    model_mob.to(device)

    mob_dataset = LeafDatasetMob(
        df=predictions_mobilenet, data_path=path, mode="test", transforms=transforms_val
    )

    mob_probs = predict_mobilenet(model_mob, mob_dataset)
    predictions_mobilenet["mobilenet"] = list(mob_probs)

    del model_mob
    gc.collect()
else:
    predictions_mobilenet = None



## === cell 6
if predictions_resnext is not None:
    submission_df = submission_df.merge(
        predictions_resnext[["image_id", "resnext"]], on="image_id", how="left"
    )
if predictions_vit is not None:
    submission_df = submission_df.merge(
        predictions_vit[["image_id", "vit"]], on="image_id", how="left"
    )
if predictions_mobilenet is not None:
    submission_df = submission_df.merge(
        predictions_mobilenet[["image_id", "mobilenet"]], on="image_id", how="left"
    )

available_models = [k for k in used_models_pytorch.keys() if k in submission_df.columns]

if available_models:

    def combine_row(row):
        probs = [row[m] for m in available_models]
        summed = np.sum(probs, axis=0)  # sum across models
        return int(np.argmax(summed))

    submission_df["label"] = submission_df.apply(combine_row, axis=1)
else:
    submission_df["label"] = 0



## === cell 7
submission_path = "submission.csv"
submission_df[["image_id", "label"]].to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")

## --- ERROR in outputing the csv:
Invalid submission: Submission must have the same length as the answers.
