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
from albumentations import (
    Compose,
    OneOf,
    Normalize,
    Resize,
    RandomResizedCrop,
    RandomCrop,
    CenterCrop,
    HorizontalFlip,
    VerticalFlip,
    Rotate,
    ShiftScaleRotate,
    Transpose,
)
from albumentations.pytorch import ToTensorV2
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from transformers import ViTForImageClassification
import math



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
path = "/kaggle/input/cassava-leaf-disease-classification/"
image_path = os.path.join(path, "test_images/")

IMAGE_SIZE = (512, 512)
submission_df = pd.DataFrame(columns=["image_id", "label"])
submission_df["image_id"] = os.listdir(image_path)
submission_df["label"] = 0



## === cell 2
onlykeras = False
used_models_pytorch = {"mobilenet": f"../input/model-mobilenet/mn3_bt20_ep5_lr1.pth"}
stacked_mean = False




## === cell 3
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


if "resnext" in used_models_pytorch:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_transforms():
        return Compose(
            [
                Resize(512, 512),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )

    def inference(model, states, test_loader, device):
        model.to(device)
        probabilities = []
        for images in test_loader:
            images = images.to(device)
            avg_preds = []
            for state in states:
                model.load_state_dict(state["model"])
                model.eval()
                with torch.no_grad():
                    y_preds = model(images)
                avg_preds.append(y_preds.softmax(1).cpu().numpy())
            avg_preds = np.mean(avg_preds, axis=0)
            probabilities.append(avg_preds)
        return np.concatenate(probabilities)

    predictions_resnext = pd.DataFrame(columns=["image_id"])
    predictions_resnext["image_id"] = submission_df["image_id"].values
    predictions_resnext["image_path_id"] = image_path + predictions_resnext[
        "image_id"
    ].astype(str)

    model = CustomResNext("resnext50_32x4d", pretrained=False)
    states = [torch.load(f) for f in used_models_pytorch["resnext"]]

    test_dataset = TestDataset(predictions_resnext, transform=get_transforms())
    test_loader = DataLoader(
        test_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True
    )
    predictions = inference(model, states, test_loader, device)
    predictions_resnext["resnext"] = [np.squeeze(p) for p in predictions]
    predictions_resnext = predictions_resnext.drop(columns=["image_path_id"])
    torch.cuda.empty_cache()
    del model, states
    gc.collect()



## === cell 4
if "vit" in used_models_pytorch:
    IMG_SIZE = 224
    BATCH_SIZE = 16
    num_classes = 5
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

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

    transforms_val = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    def predict(model, test_dataset):
        preds = []
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        for batch in tqdm(test_loader):
            batch = batch.to(device)
            model.eval()
            with torch.no_grad():
                out = model(batch)
            preds.extend(out.logits.softmax(1).cpu().numpy())
        return preds

    predictions_vit = pd.DataFrame(columns=["image_id"])
    predictions_vit["image_id"] = submission_df["image_id"].values

    model = ViTForImageClassification.from_pretrained(
        used_models_pytorch["vit"], num_labels=num_classes
    )
    test_dataset = LeafDataset(
        df=predictions_vit, data_path=path, mode="test", transforms=transforms_val
    )
    predictions_raw_vit = predict(model, test_dataset)
    predictions_vit["vit"] = [np.squeeze(p) for p in predictions_raw_vit]
    torch.cuda.empty_cache()
    del model
    gc.collect()



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
            nn.Conv2d(inp, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup), h_swish()
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
        loader = torch.utils.data.DataLoader(test_dataset, batch_size=20)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        for batch in tqdm(loader):
            batch = batch.to(device)
            model.eval()
            with torch.no_grad():
                out = model(batch)
            preds.extend(out[:, :5].softmax(1).cpu().numpy())
        return preds

    transforms_val = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    predictions_mobilenet = pd.DataFrame(columns=["image_id"])
    predictions_mobilenet["image_id"] = submission_df["image_id"].values

    model = mobilenetv3_large()
    model.load_state_dict(
        torch.load(used_models_pytorch["mobilenet"], map_location=device)
    )
    model.to(device)

    test_dataset = LeafDataset(
        df=predictions_mobilenet, data_path=path, mode="test", transforms=transforms_val
    )

    predictions_raw_mobilenet = predict(model, test_dataset)
    predictions_mobilenet["mobilenet"] = [
        np.squeeze(p) for p in predictions_raw_mobilenet
    ]
    torch.cuda.empty_cache()
    del model
    gc.collect()



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/380612585.py in <cell line: 0>()
    213     model = mobilenetv3_large()
    214     model.load_state_dict(
--> 215         torch.load(used_models_pytorch["mobilenet"], map_location=device)
    216     )
    217     model.to(device)

/usr/local/lib/python3.11/dist-packages/torch/serialization.py in load(f, map_location, pickle_module, weights_only, mmap, **pickle_load_args)
   1423         pickle_load_args["encoding"] = "utf-8"
   1424 
-> 1425     with _open_file_like(f, "rb") as opened_file:
   1426         if _is_zipfile(opened_file):
   1427             # The zipfile reader is going to advance the current file position.

/usr/local/lib/python3.11/dist-packages/torch/serialization.py in _open_file_like(name_or_buffer, mode)
    749 def _open_file_like(name_or_buffer, mode):
    750     if _is_path(name_or_buffer):
--> 751         return _open_file(name_or_buffer, mode)
    752     else:
    753         if "w" in mode:

/usr/local/lib/python3.11/dist-packages/torch/serialization.py in __init__(self, name, mode)
    730 class _open_file(_opener):
    731     def __init__(self, name, mode):
--> 732         super().__init__(open(name, mode))
    733 
    734     def __exit__(self, *args):

FileNotFoundError: [Errno 2] No such file or directory: '../input/model-mobilenet/mn3_bt20_ep5_lr1.pth'

## === cell 6
submission_df["label"] = 0

if "resnext" in used_models_pytorch:
    submission_df = submission_df.merge(predictions_resnext, on="image_id")
if "mobilenet" in used_models_pytorch:
    submission_df = submission_df.merge(predictions_mobilenet, on="image_id")
if "vit" in used_models_pytorch:
    submission_df = submission_df.merge(predictions_vit, on="image_id")

if stacked_mean:
    submission_df["stage_1"] = submission_df.apply(
        lambda row: [np.mean(e) for e in zip(row["vit2020"], row["resnext"])], axis=1
    )
    submission_df["label"] = submission_df.apply(
        lambda row: np.argmax(
            [
                np.sum(e)
                for e in zip(row["mobilenet"], row["stage_1"], row["efficientnetb4"])
            ]
        ),
        axis=1,
    )
else:
    def combine_row(row):
        probs = [row[m] for m in list(used_models_pytorch.keys())]
        summed = [np.sum(e) for e in zip(*probs)]
        return np.argmax(summed)

    submission_df["label"] = submission_df.apply(combine_row, axis=1)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3804         try:
-> 3805             return self._engine.get_loc(casted_key)
   3806         except KeyError as err:

index.pyx in pandas._libs.index.IndexEngine.get_loc()

index.pyx in pandas._libs.index.IndexEngine.get_loc()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

KeyError: 'mobilenet'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_55/4174061231.py in <cell line: 0>()
     30         return np.argmax(summed)
     31 
---> 32     submission_df["label"] = submission_df.apply(combine_row, axis=1)
     33 

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in apply(self, func, axis, raw, result_type, args, by_row, engine, engine_kwargs, **kwargs)
  10372             kwargs=kwargs,
  10373         )
> 10374         return op.apply().__finalize__(self, method="apply")
  10375 
  10376     def map(

/usr/local/lib/python3.11/dist-packages/pandas/core/apply.py in apply(self)
    914             return self.apply_raw(engine=self.engine, engine_kwargs=self.engine_kwargs)
    915 
--> 916         return self.apply_standard()
    917 
    918     def agg(self):

/usr/local/lib/python3.11/dist-packages/pandas/core/apply.py in apply_standard(self)
   1061     def apply_standard(self):
   1062         if self.engine == "python":
-> 1063             results, res_index = self.apply_series_generator()
   1064         else:
   1065             results, res_index = self.apply_series_numba()

/usr/local/lib/python3.11/dist-packages/pandas/core/apply.py in apply_series_generator(self)
   1079             for i, v in enumerate(series_gen):
   1080                 # ignore SettingWithCopy here in case the user mutates
-> 1081                 results[i] = self.func(v, *self.args, **self.kwargs)
   1082                 if isinstance(results[i], ABCSeries):
   1083                     # If we have a view on v, we need to make a copy because

/tmp/ipykernel_55/4174061231.py in combine_row(row)
     25     # Combine the probability vectors from all available models
     26     def combine_row(row):
---> 27         probs = [row[m] for m in list(used_models_pytorch.keys())]
     28         # zip over the list of probability arrays to sum per class
     29         summed = [np.sum(e) for e in zip(*probs)]

/tmp/ipykernel_55/4174061231.py in <listcomp>(.0)
     25     # Combine the probability vectors from all available models
     26     def combine_row(row):
---> 27         probs = [row[m] for m in list(used_models_pytorch.keys())]
     28         # zip over the list of probability arrays to sum per class
     29         summed = [np.sum(e) for e in zip(*probs)]

/usr/local/lib/python3.11/dist-packages/pandas/core/series.py in __getitem__(self, key)
   1119 
   1120         elif key_is_scalar:
-> 1121             return self._get_value(key)
   1122 
   1123         # Convert generator to list before going through hashable part

/usr/local/lib/python3.11/dist-packages/pandas/core/series.py in _get_value(self, label, takeable)
   1235 
   1236         # Similar to Index.get_value, but we do not fall back to positional
-> 1237         loc = self.index.get_loc(label)
   1238 
   1239         if is_integer(loc):

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3810             ):
   3811                 raise InvalidIndexError(key)
-> 3812             raise KeyError(key) from err
   3813         except TypeError:
   3814             # If we have a listlike key, _check_indexing_error will raise

KeyError: 'mobilenet'

## === cell 7
submission_df[["image_id", "label"]].to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")

## --- ERROR in outputing the csv:
Invalid submission: Submission must have the same length as the answers.
