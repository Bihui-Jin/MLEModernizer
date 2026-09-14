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

albumentations==2.0.8
geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
seaborn==0.12.2
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

0.8939256572982774

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.11584) has done: 'I fix the incorrect `TEST_PATH` selection and file listing that currently includes a nested `test_images` directory name, which causes `cv2.imread` to fail with `.../test_images/test_images`. I make the test-file discovery robust by reading `sample_submission.csv` (authoritative image_id list) when available and by filtering `os.listdir` to image files only, preventing directories from entering the dataset. I also fix the Albumentations `RandomResizedCrop` API usage for v2 and make the forward pass safe for batch size 1 by replacing `squeeze()` with `flatten(1)`. These changes are execution/format correctness fixes (and should improve score versus the current “not yielded” state) while preserving the existing ensemble/TTA core logic.'
- What this solution (achieved 0.10762) has done: 'I fix the Albumentations v2 `RandomResizedCrop` API mismatch that currently crashes transform creation, which prevents `transform` from being defined and cascades into later NameError/KeyError failures. I keep the exact TTA/ensemble logic intact, only updating the transform call signature to the required `size=(H,W)` form so the pipeline runs end-to-end. I also make the prediction call robust to the wrapper model’s `forward` signature by passing a dummy tensor for `labels` during test (instead of a Python `False`), avoiding potential type/length issues. With these fixes, the script complete and write a valid `submission.csv` with the required columns and ordering from `sample_submission.csv`, which should materially improve accuracy versus the current 0.11584.'
- What this solution (achieved 0.10762) has done: 'Your very low accuracy is most consistent with the test-time transforms being incompatible with how these pretrained checkpoints were trained (notably `CenterCrop(512)` without a prior resize, plus stochastic `RandomResizedCrop` at inference). To move the score up toward the 0.894 target while preserving the ensemble/TTA core logic, I make the test-time preprocessing deterministic and standard for ImageNet-style models: `Resize(SIZE,SIZE)` followed by optional flips/rotate (no random crop). I also disable CuDNN benchmark (your code currently sets deterministic=True and benchmark=True simultaneously, which can create nondeterministic kernels and slightly unstable outputs). These are minimal, metric-aligned changes that keep the same models/loops/softmax averaging but should substantially improve correctness and thus accuracy.'
- What this solution (achieved 0.0) has done: 'Your current score (0.10762) is far below the target (0.8939), so we should only make small, low-risk changes that improve correctness of inference without changing the model ensemble/TTA core idea. The biggest likely issue is that your wrapper models create a *new randomly initialized final classifier layer* (fc/classifier) but then you load checkpoints with `strict=True`, which either fails silently in some environments or (more likely here) ends up not using the intended trained classifier weights—leading to near-random predictions. I minimally change the wrappers so they **do not replace the classifier layer at init**, and instead use the classifier that comes from the checkpoint (keeping the same forward logic). I also make checkpoint loading robust to common saved-dict formats (`state_dict` key, `module.` prefixes) while still preferring strict loading whenever keys match, because that should move accuracy sharply upward toward the target without changing the architecture or inference averaging.'

# 9. Code solution

## === cell 0
import numpy as np
import glob



## === cell 1
pretrained_models = glob.glob(f"../input/densenet201-04-2019data/*.pth") + glob.glob(
    f"../input/eb7m-seed70/*.pth"
)

print(f"{len(pretrained_models)} models found.")
if len(pretrained_models) > 0:
    print("\n".join(np.sort(pretrained_models)))



## === cell 2
import pandas as pd

import torch
import torch.nn as nn
import torch.utils.data as data

import torchvision
from torchvision import models

import albumentations as A
from albumentations import Compose
from albumentations.pytorch import ToTensorV2

import os
from pathlib import Path
import random
import time
import sys

from tqdm import tqdm

import cv2


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


SEED = 42
seed_everything(seed=SEED)



## === cell 3
from torchvision.models import efficientnet_b7



## === cell 4
SIZE = 512
num_classes = 5



## === cell 5
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用デバイス: {device}")



## === cell 6
DEFAULT_BASE_DIR = "../input/cassava-leaf-disease-classification"
if not os.path.isdir(DEFAULT_BASE_DIR):
    DEFAULT_BASE_DIR = "/kaggle/input/cassava-leaf-disease-classification"

BASE_DIR = DEFAULT_BASE_DIR

test_dir = f"{BASE_DIR}/test_images"
train_dir = f"{BASE_DIR}/train_images"

if os.path.isdir(test_dir):
    TEST_PATH = test_dir
else:
    TEST_PATH = train_dir  # local/debug fallback only

sample_sub_path = f"{BASE_DIR}/sample_submission.csv"
if os.path.isfile(sample_sub_path) and os.path.isdir(test_dir):
    df_test = pd.read_csv(sample_sub_path)
    test_files = df_test["image_id"].tolist()
else:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    test_files = [
        f
        for f in os.listdir(TEST_PATH)
        if os.path.splitext(f.lower())[1] in exts
        and os.path.isfile(os.path.join(TEST_PATH, f))
    ]
    df_test = pd.DataFrame(test_files, columns=["image_id"])
    df_test["label"] = 1

print(f"BASE_DIR={BASE_DIR}")
print(f"TEST_PATH={TEST_PATH}")
print(f"Number of test images: {len(test_files)}")
print(df_test.head())



## === cell 7
if len(df_test) == 1:
    df_test.loc[1] = df_test.loc[0]
    print(df_test)



## === cell 8
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

transform = {
    "test": [
        Compose(
            [
                A.Resize(height=SIZE, width=SIZE, interpolation=cv2.INTER_LINEAR),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.HorizontalFlip(p=1.0),
                A.Resize(height=SIZE, width=SIZE, interpolation=cv2.INTER_LINEAR),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.VerticalFlip(p=1.0),
                A.Resize(height=SIZE, width=SIZE, interpolation=cv2.INTER_LINEAR),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.Rotate(limit=30, p=1.0),
                A.Resize(height=SIZE, width=SIZE, interpolation=cv2.INTER_LINEAR),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
    ]
}




## === cell 9
class FinalLayerMixupModel(nn.Module):
    def __init__(self, model, criterion):
        super(FinalLayerMixupModel, self).__init__()
        self.convlayer = torch.nn.Sequential(*(list(model.children())[:-1]))
        self.fc = (
            model.fc
        )  # keep original fc; checkpoint should contain trained weights
        self.criterion = criterion

    def forward(self, inputs, labels, phase):
        if phase == "val":
            x = self.convlayer(inputs)
            x = x.flatten(1)
            outputs = self.fc(x)
            loss = self.criterion(outputs, labels)
            return outputs, loss

        if phase == "test":
            x = self.convlayer(inputs)
            x = x.flatten(1)
            outputs = self.fc(x)
            return outputs

        print("ここにきてはいけない")
        sys.exit()




## === cell 10
class FinalLayerMixupModelDenseNet(nn.Module):
    def __init__(self, model, criterion):
        super(FinalLayerMixupModelDenseNet, self).__init__()
        self.convlayer = model.features
        self.AdaptiveAvgPool2d = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = (
            model.classifier
        )  # keep original classifier; checkpoint should contain weights
        self.criterion = criterion

    def forward(self, inputs, labels, phase):
        if phase == "val":
            x = self.convlayer(inputs)
            x = self.AdaptiveAvgPool2d(x)
            x = x.flatten(1)
            outputs = self.fc(x)
            loss = self.criterion(outputs, labels)
            return outputs, loss

        if phase == "test":
            x = self.convlayer(inputs)
            x = self.AdaptiveAvgPool2d(x)
            x = x.flatten(1)
            outputs = self.fc(x)
            return outputs

        print("ここにきてはいけない")
        sys.exit()




## === cell 11
class FinalLayerMixupModelEN(nn.Module):
    def __init__(self, model, criterion):
        super(FinalLayerMixupModelEN, self).__init__()
        self.model = model
        self.criterion = criterion

    def forward(self, inputs, labels, phase):
        if phase == "val":
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            return outputs, loss

        if phase == "test":
            outputs = self.model(inputs)
            return outputs

        print("ここにきてはいけない")
        sys.exit()




## === cell 12
class TestDataset(data.Dataset):
    def __init__(self, df, transform=None):
        super().__init__()
        self.image_ids = df.image_id.tolist()
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def load_image(self, image_id):
        img_path = f"{TEST_PATH}/{image_id}"
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found or unreadable: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        img = self.load_image(image_id)
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, image_id




## === cell 13
def predict_model(basename, net, dataloader):
    model_start_time = time.time()

    net.to(device)
    net.eval()
    torch.set_grad_enabled(False)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    probability = []

    for phase in ["test"]:
        progress = tqdm(dataloader[phase], desc=f"{basename}: ")
        for inputs, image_ids in progress:
            inputs = inputs.to(device)
            dummy_labels = torch.zeros(
                inputs.size(0), dtype=torch.long, device=inputs.device
            )
            outputs = net(inputs, dummy_labels, "test")
            probability.append(torch.softmax(outputs, dim=1).cpu().numpy())

    print(f"{basename} time: {time.time() - model_start_time:.2f}[sec]")
    return np.concatenate(probability, axis=0)




## === cell 14
def _extract_state_dict(obj):
    if (
        isinstance(obj, dict)
        and "state_dict" in obj
        and isinstance(obj["state_dict"], dict)
    ):
        return obj["state_dict"]
    return obj


def _strip_module_prefix(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict
    if any(k.startswith("module.") for k in state_dict.keys()):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict


if len(pretrained_models) == 0:
    print(
        "WARNING: No pretrained .pth models found. Falling back to torchvision resnet18 weights for a valid submission."
    )
    pretrained_models = ["__TORCHVISION_RESNET18_FALLBACK__"]

probability = []
start_time = time.time()

for pretrained_model in pretrained_models:
    basename = (
        os.path.splitext(os.path.basename(pretrained_model))[0]
        if pretrained_model != "__TORCHVISION_RESNET18_FALLBACK__"
        else "resnet18_fallback"
    )

    criterion = nn.CrossEntropyLoss()

    if "resnet18" in basename:
        MODEL_NAME = "resnet18"
        net = models.resnet18(pretrained=False)
        net = FinalLayerMixupModel(net, criterion)
        BATCH_SIZE = 64
    elif "resnet50" in basename:
        MODEL_NAME = "resnet50"
        net = models.resnet50(pretrained=False)
        net = FinalLayerMixupModel(net, criterion)
        BATCH_SIZE = 32
    elif "resnet152" in basename:
        MODEL_NAME = "resnet152"
        net = models.resnet152(pretrained=False)
        net = FinalLayerMixupModel(net, criterion)
        BATCH_SIZE = 16
    elif "resnext101" in basename:
        MODEL_NAME = "resnext101"
        net = models.resnext101_32x8d(pretrained=False)
        net = FinalLayerMixupModel(net, criterion)
        BATCH_SIZE = 12
    elif "densenet201" in basename:
        MODEL_NAME = "densenet201"
        net = models.densenet201(pretrained=False)
        net = FinalLayerMixupModelDenseNet(net, criterion)
        BATCH_SIZE = 12
    elif "efficientnet-b7" in basename:
        MODEL_NAME = "efficientnet-b7"
        net = efficientnet_b7(weights=None)
        net = FinalLayerMixupModelEN(net, criterion)
        BATCH_SIZE = 10
    elif pretrained_model == "__TORCHVISION_RESNET18_FALLBACK__":
        MODEL_NAME = "resnet18"
        net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        net = FinalLayerMixupModel(net, criterion)
        BATCH_SIZE = 64
    else:
        print(f"{basename} is not supported.")
        sys.exit()

    print(f"{basename}: {MODEL_NAME}")

    if pretrained_model != "__TORCHVISION_RESNET18_FALLBACK__":
        raw = torch.load(pretrained_model, map_location="cpu")
        state = _strip_module_prefix(_extract_state_dict(raw))

        loaded = False
        try:
            net.load_state_dict(state, strict=True)
            loaded = True
        except RuntimeError:
            if hasattr(net, "model"):
                try:
                    net.model.load_state_dict(state, strict=True)
                    loaded = True
                except RuntimeError:
                    loaded = False
            else:
                loaded = False

        if not loaded:
            if hasattr(net, "model"):
                missing, unexpected = net.model.load_state_dict(state, strict=False)
            else:
                missing, unexpected = net.load_state_dict(state, strict=False)
            print(
                f"WARNING: Non-strict load used for {basename}. missing={len(missing)} unexpected={len(unexpected)}"
            )

    for param in net.parameters():
        param.requires_grad = False

    for tid, transform_ in enumerate(transform["test"]):
        print(f"transform loop={tid}")
        dataset = {"test": TestDataset(df_test, transform=transform_)}
        dataloader = {
            "test": torch.utils.data.DataLoader(
                dataset["test"],
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=2,
                pin_memory=True,
            )
        }
        proba = predict_model(basename, net, dataloader)
        probability.append(proba)

    del net
    torch.cuda.empty_cache()

prob_stack = np.stack(probability, axis=0)
df_test["mean"] = prob_stack.mean(axis=0).argmax(axis=1)

print(f"total time: {time.time() - start_time:.2f}[sec]")



## === cell 15
df_test["label"] = df_test["mean"].astype(int)



## === cell 16
df_test



## === cell 17
sub = df_test[["image_id", "label"]].copy()
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv", sub.shape)
print(sub.head())
