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

0.8948322756119673

# 6. Current score

0.05531

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.05531) has done: 'The script now detects the correct data directory, safely loads EfficientNet from torchvision when the external package is unavailable, fixes Albumentations transform arguments, correctly builds the test DataFrame, aggregates model probabilities, and writes a proper `submission.csv` with the required columns.'
- What this solution (achieved 0.05531) has done: 'Implemented fixes to make the pipeline run end‑to‑end and produce a valid submission:

1. Corrected Albumentations `RandomResizedCrop` usage to match the installed v2 API (`size=SIZE`).
2. Added a fallback when no pretrained `.pth` files are found: the script now builds a ResNet‑50 pretrained on ImageNet, skips loading external weights, and still runs inference.
3. Adjusted the inference loop to handle the fallback model correctly and ensure at least one probability array is generated, avoiding the “need at least one array to stack” error.'
- What this solution (achieved 0.05531) has done: 'The fixes correct the Albumentations `RandomResizedCrop` arguments (it now receives a height‑width tuple instead of a single integer) and ensure the transform dictionary is properly created, which resolves the `NameError` in the inference loop. This restores end‑to‑end execution and produces a valid `submission.csv` while keeping the original model logic unchanged.'
- What this solution (achieved 0.05531) has done: 'The fixes address two run‑time errors and improve the fallback model:
1. `RandomResizedCrop` now uses the correct `size` argument for Albumentations v2, preventing the validation error that stopped the script and left `transform` undefined.  
2. When no external checkpoints are found, the fallback ResNet‑50 is instantiated with ImageNet‑pretrained weights (`pretrained=True`) instead of random weights, yielding much better predictions while keeping the original model logic unchanged.'
- What this solution (achieved 0.10837) has done: 'The fix corrects the Albumentations `RandomResizedCrop` size argument, updates the pretrained‑model search paths to the proper Kaggle `/kaggle/input` location, and ensures the submission file is written to the Kaggle working directory. These changes resolve the runtime errors, allow the fallback ResNet‑50 (or any found checkpoints) to be used, and produce a valid `submission.csv` that can achieve a much higher accuracy.'
- What this solution (achieved 0.1136) has done: 'The fix ensures a usable fallback model: when no checkpoint files are found the placeholder “fallback_resnet50” is now interpreted as an ImageNet‑pretrained ResNet‑50, so the inference loop actually runs and produces predictions, giving a higher validation score while keeping the original architecture unchanged.'
- What this solution (achieved 0.11435) has done: 'I add a lightweight fine‑tuning stage that runs only when no external checkpoints are found. The script load the training CSV, create a simple augmentation pipeline, train the fallback ResNet‑50 for a couple of epochs, and then use the fine‑tuned model for inference. This keeps the original architecture and inference logic unchanged while providing a realistic boost in accuracy toward the target score.'
- What this solution (achieved 0.11173) has done: 'I fixed the Albumentations `RandomResizedCrop` usage in the training transform (now uses the required `size=(SIZE, SIZE)` signature) and extended the fallback fine‑tuning to five epochs with a slightly higher learning rate to boost model performance while keeping the original architecture unchanged.'
- What this solution (achieved 0.10949) has done: 'I increase the fine‑tuning effort for the fallback ResNet‑50 by training longer (20 epochs) with a slightly lower learning rate, which should raise validation accuracy and move the Kaggle score closer to the target while keeping the original architecture and inference pipeline unchanged.'
- What this solution (achieved 0.11211) has done: 'I extend the fallback fine‑tuning by training longer and adding a simple learning‑rate decay schedule. This keeps the original model architecture and inference pipeline intact while giving the network more opportunity to learn from the training data, which should raise the validation‑style accuracy and move the Kaggle score closer to the target.'
- What this solution (achieved 0.10949) has done: 'I set a modest mix‑up strength (α = 0.4) for the fallback ResNet‑50 fine‑tuning.  
Changing the `alpha` argument from `False` to a small positive value lets the model blend samples during training, which typically improves generalisation and raises the validation‑style accuracy while keeping the original architecture and training loop unchanged. This small tweak should move the Kaggle score noticeably toward the target without altering any other logic.'
- What this solution (achieved 0.11099) has done: 'The changes ensure the model is always fine‑tuned on the Cassava training data by always adding a fallback ResNet‑50 to the list of models to evaluate, regardless of whether external checkpoints are found. This guarantees that a properly trained model (instead of an un‑trained ImageNet‑only model) is used for inference, moving the validation‑style accuracy much closer to the target score. Only the logic that decides which models to run is altered; all other architecture and training code remains unchanged.'
- What this solution (achieved 0.05531) has done: 'I fix the training loop mismatch by adjusting the mix‑up model so that its “train” forward pass returns only the logits and loss (as the training code expects). This enables the fallback ResNet‑50 fine‑tuning to actually run, which should markedly improve the validation‑style accuracy and move the Kaggle score much closer to the target.'

# 9. Code solution

## === cell 0
pretrained_models = (
    glob.glob("/kaggle/input/resnet50-04-2019/*.pth")
    + glob.glob("/kaggle/input/resnet152-04-2019data/*.pth")
    + glob.glob("/kaggle/input/eb7-00-baseline/*.pth")
)
if len(pretrained_models) == 0:
    print(
        "⚠️ No external .pth models found – will use a fallback ImageNet pretrained ResNet‑50."
    )
else:
    print(f"{len(pretrained_models)} models found.")
    print("\n".join(np.sort(pretrained_models)))



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/3204161543.py in <cell line: 0>()
      1 pretrained_models = (
----> 2     glob.glob("/kaggle/input/resnet50-04-2019/*.pth")
      3     + glob.glob("/kaggle/input/resnet152-04-2019data/*.pth")
      4     + glob.glob("/kaggle/input/eb7-00-baseline/*.pth")
      5 )

NameError: name 'glob' is not defined

## === cell 1
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
from torchvision import models, transforms
import albumentations as A
from albumentations import Compose
from albumentations.pytorch import ToTensorV2
import os
from pathlib import Path
import random
import json
import time
import tqdm
import cv2


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


SEED = 42
seed_everything(SEED)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/2881071078.py in <cell line: 0>()
     28 
     29 SEED = 42
---> 30 seed_everything(SEED)
     31 

/tmp/ipykernel_56/2881071078.py in seed_everything(seed)
     20     random.seed(seed)
     21     os.environ["PYTHONHASHSEED"] = str(seed)
---> 22     np.random.seed(seed)
     23     torch.manual_seed(seed)
     24     torch.cuda.manual_seed(seed)

NameError: name 'np' is not defined

## === cell 2
import sys

try:
    sys.path.append("/kaggle/input/package/EfficientNet-PyTorch-1.0")
    from efficientnet_pytorch import EfficientNet
except Exception:
    from torchvision.models import efficientnet_b7 as EfficientNet  # fallback



## === cell 3
SIZE = 512  # image size
num_classes = 5



## === cell 4
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用デバイス: {device}")



## === cell 5
possible_dirs = [
    Path("../input/cassava-leaf-disease-classification"),
    Path("/kaggle/input/cassava-leaf-disease-classification"),
    Path("data/cassava-leaf-disease-classification"),
    Path("./cassava-leaf-disease-classification"),
]

BASE_DIR = next((p for p in possible_dirs if p.exists()), Path("."))
TEST_PATH = BASE_DIR / "test_images"

if not TEST_PATH.exists():
    raise FileNotFoundError(f"Test images directory not found: {TEST_PATH}")

test_files = sorted(
    [
        p.name
        for p in TEST_PATH.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]
)
print(f"Number of test images: {len(test_files)}")



## === cell 6
df_test = pd.DataFrame(test_files, columns=["image_id"])
df_test["label"] = 0



## === cell 7
if len(df_test) == 1:
    df_test = pd.concat([df_test, df_test], ignore_index=True)
    print(df_test)



## === cell 8
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

transform = {
    "test": [
        Compose(
            [
                A.CenterCrop(height=SIZE, width=SIZE),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.HorizontalFlip(p=1.0),
                A.CenterCrop(height=SIZE, width=SIZE),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.RandomResizedCrop(size=(SIZE, SIZE), p=1.0),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.RandomResizedCrop(size=(SIZE, SIZE), p=1.0),
                A.HorizontalFlip(p=1.0),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.RandomResizedCrop(size=(SIZE, SIZE), p=1.0),
                A.VerticalFlip(p=1.0),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.Rotate(p=1.0),
                A.RandomResizedCrop(size=(SIZE, SIZE), p=1.0),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
    ]
}



## === cell 9
train_transform = Compose(
    [
        A.RandomResizedCrop(size=(SIZE, SIZE), scale=(0.8, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
        ToTensorV2(p=1.0),
    ],
    p=1.0,
)




## === cell 10
class FinalLayerMixupModel(nn.Module):
    def __init__(self, model, criterion, num_classes, alpha):
        super(FinalLayerMixupModel, self).__init__()
        self.convlayer = torch.nn.Sequential(*(list(model.children())[:-1]))
        num_ftrs = model.fc.in_features
        self.fc = nn.Linear(num_ftrs, num_classes)
        self.criterion = criterion
        self.alpha = alpha

    def forward(self, inputs, labels, phase):
        if phase == "val" or phase == "test":
            x = self.convlayer(inputs)
            x = x.squeeze()
            outputs = self.fc(x)
            if phase == "val":
                loss = self.criterion(outputs, labels)
                return outputs, loss
            else:  # test
                return outputs

        alpha = self.alpha
        lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
        index = torch.randperm(labels.size(0)).to(labels.device)

        mixed_inputs = lam * inputs + (1 - lam) * inputs[index]

        x = self.convlayer(mixed_inputs)
        x = x.squeeze()
        outputs = self.fc(x)

        loss = lam * self.criterion(outputs, labels) + (1 - lam) * self.criterion(
            outputs, labels[index]
        )
        return outputs, loss




## === cell 11
class FinalLayerMixupModelDenseNet(nn.Module):
    def __init__(self, model, criterion, num_classes, alpha):
        super(FinalLayerMixupModelDenseNet, self).__init__()
        self.convlayer = model.features
        self.AdaptiveAvgPool2d = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        num_ftrs = model.classifier.in_features
        self.fc = nn.Linear(num_ftrs, num_classes)
        self.criterion = criterion
        self.alpha = alpha

    def forward(self, inputs, labels, phase):
        if phase == "val":
            x = self.convlayer(inputs)
            x = self.AdaptiveAvgPool2d(x)
            x = x.squeeze()
            outputs = self.fc(x)
            loss = self.criterion(outputs, labels)
            return outputs, loss

        if phase == "test":
            x = self.convlayer(inputs)
            x = self.AdaptiveAvgPool2d(x)
            x = x.squeeze()
            outputs = self.fc(x)
            return outputs

        alpha = self.alpha
        lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
        index = torch.randperm(len(labels))
        x1 = self.convlayer(inputs)
        x2 = self.convlayer(inputs[index])
        x1 = self.AdaptiveAvgPool2d(x1)
        x2 = self.AdaptiveAvgPool2d(x2)
        mixed_x = lam * x1 + (1 - lam) * x2
        mixed_x = mixed_x.squeeze()
        outputs = self.fc(mixed_x)
        loss = lam * self.criterion(outputs, labels) + (1 - lam) * self.criterion(
            outputs, labels[index]
        )
        return outputs, loss, labels, labels[index], lam




## === cell 12
class FinalLayerMixupModelEN(nn.Module):
    def __init__(self, model, criterion, num_classes, alpha):
        super(FinalLayerMixupModelEN, self).__init__()
        num_ftrs = model._fc.in_features
        model._fc = nn.Linear(num_ftrs, num_classes)
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
        raise RuntimeError("Unexpected phase during inference")




## === cell 13
class TestDataset(data.Dataset):
    def __init__(self, df, transform=None):
        super().__init__()
        self.image_ids = df.image_id.tolist()
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def load_image(self, image_id):
        img_path = TEST_PATH / image_id
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        img = self.load_image(image_id)
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, image_id




## === cell 14
class TrainDataset(data.Dataset):
    def __init__(self, df, transform=None, images_dir=None):
        super().__init__()
        self.df = df
        self.image_ids = df["image_id"].tolist()
        self.labels = df["label"].tolist()
        self.transform = transform
        self.images_dir = images_dir

    def __len__(self):
        return len(self.image_ids)

    def load_image(self, image_id):
        img_path = self.images_dir / image_id
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        label = self.labels[idx]
        img = self.load_image(image_id)
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, label




## === cell 15
def predict_model(basename, net, dataloader):
    net.to(device)
    net.eval()
    torch.set_grad_enabled(False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    probability = []
    for inputs, image_ids in tqdm.tqdm(dataloader["test"], desc=f"{basename}: "):
        inputs = inputs.to(device)
        outputs = net(inputs, False, "test")
        probability.append(torch.softmax(outputs, dim=1).cpu().numpy())
    return np.concatenate(probability, axis=0)




## === cell 16
probability = []
start_time = time.time()

fallback_mode = True
if len(pretrained_models) == 0:
    pretrained_models = ["fallback_resnet50"]
else:
    pretrained_models.append("fallback_resnet50")

for pretrained_model in pretrained_models:
    basename = os.path.splitext(os.path.basename(pretrained_model))[0]
    criterion = nn.CrossEntropyLoss()

    if "resnet18" in basename:
        net_core = models.resnet18(pretrained=False)
        net = FinalLayerMixupModel(
            net_core, criterion, num_classes, 0.4
        )  # use mixup α=0.4
        BATCH_SIZE = 64
    elif "resnet50" in basename:
        net_core = models.resnet50(pretrained=True)
        net = FinalLayerMixupModel(
            net_core, criterion, num_classes, 0.4
        )  # use mixup α=0.4
        BATCH_SIZE = 32
    elif "resnet152" in basename:
        net_core = models.resnet152(pretrained=False)
        net = FinalLayerMixupModel(
            net_core, criterion, num_classes, 0.4
        )  # use mixup α=0.4
        BATCH_SIZE = 16
    elif "resnext101" in basename:
        net_core = models.resnext101_32x8d(pretrained=False)
        net = FinalLayerMixupModel(
            net_core, criterion, num_classes, 0.4
        )  # use mixup α=0.4
        BATCH_SIZE = 12
    elif "densenet201" in basename:
        net_core = models.densenet201(pretrained=False)
        net = FinalLayerMixupModelDenseNet(
            net_core, criterion, num_classes, 0.4
        )  # use mixup α=0.4
        BATCH_SIZE = 12
    elif "efficientnet-b7" in basename:
        print(f"{basename} (EfficientNet) not supported in current setup – skipping.")
        continue
    elif fallback_mode and basename == "fallback_resnet50":
        net_core = models.resnet50(pretrained=True)
        net = FinalLayerMixupModel(
            net_core, criterion, num_classes, 0.4
        )  # use mixup α=0.4
        BATCH_SIZE = 32

        train_csv_path = BASE_DIR / "train.csv"
        train_df = pd.read_csv(train_csv_path)
        train_images_dir = BASE_DIR / "train_images"

        train_dataset = TrainDataset(
            train_df, transform=train_transform, images_dir=train_images_dir
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=64,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )

        optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

        net.to(device)
        net.train()
        epochs = 60  # training epochs for fallback fine‑tuning
        for epoch in range(epochs):
            epoch_loss = 0.0
            for imgs, lbls in train_loader:
                imgs = imgs.to(device)
                lbls = lbls.to(device)
                optimizer.zero_grad()
                outputs, loss = net(imgs, lbls, "train")
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            scheduler.step()
            print(
                f"Fallback fine‑tune epoch {epoch+1}/{epochs}, loss: {epoch_loss/len(train_loader):.4f}"
            )
        net.eval()
    else:
        print(f"{basename} is not supported – skipping.")
        continue

    for param in net.parameters():
        param.requires_grad = False

    for tid, transform_ in enumerate(transform["test"]):
        dataset = TestDataset(df_test, transform=transform_)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )
        proba = predict_model(basename, net, {"test": loader})
        probability.append(proba)

    del net
    torch.cuda.empty_cache()

prob_array = np.stack(probability, axis=0)  # (N_models_aug, N_images, 5)
mean_probs = prob_array.mean(axis=0)  # (N_images, 5)
df_test["label"] = mean_probs.argmax(axis=1)
print(f"total time: {time.time() - start_time:.2f}[sec]")



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/901527055.py in <cell line: 0>()
      3 
      4 fallback_mode = True
----> 5 if len(pretrained_models) == 0:
      6     pretrained_models = ["fallback_resnet50"]
      7 else:

NameError: name 'pretrained_models' is not defined

## === cell 17
submission = df_test[["image_id", "label"]]
submission_path = "/kaggle/working/submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
