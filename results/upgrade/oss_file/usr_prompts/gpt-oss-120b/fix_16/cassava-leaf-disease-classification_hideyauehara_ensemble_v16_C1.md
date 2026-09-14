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

0.8934723481414325

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.05531) has done: 'I fix the failure caused by the Albumentations RandomResizedCrop transform (it requires a “size” argument in the installed version) by replacing it with a simple Resize operation, and I also filter the test directory so that only image files are loaded, preventing extra non‑image entries that broke the submission length check. These minimal changes let the notebook run end‑to‑end and generate a correctly sized `submission.csv`.'
- What this solution (achieved 0.05531) has done: 'I correct the dataset and model directory paths so the pretrained weights are actually found and loaded, which replace the fallback constant‑zero predictions with real model outputs and therefore raise the accuracy toward the target score.'
- What this solution (achieved 0.05531) has done: 'I adjust the model weight loading to use `strict=False` so that any mismatched keys (e.g., from a different final‑layer size) don’t prevent loading the pretrained convolutional features. This small change lets the networks use their learned feature extractors, which should raise the accuracy markedly toward the target while keeping all other logic unchanged.'
- What this solution (achieved 0.05531) has done: 'I limit the test‑time augmentations to a single deterministic resize+normalize transform. Using multiple aggressive transforms (flips, rotations, etc.) can introduce noise that harms prediction quality, especially when the model was trained without such test‑time augmentations. By applying only the first, simple transform we keep the core model logic unchanged while making the predictions more stable, which should move the validation accuracy toward the target score.'
- What this solution (achieved 0.0) has done: 'Implemented two key fixes to raise the Kaggle accuracy toward the target:

1. **Recursive model search** – Updated the glob pattern to locate any *.pth* files anywhere under `/kaggle/input`, ensuring pretrained weights are actually found.
2. **Fallback inference** – Added a safe fallback that runs an ImageNet‑pretrained ResNet‑50 when no custom weights are located, guaranteeing non‑random predictions instead of all‑zero outputs.

These minimal changes keep the original pipeline intact while providing realistic model predictions, moving the score much closer to the target.'
- What this solution (achieved 0.10762) has done: 'I add a small ensemble fallback: when no pretrained .pth files are found, the script run two ImageNet‑pretrained models (ResNet‑50 and DenseNet‑201) with the same deterministic resize‑normalize transform, average their softmax predictions, and use that as the final probability. This keeps the original pipeline intact while giving a better-than‑zero prediction, moving the score toward the target.'
- What this solution (achieved 0.0) has done: 'Implemented a fix for the fallback inference path that caused a runtime error when converting tensors that require gradients to NumPy arrays. The tensors are now detached before calling `.numpy()`, and the inference loop is wrapped in a `torch.no_grad()` block to avoid gradient tracking. This resolves the errors in cells 15 and 16, allowing the script to generate a correct `submission.csv` and move the validation score toward the target.'
- What this solution (achieved 0.10762) has done: 'I add a lightweight training stage that runs when no pretrained *.pth models are found. It loads the training images, fine‑tunes a pretrained ResNet‑50 for a couple of epochs, then uses this trained model (instead of the previous ImageNet‑only fallback) to generate predictions. This small addition should raise the accuracy far above the current 0 score and move it toward the target 0.893 while keeping the original inference logic intact.'
- What this solution (achieved 0.76121) has done: 'The fix replaces the ambiguous truth‑value check on the prediction array with a robust handling that works whether `probability` is a list of arrays (ensemble) or a single NumPy array (single model). It always creates `avg_probs` correctly and adds the “mean” column, allowing the subsequent cells to run without errors and produce a valid `submission.csv`.'
- What this solution (achieved 0.76383) has done: 'The change expands the test‑time augmentation loop to use **all** defined transforms instead of only the first one, so the ensemble averages predictions over multiple augmentations. This modest adjustment usually improves validation accuracy and moves the score closer to the target while keeping the core pipeline unchanged.'

# 9. Code solution

## === cell 0
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
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import sys


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


SEED = 42
seed_everything(seed=SEED)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_54/475221999.py in <cell line: 0>()
     32 
     33 SEED = 42
---> 34 seed_everything(seed=SEED)
     35 
     36 

/tmp/ipykernel_54/475221999.py in seed_everything(seed)
     24     random.seed(seed)
     25     os.environ["PYTHONHASHSEED"] = str(seed)
---> 26     np.random.seed(seed)
     27     torch.manual_seed(seed)
     28     torch.cuda.manual_seed(seed)

NameError: name 'np' is not defined

## === cell 1
sys.path.append("/kaggle/input/package/EfficientNet-PyTorch-1.0")
try:
    from efficientnet_pytorch import EfficientNet
except ModuleNotFoundError:
    EfficientNet = None
    print("EfficientNet-PyTorch not available; efficientnet models will be skipped.")




## === cell 2
SIZE = 512  # image size
num_classes = 5




## === cell 3
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用デバイス: {device}")




## === cell 4
BASE_DIR = f"{INPUT_ROOT}/cassava-leaf-disease-classification"
TEST_PATH = f"{BASE_DIR}/test_images"
if not os.path.isdir(TEST_PATH):
    TEST_PATH = f"{BASE_DIR}/train_images"  # fallback for debugging

test_files = [
    f for f in os.listdir(TEST_PATH) if f.lower().endswith((".jpg", ".jpeg", ".png"))
]
print(f"Number of test images: {len(test_files)}")

pretrained_models = glob.glob(f"{INPUT_ROOT}/**/*.pth", recursive=True)
print(f"Found {len(pretrained_models)} pretrained model(s).")




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_54/1939941112.py in <cell line: 0>()
----> 1 BASE_DIR = f"{INPUT_ROOT}/cassava-leaf-disease-classification"
      2 TEST_PATH = f"{BASE_DIR}/test_images"
      3 if not os.path.isdir(TEST_PATH):
      4     TEST_PATH = f"{BASE_DIR}/train_images"  # fallback for debugging
      5 

NameError: name 'INPUT_ROOT' is not defined

## === cell 5
df_test = pd.DataFrame(test_files, columns=["image_id"])
df_test["label"] = 1  # placeholder, will be overwritten later





## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_54/984425213.py in <cell line: 0>()
----> 1 df_test = pd.DataFrame(test_files, columns=["image_id"])
      2 df_test["label"] = 1  # placeholder, will be overwritten later
      3 
      4 # Create a single TestDataset instance that loads all images into memory once
      5 # This avoids repeated disk I/O for each model in the ensemble.

NameError: name 'test_files' is not defined

## === cell 6
if len(df_test) == 1:
    df_test = pd.concat([df_test, df_test], ignore_index=True)
    print(df_test)




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_54/1105722435.py in <cell line: 0>()
----> 1 if len(df_test) == 1:
      2     df_test = pd.concat([df_test, df_test], ignore_index=True)
      3     print(df_test)
      4 
      5 

NameError: name 'df_test' is not defined

## === cell 7
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

transform = {
    "test": [
        Compose(
            [
                A.Resize(height=SIZE, width=SIZE),
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
                A.Resize(height=SIZE, width=SIZE),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.HorizontalFlip(p=1.0),
                A.Resize(height=SIZE, width=SIZE),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.VerticalFlip(p=1.0),
                A.Resize(height=SIZE, width=SIZE),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.Rotate(p=1.0),
                A.Resize(height=SIZE, width=SIZE),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
    ]
}




## === cell 8
class FinalLayerMixupModel(nn.Module):
    def __init__(self, model, criterion, num_classes, alpha):
        super(FinalLayerMixupModel, self).__init__()
        self.convlayer = torch.nn.Sequential(*(list(model.children())[:-1]))
        num_ftrs = model.fc.in_features
        self.fc = nn.Linear(num_ftrs, num_classes)
        self.criterion = criterion
        self.alpha = alpha

    def forward(self, inputs, labels, phase):
        if phase == "val":
            x = self.convlayer(inputs)
            x = x.squeeze()
            outputs = self.fc(x)
            loss = self.criterion(outputs, labels)
            return outputs, loss

        if phase == "test":
            x = self.convlayer(inputs)
            x = x.squeeze()
            outputs = self.fc(x)
            return outputs

        alpha = self.alpha
        lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
        index = torch.randperm(len(labels))
        x1 = self.convlayer(inputs)
        x2 = self.convlayer(inputs[index])
        mixed_x = lam * x1 + (1 - lam) * x2
        mixed_x = mixed_x.squeeze()
        outputs = self.fc(mixed_x)
        loss = lam * self.criterion(outputs, labels) + (1 - lam) * self.criterion(
            outputs, labels[index]
        )
        return outputs, loss, labels, labels[index], lam




## === cell 9
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
            x = self.AdaptiveAvgPool2d(x).squeeze()
            outputs = self.fc(x)
            loss = self.criterion(outputs, labels)
            return outputs, loss

        if phase == "test":
            x = self.convlayer(inputs)
            x = self.AdaptiveAvgPool2d(x).squeeze()
            outputs = self.fc(x)
            return outputs

        alpha = self.alpha
        lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
        index = torch.randperm(len(labels))
        x1 = self.AdaptiveAvgPool2d(self.convlayer(inputs))
        x2 = self.AdaptiveAvgPool2d(self.convlayer(inputs[index]))
        mixed_x = lam * x1 + (1 - lam) * x2
        mixed_x = mixed_x.squeeze()
        outputs = self.fc(mixed_x)
        loss = lam * self.criterion(outputs, labels) + (1 - lam) * self.criterion(
            outputs, labels[index]
        )
        return outputs, loss, labels, labels[index], lam




## === cell 10
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
        print("Unexpected phase")
        sys.exit()




## === cell 11
class TestDataset(data.Dataset):
    """
    Loads all test images into RAM once and returns a stacked tensor of
    all test‑time augmentations for a single image.
    """

    def __init__(self, df, transforms):
        super().__init__()
        self.image_ids = df["image_id"].tolist()
        self.transforms = transforms  # list of albumentations Compose objects
        self._cache = {}
        for img_id in tqdm(self.image_ids, desc="Loading test images into memory"):
            img_path = f"{TEST_PATH}/{img_id}"
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self._cache[img_id] = img

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img = self._cache[image_id]
        aug_imgs = [t(image=img)["image"] for t in self.transforms]
        stacked = torch.stack(aug_imgs)  # shape (N_aug, C, H, W)
        return stacked, image_id




## === cell 12
def predict_model_multi(basename, net, dataloader, n_aug):
    """
    Runs inference for a model, averaging over the provided augmentations.
    Handles both custom models (with (inputs, labels, phase) signature)
    and plain torchvision models (inputs‑only).
    """
    start = time.time()
    net.to(device)
    net.eval()
    torch.set_grad_enabled(False)

    all_probs = []
    for aug_stack, _ in tqdm(dataloader, desc=f"{basename}: "):
        B = aug_stack.size(0)
        aug_stack = aug_stack.view(
            -1, aug_stack.size(2), aug_stack.size(3), aug_stack.size(4)
        )
        aug_stack = aug_stack.to(device)
        try:
            outputs = net(aug_stack, None, "test")  # custom models
        except TypeError:
            outputs = net(aug_stack)  # plain torchvision models
        probs = torch.softmax(outputs, dim=1).cpu()
        probs = probs.view(B, n_aug, num_classes).mean(dim=1)  # (B, num_classes)
        all_probs.append(probs.numpy())

    print(f"{basename} time: {time.time() - start:.2f}[sec]")
    return np.concatenate(all_probs, axis=0)




## === cell 13
TRAIN_PATH = f"{BASE_DIR}/train_images"
train_df = pd.read_csv(f"{BASE_DIR}/train.csv")
train_df.head()


class TrainDataset(data.Dataset):
    def __init__(self, df, transform=None):
        super().__init__()
        self.image_ids = df["image_id"].tolist()
        self.labels = df["label"].tolist()
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def load_image(self, image_id):
        img = cv2.imread(f"{TRAIN_PATH}/{image_id}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        label = self.labels[idx]
        img = self.load_image(image_id)
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, label




## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_54/754336612.py in <cell line: 0>()
----> 1 TRAIN_PATH = f"{BASE_DIR}/train_images"
      2 train_df = pd.read_csv(f"{BASE_DIR}/train.csv")
      3 train_df.head()
      4 
      5 

NameError: name 'BASE_DIR' is not defined

## === cell 14
test_dataset = TestDataset(df_test, transforms=transform["test"])

if not pretrained_models:
    print(
        "No pretrained .pth models found – training a ResNet‑50 on the provided data."
    )
    train_transform = Compose(
        [
            A.Resize(height=SIZE, width=SIZE),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ],
        p=1.0,
    )
    train_dataset = TrainDataset(train_df, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    backbone = models.resnet50(pretrained=True)
    backbone.fc = nn.Linear(backbone.fc.in_features, num_classes)
    backbone = backbone.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(backbone.parameters(), lr=1e-4)

    epochs = 5
    backbone.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for imgs, lbls in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            imgs = imgs.to(device)
            lbls = lbls.to(device)
            optimizer.zero_grad()
            outputs = backbone(imgs)
            loss = criterion(outputs, lbls)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1} loss: {epoch_loss/len(train_loader):.4f}")

    backbone.eval()
    probability = []
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    probs = predict_model_multi(
        "resnet50_trained", backbone, test_loader, len(transform["test"])
    )
    probability.append(probs)

else:
    probability = []
    start_time = time.time()

    for pretrained_model in pretrained_models:
        basename = os.path.splitext(os.path.basename(pretrained_model))[0]
        criterion = nn.CrossEntropyLoss()

        if "resnet18" in basename:
            net = models.resnet18(pretrained=False)
            net = FinalLayerMixupModel(net, criterion, num_classes, False)
            BATCH_SIZE = 64
        elif "resnet50" in basename:
            net = models.resnet50(pretrained=False)
            net = FinalLayerMixupModel(net, criterion, num_classes, False)
            BATCH_SIZE = 32
        elif "resnet152" in basename:
            net = models.resnet152(pretrained=False)
            net = FinalLayerMixupModel(net, criterion, num_classes, False)
            BATCH_SIZE = 16
        elif "resnext101" in basename:
            net = models.resnext101_32x8d(pretrained=False)
            net = FinalLayerMixupModel(net, criterion, num_classes, False)
            BATCH_SIZE = 12
        elif "densenet201" in basename:
            net = models.densenet201(pretrained=False)
            net = FinalLayerMixupModelDenseNet(net, criterion, num_classes, False)
            BATCH_SIZE = 12
        elif "efficientnet-b7" in basename:
            if EfficientNet is None:
                print(f"Skipping {basename} because EfficientNet is unavailable.")
                continue
            net = EfficientNet.from_name("efficientnet-b7")
            net = FinalLayerMixupModelEN(net, criterion, num_classes, False)
            BATCH_SIZE = 10
        else:
            print(f"{basename} is not supported, skipping.")
            continue

        print(f"{basename}: loading model")
        state_dict = torch.load(pretrained_model, map_location=device)
        if "efficientnet" in basename:
            net.model.load_state_dict(state_dict, strict=False)
        else:
            net.load_state_dict(state_dict, strict=False)

        for param in net.parameters():
            param.requires_grad = False

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )
        proba = predict_model_multi(basename, net, test_loader, len(transform["test"]))
        probability.append(proba)

        del net
        torch.cuda.empty_cache()

    print(f"model ensemble total time: {time.time() - start_time:.2f}[sec]")

if isinstance(probability, list) and len(probability) > 0:
    avg_probs = np.mean(np.stack(probability), axis=0)
else:
    avg_probs = np.zeros((len(df_test), num_classes))

df_test["mean"] = avg_probs.argmax(axis=1)




## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_54/3360411259.py in <cell line: 0>()
      1 # Instantiate the TestDataset once so every model re‑uses the in‑memory images.
----> 2 test_dataset = TestDataset(df_test, transforms=transform["test"])
      3 
      4 if not pretrained_models:
      5     print(

NameError: name 'df_test' is not defined

## === cell 15
df_test["label"] = df_test["mean"].astype(int)




## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_54/3181488290.py in <cell line: 0>()
----> 1 df_test["label"] = df_test["mean"].astype(int)
      2 
      3 

NameError: name 'df_test' is not defined

## === cell 16
df_test.head()




## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_54/533082530.py in <cell line: 0>()
----> 1 df_test.head()
      2 
      3 

NameError: name 'df_test' is not defined

## === cell 17
df_test[["image_id", "label"]].to_csv("submission.csv", index=False)

## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_54/121106179.py in <cell line: 0>()
----> 1 df_test[["image_id", "label"]].to_csv("submission.csv", index=False)

NameError: name 'df_test' is not defined
