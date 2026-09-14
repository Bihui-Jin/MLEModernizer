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
pillow==11.3.0
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

0.8679359323058325

# 6. Current score

0.19507

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.61099) has done: 'I remove/guard the notebook-only `!pip`/`!ls` cells and fix the two import/runtime blockers: TensorBoard `SummaryWriter` import (it crashes in this environment) and the missing `efficientnet_pytorch` dependency by switching to `torchvision.models.efficientnet_b7` while preserving the EfficientNet-B7 architecture and 5-class head. I also update the Albumentations imports to the v2 API (the old IAA* transforms were removed) without changing the actual augmentation pipeline used. Finally, I make inference device-safe (don’t call `.cuda()` unconditionally) and ensure a valid `submission.csv` with `image_id,label` is written end-to-end.'
- What this solution (achieved 0.22347) has done: 'I fix the Albumentations v2 breaking change that prevents `transforms_train`/`transforms_valid` from being created (this is the root cause of the downstream `NameError`s). Specifically, `A.RandomResizedCrop` now requires a `size=(H, W)` argument instead of `height=`/`width=` in your installed albumentations==2.0.8, so I update that call while keeping the augmentation pipeline semantics the same. Then the DataLoaders be constructed successfully and inference run to completion. Finally, I keep the submission writing logic unchanged but add a small safety assertion to ensure the output length matches the sample submission and the CSV is actually written.'
- What this solution (achieved 0.17825) has done: 'Your score is extremely low for a pretrained EfficientNet-B7 on Cassava, which strongly suggests an input preprocessing mismatch during inference (most commonly: evaluating a model trained at 512/600px using 224px crops/resizes). To move toward your target with minimal, metric-aligned change, I only adjust inference-time image size to match EfficientNet-B7’s native pretrained resolution (600) and ensure the validation/test transforms use that same size, leaving the model, head, loss, and training loop untouched. This change is safe, keeps core logic identical, and should substantially increase accuracy if the provided `weight.pth` (or the implicit pretrained backbone) expects larger inputs. I also keep the submission writing and ordering exactly the same, with the existing safety assertion.'
- What this solution (achieved 0.17825) has done: 'Your current score (0.17825) is far below the target (0.8679), and the biggest likely cause in this exact pipeline is a train/test preprocessing mismatch: you apply heavy training-time augmentations (RandomResizedCrop, flips, transpose, rotate) even when TRAINING=False, which makes the model see “randomly augmented” images at inference and destroys accuracy. I keep the model, head, loss, and training loop unchanged, but make the train DataLoader use the *validation/inference* transform when not training (so inference is deterministic and matches normalization/resize). I also make test-time inference deterministic by enabling `torch.inference_mode()` and setting `cudnn.benchmark=False` when deterministic is requested, without changing the evaluation semantics. The script still run end-to-end and write a valid `submission.csv` with `image_id,label`.'
- What this solution (achieved 0.19507) has done: 'Your current score is far below the target, so the most likely issue is still a train/test preprocessing mismatch at inference time. I keep the model, head, loss, and training loop unchanged, but make the validation/test preprocessing match the official EfficientNet-B7 pretrained recipe by using the built-in torchvision weights’ `transforms()` (same resize/crop + normalization) instead of a custom Albumentations resize. This is a minimal change that directly affects the metric (accuracy) and often yields a large jump when the model was trained/initialized with ImageNet weights. I also keep deterministic settings and ensure the submission order/length exactly matches `sample_submission.csv`.'

# 9. Code solution

## === cell 0
import os
import sys
import time
import random
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from sklearn.model_selection import StratifiedKFold
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import albumentations as A
from albumentations.pytorch import ToTensorV2

from torchvision import models



## === cell 1
pass



## === cell 2
pass



## === cell 3
pass



## === cell 4
pass



## === cell 5
pass



## === cell 6
pass



## === cell 7
TRAINING = False




## === cell 8
class _NoOpWriter:
    def add_scalar(self, *args, **kwargs):
        return None

    def flush(self):
        return None

    def close(self):
        return None


writer = _NoOpWriter()



## === cell 9
pass



## === cell 10
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device



## === cell 11
if torch.cuda.is_available():
    print(torch.cuda.device_count())



## === cell 12
writer



## === cell 13
SEED = 42
N_FOLDS = 5
N_EPOCHS = 10
BATCH_SIZE = 16

IMG_SIZE = 600

LR = 5e-4
NUM_CLASSES = 5




## === cell 14
def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


seed_everything(SEED)



## === cell 15
base_path = "../input/cassava-leaf-disease-classification/"



## === cell 16
train_path = base_path + "train_images/"
test_path = base_path + "test_images/"

train_csv = pd.read_csv(base_path + "train.csv")
sample = pd.read_csv(base_path + "sample_submission.csv")



## === cell 17
train_csv.head()



## === cell 18
train_csv_disease = train_csv.label.map(
    {
        0: "Cassava Bacterial Blight (CBB)",
        1: "Cassava Brown Streak Disease (CBSD)",
        2: "Cassava Green Mottle (CGM)",
        3: "Cassava Mosaic Disease (CMD)",
        4: "Healthy",
    }
)
diseases = train_csv_disease.value_counts()



## === cell 19
diseases



## === cell 20
try:
    diseases.plot.pie()
    plt.show()
except Exception:
    pass



## === cell 21
pass




## === cell 22
class CasavaDataset(Dataset):
    def __init__(self, dataframe, transforms=None, test=False):
        self.df = dataframe.reset_index(drop=True)
        self.transforms = transforms
        self.test = test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        label = int(self.df.loc[idx, "label"]) if "label" in self.df.columns else -1
        p = self.df.loc[idx, "image_id"]

        p_path = (test_path if self.test else train_path) + p

        image = Image.open(p_path).convert("RGB")
        image = np.array(image)

        if self.transforms is not None:
            transformed = self.transforms(image=image)
            image = transformed["image"]

        return image, label




## === cell 23
pass



## === cell 24
transforms_train = A.Compose(
    [
        A.RandomResizedCrop(size=(IMG_SIZE, IMG_SIZE)),
        A.Transpose(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(p=0.5),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            max_pixel_value=255.0,
            p=1.0,
        ),
        ToTensorV2(p=1.0),
    ],
    p=1.0,
)


class TorchvisionWeightsWrapper:
    def __init__(self, tv_transform):
        self.tv_transform = tv_transform

    def __call__(self, image=None, **kwargs):
        if image is None:
            raise ValueError("Expected `image` key for transform.")
        pil = Image.fromarray(image)
        t = self.tv_transform(pil)  # returns torch.Tensor CHW float normalized
        return {"image": t}


try:
    _eff_w = models.EfficientNet_B7_Weights.IMAGENET1K_V1
except Exception:
    _eff_w = None

if _eff_w is not None:
    transforms_valid = TorchvisionWeightsWrapper(_eff_w.transforms())
else:
    transforms_valid = A.Compose(
        [
            A.Resize(height=IMG_SIZE, width=IMG_SIZE),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_pixel_value=255.0,
                p=1.0,
            ),
            ToTensorV2(p=1.0),
        ],
        p=1.0,
    )



## === cell 25
folds = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)



## === cell 26
train_csv.shape



## === cell 27
model_name = "efficientnet-b7"

if TRAINING:
    model = models.efficientnet_b7(weights=None)
else:
    try:
        model = models.efficientnet_b7(
            weights=models.EfficientNet_B7_Weights.IMAGENET1K_V1
        )
    except Exception:
        model = models.efficientnet_b7(weights="IMAGENET1K_V1")

in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)

weights_file = "../input/ramki-cassava-weights/weight.pth"
if not TRAINING:
    if os.path.exists(weights_file):
        state = torch.load(weights_file, map_location="cpu")
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"model loaded from weight file: {weights_file}")
        if len(missing) or len(unexpected):
            print(
                f"load_state_dict non-strict: missing={len(missing)}, unexpected={len(unexpected)}"
            )
    else:
        print(
            f"NOTE: weights file not found at {weights_file}; using torchvision pretrained weights."
        )

model.to(device)



## === cell 28
pass



## === cell 29
layer = 0
for child in model.children():
    layer += 1
print(layer)



## === cell 30
pass



## === cell 31
pass



## === cell 32
pass



## === cell 33
pass



## === cell 34
train_transforms_for_loader = transforms_train if TRAINING else transforms_valid

trainset = CasavaDataset(train_csv, transforms=train_transforms_for_loader, test=False)
train_loader = DataLoader(
    trainset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=torch.cuda.is_available(),
)

testset = CasavaDataset(sample, transforms=transforms_valid, test=True)
test_loader = DataLoader(
    testset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=torch.cuda.is_available(),
)



## === cell 35
len(train_loader)



## === cell 36
len(test_loader)



## === cell 37
BATCH_SIZE




## === cell 38
class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        val = float(val)
        self.val = val
        self.sum += val * n
        self.count += int(n)
        self.avg = self.sum / max(1, self.count)




## === cell 39
def show_metrics(model, epoch, dataloader, criterion, optimizer):
    model.eval()

    losses = AverageMeter()
    accs = AverageMeter()

    complete_preds = []
    complete_labels = []

    tk = tqdm(dataloader, total=len(dataloader), position=0, leave=True)
    with torch.no_grad():
        for idx, (imgs, labels) in enumerate(tk):
            imgs = imgs.to(device)
            labels = labels.to(device).long()

            output = model(imgs)
            loss = criterion(output, labels)

            preds = output.argmax(1)
            complete_preds.append(preds.cpu().numpy())
            complete_labels.append(labels.cpu().numpy())

            correctly_identified_sum = (preds == labels).sum().item()
            number_of_images = imgs.size(0)

            accs.update(correctly_identified_sum / number_of_images, number_of_images)
            losses.update(loss.item(), number_of_images)

            tk.set_postfix(loss=losses.avg, acc=accs.avg)

    complete_preds = np.concatenate(complete_preds)
    complete_labels = np.concatenate(complete_labels)

    cf_matrix = confusion_matrix(complete_labels, complete_preds)
    sns.heatmap(cf_matrix, annot=True, fmt="d", cmap="YlGnBu")
    plt.show()

    from sklearn.metrics import classification_report

    target_names = [
        "Cassava Bacterial Blight (CBB)",
        "Cassava Brown Streak Disease (CBSD)",
        "Cassava Green Mottle (CGM)",
        "Cassava Mosaic Disease (CMD)",
        "Healthy",
    ]
    print(
        classification_report(
            complete_labels, complete_preds, target_names=target_names
        )
    )
    return losses.avg, complete_preds, complete_labels




## === cell 40
(
    "Cassava Bacterial Blight (CBB)",
    "Cassava Brown Streak Disease (CBSD)",
    "Cassava Green Mottle (CGM)",
    "Cassava Mosaic Disease (CMD)",
    "Healthy",
)




## === cell 41
def train_model(model, epoch, dataloader_train, criterion, optimizer):
    model.train()

    losses = AverageMeter()
    accs = AverageMeter()
    tk = tqdm(dataloader_train, total=len(dataloader_train), position=0, leave=True)

    for idx, (imgs, labels) in enumerate(tk):
        imgs_train = imgs.to(device)
        labels_train = labels.to(device).long()

        output_train = model(imgs_train)
        loss = criterion(output_train, labels_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        predicted_classes = output_train.argmax(1)
        correctly_identified_sum = (predicted_classes == labels_train).sum().item()
        number_of_images = imgs_train.size(0)

        accs.update(correctly_identified_sum / number_of_images, number_of_images)
        losses.update(loss.item(), number_of_images)

        tk.set_postfix(loss=losses.avg, acc=accs.avg)

    return losses.avg, accs.avg


def test_model(model, dataloader_valid, criterion):
    model.eval()

    losses = AverageMeter()
    accs = AverageMeter()

    with torch.no_grad():
        tk = tqdm(dataloader_valid, total=len(dataloader_valid), position=0, leave=True)
        for idx, (imgs, labels) in enumerate(tk):
            imgs_valid = imgs.to(device)
            labels_valid = labels.to(device).long()
            output_valid = model(imgs_valid)

            loss = criterion(output_valid, labels_valid)
            losses.update(loss.item(), imgs_valid.size(0))
            accs.update(
                (output_valid.argmax(1) == labels_valid).sum().item()
                / imgs_valid.size(0),
                imgs_valid.size(0),
            )

            tk.set_postfix(loss=losses.avg, acc=accs.avg)

    return losses.avg, accs.avg




## === cell 42
X = train_csv.iloc[:, :-1]
y = train_csv.iloc[:, -1]



## === cell 43
if TRAINING:
    for i_fold, (train_idx, valid_idx) in enumerate(folds.split(X, y)):
        print("Fold {}/{}".format(i_fold + 1, N_FOLDS))

        train = train_csv.iloc[train_idx].reset_index(drop=True)
        valid = train_csv.iloc[valid_idx].reset_index(drop=True)

        dataset_train = CasavaDataset(train, transforms=transforms_train, test=False)
        dataset_valid = CasavaDataset(valid, transforms=transforms_valid, test=False)

        dataloader_train = DataLoader(
            dataset_train, batch_size=BATCH_SIZE, num_workers=4, shuffle=True
        )
        dataloader_valid = DataLoader(
            dataset_valid, batch_size=BATCH_SIZE, num_workers=4, shuffle=False
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0)
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=1, verbose=True, min_lr=1e-5
        )

        best_acc = 0.0
        os.makedirs("weights1", exist_ok=True)

        for epoch in range(N_EPOCHS):
            train_loss, train_acc = train_model(
                model, epoch, dataloader_train, criterion, optimizer
            )
            val_loss, acc = test_model(model, dataloader_valid, criterion)

            writer.add_scalar("training acc", train_acc, i_fold * N_EPOCHS + epoch + 1)
            writer.add_scalar(
                "training loss", train_loss, i_fold * N_EPOCHS + epoch + 1
            )
            writer.add_scalar(
                "validation loss", val_loss, i_fold * N_EPOCHS + epoch + 1
            )
            writer.add_scalar("validation Acc", acc, i_fold * N_EPOCHS + epoch + 1)
            writer.flush()

            scheduler.step(acc)

            if acc > best_acc:
                best_acc = acc
                torch.save(
                    model.state_dict(),
                    f"weights1/weight-at-fold-{i_fold+1}-epoch-{epoch}-acc-{best_acc:.5f}.pth",
                )

            print("current_val_acc:", acc, "best_val_acc:", best_acc)



## === cell 44
test_pred = []
model.eval()

with torch.inference_mode():
    for i, (images, _) in enumerate(tqdm(test_loader, position=0, leave=True)):
        images = images.to(device)
        pred = model(images)
        pred = pred.argmax(1).detach().cpu().numpy().astype("int")
        test_pred.extend(pred.tolist())

assert len(test_pred) == len(
    sample
), f"Pred len {len(test_pred)} != sample len {len(sample)}"

sample["label"] = test_pred
sample.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", sample.shape)
print(sample.head())



## === cell 45
sample



## === cell 46
with open("submission.csv", "r", encoding="utf-8") as f:
    for _ in range(5):
        print(f.readline().rstrip())
