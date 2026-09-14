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

0.8933212450891508

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I filter the test image list to include only actual JPEG files (recursively) so that the DataLoader never tries to read a directory as an image, fixing the OpenCV error. This also ensures the `label` column is created, allowing the final submission CSV to be written successfully.'
- What this solution (achieved 0.0) has done: 'I add a second pretrained model (ResNet‑152) to the inference pipeline and average its predictions together with the existing ResNet‑50 ensemble. This keeps the original logic intact, only expands the model list, and is expected to raise the accuracy toward the target score while still producing a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'I add a short training stage that fine‑tunes a pretrained ResNet‑50 on the provided training CSV for a couple of epochs, then use this fine‑tuned model for inference (instead of the uniform dummy model). This minimal change keeps the original architecture and inference pipeline while giving a realistic boost in accuracy toward the target score. The script now creates a valid `submission.csv` and leverages the trained weights during the ensemble stage.'
- What this solution (achieved 0.0) has done: 'I replace the outdated `RandomResizedCrop` call in the training augmentation with the current Albumentations signature (using the required `size` argument). This fixes the validation error that stops the script, allowing the fine‑tuning, inference, and CSV creation to run and produce a valid `submission.csv`. No other logic is changed, preserving the existing model ensemble and expected score.'

# 9. Code solution

## === cell 0
import os
import json
import time
import random
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
from torchvision import models, transforms
import albumentations as A
from albumentations import Compose
from albumentations.pytorch import ToTensorV2
import cv2
from tqdm import tqdm
import glob  # added for recursive image discovery


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




## === cell 1
possible_paths = [
    "/kaggle/input/cassava-leaf-disease-classification",
    "/kaggle/working/input/cassava-leaf-disease-classification",
    os.path.abspath(
        os.path.join(
            os.getcwd(),
            "kaggle",
            "input",
            "cassava-leaf-disease-classification",
        )
    ),
    os.path.abspath(
        os.path.join(os.getcwd(), "input", "cassava-leaf-disease-classification")
    ),
]
BASE_DIR = None
for p in possible_paths:
    if os.path.isdir(p):
        BASE_DIR = p
        break
if BASE_DIR is None:
    raise FileNotFoundError(
        "Could not locate the cassava-leaf-disease-classification data directory."
    )
pretrained_models = []  # list of checkpoint paths (empty -> use ResNet‑50)




## === cell 2
SIZE = 512  # image size
num_classes = 5




## === cell 3
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用デバイス: {device}")




## === cell 4
train_csv_path = os.path.join(BASE_DIR, "train.csv")
train_df = pd.read_csv(train_csv_path)

TRAIN_PATH = os.path.join(BASE_DIR, "train_images")
if not os.path.isdir(TRAIN_PATH):
    raise FileNotFoundError(f"train_images folder not found under {BASE_DIR}")


class TrainDataset(data.Dataset):
    def __init__(self, df, transform=None):
        self.image_ids = df["image_id"].tolist()
        self.labels = df["label"].tolist()
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def load_image(self, image_id):
        img_path = os.path.join(TRAIN_PATH, image_id)
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img = self.load_image(image_id)
        label = self.labels[idx]
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, label


train_transform = Compose(
    [
        A.RandomResizedCrop(size=(SIZE, SIZE), scale=(0.8, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0
        ),
        ToTensorV2(),
    ],
    p=1.0,
)

train_dataset = TrainDataset(train_df, transform=train_transform)
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
)

base_model = models.resnet50(pretrained=True)
num_ftrs = base_model.fc.in_features
base_model.fc = nn.Linear(num_ftrs, num_classes)
base_model = base_model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(base_model.parameters(), lr=1e-4)

EPOCHS = 2  # short fine‑tuning to keep runtime low
base_model.train()
for epoch in range(EPOCHS):
    epoch_loss = 0.0
    correct = 0
    total = 0
    for imgs, lbls in tqdm(train_loader, desc=f"Fine‑tune epoch {epoch+1}/{EPOCHS}"):
        imgs = imgs.to(device)
        lbls = lbls.to(device)
        optimizer.zero_grad()
        outputs = base_model(imgs)
        loss = criterion(outputs, lbls)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * imgs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == lbls).sum().item()
        total += lbls.size(0)

    print(f"Epoch {epoch+1} – loss: {epoch_loss/total:.4f} – acc: {correct/total:.4f}")

fine_tuned_resnet50 = base_model
fine_tuned_resnet50.eval()




## === cell 5
TEST_PATH = os.path.join(BASE_DIR, "test_images")
if not os.path.isdir(TEST_PATH):
    TEST_PATH = os.path.join(BASE_DIR, "train_images")
    if not os.path.isdir(TEST_PATH):
        raise FileNotFoundError(
            f"Neither test nor train image folders exist under {BASE_DIR}"
        )

test_files = sorted(
    [
        os.path.relpath(p, TEST_PATH)
        for p in glob.glob(os.path.join(TEST_PATH, "**/*.jpg"), recursive=True)
    ]
)
print(f"Number of test images (full set): {len(test_files)}")
df_test = pd.DataFrame(test_files, columns=["image_id"])




## === cell 6
if len(df_test) == 1:
    df_test.loc[1] = df_test.loc[0]
    print(df_test)




## === cell 7
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
                A.RandomResizedCrop(size=(SIZE, SIZE)),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.RandomResizedCrop(size=(SIZE, SIZE)),
                A.HorizontalFlip(p=1.0),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.RandomResizedCrop(size=(SIZE, SIZE)),
                A.VerticalFlip(p=1.0),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.Rotate(p=1.0),
                A.RandomResizedCrop(size=(SIZE, SIZE)),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
    ]
}




## === cell 8
class TestDataset(data.Dataset):
    def __init__(self, df, transform=None):
        super().__init__()
        self.image_ids = df["image_id"].tolist()
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def load_image(self, image_id):
        img_path = os.path.join(TEST_PATH, image_id)
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found or unable to read: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        img = self.load_image(image_id)
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, image_id




## === cell 9
def predict_model(basename, net, dataloader):
    model_start_time = time.time()
    net.to(device)
    net.eval()
    torch.set_grad_enabled(False)

    probability = []

    for inputs, image_ids in tqdm(dataloader["test"], desc=f"{basename}: "):
        inputs = inputs.to(device)
        outputs = net(inputs, None, "test")
        probability.append(torch.softmax(outputs, dim=1).cpu().numpy())

    print(f"{basename} time: {time.time() - model_start_time:.2f}[sec]")
    return np.concatenate(probability, axis=0)




## === cell 10
class DummyModel(nn.Module):
    """Fallback model that returns uniform probabilities."""

    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, inputs, labels, phase):
        batch = inputs.shape[0]
        return torch.full(
            (batch, self.num_classes), 1.0 / self.num_classes, device=inputs.device
        )


class PretrainedWrapper(nn.Module):
    """Wrap a pretrained torchvision model for the same forward signature."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inputs, labels, phase):
        return self.model(inputs)




## === cell 11
probability = []
start_time = time.time()

default_models = [
    ("resnet50", models.resnet50, 64),
    ("resnet152", models.resnet152, 32),  # added second, stronger model
]

if not pretrained_models:
    print(
        "No pretrained checkpoints found – using pretrained ResNet models for inference."
    )
    for mdl_name, mdl_ctor, BATCH_SIZE in default_models:
        print(f"Loading {mdl_name} (ImageNet pretrained)")

        if mdl_name == "resnet50" and "fine_tuned_resnet50" in globals():
            net = PretrainedWrapper(fine_tuned_resnet50)
        else:
            base_model = mdl_ctor(pretrained=True).to(device)
            net = PretrainedWrapper(base_model)

        for tid, transform_ in enumerate(transform["test"]):
            print(f"{mdl_name} – transform loop={tid}")
            dataset = TestDataset(df_test, transform=transform_)
            dataloader = {
                "test": torch.utils.data.DataLoader(
                    dataset,
                    batch_size=BATCH_SIZE,
                    shuffle=False,
                    num_workers=2,
                    pin_memory=True,
                )
            }
            proba = predict_model(mdl_name, net, dataloader)
            probability.append(proba)

        del net
        torch.cuda.empty_cache()
else:
    for pretrained_model in pretrained_models:
        basename = os.path.splitext(os.path.basename(pretrained_model))[0]

        criterion = nn.CrossEntropyLoss()

        if "resnet18" in basename:
            MODEL_NAME = "resnet18"
            net_base = models.resnet18(pretrained=False)
            net = FinalLayerMixupModel(net_base, criterion, num_classes, False)
            BATCH_SIZE = 64
        elif "resnet50" in basename:
            MODEL_NAME = "resnet50"
            net_base = models.resnet50(pretrained=False)
            net = FinalLayerMixupModel(net_base, criterion, num_classes, False)
            BATCH_SIZE = 32
        elif "resnet152" in basename:
            MODEL_NAME = "resnet152"
            net_base = models.resnet152(pretrained=False)
            net = FinalLayerMixupModel(net_base, criterion, num_classes, False)
            BATCH_SIZE = 16
        elif "resnext101" in basename:
            MODEL_NAME = "resnext101"
            net_base = models.resnext101_32x8d(pretrained=False)
            net = FinalLayerMixupModel(net_base, criterion, num_classes, False)
            BATCH_SIZE = 16
        elif "densenet201" in basename:
            MODEL_NAME = "densenet201"
            net_base = models.densenet201(pretrained=False)
            net = FinalLayerMixupModelDenseNet(net_base, criterion, num_classes, False)
            BATCH_SIZE = 16
        elif "efficientnet-b7" in basename:
            MODEL_NAME = "efficientnet-b7"
            from efficientnet_pytorch import EfficientNet  # lazy import

            net_base = EfficientNet.from_name(MODEL_NAME)
            net = FinalLayerMixupModelEN(net_base, criterion, num_classes, False)
            BATCH_SIZE = 16
        else:
            print(f"{basename} is not supported.")
            continue

        print(f"{basename}: {MODEL_NAME}")

        state_dict = torch.load(pretrained_model, map_location="cpu")
        if MODEL_NAME == "efficientnet-b7":
            net.model.load_state_dict(state_dict)
        else:
            net.load_state_dict(state_dict)

        for param in net.parameters():
            param.requires_grad = False

        for tid, transform_ in enumerate(transform["test"]):
            print(f"transform loop={tid}")
            dataset = TestDataset(df_test, transform=transform_)
            dataloader = {
                "test": torch.utils.data.DataLoader(
                    dataset,
                    batch_size=BATCH_SIZE,
                    shuffle=False,
                    num_workers=4,
                    pin_memory=True,
                )
            }
            proba = predict_model(basename, net, dataloader)
            probability.append(proba)

        del net
        torch.cuda.empty_cache()

mean_proba = np.mean(np.stack(probability, axis=0), axis=0)  # (num_images, num_classes)
df_test["label"] = mean_proba.argmax(axis=1).astype(int)
print(f"total time: {time.time() - start_time:.2f}[sec]")




## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/3658555166.py in <cell line: 0>()
    109         torch.cuda.empty_cache()
    110 
--> 111 mean_proba = np.mean(np.stack(probability, axis=0), axis=0)  # (num_images, num_classes)
    112 df_test["label"] = mean_proba.argmax(axis=1).astype(int)
    113 print(f"total time: {time.time() - start_time:.2f}[sec]")

/usr/local/lib/python3.11/dist-packages/numpy/core/shape_base.py in stack(arrays, axis, out, dtype, casting)
    447     shapes = {arr.shape for arr in arrays}
    448     if len(shapes) != 1:
--> 449         raise ValueError('all input arrays must have the same shape')
    450 
    451     result_ndim = arrays[0].ndim + 1

ValueError: all input arrays must have the same shape

## === cell 12
submission = df_test[["image_id", "label"]]
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")

## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_55/3738666771.py in <cell line: 0>()
----> 1 submission = df_test[["image_id", "label"]]
      2 submission_path = "submission.csv"
      3 submission.to_csv(submission_path, index=False)
      4 print(f"Submission saved to {submission_path}")

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4106             if is_iterator(key):
   4107                 key = list(key)
-> 4108             indexer = self.columns._get_indexer_strict(key, "columns")[1]
   4109 
   4110         # take() does not accept boolean indexers

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _get_indexer_strict(self, key, axis_name)
   6198             keyarr, indexer, new_indexer = self._reindex_non_unique(keyarr)
   6199 
-> 6200         self._raise_if_missing(keyarr, indexer, axis_name)
   6201 
   6202         keyarr = self.take(indexer)

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _raise_if_missing(self, key, indexer, axis_name)
   6250 
   6251             not_found = list(ensure_index(key)[missing_mask.nonzero()[0]].unique())
-> 6252             raise KeyError(f"{not_found} not in index")
   6253 
   6254     @overload

KeyError: "['label'] not in index"
