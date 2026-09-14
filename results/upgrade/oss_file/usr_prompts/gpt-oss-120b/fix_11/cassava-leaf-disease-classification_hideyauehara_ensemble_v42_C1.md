# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os, glob
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
    torch.backends.cudnn.benchmark = True


SEED = 42
seed_everything(seed=SEED)




## === cell 1
try:
    sys.path.append("/kaggle/input/package/EfficientNet-PyTorch-1.0")
    from efficientnet_pytorch import EfficientNet

    HAVE_EFFICIENTNET = True
except Exception:
    HAVE_EFFICIENTNET = False
    print("EfficientNet library not found – related models will be skipped.")




## === cell 2
SIZE = 512  # image size
num_classes = 5




## === cell 3
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用デバイス: {device}")




## === cell 4
if os.getenv("KAGGLE_KERNEL_RUN_TYPE") == "Interactive":
    BASE_DIR = "/kaggle/input/cassava-leaf-disease-classification"
elif os.getenv("KAGGLE_KERNEL_RUN_TYPE") == "Batch":
    BASE_DIR = "/kaggle/input/cassava-leaf-disease-classification"
else:
    possible_dirs = [
        "data/cassava-leaf-disease-classification",
        "kaggle/input/cassava-leaf-disease-classification",
        "input/cassava-leaf-disease-classification",
        "cassava-leaf-disease-classification",
    ]
    BASE_DIR = next((d for d in possible_dirs if os.path.isdir(d)), "data")

TEST_PATH = os.path.join(BASE_DIR, "test_images")
if not os.path.isdir(TEST_PATH):
    TEST_PATH = os.path.join(BASE_DIR, "train_images")
    print(f"Fallback to train_images at {TEST_PATH}")

test_files = [f for f in os.listdir(TEST_PATH) if f.lower().endswith(".jpg")]
test_files.sort()
print(f"Number of test images found: {len(test_files)}")




## === cell 5
df_test = pd.DataFrame(test_files, columns=["image_id"])
df_test["label"] = 1  # placeholder; will be overwritten later




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
                A.CenterCrop(SIZE, SIZE),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.HorizontalFlip(p=1.0),
                A.CenterCrop(SIZE, SIZE),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.Resize(SIZE, SIZE),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.Resize(SIZE, SIZE),
                A.HorizontalFlip(p=1.0),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.Resize(SIZE, SIZE),
                A.VerticalFlip(p=1.0),
                A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ],
            p=1.0,
        ),
        Compose(
            [
                A.Rotate(p=1.0),
                A.Resize(SIZE, SIZE),
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
        if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
            num_ftrs = model.fc.in_features
        else:
            with torch.no_grad():
                dummy = torch.randn(1, 3, SIZE, SIZE)
                out = self.convlayer(dummy)
                num_ftrs = out.shape[1] * out.shape[2] * out.shape[3]
        self.fc = nn.Linear(num_ftrs, num_classes)
        self.criterion = criterion
        self.alpha = alpha

    def forward(self, inputs, labels, phase):
        if phase in ("val", "train"):
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
        raise ValueError(f"Unsupported phase: {phase}")




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
        if phase in ("val", "train"):
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
        raise ValueError(f"Unsupported phase: {phase}")




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
IMAGE_CACHE = {}


class TestDataset(data.Dataset):
    def __init__(self, df, transform=None):
        super().__init__()
        self.image_ids = df.image_id.tolist()
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def load_image(self, image_id):
        if image_id in IMAGE_CACHE:
            return IMAGE_CACHE[image_id]
        img_path = f"{TEST_PATH}/{image_id}"
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Unable to read image {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        IMAGE_CACHE[image_id] = img
        return img

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        img = self.load_image(image_id)
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, image_id




## === cell 12
def predict_model(basename, net, dataloader):
    model_start_time = time.time()
    net.to(device)
    net.eval()
    net.half()
    torch.set_grad_enabled(False)
    probability = []
    for inputs, image_ids in tqdm(dataloader["test"], desc=f"{basename}: "):
        inputs = inputs.to(device).half()  # FP16 input
        outputs = net(inputs, None, "test")
        prob = torch.softmax(outputs, dim=1).cpu().numpy()
        probability.append(prob)
    print(f"{basename} time: {time.time() - model_start_time:.2f}[sec]")
    return np.concatenate(probability, axis=0)




## === cell 13
probability = []
start_time = time.time()

pretrained_models = []  # ensure defined for the logic below
search_dirs = ["/kaggle/input", "input", "data", "/kaggle/working"]
for d in search_dirs:
    if os.path.isdir(d):
        pretrained_models.extend(glob.glob(os.path.join(d, "**/*.pth"), recursive=True))
print(f"{len(pretrained_models)} models found.")

if not pretrained_models:
    print("No pretrained models found – training a quick ResNet‑18.")
    train_csv_path = os.path.join(BASE_DIR, "train.csv")
    df_train = pd.read_csv(train_csv_path)

    class TrainDataset(data.Dataset):
        def __init__(self, df, transform=None):
            self.image_ids = df["image_id"].tolist()
            self.labels = df["label"].astype(int).tolist()
            self.transform = transform

        def __len__(self):
            return len(self.image_ids)

        def load_image(self, image_id):
            img_path = f"{BASE_DIR}/train_images/{image_id}"
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img

        def __getitem__(self, idx):
            img = self.load_image(self.image_ids[idx])
            if self.transform:
                img = self.transform(image=img)["image"]
            label = self.labels[idx]
            return img, label

    train_transform = Compose(
        [
            A.Resize(SIZE, SIZE),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ],
        p=1.0,
    )

    val_frac = 0.1
    n_total = len(df_train)
    n_val = int(n_total * val_frac)
    df_val = df_train.iloc[:n_val]
    df_trn = df_train.iloc[n_val:]

    train_dataset = TrainDataset(df_trn, transform=train_transform)
    val_dataset = TrainDataset(df_val, transform=train_transform)

    BATCH_SIZE = 64
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    base_model = models.resnet18(pretrained=True)
    criterion = nn.CrossEntropyLoss()
    net = FinalLayerMixupModel(base_model, criterion, num_classes, alpha=False)

    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    epochs = 8  # longer training for better accuracy
    for epoch in range(epochs):
        net.train()
        epoch_loss = 0.0
        for imgs, lbls in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            imgs = imgs.to(device)
            lbls = lbls.to(device)
            optimizer.zero_grad()
            outputs, loss = net(imgs, lbls, "train")
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1} train loss: {epoch_loss/len(train_loader):.4f}")

        net.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs = imgs.to(device)
                lbls = lbls.to(device)
                outputs = net(imgs, lbls, "val")[0]
                preds = outputs.argmax(dim=1)
                correct += (preds == lbls).sum().item()
                total += lbls.size(0)
        print(f"Epoch {epoch+1} val acc: {correct/total:.4f}")

    pretrained_models = ["trained_resnet18"]
    net_wrapped = net
else:
    net_wrapped = None

for pretrained_model in pretrained_models:
    basename = os.path.splitext(os.path.basename(pretrained_model))[0]
    criterion = nn.CrossEntropyLoss()
    if pretrained_model == "trained_resnet18":
        net = net_wrapped
        BATCH_SIZE = 64
    elif "resnet18" in basename:
        MODEL_NAME = "resnet18"
        net = models.resnet18(pretrained=False)
        net = FinalLayerMixupModel(net, criterion, num_classes, False)
        BATCH_SIZE = 64
        net.load_state_dict(torch.load(pretrained_model, map_location=device))
    elif "resnet50" in basename:
        MODEL_NAME = "resnet50"
        net = models.resnet50(pretrained=False)
        net = FinalLayerMixupModel(net, criterion, num_classes, False)
        BATCH_SIZE = 32
        net.load_state_dict(torch.load(pretrained_model, map_location=device))
    elif "resnet152" in basename:
        MODEL_NAME = "resnet152"
        net = models.resnet152(pretrained=False)
        net = FinalLayerMixupModel(net, criterion, num_classes, False)
        BATCH_SIZE = 16
        net.load_state_dict(torch.load(pretrained_model, map_location=device))
    elif "resnext101" in basename:
        MODEL_NAME = "resnext101"
        net = models.resnext101_32x8d(pretrained=False)
        net = FinalLayerMixupModel(net, criterion, num_classes, False)
        BATCH_SIZE = 12
        net.load_state_dict(torch.load(pretrained_model, map_location=device))
    elif "densenet201" in basename:
        MODEL_NAME = "densenet201"
        net = models.densenet201(pretrained=False)
        net = FinalLayerMixupModelDenseNet(net, criterion, num_classes, False)
        BATCH_SIZE = 12
        net.load_state_dict(torch.load(pretrained_model, map_location=device))
    elif "efficientnet-b7" in basename and HAVE_EFFICIENTNET:
        MODEL_NAME = "efficientnet-b7"
        net = EfficientNet.from_name(MODEL_NAME)
        net = FinalLayerMixupModelEN(net, criterion, num_classes, False)
        BATCH_SIZE = 10
        net.model.load_state_dict(torch.load(pretrained_model, map_location=device))
    else:
        print(f"{basename} is not supported – skipping.")
        continue

    print(f"{basename}: model ready")
    for tid, trf in enumerate(transform["test"]):
        print(f"transform loop={tid}")
        dataset = TestDataset(df_test, transform=trf)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )
        proba = predict_model(basename, net, {"test": loader})
        probability.append(proba)
    if pretrained_model != "trained_resnet18":
        del net
        torch.cuda.empty_cache()

if probability:
    prob_array = np.stack(probability)  # (n_runs, n_samples, n_classes)
    mean_probs = prob_array.mean(axis=0)  # (n_samples, n_classes)
    df_test["mean"] = mean_probs.argmax(axis=1)
else:
    df_test["mean"] = 0

print(f"total time: {time.time() - start_time:.2f}[sec]")




## === cell 14
if len(df_test) == 2 and df_test.loc[0, "image_id"] == df_test.loc[1, "image_id"]:
    df_test = pd.read_csv(f"{BASE_DIR}/sample_submission.csv")
else:
    df_test["label"] = df_test["mean"]




## === cell 15
df_test.head()




## === cell 16
df_test[["image_id", "label"]].to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")
