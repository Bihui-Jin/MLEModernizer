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

3.11

# 3. Installed packages

albumentations==2.0.8
geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
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
import os
import random
import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms.functional as TF

import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image

BASE_INPUT = Path("/kaggle/input/cassava-leaf-disease-classification")
TRAIN_CSV_PATH = BASE_INPUT / "train.csv"
TRAIN_IMAGE_PATH = BASE_INPUT / "train_images"
TEST_IMAGE_PATH = BASE_INPUT / "test_images"
SUBMISSION_PATH = "submission.csv"

if torch.cuda.device_count() > 0:
    DEVICES = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
else:
    DEVICES = [torch.device("cpu")]

OUT_FEATURES = 5
NUM_EPOCHS = 20
BATCH_SIZE = 16
IMAGE_SIZE = 224
OPTIMIZER = torch.optim.AdamW
SEED = 42
LR_START = 1e-5
LR_MAX = 2e-4
LR_FINAL = 1e-5
TTA = 3

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True




## === cell 1
def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False  # speed‑up
    torch.backends.cudnn.benchmark = True


seed_everything(SEED)




## === cell 2
def lr_tune(epoch: int) -> float:
    """cosine annealing from LR_START → LR_MAX → LR_FINAL over NUM_EPOCHS."""
    if epoch < NUM_EPOCHS / 2:
        t = epoch / (NUM_EPOCHS / 2)
        return LR_START + 0.5 * (LR_MAX - LR_START) * (1 - np.cos(np.pi * t))
    else:
        t = (epoch - NUM_EPOCHS / 2) / (NUM_EPOCHS / 2)
        return LR_MAX + 0.5 * (LR_FINAL - LR_MAX) * (1 - np.cos(np.pi * t))


train_augs = A.Compose(
    [
        A.RandomResizedCrop(size=(IMAGE_SIZE, IMAGE_SIZE), scale=(0.8, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)

valid_augs = A.Compose(
    [
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)




## === cell 3
class MyCassavaLeafDataset(torch.utils.data.Dataset):
    """
    Lazy‑loading dataset: stores image file paths and reads each image only when accessed.
    This eliminates the costly upfront RAM load while preserving all augmentations.
    """

    def __init__(
        self, csv_path, images_path, transform, mode="train", val_split=0.1, seed=SEED
    ):
        self.df = pd.read_csv(csv_path)
        self.images_path = Path(images_path)
        self.transform = transform
        self.mode = mode

        if mode == "train" or mode == "valid":
            self.indices = self.df.index.tolist()
        else:
            raise ValueError("mode must be 'train' or 'valid'")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        img_name = self.df.iloc[real_idx]["image_id"]
        img_path = self.images_path / img_name
        image = np.array(Image.open(img_path).convert("RGB"))
        if self.transform:
            image = self.transform(image=image)["image"]
        label = int(self.df.iloc[real_idx]["label"])
        return image, label


base_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
num_features = base_model.classifier[1].in_features
base_model.classifier = nn.Sequential(
    nn.Dropout(p=0.2), nn.Linear(num_features, OUT_FEATURES)
)

criterion = nn.CrossEntropyLoss()




## === cell 4
class MyTrainer:
    @staticmethod
    def accurate_count(y_hat, y_true):
        y_hat = y_hat.argmax(dim=1)
        y_true = y_true.argmax(dim=1)
        correct = (y_hat == y_true).sum().item()
        return float(correct)

    @staticmethod
    def calc_valid_acc(model, valid_dataloader):
        model.eval()
        device = next(model.parameters()).device
        total = 0
        correct = 0
        with torch.no_grad():
            for x, y in valid_dataloader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                preds = model(x)
                correct += (preds.argmax(dim=1) == y).sum().item()
                total += len(y)
        return correct / total if total > 0 else 0.0

    def __init__(
        self,
        optimizer_cls,
        model,
        criterion,
        train_loader,
        valid_loader,
        learning_rate_fn=lr_tune,
        num_epochs=NUM_EPOCHS,
        devices=DEVICES,
    ):
        self.optimizer_cls = optimizer_cls
        self.model = model
        self.criterion = criterion
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.lr_fn = learning_rate_fn
        self.epochs = num_epochs
        self.devices = devices

        base_params = [
            p for n, p in self.model.named_parameters() if "classifier" not in n
        ]
        classifier_params = self.model.classifier.parameters()
        self.optimizer = self.optimizer_cls(
            [
                {"params": base_params},
                {"params": classifier_params, "lr": self.lr_fn(0) * 10},
            ],
            lr=self.lr_fn(0),
            weight_decay=0.001,
        )

        self.scaler = (
            torch.cuda.amp.GradScaler()
            if any(d.type.startswith("cuda") for d in self.devices)
            else None
        )

    def train_epoch(self, epoch):
        self.model.train()
        device = self.devices[0]
        total_loss = 0.0
        total_samples = 0
        correct = 0

        lr = self.lr_fn(epoch)
        self.optimizer.param_groups[0]["lr"] = lr
        self.optimizer.param_groups[1]["lr"] = lr * 10

        for x, y in tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs}"):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            self.optimizer.zero_grad()
            if self.scaler:
                with torch.cuda.amp.autocast():
                    logits = self.model(x)
                    loss = self.criterion(logits, y)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(x)
                loss = self.criterion(logits, y)
                loss.backward()
                self.optimizer.step()

            batch_size = len(y)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            correct += (logits.argmax(dim=1) == y).sum().item()

        avg_loss = total_loss / total_samples
        avg_acc = correct / total_samples
        return avg_loss, avg_acc

    def train(self):
        if len(self.devices) > 1:
            self.model = nn.DataParallel(self.model, device_ids=self.devices).to(
                self.devices[0]
            )
        else:
            self.model = self.model.to(self.devices[0])

        best_acc = 0.0
        for epoch in range(self.epochs):
            train_loss, train_acc = self.train_epoch(epoch)
            valid_acc = self.calc_valid_acc(self.model, self.valid_loader)
            if valid_acc > best_acc:
                best_acc = valid_acc
                torch.save(self.model.state_dict(), "best_model.pth")
            print(
                f"Epoch {epoch+1}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, valid_acc={valid_acc:.4f}"
            )
        self.model.load_state_dict(torch.load("best_model.pth"))
        self.model.eval()
        return self.model




## === cell 5
full_df = pd.read_csv(TRAIN_CSV_PATH)
full_df = full_df.sample(frac=1, random_state=SEED).reset_index(drop=True)
val_frac = 0.1
val_size = int(len(full_df) * val_frac)
train_df = full_df.iloc[val_size:].reset_index(drop=True)
valid_df = full_df.iloc[:val_size].reset_index(drop=True)

train_df_path = "train_split.csv"
valid_df_path = "valid_split.csv"
train_df.to_csv(train_df_path, index=False)
valid_df.to_csv(valid_df_path, index=False)

train_set = MyCassavaLeafDataset(
    csv_path=train_df_path,
    images_path=TRAIN_IMAGE_PATH,
    transform=train_augs,
    mode="train",
)
my_train_dataloader = torch.utils.data.DataLoader(
    train_set,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=min(8, os.cpu_count()),
    pin_memory=True,
    persistent_workers=True,
)

valid_set = MyCassavaLeafDataset(
    csv_path=valid_df_path,
    images_path=TRAIN_IMAGE_PATH,
    transform=valid_augs,
    mode="valid",
)
my_valid_dataloader = torch.utils.data.DataLoader(
    valid_set,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=min(8, os.cpu_count()),
    pin_memory=True,
    persistent_workers=True,
)

trainer = MyTrainer(
    optimizer_cls=OPTIMIZER,
    model=base_model,
    criterion=criterion,
    train_loader=my_train_dataloader,
    valid_loader=my_valid_dataloader,
    learning_rate_fn=lr_tune,
    num_epochs=NUM_EPOCHS,
    devices=DEVICES,
)

trained_model = trainer.train()

test_files = sorted([p.name for p in Path(TEST_IMAGE_PATH).glob("*.jpg")])
submission = []

tta_aug = A.Compose(
    [
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)

trained_model = trained_model.to(DEVICES[0])
trained_model.eval()

with torch.no_grad():
    for img_name in tqdm(test_files, desc="Predicting"):
        img_path = Path(TEST_IMAGE_PATH) / img_name
        image = np.array(Image.open(img_path).convert("RGB"))
        logits_sum = None
        for _ in range(TTA):
            tensor = tta_aug(image=image)["image"].unsqueeze(0).to(DEVICES[0])
            logits = trained_model(tensor)
            logits_sum = logits if logits_sum is None else logits_sum + logits
        avg_logits = logits_sum / TTA
        pred_label = int(avg_logits.argmax(dim=1).cpu().item())
        submission.append({"image_id": img_name, "label": pred_label})

submission_df = pd.DataFrame(submission)
submission_df.to_csv(SUBMISSION_PATH, index=False)
print(f"Submission saved to {SUBMISSION_PATH}")

## --- ERROR in outputing the csv:
Invalid submission: Submission must have the same length as the answers.
