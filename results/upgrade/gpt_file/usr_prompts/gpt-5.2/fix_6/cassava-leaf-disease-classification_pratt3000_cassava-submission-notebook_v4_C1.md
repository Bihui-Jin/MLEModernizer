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

No external packages required in the script and installed.

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

0.8822907222725899

# 6. Current score

0.72123

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.09492) has done: 'I fix the early crash caused by the notebook `tqdm.notebook` import (it triggers a protobuf `MessageFactory` error in Kaggle) by switching to standard `tqdm`, and I remove the directory-walk printing that isn’t needed. I also make the imports and cell ordering consistent so `torch`, `Dataset`, `nn`, and `T` are defined before use, which resolves the cascade of `NameError`s. The provided model path points to a different/non-existent dataset, so I keep the same core inference pipeline but instantiate the defined WideResNet model with ImageNet weights (no training) to ensure end-to-end execution and a valid `submission.csv`. Finally, I fix inference concatenation/argmax shape logic and ensure predictions align with `sample_submission.csv` ordering.'
- What this solution (achieved 0.77018) has done: 'Your current score is low because the model is a pretrained ImageNet WideResNet with a randomly initialized 5-class head that is never trained, so predictions are essentially random. To move the accuracy toward the target, the smallest change that preserves your core architecture and loss is to actually train only the final `fc` layer on `train.csv` for a small number of epochs, then run the same inference code. I keep your dataset class, transforms, model definition, and cross-entropy objective; I add a simple train/valid split plus a standard PyTorch training loop (no early stopping or approximations). I also match transforms between train/valid/test and ensure the submission stays aligned to `sample_submission.csv` order.'
- What this solution (achieved 0.71076) has done: 'Your gap to target is ~0.112 (0.77018 → 0.88229), so we should improve accuracy with the smallest changes that keep your architecture and training semantics intact. The main issue is you only train the randomly initialized 5-class head for 3 epochs at 512px, which underfits; we keep “train only fc” but (1) use a more standard ImageNet input size (224) to let the pretrained backbone features transfer better, (2) train the fc head longer with a slightly lower LR, and (3) add lightweight label smoothing in the same cross-entropy objective to stabilize generalization. We also make runs deterministic and ensure the submission remains aligned to `sample_submission.csv` order and format. These are minimal, safe tweaks that should move the score upward toward your target without changing the core model/training approach.'
- What this solution (achieved 0.71114) has done: 'Your score is far below the target (0.71076 vs 0.88229), so we should improve generalization with minimal changes while keeping the same model (WideResNet101_2), same “train only fc” approach, and same cross-entropy objective. The smallest high-impact fix is to correct the training augmentation: currently you only do horizontal flip, which is too weak; adding standard small geometric and color jitter augmentations (still at 224 and same normalization) typically improves transfer performance without changing core logic. I also add a simple LR schedule (CosineAnnealingLR) while keeping the same optimizer and number of epochs; this usually improves convergence of the fc head without changing semantics or adding early stopping. Finally, I make DataLoader workers/persistent settings slightly more stable and keep the submission writing exactly the same.'
- What this solution (achieved 0.72123) has done: 'Your current score (0.71114) is well below the target (0.88229), so we should improve accuracy with small, safe changes that keep your exact core model (WideResNet101_2), “train only fc” approach, and cross-entropy objective. The biggest likely issue is distribution shift from applying augmentations only during training: we use simple test-time augmentation (TTA) by running inference on a horizontally flipped version too and averaging probabilities, which often gives a meaningful bump without changing training. We also switch `Resize` to `RandomResizedCrop` (train) and `Resize+CenterCrop` (valid/test) to better match ImageNet-style preprocessing while keeping the same 224 resolution and normalization. Finally, we keep everything deterministic and preserve your submission alignment to `sample_submission.csv`.'

# 9. Code solution

## === cell 0
import os
import json
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.models as models
import torchvision.transforms as T

from PIL import Image
from tqdm import tqdm

torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DATA_ROOT = "/kaggle/input/cassava-leaf-disease-classification"
TEST_DIR = os.path.join(DATA_ROOT, "test_images")
TRAIN_DIR = os.path.join(DATA_ROOT, "train_images")
SAMPLE_SUB_PATH = os.path.join(DATA_ROOT, "sample_submission.csv")
TRAIN_CSV_PATH = os.path.join(DATA_ROOT, "train.csv")

assert os.path.exists(TEST_DIR), f"Missing test directory: {TEST_DIR}"
assert os.path.exists(TRAIN_DIR), f"Missing train directory: {TRAIN_DIR}"
assert os.path.exists(
    SAMPLE_SUB_PATH
), f"Missing sample_submission.csv: {SAMPLE_SUB_PATH}"
assert os.path.exists(TRAIN_CSV_PATH), f"Missing train.csv: {TRAIN_CSV_PATH}"




## === cell 1
def get_image(path):
    img = Image.open(path).convert("RGB")
    return img


class GetDataset(Dataset):
    def __init__(self, df, data_root, transforms=None, output_label=True):
        super().__init__()
        self.df = df.reset_index(drop=True).copy()
        self.data_root = data_root
        self.transforms = transforms
        self.output_label = output_label

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index: int):
        path = "{}/{}".format(self.data_root, self.df.iloc[index]["image_id"])
        img = get_image(path)

        if self.transforms:
            img = self.transforms(img)

        if self.output_label:
            label = int(self.df.iloc[index]["label"])
            return img, label
        else:
            return img




## === cell 2
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader:
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        for x in self.dl:
            yield to_device(x, self.device)

    def __len__(self):
        return len(self.dl)


device = get_device()
device




## === cell 3
def accuracy(out, labels):
    _, preds = torch.max(out, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels, label_smoothing=0.1)
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels, label_smoothing=0.1)
        acc = accuracy(out, labels)
        return {"val_loss": loss.detach(), "val_acc": acc}

    def validation_epoch_end(self, outputs):
        batch_loss = [x["val_loss"] for x in outputs]
        epoch_loss = torch.stack(batch_loss).mean()
        batch_acc = [x["val_acc"] for x in outputs]
        epoch_acc = torch.stack(batch_acc).mean()
        return {"val_loss": epoch_loss.item(), "val_acc": epoch_acc.item()}

    def epoch_end(self, epoch, epochs, result):
        print(
            "Epoch: [{}/{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
                epoch,
                epochs,
                result["train_loss"],
                result["val_loss"],
                result["val_acc"],
            )
        )




## === cell 4
class Classifier(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        self.network = models.wide_resnet101_2(pretrained=True)
        number_of_features = self.network.fc.in_features
        self.network.fc = nn.Linear(number_of_features, 5)

    def forward(self, xb):
        return self.network(xb)

    def freeze(self):
        for param in self.network.parameters():
            param.requires_grad = False
        for param in self.network.fc.parameters():
            param.requires_grad = True

    def unfreeze(self):
        for param in self.network.parameters():
            param.requires_grad = True




## === cell 5
IMG_SIZE = 224
IMG_SHAPE = (IMG_SIZE, IMG_SIZE)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

train_transforms = T.Compose(
    [
        T.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomApply(
            [T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)],
            p=0.5,
        ),
        T.RandomRotation(degrees=10),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)

valid_transforms = T.Compose(
    [
        T.Resize(int(IMG_SIZE * 256 / 224)),
        T.CenterCrop(IMG_SIZE),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)

test_transforms = valid_transforms

train_df = pd.read_csv(TRAIN_CSV_PATH)
assert "image_id" in train_df.columns and "label" in train_df.columns
train_df["label"] = train_df["label"].astype(int)

perm = np.random.RandomState(42).permutation(len(train_df))
val_size = int(0.1 * len(train_df))
val_idx = perm[:val_size]
trn_idx = perm[val_size:]

trn_df = train_df.iloc[trn_idx].reset_index(drop=True)
val_df = train_df.iloc[val_idx].reset_index(drop=True)

trn_ds = GetDataset(
    trn_df[["image_id", "label"]],
    TRAIN_DIR,
    transforms=train_transforms,
    output_label=True,
)
val_ds = GetDataset(
    val_df[["image_id", "label"]],
    TRAIN_DIR,
    transforms=valid_transforms,
    output_label=True,
)

BATCH_SIZE = 64

num_workers = 2
persistent_workers = True if num_workers > 0 else False

trn_loader = DataLoader(
    trn_ds,
    batch_size=BATCH_SIZE,
    num_workers=num_workers,
    shuffle=True,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=persistent_workers,
)
val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    num_workers=num_workers,
    shuffle=False,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=persistent_workers,
)

trn_loader = DeviceDataLoader(trn_loader, device)
val_loader = DeviceDataLoader(val_loader, device)

model = Classifier()
model.freeze()
model = model.to(device)




## === cell 6
def evaluate(model, val_loader):
    model.eval()
    outputs = []
    with torch.no_grad():
        for batch in val_loader:
            outputs.append(model.validation_step(batch))
    return model.validation_epoch_end(outputs)


def fit_fc(epochs, lr, model, train_loader, val_loader):
    optimizer = torch.optim.Adam(
        model.network.fc.parameters(), lr=lr, weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        tk0 = tqdm(train_loader, total=len(train_loader))
        for batch in tk0:
            loss = model.training_step(batch)
            train_losses.append(loss.detach())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        scheduler.step()

        result = evaluate(model, val_loader)
        result["train_loss"] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, epochs, result)

    return model


EPOCHS = 8
LR = 5e-4
model = fit_fc(EPOCHS, LR, model, trn_loader, val_loader)



## === cell 7
test_csv = pd.read_csv(SAMPLE_SUB_PATH)
assert "image_id" in test_csv.columns and "label" in test_csv.columns

test_ds = GetDataset(
    test_csv[["image_id"]].copy(),
    TEST_DIR,
    transforms=test_transforms,
    output_label=False,
)

test_loader = DataLoader(
    test_ds,
    batch_size=256,
    num_workers=2,
    shuffle=False,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=True,
)
test_loader = DeviceDataLoader(test_loader, device)

len(test_csv), len(test_ds)




## === cell 8
def inference_tta_hflip(model, test_loader, device):
    model.to(device)
    model.eval()

    probs = []
    tk0 = tqdm(enumerate(test_loader), total=len(test_loader))
    for i, images in tk0:
        with torch.no_grad():
            logits1 = model(images)
            logits2 = model(
                torch.flip(images, dims=[3])
            )  # horizontal flip on tensor (N,C,H,W)
            p1 = logits1.softmax(1)
            p2 = logits2.softmax(1)
            batch_probs = ((p1 + p2) * 0.5).detach().cpu().numpy()  # (bs, 5)
        probs.append(batch_probs)

    probs = np.concatenate(probs, axis=0)  # (N, 5)
    print("predictions shape:", probs.shape)
    return probs


predictions = inference_tta_hflip(model, test_loader, device)

assert predictions.shape[0] == len(test_csv), (predictions.shape, len(test_csv))

test_csv["label"] = predictions.argmax(axis=1).astype(int)
submission_path = "./submission.csv"
test_csv[["image_id", "label"]].to_csv(submission_path, index=False)

print(test_csv.head())
print("Wrote:", submission_path, "rows:", len(test_csv))
test_csv
