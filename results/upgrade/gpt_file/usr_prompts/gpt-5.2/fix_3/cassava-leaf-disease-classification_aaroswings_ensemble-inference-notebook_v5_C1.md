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

0.8773043215472952

# 6. Current score

0.47534

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.10762) has done: 'I fix the Albumentations import breakage by using the v2-compatible import paths and explicitly importing the needed transforms. Since the referenced pretrained model directories don’t exist in this Kaggle environment, I keep the same “ensemble + argmax over summed logits” inference logic but replace the missing external model loads with a small torchvision model created locally so the notebook runs end-to-end. I also make the test dataset enumerate files in a deterministic order and ensure the submission exactly matches `sample_submission.csv` length and image_id order (this resolves the “same length as the answers” error). Finally, I write `submission.csv` with the required columns.'
- What this solution (achieved 0.47534) has done: 'Your current score is far below the target because the fallback model is untrained and the test-time transform uses `RandomResizedCrop`, making predictions essentially random; we can improve a lot without changing the core “ensemble + summed logits + argmax” inference logic. I (1) switch test transforms to deterministic resize/center-crop to stabilize predictions, and (2) train the same ResNet18 (when external pretrained models are missing) on `train.csv` + `train_images` for a small number of epochs within the time limit, then run the exact same inference pipeline. I also ensure normalization matches ImageNet stats (common for ResNets) and keep submission alignment identical to `sample_submission.csv`. These are minimal, metric-aligned changes that should move accuracy substantially toward your target.'

# 9. Code solution

## === cell 0
import io
import os
import random
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False




## === cell 1
batch_size = 32
valid_input_size = 600

test_img_path = "../input/cassava-leaf-disease-classification/test_images"
sample_sub_path = "../input/cassava-leaf-disease-classification/sample_submission.csv"

train_csv_path = "../input/cassava-leaf-disease-classification/train.csv"
train_img_path = "../input/cassava-leaf-disease-classification/train_images"

assert Path(test_img_path).exists(), f"Missing test images folder: {test_img_path}"
assert Path(
    sample_sub_path
).exists(), f"Missing sample_submission.csv: {sample_sub_path}"
assert Path(train_csv_path).exists(), f"Missing train.csv: {train_csv_path}"
assert Path(train_img_path).exists(), f"Missing train images folder: {train_img_path}"




## === cell 2
import torchvision


def _build_model(num_classes: int = 5):
    model = torchvision.models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model


model_fnames = []
p1 = Path("../input/cassava-notebook-16-models")
p2 = Path("../input/cassava-notebook-15-resnext-models/resnext101wsl_epoch_6.pickle")

if p1.exists():
    model_fnames += [x for x in p1.iterdir() if x.is_file()]
if p2.exists():
    model_fnames += [str(p2)]

models = []
loaded_external = False
if len(model_fnames) > 0:
    for x in model_fnames:
        models.append(torch.load(x, map_location=device))
    loaded_external = True
else:
    models = [_build_model(num_classes=5)]

models = [m.to(device).eval() for m in models]
print("Loaded models:", len(models), "external_pretrained_found:", loaded_external)




## === cell 3
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

test_tfms = A.Compose(
    [
        A.LongestMaxSize(max_size=valid_input_size),
        A.PadIfNeeded(
            min_height=valid_input_size,
            min_width=valid_input_size,
            border_mode=0,  # constant
            value=(0, 0, 0),
        ),
        A.CenterCrop(height=valid_input_size, width=valid_input_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ]
)

train_tfms = A.Compose(
    [
        A.RandomResizedCrop(
            size=(valid_input_size, valid_input_size),
            scale=(0.7, 1.0),
            ratio=(0.75, 1.33),
            p=1.0,
        ),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ]
)


class TestDataset(Dataset):
    def __init__(self, path, tfms):
        super().__init__()
        self.path = Path(path)
        self.tfms = tfms
        self.files = sorted([p for p in self.path.iterdir() if p.is_file()])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        filename = file.name
        with open(file, "rb") as f:
            img_bytes = f.read()
        img = np.array(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
        img = self.tfms(image=img)["image"]
        return filename, img


class TrainDataset(Dataset):
    def __init__(self, csv_path, img_dir, tfms):
        super().__init__()
        self.df = pd.read_csv(csv_path)
        self.img_dir = Path(img_dir)
        self.tfms = tfms

        assert "image_id" in self.df.columns and "label" in self.df.columns
        self.df["image_id"] = self.df["image_id"].astype(str)
        self.df["label"] = self.df["label"].astype(int)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fname = row["image_id"]
        y = int(row["label"])
        fpath = self.img_dir / fname
        with open(fpath, "rb") as f:
            img_bytes = f.read()
        img = np.array(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
        img = self.tfms(image=img)["image"]
        return img, y


test_ds = TestDataset(test_img_path, test_tfms)
test_loader = DataLoader(
    test_ds,
    batch_size=batch_size,
    num_workers=min(4, os.cpu_count() or 1),
    pin_memory=torch.cuda.is_available(),
    drop_last=False,
)
print("test images:", len(test_ds))




## === cell 4
def _train_fallback_model(model: torch.nn.Module):
    model.train()

    train_ds = TrainDataset(train_csv_path, train_img_path, train_tfms)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=min(4, os.cpu_count() or 1),
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    criterion = torch.nn.CrossEntropyLoss()

    epochs = 2
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        correct = 0
        total = 0

        for imgs, y in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits = model(imgs)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += float(loss.detach().cpu().item()) * imgs.size(0)
            pred = torch.argmax(logits.detach(), dim=1)
            correct += int((pred == y).sum().detach().cpu().item())
            total += int(imgs.size(0))

        print(
            f"epoch {epoch}/{epochs} - loss: {running_loss/total:.4f} - train_acc: {correct/total:.4f}"
        )

    model.eval()
    return model


if not loaded_external:
    print("Training fallback model (no external pretrained models found)...")
    models = [_train_fallback_model(models[0].to(device))]
else:
    models = [m.to(device).eval() for m in models]




## === cell 5
class EnsemblePredictor:
    def __init__(self, models):
        super().__init__()
        self.models = models

    def predict_on_loader(self, loader):
        predictions = []
        filenames = []

        for file, img in loader:
            img = img.to(device, non_blocking=True)
            pred = None
            for model in self.models:
                with torch.no_grad():
                    out = model(img)
                    if isinstance(out, (tuple, list)):
                        out = out[0]
                    out = out.detach().float().cpu().numpy()
                pred = out if pred is None else (pred + out)

            predictions += [int(np.argmax(x)) for x in pred]
            filenames += list(file)

        return predictions, filenames


predictor = EnsemblePredictor(models)
predictions, filenames = predictor.predict_on_loader(test_loader)

print("preds:", len(predictions), "files:", len(filenames))
assert len(predictions) == len(filenames) == len(test_ds)




## === cell 6
sample_sub = pd.read_csv(sample_sub_path)

pred_map = dict(zip(filenames, predictions))
missing = [
    img_id for img_id in sample_sub["image_id"].tolist() if img_id not in pred_map
]
if len(missing) > 0:
    for m in missing:
        pred_map[m] = 0

sub_df = sample_sub.copy()
sub_df["label"] = sub_df["image_id"].map(pred_map).astype(int)

assert sub_df.shape[0] == sample_sub.shape[0]
assert list(sub_df.columns) == ["image_id", "label"]

sub_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with rows:", len(sub_df))




## === cell 7
print(sub_df.head(10).to_string(index=False))
