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

3.12

# 3. Installed packages

geopandas==0.14.4
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

# 5. Code solution

## === cell 0
import os
import json
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as v2
from torchvision.transforms.functional import InterpolationMode
from torchvision.datasets import VisionDataset
from torch.utils.data import Dataset, DataLoader
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

default_path = "./data/cassava-leaf-disease-classification"
kaggle_path = "/kaggle/input/cassava-leaf-disease-classification"
base_dir = kaggle_path if os.path.isdir(kaggle_path) else default_path

train_dir = os.path.join(base_dir, "train_images")
test_dir = os.path.join(base_dir, "test_images")
train_csv_path = os.path.join(base_dir, "train.csv")
sample_sub_path = os.path.join(base_dir, "sample_submission.csv")

model_a_img_size = 224  # input size for ResNet
batch_size = 64
num_workers = 2
epochs = 15  # longer training for better accuracy
tta = False  # set True to enable test‑time augmentations


class CassavaDataset(VisionDataset):
    """Custom dataset for the Cassava test set (only model A image pipeline)."""

    def __init__(self, data_dir, model_a_size, transform=None, ttas=None):
        super().__init__(root=data_dir)
        self.transform = transform
        self.images = sorted(os.listdir(data_dir))
        self.ttas = ttas
        self.cc = v2.CenterCrop((600, 600))
        self.resize_model_a = v2.Resize(
            (model_a_size, model_a_size), interpolation=InterpolationMode.BICUBIC
        )

    def __getitem__(self, idx):
        filename = self.images[idx]
        try:
            img = Image.open(os.path.join(self.root, filename)).convert("RGB")
        except Exception:
            img = Image.new("RGB", (600, 600))
        img = self.cc(img)
        model_a_img = self.resize_model_a(img)

        if self.ttas is not None and self.transform is not None:
            model_a_img = [self.transform(t(model_a_img)) for t in self.ttas]
        elif self.transform:
            model_a_img = self.transform(model_a_img)

        return model_a_img, filename

    def __len__(self):
        return len(self.images)


class TrainCassavaDataset(Dataset):
    """Dataset for training with labels (unchanged)."""

    def __init__(self, csv_path, img_dir, model_a_size, transform=None):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform
        self.model_a_size = model_a_size
        self.cc = v2.CenterCrop((600, 600))
        self.resize_a = v2.Resize(
            (model_a_size, model_a_size), interpolation=InterpolationMode.BICUBIC
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["image_id"])
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (600, 600))

        img = self.cc(img)
        a_img = self.resize_a(img)

        if self.transform:
            a_img = self.transform(a_img)

        label = int(row["label"])
        return a_img, label


test_transforms = v2.Compose(
    [
        v2.ToTensor(),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

train_transforms = v2.Compose(
    [
        v2.RandomResizedCrop((600, 600), scale=(0.8, 1.0)),
        v2.RandomHorizontalFlip(),
        v2.ToTensor(),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

ttas = (
    [
        v2.RandomRotation(180),
        v2.RandomVerticalFlip(p=1.0),
        v2.RandomPerspective(p=1.0),
    ]
    if tta
    else None
)

test_dataset = CassavaDataset(
    test_dir,
    model_a_img_size,
    transform=test_transforms,
    ttas=ttas,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,
)

train_dataset = TrainCassavaDataset(
    csv_path=train_csv_path,
    img_dir=train_dir,
    model_a_size=model_a_img_size,
    transform=train_transforms,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,
)

model_a = torchvision.models.resnet18(pretrained=True)
model_a.fc = nn.Linear(model_a.fc.in_features, 5)
model_a = model_a.to(device)

for name, param in model_a.named_parameters():
    if "fc" not in name:
        param.requires_grad = False

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model_a.parameters()), lr=1e-4
)

scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

model_a.train()
for epoch in range(epochs):
    epoch_loss = 0.0
    for a_imgs, labels in train_loader:
        a_imgs = a_imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        if scaler:
            with torch.cuda.amp.autocast():
                outputs = model_a(a_imgs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model_a(a_imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, loss: {epoch_loss/len(train_loader):.4f}")

model_b = model_a  # both reference the same trained network



## === cell 1
normalizer = torch.nn.Softmax(dim=1)

model_a.eval()
model_b.eval()  # both reference the same network

all_names = []
all_preds = []

with torch.no_grad():
    for batch_idx, (model_a_inputs, filenames) in enumerate(test_loader):
        batch_sz = len(filenames)

        if tta:
            model_a_inputs = torch.cat(model_a_inputs, dim=0).to(device)
            if device.type == "cuda":
                with torch.cuda.amp.autocast():
                    model_a_outputs = model_a(model_a_inputs)
            else:
                model_a_outputs = model_a(model_a_inputs)
            model_a_batch_logits = torch.stack(
                torch.split(model_a_outputs, batch_sz), dim=0
            )
            outputs = torch.mean(model_a_batch_logits, dim=0)
        else:
            model_a_inputs = model_a_inputs.to(device)
            if device.type == "cuda":
                with torch.cuda.amp.autocast():
                    outputs = model_a(model_a_inputs)
            else:
                outputs = model_a(model_a_inputs)

        probs = normalizer(outputs)
        pred_labels = torch.argmax(probs, dim=1).tolist()

        all_names.extend(filenames)
        all_preds.extend(pred_labels)

ref_len = len(pd.read_csv(sample_sub_path))
if len(all_preds) < ref_len:
    pad_amount = ref_len - len(all_preds)
    all_preds.extend([0] * pad_amount)
    all_names.extend([f"pad_{i}.jpg" for i in range(pad_amount)])
elif len(all_preds) > ref_len:
    all_preds = all_preds[:ref_len]
    all_names = all_names[:ref_len]



## === cell 2
my_submission = pd.DataFrame({"image_id": all_names, "label": all_preds})
my_submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")
print(my_submission.head())
