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

# 5. Target score

0.8878815352070112

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import pandas as pd
import torch
from PIL import Image
from torch.backends import cudnn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import VisionDataset
from torchvision.transforms import InterpolationMode, v2
from torchvision import models
from sklearn.model_selection import train_test_split

torch.set_float32_matmul_precision("high")

torch.manual_seed(3407)
torch.cuda.manual_seed(3407)
cudnn.deterministic = False  # allow nondeterministic kernels for speed
cudnn.benchmark = True  # enable auto‑tuner
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

test_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images"
if os.path.isdir(os.path.join(test_dir, "test_images")):
    test_dir = os.path.join(test_dir, "test_images")
train_dir = "/kaggle/input/cassava-leaf-disease-classification/train_images/"
train_csv_path = "/kaggle/input/cassava-leaf-disease-classification/train.csv"

img_size = 224
batch_size = 32  # larger batch to reduce iteration overhead
num_workers = min(8, os.cpu_count())  # more workers for I/O
num_classes = 5
tta = False

model_path = "/kaggle/input/vit-v1-update/vit_v1_1.pt"
try:
    vit_model = torch.load(model_path, map_location=device)
    print("Loaded checkpoint from", model_path)
except FileNotFoundError:
    print("Checkpoint not found, building a pretrained ViT.")
    vit_model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)

    if isinstance(vit_model.heads, torch.nn.Sequential):
        in_features = vit_model.heads[-1].in_features
        vit_model.heads[-1] = torch.nn.Linear(in_features, num_classes)
    else:
        in_features = vit_model.heads.in_features
        vit_model.heads = torch.nn.Linear(in_features, num_classes)

    vit_model = vit_model.to(device)

    if device.type == "cuda":
        vit_model = torch.compile(vit_model)

    vit_model.train()

    class CassavaTrainDataset(VisionDataset):
        def __init__(self, img_dir, df, transform=None):
            super().__init__(root=img_dir)
            self.img_dir = img_dir
            self.image_ids = df["image_id"].tolist()
            self.labels = df["label"].astype(int).tolist()
            self.transform = transform

        def __len__(self):
            return len(self.image_ids)

        def __getitem__(self, idx):
            img_path = os.path.join(self.img_dir, self.image_ids[idx])
            img = Image.open(img_path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            label = self.labels[idx]
            return img, label

    train_transform = v2.Compose(
        [
            v2.RandomResizedCrop((img_size, img_size), scale=(0.8, 1.0)),
            v2.RandomHorizontalFlip(),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_df = pd.read_csv(train_csv_path)
    train_df, val_df = train_test_split(
        train_df, test_size=0.05, stratify=train_df["label"], random_state=3407
    )

    train_dataset = CassavaTrainDataset(train_dir, train_df, transform=train_transform)
    val_dataset = CassavaTrainDataset(train_dir, val_df, transform=train_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(vit_model.parameters(), lr=3e-4)

    scaler = torch.cuda.amp.GradScaler()

    epochs = 5
    for epoch in range(epochs):
        vit_model.train()
        running_loss = 0.0
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device, non_blocking=True), lbls.to(
                device, non_blocking=True
            )
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = vit_model(imgs)
                loss = criterion(outputs, lbls)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * imgs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{epochs} - Train loss: {epoch_loss:.4f}")

        vit_model.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(device, non_blocking=True), lbls.to(
                    device, non_blocking=True
                )
                with torch.cuda.amp.autocast():
                    preds = vit_model(imgs).argmax(dim=1)
                correct += (preds == lbls).sum().item()
                total += lbls.size(0)
        acc = correct / total if total > 0 else 0
        print(f" Validation accuracy: {acc:.4f}")

    vit_model.eval()
    print("Training complete – model ready for inference.")




## === cell 1
class CassavaDataset(VisionDataset):
    """Dataset for test images (no labels)."""

    def __init__(self, data_dir, transform=None, ttas=None):
        super().__init__(root=data_dir)
        self.transform = transform
        self.images = sorted(
            [
                f
                for f in os.listdir(data_dir)
                if os.path.isfile(os.path.join(data_dir, f))
                and f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
        )
        self.ttas = ttas

    def __getitem__(self, idx):
        filename = self.images[idx]
        img_path = os.path.join(self.root, filename)
        img = Image.open(img_path).convert("RGB")

        if self.ttas is not None and self.transform is not None:
            imgs = [self.transform(t(img)) for t in self.ttas]
        elif self.transform:
            imgs = self.transform(img)
        else:
            imgs = img
        return imgs, filename

    def __len__(self):
        return len(self.images)




## === cell 2
test_transforms = v2.Compose(
    [
        v2.Resize((img_size, img_size), interpolation=InterpolationMode.BICUBIC),
        v2.CenterCrop((img_size, img_size)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

if tta:
    ttas = [
        v2.RandomResizedCrop((img_size, img_size), scale=(0.5, 1.0)),
        v2.RandomRotation(180),
        v2.RandomVerticalFlip(),
        v2.RandomAffine(degrees=180),
        v2.RandomPerspective(distortion_scale=0.5, p=1.0),
    ]
else:
    ttas = None

test_dataset = CassavaDataset(test_dir, transform=test_transforms, ttas=ttas)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,  # keep order for submission
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,
)

normalizer = torch.nn.Softmax(dim=1)



## === cell 3
all_names = []
all_preds = []

vit_model.eval()
if not isinstance(vit_model, torch.nn.Module):
    vit_model = vit_model.to(device)  # ensure model on device if loaded from checkpoint

with torch.no_grad():
    for inputs, filenames in test_loader:
        if tta:
            inputs = torch.stack(inputs).to(device)  # (tta, B, C, H, W)
            B = inputs.shape[1]
            inputs = inputs.view(-1, *inputs.shape[2:])  # (tta*B, C, H, W)
            logits = vit_model(inputs)
            probs = normalizer(logits)
            probs = probs.view(len(ttas), B, -1)  # (tta, B, num_classes)
            avg_probs = probs.mean(dim=0)  # (B, num_classes)
            pred_labels = torch.argmax(avg_probs, dim=1).cpu().tolist()
        else:
            inputs = inputs.to(device)
            logits = vit_model(inputs)
            probs = normalizer(logits)
            pred_labels = torch.argmax(probs, dim=1).cpu().tolist()

        all_names.extend(filenames)
        all_preds.extend(pred_labels)



## === cell 4
submission_path = "submission.csv"
my_submission = pd.DataFrame({"image_id": all_names, "label": all_preds})
my_submission.to_csv(submission_path, index=False)
print(f"Saved submission with {len(my_submission)} rows to {submission_path}")



## === cell 5
my_submission.head()
