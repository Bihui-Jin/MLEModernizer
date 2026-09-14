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

0.858114233907525

# 6. Current score

0.72197

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.2216) has done: 'I fix the model loading and size mismatch errors: set the image size to 224 (the ViT default), replace the classifier head safely by extracting the correct `in_features`, and keep the rest of the pipeline unchanged so a valid CSV submission is produced.'
- What this solution (achieved 0.70852) has done: 'I add a lightweight fine‑tuning stage on the provided training CSV so the ViT model learns the cassava classes instead of using only ImageNet weights. After a short training loop (3 epochs) the model is re‑used for test inference, overwriting the earlier low‑accuracy predictions and producing a new `submission.csv`. This keeps the original architecture and most of the pipeline unchanged while substantially raising the validation accuracy toward the target score.'
- What this solution (achieved 0.15433) has done: 'I enable test‑time augmentation (set `tta = True`) so predictions benefit from diverse crops and flips, and I extend the fine‑tuning to 6 epochs (instead of 3) to let the model learn the cassava classes a bit more without altering the architecture or training logic. These modest changes are expected to raise validation accuracy toward the target while keeping the core pipeline intact.'
- What this solution (achieved 0.73244) has done: 'I speed up the pipeline by (1) adding persistent workers to keep DataLoader subprocesses alive across epochs, (2) using automatic mixed‑precision (AMP) for both training and inference, and (3) removing the duplicated early inference (cell 4) so the model is evaluated only once after training. These changes keep the exact model architecture, loss, optimizer, epoch count and augmentations, while reducing I/O and computation overhead.'
- What this solution (achieved 0.72197) has done: 'I fixed the test dataset loader to skip sub‑folders and only read image files, added a lightweight fine‑tuning loop (using the provided training CSV) so the ViT model learns the cassava classes, and kept the original inference logic. These changes eliminate the DataLoader crash and are expected to raise validation accuracy toward the target while preserving the original model architecture.'

# 9. Code solution

## === cell 0
import os, json, pandas as pd
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.models as models
import torchvision.transforms.v2 as v2
import torchvision.io as io  # proper image reading


def get_train_transforms(img_size):
    return v2.Compose(
        [
            v2.RandomResizedCrop((img_size, img_size), scale=(0.8, 1.0)),
            v2.RandomHorizontalFlip(),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def get_test_transforms(img_size):
    return v2.Compose(
        [
            v2.Resize((img_size, img_size)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


class CassavaTrainDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["image_id"])
        img = io.read_image(img_path)
        if self.transform:
            img = self.transform(img)
        label = int(row["label"])
        return img, label


class CassavaTestDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.filenames = sorted(
            [
                f
                for f in os.listdir(img_dir)
                if os.path.isfile(os.path.join(img_dir, f))
                and f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
        )
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_name = self.filenames[idx]
        img_path = os.path.join(self.img_dir, img_name)
        img = io.read_image(img_path)
        if self.transform:
            img = self.transform(img)
        return img, img_name


normalizer = nn.Identity()




## === cell 1
torch.manual_seed(3407)
torch.cuda.manual_seed_all(3407)

cudnn.deterministic = True
cudnn.benchmark = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

torch.set_float32_matmul_precision("high")

train_csv = "/kaggle/input/cassava-leaf-disease-classification/train.csv"
train_dir = "/kaggle/input/cassava-leaf-disease-classification/train_images/"
test_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images/"

img_size = 224
batch_size = 32
num_workers = 4
num_classes = 5
tta = False  # keep deterministic for now

try:
    vit_model = torch.load("/kaggle/input/vit-trial/vit_trial.pt", map_location=device)
    print("Loaded fine‑tuned ViT checkpoint.")
except FileNotFoundError:
    print("Fine‑tuned checkpoint not found – using pretrained ViT as fallback.")
    vit_model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)

if isinstance(vit_model.heads, nn.Linear):
    in_features = vit_model.heads.in_features
elif isinstance(vit_model.heads, nn.Sequential):
    linear_layer = next(
        (m for m in vit_model.heads.modules() if isinstance(m, nn.Linear)), None
    )
    if linear_layer is None:
        raise RuntimeError("Unable to locate Linear layer in vit_model.heads")
    in_features = linear_layer.in_features
else:
    raise RuntimeError("Unexpected type for vit_model.heads")
vit_model.heads = nn.Linear(in_features, num_classes)

vit_model = torch.compile(vit_model)
vit_model.to(device)




## === cell 2
print("Preparing training data …")
full_train_dataset = CassavaTrainDataset(
    train_csv, train_dir, transform=get_train_transforms(img_size)
)

train_len = int(0.9 * len(full_train_dataset))
val_len = len(full_train_dataset) - train_len
train_dataset, val_dataset = random_split(
    full_train_dataset,
    [train_len, val_len],
    generator=torch.Generator().manual_seed(42),
)

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

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(vit_model.parameters(), lr=3e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

epochs = 5
best_val_acc = 0.0

for epoch in range(1, epochs + 1):
    vit_model.train()
    epoch_loss = 0.0
    for imgs, labels in train_loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            logits = vit_model(imgs)
            loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * imgs.size(0)

    epoch_loss /= len(train_loader.dataset)

    vit_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.cuda.amp.autocast():
                logits = vit_model(imgs)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_acc = correct / total
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(vit_model.state_dict(), "best_vit_cassava.pt")
    scheduler.step()

    print(
        f"Epoch {epoch}/{epochs} | Train loss: {epoch_loss:.4f} | "
        f"Val acc: {val_acc:.4f} (best {best_val_acc:.4f})"
    )

vit_model.load_state_dict(torch.load("best_vit_cassava.pt"))
vit_model.eval()




## === cell 3
print("Preparing test data …")
test_dataset = CassavaTestDataset(test_dir, transform=get_test_transforms(img_size))
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,
)




## === cell 4
all_names = []
all_preds = []

with torch.no_grad():
    for inputs, filenames in test_loader:
        if tta:
            inputs = torch.stack(inputs)  # placeholder
        inputs = inputs.to(device, non_blocking=True)
        with torch.cuda.amp.autocast():
            preds = vit_model(inputs)
        pred_labels = torch.argmax(preds, dim=1).cpu().tolist()

        all_names.extend(filenames)
        all_preds.extend(pred_labels)

submission_path = "submission.csv"
pd.DataFrame({"image_id": all_names, "label": all_preds}).to_csv(
    submission_path, index=False
)
print(f"Final submission saved with {len(all_names)} rows to {submission_path}")
