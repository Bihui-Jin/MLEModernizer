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

0.8933212450891508

# 6. Current score

0.76158

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.4787) has done: 'I fix the pipeline so it runs end-to-end and writes a valid `submission.csv` by (1) removing the missing external model-file dependency that currently crashes the notebook, (2) ensuring the test DataLoader iteration order is deterministic and aligned to the provided `sample_submission.csv`, and (3) guaranteeing the submission has exactly the required rows/columns in the correct order. The core inference semantics remain “predict a class label per test image”; the only modeling change is swapping the unavailable pretrained checkpoints for a locally-available torchvision pretrained model so predictions can be produced in this environment. I also fix minor dataset/image loading robustness (RGB conversion) to prevent runtime errors. Finally, I enforce strict row-count matching and ordering using `sample_submission.csv` to avoid the “same length as the answers” submission error.'
- What this solution (achieved 0.76158) has done: 'I fix the immediate crash by handling the torchvision weights transform API change (the returned object is callable and doesn’t expose a `.transforms` list), while keeping the same preprocessing semantics. Then I ensure the train/val loaders are always defined (cell 3 must run) and make the test loader deterministic and aligned to `sample_submission.csv` as intended. Finally, I harden the submission mapping step so it cannot silently produce NaNs (which caused the `IntCastingNaNError`) by checking filename coverage and falling back only if any unexpected mismatch occurs. These changes are execution/stability fixes and should also improve score versus “random-ish” output because training actually run and inference use consistent transforms.'

# 9. Code solution

## === cell 0
import os

import pandas as pd
import torch
from PIL import Image
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
from torchvision.transforms import v2



## === cell 1
torch.manual_seed(3407)
if torch.cuda.is_available():
    torch.cuda.manual_seed(3407)

cudnn.deterministic = False
cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification"
test_dir = f"{DATA_DIR}/test_images/"
train_dir = f"{DATA_DIR}/train_images/"
sample_sub_path = f"{DATA_DIR}/sample_submission.csv"
train_csv_path = f"{DATA_DIR}/train.csv"

batch_size = 32
num_workers = 4
num_classes = 5

import torchvision

weights = torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1
model = torchvision.models.efficientnet_b0(weights=weights)
in_features = model.classifier[1].in_features
model.classifier[1] = torch.nn.Linear(in_features, num_classes)
model = model.to(device)

normalizer = torch.nn.Softmax(dim=1)




## === cell 2
class CassavaDataset(VisionDataset):
    """Custom dataset for Cassava images.

    Returns:
        image_tensor, (label if available else filename)
    """

    def __init__(
        self, data_dir, transform=None, images=None, labels=None, return_filename=False
    ):
        super().__init__(root=data_dir)
        self.transform = transform
        self.return_filename = return_filename

        if images is None:
            self.images = sorted(os.listdir(data_dir))
        else:
            self.images = list(images)

        self.labels = None if labels is None else list(labels)
        if self.labels is not None:
            assert len(self.images) == len(self.labels), "images/labels length mismatch"

    def __getitem__(self, idx):
        filename = self.images[idx]
        img_path = os.path.join(self.root, filename)
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        if self.labels is None:
            return img, filename
        if self.return_filename:
            return img, int(self.labels[idx]), filename
        return img, int(self.labels[idx])

    def __len__(self):
        return len(self.images)




## === cell 3
base_transforms = weights.transforms()

train_transforms = v2.Compose(
    [
        v2.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
        v2.RandomHorizontalFlip(p=0.5),
        base_transforms,
    ]
)
val_transforms = base_transforms
test_transforms = base_transforms

train_df = pd.read_csv(train_csv_path)

from sklearn.model_selection import train_test_split

tr_df, va_df = train_test_split(
    train_df,
    test_size=0.1,
    random_state=3407,
    stratify=train_df["label"],
)

train_dataset = CassavaDataset(
    train_dir,
    transform=train_transforms,
    images=tr_df["image_id"].tolist(),
    labels=tr_df["label"].tolist(),
)
val_dataset = CassavaDataset(
    train_dir,
    transform=val_transforms,
    images=va_df["image_id"].tolist(),
    labels=va_df["label"].tolist(),
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
)



## === cell 4
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

epochs = 3


def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = model(xb)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == yb).sum().item()
            total += yb.numel()
    return correct / max(1, total)


scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

for ep in range(epochs):
    model.train()
    running_loss = 0.0
    for xb, yb in train_loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            logits = model(xb)
            loss = criterion(logits, yb)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * xb.size(0)

    train_loss = running_loss / len(train_loader.dataset)
    val_acc = evaluate(model, val_loader)
    print(f"epoch={ep+1}/{epochs} train_loss={train_loss:.4f} val_acc={val_acc:.4f}")

model.eval()



## === cell 5
sample_df = pd.read_csv(sample_sub_path)
test_image_ids = sample_df["image_id"].tolist()

test_dataset = CassavaDataset(
    test_dir, transform=test_transforms, images=test_image_ids
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
)



## === cell 6
all_names = []
all_preds = []

model.eval()
with torch.no_grad():
    for batch_idx, (inputs, filenames) in enumerate(test_loader):
        inputs = inputs.to(device, non_blocking=True)
        logits = model(inputs)
        probs = normalizer(logits)
        pred_labels = torch.argmax(probs, dim=1).tolist()

        all_names.extend(list(filenames))
        all_preds.extend(pred_labels)

if len(all_names) != len(test_image_ids):
    raise RuntimeError(
        f"Pred count mismatch: {len(all_names)} vs {len(test_image_ids)}"
    )

pred_map = dict(zip(all_names, all_preds))

missing = set(test_image_ids) - set(pred_map.keys())
extra = set(pred_map.keys()) - set(test_image_ids)
if missing or extra:
    raise RuntimeError(
        f"Filename mismatch vs sample_submission. missing={len(missing)} extra={len(extra)}"
    )

submission = sample_df.copy()
submission["label"] = submission["image_id"].map(pred_map)

if submission["label"].isna().any():
    na_rows = submission[submission["label"].isna()].head(10)
    raise RuntimeError(f"Some labels are missing after mapping. Examples:\n{na_rows}")

submission["label"] = submission["label"].astype(int)

submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
