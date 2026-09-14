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

3.13

# 3. Installed packages

albumentations==2.0.8
geopandas==0.14.4
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

0.8785131459655485

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np, pandas as pd, os, torch, torch.nn as nn, torch.optim as optim, torch.nn.functional as F
from torchvision import models
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import cv2, albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True




## === cell 1
train_csv_path = "/kaggle/input/cassava-leaf-disease-classification/train.csv"
test_csv_path = (
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)
train_img_dir = "/kaggle/input/cassava-leaf-disease-classification/train_images"
test_img_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images"




## === cell 2
train_df = pd.read_csv(train_csv_path)
test_df = pd.read_csv(test_csv_path)




## === cell 3
train_transform = A.Compose(
    [
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
        A.Resize(384, 384, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(0.1, 0.1, 20, p=0.7),
        A.HueSaturationValue(10, 15, 10, p=0.5),
        A.RandomBrightnessContrast(0.2, 0.2, p=0.5),
        A.GaussNoise((10.0, 50.0), p=0.4),
        A.Blur(blur_limit=3, p=0.3),
        A.CoarseDropout(8, 20, 20, 1, 10, 10, 0, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)
val_transform = A.Compose(
    [
        A.Resize(384, 384, p=1.0),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)
tta_transform = A.Compose(
    [
        A.ShiftScaleRotate(0.1, 0.1, 20, p=0.7),
        A.HueSaturationValue(10, 15, 10, p=0.5),
        A.RandomBrightnessContrast(0.2, 0.2, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.GaussNoise((10.0, 50.0), p=0.4),
        A.Blur(blur_limit=3, p=0.3),
        A.CoarseDropout(8, 20, 20, 1, 10, 10, 0, p=0.5),
        A.Resize(384, 384, p=1.0),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
ValidationError                           Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/albumentations/core/validation.py in _validate_parameters(schema_cls, full_kwargs, param_names, strict)
     66             schema_kwargs["strict"] = strict
---> 67             config = schema_cls(**schema_kwargs)
     68             validated_kwargs = config.model_dump()

/usr/local/lib/python3.11/dist-packages/pydantic/main.py in __init__(self, **data)
    249         __tracebackhide__ = True
--> 250         validated_self = self.__pydantic_validator__.validate_python(data, self_instance=self)
    251         if self is not validated_self:

ValidationError: 1 validation error for InitSchema
std_range
  Value error, All values in (10.0, 50.0) must be >= 0 and <= 1 [type=value_error, input_value=(10.0, 50.0), input_type=tuple]
    For further information visit https://errors.pydantic.dev/2.12/v/value_error

The above exception was the direct cause of the following exception:

ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/1839238059.py in <cell line: 0>()
      7         A.HueSaturationValue(10, 15, 10, p=0.5),
      8         A.RandomBrightnessContrast(0.2, 0.2, p=0.5),
----> 9         A.GaussNoise((10.0, 50.0), p=0.4),
     10         A.Blur(blur_limit=3, p=0.3),
     11         A.CoarseDropout(8, 20, 20, 1, 10, 10, 0, p=0.5),

/usr/local/lib/python3.11/dist-packages/albumentations/core/validation.py in custom_init(self, *args, **kwargs)
    103                 full_kwargs, param_names, strict = cls._process_init_parameters(original_init, args, kwargs)
    104 
--> 105                 validated_kwargs = cls._validate_parameters(
    106                     dct["InitSchema"],
    107                     full_kwargs,

/usr/local/lib/python3.11/dist-packages/albumentations/core/validation.py in _validate_parameters(schema_cls, full_kwargs, param_names, strict)
     69             validated_kwargs.pop("strict", None)
     70         except ValidationError as e:
---> 71             raise ValueError(str(e)) from e
     72         except Exception as e:
     73             if strict:

ValueError: 1 validation error for InitSchema
std_range
  Value error, All values in (10.0, 50.0) must be >= 0 and <= 1 [type=value_error, input_value=(10.0, 50.0), input_type=tuple]
    For further information visit https://errors.pydantic.dev/2.12/v/value_error

## === cell 4
class CassavaDataset(Dataset):
    """Pre‑load and resize all training images once to avoid per‑epoch disk I/O."""

    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.images = []
        for img_name in self.df["image_id"]:
            img_path = os.path.join(self.img_dir, img_name)
            img = cv2.imread(img_path)
            if img is None:
                img = np.zeros((384, 384, 3), dtype=np.uint8)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (384, 384), interpolation=cv2.INTER_LINEAR)
            self.images.append(img)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = int(self.df.iloc[idx]["label"])
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, label


class CassavaTestDataset(Dataset):
    """Pre‑load all test images into memory once for fast TTA."""

    def __init__(self, df, img_dir):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.images = []
        for img_name in self.df["image_id"]:
            img_path = os.path.join(self.img_dir, img_name)
            img = cv2.imread(img_path)
            if img is None:
                img = np.zeros((384, 384, 3), dtype=np.uint8)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.images.append(img)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.images[idx], self.df.iloc[idx]["image_id"]




## === cell 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




## === cell 6
train_split, val_split = train_test_split(
    train_df, test_size=0.1, stratify=train_df["label"], random_state=42
)
train_dataset = CassavaDataset(train_split, train_img_dir, transform=train_transform)
val_dataset = CassavaDataset(val_split, train_img_dir, transform=val_transform)

num_workers = min(os.cpu_count() or 4, 4)

train_loader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=128,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
)




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/901718751.py in <cell line: 0>()
      2     train_df, test_size=0.1, stratify=train_df["label"], random_state=42
      3 )
----> 4 train_dataset = CassavaDataset(train_split, train_img_dir, transform=train_transform)
      5 val_dataset = CassavaDataset(val_split, train_img_dir, transform=val_transform)
      6 

NameError: name 'train_transform' is not defined

## === cell 7
model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
num_features = model.classifier[1].in_features
model.classifier = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(num_features, 5))
model = model.to(device)
model = torch.compile(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scaler = GradScaler()




## === cell 8
def train_one_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    running_loss = 0.0
    for imgs, targets in loader:  # tqdm removed for speed
        imgs, targets = imgs.to(device, non_blocking=True), targets.to(
            device, non_blocking=True
        )
        optimizer.zero_grad()
        with autocast():
            outputs = model(imgs)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item() * imgs.size(0)
    return running_loss / len(loader.dataset)


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.inference_mode():
        for imgs, targets in loader:  # tqdm removed
            imgs, targets = imgs.to(device, non_blocking=True), targets.to(
                device, non_blocking=True
            )
            with autocast():
                outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    return correct / total




## === cell 9
epochs = 5
for epoch in range(epochs):
    train_loss = train_one_epoch(
        model, train_loader, optimizer, criterion, device, scaler
    )
    val_acc = evaluate(model, val_loader, device)
    print(
        f"Epoch {epoch+1}/{epochs} - Train loss: {train_loss:.4f} - Val acc: {val_acc:.4f}"
    )




## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1405283717.py in <cell line: 0>()
      2 for epoch in range(epochs):
      3     train_loss = train_one_epoch(
----> 4         model, train_loader, optimizer, criterion, device, scaler
      5     )
      6     val_acc = evaluate(model, val_loader, device)

NameError: name 'train_loader' is not defined

## === cell 10
def tta_predict_batch_batch(model, images_np, tta_transform, device, n_tta=5):
    """
    images_np: (B, H, W, C) uint8
    Returns mean probability tensor of shape (B, 5)
    """
    model.eval()
    B = images_np.shape[0]
    probs_sum = torch.zeros((B, 5), device=device, dtype=torch.float32)
    with torch.inference_mode():
        for _ in range(n_tta):
            aug_batch = [tta_transform(image=images_np[i])["image"] for i in range(B)]
            batch_tensor = torch.stack(aug_batch).to(device, non_blocking=True)
            with autocast():
                out = model(batch_tensor)
            probs = F.softmax(out, dim=1)
            probs_sum += probs
    return probs_sum / n_tta




## === cell 11
test_dataset = CassavaTestDataset(test_df, test_img_dir)
test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
)




## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/491860068.py in <cell line: 0>()
      4     batch_size=64,
      5     shuffle=False,
----> 6     num_workers=num_workers,
      7     pin_memory=True,
      8     persistent_workers=True,

NameError: name 'num_workers' is not defined

## === cell 12
ensemble_predictions, image_names = [], []
with torch.no_grad():
    for batch_imgs, batch_names in test_loader:
        batch_np = np.stack(batch_imgs)  # (B, H, W, C)
        probs = tta_predict_batch_batch(model, batch_np, tta_transform, device, n_tta=5)
        preds = probs.argmax(dim=1).cpu().tolist()
        ensemble_predictions.extend(preds)
        image_names.extend(batch_names)




## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1374191989.py in <cell line: 0>()
      1 ensemble_predictions, image_names = [], []
      2 with torch.no_grad():
----> 3     for batch_imgs, batch_names in test_loader:
      4         batch_np = np.stack(batch_imgs)  # (B, H, W, C)
      5         probs = tta_predict_batch_batch(model, batch_np, tta_transform, device, n_tta=5)

NameError: name 'test_loader' is not defined

## === cell 13
submission_df = pd.DataFrame({"image_id": image_names, "label": ensemble_predictions})
submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)
print(f"Submission file saved as '{submission_path}' with {len(submission_df)} rows.")

## --- ERROR in outputing the csv:
Invalid submission: Submission must have the same length as the answers.
