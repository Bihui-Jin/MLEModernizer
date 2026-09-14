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

# 5. Code solution

## === cell 0
import numpy as np, pandas as pd, os, cv2, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.models import EfficientNet_V2_S_Weights, MobileNet_V3_Large_Weights
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from torch.cuda.amp import autocast, GradScaler

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")



## === cell 1
num_tta = 5



## === cell 2
test_image_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images"
test_df = pd.read_csv(
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)
train_df = pd.read_csv("/kaggle/input/cassava-leaf-disease-classification/train.csv")



## === cell 3
efficientnet_transforms = A.Compose(
    [
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
        A.Resize(384, 384),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)




## === cell 4
def list_collate(batch):
    imgs, names = zip(*batch)
    return list(imgs), list(names)




## === cell 5
class CassavaTestDataset(Dataset):
    def __init__(self, dataframe, image_dir):
        self.image_names = dataframe["image_id"].values
        self.dir = image_dir

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.dir, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img, img_name




## === cell 6
common_transforms = [
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=20, p=0.7),
    A.HueSaturationValue(
        hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.5
    ),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.HorizontalFlip(p=0.5),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.4),
]

tta_transform_efficientnet = A.Compose(
    common_transforms
    + [
        A.RandomResizedCrop(size=(384, 384), scale=(0.8, 1.0), p=1.0),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)

tta_transform_mobilenet = A.Compose(
    common_transforms
    + [
        A.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0), p=1.0),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)




## === cell 7
def tta_predict_single_model(model, image, tta_transform, device, n_tta=5):
    """
    Apply TTA by creating a batch of n_tta augmented versions, run a single forward
    pass, and average the softmax probabilities.
    """
    model.eval()
    aug_batch = []
    for _ in range(n_tta):
        aug = tta_transform(image=image)["image"]
        aug_batch.append(aug)
    aug_batch = torch.stack(aug_batch).to(device)

    with torch.no_grad():
        if device.type == "cuda":
            with autocast():
                out = model(aug_batch)  # (n_tta, num_classes)
        else:
            out = model(aug_batch)
        probs = F.softmax(out, dim=1)  # (n_tta, num_classes)
    return probs.mean(dim=0, keepdim=True)  # (1, num_classes)




## === cell 8
test_dataset = CassavaTestDataset(test_df, test_image_dir)
test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=4,  # reduced workers to lower overhead
    collate_fn=list_collate,
    pin_memory=True,
    persistent_workers=True,
)



## === cell 9
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")



## === cell 10
efficientnet_model = models.efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
num_ftrs = efficientnet_model.classifier[1].in_features
efficientnet_model.classifier[1] = nn.Linear(num_ftrs, 5)
efficientnet_model = efficientnet_model.to(device)

if hasattr(torch, "compile"):
    efficientnet_model = torch.compile(efficientnet_model)

efficientnet_model.train()  # will be fine‑tuned later

mobile_model = models.mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
num_ftrs = mobile_model.classifier[0].in_features
mobile_model.classifier = nn.Sequential(nn.Linear(num_ftrs, 5))
mobile_model = mobile_model.to(device)

if hasattr(torch, "compile"):
    mobile_model = torch.compile(mobile_model)

mobile_model.eval()  # keep as pretrained




## === cell 11
class CassavaTrainDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform):
        self.image_names = dataframe["image_id"].values
        self.labels = dataframe["label"].astype(int).values
        self.dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        label = int(self.labels[idx])
        img_path = os.path.join(self.dir, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        transformed = self.transform(image=img)
        img_tensor = transformed["image"]
        return img_tensor, torch.tensor(label, dtype=torch.long)


train_image_dir = "/kaggle/input/cassava-leaf-disease-classification/train_images"
train_dataset = CassavaTrainDataset(train_df, train_image_dir, efficientnet_transforms)
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=4,  # reduced workers to lower overhead
    pin_memory=True,
    persistent_workers=True,
)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(efficientnet_model.parameters(), lr=1e-4)
num_epochs = 3

scaler = GradScaler()

efficientnet_model.train()
for epoch in range(num_epochs):
    running_loss = 0.0
    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        with autocast():
            outputs = efficientnet_model(imgs)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * imgs.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch {epoch+1} loss: {epoch_loss:.4f}")

efficientnet_model.eval()  # ready for inference




## === cell 12
def batch_tta_predict(model, images, tta_transform, device, n_tta=5):
    """
    images: list of numpy arrays (H, W, C)
    Returns: tensor of shape (batch_size, num_classes) with averaged probabilities.
    """
    model.eval()
    aug_list = [
        tta_transform(image=img)["image"] for img in images for _ in range(n_tta)
    ]
    batch_tensor = torch.stack(aug_list).to(device)  # (B*n_tta, C, H, W)

    with torch.no_grad():
        if device.type == "cuda":
            with autocast():
                out = model(batch_tensor)
        else:
            out = model(batch_tensor)
        probs = F.softmax(out, dim=1)

    B = len(images)
    probs = probs.view(B, n_tta, -1).mean(dim=1)  # (B, num_classes)
    return probs




## === cell 13
ensemble_predictions = []
image_names = []

with torch.inference_mode():
    for imgs_batch, names_batch in tqdm(test_loader, desc="Inference"):
        probs_efficientnet = batch_tta_predict(
            efficientnet_model,
            imgs_batch,
            tta_transform_efficientnet,
            device,
            n_tta=num_tta,
        )
        probs_mobilenet = batch_tta_predict(
            mobile_model,
            imgs_batch,
            tta_transform_mobilenet,
            device,
            n_tta=num_tta,
        )
        combined = (probs_efficientnet + probs_mobilenet) / 2
        preds = combined.argmax(dim=1).cpu().tolist()
        ensemble_predictions.extend(preds)
        image_names.extend(names_batch)



## === cell 14
submission_df = pd.DataFrame({"image_id": image_names, "label": ensemble_predictions})
submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)
print(f"Submission file saved as '{submission_path}' with {len(submission_df)} rows.")
