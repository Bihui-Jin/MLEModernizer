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
pillow==11.3.0
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

# 5. Target score

0.8482925355092172

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.22384) has done: 'Implemented fixes to resolve import errors, correct EfficientNet classifier handling, and make weight loading robust. Added a safe dummy `SummaryWriter`, adjusted model creation to fetch `in_features` correctly, default to pretrained ImageNet weights when no custom weights are found, and ensured the script always defines `model` before inference. The pipeline now runs end‑to‑end and writes a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os, random, json, glob
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
from torchvision import transforms as T
from PIL import Image

SEED = 42
N_FOLDS = 5
N_EPOCHS = 5  # modest fine‑tuning
BATCH_SIZE = 16
IMG_SIZE = 224
LR = 5e-4
NUM_CLASSES = 5
TRAINING = True  # enable quick training to improve score
WEIGHT_FILE = "/kaggle/input/ramki-cassava-weights/weight-at-epoch-14-acc-0.85109.pth"


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


seed_everything(SEED)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:", device)




## === cell 1
def create_model(pretrained: bool = True) -> nn.Module:
    """
    Builds an EfficientNet‑B0 model adapted for the Cassava 5‑class task.
    """
    try:
        model = torchvision.models.efficientnet_b0(
            weights="IMAGENET1K_V1" if pretrained else None
        )
    except Exception:
        model = torchvision.models.efficientnet_b0(pretrained=pretrained)

    if hasattr(model, "classifier"):
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)
    else:
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, NUM_CLASSES)

    return model




## === cell 2
model = create_model(pretrained=True)

if os.path.exists(WEIGHT_FILE):
    try:
        state_dict = torch.load(WEIGHT_FILE, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded custom weights from {WEIGHT_FILE}")
    except Exception as e:
        print(
            f"Warning: failed to load weights ({e}); using ImageNet pretrained model."
        )
else:
    alt_path = "/kaggle/input/ramki-cassava-weights"
    if os.path.isdir(alt_path):
        possible = [
            os.path.join(alt_path, f)
            for f in os.listdir(alt_path)
            if f.lower().endswith(".pth")
        ]
        if possible:
            try:
                state_dict = torch.load(possible[0], map_location=device)
                model.load_state_dict(state_dict, strict=False)
                print(f"Loaded custom weights from {possible[0]}")
            except Exception as e2:
                print(
                    f"Warning: failed to load alternative weights ({e2}); using ImageNet pretrained model."
                )
        else:
            print(
                f"Warning: no .pth files found in {alt_path}; using ImageNet pretrained model."
            )
    else:
        print(
            f"Warning: weight directory not found at {alt_path}; using ImageNet pretrained model."
        )

model.to(device)
print("Model ready – training mode:", TRAINING)




## === cell 3
if TRAINING:
    class TrainImageDataset(Dataset):
        def __init__(self, df, img_root, transform=None):
            self.df = df.reset_index(drop=True)
            self.img_root = img_root
            self.transform = transform

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            img_path = os.path.join(self.img_root, row["image_id"])
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            label = int(row["label"])
            return image, label

    train_csv_path = glob.glob("/kaggle/input/**/train.csv", recursive=True)
    if not train_csv_path:
        raise RuntimeError("train.csv not found")
    train_df = pd.read_csv(train_csv_path[0])
    img_root_candidates = [
        "/kaggle/input/**/train_images",
        "/kaggle/input/**/train_images/",
        "/kaggle/input/**/train_images/",
        "/kaggle/input/**/train_images/",
        "/kaggle/input/**/train_images/",
        "/kaggle/input/**/train_images/",
        "/kaggle/input/**/train_images/",
        "/kaggle/input/**/train_images/",
        "/kaggle/input/**/train_images/",
        "/kaggle/input/**/train_images/",
    ]
    img_root = None
    for pat in img_root_candidates:
        matches = glob.glob(pat, recursive=True)
        if matches:
            img_root = matches[0]
            break
    if img_root is None:
        raise RuntimeError("train_images directory not found")
    print("Training images root:", img_root)

    train_len = int(0.9 * len(train_df))
    val_len = len(train_df) - train_len
    train_subset, val_subset = random_split(
        train_df, [train_len, val_len], generator=torch.Generator().manual_seed(SEED)
    )

    train_tf = T.Compose(
        [
            T.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_tf = T.Compose(
        [
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = TrainImageDataset(train_subset, img_root, transform=train_tf)
    val_dataset = TrainImageDataset(val_subset, img_root, transform=val_tf)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    model.train()
    for epoch in range(1, N_EPOCHS + 1):
        epoch_loss = 0.0
        for imgs, labels in tqdm(
            train_loader, desc=f"Epoch {epoch}/{N_EPOCHS} - Training"
        ):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * imgs.size(0)

        avg_loss = epoch_loss / len(train_loader.dataset)
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in tqdm(
                val_loader, desc=f"Epoch {epoch}/{N_EPOCHS} - Validation"
            ):
                imgs = imgs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                logits = model(imgs)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / total
        print(f"Epoch {epoch} – Train loss: {avg_loss:.4f} – Val Acc: {val_acc:.4f}")
        model.train()  # back to training mode after validation

    model.eval()
else:
    model.eval()




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_56/200592650.py in <cell line: 0>()
     72     )
     73 
---> 74     train_dataset = TrainImageDataset(train_subset, img_root, transform=train_tf)
     75     val_dataset = TrainImageDataset(val_subset, img_root, transform=val_tf)
     76 

/tmp/ipykernel_56/200592650.py in __init__(self, df, img_root, transform)
      3     class TrainImageDataset(Dataset):
      4         def __init__(self, df, img_root, transform=None):
----> 5             self.df = df.reset_index(drop=True)
      6             self.img_root = img_root
      7             self.transform = transform

AttributeError: 'Subset' object has no attribute 'reset_index'

## === cell 4
class TestImageDataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, os.path.basename(img_path)


test_image_paths = glob.glob("/kaggle/input/**/test_images/*.jpg", recursive=True)
if not test_image_paths:
    raise RuntimeError("No test images found under /kaggle/input/**/test_images/")

test_transforms = T.Compose(
    [
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

test_dataset = TestImageDataset(test_image_paths, transform=test_transforms)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
)




## === cell 5
preds = []
ids = []

with torch.no_grad():
    for imgs, img_ids in tqdm(test_loader, desc="Predicting"):
        imgs = imgs.to(device, non_blocking=True)

        logits_orig = model(imgs)

        imgs_hflip = torch.flip(imgs, dims=[3])  # width dimension
        logits_hflip = model(imgs_hflip)

        imgs_vflip = torch.flip(imgs, dims=[2])  # height dimension
        logits_vflip = model(imgs_vflip)

        logits_avg = (logits_orig + logits_hflip + logits_vflip) / 3.0

        batch_preds = torch.argmax(logits_avg, dim=1).cpu().numpy()
        preds.extend(batch_preds.tolist())
        ids.extend(img_ids)

submission = pd.DataFrame({"image_id": ids, "label": preds})
submission = submission.sort_values("image_id").reset_index(drop=True)

output_path = "submission.csv"
submission.to_csv(output_path, index=False)
print(f"Submission written to {output_path} ({submission.shape[0]} rows)")




## === cell 6
sample_path = glob.glob("/kaggle/input/**/sample_submission.csv", recursive=True)
if sample_path:
    sample = pd.read_csv(sample_path[0])
    assert list(sample.columns) == [
        "image_id",
        "label",
    ], "Sample submission columns mismatch"
    print("Sample submission format verified.")
else:
    print("Sample submission file not found; assuming column order is correct.")

## --- ERROR in outputing the csv:
Invalid submission: Submission must have the same length as the answers.
