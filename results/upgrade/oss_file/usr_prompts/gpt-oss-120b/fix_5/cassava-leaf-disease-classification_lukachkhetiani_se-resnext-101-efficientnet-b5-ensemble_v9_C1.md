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

# 5. Code solution

## === cell 0
import os, sys, random, glob, tqdm
import torch, torch.nn as nn, torch.nn.functional as F
import cv2, pandas as pd
from torch.utils.data import Dataset, DataLoader

try:
    from efficientnet_pytorch import EfficientNet
except ModuleNotFoundError:
    EfficientNet = None

import timm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True  # enable fast kernel selection


def try_load(path):
    if os.path.exists(path):
        try:
            return torch.load(path, map_location=device)
        except Exception as e:
            print(f"Could not load {path}: {e}")
    else:
        print(f"Checkpoint {path} not found.")
    return None


base_path = "/kaggle/input/cassava-leaf-disease-classification"
train_csv_path = os.path.join(base_path, "train.csv")
train_img_dir = os.path.join(base_path, "train_images")
test_img_dir = os.path.join(base_path, "test_images")

eff_path = "/kaggle/input/ensemble-2/eff_model_last.pth"
hr_path = "/kaggle/input/ensemble-2/seresnext_model_last.pth"

eff_state = try_load(eff_path)
hr_state = try_load(hr_path)

if eff_state is not None and EfficientNet is not None:
    efficient = EfficientNet.from_name("efficientnet-b5", num_classes=5)
    efficient.load_state_dict(eff_state)
    efficient = efficient.to(device).eval()
else:
    efficient = None

if hr_state is not None:
    hrnet = timm.create_model("seresnext101_32x4d", pretrained=False, num_classes=5)
    hrnet.load_state_dict(hr_state)
    hrnet = hrnet.to(device).eval()
else:
    hrnet = None

if efficient is None or hrnet is None:
    print(
        "Training a fresh EfficientNet-B5 model because checkpoints are unavailable..."
    )

    class CassavaDataset(Dataset):
        def __init__(self, df, img_dir, transform=None):
            self.df = df.reset_index(drop=True)
            self.img_dir = img_dir
            self.transform = transform

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            img_path = os.path.join(self.img_dir, row["image_id"])
            img = cv2.imread(img_path)
            img = cv2.resize(img, (299, 299))
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = clahe.apply(l)
            lab = cv2.merge((l, a, b))
            img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            tensor = torch.tensor(img.transpose(2, 0, 1), dtype=torch.float) / 255.0
            label = int(row["label"])
            return tensor, label

    full_train_df = pd.read_csv(train_csv_path)

    from sklearn.model_selection import train_test_split

    train_df, val_df = train_test_split(
        full_train_df,
        test_size=0.2,
        stratify=full_train_df["label"],
        random_state=42,
    )

    num_workers = min(4, os.cpu_count() or 2)
    train_dataset = CassavaDataset(train_df, train_img_dir)
    val_dataset = CassavaDataset(val_df, train_img_dir)

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    efficient = timm.create_model("efficientnet_b5", pretrained=True, num_classes=5)
    efficient = efficient.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(efficient.parameters(), lr=1e-4)

    best_val_acc = 0.0
    for epoch in range(5):  # a modest increase for better accuracy
        efficient.train()
        for imgs, targets in tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/5 [train]"):
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs = efficient(imgs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        efficient.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, targets in tqdm.tqdm(val_loader, desc=f"Epoch {epoch+1}/5 [val]"):
                imgs = imgs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                outputs = efficient(imgs)
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)
        val_acc = correct / total
        print(f"Validation accuracy after epoch {epoch+1}: {val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(efficient.state_dict(), "eff_fresh.pth")

    efficient.load_state_dict(torch.load("eff_fresh.pth", map_location=device))
    efficient.eval()
    hrnet = efficient  # use the same model for both ensemble members
    print("Training completed; both ensemble members are now ready.")
else:
    print("Both pretrained checkpoints loaded successfully.")



## === cell 1
files = glob.glob(os.path.join(test_img_dir, "**", "*.jpg"), recursive=True)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def process(image):
    img = cv2.resize(image, (299, 299))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    tensor = torch.tensor(img.transpose(2, 0, 1), dtype=torch.float).to(
        device
    )  # C,H,W on device
    return tensor




## === cell 2
batch_size = 32
names, labels = [], []

batch_tensors = []
batch_names = []

for file in tqdm.tqdm(files, desc="Predicting"):
    img = cv2.imread(file)
    if img is None:
        continue  # skip unreadable files
    tensor = process(img)  # C,H,W on device
    batch_tensors.append(tensor)
    batch_names.append(os.path.basename(file))

    if len(batch_tensors) == batch_size:
        batch = torch.stack(batch_tensors)  # B,C,H,W
        with torch.no_grad():
            hr_out = hrnet(batch)
            eff_out = efficient(batch)
            total = (hr_out + eff_out) / 2
            probs = F.softmax(total, dim=1)
            preds = torch.argmax(probs, dim=1).cpu().numpy()
        names.extend(batch_names)
        labels.extend(preds.tolist())
        batch_tensors, batch_names = [], []

if batch_tensors:
    batch = torch.stack(batch_tensors)
    with torch.no_grad():
        hr_out = hrnet(batch)
        eff_out = efficient(batch)
        total = (hr_out + eff_out) / 2
        probs = F.softmax(total, dim=1)
        preds = torch.argmax(probs, dim=1).cpu().numpy()
    names.extend(batch_names)
    labels.extend(preds.tolist())

print(f"Predicted {len(names)} images.")



## === cell 3
submission = pd.DataFrame({"image_id": names, "label": labels})
submission_path = "/kaggle/working/submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission file saved as {submission_path} with {len(submission)} rows.")
