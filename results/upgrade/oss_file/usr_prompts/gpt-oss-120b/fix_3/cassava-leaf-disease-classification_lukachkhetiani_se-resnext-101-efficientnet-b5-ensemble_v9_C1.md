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

0.8038682381384104

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, sys, subprocess, random
import torch, torch.nn as nn, torch.nn.functional as F
import cv2, glob, pandas as pd, tqdm
from torch.utils.data import Dataset, DataLoader
from efficientnet_pytorch import EfficientNet
import timm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def try_load(path):
    if os.path.exists(path):
        try:
            return torch.load(path, map_location=device)
        except Exception as e:
            print(f"Could not load {path}: {e}")
    else:
        print(f"Checkpoint {path} not found.")
    return None


eff_path = "../input/ensemble-2/eff_model_last.pth"
hr_path = "../input/ensemble-2/seresnext_model_last.pth"

eff_state = try_load(eff_path)
hr_state = try_load(hr_path)

if eff_state is not None:
    efficient = EfficientNet.from_name("efficientnet-b5", num_classes=5)
    efficient.load_state_dict(eff_state)
    efficient = efficient.to(device).eval()
else:
    efficient = None

if hr_state is not None:
    hrnet = timm.create_model("seresnext101_32x4d", num_classes=5, pretrained=False)
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

    train_csv_path = "../input/cassava-leaf-disease-classification/train.csv"
    train_img_dir = "../input/cassava-leaf-disease-classification/train_images"

    full_train_df = pd.read_csv(train_csv_path)

    from sklearn.model_selection import train_test_split

    train_df, val_df = train_test_split(
        full_train_df,
        test_size=0.2,
        stratify=full_train_df["label"],
        random_state=42,
    )

    train_dataset = CassavaDataset(train_df, train_img_dir)
    val_dataset = CassavaDataset(val_df, train_img_dir)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

    efficient = timm.create_model("efficientnet_b5", pretrained=True, num_classes=5)
    efficient = efficient.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(efficient.parameters(), lr=1e-4)

    best_val_acc = 0.0
    for epoch in range(3):  # a few epochs are enough for a decent baseline
        efficient.train()
        for imgs, targets in tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/3 [train]"):
            imgs = imgs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = efficient(imgs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        efficient.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, targets in tqdm.tqdm(val_loader, desc=f"Epoch {epoch+1}/3 [val]"):
                imgs = imgs.to(device)
                targets = targets.to(device)
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

    hrnet = efficient
    print("Training completed; both ensemble members are now ready.")
else:
    print("Both pretrained checkpoints loaded successfully.")



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_55/1450835113.py in <cell line: 0>()
      3 import cv2, glob, pandas as pd, tqdm
      4 from torch.utils.data import Dataset, DataLoader
----> 5 from efficientnet_pytorch import EfficientNet
      6 import timm
      7 

ModuleNotFoundError: No module named 'efficientnet_pytorch'

## === cell 1
files = glob.glob("../input/cassava-leaf-disease-classification/test_images/*")
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def process(image):
    img = cv2.resize(image, (299, 299))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    tensor = (
        torch.tensor(img.transpose(2, 0, 1), dtype=torch.float).unsqueeze(0).to(device)
    )
    return tensor




## === cell 2
names, labels = [], []

for file in tqdm.tqdm(files, desc="Predicting"):
    img = cv2.imread(file)
    topred = process(img)
    with torch.no_grad():
        hr_out = hrnet(topred)
        eff_out = efficient(topred)
        total = (hr_out + eff_out) / 2
        pred = int(torch.argmax(F.softmax(total, dim=1)).cpu().item())
    names.append(os.path.basename(file))
    labels.append(pred)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4069075878.py in <cell line: 0>()
      3 for file in tqdm.tqdm(files, desc="Predicting"):
      4     img = cv2.imread(file)
----> 5     topred = process(img)
      6     with torch.no_grad():
      7         hr_out = hrnet(topred)

/tmp/ipykernel_55/1139094098.py in process(image)
     11     img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
     12     tensor = (
---> 13         torch.tensor(img.transpose(2, 0, 1), dtype=torch.float).unsqueeze(0).to(device)
     14     )
     15     return tensor

NameError: name 'device' is not defined

## === cell 3
submission = pd.DataFrame({"image_id": names, "label": labels})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission file saved as {submission_path} with {len(submission)} rows.")

## --- ERROR in outputing the csv:
Invalid submission: Submission must have the same length as the answers.
