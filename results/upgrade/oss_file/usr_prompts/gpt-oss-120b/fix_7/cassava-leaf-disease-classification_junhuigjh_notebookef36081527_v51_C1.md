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

0.8566032033847084

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.05531) has done: 'I fix the path detection for the test images and safely convert the list of probability vectors into a NumPy array. This prevents the “need at least one array to stack” error and ensures the subsequent slicing works, allowing the script to finish and write a proper `submission.csv`.'
- What this solution (achieved 0.05531) has done: 'I replace the dummy TensorFlow model with the real PyTorch model (which is likely pre‑trained) and skip the unused TF predictions. By applying a softmax to the PyTorch output and using those probabilities directly for the final arg‑max, the predictions become meaningful instead of uniform, which should raise the accuracy toward the target score.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_or_initialize_model():
    pt_path = (
        "/kaggle/input/resnet50_70_512x512/pytorch/default/1/Resnet50_70_512x512.pth"
    )
    try:
        model = torch.load(pt_path, map_location=device)
        model.to(device)
        model.eval()
        print("Loaded pretrained checkpoint.")
        return model
    except Exception as e:
        print(f"Checkpoint not found or load failed ({e}); initializing new ResNet‑50.")
        backbone = models.resnet50(pretrained=True)
        backbone.fc = torch.nn.Linear(backbone.fc.in_features, 5)
        backbone = backbone.to(device)
        backbone.train()
        return backbone


model2 = load_or_initialize_model()

possible_train_csv_paths = [
    Path("/kaggle/input/cassava-leaf-disease-classification/train.csv"),
    Path("./train.csv"),
    Path("./input/cassava-leaf-disease-classification/train.csv"),
    Path("/kaggle/input/train.csv"),
]
train_csv_path = next((p for p in possible_train_csv_paths if p.is_file()), None)
if train_csv_path is None:
    raise RuntimeError("train.csv not found in any of the expected locations.")

train_df = pd.read_csv(train_csv_path)
train_df["label"] = train_df["label"].astype(int)

possible_train_dirs = [
    Path("/kaggle/input/cassava-leaf-disease-classification/train_images"),
    Path("./train_images"),
    Path("./input/cassava-leaf-disease-classification/train_images"),
    Path("./kaggle/input/cassava-leaf-disease-classification/train_images"),
]
train_dir = next((p for p in possible_train_dirs if p.is_dir()), None)
if train_dir is None:
    raise RuntimeError("Train images directory not found.")


class CassavaDataset(Dataset):
    def __init__(self, df, img_dir, transform):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.img_dir / row["image_id"]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        label = row["label"]
        return img, label


torch_transforms = transforms.Compose(
    [
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)

if isinstance(model2, torch.nn.Module) and not any(p for p in model2.parameters()):
    pass

if not isinstance(model2, torch.nn.Module) or not hasattr(model2, "fc"):
    train_dataset = CassavaDataset(train_df, train_dir, torch_transforms)
    train_loader = DataLoader(
        train_dataset, batch_size=64, shuffle=True, num_workers=2, pin_memory=True
    )

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model2.parameters(), lr=1e-4)

    model2.train()
    for epoch in range(1):  # single epoch to keep runtime low
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model2(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/1 - Loss: {epoch_loss:.4f}")

    model2.eval()

possible_test_dirs = [
    Path("/kaggle/input/cassava-leaf-disease-classification/test_images"),
    Path("./test_images"),
    Path("./input/cassava-leaf-disease-classification/test_images"),
    Path("./kaggle/input/cassava-leaf-disease-classification/test_images"),
]
test_dir = next((p for p in possible_test_dirs if p.is_dir()), None)
if test_dir is None:
    raise RuntimeError("Test images directory not found.")

test_images = sorted([f.name for f in test_dir.iterdir() if f.is_file()])

image_ids = []
combined_probs = []

for idx, img_name in enumerate(test_images, start=1):
    image_ids.append(img_name)
    img_path = test_dir / img_name
    pil_img = Image.open(img_path).convert("RGB")

    img_torch = torch_transforms(pil_img).unsqueeze(0).to(device)  # (1, 3, 512, 512)
    with torch.no_grad():
        pred_pt = model2(img_torch)  # (1, 5)
    pred_pt = pred_pt.squeeze(0)  # (5,)
    pred_pt = F.softmax(pred_pt, dim=0)  # probabilities
    combined_probs.append(pred_pt.cpu().numpy())

    if idx % 100 == 0 or idx == len(test_images):
        print(f"Processed {idx}/{len(test_images)} images", end="\r")

if not combined_probs:
    raise RuntimeError("No test images were processed; check the data path.")
combined_probs = np.array(combined_probs)  # shape (N, 5)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/1747710053.py in <cell line: 0>()
     90 
     91 # If the loaded model lacks parameters (unlikely), skip training.
---> 92 if isinstance(model2, torch.nn.Module) and not any(p for p in model2.parameters()):
     93     pass
     94 

RuntimeError: Boolean value of Tensor with more than one value is ambiguous

## === cell 1
prob_pt = combined_probs  # shape (N, 5)
prediction = np.argmax(prob_pt, axis=1)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4181920427.py in <cell line: 0>()
----> 1 prob_pt = combined_probs  # shape (N, 5)
      2 prediction = np.argmax(prob_pt, axis=1)
      3 

NameError: name 'combined_probs' is not defined

## === cell 2
submission = pd.DataFrame({"image_id": image_ids, "label": prediction})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"\nSubmission saved to {submission_path}")



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/33628752.py in <cell line: 0>()
----> 1 submission = pd.DataFrame({"image_id": image_ids, "label": prediction})
      2 submission_path = "submission.csv"
      3 submission.to_csv(submission_path, index=False)
      4 print(f"\nSubmission saved to {submission_path}")
      5 

NameError: name 'image_ids' is not defined

## === cell 3
submission.head()

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3365464162.py in <cell line: 0>()
----> 1 submission.head()

NameError: name 'submission' is not defined
