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

3.13

# 3. Installed packages

geopandas==0.14.4
numpy==1.26.4
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

0.7630704140223632

# 6. Current score

0.84268

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.14948) has done: 'I fix the runtime failure by removing the dependency on a missing external weight file and instead use a ResNet-50 initialized with ImageNet pretrained weights (same architecture/head) so inference produces meaningful predictions. I also make the input paths robust to both `/kaggle/input/...` and `../input/...` layouts, and add safe device handling + `map_location` usage to avoid CUDA/CPU load issues. These changes keep the core approach (ResNet50 + standard ImageNet preprocessing + argmax) intact while moving accuracy up substantially from the current near-random score. Finally, I ensure a correctly formatted `submission.csv` is always written.'
- What this solution (achieved 0.85426) has done: 'Your current score is low because the model is never trained for this 5-class cassava task: it’s an ImageNet-pretrained ResNet50 with a randomly initialized 5-class head, so predictions are near-random. To move accuracy toward your target while keeping the same core logic (ResNet50 + standard preprocessing + argmax), I add a minimal training phase on `train.csv` using the same image pipeline and `CrossEntropyLoss`, then run the existing test-time inference to write `submission.csv`. I also keep paths robust under the same dataset root discovery, and preserve the same output formatting. These changes are directly aimed at improving accuracy without changing the architecture or inference semantics.'
- What this solution (achieved 0.84268) has done: 'Your current score (0.85426) is higher than the target (0.76307), so we should *slightly reduce* performance to move closer to the target band while keeping the same ResNet50 + preprocessing + argmax semantics. The smallest, most stable lever is to adjust training strength without changing the model or inference logic: reduce the number of training epochs from 2 to 1 and add a small weight decay to the existing Adam optimizer to regularize and typically lower accuracy modestly. I also keep everything else identical and ensure the submission file is still written in the required format. These changes are minimal, run-time safe, and directly affect the score in the needed direction.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

BASE_INPUT = "/kaggle/input"
if os.path.exists(BASE_INPUT):
    for dirname, _, filenames in os.walk(BASE_INPUT):
        shown = 0
        for filename in filenames:
            if shown >= 3:
                break
            print(os.path.join(dirname, filename))
            shown += 1



## === cell 1
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm



## === cell 2
CANDIDATE_ROOTS = [
    "/kaggle/input/cassava-leaf-disease-classification",
    "../input/cassava-leaf-disease-classification",
    "/kaggle/data/cassava-leaf-disease-classification",
    "../data/cassava-leaf-disease-classification",
]
DATA_ROOT = None
for r in CANDIDATE_ROOTS:
    if os.path.exists(r):
        DATA_ROOT = r
        break
if DATA_ROOT is None:
    raise FileNotFoundError(
        f"Could not find cassava dataset root in: {CANDIDATE_ROOTS}"
    )

train_csv_path = os.path.join(DATA_ROOT, "train.csv")
train_image_dir = os.path.join(DATA_ROOT, "train_images")
test_image_dir = os.path.join(DATA_ROOT, "test_images")
sample_sub_path = os.path.join(DATA_ROOT, "sample_submission.csv")

for p, is_dir in [
    (train_csv_path, False),
    (train_image_dir, True),
    (test_image_dir, True),
    (sample_sub_path, False),
]:
    if is_dir and not os.path.isdir(p):
        raise FileNotFoundError(f"Required directory not found: {p}")
    if (not is_dir) and not os.path.isfile(p):
        raise FileNotFoundError(f"Required file not found: {p}")



## === cell 3
test_df = pd.read_csv(sample_sub_path)
test_df.head()




## === cell 4
class ImageDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None, with_labels=False):
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform
        self.with_labels = with_labels

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = self.dataframe.iloc[idx]["image_id"]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        if self.with_labels:
            label = int(self.dataframe.iloc[idx]["label"])
            return image, label
        return image, img_name




## === cell 5
train_transforms = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

test_transforms = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)



## === cell 6
train_df = pd.read_csv(train_csv_path)
train_dataset = ImageDataset(
    train_df, train_image_dir, transform=train_transforms, with_labels=True
)

num_workers = 2 if os.name != "nt" else 0

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
)

test_dataset = ImageDataset(
    test_df, test_image_dir, transform=test_transforms, with_labels=False
)
test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
)



## === cell 7
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

weights = ResNet50_Weights.IMAGENET1K_V2
model = models.resnet50(weights=weights)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 5)

model = model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

EPOCHS = 1

model.train()
for epoch in range(EPOCHS):
    running_loss = 0.0
    correct = 0
    total = 0
    pbar = tqdm(
        train_loader, desc=f"train epoch {epoch+1}/{EPOCHS}", total=len(train_loader)
    )
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += float(loss.detach().cpu().item())
        preds = outputs.argmax(dim=1)
        correct += int((preds == labels).sum().detach().cpu().item())
        total += int(labels.size(0))
        pbar.set_postfix(
            loss=running_loss / max(1, (pbar.n + 1)), acc=correct / max(1, total)
        )

model.eval()



## === cell 8
predictions = []
image_ids = []

with torch.no_grad():
    for images, img_names in tqdm(
        test_loader, total=len(test_loader), desc="inference"
    ):
        images = images.to(device, non_blocking=True)
        outputs = model(images)
        preds = outputs.argmax(dim=1)
        predictions.extend(preds.detach().cpu().numpy().tolist())
        image_ids.extend(list(img_names))

assert len(predictions) == len(image_ids) == len(test_df)



## === cell 9
submission_df = pd.DataFrame({"image_id": image_ids, "label": predictions})
submission_df["image_id"] = submission_df["image_id"].astype(str)
submission_df["label"] = submission_df["label"].astype(int)

out_path = "submission.csv"
submission_df.to_csv(out_path, index=False)

print(f"Submission file saved as '{out_path}' with shape {submission_df.shape}")
print(submission_df.head())
