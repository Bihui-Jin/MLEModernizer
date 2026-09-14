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

0.8224539135690541

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.10762) has done: 'I fix the runtime errors that prevent a submission file from being created and keep the model logic unchanged.  
- Guard the pre‑trained weight loading with a file‑existence check and fall back to random initialization if the file is missing.  
- Correct the misuse of `torch.cuda.is_available` (add parentheses) when moving tensors/weights to GPU.  
- Add a short comment explaining each change so the intent (producing a valid CSV and moving the score toward the target) is clear.'

# 9. Code solution

## === cell 0
import os
import torch
import pandas as pd
from copy import deepcopy
from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset
from torchvision import transforms, models
from skimage import io

torch.backends.cudnn.benchmark = True

is_submission = False  # set False to train then create submission
data_path = "/kaggle/input/cassava-leaf-disease-classification"
csv_file_name = "train.csv" if not is_submission else "sample_submission.csv"
train_batch_size, val_batch_size, sub_batch_size = 16, 32, 32
train_rate = 0.7  # proportion for training split
num_workers = min(4, os.cpu_count() or 1)

mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

use_balanced_sample = False
use_class_weight = not use_balanced_sample
mul_weight = torch.tensor([3.0, 1.5, 1.0, 1.0, 4.5])
learning_rate = 5e-4
weight_decay = 2e-5
efficient_net_version = 3  # placeholder, using EfficientNet-B0

train_epoch = (0, 12)  # train for 12 epochs (was 5)
debug = not is_submission

use_pre_trained_weight = True
pre_trained_weight_path = (
    "../input/cassava-leaf-disease-classification-weight/weight.pth"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




## === cell 1
class CassavaDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, "train_images", self.df.iloc[idx, 0])
        img = io.imread(img_name)
        if self.transform:
            img = self.transform(img)
        label = int(self.df.iloc[idx, 1])
        return img, label


train_transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
)

val_transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
)

full_dataset = CassavaDataset(
    csv_file=os.path.join(data_path, "train.csv"),
    root_dir=data_path,
    transform=train_transform,
)

train_len = int(len(full_dataset) * train_rate)
val_len = len(full_dataset) - train_len
train_dataset, val_subset = random_split(
    full_dataset,
    [train_len, val_len],
    generator=torch.Generator().manual_seed(42),
)
val_subset.dataset.transform = val_transform

val_images = torch.stack([val_subset[i][0] for i in range(len(val_subset))])
val_labels = torch.tensor(
    [val_subset[i][1] for i in range(len(val_subset))], dtype=torch.long
)
val_tensor_dataset = TensorDataset(val_images, val_labels)

train_loader = DataLoader(
    train_dataset,
    batch_size=train_batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,
)

val_loader = DataLoader(
    val_tensor_dataset,
    batch_size=val_batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,
)

if not is_submission and use_class_weight:
    all_labels = torch.tensor([label for _, label in train_dataset], dtype=torch.long)
    label_counts = torch.bincount(all_labels, minlength=5).tolist()
    total = sum(label_counts)
    weight_vals = [total / (2 * cnt) if cnt > 0 else 0.0 for cnt in label_counts]
    loss_weight = torch.tensor(weight_vals, dtype=torch.float32) * mul_weight
    loss_weight = loss_weight.to(device)
else:
    loss_weight = None




## === cell 2
model = models.efficientnet_b0(pretrained=True)
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 5)
model = model.to(device)

if use_pre_trained_weight and os.path.isfile(pre_trained_weight_path):
    try:
        state = torch.load(pre_trained_weight_path, map_location=device)
        model.load_state_dict(state)
        print("Loaded external pretrained weights.")
    except Exception as e:
        print(f"Could not load external weights: {e}")

criterion = (
    torch.nn.CrossEntropyLoss(weight=loss_weight)
    if loss_weight is not None
    else torch.nn.CrossEntropyLoss()
)
optimizer = torch.optim.Adam(
    model.parameters(), lr=learning_rate, weight_decay=weight_decay
)

best_accuracy = 0.0
best_state = deepcopy(model.state_dict())

for epoch in range(train_epoch[0], train_epoch[1]):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(
            device, non_blocking=True
        )
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    train_loss = running_loss / total
    train_acc = correct / total

    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(
                device, non_blocking=True
            )
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)
    val_acc = val_correct / val_total if val_total > 0 else 0.0

    if val_acc > best_accuracy:
        best_accuracy = val_acc
        best_state = deepcopy(model.state_dict())
        torch.save(best_state, "weight.pth")
    print(
        f"Epoch {epoch+1}/{train_epoch[1]} - "
        f"Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f}, "
        f"Val acc: {val_acc:.4f}"
    )

model.load_state_dict(best_state)
model.eval()




## === cell 3
test_csv_path = os.path.join(data_path, "test.csv")
if not os.path.isfile(test_csv_path):
    test_images_dir = os.path.join(data_path, "test_images")
    test_filenames = sorted(
        [f for f in os.listdir(test_images_dir) if f.lower().endswith((".jpg", ".png"))]
    )
    test_df = pd.DataFrame({"image_id": test_filenames})
    test_csv_path = "generated_test.csv"
    test_df.to_csv(test_csv_path, index=False)


class TestDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, "test_images", self.df.iloc[idx, 0])
        img = io.imread(img_name)
        if self.transform:
            img = self.transform(img)
        return img


test_dataset = TestDataset(
    csv_file=test_csv_path, root_dir=data_path, transform=val_transform
)

test_images = torch.stack([test_dataset[i] for i in range(len(test_dataset))])
test_tensor_dataset = TensorDataset(test_images)

test_loader = DataLoader(
    test_tensor_dataset,
    batch_size=sub_batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,
)

predictions = []
with torch.no_grad():
    for (inputs,) in test_loader:  # TensorDataset returns a tuple
        inputs = inputs.to(device, non_blocking=True)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        predictions.extend(preds.cpu().numpy())

submission = pd.DataFrame(
    {"image_id": pd.read_csv(test_csv_path)["image_id"], "label": predictions}
)
submission.to_csv("submission.csv", index=False)
print("Submission file 'submission.csv' created.")
