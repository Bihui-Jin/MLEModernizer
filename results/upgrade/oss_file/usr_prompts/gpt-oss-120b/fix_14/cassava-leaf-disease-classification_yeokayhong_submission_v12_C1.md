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
import os, random, json
import pandas as pd
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm.auto import tqdm
import numpy as np

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

test_data_directory = "/kaggle/input/cassava-leaf-disease-classification/test_images"
train_data_directory = "/kaggle/input/cassava-leaf-disease-classification/train_images"
train_csv_path = "/kaggle/input/cassava-leaf-disease-classification/train.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 5
model_select = "vit"  # choose "vit" or "en"

model_image_size = 224 if model_select == "vit" else 480

torch.backends.cudnn.benchmark = True

train_transforms = transforms.Compose(
    [
        transforms.Resize((model_image_size, model_image_size)),
        transforms.RandomResizedCrop(model_image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

val_transforms = transforms.Compose(
    [
        transforms.Resize((model_image_size, model_image_size)),
        transforms.CenterCrop(model_image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

if model_select == "en":
    try:
        model = models.efficientnet_v2_l(pretrained=True)
    except AttributeError:
        model = models.resnet18(pretrained=True)
else:
    try:
        model = models.vit_b_16(pretrained=True)
    except AttributeError:
        model = models.resnet18(pretrained=True)

if hasattr(model, "heads"):  # Vision Transformer
    if isinstance(model.heads, nn.Linear):
        model.heads = nn.Linear(model.heads.in_features, num_classes)
    elif isinstance(model.heads, nn.Sequential):
        if isinstance(model.heads[-1], nn.Linear):
            in_feat = model.heads[-1].in_features
            model.heads[-1] = nn.Linear(in_feat, num_classes)
        else:
            raise RuntimeError("Unexpected head structure in ViT model.")
    else:
        raise RuntimeError("Unsupported 'heads' type.")
elif hasattr(model, "classifier"):  # EfficientNet
    if isinstance(model.classifier, nn.Sequential):
        if isinstance(model.classifier[-1], nn.Linear):
            in_feat = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_feat, num_classes)
        else:
            raise RuntimeError("Unexpected classifier structure.")
    else:
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
elif hasattr(model, "fc"):  # ResNet fallback
    model.fc = nn.Linear(model.fc.in_features, num_classes)

model = model.to(device)




## === cell 1
class CassavaDataset(Dataset):
    """
    Training dataset that pre‑loads all raw PIL images into memory once.
    Random augmentations are applied on‑the‑fly each __getitem__ call.
    """

    def __init__(self, csv_path, img_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.images = []
        for _, row in self.df.iterrows():
            img_path = os.path.join(img_dir, row["image_id"])
            img = Image.open(img_path).convert("RGB")
            self.images.append(img)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image = self.images[idx]
        if self.transform:
            image = self.transform(image)
        label = int(self.df.iloc[idx]["label"])
        return image, label


train_dataset = CassavaDataset(
    train_csv_path, train_data_directory, transform=train_transforms
)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,  # increased from 8
    shuffle=True,
    num_workers=os.cpu_count(),  # use full CPU count for max parallelism
    pin_memory=True,
    persistent_workers=True,
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    model.parameters(), lr=5e-4
)  # slightly lower LR for better convergence
scaler = torch.cuda.amp.GradScaler()
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=4, gamma=0.5
)  # halve LR after 4 epochs

num_epochs = 12  # train longer to improve accuracy
model.train()
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for imgs, lbls in tqdm(train_loader, desc=f"Training epoch {epoch+1}/{num_epochs}"):
        imgs = imgs.to(device, non_blocking=True)
        lbls = lbls.to(device, non_blocking=True)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(imgs)
            loss = criterion(outputs, lbls)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item() * imgs.size(0)

    scheduler.step()  # update learning rate
    avg_loss = epoch_loss / len(train_loader.dataset)
    print(f"Epoch {epoch+1}/{num_epochs} – Average loss: {avg_loss:.4f}")

model.eval()




## === cell 2
class TestCassavaDataset(Dataset):
    """
    Test dataset that pre‑computes the deterministic validation transforms,
    storing the resulting tensors to avoid repeated per‑batch processing.
    """

    def __init__(self, img_dir, transform=None):
        self.transform = transform
        self.image_paths = []
        for root, _, files in os.walk(img_dir):
            for fname in files:
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.image_paths.append(os.path.join(root, fname))
        self.image_paths.sort()  # deterministic order

        self.tensors = []
        self.ids = []
        for path in self.image_paths:
            image_id = os.path.basename(path)
            try:
                img = Image.open(path).convert("RGB")
            except Exception:
                img = Image.new("RGB", (model_image_size, model_image_size))
            if self.transform:
                img = self.transform(img)
            self.tensors.append(img)
            self.ids.append(image_id)

    def __len__(self):
        return len(self.tensors)

    def __getitem__(self, idx):
        return self.tensors[idx], self.ids[idx]


test_dataset = TestCassavaDataset(test_data_directory, transform=val_transforms)
test_loader = DataLoader(
    test_dataset,
    batch_size=16,  # match training batch size for efficiency
    shuffle=False,
    num_workers=os.cpu_count(),
    pin_memory=True,
    persistent_workers=True,
)

predictions = []
image_ids = []

for batch_imgs, batch_ids in tqdm(test_loader, desc="Test"):
    batch_imgs = batch_imgs.to(device, non_blocking=True)
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            outputs = model(batch_imgs)
        _, preds = torch.max(outputs, 1)
    predictions.extend(preds.cpu().tolist())
    image_ids.extend(batch_ids)




## === cell 3
submission_df = pd.DataFrame({"image_id": image_ids, "label": predictions})
submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)
print(f"Submission file created: {submission_path} (rows: {len(submission_df)})")
