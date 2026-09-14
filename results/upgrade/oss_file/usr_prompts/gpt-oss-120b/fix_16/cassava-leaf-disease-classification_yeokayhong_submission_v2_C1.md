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
import os
import torch
import pandas as pd
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from torchvision.models import vit_l_16, ViT_L_16_Weights

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True

test_data_directory = "/kaggle/input/cassava-leaf-disease-classification/test_images"
train_data_directory = "/kaggle/input/cassava-leaf-disease-classification/train_images"
train_csv_path = "/kaggle/input/cassava-leaf-disease-classification/train.csv"
model_path = "/kaggle/input/vit_l_cassava/pytorch/default/1/model_weights_3.pth"

image_size = 224
num_classes = 5

pretrained_weight = (
    ViT_L_16_Weights.IMAGENET1K_V1
    if hasattr(ViT_L_16_Weights, "IMAGENET1K_V1")
    else ViT_L_16_Weights.DEFAULT
)

val_transforms = pretrained_weight.transforms()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

model = vit_l_16(weights=pretrained_weight, image_size=image_size)

if hasattr(model, "heads"):
    if isinstance(model.heads, torch.nn.Linear):
        in_features = model.heads.in_features
        model.heads = torch.nn.Linear(in_features, num_classes)
    elif isinstance(model.heads, torch.nn.Sequential):
        if hasattr(model.heads[-1], "in_features"):
            in_features = model.heads[-1].in_features
        else:
            in_features = 1024
        model.heads[-1] = torch.nn.Linear(in_features, num_classes)
    else:
        in_features = getattr(model, "hidden_dim", 1024)
        model.heads = torch.nn.Linear(in_features, num_classes)
else:
    in_features = 1024
    model.heads = torch.nn.Linear(in_features, num_classes)

skip_training = False
if os.path.isfile(model_path):
    try:
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded fine‑tuned weights from {model_path}")
        skip_training = True
    except Exception as e:
        print(f"Warning: could not load checkpoint ({e}). Using pretrained weights.")
else:
    print(f"Warning: checkpoint not found at {model_path}. Using pretrained weights.")

model.to(device)

if torch.cuda.is_available():
    try:
        model = torch.compile(model, mode="max-autotune")
    except Exception as e:
        print(f"Compilation warning: {e}")

MAX_WORKERS = min(8, os.cpu_count() or 1)




## === cell 1
class CassavaDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, img_dir, transform):
        df = pd.read_csv(csv_path)
        self.images = []
        self.labels = []
        for _, row in df.iterrows():
            img_path = os.path.join(img_dir, row["image_id"])
            image = Image.open(img_path).convert("RGB")
            image = transform(image)  # Tensor, already normalized
            self.images.append(image)
            self.labels.append(int(row["label"]))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


train_dataset = CassavaDataset(train_csv_path, train_data_directory, val_transforms)
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True,
    num_workers=0,  # no workers needed; data is in memory
    pin_memory=True,
    persistent_workers=False,
)



## === cell 2
for param in model.parameters():
    param.requires_grad = False

if isinstance(model.heads, torch.nn.Sequential):
    if isinstance(model.heads[-1], torch.nn.Linear):
        model.heads[-1].requires_grad_(True)
else:
    model.heads.requires_grad_(True)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4
)

scaler = torch.cuda.amp.GradScaler()
model.train()
epochs = 10  # keep original epoch count

if not skip_training:
    for epoch in range(epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0
        for imgs, lbls in tqdm(train_loader, desc=f"Training epoch {epoch+1}/{epochs}"):
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
            _, preds = torch.max(outputs, 1)
            correct += (preds == lbls).sum().item()
            total += lbls.size(0)

        epoch_acc = correct / total
        print(f"Epoch {epoch+1}: loss={epoch_loss/total:.4f}, accuracy={epoch_acc:.4f}")
else:
    print("Skipping training because fine‑tuned checkpoint was loaded.")

model.eval()  # Switch back to eval mode for inference




## === cell 3
class TestDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, transform):
        self.transform = transform
        self.filenames = [
            f
            for f in sorted(os.listdir(img_dir))
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
            and os.path.isfile(os.path.join(img_dir, f))
        ]
        self.images = []
        for fname in self.filenames:
            img_path = os.path.join(img_dir, fname)
            image = Image.open(img_path).convert("RGB")
            image = transform(image)
            self.images.append(image)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.filenames[idx]


test_dataset = TestDataset(test_data_directory, val_transforms)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=128,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
)

test_predictions = []
image_ids = []

model.to(device)  # ensure model is on the correct device

with torch.inference_mode():
    for batch_imgs, batch_names in tqdm(test_loader, desc="Test"):
        batch_imgs = batch_imgs.to(device, non_blocking=True)
        outputs = model(batch_imgs)
        _, preds = torch.max(outputs, dim=1)
        test_predictions.extend(preds.cpu().tolist())
        image_ids.extend(batch_names)

submission_df = pd.DataFrame({"image_id": image_ids, "label": test_predictions})
submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)
print(f"Submission file created: {submission_path}")
