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
import os
import random
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



## === cell 1
test_data_directory = "/kaggle/input/cassava-leaf-disease-classification/test_images"
train_data_directory = "/kaggle/input/cassava-leaf-disease-classification/train_images"
train_csv_path = "/kaggle/input/cassava-leaf-disease-classification/train.csv"

sample_sub_path = (
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 5

vit_model_path = "/kaggle/input/main_model/pytorch/default/1/ViT_H_14_518.pth"
vit_image_size = 518

resnet_model_path = "/kaggle/input/abc/keras/default/1/newModel7.keras"

assert os.path.isdir(
    test_data_directory
), f"Missing test image directory: {test_data_directory}"
assert os.path.isfile(
    sample_sub_path
), f"Missing sample_submission.csv: {sample_sub_path}"



## === cell 2
vit_preprocess_custom = transforms.Compose(
    [
        transforms.Resize((vit_image_size, vit_image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)



## === cell 3
vit_weights = None
using_custom_checkpoint = os.path.isfile(vit_model_path)

if using_custom_checkpoint:
    vit_model = models.vit_h_14(weights=None, image_size=vit_image_size)
    vit_model.heads.head = torch.nn.Linear(
        vit_model.heads.head.in_features, num_classes
    )

    state = torch.load(vit_model_path, map_location=device, weights_only=False)
    vit_model.load_state_dict(state)
    vit_preprocess = vit_preprocess_custom
else:
    try:
        vit_weights = models.ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1
    except Exception:
        vit_weights = models.ViT_H_14_Weights.IMAGENET1K_SWAG_LINEAR_V1

    vit_model = models.vit_h_14(weights=vit_weights)
    vit_preprocess = vit_weights.transforms()

vit_model.to(device)
vit_model.eval()

print(
    f"ViT model ready. Custom checkpoint loaded: {using_custom_checkpoint} (fallback pretrained used: {not using_custom_checkpoint})"
)
if vit_weights is not None:
    print(f"Fallback weights: {vit_weights}")




## === cell 4
class CassavaTrainDataset(Dataset):
    def __init__(self, df, image_dir, transform):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_id = self.df.loc[idx, "image_id"]
        label = int(self.df.loc[idx, "label"])
        img_path = os.path.join(self.image_dir, image_id)
        img = Image.open(img_path).convert("RGB")
        x = self.transform(img)
        return x, label


def train_linear_head_on_frozen_vit(
    vit_model,
    vit_preprocess,
    device,
    train_csv_path,
    train_data_directory,
    num_classes=5,
):
    assert os.path.isfile(train_csv_path), f"Missing train.csv: {train_csv_path}"
    assert os.path.isdir(
        train_data_directory
    ), f"Missing train_images directory: {train_data_directory}"

    train_df = pd.read_csv(train_csv_path)
    train_df = train_df[["image_id", "label"]].dropna()
    train_df["label"] = train_df["label"].astype(int)

    ds = CassavaTrainDataset(train_df, train_data_directory, vit_preprocess)
    dl = DataLoader(
        ds,
        batch_size=16,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    vit_model.train()
    for p in vit_model.parameters():
        p.requires_grad = False

    in_features = vit_model.heads.head.in_features
    vit_model.heads.head = torch.nn.Linear(in_features, num_classes).to(device)
    for p in vit_model.heads.head.parameters():
        p.requires_grad = True

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        vit_model.heads.head.parameters(), lr=3e-4, weight_decay=0.01
    )

    epochs = 1

    scaler = None
    use_amp = torch.cuda.is_available()
    if use_amp:
        scaler = torch.cuda.amp.GradScaler()

    vit_model.to(device)

    for ep in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(
            dl, desc=f"Training linear head (epoch {ep+1}/{epochs})", total=len(dl)
        )
        for xb, yb in pbar:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                with torch.cuda.amp.autocast():
                    logits = vit_model(xb)
                    loss = criterion(logits, yb)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = vit_model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

            running_loss += float(loss.item()) * xb.size(0)
            preds = torch.argmax(logits.detach(), dim=1)
            correct += int((preds == yb).sum().item())
            total += int(xb.size(0))
            pbar.set_postfix(
                loss=running_loss / max(total, 1), acc=correct / max(total, 1)
            )

    vit_model.eval()
    return vit_model


if not using_custom_checkpoint:
    vit_model = train_linear_head_on_frozen_vit(
        vit_model=vit_model,
        vit_preprocess=vit_preprocess,
        device=device,
        train_csv_path=train_csv_path,
        train_data_directory=train_data_directory,
        num_classes=num_classes,
    )
    print(
        "Fallback pretrained ViT head trained on cassava train.csv (backbone frozen)."
    )



## === cell 5
sample_sub = pd.read_csv(sample_sub_path)
test_image_ids = sample_sub["image_id"].tolist()

missing = [
    img_id
    for img_id in test_image_ids
    if not os.path.isfile(os.path.join(test_data_directory, img_id))
]
if missing:
    print(
        f"Warning: {len(missing)} images listed in sample_submission not found in folder (will predict label 0 for them). Example: {missing[:3]}"
    )



## === cell 6
pred_labels = []

with torch.no_grad():
    for image_id in tqdm(
        test_image_ids, desc="Test inference", total=len(test_image_ids)
    ):
        img_path = os.path.join(test_data_directory, image_id)
        if not os.path.isfile(img_path):
            pred_labels.append(0)
            continue

        img = Image.open(img_path).convert("RGB")
        x = vit_preprocess(img).unsqueeze(0).to(device)

        logits = vit_model(x)

        probs = torch.softmax(logits, dim=1)
        pred = int(torch.argmax(probs, dim=1).item())

        pred_labels.append(pred)



## === cell 7
submission_df = pd.DataFrame({"image_id": test_image_ids, "label": pred_labels})

assert submission_df.shape[0] == sample_sub.shape[0]
assert list(submission_df.columns) == ["image_id", "label"]



## === cell 8
submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)
print(f"Submission file created: {submission_path}")
print(submission_df.head())



## === cell 9
submission_df
