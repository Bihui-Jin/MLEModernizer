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

3.12

# 3. Installed packages

geopandas==0.14.4
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

0.894076760350559

# 6. Current score

0.74477

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.51868) has done: 'I fix the immediate runtime blocker by removing the dependency on missing external model files and replacing it with a lightweight, standard torchvision model available in the Kaggle runtime. I also fix the submission length issue by ensuring predictions are generated in the exact order of `sample_submission.csv` (and disabling shuffle), so `image_id` alignment and row count match the test set. To preserve the “core semantics” of image classification inference, the pipeline still load images, apply the same normalization, run a CNN, and output integer class labels 0–4. Finally, I add robust path detection for the dataset directory so it runs end-to-end and always writes a valid `submission.csv`.'
- What this solution (achieved 0.74477) has done: 'Your current score is low mainly because the model is an ImageNet-pretrained EfficientNet with a randomly initialized 5-class head, and you never fine-tune it on `train.csv`, so predictions are essentially near-random. To move toward the target accuracy with minimal semantic change, I keep the same EfficientNet-B0 architecture, same transforms, same DataLoader/inference pattern, and add a short supervised fine-tuning phase on the provided training images (train/val split) using standard cross-entropy. I also switch the center-crop to the standard EfficientNet resize/crop behavior (already implied by the pretrained weights) to reduce harmful cropping mismatch, while still keeping the same normalization and producing the same submission format and ordering. The resulting script still runs end-to-end and writes `submission.csv`.'

# 9. Code solution

## === cell 0
import os

import pandas as pd
import torch
from PIL import Image
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
from torchvision.transforms import InterpolationMode, v2



## === cell 1
torch.manual_seed(3407)
torch.cuda.manual_seed(3407)

cudnn.deterministic = False
cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

base_dir_candidates = [
    "/kaggle/input/cassava-leaf-disease-classification/",
    "/kaggle/data/cassava-leaf-disease-classification/",
]
base_dir = next((p for p in base_dir_candidates if os.path.exists(p)), None)
if base_dir is None:
    raise FileNotFoundError(
        f"Could not find dataset base dir in: {base_dir_candidates}"
    )

test_dir = os.path.join(base_dir, "test_images")
train_dir = os.path.join(base_dir, "train_images")
sample_path = os.path.join(base_dir, "sample_submission.csv")
train_csv_path = os.path.join(base_dir, "train.csv")

model_b_img_size = 224
model_a_img_size = 224

batch_size = 32
num_workers = 4
num_classes = 5
tta = False

finetune_epochs = 2
finetune_lr = 3e-4
val_fraction = 0.1

import torchvision

weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
model = torchvision.models.efficientnet_b0(weights=weights)
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
model = model.to(device)

normalizer = torch.nn.Softmax(dim=1)




## === cell 2
class CassavaDataset(VisionDataset):
    """Dataset for Cassava images.

    - For test: reads files in the exact order provided by sample_submission.csv
      to guarantee correct alignment/length for the submission.
    - For train/val: reads image_ids from train.csv and returns labels.
    """

    def __init__(
        self,
        data_dir,
        image_ids,
        model_a_size,
        model_b_size,
        transform=None,
        ttas=None,
        labels=None,  # optional list/series aligned with image_ids
    ):
        super().__init__(root=data_dir)
        self.transform = transform
        self.images = list(image_ids)
        self.ttas = ttas
        self.labels = None if labels is None else list(labels)

        self.resize_for_crop_a = v2.Resize(
            (model_a_size, model_a_size), interpolation=InterpolationMode.BICUBIC
        )
        self.resize_for_crop_b = v2.Resize(
            (model_b_size, model_b_size), interpolation=InterpolationMode.BICUBIC
        )

    def __getitem__(self, idx):
        filename = self.images[idx]
        img_path = os.path.join(self.root, filename)
        img = Image.open(img_path).convert("RGB")

        model_a_img = self.resize_for_crop_a(img)
        model_b_img = self.resize_for_crop_b(img)

        if self.ttas is not None and self.transform is not None:
            model_a_img = [self.transform(t(model_a_img)) for t in self.ttas]
            model_b_img = [self.transform(t(model_b_img)) for t in self.ttas]
        elif self.transform:
            model_a_img = self.transform(model_a_img)
            model_b_img = self.transform(model_b_img)

        if self.labels is None:
            return model_a_img, model_b_img, filename
        else:
            return model_a_img, model_b_img, filename, int(self.labels[idx])

    def __len__(self):
        return len(self.images)




## === cell 3
train_transforms = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

test_transforms = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

if tta:
    ttas = [
        v2.RandomRotation(180),
        v2.RandomVerticalFlip(1),
        v2.RandomPerspective(p=1),
    ]
else:
    ttas = None

sample_sub = pd.read_csv(sample_path)
test_image_ids = sample_sub["image_id"].tolist()

train_df = pd.read_csv(train_csv_path)
train_image_ids = train_df["image_id"].tolist()
train_labels = train_df["label"].astype(int).tolist()

n = len(train_df)
val_size = int(n * val_fraction)
g = torch.Generator()
g.manual_seed(3407)
perm = torch.randperm(n, generator=g).tolist()
val_idx = set(perm[:val_size])
tr_idx = [i for i in range(n) if i not in val_idx]
va_idx = [i for i in range(n) if i in val_idx]

tr_ids = [train_image_ids[i] for i in tr_idx]
tr_y = [train_labels[i] for i in tr_idx]
va_ids = [train_image_ids[i] for i in va_idx]
va_y = [train_labels[i] for i in va_idx]

train_dataset = CassavaDataset(
    train_dir,
    tr_ids,
    model_a_img_size,
    model_b_img_size,
    transform=train_transforms,
    ttas=None,
    labels=tr_y,
)

val_dataset = CassavaDataset(
    train_dir,
    va_ids,
    model_a_img_size,
    model_b_img_size,
    transform=test_transforms,
    ttas=None,
    labels=va_y,
)

test_dataset = CassavaDataset(
    test_dir,
    test_image_ids,
    model_a_img_size,
    model_b_img_size,
    transform=test_transforms,
    ttas=ttas,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,  # IMPORTANT: preserve order for submission alignment
    num_workers=num_workers,
    pin_memory=True,
)



## === cell 4
import torch.nn as nn

criterion = nn.CrossEntropyLoss()

for p in model.parameters():
    p.requires_grad = False
for p in model.classifier.parameters():
    p.requires_grad = True
for p in model.features[-1].parameters():
    p.requires_grad = True

optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=finetune_lr,
    weight_decay=1e-4,
)


def _accuracy_from_logits(logits, y):
    preds = torch.argmax(logits, dim=1)
    return (preds == y).float().mean().item()


model.train()
for epoch in range(finetune_epochs):
    running_loss = 0.0
    running_acc = 0.0
    steps = 0

    for model_a_inputs, _, _, y in train_loader:
        inputs = model_a_inputs.to(device, non_blocking=True)
        y = torch.as_tensor(y, device=device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_acc += _accuracy_from_logits(logits.detach(), y)
        steps += 1

    model.eval()
    val_acc = 0.0
    vsteps = 0
    with torch.no_grad():
        for model_a_inputs, _, _, y in val_loader:
            inputs = model_a_inputs.to(device, non_blocking=True)
            y = torch.as_tensor(y, device=device)
            logits = model(inputs)
            val_acc += _accuracy_from_logits(logits, y)
            vsteps += 1

    print(
        f"epoch={epoch+1}/{finetune_epochs} "
        f"train_loss={running_loss/max(1,steps):.4f} "
        f"train_acc={running_acc/max(1,steps):.4f} "
        f"val_acc={val_acc/max(1,vsteps):.4f}"
    )
    model.train()

model.eval()



## === cell 5
all_names = []
all_preds = []

with torch.no_grad():
    for batch_idx, batch in enumerate(test_loader):
        model_a_inputs, model_b_inputs, filenames = batch

        if tta:
            bs = len(filenames)
            inputs = torch.cat(model_a_inputs, dim=0).to(device)

            logits = model(inputs)
            logits = torch.stack(torch.split(logits, bs), dim=0).mean(dim=0)

            preds = normalizer(logits)
            pred_labels = torch.argmax(preds, 1).tolist()
        else:
            inputs = model_a_inputs.to(device)
            logits = model(inputs)
            preds = normalizer(logits)
            pred_labels = torch.argmax(preds, 1).tolist()

        all_names.extend(list(filenames))
        all_preds.extend(pred_labels)

if len(all_names) != len(sample_sub):
    raise RuntimeError(
        f"Prediction length mismatch: got {len(all_names)} preds but sample has {len(sample_sub)} rows."
    )

pred_df = pd.DataFrame({"image_id": all_names, "label": all_preds})
pred_df = sample_sub[["image_id"]].merge(pred_df, on="image_id", how="left")

if pred_df["label"].isna().any():
    missing = pred_df[pred_df["label"].isna()]["image_id"].head(5).tolist()
    raise RuntimeError(f"Missing predictions for some images, e.g.: {missing}")

pred_df["label"] = pred_df["label"].astype(int)



## === cell 6
my_submission = pred_df[["image_id", "label"]]
my_submission.to_csv("submission.csv", index=False)
print(my_submission.shape)
print(my_submission.head())
