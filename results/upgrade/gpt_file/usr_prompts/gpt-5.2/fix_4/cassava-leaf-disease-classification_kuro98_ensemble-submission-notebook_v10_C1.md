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

0.8986098519190088

# 6. Current score

0.7784

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.05531) has done: 'I make the solution run end-to-end by (1) replacing the missing external model checkpoints with torchvision ImageNet backbones (keeping the same “two models + linear head ensemble” core idea), (2) fixing test dataset ordering and DataLoader shuffling so predictions align exactly with `sample_submission.csv`, and (3) making TTA deterministic and correctly averaged so it doesn’t change the submission length or misalign filenames. These changes directly address the FileNotFound/NameError crashes and the “wrong length” submission error. The code always write a valid `submission.csv` with the required columns and row count.'
- What this solution (achieved 0.10987) has done: 'I fix the runtime crash by aligning the ViT input resolution to what `torchvision.models.vit_b_16` expects (224×224), while keeping your two-backbone + concatenation + linear head ensemble logic unchanged. I also correct the linear head input dimension to match the actual output sizes of ViT-B/16 (1000) and EfficientNet-B0 (1000) so the forward pass is consistent. Finally, I keep the deterministic TTA behavior and ensure the written `submission.csv` is correctly aligned to `sample_submission.csv` with integer labels and the correct row count.'
- What this solution (achieved 0.7784) has done: 'Your current score is low because the linear head is never trained (it’s randomly initialized), so predictions are essentially random; the smallest legitimate step toward the target is to train only that linear head on the provided `train.csv` while keeping both ImageNet backbones frozen and keeping the same “two models + concatenation + linear head” core logic. I add a minimal training dataset/loader and a short training loop (cross-entropy on 5 classes), then reuse your existing deterministic TTA inference and submission alignment. I also keep image preprocessing consistent between train and test, and ensure the script still finishes under the time limit by training only the small head for a few epochs. This should move accuracy much closer to the target without changing the architecture or inference semantics.'

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

data_root = "/kaggle/input/cassava-leaf-disease-classification/"
train_csv_path = os.path.join(data_root, "train.csv")
train_dir = os.path.join(data_root, "train_images/")
test_dir = os.path.join(data_root, "test_images/")
sample_sub_path = os.path.join(data_root, "sample_submission.csv")

eff_img_size = 528
vit_img_size = 224

batch_size = 16
num_workers = 4
num_classes = 5
tta = True

train_head = True
head_epochs = 3
head_lr = 2e-3

import torchvision

vit_model = torchvision.models.vit_b_16(
    weights=torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1
).to(device)
eff_model = torchvision.models.efficientnet_b0(
    weights=torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1
).to(device)

linear_head = torch.nn.Linear(2000, num_classes).to(device)




## === cell 2
class CassavaTrainDataset(VisionDataset):
    """Train dataset returning (vit_img, eff_img, label)."""

    def __init__(self, df, data_dir, vit_size, efficient_size, transform=None):
        super().__init__(root=data_dir)
        self.df = df.reset_index(drop=True)
        self.transform = transform

        self.cc = v2.CenterCrop((600, 600))
        self.resize_vit = v2.Resize(
            (vit_size, vit_size), interpolation=InterpolationMode.BICUBIC
        )
        self.resize_efficient = v2.Resize(
            (efficient_size, efficient_size), interpolation=InterpolationMode.BICUBIC
        )

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = row["image_id"]
        label = int(row["label"])

        img = Image.open(os.path.join(self.root, filename)).convert("RGB")
        img = self.cc(img)
        vit_img = self.resize_vit(img)
        eff_img = self.resize_efficient(img)

        if self.transform:
            vit_img = self.transform(vit_img)
            eff_img = self.transform(eff_img)

        return vit_img, eff_img, torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return len(self.df)


class CassavaDataset(VisionDataset):
    """Custom dataset for the Cassava test data.

    Returns:
        vit_img: Tensor [3,H,W] or list[Tensor] if TTA enabled
        eff_img: Tensor [3,H,W] or list[Tensor] if TTA enabled
        filename: image filename
    """

    def __init__(self, data_dir, vit_size, efficient_size, transform=None, ttas=None):
        super().__init__(root=data_dir)
        self.transform = transform

        self.images = sorted(
            [f for f in os.listdir(data_dir) if f.lower().endswith(".jpg")]
        )

        self.ttas = ttas
        self.cc = v2.CenterCrop((600, 600))
        self.resize_vit = v2.Resize(
            (vit_size, vit_size), interpolation=InterpolationMode.BICUBIC
        )
        self.resize_efficient = v2.Resize(
            (efficient_size, efficient_size), interpolation=InterpolationMode.BICUBIC
        )

    def __getitem__(self, idx):
        filename = self.images[idx]
        img = Image.open(os.path.join(self.root, filename)).convert("RGB")

        img = self.cc(img)
        vit_img = self.resize_vit(img)
        eff_img = self.resize_efficient(img)

        if self.ttas is not None and self.transform is not None:
            vit_img = [self.transform(t(vit_img)) for t in self.ttas]
            eff_img = [self.transform(t(eff_img)) for t in self.ttas]
        elif self.transform:
            vit_img = self.transform(vit_img)
            eff_img = self.transform(eff_img)

        return vit_img, eff_img, filename

    def __len__(self):
        return len(self.images)




## === cell 3
test_transforms = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

train_transforms = test_transforms

if tta:
    ttas = [
        v2.Identity(),  # original
        v2.RandomHorizontalFlip(p=1.0),  # deterministic because p=1.0
        v2.RandomVerticalFlip(p=1.0),  # deterministic because p=1.0
    ]
else:
    ttas = None

train_df = pd.read_csv(train_csv_path)
train_dataset = CassavaTrainDataset(
    train_df, train_dir, vit_img_size, eff_img_size, transform=train_transforms
)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
)

test_dataset = CassavaDataset(
    test_dir, vit_img_size, eff_img_size, transform=test_transforms, ttas=ttas
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
)

normalizer = torch.nn.Softmax(dim=1)



## === cell 4
for p in vit_model.parameters():
    p.requires_grad = False
for p in eff_model.parameters():
    p.requires_grad = False

vit_model.eval()
eff_model.eval()

if train_head:
    linear_head.train()
    optimizer = torch.optim.AdamW(linear_head.parameters(), lr=head_lr)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(head_epochs):
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for vit_inputs, eff_inputs, labels in train_loader:
            vit_inputs = vit_inputs.to(device, non_blocking=True)
            eff_inputs = eff_inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.no_grad():
                vit_outputs = vit_model(vit_inputs)  # [B,1000]
                eff_outputs = eff_model(eff_inputs)  # [B,1000]

            logit_inputs = torch.cat([vit_outputs, eff_outputs], dim=1)  # [B,2000]
            outputs = linear_head(logit_inputs)  # [B,5]

            loss = criterion(outputs, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * labels.size(0)
            preds = outputs.argmax(dim=1)
            running_correct += int((preds == labels).sum().item())
            running_total += int(labels.size(0))

        print(
            f"epoch {epoch+1}/{head_epochs} "
            f"loss={running_loss/max(running_total,1):.4f} "
            f"acc={running_correct/max(running_total,1):.4f}"
        )

linear_head.eval()



## === cell 5
all_names = []
all_preds = []

vit_model.eval()
eff_model.eval()
linear_head.eval()

with torch.no_grad():
    for batch_idx, (vit_inputs, eff_inputs, filenames) in enumerate(test_loader):
        base_batch_size = len(filenames)

        if tta:
            vit_inputs = torch.cat(vit_inputs, dim=0).to(device)  # [T*B,3,H,W]
            eff_inputs = torch.cat(eff_inputs, dim=0).to(device)  # [T*B,3,H,W]

            vit_outputs = vit_model(vit_inputs)  # [T*B,1000]
            eff_outputs = eff_model(eff_inputs)  # [T*B,1000]

            vit_batch_logits = torch.stack(
                torch.split(vit_outputs, base_batch_size), dim=0
            )  # [T,B,1000]
            vit_mean_logits = torch.mean(vit_batch_logits, dim=0)  # [B,1000]

            eff_batch_logits = torch.stack(
                torch.split(eff_outputs, base_batch_size), dim=0
            )  # [T,B,1000]
            eff_mean_logits = torch.mean(eff_batch_logits, dim=0)  # [B,1000]

            logit_inputs = torch.cat(
                [vit_mean_logits, eff_mean_logits], dim=1
            )  # [B,2000]
            outputs = linear_head(logit_inputs)  # [B,5]

            probs = normalizer(outputs)
            pred_labels = torch.argmax(probs, 1).tolist()
        else:
            vit_inputs = vit_inputs.to(device)
            eff_inputs = eff_inputs.to(device)

            vit_outputs = vit_model(vit_inputs)
            eff_outputs = eff_model(eff_inputs)

            logit_inputs = torch.cat([vit_outputs, eff_outputs], dim=1)
            outputs = linear_head(logit_inputs)

            probs = normalizer(outputs)
            pred_labels = torch.argmax(probs, 1).tolist()

        all_names.extend(list(filenames))
        all_preds.extend(pred_labels)

print("Preds:", len(all_preds), "Names:", len(all_names))



## === cell 6
sample_sub = pd.read_csv(sample_sub_path)

pred_df = pd.DataFrame({"image_id": all_names, "label": all_preds})

merged = sample_sub[["image_id"]].merge(pred_df, on="image_id", how="left")
if merged["label"].isna().any():
    fill_label = int(pred_df["label"].mode().iloc[0]) if len(pred_df) else 0
    merged["label"] = merged["label"].fillna(fill_label).astype(int)
else:
    merged["label"] = merged["label"].astype(int)

merged.to_csv("submission.csv", index=False)
print(merged.head())
print("submission.csv rows:", len(merged))
print("label value counts:\n", merged["label"].value_counts(dropna=False).sort_index())
