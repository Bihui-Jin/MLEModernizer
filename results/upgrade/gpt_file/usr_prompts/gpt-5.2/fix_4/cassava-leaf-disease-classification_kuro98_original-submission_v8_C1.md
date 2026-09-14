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

0.8921124206708976

# 6. Current score

0.75262

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.16816) has done: 'I fix the runtime error by ensuring the ViT input size matches what the pretrained `vit_b_16` expects (224x224), which eliminates the “Expected 224 but got 384” crash. I also fix the TTA pipeline: your current `tta_transforms` are not valid callables (they’re transforms, not functions), so I wrap them as callable pipelines that start from a PIL image and end as a normalized tensor of the correct shape. Finally, I keep the non-shuffled loader and the strict order check so the produced `submission.csv` always matches `sample_submission.csv` length and ordering.'
- What this solution (achieved 0.75262) has done: 'Your score is low because the current script never loads any trained cassava weights, so it’s effectively an ImageNet-pretrained ViT with a randomly initialized 5-class head (near-random predictions). The smallest legitimate improvement that preserves your core inference logic is to add a minimal training step on `train.csv` using the same ViT model and standard cross-entropy, then run your existing (submission-aligned) TTA inference. I keep the model architecture, transforms intent, and overall pipeline intact, but add a simple train/val split for sanity plus a short training loop that fits within the time budget. I also slightly tone down the strongest TTA transforms (keeping TTA enabled) because overly aggressive random geometry can hurt accuracy on this dataset, and this change tends to move accuracy upward toward your target.'

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

cudnn.deterministic = True
cudnn.benchmark = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification"
test_dir = f"{DATA_DIR}/test_images/"
train_dir = f"{DATA_DIR}/train_images/"
sample_path = f"{DATA_DIR}/sample_submission.csv"
train_csv_path = f"{DATA_DIR}/train.csv"

img_size = 224

batch_size = 16
num_workers = 4
num_classes = 5
tta = True

finetune_epochs = 2
finetune_lr = 3e-4

import torchvision

vit_model = torchvision.models.vit_b_16(
    weights=torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1
)
vit_model.heads.head = torch.nn.Linear(vit_model.heads.head.in_features, num_classes)
vit_model.to(device)




## === cell 2
class CassavaDataset(VisionDataset):
    """Custom dataset for the Cassava test data (submission-aligned).

    IMPORTANT: we must keep predictions aligned to sample_submission.csv.
    """

    def __init__(self, data_dir, image_ids, transform=None, tta_transforms=None):
        super().__init__(root=data_dir)
        self.transform = transform
        self.image_ids = list(image_ids)
        self.tta_transforms = tta_transforms

    def __getitem__(self, idx):
        filename = self.image_ids[idx]
        img = Image.open(os.path.join(self.root, filename)).convert("RGB")

        if self.tta_transforms is not None and self.transform is not None:
            imgs = [t(img) for t in self.tta_transforms]
            return imgs, filename

        if self.transform is not None:
            img = self.transform(img)
        return img, filename

    def __len__(self):
        return len(self.image_ids)


class CassavaTrainDataset(VisionDataset):
    def __init__(self, data_dir, df, transform=None):
        super().__init__(root=data_dir)
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = row["image_id"]
        label = int(row["label"])
        img = Image.open(os.path.join(self.root, filename)).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.df)




## === cell 3
test_transforms = v2.Compose(
    [
        v2.Resize((img_size, img_size), interpolation=InterpolationMode.BICUBIC),
        v2.CenterCrop((img_size, img_size)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

train_transforms = v2.Compose(
    [
        v2.RandomResizedCrop(
            (img_size, img_size),
            scale=(0.8, 1.0),
            interpolation=InterpolationMode.BICUBIC,
        ),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

if tta:
    tta_transforms = [
        v2.Compose(
            [
                v2.Resize(
                    (img_size, img_size), interpolation=InterpolationMode.BICUBIC
                ),
                v2.CenterCrop((img_size, img_size)),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        ),
        v2.Compose(
            [
                v2.Resize(
                    (img_size, img_size), interpolation=InterpolationMode.BICUBIC
                ),
                v2.CenterCrop((img_size, img_size)),
                v2.RandomHorizontalFlip(p=1.0),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        ),
        v2.Compose(
            [
                v2.Resize(
                    (img_size, img_size), interpolation=InterpolationMode.BICUBIC
                ),
                v2.CenterCrop((img_size, img_size)),
                v2.RandomVerticalFlip(p=1.0),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        ),
    ]
else:
    tta_transforms = None

sample_df = pd.read_csv(sample_path)
test_image_ids = sample_df["image_id"].tolist()

test_dataset = CassavaDataset(
    test_dir,
    image_ids=test_image_ids,
    transform=test_transforms,
    tta_transforms=tta_transforms,
)


def _seed_worker(worker_id):
    worker_seed = 3407 + worker_id
    torch.manual_seed(worker_seed)


g = torch.Generator()
g.manual_seed(3407)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,  # must not shuffle to preserve submission alignment
    num_workers=num_workers,
    pin_memory=True,
    worker_init_fn=_seed_worker,
    generator=g,
)

normalizer = torch.nn.Softmax(dim=1)



## === cell 4
train_df = pd.read_csv(train_csv_path)

perm = torch.randperm(len(train_df), generator=g).tolist()
split = int(0.95 * len(perm))
train_idx, val_idx = perm[:split], perm[split:]
train_df_split = train_df.iloc[train_idx].reset_index(drop=True)
val_df_split = train_df.iloc[val_idx].reset_index(drop=True)

train_dataset = CassavaTrainDataset(
    train_dir, train_df_split, transform=train_transforms
)
val_dataset = CassavaTrainDataset(train_dir, val_df_split, transform=test_transforms)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
    worker_init_fn=_seed_worker,
    generator=g,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    worker_init_fn=_seed_worker,
    generator=g,
)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(vit_model.parameters(), lr=finetune_lr, weight_decay=0.05)

for p in vit_model.parameters():
    p.requires_grad = False
for p in vit_model.heads.parameters():
    p.requires_grad = True


def _evaluate_acc(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()
    return correct / max(total, 1)


vit_model.train()
for epoch in range(finetune_epochs):
    vit_model.train()
    running_loss = 0.0
    for x, y in train_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = vit_model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    val_acc = _evaluate_acc(vit_model, val_loader)
    print(
        f"epoch={epoch+1}/{finetune_epochs} train_loss={running_loss/len(train_loader):.4f} val_acc={val_acc:.4f}"
    )



## === cell 5
all_names = []
all_preds = []

vit_model.eval()

with torch.no_grad():
    for inputs, filenames in test_loader:
        filenames = list(filenames)

        if tta:
            inputs_cat = torch.cat(inputs, dim=0).to(
                device, non_blocking=True
            )  # [T*B, C, H, W]
            preds = normalizer(vit_model(inputs_cat))  # [T*B, num_classes]

            T = len(inputs)
            B = len(filenames)
            preds = preds.view(T, B, -1)  # [T, B, num_classes]
            mean_preds = preds.mean(dim=0)  # [B, num_classes]
            pred_labels = mean_preds.argmax(dim=1).tolist()
        else:
            inputs = inputs.to(device, non_blocking=True)
            preds = normalizer(vit_model(inputs))
            pred_labels = preds.argmax(dim=1).tolist()

        all_names.extend(filenames)
        all_preds.extend(pred_labels)

assert len(all_names) == len(
    sample_df
), f"Pred length {len(all_names)} != sample length {len(sample_df)}"
assert all_names == test_image_ids, "Image order mismatch vs sample_submission.csv"



## === cell 6
my_submission = pd.DataFrame({"image_id": all_names, "label": all_preds})
my_submission.to_csv("submission.csv", index=False)

print(my_submission.shape)
print(my_submission.head())



## === cell 7
my_submission
