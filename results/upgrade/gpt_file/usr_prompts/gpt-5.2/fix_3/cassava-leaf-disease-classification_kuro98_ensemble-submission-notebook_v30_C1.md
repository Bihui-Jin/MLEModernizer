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

0.8933212450891508

# 6. Current score

0.77803

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5142) has done: 'I fix the immediate runtime blocker by removing the dependency on missing external model files (`/kaggle/input/vit-v1`, etc.) and replacing it with a small self-contained torchvision backbone so the notebook runs end-to-end in this environment. I also fix submission length/order issues by reading `sample_submission.csv` and predicting exactly in that order (no `os.listdir` randomness, no `shuffle=True`). Finally, I ensure images are always converted to RGB (some PIL files can be palette/gray) and keep the existing inference semantics (softmax + argmax) while making the pipeline deterministic and submission-valid.'
- What this solution (achieved 0.77803) has done: 'Your current score is low mainly because you’re using ImageNet-pretrained backbones with randomly initialized 5-class classifiers (no cassava training), so predictions are close to arbitrary. To move the score toward the 0.893 target while keeping the same overall inference pipeline, I add a minimal training phase on `train.csv` using the same two backbones and the same softmax+argmax inference semantics. I keep the dataset/transforms structure, add a simple train/val split for sanity (not for early stopping), train only the classifier layers first (fast, stable), then run the same test inference and write `submission.csv` in `sample_submission.csv` order. This is the smallest change that legitimately improves accuracy without changing your core model choices or output logic.'

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

base_dir = "/kaggle/input/cassava-leaf-disease-classification/"
test_dir = os.path.join(base_dir, "test_images")
train_dir = os.path.join(base_dir, "train_images")
sample_path = os.path.join(base_dir, "sample_submission.csv")
train_csv_path = os.path.join(base_dir, "train.csv")

model_b_img_size = 384
model_a_img_size = 384
batch_size = 32
num_workers = 4
num_classes = 5
tta = False

import torchvision

model_a = torchvision.models.efficientnet_b0(
    weights=torchvision.models.EfficientNet_B0_Weights.DEFAULT
)
model_a.classifier[1] = torch.nn.Linear(model_a.classifier[1].in_features, num_classes)

model_b = torchvision.models.mobilenet_v3_small(
    weights=torchvision.models.MobileNet_V3_Small_Weights.DEFAULT
)
model_b.classifier[3] = torch.nn.Linear(model_b.classifier[3].in_features, num_classes)

linear_head = torch.nn.Identity()

model_a.to(device)
model_b.to(device)
linear_head.to(device)




## === cell 2
class CassavaDataset(VisionDataset):
    """Custom dataset for Cassava images (order controlled by provided list)."""

    def __init__(
        self,
        data_dir,
        model_a_size,
        model_b_size,
        image_ids,
        labels=None,
        transform=None,
        ttas=None,
    ):
        super().__init__(root=data_dir)
        self.transform = transform
        self.images = list(image_ids)
        self.labels = None if labels is None else list(labels)
        self.ttas = ttas
        self.cc = v2.CenterCrop((600, 600))
        self.resize_model_a = v2.Resize(
            (model_a_size, model_a_size), interpolation=InterpolationMode.BICUBIC
        )
        self.resize_model_b = v2.Resize(
            (model_b_size, model_b_size), interpolation=InterpolationMode.BICUBIC
        )

    def __getitem__(self, idx):
        filename = self.images[idx]
        img_path = os.path.join(self.root, filename)
        img = Image.open(img_path).convert("RGB")
        img = self.cc(img)

        model_a_img = self.resize_model_a(img)
        model_b_img = self.resize_model_b(img)

        if self.ttas is not None and self.transform is not None:
            model_a_img = [self.transform(t(model_a_img)) for t in self.ttas]
            model_b_img = [self.transform(t(model_b_img)) for t in self.ttas]
        elif self.transform:
            model_a_img = self.transform(model_a_img)
            model_b_img = self.transform(model_b_img)

        if self.labels is None:
            return model_a_img, model_b_img, filename
        else:
            return model_a_img, model_b_img, int(self.labels[idx])

    def __len__(self):
        return len(self.images)




## === cell 3
train_transforms = v2.Compose(
    [
        v2.ToImage(),
        v2.RandomResizedCrop(
            (model_a_img_size, model_a_img_size),
            scale=(0.7, 1.0),
            interpolation=InterpolationMode.BICUBIC,
        ),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ToDtype(torch.float32, scale=True),
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



## === cell 4
train_df = pd.read_csv(train_csv_path)
train_df["image_id"] = train_df["image_id"].astype(str)
train_df["label"] = train_df["label"].astype(int)

g = torch.Generator().manual_seed(3407)
perm = torch.randperm(len(train_df), generator=g).tolist()
val_size = int(0.1 * len(train_df))
val_idx = set(perm[:val_size])
trn_rows = [i for i in range(len(train_df)) if i not in val_idx]
val_rows = [i for i in range(len(train_df)) if i in val_idx]

trn_df = train_df.iloc[trn_rows].reset_index(drop=True)
val_df = train_df.iloc[val_rows].reset_index(drop=True)

train_dataset = CassavaDataset(
    train_dir,
    model_a_img_size,
    model_b_img_size,
    image_ids=trn_df["image_id"].tolist(),
    labels=trn_df["label"].tolist(),
    transform=train_transforms,
    ttas=None,
)

val_dataset = CassavaDataset(
    train_dir,
    model_a_img_size,
    model_b_img_size,
    image_ids=val_df["image_id"].tolist(),
    labels=val_df["label"].tolist(),
    transform=test_transforms,
    ttas=None,
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



## === cell 5
for p in model_a.parameters():
    p.requires_grad = False
for p in model_b.parameters():
    p.requires_grad = False

for p in model_a.classifier.parameters():
    p.requires_grad = True
for p in model_b.classifier.parameters():
    p.requires_grad = True

criterion = torch.nn.CrossEntropyLoss()

opt_a = torch.optim.AdamW(model_a.classifier.parameters(), lr=3e-3, weight_decay=1e-4)
opt_b = torch.optim.AdamW(model_b.classifier.parameters(), lr=3e-3, weight_decay=1e-4)

epochs = 2

model_a.train()
model_b.train()

for epoch in range(1, epochs + 1):
    total_loss = 0.0
    n = 0
    for model_a_inputs, model_b_inputs, y in train_loader:
        model_a_inputs = model_a_inputs.to(device, non_blocking=True)
        model_b_inputs = model_b_inputs.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        opt_a.zero_grad(set_to_none=True)
        opt_b.zero_grad(set_to_none=True)

        out_a = model_a(model_a_inputs)
        out_b = model_b(model_b_inputs)

        logits = (0.95 * out_a + 0.05 * out_b) / 2.0

        loss = criterion(logits, y)
        loss.backward()

        opt_a.step()
        opt_b.step()

        total_loss += float(loss.item()) * y.size(0)
        n += y.size(0)

    avg_loss = total_loss / max(1, n)

    model_a.eval()
    model_b.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for model_a_inputs, model_b_inputs, y in val_loader:
            model_a_inputs = model_a_inputs.to(device, non_blocking=True)
            model_b_inputs = model_b_inputs.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            out_a = model_a(model_a_inputs)
            out_b = model_b(model_b_inputs)
            logits = (0.95 * out_a + 0.05 * out_b) / 2.0
            pred = torch.argmax(logits, dim=1)
            correct += int((pred == y).sum().item())
            total += int(y.size(0))

    val_acc = correct / max(1, total)
    print(f"epoch={epoch} train_loss={avg_loss:.4f} val_acc={val_acc:.4f}")

    model_a.train()
    model_b.train()



## === cell 6
sample_sub = pd.read_csv(sample_path)
test_image_ids = sample_sub["image_id"].tolist()

test_dataset = CassavaDataset(
    test_dir,
    model_a_img_size,
    model_b_img_size,
    image_ids=test_image_ids,
    labels=None,
    transform=test_transforms,
    ttas=ttas,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
)

normalizer = torch.nn.Softmax(dim=1)



## === cell 7
all_names = []
all_preds = []

model_a.eval()
model_b.eval()
linear_head.eval()

with torch.no_grad():
    for model_a_inputs, model_b_inputs, filenames in test_loader:
        bs = len(filenames)

        if tta:
            model_a_inputs = torch.cat(model_a_inputs, dim=0).to(device)
            model_b_inputs = torch.cat(model_b_inputs, dim=0).to(device)
            filenames = list(filenames)

            model_a_outputs = model_a(model_a_inputs)
            model_b_outputs = model_b(model_b_inputs)

            model_a_batch_logits = torch.stack(torch.split(model_a_outputs, bs), dim=0)
            model_a_mean_logits = torch.mean(model_a_batch_logits, dim=0)

            model_b_batch_logits = torch.stack(torch.split(model_b_outputs, bs), dim=0)
            model_b_mean_logits = torch.mean(model_b_batch_logits, dim=0)

            outputs = (0.9 * model_a_mean_logits + 0.1 * model_b_mean_logits) / 2.0
            preds = normalizer(outputs)
            pred_labels = torch.argmax(preds, 1).tolist()
        else:
            model_a_inputs = model_a_inputs.to(device, non_blocking=True)
            model_b_inputs = model_b_inputs.to(device, non_blocking=True)
            filenames = list(filenames)

            model_a_outputs = model_a(model_a_inputs)
            model_b_outputs = model_b(model_b_inputs)

            outputs = (0.95 * model_a_outputs + 0.05 * model_b_outputs) / 2.0
            preds = normalizer(outputs)
            pred_labels = torch.argmax(preds, 1).tolist()

        all_names.extend(filenames)
        all_preds.extend(pred_labels)

assert len(all_names) == len(sample_sub), (len(all_names), len(sample_sub))
assert (
    all_names == test_image_ids
), "Prediction order mismatch vs sample_submission order."



## === cell 8
my_submission = pd.DataFrame({"image_id": all_names, "label": all_preds})
my_submission.to_csv("submission.csv", index=False)

print(my_submission.head())
print("Saved submission.csv with rows:", len(my_submission))
my_submission
