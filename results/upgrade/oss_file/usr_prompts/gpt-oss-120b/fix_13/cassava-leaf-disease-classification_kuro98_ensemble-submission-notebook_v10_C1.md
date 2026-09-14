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

0.78251

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.10463) has done: 'Implemented a robust dataset loader that skips directory entries, preventing PIL from trying to open a folder as an image. This resolves the `IsADirectoryError` and ensures the number of predictions matches the expected test set size, allowing a valid `submission.csv` to be generated.'
- What this solution (achieved 0.76794) has done: 'I replace the missing pretrained models with a simple ImageNet‑pretrained EfficientNet backbone, freeze its weights, and train only a small linear head on the provided training set for one epoch. This keeps the original architecture (two feature extractors whose outputs are concatenated) while adding a lightweight fine‑tuning step that should raise accuracy well above the random‑guess baseline and move the score toward the target.'
- What this solution (achieved 0.10837) has done: 'I compile the models with torch.compile to reduce the per‑step compute cost, enable TF‑32 and high‑precision matrix math, and increase the data‑loader workers to better utilize CPU cores. These changes keep the exact architecture, loss, optimizer and training schedule, so model behavior and accuracy remain unchanged while runtime is shortened.'
- What this solution (achieved 0.78961) has done: 'We pre‑compute the frozen backbone features once before training so the 5‑epoch loop only updates the tiny linear head, eliminating repeated costly forward passes through the Vision Transformers and EfficientNet. This keeps the exact same inputs to the linear head (identical features and labels), preserving model semantics while cutting runtime dramatically. We also use a TensorDataset for the pre‑computed features to simplify the training loop.'
- What this solution (achieved 0.78251) has done: 'I keep the overall architecture unchanged (two frozen backbones with a linear head) but train the head more effectively. The changes increase the number of epochs, add a cosine‑annealing learning‑rate scheduler with a small weight decay, and use label‑smoothing in the cross‑entropy loss. These minimal adjustments should raise validation accuracy toward the target while still producing a correct `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.backends import cudnn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import VisionDataset
from torchvision.transforms import InterpolationMode, v2
import torchvision.models as models
from torch.cuda import amp  # mixed precision

import torch._dynamo

torch._dynamo.config.suppress_errors = True

torch.manual_seed(3407)
torch.cuda.manual_seed(3407)

cudnn.deterministic = False
cudnn.benchmark = True

torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

test_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images/"

eff_img_size = 528
vit_img_size = 384
batch_size = 16
num_workers = min(8, os.cpu_count() or 1)
num_classes = 5
tta = False  # disabled to simplify inference


def get_backbone():
    model = models.efficientnet_b0(weights="DEFAULT")
    model.classifier = nn.Identity()
    return model.to(device)


try:
    vit_model = torch.load(
        "/kaggle/input/vit-v1-update/vit_v1_1.pt", map_location=device
    ).to(device)
except FileNotFoundError:
    print("vit model not found, using EfficientNet backbone.")
    vit_model = get_backbone()

try:
    eff_model = torch.load(
        "/kaggle/input/efficient-net/vit_cont_3.pt", map_location=device
    ).to(device)
except FileNotFoundError:
    print("efficient model not found, using EfficientNet backbone.")
    eff_model = get_backbone()

for param in vit_model.parameters():
    param.requires_grad = False
for param in eff_model.parameters():
    param.requires_grad = False

linear_head = nn.Linear(2 * 1280, num_classes).to(device)

vit_model.eval()
eff_model.eval()
linear_head.eval()

if hasattr(torch, "compile"):
    try:
        vit_model = torch.compile(vit_model)
        eff_model = torch.compile(eff_model)
        linear_head = torch.compile(linear_head)
    except Exception as e:
        print("torch.compile failed, proceeding with eager mode:", e)




## === cell 1
class CassavaDataset(VisionDataset):
    """Custom dataset for the Cassava data (test mode)."""

    def __init__(
        self,
        data_dir,
        vit_size,
        efficient_size,
        transform=None,
        ttas=None,
        img_size=384,
    ):
        super().__init__(root=data_dir)

        self.transform = transform
        self.images = sorted(
            [
                f
                for f in os.listdir(data_dir)
                if os.path.isfile(os.path.join(data_dir, f)) and not f.startswith(".")
            ]
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




## === cell 2
class CassavaTrainDataset(VisionDataset):
    """Dataset for training – returns images and integer labels."""

    def __init__(
        self,
        csv_path,
        img_dir,
        vit_size,
        efficient_size,
        transform=None,
        img_size=384,
    ):
        super().__init__(root=img_dir)
        self.df = pd.read_csv(csv_path)
        self.image_ids = self.df["image_id"].tolist()
        self.labels = self.df["label"].tolist()
        self.transform = transform
        self.cc = v2.CenterCrop((600, 600))
        self.resize_vit = v2.Resize(
            (vit_size, vit_size), interpolation=InterpolationMode.BICUBIC
        )
        self.resize_efficient = v2.Resize(
            (efficient_size, efficient_size), interpolation=InterpolationMode.BICUBIC
        )

    def __getitem__(self, idx):
        filename = self.image_ids[idx]
        label = self.labels[idx]
        img = Image.open(os.path.join(self.root, filename)).convert("RGB")
        img = self.cc(img)
        vit_img = self.resize_vit(img)
        eff_img = self.resize_efficient(img)

        if self.transform:
            vit_img = self.transform(vit_img)
            eff_img = self.transform(eff_img)

        return vit_img, eff_img, label

    def __len__(self):
        return len(self.image_ids)




## === cell 3
test_transforms = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

train_transforms = v2.Compose(
    [
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),
        v2.RandomRotation(degrees=30),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

if tta:
    ttas = [
        v2.RandomRotation(180),
        v2.RandomVerticalFlip(1),
        v2.RandomAffine(180),
        v2.RandomPerspective(p=1),
    ]
else:
    ttas = None

test_dataset = CassavaDataset(
    test_dir, vit_img_size, eff_img_size, transform=test_transforms, ttas=ttas
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size * 2,  # increased inference batch size for speed
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,
)

train_csv = "/kaggle/input/cassava-leaf-disease-classification/train.csv"
train_dir = "/kaggle/input/cassava-leaf-disease-classification/train_images/"

train_dataset = CassavaTrainDataset(
    train_csv,
    train_dir,
    vit_img_size,
    eff_img_size,
    transform=train_transforms,
)

print("Pre‑computing backbone features for training data...")
precompute_loader = DataLoader(
    train_dataset,
    batch_size=batch_size * 2,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,
)

all_feats = []
all_lbls = []
vit_model.eval()
eff_model.eval()
with torch.no_grad():
    for vit_imgs, eff_imgs, lbls in precompute_loader:
        vit_imgs = vit_imgs.to(device, non_blocking=True)
        eff_imgs = eff_imgs.to(device, non_blocking=True)
        vit_feat = vit_model(vit_imgs)  # [B, 1280]
        eff_feat = eff_model(eff_imgs)  # [B, 1280]
        feats = torch.cat([vit_feat, eff_feat], dim=1).cpu()  # [B, 2560]
        all_feats.append(feats)
        all_lbls.append(lbls)

train_features = torch.cat(all_feats)  # shape (N, 2560)
train_labels = torch.cat(all_lbls)  # shape (N,)

feature_dataset = TensorDataset(train_features, train_labels)
train_loader = DataLoader(
    feature_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,  # small tensor dataset – no need for many workers
    pin_memory=True,
)

normalizer = torch.nn.Softmax(dim=1)




## === cell 4
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.Adam(
    linear_head.parameters(),
    lr=1e-3,
    weight_decay=1e-4,
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=15, eta_min=1e-5
)

scaler = amp.GradScaler()  # mixed‑precision scaler

linear_head.train()

epochs = 15  # increased from 5 to allow better fitting
for epoch in range(epochs):
    epoch_loss = 0.0
    for feats, labels in train_loader:
        feats = feats.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        with amp.autocast():
            logits = linear_head(feats)  # [B, 5]
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()
    scheduler.step()
    print(f"Epoch {epoch+1}/{epochs}, loss: {epoch_loss/len(train_loader):.4f}")

vit_model.eval()
eff_model.eval()
linear_head.eval()
torch.cuda.empty_cache()  # free unused memory




## === cell 5
all_names = []
all_preds = []

with torch.no_grad():
    for batch_idx, (vit_inputs, eff_inputs, filenames) in enumerate(test_loader):
        vit_inputs = vit_inputs.to(device, non_blocking=True)
        eff_inputs = eff_inputs.to(device, non_blocking=True)

        with amp.autocast():
            vit_feats = vit_model(vit_inputs)
            eff_feats = eff_model(eff_inputs)
            logits = linear_head(torch.cat([vit_feats, eff_feats], dim=1))
            probs = normalizer(logits)

        pred_labels = torch.argmax(probs, dim=1).cpu().tolist()
        all_names.extend(filenames)
        all_preds.extend(pred_labels)




## === cell 6
my_submission = pd.DataFrame({"image_id": all_names, "label": all_preds})
my_submission.to_csv("submission.csv", index=False)
my_submission
