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

0.8990631610758537

# 6. Current score

0.76719

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.61099) has done: 'I wrap the model loading in a safe try‑except so missing weight files no longer raise an error, then fall back to a simple baseline that predicts the most common label from the training set. The inference loop is replaced with logic that reads the test image names (or the sample submission file) and assigns this majority label, guaranteeing a correctly‑sized CSV submission with the required columns.'
- What this solution (achieved 0.61099) has done: 'I disable test‑time augmentation because the aggressive random transforms can hurt accuracy, and keep the existing model‑based inference path (which already outperforms the majority‑label fallback). This minimal change keeps the core logic intact while moving the validation score closer to the target.'
- What this solution (achieved 0.24925) has done: 'I add a lightweight “class‑mean” baseline that is used when the pretrained models are unavailable.  
It computes average RGB values per disease from a random subset of the training images and assigns each test image to the nearest class mean, which is expected to improve accuracy beyond the simple majority‑label fallback while keeping the original model‑based path unchanged.'
- What this solution (achieved 0.7216) has done: 'I add a lightweight fallback that trains a tiny linear head on top of frozen ImageNet‑pretrained EfficientNet‑B0 features when the custom ViT/EfficientNet models are missing. This keeps the original flow but replaces the very weak class‑mean baseline with a learned classifier, which should raise accuracy substantially toward the target while preserving the core logic. The change is limited to the fallback section and adds only the necessary imports and tiny training loop.'
- What this solution (achieved 0.76532) has done: 'I boost the fallback classifier by training it on the full training set instead of a 2k sample and by doubling the training epochs from 5 to 10. These minimal adjustments keep the original architecture and inference flow unchanged while giving the linear head more data and learning time, which should raise validation accuracy toward the target score.'
- What this solution (achieved 0.77167) has done: 'We keep the overall structure and logic unchanged but accelerate the fallback training by using mixed‑precision (AMP) and a few DataLoader tweaks that reduce CPU‑GPU transfer overhead. Mixed‑precision yields the same predictions up to negligible floating‑point differences while cutting compute time, especially on GPU. Adding `persistent_workers=True` and increasing `num_workers` lets the loader keep workers alive between epochs, avoiding repeated process startup costs.'
- What this solution (achieved 0.76457) has done: 'I increase the training epochs of the fallback EfficientNet‑B0 linear head from 10 to 12, which modestly strengthens the learned classifier without altering the overall architecture or inference flow. This change is small, keeps the core logic intact, and is expected to raise the validation accuracy toward the target score.'
- What this solution (achieved 0.76719) has done: 'I add lightweight data‑augmentation to the EfficientNet fallback transform and train the linear head for a few more epochs (15 instead of 12). These changes keep the overall architecture and inference flow unchanged while giving the fallback model a modest boost in generalisation, which should raise the validation accuracy toward the target score.'

# 9. Code solution

## === cell 0
import os

import pandas as pd
import torch
from torch.backends import cudnn
from torch.cuda.amp import autocast, GradScaler  # <-- added for mixed precision
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision.datasets import VisionDataset
from torchvision.transforms import InterpolationMode, v2

from torchvision import models

torch.manual_seed(3407)
torch.cuda.manual_seed(3407)

cudnn.deterministic = False
cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

test_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images/"

eff_img_size = 528
vit_img_size = 384
batch_size = 16
num_workers = 8  # increased to utilize more CPU cores
num_classes = 5
tta = False

try:
    vit_model = torch.load(
        "/kaggle/input/vit-v1-update/vit_v1_1.pt", map_location=device
    ).to(device)
except FileNotFoundError:
    print("vit model not found, using fallback.")
    vit_model = None

try:
    eff_model = torch.load(
        "/kaggle/input/efficient-net/vit_cont_3.pt", map_location=device
    ).to(device)
except FileNotFoundError:
    print("efficient model not found, using fallback.")
    eff_model = None

try:
    linear_head = torch.load(
        "/kaggle/input/linear-head/linear_cls.pt", map_location=device
    ).to(device)
except FileNotFoundError:
    print("linear head not found, using fallback.")
    linear_head = None




## === cell 1
class CassavaDataset(VisionDataset):
    """Custom dataset for the Cassava data."""

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
        self.images = sorted(os.listdir(data_dir))
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
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,  # keep workers alive across epochs/inference
)

normalizer = torch.nn.Softmax(dim=1)




## === cell 3
train_csv_path = "/kaggle/input/cassava-leaf-disease-classification/train.csv"
train_df = pd.read_csv(train_csv_path)
majority_label = int(train_df["label"].mode()[0])
print(f"Majority label from training data: {majority_label}")

train_images_dir = "/kaggle/input/cassava-leaf-disease-classification/train_images/"

train_sample = train_df.sample(n=min(2000, len(train_df)), random_state=42)

baseline_preproc = v2.Compose(
    [
        v2.CenterCrop((600, 600)),
        v2.Resize((384, 384), interpolation=InterpolationMode.BICUBIC),
        v2.ToTensor(),  # yields values in [0, 1]
    ]
)

class_sums = torch.zeros(num_classes, 3)
class_counts = torch.zeros(num_classes)

for _, row in train_sample.iterrows():
    img_path = os.path.join(train_images_dir, row["image_id"])
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception:
        continue  # skip corrupted files
    tensor = baseline_preproc(img)  # C x H x W
    mean_vec = tensor.mean(dim=(1, 2))  # per‑channel mean
    label = int(row["label"])
    class_sums[label] += mean_vec
    class_counts[label] += 1

class_means = class_sums / class_counts.unsqueeze(1).clamp(min=1)

all_names = []
all_preds = []

if vit_model is not None and eff_model is not None:
    vit_model.eval()
    eff_model.eval()
    if linear_head is not None:
        linear_head.eval()

    with torch.no_grad():
        for batch_idx, (vit_inputs, eff_inputs, filenames) in enumerate(test_loader):
            batch_sz = len(filenames)

            if tta:
                vit_inputs, eff_inputs, filenames = (
                    torch.cat(vit_inputs, dim=0).to(device),
                    torch.cat(eff_inputs, dim=0).to(device),
                    list(filenames),
                )

                vit_outputs = vit_model(vit_inputs)
                eff_outputs = eff_model(eff_inputs)

                vit_batch_logits = torch.stack(
                    torch.split(vit_outputs, batch_sz), dim=0
                )
                vit_mean_logits = torch.mean(vit_batch_logits, dim=0)

                eff_batch_logits = torch.stack(
                    torch.split(eff_outputs, batch_sz), dim=0
                )
                eff_mean_logits = torch.mean(eff_batch_logits, dim=0)

                outputs = 0.45 * vit_mean_logits + 0.55 * eff_mean_logits
                mean_preds = normalizer(outputs)
                pred_labels = torch.argmax(mean_preds, 1).tolist()
            else:
                vit_inputs, eff_inputs, filenames = (
                    vit_inputs.to(device),
                    eff_inputs.to(device),
                    list(filenames),
                )

                vit_outputs = vit_model(vit_inputs)
                eff_outputs = eff_model(eff_inputs)

                outputs = (vit_outputs + eff_outputs) / 2
                preds = normalizer(outputs)
                pred_labels = torch.argmax(preds, 1).tolist()

            all_names.extend(filenames)
            all_preds.extend(pred_labels)
else:
    print(
        "Both pretrained models missing – training lightweight EfficientNet fallback."
    )

    class SimpleCassavaTrain(Dataset):
        def __init__(self, df, img_dir, transform):
            self.df = df.reset_index(drop=True)
            self.img_dir = img_dir
            self.transform = transform

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            img_path = os.path.join(self.img_dir, row["image_id"])
            img = Image.open(img_path).convert("RGB")
            img = self.transform(img)
            label = int(row["label"])
            return img, label

    effnet_transform = v2.Compose(
        [
            v2.CenterCrop((600, 600)),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
            v2.ToTensor(),
            v2.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    effnet_full = models.efficientnet_b0(
        weights=models.EfficientNet_B0_Weights.DEFAULT
    ).to(device)
    effnet_full.eval()
    for param in effnet_full.parameters():
        param.requires_grad = False

    feature_extractor = torch.nn.Sequential(
        effnet_full.features, torch.nn.AdaptiveAvgPool2d(1)
    ).to(device)

    train_subset = train_df  # use all available training data

    train_dataset = SimpleCassavaTrain(train_subset, train_images_dir, effnet_transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    feature_dim = 1280  # EfficientNet‑B0 final channel count
    classifier = torch.nn.Linear(feature_dim, num_classes).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)

    scaler = GradScaler()  # <-- mixed‑precision scaler

    classifier.train()
    for epoch in range(15):
        epoch_loss = 0.0
        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            with autocast():  # <-- mixed‑precision forward
                with torch.no_grad():
                    feats = feature_extractor(imgs)  # (B, C, 1, 1)
                    feats = feats.view(feats.size(0), -1)  # (B, C)

                logits = classifier(feats)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()  # <-- scaled backward
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

    classifier.eval()

    sample_sub_path = (
        "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
    )
    sample_sub = pd.read_csv(sample_sub_path)
    test_names = sample_sub["image_id"].tolist()
    all_names = test_names

    for img_name in test_names:
        img_path = os.path.join(test_dir, img_name)
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            all_preds.append(majority_label)
            continue
        img_tensor = effnet_transform(img).unsqueeze(0).to(device)  # (1, C, H, W)

        with torch.no_grad(), autocast():  # <-- mixed‑precision inference
            feats = feature_extractor(img_tensor)
            feats = feats.view(feats.size(0), -1)
            logits = classifier(feats)
            pred = torch.argmax(logits, dim=1).item()
        all_preds.append(int(pred))

print(f"Generated predictions for {len(all_names)} images.")




## === cell 4
submission_path = "submission.csv"
my_submission = pd.DataFrame({"image_id": all_names, "label": all_preds})
my_submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")
my_submission.head()
