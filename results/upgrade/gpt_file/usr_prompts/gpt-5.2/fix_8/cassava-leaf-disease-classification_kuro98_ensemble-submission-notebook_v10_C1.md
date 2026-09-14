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

0.69768

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.05531) has done: 'I make the solution run end-to-end by (1) replacing the missing external model checkpoints with torchvision ImageNet backbones (keeping the same “two models + linear head ensemble” core idea), (2) fixing test dataset ordering and DataLoader shuffling so predictions align exactly with `sample_submission.csv`, and (3) making TTA deterministic and correctly averaged so it doesn’t change the submission length or misalign filenames. These changes directly address the FileNotFound/NameError crashes and the “wrong length” submission error. The code always write a valid `submission.csv` with the required columns and row count.'
- What this solution (achieved 0.10987) has done: 'I fix the runtime crash by aligning the ViT input resolution to what `torchvision.models.vit_b_16` expects (224×224), while keeping your two-backbone + concatenation + linear head ensemble logic unchanged. I also correct the linear head input dimension to match the actual output sizes of ViT-B/16 (1000) and EfficientNet-B0 (1000) so the forward pass is consistent. Finally, I keep the deterministic TTA behavior and ensure the written `submission.csv` is correctly aligned to `sample_submission.csv` with integer labels and the correct row count.'
- What this solution (achieved 0.7784) has done: 'Your current score is low because the linear head is never trained (it’s randomly initialized), so predictions are essentially random; the smallest legitimate step toward the target is to train only that linear head on the provided `train.csv` while keeping both ImageNet backbones frozen and keeping the same “two models + concatenation + linear head” core logic. I add a minimal training dataset/loader and a short training loop (cross-entropy on 5 classes), then reuse your existing deterministic TTA inference and submission alignment. I also keep image preprocessing consistent between train and test, and ensure the script still finishes under the time limit by training only the small head for a few epochs. This should move accuracy much closer to the target without changing the architecture or inference semantics.'
- What this solution (achieved 0.05531) has done: 'I make two minimal, score-relevant fixes while keeping your “two frozen ImageNet backbones + concatenation + linear head” core logic unchanged. First, I switch the EfficientNet input size from 528 to its native 224 to remove the train/test feature distribution mismatch and reduce wasted resizing (this typically gives a noticeable accuracy gain without changing the model). Second, I align preprocessing to the official torchvision weights for each backbone (same normalization/statistics) and use autocast during inference/training to stay comfortably under the time limit without changing semantics. Everything else (frozen backbones, same linear head, same TTA averaging, same submission alignment) stays the same and still writes a valid `submission.csv`.'
- What this solution (achieved 0.05531) has done: 'The main crash is coming from `weights.transforms()` returning a `torchvision.transforms.v2._presets.ImageClassification` object in your torchvision version, which does not expose a `.transforms` attribute as your `_strip_resize_crop()` expects; fixing this allow the preprocessing/transforms and DataLoaders to be constructed so later cells (training/inference) can run. I replace the transform-stripping logic with a minimal, stable implementation that keeps the same intent: you do explicit crop/resize in the Dataset, and only apply `ToImage -> ToDtype(float, scale=True) -> Normalize(mean/std)` in the collate step, using the weights’ canonical mean/std. This is a correctness/runtime fix and is score-positive because it ensures the models receive properly normalized inputs consistent with their pretrained weights. No changes to the core “two frozen ImageNet backbones + concatenation + linear head, optional deterministic TTA, train only head” approach are made.'
- What this solution (achieved 0.77093) has done: 'The crash comes from a mismatch between what `CassavaDataset.__getitem__` returns under TTA and what `test_collate` assumes: right now `__getitem__` applies `t(vit_img)` before `self.transform`, but `self.transform` is `None`, so it returns a single PIL image (not a list), and then `len(vit_items[0])` fails. I fix this by making `CassavaDataset` always return a list of PIL images when `ttas` is provided (independent of `transform`), and letting the collate function apply the correct normalization transforms per backbone. This is a minimal runtime/correctness fix that preserves your model/ensemble/training logic and should also improve score by actually enabling the intended deterministic TTA averaging. The rest of the pipeline (head training, inference, and submission alignment to `sample_submission.csv`) remains unchanged.'
- What this solution (achieved 0.69768) has done: 'I make two minimal, score-positive adjustments while keeping your “two frozen ImageNet backbones + concatenation + linear head trained on train.csv + deterministic TTA inference” core logic unchanged. First, I switch the training crop from a hard CenterCrop(600) to a RandomResizedCrop around 600 (still resized to 224), which adds essential train-time augmentation and reduces overfitting of the small linear head. Second, I add label smoothing to CrossEntropyLoss (small, stable regularization) and ensure all randomness is properly seeded across DataLoader workers so the augmentation is deterministic/reproducible. Everything else (models, head dimension, TTA, inference averaging, submission alignment) remains the same and it still writes a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import random

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
random.seed(3407)

cudnn.deterministic = False
cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

data_root = "/kaggle/input/cassava-leaf-disease-classification/"
train_csv_path = os.path.join(data_root, "train.csv")
train_dir = os.path.join(data_root, "train_images/")
test_dir = os.path.join(data_root, "test_images/")
sample_sub_path = os.path.join(data_root, "sample_submission.csv")

eff_img_size = 224
vit_img_size = 224

batch_size = 16
num_workers = 4
num_classes = 5
tta = True

train_head = True
head_epochs = 3
head_lr = 2e-3

label_smoothing = 0.05

import torchvision

vit_weights = torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1
eff_weights = torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1

vit_model = torchvision.models.vit_b_16(weights=vit_weights).to(device)
eff_model = torchvision.models.efficientnet_b0(weights=eff_weights).to(device)

linear_head = torch.nn.Linear(2000, num_classes).to(device)

use_amp = torch.cuda.is_available()
amp_dtype = torch.float16


def seed_worker(worker_id: int):
    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


g = torch.Generator()
g.manual_seed(3407)




## === cell 2
class CassavaTrainDataset(VisionDataset):
    """Train dataset returning (vit_img, eff_img, label)."""

    def __init__(self, df, data_dir, vit_size, efficient_size, transform=None):
        super().__init__(root=data_dir)
        self.df = df.reset_index(drop=True)
        self.transform = transform

        self.train_crop = v2.RandomResizedCrop(
            size=(600, 600),
            scale=(0.70, 1.0),
            ratio=(0.90, 1.10),
            interpolation=InterpolationMode.BICUBIC,
        )

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

        img = self.train_crop(img)
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
        vit_img: PIL.Image or list[PIL.Image] if TTA enabled
        eff_img: PIL.Image or list[PIL.Image] if TTA enabled
        filename: image filename
    """

    def __init__(self, data_dir, vit_size, efficient_size, transform=None, ttas=None):
        super().__init__(root=data_dir)
        self.transform = (
            transform  # kept for backward-compat; collate applies normalization
        )
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

        if self.ttas is not None:
            vit_img = [t(vit_img) for t in self.ttas]
            eff_img = [t(eff_img) for t in self.ttas]

        return vit_img, eff_img, filename

    def __len__(self):
        return len(self.images)




## === cell 3
def _norm_only_from_weights(weights):
    mean = weights.meta.get("mean", [0.485, 0.456, 0.406])
    std = weights.meta.get("std", [0.229, 0.224, 0.225])
    return v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=mean, std=std),
        ]
    )


vit_transforms = _norm_only_from_weights(vit_weights)
eff_transforms = _norm_only_from_weights(eff_weights)


class DualTransform:
    def __init__(self, vit_t, eff_t):
        self.vit_t = vit_t
        self.eff_t = eff_t

    def __call__(self, x):
        return x


dual_transform = DualTransform(vit_transforms, eff_transforms)

if tta:
    ttas = [
        v2.Identity(),
        v2.RandomHorizontalFlip(p=1.0),
        v2.RandomVerticalFlip(p=1.0),
    ]
else:
    ttas = None

train_df = pd.read_csv(train_csv_path)

train_dataset = CassavaTrainDataset(
    train_df, train_dir, vit_img_size, eff_img_size, transform=None
)
test_dataset = CassavaDataset(
    test_dir, vit_img_size, eff_img_size, transform=None, ttas=ttas
)


def train_collate(batch):
    vit_imgs, eff_imgs, labels = zip(*batch)
    vit_imgs = torch.stack([dual_transform.vit_t(im) for im in vit_imgs], dim=0)
    eff_imgs = torch.stack([dual_transform.eff_t(im) for im in eff_imgs], dim=0)
    labels = torch.stack(labels, dim=0)
    return vit_imgs, eff_imgs, labels


def test_collate(batch):
    vit_items, eff_items, fnames = zip(*batch)
    if tta:
        T = len(vit_items[0])
        vit_out = [
            torch.stack(
                [dual_transform.vit_t(vit_items[b][t]) for b in range(len(batch))],
                dim=0,
            )
            for t in range(T)
        ]
        eff_out = [
            torch.stack(
                [dual_transform.eff_t(eff_items[b][t]) for b in range(len(batch))],
                dim=0,
            )
            for t in range(T)
        ]
        return vit_out, eff_out, list(fnames)
    else:
        vit_imgs = torch.stack([dual_transform.vit_t(im) for im in vit_items], dim=0)
        eff_imgs = torch.stack([dual_transform.eff_t(im) for im in eff_items], dim=0)
        return vit_imgs, eff_imgs, list(fnames)


train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
    collate_fn=train_collate,
    worker_init_fn=seed_worker,
    generator=g,
    persistent_workers=(num_workers > 0),
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    collate_fn=test_collate,
    worker_init_fn=seed_worker,
    generator=g,
    persistent_workers=(num_workers > 0),
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

    criterion = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    for epoch in range(head_epochs):
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for vit_inputs, eff_inputs, labels in train_loader:
            vit_inputs = vit_inputs.to(device, non_blocking=True)
            eff_inputs = eff_inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.no_grad(), torch.autocast(
                device_type="cuda", dtype=amp_dtype, enabled=use_amp
            ):
                vit_outputs = vit_model(vit_inputs)  # [B,1000]
                eff_outputs = eff_model(eff_inputs)  # [B,1000]

            logit_inputs = torch.cat(
                [vit_outputs.float(), eff_outputs.float()], dim=1
            )  # [B,2000]
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
            vit_inputs = torch.cat(vit_inputs, dim=0).to(device, non_blocking=True)
            eff_inputs = torch.cat(eff_inputs, dim=0).to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                vit_outputs = vit_model(vit_inputs)
                eff_outputs = eff_model(eff_inputs)

            vit_outputs = vit_outputs.float()
            eff_outputs = eff_outputs.float()

            vit_batch_logits = torch.stack(
                torch.split(vit_outputs, base_batch_size), dim=0
            )  # [T,B,1000]
            vit_mean_logits = torch.mean(vit_batch_logits, dim=0)  # [B,1000]

            eff_batch_logits = torch.stack(
                torch.split(eff_outputs, base_batch_size), dim=0
            )  # [T,B,1000]
            eff_mean_logits = torch.mean(eff_batch_logits, dim=0)  # [B,1000]

            logit_inputs = torch.cat([vit_mean_logits, eff_mean_logits], dim=1)
            outputs = linear_head(logit_inputs)

            probs = normalizer(outputs)
            pred_labels = torch.argmax(probs, 1).tolist()
        else:
            vit_inputs = vit_inputs.to(device, non_blocking=True)
            eff_inputs = eff_inputs.to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                vit_outputs = vit_model(vit_inputs)
                eff_outputs = eff_model(eff_inputs)

            logit_inputs = torch.cat([vit_outputs.float(), eff_outputs.float()], dim=1)
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
