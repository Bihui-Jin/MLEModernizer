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

0.8939256572982774

# 6. Current score

0.72272

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.05531) has done: 'I fix the pipeline so it runs end-to-end in the Kaggle filesystem by removing the hard dependency on missing external `.pt` model files and replacing them with a lightweight, local torchvision model that can be loaded without extra inputs. I also fix submission-length/alignment issues by iterating the test set in a deterministic order (no shuffle) and writing predictions in exactly the same order as `sample_submission.csv` expects. To keep runtime safe and produce a valid CSV reliably, the model run in `eval()` with `no_grad()` and map ImageNet logits to 5 classes via a fixed linear head (score be modest but valid, and the main goal here is correctness + submission creation). Finally, I add robust path handling so it works whether the dataset is under `/kaggle/input/...` or mirrored under `/kaggle/data/...`.'
- What this solution (achieved 0.66069) has done: 'Your current 0.05531 score is mainly because the model head is initialized to all-zeros, so it predicts the same class for every image. To move the score toward the 0.8939 target while keeping your core inference-only pipeline intact, I train only the existing linear projection head (`proj`) on `train.csv` using frozen EfficientNet-B0 features, then reuse the same preprocessing and ordered test inference to write `submission.csv`. This is a minimal change: same backbone, same head type, same transforms, and no architectural changes—just fitting the already-present head on the provided training labels. I also ensure label types and file alignment remain identical to `sample_submission.csv`.'
- What this solution (achieved 0.71824) has done: 'Main bottlenecks are (1) repeatedly decoding/resizing images on CPU for every epoch, (2) running the frozen EfficientNet backbone inside the training loop for all 18.7k images × 6 epochs, and (3) DataLoader overhead without persistent workers/prefetch. To preserve identical training semantics while cutting runtime drastically, the optimized script precomputes the frozen backbone’s 1000-dim features for the entire training set once (same transforms), then trains the linear projection on those cached features for the same number of epochs/batches. It also enables DataLoader performance flags (persistent_workers, prefetch_factor) and avoids redundant model_b image work since model_b inputs are never used. Inference logic and submission formatting remain unchanged.'
- What this solution (achieved 0.76756) has done: 'Your current gap to the target is large (0.71824 → 0.89393), so the smallest score-moving change is to fix the training mismatch: EfficientNet’s forward output is 1000-class logits, not stable “features”, so training `proj` on those logits is weak. We keep the same backbone (EfficientNet-B0) and the same linear head + loss/training loop, but we extract proper penultimate features (1280-d) from the model’s `features + avgpool` and update `proj` accordingly. This preserves the approach (frozen backbone + train only a linear head) while materially improving separability and should move accuracy substantially toward your target. Submission ordering/format and the rest of the pipeline remain unchanged.'
- What this solution (achieved 0.76046) has done: 'I fix the immediate runtime blockers so the notebook runs end-to-end: (1) `weights.meta["mean"/"std"]` is not present in this torchvision version, so I use the standard ImageNet mean/std (matching the pretrained weights) to build `train_transforms`. Because cell 2 currently crashes, downstream variables like `train_transforms`, `sample_sub`, and `test_loader` never get defined; fixing cell 2 resolves the cascade of `NameError`s in later cells. I keep the core approach unchanged (frozen EfficientNet-B0 feature extractor + trained linear head) and preserve submission ordering by iterating exactly in `sample_submission.csv` order. Finally, I add a tiny safety check to ensure all mapped labels are present before writing `submission.csv`.'
- What this solution (achieved 0.76495) has done: 'Your current score (0.76046) is well below the target (0.89393), so we should improve accuracy with the smallest change that keeps your frozen EfficientNet-B0 + linear head approach intact. The biggest low-risk gain here is to remove the train/test preprocessing mismatch: you currently train with strong random crops but infer with EfficientNet’s center-crop preprocessing, which can hurt generalization for a linear head. I switch training to use the exact same `weights.transforms()` preprocessing as test (no augmentation), keeping the same backbone, head, loss, optimizer, and epoch count. This typically increases stability and accuracy for this “precompute features + train linear head” pipeline without altering core logic.'
- What this solution (achieved 0.71749) has done: 'Your current score (0.76495) is well below the target (0.89393), so we should improve accuracy with the smallest change that keeps your frozen EfficientNet-B0 + linear head approach intact. The lowest-risk gain is to fix class-imbalance bias in the linear-head training by using weighted cross-entropy computed from `train.csv` label frequencies; this does not change the model, features, or loop structure, only the loss weighting. I also make the run fully deterministic (same semantics, just reduced randomness) so the score is more stable and easier to move toward the target. Everything else (precompute frozen features once, train only `proj`, same preprocessing for train/test, and submission ordering) remains unchanged.'
- What this solution (achieved 0.72272) has done: 'Your current score (0.71749) is far below the target (0.89393), so we should improve accuracy with the smallest change that preserves your core approach (frozen EfficientNet-B0 feature extractor + trained linear head on cached features). The most direct, low-risk gain is to train the linear head with a simple validation split and pick the best-epoch weights (still same model, same loss, same optimizer, same number of epochs; we just keep the best checkpoint instead of the last one). This typically avoids ending on an overfit epoch and moves accuracy upward without changing inference semantics or submission formatting. I also keep your deterministic setup and ensure feature extraction is still done once (fast) and the final submission ordering remains identical to `sample_submission.csv`.'

# 9. Code solution

## === cell 0
import os
import random

import pandas as pd
import torch
from PIL import Image
from torch.backends import cudnn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import VisionDataset
from torchvision.transforms import v2
import torchvision

torch.manual_seed(3407)
random.seed(3407)
if torch.cuda.is_available():
    torch.cuda.manual_seed(3407)
    torch.cuda.manual_seed_all(3407)

cudnn.deterministic = True
cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

DATA_ROOT_CANDIDATES = [
    "/kaggle/input/cassava-leaf-disease-classification",
    "/kaggle/data/cassava-leaf-disease-classification",
]
DATA_ROOT = None
for p in DATA_ROOT_CANDIDATES:
    if os.path.exists(p):
        DATA_ROOT = p
        break
if DATA_ROOT is None:
    raise FileNotFoundError(f"Could not find dataset root in: {DATA_ROOT_CANDIDATES}")

test_dir = os.path.join(DATA_ROOT, "test_images")
train_dir = os.path.join(DATA_ROOT, "train_images")
sample_sub_path = os.path.join(DATA_ROOT, "sample_submission.csv")
train_csv_path = os.path.join(DATA_ROOT, "train.csv")

weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
preprocess = weights.transforms()  # Resize+CenterCrop+Normalize
model_input_size = 224

model_b_img_size = model_input_size
model_a_img_size = model_input_size
batch_size = 64
num_workers = 2 if os.cpu_count() is None else min(4, os.cpu_count())
num_classes = 5
tta = False

backbone = torchvision.models.efficientnet_b0(weights=weights).to(device).eval()
feat_dim = backbone.classifier[1].in_features  # 1280

proj = torch.nn.Linear(feat_dim, num_classes, bias=True).to(device)
torch.nn.init.normal_(proj.weight, mean=0.0, std=0.01)
torch.nn.init.zeros_(proj.bias)

pin_memory = torch.cuda.is_available()
loader_kwargs = dict(
    num_workers=num_workers,
    pin_memory=pin_memory,
)
if num_workers > 0:
    loader_kwargs.update(dict(persistent_workers=True, prefetch_factor=4))




## === cell 1
class CassavaDataset(VisionDataset):
    """Custom dataset for Cassava images (test/inference-style)."""

    def __init__(
        self,
        data_dir,
        model_a_size,
        model_b_size,
        transform=None,
        ttas=None,
        img_size=384,
        images=None,
    ):
        super().__init__(root=data_dir)
        self.transform = transform
        self.ttas = ttas

        if images is None:
            self.images = sorted(
                [f for f in os.listdir(data_dir) if f.lower().endswith(".jpg")]
            )
        else:
            self.images = list(images)

        self.cc = None
        self.resize_model_a = None
        self.resize_model_b = None

    def __getitem__(self, idx):
        filename = self.images[idx]
        img = Image.open(os.path.join(self.root, filename)).convert("RGB")

        model_a_img = img
        model_b_img = img

        if self.ttas is not None and self.transform is not None:
            model_a_img = [self.transform(t(model_a_img)) for t in self.ttas]
            model_b_img = [self.transform(t(model_b_img)) for t in self.ttas]
        elif self.transform:
            model_a_img = self.transform(model_a_img)
            model_b_img = self.transform(model_b_img)

        return model_a_img, model_b_img, filename

    def __len__(self):
        return len(self.images)


class CassavaTrainDataset(VisionDataset):
    """Train dataset returning image tensor + label, using same preprocessing as test."""

    def __init__(self, data_dir, df, model_a_size, model_b_size, transform=None):
        super().__init__(root=data_dir)
        self.df = df.reset_index(drop=True)
        self.transform = transform

        self.cc = None
        self.resize_model_a = None
        self.resize_model_b = None

    def __getitem__(self, idx):
        filename = self.df.loc[idx, "image_id"]
        y = int(self.df.loc[idx, "label"])
        img = Image.open(os.path.join(self.root, filename)).convert("RGB")

        model_a_img = img
        model_b_img = img

        if self.transform:
            model_a_img = self.transform(model_a_img)
            model_b_img = self.transform(model_b_img)

        return model_a_img, model_b_img, torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.df)


class CassavaTrainDatasetAOnly(VisionDataset):
    def __init__(self, data_dir, df, model_a_size, transform=None):
        super().__init__(root=data_dir)
        self.df = df.reset_index(drop=True)
        self.transform = transform

        self.cc = None
        self.resize_model_a = None

    def __getitem__(self, idx):
        filename = self.df.loc[idx, "image_id"]
        y = int(self.df.loc[idx, "label"])
        img = Image.open(os.path.join(self.root, filename)).convert("RGB")
        model_a_img = img
        if self.transform:
            model_a_img = self.transform(model_a_img)
        return model_a_img, torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.df)




## === cell 2
test_transforms = preprocess
train_transforms = preprocess

if tta:
    ttas = [
        v2.RandomRotation(180),
        v2.RandomVerticalFlip(1),
        v2.RandomPerspective(p=1),
    ]
else:
    ttas = None

sample_sub = pd.read_csv(sample_sub_path)
test_images_order = sample_sub["image_id"].tolist()

test_dataset = CassavaDataset(
    test_dir,
    model_a_img_size,
    model_b_img_size,
    transform=test_transforms,
    ttas=ttas,
    images=test_images_order,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    **loader_kwargs,
)




## === cell 3
def extract_effnet_features(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    x = model.features(x)
    x = model.avgpool(x)
    x = torch.flatten(x, 1)  # [B, 1280]
    return x


train_df = pd.read_csv(train_csv_path)
if not {"image_id", "label"}.issubset(train_df.columns):
    raise RuntimeError("train.csv must contain image_id and label columns")

perm = torch.randperm(
    len(train_df), generator=torch.Generator().manual_seed(3407)
).tolist()
val_frac = 0.10
val_n = max(1, int(len(train_df) * val_frac))
val_idx = set(perm[:val_n])
train_idx = [i for i in perm[val_n:]]

train_df_tr = train_df.iloc[train_idx].reset_index(drop=True)
train_df_va = train_df.iloc[list(val_idx)].reset_index(drop=True)

label_counts = train_df_tr["label"].value_counts().sort_index()
for k in range(num_classes):
    if k not in label_counts.index:
        raise RuntimeError(
            f"Missing class {k} in train split; cannot compute weights safely."
        )
counts = label_counts.values.astype("float64")
inv_freq = counts.sum() / (counts + 1e-12)
class_weights = inv_freq / inv_freq.mean()
class_weights_t = torch.tensor(class_weights, dtype=torch.float32, device=device)

train_img_dataset_tr = CassavaTrainDatasetAOnly(
    train_dir,
    train_df_tr,
    model_a_img_size,
    transform=train_transforms,
)
train_img_dataset_va = CassavaTrainDatasetAOnly(
    train_dir,
    train_df_va,
    model_a_img_size,
    transform=train_transforms,
)

feat_loader_tr = DataLoader(
    train_img_dataset_tr,
    batch_size=batch_size,
    shuffle=False,
    **loader_kwargs,
)
feat_loader_va = DataLoader(
    train_img_dataset_va,
    batch_size=batch_size,
    shuffle=False,
    **loader_kwargs,
)

for p in backbone.parameters():
    p.requires_grad = False
backbone.eval()


def precompute_feats(loader):
    all_feats = []
    all_y = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            feats = extract_effnet_features(backbone, x)  # [B, 1280]
            all_feats.append(feats.detach().cpu())
            all_y.append(y.cpu())
    return torch.cat(all_feats, dim=0), torch.cat(all_y, dim=0)


train_feats_tr, train_y_tr = precompute_feats(feat_loader_tr)
train_feats_va, train_y_va = precompute_feats(feat_loader_va)

train_tensor_ds_tr = TensorDataset(train_feats_tr, train_y_tr)
train_loader = DataLoader(
    train_tensor_ds_tr,
    batch_size=batch_size,
    shuffle=True,  # preserve training semantics
    **loader_kwargs,
)

proj.train()
optimizer = torch.optim.AdamW(proj.parameters(), lr=2e-3, weight_decay=1e-2)
criterion = torch.nn.CrossEntropyLoss(weight=class_weights_t)

epochs = 6
best_state = None
best_val_acc = -1.0

for ep in range(epochs):
    running_loss = 0.0
    n = 0
    for feats_cpu, y_cpu in train_loader:
        feats = feats_cpu.to(device, non_blocking=True)
        y = y_cpu.to(device, non_blocking=True)

        logits = proj(feats)  # [B, 5]
        loss = criterion(logits, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        running_loss += float(loss.item()) * feats.size(0)
        n += feats.size(0)

    proj.eval()
    with torch.no_grad():
        feats = train_feats_va.to(device, non_blocking=True)
        y = train_y_va.to(device, non_blocking=True)
        logits = proj(feats)
        pred = torch.argmax(logits, dim=1)
        val_acc = (pred == y).float().mean().item()

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_state = {k: v.detach().cpu().clone() for k, v in proj.state_dict().items()}

    proj.train()
    print(
        f"epoch {ep+1}/{epochs} - train_loss: {running_loss/max(n,1):.5f} - val_acc: {val_acc:.5f} - best_val_acc: {best_val_acc:.5f}"
    )

if best_state is not None:
    proj.load_state_dict({k: v.to(device) for k, v in best_state.items()})

proj.eval()



## === cell 4
all_names = []
all_preds = []

backbone.eval()
proj.eval()

with torch.no_grad():
    for batch_idx, (model_a_inputs, model_b_inputs, filenames) in enumerate(
        test_loader
    ):
        if tta:
            bsz = len(filenames)
            x = torch.cat(model_a_inputs, dim=0).to(device, non_blocking=True)
            feats = extract_effnet_features(backbone, x)
            logits_5 = proj(feats)

            batch_logits = torch.stack(torch.split(logits_5, bsz), dim=0)
            mean_logits = torch.mean(batch_logits, dim=0)

            pred_labels = torch.argmax(mean_logits, 1).tolist()
        else:
            x = model_a_inputs.to(device, non_blocking=True)
            feats = extract_effnet_features(backbone, x)
            logits_5 = proj(feats)
            pred_labels = torch.argmax(logits_5, 1).tolist()

        all_names.extend(list(filenames))
        all_preds.extend(pred_labels)

if len(all_names) != len(sample_sub):
    raise RuntimeError(
        f"Prediction count mismatch: got {len(all_names)} preds, expected {len(sample_sub)}"
    )



## === cell 5
pred_map = dict(zip(all_names, all_preds))
submission = sample_sub.copy()
submission["label"] = submission["image_id"].map(pred_map)

missing = submission["label"].isna().sum()
if missing:
    raise RuntimeError(
        f"Found {missing} test images without predictions (mapping failed)."
    )

submission["label"] = submission["label"].astype(int)

out_path = "submission.csv"
submission.to_csv(out_path, index=False)

print("Wrote:", out_path)
print(submission.head())
print("rows:", len(submission), "cols:", submission.shape[1])
