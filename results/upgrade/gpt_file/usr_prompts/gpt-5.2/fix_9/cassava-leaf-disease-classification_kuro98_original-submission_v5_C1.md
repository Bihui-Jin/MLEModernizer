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

0.858114233907525

# 6. Current score

0.76607

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.05531) has done: 'I fix the missing model file issue by switching from the unavailable `torch.load("/kaggle/input/vit-trial/vit_trial.pt")` to a built-in torchvision ViT (same overall inference flow: softmax + argmax). I also fix the submission-length error by ensuring we predict exactly once per test image, in the exact order of `sample_submission.csv`, and by using `shuffle=False` in the test DataLoader. Finally, I make image loading robust (RGB conversion) and ensure the CSV has the required columns and row count.'
- What this solution (achieved 0.2216) has done: 'I fix the runtime error by making the inference transform use the exact image size expected by `vit_b_16` (224) rather than 384, which currently triggers the assertion inside torchvision’s ViT forward pass. I keep the model/inference flow the same (softmax + argmax, same DataLoader ordering) and only align preprocessing to the pretrained weights’ recommended pipeline so predictions are meaningful and the score moves up toward the target. I also ensure `vit_model` is put on the correct device before inference and keep submission alignment exactly matching `sample_submission.csv`. These changes are minimal, directly address the crash, and should substantially improve accuracy versus the current near-random score.'
- What this solution (achieved 0.75747) has done: 'Your current score is far below the target (0.2216 vs 0.8581), so we should legitimately improve model performance with minimal changes that keep the same ViT-B/16 architecture and the same inference semantics (softmax + argmax). The biggest issue is that your classification head is randomly initialized (you replaced the 1000-class ImageNet head with a new 5-class head but never trained/loaded cassava weights), which makes predictions near-random; we fix this by training only the new head briefly on `train.csv` while freezing the backbone. We also switch preprocessing to the exact transforms recommended by the pretrained weights (same resizing intent, but using the official weights pipeline for better feature compatibility) and keep deterministic ordering to produce a correct submission. The result should move accuracy substantially toward the target without changing the core model, loss (cross-entropy), or overall approach.'
- What this solution (achieved 0.76794) has done: 'Your current score (0.75747) is below the target (0.85811), so we should make a small, legitimate improvement without changing the ViT-B/16 core logic. The biggest low-risk gain is to stop training on the full training set blindly and instead use a stratified validation split to tune only the head-training hyperparameters (epochs and lr) to a setting that generalizes better, then retrain the head once on the full data with the chosen setting. This keeps the same architecture, loss (cross-entropy), and training loop style (simple supervised head training), but typically improves generalization and moves accuracy upward toward the target. I also enable AMP during training/inference to fit a few more epochs within the time budget without changing semantics, and keep submission ordering exactly aligned to `sample_submission.csv`.'
- What this solution (achieved 0.76719) has done: 'Your main timeout is coming from repeatedly running the full ViT encoder for every batch during head-only training (two hyperparameter trials + final fit), plus slow Python/PIL image loading without persistent workers/prefetch. To keep the exact same model and training semantics, we can cache the frozen-backbone features (the CLS embedding after the encoder) once for each image and then train the head on those cached tensors (this is mathematically equivalent to the current `_forward_frozen_backbone` path since the backbone is frozen and run in `no_grad`). We also speed up data loading by enabling `persistent_workers`, `prefetch_factor`, and keeping deterministic settings unchanged. Inference stays the same except for faster DataLoader settings; no approximations, no fewer epochs, no architecture changes.'
- What this solution (achieved 0.76906) has done: 'Your current score (0.76719) is below the target (0.85811), so we should make a small, legitimate improvement without changing the ViT-B/16 core or the head-only training approach. The biggest low-risk gain is to make the frozen-feature extraction *numerically consistent* with ViT’s normal forward path by applying the encoder’s layernorm (`encoder.ln`) before the head; your current feature cache skips that, which hurts head training quality. I also ensure AMP scaler usage is correct on CPU-only runs and keep all ordering/submission alignment identical. These changes keep the exact same model, loss, and training loop semantics, but should move accuracy upward toward the target.'
- What this solution (achieved 0.76607) has done: 'We make two minimal, score-relevant fixes while preserving your ViT-B/16 + frozen-backbone + head-only training approach. First, your cached features are normalized with `encoder.ln`, but your inference path uses `vit_model(inputs)` (which applies the correct internal normalization); this mismatch can hurt, so we make feature extraction match the *exact* forward path by using `vit_model.forward_features()` when available. Second, your “full” feature extraction currently uses a shuffled loader, which can slightly inject noise via batch ordering and makes caching less reproducible; we extract full-data features with `shuffle=False` and only shuffle in the feature DataLoader. These changes keep the same model, loss, and training semantics, but should legitimately improve generalization toward the target.'

# 9. Code solution

## === cell 0
import os

import pandas as pd
import torch
from PIL import Image
from torch.backends import cudnn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import VisionDataset
from torchvision.transforms import v2

torch.manual_seed(3407)
if torch.cuda.is_available():
    torch.cuda.manual_seed(3407)

cudnn.deterministic = True
cudnn.benchmark = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

DATA_ROOT = "/kaggle/input/cassava-leaf-disease-classification"
train_csv_path = f"{DATA_ROOT}/train.csv"
train_dir = f"{DATA_ROOT}/train_images/"
test_dir = f"{DATA_ROOT}/test_images/"
sample_sub_path = f"{DATA_ROOT}/sample_submission.csv"

batch_size = 32
num_workers = 4
num_classes = 5
tta = False

use_amp = torch.cuda.is_available()
scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None

from torchvision.models import vit_b_16, ViT_B_16_Weights

weights = ViT_B_16_Weights.IMAGENET1K_V1
vit_model = vit_b_16(weights=weights)

vit_model.heads.head = torch.nn.Linear(vit_model.heads.head.in_features, num_classes)

for p in vit_model.parameters():
    p.requires_grad = False
for p in vit_model.heads.head.parameters():
    p.requires_grad = True

vit_model.to(device)




## === cell 1
class CassavaDataset(VisionDataset):
    """Dataset for Cassava images.

    - robust RGB loading
    - supports optional labels (train) and deterministic ordering (test)
    """

    def __init__(self, data_dir, images, labels=None, transform=None, ttas=None):
        super().__init__(root=data_dir)
        self.transform = transform
        self.ttas = ttas
        self.images = list(images)
        self.labels = None if labels is None else list(labels)

    def __getitem__(self, idx):
        filename = self.images[idx]
        img_path = os.path.join(self.root, filename)
        img = Image.open(img_path).convert("RGB")

        if self.ttas is not None and self.transform is not None:
            img = [self.transform(t(img)) for t in self.ttas]
        elif self.transform:
            img = self.transform(img)

        if self.labels is None:
            return img, filename
        return img, int(self.labels[idx])

    def __len__(self):
        return len(self.images)




## === cell 2
train_transforms = weights.transforms(antialias=True)
test_transforms = weights.transforms(antialias=True)

if tta:
    ttas = [
        v2.RandomResizedCrop((224, 224), (0.5, 1)),
        v2.RandomRotation(180),
        v2.RandomVerticalFlip(1),
        v2.RandomAffine(180),
        v2.RandomPerspective(p=1),
    ]
else:
    ttas = None

train_df = pd.read_csv(train_csv_path)

sample_sub = pd.read_csv(sample_sub_path)
test_images_ordered = sample_sub["image_id"].tolist()
test_dataset = CassavaDataset(
    test_dir,
    images=test_images_ordered,
    labels=None,
    transform=test_transforms,
    ttas=ttas,
)

_loader_kwargs = dict(
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else None,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,  # maintain 1:1 alignment and stable ordering
    **{k: v for k, v in _loader_kwargs.items() if v is not None},
)

normalizer = torch.nn.Softmax(dim=1)




## === cell 3
def _stratified_split_indices(labels, val_frac=0.1, seed=3407):
    g = torch.Generator()
    g.manual_seed(seed)
    labels = torch.as_tensor(labels, dtype=torch.long)
    idx_all = torch.arange(labels.numel())

    tr_idx = []
    va_idx = []
    for c in range(int(labels.max().item()) + 1):
        cls_idx = idx_all[labels == c]
        if cls_idx.numel() == 0:
            continue
        perm = cls_idx[torch.randperm(cls_idx.numel(), generator=g)]
        n_val = max(1, int(round(val_frac * perm.numel())))
        va_idx.append(perm[:n_val])
        tr_idx.append(perm[n_val:])
    tr_idx = torch.cat(tr_idx).tolist()
    va_idx = torch.cat(va_idx).tolist()
    return tr_idx, va_idx


def _make_loader(df, indices, shuffle):
    sub = df.iloc[indices].reset_index(drop=True)
    ds = CassavaDataset(
        train_dir,
        images=sub["image_id"].tolist(),
        labels=sub["label"].tolist(),
        transform=train_transforms,
        ttas=None,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        **{k: v for k, v in _loader_kwargs.items() if v is not None},
    )


def _forward_frozen_backbone(model, x):
    x = model._process_input(x)
    n = x.shape[0]
    batch_class_token = model.class_token.expand(n, -1, -1)
    x = torch.cat([batch_class_token, x], dim=1)

    with torch.no_grad():
        x = model.encoder(x)

    x = x[:, 0]
    x = model.heads(x)
    return x


def _eval_acc(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(inputs)
            preds = torch.argmax(logits, dim=1)
            correct += int((preds == labels).sum().item())
            total += int(labels.size(0))
    return correct / max(total, 1)


@torch.no_grad()
def _extract_frozen_features(model, loader, feat_dim, device):
    model.eval()
    feats = torch.empty(
        (len(loader.dataset), feat_dim), dtype=torch.float32, device="cpu"
    )
    labels_out = torch.empty((len(loader.dataset),), dtype=torch.long, device="cpu")

    use_forward_features = hasattr(model, "forward_features")

    write_pos = 0
    for inputs, labels in loader:
        bs = inputs.size(0)
        inputs = inputs.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            if use_forward_features:
                x = model.forward_features(inputs)
            else:
                x = model._process_input(inputs)
                n = x.shape[0]
                batch_class_token = model.class_token.expand(n, -1, -1)
                x = torch.cat([batch_class_token, x], dim=1)
                x = model.encoder(x)
                x = model.encoder.ln(x)
                x = x[:, 0]

        x = x.detach()

        feats[write_pos : write_pos + bs].copy_(x.float().cpu(), non_blocking=False)
        labels_out[write_pos : write_pos + bs].copy_(labels, non_blocking=False)
        write_pos += bs

    return feats, labels_out


def _make_feature_loader(feats_cpu, labels_cpu, shuffle):
    ds = TensorDataset(feats_cpu, labels_cpu)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        **{k: v for k, v in _loader_kwargs.items() if v is not None},
    )


criterion = torch.nn.CrossEntropyLoss()

tr_idx, va_idx = _stratified_split_indices(
    train_df["label"].tolist(), val_frac=0.1, seed=3407
)
train_loader_split = _make_loader(train_df, tr_idx, shuffle=True)
val_loader_split = _make_loader(train_df, va_idx, shuffle=False)

feat_dim = vit_model.heads.head.in_features

train_feats_cpu, train_labels_cpu = _extract_frozen_features(
    vit_model, train_loader_split, feat_dim=feat_dim, device=device
)
val_feats_cpu, val_labels_cpu = _extract_frozen_features(
    vit_model, val_loader_split, feat_dim=feat_dim, device=device
)
train_feat_loader = _make_feature_loader(
    train_feats_cpu, train_labels_cpu, shuffle=True
)
val_feat_loader = _make_feature_loader(val_feats_cpu, val_labels_cpu, shuffle=False)


def _reset_head_parameters():
    torch.manual_seed(3407)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(3407)
    vit_model.heads.head.reset_parameters()


@torch.no_grad()
def _eval_acc_from_features(model, feat_loader):
    model.eval()
    correct = 0
    total = 0
    for feats, labels in feat_loader:
        feats = feats.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model.heads(feats)
        preds = torch.argmax(logits, dim=1)
        correct += int((preds == labels).sum().item())
        total += int(labels.size(0))
    return correct / max(total, 1)


candidates = [
    {"lr": 3e-4, "epochs": 3},
    {"lr": 2e-4, "epochs": 4},
]

best = None
best_acc = -1.0

for cfg in candidates:
    _reset_head_parameters()
    optimizer = torch.optim.AdamW(
        vit_model.heads.head.parameters(), lr=cfg["lr"], weight_decay=1e-2
    )

    vit_model.eval()
    vit_model.heads.head.train()

    for epoch in range(cfg["epochs"]):
        running_loss = 0.0
        total = 0
        for feats, labels in train_feat_loader:
            feats = feats.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = vit_model.heads(feats)
                loss = criterion(logits, labels)

            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            running_loss += float(loss.item()) * labels.size(0)
            total += int(labels.size(0))

        print(
            f"trial lr={cfg['lr']} epoch={epoch+1}/{cfg['epochs']} train_loss={running_loss/max(total,1):.4f}"
        )

    val_acc = _eval_acc_from_features(vit_model, val_feat_loader)
    print(f"trial lr={cfg['lr']} epochs={cfg['epochs']} val_acc={val_acc:.5f}")

    if val_acc > best_acc:
        best_acc = val_acc
        best = cfg

print("Selected head-train config:", best, "val_acc=", best_acc)

_reset_head_parameters()

full_train_dataset = CassavaDataset(
    train_dir,
    images=train_df["image_id"].tolist(),
    labels=train_df["label"].tolist(),
    transform=train_transforms,
    ttas=None,
)
full_train_loader = DataLoader(
    full_train_dataset,
    batch_size=batch_size,
    shuffle=False,
    **{k: v for k, v in _loader_kwargs.items() if v is not None},
)

full_feats_cpu, full_labels_cpu = _extract_frozen_features(
    vit_model, full_train_loader, feat_dim=feat_dim, device=device
)
full_feat_loader = _make_feature_loader(full_feats_cpu, full_labels_cpu, shuffle=True)

optimizer = torch.optim.AdamW(
    vit_model.heads.head.parameters(), lr=best["lr"], weight_decay=1e-2
)

vit_model.eval()
vit_model.heads.head.train()

for epoch in range(best["epochs"]):
    running_loss = 0.0
    correct = 0
    total = 0
    for feats, labels in full_feat_loader:
        feats = feats.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = vit_model.heads(feats)
            loss = criterion(logits, labels)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += float(loss.item()) * labels.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += int((preds == labels).sum().item())
        total += int(labels.size(0))

    print(
        f"final epoch={epoch+1}/{best['epochs']} "
        f"train_loss={running_loss/max(total,1):.4f} "
        f"train_acc={correct/max(total,1):.4f}"
    )



## === cell 4
all_names = []
all_preds = []

vit_model.eval()

with torch.no_grad():
    for batch_idx, (inputs, filenames) in enumerate(test_loader):
        if tta:
            n_imgs = len(filenames)
            n_tta = len(ttas)
            inputs = torch.cat(inputs, dim=0).to(device, non_blocking=True)
            filenames = list(filenames)

            with torch.amp.autocast("cuda", enabled=use_amp):
                preds = normalizer(vit_model(inputs))

            preds = preds.view(n_tta, n_imgs, -1).mean(dim=0)
            pred_labels = torch.argmax(preds, 1).tolist()
        else:
            inputs = inputs.to(device, non_blocking=True)
            filenames = list(filenames)

            with torch.amp.autocast("cuda", enabled=use_amp):
                preds = normalizer(vit_model(inputs))
            pred_labels = torch.argmax(preds, 1).tolist()

        all_names.extend(filenames)
        all_preds.extend(pred_labels)



## === cell 5
pred_df = pd.DataFrame({"image_id": all_names, "label": all_preds})

pred_df = sample_sub[["image_id"]].merge(pred_df, on="image_id", how="left")
pred_df["label"] = pred_df["label"].fillna(0).astype(int)

pred_df.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", pred_df.shape)
print(pred_df.head())



## === cell 6
pred_df
