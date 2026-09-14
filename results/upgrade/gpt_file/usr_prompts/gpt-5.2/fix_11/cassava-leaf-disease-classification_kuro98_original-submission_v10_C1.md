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

0.8948322756119673

# 6. Current score

0.53662

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.05531) has done: 'I first fix the missing model file by loading a ViT model from `torchvision` (same overall approach: single forward pass classification with optional TTA) and, if available, load weights from a local Kaggle input path; otherwise it run with ImageNet weights so a valid submission is always produced. I fix the TTA batching bug (the current split logic uses the wrong chunk size) and make the test loader non-shuffled so filenames and predictions stay aligned deterministically. I also ensure images are converted to RGB and the submission rows exactly match `sample_submission.csv` (same order/length), which fixes the “same length as the answers” error. These changes are execution-blocking/format-correctness fixes and should move accuracy up compared to an invalid submission.'
- What this solution (achieved 0.17638) has done: 'I fix the runtime error by making the ViT input size match what `torchvision.models.vit_b_16` expects (224×224), which resolves the assertion failure during forward. I also make the TTA transforms deterministic-at-inference by replacing stochastic `Random*` transforms with their deterministic equivalents, so predictions are stable and typically stronger. Finally, I keep the submission aligned exactly to `sample_submission.csv` order and ensure `submission.csv` is always written.'
- What this solution (achieved 0.66106) has done: 'Your low accuracy is mostly because the inference is effectively “untrained for cassava”: you replace the ImageNet head with a fresh 5-class layer and (unless a hidden checkpoint exists) never train it, so predictions are near-random. To move the score toward the target with minimal disruption, I keep the same ViT model and inference logic, but add a small, deterministic supervised fine-tuning step on `train.csv` using the same preprocessing (center-crop/resize/normalize). I also make the current TTA deterministic by replacing the stochastic `RandomRotation`/`RandomVerticalFlip` with deterministic equivalents (rotate/flip), so you get stable and stronger averaging without changing the TTA structure. The submission writing stays aligned to `sample_submission.csv` exactly as before.'
- What this solution (achieved 0.71188) has done: 'Your score gap to the target is large (0.66106 → 0.89483), and the main bottleneck is that you fine-tune on the full training set for only 2 epochs at a relatively high LR, which tends to underfit and/or destabilize the pretrained representation (especially without any train/val monitoring). To move accuracy upward while keeping the same core ViT model, loss, and training loop structure, I (1) add a lightweight stratified validation split for monitoring, (2) freeze the ViT backbone and train only the classification head for the first epoch, then unfreeze for the remaining epoch(s), and (3) add a conservative LR schedule (cosine) without changing epochs or adding early stopping. These are minimal, standard fine-tuning stabilizers that usually yield a sizable accuracy jump on Cassava without altering evaluation semantics, and they keep inference/submission formatting exactly the same.'
- What this solution (achieved 0.76046) has done: 'Your score is still far below the target (0.71188 vs 0.89483), so we should improve accuracy with minimal, stability-focused changes that keep the same ViT, loss, and single training loop. The biggest low-risk gain for Cassava is to use standard *training-time augmentation* (while keeping test-time transforms deterministic) and to use the pretrained ViT’s expected normalization constants from `ViT_B_16_Weights` to reduce preprocessing mismatch. I also make the fine-tune optimizer/scheduler persist across epochs (instead of being re-created each epoch), which keeps the same approach but avoids effectively “restarting” learning rate dynamics after the head-only warmup. These changes are small, don’t alter the core architecture/inference semantics, and should push accuracy upward toward the target.'
- What this solution (achieved 0.66368) has done: 'To move accuracy up toward your target with minimal disruption, I keep the same ViT model, loss, and training loop, but fix the main training-data mismatch: your `RandomResizedCrop` is currently applied *after* a fixed `CenterCrop`, which can remove relevant leaf context and harms generalization. I replace that with a single `RandomResizedCrop` from the full image (same output size/normalization), and I also use light label-smoothing in `CrossEntropyLoss` to reduce overconfidence and improve test accuracy without changing the objective type. Finally, I make the head-only → full fine-tune transition safe by rebuilding the optimizer when unfreezing (avoids missing/frozen params issues) while keeping the same optimizer/scheduler approach and epochs.'
- What this solution (achieved 0.73879) has done: 'Your current score (0.66368) is far below the target (0.89483), so we should improve generalization with minimal, stability-focused changes while keeping the same ViT model, loss type (cross-entropy), and the same overall fine-tuning + inference structure. The biggest low-risk issue is the heavy information loss from `CenterCrop((600,600))` in both validation and test preprocessing; switching to the standard ViT pipeline (`Resize` shorter side then `CenterCrop(224)`) typically improves cassava accuracy materially without changing the core logic. I also make the cosine LR schedule correct (per-epoch `T_max=len(train_loader)` rather than `finetune_epochs*len(train_loader)` while stepping every batch), which prevents LR from staying too high/low for the actual number of steps and usually improves fine-tuning stability. Finally, I keep submission alignment identical to `sample_submission.csv` and keep deterministic inference/TTA as you already do.'
- What this solution (achieved 0.68087) has done: 'Your current score (0.73879) is well below the target (0.89483), so we should improve generalization with minimal, stability-focused changes while keeping the same ViT model, cross-entropy loss, and overall fine-tune→infer pipeline. The main low-risk gain is to stop reinitializing the optimizer/scheduler each epoch: right now AdamW “forgets” momentum/state at the head-only→full transition and again every epoch, which commonly hurts fine-tuning accuracy; we instead keep one optimizer and only rebuild it once when unfreezing (so new trainable params are included). We also use parameter groups so the head learns faster than the backbone during full fine-tuning (same optimizer type/loop, just safer LR assignment), and add simple gradient clipping to prevent occasional unstable updates at this LR. These changes are directly tied to improving the learned weights (not submission formatting) and should push accuracy upward toward your target without changing architecture or inference semantics.'
- What this solution (achieved 0.52504) has done: 'Your current gap to the target is large, so the smallest likely-win is to improve fine-tuning quality without changing the ViT architecture or the overall train→infer pipeline. I keep your exact model, loss type, and TTA structure, but (1) add mixed-precision training/inference (same math, typically better convergence and faster within the 600s budget), and (2) use a class-weighted cross-entropy (still cross-entropy) to counter Cassava’s label imbalance, which commonly boosts accuracy materially. I also make the train dataloader use a deterministic generator (so your seeds actually control shuffling) and enable cuDNN benchmark only for the non-deterministic path (we keep determinism as you set it). Submission formatting and ordering stay exactly aligned to `sample_submission.csv`.'
- What this solution (achieved 0.53662) has done: 'Your current score is far below the target, so we should make a small, stability-focused training change that legitimately improves accuracy without changing the model, loss type, or overall train→infer pipeline. The biggest issue hurting generalization is that `cudnn.deterministic=True` (plus fully deterministic settings) can reduce GPU kernel choices and, with AMP, sometimes harms convergence; we keep inference deterministic but allow non-deterministic cuDNN during *training only* to improve optimization stability toward the target. We also fix a subtle but important data imbalance bug: you compute class weights from `train_df` only (after stratified split), which slightly distorts weights vs the real distribution; we compute weights from `full_df` instead. Finally, we keep everything else identical (same ViT, same epochs, same augmentations, same TTA, same submission alignment) to minimize disruption.'

# 9. Code solution

## === cell 0
import os
import glob

import pandas as pd
import torch
from PIL import Image
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
from torchvision.transforms import InterpolationMode, v2
from torchvision import models



## === cell 1
torch.manual_seed(3407)
torch.cuda.manual_seed(3407)

cudnn.deterministic = True
cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

test_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images/"
train_csv_path = "/kaggle/input/cassava-leaf-disease-classification/train.csv"
train_dir = "/kaggle/input/cassava-leaf-disease-classification/train_images/"
sample_sub_path = (
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)

img_size = 224

batch_size = 16
num_workers = 4
num_classes = 5
tta = True

finetune = True
finetune_epochs = 2
finetune_lr = 3e-4

val_frac = 0.1

use_amp = torch.cuda.is_available()
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)


def _build_vit(num_classes: int):
    try:
        weights = models.ViT_B_16_Weights.IMAGENET1K_V1
        vit = models.vit_b_16(weights=weights)
    except Exception:
        vit = models.vit_b_16(weights=None)
    in_features = vit.heads.head.in_features
    vit.heads.head = torch.nn.Linear(in_features, num_classes)
    return vit


vit_model = _build_vit(num_classes=num_classes)

candidate_files = [
    "/kaggle/input/vit-v1-update/vit_v1_1.pt",
    "/kaggle/input/vit-v1-update/vit_v1_1.pth",
    "/kaggle/input/vit-v1-update/vit_v1_1.bin",
]
candidate_files += glob.glob("/kaggle/input/**/vit_v1_1.pt", recursive=True)
candidate_files += glob.glob("/kaggle/input/**/vit_v1_1.pth", recursive=True)

loaded = False
for ckpt_path in candidate_files:
    if os.path.exists(ckpt_path):
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            if isinstance(ckpt, torch.nn.Module):
                vit_model = ckpt
            elif (
                isinstance(ckpt, dict)
                and "state_dict" in ckpt
                and isinstance(ckpt["state_dict"], dict)
            ):
                vit_model.load_state_dict(ckpt["state_dict"], strict=False)
            elif isinstance(ckpt, dict):
                vit_model.load_state_dict(ckpt, strict=False)
            loaded = True
            print(f"Loaded weights from: {ckpt_path}")
            break
        except Exception as e:
            print(
                f"Found checkpoint at {ckpt_path} but failed to load ({type(e).__name__}: {e}). Continuing..."
            )

if not loaded:
    print(
        "No custom vit_v1_1 checkpoint found; using torchvision ViT initialization/weights (will fine-tune on train.csv if enabled)."
    )

vit_model.to(device)




## === cell 2
class CassavaDataset(VisionDataset):
    """Custom dataset for the Cassava test data (image_id only)."""

    def __init__(self, data_dir, transform=None, ttas=None):
        super().__init__(root=data_dir)
        self.transform = transform
        self.images = sorted(
            [
                f
                for f in os.listdir(data_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
        )
        self.ttas = ttas

    def __getitem__(self, idx):
        filename = self.images[idx]
        img = Image.open(os.path.join(self.root, filename)).convert("RGB")

        if self.ttas is not None and self.transform is not None:
            img = [self.transform(t(img)) for t in self.ttas]
        elif self.transform:
            img = self.transform(img)

        return img, filename

    def __len__(self):
        return len(self.images)


class CassavaTrainDataset(VisionDataset):
    """Dataset for Cassava train data (image + label)."""

    def __init__(self, csv_path, image_dir, transform=None):
        super().__init__(root=image_dir)
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.image_dir = image_dir

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = row["image_id"]
        label = int(row["label"])
        img = Image.open(os.path.join(self.image_dir, filename)).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.df)


class CassavaTrainDatasetFromDF(VisionDataset):
    """Dataset for Cassava train data using a pre-split dataframe (image + label)."""

    def __init__(self, df, image_dir, transform=None):
        super().__init__(root=image_dir)
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.image_dir = image_dir

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = row["image_id"]
        label = int(row["label"])
        img = Image.open(os.path.join(self.image_dir, filename)).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.df)




## === cell 3
try:
    _w = models.ViT_B_16_Weights.IMAGENET1K_V1
    _mean = list(_w.transforms().mean)
    _std = list(_w.transforms().std)
except Exception:
    _mean = [0.485, 0.456, 0.406]
    _std = [0.229, 0.224, 0.225]

test_transforms = v2.Compose(
    [
        v2.Resize(256, interpolation=InterpolationMode.BICUBIC),
        v2.CenterCrop((img_size, img_size)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=_mean, std=_std),
    ]
)

if tta:

    class _Rotate180:
        def __call__(self, img):
            return img.rotate(180, resample=Image.BICUBIC, expand=False)

    class _VFlip:
        def __call__(self, img):
            return img.transpose(Image.FLIP_TOP_BOTTOM)

    ttas = [v2.Identity(), _Rotate180(), _VFlip()]
else:
    ttas = None

test_dataset = CassavaDataset(test_dir, transform=test_transforms, ttas=ttas)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
)

normalizer = torch.nn.Softmax(dim=1)

train_transforms = v2.Compose(
    [
        v2.RandomResizedCrop(
            (img_size, img_size),
            scale=(0.7, 1.0),
            ratio=(0.9, 1.1),
            interpolation=InterpolationMode.BICUBIC,
        ),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=_mean, std=_std),
    ]
)



## === cell 4
if finetune:
    full_df = pd.read_csv(train_csv_path)
    full_df["label"] = full_df["label"].astype(int)

    gen = torch.Generator().manual_seed(3407)
    train_parts = []
    val_parts = []
    for lbl in sorted(full_df["label"].unique().tolist()):
        part = full_df[full_df["label"] == lbl]
        idx = torch.randperm(len(part), generator=gen).tolist()
        n_val = max(1, int(round(len(part) * val_frac)))
        val_idx = idx[:n_val]
        tr_idx = idx[n_val:]
        val_parts.append(part.iloc[val_idx])
        train_parts.append(part.iloc[tr_idx])

    train_df = (
        pd.concat(train_parts, axis=0)
        .sample(frac=1.0, random_state=3407)
        .reset_index(drop=True)
    )
    val_df = pd.concat(val_parts, axis=0).reset_index(drop=True)

    print(
        f"Train size: {len(train_df)} | Val size: {len(val_df)} | Full: {len(full_df)}"
    )

    train_dataset = CassavaTrainDatasetFromDF(
        train_df, train_dir, transform=train_transforms
    )
    val_dataset = CassavaTrainDatasetFromDF(
        val_df, train_dir, transform=test_transforms
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        generator=torch.Generator().manual_seed(3407),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    def _set_trainable_head_only(model: torch.nn.Module, head_only: bool):
        if head_only:
            for p in model.parameters():
                p.requires_grad = False
            for p in model.heads.parameters():
                p.requires_grad = True
        else:
            for p in model.parameters():
                p.requires_grad = True

    prev_cudnn_det = cudnn.deterministic
    prev_cudnn_bench = cudnn.benchmark
    cudnn.deterministic = False
    cudnn.benchmark = True

    vit_model.train()

    label_counts = (
        full_df["label"].value_counts().reindex(range(num_classes), fill_value=1).values
    )
    class_weights = (label_counts.sum() / (num_classes * label_counts)).astype(
        "float32"
    )
    class_weights_t = torch.tensor(class_weights, device=device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights_t, label_smoothing=0.05)

    def _make_optimizer_with_groups(
        model: torch.nn.Module, base_lr: float, head_lr_mult: float = 3.0
    ):
        head_params = [p for p in model.heads.parameters() if p.requires_grad]
        backbone_params = [
            p
            for n, p in model.named_parameters()
            if p.requires_grad and not n.startswith("heads.")
        ]
        param_groups = []
        if backbone_params:
            param_groups.append({"params": backbone_params, "lr": base_lr})
        if head_params:
            param_groups.append({"params": head_params, "lr": base_lr * head_lr_mult})
        return torch.optim.AdamW(param_groups, lr=base_lr, weight_decay=1e-4)

    optimizer = None
    scheduler = None

    for epoch in range(finetune_epochs):
        head_only = epoch == 0  # first epoch: head-only; second: full fine-tune
        _set_trainable_head_only(vit_model, head_only=head_only)

        if optimizer is None or (epoch == 1):
            optimizer = _make_optimizer_with_groups(
                vit_model, base_lr=finetune_lr, head_lr_mult=3.0
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, len(train_loader))
            )

        vit_model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for imgs, labels in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = vit_model(imgs)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(vit_model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += loss.item() * imgs.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)

        epoch_loss = running_loss / max(total, 1)
        epoch_acc = correct / max(total, 1)

        vit_model.eval()
        v_correct = 0
        v_total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    logits = vit_model(imgs)
                preds = logits.argmax(dim=1)
                v_correct += (preds == labels).sum().item()
                v_total += imgs.size(0)
        v_acc = v_correct / max(v_total, 1)

        print(
            f"Finetune epoch {epoch+1}/{finetune_epochs} "
            f"({'head-only' if head_only else 'full'}) - loss: {epoch_loss:.4f} - train_acc: {epoch_acc:.4f} - val_acc: {v_acc:.4f}"
        )

    cudnn.deterministic = prev_cudnn_det
    cudnn.benchmark = prev_cudnn_bench



## === cell 5
all_names = []
all_preds = []

vit_model.eval()

with torch.no_grad():
    for batch_idx, (inputs, filenames) in enumerate(test_loader):
        if tta:
            n_ttas = len(inputs)
            bsz = len(filenames)

            inputs_cat = torch.cat(inputs, dim=0).to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = vit_model(inputs_cat)
            preds = normalizer(logits)

            preds = preds.view(n_ttas, bsz, -1)
            mean_preds = preds.mean(dim=0)
            pred_labels = mean_preds.argmax(dim=1).tolist()
        else:
            inputs = inputs.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                preds = normalizer(vit_model(inputs))
            pred_labels = preds.argmax(dim=1).tolist()

        all_names.extend(list(filenames))
        all_preds.extend(pred_labels)

print(f"Predicted {len(all_preds)} labels for {len(all_names)} images.")
print("Unique labels:", sorted(set(all_preds)))



## === cell 6
sample_sub = pd.read_csv(sample_sub_path)
pred_map = dict(zip(all_names, all_preds))

missing = [
    img_id for img_id in sample_sub["image_id"].tolist() if img_id not in pred_map
]
if missing:
    print(
        f"Warning: {len(missing)} image_ids missing from predictions. Filling with 0."
    )
for img_id in missing:
    pred_map[img_id] = 0

my_submission = pd.DataFrame(
    {
        "image_id": sample_sub["image_id"],
        "label": sample_sub["image_id"].map(pred_map).astype(int),
    }
)

assert list(my_submission.columns) == ["image_id", "label"]
assert len(my_submission) == len(sample_sub)

my_submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", my_submission.shape)



## === cell 7
my_submission.head()
