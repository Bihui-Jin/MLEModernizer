# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.889694771834391

# 6. Current score

0.48991

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.17526) has done: 'I fix the missing model file issue by loading the model weights from a path that actually exists in this Kaggle environment, falling back to a strong torchvision ImageNet backbone if no external .pt is available (so the notebook always runs end-to-end). I also fix the test DataLoader logic errors that cause invalid submissions: `shuffle` must be `False`, filenames must be kept in dataset order, and the final submission must be aligned exactly to `sample_submission.csv` image_id order. Finally, I make TTA deterministic and shape-safe (by repeating the same batch `N` times rather than relying on per-item random transforms) so predictions and row counts remain correct and stable.'
- What this solution (achieved 0.76383) has done: 'Your current score (0.17526) is far below the target (0.88969), and the main cause is that you’re using an ImageNet backbone with a randomly initialized 5-class head (no cassava training), so predictions are nearly random. To move the score sharply upward while preserving your core inference logic, I add a minimal training cell that fine-tunes only the classifier head on `train.csv` images using the same EfficientNet-V2-S backbone, then reuse your exact test DataLoader + submission alignment. I also make the “TTA” loop meaningful but still minimal by switching it to deterministic flip-averaging (instead of repeating identical inputs). This keeps the architecture and overall approach intact while making the model actually learn the cassava labels and should bring accuracy much closer to your target band.'
- What this solution (achieved 0.74253) has done: 'Your current score (0.76383) is well below the target (0.88969), so we should improve accuracy while keeping the same EfficientNet-V2-S backbone + “train only classifier head” core logic. The biggest low-risk gain is to add a proper train/validation split and train against that split (rather than measuring training accuracy), using class-weighted cross entropy to handle imbalance and a standard LR schedule to make 2 epochs more effective. We keep the same dataset, transforms, loss family (cross-entropy), and inference/TTA + submission alignment, but make training more stable and better calibrated. These changes are small, should stay within the 600s budget, and are directly aimed at moving accuracy upward toward the target band.'
- What this solution (achieved 0.52354) has done: 'We’re still well below the target (0.7425 vs 0.8897), so we should increase accuracy with the smallest changes that preserve your EfficientNet-V2-S + “train only classifier head” logic. The most effective minimal tweak is to feed the model its native input resolution (384 for EfficientNet-V2-S) instead of 528, which reduces resize distortion and train/infer mismatch and typically improves accuracy while also speeding training. I also make the flip-TTA consistent with natural-image symmetry by averaging both horizontal and vertical flips (still deterministic, same semantics). Finally, I add a safe label-smoothing term to the existing CrossEntropyLoss (same loss family) to improve calibration/generalization without changing the training loop structure.'
- What this solution (achieved 0.67265) has done: 'Your current score (0.52354) is far below the target (0.88969), so we should increase accuracy with the smallest changes that keep your EfficientNet-V2-S + “train only classifier head” approach intact. The biggest low-risk issue is that your training augmentation is too weak for this dataset (only horizontal flip), so I add standard light augmentations (small random rotation, color jitter, random resized crop) while keeping the same input size, backbone, and loss family. I also ensure the EfficientNet preprocessing exactly matches its pretrained weights (using the official weights transforms), which typically gives a meaningful jump without changing the model or loop. Finally, I keep your deterministic flip-TTA and submission alignment unchanged, just making inference preprocessing consistent with training.'
- What this solution (achieved 0.68685) has done: 'Your gap to the target is still large (0.67265 → 0.88969, higher is better), so we should improve generalization while keeping your EfficientNet-V2-S + “train only classifier head” pipeline intact. The lowest-risk gain is to make validation preprocessing consistent with training by using a deterministic center-crop style eval transform (instead of resize-stretch), and to use the same eval transform at test time as well. I also fix a subtle TTA mismatch: if you apply flips at inference, you should also flip the resulting probabilities back into the same “view” implicitly by averaging probabilities from flipped inputs (which you already do), but we add a center-crop baseline that aligns better and typically improves accuracy. Finally, I keep everything else (architecture, loss family, training loop, epochs) the same to stay minimal and stable.'
- What this solution (achieved 0.72459) has done: 'Your current score (0.68685) is far below the target (0.88969), so we should improve accuracy with the smallest changes that preserve your EfficientNet-V2-S + “train only classifier head” training loop and the same inference/submission semantics. The main limitation is that training only the head for 2 epochs underfits; we can move toward the target by fine-tuning a small part of the backbone (just the last EfficientNet “features” block) while keeping everything else identical. To keep this stable and fast, we use a smaller LR for the unfrozen block, keep your class-weighted cross-entropy with label smoothing, and keep your deterministic flip-TTA and submission alignment unchanged. These changes are minimal, directly aimed at improving generalization, and should run within the time budget.'
- What this solution (achieved 0.61099) has done: 'We need to move your accuracy up toward the 0.8897 target (current 0.7246, higher-is-better), so the smallest effective change is to reduce underfitting without changing your core model/loop: keep EfficientNet-V2-S and the same training procedure, but unfreeze one additional top backbone block (last two blocks instead of only the last) and use a slightly longer fine-tuning schedule (more epochs) with the same optimizer/loss/scheduler family. This keeps architecture, loss, and inference semantics identical while letting the model adapt enough to cassava. I also make the validation split deterministic and stable (same logic, just ensure label indices are concatenated safely) and keep submission alignment unchanged. The code still runs end-to-end within Kaggle constraints and writes a valid `submission.csv`.'
- What this solution (achieved 0.48991) has done: 'Main runtime is dominated by (1) PIL image decoding + heavy per-sample augmentations for 7 epochs over ~16.8k images and (2) triple-forward TTA on the full test set. To fit the 600s budget without changing the model or training semantics, I remove avoidable Python/Pandas overhead in the datasets (precompute paths/labels), enable faster and deterministic CPU decoding via `torchvision.io` (with a safe PIL fallback), and make the DataLoader pipeline more efficient (larger `num_workers`, `torch.set_num_threads`, `torch.backends.cuda.matmul.allow_tf32=False` to preserve numerics, pinned memory already). I also avoid redundant work in TTA by computing softmax only once per summed logits (mathematically equivalent for argmax of averaged probs) while keeping identical prediction semantics. These changes keep the same architecture, losses, epochs, and transforms, but reduce constant factors substantially.'

# 9. Code solution

## === cell 0
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
from torchvision.transforms import InterpolationMode, v2

SEED = 3407
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

cudnn.deterministic = True
cudnn.benchmark = False

CPU_COUNT = os.cpu_count() or 4
torch.set_num_threads(min(8, CPU_COUNT))
torch.set_num_interop_threads(min(4, CPU_COUNT))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

data_root = "/kaggle/input/cassava-leaf-disease-classification/"
train_csv_path = os.path.join(data_root, "train.csv")
train_dir = os.path.join(data_root, "train_images/")
test_dir = os.path.join(data_root, "test_images/")
sample_path = os.path.join(data_root, "sample_submission.csv")

img_size = 384

batch_size = 16
num_workers = 4
num_classes = 5

tta = True
tta_runs = 2  # kept for compatibility (not used directly)

train_epochs = 7

lr = 2e-3
ft_backbone_lr = 5e-4

val_frac = 0.1

label_smoothing = 0.05

ft_unfreeze_last_block = True
ft_unfreeze_last_n_blocks = 2

num_workers = max(num_workers, min(12, CPU_COUNT))
PERSISTENT_WORKERS = num_workers > 0
PREFETCH_FACTOR = 4 if num_workers > 0 else None

USE_CHANNELS_LAST = device.type == "cuda"

if device.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _try_load_model():
    candidates = [
        "/kaggle/input/efficient-net/vit_cont_3.pt",
        "/kaggle/input/efficient-net/vit_cont_3.pth",
        "/kaggle/input/efficient-net/vit_cont_3.bin",
    ]
    for p in candidates:
        if os.path.exists(p):
            obj = torch.load(p, map_location="cpu")
            return obj, f"loaded torch model from {p}"

    import torchvision

    weights = torchvision.models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
    model = torchvision.models.efficientnet_v2_s(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(in_features, num_classes)
    return (
        model,
        "fallback: torchvision efficientnet_v2_s (imagenet weights) with new 5-class head",
    )


model, model_msg = _try_load_model()
print(model_msg)

if isinstance(model, dict):
    state = model.get("state_dict", model.get("model_state_dict", model))
    import torchvision

    weights = torchvision.models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
    model_obj = torchvision.models.efficientnet_v2_s(weights=weights)
    in_features = model_obj.classifier[1].in_features
    model_obj.classifier[1] = torch.nn.Linear(in_features, num_classes)

    new_state = {}
    for k, v in state.items():
        nk = k.replace("module.", "")
        new_state[nk] = v
    missing, unexpected = model_obj.load_state_dict(new_state, strict=False)
    print(
        "Loaded state_dict. Missing keys:",
        len(missing),
        "Unexpected keys:",
        len(unexpected),
    )
    model = model_obj

model = model.to(device)
if USE_CHANNELS_LAST:
    model = model.to(memory_format=torch.channels_last)

if device.type == "cuda":
    cudnn.benchmark = True



## === cell 1
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms.v2 import functional as Fv2


class CassavaTrainDataset(VisionDataset):
    """Train dataset: returns (image_tensor, label_int)."""

    def __init__(self, df: pd.DataFrame, data_dir: str, transform=None):
        super().__init__(root=data_dir)
        df = df.reset_index(drop=True)
        self.paths = [os.path.join(self.root, f) for f in df["image_id"].tolist()]
        self.labels = df["label"].astype(np.int64).to_numpy()
        self.transform = transform

    def __getitem__(self, idx):
        path = self.paths[idx]
        label = int(self.labels[idx])
        try:
            img = read_image(path, mode=ImageReadMode.RGB)  # uint8 CHW
            img = Fv2.to_image(img)  # tv_tensors.Image (CHW)
        except Exception:
            img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.labels)


class CassavaTestDataset(VisionDataset):
    """Test dataset: returns (image_tensor, filename)."""

    def __init__(self, data_dir, transform=None):
        super().__init__(root=data_dir)
        self.transform = transform
        self.images = sorted(
            [f for f in os.listdir(data_dir) if f.lower().endswith(".jpg")]
        )
        self.paths = [os.path.join(self.root, f) for f in self.images]

    def __getitem__(self, idx):
        filename = self.images[idx]
        path = self.paths[idx]
        try:
            img = read_image(path, mode=ImageReadMode.RGB)
            img = Fv2.to_image(img)
        except Exception:
            img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, filename

    def __len__(self):
        return len(self.images)




## === cell 2
import torchvision

weights = torchvision.models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
mean = list(weights.transforms().mean)
std = list(weights.transforms().std)

train_transforms = v2.Compose(
    [
        v2.RandomResizedCrop(
            size=(img_size, img_size),
            scale=(0.80, 1.00),
            ratio=(0.90, 1.10),
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        ),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomRotation(degrees=10, interpolation=InterpolationMode.BICUBIC),
        v2.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=mean, std=std),
    ]
)

eval_transforms = v2.Compose(
    [
        v2.Resize(
            int(round(img_size * 256 / 224)),  # standard eval scaling rule (short side)
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        ),
        v2.CenterCrop((img_size, img_size)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=mean, std=std),
    ]
)

train_df = pd.read_csv(train_csv_path)


def _stratified_split(df: pd.DataFrame, val_frac: float, seed: int):
    rng = np.random.default_rng(seed)
    val_idx_parts = []
    for c in sorted(df["label"].unique()):
        idx = df.index[df["label"] == c].to_numpy()
        rng.shuffle(idx)
        n_val = max(1, int(round(len(idx) * val_frac)))
        val_idx_parts.append(idx[:n_val])
    val_idx = (
        np.concatenate(val_idx_parts) if len(val_idx_parts) else np.array([], dtype=int)
    )
    val_mask = df.index.isin(val_idx)
    return df.loc[~val_mask].reset_index(drop=True), df.loc[val_mask].reset_index(
        drop=True
    )


train_df_split, val_df_split = _stratified_split(train_df, val_frac=val_frac, seed=SEED)
print("train/val sizes:", len(train_df_split), len(val_df_split))
print(
    "train label dist:\n",
    train_df_split["label"].value_counts(normalize=True).sort_index(),
)
print(
    "val label dist:\n", val_df_split["label"].value_counts(normalize=True).sort_index()
)

train_dataset = CassavaTrainDataset(
    train_df_split, train_dir, transform=train_transforms
)
val_dataset = CassavaTrainDataset(val_df_split, train_dir, transform=eval_transforms)

g = torch.Generator()
g.manual_seed(SEED)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    generator=g,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=PERSISTENT_WORKERS,
    prefetch_factor=PREFETCH_FACTOR,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=PERSISTENT_WORKERS,
    prefetch_factor=PREFETCH_FACTOR,
)

for p in model.parameters():
    p.requires_grad = False
for p in model.classifier.parameters():
    p.requires_grad = True

backbone_params = []
if ft_unfreeze_last_block:
    if (
        hasattr(model, "features")
        and isinstance(model.features, torch.nn.Sequential)
        and len(model.features) > 0
    ):
        n = int(ft_unfreeze_last_n_blocks)
        n = max(1, min(n, len(model.features)))
        blocks = list(model.features)[-n:]
        for blk in blocks:
            for p in blk.parameters():
                p.requires_grad = True
        backbone_params = [p for blk in blocks for p in blk.parameters()]
        print(
            f"Unfroze last {n} backbone block(s) params:",
            sum(p.numel() for p in backbone_params),
        )
    else:
        print("Warning: could not locate model.features; proceeding head-only.")

counts = train_df_split["label"].value_counts().sort_index()
freq = counts.to_numpy(dtype=np.float64)
freq = np.maximum(freq, 1.0)

weights_ce = freq.sum() / (num_classes * freq)
weights_ce = torch.tensor(weights_ce, dtype=torch.float32, device=device)
print("class weights:", weights_ce.detach().cpu().numpy().round(3).tolist())

criterion = torch.nn.CrossEntropyLoss(
    weight=weights_ce, label_smoothing=label_smoothing
)

param_groups = [{"params": model.classifier.parameters(), "lr": lr}]
if len(backbone_params) > 0:
    param_groups.append({"params": backbone_params, "lr": ft_backbone_lr})

optimizer = torch.optim.AdamW(param_groups)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_epochs)


def _to_device_inputs(x):
    x = x.to(device, non_blocking=True)
    if USE_CHANNELS_LAST:
        x = x.contiguous(memory_format=torch.channels_last)
    return x


def _eval_acc(m, loader):
    m.eval()
    correct = 0
    total = 0
    with torch.inference_mode():
        for inputs, labels in loader:
            inputs = _to_device_inputs(inputs)
            labels = labels.to(device, non_blocking=True)
            logits = m(inputs)
            preds = torch.argmax(logits, dim=1)
            correct += int((preds == labels).sum().item())
            total += int(labels.numel())
    return correct / max(1, total)


model.train()
for epoch in range(train_epochs):
    running_loss = 0.0
    total = 0

    for inputs, labels in train_loader:
        inputs = _to_device_inputs(inputs)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += float(loss.item()) * inputs.size(0)
        total += int(labels.numel())

    scheduler.step()

    val_acc = _eval_acc(model, val_loader)
    lrs = [pg["lr"] for pg in optimizer.param_groups]
    print(
        f"epoch {epoch+1}/{train_epochs} - loss: {running_loss/total:.4f} - val_acc: {val_acc:.4f} - lrs: {['%.2e'%x for x in lrs]}"
    )

model.eval()



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/844385241.py in <cell line: 0>()
    170     total = 0
    171 
--> 172     for inputs, labels in train_loader:
    173         inputs = _to_device_inputs(inputs)
    174         labels = labels.to(device, non_blocking=True)

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in __next__(self)
    706                 # TODO(https://github.com/pytorch/pytorch/issues/76750)
    707                 self._reset()  # type: ignore[call-arg]
--> 708             data = self._next_data()
    709             self._num_yielded += 1
    710             if (

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _next_data(self)
   1478                 del self._task_info[idx]
   1479                 self._rcvd_idx += 1
-> 1480                 return self._process_data(data)
   1481 
   1482     def _try_put_index(self):

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _process_data(self, data)
   1503         self._try_put_index()
   1504         if isinstance(data, ExceptionWrapper):
-> 1505             data.reraise()
   1506         return data
   1507 

/usr/local/lib/python3.11/dist-packages/torch/_utils.py in reraise(self)
    731             # instantiate since we don't know how to
    732             raise RuntimeError(msg) from None
--> 733         raise exception
    734 
    735 

ValueError: Caught ValueError in DataLoader worker process 0.
Original Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/worker.py", line 349, in _worker_loop
    data = fetcher.fetch(index)  # type: ignore[possibly-undefined]
           ^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py", line 52, in fetch
    data = [self.dataset[idx] for idx in possibly_batched_index]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py", line 52, in <listcomp>
    data = [self.dataset[idx] for idx in possibly_batched_index]
            ~~~~~~~~~~~~^^^^^
  File "/tmp/ipykernel_55/2516142540.py", line 27, in __getitem__
    img = self.transform(img)
          ^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torchvision/transforms/v2/_container.py", line 51, in forward
    outputs = transform(*inputs)
              ^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torchvision/transforms/v2/_transform.py", line 68, in forward
    flat_outputs = [
                   ^
  File "/usr/local/lib/python3.11/dist-packages/torchvision/transforms/v2/_transform.py", line 69, in <listcomp>
    self.transform(inpt, params) if needs_transform else inpt
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torchvision/transforms/v2/_geometry.py", line 627, in transform
    return self._call_kernel(
           ^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torchvision/transforms/v2/_transform.py", line 49, in _call_kernel
    return kernel(inpt, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torchvision/transforms/v2/functional/_utils.py", line 31, in wrapper
    output = kernel(inpt.as_subclass(torch.Tensor), *args, **kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torchvision/transforms/v2/functional/_geometry.py", line 1031, in rotate_image
    _assert_grid_transform_inputs(image, matrix, interpolation.value, fill, ["nearest", "bilinear"])
  File "/usr/local/lib/python3.11/dist-packages/torchvision/transforms/v2/functional/_geometry.py", line 672, in _assert_grid_transform_inputs
    raise ValueError(f"Interpolation mode '{interpolation}' is unsupported with Tensor input")
ValueError: Interpolation mode 'bicubic' is unsupported with Tensor input


## === cell 3
test_dataset = CassavaTestDataset(test_dir, transform=eval_transforms)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=PERSISTENT_WORKERS,
    prefetch_factor=PREFETCH_FACTOR,
)

all_names = []
all_preds = []

with torch.inference_mode():
    for inputs, filenames in test_loader:
        inputs = _to_device_inputs(inputs)

        if tta:
            logits0 = model(inputs)
            logits_h = model(torch.flip(inputs, dims=[3]))  # horizontal flip
            logits_v = model(torch.flip(inputs, dims=[2]))  # vertical flip

            probs = torch.softmax(logits0, dim=1)
            probs.add_(torch.softmax(logits_h, dim=1))
            probs.add_(torch.softmax(logits_v, dim=1))
            probs.div_(3.0)
            pred_labels = torch.argmax(probs, dim=1).tolist()
        else:
            logits = model(inputs)
            pred_labels = torch.argmax(logits, dim=1).tolist()

        all_names.extend(filenames)
        all_preds.extend(pred_labels)

print("preds:", len(all_preds), "names:", len(all_names), "dataset:", len(test_dataset))



## === cell 4
sample = pd.read_csv(sample_path)

pred_map = dict(zip(all_names, all_preds))
sample["label"] = sample["image_id"].map(pred_map).fillna(0).astype(int)

submission_path = "submission.csv"
sample.to_csv(submission_path, index=False)

print("Wrote:", submission_path, "rows:", len(sample))
print(sample.head())



## === cell 5
sample
