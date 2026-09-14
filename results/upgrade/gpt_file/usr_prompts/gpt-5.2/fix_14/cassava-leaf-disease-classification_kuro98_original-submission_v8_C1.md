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

0.76495

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.16816) has done: 'I fix the runtime error by ensuring the ViT input size matches what the pretrained `vit_b_16` expects (224x224), which eliminates the “Expected 224 but got 384” crash. I also fix the TTA pipeline: your current `tta_transforms` are not valid callables (they’re transforms, not functions), so I wrap them as callable pipelines that start from a PIL image and end as a normalized tensor of the correct shape. Finally, I keep the non-shuffled loader and the strict order check so the produced `submission.csv` always matches `sample_submission.csv` length and ordering.'
- What this solution (achieved 0.75262) has done: 'Your score is low because the current script never loads any trained cassava weights, so it’s effectively an ImageNet-pretrained ViT with a randomly initialized 5-class head (near-random predictions). The smallest legitimate improvement that preserves your core inference logic is to add a minimal training step on `train.csv` using the same ViT model and standard cross-entropy, then run your existing (submission-aligned) TTA inference. I keep the model architecture, transforms intent, and overall pipeline intact, but add a simple train/val split for sanity plus a short training loop that fits within the time budget. I also slightly tone down the strongest TTA transforms (keeping TTA enabled) because overly aggressive random geometry can hurt accuracy on this dataset, and this change tends to move accuracy upward toward your target.'
- What this solution (achieved 0.77541) has done: 'The timeout is dominated by training a full ViT-B/16 for multiple epochs with expensive input pipeline + validation passes; the fastest safe wins are (1) cutting per-step overhead via TF32/cuDNN/autotune settings that preserve semantics within negligible FP differences, (2) using a faster, equivalent collate path for TTA to avoid Python list handling per batch, and (3) avoiding redundant work in validation caching and inference (fewer tensor conversions/contiguous calls, better non-blocking transfers). I keep the exact same model, transforms, datasets, losses, epochs, and training loops, but make the dataloaders and batch assembly more efficient and reduce CPU/GPU synchronization points. I also fix a determinism contradiction (deterministic=True with benchmark=True) in a way that keeps results stable and avoids slow deterministic algorithms where possible without changing the algorithm. All file paths, architecture, and evaluation semantics remain unchanged.'
- What this solution (achieved 0.77055) has done: 'Your current gap to the target is large (0.77541 → 0.89211), so the most direct way to move accuracy upward without changing the core ViT/CE training logic is to fix two data/optimization mismatches that commonly cap this baseline: (1) use a stratified train/val split so all classes are represented similarly during training (reduces overfitting/underfitting from label imbalance), and (2) add a standard LR scheduler (CosineAnnealingLR) to stabilize finetuning at your existing epoch counts and learning rates. These changes keep the same model, loss, transforms intent, and training loops, but typically yield a meaningful accuracy lift on Cassava within the same runtime budget. Submission writing, ordering guarantees, and TTA inference remain unchanged.'
- What this solution (achieved 0.73019) has done: 'Your current score (0.77055) is well below the target (0.89211), so we should make small, safe improvements that typically boost Cassava accuracy without changing the ViT model or the overall training/inference structure. The biggest low-risk gain here is to address class imbalance: replace plain CrossEntropy with class-weighted CrossEntropy (same loss family/semantics) so minority classes aren’t ignored during finetuning. Second, make the training augmentation slightly less destructive for leaf textures by narrowing the RandomResizedCrop scale range (still the same augmentation type), which often improves generalization. Finally, keep your existing TTA/inference/submission alignment intact, only touching the parts that influence training signal.'
- What this solution (achieved 0.76906) has done: 'Your current score is far below the target, so we should make small, low-risk training-signal improvements without changing the ViT architecture, loss family, or overall train→infer→submit flow. The biggest issue is that class-weighted CE can overcorrect and hurt top-1 accuracy here; switching to label smoothing CrossEntropy (same criterion type/semantics) typically improves generalization for Cassava. We also add a standard RandomErasing augmentation (common for image classification) and use a slightly more realistic crop scale range to reduce overfitting while keeping the same augmentation category (RandomResizedCrop). Finally, we keep your submission alignment and TTA inference intact, only touching training transforms and criterion in a minimal way.'
- What this solution (achieved 0.76495) has done: 'We keep your ViT-B/16 + two-phase finetune + label-smoothed CE + TTA pipeline intact, and only make small changes that typically lift Cassava accuracy toward your target. The biggest low-risk gain is to match the pretrained ViT’s expected training recipe by using its provided training transforms (same image size/normalization, better augmentation) instead of a hand-rolled pipeline that may be suboptimal. We also set the classifier head dropout to the standard 0.1 (architecture unchanged; it’s already a dropout layer in the model) to improve generalization during finetuning. Finally, we fix one subtle but important data-path issue: your current training split uses a fixed generator `g` without advancing it across epochs/workers, so we keep determinism but ensure per-epoch shuffling is handled by DataLoader (already true) while leaving everything else unchanged.'
- What this solution (achieved 0.76495) has done: 'Your current score (0.76495) is well below the target (0.89211), so we should make small, safe training-signal improvements without changing the ViT architecture, loss family, or the overall train→infer→submit flow. The biggest low-risk gain here is to unfreeze and fine-tune a bit more of the encoder in phase 2 (still the exact same two-phase loop and optimizer/scheduler types), because unfreezing only the last block often underfits Cassava. Additionally, we should apply the standard weight-decay exclusion for biases/LayerNorm (common AdamW setup) to improve fine-tuning stability/generalization without changing the core optimization algorithm. Finally, we keep your submission alignment and TTA logic intact, only touching phase-2 unfreezing depth and optimizer parameter grouping.'

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

torch.manual_seed(3407)
torch.cuda.manual_seed(3407)

cudnn.deterministic = False
cudnn.benchmark = True
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification"
test_dir = f"{DATA_DIR}/test_images/"
train_dir = f"{DATA_DIR}/train_images/"
sample_path = f"{DATA_DIR}/sample_submission.csv"
train_csv_path = f"{DATA_DIR}/train.csv"

img_size = 224

batch_size = 16

_cpu = os.cpu_count() or 4
num_workers = min(12, _cpu)  # allow a bit more parallel JPEG decode (bounded)
prefetch_factor = 4 if num_workers > 0 else None
persistent_workers = True if num_workers > 0 else False
pin_memory = True

num_classes = 5
tta = True

finetune_epochs = 4  # head-only phase (was 2)

unfreeze_last_n_blocks = 3  # was 1

finetune2_epochs = 2  # short second phase
finetune_lr = 3e-4
finetune2_lr = 5e-5

import torchvision

vit_weights = torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1
vit_model = torchvision.models.vit_b_16(weights=vit_weights)

if hasattr(vit_model, "heads") and hasattr(vit_model.heads, "dropout"):
    vit_model.heads.dropout.p = 0.1

vit_model.heads.head = torch.nn.Linear(vit_model.heads.head.in_features, num_classes)
vit_model.to(device)

if device.type == "cuda":
    vit_model = vit_model.to(memory_format=torch.channels_last)

if device.type == "cuda":
    try:
        vit_model = torch.compile(vit_model, mode="reduce-overhead")
    except Exception as e:
        print(f"torch.compile unavailable/failed; continuing eager. reason={e}")



## === cell 1
from torchvision.io import ImageReadMode, decode_jpeg, read_file


def _read_rgb_tensor(path: str) -> torch.Tensor:
    data = read_file(path)
    return decode_jpeg(data, mode=ImageReadMode.RGB)  # uint8 [C,H,W]


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
        path = os.path.join(self.root, filename)
        img = _read_rgb_tensor(path)  # uint8 [C,H,W]

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
        self.image_ids = df["image_id"].to_numpy()
        self.labels = df["label"].to_numpy(dtype="int64")
        self.transform = transform

    def __getitem__(self, idx):
        filename = self.image_ids[idx]
        label = int(self.labels[idx])
        path = os.path.join(self.root, filename)
        img = _read_rgb_tensor(path)  # uint8 [C,H,W]
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.labels)




## === cell 2
weights_transforms = vit_weights.transforms()
mean = list(weights_transforms.mean)
std = list(weights_transforms.std)

test_transforms = v2.Compose(
    [
        v2.Resize((img_size, img_size), interpolation=InterpolationMode.BICUBIC),
        v2.CenterCrop((img_size, img_size)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=mean, std=std),
    ]
)

train_transforms = weights_transforms

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
                v2.Normalize(mean=mean, std=std),
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
                v2.Normalize(mean=mean, std=std),
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
                v2.Normalize(mean=mean, std=std),
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


def _collate_tta(batch):
    imgs_lists, filenames = zip(*batch)
    T = len(imgs_lists[0])
    stacked = torch.stack(
        [
            torch.stack([imgs_lists[b][t] for b in range(len(imgs_lists))], dim=0)
            for t in range(T)
        ],
        dim=0,
    )
    return stacked, list(filenames)


test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,  # must not shuffle to preserve submission alignment
    num_workers=num_workers,
    pin_memory=pin_memory,
    worker_init_fn=_seed_worker,
    generator=g,
    persistent_workers=persistent_workers,
    prefetch_factor=prefetch_factor,
    collate_fn=_collate_tta if tta else None,
)

normalizer = torch.nn.Softmax(dim=1)



## === cell 3
train_df = pd.read_csv(train_csv_path)

labels = train_df["label"].to_numpy()
idx_by_class = {}
for i, y in enumerate(labels):
    idx_by_class.setdefault(int(y), []).append(i)

train_idx = []
val_idx = []
for c, idxs in idx_by_class.items():
    perm_c = torch.randperm(len(idxs), generator=g).tolist()
    idxs_shuf = [idxs[j] for j in perm_c]
    split_c = int(0.95 * len(idxs_shuf))
    train_idx.extend(idxs_shuf[:split_c])
    val_idx.extend(idxs_shuf[split_c:])

train_idx = [train_idx[i] for i in torch.randperm(len(train_idx), generator=g).tolist()]
val_idx = [val_idx[i] for i in torch.randperm(len(val_idx), generator=g).tolist()]

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
    pin_memory=pin_memory,
    worker_init_fn=_seed_worker,
    generator=g,
    persistent_workers=persistent_workers,
    prefetch_factor=prefetch_factor,
    drop_last=True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=pin_memory,
    worker_init_fn=_seed_worker,
    generator=g,
    persistent_workers=persistent_workers,
    prefetch_factor=prefetch_factor,
)

criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)


def _materialize_val_cache_via_loader(loader):
    xs = []
    ys = []
    n = 0
    for x, y in loader:
        xs.append(x)  # CPU
        ys.append(y)
        n += x.shape[0]
    x_all = torch.cat(xs, dim=0).contiguous()
    y_all = torch.cat(ys, dim=0).to(dtype=torch.long).contiguous()
    assert x_all.shape[0] == n and y_all.shape[0] == n
    return x_all, y_all


val_x_cpu, val_y_cpu = _materialize_val_cache_via_loader(val_loader)


def _evaluate_acc_cached(model, x_cpu, y_cpu, batch=256):
    model.eval()
    total = int(y_cpu.numel())
    correct = 0
    with torch.no_grad():
        for i in range(0, total, batch):
            xb = x_cpu[i : i + batch]
            yb = y_cpu[i : i + batch]
            if device.type == "cuda":
                xb = xb.to(device, non_blocking=True).contiguous(
                    memory_format=torch.channels_last
                )
                yb = yb.to(device, non_blocking=True)
            else:
                xb = xb.to(device)
                yb = yb.to(device)
            logits = model(xb)
            pred = logits.argmax(dim=1)
            correct += (pred == yb).sum().item()
    return correct / max(total, 1)


def _adamw_param_groups(model, weight_decay: float):
    decay = []
    no_decay = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        n = name.lower()
        if n.endswith(".bias") or ("ln" in n) or ("layernorm" in n) or ("norm" in n):
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


for p in vit_model.parameters():
    p.requires_grad = False
for p in vit_model.heads.parameters():
    p.requires_grad = True

optimizer = torch.optim.AdamW(
    _adamw_param_groups(vit_model, weight_decay=0.05),
    lr=finetune_lr,
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=finetune_epochs, eta_min=finetune_lr * 0.05
)

for epoch in range(finetune_epochs):
    vit_model.train()
    running_loss = 0.0
    for x, y in train_loader:
        if device.type == "cuda":
            x = x.to(device, non_blocking=True).contiguous(
                memory_format=torch.channels_last
            )
            y = y.to(device, non_blocking=True)
        else:
            x = x.to(device)
            y = y.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = vit_model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    scheduler.step()
    val_acc = _evaluate_acc_cached(vit_model, val_x_cpu, val_y_cpu, batch=256)
    print(
        f"phase1 epoch={epoch+1}/{finetune_epochs} lr={scheduler.get_last_lr()[0]:.6f} train_loss={running_loss/len(train_loader):.4f} val_acc={val_acc:.4f}"
    )

if (
    hasattr(vit_model, "encoder")
    and hasattr(vit_model.encoder, "layers")
    and unfreeze_last_n_blocks > 0
):
    for p in vit_model.parameters():
        p.requires_grad = False
    for p in vit_model.heads.parameters():
        p.requires_grad = True

    layers = vit_model.encoder.layers
    n = len(layers)
    for li in range(max(0, n - unfreeze_last_n_blocks), n):
        for p in layers[li].parameters():
            p.requires_grad = True

    optimizer2 = torch.optim.AdamW(
        _adamw_param_groups(vit_model, weight_decay=0.05),
        lr=finetune2_lr,
    )

    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer2, T_max=finetune2_epochs, eta_min=finetune2_lr * 0.05
    )

    for epoch in range(finetune2_epochs):
        vit_model.train()
        running_loss = 0.0
        for x, y in train_loader:
            if device.type == "cuda":
                x = x.to(device, non_blocking=True).contiguous(
                    memory_format=torch.channels_last
                )
                y = y.to(device, non_blocking=True)
            else:
                x = x.to(device)
                y = y.to(device)

            optimizer2.zero_grad(set_to_none=True)
            logits = vit_model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer2.step()
            running_loss += loss.item()

        scheduler2.step()
        val_acc = _evaluate_acc_cached(vit_model, val_x_cpu, val_y_cpu, batch=256)
        print(
            f"phase2 epoch={epoch+1}/{finetune2_epochs} lr={scheduler2.get_last_lr()[0]:.6f} train_loss={running_loss/len(train_loader):.4f} val_acc={val_acc:.4f}"
        )



## === cell 4
all_names = []
all_preds = []

vit_model.eval()

with torch.no_grad():
    for inputs, filenames in test_loader:
        if tta:
            T, B = inputs.shape[0], inputs.shape[1]
            inputs_cat = inputs.reshape(T * B, *inputs.shape[2:]).contiguous()
            if device.type == "cuda":
                inputs_cat = inputs_cat.to(device, non_blocking=True).contiguous(
                    memory_format=torch.channels_last
                )
            else:
                inputs_cat = inputs_cat.to(device)
            preds = normalizer(vit_model(inputs_cat))  # [T*B, num_classes]
            preds = preds.view(T, B, -1)  # [T, B, num_classes]
            mean_preds = preds.mean(dim=0)  # [B, num_classes]
            pred_labels = mean_preds.argmax(dim=1).tolist()
        else:
            if device.type == "cuda":
                inputs = inputs.to(device, non_blocking=True).contiguous(
                    memory_format=torch.channels_last
                )
            else:
                inputs = inputs.to(device)
            preds = normalizer(vit_model(inputs))
            pred_labels = preds.argmax(dim=1).tolist()

        all_names.extend(filenames)
        all_preds.extend(pred_labels)

assert len(all_names) == len(
    sample_df
), f"Pred length {len(all_names)} != sample length {len(sample_df)}"
assert all_names == test_image_ids, "Image order mismatch vs sample_submission.csv"



## === cell 5
my_submission = pd.DataFrame({"image_id": all_names, "label": all_preds})
my_submission.to_csv("submission.csv", index=False)

print(my_submission.shape)
print(my_submission.head())



## === cell 6
my_submission
