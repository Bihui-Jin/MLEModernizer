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

0.8702024780900575

# 6. Current score

0.73281

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.20478) has done: 'I fix the dataset file listing to ignore subdirectories (your `test_images` folder contains a nested `test_images/` directory), which is what caused the `IsADirectoryError` and the later “missing predictions” error. I also load the best-available checkpoint if one exists (common in Kaggle notebooks) while keeping the same ViT model/heads and inference logic; otherwise it run with ImageNet weights as before. Finally, I add a small amount of defensive I/O (filtering by image extensions) so the submission is always complete and correctly aligned to `sample_submission.csv`.'
- What this solution (achieved 0.77952) has done: 'Your low score is consistent with running a randomly initialized classification head (ImageNet ViT backbone + new 5-class head) with no fine-tuning, so the smallest score-improving change is to actually train that head on `train.csv` while keeping the exact same model/architecture, transforms, loss (cross-entropy), and inference semantics. I add a short train/val split and a brief training loop (few epochs) that updates only the head (backbone frozen) to move accuracy sharply upward without changing the core approach. I also reuse the same normalization/resize pipeline for train/val to avoid train–test mismatch and keep submission writing/alignment exactly as you already do. This should move the score much closer to the 0.87 target band while staying within the 600s runtime budget.'
- What this solution (achieved 0.73281) has done: 'The timeout is dominated by training the ViT backbone even with only the head unfrozen, plus avoidable data-loading overhead (PIL decode + repeated open to check crop, expensive random transforms, and conservative cuDNN settings). I keep the same model, loss, epochs, and train/eval loops, but make the pipeline faster by (1) enabling cuDNN benchmark (safe here because all tensors are fixed-size), (2) removing the extra per-sample image open for `_needs_center_crop` by computing the condition from the already-opened image (equivalent result), and (3) optimizing DataLoader settings (more workers, persistent workers, pin memory, and faster inter-process sharing) to maximize GPU utilization. These changes preserve evaluation semantics and accuracy while cutting constant-factor runtime substantially.'

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

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

torch.manual_seed(3407)
torch.cuda.manual_seed(3407)
random.seed(3407)

cudnn.deterministic = True

cudnn.benchmark = True

try:
    torch.multiprocessing.set_sharing_strategy("file_system")
except Exception:
    pass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

test_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images/"
if not os.path.isdir(test_dir):
    test_dir = "/kaggle/data/cassava-leaf-disease-classification/test_images/"
assert os.path.isdir(test_dir), f"Test image directory not found: {test_dir}"

train_csv_path = "/kaggle/input/cassava-leaf-disease-classification/train.csv"
train_img_dir = "/kaggle/input/cassava-leaf-disease-classification/train_images/"
if not os.path.isfile(train_csv_path):
    train_csv_path = "/kaggle/data/cassava-leaf-disease-classification/train.csv"
    train_img_dir = "/kaggle/data/cassava-leaf-disease-classification/train_images/"
assert os.path.isfile(train_csv_path), f"Train CSV not found: {train_csv_path}"
assert os.path.isdir(train_img_dir), f"Train image directory not found: {train_img_dir}"

batch_size = 16

_cpu = os.cpu_count() or 4
num_workers = min(12, max(4, _cpu - 1))
prefetch_factor = 4
persistent_workers = num_workers > 0

num_classes = 5
tta = True

from torchvision.models import vit_b_16, ViT_B_16_Weights

weights = ViT_B_16_Weights.IMAGENET1K_V1
img_size = int(weights.transforms().crop_size[0])

model = vit_b_16(weights=weights)
model.heads.head = torch.nn.Linear(model.heads.head.in_features, num_classes)


def _try_load_checkpoint(m, device):
    candidates = [
        "/kaggle/input/model.pth",
        "/kaggle/input/model.pt",
        "/kaggle/working/model.pth",
        "/kaggle/working/model.pt",
        "/kaggle/working/best.pth",
        "/kaggle/working/best.pt",
        "/kaggle/working/checkpoint.pth",
        "/kaggle/working/checkpoint.pt",
    ]
    for p in candidates:
        if os.path.isfile(p):
            ckpt = torch.load(p, map_location="cpu")
            state = ckpt.get("state_dict", ckpt)
            cleaned = {}
            for k, v in state.items():
                if k.startswith("model."):
                    k = k[len("model.") :]
                if k.startswith("module."):
                    k = k[len("module.") :]
                cleaned[k] = v
            missing, unexpected = m.load_state_dict(cleaned, strict=False)
            print(f"Loaded checkpoint: {p}")
            if missing:
                print(f"  Missing keys (partial load, ok): {len(missing)}")
            if unexpected:
                print(f"  Unexpected keys (ignored): {len(unexpected)}")
            return True
    print(
        "No checkpoint found; using ImageNet-pretrained backbone with randomly initialized head."
    )
    return False


has_ckpt = _try_load_checkpoint(model, device)
model.to(device)




## === cell 1
try:
    import torchvision

    torchvision.set_image_backend("accimage")
    _IMG_BACKEND = "accimage"
except Exception:
    _IMG_BACKEND = "pil"


def _needs_center_crop_from_size(w: int, h: int) -> bool:
    return (w >= 500) and (h >= 500)


class CassavaDataset(VisionDataset):
    """Custom dataset for the Cassava test data."""

    def __init__(self, data_dir, transform=None, ttas=None, img_size=384):
        super().__init__(root=data_dir)
        self.transform = transform

        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        files = []
        for name in os.listdir(data_dir):
            p = os.path.join(data_dir, name)
            if os.path.isfile(p) and os.path.splitext(name.lower())[1] in exts:
                files.append(name)
        self.images = sorted(files)  # deterministic order

        self.ttas = ttas
        self.cc = v2.CenterCrop((500, 500))

    def __getitem__(self, idx):
        filename = self.images[idx]
        img_path = os.path.join(self.root, filename)

        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        if _needs_center_crop_from_size(w, h):
            img = self.cc(img)

        if self.ttas is not None and self.transform is not None:
            img = [self.transform(t(img)) for t in self.ttas]
        elif self.transform:
            img = self.transform(img)

        return img, filename

    def __len__(self):
        return len(self.images)




## === cell 2
class CassavaTrainDataset(VisionDataset):
    def __init__(self, df, img_dir, transform=None):
        super().__init__(root=img_dir)
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.cc = v2.CenterCrop((500, 500))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fname = row["image_id"]
        y = int(row["label"])
        p = os.path.join(self.root, fname)

        img = Image.open(p).convert("RGB")
        w, h = img.size
        if _needs_center_crop_from_size(w, h):
            img = self.cc(img)

        if self.transform is not None:
            img = self.transform(img)
        return img, y




## === cell 3
test_transforms = v2.Compose(
    [
        v2.Resize((img_size, img_size), interpolation=InterpolationMode.BICUBIC),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

train_transforms = v2.Compose(
    [
        v2.Resize((img_size, img_size), interpolation=InterpolationMode.BICUBIC),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.2),
        v2.RandomRotation(degrees=10),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

if tta:
    ttas = [
        v2.Identity(),
        v2.RandomRotation((180, 180)),  # deterministic 180-degree rotation
        v2.RandomVerticalFlip(p=1.0),  # deterministic flip
        v2.RandomHorizontalFlip(p=1.0),  # deterministic flip
    ]
else:
    ttas = None

test_dataset = CassavaDataset(test_dir, transform=test_transforms, ttas=ttas)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=persistent_workers,
    prefetch_factor=prefetch_factor if persistent_workers else None,
)

normalizer = torch.nn.Softmax(dim=1)




## === cell 4
if not has_ckpt:
    train_df = pd.read_csv(train_csv_path)

    g = torch.Generator().manual_seed(3407)
    tr_parts, va_parts = [], []
    for c in sorted(train_df["label"].unique().tolist()):
        df_c = train_df[train_df["label"] == c].reset_index(drop=True)
        perm = torch.randperm(len(df_c), generator=g).tolist()
        split = int(0.9 * len(perm))
        tr_parts.append(df_c.iloc[perm[:split]])
        va_parts.append(df_c.iloc[perm[split:]])
    tr_df = (
        pd.concat(tr_parts, ignore_index=True)
        .sample(frac=1.0, random_state=3407)
        .reset_index(drop=True)
    )
    va_df = pd.concat(va_parts, ignore_index=True).reset_index(drop=True)

    train_ds = CassavaTrainDataset(tr_df, train_img_dir, transform=train_transforms)
    val_ds = CassavaTrainDataset(va_df, train_img_dir, transform=test_transforms)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor if persistent_workers else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor if persistent_workers else None,
    )

    for p in model.parameters():
        p.requires_grad = False
    for p in model.heads.head.parameters():
        p.requires_grad = True

    counts = tr_df["label"].value_counts().sort_index()
    counts = counts.reindex(range(num_classes), fill_value=1)
    w = (counts.sum() / counts).astype("float32").values
    w = torch.tensor(w, device=device)
    w = w / w.mean()
    criterion = torch.nn.CrossEntropyLoss(weight=w)

    optimizer = torch.optim.AdamW(
        model.heads.head.parameters(), lr=5e-4, weight_decay=1e-2
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=6)

    def _eval_acc():
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                logits = model(xb)
                pred = torch.argmax(logits, dim=1)
                correct += (pred == yb).sum().item()
                total += yb.numel()
        return correct / max(1, total)

    model.train()
    epochs = 6
    for ep in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        scheduler.step()
        acc = _eval_acc()
        lr = scheduler.get_last_lr()[0]
        print(f"Epoch {ep+1}/{epochs} - lr: {lr:.2e} - val acc: {acc:.4f}")




## === cell 5
all_names = []
all_preds = []

model.eval()

with torch.inference_mode():
    for batch_idx, (inputs, filenames) in enumerate(test_loader):
        filenames = list(filenames)

        if tta:
            inputs_cat = torch.cat(inputs, dim=0).to(device, non_blocking=True)

            preds = normalizer(model(inputs_cat))  # [n_tta*B, num_classes]
            n_tta = len(inputs)
            bsz = len(filenames)

            preds = preds.view(n_tta, bsz, -1)
            mean_preds = preds.mean(dim=0)  # [B, num_classes]
            pred_labels = torch.argmax(mean_preds, dim=1).tolist()
        else:
            inputs = inputs.to(device, non_blocking=True)
            preds = normalizer(model(inputs))
            pred_labels = torch.argmax(preds, dim=1).tolist()

        all_names.extend(filenames)
        all_preds.extend(pred_labels)

assert len(all_names) == len(test_dataset), (len(all_names), len(test_dataset))
assert len(all_preds) == len(test_dataset), (len(all_preds), len(test_dataset))




## === cell 6
sample_path = "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
if not os.path.isfile(sample_path):
    sample_path = (
        "/kaggle/data/cassava-leaf-disease-classification/sample_submission.csv"
    )
sample = pd.read_csv(sample_path)

pred_df = pd.DataFrame({"image_id": all_names, "label": all_preds})

pred_df = sample[["image_id"]].merge(pred_df, on="image_id", how="left")
if pred_df["label"].isna().any():
    missing_ids = pred_df.loc[pred_df["label"].isna(), "image_id"].head(10).tolist()
    missing = int(pred_df["label"].isna().sum())
    raise RuntimeError(
        f"Missing predictions for {missing} test images; e.g. {missing_ids}. "
        f"Check test_dir contents: {test_dir}"
    )
pred_df["label"] = pred_df["label"].astype(int)

pred_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", pred_df.shape)




## === cell 7
pred_df.head()
