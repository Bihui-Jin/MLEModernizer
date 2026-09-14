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

0.8842550619522515

# 6. Current score

0.67003

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.09903) has done: 'I fix the runtime error by making the inference image size match the loaded ViT model’s expected `image_size` (224 for `vit_b_16`), which is why it currently crashes at forward. I also make TTA deterministic and safe by keeping it, but switching to non-random, inference-appropriate transforms so it doesn’t introduce stochasticity across runs. Finally, I ensure `my_submission` is always created and written even if something goes wrong earlier, and I keep the submission ordering aligned to `sample_submission.csv` so Kaggle accepts the file. These changes preserve the existing core approach (ViT + transforms + optional TTA + argmax labels) while making the pipeline run end-to-end and produce `submission.csv`.'
- What this solution (achieved 0.09903) has done: 'Your very low accuracy (0.099) strongly suggests you are using a random-weight fallback ViT because no valid trained checkpoint is being loaded, so the smallest effective fix is to (1) load the model in a way that can handle common “state_dict-only” checkpoints, and (2) stop “guessing” a random checkpoint anywhere under `/kaggle/input` which can easily pick the wrong file. I keep the same ViT_B_16 architecture, same preprocessing, same TTA/argmax semantics, but make checkpoint discovery prioritize the competition dataset folder and make loading robust to `state_dict`/`model_state_dict` formats. If no checkpoint is found, it still produce a valid submission, but the goal is to actually load your trained weights so the score moves toward the 0.884 target.'
- What this solution (achieved 0.61136) has done: 'Your score (0.099) is consistent with running an untrained/random ViT because the script never trains and is unlikely to find a usable checkpoint inside `/kaggle/input` (competition dataset doesn’t ship model weights). To move toward the 0.884 target while preserving your ViT+argmax inference semantics, the minimal legitimate fix is to add a standard training step on `train_images/train.csv` (same model architecture and loss: CrossEntropy), then run the same inference pipeline on the test set. I keep your transforms/TTA/loader structure, but add a train dataset/loader, train for a small fixed number of epochs within the 600s budget, save/load the best weights, and then generate `submission.csv` aligned to `sample_submission.csv`. This should materially increase accuracy compared with random weights without changing the core approach.'
- What this solution (achieved 0.67003) has done: 'Your current score is far below the target, and the main bottleneck is that you’re training ViT from scratch for only 2 epochs on 18k images, which can’t reach ~0.88 accuracy. Keeping the same ViT_B_16 architecture, CrossEntropy loss, and overall train→infer pipeline, the smallest high-impact change is to switch to ImageNet-pretrained weights and fine-tune (still the same model) while updating normalization to the official ViT weights’ preprocessing. I also add a simple stratified train/val split and save the best checkpoint by validation accuracy (aligned to the competition metric) rather than training loss, without changing inference semantics (argmax of mean TTA probabilities). These changes should move the score materially upward toward the 0.884 target while staying within runtime and keeping core logic intact.'

# 9. Code solution

## === cell 0
import os
import glob
import time
import random

import pandas as pd
import torch
from PIL import Image
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
from torchvision.transforms import InterpolationMode, v2
import torchvision

torch.manual_seed(3407)
random.seed(3407)
if torch.cuda.is_available():
    torch.cuda.manual_seed(3407)

cudnn.deterministic = True
cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

base_dir = "/kaggle/input/cassava-leaf-disease-classification"
train_csv_path = f"{base_dir}/train.csv"
train_dir = f"{base_dir}/train_images/"
test_dir = f"{base_dir}/test_images/"
sample_sub_path = f"{base_dir}/sample_submission.csv"

img_size = 224

batch_size = 16
num_workers = 4
num_classes = 5
tta = True

epochs = 2
lr = 3e-4
weight_decay = 1e-4
train_print_every = 50

use_pretrained = True


def _find_checkpoint():
    local_candidates = [
        "best_vit_b16.pth",
        "vit_b16_trained.pth",
        "submission_model.pth",
    ]
    for p in local_candidates:
        if os.path.exists(p):
            return p

    search_roots = [
        "/kaggle/input/cassava-leaf-disease-classification",
        "/kaggle/working",
        "/kaggle/input",
    ]
    patterns = ("*.pt", "*.pth", "*.bin")

    candidates = []
    for root in search_roots:
        for pat in patterns:
            candidates.extend(glob.glob(os.path.join(root, "**", pat), recursive=True))

    preferred_names = {
        "vit_v6.pt",
        "vit-v6.pt",
        "vitv6.pt",
        "model.pth",
        "checkpoint.pth",
        "best.pth",
        "best_vit_b16.pth",
    }
    preferred = [
        p for p in candidates if os.path.basename(p).lower() in preferred_names
    ]
    if preferred:
        return preferred[0]

    def _rank(p):
        p_lower = p.lower()
        in_comp = "cassava-leaf-disease-classification" in p_lower
        depth = p_lower.count(os.sep)
        return (0 if in_comp else 1, depth)

    candidates = sorted(set(candidates), key=_rank)
    return candidates[0] if candidates else None


def _build_model(num_classes=5, use_pretrained=True):
    if use_pretrained:
        weights = torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1
        m = torchvision.models.vit_b_16(weights=weights)
    else:
        m = torchvision.models.vit_b_16(weights=None)

    in_features = m.heads.head.in_features
    m.heads.head = torch.nn.Linear(in_features, num_classes)
    return m


def _load_model_from_checkpoint(ckpt_path, num_classes=5, use_pretrained=True):
    m = _build_model(num_classes=num_classes, use_pretrained=use_pretrained)

    obj = None
    try:
        obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        obj = torch.load(ckpt_path, map_location="cpu")
    except Exception as e:
        print("torch.load failed:", repr(e))
        return None

    if isinstance(obj, torch.nn.Module):
        return obj

    if isinstance(obj, dict):
        state = None
        for k in ("state_dict", "model_state_dict", "model", "net", "weights"):
            if k in obj and isinstance(obj[k], dict):
                state = obj[k]
                break
        if state is None and all(isinstance(k, str) for k in obj.keys()):
            state = obj

        if state is None:
            print("Checkpoint dict did not contain a usable state_dict-like object.")
            return None

        cleaned = {}
        for k, v in state.items():
            nk = k
            if nk.startswith("module."):
                nk = nk[len("module.") :]
            if nk.startswith("model."):
                nk = nk[len("model.") :]
            cleaned[nk] = v

        missing, unexpected = m.load_state_dict(cleaned, strict=False)
        print(
            "Loaded state_dict with strict=False. Missing keys:",
            len(missing),
            "Unexpected keys:",
            len(unexpected),
        )
        return m

    print("Unsupported checkpoint object type:", type(obj))
    return None


ckpt_path = _find_checkpoint()
print("Checkpoint found:", ckpt_path)

model = None
if ckpt_path is not None and os.path.exists(ckpt_path):
    model = _load_model_from_checkpoint(
        ckpt_path, num_classes=num_classes, use_pretrained=use_pretrained
    )
    if model is not None:
        print("Loaded checkpoint:", ckpt_path)
    else:
        print("Failed to interpret checkpoint contents; will train a new model.")

if model is None:
    model = _build_model(num_classes=num_classes, use_pretrained=use_pretrained)
    print("Initialized torchvision vit_b_16 (will train). Pretrained:", use_pretrained)

if hasattr(model, "image_size"):
    try:
        img_size = int(model.image_size)
        print("Using model.image_size:", img_size)
    except Exception:
        pass

model = model.to(device)




## === cell 1
class CassavaTrainDataset(VisionDataset):
    """Train dataset (image_id + label) using the same image loading path as test."""

    def __init__(self, data_dir, df, transform=None):
        super().__init__(root=data_dir)
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __getitem__(self, idx):
        image_id = self.df.loc[idx, "image_id"]
        label = int(self.df.loc[idx, "label"])
        img = Image.open(os.path.join(self.root, image_id)).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.df)


class CassavaDataset(VisionDataset):
    """Custom dataset for Cassava test data (image_id only)."""

    def __init__(self, data_dir, image_ids=None, transform=None, ttas=None):
        super().__init__(root=data_dir)
        self.transform = transform
        self.ttas = ttas
        self.cc = v2.CenterCrop((600, 600))

        if image_ids is None:
            self.images = sorted(os.listdir(data_dir))
        else:
            self.images = list(image_ids)

    def __getitem__(self, idx):
        filename = self.images[idx]
        img = Image.open(os.path.join(self.root, filename)).convert("RGB")
        img = self.cc(img)

        if self.ttas is not None and self.transform is not None:
            img = [self.transform(t(img)) for t in self.ttas]
        elif self.transform:
            img = self.transform(img)

        return img, filename

    def __len__(self):
        return len(self.images)




## === cell 2
vit_weights = torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1
vit_mean = list(vit_weights.transforms().mean)
vit_std = list(vit_weights.transforms().std)

train_transforms = v2.Compose(
    [
        v2.Resize((img_size, img_size), interpolation=InterpolationMode.BICUBIC),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=vit_mean, std=vit_std),
    ]
)

test_transforms = v2.Compose(
    [
        v2.Resize((img_size, img_size), interpolation=InterpolationMode.BICUBIC),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=vit_mean, std=vit_std),
    ]
)

if tta:
    ttas = [
        v2.Identity(),
        v2.RandomHorizontalFlip(p=1.0),
        v2.RandomVerticalFlip(p=1.0),
    ]
else:
    ttas = None

train_df = pd.read_csv(train_csv_path)
sample_sub = pd.read_csv(sample_sub_path)
test_image_ids = sample_sub["image_id"].tolist()

val_frac = 0.1
train_df = train_df.sample(frac=1.0, random_state=3407).reset_index(drop=True)

val_parts = []
train_parts = []
for lbl, g in train_df.groupby("label", sort=False):
    n_val = max(1, int(round(len(g) * val_frac)))
    val_parts.append(g.iloc[:n_val])
    train_parts.append(g.iloc[n_val:])
train_df_split = pd.concat(train_parts, axis=0).reset_index(drop=True)
val_df_split = pd.concat(val_parts, axis=0).reset_index(drop=True)

train_dataset = CassavaTrainDataset(
    train_dir, train_df_split, transform=train_transforms
)
val_dataset = CassavaTrainDataset(train_dir, val_df_split, transform=test_transforms)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
)

test_dataset = CassavaDataset(
    test_dir, image_ids=test_image_ids, transform=test_transforms, ttas=ttas
)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
)

normalizer = torch.nn.Softmax(dim=1)



## === cell 3
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

best_acc = -1.0
best_path = "best_vit_b16.pth"

start_time = time.time()
for epoch in range(1, epochs + 1):
    model.train()
    running_loss = 0.0
    n_seen = 0

    for step, (x, y) in enumerate(train_loader, start=1):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        bsz = x.size(0)
        running_loss += loss.item() * bsz
        n_seen += bsz

        if step % train_print_every == 0:
            print(
                f"epoch {epoch}/{epochs} step {step}/{len(train_loader)} "
                f"loss {running_loss/n_seen:.4f} elapsed {time.time()-start_time:.1f}s"
            )

    epoch_loss = running_loss / max(1, n_seen)
    print(f"epoch {epoch}/{epochs} train_loss {epoch_loss:.4f}")

    model.eval()
    correct = 0
    total = 0
    val_loss_sum = 0.0
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)
            val_loss_sum += loss.item() * x.size(0)

            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.numel()

    val_acc = correct / max(1, total)
    val_loss = val_loss_sum / max(1, total)
    print(f"epoch {epoch}/{epochs} val_loss {val_loss:.4f} val_acc {val_acc:.4f}")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save({"state_dict": model.state_dict()}, best_path)
        print("Saved best checkpoint to:", best_path, "val_acc:", best_acc)

loaded = _load_model_from_checkpoint(
    best_path, num_classes=num_classes, use_pretrained=use_pretrained
)
if loaded is not None:
    model = loaded.to(device)
    print("Reloaded best checkpoint for inference:", best_path)
else:
    print("Warning: failed to reload best checkpoint; using current in-memory model.")



## === cell 4
all_names = []
all_preds = []

model.eval()

with torch.no_grad():
    for batch_idx, (inputs, filenames) in enumerate(test_loader):
        if tta:
            bsz = len(filenames)
            inputs = torch.cat(inputs, dim=0).to(
                device, non_blocking=True
            )  # [T*B, C, H, W]
            preds = normalizer(model(inputs))  # [T*B, num_classes]

            t = len(ttas)
            preds = preds.view(t, bsz, -1).mean(dim=0)  # [B, num_classes]
            pred_labels = torch.argmax(preds, dim=1).tolist()
        else:
            inputs = inputs.to(device, non_blocking=True)
            preds = normalizer(model(inputs))
            pred_labels = torch.argmax(preds, dim=1).tolist()

        all_names.extend(list(filenames))
        all_preds.extend(pred_labels)

assert len(all_names) == len(
    test_image_ids
), f"Pred count {len(all_names)} != expected {len(test_image_ids)}"
assert len(all_preds) == len(
    test_image_ids
), f"Label count {len(all_preds)} != expected {len(test_image_ids)}"

my_submission = pd.DataFrame({"image_id": all_names, "label": all_preds})



## === cell 5
my_submission = (
    my_submission.set_index("image_id").loc[sample_sub["image_id"]].reset_index()
)
my_submission["label"] = my_submission["label"].astype(int)

out_path = "submission.csv"
my_submission.to_csv(out_path, index=False)
print("Wrote:", out_path, "rows:", len(my_submission))
print(my_submission.head())
