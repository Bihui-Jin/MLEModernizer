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

3.9

# 3. Installed packages

albumentations==2.0.8
geopandas==0.14.4
numpy==1.26.4
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

0.8278936234511937

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.61061) has done: 'I make the notebook run end-to-end by (1) fixing the missing model weights path issue by falling back to a local lightweight training of the same SqueezeNet-1.0 head when the external `.pth` file isn’t available, (2) updating the Albumentations `RandomResizedCrop`/deprecated transforms to the v2 API so augmentations build correctly, and (3) ensuring device handling, image mode conversion, and submission writing are robust and match the required `image_id,label` format. These changes preserve the core model architecture (SqueezeNet with a 5-class conv head) and the same inference approach (TTA averaging over multiple augmented views), while guaranteeing a valid `submission.csv` is produced. Since no score was yielded previously, the goal is correctness + a reasonable baseline accuracy from quick training within the time limit.'
- What this solution (achieved 0.61771) has done: 'Your current score (0.61061) is well below the target (0.82789), so we should make small, low-risk changes that legitimately improve accuracy without changing the model or training loop structure. The biggest issue is that SqueezeNet expects 224×224 inputs; using 256×256 plus heavy TTA-time augmentations (including CoarseDropout) can hurt accuracy, so we align train/val/test preprocessing to a standard 224 resize/crop and make test-time augmentation “safe” (flip-only, no dropout/color/rotate). We also fix a subtle but important bug: you currently apply `ToTensor()` after Albumentations `Normalize`, which makes tensors be re-scaled by 1/255 again; instead we have Albumentations output a PyTorch tensor directly via `ToTensorV2()` to preserve correct normalization. These changes keep the same architecture, loss, optimizer, and overall training/inference approach (still TTA averaging), but should move the score substantially toward the target.'
- What this solution (achieved 0.05531) has done: 'Your current score (0.61771) is far below the target (0.82789), so we should make a small but meaningful accuracy improvement without changing the core model or training/inference structure. The biggest gain with minimal risk is to address class imbalance using class-weighted CrossEntropy (same loss, just weighted), and to make the fallback training slightly stronger by training for a few more epochs when external weights aren’t available. These changes preserve the SqueezeNet architecture, the same optimization approach (Adam), and the same TTA averaging at inference, but should move accuracy materially upward. I’m also keeping preprocessing and the ToTensorV2 normalization flow intact since that was already a key correctness fix.'
- What this solution (achieved 0.05531) has done: 'Your current score (0.05531) is far below the target (0.82789), and the most likely cause is a mismatch between the model weights you load and the classifier head you define (so `weights_loaded` can be true but the head is effectively random/misaligned). I minimally fix weight loading to be robust: load with `strict=False`, and if keys don’t match (or file missing) we fall back to the existing local training path. I also make training deterministic and ensure the model is set to `train()` before training (your code already mostly does this), without changing the architecture, loss type, optimizer type, or inference/TTA semantics. This should restore the expected baseline accuracy and move the score materially toward the target.'
- What this solution (achieved 0.61099) has done: 'Your current score (0.05531) is far below the target (0.82789), so we should make the smallest fixes that plausibly recover the expected baseline accuracy without changing the model/training core. The most likely cause is that test-time augmentation is effectively random (horizontal flip p=0.5) and repeated 6 times per image, which just injects noise and can collapse accuracy; we replace it with deterministic TTA (original + horizontal flip) while keeping the same “TTA averaging logits” inference semantics. We also stop using `sample_submission.iterrows()` for inference order and instead run a proper `Dataset/DataLoader` over `sample_submission` to avoid any subtle dtype/order issues and to make inference consistent and faster (no change to prediction logic). Finally, we keep the same SqueezeNet architecture, same loss type, same optimizer type, and same training loop structure, only improving determinism and correctness in inference.'
- What this solution (achieved 0.05531) has done: 'We keep your SqueezeNet architecture, loss, optimizer type, and the same “TTA = average logits over (base + hflip)” inference semantics, but fix two accuracy killers with minimal risk. First, we align the train/val split to be stratified by label (instead of pure random), which stabilizes training and improves generalization without changing the training loop itself. Second, we add label-smoothing to CrossEntropy (still CrossEntropyLoss) to reduce overconfidence and typically improve top-1 accuracy on this dataset; class-weights are preserved. These changes should move your score upward toward the 0.8279 target while remaining within Kaggle constraints and producing the same submission format.'
- What this solution (achieved 0.05531) has done: 'Your current score (0.05531) is far below the target (0.82789), so we should make the smallest fixes that plausibly recover a sane accuracy baseline without changing the model architecture or training/inference semantics. The most likely accuracy killer left is that, when external weights are missing, you train the whole SqueezeNet from scratch for only a few epochs—this typically won’t reach good accuracy. I keep the exact same SqueezeNet-1.0 + 5-class conv head and the same training loop, but switch the backbone initialization to ImageNet pretrained weights only in the fallback-training path (this is not an architecture change, just a standard initialization). I also fix deterministic seeding more completely (incl. CuDNN determinism) to stabilize results and prevent accidental regressions.'
- What this solution (achieved 0.05531) has done: 'Your score (0.05531) is far below the target (0.82789), so we should make the smallest changes that plausibly recover a sane baseline without changing your SqueezeNet architecture, loss type, optimizer type, or the overall training/inference (TTA-averaged logits) semantics. The biggest likely remaining issue is that the model’s outputs are spatial (N,5,H,W) and need global-average pooling to become class logits (N,5); otherwise `argmax(dim=1)` produces (N,H,W) and the whole pipeline silently breaks (and training loss shape is wrong/unstable). I add a minimal wrapper that performs the required adaptive average pooling both during training and inference (no new layers/parameters; it’s just the missing reduction step). I also make weight-loading/training use this wrapper consistently so predictions are well-formed and aligned with the submission format.'

# 9. Code solution

## === cell 0
import os

import albumentations
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models



## === cell 1
model_path = "../input/sn-wc-aug/model(4).pth"
sample_sub_path = "../input/cassava-leaf-disease-classification/sample_submission.csv"
test_images_path = "../input/cassava-leaf-disease-classification/test_images"

train_csv_path = "../input/cassava-leaf-disease-classification/train.csv"
train_images_path = "../input/cassava-leaf-disease-classification/train_images"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(False)




## === cell 2
class SqueezeNetImageLogits(nn.Module):
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base = base_model

    def forward(self, x):
        out = self.base(x)  # may be (N, C) or (N, C, H, W)
        if out.dim() == 4:
            out = torch.nn.functional.adaptive_avg_pool2d(out, (1, 1)).flatten(1)
        return out


def build_model(pretrained: bool) -> nn.Module:
    base_model = models.squeezenet1_0(pretrained=pretrained)
    base_model.classifier[1] = nn.Conv2d(
        in_channels=512, out_channels=5, kernel_size=(1, 1), stride=(1, 1)
    )
    base_model.num_classes = 5
    return SqueezeNetImageLogits(base_model)


model = build_model(pretrained=False).to(device)


def _extract_state_dict(obj):
    if (
        isinstance(obj, dict)
        and "state_dict" in obj
        and isinstance(obj["state_dict"], dict)
    ):
        return obj["state_dict"]
    if isinstance(obj, dict):
        return obj
    return None


def _try_load_with_prefix_handling(m: nn.Module, raw_state: dict):
    """
    Returns (loaded_ok: bool, missing: list, unexpected: list)
    loaded_ok is True only if the classifier conv weights are loaded.
    """
    candidates = []

    candidates.append(raw_state)

    cand = {}
    for k, v in raw_state.items():
        nk = k[len("module.") :] if k.startswith("module.") else k
        cand[nk] = v
    candidates.append(cand)

    cand2 = {}
    for k, v in cand.items():
        nk = k
        if not nk.startswith("base."):
            nk = "base." + nk
        cand2[nk] = v
    candidates.append(cand2)

    cand3 = {}
    for k, v in cand2.items():
        nk = k.replace("base.base.", "base.")
        cand3[nk] = v
    candidates.append(cand3)

    head_w_key = "base.classifier.1.weight"
    head_b_key = "base.classifier.1.bias"
    for st in candidates:
        try:
            missing, unexpected = m.load_state_dict(st, strict=False)

            provided_has_head = (head_w_key in st) and (head_b_key in st)
            head_missing = (head_w_key in missing) or (head_b_key in missing)

            if provided_has_head and (not head_missing):
                return True, missing, unexpected
        except Exception:
            continue
    return False, [], []


weights_loaded = False
if os.path.exists(model_path):
    state_obj = torch.load(model_path, map_location="cpu")
    state = _extract_state_dict(state_obj)
    if isinstance(state, dict):
        weights_loaded, missing, unexpected = _try_load_with_prefix_handling(
            model, state
        )

model.eval()
print(f"Device: {device} | External weights loaded: {weights_loaded}")



## === cell 3
from torch.utils.data import Dataset, DataLoader
from albumentations.pytorch import ToTensorV2


class CassavaTrainDataset(Dataset):
    def __init__(self, df, images_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.images_dir, row.image_id)
        img = Image.open(img_path).convert("RGB")
        img = np.array(img)
        if self.transform is not None:
            img = self.transform(image=img)["image"]
        y = int(row.label)
        return img, y


IMG_SIZE = 224

train_aug = albumentations.Compose(
    [
        albumentations.RandomResizedCrop(
            size=(IMG_SIZE, IMG_SIZE),
            scale=(0.8, 1.0),
            ratio=(0.75, 1.3333333333),
            p=1.0,
        ),
        albumentations.HorizontalFlip(p=0.5),
        albumentations.VerticalFlip(p=0.2),
        albumentations.ShiftScaleRotate(
            shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5
        ),
        albumentations.HueSaturationValue(
            hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.5
        ),
        albumentations.RandomBrightnessContrast(
            brightness_limit=0.1, contrast_limit=0.1, p=0.5
        ),
        albumentations.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0,
        ),
        ToTensorV2(),
    ],
    p=1.0,
)



## === cell 4
sub_aug_base = albumentations.Compose(
    [
        albumentations.Resize(IMG_SIZE, IMG_SIZE),
        albumentations.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0,
        ),
        ToTensorV2(),
    ],
    p=1.0,
)

sub_aug_hflip = albumentations.Compose(
    [
        albumentations.Resize(IMG_SIZE, IMG_SIZE),
        albumentations.HorizontalFlip(p=1.0),
        albumentations.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0,
        ),
        ToTensorV2(),
    ],
    p=1.0,
)



## === cell 5
if not weights_loaded:
    model = build_model(pretrained=True).to(device)

    df = pd.read_csv(train_csv_path)

    rng = np.random.RandomState(SEED)
    train_parts = []
    val_parts = []
    for lab, g in df.groupby("label"):
        idx = g.index.to_numpy()
        rng.shuffle(idx)
        split = int(0.9 * len(idx))
        train_parts.append(df.loc[idx[:split]])
        val_parts.append(df.loc[idx[split:]])
    train_df = (
        pd.concat(train_parts, axis=0)
        .sample(frac=1.0, random_state=SEED)
        .reset_index(drop=True)
    )
    val_df = pd.concat(val_parts, axis=0).reset_index(drop=True)

    train_ds = CassavaTrainDataset(train_df, train_images_path, transform=train_aug)
    val_ds = CassavaTrainDataset(
        val_df,
        train_images_path,
        transform=albumentations.Compose(
            [
                albumentations.Resize(IMG_SIZE, IMG_SIZE),
                albumentations.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0,
                    p=1.0,
                ),
                ToTensorV2(),
            ],
            p=1.0,
        ),
    )

    batch_size = 64 if torch.cuda.is_available() else 16
    num_workers = 2

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    class_counts = train_df["label"].value_counts().sort_index()
    counts = np.zeros(5, dtype=np.float32)
    for k, v in class_counts.items():
        if 0 <= int(k) < 5:
            counts[int(k)] = float(v)
    counts = np.maximum(counts, 1.0)
    inv = 1.0 / counts
    weights = inv * (len(train_df) / inv.sum())
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    def accuracy_on_loader(m, loader):
        m.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                logits = m(xb)
                preds = logits.argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += yb.numel()
        return correct / max(1, total)

    epochs = 4 if torch.cuda.is_available() else 2

    for ep in range(epochs):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * yb.size(0)

        train_loss = running_loss / len(train_ds)
        val_acc = accuracy_on_loader(model, val_loader)
        print(
            f"Epoch {ep+1}/{epochs} | train_loss={train_loss:.4f} | val_acc={val_acc:.4f}"
        )

    model.eval()




## === cell 6
class CassavaTestDataset(Dataset):
    def __init__(self, df, images_dir):
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_id = self.df.loc[idx, "image_id"]
        img_path = os.path.join(self.images_dir, image_id)
        img = Image.open(img_path).convert("RGB")
        img = np.array(img)
        return image_id, img


sample_sub = pd.read_csv(sample_sub_path)
test_ds = CassavaTestDataset(sample_sub, test_images_path)

infer_bs = 128 if torch.cuda.is_available() else 16
test_loader = DataLoader(
    test_ds,
    batch_size=infer_bs,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)

predictions = []
model.eval()

with torch.no_grad():
    for image_ids, base_imgs in test_loader:
        logits_sum = None
        for aug in (sub_aug_base, sub_aug_hflip):
            batch_tensors = [aug(image=img)["image"] for img in base_imgs]
            xb = torch.stack(batch_tensors, dim=0).to(device, non_blocking=True)
            out = model(xb)
            logits_sum = out if logits_sum is None else (logits_sum + out)

        logits_mean = logits_sum / 2.0
        pred_labels = logits_mean.argmax(dim=1).detach().cpu().numpy().astype(int)

        for iid, lab in zip(list(image_ids), list(pred_labels)):
            predictions.append([iid, int(lab)])

sub_df = pd.DataFrame(predictions, columns=["image_id", "label"])
sub_df.to_csv("submission.csv", index=False)
print(sub_df.head())
print(f"Wrote submission.csv with {len(sub_df)} rows")

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2536694244.py in <cell line: 0>()
     36         logits_sum = None
     37         for aug in (sub_aug_base, sub_aug_hflip):
---> 38             batch_tensors = [aug(image=img)["image"] for img in base_imgs]
     39             xb = torch.stack(batch_tensors, dim=0).to(device, non_blocking=True)
     40             out = model(xb)

/tmp/ipykernel_55/2536694244.py in <listcomp>(.0)
     36         logits_sum = None
     37         for aug in (sub_aug_base, sub_aug_hflip):
---> 38             batch_tensors = [aug(image=img)["image"] for img in base_imgs]
     39             xb = torch.stack(batch_tensors, dim=0).to(device, non_blocking=True)
     40             out = model(xb)

/usr/local/lib/python3.11/dist-packages/albumentations/core/composition.py in __call__(self, force_apply, *args, **data)
    605             return data
    606 
--> 607         self.preprocess(data)
    608 
    609         for t in self.transforms:

/usr/local/lib/python3.11/dist-packages/albumentations/core/composition.py in preprocess(self, data)
    628                     continue
    629 
--> 630                 shape = self._get_data_shape(data_name, internal_name, data_value)
    631                 if shape is not None:
    632                     if internal_name in CHECKED_VOLUME | CHECKED_MASK3D:

/usr/local/lib/python3.11/dist-packages/albumentations/core/composition.py in _get_data_shape(self, data_name, internal_name, data)
    894         # Handle single images and masks
    895         if internal_name in CHECKED_SINGLE:
--> 896             return self._get_single_data_shape(data_name, data)
    897 
    898         # Handle volumes

/usr/local/lib/python3.11/dist-packages/albumentations/core/composition.py in _get_single_data_shape(self, data_name, data)
    913         """Get shape of single image or mask."""
    914         if not isinstance(data, np.ndarray):
--> 915             raise TypeError(f"{data_name} must be numpy array type")
    916         return data.shape
    917 

TypeError: image must be numpy array type
