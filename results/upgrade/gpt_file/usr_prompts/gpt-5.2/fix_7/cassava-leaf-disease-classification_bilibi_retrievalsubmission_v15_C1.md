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

3.9

# 3. Installed packages

geopandas==0.14.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
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

0.8689936536718041

# 6. Current score

0.76196

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.10314) has done: 'The main issue is that this notebook depends on external Kaggle Dataset code (`../input/baseline-infer`, `../input/configs`, `../input/baseline-weights`) that is not available in your environment, causing import failures and preventing a valid `submission.csv` from being produced. To keep the solution runnable end-to-end while preserving the overall inference workflow (load model → build dataset/dataloader → predict → write `image_id,label`), I replace those missing modules with a self-contained PyTorch inference baseline using `torchvision`’s ResNet50 and standard imagenet preprocessing. I also fix the submission-length issue by reading `sample_submission.csv` and predicting exactly in that order (no reliance on `os.listdir` ordering), guaranteeing the required 2676 rows. Finally, I ensure the script writes `submission.csv` with the correct columns and integer labels 0–4.'
- What this solution (achieved 0.75224) has done: 'Your current score is low because the model’s 5-class classification head is random unless the missing external weights exist, so predictions are essentially arbitrary. To move toward the target score while preserving your exact inference pipeline (ResNet50 + argmax logits + same preprocessing + same dataloader loop), the minimal legitimate fix is to actually train that final head on the provided `train.csv`/`train_images` in this environment and then run inference on the test set. I freeze the ImageNet-pretrained backbone and train only `model.fc` for a small number of epochs, which keeps the core model architecture and prediction semantics unchanged while substantially improving accuracy. I also keep the submission ordering tied to `sample_submission.csv` exactly as you already do, ensuring a valid 2676-row `submission.csv`.'
- What this solution (achieved 0.69058) has done: 'Your gap to the target is ~0.1168 (0.75224 → 0.86899), so we should improve accuracy with minimal, low-risk changes without altering the core pipeline (ResNet50 backbone + linear fc head + CrossEntropy + argmax). The biggest limitation is training only the final head from a random init on a highly imbalanced dataset; we can move closer to the target by (1) initializing `model.fc` using a closed-form linear classifier (ridge regression) on frozen ResNet features to give the head a strong starting point, and then (2) fine-tuning that same head with your existing Adam loop. Additionally, using class-weighted CrossEntropy counters label imbalance and typically boosts accuracy for Cassava without changing evaluation semantics. These changes keep architecture and inference identical, only improving the head’s training dynamics.'
- What this solution (achieved 0.68647) has done: 'To move your score upward toward 0.86899 without changing the core ResNet50+linear-head+CrossEntropy/argmax pipeline, I make the head-training more effective while keeping the same overall approach (freeze backbone, train only `model.fc`, same preprocessing and inference). The main minimal lift is to use the already-computed frozen features to train the linear head with a proper softmax linear classifier objective (multinomial logistic regression) rather than squared-loss ridge, then keep your exact Adam fine-tuning loop. I also switch the loss weighting from inverse-frequency to a gentler, commonly effective `1/sqrt(freq)` weighting to avoid over-correcting imbalance (which can hurt overall accuracy). Finally, I add a lightweight, deterministic train/val split purely for monitoring (no early stopping, no budget changes) to catch obvious issues without affecting submission semantics.'
- What this solution (achieved 0.77392) has done: 'Your current score (0.68647) is far below the target (0.86899), so we should improve accuracy with the smallest changes that keep your exact ResNet50+linear-head, CrossEntropy, and argmax inference semantics intact. The biggest low-risk gain here is to unfreeze and fine-tune the last ResNet block (`layer4`) together with `fc` after your existing logistic-regression head initialization, using a smaller LR for `layer4` and keeping the same epoch budget to stay within time. This typically yields a material jump on Cassava vs training only `fc`, without changing architecture or data processing. I also switch the resize interpolation to `INTER_LINEAR` (closer to torchvision’s default) to better match ImageNet-pretraining behavior, which is a minimal preprocessing tweak that often improves transfer accuracy.'
- What this solution (achieved 0.76196) has done: 'Your current score (0.77392) is below the target (0.86899), so we should make small, low-risk changes that keep the same ResNet50 + CrossEntropy + argmax semantics while improving generalization. The highest-impact minimal fix is to add standard ImageNet-style training augmentation (random resized crop + horizontal flip) only for the training dataset; inference preprocessing remains unchanged and aligned to the submission metric. To better match ResNet50 pretraining and stabilize fine-tuning, we also keep the backbone frozen except `layer4`+`fc` as you already do, but add a small weight decay to Adam and use a slightly more appropriate split (stratified) for monitoring without changing the training loop structure. These changes are targeted to raise accuracy toward the target without altering the model architecture or prediction format.'

# 9. Code solution

## === cell 0
import os
import sys
import copy
import datetime
import random

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import cv2
from torchvision import models


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)

DATA_ROOT = "../input/cassava-leaf-disease-classification"
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test_images")
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train_images")
SAMPLE_SUB_PATH = os.path.join(DATA_ROOT, "sample_submission.csv")
TRAIN_CSV_PATH = os.path.join(DATA_ROOT, "train.csv")

assert os.path.exists(SAMPLE_SUB_PATH), f"Missing {SAMPLE_SUB_PATH}"
assert os.path.isdir(TEST_IMG_DIR), f"Missing {TEST_IMG_DIR}"
assert os.path.exists(TRAIN_CSV_PATH), f"Missing {TRAIN_CSV_PATH}"
assert os.path.isdir(TRAIN_IMG_DIR), f"Missing {TRAIN_IMG_DIR}"

sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
assert list(sample_sub.columns) == [
    "image_id",
    "label",
], "Unexpected sample_submission.csv columns"

train_df = pd.read_csv(TRAIN_CSV_PATH)
assert list(train_df.columns) == ["image_id", "label"], "Unexpected train.csv columns"

print("sample_submission rows:", len(sample_sub))
print("train rows:", len(train_df))
print(
    "test_images files:",
    len([x for x in os.listdir(TEST_IMG_DIR) if x.endswith(".jpg")]),
)
print(
    "train_images files:",
    len([x for x in os.listdir(TRAIN_IMG_DIR) if x.endswith(".jpg")]),
)



## === cell 1
NUM_CLASSES = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, NUM_CLASSES)
model = model.to(device)

weights_path = "../input/baseline-weights/weight_epoch_21.pth"
loaded_external = False
if os.path.exists(weights_path):
    state = torch.load(weights_path, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    print("Loaded external weights:", weights_path)
    print("Missing keys:", len(missing), "Unexpected keys:", len(unexpected))
    loaded_external = True
else:
    print("No external weights found at:", weights_path)
    print(
        "Will train a lightweight head on train.csv to improve accuracy in this environment."
    )



## === cell 2
IM_SIZE = 224
IM_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IM_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_bgr_uint8(img_bgr: np.ndarray) -> torch.Tensor:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (IM_SIZE, IM_SIZE), interpolation=cv2.INTER_LINEAR)
    x = img_rgb.astype(np.float32) / 255.0
    x = (x - IM_MEAN) / IM_STD
    x = np.transpose(x, (2, 0, 1))  # CHW
    return torch.from_numpy(x).float()


def preprocess_train_bgr_uint8(
    img_bgr: np.ndarray, rng: np.random.RandomState
) -> torch.Tensor:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]

    area = h * w
    for _ in range(10):
        target_area = rng.uniform(0.70, 1.00) * area
        aspect = rng.uniform(3.0 / 4.0, 4.0 / 3.0)
        crop_w = int(round(np.sqrt(target_area * aspect)))
        crop_h = int(round(np.sqrt(target_area / aspect)))
        if crop_w <= w and crop_h <= h and crop_w > 0 and crop_h > 0:
            x1 = rng.randint(0, w - crop_w + 1)
            y1 = rng.randint(0, h - crop_h + 1)
            img_rgb = img_rgb[y1 : y1 + crop_h, x1 : x1 + crop_w]
            break
    else:
        side = min(h, w)
        x1 = (w - side) // 2
        y1 = (h - side) // 2
        img_rgb = img_rgb[y1 : y1 + side, x1 : x1 + side]

    if rng.rand() < 0.5:
        img_rgb = np.ascontiguousarray(img_rgb[:, ::-1, :])

    img_rgb = cv2.resize(img_rgb, (IM_SIZE, IM_SIZE), interpolation=cv2.INTER_LINEAR)
    x = img_rgb.astype(np.float32) / 255.0
    x = (x - IM_MEAN) / IM_STD
    x = np.transpose(x, (2, 0, 1))
    return torch.from_numpy(x).float()


class TrainSet(Dataset):
    def __init__(
        self, img_dir: str, df: pd.DataFrame, augment: bool = False, seed: int = 42
    ):
        self.img_dir = img_dir
        self.image_ids = df["image_id"].tolist()
        self.labels = df["label"].astype(int).tolist()
        self.augment = augment
        self.base_seed = int(seed)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        fn = self.image_ids[idx]
        y = self.labels[idx]
        path = os.path.join(self.img_dir, fn)
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {path}")
        if self.augment:
            rng = np.random.RandomState(self.base_seed + idx)
            x = preprocess_train_bgr_uint8(img, rng)
        else:
            x = preprocess_bgr_uint8(img)
        return x, torch.tensor(y, dtype=torch.long)


class InferSet(Dataset):
    def __init__(self, img_dir: str, image_ids: list[str]):
        self.img_dir = img_dir
        self.image_ids = image_ids

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        fn = self.image_ids[idx]
        path = os.path.join(self.img_dir, fn)
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {path}")
        x = preprocess_bgr_uint8(img)
        return x, fn




## === cell 3
if not loaded_external:
    for p in model.parameters():
        p.requires_grad = False
    for p in model.fc.parameters():
        p.requires_grad = True

    rng = np.random.RandomState(42)
    val_frac = 0.1
    val_indices = []
    for c in range(NUM_CLASSES):
        idx_c = np.where(train_df["label"].values == c)[0]
        rng.shuffle(idx_c)
        n_val_c = int(round(val_frac * len(idx_c)))
        val_indices.append(idx_c[:n_val_c])
    val_idx = np.concatenate(val_indices)
    val_mask = np.zeros(len(train_df), dtype=bool)
    val_mask[val_idx] = True
    trn_idx = np.where(~val_mask)[0]

    train_df_trn = train_df.iloc[trn_idx].reset_index(drop=True)
    train_df_val = train_df.iloc[val_idx].reset_index(drop=True)

    train_ds_noaug = TrainSet(TRAIN_IMG_DIR, train_df_trn, augment=False, seed=42)
    val_ds = TrainSet(TRAIN_IMG_DIR, train_df_val, augment=False, seed=42)

    feat_loader = DataLoader(
        train_ds_noaug,
        batch_size=64,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    backbone = copy.deepcopy(model).to(device)
    backbone.fc = nn.Identity()
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    X_chunks = []
    y_chunks = []
    with torch.no_grad():
        for xb, yb in feat_loader:
            xb = xb.to(device, non_blocking=True)
            feats = backbone(xb).detach().cpu()  # [B, 2048]
            X_chunks.append(feats)
            y_chunks.append(yb.cpu())
    X = torch.cat(X_chunks, dim=0).float()  # [N, D]
    y = torch.cat(y_chunks, dim=0).long()  # [N]
    N, D = X.shape

    Xb = torch.cat([X, torch.ones(N, 1)], dim=1)  # [N, D+1]
    y_np = y.numpy()

    counts = train_df_trn["label"].value_counts().sort_index()
    counts = counts.reindex(range(NUM_CLASSES), fill_value=0).values.astype(np.float32)
    w = 1.0 / np.sqrt(np.maximum(counts, 1.0))
    w = w / w.mean()
    class_weights_t = torch.tensor(w, dtype=torch.float32, device=device)

    Xb_cpu = Xb
    y_cpu = y

    W_t = torch.zeros(D + 1, NUM_CLASSES, dtype=torch.float32, requires_grad=True)
    opt = torch.optim.LBFGS([W_t], lr=1.0, max_iter=60, line_search_fn="strong_wolfe")

    sample_w = torch.tensor(w[y_np], dtype=torch.float32)

    def closure():
        opt.zero_grad(set_to_none=True)
        logits = Xb_cpu @ W_t  # [N, C]
        loss_per = torch.nn.functional.cross_entropy(logits, y_cpu, reduction="none")
        loss = (loss_per * sample_w).mean()
        loss = loss + 1e-4 * (W_t[:-1].pow(2).sum())
        loss.backward()
        return loss

    opt.step(closure)
    W = W_t.detach()

    with torch.no_grad():
        model.fc.weight.copy_(W[:D, :].T.contiguous())
        model.fc.bias.copy_(W[D, :].contiguous())

    train_ds = TrainSet(TRAIN_IMG_DIR, train_df_trn, augment=True, seed=42)

    train_loader = DataLoader(
        train_ds,
        batch_size=64,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=64,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    criterion = nn.CrossEntropyLoss(weight=class_weights_t)

    for p in model.layer4.parameters():
        p.requires_grad = True

    optimizer = torch.optim.Adam(
        [
            {"params": model.layer4.parameters(), "lr": 1e-4},
            {"params": model.fc.parameters(), "lr": 1e-3},
        ],
        weight_decay=1e-4,
    )

    model.train()
    epochs = 4  # keep same training budget to stay within time and preserve core loop
    for epoch in range(epochs):
        running_loss = 0.0
        seen = 0
        correct = 0

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            bs = xb.size(0)
            running_loss += loss.item() * bs
            seen += bs
            correct += (logits.argmax(1) == yb).sum().item()

        model.eval()
        v_seen, v_correct = 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                logits = model(xb)
                v_seen += xb.size(0)
                v_correct += (logits.argmax(1) == yb).sum().item()
        model.train()

        print(
            f"epoch {epoch+1}/{epochs} - loss: {running_loss/seen:.4f} - train_acc: {correct/seen:.4f} - val_acc: {v_correct/v_seen:.4f}"
        )

    model.eval()
else:
    model.eval()

infer_ds = InferSet(TEST_IMG_DIR, sample_sub["image_id"].tolist())
infer_loader = DataLoader(
    infer_ds,
    batch_size=64,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)

xb, fnb = next(iter(infer_loader))
print("batch tensor shape:", xb.shape, "first filename:", fnb[0])



## === cell 4
pred_labels = []

with torch.no_grad():
    for xb, fnb in infer_loader:
        xb = xb.to(device, non_blocking=True)
        logits = model(xb)
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy().astype(int).tolist()
        pred_labels.extend(preds)

assert len(pred_labels) == len(
    sample_sub
), f"Pred length {len(pred_labels)} != sample_sub length {len(sample_sub)}"

sub = pd.DataFrame({"image_id": sample_sub["image_id"].values, "label": pred_labels})
sub["label"] = sub["label"].astype(int)

out_path = "./submission.csv"
sub.to_csv(out_path, index=False)

print("Wrote:", out_path)
print(sub.head())
print("rows:", len(sub), "cols:", sub.columns.tolist())



## === cell 5
chk = pd.read_csv("./submission.csv")
assert list(chk.columns) == ["image_id", "label"]
assert len(chk) == len(sample_sub)
assert chk["label"].between(0, 4).all()
print("Submission looks valid. Length:", len(chk))
