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

0.8904502870957993

# 6. Current score

0.77653

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.05531) has done: 'I fix the pipeline so it can always load a model in this Kaggle environment (your hardcoded `/kaggle/input/efficient-net/vit_cont_3.pt` path doesn’t exist), while preserving the same inference approach (Softmax + argmax, optional TTA). I also fix the TTA batching logic (it currently mis-splits predictions) and make the DataLoader deterministic and ordered (no shuffle) so `image_id` alignment is stable. Finally, I ensure the submission has exactly the same `image_id` list/order/length as `sample_submission.csv`, which resolves the “same length as the answers” invalid submission error. These changes are bug-fixes and should improve accuracy versus the broken TTA averaging.'
- What this solution (achieved 0.05531) has done: 'I fix the immediate runtime failure by removing the hard dependency on an external `.pt/.pth` checkpoint (none exists in this environment) and instead instantiate a torchvision pretrained model so inference can run end-to-end. To move the score toward your target (and far above the current ~0.055), I use an ImageNet-pretrained EfficientNet-V2 backbone with the correct input normalization and size, while keeping your same inference semantics (Softmax + argmax, optional TTA averaging) and the same submission alignment safeguards. I also make TTA transforms deterministic per-worker and ensure the dataset reads the `sample_submission.csv` image order so `image_id` alignment stays correct. The output always be a valid `submission.csv` with the required columns and length.'
- What this solution (achieved 0.05531) has done: 'I fix the immediate runtime error by using the correct normalization metadata keys for torchvision 0.21 (it uses `categories`/`transforms` and doesn’t always expose `meta['mean']`/`meta['std']`). I also fix the logic issue that currently applies the main transform twice during TTA (once inside the TTA branch and again inside `self.transform(t(img))`), which badly corrupts inputs and explains the extremely low score. To keep core semantics the same, I preserve the same model, Softmax+argmax prediction, TTA averaging, and submission alignment, but I apply augmentation then the base preprocessing exactly once per TTA view. These changes are bug fixes (not architectural changes) and should move accuracy substantially toward the target by restoring valid preprocessing.'
- What this solution (achieved 0.10239) has done: 'I fix the TTA batching bug by making the DataLoader collate function explicitly preserve the per-sample list-of-T views (instead of PyTorch’s default collate trying to stack/transpose it into an unexpected structure). This directly resolves the `torch.stack(): ... must be tuple of Tensors, not Tensor` error and lets inference run end-to-end. I also fix the model head mismatch by replacing the ImageNet 1000-class classifier with a 5-class head (same backbone/inference semantics), which should move accuracy substantially toward your target because predictions be in the correct label space. Finally, I keep the submission alignment logic so the produced `submission.csv` matches `sample_submission.csv` exactly.'
- What this solution (achieved 0.75112) has done: 'Your score is far below the target, so the smallest safe way to move toward ~0.89 is to fix the main accuracy limiter: you’re doing inference with an ImageNet-pretrained backbone but a randomly-initialized 5-class head, so predictions are essentially random. To preserve your core inference semantics (same EfficientNetV2-S backbone, Softmax+argmax, optional TTA averaging, same submission alignment), I add a minimal training step on `train.csv` using standard cross-entropy and the same normalization/resize you already use, then run your existing test inference. I also make TTA less destructive by using simple geometric flips (still TTA averaging, but avoids heavy random crops/warps that tend to hurt without cassava-specific fine-tuning). The output remains a valid `submission.csv` matching `sample_submission.csv` exactly.'
- What this solution (achieved 0.77466) has done: 'You’re currently under the target (0.75112 vs 0.89045), so we should improve accuracy with the smallest changes that don’t alter your core model/inference semantics. The main limiter is that you only train the classifier head while keeping the backbone frozen; on cassava this typically leaves a lot of performance on the table, so we minimally fine-tune the last EfficientNetV2 block as well (not changing architecture/loss/inference). To keep it stable and within the time budget, we use a small LR for the unfrozen block and keep your same 2 epochs, data pipeline, Softmax+argmax, and TTA averaging. This usually yields a meaningful boost toward ~0.89 while remaining a minimal, legitimate change.'
- What this solution (achieved 0.75374) has done: 'Your current score (0.77466) is still well below the target (0.89045), so we should improve accuracy with the smallest legitimate change that doesn’t alter your core model/inference semantics. The main limiting factor now is the training augmentation: you are training on only resized images (no augmentation), which tends to overfit and generalize poorly for Cassava; adding a minimal set of standard augmentations (random resized crop + horizontal flip) while keeping the same normalization, loss, optimizer, epochs, and TTA inference usually provides a sizable boost. I also make the training loader deterministic (so runs are stable) and add very light label smoothing in the same CrossEntropyLoss family to improve calibration/generalization without changing the training approach. All paths, the EfficientNetV2-S backbone + 5-class head, 2-epoch training loop, and Softmax+argmax + TTA averaging inference remain the same, and the script still writes a valid `submission.csv`.'
- What this solution (achieved 0.79596) has done: 'Your current gap to target is large (0.75374 vs 0.89045; higher is better), so we should make the smallest changes that legitimately improve generalization without changing your core model/inference semantics. The biggest low-risk win here is to fix the train/test preprocessing mismatch: you train with RandomResizedCrop but test with a plain Resize, which often costs accuracy; switching test preprocessing to a standard Resize+CenterCrop (still deterministic, still same normalization/size) typically improves ImageNet-backbone transfer. Next, we enable the already-imported cuDNN benchmarking (since determinism is not required by Kaggle scoring) to slightly improve numeric/throughput stability in the time budget, and we add a very light gradient clipping to prevent rare unstable steps without changing the training loop structure. Everything else (EfficientNetV2-S backbone, 5-class head, 2-epoch training, CE loss with label smoothing, Softmax+argmax, same TTA averaging, and exact submission alignment to sample_submission.csv) stays the same.'
- What this solution (achieved 0.77653) has done: 'Your current score (0.79596) is below the target (0.89045), so we should make small, low-risk changes that improve generalization without changing the core model or training loop. The biggest gap is that you train on the full training set with no validation split, which makes it hard to control overfitting; we add a small stratified validation split and pick the best of the 2 epochs (same epochs, same optimizer/loss, same architecture) based on validation accuracy. We also switch training-time sampling to a class-balanced sampler (keeps the same data and loss, just changes batch composition) which is commonly beneficial for Cassava’s class imbalance and tends to move accuracy upward. Finally, we keep your submission alignment logic untouched and ensure the script still runs end-to-end and writes a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import random

import numpy as np
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
np.random.seed(3407)
random.seed(3407)
if torch.cuda.is_available():
    torch.cuda.manual_seed(3407)

cudnn.deterministic = False
cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

test_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images/"
train_dir = "/kaggle/input/cassava-leaf-disease-classification/train_images/"
train_csv_path = "/kaggle/input/cassava-leaf-disease-classification/train.csv"
sample_sub_path = (
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)

img_size = 384  # EfficientNetV2-S default.
batch_size = 32
num_workers = 4
num_classes = 5
tta = True

weights = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
model = models.efficientnet_v2_s(weights=weights)

in_features = model.classifier[1].in_features
model.classifier[1] = torch.nn.Linear(in_features, num_classes)

model.to(device)




## === cell 2
class CassavaDataset(VisionDataset):
    """Custom dataset for Cassava data.
    - If ttas is provided: returns a list of T transformed tensors, one per TTA transform.
    - Else: returns a single transformed tensor.
    """

    def __init__(self, data_dir, transform=None, ttas=None):
        super().__init__(root=data_dir)
        self.transform = transform
        if os.path.exists(sample_sub_path) and "test_images" in data_dir:
            self.images = pd.read_csv(sample_sub_path)["image_id"].tolist()
            self.labels = None
        else:
            self.images = sorted(os.listdir(data_dir))
            self.labels = None
        self.ttas = ttas

    def __getitem__(self, idx):
        filename = self.images[idx]
        img = Image.open(os.path.join(self.root, filename)).convert("RGB")

        if self.ttas is not None and self.transform is not None:
            img_list = [self.transform(t(img)) for t in self.ttas]
            return img_list, filename
        elif self.transform is not None:
            return self.transform(img), filename
        else:
            return img, filename

    def __len__(self):
        return len(self.images)


class CassavaTrainDataset(VisionDataset):
    """Minimal train dataset reading train.csv; returns (tensor, label)."""

    def __init__(self, data_dir, csv_path, transform=None):
        super().__init__(root=data_dir)
        self.df = pd.read_csv(csv_path)
        self.transform = transform

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = row["image_id"]
        y = int(row["label"])
        img = Image.open(os.path.join(self.root, filename)).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, y

    def __len__(self):
        return len(self.df)




## === cell 3
weights_tfm = weights.transforms()

train_transforms = v2.Compose(
    [
        v2.RandomResizedCrop(
            size=(img_size, img_size),
            scale=(0.75, 1.0),
            ratio=(0.9, 1.1),
            interpolation=InterpolationMode.BICUBIC,
        ),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=weights_tfm.mean, std=weights_tfm.std),
    ]
)

base_transforms = v2.Compose(
    [
        v2.Resize(int(img_size * 1.15), interpolation=InterpolationMode.BICUBIC),
        v2.CenterCrop((img_size, img_size)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=weights_tfm.mean, std=weights_tfm.std),
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

full_df = pd.read_csv(train_csv_path)
rng = np.random.RandomState(3407)

val_frac = 0.10
train_idx = []
val_idx = []
for c in sorted(full_df["label"].unique()):
    idxs = np.where(full_df["label"].values == c)[0]
    rng.shuffle(idxs)
    n_val = max(1, int(len(idxs) * val_frac))
    val_idx.extend(idxs[:n_val].tolist())
    train_idx.extend(idxs[n_val:].tolist())

train_idx = np.array(train_idx, dtype=np.int64)
val_idx = np.array(val_idx, dtype=np.int64)

train_dataset = CassavaTrainDataset(
    train_dir, train_csv_path, transform=train_transforms
)
val_dataset = CassavaTrainDataset(
    train_dir,
    train_csv_path,
    transform=base_transforms,  # deterministic eval preprocessing
)

test_dataset = CassavaDataset(test_dir, transform=base_transforms, ttas=ttas)


def _seed_worker(worker_id: int):
    worker_seed = (3407 + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


g = torch.Generator()
g.manual_seed(3407)


def tta_collate_fn(batch):
    inputs, filenames = zip(*batch)
    return list(inputs), list(filenames)


train_labels = full_df["label"].values[train_idx]
class_counts = np.bincount(train_labels, minlength=num_classes).astype(np.float64)
class_counts[class_counts == 0] = 1.0
class_weights = 1.0 / class_counts
sample_weights = class_weights[train_labels]
sample_weights_t = torch.as_tensor(sample_weights, dtype=torch.double)

sampler = torch.utils.data.WeightedRandomSampler(
    weights=sample_weights_t,
    num_samples=len(sample_weights_t),
    replacement=True,
    generator=g,
)

train_loader = DataLoader(
    torch.utils.data.Subset(train_dataset, train_idx.tolist()),
    batch_size=batch_size,
    shuffle=False,
    sampler=sampler,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    worker_init_fn=_seed_worker,
    generator=g,
)

val_loader = DataLoader(
    torch.utils.data.Subset(val_dataset, val_idx.tolist()),
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    worker_init_fn=_seed_worker,
    generator=g,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    worker_init_fn=_seed_worker,
    generator=g,
    collate_fn=tta_collate_fn if tta else None,
)

normalizer = torch.nn.Softmax(dim=1)



## === cell 4
criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.05)

for p in model.parameters():
    p.requires_grad = False

for p in model.classifier.parameters():
    p.requires_grad = True

for p in model.features[-1].parameters():
    p.requires_grad = True

optimizer = torch.optim.AdamW(
    [
        {"params": model.classifier.parameters(), "lr": 3e-3, "weight_decay": 1e-4},
        {"params": model.features[-1].parameters(), "lr": 3e-4, "weight_decay": 1e-4},
    ]
)

best_state = None
best_val_acc = -1.0

epochs = 2  # keep identical training length for minimal change and time safety

for ep in range(epochs):
    model.train()
    running = 0.0
    seen = 0
    for xb, yb in train_loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        running += loss.item() * xb.size(0)
        seen += xb.size(0)

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = model(xb)
            pred = torch.argmax(logits, dim=1)
            correct += (pred == yb).sum().item()
            total += yb.numel()
    val_acc = correct / max(1, total)

    print(
        f"epoch {ep+1}/{epochs} - train_loss: {running/seen:.4f} - val_acc: {val_acc:.4f}"
    )

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_state = {
            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
        }

if best_state is not None:
    model.load_state_dict(best_state)



## === cell 5
all_names = []
all_preds = []

model.eval()

with torch.no_grad():
    for _, (inputs, filenames) in enumerate(test_loader):
        if tta:
            T = len(ttas)
            per_sample = [torch.stack(list(x), dim=0) for x in inputs]  # [T,C,H,W]
            batch = torch.stack(per_sample, dim=0)  # [B,T,C,H,W]
            B = batch.shape[0]
            flat = batch.view(B * T, *batch.shape[2:]).to(device, non_blocking=True)

            preds = normalizer(model(flat))  # [B*T, 5]
            preds = preds.view(B, T, -1)  # [B,T,5]
            mean_preds = preds.mean(dim=1)  # [B,5]
            pred_labels = torch.argmax(mean_preds, dim=1).tolist()
        else:
            inputs = inputs.to(device, non_blocking=True)
            preds = normalizer(model(inputs))
            pred_labels = torch.argmax(preds, dim=1).tolist()

        pred_labels = [int(p) for p in pred_labels]

        all_names.extend(list(filenames))
        all_preds.extend(pred_labels)



## === cell 6
sample_sub = pd.read_csv(sample_sub_path)
pred_df = pd.DataFrame({"image_id": all_names, "label": all_preds})

pred_df = pred_df.drop_duplicates(subset=["image_id"], keep="first")
pred_df = sample_sub[["image_id"]].merge(pred_df, on="image_id", how="left")
pred_df["label"] = pred_df["label"].fillna(0).astype(int)

pred_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", pred_df.shape)
print(pred_df.head())



## === cell 7
pred_df
