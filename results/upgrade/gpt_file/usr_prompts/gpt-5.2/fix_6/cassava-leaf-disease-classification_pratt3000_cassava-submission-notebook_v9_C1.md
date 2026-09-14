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

No external packages required in the script and installed.

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

0.8874282260501662

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.12332) has done: 'I fix the early import crash by removing the `/kaggle/input` recursive walk/print (it triggers the protobuf `MessageFactory.GetPrototype` error in this environment) and keeping only the necessary imports. Then I fix the cascading `NameError`s by ensuring `torch/nn/Dataset/torchvision.transforms as T` are imported before use and by renumbering cells to start at 1 while preserving the same modeling/inference logic. I also make the model loading robust to the missing external checkpoint by falling back to the defined ResNeXt50 classifier if the `.pth` file is not present, so the notebook always runs end-to-end. Finally, I fix the inference aggregation/shape logic and write a valid `submission.csv` with exactly `image_id,label`.'
- What this solution (achieved 0.73954) has done: 'Your current score is far below the target because the fallback model is an ImageNet-pretrained ResNeXt50 with a randomly initialized 5-class head, so its predictions are essentially untrained noise. To move accuracy upward toward the target while preserving your core architecture and inference semantics, I add a minimal fine-tuning step using `train.csv` with a stratified train/validation split and the same normalization transforms. I keep the same model class (ResNeXt50_32x4d + linear 5-way head) and use standard cross-entropy training without changing the inference pipeline. This should raise the score substantially toward the 0.887 band, while staying within Kaggle runtime by training only the classifier head for a small number of epochs.'
- What this solution (achieved 0.75673) has done: 'Your current model is only training the final classification head for 2 epochs with a relatively high LR, which typically underfits this dataset and lands far below your target accuracy. To move the score upward toward the 0.887 band while keeping the exact same architecture/training loop/loss/inference semantics, I (1) train the head a bit longer with a slightly smaller LR, and (2) use a class-weighted cross-entropy (same loss function family) to better handle label imbalance, which usually boosts accuracy with minimal risk. I also enable a couple of safe DataLoader throughput flags (persistent workers/prefetch) to keep runtime within limits without changing learning behavior. No changes are made to the model structure or prediction formatting, and the script still writes a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import json
import random
import numpy as np
import pandas as pd

from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision.models as models
import torchvision.transforms as T

from tqdm.auto import tqdm  # faster/safer than tqdm.notebook in script-like runs

try:
    import cv2  # noqa: F401
except Exception:
    cv2 = None

INPUT_ROOT = "/kaggle/input/cassava-leaf-disease-classification"
WORKING_ROOT = "/kaggle/working"

print("INPUT_ROOT exists:", os.path.exists(INPUT_ROOT))
print("WORKING_ROOT exists:", os.path.exists(WORKING_ROOT))



## === cell 1
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


def get_image(path):
    with Image.open(path) as img:
        return img.convert("RGB")




## === cell 2
class GetDataset(Dataset):
    def __init__(self, df, data_root, transforms=None, output_label=True):
        super().__init__()
        df = df.reset_index(drop=True)
        self.data_root = data_root
        self.transforms = transforms
        self.output_label = output_label

        image_ids = df["image_id"].to_numpy()
        self.paths = np.char.add(data_root + "/", image_ids).tolist()

        if output_label:
            self.labels = df["label"].to_numpy(dtype=np.int64)
        else:
            self.labels = None

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index: int):
        img = get_image(self.paths[index])

        if self.transforms:
            img = self.transforms(img)

        if self.output_label:
            return img, int(self.labels[index])
        else:
            return img




## === cell 3
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader:
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        for x in self.dl:
            yield to_device(x, self.device)

    def __len__(self):
        return len(self.dl)


device = get_device()
device




## === cell 4
def accuracy(out, labels):
    _, preds = torch.max(out, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


LOSS_WEIGHTS = None


class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels, weight=LOSS_WEIGHTS)
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels, weight=LOSS_WEIGHTS)
        acc = accuracy(out, labels)
        return {"val_loss": loss.detach(), "val_acc": acc}

    def validation_epoch_end(self, outputs):
        batch_loss = [x["val_loss"] for x in outputs]
        epoch_loss = torch.stack(batch_loss).mean()
        batch_acc = [x["val_acc"] for x in outputs]
        epoch_acc = torch.stack(batch_acc).mean()
        return {"val_loss": epoch_loss.item(), "val_acc": epoch_acc.item()}

    def epoch_end(self, epoch, epochs, result):
        print(
            "Epoch: [{}/{}], last_lr: {:.6f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
                epoch,
                epochs,
                result["lrs"][-1],
                result["train_loss"],
                result["val_loss"],
                result["val_acc"],
            )
        )




## === cell 5
class Classifier(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        self.network = models.resnext50_32x4d(pretrained=True)
        number_of_features = self.network.fc.in_features
        self.network.fc = nn.Linear(number_of_features, 5)

    def forward(self, xb):
        return self.network(xb)

    def freeze(self):
        for param in self.network.parameters():
            param.requires_grad = False
        for param in self.network.fc.parameters():
            param.requires_grad = True

    def unfreeze(self):
        for param in self.network.parameters():
            param.requires_grad = True




## === cell 6
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = True


seed_everything(42)

ckpt_path = "/kaggle/input/cassava-leaf-disease-detection-resnext101-32x8d/mod.pth"

if os.path.exists(ckpt_path):
    model = torch.load(ckpt_path, map_location=device)
    print("Loaded checkpoint model from:", ckpt_path)
    model = model.to(device)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    NEED_TRAIN = False
else:
    model = Classifier()
    print("Checkpoint not found; will fine-tune ResNeXt50_32x4d head on train.csv.")
    model = model.to(device)
    NEED_TRAIN = True




## === cell 7
@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = []
    for batch in val_loader:
        out = model.validation_step(batch)
        outputs.append(out)
    return model.validation_epoch_end(outputs)


def fit(epochs, lr, model, train_loader, val_loader):
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
    history = []
    for epoch in range(epochs):
        model.train()
        train_losses = []
        lrs = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss.detach())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(
                set_to_none=True
            )  # speed: less memory ops; same semantics
            lrs.append(lr)

        result = evaluate(model, val_loader)
        result["train_loss"] = torch.stack(train_losses).mean().item()
        result["lrs"] = lrs
        model.epoch_end(epoch + 1, epochs, result)
        history.append(result)
    return history




## === cell 8
BATCH_SIZE = 128
IMG_SIZE = 512
IMG_SHAPE = (IMG_SIZE, IMG_SIZE)

TRAIN_DIR = f"{INPUT_ROOT}/train_images"
TEST_DIR = f"{INPUT_ROOT}/test_images"

train_transforms = T.Compose(
    [
        T.Resize(IMG_SHAPE),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.2),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

val_transforms = T.Compose(
    [
        T.Resize(IMG_SHAPE),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

test_transforms = val_transforms



## === cell 9
train_csv_path = f"{INPUT_ROOT}/train.csv"
train_df = pd.read_csv(train_csv_path)

val_frac = 0.1
rng = np.random.RandomState(42)
val_indices = []
for lbl, g in train_df.groupby("label"):
    idx = g.index.values.copy()
    rng.shuffle(idx)
    n_val = max(1, int(len(idx) * val_frac))
    val_indices.extend(idx[:n_val].tolist())

val_df = train_df.loc[val_indices].reset_index(drop=True)
trn_df = train_df.drop(index=val_indices).reset_index(drop=True)

print("Train size:", len(trn_df), "Val size:", len(val_df))
print("Train label counts:", trn_df["label"].value_counts().sort_index().to_dict())
print("Val label counts:", val_df["label"].value_counts().sort_index().to_dict())

label_counts = trn_df["label"].value_counts().sort_index()
counts = label_counts.values.astype(np.float32)
weights = counts.sum() / (len(counts) * counts)
weights = weights / weights.mean()
LOSS_WEIGHTS = torch.tensor(weights, dtype=torch.float32, device=device)
print("Using class weights:", {i: float(w) for i, w in enumerate(weights)})

train_ds = GetDataset(trn_df, TRAIN_DIR, transforms=train_transforms, output_label=True)
val_ds = GetDataset(val_df, TRAIN_DIR, transforms=val_transforms, output_label=True)

use_cuda = torch.cuda.is_available()
cpu_count = os.cpu_count() or 2
num_workers = min(8, max(2, cpu_count // 2))


def seed_worker(worker_id):
    worker_seed = 42 + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


g = torch.Generator()
g.manual_seed(42)

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    num_workers=num_workers,
    shuffle=True,
    pin_memory=use_cuda,
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else None,
    worker_init_fn=seed_worker,
    generator=g,
)
val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    num_workers=num_workers,
    shuffle=False,
    pin_memory=use_cuda,
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else None,
    worker_init_fn=seed_worker,
)

train_loader = DeviceDataLoader(train_loader, device)
val_loader = DeviceDataLoader(val_loader, device)



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3878760686.py in <cell line: 0>()
     25 print("Using class weights:", {i: float(w) for i, w in enumerate(weights)})
     26 
---> 27 train_ds = GetDataset(trn_df, TRAIN_DIR, transforms=train_transforms, output_label=True)
     28 val_ds = GetDataset(val_df, TRAIN_DIR, transforms=val_transforms, output_label=True)
     29 

/tmp/ipykernel_55/3454705498.py in __init__(self, df, data_root, transforms, output_label)
     10 
     11         image_ids = df["image_id"].to_numpy()
---> 12         self.paths = np.char.add(data_root + "/", image_ids).tolist()
     13 
     14         if output_label:

/usr/local/lib/python3.11/dist-packages/numpy/core/defchararray.py in add(x1, x2)
    330         # object dtype itemsize as num chars (worked on short strings).
    331         # bytes + void worked but promoting void->bytes is dubious also.
--> 332         raise TypeError(
    333             "np.char.add() requires both arrays of the same dtype kind, but "
    334             f"got dtypes: '{arr1.dtype}' and '{arr2.dtype}' (the few cases "

TypeError: np.char.add() requires both arrays of the same dtype kind, but got dtypes: '<U63' and 'object' (the few cases where this used to work often lead to incorrect results).

## === cell 10
if NEED_TRAIN:
    model.freeze()

    EPOCHS = 10
    LR = 5e-4
    _ = fit(EPOCHS, LR, model, train_loader, val_loader)

    for p in model.parameters():
        p.requires_grad = False
    model.eval()



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4108149750.py in <cell line: 0>()
      4     EPOCHS = 10
      5     LR = 5e-4
----> 6     _ = fit(EPOCHS, LR, model, train_loader, val_loader)
      7 
      8     for p in model.parameters():

NameError: name 'train_loader' is not defined

## === cell 11
sample_path = f"{INPUT_ROOT}/sample_submission.csv"
test_csv = pd.read_csv(sample_path)

test_ds = GetDataset(test_csv, TEST_DIR, transforms=test_transforms, output_label=False)

test_loader = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    num_workers=num_workers,
    shuffle=False,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else None,
    worker_init_fn=seed_worker,
)
test_loader = DeviceDataLoader(test_loader, device)

len(test_ds), test_csv.head()




## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_55/60493518.py in <cell line: 0>()
      2 test_csv = pd.read_csv(sample_path)
      3 
----> 4 test_ds = GetDataset(test_csv, TEST_DIR, transforms=test_transforms, output_label=False)
      5 
      6 # Speed: reuse worker count and stronger prefetching; inference ordering unchanged (shuffle=False).

/tmp/ipykernel_55/3454705498.py in __init__(self, df, data_root, transforms, output_label)
     10 
     11         image_ids = df["image_id"].to_numpy()
---> 12         self.paths = np.char.add(data_root + "/", image_ids).tolist()
     13 
     14         if output_label:

/usr/local/lib/python3.11/dist-packages/numpy/core/defchararray.py in add(x1, x2)
    330         # object dtype itemsize as num chars (worked on short strings).
    331         # bytes + void worked but promoting void->bytes is dubious also.
--> 332         raise TypeError(
    333             "np.char.add() requires both arrays of the same dtype kind, but "
    334             f"got dtypes: '{arr1.dtype}' and '{arr2.dtype}' (the few cases "

TypeError: np.char.add() requires both arrays of the same dtype kind, but got dtypes: '<U62' and 'object' (the few cases where this used to work often lead to incorrect results).

## === cell 12
def inference(model, test_loader, device):
    model.to(device)
    model.eval()

    n = (
        len(test_loader.dl.dataset)
        if isinstance(test_loader, DeviceDataLoader)
        else len(test_loader.dataset)
    )
    out_probs = np.empty((n, 5), dtype=np.float32)

    bs = (
        test_loader.dl.batch_size
        if isinstance(test_loader, DeviceDataLoader)
        else getattr(test_loader, "batch_size", None)
    )
    if bs is None:
        bs = BATCH_SIZE

    idx = 0
    it = tqdm(test_loader, total=len(test_loader), leave=False)
    with torch.inference_mode():
        for images in it:
            logits = model(images)
            prob = logits.softmax(1).detach().cpu().numpy()
            b = prob.shape[0]
            out_probs[idx : idx + b] = prob
            idx += b

    if idx != n:
        out_probs = out_probs[:idx]
    print("predictions shape:", out_probs.shape)
    return out_probs


predictions = inference(model, test_loader, device)



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2115975122.py in <cell line: 0>()
     35 
     36 
---> 37 predictions = inference(model, test_loader, device)
     38 

NameError: name 'test_loader' is not defined

## === cell 13
test_csv["label"] = predictions.argmax(1).astype(int)
sub_path = os.path.join(WORKING_ROOT, "submission.csv")
test_csv[["image_id", "label"]].to_csv(sub_path, index=False)

print("Saved:", sub_path)
print(test_csv.head())
print(
    "Rows:",
    len(test_csv),
    "Unique labels:",
    sorted(test_csv["label"].unique().tolist()),
)

## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4060652835.py in <cell line: 0>()
----> 1 test_csv["label"] = predictions.argmax(1).astype(int)
      2 sub_path = os.path.join(WORKING_ROOT, "submission.csv")
      3 test_csv[["image_id", "label"]].to_csv(sub_path, index=False)
      4 
      5 print("Saved:", sub_path)

NameError: name 'predictions' is not defined
