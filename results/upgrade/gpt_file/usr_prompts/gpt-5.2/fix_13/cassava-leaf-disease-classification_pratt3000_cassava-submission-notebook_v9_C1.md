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
import math
import numpy as np
import pandas as pd

from PIL import Image, ImageFile
from io import BytesIO

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, IterableDataset, get_worker_info

import torchvision.models as models
import torchvision.transforms as T

from tqdm.auto import tqdm  # faster/safer than tqdm.notebook in script-like runs

try:
    from torchvision.io import read_image, ImageReadMode, decode_jpeg  # type: ignore

    _HAS_TVIO = True
except Exception:
    read_image = None
    decode_jpeg = None
    ImageReadMode = None
    _HAS_TVIO = False

try:
    import tensorflow as tf  # type: ignore

    _HAS_TF = True
except Exception:
    tf = None
    _HAS_TF = False

try:
    import cv2  # noqa: F401
except Exception:
    cv2 = None

INPUT_ROOT = "/kaggle/input/cassava-leaf-disease-classification"
WORKING_ROOT = "/kaggle/working"

print("INPUT_ROOT exists:", os.path.exists(INPUT_ROOT))
print("WORKING_ROOT exists:", os.path.exists(WORKING_ROOT))




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

try:
    ImageFile.LOAD_TRUNCATED_IMAGES = True
except Exception:
    pass


def get_image(path):
    if _HAS_TVIO:
        return read_image(path, mode=ImageReadMode.RGB)  # uint8, [C,H,W]
    with Image.open(path) as img:
        return img.convert("RGB")


def _bytes_to_pil_rgb(jpeg_bytes: bytes) -> Image.Image:
    if _HAS_TVIO and decode_jpeg is not None:
        t = torch.frombuffer(memoryview(jpeg_bytes), dtype=torch.uint8)
        img = decode_jpeg(t, mode=ImageReadMode.RGB)
        return T.ToPILImage()(img)
    return Image.open(BytesIO(jpeg_bytes)).convert("RGB")




## === cell 2
class GetDataset(Dataset):
    def __init__(self, df, data_root, transforms=None, output_label=True):
        super().__init__()
        df = df.reset_index(drop=True)
        self.data_root = data_root
        self.transforms = transforms
        self.output_label = output_label

        image_ids = df["image_id"].astype(str).to_list()
        self.paths = [os.path.join(data_root, img_id) for img_id in image_ids]

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


class CassavaTFRecordIterable(IterableDataset):
    def __init__(
        self,
        tfrec_paths,
        transforms=None,
        output_label=True,
        with_image_id=False,
        seed=42,
    ):
        super().__init__()
        self.tfrec_paths = list(tfrec_paths)
        self.transforms = transforms
        self.output_label = output_label
        self.with_image_id = with_image_id
        self.seed = int(seed)

        if not _HAS_TF:
            raise RuntimeError(
                "TensorFlow is required to read TFRecords but is not available."
            )

        self._feature_spec = {
            "image": tf.io.FixedLenFeature([], tf.string),
            "image_name": tf.io.FixedLenFeature([], tf.string),
        }
        if output_label:
            self._feature_spec["target"] = tf.io.FixedLenFeature([], tf.int64)

    def __len__(self):
        raise TypeError("CassavaTFRecordIterable does not support __len__")

    def _iter_files_for_worker(self):
        wi = get_worker_info()
        if wi is None:
            return self.tfrec_paths
        return self.tfrec_paths[wi.id :: wi.num_workers]

    def __iter__(self):
        files = self._iter_files_for_worker()
        for path in files:
            ds = tf.data.TFRecordDataset(path, num_parallel_reads=1)
            for raw in ds:
                ex = tf.io.parse_single_example(raw, self._feature_spec)
                img_bytes = ex["image"].numpy()
                pil_img = _bytes_to_pil_rgb(img_bytes)
                if self.transforms:
                    x = self.transforms(pil_img)
                else:
                    x = pil_img
                if self.output_label:
                    y = int(ex["target"].numpy())
                    if self.with_image_id:
                        image_id = ex["image_name"].numpy().decode("utf-8")
                        yield x, y, image_id
                    else:
                        yield x, y
                else:
                    if self.with_image_id:
                        image_id = ex["image_name"].numpy().decode("utf-8")
                        yield x, image_id
                    else:
                        yield x




## === cell 3
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    if torch.is_tensor(data):
        if device.type == "cuda" and data.dim() == 4:
            data = data.contiguous(memory_format=torch.channels_last)
        return data.to(device, non_blocking=True)
    return data


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
    torch.backends.cudnn.benchmark = False


seed_everything(42)

try:
    torch.set_num_threads(min(4, os.cpu_count() or 2))
except Exception:
    pass


def _find_checkpoint_path(preferred: str) -> str:
    if os.path.exists(preferred):
        return preferred
    base = "/kaggle/input"
    target_name = os.path.basename(preferred)
    for root, _, files in os.walk(base):
        if target_name in files:
            return os.path.join(root, target_name)
    return preferred


ckpt_path = "/kaggle/input/cassava-leaf-disease-detection-resnext101-32x8d/mod.pth"
ckpt_path = _find_checkpoint_path(ckpt_path)


def _load_checkpoint_model(path: str, device: torch.device) -> nn.Module:
    obj = torch.load(path, map_location=device)

    if isinstance(obj, nn.Module):
        return obj

    state = None
    if isinstance(obj, dict):
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            state = obj["state_dict"]
        elif "model" in obj and isinstance(obj["model"], dict):
            state = obj["model"]
        else:
            tensor_vals = [v for v in obj.values() if torch.is_tensor(v)]
            if len(tensor_vals) >= max(1, len(obj) // 2):
                state = obj

    if state is None:
        raise RuntimeError("Unrecognized checkpoint format at: " + path)

    new_state = {}
    for k, v in state.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module.") :]
        if nk.startswith("network."):
            nk = nk[len("network.") :]
        new_state[nk] = v

    m = Classifier().to(device)
    try:
        m.load_state_dict(
            {("network." + k): v for k, v in new_state.items()}, strict=False
        )
        missing, unexpected = m.load_state_dict(
            {("network." + k): v for k, v in new_state.items()}, strict=False
        )
    except Exception:
        missing, unexpected = m.load_state_dict(new_state, strict=False)

    return m


if os.path.exists(ckpt_path):
    model = _load_checkpoint_model(ckpt_path, device)
    print("Loaded checkpoint model from:", ckpt_path)
    model = model.to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    NEED_TRAIN = False
else:
    raise FileNotFoundError(
        f"Checkpoint not found at: {ckpt_path}. "
        "This solution is designed to run inference from the provided checkpoint; "
        "training fallback exceeds the 600-second timeout."
    )




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/883981073.py in <cell line: 0>()
     92     # TIMEOUT FIX: do not silently fall back to long training which can't fit the 600s budget.
     93     # This preserves the intended solution semantics (checkpoint-based inference).
---> 94     raise FileNotFoundError(
     95         f"Checkpoint not found at: {ckpt_path}. "
     96         "This solution is designed to run inference from the provided checkpoint; "

FileNotFoundError: Checkpoint not found at: /kaggle/input/cassava-leaf-disease-detection-resnext101-32x8d/mod.pth. This solution is designed to run inference from the provided checkpoint; training fallback exceeds the 600-second timeout.

## === cell 7
@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    loss_sum = 0.0
    acc_sum = 0.0
    n_batches = 0
    for batch in val_loader:
        out = model.validation_step(batch)
        loss_sum += float(out["val_loss"])
        acc_sum += float(out["val_acc"])
        n_batches += 1
    return {
        "val_loss": loss_sum / max(1, n_batches),
        "val_acc": acc_sum / max(1, n_batches),
    }


def fit(epochs, lr, model, train_loader, val_loader):
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
    history = []
    for epoch in range(epochs):
        model.train()

        train_loss_sum = 0.0
        train_batches = 0
        lrs = []

        for batch in train_loader:
            loss = model.training_step(batch)
            train_loss_sum += float(loss.detach())
            train_batches += 1
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            lrs.append(lr)

        result = evaluate(model, val_loader)
        result["train_loss"] = train_loss_sum / max(1, train_batches)
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
        T.Resize(IMG_SHAPE, antialias=True),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.2),
        T.ConvertImageDtype(torch.float32),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

val_transforms = T.Compose(
    [
        T.Resize(IMG_SHAPE, antialias=True),
        T.ConvertImageDtype(torch.float32),
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
for lbl, gdf in train_df.groupby("label"):
    idx = gdf.index.values.copy()
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

use_cuda = torch.cuda.is_available()
cpu_count = os.cpu_count() or 2

num_workers = min(4, max(2, cpu_count - 1))


def seed_worker(worker_id):
    worker_seed = 42 + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


g = torch.Generator()
g.manual_seed(42)

TRAIN_TFREC_DIR = f"{INPUT_ROOT}/train_tfrecords"
TEST_TFREC_DIR = f"{INPUT_ROOT}/test_tfrecords"
train_tfrec_paths = []
test_tfrec_paths = []
if os.path.isdir(TRAIN_TFREC_DIR):
    train_tfrec_paths = sorted(
        [
            os.path.join(TRAIN_TFREC_DIR, f)
            for f in os.listdir(TRAIN_TFREC_DIR)
            if f.endswith(".tfrec")
        ]
    )
if os.path.isdir(TEST_TFREC_DIR):
    test_tfrec_paths = sorted(
        [
            os.path.join(TEST_TFREC_DIR, f)
            for f in os.listdir(TEST_TFREC_DIR)
            if f.endswith(".tfrec")
        ]
    )

USE_TFRECORDS = _HAS_TF and (len(train_tfrec_paths) > 0) and (len(test_tfrec_paths) > 0)
print(
    "USE_TFRECORDS:",
    USE_TFRECORDS,
    "| tf:",
    _HAS_TF,
    "| n_train_tfrec:",
    len(train_tfrec_paths),
    "| n_test_tfrec:",
    len(test_tfrec_paths),
)

train_ds = GetDataset(trn_df, TRAIN_DIR, transforms=train_transforms, output_label=True)
val_ds = GetDataset(val_df, TRAIN_DIR, transforms=val_transforms, output_label=True)

prefetch = 4 if num_workers > 0 else None

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    num_workers=num_workers,
    shuffle=True,
    pin_memory=use_cuda,
    persistent_workers=(num_workers > 0),
    prefetch_factor=prefetch,
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
    prefetch_factor=prefetch,
    worker_init_fn=seed_worker,
)

train_loader = DeviceDataLoader(train_loader, device)
val_loader = DeviceDataLoader(val_loader, device)




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
/tmp/ipykernel_55/805274031.py in <cell line: 0>()
----> 1 if NEED_TRAIN:
      2     model.freeze()
      3 
      4     EPOCHS = 10
      5     LR = 5e-4

NameError: name 'NEED_TRAIN' is not defined

## === cell 11
sample_path = f"{INPUT_ROOT}/sample_submission.csv"
test_csv = pd.read_csv(sample_path)

FORCE_IMAGE_FOLDER_INFERENCE = True

if USE_TFRECORDS and not FORCE_IMAGE_FOLDER_INFERENCE:
    test_ds = CassavaTFRecordIterable(
        test_tfrec_paths,
        transforms=test_transforms,
        output_label=False,
        with_image_id=True,
        seed=42,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
        prefetch_factor=prefetch,
        worker_init_fn=seed_worker,
    )
    USING_TFREC_FOR_TEST = True
else:
    test_ds = GetDataset(
        test_csv, TEST_DIR, transforms=test_transforms, output_label=False
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
        prefetch_factor=prefetch,
        worker_init_fn=seed_worker,
    )
    USING_TFREC_FOR_TEST = False

test_loader = DeviceDataLoader(test_loader, device)

len(test_csv), test_csv.head()




## === cell 12
def inference(model, test_loader, device):
    model.to(device)
    model.eval()

    with torch.inference_mode():
        if USING_TFREC_FOR_TEST:
            probs_by_id = {}
            it = tqdm(test_loader, total=None, leave=False)
            for images, image_ids in it:
                logits = model(images)
                prob = torch.softmax(logits, dim=1).cpu().numpy()
                for img_id, p in zip(image_ids, prob):
                    probs_by_id[str(img_id)] = p
            print("predictions collected:", len(probs_by_id))
            return probs_by_id
        else:
            n = (
                len(test_loader.dl.dataset)
                if isinstance(test_loader, DeviceDataLoader)
                else len(test_loader.dataset)
            )
            out_probs = np.empty((n, 5), dtype=np.float32)
            idx = 0
            it = tqdm(test_loader, total=len(test_loader), leave=False)
            for images in it:
                logits = model(images)
                prob = torch.softmax(logits, dim=1).cpu().numpy()
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
/tmp/ipykernel_55/1718609626.py in <cell line: 0>()
     35 
     36 
---> 37 predictions = inference(model, test_loader, device)
     38 
     39 

NameError: name 'model' is not defined

## === cell 13
if USING_TFREC_FOR_TEST:
    ordered = np.stack(
        [predictions[img_id] for img_id in test_csv["image_id"].astype(str).tolist()],
        axis=0,
    )
    test_csv["label"] = ordered.argmax(1).astype(int)
else:
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
/tmp/ipykernel_55/3288417140.py in <cell line: 0>()
      6     test_csv["label"] = ordered.argmax(1).astype(int)
      7 else:
----> 8     test_csv["label"] = predictions.argmax(1).astype(int)
      9 
     10 sub_path = os.path.join(WORKING_ROOT, "submission.csv")

NameError: name 'predictions' is not defined
