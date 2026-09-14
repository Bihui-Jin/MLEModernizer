# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sklearn-pandas==2.2.0
timm==1.0.19
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124
tqdm==4.67.1

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

# 5. Code solution

## === cell 0
import os
import sys
import time
import copy
import random
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset

from torch.cuda import amp

from sklearn.model_selection import StratifiedKFold

import albumentations as A
from albumentations.pytorch import ToTensorV2

import timm



## === cell 1
ROOT_DIR = "../input/cassava-leaf-disease-classification"
TRAIN_DIR = "../input/cassava-leaf-disease-classification/train_images"
TEST_DIR = "../input/cassava-leaf-disease-classification/test_images"

if not os.path.exists(ROOT_DIR):
    ROOT_DIR = "/kaggle/input/cassava-leaf-disease-classification"
    TRAIN_DIR = f"{ROOT_DIR}/train_images"
    TEST_DIR = f"{ROOT_DIR}/test_images"

assert os.path.exists(f"{ROOT_DIR}/train.csv"), f"train.csv not found under {ROOT_DIR}"




## === cell 2
class CFG:
    model_name = "tf_efficientnet_b4_ns"
    img_size = 512
    scheduler = "CosineAnnealingWarmRestarts"
    T_max = 10
    T_0 = 10
    lr = 1e-4
    min_lr = 1e-6
    batch_size = 16
    weight_decay = 1e-6
    seed = 42
    num_classes = 5
    num_epochs = 10
    n_fold = 5
    smoothing = 0.2
    t1 = 0.8
    t2 = 1.4
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    savemodel = True
    savemodelpath = "/kaggle/working/bitemp-21.pth"
    dosubmission = True
    crop_size = 256
    crop_p = 0.9




## === cell 3
def set_seed(seed=42):
    """Reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


set_seed(CFG.seed)

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True



## === cell 4
df = pd.read_csv(f"{ROOT_DIR}/train.csv")
assert {"image_id", "label"}.issubset(df.columns)

skf = StratifiedKFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed)
for fold, (_, val_) in enumerate(skf.split(X=df, y=df.label)):
    df.loc[val_, "kfold"] = int(fold)
df["kfold"] = df["kfold"].astype(int)




## === cell 5
class CassavaLeafDataset(Dataset):
    def __init__(self, root_dir, df, transforms=None, has_labels=True):
        self.root_dir = root_dir
        self.df = df.reset_index(drop=True)
        self.transforms = transforms
        self.has_labels = has_labels and ("label" in self.df.columns)

        if "image_id" in self.df.columns:
            image_ids = self.df["image_id"].astype(str).values
        else:
            image_ids = self.df.iloc[:, 0].astype(str).values

        self.img_paths = [os.path.join(self.root_dir, x) for x in image_ids]

        if self.has_labels:
            self.labels = self.df["label"].astype(np.int64).values
        else:
            self.labels = None

    def __len__(self):
        return len(self.img_paths)

    @staticmethod
    def _read_rgb(path: str):
        data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return np.ascontiguousarray(img)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = self._read_rgb(img_path)

        if self.transforms is not None:
            img = self.transforms(image=img)["image"]
        else:
            img = ToTensorV2()(image=img)["image"]

        if self.has_labels:
            return img, int(self.labels[index])
        return img, -1




## === cell 6
data_transforms = {
    "train": A.Compose(
        [
            A.RandomResizedCrop(
                size=(CFG.img_size, CFG.img_size),
                p=1.0,
            ),
            A.CenterCrop(
                height=CFG.crop_size,
                width=CFG.crop_size,
                p=CFG.crop_p,
            ),
            A.Transpose(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(p=0.5),
            A.HueSaturationValue(
                hue_shift_limit=0.2,
                sat_shift_limit=0.2,
                val_shift_limit=0.2,
                p=0.5,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=(-0.1, 0.1),
                contrast_limit=(-0.1, 0.1),
                p=0.5,
            ),
            A.Resize(height=CFG.img_size, width=CFG.img_size, p=1.0),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0,
            ),
            A.CoarseDropout(p=0.5),
            ToTensorV2(),
        ],
        p=1.0,
    ),
    "valid": A.Compose(
        [
            A.CenterCrop(height=CFG.crop_size, width=CFG.crop_size, p=CFG.crop_p),
            A.Resize(height=CFG.img_size, width=CFG.img_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0,
            ),
            ToTensorV2(),
        ],
        p=1.0,
    ),
}




## === cell 7
def log_t(u, t):
    if t == 1.0:
        return u.log()
    else:
        return (u.pow(1.0 - t) - 1.0) / (1.0 - t)


def exp_t(u, t):
    if t == 1:
        return u.exp()
    else:
        return (1.0 + (1.0 - t) * u).relu().pow(1.0 / (1.0 - t))


def compute_normalization_fixed_point(activations, t, num_iters):
    mu, _ = torch.max(activations, -1, keepdim=True)
    normalized_activations_step_0 = activations - mu
    normalized_activations = normalized_activations_step_0
    for _ in range(num_iters):
        logt_partition = torch.sum(exp_t(normalized_activations, t), -1, keepdim=True)
        normalized_activations = normalized_activations_step_0 * logt_partition.pow(
            1.0 - t
        )
    logt_partition = torch.sum(exp_t(normalized_activations, t), -1, keepdim=True)
    normalization_constants = -log_t(1.0 / logt_partition, t) + mu
    return normalization_constants


def compute_normalization_binary_search(activations, t, num_iters):
    mu, _ = torch.max(activations, -1, keepdim=True)
    normalized_activations = activations - mu
    effective_dim = torch.sum(
        (normalized_activations > -1.0 / (1.0 - t)).to(torch.int32),
        dim=-1,
        keepdim=True,
    ).to(activations.dtype)
    shape_partition = activations.shape[:-1] + (1,)
    lower = torch.zeros(
        shape_partition, dtype=activations.dtype, device=activations.device
    )
    upper = -log_t(1.0 / effective_dim, t) * torch.ones_like(lower)
    for _ in range(num_iters):
        logt_partition = (upper + lower) / 2.0
        sum_probs = torch.sum(
            exp_t(normalized_activations - logt_partition, t), dim=-1, keepdim=True
        )
        update = (sum_probs < 1.0).to(activations.dtype)
        lower = torch.reshape(
            lower * update + (1.0 - update) * logt_partition, shape_partition
        )
        upper = torch.reshape(
            upper * (1.0 - update) + update * logt_partition, shape_partition
        )
    logt_partition = (upper + lower) / 2.0
    return logt_partition + mu


class ComputeNormalization(torch.autograd.Function):
    @staticmethod
    def forward(ctx, activations, t, num_iters):
        if t < 1.0:
            normalization_constants = compute_normalization_binary_search(
                activations, t, num_iters
            )
        else:
            normalization_constants = compute_normalization_fixed_point(
                activations, t, num_iters
            )
        ctx.save_for_backward(activations, normalization_constants)
        ctx.t = t
        return normalization_constants

    @staticmethod
    def backward(ctx, grad_output):
        activations, normalization_constants = ctx.saved_tensors
        t = ctx.t
        normalized_activations = activations - normalization_constants
        probabilities = exp_t(normalized_activations, t)
        escorts = probabilities.pow(t)
        escorts = escorts / escorts.sum(dim=-1, keepdim=True)
        grad_input = escorts * grad_output
        return grad_input, None, None


def compute_normalization(activations, t, num_iters=5):
    return ComputeNormalization.apply(activations, t, num_iters)


def tempered_softmax(activations, t, num_iters=5):
    if t == 1.0:
        return activations.softmax(dim=-1)
    normalization_constants = compute_normalization(activations, t, num_iters)
    return exp_t(activations - normalization_constants, t)


def bi_tempered_logistic_loss(
    activations, labels, t1, t2, label_smoothing=0.0, num_iters=5, reduction="mean"
):
    if len(labels.shape) < len(activations.shape):
        labels_onehot = torch.zeros_like(activations)
        labels_onehot.scatter_(1, labels[..., None], 1)
    else:
        labels_onehot = labels

    if label_smoothing > 0:
        num_classes = labels_onehot.shape[-1]
        labels_onehot = (
            1 - label_smoothing * num_classes / (num_classes - 1)
        ) * labels_onehot + label_smoothing / (num_classes - 1)

    probabilities = tempered_softmax(activations, t2, num_iters)

    loss_values = (
        labels_onehot * log_t(labels_onehot + 1e-10, t1)
        - labels_onehot * log_t(probabilities, t1)
        - labels_onehot.pow(2.0 - t1) / (2.0 - t1)
        + probabilities.pow(2.0 - t1) / (2.0 - t1)
    ).sum(dim=-1)

    if reduction == "none":
        return loss_values
    if reduction == "sum":
        return loss_values.sum()
    return loss_values.mean()




## === cell 8
class CUDAPrefetcher:
    def __init__(self, loader, device, use_channels_last: bool):
        self.loader = loader
        self.device = device
        self.use_channels_last = use_channels_last and (device.type == "cuda")
        self.stream = torch.cuda.Stream() if device.type == "cuda" else None

    def __iter__(self):
        if self.device.type != "cuda":
            for batch in self.loader:
                yield batch
            return

        first = True
        it = iter(self.loader)
        next_inputs, next_labels = None, None

        def _preload():
            nonlocal next_inputs, next_labels
            try:
                inputs, labels = next(it)
            except StopIteration:
                next_inputs, next_labels = None, None
                return
            with torch.cuda.stream(self.stream):
                inputs = inputs.to(self.device, non_blocking=True)
                if self.use_channels_last:
                    inputs = inputs.to(memory_format=torch.channels_last)
                labels = labels.to(self.device, non_blocking=True)
            next_inputs, next_labels = inputs, labels

        _preload()
        while next_inputs is not None:
            torch.cuda.current_stream().wait_stream(self.stream)
            inputs, labels = next_inputs, next_labels
            if first:
                first = False
            _preload()
            yield inputs, labels

    def __len__(self):
        return len(self.loader)


def train_model(
    model, optimizer, scheduler, num_epochs, dataloaders, dataset_sizes, device, fold
):
    start = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    history = defaultdict(list)
    scaler = amp.GradScaler(enabled=(device.type == "cuda"))

    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    prefetched = {}
    for phase in ["train", "valid"]:
        prefetched[phase] = CUDAPrefetcher(
            dataloaders[phase], device=device, use_channels_last=True
        )

    for epoch in range(1, num_epochs + 1):
        print(f"Epoch {epoch}/{num_epochs}")
        print("-" * 10)

        for phase in ["train", "valid"]:
            model.train() if phase == "train" else model.eval()

            running_loss = 0.0
            running_corrects = 0.0

            ctx = torch.enable_grad() if phase == "train" else torch.inference_mode()
            with ctx:
                for batch in prefetched[phase]:
                    if device.type == "cuda":
                        inputs, labels = batch  # already on GPU
                    else:
                        inputs, labels = batch
                        inputs = inputs.to(device, non_blocking=True)
                        labels = labels.to(device, non_blocking=True)

                    optimizer.zero_grad(set_to_none=True)

                    with torch.set_grad_enabled(phase == "train"):
                        with torch.amp.autocast(
                            device_type=device.type, enabled=(device.type == "cuda")
                        ):
                            outputs = model(inputs)
                            _, preds = torch.max(outputs, 1)
                            loss = bi_tempered_logistic_loss(
                                outputs,
                                labels,
                                t1=CFG.t1,
                                t2=CFG.t2,
                                label_smoothing=CFG.smoothing,
                            )

                        if phase == "train":
                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scaler.update()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += (preds == labels).sum().item()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            history[phase + " loss"].append(epoch_loss)
            history[phase + " acc"].append(epoch_acc)

            if phase == "train" and scheduler is not None:
                scheduler.step()

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == "valid" and epoch_acc >= best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - start
    print(
        "Training complete in {:.0f}h {:.0f}m {:.0f}s".format(
            time_elapsed // 3600,
            (time_elapsed % 3600) // 60,
            (time_elapsed % 3600) % 60,
        )
    )
    print("Best Accuracy", best_acc)

    model.load_state_dict(best_model_wts)
    return model, history




## === cell 9
def run_fold(model, optimizer, scheduler, device, fold, num_epochs=10):
    valid_df = df[df.kfold == fold].reset_index(drop=True)
    train_df = df[df.kfold != fold].reset_index(drop=True)

    train_data = CassavaLeafDataset(
        TRAIN_DIR, train_df, transforms=data_transforms["train"], has_labels=True
    )
    valid_data = CassavaLeafDataset(
        TRAIN_DIR, valid_df, transforms=data_transforms["valid"], has_labels=True
    )

    dataset_sizes = {"train": len(train_data), "valid": len(valid_data)}

    cpu_count = os.cpu_count() or 2
    num_workers = min(8, max(2, cpu_count // 2))

    common_loader_kwargs = dict(
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
    )
    if common_loader_kwargs["prefetch_factor"] is None:
        common_loader_kwargs.pop("prefetch_factor")

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=CFG.batch_size,
        shuffle=True,
        drop_last=False,
        **common_loader_kwargs,
    )
    valid_loader = DataLoader(
        dataset=valid_data,
        batch_size=CFG.batch_size,
        shuffle=False,
        drop_last=False,
        **common_loader_kwargs,
    )

    dataloaders = {"train": train_loader, "valid": valid_loader}

    model, history = train_model(
        model,
        optimizer,
        scheduler,
        num_epochs,
        dataloaders,
        dataset_sizes,
        device,
        fold,
    )
    return model, history




## === cell 10
model = timm.create_model(CFG.model_name, pretrained=True)

if hasattr(model, "classifier") and isinstance(model.classifier, nn.Module):
    num_features = model.classifier.in_features
    model.classifier = nn.Linear(num_features, CFG.num_classes)
elif hasattr(model, "fc") and isinstance(model.fc, nn.Module):
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, CFG.num_classes)
elif hasattr(model, "head") and isinstance(model.head, nn.Module):
    num_features = model.head.in_features
    model.head = nn.Linear(num_features, CFG.num_classes)
else:
    raise AttributeError(
        "Could not locate classifier/fc/head to replace for this timm model."
    )

model.to(CFG.device)

optimizer = optim.Adam(
    model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay, amsgrad=False
)


def fetch_scheduler(optimizer):
    if CFG.scheduler == "CosineAnnealingLR":
        return lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=CFG.T_max, eta_min=CFG.min_lr
        )
    elif CFG.scheduler == "CosineAnnealingWarmRestarts":
        return lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=CFG.T_0, T_mult=1, eta_min=CFG.min_lr
        )
    return None


scheduler = fetch_scheduler(optimizer)

if CFG.device.type == "cuda":
    try:
        model = torch.compile(model, mode="max-autotune", fullgraph=False)
    except Exception as e:
        print(f"torch.compile unavailable/failed, continuing eager. Reason: {e}")



## === cell 11
model, history = run_fold(
    model, optimizer, scheduler, device=CFG.device, fold=0, num_epochs=CFG.num_epochs
)



## === cell 12
if CFG.savemodel:
    torch.save(model, CFG.savemodelpath)



## === cell 13
if CFG.dosubmission:
    t_df = pd.read_csv(f"{ROOT_DIR}/sample_submission.csv")
    assert "image_id" in t_df.columns

    t_data = CassavaLeafDataset(
        TEST_DIR, t_df, transforms=data_transforms["valid"], has_labels=False
    )

    cpu_count = os.cpu_count() or 2
    num_workers = min(8, max(2, cpu_count // 2))

    t_loader = DataLoader(
        dataset=t_data,
        batch_size=32,  # speed-up; score-neutral
        num_workers=num_workers,
        pin_memory=(CFG.device.type == "cuda"),
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else 2,
        shuffle=False,
        drop_last=False,
    )

    model.eval()
    if CFG.device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    all_preds = []
    with torch.inference_mode():
        if CFG.device.type == "cuda":
            t_iter = CUDAPrefetcher(t_loader, device=CFG.device, use_channels_last=True)
            for inputs, _ in t_iter:
                with torch.amp.autocast(device_type=CFG.device.type, enabled=True):
                    outputs = model(inputs)
                all_preds.append(torch.argmax(outputs, dim=1).detach().cpu().numpy())
        else:
            for inputs, _ in t_loader:
                inputs = inputs.to(CFG.device, non_blocking=True)
                outputs = model(inputs)
                all_preds.append(torch.argmax(outputs, dim=1).detach().cpu().numpy())

    all_preds = np.concatenate(all_preds).astype(int)

    submit_df = pd.DataFrame({"image_id": t_df["image_id"].values, "label": all_preds})
    out_path = "/kaggle/working/submission.csv"
    submit_df.to_csv(out_path, index=False)
    print(f"Wrote submission to: {out_path} (rows={len(submit_df)})")
    print(submit_df.head())
