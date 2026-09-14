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

3.11

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
import random
import numpy as np
import pandas as pd



## === cell 1
try:
    sys.path.append("../input/pytorchimagemodels/pytorch-image-models-main")
except Exception:
    pass
import timm



## === cell 2
import os
import pandas as pd
import numpy as np
import random
import cv2

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

import timm
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedKFold

import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

try:
    from torchvision.io import read_file as tv_read_file
    from torchvision.io import decode_jpeg as tv_decode_jpeg

    _HAS_TVJPEG = True
except Exception:
    _HAS_TVJPEG = False




## === cell 3
def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    try:
        cv2.setNumThreads(min(8, os.cpu_count() or 1))
    except Exception:
        pass




## === cell 4
class Config:
    seed = 42
    data_dir = "../input/cassava-leaf-disease-classification/"
    train_data_dir = data_dir + "train_images/"
    train_csv_path = data_dir + "train.csv"

    arch = "maxxvit_rmlp_nano_rw_256"  ## model name
    device = "cuda" if torch.cuda.is_available() else "cpu"
    debug = True

    image_size = 256
    train_batch_size = 16
    val_batch_size = 32
    epochs = 10
    freeze_bn_epochs = 5

    lr = 1e-4
    min_lr = 1e-6
    weight_decay = 1e-6

    num_workers = min(8, os.cpu_count() or 1)

    num_splits = 5
    num_classes = 5
    T_0 = 10
    T_mult = 1
    accum_iter = 2
    verbose_step = 1

    criterion = "LabelSmoothingCrossEntropy"
    label_smoothing = 0.3

    train_id = [0, 1, 2, 3, 4]




## === cell 5
def load_image(image_path):
    if _HAS_TVJPEG:
        try:
            data = tv_read_file(image_path)  # uint8 1D tensor (CPU)
            img = tv_decode_jpeg(data, device="cpu")  # CHW, uint8, RGB
            img = img.permute(1, 2, 0).contiguous().numpy()  # HWC, contiguous
            return img
        except Exception:
            pass

    data = np.fromfile(image_path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if not img.flags["C_CONTIGUOUS"]:
        img = np.ascontiguousarray(img)
    return img




## === cell 6
from collections import OrderedDict


class _LRUCache:
    __slots__ = ("max_items", "d")

    def __init__(self, max_items=0):
        self.max_items = int(max_items)
        self.d = OrderedDict()

    def get(self, key):
        if self.max_items <= 0:
            return None
        v = self.d.get(key, None)
        if v is None:
            return None
        self.d.move_to_end(key, last=True)
        return v

    def set(self, key, value):
        if self.max_items <= 0:
            return
        if key in self.d:
            self.d[key] = value
            self.d.move_to_end(key, last=True)
            return
        self.d[key] = value
        if len(self.d) > self.max_items:
            self.d.popitem(last=False)




## === cell 7
class CassavaDataset(Dataset):
    def __init__(self, data_dir, df, transforms=None, output_label=True):
        self.data_dir = data_dir
        self.df = df
        self.transforms = transforms
        self.output_label = output_label

        self._image_ids = df["image_id"].values
        self._labels = df["label"].values if "label" in df.columns else None

        self.enable_cache = False
        self._cache = _LRUCache(max_items=0)

    def set_cache(self, enabled: bool, max_items: int = 0):
        self.enable_cache = bool(enabled)
        self._cache = _LRUCache(max_items=max_items if enabled else 0)

    def __len__(self):
        return len(self._image_ids)

    def __getitem__(self, index):
        image_id = self._image_ids[index]
        image_path = os.path.join(self.data_dir, image_id)

        if self.enable_cache:
            image = self._cache.get(image_id)
            if image is None:
                image = load_image(image_path)
                if image is None:
                    raise FileNotFoundError(image_path)
                if not image.flags["C_CONTIGUOUS"]:
                    image = np.ascontiguousarray(image)
                self._cache.set(image_id, image)
        else:
            image = load_image(image_path)
            if image is None:
                raise FileNotFoundError(image_path)

        if self.transforms is not None:
            image = self.transforms(image=image)["image"]
            if image.dtype != torch.float32:
                image = image.float()
        else:
            if image.ndim == 3:
                image = np.transpose(image, (2, 0, 1))  # HWC -> CHW
            image = torch.from_numpy(image).float()

        if not image.is_contiguous():
            image = image.contiguous()

        if self.output_label:
            return image, int(self._labels[index])
        else:
            return image




## === cell 8
class CassavaClassifier(nn.Module):
    def __init__(self, model_arch, num_classes, pretrained=False):
        super().__init__()
        self.model = timm.create_model(
            model_arch, pretrained=pretrained, num_classes=num_classes
        )

    def forward(self, x):
        x = self.model(x)
        return x




## === cell 9
_TRANSFORM_CACHE = {}


def get_train_transforms(CFG):
    key = ("train", CFG.image_size)
    if key in _TRANSFORM_CACHE:
        return _TRANSFORM_CACHE[key]
    tfm = A.Compose(
        [
            A.RandomResizedCrop(size=(CFG.image_size, CFG.image_size), p=0.5),
            A.Transpose(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(p=0.5),
            A.HueSaturationValue(
                hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5
            ),
            A.RandomBrightnessContrast(
                brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5
            ),
            A.CenterCrop(height=CFG.image_size, width=CFG.image_size, p=1.0),
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
    )
    _TRANSFORM_CACHE[key] = tfm
    return tfm


def get_val_transforms(CFG):
    key = ("val", CFG.image_size)
    if key in _TRANSFORM_CACHE:
        return _TRANSFORM_CACHE[key]
    tfm = A.Compose(
        [
            A.Resize(height=CFG.image_size, width=CFG.image_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0,
            ),
            ToTensorV2(),
        ],
        p=1.0,
    )
    _TRANSFORM_CACHE[key] = tfm
    return tfm




## === cell 10
def get_inference_transforms(CFG):
    key = ("infer", CFG.image_size)
    if key in _TRANSFORM_CACHE:
        return _TRANSFORM_CACHE[key]
    tfm = A.Compose(
        [
            A.Resize(height=CFG.image_size, width=CFG.image_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0,
            ),
            ToTensorV2(p=1.0),
        ],
        p=1.0,
    )
    _TRANSFORM_CACHE[key] = tfm
    return tfm




## === cell 11
def train_one_epoch(
    epoch,
    model,
    loss_fn,
    optimizer,
    train_loader,
    device,
    scaler,
    scheduler=None,
    schd_batch_update=False,
):
    model.train()
    running_loss = None

    accum_iter = CFG.accum_iter
    dev_is_cuda = device.type == "cuda"

    pbar = tqdm(enumerate(train_loader), total=len(train_loader), mininterval=20.0)
    optimizer.zero_grad(set_to_none=True)

    for step, (images, targets) in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True).long()

        with autocast(enabled=dev_is_cuda):
            preds = model(images)
            loss = loss_fn(preds, targets)
            loss = loss / accum_iter

        scaler.scale(loss).backward()

        loss_item = loss.detach().item() * accum_iter
        if running_loss is None:
            running_loss = loss_item
        else:
            running_loss = running_loss * 0.99 + loss_item * 0.01

        if ((step + 1) % accum_iter == 0) or ((step + 1) == len(train_loader)):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            if scheduler is not None and schd_batch_update:
                scheduler.step()

        if (step % max(1, CFG.verbose_step * 200) == 0) or (
            step + 1 == len(train_loader)
        ):
            pbar.set_description(f"Train epoch {epoch} loss: {running_loss:.5f}")

    if scheduler is not None and (not schd_batch_update):
        pass




## === cell 12
def _seed_worker(worker_id):
    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _fast_collate(batch):
    b0 = batch[0]
    if isinstance(b0, tuple) and len(b0) == 2 and torch.is_tensor(b0[0]):
        imgs = torch.stack([b[0] for b in batch], dim=0)
        targets = torch.as_tensor([b[1] for b in batch], dtype=torch.int64)
        return imgs, targets
    if torch.is_tensor(b0):
        return torch.stack(batch, dim=0)
    from torch.utils.data._utils.collate import default_collate

    return default_collate(batch)


def load_dataloader(CFG, df, idx, mode="train"):
    df_sel = df.loc[idx, :].reset_index(drop=True)

    if mode == "train":
        dataset = CassavaDataset(
            CFG.train_data_dir,
            df_sel,
            transforms=get_train_transforms(CFG),
            output_label=True,
        )
        dataset.set_cache(enabled=True, max_items=2048)

        bs = CFG.train_batch_size
        shuffle = True
        drop_last = True
        num_workers = CFG.num_workers
        persistent_workers = num_workers > 0
        prefetch = 6 if num_workers > 0 else None
    elif mode == "val":
        dataset = CassavaDataset(
            CFG.train_data_dir,
            df_sel,
            transforms=get_val_transforms(CFG),
            output_label=True,
        )
        num_workers = min(4, os.cpu_count() or 1)
        persistent_workers = num_workers > 0
        dataset.set_cache(enabled=True, max_items=1024)
        bs = CFG.val_batch_size
        shuffle = False
        drop_last = False
        prefetch = 6 if num_workers > 0 else None
    else:
        raise ValueError("mode must be 'train' or 'val'")

    g = torch.Generator()
    g.manual_seed(CFG.seed)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=bs,
        pin_memory=(CFG.device == "cuda"),
        pin_memory_device="cuda" if (CFG.device == "cuda") else "",
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch,
        worker_init_fn=_seed_worker if num_workers > 0 else None,
        generator=g if shuffle else None,
        drop_last=drop_last,
        collate_fn=_fast_collate,
        in_order=False if num_workers > 0 else True,
    )
    return loader




## === cell 13
def valid_one_epoch(
    epoch, model, loss_fn, val_loader, device, scheduler=None, schd_loss_update=False
):
    model.eval()

    loss_sum = 0.0
    sample_num = 0
    correct = 0
    dev_is_cuda = device.type == "cuda"

    pbar = tqdm(enumerate(val_loader), total=len(val_loader), mininterval=20.0)
    for step, (images, targets) in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True).long()
        with torch.no_grad():
            with autocast(enabled=dev_is_cuda):
                preds = model(images)
                loss = loss_fn(preds, targets)

        batch_preds = torch.argmax(preds, 1)
        correct += (batch_preds == targets).sum().item()

        bs = targets.shape[0]
        loss_sum += loss.item() * bs
        sample_num += bs

        if (step % max(1, CFG.verbose_step * 200) == 0) or (
            step + 1 == len(val_loader)
        ):
            pbar.set_description(f"Val epoch {epoch} loss: {loss_sum / sample_num:.5f}")

    accuracy = correct / sample_num
    print(f"Validation multi-class accuracy = {accuracy:.5f}")

    if scheduler is not None:
        if schd_loss_update:
            scheduler.step(loss_sum / sample_num)
        else:
            scheduler.step()

    return accuracy




## === cell 14
def inference_one_epoch(model, data_loader, device):
    model.eval()
    image_preds_all = []
    dev_is_cuda = device.type == "cuda"

    pbar = tqdm(enumerate(data_loader), total=len(data_loader), mininterval=20.0)
    with torch.inference_mode():
        for step, imgs in pbar:
            imgs = imgs.to(device, non_blocking=True)
            with autocast(enabled=dev_is_cuda):
                image_preds = model(imgs)
                image_preds = torch.softmax(image_preds, 1)
            image_preds_all.append(image_preds.detach().cpu().numpy())

            if (step % 400 == 0) or (step + 1 == len(data_loader)):
                pbar.set_description("Inference")

    image_preds_all = np.concatenate(image_preds_all, axis=0)
    return image_preds_all




## === cell 15
def _collect_norm_modules(net):
    norms = []
    for m in net.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
            norms.append(m)
    return norms


def freeze_batchnorm_stats(net, _norms=None):
    try:
        norms = _norms if _norms is not None else _collect_norm_modules(net)
        for m in norms:
            m.eval()
    except ValueError:
        print("error with batchnorm2d or layernorm")
        return




## === cell 16
class LabelSmoothingCrossEntropy(nn.Module):
    """
    NLL loss with label smoothing.
    """

    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, x, target):
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()




## === cell 17
train = pd.read_csv("../input/cassava-leaf-disease-classification/train.csv")
train.head()



## === cell 18
if __name__ == "__main__":
    CFG = Config
    seed_everything(CFG.seed)

    try:
        torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    device = torch.device(CFG.device)

    folds = StratifiedKFold(
        n_splits=CFG.num_splits, shuffle=True, random_state=CFG.seed
    ).split(np.arange(train.shape[0]), train.label.values)

    sample_sub = pd.read_csv(
        "../input/cassava-leaf-disease-classification/sample_submission.csv"
    )
    test = sample_sub[["image_id"]].copy()

    test_data_dir = "../input/cassava-leaf-disease-classification/test_images/"
    test_dataset = CassavaDataset(
        data_dir=test_data_dir,
        df=test,
        transforms=get_inference_transforms(CFG),
        output_label=False,
    )
    test_dataset.set_cache(enabled=True, max_items=2048)

    infer_num_workers = min(4, os.cpu_count() or 1)
    tst_loader = DataLoader(
        test_dataset,
        batch_size=CFG.val_batch_size,
        shuffle=False,
        num_workers=infer_num_workers,
        pin_memory=(CFG.device == "cuda"),
        pin_memory_device="cuda" if (CFG.device == "cuda") else "",
        persistent_workers=(infer_num_workers > 0),
        prefetch_factor=6 if infer_num_workers > 0 else None,
        worker_init_fn=_seed_worker if infer_num_workers > 0 else None,
        generator=None,
        collate_fn=_fast_collate,
        in_order=False if infer_num_workers > 0 else True,
    )

    model = CassavaClassifier(CFG.arch, CFG.num_classes, pretrained=True).to(device)

    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    _norms = _collect_norm_modules(model)

    if os.environ.get("ENABLE_TORCH_COMPILE", "0") == "1" and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode="default", fullgraph=False)
        except Exception:
            pass

    loss_fn = LabelSmoothingCrossEntropy(smoothing=CFG.label_smoothing).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=CFG.T_0, T_mult=CFG.T_mult, eta_min=CFG.min_lr
    )
    scaler = GradScaler(enabled=(device.type == "cuda"))

    for fold, (trn_idx, val_idx) in enumerate(folds):
        if fold > 0:
            break

        train_loader = load_dataloader(CFG, train, trn_idx, mode="train")
        val_loader = load_dataloader(CFG, train, val_idx, mode="val")

        best_acc = -1.0
        best_state = None

        for epoch in range(CFG.epochs):
            if epoch < CFG.freeze_bn_epochs:
                freeze_batchnorm_stats(model, _norms=_norms)

            train_one_epoch(
                epoch=epoch,
                model=model,
                loss_fn=loss_fn,
                optimizer=optimizer,
                train_loader=train_loader,
                device=device,
                scaler=scaler,
                scheduler=scheduler,
                schd_batch_update=False,
            )
            acc = valid_one_epoch(
                epoch=epoch,
                model=model,
                loss_fn=loss_fn,
                val_loader=val_loader,
                device=device,
                scheduler=scheduler,
                schd_loss_update=False,
            )

            if acc > best_acc:
                best_acc = acc
                best_state = {
                    k: v.detach().cpu() for k, v in model.state_dict().items()
                }

        if best_state is not None:
            model.load_state_dict(best_state, strict=True)

    tst_preds = inference_one_epoch(model, tst_loader, device)
    test["label"] = np.argmax(tst_preds, axis=1).astype(int)

    test[["image_id", "label"]].to_csv("submission.csv", index=False)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
