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

0.8224539135690541

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.11584) has done: 'Your code didn’t yield a score mainly because it is likely failing before creating `submission.csv`: it imports `skimage` (often not available) and it references a pretrained weight file path that is not present in your provided data tree. I keep the model and inference logic identical, but switch image loading to PIL/torchvision (already available via `torchvision`) and make pretrained weight loading conditional on file existence so submission always runs. I also fix two small correctness issues that can silently hurt accuracy: use `torch.cuda.is_available()` correctly and ensure transforms are applied in the safe order (Resize before ToTensor for PIL images), without changing the evaluation semantics. Finally, I enforce that predictions align 1:1 with `sample_submission.csv` row order and write a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import math
import random
from copy import deepcopy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torchvision import transforms
from PIL import Image

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True  # keep determinism
torch.backends.cudnn.benchmark = (
    True  # faster for fixed 512x512 convs; deterministic True keeps determinism
)

is_submission = False

data_path = "/kaggle/input/cassava-leaf-disease-classification"
csv_file_name = "train.csv" if not is_submission else "sample_submission.csv"
train_batch_size, val_batch_size, sub_batch_size = 4, 8, 8
train_rate = 0.7  # rate for train dataset
num_workers = 2
mean, std = 0.5, 0.5

use_balanced_sample = False
use_class_weight = not use_balanced_sample
mul_weight = torch.tensor([3.0, 1.5, 1.0, 1.0, 4.5])
learning_rate = 0.001
weight_decay = 0.00002
efficient_net_version = 3  # (1.4, 1.2)

train_epoch = (0, 20)
debug = not is_submission

use_pre_trained_weight = True
pre_trained_weight_path = (
    "../input/cassava-leaf-disease-classification-weight/weight.pth"
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PIN_MEMORY = torch.cuda.is_available()
PERSISTENT_WORKERS = num_workers > 0
PREFETCH_FACTOR = 4 if num_workers > 0 else None


def seed_worker(worker_id: int):
    worker_seed = (SEED + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


dl_generator = torch.Generator()
dl_generator.manual_seed(SEED)




## === cell 1
class CassavaLeafDiseaseDataset(Dataset):
    "Cassava Leaf Disease"

    def __init__(self, csv_file, root_dir, transform=None, train=True):
        """
        Args:
            csv_file (string): csv file path
            root_dir (string): directory path of exist all image
            transform (callable, optional): Optional transform for sample
        """
        self.cassava_leaf_disease = pd.read_csv(csv_file).reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform
        self.train = train
        self.folder = "train_images" if train else "test_images"

        self._image_ids = self.cassava_leaf_disease["image_id"].to_numpy()
        self._has_label = self.train and ("label" in self.cassava_leaf_disease.columns)
        self._labels = (
            self.cassava_leaf_disease["label"].to_numpy(dtype=np.int64)
            if self._has_label
            else None
        )

    def __len__(self):
        return len(self._image_ids)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = "/".join([self.root_dir, self.folder, self._image_ids[idx]])
        img = Image.open(img_name).convert("RGB")

        if self.transform:
            img = self.transform(img)

        label = int(self._labels[idx]) if self._has_label else 0
        sample = (img, label)
        return sample


def split_dataset(dataset, rate):
    train_length = int(len(dataset) * rate)
    val_length = len(dataset) - train_length
    g = torch.Generator()
    g.manual_seed(SEED)
    train_set, val_set = torch.utils.data.random_split(
        dataset, [train_length, val_length], generator=g
    )
    return train_set, val_set




## === cell 2
def _list_tfrec_files(root, train=True):
    folder = "train_tfrecords" if train else "test_tfrecords"
    p = os.path.join(root, folder)
    if not os.path.isdir(p):
        return []
    files = [os.path.join(p, f) for f in os.listdir(p) if f.endswith(".tfrec")]
    return sorted(files)


_HAS_TF = False
try:
    import tensorflow as tf  # type: ignore

    _HAS_TF = True
except Exception:
    _HAS_TF = False


class CassavaTFRecordIterable(IterableDataset):
    def __init__(self, tfrec_files, batch_size, is_train, mean, std, seed=42):
        super().__init__()
        self.tfrec_files = list(tfrec_files)
        self.batch_size = int(batch_size)
        self.is_train = bool(is_train)
        self.mean = float(mean)
        self.std = float(std)
        self.seed = int(seed)

        self._feature_desc = {
            "image": tf.io.FixedLenFeature([], tf.string),
            "image_id": tf.io.FixedLenFeature([], tf.string),
        }
        if self.is_train:
            self._feature_desc["label"] = tf.io.FixedLenFeature([], tf.int64)

    def _parse(self, ex):
        ex = tf.io.parse_single_example(ex, self._feature_desc)
        img = tf.io.decode_jpeg(ex["image"], channels=3)
        img = tf.image.resize(
            img, [512, 512], method=tf.image.ResizeMethod.BILINEAR, antialias=True
        )

        if self.is_train:
            angle = tf.random.stateless_uniform(
                [], seed=[self.seed, 1], minval=-30.0, maxval=30.0
            ) * (math.pi / 180.0)
            img = tf.image.rotate(img, angles=angle, interpolation="BILINEAR")
            img = tf.image.stateless_random_flip_left_right(img, seed=[self.seed, 2])
            img = tf.image.stateless_random_flip_up_down(img, seed=[self.seed, 3])

        img = tf.cast(img, tf.float32) / 255.0
        img = (img - self.mean) / self.std
        img = tf.transpose(img, [2, 0, 1])  # CHW for PyTorch

        if self.is_train:
            label = tf.cast(ex["label"], tf.int64)
        else:
            label = tf.constant(0, dtype=tf.int64)
        return img, label

    def __iter__(self):
        ds = tf.data.TFRecordDataset(
            self.tfrec_files, num_parallel_reads=tf.data.AUTOTUNE
        )
        opts = tf.data.Options()
        opts.experimental_deterministic = True
        ds = ds.with_options(opts)

        if self.is_train:
            ds = ds.shuffle(2048, seed=self.seed, reshuffle_each_iteration=True)

        ds = ds.map(self._parse, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(self.batch_size, drop_remainder=False)
        ds = ds.prefetch(tf.data.AUTOTUNE)

        for imgs, labels in ds.as_numpy_iterator():
            yield torch.from_numpy(imgs), torch.from_numpy(labels)


try:
    from torchvision.transforms import v2 as T

    _HAS_V2 = True
except Exception:
    _HAS_V2 = False

if _HAS_V2:
    if is_submission:
        transform = T.Compose(
            [
                T.Resize((512, 512), antialias=True),
                T.ToImage(),
                T.ToDtype(torch.float32, scale=True),
                T.Normalize((mean, mean, mean), (std, std, std)),
            ]
        )
    else:
        transform = T.Compose(
            [
                T.Resize((512, 512), antialias=True),
                T.RandomRotation(30),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.ToImage(),
                T.ToDtype(torch.float32, scale=True),
                T.Normalize((mean, mean, mean), (std, std, std)),
            ]
        )
else:
    if is_submission:
        transform = transforms.Compose(
            [
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
                transforms.Normalize((mean, mean, mean), (std, std, std)),
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.Resize((512, 512)),
                transforms.RandomRotation(30),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((mean, mean, mean), (std, std, std)),
            ]
        )

dataset = CassavaLeafDiseaseDataset(
    csv_file="/".join([data_path, csv_file_name]),
    root_dir=data_path,
    transform=transform,
    train=not is_submission,
)

if is_submission:
    sub_loader = DataLoader(
        dataset,
        batch_size=sub_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS,
        prefetch_factor=PREFETCH_FACTOR,
        worker_init_fn=seed_worker,
        generator=dl_generator,
    )
else:
    train_set, val_set = split_dataset(dataset, train_rate)

    train_tfrec_files = _list_tfrec_files(data_path, train=True) if _HAS_TF else []
    if _HAS_TF and len(train_tfrec_files) > 0:
        train_loader = DataLoader(
            CassavaTFRecordIterable(
                train_tfrec_files, train_batch_size, True, mean, std, seed=SEED
            ),
            batch_size=None,
            num_workers=0,
            pin_memory=PIN_MEMORY,
        )
        val_loader = DataLoader(
            CassavaTFRecordIterable(
                train_tfrec_files, val_batch_size, False, mean, std, seed=SEED
            ),
            batch_size=None,
            num_workers=0,
            pin_memory=PIN_MEMORY,
        )
    else:
        train_loader = DataLoader(
            train_set,
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=PIN_MEMORY,
            persistent_workers=PERSISTENT_WORKERS,
            prefetch_factor=PREFETCH_FACTOR,
            worker_init_fn=seed_worker,
            generator=dl_generator,
        )

        val_loader = DataLoader(
            val_set,
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=PIN_MEMORY,
            persistent_workers=PERSISTENT_WORKERS,
            prefetch_factor=PREFETCH_FACTOR,
            worker_init_fn=seed_worker,
            generator=dl_generator,
        )



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 3
if (not is_submission) and use_class_weight:

    def class_weight(label_count):
        total = sum(label_count)
        weight = [total / (2 * max(1, count)) for count in label_count]
        return torch.tensor(weight, dtype=torch.float32)

    vc = dataset.cassava_leaf_disease["label"].value_counts().to_dict()
    label_count = [int(vc.get(i, 0)) for i in range(5)]
    print("Label counts:", label_count)

    loss_weight = class_weight(label_count)
    loss_weight *= mul_weight

    if torch.cuda.is_available():
        loss_weight = loss_weight.cuda()
        print("Loss weight (cuda):", loss_weight)




## === cell 4
def progress_print(comment, current, total, num_of_print=50, slow=5):
    end = "\r" if current < total else "\n"
    symbol = ["\\", "/", "-"]
    done = int((current - 1) / total * num_of_print) if total > 0 else 0
    doing = symbol[(current % (len(symbol) * slow) // slow)] if current < total else ""
    yet = max(0, num_of_print - 1 - done)
    print(comment + ": " + "#" * done + doing + "*" * yet, end=end)


def num_of_correct(outputs, labels):
    _, predicted = torch.max(outputs, 1)
    c = (predicted == labels).squeeze()
    return c.sum().item()


def train(
    model=None,
    criterion=None,
    optimizer=None,
    data_loader=None,
    debug=True,
    epoch=0,
    debug_rate=0.01,
):
    model.train()
    data_loader_length = len(data_loader) if hasattr(data_loader, "__len__") else -1
    correct, total = 0, 0
    running_loss = 0.0

    for i, data in enumerate(data_loader):
        inputs, labels = data
        if torch.cuda.is_available():
            inputs = inputs.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        correct += num_of_correct(outputs, labels)
        total += len(inputs)
        running_loss += loss.item() * len(inputs)

    if debug:
        print(
            "epoch: %d (%s/%s),\tloss: %.3f"
            % (
                epoch + 1,
                str(data_loader_length),
                str(data_loader_length),
                running_loss / max(1, total),
            )
        )

    return correct / total, running_loss / total


def valid(model=None, data_loader=None, debug=True):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            inputs, labels = data
            if torch.cuda.is_available():
                inputs = inputs.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
            outputs = model(inputs)

            correct += num_of_correct(outputs, labels)
            total += len(inputs)
    if debug:
        print(
            "validation done (%d batches)"
            % (len(data_loader) if hasattr(data_loader, "__len__") else -1)
        )
    return correct / total


def get_confusion_matrix(model, data_loader, num_classes):
    model.eval()
    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    with torch.no_grad():
        for inputs, labels in data_loader:
            if torch.cuda.is_available():
                inputs = inputs.cuda(non_blocking=True)
            outputs = model(inputs)
            predicted = outputs.argmax(dim=1).detach().cpu()
            labels = labels.detach().cpu()

            idx = predicted * num_classes + labels
            cm += torch.bincount(idx, minlength=num_classes * num_classes).view(
                num_classes, num_classes
            )
    return cm.numpy().astype(np.int32)




## === cell 5
class ConvUnit(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int = 0,
        stride: int = 1,
        groups: int = 1,
        batch_norm=True,
        activation=True,
    ):
        super().__init__()
        modules = [
            torch.nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
                groups=groups,
                bias=not batch_norm,
            )
        ]
        if batch_norm:
            modules.append(torch.nn.BatchNorm2d(out_channels))
        if activation:
            modules.append(torch.nn.LeakyReLU())
        self.sequence = torch.nn.Sequential(*modules)
        self.out_channels = out_channels

    def forward(self, x):
        x = self.sequence(x)
        return x


def DConvUnit(in_channels: int, kernel_size: int, stride: int = 1) -> torch.nn.Module:
    padding = kernel_size // 2
    return ConvUnit(in_channels, in_channels, kernel_size, padding, stride, in_channels)


def EConvUnit(in_channels: int, factor: int = 6) -> torch.nn.Module:
    return ConvUnit(in_channels, factor * in_channels, 1)


def PConvUnit(in_channels, out_channels, activation=False) -> torch.nn.Module:
    return ConvUnit(in_channels, out_channels, 1, activation=activation)


class DSConvUnit(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        activation=True,
    ):
        super().__init__()
        modules = [
            DConvUnit(in_channels, kernel_size, stride),
            PConvUnit(in_channels, out_channels, activation),
        ]
        self.sequence = torch.nn.Sequential(*modules)
        self.out_channels = out_channels

    def forward(self, x):
        x = self.sequence(x)
        return x


class SEUnit(torch.nn.Module):
    def __init__(self, in_channels: int, se_ratio: float):
        super().__init__()
        se_channels = max(1, int(in_channels * se_ratio))
        self.sequence = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            ConvUnit(in_channels, se_channels, 1, batch_norm=False),
            ConvUnit(se_channels, in_channels, 1, batch_norm=False, activation=False),
            torch.nn.Sigmoid(),
        )
        self.out_channels = in_channels

    def forward(self, x):
        y = self.sequence(x)
        z = x * y
        return z


class BottleneckUnit(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        factor: int = 6,
        kernel_size: int = 3,
        stride: int = 1,
        has_se: bool = True,
        se_ratio: float = 0.25,
    ):
        super().__init__()
        modules = []
        self.residual = stride == 1 and in_channels == out_channels
        hid_channels = factor * in_channels
        if factor > 1:
            modules.append(EConvUnit(in_channels, factor))
        modules.append(DConvUnit(hid_channels, kernel_size, stride))
        if has_se:
            calced_se_ratio = se_ratio / factor
            modules.append(SEUnit(hid_channels, calced_se_ratio))
        modules.append(PConvUnit(hid_channels, out_channels, False))
        self.sequence = torch.nn.Sequential(*modules)
        self.activation = torch.nn.LeakyReLU()
        self.out_channels = in_channels

    def forward(self, x):
        y = self.sequence(x)
        z = self.activation(x + y if self.residual else y)
        return z




## === cell 6
def round_filters(filters, coef: float, divisor: int) -> int:
    filters *= coef
    filters = max(divisor, int(filters + divisor / 2) // divisor * divisor)
    return filters


def round_repeats(repeats, coef):
    repeats = int(math.ceil(repeats * coef)) if repeats > 0 else -repeats
    return repeats


class BlockArgs:
    def __init__(
        self, module: torch.nn.Module, filters: int, num_repeats: int = 1, **kwargs
    ):
        self.module = module
        self.filters = filters
        self.num_repeats = num_repeats
        self.kwargs = kwargs

    def __call__(
        self,
        in_channels: int,
        width_coef: float = 1.0,
        depth_coef: float = 1.0,
        divisor: int = 8,
    ) -> torch.nn.Module:
        out_channels = round_filters(self.filters, width_coef, divisor)
        num_repeats = round_repeats(self.num_repeats, depth_coef)
        modules = []
        kwargs_local = dict(self.kwargs)
        for _ in range(num_repeats):
            modules.append(self.module(in_channels, out_channels, **kwargs_local))
            if "stride" in kwargs_local:
                del kwargs_local["stride"]
            in_channels = out_channels
        return torch.nn.Sequential(*modules), in_channels


EFFICIENT_NET_BLOCK_ARGS = [
    BlockArgs(ConvUnit, 32, -1, kernel_size=3, stride=2, padding=1),
    BlockArgs(BottleneckUnit, 16, 1, factor=1),
    BlockArgs(BottleneckUnit, 24, 2, stride=2),
    BlockArgs(BottleneckUnit, 40, 2, kernel_size=5, stride=2),
    BlockArgs(BottleneckUnit, 80, 3, stride=2),
    BlockArgs(BottleneckUnit, 112, 3, kernel_size=5),
    BlockArgs(BottleneckUnit, 192, 4, kernel_size=5, stride=2),
    BlockArgs(BottleneckUnit, 320, 1),
]

EFFICIENT_NET_COMPOUND_COEF = [
    (1.0, 1.0),
    (1.1, 1.0),
    (1.2, 1.1),
    (1.4, 1.2),
    (1.8, 1.4),
    (2.2, 1.6),
    (2.6, 1.8),
    (3.1, 2.0),
]


class EfficientNet(torch.nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_of_class: int = 1000,
        depth_coef: float = 1.0,
        width_coef: float = 1.0,
        hidden_channels: int = 1280,
        block_args=EFFICIENT_NET_BLOCK_ARGS,
    ):
        super().__init__()
        modules = []
        self.hidden_channels = round_filters(hidden_channels, width_coef, 8)
        for block_arg in block_args:
            module, in_channels = block_arg(in_channels, width_coef, depth_coef)
            modules.append(module)
        modules.append(PConvUnit(in_channels, self.hidden_channels, True))
        modules.append(torch.nn.AdaptiveAvgPool2d((1, 1)))
        self.sequence = torch.nn.Sequential(*modules)
        self.linear = torch.nn.Linear(self.hidden_channels, num_of_class)

    def forward(self, x):
        x = self.sequence(x)
        x = x.view(-1, self.hidden_channels)
        x = self.linear(x)
        return x




## === cell 7
model = EfficientNet(3, 5, *EFFICIENT_NET_COMPOUND_COEF[efficient_net_version])

if use_pre_trained_weight and os.path.exists(pre_trained_weight_path):
    pre_trained_weight = torch.load(pre_trained_weight_path, map_location="cpu")
    model.load_state_dict(pre_trained_weight)

model = model.to(DEVICE)

try:
    if hasattr(torch, "compile"):
        model = torch.compile(model, mode="reduce-overhead")
except Exception:
    pass



## === cell 8
if not is_submission:
    if use_class_weight:
        criterion = torch.nn.CrossEntropyLoss(weight=loss_weight)
    elif mul_weight is not None:
        mul_weight = mul_weight.to(DEVICE) if torch.cuda.is_available() else mul_weight
        criterion = torch.nn.CrossEntropyLoss(weight=mul_weight)
    else:
        criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    train_accuracy = []
    train_losses = []
    val_accuracy = []
    best_accuracy = 0.0 if len(val_accuracy) == 0 else min(val_accuracy)
    best_model = deepcopy(model.state_dict())



## === cell 9
if not is_submission:
    for epoch in range(*train_epoch):
        train_acc, train_loss = train(
            model,
            criterion,
            optimizer,
            train_loader,
            epoch=epoch,
            debug=debug,
            debug_rate=0.1,
        )
        val_acc = valid(model, val_loader, debug=debug)
        train_accuracy.append(train_acc)
        train_losses.append(train_loss)
        val_accuracy.append(val_acc)
        if best_accuracy < val_acc:
            best_model = deepcopy(model.state_dict())
            best_accuracy = val_acc
            print("Best model is updated.")
        print(train_acc, val_acc)
    torch.save(best_model, "weight.pth")



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_55/56289705.py in <cell line: 0>()
      1 if not is_submission:
      2     for epoch in range(*train_epoch):
----> 3         train_acc, train_loss = train(
      4             model,
      5             criterion,

/tmp/ipykernel_55/624126785.py in train(model, criterion, optimizer, data_loader, debug, epoch, debug_rate)
     24 ):
     25     model.train()
---> 26     data_loader_length = len(data_loader) if hasattr(data_loader, "__len__") else -1
     27     correct, total = 0, 0
     28     running_loss = 0.0

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in __len__(self)
    525 
    526             # Cannot statically verify that dataset is Sized
--> 527             length = self._IterableDataset_len_called = len(self.dataset)  # type: ignore[assignment, arg-type]
    528             if (
    529                 self.batch_size is not None

TypeError: object of type 'CassavaTFRecordIterable' has no len()

## === cell 10
if not is_submission and debug:
    print(train_accuracy)
    print(train_losses)
    print(val_accuracy)



## === cell 11
if not is_submission and debug:
    plt.plot(train_accuracy, label="train")
    plt.plot(train_losses, label="loss")
    plt.plot(val_accuracy, label="validation")
    plt.legend(loc="upper left")
    plt.show()




## === cell 12
def show_confusion_matrix(confusion_matrix):
    row_sums = confusion_matrix.sum(axis=1)
    normalized_confusion_matrix = confusion_matrix / row_sums[:, np.newaxis]
    diagonal_confusion_matrix = deepcopy(confusion_matrix)
    np.fill_diagonal(diagonal_confusion_matrix, 0)
    row_sums = diagonal_confusion_matrix.sum(axis=1)
    normalized_diagonal_confusion_matrix = (
        diagonal_confusion_matrix / row_sums[:, np.newaxis]
    )
    plt.figure(figsize=(2, 1))
    plt.matshow(normalized_confusion_matrix, cmap="gray")
    plt.matshow(normalized_diagonal_confusion_matrix, cmap="gray")
    plt.show()
    print(confusion_matrix)
    print(normalized_confusion_matrix)




## === cell 13
if not is_submission and debug:
    confusion_matrix = get_confusion_matrix(model, val_loader, 5)
    show_confusion_matrix(confusion_matrix)



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
InvalidArgumentError                      Traceback (most recent call last)
/tmp/ipykernel_55/1167571672.py in <cell line: 0>()
      2 # Keep it available but only run when debug is True.
      3 if not is_submission and debug:
----> 4     confusion_matrix = get_confusion_matrix(model, val_loader, 5)
      5     show_confusion_matrix(confusion_matrix)
      6 

/tmp/ipykernel_55/624126785.py in get_confusion_matrix(model, data_loader, num_classes)
     83     cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)
     84     with torch.no_grad():
---> 85         for inputs, labels in data_loader:
     86             if torch.cuda.is_available():
     87                 inputs = inputs.cuda(non_blocking=True)

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in __next__(self)
    706                 # TODO(https://github.com/pytorch/pytorch/issues/76750)
    707                 self._reset()  # type: ignore[call-arg]
--> 708             data = self._next_data()
    709             self._num_yielded += 1
    710             if (

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _next_data(self)
    762     def _next_data(self):
    763         index = self._next_index()  # may raise StopIteration
--> 764         data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
    765         if self._pin_memory:
    766             data = _utils.pin_memory.pin_memory(data, self._pin_memory_device)

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py in fetch(self, possibly_batched_index)
     40                 raise StopIteration
     41         else:
---> 42             data = next(self.dataset_iter)
     43         return self.collate_fn(data)
     44 

/tmp/ipykernel_55/745255779.py in __iter__(self)
     82         ds = ds.prefetch(tf.data.AUTOTUNE)
     83 
---> 84         for imgs, labels in ds.as_numpy_iterator():
     85             yield torch.from_numpy(imgs), torch.from_numpy(labels)
     86 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/dataset_ops.py in __next__(self)
   4786       return numpy
   4787 
-> 4788     return nest.map_structure(to_numpy, next(self._iterator))
   4789 
   4790   def next(self):

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/iterator_ops.py in __next__(self)
    824   def __next__(self):
    825     try:
--> 826       return self._next_internal()
    827     except errors.OutOfRangeError:
    828       raise StopIteration

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/iterator_ops.py in _next_internal(self)
    774     # to communicate that there is no more data to iterate over.
    775     with context.execution_mode(context.SYNC):
--> 776       ret = gen_dataset_ops.iterator_get_next(
    777           self._iterator_resource,
    778           output_types=self._flat_output_types,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/gen_dataset_ops.py in iterator_get_next(iterator, output_types, output_shapes, name)
   3084       return _result
   3085     except _core._NotOkStatusException as e:
-> 3086       _ops.raise_from_not_ok_status(e, name)
   3087     except _core._FallbackException:
   3088       pass

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py in raise_from_not_ok_status(e, name)
   6000 def raise_from_not_ok_status(e, name) -> NoReturn:
   6001   e.message += (" name: " + str(name if name is not None else ""))
-> 6002   raise core._status_to_exception(e) from None  # pylint: disable=protected-access
   6003 
   6004 

InvalidArgumentError: {{function_node __wrapped__IteratorGetNext_output_types_2_device_/job:localhost/replica:0/task:0/device:CPU:0}} Feature: image_id (data type: string) is required but could not be found.
	 [[{{node ParseSingleExample/ParseExample/ParseExampleV2}}]] [Op:IteratorGetNext] name: 

## === cell 14
if not is_submission:
    if os.path.exists("weight.pth"):
        model.load_state_dict(torch.load("weight.pth", map_location="cpu"))
        model = model.to(DEVICE)

    if _HAS_V2:
        test_transform = T.Compose(
            [
                T.Resize((512, 512), antialias=True),
                T.ToImage(),
                T.ToDtype(torch.float32, scale=True),
                T.Normalize((mean, mean, mean), (std, std, std)),
            ]
        )
    else:
        test_transform = transforms.Compose(
            [
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
                transforms.Normalize((mean, mean, mean), (std, std, std)),
            ]
        )

    test_dataset = CassavaLeafDiseaseDataset(
        csv_file="/".join([data_path, "sample_submission.csv"]),
        root_dir=data_path,
        transform=test_transform,
        train=False,
    )

    test_tfrec_files = _list_tfrec_files(data_path, train=False) if _HAS_TF else []
    if _HAS_TF and len(test_tfrec_files) > 0:
        sub_loader = DataLoader(
            CassavaTFRecordIterable(
                test_tfrec_files, sub_batch_size, False, mean, std, seed=SEED
            ),
            batch_size=None,
            num_workers=0,
            pin_memory=PIN_MEMORY,
        )
    else:
        sub_loader = DataLoader(
            test_dataset,
            batch_size=sub_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=PIN_MEMORY,
            persistent_workers=PERSISTENT_WORKERS,
            prefetch_factor=PREFETCH_FACTOR,
            worker_init_fn=seed_worker,
            generator=dl_generator,
        )

    model.eval()
    submission = test_dataset.cassava_leaf_disease.copy()
    predicts = []
    with torch.no_grad():
        for i, data in enumerate(sub_loader):
            inputs, _ = data
            inputs = inputs.to(DEVICE, non_blocking=True)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            predicts.extend(predicted.cpu().numpy().tolist())

    assert len(predicts) == len(
        submission
    ), f"Pred length {len(predicts)} != submission length {len(submission)}"

    submission["label"] = predicts
    submission.to_csv("submission.csv", index=False)
    print("Wrote submission.csv with shape:", submission.shape)

## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
InvalidArgumentError                      Traceback (most recent call last)
/tmp/ipykernel_55/927333871.py in <cell line: 0>()
     57     predicts = []
     58     with torch.no_grad():
---> 59         for i, data in enumerate(sub_loader):
     60             inputs, _ = data
     61             inputs = inputs.to(DEVICE, non_blocking=True)

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in __next__(self)
    706                 # TODO(https://github.com/pytorch/pytorch/issues/76750)
    707                 self._reset()  # type: ignore[call-arg]
--> 708             data = self._next_data()
    709             self._num_yielded += 1
    710             if (

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _next_data(self)
    762     def _next_data(self):
    763         index = self._next_index()  # may raise StopIteration
--> 764         data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
    765         if self._pin_memory:
    766             data = _utils.pin_memory.pin_memory(data, self._pin_memory_device)

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py in fetch(self, possibly_batched_index)
     40                 raise StopIteration
     41         else:
---> 42             data = next(self.dataset_iter)
     43         return self.collate_fn(data)
     44 

/tmp/ipykernel_55/745255779.py in __iter__(self)
     82         ds = ds.prefetch(tf.data.AUTOTUNE)
     83 
---> 84         for imgs, labels in ds.as_numpy_iterator():
     85             yield torch.from_numpy(imgs), torch.from_numpy(labels)
     86 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/dataset_ops.py in __next__(self)
   4786       return numpy
   4787 
-> 4788     return nest.map_structure(to_numpy, next(self._iterator))
   4789 
   4790   def next(self):

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/iterator_ops.py in __next__(self)
    824   def __next__(self):
    825     try:
--> 826       return self._next_internal()
    827     except errors.OutOfRangeError:
    828       raise StopIteration

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/iterator_ops.py in _next_internal(self)
    774     # to communicate that there is no more data to iterate over.
    775     with context.execution_mode(context.SYNC):
--> 776       ret = gen_dataset_ops.iterator_get_next(
    777           self._iterator_resource,
    778           output_types=self._flat_output_types,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/gen_dataset_ops.py in iterator_get_next(iterator, output_types, output_shapes, name)
   3084       return _result
   3085     except _core._NotOkStatusException as e:
-> 3086       _ops.raise_from_not_ok_status(e, name)
   3087     except _core._FallbackException:
   3088       pass

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py in raise_from_not_ok_status(e, name)
   6000 def raise_from_not_ok_status(e, name) -> NoReturn:
   6001   e.message += (" name: " + str(name if name is not None else ""))
-> 6002   raise core._status_to_exception(e) from None  # pylint: disable=protected-access
   6003 
   6004 

InvalidArgumentError: {{function_node __wrapped__IteratorGetNext_output_types_2_device_/job:localhost/replica:0/task:0/device:CPU:0}} Feature: image_id (data type: string) is required but could not be found.
	 [[{{node ParseSingleExample/ParseExample/ParseExampleV2}}]] [Op:IteratorGetNext] name:
