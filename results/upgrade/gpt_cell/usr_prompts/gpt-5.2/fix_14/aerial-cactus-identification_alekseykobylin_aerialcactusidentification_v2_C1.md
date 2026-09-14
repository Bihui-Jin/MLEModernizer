# Goal

You will receive environment details and a partial notebook export.

# Requirements

- Fix the bug that causes the error in cell k.
- Do NOT adjust any other non-buggy cells.
- You may reference cell k+1 only to preserve variable/interface compatibility.
- Do not complete or extend code logic in cell k, k+1, or later cells.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (bug fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Output must follow your strict format: Diagnosis / Patch summary / Updated cells / Compatibility notes for cell k+1 / Assumptions.


# 1. Python version

3.8

# 2. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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
tqdm==4.67.1

# 3. Data file paths

```
/
    kaggle/
        data/
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
        input/
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
        working/
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
```

-> data/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> data/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> (stopped after 10 files for performance)

# 4. Code solution

## === cell 0
!ls ../input/aerial-cactus-identification
!mkdir data
!cp ../input/aerial-cactus-identification/* data
!ls data


## === cell 1
!unzip -o data/test.zip -d data
!unzip -o data/train.zip -d data
!rm -f data/*.zip
!ls data


## === cell 2
import os
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.transforms as transforms
import torchvision
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score
from PIL import Image


## === cell 3
CLASS_NAMES = ('0', '1')
data_root = './data'
train_data_path = os.path.join(data_root, 'train')
test_data_path = os.path.join(data_root, 'test')
train_set_path = os.path.join(train_data_path, 'train')
dev_set_path = os.path.join(train_data_path, 'dev')
test_set_path = os.path.join(train_data_path, 'test')
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]


## === cell 4
train_targets = pd.read_csv(os.path.join(data_root, 'train.csv'))
train_targets.head()


## === cell 5
X = train_targets['id'].values
y = train_targets['has_cactus'].values.astype(int)
print(X.shape, y.shape)


## === cell 6
from sklearn.model_selection import train_test_split
X_train, _X_test, y_train, _y_test = train_test_split(X, y, test_size=0.1, shuffle=True, stratify=y)
X_dev, X_test, y_dev, y_test = train_test_split(_X_test, _y_test, test_size=0.5, shuffle=True, stratify=_y_test)
print(X_train.shape, X_dev.shape, X_test.shape)
no_cactus_weight = (y_train==1).sum() / y_train.shape[0]
has_cactus_weight = (y_train==0).sum() / y_train.shape[0]
print(has_cactus_weight, no_cactus_weight)


## === cell 7
for set_path in (train_set_path, dev_set_path, test_set_path):
  os.system(f"mkdir {set_path}")
  for class_name in CLASS_NAMES:
    os.system(f"mkdir {os.path.join(set_path, class_name)}")


## === cell 8
for path, file_names, labels in ((train_set_path, X_train, y_train), (dev_set_path, X_dev, y_dev), (test_set_path, X_test, y_test)):
  print(path)
  for file_name, label in zip(file_names, labels):
    os.system(f"mv -f {os.path.join(train_data_path, file_name)} {os.path.join(path, str(label), file_name)}")


## === cell 9
import os
import math
import random
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
import matplotlib.pyplot as plt


class SummaryWriter:  # minimal no-op drop-in replacement
    def __init__(self, *args, **kwargs):
        pass

    def add_scalar(self, *args, **kwargs):
        pass

    def add_scalars(self, *args, **kwargs):
        pass

    def add_figure(self, *args, **kwargs):
        pass

    def close(self):
        pass


from tqdm.autonotebook import tqdm


METRICS = {
    "accuracy": {"f": balanced_accuracy_score, "args": {}},
}


NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]


def make_image_label_grid(images, labels=None, class_names=None):
    channels = images.shape[1]
    if channels not in (3, 1):
        raise ValueError("Images must have 1 or 3 channels")
    mean = NORM_MEAN if channels == 3 else [sum(NORM_MEAN) / 3]
    std = NORM_STD if channels == 3 else [sum(NORM_STD) / 3]
    mean = torch.tensor(mean)
    std = torch.tensor(std)
    mean = (-mean / std).tolist()
    std = (1.0 / std).tolist()
    img_grid = torchvision.utils.make_grid(images)
    img_grid = F.normalize(img_grid, mean=mean, std=std)
    return img_grid


def make_image_label_figure(images, labels=None, class_names=None):
    channels = images.shape[1]
    if channels not in (3, 1):
        raise ValueError("Images must have 1 or 3 channels")
    mean = NORM_MEAN if channels == 3 else [sum(NORM_MEAN) / 3]
    std = NORM_STD if channels == 3 else [sum(NORM_STD) / 3]
    mean = torch.tensor(mean)
    std = torch.tensor(std)
    mean = (-mean / std).tolist()
    std = (1.0 / std).tolist()
    n = int(math.sqrt(len(images)))
    figure = plt.figure(figsize=(n, n))
    figure.subplots_adjust(hspace=0.4, wspace=0.4)
    for i in range(n * n):
        image, label = images[i], (0 if labels is None else labels[i])
        image = F.normalize(image, mean=mean, std=std)
        image = image.permute(1, 2, 0)
        image = torch.squeeze(image)
        image = (image * 255).int()
        plt.subplot(
            n, n, i + 1, title="NA" if class_names is None else class_names[label]
        )
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image, cmap="gray" if channels == 1 else None)
    return figure


class Transforms(transforms.Compose):

    def __init__(self, in_channels=1, out_channels=1, size=(32, 32)):
        if out_channels not in (3, 1) or in_channels not in (3, 1):
            raise ValueError("Images must have 1 or 3 channels")
        mean = NORM_MEAN if out_channels == 3 else [sum(NORM_MEAN) / 3]
        std = NORM_STD if out_channels == 3 else [sum(NORM_STD) / 3]
        transforms_list = []
        if in_channels != out_channels:
            transforms_list.append(transforms.Grayscale(out_channels))
        transforms_list.extend(
            [
                transforms.Resize(size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
        super(Transforms, self).__init__(transforms_list)


class TrainTransforms(transforms.Compose):

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        size=(32, 32),
        random_crop=False,
        random_affine=False,
        horizontal_flip=False,
        color_jitter=False,
        random_erasing=False,
    ):
        if out_channels not in (3, 1) or in_channels not in (3, 1):
            raise ValueError("Images must have 1 or 3 channels")
        mean = NORM_MEAN if out_channels == 3 else [sum(NORM_MEAN) / 3]
        std = NORM_STD if out_channels == 3 else [sum(NORM_STD) / 3]
        transforms_list = []
        if in_channels != out_channels:
            transforms_list.append(transforms.Grayscale(out_channels))
        if random_affine:
            transforms_list.append(
                transforms.RandomAffine(
                    degrees=10.0, translate=(0.25, 0.25), shear=(-10, 10, -10, 10)
                )
            )
        if random_crop:
            transforms_list.append(
                transforms.RandomResizedCrop(size, scale=(0.9, 1.1), ratio=(0.75, 1.33))
            )
        else:
            transforms_list.append(transforms.Resize(size))
        if horizontal_flip:
            transforms_list.append(transforms.RandomHorizontalFlip(p=0.5))
        if color_jitter:
            transforms_list.append(
                transforms.ColorJitter(
                    brightness=(0.75, 1.5),
                    contrast=(0.75, 1.5),
                    saturation=(0.75, 1.5),
                    hue=(-0.1, 0.1),
                )
            )
        transforms_list.append(transforms.ToTensor())
        if random_erasing:
            transforms_list.append(
                transforms.RandomErasing(
                    p=0.2, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0
                )
            )
        transforms_list.append(transforms.Normalize(mean=mean, std=std))
        super(TrainTransforms, self).__init__(transforms_list)


class TrainerProgressBar(tqdm):

    def __init__(self, desc=None, total=10, unit="it", position=None):
        super(TrainerProgressBar, self).__init__(
            desc=desc,
            total=total,
            leave=True,
            unit=unit,
            position=position,
            dynamic_ncols=True,
        )

    def reset(self, total=None, desc=None, ordered_dict=None):
        self.last_print_n = self.n = 0
        self.last_print_t = self.start_t = self._time()
        if total is not None:
            self.total = total
        super(TrainerProgressBar, self).refresh()
        if desc is not None:
            super(TrainerProgressBar, self).set_description(desc)
        if ordered_dict is not None:
            super(TrainerProgressBar, self).set_postfix(ordered_dict)

    def update(self, desc=None, ordered_dict=None, n=1):
        if desc is not None:
            super(TrainerProgressBar, self).set_description(desc)
        if ordered_dict is not None:
            super(TrainerProgressBar, self).set_postfix(ordered_dict)
        super(TrainerProgressBar, self).update(n)


class TensorBoardLogger(SummaryWriter):

    def __init__(
        self,
        log_dir=None,
        comment="",
        purge_step=None,
        max_queue=10,
        flush_secs=120,
        filename_suffix="",
        class_names=None,
        testloader=None,
        device=None,
    ):
        super(TensorBoardLogger, self).__init__(
            log_dir=log_dir,
            comment=comment,
            purge_step=purge_step,
            max_queue=max_queue,
            flush_secs=flush_secs,
            filename_suffix=filename_suffix,
        )
        self.class_names = class_names
        self.testloader = testloader
        self.device = (
            torch.device(device)
            if device
            else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        )

    def epoch_callback(
        self, net, epoch, lr, train_loss, val_loss, train_metrics_dict, val_metrics_dict
    ):
        super(TensorBoardLogger, self).add_scalar(f"epoch/lr", lr, epoch)
        super(TensorBoardLogger, self).add_scalars(
            f"epoch/loss", {"train": train_loss, "val": val_loss}, epoch
        )
        for metric_name in train_metrics_dict:
            super(TensorBoardLogger, self).add_scalars(
                f"epoch/{metric_name}",
                {
                    "train": train_metrics_dict[metric_name],
                    "val": val_metrics_dict[metric_name],
                },
                epoch,
            )
        if self.testloader:
            inputs, targets = next(iter(self.testloader))
            _inputs = inputs.to(self.device)
            predictions = net.forward(_inputs).argmax(dim=1).data.cpu()
            start = random.randrange(0, len(targets) - 9)
            stop = start + 9
            super(TensorBoardLogger, self).add_figure(
                "examples/real",
                make_image_label_figure(
                    inputs[start:stop], targets[start:stop], self.class_names
                ),
            )
            super(TensorBoardLogger, self).add_figure(
                "examples/predicted",
                make_image_label_figure(
                    inputs[start:stop], predictions[start:stop], self.class_names
                ),
            )

    def batch_callback(self, train, epoch, batch, batches, loss, metrics_dict):
        section = "train_batch" if train else "val_batch"
        super(TensorBoardLogger, self).add_scalar(
            f"{section}/loss", loss, batch + epoch * batches
        )
        for metric_name, metric_value in metrics_dict.items():
            if isinstance(metric_value, dict):
                super(TensorBoardLogger, self).add_scalars(
                    f"{section}/{metric_name}", metric_value, batch + epoch * batches
                )
            else:
                super(TensorBoardLogger, self).add_scalar(
                    f"{section}/{metric_name}", metric_value, batch + epoch * batches
                )


class PyTorchTrainer(object):

    def __init__(
        self, device=None, metrics=None, epoch_callback=None, batch_callback=None
    ):
        self.device = (
            torch.device(device)
            if device
            else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        )
        self.metrics = metrics or METRICS
        self.epoch_callback = epoch_callback
        self.batch_callback = batch_callback

        self.train_pb = None
        self.epoch_train_pb = None
        self.epoch_val_pb = None

    def train(
        self,
        model,
        optimizer,
        loss_criterion,
        train_data_loader,
        val_data_loader,
        scheduler=None,
        epochs=10,
    ):
        print(
            f"=============================== Training NN ==============================="
        )
        print(f"== Epochs:              {epochs:6d}")
        print(f"== Train batch size:    {train_data_loader.batch_size:6d}")
        print(f"== Train batches:       {len(train_data_loader):6d}")
        print(f"== Validate batch size: {val_data_loader.batch_size:6d}")
        print(f"== Validate batches:    {len(val_data_loader):6d}")
        print(
            f"==========================================================================="
        )
        self.train_pb = TrainerProgressBar(
            desc=f"== Epoch {1}", total=epochs, unit="epoch", position=0
        )
        self.epoch_train_pb = TrainerProgressBar(
            desc=f"== Train {1}", total=len(train_data_loader), unit="batch", position=1
        )
        self.epoch_val_pb = TrainerProgressBar(
            desc=f"== Val {1}", total=len(val_data_loader), unit="batch", position=2
        )
        self.train_pb.reset(total=epochs)
        self.epoch_train_pb.reset(total=len(train_data_loader))
        self.epoch_val_pb.reset(total=len(val_data_loader))
        for epoch in range(epochs):
            train_loss, train_metrics_dict = self.forward_batches(
                model, optimizer, loss_criterion, train_data_loader, epoch, train=True
            )
            val_loss, val_metrics_dict = self.forward_batches(
                model, optimizer, loss_criterion, val_data_loader, epoch, train=False
            )
            if scheduler:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(train_loss, epoch=epoch)
                else:
                    scheduler.step(epoch=epoch)
            lr = optimizer.param_groups[0]["lr"]
            if self.epoch_callback:
                self.epoch_callback(
                    model,
                    epoch,
                    lr,
                    train_loss,
                    val_loss,
                    train_metrics_dict,
                    val_metrics_dict,
                )
            metrics_dict = {"lr": lr}
            for metric_name in train_metrics_dict:
                metrics_dict[f"train_{metric_name}"] = train_metrics_dict[metric_name]
                metrics_dict[f"val_{metric_name}"] = val_metrics_dict[metric_name]
            metrics_dict.update({"train_loss": train_loss, "val_loss": val_loss})
            self.train_pb.update(desc=f"== Epoch {epoch+1}", ordered_dict=metrics_dict)
        self.train_pb.close()
        self.epoch_train_pb.close()
        self.epoch_val_pb.close()
        print(
            f"==========================================================================="
        )

    def forward_batches(
        self, model, optimizer, loss_criterion, data_loader, epoch, train=True
    ):
        if train:
            model.train()
        else:
            model.eval()
        avg_loss_value = 0
        avg_metrics_dict = None
        batches = len(data_loader)
        if train:
            self.epoch_train_pb.reset(batches, f"== Train {epoch+1}")
        else:
            self.epoch_val_pb.reset(batches, f"== Val {epoch+1}")
        for batch_i, data in enumerate(data_loader, 1):
            loss_value, predictions, targets = self.forward_batch(
                model, optimizer, loss_criterion, data, train=train
            )
            metrics_dict = self.metrics_dict(predictions, targets)
            avg_loss_value += loss_value
            if avg_metrics_dict is None:
                avg_metrics_dict = metrics_dict.copy()
            else:
                for metric_name in avg_metrics_dict:
                    avg_metrics_dict[metric_name] += metrics_dict[metric_name]
            if self.batch_callback:
                self.batch_callback(
                    train, epoch, batch_i, batches, loss_value, metrics_dict
                )
            metrics_dict.update({"loss": avg_loss_value / batch_i})
            if train:
                self.epoch_train_pb.update(ordered_dict=metrics_dict)
            else:
                self.epoch_val_pb.update(ordered_dict=metrics_dict)
        avg_loss_value /= batches
        for metric_name in avg_metrics_dict:
            avg_metrics_dict[metric_name] /= batches
        return avg_loss_value, avg_metrics_dict

    def forward_batch(self, model, optimizer, loss_criterion, batch_data, train=True):
        inputs, targets = batch_data
        _inputs = inputs.to(self.device)
        _targets = targets.to(self.device)
        with torch.set_grad_enabled(train):
            _outputs = model.forward(_inputs)
            loss = loss_criterion(_outputs, _targets)
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss_value = loss.item()
        predictions = _outputs.argmax(dim=1).data.cpu()
        targets = targets.data
        return loss_value, predictions, targets

    def metrics_dict(self, predictions, targets):
        d = {}
        for metric_name in self.metrics:
            metric_value = self.metrics[metric_name]["f"](
                predictions, targets, **self.metrics[metric_name]["args"]
            )
            d[metric_name] = metric_value

        return d


## === cell 10
class TrainTransforms(transforms.Compose):

    def __init__(self):
        super(TrainTransforms, self).__init__([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
        ])


class TestTransforms(transforms.Compose):

    def __init__(self):
        super(TestTransforms, self).__init__([
            transforms.ToTensor(),
            transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
        ])


## === cell 11


def _count_images_in_dir(
    root_dir,
    exts=(".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp"),
):
    if not os.path.isdir(root_dir):
        return 0
    n = 0
    for r, _, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith(exts):
                n += 1
    return n


def _restore_train_sources_from_splits(train_data_path, split_paths):
    exts = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")
    os.makedirs(train_data_path, exist_ok=True)
    for split_path in split_paths:
        if not os.path.isdir(split_path):
            continue
        for class_name in CLASS_NAMES:
            class_dir = os.path.join(split_path, class_name)
            if not os.path.isdir(class_dir):
                continue
            for f in os.listdir(class_dir):
                if f.lower().endswith(exts):
                    src = os.path.join(class_dir, f)
                    dst = os.path.join(train_data_path, f)
                    if os.path.exists(src) and (not os.path.exists(dst)):
                        os.rename(src, dst)


def _find_train_image_root(candidates):
    best_path = None
    best_count = -1
    for cand in candidates:
        c = _count_images_in_dir(cand)
        if c > best_count:
            best_count = c
            best_path = cand
    if best_count <= 0:
        for cand in candidates:
            if not os.path.isdir(cand):
                continue
            for name in os.listdir(cand):
                sub = os.path.join(cand, name)
                if os.path.isdir(sub):
                    c = _count_images_in_dir(sub)
                    if c > best_count:
                        best_count = c
                        best_path = sub
    return best_path if best_count > 0 else candidates[0]


def _build_filename_to_path_map(
    root_dir,
    exts=(".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp"),
):
    m = {}
    if not os.path.isdir(root_dir):
        return m
    for r, _, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith(exts):
                m.setdefault(f, os.path.join(r, f))
    return m


def _split_has_all_classes_images(split_path):
    for class_name in CLASS_NAMES:
        class_dir = os.path.join(split_path, class_name)
        if _count_images_in_dir(class_dir) == 0:
            return False
    return True


def _resolve_src_path(file_id, fname_to_path, base_dir):
    exts = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")
    if file_id in fname_to_path:
        return fname_to_path[file_id]
    direct = os.path.join(base_dir, file_id)
    if os.path.exists(direct):
        return direct
    for ext in exts:
        cand_name = f"{file_id}{ext}"
        if cand_name in fname_to_path:
            return fname_to_path[cand_name]
        cand_path = os.path.join(base_dir, cand_name)
        if os.path.exists(cand_path):
            return cand_path
    return direct  # fallback; existence will be checked by caller


import shutil

_train_img_root_candidates = (
    train_data_path,
    os.path.join(train_data_path, "train"),
    os.path.join(train_data_path, "train", "train"),
    os.path.join(data_root, "aerial-cactus-identification", "train"),
    os.path.join(data_root, "aerial-cactus-identification", "train", "train"),
    os.path.join(data_root, "train"),
    os.path.join(data_root, "train", "train"),
)
train_data_path = _find_train_image_root(_train_img_root_candidates)

needs_populate = not (
    _split_has_all_classes_images(train_set_path)
    and _split_has_all_classes_images(dev_set_path)
    and _split_has_all_classes_images(test_set_path)
)
if (
    _count_images_in_dir(train_set_path) == 0
    or _count_images_in_dir(dev_set_path) == 0
    or _count_images_in_dir(test_set_path) == 0
):
    needs_populate = True

if needs_populate and _count_images_in_dir(train_data_path) == 0:
    _restore_train_sources_from_splits(
        train_data_path, (train_set_path, dev_set_path, test_set_path)
    )

if needs_populate and _count_images_in_dir(train_data_path) == 0:
    _fallback_candidates = (
        os.path.join(data_root, "train"),
        os.path.join(data_root, "train", "train"),
        os.path.join(data_root, "aerial-cactus-identification", "train"),
        os.path.join(data_root, "aerial-cactus-identification", "train", "train"),
        train_set_path,
        dev_set_path,
        test_set_path,
    )
    train_data_path = _find_train_image_root(_fallback_candidates)

if needs_populate:
    for set_path in (train_set_path, dev_set_path, test_set_path):
        os.makedirs(set_path, exist_ok=True)
        for class_name in CLASS_NAMES:
            os.makedirs(os.path.join(set_path, class_name), exist_ok=True)

    _fname_to_path = _build_filename_to_path_map(train_data_path)

    for path, file_names, labels in (
        (train_set_path, X_train, y_train),
        (dev_set_path, X_dev, y_dev),
        (test_set_path, X_test, y_test),
    ):
        for file_name, label in zip(file_names, labels):
            src = _resolve_src_path(file_name, _fname_to_path, train_data_path)

            dst_name = (
                f"{file_name}"
                if os.path.splitext(str(file_name))[1]
                else f"{file_name}.jpg"
            )
            if os.path.exists(src):
                dst_name = os.path.basename(src)

            dst = os.path.join(path, str(int(label)), dst_name)

            if os.path.exists(src) and (not os.path.exists(dst)):
                shutil.copy2(src, dst)

batch_size = 1024
train_dataset = torchvision.datasets.ImageFolder(
    train_set_path, transform=TrainTransforms()
)
dev_dataset = torchvision.datasets.ImageFolder(dev_set_path, transform=TestTransforms())
test_dataset = torchvision.datasets.ImageFolder(
    test_set_path, transform=TestTransforms()
)
train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
)
dev_dataloader = torch.utils.data.DataLoader(
    dev_dataset, batch_size=batch_size, shuffle=True, num_workers=4
)
test_dataloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=True, num_workers=4
)


## --- ERROR in cell 11, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mFileNotFoundError[0m                         Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/27816949.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m    166[0m [0;34m[0m[0m
[1;32m    167[0m [0mbatch_size[0m [0;34m=[0m [0;36m1024[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 168[0;31m train_dataset = torchvision.datasets.ImageFolder(
[0m[1;32m    169[0m     [0mtrain_set_path[0m[0;34m,[0m [0mtransform[0m[0;34m=[0m[0mTrainTransforms[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    170[0m )

[0;32m/usr/local/lib/python3.11/dist-packages/torchvision/datasets/folder.py[0m in [0;36m__init__[0;34m(self, root, transform, target_transform, loader, is_valid_file, allow_empty)[0m
[1;32m    326[0m         [0mallow_empty[0m[0;34m:[0m [0mbool[0m [0;34m=[0m [0;32mFalse[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m    327[0m     ):
[0;32m--> 328[0;31m         super().__init__(
[0m[1;32m    329[0m             [0mroot[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m    330[0m             [0mloader[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torchvision/datasets/folder.py[0m in [0;36m__init__[0;34m(self, root, loader, extensions, transform, target_transform, is_valid_file, allow_empty)[0m
[1;32m    148[0m         [0msuper[0m[0;34m([0m[0;34m)[0m[0;34m.[0m[0m__init__[0m[0;34m([0m[0mroot[0m[0;34m,[0m [0mtransform[0m[0;34m=[0m[0mtransform[0m[0;34m,[0m [0mtarget_transform[0m[0;34m=[0m[0mtarget_transform[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    149[0m         [0mclasses[0m[0;34m,[0m [0mclass_to_idx[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0mfind_classes[0m[0;34m([0m[0mself[0m[0;34m.[0m[0mroot[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 150[0;31m         samples = self.make_dataset(
[0m[1;32m    151[0m             [0mself[0m[0;34m.[0m[0mroot[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m    152[0m             [0mclass_to_idx[0m[0;34m=[0m[0mclass_to_idx[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torchvision/datasets/folder.py[0m in [0;36mmake_dataset[0;34m(directory, class_to_idx, extensions, is_valid_file, allow_empty)[0m
[1;32m    201[0m             [0;31m# is potentially overridden and thus could have a different logic.[0m[0;34m[0m[0;34m[0m[0m
[1;32m    202[0m             [0;32mraise[0m [0mValueError[0m[0;34m([0m[0;34m"The class_to_idx parameter cannot be None."[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 203[0;31m         return make_dataset(
[0m[1;32m    204[0m             [0mdirectory[0m[0;34m,[0m [0mclass_to_idx[0m[0;34m,[0m [0mextensions[0m[0;34m=[0m[0mextensions[0m[0;34m,[0m [0mis_valid_file[0m[0;34m=[0m[0mis_valid_file[0m[0;34m,[0m [0mallow_empty[0m[0;34m=[0m[0mallow_empty[0m[0;34m[0m[0;34m[0m[0m
[1;32m    205[0m         )

[0;32m/usr/local/lib/python3.11/dist-packages/torchvision/datasets/folder.py[0m in [0;36mmake_dataset[0;34m(directory, class_to_idx, extensions, is_valid_file, allow_empty)[0m
[1;32m    102[0m         [0;32mif[0m [0mextensions[0m [0;32mis[0m [0;32mnot[0m [0;32mNone[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    103[0m             [0mmsg[0m [0;34m+=[0m [0;34mf"Supported extensions are: {extensions if isinstance(extensions, str) else ', '.join(extensions)}"[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 104[0;31m         [0;32mraise[0m [0mFileNotFoundError[0m[0;34m([0m[0mmsg[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    105[0m [0;34m[0m[0m
[1;32m    106[0m     [0;32mreturn[0m [0minstances[0m[0;34m[0m[0;34m[0m[0m

[0;31mFileNotFoundError[0m: Found no valid file for the classes 0, 1. Supported extensions are: .jpg, .jpeg, .png, .ppm, .bmp, .pgm, .tif, .tiff, .webp

## === cell 12
images, targets = next(iter(train_dataloader))
print(targets[:9])
fig = make_image_label_figure(images[:9], targets[:9], CLASS_NAMES)
