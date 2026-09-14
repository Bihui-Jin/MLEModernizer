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
Create a classifier to predict whether an image contains a cactus.

## Metric
Area under the ROC curve.

## Submission Format
For each ID in the test set, you must predict a probability for the `has_cactus` variable. The file should contain a header and have the following format:

```
id,has_cactus
000940378805c44108d287872b2f04ce.jpg,0.5
0017242f54ececa4512b4d7937d1e21e.jpg,0.5
001ee6d8564003107853118ab87df407.jpg,0.5
etc.
```

## Dataset
This dataset contains a large number of 32 x 32 thumbnail images containing aerial photos of a cactus. The file name of an image corresponds to its `id`.

- **train/** - the training set images
- **test/** - the test set images (you must predict the labels of these)
- **train.csv** - the training set labels, indicates whether the image has a cactus (`has_cactus = 1`)
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.8

# 3. Installed packages

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

# 4. Data file paths

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

# 5. Target score

0.9833

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

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
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from tqdm.autonotebook import tqdm


METRICS = {
    'accuracy': {
        'f': balanced_accuracy_score,
        'args': {}
    },
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
    for i in range(n*n):
        image, label = images[i], (0 if labels is None else labels[i])
        image = F.normalize(image, mean=mean, std=std)
        image = image.permute(1, 2, 0)
        image = torch.squeeze(image)
        image = (image * 255).int()
        plt.subplot(n, n, i + 1, title='NA' if class_names is None else class_names[label])
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image, cmap='gray' if channels == 1 else None)
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
        transforms_list.extend([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])
        super(Transforms, self).__init__(transforms_list)


class TrainTransforms(transforms.Compose):

    def __init__(self, in_channels=1, out_channels=1, size=(32, 32), random_crop=False, random_affine=False,
                 horizontal_flip=False, color_jitter=False, random_erasing=False):
        if out_channels not in (3, 1) or in_channels not in (3, 1):
            raise ValueError("Images must have 1 or 3 channels")
        mean = NORM_MEAN if out_channels == 3 else [sum(NORM_MEAN) / 3]
        std = NORM_STD if out_channels == 3 else [sum(NORM_STD) / 3]
        transforms_list = []
        if in_channels != out_channels:
            transforms_list.append(transforms.Grayscale(out_channels))
        if random_affine:
            transforms_list.append(transforms.RandomAffine(degrees=10.0, translate=(0.25, 0.25),
                                                           shear=(-10, 10, -10, 10)))
        if random_crop:
            transforms_list.append(transforms.RandomResizedCrop(size, scale=(0.9, 1.1), ratio=(0.75, 1.33)))
        else:
            transforms_list.append(transforms.Resize(size))
        if horizontal_flip:
            transforms_list.append(transforms.RandomHorizontalFlip(p=0.5))
        if color_jitter:
            transforms_list.append(transforms.ColorJitter(brightness=(0.75, 1.5), contrast=(0.75, 1.5),
                                                          saturation=(0.75, 1.5), hue=(-0.1, 0.1)))
        transforms_list.append(transforms.ToTensor())
        if random_erasing:
            transforms_list.append(transforms.RandomErasing(p=0.2, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0))
        transforms_list.append(transforms.Normalize(mean=mean, std=std))
        super(TrainTransforms, self).__init__(transforms_list)


class TrainerProgressBar(tqdm):

    def __init__(self, desc=None, total=10, unit='it', position=None):
        super(TrainerProgressBar, self).__init__(
            desc=desc, total=total, leave=True, unit=unit, position=position, dynamic_ncols=True
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

    def __init__(self, log_dir=None, comment='', purge_step=None, max_queue=10,
                 flush_secs=120, filename_suffix='', class_names=None, testloader=None, device=None):
        super(TensorBoardLogger, self).__init__(log_dir=log_dir, comment=comment, purge_step=purge_step,
                                                max_queue=max_queue, flush_secs=flush_secs,
                                                filename_suffix=filename_suffix)
        self.class_names = class_names
        self.testloader = testloader
        self.device = torch.device(device) if device else torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def epoch_callback(self, net, epoch, lr, train_loss, val_loss, train_metrics_dict, val_metrics_dict):
        super(TensorBoardLogger, self).add_scalar(f"epoch/lr", lr, epoch)
        super(TensorBoardLogger, self).add_scalars(f"epoch/loss", {'train': train_loss, 'val': val_loss}, epoch)
        for metric_name in train_metrics_dict:
            super(TensorBoardLogger, self).add_scalars(f"epoch/{metric_name}",
                                                       {'train': train_metrics_dict[metric_name],
                                                        'val': val_metrics_dict[metric_name]}, epoch)
        if self.testloader:
            inputs, targets = next(iter(self.testloader))
            _inputs = inputs.to(self.device)
            predictions = net.forward(_inputs).argmax(dim=1).data.cpu()
            start = random.randrange(0, len(targets)-9)
            stop = start + 9
            super(TensorBoardLogger, self).add_figure(
                'examples/real', make_image_label_figure(inputs[start:stop], targets[start:stop], self.class_names)
            )
            super(TensorBoardLogger, self).add_figure(
                'examples/predicted', make_image_label_figure(inputs[start:stop], predictions[start:stop],
                                                              self.class_names)
            )

    def batch_callback(self, train, epoch, batch, batches, loss, metrics_dict):
        section = 'train_batch' if train else 'val_batch'
        super(TensorBoardLogger, self).add_scalar(f"{section}/loss", loss, batch + epoch * batches)
        for metric_name, metric_value in metrics_dict.items():
            if isinstance(metric_value, dict):
                super(TensorBoardLogger, self).add_scalars(f"{section}/{metric_name}", metric_value,
                                                           batch + epoch * batches)
            else:
                super(TensorBoardLogger, self).add_scalar(f"{section}/{metric_name}", metric_value,
                                                          batch + epoch * batches)


class PyTorchTrainer(object):

    def __init__(self, device=None, metrics=None, epoch_callback=None, batch_callback=None):
        self.device = torch.device(device) if device else torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.metrics = metrics or METRICS
        self.epoch_callback = epoch_callback
        self.batch_callback = batch_callback

        self.train_pb = None
        self.epoch_train_pb = None
        self.epoch_val_pb = None

    def train(self, model, optimizer, loss_criterion, train_data_loader, val_data_loader, scheduler=None,
              epochs=10):
        print(f"=============================== Training NN ===============================")
        print(f"== Epochs:              {epochs:6d}")
        print(f"== Train batch size:    {train_data_loader.batch_size:6d}")
        print(f"== Train batches:       {len(train_data_loader):6d}")
        print(f"== Validate batch size: {val_data_loader.batch_size:6d}")
        print(f"== Validate batches:    {len(val_data_loader):6d}")
        print(f"===========================================================================")
        self.train_pb = TrainerProgressBar(desc=f'== Epoch {1}', total=epochs, unit='epoch', position=0)
        self.epoch_train_pb = TrainerProgressBar(desc=f'== Train {1}', total=len(train_data_loader),
                                                 unit='batch', position=1)
        self.epoch_val_pb = TrainerProgressBar(desc=f'== Val {1}', total=len(val_data_loader), unit='batch',
                                               position=2)
        self.train_pb.reset(total=epochs)
        self.epoch_train_pb.reset(total=len(train_data_loader))
        self.epoch_val_pb.reset(total=len(val_data_loader))
        for epoch in range(epochs):
            train_loss, train_metrics_dict = self.forward_batches(model, optimizer, loss_criterion,
                                                                  train_data_loader, epoch, train=True)
            val_loss, val_metrics_dict = self.forward_batches(model, optimizer, loss_criterion,
                                                              val_data_loader, epoch, train=False)
            if scheduler:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(train_loss, epoch=epoch)
                else:
                    scheduler.step(epoch=epoch)
            lr = optimizer.param_groups[0]['lr']
            if self.epoch_callback:
                self.epoch_callback(model, epoch, lr, train_loss, val_loss, train_metrics_dict, val_metrics_dict)
            metrics_dict = {'lr': lr}
            for metric_name in train_metrics_dict:
                metrics_dict[f"train_{metric_name}"] = train_metrics_dict[metric_name]
                metrics_dict[f"val_{metric_name}"] = val_metrics_dict[metric_name]
            metrics_dict.update({'train_loss': train_loss, 'val_loss': val_loss})
            self.train_pb.update(desc=f'== Epoch {epoch+1}', ordered_dict=metrics_dict)
        self.train_pb.close()
        self.epoch_train_pb.close()
        self.epoch_val_pb.close()
        print(f"===========================================================================")

    def forward_batches(self, model, optimizer, loss_criterion, data_loader, epoch, train=True):
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
            loss_value, predictions, targets = self.forward_batch(model, optimizer, loss_criterion, data,
                                                                  train=train)
            metrics_dict = self.metrics_dict(predictions, targets)
            avg_loss_value += loss_value
            if avg_metrics_dict is None:
                avg_metrics_dict = metrics_dict.copy()
            else:
                for metric_name in avg_metrics_dict:
                    avg_metrics_dict[metric_name] += metrics_dict[metric_name]
            if self.batch_callback:
                self.batch_callback(train, epoch, batch_i, batches, loss_value, metrics_dict)
            metrics_dict.update({'loss': avg_loss_value/batch_i})
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
            metric_value = self.metrics[metric_name]['f'](predictions, targets, **self.metrics[metric_name]['args'])
            d[metric_name] = metric_value

        return d


## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/tensorboard/compat/__init__.py in tf()
     41     try:
---> 42         from tensorboard.compat import notf  # noqa: F401
     43     except ImportError:

ImportError: cannot import name 'notf' from 'tensorboard.compat' (/usr/local/lib/python3.11/dist-packages/tensorboard/compat/__init__.py)

During handling of the above exception, another exception occurred:

AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

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
batch_size = 1024
train_dataset = torchvision.datasets.ImageFolder(train_set_path, transform=TrainTransforms())
dev_dataset = torchvision.datasets.ImageFolder(dev_set_path, transform=TestTransforms())
test_dataset = torchvision.datasets.ImageFolder(test_set_path, transform=TestTransforms())
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
dev_dataloader = torch.utils.data.DataLoader(dev_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)


## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/3881876403.py in <cell line: 0>()
      3 batch_size = 1024
      4 # Datasets
----> 5 train_dataset = torchvision.datasets.ImageFolder(train_set_path, transform=TrainTransforms())
      6 dev_dataset = torchvision.datasets.ImageFolder(dev_set_path, transform=TestTransforms())
      7 test_dataset = torchvision.datasets.ImageFolder(test_set_path, transform=TestTransforms())

/usr/local/lib/python3.11/dist-packages/torchvision/datasets/folder.py in __init__(self, root, transform, target_transform, loader, is_valid_file, allow_empty)
    326         allow_empty: bool = False,
    327     ):
--> 328         super().__init__(
    329             root,
    330             loader,

/usr/local/lib/python3.11/dist-packages/torchvision/datasets/folder.py in __init__(self, root, loader, extensions, transform, target_transform, is_valid_file, allow_empty)
    147     ) -> None:
    148         super().__init__(root, transform=transform, target_transform=target_transform)
--> 149         classes, class_to_idx = self.find_classes(self.root)
    150         samples = self.make_dataset(
    151             self.root,

/usr/local/lib/python3.11/dist-packages/torchvision/datasets/folder.py in find_classes(self, directory)
    232             (Tuple[List[str], Dict[str, int]]): List of all classes and dictionary mapping each class to an index.
    233         """
--> 234         return find_classes(directory)
    235 
    236     def __getitem__(self, index: int) -> Tuple[Any, Any]:

/usr/local/lib/python3.11/dist-packages/torchvision/datasets/folder.py in find_classes(directory)
     39     See :class:`DatasetFolder` for details.
     40     """
---> 41     classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
     42     if not classes:
     43         raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

FileNotFoundError: [Errno 2] No such file or directory: './data/train/train'

## === cell 12
images, targets = next(iter(train_dataloader))
print(targets[:9])
fig = make_image_label_figure(images[:9], targets[:9], CLASS_NAMES)


## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3696156067.py in <cell line: 0>()
      1 # Show train batch
----> 2 images, targets = next(iter(train_dataloader))
      3 print(targets[:9])
      4 fig = make_image_label_figure(images[:9], targets[:9], CLASS_NAMES)

NameError: name 'train_dataloader' is not defined

## === cell 13
images, targets = next(iter(dev_dataloader))
print(targets[:9])
fig = make_image_label_figure(images[:9], targets[:9], CLASS_NAMES)


## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/607135720.py in <cell line: 0>()
      1 # Show dev batch
----> 2 images, targets = next(iter(dev_dataloader))
      3 print(targets[:9])
      4 fig = make_image_label_figure(images[:9], targets[:9], CLASS_NAMES)

NameError: name 'dev_dataloader' is not defined

## === cell 14
images, targets = next(iter(test_dataloader))
print(targets[:9])
fig = make_image_label_figure(images[:9], targets[:9], CLASS_NAMES)


## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1082004999.py in <cell line: 0>()
      1 # Show test batch
----> 2 images, targets = next(iter(test_dataloader))
      3 print(targets[:9])
      4 fig = make_image_label_figure(images[:9], targets[:9], CLASS_NAMES)

NameError: name 'test_dataloader' is not defined

## === cell 15
class Conv2dBNReLU(nn.Sequential):
    '''Convolution2d + BatchNormalization2d + ReLU Activation'''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=True):
        super(Conv2dBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class DOLinearBNReLU(nn.Sequential):
    '''Droupout + Linear'''
    def __init__(self, in_features, out_features, bias=True):
        super(DOLinearBNReLU, self).__init__(
            nn.Dropout(0.2),
            nn.Linear(in_features, out_features, bias=bias),
            nn.BatchNorm1d(out_features),
            nn.ReLU(inplace=True)
        )


class DOLinear(nn.Sequential):
    '''Droupout + Linear'''
    def __init__(self, in_features, out_features, bias=True):
        super(DOLinear, self).__init__(
            nn.Dropout(0.2),
            nn.Linear(in_features, out_features, bias=bias)
        )


class Net(nn.Module):

    def __init__(self, in_channels=1, classes=10):
        super(Net, self).__init__()

        self.feature_extractor = nn.Sequential(
            Conv2dBNReLU(in_channels, 32, kernel_size=3, padding=1, bias=False),
            
            Conv2dBNReLU(32, 32, kernel_size=3, padding=1, bias=False),
            Conv2dBNReLU(32, 64, kernel_size=3, padding=1, bias=False),

            nn.MaxPool2d(2),

            Conv2dBNReLU(64, 64, kernel_size=3, padding=1, bias=False),
            Conv2dBNReLU(64, 64, kernel_size=3, padding=1, bias=False),
            Conv2dBNReLU(64, 128, kernel_size=3, padding=1, bias=False),

            nn.MaxPool2d(2),

            Conv2dBNReLU(128, 128, kernel_size=3, padding=1, bias=False),
            Conv2dBNReLU(128, 128, kernel_size=3, padding=1, bias=False),
            Conv2dBNReLU(128, 256, kernel_size=3, padding=1, bias=False),

            nn.MaxPool2d(2),

            Conv2dBNReLU(256, 256, kernel_size=3, padding=1, bias=False),
            Conv2dBNReLU(256, 256, kernel_size=3, padding=1, bias=False),
            Conv2dBNReLU(256, 256, kernel_size=3, padding=1, bias=False),
        )
        self.pool = nn.AdaptiveAvgPool2d((2, 2))
        self.classifier = DOLinear(2 * 2 * 256, classes)
        self.sm = nn.Softmax()

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.pool(x)
        x = x.view(-1, 2 * 2 * 256)
        x = self.classifier(x)
        return x

    def predict(self, x):
        x = self.forward(x)
        x = self.sm(x)
        return x


## === cell 16
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

net = Net(in_channels=3, classes=2)
net = net.to(torch.device(device))

criterion = nn.CrossEntropyLoss(weight=torch.tensor([no_cactus_weight, has_cactus_weight], dtype=torch.float32).to(torch.device(device)))
optimizer = optim.Adam(net.parameters(), lr=0.0001, weight_decay=0.0001)
scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

metrics = {
    'accuracy': {
        'f': accuracy_score,
        'args': {}
    },
    'balanced_accuracy': {
        'f': balanced_accuracy_score,
        'args': {}
    },
    'f1': {
        'f': f1_score,
        'args': {'average': 'weighted'}
    }
}

trainer = PyTorchTrainer(device=device, metrics=metrics)
trainer.train(net, optimizer, criterion, train_dataloader, dev_dataloader, scheduler=scheduler, epochs=50)


## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/336144768.py in <cell line: 0>()
     25 
     26 trainer = PyTorchTrainer(device=device, metrics=metrics)
---> 27 trainer.train(net, optimizer, criterion, train_dataloader, dev_dataloader, scheduler=scheduler, epochs=50)

NameError: name 'train_dataloader' is not defined

## === cell 17
net.eval()
for parameter in net.parameters():
  parameter.requires_grad = False


## === cell 18
def get_metrics_dict(predictions, targets):
  d = {}
  for metric_name in metrics:
    metric_value = metrics[metric_name]['f'](predictions, targets, **metrics[metric_name]['args'])
    d[metric_name] = metric_value
  return d

avg_metrics_dict = None
batches = len(test_dataloader)
for batch_data in test_dataloader:
  inputs, targets = batch_data
  _inputs = inputs.to(torch.device(device))
  _outputs = net.forward(_inputs)
  predictions = _outputs.argmax(dim=1).data.cpu()
  metrics_dict = get_metrics_dict(predictions, targets)
  if avg_metrics_dict is None:
    avg_metrics_dict = metrics_dict.copy()
  else:
    for metric_name in avg_metrics_dict:
      avg_metrics_dict[metric_name] += metrics_dict[metric_name]

for metric_name in avg_metrics_dict:
  avg_metrics_dict[metric_name] /= batches
  print(metric_name, avg_metrics_dict[metric_name])


## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1866872371.py in <cell line: 0>()
      8 # Forward test set
      9 avg_metrics_dict = None
---> 10 batches = len(test_dataloader)
     11 for batch_data in test_dataloader:
     12   # Get Inputs and put them to device

NameError: name 'test_dataloader' is not defined

## === cell 19
class SubmissionDataset(torch.utils.data.Dataset):
  
  def __init__(self, root, transform=None):
    super(SubmissionDataset, self).__init__()
    self.transform = transform
    self.image_filenames = []
    self.image_paths = []
    for dirname, _, filenames in os.walk(root):
      for filename in filenames:
        self.image_filenames.append(filename)
        self.image_paths.append(os.path.join(root, filename))
  
  def __len__(self):
    return len(self.image_filenames)
      
  def __getitem__(self, index):
    image = Image.open(self.image_paths[index])
    return image if self.transform is None else self.transform(image)


## === cell 20
submission_dataset = SubmissionDataset(test_data_path, transform=TestTransforms())
submission_dataloader = torch.utils.data.DataLoader(submission_dataset, batch_size=batch_size, shuffle=False, num_workers=4)


## === cell 21
images = next(iter(submission_dataloader))
fig = make_image_label_figure(images[:9])


## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
StopIteration                             Traceback (most recent call last)
/tmp/ipykernel_11/3878271972.py in <cell line: 0>()
      1 # Show submission batch
----> 2 images = next(iter(submission_dataloader))
      3 fig = make_image_label_figure(images[:9])

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in __next__(self)
    706                 # TODO(https://github.com/pytorch/pytorch/issues/76750)
    707                 self._reset()  # type: ignore[call-arg]
--> 708             data = self._next_data()
    709             self._num_yielded += 1
    710             if (

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _next_data(self)
   1445                 if not self._persistent_workers:
   1446                     self._shutdown_workers()
-> 1447                 raise StopIteration
   1448 
   1449             # Now `self._rcvd_idx` is the batch index we want to fetch

StopIteration: 

## === cell 22
test_predictions = None
for batch_data in submission_dataloader:
  inputs = batch_data
  _inputs = inputs.to(torch.device(device))
  _outputs = net.forward(_inputs)
  predictions = _outputs.argmax(dim=1).data
  test_predictions = predictions if test_predictions is None else torch.cat((test_predictions, predictions))
test_predictions = test_predictions.cpu().numpy()


## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/3562378272.py in <cell line: 0>()
     10   predictions = _outputs.argmax(dim=1).data
     11   test_predictions = predictions if test_predictions is None else torch.cat((test_predictions, predictions))
---> 12 test_predictions = test_predictions.cpu().numpy()

AttributeError: 'NoneType' object has no attribute 'cpu'

## === cell 23
submission_df = pd.DataFrame(np.c_[np.array(submission_dataset.image_filenames)[:,None], test_predictions], columns=['id', 'has_cactus'])
submission_df.head()


## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3933788243.py in <cell line: 0>()
      1 # Write submission to pd.DataFrame
----> 2 submission_df = pd.DataFrame(np.c_[np.array(submission_dataset.image_filenames)[:,None], test_predictions], columns=['id', 'has_cactus'])
      3 submission_df.head()

/usr/local/lib/python3.11/dist-packages/numpy/lib/index_tricks.py in __getitem__(self, key)
    416                           ndmin=ndmin, dtype=final_dtype) for obj in objs]
    417 
--> 418         res = self.concatenate(tuple(objs), axis=axis)
    419 
    420         if matrix:

ValueError: all the input array dimensions except for the concatenation axis must match exactly, but along dimension 0, the array at index 0 has size 0 and the array at index 1 has size 1

## === cell 24
submission_df.to_csv('submission.csv', index=False)


## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1052958295.py in <cell line: 0>()
      1 # Write submission DataFrame to csv
----> 2 submission_df.to_csv('submission.csv', index=False)

NameError: name 'submission_df' is not defined

## === cell 25
!rm -rf data
