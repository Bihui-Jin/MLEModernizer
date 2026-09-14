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

Higher is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 1
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
import math
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from copy import deepcopy

## === cell 2
is_submission = True
data_path = '/kaggle/input/cassava-leaf-disease-classification'
csv_file_name = 'train.csv' if not is_submission else 'sample_submission.csv'
train_batch_size, val_batch_size, sub_batch_size = 4, 8, 8
train_rate = 0.7 # rate for train dataset
num_workers = 2
mean, std = 0.5, 0.5

use_balanced_sample = False
use_class_weight = not use_balanced_sample
mul_weight = torch.tensor([3.0, 1.5, 1.0, 1.0, 4.5])
learning_rate = 0.001
weight_decay = 0.00002
efficient_net_version = 3 # (1.4, 1.2)

train_epoch = (0, 20)
debug = not is_submission

use_pre_trained_weight = True
pre_trained_weight_path = '../input/cassava-leaf-disease-classification-weight/weight.pth'

## === cell 3
class CassavaLeafDiseaseDataset(Dataset):
    "Cassava Leaf Disease"

    def __init__(self, csv_file, root_dir, transform=None, train=True):
        """
        Args:
            csv_file (string): csv file path
            root_dir (string): directory path of exist all image
            transform (callable, optional): Optional transform for sample
        """
        self.cassava_leaf_disease = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.folder = 'train_images' if train else 'test_images'

    def __len__(self):
        return len(self.cassava_leaf_disease)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = '/'.join([self.root_dir,
                             self.folder,
                             self.cassava_leaf_disease.image_id[idx]])
        img = io.imread(img_name)
        if self.transform:
            img = self.transform(img)
        label = self.cassava_leaf_disease.label[idx]
        sample = (img, label)

        return sample
    
def split_dataset(dataset, rate):
    train_length = int(len(dataset) * rate)
    val_length = len(dataset) - train_length
    train_set, val_set = torch.utils.data.random_split(
        dataset, [train_length, val_length])
    return train_set, val_set

## === cell 4
if is_submission:
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize((512, 512)),
                                    transforms.Normalize((mean, mean, mean), (std, std, std))])
else:
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize((512, 512)),
                                    transforms.RandomRotation(30),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomVerticalFlip(),
                                    transforms.Normalize((mean, mean, mean), (std, std, std))])

dataset = CassavaLeafDiseaseDataset(csv_file='/'.join([data_path, csv_file_name]),
                                    root_dir=data_path,
                                    transform=transform,
                                    train=not is_submission)

if is_submission:
    sub_loader = DataLoader(dataset,
                            batch_size=sub_batch_size,
                            shuffle=False,
                            num_workers=num_workers)
else:
    if use_balanced_sample:
        indices_for_each_label = [[] for i in range(5)]

        for i in range(len(dataset)):
            label = dataset.cassava_leaf_disease.label[i]
            indices_for_each_label[label].append(i)

        dataset_for_each_label = [torch.utils.data.dataset.Subset(dataset, indices) for indices in indices_for_each_label]
        dataset_for_each_label[3], _ = torch.utils.data.random_split(dataset_for_each_label[3], [2500, 13158 - 2500])

        train_set_for_each_label = []
        val_set_for_each_label = []
        for sub_dataset in dataset_for_each_label:
            sub_train_set, sub_val_set = split_dataset(sub_dataset, train_rate)
            train_set_for_each_label.append(sub_train_set)
            val_set_for_each_label.append(sub_val_set)

        train_set_for_each_label[0] = torch.utils.data.ConcatDataset([train_set_for_each_label[0],
                                                                      train_set_for_each_label[0]])
        val_set_for_each_label[0] = torch.utils.data.ConcatDataset([val_set_for_each_label[0],
                                                                    val_set_for_each_label[0]])

        train_set = torch.utils.data.ConcatDataset(train_set_for_each_label)
        val_set = torch.utils.data.ConcatDataset(val_set_for_each_label)
    else:
        train_set, val_set = split_dataset(dataset, train_rate)
    
    train_loader = DataLoader(train_set,
                          batch_size=train_batch_size,
                          shuffle=True,
                          num_workers=num_workers)

    val_loader = DataLoader(val_set,
                            batch_size=val_batch_size,
                            shuffle=False,
                            num_workers=num_workers)

## === cell 5
if use_class_weight:
    def class_weight(label_count):
        total = sum(label_count)
        weight = [total / (2 * count) for count in label_count]
        return torch.tensor(weight)

    label_count = [0 for i in range(5)]

    for i in range(len(dataset)):
        label = dataset.cassava_leaf_disease.label[i]
        label_count[label] += 1
    print(label_count)

    loss_weight = class_weight(label_count)
    loss_weight *= mul_weight
    if torch.cuda.is_available:
        loss_weight = loss_weight.cuda()
        print(loss_weight)

## === cell 6
def progress_print(comment, current, total, num_of_print=50, slow=5):
    end = '\r' if current < total else '\n'
    symbol = ['\\', '/', '-']
    done = int((current - 1) / total * num_of_print)
    doing = symbol[(current % (len(symbol) * slow) // slow)] if current < total else ''
    yet = max(0, num_of_print - 1 - done)
    print(comment + ': ' + '#' * done + doing + '*' * yet, end=end)
    
def num_of_correct(outputs, labels):
    _, predicted = torch.max(outputs, 1)
    c = (predicted == labels).squeeze()
    return c.sum().item()


def train(model=None, criterion=None,
          optimizer=None, data_loader=None,
          debug=True, epoch=0, debug_rate=0.01):
    model.train()
    data_loader_length = len(data_loader)
    correct, total = 0, 0
    running_loss = 0.0
    running_debug_rate = debug_rate
    running_debug_step = int(data_loader_length * running_debug_rate)
    for i, data in enumerate(data_loader):
        inputs, labels = data
        if torch.cuda.is_available():
            inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        correct += num_of_correct(outputs, labels)
        total += len(inputs)

        running_loss += loss.item() * len(inputs)
        if debug:
            progress_print('train', i, running_debug_step, slow=3)
        if debug and i == running_debug_step - 1:
            print('epoch: %d (%d/%d),\tloss: %.3f' % (epoch + 1,
                                                     i + 1,
                                                     data_loader_length,
                                                     running_loss / total))
            running_debug_rate += debug_rate
            running_debug_step = min(data_loader_length,
                                     int(data_loader_length * running_debug_rate))
    return correct / total, running_loss / total

def valid(model=None, data_loader=None, debug=True):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            inputs, labels = data
            if torch.cuda.is_available():
                inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)

            correct += num_of_correct(outputs, labels)
            total += len(inputs)
            if debug:
                progress_print('validation', i, len(data_loader), slow=3)
    return correct / total

def get_confusion_matrix(model, data_loader, num_classes):
    model.eval()
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int32)
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            inputs, labels = data
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            predicts = predicted.squeeze()
            for p, l in zip(predicts, labels):
                confusion_matrix[p.item()][l] += 1
            
            progress_print('confusion_matrix', i, len(data_loader), slow=3)
    return confusion_matrix

## === cell 7
class ConvUnit(torch.nn.Module):
    def __init__(self,
                 in_channels:int,
                 out_channels:int,
                 kernel_size:int,
                 padding:int=0,
                 stride:int=1,
                 groups:int=1,
                 batch_norm=True,
                 activation=True):
        super().__init__()
        modules = [torch.nn.Conv2d(in_channels,
                                   out_channels,
                                   kernel_size=kernel_size,
                                   padding=padding,
                                   stride=stride,
                                   groups=groups,
                                   bias=not batch_norm)]
        if batch_norm:
            modules.append(torch.nn.BatchNorm2d(out_channels))
        if activation:
            modules.append(torch.nn.LeakyReLU())
        self.sequence = torch.nn.Sequential(*modules)
        self.out_channels = out_channels
    
    def forward(self, x):
        x = self.sequence(x)
        return x

def DConvUnit(in_channels:int, kernel_size:int, stride:int=1) -> torch.nn.Module:
    padding = kernel_size // 2
    return ConvUnit(in_channels, in_channels, kernel_size, padding, stride, in_channels)

def EConvUnit(in_channels:int, factor:int=6) -> torch.nn.Module:
    return ConvUnit(in_channels, factor * in_channels, 1)

def PConvUnit(in_channels, out_channels, activation=False) -> torch.nn.Module:
    return ConvUnit(in_channels, out_channels, 1, activation=activation)

class DSConvUnit(torch.nn.Module):
    def __init__(self,
                 in_channels:int,
                 out_channels:int,
                 kernel_size:int=3,
                 stride:int=1,
                 activation=True):
        super().__init__()
        modules = [
            DConvUnit(in_channels, kernel_size, stride),
            PConvUnit(in_channels, out_channels, activation)
        ]
        self.sequence = torch.nn.Sequential(*modules)
        self.out_channels = out_channels
    
    def forward(self, x):
        x = self.sequence(x)
        return x

class SEUnit(torch.nn.Module):
    def __init__(self,
                 in_channels:int,
                 se_ratio:float):
        super().__init__()
        se_channels = max(1, int(in_channels * se_ratio))
        self.sequence = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            ConvUnit(in_channels, se_channels, 1, batch_norm=False),
            ConvUnit(se_channels, in_channels, 1, batch_norm=False, activation=False),
            torch.nn.Sigmoid()
        )
        self.out_channels = in_channels
    
    def forward(self, x):
        y = self.sequence(x)
        z = x * y
        return z
        

class BottleneckUnit(torch.nn.Module):
    def __init__(self,
                 in_channels:int,
                 out_channels:int,
                 factor:int=6,
                 kernel_size:int=3,
                 stride:int=1,
                 has_se:bool=True,
                 se_ratio:float=0.25):
        super().__init__()
        modules = []
        self.residual = stride==1 and in_channels == out_channels
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

## === cell 8
def round_filters(filters, coef:float, divisor:int) -> int:
    filters *= coef
    filters = max(divisor, int(filters + divisor / 2) // divisor * divisor)
    return filters

def round_repeats(repeats, coef):
    repeats = int(math.ceil(repeats * coef)) if repeats > 0 else -repeats
    return repeats

class BlockArgs():
    def __init__(self,
                 module:torch.nn.Module,
                 filters:int,
                 num_repeats:int=1,
                 **kwargs):
        self.module = module
        self.filters = filters
        self.num_repeats = num_repeats
        self.kwargs = kwargs
    
    def __call__(self,
                 in_channels:int,
                 width_coef:float=1.0,
                 depth_coef:float=1.0,
                 divisor:int=8) -> torch.nn.Module:
        out_channels = round_filters(self.filters, width_coef, divisor)
        num_repeats = round_repeats(self.num_repeats, depth_coef)
        modules = []
        for _ in range(num_repeats):
            modules.append(self.module(in_channels, out_channels, **self.kwargs))
            if 'stride' in self.kwargs:
                del self.kwargs['stride']
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
    BlockArgs(BottleneckUnit, 320, 1)
]

EFFICIENT_NET_COMPOUND_COEF = [
    (1.0, 1.0),
    (1.1, 1.0),
    (1.2, 1.1),
    (1.4, 1.2),
    (1.8, 1.4),
    (2.2, 1.6),
    (2.6, 1.8),
    (3.1, 2.0)
]

class EfficientNet(torch.nn.Module):
    def __init__(self,
                 in_channels:int=3,
                 num_of_class:int=1000,
                 depth_coef:float=1.0,
                 width_coef:float=1.0,
                 hidden_channels:int=1280,
                 block_args=EFFICIENT_NET_BLOCK_ARGS):
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

## === cell 9
model = EfficientNet(3, 5, *EFFICIENT_NET_COMPOUND_COEF[efficient_net_version])
if use_pre_trained_weight:
    pre_trained_weight = torch.load(pre_trained_weight_path)
    model.load_state_dict(pre_trained_weight)
if torch.cuda.is_available():
    model = model.cuda()

## === cell 11
if not is_submission:
    if use_class_weight:
        criterion = torch.nn.CrossEntropyLoss(weight=loss_weight)
    elif mul_weight is not None:
        if torch.cuda.is_available():
            mul_weight = mul_weight.cuda()
        criterion = torch.nn.CrossEntropyLoss(weight=mul_weight)
    else:
        criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=learning_rate,
                                 weight_decay=weight_decay)
    train_accuracy = []
    train_losses = []
    val_accuracy= []
    best_accuracy = 0.0 if len(val_accuracy) == 0 else min(val_accuracy)
    best_model = deepcopy(model.state_dict())

## === cell 12
if not is_submission:
    for epoch in range(*train_epoch):
        train_acc, train_loss = train(model, criterion, optimizer, train_loader, epoch=epoch, debug=debug, debug_rate=0.1)
        val_acc = valid(model, val_loader, debug=debug)
        train_accuracy.append(train_acc)
        train_losses.append(train_loss)
        val_accuracy.append(val_acc)
        if best_accuracy < val_acc:
            best_model = deepcopy(model.state_dict())
            best_accuracy = val_acc
            print('Best model is updated.')
        print(train_acc, val_acc)
    torch.save(best_model, 'weight.pth')

## === cell 13
if not is_submission:
    print(train_accuracy)
    print(train_losses)
    print(val_accuracy)

## === cell 14
if not is_submission:
    plt.plot(train_accuracy, label='train')
    plt.plot(train_losses, label='loss')
    plt.plot(val_accuracy, label='validation')
    plt.legend(loc='upper left')
    plt.show()

## === cell 16
def show_confusion_matrix(confusion_matrix):
    row_sums = confusion_matrix.sum(axis=1)
    normalized_confusion_matrix = confusion_matrix / row_sums[:, np.newaxis]
    diagonal_confusion_matrix = deepcopy(confusion_matrix)
    np.fill_diagonal(diagonal_confusion_matrix, 0)
    row_sums = diagonal_confusion_matrix.sum(axis=1)
    normalized_diagonal_confusion_matrix = diagonal_confusion_matrix / row_sums[:, np.newaxis]
    plt.figure(figsize=(2,1))
    plt.matshow(normalized_confusion_matrix, cmap='gray')
    plt.matshow(normalized_diagonal_confusion_matrix, cmap='gray')
    plt.show()
    print(confusion_matrix)
    print(normalized_confusion_matrix)

## === cell 17
if not is_submission:
    confusion_matrix = get_confusion_matrix(model, val_loader, 5)
    show_confusion_matrix(confusion_matrix)

## === cell 19
if is_submission:
    model.eval()
    submission = dataset.cassava_leaf_disease.copy()
    predicts = []
    with torch.no_grad():
        for i, data in enumerate(sub_loader):
            inputs, labels = data
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            predicts.extend(predicted.cpu().numpy())
            
    submission['label'] = predicts
    submission.to_csv('submission.csv', index=False)
