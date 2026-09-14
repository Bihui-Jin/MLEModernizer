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

geopandas==0.14.4
imbalanced-learn==0.13.0
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

0.7097310365669387

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import pandas as pd

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

import PIL
from PIL import Image
import matplotlib.pyplot as plt

from torch.autograd import Variable


from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
import torchvision.models as models


from sklearn import metrics, model_selection, preprocessing
from imblearn.over_sampling import SMOTE

from tqdm.notebook import tqdm

from sklearn.model_selection import StratifiedKFold
import time
import datetime



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_55/2253081618.py in <cell line: 0>()
     22 
     23 from sklearn import metrics, model_selection, preprocessing
---> 24 from imblearn.over_sampling import SMOTE
     25 
     26 from tqdm.notebook import tqdm

/usr/local/lib/python3.11/dist-packages/imblearn/__init__.py in <module>
     50     # process, as it may not be compiled yet
     51 else:
---> 52     from . import (
     53         combine,
     54         ensemble,

/usr/local/lib/python3.11/dist-packages/imblearn/combine/__init__.py in <module>
      3 """
      4 
----> 5 from ._smote_enn import SMOTEENN
      6 from ._smote_tomek import SMOTETomek
      7 

/usr/local/lib/python3.11/dist-packages/imblearn/combine/_smote_enn.py in <module>
     10 from sklearn.utils import check_X_y
     11 
---> 12 from ..base import BaseSampler
     13 from ..over_sampling import SMOTE
     14 from ..over_sampling.base import BaseOverSampler

/usr/local/lib/python3.11/dist-packages/imblearn/base.py in <module>
     10 from sklearn.base import BaseEstimator, OneToOneFeatureMixin
     11 from sklearn.preprocessing import label_binarize
---> 12 from sklearn.utils._metadata_requests import METHODS
     13 from sklearn.utils.multiclass import check_classification_targets
     14 

ModuleNotFoundError: No module named 'sklearn.utils._metadata_requests'

## === cell 1
torch.cuda.is_available()



## === cell 2

dfx = pd.read_csv("../input/cassava-leaf-disease-classification/train.csv")

image_path = "../input/cassava-leaf-disease-classification/train_images/"
data_image_paths = [os.path.join(image_path, x) for x in dfx.image_id.values]
data_targets = dfx.label.values
data_ids = dfx.image_id.values



## === cell 3
"""torch module dataset"""


class CassavaDataset(Dataset):  # Override torch.utils.data.Dataset
    def __init__(self, data, targets, dataset, transform=None):
        """
        Args:
          csv_file    (string): path of csv file
          dir         (string): path of images
          transoform  (callable, optional): Optional transform - IDK
        """
        self.files = data
        self.targets = targets
        self.classes = list(set(targets))
        self.transform = transform
        self.dataset = dataset

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        name = self.files[idx]
        img_name = os.path.join(name)
        image = Image.open(img_name)
        """
    Need to transform image size
    ------------------------------------------------------------"""
        input_size = 384
        imagenet_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        if self.dataset == "train":
            transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop((input_size, input_size)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.ToTensor(),
                    transforms.Normalize(*imagenet_stats),
                ]
            )
            image = transform(image)
        elif self.dataset == "test":
            transform = transforms.Compose(
                [
                    transforms.Resize((input_size, input_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(*imagenet_stats),
                ]
            )
            image = transform(image)

        """---------------------------------------------------------"""

        label = self.targets[idx]

        if self.transform:  ## IDK
            sample = self.transform(sample)

        return image, label




## === cell 4
def getkFoldLoader(fold, dfx):
    train_targets = dfx[dfx["fold"] != fold].reset_index(drop=True)
    valid_targets = dfx[dfx["fold"] == fold].reset_index(drop=True)

    train_image_paths = [
        os.path.join(image_path, x) for x in train_targets.image_id.values
    ]
    valid_image_paths = [
        os.path.join(image_path, x) for x in valid_targets.image_id.values
    ]

    train_targets = train_targets.label.values
    valid_targets = valid_targets.label.values

    cassava_train = CassavaDataset(train_image_paths, train_targets, "train")
    cassava_test = CassavaDataset(valid_image_paths, valid_targets, "test")

    train_loader = DataLoader(
        cassava_train, batch_size=batch_size, shuffle=True, num_workers=2
    )
    test_loader = DataLoader(
        cassava_test, batch_size=batch_size, shuffle=True, num_workers=2
    )

    return train_loader, test_loader




## === cell 5
"""Dataset Initialization"""




## === cell 6
batch_size = 16




## === cell 7
def denormalize(images, means, stds):
    if len(images.shape) == 3:
        images = images.unsqueeze(0)
    means = torch.tensor(means).reshape(1, 3, 1, 1)
    stds = torch.tensor(stds).reshape(1, 3, 1, 1)
    return images * stds + means


imagenet_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def show_image(img_tensor, label):
    print("Label:", cassava_data.classes[label], "(" + str(label) + ")")
    img_tensor = denormalize(img_tensor, *imagenet_stats)[0].permute((1, 2, 0))
    img_tensor = img_tensor[0].permute((1, 2, 0))
    plt.imshow(img_tensor)


def imshow(img, label):
    npimg = img.numpy()
    print("Label:", cassava_data.classes[label], "(" + str(label) + ")")
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()




## === cell 8
class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:k].reshape(-1).float().sum(0) * 100.0 / batch_size for k in topk]




## === cell 9
"""optimizer setting"""
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(resnet.parameters(), lr=0.01, momentum=0.9)




## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/686889312.py in <cell line: 0>()
      4 
      5 criterion = nn.CrossEntropyLoss()
----> 6 optimizer = optim.SGD(resnet.parameters(), lr=0.01, momentum=0.9)
      7 
      8 

NameError: name 'resnet' is not defined

## === cell 10
def reset_weights(m):
    """
    Try resetting model weights to avoid
    weight leakage.
    """
    for layer in m.children():
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()




## === cell 11
def train_epoch(model, loader, device, loss_func, optimizer):
    model.train()
    summary_loss = AverageMeter()  # track running loss
    summary_acc = AverageMeter()  # track running accuracy
    start = time.time()  # track time

    n = len(loader)

    for batch in tqdm(loader):

        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)

        out = model(images)  # Generate predictions
        loss = loss_func(out, labels)  # Calculate loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        with torch.no_grad():
            acc = accuracy(out, labels)[0]

        summary_loss.update(loss.detach().item(), batch_size)
        summary_acc.update(acc.detach().item(), batch_size)

    train_time = str(datetime.timedelta(seconds=time.time() - start))
    print(
        "Train loss: {:.5f} - Train acc: {:.2f}% - time: {}".format(
            summary_loss.avg, summary_acc.avg, train_time
        )
    )
    return summary_loss, summary_acc




## === cell 12
def evaluate_epoch(model, loader, device, loss_func):
    model.eval()
    summary_loss = AverageMeter()  # track running loss
    summary_acc = AverageMeter()  # track running accuracy
    start = time.time()  # track time

    n = len(loader)

    for batch in tqdm(loader):
        with torch.no_grad():
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)

            out = model(images)  # Generate predictions
            loss = loss_func(out, labels)  # Calculate loss

            acc = accuracy(out, labels)[0]

            summary_loss.update(loss.detach().item(), batch_size)
            summary_acc.update(acc.detach().item(), batch_size)

    eval_time = str(datetime.timedelta(seconds=time.time() - start))
    print(
        "Val loss: {:.5f} - Val acc: {:.2f}% - time: {}".format(
            summary_loss.avg, summary_acc.avg, eval_time
        )
    )
    return summary_loss, summary_acc




## === cell 13
PATH = "../input/resnext3/resnext_epoch7.pth"

resnet = models.resnext50_32x4d()

num_ftrs = resnet.fc.in_features
resnet.fc = nn.Linear(num_ftrs, 5)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
resnet.to(device)

resnet.load_state_dict(torch.load(PATH))

resnet.eval()



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/870230272.py in <cell line: 0>()
      8 resnet.to(device)
      9 
---> 10 resnet.load_state_dict(torch.load(PATH))
     11 
     12 resnet.eval()

/usr/local/lib/python3.11/dist-packages/torch/serialization.py in load(f, map_location, pickle_module, weights_only, mmap, **pickle_load_args)
   1423         pickle_load_args["encoding"] = "utf-8"
   1424 
-> 1425     with _open_file_like(f, "rb") as opened_file:
   1426         if _is_zipfile(opened_file):
   1427             # The zipfile reader is going to advance the current file position.

/usr/local/lib/python3.11/dist-packages/torch/serialization.py in _open_file_like(name_or_buffer, mode)
    749 def _open_file_like(name_or_buffer, mode):
    750     if _is_path(name_or_buffer):
--> 751         return _open_file(name_or_buffer, mode)
    752     else:
    753         if "w" in mode:

/usr/local/lib/python3.11/dist-packages/torch/serialization.py in __init__(self, name, mode)
    730 class _open_file(_opener):
    731     def __init__(self, name, mode):
--> 732         super().__init__(open(name, mode))
    733 
    734     def __exit__(self, *args):

FileNotFoundError: [Errno 2] No such file or directory: '../input/resnext3/resnext_epoch7.pth'

## === cell 14
submission_df = pd.read_csv(
    "../input/cassava-leaf-disease-classification/sample_submission.csv"
)
submission_df.head()



## === cell 15
"""TTA"""
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms

input_size = 384
stats = ([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])

transform = transforms.Compose(
    [
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(*stats),
    ]
)

trans1 = transforms.Compose(
    [
        transforms.Resize((input_size, input_size)),
        transforms.Pad(8, padding_mode="reflect"),
        transforms.ToTensor(),
        transforms.Normalize(*stats),
    ]
)

trans2 = transforms.Compose(
    [
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomResizedCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(*stats),
    ]
)

trans3 = transforms.Compose(
    [
        transforms.Resize((input_size, input_size)),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomResizedCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(*stats),
    ]
)

trans4 = transforms.Compose(
    [
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomResizedCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(*stats),
    ]
)
transs = [transform, trans1, trans2, trans3, trans4]



## === cell 16
"""Inference"""

from PIL import Image

test_path = "../input/cassava-leaf-disease-classification/test_images/"
test_images = os.listdir(test_path)

y_preds = []

for img_name in tqdm(test_images):
    image = Image.open(os.path.join(test_path, img_name)).convert("RGB")

    outs = torch.zeros((len(transs), 5), device="cpu")
    for k, trans in enumerate(transs):
        img_tensor = trans(image)  # (C, H, W)
        img_tensor = img_tensor.unsqueeze(0).to(
            device
        )  # add batch dim & move to device
        with torch.no_grad():
            out = resnet(img_tensor)  # (1, 5) on device
        outs[k] = out.squeeze().cpu()  # move back to CPU for averaging

    avg_out = outs.mean(dim=0)
    _, predicted = torch.max(avg_out, 0)
    y_preds.append(int(predicted.item()))



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1925442942.py in <cell line: 0>()
     10 
     11 # iterate over test images
---> 12 for img_name in tqdm(test_images):
     13     # open image and ensure RGB
     14     image = Image.open(os.path.join(test_path, img_name)).convert("RGB")

NameError: name 'tqdm' is not defined

## === cell 17
df_sub = pd.DataFrame({"image_id": test_images, "label": y_preds})
display(df_sub)



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/4244642311.py in <cell line: 0>()
----> 1 df_sub = pd.DataFrame({"image_id": test_images, "label": y_preds})
      2 display(df_sub)
      3 

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __init__(self, data, index, columns, dtype, copy)
    776         elif isinstance(data, dict):
    777             # GH#38939 de facto copy defaults to False only in non-dict cases
--> 778             mgr = dict_to_mgr(data, index, columns, dtype=dtype, copy=copy, typ=manager)
    779         elif isinstance(data, ma.MaskedArray):
    780             from numpy.ma import mrecords

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/construction.py in dict_to_mgr(data, index, columns, dtype, typ, copy)
    501             arrays = [x.copy() if hasattr(x, "dtype") else x for x in arrays]
    502 
--> 503     return arrays_to_mgr(arrays, columns, index, dtype=dtype, typ=typ, consolidate=copy)
    504 
    505 

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/construction.py in arrays_to_mgr(arrays, columns, index, dtype, verify_integrity, typ, consolidate)
    112         # figure out the index, if necessary
    113         if index is None:
--> 114             index = _extract_index(arrays)
    115         else:
    116             index = ensure_index(index)

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/construction.py in _extract_index(data)
    675         lengths = list(set(raw_lengths))
    676         if len(lengths) > 1:
--> 677             raise ValueError("All arrays must be of the same length")
    678 
    679         if have_dicts:

ValueError: All arrays must be of the same length

## === cell 18
df_sub.to_csv("submission.csv", index=False)

## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/622484759.py in <cell line: 0>()
----> 1 df_sub.to_csv("submission.csv", index=False)

NameError: name 'df_sub' is not defined
