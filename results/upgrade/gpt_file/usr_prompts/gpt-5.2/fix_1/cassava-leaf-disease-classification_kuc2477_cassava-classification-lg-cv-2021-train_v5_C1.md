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
imageio==2.37.0
imageio-ffmpeg==0.6.0
numpy==1.26.4
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

0.6128739800543971

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os.path as osp

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import imageio
from imgaug import augmenters as iaa
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from tqdm.notebook import tqdm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/1615853561.py in <cell line: 0>()
     11 # data pipeline
     12 import imageio
---> 13 from imgaug import augmenters as iaa
     14 from torch.utils.data import Dataset, DataLoader
     15 from torchvision import transforms

ModuleNotFoundError: No module named 'imgaug'

## === cell 2
ROOT = '/kaggle/input/cassava-leaf-disease-classification'
TRAIN_DIR = f'{ROOT}/train_images/'
TRAIN_CSV = f'{ROOT}/train.csv'
TEST_DIR = f'{ROOT}/test_images/'
TEST_CSV = f'{ROOT}/sample_submission.csv'
BEST_MODEL_PATH = '/kaggle/working/model.pt'

## === cell 3
class CassavaDataset(Dataset):
    def __init__(self, split, transform=None):
        assert split in ('train', 'val', 'test')
        self.split = split
        self.transform = transform
        if split in ('train', 'val'):
            csv = pd.read_csv(TRAIN_CSV)
            self.df = train_test_split(
                csv, test_size=0.1, random_state=0
            )[0 if split == 'train' else 1].reset_index()
        else:
            self.df = pd.read_csv(TEST_CSV)
        
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, i: int):
        base_dir = TRAIN_DIR if self.split in ('train', 'val') else TEST_DIR
        x = imageio.imread(osp.join(base_dir, self.df['image_id'][i]))
        y = self.df['label'][i] if self.split in ('train', 'val') else -1
        if self.transform:
            x = self.transform(x)
        return (x, y)

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1964992959.py in <cell line: 0>()
----> 1 class CassavaDataset(Dataset):
      2     def __init__(self, split, transform=None):
      3         assert split in ('train', 'val', 'test')
      4         self.split = split
      5         self.transform = transform

NameError: name 'Dataset' is not defined

## === cell 5
INPUT_SIZE = 128
TRANSFORMS = {
    'train': transforms.Compose([
        iaa.Sequential([
            iaa.Resize((INPUT_SIZE, INPUT_SIZE)),
        ]).augment_image,
        transforms.ToTensor(),
    ]),
    'val': transforms.Compose([
        iaa.Sequential([
            iaa.Resize((INPUT_SIZE, INPUT_SIZE)),
        ]).augment_image,
        transforms.ToTensor(),
    ]),
    'test': transforms.Compose([
        iaa.Sequential([
            iaa.Resize((INPUT_SIZE, INPUT_SIZE)),
        ]).augment_image,
        transforms.ToTensor(),
    ]),
}

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/315232484.py in <cell line: 0>()
     11 INPUT_SIZE = 128
     12 TRANSFORMS = {
---> 13     'train': transforms.Compose([
     14         iaa.Sequential([
     15             iaa.Resize((INPUT_SIZE, INPUT_SIZE)),

NameError: name 'transforms' is not defined

## === cell 7
class CassavaClassifier(nn.Module):
    
    def __init__(self):

        super().__init__()
        CH_LAST = 128
        NUM_CLASSES = 5
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, CH_LAST, kernel_size=3, padding=1)
        self.fc = nn.Linear(CH_LAST *  (INPUT_SIZE // 2 ** 3) ** 2, NUM_CLASSES)
        self.pool = nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1)
        
    def forward(self, x):
        h = self.pool(F.relu(self.conv1(x)))
        h = self.pool(F.relu(self.conv2(h)))
        h = self.pool(F.relu(self.conv3(h)))
        return self.fc(h.flatten(1))

## === cell 9
EPOCHS = 1
BATCH_SIZE = 64
MOMENTUM = 0.99
LR = 0.0001
WD = 0.0001

## === cell 10
model = CassavaClassifier()
model.cuda()
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WD)
train_dataset = CassavaDataset('train', transform=TRANSFORMS['train'])
val_dataset = CassavaDataset('val', transform=TRANSFORMS['val'])

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_11/2216315857.py in <cell line: 0>()
      9 # for their actual APIs and detailed usages.
     10 model = CassavaClassifier()
---> 11 model.cuda()
     12 optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WD)
     13 train_dataset = CassavaDataset('train', transform=TRANSFORMS['train'])

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in cuda(self, device)
   1051             Module: self
   1052         """
-> 1053         return self._apply(lambda t: t.cuda(device))
   1054 
   1055     def ipu(self: T, device: Optional[Union[int, device]] = None) -> T:

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _apply(self, fn, recurse)
    901         if recurse:
    902             for module in self.children():
--> 903                 module._apply(fn)
    904 
    905         def compute_should_use_set_data(tensor, tensor_applied):

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _apply(self, fn, recurse)
    928             # `with torch.no_grad():`
    929             with torch.no_grad():
--> 930                 param_applied = fn(param)
    931             p_should_use_set_data = compute_should_use_set_data(param, param_applied)
    932 

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in <lambda>(t)
   1051             Module: self
   1052         """
-> 1053         return self._apply(lambda t: t.cuda(device))
   1054 
   1055     def ipu(self: T, device: Optional[Union[int, device]] = None) -> T:

/usr/local/lib/python3.11/dist-packages/torch/cuda/__init__.py in _lazy_init()
    317         if "CUDA_MODULE_LOADING" not in os.environ:
    318             os.environ["CUDA_MODULE_LOADING"] = "LAZY"
--> 319         torch._C._cuda_init()
    320         # Some of the queued calls may reentrantly call _lazy_init();
    321         # we need to just return without initializing in that case.

RuntimeError: Found no NVIDIA driver on your system. Please check that you have an NVIDIA GPU and installed a driver from http://www.nvidia.com/Download/index.aspx

## === cell 11
TRAIN_AND_SAVE = True

best_val_accuracy = 0.

for e in range(EPOCHS if TRAIN_AND_SAVE else 0):
    train_loader = DataLoader(train_dataset, BATCH_SIZE, num_workers=BATCH_SIZE, pin_memory=True, shuffle=True)
    val_loader = DataLoader(val_dataset, BATCH_SIZE, num_workers=BATCH_SIZE, pin_memory=True)
    
    model.train()
    total = 0
    correct = 0
    for x, y in tqdm(train_loader, desc=f'Training the network (Epoch {e})'):
        x, y = x.cuda(), y.cuda()
        logits = model(x)
        optimizer.zero_grad()
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()
        
        _, y_hat = torch.max(logits, dim=1)
        total += x.size(0)
        correct += (y_hat == y).sum().detach().cpu().numpy()        
                
    print('==========================')
    print(f'Train accuracy (Epoch {e}): {correct / total}')
    print('==========================')
    
    model.eval()
    total = 0
    correct = 0
    for x, y in tqdm(val_loader, desc=f'Evaluating the network (Epoch {e})'):
        x, y = x.cuda(), y.cuda()
        with torch.no_grad():
            logits = model(x)
        _, y_hat = torch.max(logits, dim=1)
        total += x.size(0)
        correct += (y_hat == y).sum().detach().cpu().numpy()
    
    val_accuracy = correct / total
    
    print('==========================')
    print(f'Test accuracy (Epoch {e}): {val_accuracy}')
    print('==========================')
    
    if val_accuracy > best_val_accuracy:
        print(f'* Saving the best model with accuracy {val_accuracy}')
        torch.save(model, BEST_MODEL_PATH)

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1792099041.py in <cell line: 0>()
      5 
      6 for e in range(EPOCHS if TRAIN_AND_SAVE else 0):
----> 7     train_loader = DataLoader(train_dataset, BATCH_SIZE, num_workers=BATCH_SIZE, pin_memory=True, shuffle=True)
      8     val_loader = DataLoader(val_dataset, BATCH_SIZE, num_workers=BATCH_SIZE, pin_memory=True)
      9 

NameError: name 'DataLoader' is not defined

## === cell 13
model = torch.load(BEST_MODEL_PATH)
model.eval()
test_dataset = CassavaDataset('test', transform=TRANSFORMS['test'])
test_loader = DataLoader(test_dataset, batch_size=32)
test_csv = pd.read_csv(TEST_CSV)

y_hats = []
for x, _ in test_loader:
    y_hat = model(x.cuda())
    y_hat = torch.argmax(y_hat,dim=1)
    y_hats.extend(y_hat.cpu().detach().numpy().tolist())
    

test_csv['label'] = y_hats
test_csv[['image_id','label']].to_csv("submission.csv", index=False)
test_csv.head()

## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/2945717891.py in <cell line: 0>()
----> 1 model = torch.load(BEST_MODEL_PATH)
      2 model.eval()
      3 test_dataset = CassavaDataset('test', transform=TRANSFORMS['test'])
      4 test_loader = DataLoader(test_dataset, batch_size=32)
      5 test_csv = pd.read_csv(TEST_CSV)

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

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/working/model.pt'
