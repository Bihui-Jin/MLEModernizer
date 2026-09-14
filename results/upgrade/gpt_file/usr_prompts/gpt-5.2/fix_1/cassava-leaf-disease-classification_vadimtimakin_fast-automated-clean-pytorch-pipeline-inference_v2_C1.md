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

albumentations==2.0.8
geopandas==0.14.4
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

0.7243880326382592

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 3
import torch
import torchvision.models as models
import numpy as np
import pandas as pd
import random
import os
import cv2
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensor
from tqdm.notebook import tqdm

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/1328928470.py in <cell line: 0>()
      8 import torch.nn as nn
      9 import albumentations as A
---> 10 from albumentations.pytorch import ToTensor
     11 from tqdm.notebook import tqdm

ImportError: cannot import name 'ToTensor' from 'albumentations.pytorch' (/usr/local/lib/python3.11/dist-packages/albumentations/pytorch/__init__.py)

## === cell 5
class cfg:
    """Main config."""
    
    NUMCLASSES = 5  # CONST
    seed = 42       # random seed
    
    pathtoimgs = "../input/cassava-leaf-disease-classification/test_images"  # Path to folder with train images
    pathtocsv = "../input/cassava-leaf-disease-classification/sample_submission.csv"  # Path to csv-file with targets
    chk = "../input/cassava/weights.pt"  # Path to model checkpoint (weights)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Device
    modelname = "resnext101_32x8d"  # PyTorch model
    
    batchsize = 1   # BatchSize
    numworkers = 4  # Number of workers
    
    transforms = [
        dict(
            name="Normalize",
            params=dict(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0
            )
        ),
        dict(
            name="/custom/totensor",
            params=dict(
            )
        ),
    ]

## === cell 6
print(cfg.device)

## === cell 8
def totensor():
    return A.pytorch.ToTensor()

## === cell 10
def fullseed(seed=42):
    """Sets the random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    os.environ['PYTHONHASHSEED'] = str(seed)

    
fullseed(cfg.seed)

## === cell 12
def get_model(cfg):
    """Get PyTorch model."""
    model = getattr(models, cfg.modelname)(pretrained=False)
    lastlayer = list(model._modules)[-1]
    setattr(model, lastlayer, nn.Linear(in_features=getattr(model, lastlayer).in_features,
                                        out_features=cfg.NUMCLASSES, bias=True))
    cp = torch.load(cfg.chk)
    if 'model' in cp:
        model.load_state_dict(cp['model'])
    else:
        model.load_state_dict(cp)
    if 'epoch' in cp:
        epoch = int(cp['epoch'])
    if 'trainloss' in cp:
        trainloss = cp['trainloss']
    if 'valloss' in cp:
        valloss = cp['valloss']
    if 'metric' in cp:
        metric = cp['metric']
    if 'lr' in cp:
        lr = cp['lr']
        cfg.lr = lr
    if 'stopflag' in cp:
        stopflag = cp['stopflag']
        cfg.stopflag = stopflag
    return model.to(cfg.device)


def get_transforms(cfg):
    """Get train and test augmentations."""
    transforms = [globals()[item["name"][8:]](**item["params"]) if item["name"].startswith("/custom/") 
                     else getattr(A, item["name"])(**item["params"]) for item in cfg.transforms]
    return A.Compose(transforms)

## === cell 13
class CassavaDataset(torch.utils.data.Dataset):
    """Cassava Dataset for uploading images and targets."""
    
    def __init__(self, cfg, images, transforms):
        self.images = images           # List with images
        self.transforms = transforms   # Transforms
        self.cfg = cfg                 # Config
        
    def __getitem__(self, idx):
        img = cv2.imread(os.path.join(self.cfg.pathtoimgs, self.images[idx]))
        img = torch.FloatTensor(self.transforms(image=img)["image"])
        return img

    def __len__(self):
        return len(self.images)

## === cell 14
def get_loader(cfg):
    """Getting dataloaders for train, validation (and test, if needed)."""
    data = pd.read_csv(cfg.pathtocsv)
    imgs = list(data["image_id"])
    transforms = get_transforms(cfg)
    dataset = CassavaDataset(cfg, imgs, transforms)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             shuffle=False,
                                             batch_size=cfg.batchsize,
                                             pin_memory=True,
                                             num_workers=cfg.numworkers)
    return dataloader

## === cell 16
torch.cuda.empty_cache()
dataloader = get_loader(cfg)
model = get_model(cfg) 
model.eval()

preds = []
with torch.no_grad():
    for img in tqdm(dataloader):
        outputs = model(img.to(cfg.device))
        preds.append(np.argmax(outputs.to('cpu').numpy()).tolist())

## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/3271166978.py in <cell line: 0>()
      1 torch.cuda.empty_cache()
----> 2 dataloader = get_loader(cfg)
      3 model = get_model(cfg)
      4 model.eval()
      5 

/tmp/ipykernel_11/296893914.py in get_loader(cfg)
      3     data = pd.read_csv(cfg.pathtocsv)
      4     imgs = list(data["image_id"])
----> 5     transforms = get_transforms(cfg)
      6     dataset = CassavaDataset(cfg, imgs, transforms)
      7     dataloader = torch.utils.data.DataLoader(dataset,

/tmp/ipykernel_11/2204067274.py in get_transforms(cfg)
     29 def get_transforms(cfg):
     30     """Get train and test augmentations."""
---> 31     transforms = [globals()[item["name"][8:]](**item["params"]) if item["name"].startswith("/custom/") 
     32                      else getattr(A, item["name"])(**item["params"]) for item in cfg.transforms]
     33     return A.Compose(transforms)

/tmp/ipykernel_11/2204067274.py in <listcomp>(.0)
     29 def get_transforms(cfg):
     30     """Get train and test augmentations."""
---> 31     transforms = [globals()[item["name"][8:]](**item["params"]) if item["name"].startswith("/custom/") 
     32                      else getattr(A, item["name"])(**item["params"]) for item in cfg.transforms]
     33     return A.Compose(transforms)

/tmp/ipykernel_11/3739077452.py in totensor()
      1 def totensor():
----> 2     return A.pytorch.ToTensor()

AttributeError: module 'albumentations.pytorch' has no attribute 'ToTensor'

## === cell 18
df = pd.read_csv(cfg.pathtocsv)
df["label"] = preds
df.to_csv('submission.csv', index=False)
df.head()

## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2690803932.py in <cell line: 0>()
      1 df = pd.read_csv(cfg.pathtocsv)
----> 2 df["label"] = preds
      3 df.to_csv('submission.csv', index=False)
      4 df.head()

NameError: name 'preds' is not defined
