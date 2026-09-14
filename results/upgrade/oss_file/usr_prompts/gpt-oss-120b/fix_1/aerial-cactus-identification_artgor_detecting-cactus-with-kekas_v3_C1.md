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

3.7

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

0.3983

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import torch
from torch.utils.data import TensorDataset, DataLoader,Dataset
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.optim import lr_scheduler
import time 
from PIL import Image
train_on_gpu = True
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR
from sklearn.metrics import accuracy_score
import cv2


## === cell 1
!pip install albumentations > /dev/null 2>&1
!pip install pretrainedmodels > /dev/null 2>&1
!pip install kekas > /dev/null 2>&1
!pip install adabound > /dev/null 2>&1


## === cell 2
import albumentations
from albumentations import torch as AT
import pretrainedmodels
import adabound

from kekas import Keker, DataOwner, DataKek
from kekas.transformations import Transformer, to_torch, normalize
from kekas.metrics import accuracy
from kekas.modules import Flatten, AdaptiveConcatPool2d
from kekas.callbacks import Callback, Callbacks, DebuggerCallback
from kekas.utils import DotDict


## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/3528655446.py in <cell line: 0>()
      1 # more imports
      2 import albumentations
----> 3 from albumentations import torch as AT
      4 import pretrainedmodels
      5 import adabound

ImportError: cannot import name 'torch' from 'albumentations' (/usr/local/lib/python3.11/dist-packages/albumentations/__init__.py)

## === cell 3
labels = pd.read_csv('../input/train.csv')
fig = plt.figure(figsize=(25, 8))
train_imgs = os.listdir("../input/train/train")
for idx, img in enumerate(np.random.choice(train_imgs, 20)):
    ax = fig.add_subplot(4, 20//4, idx+1, xticks=[], yticks=[])
    im = Image.open("../input/train/train/" + img)
    plt.imshow(im)
    lab = labels.loc[labels['id'] == img, 'has_cactus'].values[0]
    ax.set_title(f'Label: {lab}')


## === cell 4
test_img = os.listdir('../input/test/test')
test_df = pd.DataFrame(test_img, columns=['id'])
test_df['has_cactus'] = -1
test_df['data_type'] = 'test'

labels['has_cactus'] = labels['has_cactus'].astype(int)
labels['data_type'] = 'train'

labels.head()


## === cell 5
labels.loc[labels['data_type'] == 'train', 'has_cactus'].value_counts()


## === cell 6
train, valid = train_test_split(labels, stratify=labels.has_cactus, test_size=0.2)


## === cell 7
def reader_fn(i, row):
    image = cv2.imread(f"../input/{row['data_type']}/{row['data_type']}/{row['id']}")[:,:,::-1] # BGR -> RGB
    label = torch.Tensor([row["has_cactus"]])
    return {"image": image, "label": label}


## === cell 8
def augs(p=0.5):
    return albumentations.Compose([
        albumentations.HorizontalFlip(),
        albumentations.Transpose(),
        albumentations.Flip(),
    ], p=p)


## === cell 9
def get_transforms(dataset_key, size, p):

    PRE_TFMS = Transformer(dataset_key, lambda x: cv2.resize(x, (size, size)))

    AUGS = Transformer(dataset_key, lambda x: augs()(image=x)["image"])

    NRM_TFMS = transforms.Compose([
        Transformer(dataset_key, to_torch()),
        Transformer(dataset_key, normalize())
    ])
    
    train_tfms = transforms.Compose([PRE_TFMS, AUGS, NRM_TFMS])
    val_tfms = transforms.Compose([PRE_TFMS, NRM_TFMS])
    
    return train_tfms, val_tfms


## === cell 10
train_tfms, val_tfms = get_transforms("image", 32, 0.5)


## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/796524953.py in <cell line: 0>()
----> 1 train_tfms, val_tfms = get_transforms("image", 32, 0.5)

/tmp/ipykernel_11/504960847.py in get_transforms(dataset_key, size, p)
      1 def get_transforms(dataset_key, size, p):
      2 
----> 3     PRE_TFMS = Transformer(dataset_key, lambda x: cv2.resize(x, (size, size)))
      4 
      5     AUGS = Transformer(dataset_key, lambda x: augs()(image=x)["image"])

NameError: name 'Transformer' is not defined

## === cell 11
train_dk = DataKek(df=train, reader_fn=reader_fn, transforms=train_tfms)
val_dk = DataKek(df=valid, reader_fn=reader_fn, transforms=val_tfms)

batch_size = 64
workers = 0

train_dl = DataLoader(train_dk, batch_size=batch_size, num_workers=workers, shuffle=True, drop_last=True)
val_dl = DataLoader(val_dk, batch_size=batch_size, num_workers=workers, shuffle=False)


## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1192493966.py in <cell line: 0>()
----> 1 train_dk = DataKek(df=train, reader_fn=reader_fn, transforms=train_tfms)
      2 val_dk = DataKek(df=valid, reader_fn=reader_fn, transforms=val_tfms)
      3 
      4 batch_size = 64
      5 workers = 0

NameError: name 'DataKek' is not defined

## === cell 12
test_dk = DataKek(df=test_df, reader_fn=reader_fn, transforms=val_tfms)
test_dl = DataLoader(test_dk, batch_size=batch_size, num_workers=workers, shuffle=False)


## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4264597454.py in <cell line: 0>()
----> 1 test_dk = DataKek(df=test_df, reader_fn=reader_fn, transforms=val_tfms)
      2 test_dl = DataLoader(test_dk, batch_size=batch_size, num_workers=workers, shuffle=False)

NameError: name 'DataKek' is not defined

## === cell 13
class Net(nn.Module):
    def __init__(
            self,
            num_classes: int,
            p: float = 0.2,
            pooling_size: int = 2,
            last_conv_size: int = 81536,
            arch: str = "densenet169",
            pretrained: str = "imagenet") -> None:
        """A simple model to finetune.
        
        Args:
            num_classes: the number of target classes, the size of the last layer's output
            p: dropout probability
            pooling_size: the size of the result feature map after adaptive pooling layer
            last_conv_size: size of the flatten last backbone conv layer
            arch: the name of the architecture form pretrainedmodels
            pretrained: the mode for pretrained model from pretrainedmodels
        """
        super().__init__()
        net = pretrainedmodels.__dict__[arch](pretrained=pretrained)
        modules = list(net.children())[:-1]  # delete last layer
        modules += [nn.Sequential(
            AdaptiveConcatPool2d(size=pooling_size),
            Flatten(),
            nn.BatchNorm1d(13312),
            nn.Dropout(p),
            nn.Linear(13312, num_classes)
        )]
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        logits = self.net(x)
        return logits


## === cell 14
dataowner = DataOwner(train_dl, val_dl, None)
model = Net(num_classes=1)
criterion = nn.BCEWithLogitsLoss()


## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3360747037.py in <cell line: 0>()
----> 1 dataowner = DataOwner(train_dl, val_dl, None)
      2 model = Net(num_classes=1)
      3 criterion = nn.BCEWithLogitsLoss()

NameError: name 'DataOwner' is not defined

## === cell 15
def step_fn(model: torch.nn.Module,
            batch: torch.Tensor) -> torch.Tensor:
    """Determine what your model will do with your data.

    Args:
        model: the pytorch module to pass input in
        batch: the batch of data from the DataLoader

    Returns:
        The models forward pass results
    """
    
    inp = batch["image"]
    return model(inp)


## === cell 16
def bce_accuracy(target: torch.Tensor,
                 preds: torch.Tensor,
                 thresh: bool = 0.5) -> float:
    target = target.cpu().detach().numpy()
    preds = (torch.sigmoid(preds).cpu().detach().numpy() > thresh).astype(int)
    return accuracy_score(target, preds)
  
def roc_auc(target: torch.Tensor,
                 preds: torch.Tensor) -> float:
    target = target.cpu().detach().numpy()
    preds = torch.sigmoid(preds).cpu().detach().numpy()
    return roc_auc_score(target, preds)


## === cell 17
keker = Keker(model=model,
              dataowner=dataowner,
              criterion=criterion,
              step_fn=step_fn,
              target_key="label",
              metrics={"acc": bce_accuracy, 'auc': roc_auc},
              opt=torch.optim.SGD,
              opt_params={"momentum": 0.99})


## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/977955414.py in <cell line: 0>()
----> 1 keker = Keker(model=model,
      2               dataowner=dataowner,
      3               criterion=criterion,
      4               step_fn=step_fn,
      5               target_key="label",

NameError: name 'Keker' is not defined

## === cell 18
keker.unfreeze(model_attr="net")

layer_num = -1
keker.freeze_to(layer_num, model_attr="net")


## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3468827365.py in <cell line: 0>()
----> 1 keker.unfreeze(model_attr="net")
      2 
      3 layer_num = -1
      4 keker.freeze_to(layer_num, model_attr="net")

NameError: name 'keker' is not defined

## === cell 19
keker.kek_one_cycle(max_lr=1e-1,                  # the maximum learning rate
                    cycle_len=5,                  # number of epochs, actually, but not exactly
                    momentum_range=(0.95, 0.85),  # range of momentum changes
                    div_factor=25,                # max_lr / min_lr
                    increase_fraction=0.3,        # the part of cycle when learning rate increases
                    logdir='train_logs')
keker.plot_kek('train_logs')


## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/507259039.py in <cell line: 0>()
----> 1 keker.kek_one_cycle(max_lr=1e-1,                  # the maximum learning rate
      2                     cycle_len=5,                  # number of epochs, actually, but not exactly
      3                     momentum_range=(0.95, 0.85),  # range of momentum changes
      4                     div_factor=25,                # max_lr / min_lr
      5                     increase_fraction=0.3,        # the part of cycle when learning rate increases

NameError: name 'keker' is not defined

## === cell 20
keker.unfreeze(model_attr="net")
keker.kek_one_cycle(max_lr=1e-2,                  # the maximum learning rate
                    cycle_len=3,                  # number of epochs, actually, but not exactly
                    momentum_range=(0.95, 0.85),  # range of momentum changes
                    div_factor=25,                # max_lr / min_lr
                    increase_fraction=0.3,        # the part of cycle when learning rate increases
                    logdir='train_logs1')
keker.plot_kek('train_logs1')


## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/489524313.py in <cell line: 0>()
----> 1 keker.unfreeze(model_attr="net")
      2 keker.kek_one_cycle(max_lr=1e-2,                  # the maximum learning rate
      3                     cycle_len=3,                  # number of epochs, actually, but not exactly
      4                     momentum_range=(0.95, 0.85),  # range of momentum changes
      5                     div_factor=25,                # max_lr / min_lr

NameError: name 'keker' is not defined

## === cell 21
preds = keker.predict_loader(loader=test_dl)


## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3109920335.py in <cell line: 0>()
----> 1 preds = keker.predict_loader(loader=test_dl)

NameError: name 'keker' is not defined

## === cell 22
flip_ = albumentations.HorizontalFlip(always_apply=True)
transpose_ = albumentations.Transpose(always_apply=True)

def insert_aug(aug, dataset_key="image", size=224):    
    PRE_TFMS = Transformer(dataset_key, lambda x: cv2.resize(x, (size, size)))
    
    AUGS = Transformer(dataset_key, lambda x: aug(image=x)["image"])
    
    NRM_TFMS = transforms.Compose([
        Transformer(dataset_key, to_torch()),
        Transformer(dataset_key, normalize())
    ])
    
    tfm = transforms.Compose([PRE_TFMS, AUGS, NRM_TFMS])
    return tfm

flip = insert_aug(flip_)
transpose = insert_aug(transpose_)

tta_tfms = {"flip": flip, "transpose": transpose}

keker.TTA(loader=test_dl,                # loader to predict on 
          tfms=tta_tfms,                # list or dict of always applying transforms
          savedir="tta_preds1",  # savedir
          prefix="preds")               # (optional) name prefix. default is 'preds'


## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2706937858.py in <cell line: 0>()
     15     return tfm
     16 
---> 17 flip = insert_aug(flip_)
     18 transpose = insert_aug(transpose_)
     19 

/tmp/ipykernel_11/2706937858.py in insert_aug(aug, dataset_key, size)
      3 
      4 def insert_aug(aug, dataset_key="image", size=224):
----> 5     PRE_TFMS = Transformer(dataset_key, lambda x: cv2.resize(x, (size, size)))
      6 
      7     AUGS = Transformer(dataset_key, lambda x: aug(image=x)["image"])

NameError: name 'Transformer' is not defined

## === cell 23
prediction = np.zeros((test_df.shape[0], 1))
for i in os.listdir('tta_preds1'):
    pr = np.load('tta_preds1/' + i)
    prediction += pr
prediction = prediction / len(os.listdir('tta_preds1'))


## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/2947314519.py in <cell line: 0>()
      1 prediction = np.zeros((test_df.shape[0], 1))
----> 2 for i in os.listdir('tta_preds1'):
      3     pr = np.load('tta_preds1/' + i)
      4     prediction += pr
      5 prediction = prediction / len(os.listdir('tta_preds1'))

FileNotFoundError: [Errno 2] No such file or directory: 'tta_preds1'

## === cell 24
test_preds = pd.DataFrame({'imgs': test_df.id.values, 'preds': prediction.reshape(-1,)})
test_preds.columns = ['id', 'has_cactus']
test_preds.to_csv('sub.csv', index=False)
test_preds.head()


## --- ERROR in outputing the csv:
Invalid submission: Submission and answers should have the same number of rows
