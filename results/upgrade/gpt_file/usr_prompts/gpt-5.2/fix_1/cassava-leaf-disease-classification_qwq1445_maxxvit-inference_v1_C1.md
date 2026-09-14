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

# 5. Target score

0.8579631308552432

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 3
import sys
sys.path.append('../input/pytorchimagemodels/pytorch-image-models-main')
import timm

## === cell 7
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

## === cell 8
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

## === cell 10
class Config:
    seed = 42
    data_dir = '../input/cassava-leaf-disease-classification/'
    train_data_dir = data_dir + 'test_images/'
    train_csv_path = data_dir + 'test.csv'
    arch = 'maxxvit_rmlp_nano_rw_256' ## model name
    device = 'cuda'
    debug = True                 ##
    
    image_size = 256    
    train_batch_size = 16
    val_batch_size = 32
    epochs = 10                 ## total train epochs
    freeze_bn_epochs = 5        ## freeze bn weights before epochs
    
    lr=1e-4                     ## init learning rate
    min_lr = 1e-6               ## min learning rate
    weight_decay = 1e-6
    num_workers = 4
    num_splits = 5             ## numbers splits
    num_classes = 5            ## numbers classes
    T_0 = 10
    T_mult = 1
    accum_iter = 2
    verbose_step = 1
    
    criterion = 'LabelSmoothingCrossEntropy' ## CrossEntropy, LabelSmoothingCrossEntropy
    label_smoothing = 0.3
    
    train_id = [0,1,2,3,4]

## === cell 12
def load_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

## === cell 14
class CassavaDataset(Dataset):
    def __init__(self, data_dir, df, transforms=None, output_label=True):
        self.data_dir = data_dir
        self.df = df
        self.transforms = transforms
        self.output_label = output_label

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        image_infos = self.df.iloc[index]
        image_path = self.data_dir + image_infos.image_id

        image = load_image(image_path)

        if image is None:
            raise FileNotFoundError(image_path)

        if self.transforms is not None:
            image = self.transforms(image=image)['image']
        else:
            image = torch.from_numpy(image)

        if self.output_label:
            return image, image_infos.label
        else:
            return image

## === cell 16
class CassavaClassifier(nn.Module):
    def __init__(self, model_arch, num_classes, pretrained=False):
        super().__init__()
        self.model = timm.create_model(model_arch, pretrained=pretrained)

        
        '''
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            #nn.Linear(num_features, hidden_size,bias=True), nn.ELU(),
            nn.Linear(num_features, num_classes, bias=True)
        )
        '''
    def forward(self, x):
        x = self.model(x)
        return x

## === cell 18
def get_train_transforms(CFG):
    return A.Compose([
            A.RandomResizedCrop(height=CFG.image_size, width=CFG.image_size, p=0.5),
            A.Transpose(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(p=0.5),
            A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
            A.CenterCrop(CFG.image_size, CFG.image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            A.CoarseDropout(p=0.5),
            A.Cutout(p=0.5),
            ToTensorV2(),
        ],p=1.0)

def get_val_transforms(cfg):
    return A.Compose([
            A.CenterCrop(CFG.image_size, CFG.image_size, p=0.5),
            A.Resize(CFG.image_size, CFG.image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            ToTensorV2(),
        ],p=1.0)

## === cell 19
def get_inference_transforms(CFG):
    return A.Compose([
            A.RandomResizedCrop(CFG.image_size, CFG.image_size),
            A.Transpose(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.0)

## === cell 22
def train_one_epoch(epoch,model,loss_fn,optimizer,train_loader,device,scheduler=None,schd_batch_update=False):
    model.train()
    lr = optimizer.state_dict()['param_groups'][0]['lr']   ##不同
    
    running_loss = None
    pbar = tqdm(enumerate(train_loader),total=len(train_loader))
    for step,(images,targets) in pbar:
        images = images.to(device).float()
        targets = targets.to(device).long()
        
        with autocast():
            preds = model(images)
            loss = loss_fn(preds,targets)
        
            scaler.scale(loss).backward()
            if running_loss is None:
                running_loss = loss.item()
            else:
                running_loss = running_loss* 0.99 + loss.item()*0.01
                
            if ((step + 1) % CFG.accum_iter == 0) or ((step + 1) == len(train_loader)):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                if scheduler is not None and schd_batch_update:
                    scheduler.step()
            if ((step + 1) % CFG.accum_iter == 0) or ((step + 1) == len(train_loader)):
                description = f'Train epoch {epoch} loss: {running_loss:.5f}'
                pbar.set_description(description)
                
    if scheduler is not None and schd_batch_update:
        scheduler.step()

## === cell 23
def load_dataloader(CFG, df, train_idx):
    df_train = df.loc[train_idx,:].reset_index(drop=True)

    train_dataset = CassavaDataset(
        CFG.train_data_dir,
        df_train,
        transforms=get_train_transforms(CFG), 
        output_label=False)


    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=CFG.train_batch_size,
        pin_memory=False,
        shuffle=False,        
        num_workers=CFG.num_workers,
    )

    
    return train_loader

## === cell 25
def valid_one_epoch(epoch,model,loss_fn,val_loader,device,scheduler=None,schd_loss_update=False):
    model.eval()
    
    loss_sum = 0
    sample_num = 0
    preds_all = []
    targets_all = []
    scores = []
    
    pbar = tqdm(enumerate(val_loader),total=len(val_loader))
    for step,(images,targets) in pbar:
        images = images.to(device).float()
        targets = targets.to(device).long()
        preds = model(images)
            
        preds_all += [torch.argmax(preds,1).detach().cpu().numpy()]
        targets_all += [targets.detach().cpu().numpy()]

        loss = loss_fn(preds,targets)
        loss_sum += loss.item()*targets.shape[0]
        sample_num += targets.shape[0]
           
        if ((step + 1) % CFG.accum_iter == 0) or ((step + 1) == len(train_loader)):
            description = f'Val epoch {epoch} loss: {loss_sum/sample_num:.5f}'
            pbar.set_description(description)
            
    preds_all = np.concatenate(preds_all)
    targets_all = np.concatenate(targets_all)
    accuracy = (preds_all == targets_all).mean()
    print(f'Validation multi-class accuracy = {accuracy:.5f}')
    
    if scheduler is not None:
        if schd_loss_update:
            scheduler.step(loss_sum/sample_num)
        else:
            scheduler.step()
    
    return accuracy

## === cell 26
def inference_one_epoch(model, data_loader, device):
    model.eval()

    image_preds_all = []
    
    pbar = tqdm(enumerate(data_loader), total=len(data_loader))
    for step, (imgs) in pbar:
        imgs = imgs.to(device).float()
        
        image_preds = model(imgs)   #output = model(input)
        image_preds_all += [torch.softmax(image_preds, 1).detach().cpu().numpy()]
        
    
    image_preds_all = np.concatenate(image_preds_all, axis=0)
    return image_preds_all

## === cell 28
def freeze_batchnorm_stats(net):
    try:
        for m in net.modules():
            if isinstance(m,nn.BatchNorm2d) or isinstance(m,nn.LayerNorm):
                m.eval()
    except ValuError:
        print('error with batchnorm2d or layernorm')
        return

## === cell 30
class LabelSmoothingCrossEntropy(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x, target):
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

## === cell 31
asd=['../input/maxxvit-ckpt/maxxvit_rmlp_nano_rw_256_fold2_best_false.ckpt']

## === cell 32
train = pd.read_csv('../input/cassava-leaf-disease-classification/train.csv')
train.head()

## === cell 34
if __name__ == '__main__':
    CFG = Config
    seed_everything(CFG.seed)
    
    folds = StratifiedKFold(n_splits=CFG.seed).split(np.arange(train.shape[0]), train.label.values)
    
    for fold, (trn_idx, val_idx) in enumerate(folds):
        if fold > 0:
            break 


        
        test = pd.DataFrame()
        test['image_id'] = list(os.listdir('../input/cassava-leaf-disease-classification/test_images/'))
        
        
        
        tst_loader=load_dataloader(CFG, test,np.array(list(test.index)))
        

        device = torch.device(CFG.device)
        model = CassavaClassifier(CFG.arch, train.label.nunique(), pretrained=False).to(device)
        tst_preds = []
        
        for i in asd:
            model.load_state_dict(torch.load(i,map_location=device))
        
            with torch.no_grad():
                tst_preds+=[inference_one_epoch(model, tst_loader, device)]
        
        tst_preds = np.mean(tst_preds, axis=0) 
        
            

        
        
        del model
        torch.cuda.empty_cache()

## --- ERROR in cell 34, traceback:
---------------------------------------------------------------------------
ValidationError                           Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/albumentations/core/validation.py in _validate_parameters(schema_cls, full_kwargs, param_names, strict)
     66             schema_kwargs["strict"] = strict
---> 67             config = schema_cls(**schema_kwargs)
     68             validated_kwargs = config.model_dump()

/usr/local/lib/python3.11/dist-packages/pydantic/main.py in __init__(self, **data)
    249         __tracebackhide__ = True
--> 250         validated_self = self.__pydantic_validator__.validate_python(data, self_instance=self)
    251         if self is not validated_self:

ValidationError: 1 validation error for InitSchema
size
  Field required [type=missing, input_value={'p': 0.5, 'scale': (0.08...: None, 'strict': False}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.12/v/missing

The above exception was the direct cause of the following exception:

ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2421579004.py in <cell line: 0>()
     36 #         )
     37 
---> 38         tst_loader=load_dataloader(CFG, test,np.array(list(test.index)))
     39 
     40 

/tmp/ipykernel_11/2456121049.py in load_dataloader(CFG, df, train_idx)
      6         CFG.train_data_dir,
      7         df_train,
----> 8         transforms=get_train_transforms(CFG),
      9         output_label=False)
     10 

/tmp/ipykernel_11/1260327829.py in get_train_transforms(CFG)
      1 def get_train_transforms(CFG):
      2     return A.Compose([
----> 3             A.RandomResizedCrop(height=CFG.image_size, width=CFG.image_size, p=0.5),
      4             A.Transpose(p=0.5),
      5             A.HorizontalFlip(p=0.5),

/usr/local/lib/python3.11/dist-packages/albumentations/core/validation.py in custom_init(self, *args, **kwargs)
    103                 full_kwargs, param_names, strict = cls._process_init_parameters(original_init, args, kwargs)
    104 
--> 105                 validated_kwargs = cls._validate_parameters(
    106                     dct["InitSchema"],
    107                     full_kwargs,

/usr/local/lib/python3.11/dist-packages/albumentations/core/validation.py in _validate_parameters(schema_cls, full_kwargs, param_names, strict)
     69             validated_kwargs.pop("strict", None)
     70         except ValidationError as e:
---> 71             raise ValueError(str(e)) from e
     72         except Exception as e:
     73             if strict:

ValueError: 1 validation error for InitSchema
size
  Field required [type=missing, input_value={'p': 0.5, 'scale': (0.08...: None, 'strict': False}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.12/v/missing

## === cell 35
tst_preds

## --- ERROR in cell 35, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3651141473.py in <cell line: 0>()
----> 1 tst_preds

NameError: name 'tst_preds' is not defined

## === cell 36
test['label'] = np.argmax(tst_preds, axis=1)
test.head()

## --- ERROR in cell 36, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/27549819.py in <cell line: 0>()
----> 1 test['label'] = np.argmax(tst_preds, axis=1)
      2 test.head()

NameError: name 'tst_preds' is not defined

## === cell 37
test.to_csv('submission.csv', index=False)

## --- ERROR in outputing the csv:
Invalid submission: Submission must have the same length as the answers.
