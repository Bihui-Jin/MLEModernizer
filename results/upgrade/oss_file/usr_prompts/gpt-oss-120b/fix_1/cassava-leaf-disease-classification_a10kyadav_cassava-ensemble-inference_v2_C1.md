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

0.8919613176186159

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
!pip install --quiet ../input/cassava-models/inputs/keras-applications
!pip install --quiet ../input/cassava-models/inputs/efficientnet_git

## === cell 1
import sys

package_path = '../input/cassava-models/inputs/Vision Transformer Pytorch/VisionTransformer-Pytorch'
sys.path.append(package_path)
from vision_transformer_pytorch import VisionTransformer

package_path = '../input/cassava-models/inputs/pytorch image models/pytorch-image-models-master'
sys.path.append(package_path)
import timm

## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/3591524633.py in <cell line: 0>()
      3 package_path = '../input/cassava-models/inputs/Vision Transformer Pytorch/VisionTransformer-Pytorch'
      4 sys.path.append(package_path)
----> 5 from vision_transformer_pytorch import VisionTransformer
      6 
      7 package_path = '../input/cassava-models/inputs/pytorch image models/pytorch-image-models-master'

ModuleNotFoundError: No module named 'vision_transformer_pytorch'

## === cell 2
import cv2
from skimage import io
import time
import random
from torchvision import transforms
from tqdm import tqdm

import matplotlib.pyplot as plt
from torch.utils.data import Dataset,DataLoader

import os, re, glob
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.layers as L
import tensorflow.keras.backend as K
from tensorflow.keras import Model
import efficientnet.tfkeras as efn

import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models
from albumentations.pytorch import ToTensorV2
import albumentations

from pathlib import Path
from contextlib import contextmanager

from scipy.special import softmax
from sklearn.metrics import accuracy_score
from tqdm.auto import tqdm
from PIL import Image


import warnings 
warnings.filterwarnings('ignore')

## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 3
IMAGE_SIZE = 512
CLASSES = 5

SEED = 99
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
seed_everything(SEED)

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/535515513.py in <cell line: 0>()
      3 
      4 SEED = 99
----> 5 device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
      6 
      7 

NameError: name 'torch' is not defined

## === cell 5
class PytorchCassavaDataset(Dataset):
    def __init__(self, df = pd.read_csv('../input/cassava-leaf-disease-classification/sample_submission.csv'),
                 data_root = "../input/cassava-leaf-disease-classification/test_images", transforms=None):
        
        super().__init__()
        self.df = df.reset_index(drop=True).copy()
        self.transforms = transforms
        self.data_root = data_root
        
    def get_img(self, path):
        im_bgr = cv2.imread(path)
        im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
        return im_rgb

    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, index: int):
        path = "{}/{}".format(self.data_root, self.df.iloc[index]['image_id'])
        img  = self.get_img(path)
        
        if self.transforms:
            img = self.transforms(image=img)['image']
        return img
    
    
AUTO = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 32

def get_name(file_path):
    parts = tf.strings.split(file_path, os.path.sep)
    name = parts[-1]
    return name

def decode_image(image_data):
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    return image

def center_crop(image):
    image = tf.reshape(image, [600, 800, 3]) # Original shape
    
    h, w = image.shape[0], image.shape[1]
    if h > w:
        image = tf.image.crop_to_bounding_box(image, (h - w) // 2, 0, w, w)
    else:
        image = tf.image.crop_to_bounding_box(image, 0, (w - h) // 2, h, h)
        
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE]) # Expected shape
    return image

def resize_image(image, label):
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
    image = tf.reshape(image, [IMAGE_SIZE, IMAGE_SIZE, 3])
    return image, label

def process_path(file_path):
    name = get_name(file_path)
    img = tf.io.read_file(file_path)
    img = decode_image(img)
    return img, name

def get_dataset(files_path, shuffled=False, tta=False, extension='jpg'):
    dataset = tf.data.Dataset.list_files(f'{files_path}*{extension}', shuffle=shuffled)
    dataset = dataset.map(process_path, num_parallel_calls=AUTO)
    dataset = dataset.map(data_augment, num_parallel_calls=AUTO)
    dataset = dataset.map(resize_image, num_parallel_calls=AUTO)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO)
    return dataset

## === cell 7
efficientnet_transforms = albumentations.Compose([
    albumentations.CenterCrop(IMAGE_SIZE, IMAGE_SIZE, p=1),
    albumentations.Resize(IMAGE_SIZE, IMAGE_SIZE),
    albumentations.Normalize(),
    ToTensorV2(p=1.0)
])

resnext_transforms = albumentations.Compose([
    albumentations.Resize(IMAGE_SIZE, IMAGE_SIZE),
    albumentations.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

vitEfficientnet_transforms = albumentations.Compose([
    albumentations.RandomResizedCrop(384, 384),
    albumentations.Transpose(p=0.5),
    albumentations.HorizontalFlip(p=0.5),
    albumentations.VerticalFlip(p=0.5),
    albumentations.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
    albumentations.RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
    albumentations.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
    ToTensorV2(p=1.0),
], p=1.)


def data_augment(image, label):
    p_spatial = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    p_rotate = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    p_pixel_1 = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    p_pixel_2 = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    p_pixel_3 = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    p_crop = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
            
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    if p_spatial > .75:
        image = tf.image.transpose(image)
        
    if p_rotate > .75:
        image = tf.image.rot90(image, k=3) # rotate 270º
    elif p_rotate > .5:
        image = tf.image.rot90(image, k=2) # rotate 180º
    elif p_rotate > .25:
        image = tf.image.rot90(image, k=1) # rotate 90º
        
    if p_pixel_1 >= .4:
        image = tf.image.random_saturation(image, lower=.7, upper=1.3)
    if p_pixel_2 >= .4:
        image = tf.image.random_contrast(image, lower=.8, upper=1.2)
    if p_pixel_3 >= .4:
        image = tf.image.random_brightness(image, max_delta=.1)
        
    if p_crop > .7:
        if p_crop > .9:
            image = tf.image.central_crop(image, central_fraction=.7)
        elif p_crop > .8:
            image = tf.image.central_crop(image, central_fraction=.8)
        else:
            image = tf.image.central_crop(image, central_fraction=.9)
    elif p_crop > .4:
        crop_size = tf.random.uniform([], int(IMAGE_SIZE*.8), IMAGE_SIZE, dtype=tf.int32)
        image = tf.image.random_crop(image, size=[crop_size, crop_size, 3])
    return image, label

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/282420146.py in <cell line: 0>()
      1 # Pytorch augmentations
----> 2 efficientnet_transforms = albumentations.Compose([
      3     albumentations.CenterCrop(IMAGE_SIZE, IMAGE_SIZE, p=1),
      4     albumentations.Resize(IMAGE_SIZE, IMAGE_SIZE),
      5     albumentations.Normalize(),

NameError: name 'albumentations' is not defined

## === cell 9
class TimmModel(nn.Module):
    def __init__(self, model_name, pretrained=False):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        n_features = self.model.fc.in_features
        self.model.fc = nn.Linear(n_features, CLASSES)

    def forward(self, x):
        x = self.model(x)
        return x
    
class enet_v2(nn.Module):
    def __init__(self, backbone, out_dim, pretrained=False):
        super(enet_v2, self).__init__()
        self.enet = timm.create_model(backbone, pretrained=pretrained)
        in_ch = self.enet.classifier.in_features
        self.myfc = nn.Linear(in_ch, out_dim)
        self.enet.classifier = nn.Identity()

    def forward(self, x):
        x = self.enet(x)
        x = self.myfc(x)
        return x
    
class CassavaImgClassifier(nn.Module):
    def __init__(self, model_arch, n_class, pretrained=False):
        super().__init__()
        self.model = timm.create_model(model_arch, pretrained=pretrained)
        n_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(n_features, n_class)
        
    def forward(self, x):
        x = self.model(x)
        return x
    
class ViTEfficientnetClassifier(nn.Module):
    def __init__(self, model_arch, n_class, pretrained=False):
        super().__init__()
        self.model1 = VisionTransformer.from_name('ViT-B_16', num_classes=5) 
        self.model1.load_state_dict(torch.load('../input/cassava-models/inputs/ViT/ViT-B_16.pt'))
        self.model2 = CassavaImgClassifier(model_arch, n_class, pretrained)
        
    def forward(self, x):
        x1 = self.model1(x)
        x2 = self.model2(x)
        return 0.6 * x1 + 0.4 * x2
    
    def load(self, state_dict):
        self.model2.load_state_dict(state_dict)
    
    
def model_fn(input_shape, n_class):
    inputs = L.Input(shape=input_shape, name='input_image')
    base_model = efn.EfficientNetB4(input_tensor=inputs, 
                                    include_top=False, 
                                    weights=None, 
                                    pooling='avg')
    x = L.Dropout(.5)(base_model.output)
    output = L.Dense(n_class, activation='softmax', name='output')(x)
    model = Model(inputs=inputs, outputs=output)

    return model

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2380070949.py in <cell line: 0>()
      1 # Pytorch models
----> 2 class TimmModel(nn.Module):
      3     def __init__(self, model_name, pretrained=False):
      4         super().__init__()
      5         self.model = timm.create_model(model_name, pretrained=pretrained)

NameError: name 'nn' is not defined

## === cell 11
class ResnextEfficientnet_CFG:
    resnext_model='resnext50_32x4d'
    efficientnet_model='tf_efficientnet_b4_ns'
    num_workers=8
    batch_size=32
    trn_fold=[0, 1, 2, 3, 4]
    efficientnet_model_path=glob.glob('../input/cassava-models/inputs/Efficientnet-B4-2/*')


def load_resnext_state(model_path):
    model = TimmModel(ResnextEfficientnet_CFG.resnext_model, pretrained=False)
    try:  # single GPU model_file
        model.load_state_dict(torch.load(model_path)['model'], strict=True)
        state_dict = torch.load(model_path)['model']
    except:  # multi GPU model_file
        state_dict = torch.load(model_path)['model']
        state_dict = {k[7:] if k.startswith('module.') else k: state_dict[k] for k in state_dict.keys()}

    return state_dict


def inference(model, states, test_loader, device):
    model.to(device)
    tk0 = tqdm(enumerate(test_loader), total=len(test_loader))
    probs = []
    for i, (images) in tk0:
        images = images.to(device)
        avg_preds = []
        for state in states:
            model.load_state_dict(state)
            model.eval()
            with torch.no_grad():
                y_preds = model(images)
            avg_preds.append(y_preds.softmax(1).to('cpu').numpy())
        avg_preds = np.mean(avg_preds, axis=0)
        probs.append(avg_preds)
    probs = np.concatenate(probs)
    return probs

def tta_inference_func(test_loader):
    model.eval()
    bar = tqdm(test_loader)
    PREDS = []
    LOGITS = []

    with torch.no_grad():
        for batch_idx, images in enumerate(bar):
            x = images.to(device)
            x = torch.stack([x,x.flip(-1),x.flip(-2),x.flip(-1,-2),
            x.transpose(-1,-2),x.transpose(-1,-2).flip(-1),
            x.transpose(-1,-2).flip(-2),x.transpose(-1,-2).flip(-1,-2)],0)
            x = x.view(-1, 3, IMAGE_SIZE, IMAGE_SIZE)
            logits = model(x)
            logits = logits.view(1, 8, -1).mean(1)
            PREDS += [torch.softmax(logits, 1).detach().cpu()]
            LOGITS.append(logits.cpu())

        PREDS = torch.cat(PREDS).cpu().numpy()
        
    return PREDS

test_dataset_efficient = PytorchCassavaDataset(transforms=efficientnet_transforms)
test_loader_efficient = torch.utils.data.DataLoader(test_dataset_efficient, batch_size=1, shuffle=False,  num_workers=4)
test_dataset_resnext = PytorchCassavaDataset(transforms=resnext_transforms)
test_loader_resnext = DataLoader(test_dataset_resnext, batch_size=ResnextEfficientnet_CFG.batch_size, shuffle=False, num_workers=ResnextEfficientnet_CFG.num_workers, pin_memory=True)

model = TimmModel(ResnextEfficientnet_CFG.resnext_model, pretrained=False)
states = [load_resnext_state(path) for path in glob.glob('../input/cassava-models/inputs/Resnext/*')]
resnext_predictions = inference(model, states, test_loader_resnext, device)

test_preds = []
for i in range(len(ResnextEfficientnet_CFG.efficientnet_model_path)):
    model = enet_v2(ResnextEfficientnet_CFG.efficientnet_model, out_dim=5)
    model = model.to(device)
    model.load_state_dict(torch.load(ResnextEfficientnet_CFG.efficientnet_model_path[i]))
    test_preds += [tta_inference_func(test_loader_efficient)]
efficientnet_predictions = np.mean(test_preds, axis=0)

test = pd.read_csv('../input/cassava-leaf-disease-classification/sample_submission.csv')
pred = 0.5*resnext_predictions + 0.5*efficientnet_predictions
test['label'] = softmax(pred).tolist()
resnext_b4_dict = dict(zip(list(test['image_id']), pred))

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1646413756.py in <cell line: 0>()
     61 
     62 # Create loaders
---> 63 test_dataset_efficient = PytorchCassavaDataset(transforms=efficientnet_transforms)
     64 test_loader_efficient = torch.utils.data.DataLoader(test_dataset_efficient, batch_size=1, shuffle=False,  num_workers=4)
     65 test_dataset_resnext = PytorchCassavaDataset(transforms=resnext_transforms)

NameError: name 'efficientnet_transforms' is not defined

## === cell 13
class ViTEfficientnet_CFG:
    efficientnet_model='tf_efficientnet_b3_ns'
    img_size=384
    valid_bs=32
    num_workers=4
    tta=1
    used_epochs=[6, 7, 8, 9]
        
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

test = pd.DataFrame()
test['image_id'] = list(os.listdir('../input/cassava-leaf-disease-classification/test_images/'))
test_ds = PytorchCassavaDataset(transforms=vitEfficientnet_transforms)

tst_loader = torch.utils.data.DataLoader(
    test_ds, 
    batch_size=ViTEfficientnet_CFG.valid_bs,
    num_workers=ViTEfficientnet_CFG.num_workers,
    shuffle=False,
    pin_memory=False,
)

model = ViTEfficientnetClassifier(ViTEfficientnet_CFG.efficientnet_model, CLASSES).to(device)

tst_preds = []
for i, epoch in enumerate(ViTEfficientnet_CFG.used_epochs):    
    model.load(torch.load('../input/cassava-models/inputs/Efficientnet-B3/{}_fold_{}_{}'.format(ViTEfficientnet_CFG.efficientnet_model, 0, epoch)))
    with torch.no_grad():
        for _ in range(ViTEfficientnet_CFG.tta):
            tst_preds += [inference_one_epoch(model, tst_loader, device)]
tst_preds = np.mean(tst_preds, axis=0)

vit_dict = dict(zip(list(os.listdir('../input/cassava-leaf-disease-classification/test_images/')), tst_preds))

del model
torch.cuda.empty_cache()

## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3195017092.py in <cell line: 0>()
     25 test = pd.DataFrame()
     26 test['image_id'] = list(os.listdir('../input/cassava-leaf-disease-classification/test_images/'))
---> 27 test_ds = PytorchCassavaDataset(transforms=vitEfficientnet_transforms)
     28 
     29 tst_loader = torch.utils.data.DataLoader(

NameError: name 'vitEfficientnet_transforms' is not defined

## === cell 15
TTA_STEPS = 3 # Do TTA if > 0

files_path = '/kaggle/input/cassava-leaf-disease-classification/test_images/'
test_size = len(os.listdir(files_path))
test_preds = np.zeros((test_size, CLASSES))

model_path_list = glob.glob('../input/cassava-models/inputs/Efficientnet-B4/*.h5')
model_path_list.sort()
model = model_fn((None, None, 3), CLASSES)

for model_path in model_path_list:
    print(model_path)
    K.clear_session()
    model.load_weights(model_path)

    if TTA_STEPS > 0:
        test_ds = get_dataset(files_path, tta=True).repeat()
        ct_steps = TTA_STEPS * ((test_size/BATCH_SIZE) + 1)
        preds = model.predict(test_ds, steps=ct_steps, verbose=1)[:(test_size * TTA_STEPS)]
        preds = np.mean(preds.reshape(test_size, TTA_STEPS, CLASSES, order='F'), axis=1)
        test_preds += preds / len(model_path_list)
    else:
        test_ds = get_dataset(files_path, tta=False)
        x_test = test_ds.map(lambda image, image_name: image)
        test_preds += model.predict(x_test) / len(model_path_list)
    
test_names_ds = get_dataset(files_path)
image_names = [img_name.numpy().decode('utf-8') for img, img_name in iter(test_names_ds.unbatch())]
b4_dict = dict(zip(image_names, test_preds))

vit_b3_b4_dict = {image_name:0.5*b4_dict[image_name]+0.5*vit_dict[image_name] for image_name in vit_dict.keys()}

## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2624783725.py in <cell line: 0>()
      8 model_path_list = glob.glob('../input/cassava-models/inputs/Efficientnet-B4/*.h5')
      9 model_path_list.sort()
---> 10 model = model_fn((None, None, 3), CLASSES)
     11 
     12 # Load model weights and make predictions

NameError: name 'model_fn' is not defined

## === cell 17
final_dict = {image_name:np.argmax(resnext_b4_dict[image_name]+vit_b3_b4_dict[image_name], axis=-1) for image_name in vit_b3_b4_dict.keys()}
submission_df = pd.DataFrame()
submission_df['image_id'] = final_dict.keys()
submission_df['label'] = final_dict.values()

## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/791992046.py in <cell line: 0>()
----> 1 final_dict = {image_name:np.argmax(resnext_b4_dict[image_name]+vit_b3_b4_dict[image_name], axis=-1) for image_name in vit_b3_b4_dict.keys()}
      2 submission_df = pd.DataFrame()
      3 submission_df['image_id'] = final_dict.keys()
      4 submission_df['label'] = final_dict.values()

NameError: name 'vit_b3_b4_dict' is not defined

## === cell 18
submission_df.to_csv('submission.csv', index=False)

## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1211055984.py in <cell line: 0>()
----> 1 submission_df.to_csv('submission.csv', index=False)

NameError: name 'submission_df' is not defined
