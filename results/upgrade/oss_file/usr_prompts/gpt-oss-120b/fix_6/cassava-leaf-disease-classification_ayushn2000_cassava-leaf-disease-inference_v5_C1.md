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
scikit-image==0.25.2
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

0.819431852523421

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
package_path = "../input/pytorch-image-dataset"
import sys

sys.path.append(package_path)




## === cell 1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
import albumentations.pytorch as Apy

from glob import glob
import os
import time
import random
import cv2
import warnings
import timm

from tqdm import tqdm
from datetime import datetime
from skimage import io
from sklearn.model_selection import GroupKFold, StratifiedKFold

import torch
import torchvision
from torch import nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler
from torch.cuda.amp import autocast, GradScaler

import sklearn
from sklearn.metrics import roc_auc_score, log_loss
from sklearn import metrics
from sklearn.metrics import log_loss




## === cell 2
config = {
    "fold_num": 5,
    "seed": 719,
    "model_arch": "vit_base_resnet50d_224",
    "img_size": 224,
    "resize_to": 224,
    "epochs": 3,
    "train_bs": 32,
    "valid_bs": 32,
    "lr": 1e-4,
    "num_workers": 4,
    "accum_iter": 1,
    "verbose_step": 1,
    "device": "cuda:0" if torch.cuda.is_available() else "cpu",
    "tta": 3,
    "used_epochs": [0, 1, 2],
    "weights": [1, 1, 1, 1],
}




## === cell 3
submission = pd.read_csv(
    "../input/cassava-leaf-disease-classification/sample_submission.csv"
)
submission.head()




## === cell 4
def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_img(path):
    """Read an image with OpenCV and convert BGR→RGB.
    Raises a clear error if the file cannot be read."""
    im_bgr = cv2.imread(path)
    if im_bgr is None:
        raise FileNotFoundError(f"Image not found or cannot be read: {path}")
    return im_bgr[:, :, ::-1]




## === cell 5
class CassavaDataset(Dataset):
    def __init__(self, df, data_root, transforms=None, output_label=True):
        super().__init__()
        self.resize_image = torchvision.transforms.Resize(
            size=(config["resize_to"], config["resize_to"])
        )
        self.df = df.reset_index(drop=True).copy()
        self.transforms = transforms
        self.data_root = data_root
        self.output_label = output_label

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index: int):
        if self.output_label:
            target = self.df.iloc[index]["label"]

        path = f"{self.data_root}/{self.df.iloc[index]['image_id']}"
        img = get_img(path)

        if self.transforms:
            img = self.transforms(image=img)["image"]

        if self.output_label:
            return img, target
        else:
            return img




## === cell 6
def get_inference_transforms():
    return A.Compose(
        [
            A.RandomResizedCrop(
                size=(config["img_size"], config["img_size"]),
                scale=(0.8, 1.0),
                ratio=(0.75, 1.33),
            ),
            A.Transpose(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.HueSaturationValue(
                hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5
            ),
            A.RandomBrightnessContrast(
                brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5
            ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0,
            ),
            Apy.ToTensorV2(p=1.0),
        ],
        p=1.0,
    )




## === cell 7
class CassvaImgClassifier(nn.Module):
    """
    Simple wrapper that creates the requested Vision Transformer model via timm.create_model.
    This avoids the deprecated `hybrid_backbone` argument that caused a TypeError.
    """

    def __init__(self, model_arch, n_class, pretrained=False):
        super().__init__()
        self.model = timm.create_model(
            model_arch, pretrained=pretrained, num_classes=n_class
        )

    def forward(self, x):
        return self.model(x)




## === cell 8
def inference_one_epoch(model, data_loader, device):
    model.eval()

    image_preds_all = []

    pbar = tqdm(enumerate(data_loader), total=len(data_loader))
    for step, imgs in pbar:  # corrected unpacking
        imgs = imgs.to(device).float()

        image_preds = model(imgs)
        image_preds_all += [torch.softmax(image_preds, 1).detach().cpu().numpy()]

    image_preds_all = np.concatenate(image_preds_all, axis=0)
    return image_preds_all




## === cell 9
test_dir = "../input/cassava-leaf-disease-classification/test_images/"
test_files = [
    f for f in os.listdir(test_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

test = pd.DataFrame()
test["image_id"] = test_files

test_ds = CassavaDataset(
    test,
    test_dir,
    transforms=get_inference_transforms(),
    output_label=False,
)

tst_loader = torch.utils.data.DataLoader(
    test_ds,
    batch_size=config["valid_bs"],
    num_workers=config["num_workers"],
    shuffle=False,
    pin_memory=False,
)

device = torch.device(config["device"])

model = CassvaImgClassifier(config["model_arch"], 5, pretrained=True).to(device)

tst_preds = []

for fold in range(config["fold_num"]):
    for i, epoch in enumerate(config["used_epochs"]):
        weight_path = f'../input/cassava-leaf-disease-classification/{config["model_arch"]}_fold_{fold}_{epoch}.pth'
        if os.path.exists(weight_path):
            model.load_state_dict(torch.load(weight_path, map_location=device))
        else:
            continue
        model.eval()
        with torch.no_grad():
            for _ in range(config["tta"]):
                preds = inference_one_epoch(model, tst_loader, device)
                weight = config["weights"][i] / sum(config["weights"]) / config["tta"]
                tst_preds.append(weight * preds)

if len(tst_preds) == 0:
    with torch.no_grad():
        tst_preds = inference_one_epoch(model, tst_loader, device)
else:
    tst_preds = np.mean(tst_preds, axis=0)

del model
torch.cuda.empty_cache()




## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/3278212097.py in <cell line: 0>()
     26 
     27 # Initialise the model with pretrained ImageNet weights (helps accuracy).
---> 28 model = CassvaImgClassifier(config["model_arch"], 5, pretrained=True).to(device)
     29 
     30 tst_preds = []

/tmp/ipykernel_55/3588253157.py in __init__(self, model_arch, n_class, pretrained)
      7     def __init__(self, model_arch, n_class, pretrained=False):
      8         super().__init__()
----> 9         self.model = timm.create_model(
     10             model_arch, pretrained=pretrained, num_classes=n_class
     11         )

/usr/local/lib/python3.11/dist-packages/timm/models/_factory.py in create_model(model_name, pretrained, pretrained_cfg, pretrained_cfg_overlay, checkpoint_path, cache_dir, scriptable, exportable, no_jit, **kwargs)
    136     create_fn = model_entrypoint(model_name)
    137     with set_layer_config(scriptable=scriptable, exportable=exportable, no_jit=no_jit):
--> 138         model = create_fn(
    139             pretrained=pretrained,
    140             pretrained_cfg=pretrained_cfg,

/usr/local/lib/python3.11/dist-packages/timm/models/vision_transformer_hybrid.py in vit_base_resnet50d_224(pretrained, **kwargs)
    373     backbone = resnet50d(pretrained=pretrained, in_chans=kwargs.get('in_chans', 3), features_only=True, out_indices=[4])
    374     model_args = dict(embed_dim=768, depth=12, num_heads=12)
--> 375     model = _create_vision_transformer_hybrid(
    376         'vit_base_resnet50d_224', backbone=backbone, pretrained=pretrained, **dict(model_args, **kwargs))
    377     return model

/usr/local/lib/python3.11/dist-packages/timm/models/vision_transformer_hybrid.py in _create_vision_transformer_hybrid(variant, backbone, embed_args, pretrained, **kwargs)
    134     kwargs.setdefault('embed_layer', embed_layer)
    135     kwargs.setdefault('patch_size', 1)  # default patch size for hybrid models if not set
--> 136     return build_model_with_cfg(
    137         VisionTransformer,
    138         variant,

/usr/local/lib/python3.11/dist-packages/timm/models/_builder.py in build_model_with_cfg(model_cls, variant, pretrained, pretrained_cfg, pretrained_cfg_overlay, model_cfg, feature_cfg, pretrained_strict, pretrained_filter_fn, cache_dir, kwargs_filter, **kwargs)
    455     num_classes_pretrained = 0 if features else getattr(model, 'num_classes', kwargs.get('num_classes', 1000))
    456     if pretrained:
--> 457         load_pretrained(
    458             model,
    459             pretrained_cfg=pretrained_cfg,

/usr/local/lib/python3.11/dist-packages/timm/models/_builder.py in load_pretrained(model, pretrained_cfg, num_classes, in_chans, filter_fn, strict, cache_dir)
    234     else:
    235         model_name = pretrained_cfg.get('architecture', 'this model')
--> 236         raise RuntimeError(f"No pretrained weights exist for {model_name}. Use `pretrained=False` for random init.")
    237 
    238     if filter_fn is not None:

RuntimeError: No pretrained weights exist for vit_base_resnet50d_224. Use `pretrained=False` for random init.

## === cell 10
test["label"] = np.argmax(tst_preds, axis=1)
test.to_csv("submission.csv", index=False)

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4111679786.py in <cell line: 0>()
----> 1 test["label"] = np.argmax(tst_preds, axis=1)
      2 test.to_csv("submission.csv", index=False)

NameError: name 'tst_preds' is not defined
