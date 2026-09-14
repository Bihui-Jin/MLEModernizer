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
import os, sys, glob, random, re, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

INPUT_DIR = "/kaggle/input/cassava-leaf-disease-classification"
TEST_IMG_DIR = f"{INPUT_DIR}/test_images"
SAMPLE_SUB_PATH = f"{INPUT_DIR}/sample_submission.csv"

assert os.path.exists(SAMPLE_SUB_PATH), f"Missing {SAMPLE_SUB_PATH}"
assert os.path.isdir(TEST_IMG_DIR), f"Missing {TEST_IMG_DIR}"



## === cell 1

import cv2

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from tqdm.auto import tqdm



## === cell 2
IMAGE_SIZE = 512
CLASSES = 5
SEED = 99

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


seed_everything(SEED)



## === cell 3
CASSAVA_MODELS_DIR = "/kaggle/input/cassava-models/inputs"
if not os.path.isdir("/kaggle/input/cassava-models"):
    raise FileNotFoundError(
        "This notebook expects the Kaggle dataset 'cassava-models' to be added as an input "
        "(path /kaggle/input/cassava-models). Please attach it in the Kaggle notebook UI."
    )




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/1933340574.py in <cell line: 0>()
      3 CASSAVA_MODELS_DIR = "/kaggle/input/cassava-models/inputs"
      4 if not os.path.isdir("/kaggle/input/cassava-models"):
----> 5     raise FileNotFoundError(
      6         "This notebook expects the Kaggle dataset 'cassava-models' to be added as an input "
      7         "(path /kaggle/input/cassava-models). Please attach it in the Kaggle notebook UI."

FileNotFoundError: This notebook expects the Kaggle dataset 'cassava-models' to be added as an input (path /kaggle/input/cassava-models). Please attach it in the Kaggle notebook UI.

## === cell 4
class PytorchCassavaDataset(Dataset):
    def __init__(self, df, data_root=TEST_IMG_DIR, transforms=None):
        super().__init__()
        self.df = df.reset_index(drop=True).copy()
        self.transforms = transforms
        self.data_root = data_root

    def get_img(self, path):
        im_bgr = cv2.imread(path)
        if im_bgr is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
        return im_rgb

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index: int):
        image_id = self.df.iloc[index]["image_id"]
        path = f"{self.data_root}/{image_id}"
        img = self.get_img(path)
        if self.transforms:
            img = self.transforms(image=img)["image"]
        return img




## === cell 5
pass



## === cell 6
efficientnet_transforms = A.Compose(
    [
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(),  # default mean/std for 0-255 -> 0-1 then normalize
        ToTensorV2(p=1.0),
    ]
)

resnext_transforms = A.Compose(
    [
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(p=1.0),
    ]
)


def data_augment(image, label):
    return image, label




## === cell 7
pass



## === cell 8


class TimmModel(nn.Module):
    def __init__(self, model_name, pretrained=False):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        n_features = self.model.fc.in_features
        self.model.fc = nn.Linear(n_features, CLASSES)

    def forward(self, x):
        return self.model(x)


class enet_v2(nn.Module):
    def __init__(self, backbone, out_dim, pretrained=False):
        super().__init__()
        self.enet = timm.create_model(backbone, pretrained=pretrained)
        in_ch = self.enet.classifier.in_features
        self.myfc = nn.Linear(in_ch, out_dim)
        self.enet.classifier = nn.Identity()

    def forward(self, x):
        x = self.enet(x)
        x = self.myfc(x)
        return x





## === cell 9
pass




## === cell 10
class ResnextEfficientnet_CFG:
    resnext_model = "resnext50_32x4d"
    efficientnet_model = "tf_efficientnet_b4_ns"
    num_workers = 2
    batch_size = 32

    resnext_model_glob = f"{CASSAVA_MODELS_DIR}/Resnext/*"
    efficientnet_model_glob = f"{CASSAVA_MODELS_DIR}/Efficientnet-B4-2/*"


def load_resnext_state(model_path):
    ckpt = torch.load(model_path, map_location="cpu")
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {
            k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()
        }
    return state_dict


def load_enet_state(model_path):
    ckpt = torch.load(model_path, map_location="cpu")
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {
            k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()
        }
    return state_dict


@torch.no_grad()
def inference_with_states(model, states, test_loader, device):
    model.to(device)
    probs = []
    for images in tqdm(test_loader, total=len(test_loader)):
        images = images.to(device)
        avg_preds = []
        for state in states:
            model.load_state_dict(state, strict=True)
            model.eval()
            y = model(images)
            avg_preds.append(y.softmax(1).detach().cpu().numpy())
        avg_preds = np.mean(avg_preds, axis=0)
        probs.append(avg_preds)
    return np.concatenate(probs, axis=0)


@torch.no_grad()
def tta_inference_func(model, test_loader, device):
    model.eval()
    preds = []
    for images in tqdm(test_loader, total=len(test_loader)):
        x = images.to(device)
        x = torch.stack(
            [
                x,
                x.flip(-1),
                x.flip(-2),
                x.flip((-1, -2)),
                x.transpose(-1, -2),
                x.transpose(-1, -2).flip(-1),
                x.transpose(-1, -2).flip(-2),
                x.transpose(-1, -2).flip((-1, -2)),
            ],
            0,
        )
        x = x.view(-1, 3, IMAGE_SIZE, IMAGE_SIZE)
        logits = model(x)
        logits = logits.view(1, 8, -1).mean(1)
        preds.append(torch.softmax(logits, 1).detach().cpu().numpy())
    return np.concatenate(preds, axis=0)


test_df = pd.read_csv(SAMPLE_SUB_PATH)
assert "image_id" in test_df.columns and len(test_df) > 0

test_dataset_efficient = PytorchCassavaDataset(
    test_df, transforms=efficientnet_transforms
)
test_loader_efficient = DataLoader(
    test_dataset_efficient, batch_size=1, shuffle=False, num_workers=2
)

test_dataset_resnext = PytorchCassavaDataset(test_df, transforms=resnext_transforms)
test_loader_resnext = DataLoader(
    test_dataset_resnext,
    batch_size=ResnextEfficientnet_CFG.batch_size,
    shuffle=False,
    num_workers=ResnextEfficientnet_CFG.num_workers,
    pin_memory=True,
)

resnext_paths = sorted(glob.glob(ResnextEfficientnet_CFG.resnext_model_glob))
efficient_paths = sorted(glob.glob(ResnextEfficientnet_CFG.efficientnet_model_glob))

if len(resnext_paths) == 0:
    raise FileNotFoundError(
        f"No ResNeXt weights found at: {ResnextEfficientnet_CFG.resnext_model_glob}"
    )
if len(efficient_paths) == 0:
    raise FileNotFoundError(
        f"No EfficientNet weights found at: {ResnextEfficientnet_CFG.efficientnet_model_glob}"
    )

resnext_model = TimmModel(ResnextEfficientnet_CFG.resnext_model, pretrained=False)
resnext_states = [load_resnext_state(p) for p in resnext_paths]
resnext_predictions = inference_with_states(
    resnext_model, resnext_states, test_loader_resnext, device
)

enet_preds = []
for p in efficient_paths:
    enet_model = enet_v2(
        ResnextEfficientnet_CFG.efficientnet_model, out_dim=CLASSES, pretrained=False
    ).to(device)
    enet_model.load_state_dict(load_enet_state(p), strict=True)
    enet_preds.append(tta_inference_func(enet_model, test_loader_efficient, device))
efficientnet_predictions = np.mean(enet_preds, axis=0)

pred = 0.5 * resnext_predictions + 0.5 * efficientnet_predictions

resnext_b4_dict = dict(zip(test_df["image_id"].tolist(), pred))



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/2774812503.py in <cell line: 0>()
    100 
    101 if len(resnext_paths) == 0:
--> 102     raise FileNotFoundError(
    103         f"No ResNeXt weights found at: {ResnextEfficientnet_CFG.resnext_model_glob}"
    104     )

FileNotFoundError: No ResNeXt weights found at: /kaggle/input/cassava-models/inputs/Resnext/*

## === cell 11
pass



## === cell 12
vit_dict = None



## === cell 13
pass



## === cell 14
vit_b3_b4_dict = None



## === cell 15
pass



## === cell 16
final_labels = [
    int(np.argmax(resnext_b4_dict[image_id], axis=-1))
    for image_id in test_df["image_id"].tolist()
]

submission_df = pd.DataFrame(
    {"image_id": test_df["image_id"].tolist(), "label": final_labels}
)
assert submission_df.shape[0] == test_df.shape[0]
assert submission_df["label"].between(0, CLASSES - 1).all()



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2001080009.py in <cell line: 0>()
      1 # Final prediction: argmax over blended logits/probabilities for each image_id in sample_submission order.
----> 2 final_labels = [
      3     int(np.argmax(resnext_b4_dict[image_id], axis=-1))
      4     for image_id in test_df["image_id"].tolist()
      5 ]

/tmp/ipykernel_55/2001080009.py in <listcomp>(.0)
      1 # Final prediction: argmax over blended logits/probabilities for each image_id in sample_submission order.
      2 final_labels = [
----> 3     int(np.argmax(resnext_b4_dict[image_id], axis=-1))
      4     for image_id in test_df["image_id"].tolist()
      5 ]

NameError: name 'resnext_b4_dict' is not defined

## === cell 17
submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)
print("Wrote", submission_path)
print(submission_df.head())

## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2725230928.py in <cell line: 0>()
      1 submission_path = "submission.csv"
----> 2 submission_df.to_csv(submission_path, index=False)
      3 print("Wrote", submission_path)
      4 print(submission_df.head())

NameError: name 'submission_df' is not defined
