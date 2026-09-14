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

0.8862194016319129

# 6. Current score

0.09193

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.09193) has done: 'I remove the failing dependency on a missing `../input/cassavalblsmoothingresnet50` checkpoint directory by switching to a timm pretrained backbone (same ResNet50 architecture) so inference can run end-to-end. I also fix a PyTorch API error in `F.softmax` (missing `dim`) and make the dataloader→batch unpacking deterministic (don’t rely on `dict.values()` order). Finally, I robustly locate the dataset path across the provided directory layout and always write a valid `submission.csv` with `image_id,label` columns.'

# 9. Code solution

## === cell 0
import random
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import timm
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.nn.functional as F

import albumentations as A
from albumentations import Compose
from albumentations.pytorch import ToTensorV2
import cv2



## === cell 1
import warnings

warnings.filterwarnings("ignore")



## === cell 2
sys.path.append("../input/pytorchimagemodelsmaster/pytorch-image-models-master")



## === cell 3
_CANDIDATE_DATA_PATHS = [
    "../input/cassava-leaf-disease-classification/",
    "/kaggle/input/cassava-leaf-disease-classification/",
    "/kaggle/data/cassava-leaf-disease-classification/",
]
DATA_PATH = None
for p in _CANDIDATE_DATA_PATHS:
    if os.path.exists(p) and os.path.exists(os.path.join(p, "sample_submission.csv")):
        DATA_PATH = p
        break
if DATA_PATH is None:
    raise FileNotFoundError(
        f"Could not find cassava dataset directory in candidates: {_CANDIDATE_DATA_PATHS}"
    )

NUM_FOLDS = 5
bs = 16
EPOCHS = 10
sz = 448
SNAPMIX_ALPHA = 5.0
SNAPMIX_PCT = 0.5
GRAD_ACCUM_STEPS = 1
TIMM_MODEL = "resnet50"




## === cell 4
def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


SEED = 1234
seed_everything(SEED)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




## === cell 5
def _assert_cv2_imread(path: str):
    im = cv2.imread(path)
    if im is None:
        raise FileNotFoundError(f"cv2.imread failed for path: {path}")
    return im




## === cell 6
class CassavaDataset(Dataset):
    def __init__(self, dataframe, root_dir, transforms=None):
        super().__init__()
        self.dataframe = dataframe.reset_index(drop=True)
        self.root_dir = root_dir
        self.transforms = transforms

    def __len__(self):
        return len(self.dataframe)

    def get_img_bgr_to_rgb(self, path):
        im_bgr = _assert_cv2_imread(path)
        im_rgb = im_bgr[:, :, ::-1]
        return im_rgb

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.dataframe.iloc[idx, 0])
        image = self.get_img_bgr_to_rgb(img_name)

        if self.transforms:
            image = self.transforms(image=image)["image"]

        label = self.dataframe.iloc[idx, 1] if "label" in self.dataframe.columns else -1
        sample = {"image": image, "label": int(label)}
        return sample




## === cell 7
def _collate_fn(batch):
    images = torch.stack([b["image"] for b in batch], dim=0)
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    return {"image": images, "label": labels}




## === cell 8
def test_transforms():
    return Compose(
        [
            A.Resize(sz, sz),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0,
            ),
            ToTensorV2(p=1.0),
        ],
        p=1.0,
    )




## === cell 10
class CassavaNet(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = timm.create_model(TIMM_MODEL, pretrained=True)
        n_features = backbone.fc.in_features
        self.backbone = nn.Sequential(*backbone.children())[:-2]
        self.classifier = nn.Linear(n_features, 5)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward_features(self, x):
        x = self.backbone(x)
        return x

    def forward(self, x):
        feats = self.forward_features(x)
        x = self.pool(feats).view(x.size(0), -1)
        x = self.classifier(x)
        return x, feats




## === cell 11
model = CassavaNet().to(device)



## === cell 13
def predict(model, ckpts, dataloader):
    if ckpts is None:
        ckpts = []
    use_ensemble = len(ckpts) > 0

    predict_list = []
    model.eval()

    with torch.no_grad():
        for data in tqdm(
            dataloader, total=len(dataloader), desc="Predict", leave=False
        ):
            images = data["image"].to(device, non_blocking=True)

            if use_ensemble:
                avg_preds = []
                for ckpt in ckpts:
                    sd = (
                        ckpt["state_dict"]
                        if isinstance(ckpt, dict) and "state_dict" in ckpt
                        else ckpt
                    )
                    model.load_state_dict(sd, strict=False)
                    model.eval()
                    outputs, _ = model(images)
                    preds = F.softmax(outputs, dim=1).detach().cpu().numpy()
                    avg_preds.append(preds)
                batch_probs = np.mean(avg_preds, axis=0)
            else:
                outputs, _ = model(images)
                batch_probs = F.softmax(outputs, dim=1).detach().cpu().numpy()

            predict_list.append(batch_probs)

    predict_probs = np.concatenate(predict_list, axis=0)
    return predict_probs.argmax(axis=1)




## === cell 15
test_df = pd.read_csv(os.path.join(DATA_PATH, "sample_submission.csv"))
test_dir = os.path.join(DATA_PATH, "test_images")

test_ds = CassavaDataset(
    dataframe=test_df, root_dir=test_dir, transforms=test_transforms()
)

test_dl = DataLoader(
    test_ds,
    batch_size=bs,
    shuffle=False,
    num_workers=4,  # safer across environments while still parallel
    pin_memory=torch.cuda.is_available(),
    collate_fn=_collate_fn,
)



## === cell 17
ckpts = []
trained_model_path = "../input/cassavalblsmoothingresnet50"
if os.path.exists(trained_model_path):
    for path in sorted(os.listdir(trained_model_path)):
        full_path = os.path.join(trained_model_path, path)
        if os.path.isfile(full_path):
            ckpts.append(torch.load(full_path, map_location="cpu"))



## === cell 19
test_predict_list = predict(model, ckpts, test_dl)



## === cell 20
test_predict_list



## === cell 22
test_df.head()



## === cell 23
test_df["label"] = test_predict_list.astype(int)
sub_path = "submission.csv"
test_df[["image_id", "label"]].to_csv(sub_path, index=False)
print(
    f"Wrote submission to: {sub_path}  rows={len(test_df)} cols={list(test_df[['image_id','label']].columns)}"
)
test_df.head()
