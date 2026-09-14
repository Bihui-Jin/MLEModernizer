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

0.1403747355696585

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.10725) has done: 'I make the model loading robust by checking if the checkpoint file exists; if it does not, the code fall back to a pretrained ResNet34 (ImageNet weights) and keep the randomly‑initialized final classification layer. This prevents the FileNotFoundError and ensures `preds` is created, allowing the script to write a valid `submission.csv` without altering the overall pipeline logic.'
- What this solution (achieved 0.10164) has done: 'I keep the overall pipeline unchanged while making two small but impactful tweaks: (1) use a slightly stronger pretrained backbone (`resnet50` instead of `resnet34`) by updating the config, and (2) add a simple test‑time augmentation – averaging the model’s logits on the original image and its horizontal flip – before taking the argmax. These changes preserve the core logic and should raise the validation accuracy from 0.10725 toward the target 0.14037 without altering the submission format.'
- What this solution (achieved 0.10314) has done: 'I increase the input resolution from 256 to 320 pixels (a modest change that often improves CNN accuracy) and expand the test‑time augmentation to four deterministic views – original, horizontal flip, vertical flip, and a 90° rotation – averaging their logits before taking the argmax. These adjustments keep the overall pipeline, model architecture, and training logic unchanged while providing a realistic boost toward the target accuracy. The script still write a correct `submission.csv` file.'
- What this solution (achieved 0.10164) has done: 'I convert each loaded image from BGR (the OpenCV default) to RGB before applying the Albumentations transforms. The pretrained ResNet backbone expects RGB images, so this small fix should raise validation accuracy modestly and move the score closer to the target without altering the overall pipeline.'
- What this solution (achieved 0.10277) has done: 'I add one more deterministic test‑time augmentation (a 270° rotation) and include its predictions in the averaging step, which should modestly boost the validation accuracy and move the score closer to the target without altering the core model or training logic.'
- What this solution (achieved 0.09791) has done: 'I keep the overall pipeline unchanged but improve the test‑time inference slightly:  
1. Convert each model output to probabilities with a softmax before averaging (logits averaging can be less stable).  
2. Add a simple multi‑scale view by down‑sampling the 320 × 320 image to 256 × 256 and back to 320 × 320, then include its predictions in the ensemble.  
3. Include an extra combined horizontal‑vertical flip view.  
These deterministic augmentations are lightweight, preserve the core logic, and are expected to raise the validation accuracy toward the target without altering the submission format.'

# 9. Code solution

## === cell 0
import torch
import torchvision.models as models
import numpy as np
import pandas as pd
import random
import os
import cv2
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm.notebook import tqdm
import torch.nn.functional as F




## === cell 1
class cfg:
    """Main config."""

    NUMCLASSES = 5  # CONST
    seed = 42  # random seed

    pathtoimgs = "../input/cassava-leaf-disease-classification/test_images"
    train_pathtoimgs = "../input/cassava-leaf-disease-classification/train_images"
    pathtocsv = "../input/cassava-leaf-disease-classification/sample_submission.csv"
    train_csv = "../input/cassava-leaf-disease-classification/train.csv"
    chk = "../input/cassava/weights.pt"  # optional checkpoint

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelname = "resnet50"  # keep same backbone

    batchsize = 32
    numworkers = 4

    transforms = [
        dict(
            name="Resize",
            params=dict(height=320, width=320, p=1.0),
        ),
        dict(
            name="Normalize",
            params=dict(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0,
            ),
        ),
        dict(name="/custom/totensor", params=dict()),
    ]




## === cell 2
print(cfg.device)




## === cell 3
def totensor():
    """Return Albumentations ToTensorV2 transform."""
    return ToTensorV2()




## === cell 4
def fullseed(seed=42):
    """Sets the random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    os.environ["PYTHONHASHSEED"] = str(seed)


fullseed(cfg.seed)




## === cell 5
def get_model(cfg):
    """Get PyTorch model, loading checkpoint if it exists."""
    model = getattr(models, cfg.modelname)(pretrained=True)
    lastlayer = list(model._modules)[-1]
    setattr(
        model,
        lastlayer,
        nn.Linear(
            in_features=getattr(model, lastlayer).in_features,
            out_features=cfg.NUMCLASSES,
            bias=True,
        ),
    )
    if os.path.exists(cfg.chk):
        cp = torch.load(cfg.chk, map_location=cfg.device)
        if isinstance(cp, dict) and "model" in cp:
            model.load_state_dict(cp["model"])
        elif isinstance(cp, dict):
            model.load_state_dict(cp)
        else:
            model.load_state_dict(cp)
    return model.to(cfg.device)


def get_transforms(cfg):
    """Get train / test Albumentations transforms."""
    transforms = [
        (
            globals()[item["name"][8:]](**item["params"])
            if item["name"].startswith("/custom/")
            else getattr(A, item["name"])(**item["params"])
        )
        for item in cfg.transforms
    ]
    return A.Compose(transforms)




## === cell 6
class CassavaDataset(torch.utils.data.Dataset):
    """Dataset for inference (images only)."""

    def __init__(self, cfg, images, transforms):
        self.images = images
        self.transforms = transforms
        self.cfg = cfg

    def __getitem__(self, idx):
        img_path = os.path.join(self.cfg.pathtoimgs, self.images[idx])
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transforms(image=img)["image"]
        return img

    def __len__(self):
        return len(self.images)


class CassavaTrainDataset(torch.utils.data.Dataset):
    """Dataset for training / validation (image + label)."""

    def __init__(self, cfg, images, labels, transforms):
        self.images = images
        self.labels = labels
        self.transforms = transforms
        self.cfg = cfg

    def __getitem__(self, idx):
        img_path = os.path.join(self.cfg.train_pathtoimgs, self.images[idx])
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transforms(image=img)["image"]
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return img, label

    def __len__(self):
        return len(self.images)


def get_loader(cfg):
    """Dataloader for test (no shuffling)."""
    data = pd.read_csv(cfg.pathtocsv)
    imgs = list(data["image_id"])
    transforms = get_transforms(cfg)
    dataset = CassavaDataset(cfg, imgs, transforms)
    return torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        batch_size=cfg.batchsize,
        pin_memory=True,
        num_workers=cfg.numworkers,
    )


def get_train_val_loaders(cfg, split_ratio=0.9):
    """Create train / validation loaders from train.csv."""
    df = pd.read_csv(cfg.train_csv)
    images = df["image_id"].tolist()
    labels = df["label"].tolist()
    transforms = get_transforms(cfg)

    n_total = len(images)
    n_train = int(split_ratio * n_total)
    n_val = n_total - n_train
    train_imgs, val_imgs = images[:n_train], images[n_train:]
    train_lbls, val_lbls = labels[:n_train], labels[n_train:]

    train_dataset = CassavaTrainDataset(cfg, train_imgs, train_lbls, transforms)
    val_dataset = CassavaTrainDataset(cfg, val_imgs, val_lbls, transforms)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=cfg.batchsize,
        pin_memory=True,
        num_workers=cfg.numworkers,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=cfg.batchsize,
        pin_memory=True,
        num_workers=cfg.numworkers,
    )
    return train_loader, val_loader




## === cell 7
model = get_model(cfg)
model.train()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

train_loader, val_loader = get_train_val_loaders(cfg, split_ratio=0.9)

best_val_acc = 0.0
best_state = None
NUM_EPOCHS = 2  # keep it short to stay within time limits

for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    running_loss = 0.0
    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch} [train]"):
        imgs = imgs.to(cfg.device)
        labels = labels.to(cfg.device)

        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in tqdm(val_loader, desc=f"Epoch {epoch} [val]"):
            imgs = imgs.to(cfg.device)
            labels = labels.to(cfg.device)
            logits = model(imgs)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_acc = correct / total
    print(f"Epoch {epoch}: train loss={epoch_loss:.4f}, val acc={val_acc:.4f}")
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_state = model.state_dict()

if best_state is not None:
    model.load_state_dict(best_state)




## === cell 8
torch.cuda.empty_cache()
dataloader = get_loader(cfg)
model.eval()

preds = []
with torch.no_grad():
    for img in tqdm(dataloader, desc="Test inference"):
        img = img.to(cfg.device)  # (1, C, H, W)

        img_hflip = torch.flip(img, dims=[-1])
        img_vflip = torch.flip(img, dims=[-2])
        img_rot90 = torch.rot90(img, k=1, dims=[-2, -1])
        img_rot270 = torch.rot90(img, k=3, dims=[-2, -1])
        img_hvflip = torch.flip(img_hflip, dims=[-2])

        img_small = F.interpolate(
            img, size=(256, 256), mode="bilinear", align_corners=False
        )
        img_scale = F.interpolate(
            img_small, size=(320, 320), mode="bilinear", align_corners=False
        )

        outs = [
            model(img),
            model(img_hflip),
            model(img_vflip),
            model(img_rot90),
            model(img_rot270),
            model(img_hvflip),
            model(img_scale),
        ]

        probs = [F.softmax(o, dim=1) for o in outs]
        avg_prob = torch.stack(probs, dim=0).mean(dim=0)  # (1, NUMCLASSES)

        preds.append(torch.argmax(avg_prob, dim=1).cpu().item())




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/3655188727.py in <cell line: 0>()
     38         avg_prob = torch.stack(probs, dim=0).mean(dim=0)  # (1, NUMCLASSES)
     39 
---> 40         preds.append(torch.argmax(avg_prob, dim=1).cpu().item())
     41 
     42 

RuntimeError: a Tensor with 32 elements cannot be converted to Scalar

## === cell 9
df = pd.read_csv(cfg.pathtocsv)
df["label"] = preds
df.to_csv("submission.csv", index=False)
print("submission.csv written, first rows:")
print(df.head())

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/3946604480.py in <cell line: 0>()
      3 # ------------------------------------------------------------------
      4 df = pd.read_csv(cfg.pathtocsv)
----> 5 df["label"] = preds
      6 df.to_csv("submission.csv", index=False)
      7 print("submission.csv written, first rows:")

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __setitem__(self, key, value)
   4309         else:
   4310             # set column
-> 4311             self._set_item(key, value)
   4312 
   4313     def _setitem_slice(self, key: slice, value) -> None:

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in _set_item(self, key, value)
   4522         ensure homogeneity.
   4523         """
-> 4524         value, refs = self._sanitize_column(value)
   4525 
   4526         if (

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in _sanitize_column(self, value)
   5264 
   5265         if is_list_like(value):
-> 5266             com.require_length_match(value, self.index)
   5267         arr = sanitize_array(value, self.index, copy=True, allow_2d=True)
   5268         if (

/usr/local/lib/python3.11/dist-packages/pandas/core/common.py in require_length_match(data, index)
    571     """
    572     if len(data) != len(index):
--> 573         raise ValueError(
    574             "Length of values "
    575             f"({len(data)}) "

ValueError: Length of values (0) does not match length of index (2676)
