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

0.10725

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.10762) has done: 'I fix the dtype mismatch that caused the model to crash by ensuring the dataset returns a float‐tensor (the model expects FloatTensor, not ByteTensor). This also guarantees that predictions are collected, so the submission dataframe gets the correct length. The change is limited to the dataset’s `__getitem__` method, preserving all other logic and model architecture.'
- What this solution (achieved 0.10762) has done: 'I add a BGR‑>RGB conversion in the dataset and include standard ImageNet normalization in the transform pipeline so the model receives inputs in the format it was trained on. These small, targeted changes should raise the validation accuracy toward the target score without altering the core architecture or training logic.'
- What this solution (achieved 0.10688) has done: 'The change switches the ResNet34 model to use ImageNet‑pretrained weights (instead of random initialization). Keeping the same architecture and all other logic unchanged, this modest improvement usually raises classification accuracy and moves the validation score nearer to the target 0.14037.'
- What this solution (achieved 0.10688) has done: 'I adjust the test‑time image size to match the ResNet‑34 ImageNet pre‑training resolution (224 × 224). This tiny change aligns the input distribution with the weights the model was trained on, which usually yields a modest boost in accuracy and moves the validation score closer to the target without altering any core architecture or training logic.'
- What this solution (achieved 0.10688) has done: 'I adjust the checkpoint path so that the model loads the fine‑tuned weights that are included with the competition data (instead of the missing placeholder path). This small fix lets the pretrained ResNet34 use the learned final layer, which should raise the validation accuracy toward the target without altering any core architecture or training logic. The rest of the code remains unchanged.'
- What this solution (achieved 0.108) has done: 'We prevent the CUDA out‑of‑memory crash by forcing inference onto the CPU and wrapping the forward passes in a `torch.no_grad()` block, which also clears unused tensors. The rest of the logic (model architecture, transforms, TTA) stays unchanged, so the score impact is minimal while guaranteeing a valid `submission.csv` is written.'
- What this solution (achieved 0.10688) has done: 'Implemented an additional vertical‑flip test‑time augmentation and adjusted the prediction aggregation to average logits from the original, horizontally‑flipped, and vertically‑flipped versions. This small, targeted change preserves the original model architecture and training logic while providing a modest boost in validation accuracy, moving the score closer to the target without altering core functionality.'
- What this solution (achieved 0.10725) has done: 'I add a fourth test‑time augmentation (a full 90° rotation) and include its predictions in the averaging step. This small change keeps the same model and training logic while giving the classifier more varied views of each leaf, which should raise the validation accuracy toward the target score.'

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
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm.notebook import tqdm
import copy




## === cell 1
class cfg:
    """Main config."""

    NUMCLASSES = 5
    seed = 42
    pathtoimgs = "../input/cassava-leaf-disease-classification/test_images"
    pathtocsv = "../input/cassava-leaf-disease-classification/sample_submission.csv"
    chk = "../input/cassava-leaf-disease-classification/weights.pt"
    device = torch.device("cpu")
    modelname = "resnet34"
    batchsize = 1
    numworkers = 4
    transforms = [
        dict(
            name="Resize",
            params=dict(height=224, width=224, p=1.0),
        ),
        dict(
            name="Normalize",
            params=dict(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
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
    """Custom wrapper that returns Albumentations' ToTensorV2."""
    return ToTensorV2()




## === cell 4
def fullseed(seed=42):
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
    try:
        cp = torch.load(cfg.chk, map_location=cfg.device)
        if "model" in cp:
            model.load_state_dict(cp["model"])
        else:
            model.load_state_dict(cp)
        if "lr" in cp:
            cfg.lr = cp["lr"]
        if "stopflag" in cp:
            cfg.stopflag = cp["stopflag"]
    except FileNotFoundError:
        pass
    return model.to(cfg.device)


def get_transforms(cfg):
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
        transformed = self.transforms(image=img)
        tensor_img = transformed["image"]
        if isinstance(tensor_img, torch.Tensor):
            tensor_img = tensor_img.float()
        else:
            tensor_img = torch.from_numpy(tensor_img).permute(2, 0, 1).float()
        return tensor_img

    def __len__(self):
        return len(self.images)




## === cell 7
def get_loader(cfg):
    data = pd.read_csv(cfg.pathtocsv)
    imgs = list(data["image_id"])
    transforms = get_transforms(cfg)
    dataset = CassavaDataset(cfg, imgs, transforms)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        batch_size=cfg.batchsize,
        pin_memory=True,
        num_workers=cfg.numworkers,
    )
    return dataloader




## === cell 8
torch.cuda.empty_cache()
loader_orig = get_loader(cfg)
model = get_model(cfg)
model.eval()

logits_orig = []
with torch.no_grad():
    for img in tqdm(loader_orig, desc="Predict original"):
        outputs = model(img.to(cfg.device))
        logits_orig.append(outputs.squeeze(0).cpu())

flip_cfg = copy.deepcopy(cfg)
flip_cfg.transforms = [
    dict(name="HorizontalFlip", params=dict(p=1.0))
] + flip_cfg.transforms
flip_transform = get_transforms(flip_cfg)

flip_imgs = list(pd.read_csv(cfg.pathtocsv)["image_id"])
flip_dataset = CassavaDataset(cfg, flip_imgs, flip_transform)
loader_flip = torch.utils.data.DataLoader(
    flip_dataset,
    shuffle=False,
    batch_size=cfg.batchsize,
    pin_memory=True,
    num_workers=cfg.numworkers,
)

logits_flip = []
with torch.no_grad():
    for img in tqdm(loader_flip, desc="Predict flipped"):
        outputs = model(img.to(cfg.device))
        logits_flip.append(outputs.squeeze(0).cpu())

vert_cfg = copy.deepcopy(cfg)
vert_cfg.transforms = [
    dict(name="VerticalFlip", params=dict(p=1.0))
] + vert_cfg.transforms
vert_transform = get_transforms(vert_cfg)

vert_imgs = list(pd.read_csv(cfg.pathtocsv)["image_id"])
vert_dataset = CassavaDataset(cfg, vert_imgs, vert_transform)
loader_vert = torch.utils.data.DataLoader(
    vert_dataset,
    shuffle=False,
    batch_size=cfg.batchsize,
    pin_memory=True,
    num_workers=cfg.numworkers,
)

logits_vert = []
with torch.no_grad():
    for img in tqdm(loader_vert, desc="Predict vertically flipped"):
        outputs = model(img.to(cfg.device))
        logits_vert.append(outputs.squeeze(0).cpu())

rot_cfg = copy.deepcopy(cfg)
rot_cfg.transforms = [
    dict(name="RandomRotate90", params=dict(p=1.0))
] + rot_cfg.transforms
rot_transform = get_transforms(rot_cfg)

rot_imgs = list(pd.read_csv(cfg.pathtocsv)["image_id"])
rot_dataset = CassavaDataset(cfg, rot_imgs, rot_transform)
loader_rot = torch.utils.data.DataLoader(
    rot_dataset,
    shuffle=False,
    batch_size=cfg.batchsize,
    pin_memory=True,
    num_workers=cfg.numworkers,
)

logits_rot = []
with torch.no_grad():
    for img in tqdm(loader_rot, desc="Predict rotated"):
        outputs = model(img.to(cfg.device))
        logits_rot.append(outputs.squeeze(0).cpu())

preds = []
for lo, lh, lv, lr in zip(logits_orig, logits_flip, logits_vert, logits_rot):
    avg_logit = (lo + lh + lv + lr) / 4.0
    preds.append(int(torch.argmax(avg_logit).item()))

print(preds)



## === cell 9
df = pd.read_csv(cfg.pathtocsv)
df["label"] = preds
df.to_csv("submission.csv", index=False)
print(df.head())
