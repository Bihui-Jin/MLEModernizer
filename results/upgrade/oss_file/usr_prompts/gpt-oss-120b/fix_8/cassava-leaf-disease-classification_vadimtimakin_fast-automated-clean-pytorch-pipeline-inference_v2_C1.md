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

0.7243880326382592

# 6. Current score

0.53176

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0938) has done: 'I fix the import error for `ToTensor` by using the correct Albumentations class `ToTensorV2`, adjust the custom `totensor` helper to return this class, and make the model load pretrained weights (which greatly improves accuracy). I also guard the checkpoint loading so the script runs even if the weight file is missing. These minimal changes resolve the runtime errors and should push the validation accuracy toward the target score.'
- What this solution (achieved 0.0938) has done: 'I correct the checkpoint path so the fine‑tuned weights are actually loaded, and I load the checkpoint before replacing the final layer – this preserves the trained classifier instead of overwriting it with random weights. These minimal adjustments let the model use the learned parameters and should raise the validation accuracy toward the target score.'
- What this solution (achieved 0.0938) has done: 'I adjust the model construction so the final classification layer is replaced **before** loading the checkpoint. This prevents the fine‑tuned weights from being overwritten by a random layer. The checkpoint is then loaded with `strict=False` to tolerate any minor key mismatches, ensuring the pretrained‑on‑cassava parameters are used and the validation accuracy moves toward the target score.'
- What this solution (achieved 0.1222) has done: 'I added a resize step (224 × 224) to the test augmentation pipeline and converted the OpenCV‑loaded BGR images to RGB before applying Albumentations, which aligns the input with the pretrained ImageNet weights and improves classification accuracy toward the target. The rest of the workflow and model loading remain unchanged, ensuring a valid submission.csv is still produced.'
- What this solution (achieved 0.11173) has done: 'Implemented missing imports, corrected configuration, built proper Albumentations transforms, added dataset loading, loader creation, and ensured the inference loop runs without errors. Fixed path handling, replaced the broken custom totensor with `ToTensorV2`, and restored pandas and tqdm usage. The script now creates a valid `submission.csv` ready for Kaggle submission.'
- What this solution (achieved 0.53176) has done: 'I adjust the model‑loading logic so that the fine‑tuned checkpoint (if present) is loaded **before** any modification of the final classification layer. This ensures the learned weights for the 5‑class head are retained, which should raise the validation accuracy substantially toward the target. The change adds a safe try/except to load with `strict=True` (fallback to `strict=False`), keeps the rest of the pipeline unchanged, and still writes a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import torch
import torch.nn as nn
import torchvision.models as models
import cv2
import pandas as pd
from tqdm.auto import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2


class cfg:
    """Main config."""

    NUMCLASSES = 5  # CONST
    seed = 42  # random seed

    _root = "/kaggle/input" if os.path.isdir("/kaggle/input") else ".."
    pathtoimgs = os.path.join(
        _root, "cassava-leaf-disease-classification", "test_images"
    )
    pathtocsv = os.path.join(
        _root, "cassava-leaf-disease-classification", "sample_submission.csv"
    )
    chk = os.path.join(_root, "cassava-leaf-disease-classification", "weights.pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Device
    modelname = "resnext101_32x8d"  # PyTorch model

    batchsize = 1  # BatchSize
    numworkers = 4  # Number of workers

    transforms = A.Compose(
        [
            A.Resize(height=224, width=224, p=1.0),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0,
            ),
            ToTensorV2(),
        ]
    )




## === cell 1
def get_model(cfg):
    """Get PyTorch model (load fine‑tuned checkpoint if available)."""
    model = getattr(models, cfg.modelname)(pretrained=True)

    if os.path.isfile(cfg.chk):
        cp = torch.load(cfg.chk, map_location=cfg.device)
        if isinstance(cp, dict) and "model" in cp:
            state_dict = cp["model"]
        else:
            state_dict = cp
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError:
            model.load_state_dict(state_dict, strict=False)

        for key in ["epoch", "trainloss", "valloss", "metric", "lr", "stopflag"]:
            if isinstance(cp, dict) and key in cp:
                setattr(cfg, key, cp[key])
    else:
        print(
            f"Checkpoint {cfg.chk} not found – using pretrained ImageNet weights only."
        )
        lastlayer_name = list(model._modules)[-1]
        lastlayer = getattr(model, lastlayer_name)
        if getattr(lastlayer, "out_features", None) != cfg.NUMCLASSES:
            setattr(
                model,
                lastlayer_name,
                nn.Linear(
                    in_features=lastlayer.in_features,
                    out_features=cfg.NUMCLASSES,
                    bias=True,
                ),
            )

    return model.to(cfg.device)




## === cell 2
class CassavaDataset(torch.utils.data.Dataset):
    """Dataset for loading test images with simple TTA (horizontal flip)."""

    def __init__(self, cfg, images, transforms):
        self.images = images  # list of image filenames
        self.transforms = transforms
        self.cfg = cfg

    def __getitem__(self, idx):
        img_path = os.path.join(self.cfg.pathtoimgs, self.images[idx])
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        transformed = self.transforms(image=img)
        img_tensor = transformed["image"]

        img_flipped = cv2.flip(img, 1)
        transformed_flipped = self.transforms(image=img_flipped)
        img_tensor_flipped = transformed_flipped["image"]

        return img_tensor, img_tensor_flipped

    def __len__(self):
        return len(self.images)




## === cell 3
def get_loader(cfg):
    """Create DataLoader for test images."""
    df = pd.read_csv(cfg.pathtocsv)
    image_list = df["image_id"].astype(str).tolist()

    dataset = CassavaDataset(cfg, image_list, cfg.transforms)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batchsize,
        shuffle=False,
        num_workers=cfg.numworkers,
        pin_memory=True,
    )
    return loader, image_list




## === cell 4
torch.cuda.empty_cache()
dataloader, image_ids = get_loader(cfg)
model = get_model(cfg)
model.eval()

preds = []
with torch.no_grad():
    for batch in tqdm(dataloader, desc="Predict"):
        img_orig, img_flip = batch  # each has shape [B, C, H, W]
        logits_orig = model(img_orig.to(cfg.device))
        logits_flip = model(img_flip.to(cfg.device))
        logits_avg = (logits_orig + logits_flip) / 2.0
        pred_class = int(torch.argmax(logits_avg, dim=1).cpu().item())
        preds.append(pred_class)




## === cell 5
df_sub = pd.read_csv(cfg.pathtocsv)
df_sub["label"] = preds
df_sub.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")
df_sub.head()
