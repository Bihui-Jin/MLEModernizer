# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

3.13

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

# 5. Code solution

## === cell 0
import os
import glob
import torch
from torch.utils.data import DataLoader
from torchvision import models, transforms
from tqdm import tqdm
from PIL import Image
import pandas as pd

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)



## === cell 1
possible_test_dirs = glob.glob("/kaggle/**/test_images", recursive=True)
if possible_test_dirs:
    test_data_directory = possible_test_dirs[0]
else:
    test_data_directory = (
        "/kaggle/input/cassava-leaf-disease-classification/test_images"
    )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 5

en_model_path = "/kaggle/input/efficientnetv2-large-test/pytorch/default/4/efficientnet_v2_l_480_8591_ISP_CBP.pth"
en_image_size = 480

vit_model_path = (
    "/kaggle/input/vit_l_cassava/pytorch/default/5/vit_h_14_518_8718_CBP.pth"
)
vit_image_size = 518

torch.backends.cudnn.benchmark = True



## === cell 2
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]




## === cell 3
def locate_checkpoint(filename_substring):
    candidates = glob.glob("/kaggle/input/**/*.pth", recursive=True)
    for cand in candidates:
        if filename_substring in os.path.basename(cand):
            return cand
    return candidates[0] if candidates else None


en_model = models.efficientnet_v2_l(weights=None)
en_model.classifier[1] = torch.nn.Linear(
    en_model.classifier[1].in_features, num_classes
)
en_ckpt = (
    en_model_path
    if os.path.exists(en_model_path)
    else locate_checkpoint("efficientnet_v2_l")
)
if en_ckpt and os.path.exists(en_ckpt):
    try:
        state = torch.load(en_ckpt, map_location=device, weights_only=True)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        en_model.load_state_dict(state, strict=False)
    except Exception:
        en_model = models.efficientnet_v2_l(
            weights=models.EfficientNet_V2_L_Weights.DEFAULT
        )
        en_model.classifier[1] = torch.nn.Linear(
            en_model.classifier[1].in_features, num_classes
        )
else:
    en_model = models.efficientnet_v2_l(
        weights=models.EfficientNet_V2_L_Weights.DEFAULT
    )
    en_model.classifier[1] = torch.nn.Linear(
        en_model.classifier[1].in_features, num_classes
    )
en_model.to(device)
en_model.eval()
if hasattr(torch, "compile"):
    en_model = torch.compile(en_model)

vit_model = models.vit_h_14(weights=None, image_size=vit_image_size)
vit_model.heads.head = torch.nn.Linear(vit_model.heads.head.in_features, num_classes)
vit_ckpt = (
    vit_model_path if os.path.exists(vit_model_path) else locate_checkpoint("vit_h_14")
)
if vit_ckpt and os.path.exists(vit_ckpt):
    try:
        state = torch.load(vit_ckpt, map_location=device, weights_only=True)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        vit_model.load_state_dict(state, strict=False)
    except Exception:
        vit_model = models.vit_h_14(weights=models.ViT_H_14_Weights.DEFAULT)
        vit_model.heads.head = torch.nn.Linear(
            vit_model.heads.head.in_features, num_classes
        )
else:
    vit_model = models.vit_h_14(weights=models.ViT_H_14_Weights.DEFAULT)
    vit_model.heads.head = torch.nn.Linear(
        vit_model.heads.head.in_features, num_classes
    )
vit_model.to(device)
vit_model.eval()
if hasattr(torch, "compile"):
    vit_model = torch.compile(vit_model)




## === cell 4
class TestDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths):
        self.paths = image_paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        tensor = transforms.functional.to_tensor(img)

        en_tensor = torch.nn.functional.interpolate(
            tensor.unsqueeze(0),
            size=(en_image_size, en_image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        en_tensor = transforms.functional.normalize(
            en_tensor, imagenet_mean, imagenet_std
        )

        vit_tensor = torch.nn.functional.interpolate(
            tensor.unsqueeze(0),
            size=(vit_image_size, vit_image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        vit_tensor = transforms.functional.normalize(
            vit_tensor, imagenet_mean, imagenet_std
        )

        image_id = os.path.basename(path)
        return en_tensor, vit_tensor, image_id


image_paths = sorted(
    [
        p
        for p in glob.glob(os.path.join(test_data_directory, "**", "*"), recursive=True)
        if os.path.isfile(p) and p.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
)

num_workers = min(8, os.cpu_count() or 1)

dataset = TestDataset(image_paths)
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    prefetch_factor=2,
)

predictions = []
image_ids = []

for en_batch, vit_batch, ids_batch in tqdm(dataloader, desc="Test"):
    en_batch = en_batch.to(device, non_blocking=True)
    vit_batch = vit_batch.to(device, non_blocking=True)

    en_flip = torch.flip(en_batch, dims=[3])
    vit_flip = torch.flip(vit_batch, dims=[3])

    en_combined = torch.cat([en_batch, en_flip], dim=0)
    vit_combined = torch.cat([vit_batch, vit_flip], dim=0)

    with torch.no_grad():
        en_out = en_model(en_combined)
        en_out_orig, en_out_flip = torch.chunk(en_out, 2, dim=0)
        en_avg = (en_out_orig + en_out_flip) / 2.0

        vit_out = vit_model(vit_combined)
        vit_out_orig, vit_out_flip = torch.chunk(vit_out, 2, dim=0)
        vit_avg = (vit_out_orig + vit_out_flip) / 2.0

        ensemble_logits = (en_avg + vit_avg) / 2.0
        batch_preds = torch.argmax(ensemble_logits, dim=1)

    predictions.extend(batch_preds.cpu().tolist())
    image_ids.extend(ids_batch)

submission_df = pd.DataFrame({"image_id": image_ids, "label": predictions})
submission_path = os.path.join("/kaggle/working", "submission.csv")
submission_df.to_csv(submission_path, index=False)
print(f"Submission file created: {submission_path}")
