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

3.12

# 3. Installed packages

geopandas==0.14.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
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

0.8915080084617709

# 6. Current score

0.15022

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.15022) has done: 'I fix the model instantiation error by keeping ViT-B/16 at its pretrained-required `image_size=224` and aligning the preprocessing `img_size` to 224 so inference runs. Then I address the missing-predictions issue by filtering the test dataset file list to only valid image files (and matching the sample submission IDs), preventing mismatches caused by stray entries (e.g., nested folders) and ensuring every `image_id` gets a prediction. Finally, I make the TTA deterministic per-image index so results are stable under `cudnn.deterministic=True`, and the script always write a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import random

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
from torchvision.transforms import InterpolationMode, v2
import torchvision



## === cell 1
seed = 3407
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

cudnn.deterministic = True
cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

_CANDIDATE_TEST_DIRS = [
    "/kaggle/input/cassava-leaf-disease-classification/test_images/",
    "/kaggle/data/cassava-leaf-disease-classification/test_images/",
]
test_dir = None
for d in _CANDIDATE_TEST_DIRS:
    if os.path.isdir(d):
        test_dir = d
        break
if test_dir is None:
    raise FileNotFoundError(
        f"Could not find test_images in candidates: {_CANDIDATE_TEST_DIRS}"
    )
print("Using test_dir:", test_dir)

img_size = 224
batch_size = 16
num_workers = 4
num_classes = 5



## === cell 2
vit_model = torchvision.models.vit_b_16(
    weights=torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1
)
in_features = vit_model.heads.head.in_features
vit_model.heads.head = torch.nn.Linear(in_features, num_classes)
vit_model.to(device)




## === cell 3
class CassavaDataset(VisionDataset):
    """Custom dataset for the Cassava test data: returns TTA-augmented tensors + filename."""

    def __init__(self, data_dir, transform=None, ttas=None, image_ids=None):
        super().__init__(root=data_dir)
        self.transform = transform
        self.ttas = ttas or []

        valid_ext = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

        if image_ids is not None:
            self.images = [
                x for x in list(image_ids) if str(x).lower().endswith(valid_ext)
            ]
        else:
            self.images = sorted(
                [
                    f
                    for f in os.listdir(data_dir)
                    if os.path.isfile(os.path.join(data_dir, f))
                    and f.lower().endswith(valid_ext)
                ]
            )

        missing = [
            f for f in self.images if not os.path.isfile(os.path.join(self.root, f))
        ]
        if missing:
            raise FileNotFoundError(
                f"Some image_ids are missing in {self.root}, e.g. {missing[:5]}"
            )

    def __getitem__(self, idx):
        filename = self.images[idx]
        img_path = os.path.join(self.root, filename)
        img = Image.open(img_path).convert("RGB")

        if len(self.ttas) == 0:
            views = [self.transform(img)]
        else:
            cpu_state = torch.get_rng_state()
            py_state = random.getstate()
            np_state = np.random.get_state()

            per_item_seed = seed + int(idx)
            torch.manual_seed(per_item_seed)
            random.seed(per_item_seed)
            np.random.seed(per_item_seed)

            views = [self.transform(t(img)) for t in self.ttas]

            torch.set_rng_state(cpu_state)
            random.setstate(py_state)
            np.random.set_state(np_state)

        return torch.stack(views, dim=0), filename

    def __len__(self):
        return len(self.images)




## === cell 4
test_transforms = v2.Compose(
    [
        v2.Resize((img_size, img_size), interpolation=InterpolationMode.BICUBIC),
        v2.CenterCrop((img_size, img_size)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

ttas = [
    v2.RandomResizedCrop(
        (img_size, img_size), scale=(0.5, 1.0), interpolation=InterpolationMode.BICUBIC
    ),
    v2.RandomRotation(180, interpolation=InterpolationMode.BILINEAR),
    v2.RandomVerticalFlip(p=1.0),
    v2.RandomAffine(degrees=180, interpolation=InterpolationMode.BILINEAR),
    v2.RandomPerspective(
        distortion_scale=0.5, p=1.0, interpolation=InterpolationMode.BILINEAR
    ),
]

sample_path = "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
if not os.path.exists(sample_path):
    sample_path = (
        "/kaggle/data/cassava-leaf-disease-classification/sample_submission.csv"
    )
sample_sub = pd.read_csv(sample_path)

if list(sample_sub.columns) != ["image_id", "label"]:
    raise ValueError(
        f"Unexpected sample_submission columns: {sample_sub.columns.tolist()}"
    )

test_dataset = CassavaDataset(
    test_dir,
    transform=test_transforms,
    ttas=ttas,
    image_ids=sample_sub["image_id"].tolist(),
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
)

normalizer = torch.nn.Softmax(dim=1)



## === cell 5
all_names = []
all_preds = []

vit_model.eval()

with torch.no_grad():
    for batch_idx, (tta_views, filenames) in enumerate(test_loader):
        tta_views = tta_views.to(device, non_blocking=True)
        B, T, C, H, W = tta_views.shape

        inputs = tta_views.view(B * T, C, H, W)
        probs = normalizer(vit_model(inputs))  # (B*T, num_classes)

        probs = probs.view(B, T, num_classes).mean(dim=1)  # (B, num_classes)
        pred_labels = torch.argmax(probs, dim=1).tolist()

        all_names.extend(list(filenames))
        all_preds.extend(pred_labels)

assert len(all_names) == len(
    test_dataset
), f"Predictions ({len(all_names)}) != test images ({len(test_dataset)})"



## === cell 6
my_submission = pd.DataFrame({"image_id": all_names, "label": all_preds})

my_submission = sample_sub[["image_id"]].merge(my_submission, on="image_id", how="left")
if my_submission["label"].isna().any():
    missing = (
        my_submission.loc[my_submission["label"].isna(), "image_id"].head(20).tolist()
    )
    raise RuntimeError(
        f"Missing predictions for some images (showing up to 20): {missing}. "
        f"Check test_dir={test_dir} and dataset file filtering."
    )

my_submission["label"] = my_submission["label"].astype(int)

assert list(my_submission.columns) == ["image_id", "label"]
assert len(my_submission) == len(
    sample_sub
), f"Submission length {len(my_submission)} != expected {len(sample_sub)}"

my_submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with", len(my_submission), "rows")



## === cell 7
my_submission.head()
