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

0.8934723481414325

# 6. Current score

0.19357

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.19283) has done: 'I fix the immediate runtime error by making the ViT input resolution consistent with the model’s expected `image_size` (224 for `vit_b_16`), while keeping the same inference logic and TTA structure. I also make the TTA deterministic by switching to fixed transforms (random rotations/perspective were producing stochastic, non-reproducible predictions) without changing the overall approach. Finally, I fix the submission mapping bug by filling missing mapped labels before casting to `int`, ensuring a valid `submission.csv` is always written in the required format.'
- What this solution (achieved 0.19357) has done: 'I fix the `IsADirectoryError` by filtering `os.listdir()` to include only image files (and/or by using the competition’s `sample_submission.csv` as the canonical ordered list of test images), since the directory contains a nested `test_images/` folder that the current dataset tries to open as an image. I also add a safe fallback to read filenames from `sample_submission.csv` to guarantee perfect alignment and row count for submission. These changes are score-neutral but unblock end-to-end execution; the low current score is likely due to a mismatched/untrained checkpoint, but without changing core modeling we focus on correctness and producing a valid `.csv`.'
- What this solution (achieved 0.19357) has done: 'Your current score (0.19357) is far below the target (0.89347), so we need a real accuracy lift while keeping the same core ViT inference pipeline. The biggest likely issue is that `torch.load()` is returning a checkpoint *state_dict* (or a dict containing it), but the code is treating it as a ready-to-run model—this typically yields random/untrained heads or incorrect weights, producing very low accuracy. I minimally change checkpoint loading to correctly reconstruct `vit_b_16` and load weights when the checkpoint is a `state_dict`/dict, while preserving your transforms, TTA, and prediction logic. I also ensure we load on CPU first (safer across formats) and then move to GPU, without changing evaluation semantics.'

# 9. Code solution

## === cell 0
import os

import pandas as pd
import torch
from PIL import Image
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
from torchvision.transforms import InterpolationMode, v2



## === cell 1
torch.manual_seed(3407)
torch.cuda.manual_seed(3407)

cudnn.deterministic = True
cudnn.benchmark = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

test_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images/"

img_size = 224

batch_size = 16
num_workers = 4
num_classes = 5
tta = True

ckpt_candidates = [
    "/kaggle/input/vit-v1-update/vit_v1_1.pt",
    "/kaggle/input/vit-v1-update/vit_v1_1.pth",
    "/kaggle/input/vit-v1-update/vit_v1_1.bin",
]

from torchvision.models import vit_b_16, ViT_B_16_Weights

weights = ViT_B_16_Weights.IMAGENET1K_V1
vit_model = vit_b_16(weights=weights)
vit_model.heads.head = torch.nn.Linear(vit_model.heads.head.in_features, num_classes)

loaded_any = False
for p in ckpt_candidates:
    if not os.path.exists(p):
        continue

    obj = torch.load(p, map_location="cpu")

    if isinstance(obj, torch.nn.Module):
        vit_model = obj
        loaded_any = True
        print(f"Loaded full model from: {p}")
        break

    state_dict = None
    if isinstance(obj, dict):
        for k in ["state_dict", "model_state_dict", "model", "net", "weights"]:
            if k in obj and isinstance(obj[k], dict):
                state_dict = obj[k]
                break
        if state_dict is None:
            if len(obj) > 0 and all(hasattr(v, "shape") for v in obj.values()):
                state_dict = obj

    if state_dict is not None:
        if any(key.startswith("module.") for key in state_dict.keys()):
            state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

        missing, unexpected = vit_model.load_state_dict(state_dict, strict=False)
        loaded_any = True
        print(f"Loaded state_dict from: {p}")
        if len(missing) > 0:
            print(f"  Missing keys (showing up to 10): {missing[:10]}")
        if len(unexpected) > 0:
            print(f"  Unexpected keys (showing up to 10): {unexpected[:10]}")
        break

if not loaded_any:
    print(
        "No checkpoint loaded; using ImageNet-pretrained ViT with randomly initialized 5-class head."
    )

vit_model.to(device)

if hasattr(vit_model, "image_size"):
    img_size = int(vit_model.image_size)




## === cell 2
class CassavaDataset(VisionDataset):
    """Custom dataset for the Cassava data.

    Args:
        data_dir: base directory to the images.
        transforms: set of transforms to be used.
        ttas: list of transforms to apply for TTA (applied before `transform`).
        image_ids: optional explicit list of image filenames to use (ensures correct ordering and ignores subdirs).
    """

    def __init__(self, data_dir, transform=None, ttas=None, image_ids=None):
        super().__init__(root=data_dir)
        self.transform = transform

        if image_ids is not None:
            self.images = list(image_ids)
        else:
            exts = {".jpg", ".jpeg", ".png", ".bmp"}
            imgs = []
            for n in os.listdir(data_dir):
                p = os.path.join(data_dir, n)
                if os.path.isfile(p) and os.path.splitext(n.lower())[1] in exts:
                    imgs.append(n)
            self.images = sorted(imgs)  # deterministic ordering

        self.ttas = ttas

    def __getitem__(self, idx):
        filename = self.images[idx]
        img = Image.open(os.path.join(self.root, filename)).convert("RGB")

        if self.ttas is not None and self.transform is not None:
            img = [self.transform(t(img)) for t in self.ttas]
        elif self.transform:
            img = self.transform(img)

        return img, filename

    def __len__(self):
        return len(self.images)




## === cell 3
test_transforms = v2.Compose(
    [
        v2.CenterCrop((600, 600)),
        v2.Resize((img_size, img_size), interpolation=InterpolationMode.BICUBIC),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

if tta:
    ttas = [
        lambda x: x,  # identity
        v2.RandomHorizontalFlip(p=1),  # fixed flip (deterministic)
        v2.RandomVerticalFlip(p=1),  # fixed flip (deterministic)
    ]
else:
    ttas = None

sample_path = "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
image_ids = None
if os.path.exists(sample_path):
    sample = pd.read_csv(sample_path)
    if "image_id" in sample.columns:
        image_ids = sample["image_id"].tolist()

test_dataset = CassavaDataset(
    test_dir, transform=test_transforms, ttas=ttas, image_ids=image_ids
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
)

normalizer = torch.nn.Softmax(dim=1)



## === cell 4
all_names = []
all_preds = []

vit_model.eval()

with torch.no_grad():
    for batch_idx, (inputs, filenames) in enumerate(test_loader):
        filenames = list(filenames)
        bsz = len(filenames)

        if tta:
            inputs = torch.cat(inputs, dim=0).to(device)  # [T*B, C, H, W]
            preds = normalizer(vit_model(inputs))  # [T*B, K]
            t = preds.shape[0] // bsz
            preds = preds.view(t, bsz, -1)
            mean_preds = preds.mean(dim=0)  # [B, K]
            pred_labels = mean_preds.argmax(dim=1).tolist()
        else:
            inputs = inputs.to(device)
            preds = normalizer(vit_model(inputs))
            pred_labels = preds.argmax(dim=1).tolist()

        all_names.extend(filenames)
        all_preds.extend(pred_labels)

assert len(all_names) == len(
    test_dataset
), f"Pred count {len(all_names)} != test size {len(test_dataset)}"
assert len(all_preds) == len(
    test_dataset
), f"Pred count {len(all_preds)} != test size {len(test_dataset)}"



## === cell 5
my_submission = pd.DataFrame({"image_id": all_names, "label": all_preds})

sample_path = "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
if os.path.exists(sample_path):
    sample = pd.read_csv(sample_path)
    pred_map = dict(
        zip(my_submission["image_id"].tolist(), my_submission["label"].tolist())
    )
    my_submission = sample.copy()

    mapped = my_submission["image_id"].map(pred_map)
    my_submission["label"] = mapped.fillna(0).astype(int)

my_submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", my_submission.shape)
print(my_submission.head())
