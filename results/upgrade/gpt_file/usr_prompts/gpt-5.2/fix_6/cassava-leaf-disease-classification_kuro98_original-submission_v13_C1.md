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

0.8948322756119673

# 6. Current score

0.17339

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.05531) has done: 'I fix the missing model file error by adding a robust fallback that loads a built-in torchvision ViT when the external checkpoint path doesn’t exist, so the notebook runs end-to-end and produces predictions. I also fix the invalid submission length issue by ensuring the DataLoader does not shuffle and by aligning/sorting predictions to exactly match `sample_submission.csv`’s `image_id` ordering (this also prevents duplicates/missing IDs). Finally, I make test-time augmentation deterministic (so it’s reproducible under the fixed seed) and remove a large debug print that can slow/timeout inference, without changing the core inference semantics (still TTA with mean probabilities and argmax). The result always write a valid `submission.csv` with the required columns and correct row count.'
- What this solution (achieved 0.05531) has done: 'I fix the runtime error by ensuring the fallback torchvision ViT model is created with `image_size=384` to match your 384×384 preprocessing (the checkpointed model likely expected 384, while the default torchvision ViT expects 224). I also make the test image listing deterministic (sorted) so filenames/predictions are stable and aligned, without changing inference semantics. Finally, I keep your submission alignment via `sample_submission.csv` unchanged so it always outputs a valid `submission.csv` with the correct rows/columns; these changes should also materially improve accuracy versus the broken 224/384 mismatch.'
- What this solution (achieved 0.13117) has done: 'I fix the immediate runtime error caused by trying to override `image_size` while also loading pretrained `ViT_B_16_Weights` (torchvision enforces 224 for that weights enum). To preserve your core ViT approach and 384×384 preprocessing, I switch the fallback to `weights=None` when `image_size=384`, which avoids the exception and ensures `vit_model` is always created (preventing the downstream `NoneType` error). I also make the TTA transforms deterministic per-sample by seeding inside `__getitem__`, keeping the same TTA/mean-prob/argmax semantics but stabilizing predictions and typically improving accuracy vs fully random TTA. The rest of the pipeline (dataset, TTA averaging, and submission alignment to `sample_submission.csv`) remains unchanged.'
- What this solution (achieved 0.13117) has done: 'I fix the crash by filtering the test directory listing so the dataset only includes actual image files (the provided directory contains a nested `test_images/` folder that was being treated like an image). I also make the checkpoint loading more robust (load either a full model object or a state_dict into the same ViT architecture) so the intended weights are used when available, which should move accuracy up toward the target without changing the core ViT+TTA inference semantics. Finally, I keep your submission alignment to `sample_submission.csv` unchanged and ensure the script always writes a valid `submission.csv`.'
- What this solution (achieved 0.17339) has done: 'Your score is far below the target, so the smallest reliable way to move accuracy upward is to make sure inference uses trained weights rather than an untrained ViT fallback. I (1) make checkpoint loading robust to common key mismatches (especially `heads.head.*` vs `head.*` vs `classifier.*`) and enforce `strict=True` when possible so we don’t silently run with random weights, and (2) if the provided checkpoint can’t be used, switch the fallback to a pretrained ViT at its native 224px and adjust the preprocessing accordingly (same ViT core logic, just avoiding the current “random weights at 384px” failure mode). I also remove the submission-time `groupby` (it can hide bugs) and instead assert one prediction per sample and align exactly to `sample_submission.csv`. These changes keep your architecture/inference approach (ViT + optional TTA + mean-prob + argmax) intact while addressing the main reason for the low accuracy.'

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

img_size = 384
batch_size = 16
num_workers = 4
num_classes = 5
tta = True

ckpt_path = "/kaggle/input/vit-v1-update/vit_v1_1.pt"

from torchvision.models import vit_b_16, ViT_B_16_Weights


def _build_vit(image_size: int, weights):
    m = vit_b_16(weights=weights, image_size=image_size)
    in_features = m.heads.head.in_features
    m.heads.head = torch.nn.Linear(in_features, num_classes)
    return m


vit_model = _build_vit(image_size=img_size, weights=None)

ckpt_loaded = False
if os.path.exists(ckpt_path):
    obj = torch.load(ckpt_path, map_location="cpu")
    try:
        if isinstance(obj, torch.nn.Module):
            vit_model = obj
            ckpt_loaded = True
        elif isinstance(obj, dict):
            state_dict = obj.get("state_dict", obj)

            cleaned = {}
            for k, v in state_dict.items():
                nk = k
                if nk.startswith("module."):
                    nk = nk[len("module.") :]
                if nk.startswith("model."):
                    nk = nk[len("model.") :]

                if nk.startswith("head."):
                    nk = "heads.head." + nk[len("head.") :]
                if nk.startswith("classifier."):
                    nk = "heads.head." + nk[len("classifier.") :]
                if nk.startswith("fc."):
                    nk = "heads.head." + nk[len("fc.") :]

                cleaned[nk] = v

            try:
                vit_model.load_state_dict(cleaned, strict=True)
                ckpt_loaded = True
            except Exception as e_strict:
                vit_model.load_state_dict(cleaned, strict=False)
                ckpt_loaded = True
                print(
                    "Warning: checkpoint loaded with strict=False (some keys mismatched). Error:",
                    repr(e_strict),
                )
    except Exception as e:
        print(
            "Warning: failed to load checkpoint object, will fall back. Error:",
            repr(e),
        )

if not ckpt_loaded:
    print(
        "Checkpoint not loaded; falling back to pretrained ViT_B_16_Weights.DEFAULT at 224px."
    )
    img_size = 224
    vit_model = _build_vit(image_size=img_size, weights=ViT_B_16_Weights.DEFAULT)

vit_model.to(device)




## === cell 2
class CassavaDataset(VisionDataset):
    """Custom dataset for the Cassava test images.
    Returns (image or list-of-tta-images, filename).
    """

    def __init__(self, data_dir, transform=None, ttas=None, img_size=384):
        super().__init__(root=data_dir)
        self.transform = transform
        self.ttas = ttas
        self.base_seed = 3407

        exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        files = []
        for name in os.listdir(data_dir):
            full = os.path.join(data_dir, name)
            if os.path.isfile(full) and name.lower().endswith(exts):
                files.append(name)
        self.images = sorted(files)

    def __getitem__(self, idx):
        filename = self.images[idx]
        img = Image.open(os.path.join(self.root, filename)).convert("RGB")

        if self.ttas is not None and self.transform is not None:
            out = []
            for j, t in enumerate(self.ttas):
                torch.manual_seed(self.base_seed + idx * 1000 + j)
                torch.cuda.manual_seed(self.base_seed + idx * 1000 + j)
                aug = t(img)
                out.append(self.transform(aug))
            img = out
        elif self.transform:
            img = self.transform(img)

        return img, filename

    def __len__(self):
        return len(self.images)




## === cell 3
test_transforms = v2.Compose(
    [
        v2.Resize(
            (img_size, img_size),
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        ),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

if tta:
    ttas = [
        v2.RandomRotation(180),
        v2.RandomAffine(180),
        v2.RandomPerspective(p=1),
    ]
else:
    ttas = None

test_dataset = CassavaDataset(test_dir, transform=test_transforms, ttas=ttas)

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

        if tta:
            inputs = torch.cat(inputs, dim=0).to(device, non_blocking=True)

            preds = normalizer(vit_model(inputs))

            b = len(filenames)
            t = len(ttas)
            batch_preds = preds.view(t, b, -1)
            mean_preds = batch_preds.mean(dim=0)
            pred_labels = torch.argmax(mean_preds, dim=1).tolist()
        else:
            inputs = inputs.to(device, non_blocking=True)
            preds = normalizer(vit_model(inputs))
            pred_labels = torch.argmax(preds, dim=1).tolist()

        all_names.extend(filenames)
        all_preds.extend(pred_labels)

assert len(all_names) == len(test_dataset), (len(all_names), len(test_dataset))
assert len(all_preds) == len(test_dataset), (len(all_preds), len(test_dataset))



## === cell 5
sample_path = "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
sample = pd.read_csv(sample_path)

pred_df = pd.DataFrame({"image_id": all_names, "label": all_preds})

if pred_df["image_id"].duplicated().any():
    pred_df = pred_df.drop_duplicates("image_id", keep="first")

submission = sample[["image_id"]].merge(pred_df, on="image_id", how="left")
submission["label"] = submission["label"].fillna(0).astype(int)

submission.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())



## === cell 6
submission
