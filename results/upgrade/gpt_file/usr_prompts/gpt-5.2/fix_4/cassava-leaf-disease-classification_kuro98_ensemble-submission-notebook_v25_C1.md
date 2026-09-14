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

0.8931701420368692

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.14312) has done: 'I fix the dataset bug that accidentally includes the nested `test_images/` directory as an “image”, which causes `IsADirectoryError` and prevents `pred_map` (and thus the submission) from being created. I make the file listing robust by filtering to actual image files (and ignoring directories/hidden files), preserving ordering so predictions align with `sample_submission.csv`. I also add a safe fallback path for `test_dir` in case the directory structure differs across Kaggle mounts, and ensure the script always writes `submission.csv` with the required columns.'

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

torch.manual_seed(3407)
if torch.cuda.is_available():
    torch.cuda.manual_seed(3407)

cudnn.deterministic = False
cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

test_dir_candidates = [
    "/kaggle/input/cassava-leaf-disease-classification/test_images/",
    "/kaggle/input/cassava-leaf-disease-classification/test_images/test_images/",
    "/kaggle/input/test_images/",
    "/kaggle/input/test_images/test_images/",
]
test_dir = None
for p in test_dir_candidates:
    if os.path.isdir(p):
        try:
            has_jpg = any(
                os.path.isfile(os.path.join(p, f))
                and f.lower().endswith((".jpg", ".jpeg", ".png"))
                for f in os.listdir(p)
            )
        except Exception:
            has_jpg = False
        if has_jpg:
            test_dir = p
            break
if test_dir is None:
    test_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images/"

sample_sub_path = (
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)
train_csv_path_candidates = [
    "/kaggle/input/cassava-leaf-disease-classification/train.csv",
    "/kaggle/input/train.csv",
]
train_csv_path = None
for p in train_csv_path_candidates:
    if os.path.exists(p):
        train_csv_path = p
        break

model_b_img_size = 384
model_a_img_size = 384
batch_size = 16
num_workers = 4
num_classes = 5
tta = False


def _safe_torch_load(path, map_location):
    try:
        if os.path.exists(path):
            return torch.load(path, map_location=map_location)
        return None
    except Exception:
        return None


model_a = _safe_torch_load("/kaggle/input/vit-v1/vit_v1.pt", map_location=device)
model_b = _safe_torch_load(
    "/kaggle/input/vit-boosted/vit_boosted.pt", map_location=device
)
linear_head = _safe_torch_load(
    "/kaggle/input/linear-head/linear_cls.pt", map_location=device
)

use_centroid_fallback = False
centroids = None  # shape [num_classes, D]
feat_dim = None
train_dir_candidates = [
    "/kaggle/input/cassava-leaf-disease-classification/train_images/",
    "/kaggle/input/cassava-leaf-disease-classification/train_images/train_images/",
    "/kaggle/input/train_images/",
    "/kaggle/input/train_images/train_images/",
]
train_dir = None
for p in train_dir_candidates:
    if os.path.isdir(p):
        try:
            has_jpg = any(
                os.path.isfile(os.path.join(p, f))
                and f.lower().endswith((".jpg", ".jpeg", ".png"))
                for f in os.listdir(p)
            )
        except Exception:
            has_jpg = False
        if has_jpg:
            train_dir = p
            break

from torchvision.models import vit_b_16, ViT_B_16_Weights

weights = ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1

if (model_a is None) or (model_b is None):
    backbone = vit_b_16(weights=weights).to(device)
    backbone.eval()
    use_centroid_fallback = True
else:
    backbone = None

if isinstance(model_a, dict) or isinstance(model_b, dict):
    model_a, model_b = None, None
    backbone = vit_b_16(weights=weights).to(device)
    backbone.eval()
    use_centroid_fallback = True

if not use_centroid_fallback:
    model_a = model_a.to(device)
    model_b = model_b.to(device)
    if isinstance(model_a, dict) and "state_dict" in model_a:
        raise RuntimeError("Loaded model_a is a checkpoint dict; expected a nn.Module.")
    if isinstance(model_b, dict) and "state_dict" in model_b:
        raise RuntimeError("Loaded model_b is a checkpoint dict; expected a nn.Module.")




## === cell 1
class CassavaDataset(VisionDataset):
    """Custom dataset for the Cassava test data.

    Args:
        data_dir: base directory to the images.
        model_a_size/model_b_size: per-model input sizes.
        transform: transforms to be used (post resize/crop).
        ttas: optional list of transforms for test-time augmentation.
        image_list: optional explicit image_id ordering (used to match sample_submission exactly).
    """

    def __init__(
        self,
        data_dir,
        model_a_size,
        model_b_size,
        transform=None,
        ttas=None,
        image_list=None,
    ):
        super().__init__(root=data_dir)

        self.transform = transform

        exts = (".jpg", ".jpeg", ".png")

        if image_list is not None:
            files = []
            for f in image_list:
                if not isinstance(f, str):
                    continue
                if f.startswith("."):
                    continue
                fp = os.path.join(data_dir, f)
                if os.path.isfile(fp) and f.lower().endswith(exts):
                    files.append(f)
            self.images = files
        else:
            files = []
            for f in os.listdir(data_dir):
                if f.startswith("."):
                    continue
                fp = os.path.join(data_dir, f)
                if os.path.isfile(fp) and f.lower().endswith(exts):
                    files.append(f)
            self.images = sorted(files)

        if len(self.images) == 0:
            raise RuntimeError(f"No image files found in test_dir={data_dir}")

        self.ttas = ttas
        self.cc = v2.CenterCrop((600, 600))
        self.resize_model_a = v2.Resize(
            (model_a_size, model_a_size), interpolation=InterpolationMode.BICUBIC
        )
        self.resize_model_b = v2.Resize(
            (model_b_size, model_b_size), interpolation=InterpolationMode.BICUBIC
        )

    def __getitem__(self, idx):
        filename = self.images[idx]
        img = Image.open(os.path.join(self.root, filename)).convert("RGB")

        img = self.cc(img)
        model_a_img = self.resize_model_a(img)
        model_b_img = self.resize_model_b(img)

        if self.ttas is not None and self.transform is not None:
            model_a_img = [self.transform(t(model_a_img)) for t in self.ttas]
            model_b_img = [self.transform(t(model_b_img)) for t in self.ttas]
        elif self.transform:
            model_a_img = self.transform(model_a_img)
            model_b_img = self.transform(model_b_img)

        return model_a_img, model_b_img, filename

    def __len__(self):
        return len(self.images)




## === cell 2
test_transforms = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

if tta:
    ttas = [
        v2.RandomRotation(180),
        v2.RandomVerticalFlip(1),
        v2.RandomPerspective(p=1),
    ]
else:
    ttas = None

sample_sub = pd.read_csv(sample_sub_path)
test_image_list = sample_sub["image_id"].tolist()

test_dataset = CassavaDataset(
    test_dir,
    model_a_img_size,
    model_b_img_size,
    transform=test_transforms,
    ttas=ttas,
    image_list=test_image_list,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
)

normalizer = torch.nn.Softmax(dim=1)


def _vit_features(vit_model, x):
    vit_model.eval()
    with torch.no_grad():
        x = vit_model._process_input(x)
        n = x.shape[0]
        batch_class_token = vit_model.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = vit_model.encoder(x)
        x = x[:, 0]  # CLS token
        x = vit_model.heads.pre_logits(x)
    return x


if "use_centroid_fallback" in globals() and use_centroid_fallback:
    if train_csv_path is None or train_dir is None:
        raise RuntimeError(
            "Centroid fallback enabled but train.csv or train_images directory not found."
        )

    train_df = pd.read_csv(train_csv_path)
    max_train_for_centroids = 8000
    train_df = train_df.iloc[:max_train_for_centroids].copy()

    class_sums = None
    class_counts = torch.zeros(num_classes, dtype=torch.long)

    train_ds = CassavaDataset(
        train_dir,
        model_a_img_size,
        model_b_img_size,
        transform=test_transforms,
        ttas=None,
        image_list=train_df["image_id"].tolist(),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    labels_map = dict(zip(train_df["image_id"].tolist(), train_df["label"].tolist()))

    backbone.eval()
    for model_a_inputs, _, filenames in train_loader:
        model_a_inputs = model_a_inputs.to(device)
        feats = _vit_features(backbone, model_a_inputs)  # [B,D]
        if class_sums is None:
            feat_dim = feats.shape[1]
            class_sums = torch.zeros(num_classes, feat_dim, device=device)

        for i, fn in enumerate(filenames):
            y = int(labels_map[str(fn)])
            class_sums[y] += feats[i]
            class_counts[y] += 1

    class_counts_safe = class_counts.clamp(min=1).to(device).unsqueeze(1)
    centroids = class_sums / class_counts_safe
    centroids = torch.nn.functional.normalize(centroids, dim=1)
    print("Built centroids with counts:", class_counts.tolist())




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_55/283801061.py in <cell line: 0>()
     95     for model_a_inputs, _, filenames in train_loader:
     96         model_a_inputs = model_a_inputs.to(device)
---> 97         feats = _vit_features(backbone, model_a_inputs)  # [B,D]
     98         if class_sums is None:
     99             feat_dim = feats.shape[1]

/tmp/ipykernel_55/283801061.py in _vit_features(vit_model, x)
     53         x = vit_model.encoder(x)
     54         x = x[:, 0]  # CLS token
---> 55         x = vit_model.heads.pre_logits(x)
     56     return x
     57 

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in __getattr__(self, name)
   1926             if name in modules:
   1927                 return modules[name]
-> 1928         raise AttributeError(
   1929             f"'{type(self).__name__}' object has no attribute '{name}'"
   1930         )

AttributeError: 'Sequential' object has no attribute 'pre_logits'

## === cell 3
all_names = []
all_preds = []

if "use_centroid_fallback" in globals() and use_centroid_fallback:
    backbone.eval()
else:
    model_a.eval()
    model_b.eval()
    if linear_head is not None and hasattr(linear_head, "eval"):
        linear_head.eval()

with torch.no_grad():
    for _, (model_a_inputs, model_b_inputs, filenames) in enumerate(test_loader):
        if tta:
            batch_n = len(filenames)
            model_a_inputs = torch.cat(model_a_inputs, dim=0).to(device)
            model_b_inputs = torch.cat(model_b_inputs, dim=0).to(device)
            filenames = list(filenames)

            if "use_centroid_fallback" in globals() and use_centroid_fallback:
                feats = _vit_features(backbone, model_a_inputs)
                feats = torch.nn.functional.normalize(feats, dim=1)
                sims = feats @ centroids.T  # cosine similarity
                sims_batch = torch.stack(torch.split(sims, batch_n), dim=0)
                sims_mean = torch.mean(sims_batch, dim=0)
                pred_labels = torch.argmax(sims_mean, 1).tolist()
            else:
                model_a_outputs = model_a(model_a_inputs)
                model_b_outputs = model_b(model_b_inputs)

                model_a_batch_logits = torch.stack(
                    torch.split(model_a_outputs, batch_n), dim=0
                )
                model_a_mean_logits = torch.mean(model_a_batch_logits, dim=0)

                model_b_batch_logits = torch.stack(
                    torch.split(model_b_outputs, batch_n), dim=0
                )
                model_b_mean_logits = torch.mean(model_b_batch_logits, dim=0)

                outputs = 0.9 * model_a_mean_logits + 0.1 * model_b_mean_logits
                preds = normalizer(outputs)
                pred_labels = torch.argmax(preds, 1).tolist()
        else:
            model_a_inputs = model_a_inputs.to(device)
            model_b_inputs = model_b_inputs.to(device)
            filenames = list(filenames)

            if "use_centroid_fallback" in globals() and use_centroid_fallback:
                feats = _vit_features(backbone, model_a_inputs)
                feats = torch.nn.functional.normalize(feats, dim=1)
                sims = feats @ centroids.T
                pred_labels = torch.argmax(sims, 1).tolist()
            else:
                model_a_outputs = model_a(model_a_inputs)
                model_b_outputs = model_b(model_b_inputs)

                outputs = 0.9 * model_a_outputs + 0.1 * model_b_outputs
                preds = normalizer(outputs)
                pred_labels = torch.argmax(preds, 1).tolist()

        all_names.extend(filenames)
        all_preds.extend(pred_labels)

pred_map = dict(zip(all_names, all_preds))




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_55/2316644378.py in <cell line: 0>()
     51 
     52             if "use_centroid_fallback" in globals() and use_centroid_fallback:
---> 53                 feats = _vit_features(backbone, model_a_inputs)
     54                 feats = torch.nn.functional.normalize(feats, dim=1)
     55                 sims = feats @ centroids.T

/tmp/ipykernel_55/283801061.py in _vit_features(vit_model, x)
     53         x = vit_model.encoder(x)
     54         x = x[:, 0]  # CLS token
---> 55         x = vit_model.heads.pre_logits(x)
     56     return x
     57 

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in __getattr__(self, name)
   1926             if name in modules:
   1927                 return modules[name]
-> 1928         raise AttributeError(
   1929             f"'{type(self).__name__}' object has no attribute '{name}'"
   1930         )

AttributeError: 'Sequential' object has no attribute 'pre_logits'

## === cell 4
ordered_preds = sample_sub["image_id"].map(pred_map)

ordered_preds = ordered_preds.fillna(0).astype(int)

my_submission = pd.DataFrame(
    {"image_id": sample_sub["image_id"], "label": ordered_preds}
)

assert len(my_submission) == len(sample_sub), (len(my_submission), len(sample_sub))
assert my_submission["image_id"].isna().sum() == 0
assert my_submission["label"].between(0, 4).all()

my_submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", my_submission.shape)
print(my_submission.head())

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/553233749.py in <cell line: 0>()
      1 # sample_sub already read in cell 3
----> 2 ordered_preds = sample_sub["image_id"].map(pred_map)
      3 
      4 # If any are missing (shouldn't happen now due to explicit ordering), default to class 0.
      5 ordered_preds = ordered_preds.fillna(0).astype(int)

NameError: name 'pred_map' is not defined
