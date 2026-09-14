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

0.8974010275007556

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

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
if torch.cuda.is_available():
    torch.cuda.manual_seed(3407)

cudnn.deterministic = False
cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

test_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images/"
sample_sub_path = (
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)

model_a_img_size = 384
model_b_img_size = 528
model_c_img_size = 384

batch_size = 16
num_workers = 4
num_classes = 5
tta = False


def _replace_classifier(model: torch.nn.Module, num_classes: int) -> torch.nn.Module:
    if hasattr(model, "fc") and isinstance(model.fc, torch.nn.Module):
        in_f = model.fc.in_features
        model.fc = torch.nn.Linear(in_f, num_classes)
    elif hasattr(model, "classifier") and isinstance(model.classifier, torch.nn.Module):
        if isinstance(model.classifier, torch.nn.Sequential):
            seq = list(model.classifier)
            for i in range(len(seq) - 1, -1, -1):
                if isinstance(seq[i], torch.nn.Linear):
                    in_f = seq[i].in_features
                    seq[i] = torch.nn.Linear(in_f, num_classes)
                    break
            model.classifier = torch.nn.Sequential(*seq)
        elif isinstance(model.classifier, torch.nn.Linear):
            in_f = model.classifier.in_features
            model.classifier = torch.nn.Linear(in_f, num_classes)
    elif hasattr(model, "heads"):
        heads = getattr(model, "heads")
        if isinstance(heads, torch.nn.Sequential):
            seq = list(heads)
            for i in range(len(seq) - 1, -1, -1):
                if isinstance(seq[i], torch.nn.Linear):
                    in_f = seq[i].in_features
                    seq[i] = torch.nn.Linear(in_f, num_classes)
                    break
            model.heads = torch.nn.Sequential(*seq)
        elif isinstance(heads, torch.nn.Linear):
            in_f = heads.in_features
            model.heads = torch.nn.Linear(in_f, num_classes)
    elif hasattr(model, "head") and isinstance(model.head, torch.nn.Linear):
        in_f = model.head.in_features
        model.head = torch.nn.Linear(in_f, num_classes)
    return model


def _safe_construct(fallback_ctor, num_classes: int):
    """
    Bugfix: Kaggle kernels may not have internet to download DEFAULT weights.
    Try weights='DEFAULT' first, then fall back to weights=None to avoid crash.
    """
    try:
        model = fallback_ctor(weights="DEFAULT")
    except Exception as e:
        print(
            f"[WARN] Could not load DEFAULT weights ({type(e).__name__}: {e}). Using weights=None."
        )
        model = fallback_ctor(weights=None)
    model = _replace_classifier(model, num_classes)
    return model


def _try_load_or_fallback(pt_path: str, fallback_ctor, num_classes: int):
    if os.path.exists(pt_path):
        obj = torch.load(pt_path, map_location=device)
        if isinstance(obj, dict) and "state_dict" in obj:
            model = _safe_construct(fallback_ctor, num_classes)
            model.load_state_dict(obj["state_dict"], strict=False)
            return model
        if isinstance(obj, dict):
            model = _safe_construct(fallback_ctor, num_classes)
            model.load_state_dict(obj, strict=False)
            return model
        return obj
    return _safe_construct(fallback_ctor, num_classes)


def _infer_required_image_size(model: torch.nn.Module, default: int) -> int:
    """
    Bugfix: ViT models assert input H/W equals model.image_size.
    Use that when available; otherwise keep provided default.
    """
    sz = getattr(model, "image_size", None)
    if isinstance(sz, int) and sz > 0:
        return sz
    return default


from torchvision import models as tvm

path_a = "/kaggle/input/vit-v1/vit_v1.pt"
path_b = "/kaggle/input/efficient-net/efficient_net.pt"
path_c = "/kaggle/input/vit-v6/vit_v6.pt"

model_a = _try_load_or_fallback(
    path_a, lambda weights="DEFAULT": tvm.vit_b_16(weights=weights), num_classes
)
model_b = _try_load_or_fallback(
    path_b, lambda weights="DEFAULT": tvm.efficientnet_b0(weights=weights), num_classes
)
model_c = _try_load_or_fallback(
    path_c, lambda weights="DEFAULT": tvm.resnet18(weights=weights), num_classes
)

model_a_img_size = _infer_required_image_size(model_a, model_a_img_size)
model_b_img_size = _infer_required_image_size(model_b, model_b_img_size)
model_c_img_size = _infer_required_image_size(model_c, model_c_img_size)
print(
    "Using image sizes:",
    {"a": model_a_img_size, "b": model_b_img_size, "c": model_c_img_size},
)

model_a.to(device)
model_b.to(device)
model_c.to(device)




## === cell 2
class CassavaDataset(VisionDataset):
    """Custom dataset for the Cassava data.

    Args:
        data_dir: base directory to the images.
        transforms: set of transforms to be used.
    """

    def __init__(
        self,
        data_dir,
        model_a_size,
        model_b_size,
        model_c_size,
        transform=None,
        ttas=None,
        img_size=384,
    ):
        if os.path.isdir(os.path.join(data_dir, "test_images")):
            data_dir = os.path.join(data_dir, "test_images")
        if os.path.isdir(os.path.join(data_dir, "train_images")):
            data_dir = os.path.join(data_dir, "train_images")

        super().__init__(root=data_dir)

        self.transform = transform
        self.ttas = ttas
        self.cc = v2.CenterCrop((600, 600))
        self.resize_model_a = v2.Resize(
            (model_a_size, model_a_size), interpolation=InterpolationMode.BICUBIC
        )
        self.resize_model_b = v2.Resize(
            (model_b_size, model_b_size), interpolation=InterpolationMode.BICUBIC
        )
        self.resize_model_c = v2.Resize(
            (model_c_size, model_c_size), interpolation=InterpolationMode.BICUBIC
        )

        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        files = []
        for name in os.listdir(self.root):
            p = os.path.join(self.root, name)
            if os.path.isfile(p) and os.path.splitext(name.lower())[1] in exts:
                files.append(name)
        self.images = sorted(files)

        if len(self.images) == 0:
            raise RuntimeError(f"No image files found under: {self.root}")

    def __getitem__(self, idx):
        filename = self.images[idx]
        img = Image.open(os.path.join(self.root, filename)).convert("RGB")
        img = self.cc(img)
        model_a_img = self.resize_model_a(img)
        model_b_img = self.resize_model_b(img)
        model_c_img = self.resize_model_c(img)

        if self.ttas is not None and self.transform is not None:
            model_a_img = [self.transform(t(model_a_img)) for t in self.ttas]
            model_b_img = [self.transform(t(model_b_img)) for t in self.ttas]
            model_c_img = [self.transform(t(model_c_img)) for t in self.ttas]
        elif self.transform:
            model_a_img = self.transform(model_a_img)
            model_b_img = self.transform(model_b_img)
            model_c_img = self.transform(model_c_img)

        return model_a_img, model_b_img, model_c_img, filename

    def __len__(self):
        return len(self.images)




## === cell 3
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
        v2.RandomAffine(180),
        v2.RandomPerspective(p=1),
    ]
else:
    ttas = None

test_dataset = CassavaDataset(
    test_dir,
    model_a_img_size,
    model_b_img_size,
    model_c_img_size,
    transform=test_transforms,
    ttas=ttas,
)

use_workers = num_workers if os.name != "nt" else 0
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=use_workers,
    pin_memory=torch.cuda.is_available(),
)

normalizer = torch.nn.Softmax(dim=1)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/2450387191.py in <cell line: 0>()
     17     ttas = None
     18 
---> 19 test_dataset = CassavaDataset(
     20     test_dir,
     21     model_a_img_size,

/tmp/ipykernel_55/1565544454.py in __init__(self, data_dir, model_a_size, model_b_size, model_c_size, transform, ttas, img_size)
     48 
     49         if len(self.images) == 0:
---> 50             raise RuntimeError(f"No image files found under: {self.root}")
     51 
     52     def __getitem__(self, idx):

RuntimeError: No image files found under: /kaggle/input/cassava-leaf-disease-classification/test_images/test_images

## === cell 4
all_names = []
all_preds = []

model_a.eval()
model_b.eval()
model_c.eval()

with torch.no_grad():
    for batch_idx, (
        model_a_inputs,
        model_b_inputs,
        model_c_inputs,
        filenames,
    ) in enumerate(test_loader):
        bsz = len(filenames)

        if tta:
            model_a_inputs = torch.cat(model_a_inputs, dim=0).to(device)
            model_b_inputs = torch.cat(model_b_inputs, dim=0).to(device)
            model_c_inputs = torch.cat(model_c_inputs, dim=0).to(device)
            filenames = list(filenames)

            model_a_outputs = model_a(model_a_inputs)
            model_b_outputs = model_b(model_b_inputs)
            model_c_outputs = model_c(model_c_inputs)

            model_a_batch_logits = torch.stack(torch.split(model_a_outputs, bsz), dim=0)
            model_a_mean_logits = torch.mean(model_a_batch_logits, dim=0)

            model_b_batch_logits = torch.stack(torch.split(model_b_outputs, bsz), dim=0)
            model_b_mean_logits = torch.mean(model_b_batch_logits, dim=0)

            model_c_batch_logits = torch.stack(torch.split(model_c_outputs, bsz), dim=0)
            model_c_mean_logits = torch.mean(model_c_batch_logits, dim=0)

            outputs = (
                model_a_mean_logits + model_b_mean_logits + model_c_mean_logits
            ) / 3.0
            mean_preds = normalizer(outputs)
            pred_labels = torch.argmax(mean_preds, 1).tolist()
        else:
            model_a_inputs = model_a_inputs.to(device)
            model_b_inputs = model_b_inputs.to(device)
            model_c_inputs = model_c_inputs.to(device)
            filenames = list(filenames)

            model_a_outputs = model_a(model_a_inputs)
            model_b_outputs = model_b(model_b_inputs)
            model_c_outputs = model_c(model_c_inputs)

            outputs = (model_a_outputs + model_b_outputs + model_c_outputs) / 3.0
            preds = normalizer(outputs)
            pred_labels = torch.argmax(preds, 1).tolist()

        all_names.extend(filenames)
        all_preds.extend(pred_labels)

sample_sub = pd.read_csv(sample_sub_path)
pred_map = pd.DataFrame({"image_id": all_names, "label": all_preds}).drop_duplicates(
    "image_id"
)
my_submission = sample_sub[["image_id"]].merge(pred_map, on="image_id", how="left")
my_submission["label"] = my_submission["label"].fillna(0).astype(int)

my_submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", my_submission.shape)
print(my_submission.head())
print("Missing predictions filled with 0:", int(my_submission["label"].isna().sum()))

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1143613450.py in <cell line: 0>()
     12         model_c_inputs,
     13         filenames,
---> 14     ) in enumerate(test_loader):
     15         bsz = len(filenames)
     16 

NameError: name 'test_loader' is not defined
