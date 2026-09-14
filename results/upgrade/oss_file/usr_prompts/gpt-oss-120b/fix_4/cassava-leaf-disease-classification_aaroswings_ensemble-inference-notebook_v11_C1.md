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

0.8733756421879721

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.05531) has done: 'I added robust path handling for the test images, wrapped model loading in a safe fallback that creates a simple dummy model when the original checkpoint files are missing, and ensured the DataLoader uses a safe `num_workers=0`. This guarantees the script runs end‑to‑end, produces a correctly‑sized CSV submission, and avoids the previous `FileNotFoundError` and `NameError` issues.'

# 9. Code solution

## === cell 0
import io
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

from albumentations import (
    Compose,
    CenterCrop,
    Normalize,
    Resize,
    HorizontalFlip,
    RandomBrightnessContrast,
)
from albumentations.pytorch.transforms import ToTensorV2

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

test_paths = [
    Path("../input/cassava-leaf-disease-classification/test_images"),
    Path("/kaggle/input/cassava-leaf-disease-classification/test_images"),
    Path("input/cassava-leaf-disease-classification/test_images"),
    Path("test_images"),
]
test_img_path = next((p for p in test_paths if p.exists()), None)
if test_img_path is None:
    raise FileNotFoundError("Test image directory not found.")

train_paths = [
    Path("../input/cassava-leaf-disease-classification/train_images"),
    Path("/kaggle/input/cassava-leaf-disease-classification/train_images"),
    Path("input/cassava-leaf-disease-classification/train_images"),
    Path("train_images"),
]
train_img_path = next((p for p in train_paths if p.exists()), None)
if train_img_path is None:
    raise FileNotFoundError("Train image directory not found.")

batch_size = 64
valid_input_size = (224, 224)  # ResNet expected size


## === cell 1
train_df = pd.read_csv(
    next(
        p
        for p in [
            Path("../input/cassava-leaf-disease-classification/train.csv"),
            Path("/kaggle/input/cassava-leaf-disease-classification/train.csv"),
            Path("input/cassava-leaf-disease-classification/train.csv"),
            Path("train.csv"),
        ]
        if p.exists()
    )
)

train_tfms = Compose(
    [
        Resize(*valid_input_size),
        HorizontalFlip(p=0.5),
        RandomBrightnessContrast(p=0.3),
        Normalize(),
        ToTensorV2(),
    ]
)


class CassavaDataset(Dataset):
    def __init__(self, df, img_dir: Path, tfms):
        self.df = df
        self.img_dir = img_dir
        self.tfms = tfms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.img_dir / row["image_id"]
        img = np.array(Image.open(img_path).convert("RGB"))
        img = self.tfms(image=img)["image"]
        label = int(row["label"])
        return img, label


train_dataset = CassavaDataset(train_df, train_img_path, train_tfms)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    pin_memory=True,
)

model = torch.hub.load("pytorch/vision", "resnet18", pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 5)  # 5 classes
model = model.to(device).train()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

epochs = 2
for epoch in range(epochs):
    epoch_loss = 0.0
    correct = 0
    total = 0
    for imgs, labels in train_loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * imgs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    print(
        f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss/total:.4f} - Acc: {correct/total:.4f}"
    )

model.eval()


## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/2998750720.py in <cell line: 0>()
     52 
     53 # ---------- model definition ----------
---> 54 model = torch.hub.load("pytorch/vision", "resnet18", pretrained=True)
     55 model.fc = nn.Linear(model.fc.in_features, 5)  # 5 classes
     56 model = model.to(device).train()

/usr/local/lib/python3.11/dist-packages/torch/hub.py in load(repo_or_dir, model, source, trust_repo, force_reload, verbose, skip_validation, *args, **kwargs)
    645         )
    646 
--> 647     model = _load_local(repo_or_dir, model, *args, **kwargs)
    648     return model
    649 

/usr/local/lib/python3.11/dist-packages/torch/hub.py in _load_local(hubconf_dir, model, *args, **kwargs)
    671     with _add_to_sys_path(hubconf_dir):
    672         hubconf_path = os.path.join(hubconf_dir, MODULE_HUBCONF)
--> 673         hub_module = _import_module(MODULE_HUBCONF, hubconf_path)
    674 
    675         entry = _load_entry_from_hubconf(hub_module, model)

/usr/local/lib/python3.11/dist-packages/torch/hub.py in _import_module(name, path)
    113     module = importlib.util.module_from_spec(spec)
    114     assert isinstance(spec.loader, Loader)
--> 115     spec.loader.exec_module(module)
    116     return module
    117 

/usr/lib/python3.11/importlib/_bootstrap_external.py in exec_module(self, module)

/usr/lib/python3.11/importlib/_bootstrap.py in _call_with_frames_removed(f, *args, **kwds)

~/.cache/torch/hub/pytorch_vision_main/hubconf.py in <module>
      2 dependencies = ["torch"]
      3 
----> 4 from torchvision.models import get_model_weights, get_weight
      5 from torchvision.models.alexnet import alexnet
      6 from torchvision.models.convnext import convnext_base, convnext_large, convnext_small, convnext_tiny

~/.cache/torch/hub/pytorch_vision_main/torchvision/__init__.py in <module>
      6 # .extension) before entering _meta_registrations.
      7 from . import extension  # usort:skip  # noqa: F401
----> 8 from torchvision import (
      9     _autograd_registrations,
     10     _meta_registrations,

~/.cache/torch/hub/pytorch_vision_main/torchvision/_meta_registrations.py in <module>
    161 
    162 
--> 163 @torch.library.register_fake("torchvision::nms")
    164 def meta_nms(dets, scores, iou_threshold):
    165     torch._check(dets.dim() == 2, lambda: f"boxes should be a 2d tensor, got {dets.dim()}D")

/usr/local/lib/python3.11/dist-packages/torch/library.py in register(func)
    826         else:
    827             use_lib = lib
--> 828         use_lib._register_fake(op_name, func, _stacklevel=stacklevel + 1)
    829         return func
    830 

/usr/local/lib/python3.11/dist-packages/torch/library.py in _register_fake(self, op_name, fn, _stacklevel)
    196             func_to_register = fn
    197 
--> 198         handle = entry.fake_impl.register(func_to_register, source)
    199         self._registration_handles.append(handle)
    200 

/usr/local/lib/python3.11/dist-packages/torch/_library/fake_impl.py in register(self, func, source)
     29                 f"{self.kernel.source}."
     30             )
---> 31         if torch._C._dispatch_has_kernel_for_dispatch_key(self.qualname, "Meta"):
     32             raise RuntimeError(
     33                 f"register_fake(...): the operator {self.qualname} "

RuntimeError: operator torchvision::nms does not exist

## === cell 2
test_tfms = Compose(
    [
        Resize(*valid_input_size),
        CenterCrop(*valid_input_size, always_apply=True),
        Normalize(),
        ToTensorV2(),
    ]
)


class TestDataset(Dataset):
    def __init__(self, path: Path, tfms):
        self.data = [
            (file.name, file.read_bytes()) for file in path.iterdir() if file.is_file()
        ]
        self.tfms = tfms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        filename, img_bytes = self.data[idx]
        img = np.array(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
        img = self.tfms(image=img)["image"]
        return filename, img


test_ds = TestDataset(test_img_path, test_tfms)
test_loader = DataLoader(
    test_ds,
    batch_size=batch_size,
    num_workers=0,
    drop_last=False,
    pin_memory=True,
)




## === cell 3
class EnsemblePredictor:
    def __init__(self, models):
        self.models = models

    def predict_on_loader(self, loader):
        predictions = []
        filenames = []

        for batch_filenames, batch_imgs in loader:
            batch_imgs = batch_imgs.to(device)
            batch_logits = None
            for model in self.models:
                with torch.no_grad():
                    logits = model(batch_imgs).cpu()
                if batch_logits is None:
                    batch_logits = logits
                else:
                    batch_logits += logits
            batch_preds = batch_logits.numpy()
            predictions.extend([int(np.argmax(p)) for p in batch_preds])
            filenames.extend(batch_filenames)

        return predictions, filenames




## === cell 4
predictor = EnsemblePredictor([model])
predictions, filenames = predictor.predict_on_loader(test_loader)


## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2846965598.py in <cell line: 0>()
      1 # Use the trained ResNet as the sole ensemble member
----> 2 predictor = EnsemblePredictor([model])
      3 predictions, filenames = predictor.predict_on_loader(test_loader)

NameError: name 'model' is not defined

## === cell 5
submission_path = Path("submission.csv")
with submission_path.open("w", newline="") as f:
    f.write("image_id,label\n")
    for filename, pred in zip(filenames, predictions):
        f.write(f"{filename},{pred}\n")

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/414679552.py in <cell line: 0>()
      2 with submission_path.open("w", newline="") as f:
      3     f.write("image_id,label\n")
----> 4     for filename, pred in zip(filenames, predictions):
      5         f.write(f"{filename},{pred}\n")

NameError: name 'filenames' is not defined

## --- ERROR in outputing the csv:
Invalid submission: Submission must have the same length as the answers.
