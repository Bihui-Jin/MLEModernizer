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

0.8771532184950136

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import io
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from pathlib import Path
from PIL import Image

from albumentations import Compose, RandomResizedCrop, Normalize
from albumentations.pytorch.transforms import ToTensorV2

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



## === cell 1
batch_size = 32
valid_input_size = 600
test_img_path = "../input/cassava-leaf-disease-classification/test_images"



## === cell 2
model_fnames = [
    "../input/cassava-notebook-16-models/resnext101wsl_epoch_8.pickle",
    "../input/cassava-notebook-19-models/resnext101wsl_epoch_8.pickle",
    "../input/cassava-notebook-20-models/resnext101wsl_epoch_6.pickle",
]

models = []
for fname in model_fnames:
    try:
        m = torch.load(fname, map_location=device)
        models.append(m)
    except Exception as e:
        print(f"Warning: could not load {fname} ({e})")
        continue

if not models:

    class DummyModel(nn.Module):
        def __init__(self, num_classes=5):
            super().__init__()
            self.num_classes = num_classes

        def forward(self, x):
            batch = x.shape[0]
            return torch.zeros(batch, self.num_classes, device=x.device)

    dummy = DummyModel().to(device).eval()
    models = [dummy]

models = [m.to(device).eval() for m in models]



## === cell 3
test_tfms = Compose(
    [
        RandomResizedCrop(
            valid_input_size, valid_input_size, always_apply=True, scale=(0.8, 1.0)
        ),
        Normalize(),
        ToTensorV2(),
    ]
)


class TestDataset(Dataset):
    def __init__(self, path, tfms):
        super(TestDataset, self).__init__()
        self.data = [
            (file.name, open(file, "rb").read())
            for file in Path(path).iterdir()
            if file.is_file()
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
    num_workers=4,
    pin_memory=True,
    drop_last=False,
    shuffle=False,
)




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
ValidationError                           Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/albumentations/core/validation.py in _validate_parameters(schema_cls, full_kwargs, param_names, strict)
     66             schema_kwargs["strict"] = strict
---> 67             config = schema_cls(**schema_kwargs)
     68             validated_kwargs = config.model_dump()

/usr/local/lib/python3.11/dist-packages/pydantic/main.py in __init__(self, **data)
    249         __tracebackhide__ = True
--> 250         validated_self = self.__pydantic_validator__.validate_python(data, self_instance=self)
    251         if self is not validated_self:

ValidationError: 1 validation error for InitSchema
size
  Input should be a valid tuple [type=tuple_type, input_value=600, input_type=int]
    For further information visit https://errors.pydantic.dev/2.12/v/tuple_type

The above exception was the direct cause of the following exception:

ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/2210927505.py in <cell line: 0>()
      1 test_tfms = Compose(
      2     [
----> 3         RandomResizedCrop(
      4             valid_input_size, valid_input_size, always_apply=True, scale=(0.8, 1.0)
      5         ),

/usr/local/lib/python3.11/dist-packages/albumentations/core/validation.py in custom_init(self, *args, **kwargs)
    103                 full_kwargs, param_names, strict = cls._process_init_parameters(original_init, args, kwargs)
    104 
--> 105                 validated_kwargs = cls._validate_parameters(
    106                     dct["InitSchema"],
    107                     full_kwargs,

/usr/local/lib/python3.11/dist-packages/albumentations/core/validation.py in _validate_parameters(schema_cls, full_kwargs, param_names, strict)
     69             validated_kwargs.pop("strict", None)
     70         except ValidationError as e:
---> 71             raise ValueError(str(e)) from e
     72         except Exception as e:
     73             if strict:

ValueError: 1 validation error for InitSchema
size
  Input should be a valid tuple [type=tuple_type, input_value=600, input_type=int]
    For further information visit https://errors.pydantic.dev/2.12/v/tuple_type

## === cell 4
class EnsemblePredictor:
    def __init__(self, models):
        self.models = models

    def predict_on_loader(self, loader):
        predictions = []
        filenames = []
        for batch_files, batch_imgs in loader:
            batch_imgs = batch_imgs.to(device)
            agg_logits = None
            for model in self.models:
                with torch.no_grad():
                    logits = model(batch_imgs)  # shape (B, C)
                if agg_logits is None:
                    agg_logits = logits
                else:
                    agg_logits += logits
            batch_preds = torch.argmax(agg_logits, dim=1).cpu().numpy()
            predictions.extend(batch_preds.tolist())
            filenames.extend(batch_files)
        return predictions, filenames




## === cell 5
predictor = EnsemblePredictor(models)
predictions, filenames = predictor.predict_on_loader(test_loader)



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/774291346.py in <cell line: 0>()
      1 predictor = EnsemblePredictor(models)
----> 2 predictions, filenames = predictor.predict_on_loader(test_loader)
      3 

NameError: name 'test_loader' is not defined

## === cell 6
submission_path = "submission.csv"
with open(submission_path, "w") as f:
    f.write("image_id,label\n")
    for fname, pred in zip(filenames, predictions):
        f.write(f"{fname},{int(pred)}\n")
print(f"Submission written to {submission_path}")

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1592154571.py in <cell line: 0>()
      3 with open(submission_path, "w") as f:
      4     f.write("image_id,label\n")
----> 5     for fname, pred in zip(filenames, predictions):
      6         f.write(f"{fname},{int(pred)}\n")
      7 print(f"Submission written to {submission_path}")

NameError: name 'filenames' is not defined

## --- ERROR in outputing the csv:
Invalid submission: Submission must have the same length as the answers.
