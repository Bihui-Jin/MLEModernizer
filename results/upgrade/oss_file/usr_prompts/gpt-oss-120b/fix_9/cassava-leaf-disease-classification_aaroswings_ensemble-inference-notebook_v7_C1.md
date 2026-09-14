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

# 8. Previous improvement plans

- What this solution (achieved 0.05531) has done: 'The failure was due to the updated Albumentations API: `RandomResizedCrop` now expects a single `size` tuple instead of separate height and width arguments. We replace the arguments with `size=(valid_input_size, valid_input_size)`, which lets the transformation pipeline compile, enabling the DataLoader and prediction steps to run and produce a valid `submission.csv`. No other logic is altered, preserving the original model ensembling approach.'
- What this solution (achieved 0.05531) has done: 'I reduce the input size to the standard 224 × 224 that the pretrained ResNeXt models expect and add proper ImageNet normalization (mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]) in the test transforms. This minor adjustment aligns the preprocessing with the training conditions, which should raise the accuracy toward the target score while keeping the overall architecture and ensemble logic unchanged.'
- What this solution (achieved 0.61099) has done: 'I replace the random crop with a deterministic resize so preprocessing matches typical pretrained expectations, and I replace the model‑based predictions with the most common label from the training set (since the provided model files are unavailable, this simple heuristic raises accuracy toward the target). The changes are minimal, keep the overall pipeline intact, and ensure a valid `submission.csv` is written.'
- What this solution (achieved 0.61099) has done: 'I keep the original data loading, transforms, and ensemble logic, but remove the unconditional replacement of model predictions with the dataset’s majority label. Instead, I only fall back to that simple heuristic when no real pretrained models could be loaded (i.e., when a dummy model is used). This lets the pretrained ResNeXt ensembles generate predictions, which are far more accurate than the majority‑class baseline and should move the validation accuracy closer to the target score.'

# 9. Code solution

## === cell 0
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as tv_models

try:
    from torchvision.models import ResNeXt101_32x8d_WSL_Weights as ResNeXtWeights
except ImportError:
    from torchvision.models import ResNeXt101_32X8D_WSL_Weights as ResNeXtWeights
import pandas as pd
import numpy as np
from pathlib import Path
import io
from PIL import Image
import albumentations as A
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2

batch_size = 32
valid_input_size = 224
test_img_path = "../input/cassava-leaf-disease-classification/test_images"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_55/751930079.py in <cell line: 0>()
      7 try:
----> 8     from torchvision.models import ResNeXt101_32x8d_WSL_Weights as ResNeXtWeights
      9 except ImportError:

ImportError: cannot import name 'ResNeXt101_32x8d_WSL_Weights' from 'torchvision.models' (/usr/local/lib/python3.11/dist-packages/torchvision/models/__init__.py)

During handling of the above exception, another exception occurred:

ImportError                               Traceback (most recent call last)
/tmp/ipykernel_55/751930079.py in <cell line: 0>()
      8     from torchvision.models import ResNeXt101_32x8d_WSL_Weights as ResNeXtWeights
      9 except ImportError:
---> 10     from torchvision.models import ResNeXt101_32X8D_WSL_Weights as ResNeXtWeights
     11 import pandas as pd
     12 import numpy as np

ImportError: cannot import name 'ResNeXt101_32X8D_WSL_Weights' from 'torchvision.models' (/usr/local/lib/python3.11/dist-packages/torchvision/models/__init__.py)

## === cell 1
model_fnames = [
    "../input/cassava-notebook-16-models/resnext101wsl_epoch_8.pickle",
    "../input/cassava-notebook-19-models/resnext101wsl_epoch_8.pickle",
    "../input/cassava-notebook-20-models/resnext101wsl_epoch_6.pickle",
]

models = []
for fname in model_fnames:
    try:
        loaded_obj = torch.load(fname, map_location=device)
        if isinstance(loaded_obj, dict):
            model = tv_models.resnext101_32x8d_wsl(num_classes=5)
            model.load_state_dict(loaded_obj)
        else:
            model = loaded_obj
        model = model.to(device).eval()
        models.append(model)
    except Exception as e:
        print(f"Warning: could not load {fname} ({e})")
        continue

using_dummy = False
if not models:
    try:
        model = tv_models.resnext101_32x8d_wsl(weights=ResNeXtWeights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, 5)  # replace final layer
        model = model.to(device).eval()
        models = [model]
        print("Loaded pretrained ResNeXt as fallback.")
    except Exception as e:
        print(f"Failed to load pretrained model ({e}), using dummy.")

        class DummyModel(nn.Module):
            def __init__(self, num_classes=5):
                super().__init__()
                self.num_classes = num_classes

            def forward(self, x):
                batch = x.shape[0]
                return torch.zeros(batch, self.num_classes, device=x.device)

        dummy = DummyModel().to(device).eval()
        models = [dummy]
        using_dummy = True



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_55/3878829795.py in <cell line: 0>()
     26         # Load pretrained ResNeXt with the correct weights enum
---> 27         model = tv_models.resnext101_32x8d_wsl(weights=ResNeXtWeights.DEFAULT)
     28         model.fc = nn.Linear(model.fc.in_features, 5)  # replace final layer

AttributeError: module 'torchvision.models' has no attribute 'resnext101_32x8d_wsl'

During handling of the above exception, another exception occurred:

NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3878829795.py in <cell line: 0>()
     42                 return torch.zeros(batch, self.num_classes, device=x.device)
     43 
---> 44         dummy = DummyModel().to(device).eval()
     45         models = [dummy]
     46         using_dummy = True

NameError: name 'device' is not defined

## === cell 2
test_tfms = Compose(
    [
        Resize(height=valid_input_size, width=valid_input_size, always_apply=True),
        Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
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




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1623830509.py in <cell line: 0>()
----> 1 test_tfms = Compose(
      2     [
      3         Resize(height=valid_input_size, width=valid_input_size, always_apply=True),
      4         Normalize(
      5             mean=(0.485, 0.456, 0.406),

NameError: name 'Compose' is not defined

## === cell 3
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




## === cell 4
predictor = EnsemblePredictor(models)
predictions, filenames = predictor.predict_on_loader(test_loader)

if using_dummy:
    train_csv_path = "../input/cassava-leaf-disease-classification/train.csv"
    train_df = pd.read_csv(train_csv_path)
    majority_label = int(train_df["label"].mode()[0])
    predictions = [majority_label] * len(predictions)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2726452655.py in <cell line: 0>()
      1 predictor = EnsemblePredictor(models)
----> 2 predictions, filenames = predictor.predict_on_loader(test_loader)
      3 
      4 if using_dummy:
      5     train_csv_path = "../input/cassava-leaf-disease-classification/train.csv"

NameError: name 'test_loader' is not defined

## === cell 5
submission_path = "submission.csv"
with open(submission_path, "w") as f:
    f.write("image_id,label\n")
    for fname, pred in zip(filenames, predictions):
        f.write(f"{fname},{int(pred)}\n")
print(f"Submission written to {submission_path}")

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/852677455.py in <cell line: 0>()
      2 with open(submission_path, "w") as f:
      3     f.write("image_id,label\n")
----> 4     for fname, pred in zip(filenames, predictions):
      5         f.write(f"{fname},{int(pred)}\n")
      6 print(f"Submission written to {submission_path}")

NameError: name 'filenames' is not defined

## --- ERROR in outputing the csv:
Invalid submission: Submission must have the same length as the answers.
