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

3.10

# 3. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

0.8055303717135086

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 1
import os
import re
import cv2
import torch
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from time import time
from torch import nn, optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader as DL
from torch.nn.utils import weight_norm as WN
from torchvision import models, transforms

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder

## === cell 3
le = LabelEncoder()

SEED = 42
SIZE = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRANSFORM_FINAL = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
TRANSFORM = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.43032, 0.49672, 0.31341], [0.20665, 0.21170, 0.18763])])

## === cell 4
def breaker(num: int = 50, char: str = "*") -> None:
    print("\n" + num*char + "\n")



    


def load_image(path: str, size: int) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2RGB)
    image = cv2.resize(src=image, dsize=(size, size), interpolation=cv2.INTER_AREA)
    return image


def show_image(image: np.ndarray, cmap: str = "gnuplot2") -> None:
    plt.figure()
    plt.imshow(image, cmap=cmap)
    plt.axis("off")
    plt.show()

## === cell 6
class DS(Dataset):
    def __init__(self, base_path: str, filenames: np.ndarray, labels: np.ndarray = None, mode: str = "train", transform = None):

        assert re.match(r"^train$", mode, re.IGNORECASE) or re.match(r"^valid$", mode, re.IGNORECASE) or re.match(r"^test$", mode, re.IGNORECASE), "Invalid Mode"
        
        self.mode = mode
        self.base_path = base_path
        self.filenames = filenames
        self.transform = transform

        if re.match(r"^train$", mode, re.IGNORECASE) or re.match(r"^valid$", mode, re.IGNORECASE):
            self.labels = labels

    def __len__(self):
        return self.filenames.shape[0]

    def __getitem__(self, idx):
        image = load_image(os.path.join(self.base_path, self.filenames[idx]), SIZE)
        if re.match(r"^train$", self.mode, re.IGNORECASE) or re.match(r"^valid$", self.mode, re.IGNORECASE):
            return self.transform(image), torch.LongTensor(self.labels[idx])
        else:
            return self.transform(image)

## === cell 8
class CNN(nn.Module):
    def __init__(self, mode: str, model_name: str):
        
        super(CNN, self).__init__()
        
        self.model_name = model_name
        self.mode = mode
        
        if re.match(r"^vgg$", self.model_name, re.IGNORECASE):
            if re.match(r"^full$", self.mode, re.IGNORECASE):
                self.model = models.vgg16_bn(pretrained=False, progress=True)
                self.model.classifier[-1] = nn.Linear(in_features=self.model.classifier[-1].in_features, out_features=10)
            elif re.match(r"^semi$", self.mode, re.IGNORECASE) or re.match(r"^final$", self.mode, re.IGNORECASE):
                self.model = models.vgg16_bn(pretrained=True, progress=True)
                self.freeze()
                self.model.classifier[-1] = nn.Linear(in_features=self.model.classifier[-1].in_features, out_features=10)
            
        elif re.match(r"^resnet$", self.model_name, re.IGNORECASE):
            if re.match(r"^full$", self.mode, re.IGNORECASE):
                self.model = models.resnet50(pretrained=False, progress=True)
                self.model.fc = nn.Linear(in_features=self.model.fc.in_features, out_features=10)
            elif re.match(r"^semi$", self.mode, re.IGNORECASE) or re.match(r"^final$", self.mode, re.IGNORECASE):
                self.model = models.resnet50(pretrained=True, progress=True)
                self.freeze()
                self.model.fc = nn.Linear(in_features=self.model.fc.in_features, out_features=10)
        
        elif re.match(r"^densenet$", self.model_name, re.IGNORECASE):
            if re.match(r"^full$", self.mode, re.IGNORECASE):
                self.model = models.densenet169(pretrained=False, progress=True)
                self.model.classifier = nn.Linear(in_features=self.model.classifier.in_features, out_features=10)
            elif re.match(r"^semi$", self.mode, re.IGNORECASE) or re.match(r"^final$", self.mode, re.IGNORECASE):
                self.model = models.densenet169(pretrained=True, progress=True)
                self.freeze()
                self.model.classifier = nn.Linear(in_features=self.model.classifier.in_features, out_features=10)
        
        elif re.match(r"^mobilenet$", self.model_name, re.IGNORECASE):
            if re.match(r"^full$", self.mode, re.IGNORECASE):
                self.model = models.mobilenet_v3_small(pretrained=False, progress=True)
                self.model.classifier[-1] = nn.Linear(in_features=self.model.classifier[-1].in_features, out_features=10)
            elif re.match(r"^semi$", self.mode, re.IGNORECASE) or re.match(r"^final$", self.mode, re.IGNORECASE):
                self.model = models.mobilenet_v3_small(pretrained=True, progress=True)
                self.freeze()
                self.model.classifier[-1] = nn.Linear(in_features=self.model.classifier[-1].in_features, out_features=10)

    def freeze(self):
        for params in self.parameters():
            params.requires_grad = False

        if re.match(r"^vgg$", self.model_name, re.IGNORECASE):
            if re.match(r"^semi$", self.mode, re.IGNORECASE):
                for names, params in self.named_parameters():
                    if re.match(r".*features.3[4-9].*", names, re.IGNORECASE) or re.match(r".*features.4[0-9].*", names, re.IGNORECASE) or re.match(r".*classifier.*", names, re.IGNORECASE):
                        params.requires_grad = True
        
        elif re.match(r"^resnet$", self.model_name, re.IGNORECASE):
            if re.match(r"^semi$", self.mode, re.IGNORECASE):
                for names, params in self.named_parameters():
                    if re.match(r".*layer4.*", names, re.IGNORECASE):
                        params.requires_grad = True
        
        elif re.match(r"^densenet$", self.model_name, re.IGNORECASE):
            if re.match(r"^semi$", self.mode, re.IGNORECASE):
                for names, params in self.named_parameters():
                    if re.match(r".*denseblock4.*", names, re.IGNORECASE) or re.match(r".*norm5.*", names, re.IGNORECASE):
                        params.requires_grad = True
        
        elif re.match(r"^mobilenet$", self.model_name, re.IGNORECASE):
            if re.match(r"^semi$", self.mode, re.IGNORECASE):
                for names, params in self.named_parameters():
                    if re.match(r".*features.9.*", names, re.IGNORECASE) or re.match(r".*features.1[0-2].*", names, re.IGNORECASE) or re.match(r".*classifier.*", names, re.IGNORECASE):
                        params.requires_grad = True
    
    def get_optimizer(self, lr=1e-3, wd=0.0):
        params = [p for p in self.parameters() if p.requires_grad]
        return optim.Adam(params, lr=lr, weight_decay=wd)

    def get_plateau_scheduler(self, optimizer=None, patience=5, eps=1e-8):
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience=patience, eps=eps)
    
    def forward(self, x):
        return nn.LogSoftmax(dim=1)(self.model(x))

## === cell 10
    
    



            



        
        
        
        




def predict(model=None, dataloader=None, path=None) -> np.ndarray:
    model.load_state_dict(torch.load(path, map_location=DEVICE)["model_state_dict"])
    model.to(DEVICE)    
    model.eval()
    
    y_pred = torch.zeros(1, 1).to(DEVICE)
    
    for X in dataloader:
        X = X.to(DEVICE)
        with torch.no_grad():
            output = torch.argmax(torch.exp(model(X)), dim=1)
        y_pred = torch.cat((y_pred, output.view(-1, 1)), dim=0)
    
    return y_pred[1:].detach().cpu().numpy()

## === cell 12
DEBUG: bool = False
    
if DEBUG:
    n_splits = 3
    epochs = 2
    mode = "semi"
    model_name = "mobilenet"
else:
    n_splits = 25
    epochs = 250
    mode = "semi"
    model_name = "resnet"

batch_size = 128
lr = 1e-6
wd = 1e-5
early_stopping = 5
patience = None
eps = None

## === cell 18
ss_df = pd.read_csv("../input/cassava-leaf-disease-classification/sample_submission.csv")
ts_filenames = ss_df.image_id.copy().values

ts_data_setup = DS("../input/cassava-leaf-disease-classification/test_images", ts_filenames, None, "test", TRANSFORM)
ts_data = DL(ts_data_setup, batch_size=batch_size, shuffle=False)

model = CNN(mode="full", model_name=model_name).to(DEVICE)
y_pred = predict(model=model, dataloader=ts_data, path="../input/cldc-r256slow/saves/state.pt")

ss_df["label"] = y_pred.astype("uint8")
ss_df.to_csv("submission.csv", index=False)

## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_10/3610609725.py in <cell line: 0>()
      6 
      7 model = CNN(mode="full", model_name=model_name).to(DEVICE)
----> 8 y_pred = predict(model=model, dataloader=ts_data, path="../input/cldc-r256slow/saves/state.pt")
      9 
     10 ss_df["label"] = y_pred.astype("uint8")

/tmp/ipykernel_10/2397443641.py in predict(model, dataloader, path)
     94 
     95 def predict(model=None, dataloader=None, path=None) -> np.ndarray:
---> 96     model.load_state_dict(torch.load(path, map_location=DEVICE)["model_state_dict"])
     97     model.to(DEVICE)
     98     model.eval()

/usr/local/lib/python3.11/dist-packages/torch/serialization.py in load(f, map_location, pickle_module, weights_only, mmap, **pickle_load_args)
   1423         pickle_load_args["encoding"] = "utf-8"
   1424 
-> 1425     with _open_file_like(f, "rb") as opened_file:
   1426         if _is_zipfile(opened_file):
   1427             # The zipfile reader is going to advance the current file position.

/usr/local/lib/python3.11/dist-packages/torch/serialization.py in _open_file_like(name_or_buffer, mode)
    749 def _open_file_like(name_or_buffer, mode):
    750     if _is_path(name_or_buffer):
--> 751         return _open_file(name_or_buffer, mode)
    752     else:
    753         if "w" in mode:

/usr/local/lib/python3.11/dist-packages/torch/serialization.py in __init__(self, name, mode)
    730 class _open_file(_opener):
    731     def __init__(self, name, mode):
--> 732         super().__init__(open(name, mode))
    733 
    734     def __exit__(self, *args):

FileNotFoundError: [Errno 2] No such file or directory: '../input/cldc-r256slow/saves/state.pt'
