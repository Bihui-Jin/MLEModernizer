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
Create a classifier to predict the severity of diabetic retinopathy.

## Metric
Quadratic weighted kappa, which measures the agreement between two ratings. This metric typically varies from 0 (random agreement between raters) to 1 (complete agreement between raters). In the event that there is less agreement between the raters than expected by chance, this metric may go below 0. The quadratic weighted kappa is calculated between the scores assigned by the human rater and the predicted scores.

Images have five possible ratings, 0,1,2,3,4.  Each image is characterized by a tuple *(e*,*e)*, which corresponds to its scores by *Rater A* (human) and *Rater B* (predicted).  The quadratic weighted kappa is calculated as follows. First, an N x N histogram matrix *O* is constructed, such that *O* corresponds to the number of images that received a rating *i* by *A* and a rating *j* by *B*. An *N-by-N* matrix of weights, *w*, is calculated based on the difference between raters' scores:

An *N-by-N* histogram matrix of expected ratings, *E*, is calculated, assuming that there is no correlation between rating scores.  This is calculated as the outer product between each rater's histogram vector of ratings, normalized such that *E* and *O* have the same sum.

## Submission Format
```
id_code,diagnosis
0005cfc8afb6,0
003f0afdcd15,0
etc.
```

## Dataset
You are provided with a large set of retina images taken using [fundus photography](https://en.wikipedia.org/wiki/Fundus_photography) under a variety of imaging conditions.

Labels are on a scale of 0 to 4:

> 0 - No DR
> 1 - Mild
> 2 - Moderate
> 3 - Severe
> 4 - Proliferative DR

Images may contain artifacts, be out of focus, underexposed, or overexposed. The images were gathered from multiple clinics using a variety of cameras over an extended period of time, which will introduce further variation.

- **train.csv** - the training labels
- **test.csv** - the test set (you must predict the `diagnosis` value for these variables)
- **sample_submission.csv** - a sample submission file in the correct format
- **train.zip** - the training set images
- **test.zip** - the public test set images

# 2. Python version

3.7

# 3. Installed packages

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
xgboost==2.0.3

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (118 lines)
            sample_submission.csv (368 lines)
            sample_submission.csv.zip (3.2 kB)
            test.csv (368 lines)
            test.csv.zip (2.9 kB)
            test.zip (160 Bytes)
            test_images.zip (902.9 MB)
            train.csv (3296 lines)
            train.csv.zip (27.5 kB)
            train.zip (162 Bytes)
            train_images.zip (7.7 GB)
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
            test_images/
                218c822a3dd9.png (5.7 MB)
                0e82bcacc475.png (5.2 MB)
                ... and 365 other files
                test_images/
            train_images/
                184a185e7447.png (337.5 kB)
                c4aef0d88d1b.png (876.6 kB)
                ... and 3293 other files
                train_images/
        input/
            description.md (118 lines)
            sample_submission.csv (368 lines)
            sample_submission.csv.zip (3.2 kB)
            test.csv (368 lines)
            test.csv.zip (2.9 kB)
            test.zip (160 Bytes)
            test_images.zip (902.9 MB)
            train.csv (3296 lines)
            train.csv.zip (27.5 kB)
            train.zip (162 Bytes)
            train_images.zip (7.7 GB)
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
            test_images/
                218c822a3dd9.png (5.7 MB)
                0e82bcacc475.png (5.2 MB)
                ... and 365 other files
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
            train_images/
                184a185e7447.png (337.5 kB)
                c4aef0d88d1b.png (876.6 kB)
                ... and 3293 other files
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
        working/
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
```

-> data/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/aptos2019-blindness-detection/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/aptos2019-blindness-detection/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> input/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> (stopped after 10 files for performance)

# 5. Target score

0.8165454803132608

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 1
import os
import pandas as pd
import numpy as np 

from PIL import Image

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms

import xgboost as xgb
from sklearn.metrics import cohen_kappa_score

import pickle

DEVICE = torch.device("cuda:0")
DATA_SOURCE = os.path.join("..","input","aptos2019-blindness-detection")
MODEL_SOURCE = os.path.join("..","input","aptos-cnn-features-extraction-xgb-baseline")


## === cell 3
class RetinopathyDatasetTest(Dataset):

    def __init__(self, eval_set=False, random_state=42):
        csv_file = os.path.join(DATA_SOURCE, "test.csv")
        df = pd.read_csv(csv_file)
        self.data = df.reset_index(drop=True)
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        folder = os.path.join(DATA_SOURCE, "test_images")
        code = str(self.data.loc[idx, 'id_code'])
        file = code + ".png"
        path = os.path.join(folder, file)
        imgpil = Image.open(path)
        base_transforms = transforms.Compose([transforms.Resize((224, 224)),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406], 
                                                                   [0.229, 0.224, 0.225])])
        img_tensor = base_transforms(imgpil)
        return {'image': img_tensor}

## === cell 5
extractor = torchvision.models.resnet101(pretrained=False)
extractor.fc = nn.Identity() 
model_path = os.path.join(MODEL_SOURCE, "resnet101.pth")
extractor.load_state_dict(torch.load(model_path))
extractor.to(DEVICE)
extractor.eval()

data_loader = torch.utils.data.DataLoader(RetinopathyDatasetTest(), 
                            batch_size=2, shuffle=False, num_workers=0, drop_last=False)

def get_extracted_data(data_loader):
    for bi, d in enumerate(data_loader):
        if bi % 32 == 0 : print(".", end="")
        img_tensor = d["image"].to(DEVICE)
        with torch.no_grad(): feature = extractor(img_tensor)
        feature = feature.cpu().detach().squeeze(0).numpy()
        if bi == 0 :
            features = feature 
        else :
            features = np.concatenate([features, feature], axis=0)
    print("")
    return features

print("...............................")
features = get_extracted_data(data_loader)

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/2406629919.py in <cell line: 0>()
      4 extractor.fc = nn.Identity()
      5 model_path = os.path.join(MODEL_SOURCE, "resnet101.pth")
----> 6 extractor.load_state_dict(torch.load(model_path))
      7 extractor.to(DEVICE)
      8 extractor.eval()

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

FileNotFoundError: [Errno 2] No such file or directory: '../input/aptos-cnn-features-extraction-xgb-baseline/resnet101.pth'

## === cell 7
XGBOOST_PARAM = {
    "random_state" : 42,
    'objective': 'multi:softmax',
    "num_class" : 5,
    "n_estimators" : 200,
    "eval_metric" : "mlogloss"
}
xgb_model_1 = xgb.XGBClassifier(**XGBOOST_PARAM)
model_path = os.path.join(MODEL_SOURCE, "xgb_model_1")
xgb_model_1 = pickle.load(open(model_path, "rb"))
prediction1 = xgb_model_1.predict_proba(features)

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/325045080.py in <cell line: 0>()
      8 xgb_model_1 = xgb.XGBClassifier(**XGBOOST_PARAM)
      9 model_path = os.path.join(MODEL_SOURCE, "xgb_model_1")
---> 10 xgb_model_1 = pickle.load(open(model_path, "rb"))
     11 prediction1 = xgb_model_1.predict_proba(features)

FileNotFoundError: [Errno 2] No such file or directory: '../input/aptos-cnn-features-extraction-xgb-baseline/xgb_model_1'

## === cell 8
xgb_model_2 = xgb.XGBClassifier(**XGBOOST_PARAM)
model_path = os.path.join(MODEL_SOURCE, "xgb_model_2")
xgb_model_2 = pickle.load(open(model_path, "rb"))
prediction2 = xgb_model_2.predict_proba(features)

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/2643751920.py in <cell line: 0>()
      1 xgb_model_2 = xgb.XGBClassifier(**XGBOOST_PARAM)
      2 model_path = os.path.join(MODEL_SOURCE, "xgb_model_2")
----> 3 xgb_model_2 = pickle.load(open(model_path, "rb"))
      4 prediction2 = xgb_model_2.predict_proba(features)

FileNotFoundError: [Errno 2] No such file or directory: '../input/aptos-cnn-features-extraction-xgb-baseline/xgb_model_2'

## === cell 10
prediction = (prediction1 + prediction2).argmax(axis=1)
csv_file = os.path.join(DATA_SOURCE, "sample_submission.csv")
df = pd.read_csv(csv_file)
df["diagnosis"] = prediction
df.to_csv('submission.csv',index=False)

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/792883186.py in <cell line: 0>()
----> 1 prediction = (prediction1 + prediction2).argmax(axis=1)
      2 csv_file = os.path.join(DATA_SOURCE, "sample_submission.csv")
      3 df = pd.read_csv(csv_file)
      4 df["diagnosis"] = prediction
      5 df.to_csv('submission.csv',index=False)

NameError: name 'prediction1' is not defined
