# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os
import warnings
import pandas as pd
import numpy as np

from PIL import Image

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms

import xgboost as xgb
import pickle

warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

DATA_SOURCE = os.path.join("..", "input", "aptos2019-blindness-detection")
MODEL_SOURCE = os.path.join("..", "input", "aptos-cnn-features-extraction-xgb-baseline")

TRAIN_CSV = os.path.join(DATA_SOURCE, "train.csv")
TEST_CSV = os.path.join(DATA_SOURCE, "test.csv")
SAMPLE_SUB = os.path.join(DATA_SOURCE, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(DATA_SOURCE, "train_images")
TEST_IMG_DIR = os.path.join(DATA_SOURCE, "test_images")

BASE_TRANSFORMS = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)




## === cell 1
class RetinopathyDatasetTrain(Dataset):
    def __init__(self):
        df = pd.read_csv(TRAIN_CSV)
        self.data = df.reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        code = str(self.data.loc[idx, "id_code"])
        path = os.path.join(TRAIN_IMG_DIR, code + ".png")
        imgpil = Image.open(path).convert("RGB")
        img_tensor = BASE_TRANSFORMS(imgpil)
        y = int(self.data.loc[idx, "diagnosis"])
        return {"image": img_tensor, "label": y}


class RetinopathyDatasetTest(Dataset):
    def __init__(self):
        df = pd.read_csv(TEST_CSV)
        self.data = df.reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        code = str(self.data.loc[idx, "id_code"])
        path = os.path.join(TEST_IMG_DIR, code + ".png")
        imgpil = Image.open(path).convert("RGB")
        img_tensor = BASE_TRANSFORMS(imgpil)
        return {"image": img_tensor}




## === cell 2
extractor = torchvision.models.resnet101(weights=None)
extractor.fc = nn.Identity()

resnet_path = os.path.join(MODEL_SOURCE, "resnet101.pth")
loaded_external = False
if os.path.exists(resnet_path):
    try:
        state = torch.load(resnet_path, map_location="cpu")
        extractor.load_state_dict(state)
        loaded_external = True
        print(f"Loaded external extractor weights: {resnet_path}")
    except Exception as e:
        print(
            f"Warning: failed to load external weights ({e}). Falling back to ImageNet weights."
        )
        loaded_external = False

if not loaded_external:
    extractor = torchvision.models.resnet101(
        weights=torchvision.models.ResNet101_Weights.IMAGENET1K_V2
    )
    extractor.fc = nn.Identity()
    print("Using ImageNet-pretrained ResNet101 weights for feature extraction.")

extractor.to(DEVICE)
extractor.eval()




## === cell 3
def get_extracted_data(data_loader):
    feats = []
    labels = []
    for bi, d in enumerate(data_loader):
        if bi % 32 == 0:
            print(".", end="")
        img_tensor = d["image"].to(DEVICE, non_blocking=True)
        with torch.no_grad():
            feature = extractor(img_tensor)  # [B, 2048]
        feature = feature.detach().cpu().numpy()
        feats.append(feature)
        if "label" in d:
            labels.append(d["label"].numpy())
    print("")
    X = np.concatenate(feats, axis=0)
    if len(labels) > 0:
        y = np.concatenate(labels, axis=0)
        return X, y
    return X




## === cell 4
train_loader = torch.utils.data.DataLoader(
    RetinopathyDatasetTrain(),
    batch_size=16,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
    drop_last=False,
)
test_loader = torch.utils.data.DataLoader(
    RetinopathyDatasetTest(),
    batch_size=16,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
    drop_last=False,
)

print("Extracting train features")
X_train, y_train = get_extracted_data(train_loader)
print("Train features:", X_train.shape, "Train labels:", y_train.shape)

print("Extracting test features")
X_test = get_extracted_data(test_loader)
print("Test features:", X_test.shape)



## === cell 5
XGBOOST_PARAM = {
    "random_state": 42,
    "objective": "multi:softprob",  # need probabilities for averaging; aligns with predict_proba usage
    "num_class": 5,
    "n_estimators": 200,
    "eval_metric": "mlogloss",
    "tree_method": "hist",
}


def load_or_train_xgb(model_filename, seed_offset=0):
    model_path = os.path.join(MODEL_SOURCE, model_filename)
    if os.path.exists(model_path):
        try:
            with open(model_path, "rb") as f:
                m = pickle.load(f)
            print(f"Loaded XGB model: {model_path}")
            return m
        except Exception as e:
            print(
                f"Warning: failed to load {model_path} ({e}). Training a new model instead."
            )

    params = dict(XGBOOST_PARAM)
    params["random_state"] = int(params["random_state"]) + int(seed_offset)
    m = xgb.XGBClassifier(**params)
    m.fit(X_train, y_train)
    return m




## === cell 6
xgb_model_1 = load_or_train_xgb("xgb_model_1", seed_offset=0)
prediction1 = xgb_model_1.predict_proba(X_test)



## === cell 7
xgb_model_2 = load_or_train_xgb("xgb_model_2", seed_offset=1)
prediction2 = xgb_model_2.predict_proba(X_test)



## === cell 8
prediction = (prediction1 + prediction2).argmax(axis=1).astype(int)



## === cell 9
df_sub = pd.read_csv(SAMPLE_SUB)
df_sub["diagnosis"] = prediction
df_sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", df_sub.shape)
print(df_sub.head())
