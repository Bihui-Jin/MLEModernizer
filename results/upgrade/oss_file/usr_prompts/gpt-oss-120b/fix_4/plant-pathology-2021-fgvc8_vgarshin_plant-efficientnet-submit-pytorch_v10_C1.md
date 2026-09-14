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
Detect apple diseases from images.

## Metric
Mean F1-Score

## Submission Format
labels should be a space-delimited list.

The file should contain a header and have the following format:

```
image, labels
85f8cb619c66b863.jpg,healthy
ad8770db05586b59.jpg,healthy
c7b03e718489f3ca.jpg,healthy
```

## Dataset
**train.csv** - the training set metadata.

- `image` - the image ID.
- `labels` - the target classes, a space delimited list of all diseases found in the image. Unhealthy leaves with too many diseases to classify visually will have the `complex` class, and may also have a subset of the diseases identified.

**sample_submission.csv** - A sample submission file in the correct format.

- `image`
- `labels`

**train_images** - The training set images.

**test_images** - The test set images. This competition has a hidden test set: only three images are provided here as samples while the remaining 5,000 images will be available to your notebook once it is submitted.

# 2. Python version

3.9

# 3. Installed packages

geopandas==0.14.4
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
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
        input/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
        working/
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
```

-> data/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> data/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> (stopped after 10 files for performance)

# 5. Target score

0.8273499538319488

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import gc
import json
import time
import cv2
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data.sampler import SequentialSampler
from torchvision import transforms, models
from IPython.display import display  # for Jupyter preview calls

KAGGLE = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    from efficientnet_pytorch import model as enet
except Exception:
    enet = None  # will use torchvision later.




## === cell 1
TEST = True
VER = "v4"
if KAGGLE:
    DATA_PATH = "../input/plant-pathology-2021-fgvc8"
    MDLS_PATH = f"../input/plant-models-{VER}"
else:
    DATA_PATH = "./data"
    MDLS_PATH = f"./models_{VER}"
TH = 0.3
TTAS = [0, 1, 2]
FOLDS = [3, 4]
IMGS_PATH = f"{DATA_PATH}/test_images" if TEST else f"{DATA_PATH}/train_images"

start_time = time.time()




## === cell 2
params_path = f"{MDLS_PATH}/params.json"
if os.path.isfile(params_path):
    with open(params_path) as file:
        params = json.load(file)
else:
    print("params.json not found – using default parameters.")
    params = {
        "img_size": 224,
        "batch_size": 16,
        "workers": 0,
        "dropout": 0.2,
        "backbone": "efficientnet-b0",
        "labels_": ["healthy", "multiple_diseases", "rust", "scab", "complex"],
        "labels": {
            "0": "healthy",
            "1": "multiple_diseases",
            "2": "rust",
            "3": "scab",
            "4": "complex",
        },
    }
LABELS_ = params["labels_"]
LABELS = params["labels"]
WORKERS = 2 if KAGGLE else params.get("workers", 0)
print("Loaded params:", params)




## === cell 3
df_sub = pd.DataFrame(os.listdir(IMGS_PATH))
df_sub.columns = ["image"]
df_sub["labels"] = "healthy"  # placeholder; will be overwritten after inference.
print("Submission template preview:")
display(df_sub.head())




## === cell 4
def flip(img, axis=0):
    if axis == 1:
        return img[::-1, :, :]
    elif axis == 2:
        return img[:, ::-1, :]
    elif axis == 3:
        return img[::-1, ::-1, :]
    else:
        return img


class PlantDataset(data.Dataset):
    def __init__(self, df, size, labels, transform=None, tta=0):
        self.df = df.reset_index(drop=True)
        self.size = size
        self.labels = labels
        self.transform = transform
        self.tta = tta

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img_name = row.image
        img_path = f"{IMGS_PATH}/{img_name}"
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.size, self.size))
        img = img.astype(np.float32) / 255.0
        if self.transform is not None:
            img = self.transform(image=img)["image"]
        if self.labels:
            img = img.transpose(2, 0, 1)
            label = np.zeros(len(self.labels), dtype=np.float32)
            for lbl in row.labels.split():
                label[self.labels[lbl]] = 1.0
            return torch.tensor(img), torch.tensor(label)
        else:
            img = flip(img, axis=self.tta)
            img = img.transpose(2, 0, 1)
            return torch.tensor(img.copy())


class EffNet(nn.Module):
    def __init__(self, params, out_dim):
        super(EffNet, self).__init__()
        backbone_name = params["backbone"]
        if enet is not None:
            self.enet = enet.EfficientNet.from_name(backbone_name)
        else:
            try:
                ctor_name = f"efficientnet_{backbone_name.split('-')[-1]}"
                self.enet = getattr(models, ctor_name)(pretrained=True)
            except Exception as e:
                raise ValueError(f"Backbone {backbone_name} not supported.") from e

        nc = (
            self.enet.classifier[1].in_features
            if hasattr(self.enet, "classifier")
            else self.enet._fc.in_features
        )
        if hasattr(self.enet, "classifier"):
            self.enet.classifier = nn.Identity()
        else:
            self.enet._fc = nn.Identity()
        self.myfc = nn.Sequential(
            nn.Dropout(params["dropout"]),
            nn.Linear(nc, nc // 4),
            nn.Dropout(params["dropout"]),
            nn.Linear(nc // 4, out_dim),
        )

    def extract(self, x):
        return self.enet(x)

    def forward(self, x):
        x = self.extract(x)
        x = self.myfc(x)
        return x




## === cell 5
models = []
for n_fold in FOLDS:
    model = EffNet(params, out_dim=len(LABELS_))
    ckpt_path = f"{MDLS_PATH}/model_best_{n_fold}.pth"
    if os.path.isfile(ckpt_path):
        state_dict = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state_dict)
        print(f"Loaded checkpoint: {ckpt_path}")
    else:
        print(f"Checkpoint not found for fold {n_fold}, using random weights.")
    model = model.to(DEVICE)
    model.eval()
    models.append(model)
del model, state_dict  # clean up




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_55/1373889181.py in __init__(self, params, out_dim)
     57                 ctor_name = f"efficientnet_{backbone_name.split('-')[-1]}"
---> 58                 self.enet = getattr(models, ctor_name)(pretrained=True)
     59             except Exception as e:

AttributeError: 'list' object has no attribute 'efficientnet_b0'

The above exception was the direct cause of the following exception:

ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/4153840684.py in <cell line: 0>()
      1 models = []
      2 for n_fold in FOLDS:
----> 3     model = EffNet(params, out_dim=len(LABELS_))
      4     ckpt_path = f"{MDLS_PATH}/model_best_{n_fold}.pth"
      5     if os.path.isfile(ckpt_path):

/tmp/ipykernel_55/1373889181.py in __init__(self, params, out_dim)
     58                 self.enet = getattr(models, ctor_name)(pretrained=True)
     59             except Exception as e:
---> 60                 raise ValueError(f"Backbone {backbone_name} not supported.") from e
     61 
     62         # Determine the number of features coming out of the backbone

ValueError: Backbone efficientnet-b0 not supported.

## === cell 6
datasets, loaders = [], []
for tta in TTAS[:1]:  # limit to a single augmentation to keep shapes consistent
    dataset = PlantDataset(
        df=df_sub, size=params["img_size"], labels=None, transform=None, tta=tta
    )
    datasets.append(dataset)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=params["batch_size"],
        sampler=SequentialSampler(dataset),
        num_workers=WORKERS,
        pin_memory=True,
    )
    loaders.append(loader)




## === cell 7
def get_labels(row, labels_dict, th):
    try:
        idxs = [i for i, x in enumerate(row) if x > th]
        names = [labels_dict[str(i)] for i in idxs]
        return "healthy" if ("healthy" in names or len(names) == 0) else " ".join(names)
    except Exception as e:
        print(f"Label decoding error: {e}")
        return "healthy"


model = models[0]
loader = loaders[0]
all_logits = []
with torch.no_grad():
    for batch in loader:
        batch = batch.to(DEVICE)
        preds = torch.sigmoid(model(batch)).cpu().numpy()
        all_logits.append(preds)
logits = np.vstack(all_logits)  # shape: (num_images, num_classes)

df_sub["labels"] = [get_labels(x, LABELS, TH) for x in logits]

elapsed = time.time() - start_time
print(f"Time elapsed: {int(elapsed // 60)} min {int(elapsed % 60)} sec")
print("Sample predictions:")
display(df_sub.head())




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
/tmp/ipykernel_55/3081047572.py in <cell line: 0>()
      9 
     10 
---> 11 model = models[0]
     12 loader = loaders[0]
     13 all_logits = []

IndexError: list index out of range

## === cell 8
output_path = "submission.csv"
df_sub.to_csv(output_path, index=False)
print(f"Submission written to {output_path}")

## --- ERROR in outputing the csv:
Invalid submission: Submission and answers DataFrames must have the same number of rows.
