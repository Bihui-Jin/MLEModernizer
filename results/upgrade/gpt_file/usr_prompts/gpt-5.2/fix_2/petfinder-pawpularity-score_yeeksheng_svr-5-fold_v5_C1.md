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
Predict engagement with a pet's profile based on the photograph for that profile.

## Metric
Root mean squared error.

## Submission Format
For each `Id` in the test set, you must predict a probability for the target variable, `Pawpularity`. The file should contain a header and have the following format:

```
Id, Pawpularity
0008dbfb52aa1dc6ee51ee02adf13537, 99.24
0014a7b528f1682f0cf3b73a991c17a0, 61.71
0019c1388dfcd30ac8b112fb4250c251, 6.23
00307b779c82716b240a24f028b0031b, 9.43
00320c6dd5b4223c62a9670110d47911, 70.89
etc.
```

## Dataset
- **train/** - Folder containing training set photos of the form **{id}.jpg**, where **{id}** is a unique Pet Profile ID.
- **train.csv** - Metadata (described below) for each photo in the training set as well as the target, the photo's Pawpularity score. The Id column gives the photo's unique Pet Profile ID corresponding the photo's file name.

The train.csv and test.csv files contain metadata for photos in the training set and test set, respectively. Each pet photo is labeled with the value of 1 (Yes) or 0 (No) for each of the following features:

- **Focus** - Pet stands out against uncluttered background, not too close / far.
- **Eyes** - Both eyes are facing front or near-front, with at least 1 eye / pupil decently clear.
- **Face** - Decently clear face, facing front or near-front.
- **Near** - Single pet taking up significant portion of photo (roughly over 50% of photo width or height).
- **Action** - Pet in the middle of an action (e.g., jumping).
- **Accessory** - Accompanying physical or digital accessory / prop (i.e. toy, digital sticker), excluding collar and leash.
- **Group** - More than 1 pet in the photo.
- **Collage** - Digitally-retouched photo (i.e. with digital photo frame, combination of multiple photos).
- **Human** - Human in the photo.
- **Occlusion** - Specific undesirable objects blocking part of the pet (i.e. human, cage or fence). Note that not all blocking objects are considered occlusion.
- **Info** - Custom-added text or labels (i.e. pet name, description).
- **Blur** - Noticeably out of focus or noisy, especially for the pet's eyes and face. For Blur entries, "Eyes" column is always set to 0.

# 2. Python version

3.13

# 3. Installed packages

albumentations==2.0.8
cuml-cu12==25.2.1
geopandas==0.14.4
joblib==1.5.2
libcuml-cu12==25.2.1
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
timm==1.0.19
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124
tqdm==4.67.1

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (132 lines)
            sample_submission.csv (993 lines)
            sample_submission.csv.zip (22.7 kB)
            test.csv (993 lines)
            test.csv.zip (22.5 kB)
            test.zip (102.2 MB)
            train.csv (8921 lines)
            train.csv.zip (213.0 kB)
            train.zip (926.9 MB)
            petfinder-pawpularity-score/
                description.md (132 lines)
                sample_submission.csv (993 lines)
                ... and 7 other files
                petfinder-pawpularity-score/
                test/
                    a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                    2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                    ... and 990 other files
                    test/
                train/
                    e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                    cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                    ... and 8918 other files
                    train/
            test/
                a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                ... and 990 other files
                test/
            train/
                e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                ... and 8918 other files
                train/
        input/
            description.md (132 lines)
            sample_submission.csv (993 lines)
            sample_submission.csv.zip (22.7 kB)
            test.csv (993 lines)
            test.csv.zip (22.5 kB)
            test.zip (102.2 MB)
            train.csv (8921 lines)
            train.csv.zip (213.0 kB)
            train.zip (926.9 MB)
            petfinder-pawpularity-score/
                description.md (132 lines)
                sample_submission.csv (993 lines)
                ... and 7 other files
                petfinder-pawpularity-score/
                test/
                    a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                    2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                    ... and 990 other files
                    test/
                train/
                    e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                    cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                    ... and 8918 other files
                    train/
            test/
                a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                ... and 990 other files
                test/
                    a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                    2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                    ... and 990 other files
                    test/
            train/
                e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                ... and 8918 other files
                train/
                    e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                    cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                    ... and 8918 other files
                    train/
        working/
            petfinder-pawpularity-score/
                description.md (132 lines)
                sample_submission.csv (993 lines)
                ... and 7 other files
                petfinder-pawpularity-score/
                test/
                    a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                    2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                    ... and 990 other files
                    test/
                train/
                    e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                    cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                    ... and 8918 other files
                    train/
```

-> data/petfinder-pawpularity-score/sample_submission.csv has 992 rows and 2 columns.
The columns are: Id, Pawpularity

-> data/petfinder-pawpularity-score/test.csv has 992 rows and 13 columns.
The columns are: Id, Subject Focus, Eyes, Face, Near, Action, Accessory, Group, Collage, Human, Occlusion, Info, Blur

-> data/petfinder-pawpularity-score/train.csv has 8920 rows and 14 columns.
The columns are: Id, Subject Focus, Eyes, Face, Near, Action, Accessory, Group, Collage, Human, Occlusion, Info, Blur, Pawpularity

-> data/sample_submission.csv has 992 rows and 2 columns.
The columns are: Id, Pawpularity

-> data/test.csv has 992 rows and 13 columns.
The columns are: Id, Subject Focus, Eyes, Face, Near, Action, Accessory, Group, Collage, Human, Occlusion, Info, Blur

-> data/train.csv has 8920 rows and 14 columns.
The columns are: Id, Subject Focus, Eyes, Face, Near, Action, Accessory, Group, Collage, Human, Occlusion, Info, Blur, Pawpularity

-> (stopped after 10 files for performance)

# 5. Target score

23.724928689248216

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import warnings
import sklearn.exceptions

warnings.filterwarnings("ignore")

import pickle
from tqdm.auto import tqdm
from collections import defaultdict
import os
import numpy as np
import pandas as pd
import random
import gc

from PIL import Image

import albumentations as A

from torch.utils.data import Dataset, DataLoader
import torch
import timm
import torch.nn as nn

from sklearn.model_selection import KFold
from sklearn.svm import SVR  # CPU fallback for cuML SVR

import glob
import joblib

gc.enable()



## === cell 1
models = {
    "beit_large_patch16_512": {
        "model_path": "/kaggle/input/beit_large__512_5fold/transformers/default/1",
        "im_size": 512,
        "train_emb": [],
        "test_emb": [],
    },
    "deit_base_distilled_patch16_384": {
        "model_path": "/kaggle/input/deit-base-5-folds/pytorch/default/1",
        "im_size": 384,
        "train_emb": [],
        "test_emb": [],
    },
    "maxvit_xlarge_tf_512": {
        "model_path": "/kaggle/input/maxvit_5folds/transformers/default/1",
        "im_size": 512,
        "train_emb": [],
        "test_emb": [],
    },
}


class Config:
    data_dir = "/kaggle/input/petfinder-pawpularity-score"
    embedding_dir = "/kaggle/input/training-embeddings/pytorch/default/1"
    svr_dir = "/kaggle/input/svr-4-models-5-folds/pytorch/default/1"
    random_seed = 555
    tta_times = 1  # 1: no TTA
    tta_beta = 1 / tta_times
    pretrained = False
    inp_channels = 3
    batch_size = 1
    num_workers = 0  # >0: OS Error
    out_features = 1
    dropout = 0.1
    scheduler_name = "OneCycleLR"




## === cell 2
def seed_everything(seed=Config.random_seed):
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


seed_everything()

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {device}")



## === cell 3
train = pd.read_csv(f"{Config.data_dir}/train.csv")
test = pd.read_csv(f"{Config.data_dir}/test.csv")

train["path"] = train["Id"].map(lambda x: f"{Config.data_dir}/train/{x}.jpg")
test["path"] = test["Id"].map(lambda x: f"{Config.data_dir}/test/{x}.jpg")

print(train.shape, test.shape)




## === cell 4
def get_train_transforms(dim):
    return A.Compose(
        [
            A.SmallestMaxSize(max_size=dim, p=1.0),
            A.RandomCrop(height=dim, width=dim, p=1.0),
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
        ]
    )


def get_inference_fixed_transforms(dim):
    return A.Compose(
        [
            A.SmallestMaxSize(max_size=dim, p=1.0),
            A.CenterCrop(height=dim, width=dim, p=1.0),
        ],
        p=1.0,
    )




## === cell 5
class PetDataset(Dataset):
    def __init__(self, image_filepaths, targets=None, transform=None):
        self.image_filepaths = image_filepaths
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.image_filepaths)

    def __getitem__(self, idx):
        image_filepath = self.image_filepaths[idx]
        with open(image_filepath, "rb") as f:
            image = Image.open(f)
            image_rgb = image.convert("RGB")
        image = np.array(image_rgb)

        if self.transform is not None:
            image = self.transform(image=image)["image"]

        image = image / 255.0
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        image = torch.tensor(image, dtype=torch.float32)

        if self.targets is not None:
            target = torch.tensor(self.targets[idx], dtype=torch.float32)
            return image, target
        else:
            return image




## === cell 6
class PetNet(nn.Module):
    def __init__(
        self,
        model_name,
        out_features=Config.out_features,
        inp_channels=Config.inp_channels,
        pretrained=Config.pretrained,
    ):
        super().__init__()
        self.model = timm.create_model(
            model_name, pretrained=pretrained, in_chans=3, num_classes=0
        )
        self.fc1 = nn.Linear(self.model.num_features, 128)
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(128, 1)

    def get_image_embedding(self, image):
        return self.model(image)

    def forward(self, image):
        output = self.model(image)
        x = self.fc1(output)
        x = self.dropout(x)
        x = self.fc2(x)
        return x




## === cell 7
def extract_embeddings(model_arch, model_path, im_size, batch_size, dataset):
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model_files = sorted(glob.glob(f"{model_path}/*.pth"))
    if len(model_files) == 0:
        raise FileNotFoundError(f"No .pth files found under: {model_path}")

    all_embeddings = None

    with torch.no_grad():
        print(
            f"Extract Embeddings for {model_arch} from {model_path} ({len(model_files)} folds/checkpoints)"
        )
        for model_file in model_files:
            current_embeddings = []

            model = PetNet(model_name=model_arch)
            state = torch.load(model_file, map_location=device)
            model.load_state_dict(state, strict=True)
            model = model.to(device)
            model.eval()

            for batch in tqdm(dataloader, leave=False):
                images = batch
                images = images.to(device, non_blocking=torch.cuda.is_available())
                outputs = model.model(
                    images
                )  # keep original core: use backbone embeddings
                current_embeddings.append(outputs.detach().cpu().numpy())

            current_embeddings = np.concatenate(current_embeddings, axis=0)

            if all_embeddings is None:
                all_embeddings = current_embeddings
            else:
                all_embeddings = np.concatenate(
                    (all_embeddings, current_embeddings), axis=1
                )

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    return all_embeddings




## === cell 8
for model_name, model_info in models.items():
    model_path = model_info["model_path"]
    im_size = model_info["im_size"]

    train_emb_path = f"{Config.embedding_dir}/{model_name}_train_embeddings.pkl"

    if os.path.exists(train_emb_path):
        train_embeddings = joblib.load(train_emb_path)
        train_embeddings = np.asarray(train_embeddings)
        if train_embeddings.ndim == 1:
            train_embeddings = train_embeddings.reshape(-1, 1)
        models[model_name]["train_emb"] = train_embeddings
        print(
            f"Loaded cached train embeddings: {model_name} -> {train_embeddings.shape}"
        )
    else:
        print(
            f"Cached train embeddings not found for {model_name} at: {train_emb_path}"
        )
        print("Computing train embeddings on-the-fly (this may take some time)...")
        train_dataset = PetDataset(
            image_filepaths=train["path"].values,
            targets=None,
            transform=get_inference_fixed_transforms(im_size),
        )
        train_embeddings = extract_embeddings(
            model_name, model_path, im_size, Config.batch_size, train_dataset
        )
        models[model_name]["train_emb"] = train_embeddings
        print(f"Computed train embeddings: {model_name} -> {train_embeddings.shape}")

    test_dataset = PetDataset(
        image_filepaths=test["path"].values,
        targets=None,
        transform=get_inference_fixed_transforms(im_size),
    )
    test_embeddings = extract_embeddings(
        model_name, model_path, im_size, Config.batch_size, test_dataset
    )
    models[model_name]["test_emb"] = test_embeddings
    print(f"Computed test embeddings: {model_name} -> {test_embeddings.shape}")



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_56/2674299098.py in <cell line: 0>()
     25             transform=get_inference_fixed_transforms(im_size),
     26         )
---> 27         train_embeddings = extract_embeddings(
     28             model_name, model_path, im_size, Config.batch_size, train_dataset
     29         )

/tmp/ipykernel_56/1636137918.py in extract_embeddings(model_arch, model_path, im_size, batch_size, dataset)
     11     model_files = sorted(glob.glob(f"{model_path}/*.pth"))
     12     if len(model_files) == 0:
---> 13         raise FileNotFoundError(f"No .pth files found under: {model_path}")
     14 
     15     all_embeddings = None

FileNotFoundError: No .pth files found under: /kaggle/input/beit_large__512_5fold/transformers/default/1

## === cell 9
TRAIN_parts = []
TEST_parts = []
for name in models.keys():
    tr = np.asarray(models[name]["train_emb"])
    te = np.asarray(models[name]["test_emb"])
    if tr.ndim == 1:
        tr = tr.reshape(-1, 1)
    if te.ndim == 1:
        te = te.reshape(-1, 1)
    TRAIN_parts.append(tr)
    TEST_parts.append(te)

TRAIN = np.concatenate(TRAIN_parts, axis=1)
TEST = np.concatenate(TEST_parts, axis=1)
targets_train = train["Pawpularity"].values.astype(np.float32)

print("TRAIN:", TRAIN.shape, "TEST:", TEST.shape, "targets:", targets_train.shape)




## === cell 10
def fit_cpu_svr(TRAIN, TEST, train_targets, n_splits=5):
    ypredtrain_ = np.zeros(TRAIN.shape[0], dtype=np.float32)
    ypredtest_ = np.zeros(TEST.shape[0], dtype=np.float32)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for fold, (train_index, valid_index) in enumerate(
        tqdm(kf.split(TRAIN), total=n_splits)
    ):
        X_train, X_valid = TRAIN[train_index], TRAIN[valid_index]
        y_train, y_valid = train_targets[train_index], train_targets[valid_index]

        model = SVR(C=16.0, kernel="rbf", degree=3, max_iter=4000)

        model.fit(X_train, np.clip(y_train, 1, 85))

        ypredtrain_[valid_index] = np.clip(model.predict(X_valid), 1, 100).astype(
            np.float32
        )
        ypredtest_ += np.clip(model.predict(TEST), 1, 100).astype(np.float32)

        del model
        gc.collect()

    ypredtest_ /= n_splits
    return ypredtrain_, ypredtest_


def svr_predict(TEST, n_splits=5):
    ypredtest_ = np.zeros(TEST.shape[0], dtype=np.float32)

    for index in range(n_splits):
        model_path = f"{Config.svr_dir}/svr_model_fold_{index}.joblib"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Missing pretrained SVR model: {model_path}")
        model = joblib.load(model_path)
        ypredtest_ += np.clip(model.predict(TEST), 1, 100).astype(np.float32)
        del model
        gc.collect()

    ypredtest_ /= n_splits
    return ypredtest_




## === cell 11
ypred_train, ypred_test = fit_cpu_svr(TRAIN, TEST, targets_train)
ypred_test[:10], ypred_test.shape



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_56/43980571.py in <cell line: 0>()
      1 # Prefer training SVR directly to ensure end-to-end execution without relying on external joblib models.
----> 2 ypred_train, ypred_test = fit_cpu_svr(TRAIN, TEST, targets_train)
      3 ypred_test[:10], ypred_test.shape
      4 

/tmp/ipykernel_56/3517191985.py in fit_cpu_svr(TRAIN, TEST, train_targets, n_splits)
      7     kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
      8 
----> 9     for fold, (train_index, valid_index) in enumerate(
     10         tqdm(kf.split(TRAIN), total=n_splits)
     11     ):

/usr/local/lib/python3.11/dist-packages/tqdm/notebook.py in __iter__(self)
    248         try:
    249             it = super().__iter__()
--> 250             for obj in it:
    251                 # return super(tqdm...) will not catch exception
    252                 yield obj

/usr/local/lib/python3.11/dist-packages/tqdm/std.py in __iter__(self)
   1179 
   1180         try:
-> 1181             for obj in iterable:
   1182                 yield obj
   1183                 # Update and possibly print the progressbar.

/usr/local/lib/python3.11/dist-packages/sklearn/model_selection/_split.py in split(self, X, y, groups)
    343         n_samples = _num_samples(X)
    344         if self.n_splits > n_samples:
--> 345             raise ValueError(
    346                 (
    347                     "Cannot have number of splits n_splits={0} greater"

ValueError: Cannot have number of splits n_splits=5 greater than the number of samples: n_samples=0.

## === cell 12
test["Pawpularity"] = np.array(ypred_test, dtype=np.float32)
output = test[["Id", "Pawpularity"]]
output.head()



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/3008047947.py in <cell line: 0>()
----> 1 test["Pawpularity"] = np.array(ypred_test, dtype=np.float32)
      2 output = test[["Id", "Pawpularity"]]
      3 output.head()
      4 

NameError: name 'ypred_test' is not defined

## === cell 13
output.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", output.shape)
print(output.describe(include="all"))

## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/687772447.py in <cell line: 0>()
      1 # Ensure a valid Kaggle submission file with correct name and columns.
----> 2 output.to_csv("submission.csv", index=False)
      3 print("Wrote submission.csv with shape:", output.shape)
      4 print(output.describe(include="all"))

NameError: name 'output' is not defined
