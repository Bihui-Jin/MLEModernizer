# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import warnings

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
from torch.optim.lr_scheduler import OneCycleLR
import torch
import timm
import torch.nn as nn
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

import glob
import joblib

try:
    from cuml.svm import SVR as CumlSVR
except Exception:
    CumlSVR = None
from sklearn.svm import SVR as SklearnSVR

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
    pretrained = True  # use pretrained weights when checkpoints are missing
    inp_channels = 3
    batch_size = 32  # increased from 1 to speed up embedding extraction
    num_workers = 8
    out_features = 1
    dropout = 0.1
    scheduler_name = "OneCycleLR"




## === cell 2
def seed_everything(seed=Config.random_seed):
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True


seed_everything()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        image_path = self.image_filepaths[idx]
        with open(image_path, "rb") as f:
            img = Image.open(f).convert("RGB")
        img = np.array(img)
        if self.transform is not None:
            img = self.transform(image=img)["image"]
        img = img / 255.0
        img = np.transpose(img, (2, 0, 1)).astype(np.float32)
        img = torch.tensor(img, dtype=torch.float)
        if self.targets is not None:
            return img, torch.tensor(self.targets[idx], dtype=torch.float)
        return img




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
            model_name, pretrained=pretrained, in_chans=inp_channels, num_classes=0
        )
        self.fc1 = nn.Linear(self.model.num_features, 128)
        self.dropout = nn.Dropout(Config.dropout)
        self.fc2 = nn.Linear(128, out_features)

    def forward(self, x):
        x = self.model(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x




## === cell 7
def extract_embeddings(model_arch, model_path, im_size, dataset):
    """
    Extract embeddings for all images once per model architecture,
    concatenating the features from all available checkpoints (or the pretrained
    weights if no checkpoint exists). This avoids re‑reading and re‑forwarding
    the whole dataset for each checkpoint, preserving the exact concatenated
    feature matrix while dramatically reducing runtime.
    """
    loader = DataLoader(
        dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    checkpoint_files = glob.glob(f"{model_path}/*.pth")
    if not checkpoint_files:
        checkpoint_files = [None]  # placeholder for "no checkpoint"

    models = []
    for ckpt in checkpoint_files:
        model = PetNet(model_name=model_arch)
        if ckpt is not None:
            try:
                model.load_state_dict(torch.load(ckpt, map_location=device))
            except Exception as e:
                print(f"Warning: failed to load {ckpt}: {e}")
        model.to(device)
        model.eval()
        models.append(model)

    all_batch_emb = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Embedding {model_arch}"):
            imgs = batch[0] if isinstance(batch, (list, tuple)) else batch
            imgs = imgs.to(device, non_blocking=True)

            batch_embs = []
            for m in models:
                emb = m.model(imgs)  # raw backbone features
                batch_embs.append(emb.cpu().numpy())
            batch_concat = np.concatenate(batch_embs, axis=1)
            all_batch_emb.append(batch_concat)

    all_emb = np.concatenate(all_batch_emb, axis=0)

    for m in models:
        del m
    torch.cuda.empty_cache()
    gc.collect()

    return all_emb




## === cell 8
train_dataset = PetDataset(
    image_filepaths=train["path"].values,
    targets=None,
    transform=get_inference_fixed_transforms(384),  # size will be overridden per model
)

test_dataset = PetDataset(
    image_filepaths=test["path"].values,
    targets=None,
    transform=get_inference_fixed_transforms(384),
)

for model_name, info in models.items():
    im_sz = info["im_size"]
    train_dataset.transform = get_inference_fixed_transforms(im_sz)
    test_dataset.transform = get_inference_fixed_transforms(im_sz)

    train_emb = extract_embeddings(model_name, info["model_path"], im_sz, train_dataset)
    models[model_name]["train_emb"] = train_emb

    test_emb = extract_embeddings(model_name, info["model_path"], im_sz, test_dataset)
    models[model_name]["test_emb"] = test_emb




## === cell 9
TRAIN = np.concatenate([models[m]["train_emb"] for m in models], axis=1)
TEST = np.concatenate([models[m]["test_emb"] for m in models], axis=1)
targets_train = train["Pawpularity"].values
print("TRAIN shape:", TRAIN.shape, "TEST shape:", TEST.shape)




## === cell 10
def fit_svr(TRAIN, TEST, y, n_splits=5):
    """Fit SVR on GPU if cuML is available, otherwise fallback to sklearn."""
    pred_train = np.zeros(TRAIN.shape[0])
    pred_test = np.zeros(TEST.shape[0])
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for fold, (tr_idx, val_idx) in enumerate(tqdm(kf.split(TRAIN), desc="CV folds")):
        X_tr, X_val = TRAIN[tr_idx], TRAIN[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        if CumlSVR is not None:
            model = CumlSVR(C=16.0, kernel="rbf", degree=3, max_iter=4000)
        else:
            model = SklearnSVR(C=16.0, kernel="rbf", degree=3, max_iter=4000)

        model.fit(X_tr, np.clip(y_tr, 1, 85))
        pred_train[val_idx] = np.clip(model.predict(X_val), 1, 100)
        pred_test += np.clip(model.predict(TEST), 1, 100)

        del model
        gc.collect()

    pred_test /= n_splits
    return pred_train, pred_test




## === cell 11
ypred_train, ypred_test = fit_svr(TRAIN, TEST, targets_train)




## === cell 12
test["Pawpularity"] = ypred_test
submission = test[["Id", "Pawpularity"]]
print(submission.head())




## === cell 13
submission.to_csv("submission.csv", index=False)
print("Saved submission.csv")
