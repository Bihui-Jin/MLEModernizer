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

23.26740549171348

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

import os
import glob
import gc
import random
from tqdm.auto import tqdm

import numpy as np
import pandas as pd

from PIL import Image

import albumentations as A

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import timm

from sklearn.model_selection import KFold
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
    "tf_efficientnet_l2": {
        "model_path": "/kaggle/input/tf_efficientnet-5-folds/pytorch/default/1",
        "im_size": 475,
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


def find_dataset_root():
    candidates = [
        Config.data_dir,
        "/kaggle/data/petfinder-pawpularity-score",
        "/kaggle/data/input/petfinder-pawpularity-score",
        "/kaggle/data/petfinder-pawpularity-score/petfinder-pawpularity-score",
        "/kaggle/data/input/petfinder-pawpularity-score/petfinder-pawpularity-score",
        "/kaggle/working/petfinder-pawpularity-score/petfinder-pawpularity-score",
    ]
    for p in candidates:
        if os.path.isfile(os.path.join(p, "train.csv")) and os.path.isfile(
            os.path.join(p, "test.csv")
        ):
            if os.path.isdir(os.path.join(p, "train")) and os.path.isdir(
                os.path.join(p, "test")
            ):
                return p

    for base in ["/kaggle/data", "/kaggle/input", "/kaggle/working"]:
        for p in glob.glob(
            os.path.join(base, "**", "petfinder-pawpularity-score"), recursive=True
        ):
            if os.path.isfile(os.path.join(p, "train.csv")) and os.path.isfile(
                os.path.join(p, "test.csv")
            ):
                if os.path.isdir(os.path.join(p, "train")) and os.path.isdir(
                    os.path.join(p, "test")
                ):
                    return p
    raise FileNotFoundError(
        "Could not locate dataset root containing train.csv/test.csv and train/test folders."
    )


Config.data_dir = find_dataset_root()
print("Resolved Config.data_dir =", Config.data_dir)



## === cell 3
train = pd.read_csv(f"{Config.data_dir}/train.csv")
test = pd.read_csv(f"{Config.data_dir}/test.csv")

train["path"] = train["Id"].map(lambda x: f"{Config.data_dir}/train/{x}.jpg")
test["path"] = test["Id"].map(lambda x: f"{Config.data_dir}/test/{x}.jpg")

missing_train = (~train["path"].map(os.path.isfile)).sum()
missing_test = (~test["path"].map(os.path.isfile)).sum()
print(train.shape, test.shape)
print(
    "Missing train images:",
    int(missing_train),
    "Missing test images:",
    int(missing_test),
)




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

        image = image / 255.0  # Convert to 0-1
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

    all_embeddings = None

    model_files = sorted(glob.glob(f"{model_path}/*.pth"))
    if len(model_files) == 0:
        raise FileNotFoundError(f"No .pth files found in model_path={model_path}")

    with torch.no_grad():
        print(
            f"Extract Embeddings for {model_arch} from {model_path} ({len(model_files)} checkpoints)"
        )
        for model_file in model_files:
            current_embeddings = []

            model = PetNet(model_name=model_arch)
            state = torch.load(model_file, map_location=device)
            model.load_state_dict(state)
            model = model.to(device)
            model.eval()

            for batch in tqdm(dataloader, leave=False):
                images = batch.to(device)
                outputs = model.model(images)
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


def _try_joblib_load(path):
    try:
        if os.path.isfile(path):
            return joblib.load(path)
    except Exception:
        return None
    return None




## === cell 8
for model_name, model_info in models.items():
    train_emb_path = f"{Config.embedding_dir}/{model_name}_train_embeddings.pkl"
    train_embeddings = _try_joblib_load(train_emb_path)

    if train_embeddings is None:
        print(
            f"[WARN] Missing precomputed train embeddings at {train_emb_path}. Extracting from images instead."
        )
        train_dataset = PetDataset(
            image_filepaths=train["path"].values,
            targets=None,
            transform=get_inference_fixed_transforms(model_info["im_size"]),
        )
        train_embeddings = extract_embeddings(
            model_arch=model_name,
            model_path=model_info["model_path"],
            im_size=model_info["im_size"],
            batch_size=Config.batch_size,
            dataset=train_dataset,
        )

    models[model_name]["train_emb"] = train_embeddings

    test_dataset = PetDataset(
        image_filepaths=test["path"].values,
        targets=None,
        transform=get_inference_fixed_transforms(model_info["im_size"]),
    )
    test_embeddings = extract_embeddings(
        model_arch=model_name,
        model_path=model_info["model_path"],
        im_size=model_info["im_size"],
        batch_size=Config.batch_size,
        dataset=test_dataset,
    )
    models[model_name]["test_emb"] = test_embeddings




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/2089208386.py in <cell line: 0>()
     14             transform=get_inference_fixed_transforms(model_info["im_size"]),
     15         )
---> 16         train_embeddings = extract_embeddings(
     17             model_arch=model_name,
     18             model_path=model_info["model_path"],

/tmp/ipykernel_55/1743807508.py in extract_embeddings(model_arch, model_path, im_size, batch_size, dataset)
     14     model_files = sorted(glob.glob(f"{model_path}/*.pth"))
     15     if len(model_files) == 0:
---> 16         raise FileNotFoundError(f"No .pth files found in model_path={model_path}")
     17 
     18     with torch.no_grad():

FileNotFoundError: No .pth files found in model_path=/kaggle/input/beit_large__512_5fold/transformers/default/1

## === cell 9
def _ensure_2d(a):
    a = np.asarray(a)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    return a


TRAIN = np.concatenate(
    [_ensure_2d(models[name]["train_emb"]) for name in models.keys()], axis=1
)
TEST = np.concatenate(
    [_ensure_2d(models[name]["test_emb"]) for name in models.keys()], axis=1
)
targets_train = train["Pawpularity"].values.astype(np.float32)

print("TRAIN:", TRAIN.shape, "TEST:", TEST.shape, "y:", targets_train.shape)




## === cell 10
def svr_predict(TEST, n_splits=5):
    ypredtest_ = np.zeros(TEST.shape[0], dtype=np.float32)

    loaded = 0
    for index in range(n_splits):
        model_path = f"{Config.svr_dir}/svr_model_fold_{index}.joblib"
        model = _try_joblib_load(model_path)
        if model is None:
            raise FileNotFoundError(
                f"Missing saved SVR model: {model_path}. "
                f"Please ensure Config.svr_dir points to available joblib models in this environment."
            )
        ypredtest_ += np.clip(model.predict(TEST), 1, 100).astype(np.float32)
        loaded += 1
        del model
        gc.collect()

    ypredtest_ /= max(1, loaded)
    return ypredtest_


ypred_test = svr_predict(TEST, n_splits=5)
print(
    "Preds:",
    ypred_test.shape,
    "min/max",
    float(ypred_test.min()),
    float(ypred_test.max()),
)



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/1693150999.py in <cell line: 0>()
     22 
     23 
---> 24 ypred_test = svr_predict(TEST, n_splits=5)
     25 print(
     26     "Preds:",

/tmp/ipykernel_55/1693150999.py in svr_predict(TEST, n_splits)
      9         model = _try_joblib_load(model_path)
     10         if model is None:
---> 11             raise FileNotFoundError(
     12                 f"Missing saved SVR model: {model_path}. "
     13                 f"Please ensure Config.svr_dir points to available joblib models in this environment."

FileNotFoundError: Missing saved SVR model: /kaggle/input/svr-4-models-5-folds/pytorch/default/1/svr_model_fold_0.joblib. Please ensure Config.svr_dir points to available joblib models in this environment.

## === cell 11
test["Pawpularity"] = np.array(ypred_test, dtype=np.float32)
output = test[["Id", "Pawpularity"]].copy()
output.head()



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1636654562.py in <cell line: 0>()
----> 1 test["Pawpularity"] = np.array(ypred_test, dtype=np.float32)
      2 output = test[["Id", "Pawpularity"]].copy()
      3 output.head()
      4 

NameError: name 'ypred_test' is not defined

## === cell 12
output.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", output.shape)
print(output.head())

## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3491186021.py in <cell line: 0>()
      1 # Ensure valid submission file with .csv suffix and correct columns.
----> 2 output.to_csv("submission.csv", index=False)
      3 print("Wrote submission.csv with shape:", output.shape)
      4 print(output.head())

NameError: name 'output' is not defined
