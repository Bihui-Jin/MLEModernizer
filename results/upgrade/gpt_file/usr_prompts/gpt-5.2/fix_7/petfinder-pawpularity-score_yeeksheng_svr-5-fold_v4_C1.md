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
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
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
    pretrained = False  # original intent for checkpoint loading; fallback will use pretrained=True
    inp_channels = 3
    batch_size = 8  # keep as given
    num_workers = 2
    out_features = 1
    dropout = 0.1
    scheduler_name = "OneCycleLR"

    require_precomputed_embeddings = True
    require_pretrained_svr = True




## === cell 2
def seed_everything(seed=Config.random_seed):
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


seed_everything()

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {device}")

if device.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def find_dataset_root():
    candidates = [
        Config.data_dir,
        "/kaggle/input/petfinder-pawpularity-score",
        "/kaggle/data/petfinder-pawpularity-score",
        "/kaggle/data/input/petfinder-pawpularity-score",
        "/kaggle/input/petfinder-pawpularity-score/petfinder-pawpularity-score",
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


def _resolve_dir(path: str) -> str:
    if path and os.path.isdir(path):
        return path
    tails = []
    if path:
        parts = path.split("/kaggle/input/")
        if len(parts) == 2:
            tails.append(parts[1])
        parts = path.split("/kaggle/data/input/")
        if len(parts) == 2:
            tails.append(parts[1])
        parts = path.split("/kaggle/data/")
        if len(parts) == 2:
            tails.append(parts[1])
        parts = path.split("/kaggle/working/")
        if len(parts) == 2:
            tails.append(parts[1])

    bases = ["/kaggle/input", "/kaggle/data/input", "/kaggle/data", "/kaggle/working"]
    for tail in tails:
        for b in bases:
            cand = os.path.join(b, tail)
            if os.path.isdir(cand):
                return cand

    leaf = os.path.basename(path.rstrip("/")) if path else ""
    if leaf:
        for b in bases:
            for p in glob.glob(os.path.join(b, "**", leaf), recursive=True):
                if os.path.isdir(p):
                    return p
    return path


Config.data_dir = find_dataset_root()
Config.embedding_dir = _resolve_dir(Config.embedding_dir)
Config.svr_dir = _resolve_dir(Config.svr_dir)

print("Resolved Config.data_dir      =", Config.data_dir)
print(
    "Resolved Config.embedding_dir =",
    Config.embedding_dir,
    "exists:",
    os.path.isdir(Config.embedding_dir),
)
print(
    "Resolved Config.svr_dir       =",
    Config.svr_dir,
    "exists:",
    os.path.isdir(Config.svr_dir),
)



## === cell 3
train = pd.read_csv(f"{Config.data_dir}/train.csv")
test = pd.read_csv(f"{Config.data_dir}/test.csv")

train["path"] = f"{Config.data_dir}/train/" + train["Id"].astype(str) + ".jpg"
test["path"] = f"{Config.data_dir}/test/" + test["Id"].astype(str) + ".jpg"

print(train.shape, test.shape)

print("Skipping per-file existence checks for speed (paths constructed as usual).")




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
        image = Image.open(image_filepath).convert("RGB")
        image = np.asarray(image)

        if self.transform is not None:
            image = self.transform(image=image)["image"]

        image = image.astype(np.float32, copy=False) / 255.0
        image = np.transpose(image, (2, 0, 1))
        image = np.ascontiguousarray(image)
        image = torch.from_numpy(image)

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
def _collate_images_only(batch):
    return torch.stack(batch, dim=0)


def _make_loader(dataset, batch_size):
    use_cuda = torch.cuda.is_available()
    nw = int(min(8, os.cpu_count() or 2))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=nw,
        pin_memory=use_cuda,
        persistent_workers=(nw > 0),
        prefetch_factor=8 if nw > 0 else None,
        collate_fn=_collate_images_only,
    )


def extract_embeddings(
    model_arch, model_path, im_size, batch_size, dataset, allow_pretrained_fallback=True
):
    """
    Bugfix retained: if checkpoints are missing, use timm pretrained backbone.
    Performance: single-pass inference with preallocation, inference_mode, optional autocast, and no redundant loader iteration.
    Core logic preserved: embeddings come from the backbone feature extractor (model.model).
    """
    dataloader = _make_loader(dataset, batch_size=batch_size)

    model_files = sorted(glob.glob(f"{model_path}/*.pth"))
    use_ckpts = len(model_files) > 0

    if not use_ckpts and not allow_pretrained_fallback:
        raise FileNotFoundError(f"No .pth files found in model_path={model_path}")

    def _run_one(model):
        model = model.to(device)
        model.eval()

        feat_dim = int(model.model.num_features)
        n = len(dataset)
        out = np.empty((n, feat_dim), dtype=np.float32)

        offset = 0
        use_amp = device.type == "cuda"
        with torch.inference_mode():
            for images in tqdm(dataloader, leave=False):
                images = images.to(device, non_blocking=True)
                if use_amp:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        outputs = model.model(images)
                else:
                    outputs = model.model(images)
                outputs = outputs.detach().to("cpu").to(torch.float32).numpy()
                bs = outputs.shape[0]
                out[offset : offset + bs] = outputs
                offset += bs

        return out

    if use_ckpts:
        print(
            f"Extract Embeddings for {model_arch} from {model_path} ({len(model_files)} checkpoints)"
        )
        all_embeddings = None
        for model_file in model_files:
            model = PetNet(model_name=model_arch, pretrained=False)
            state = torch.load(model_file, map_location="cpu")
            model.load_state_dict(state)

            cur = _run_one(model)

            if all_embeddings is None:
                all_embeddings = cur
            else:
                all_embeddings = np.concatenate((all_embeddings, cur), axis=1)

            del model, state, cur
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
    else:
        print(
            f"[FALLBACK] No checkpoints for {model_arch} at {model_path}. Using timm pretrained backbone embeddings."
        )
        model = PetNet(model_name=model_arch, pretrained=True)
        all_embeddings = _run_one(model)

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


def _ensure_2d(a):
    a = np.asarray(a)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    return a




## === cell 8
for model_name, model_info in models.items():
    train_emb_path = f"{Config.embedding_dir}/{model_name}_train_embeddings.pkl"
    train_embeddings = _try_joblib_load(train_emb_path)

    if train_embeddings is None and Config.require_precomputed_embeddings:
        raise FileNotFoundError(
            f"Required precomputed train embeddings not found: {train_emb_path}\n"
            f"To avoid 600s timeout, fallback extraction is disabled."
        )

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
    else:
        print(f"Loaded train embeddings: {model_name} -> {train_emb_path}")

    models[model_name]["train_emb"] = train_embeddings

    test_emb_path = f"{Config.embedding_dir}/{model_name}_test_embeddings.pkl"
    test_embeddings = _try_joblib_load(test_emb_path)

    if test_embeddings is None and Config.require_precomputed_embeddings:
        raise FileNotFoundError(
            f"Required precomputed test embeddings not found: {test_emb_path}\n"
            f"To avoid 600s timeout, fallback extraction is disabled."
        )

    if test_embeddings is None:
        print(
            f"[WARN] Missing precomputed test embeddings at {test_emb_path}. Extracting from images instead."
        )
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
    else:
        print(f"Loaded test embeddings: {model_name} -> {test_emb_path}")

    models[model_name]["test_emb"] = test_embeddings



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/743219394.py in <cell line: 0>()
      6     # This keeps the intended solution path (loading precomputed embeddings) unchanged and deterministic.
      7     if train_embeddings is None and Config.require_precomputed_embeddings:
----> 8         raise FileNotFoundError(
      9             f"Required precomputed train embeddings not found: {train_emb_path}\n"
     10             f"To avoid 600s timeout, fallback extraction is disabled."

FileNotFoundError: Required precomputed train embeddings not found: /kaggle/input/training-embeddings/pytorch/default/1/beit_large_patch16_512_train_embeddings.pkl
To avoid 600s timeout, fallback extraction is disabled.

## === cell 9
train_blocks = [
    _ensure_2d(models[name]["train_emb"]).astype(np.float32, copy=False)
    for name in models.keys()
]
test_blocks = [
    _ensure_2d(models[name]["test_emb"]).astype(np.float32, copy=False)
    for name in models.keys()
]

n_train = train_blocks[0].shape[0]
n_test = test_blocks[0].shape[0]
total_dim = int(sum(b.shape[1] for b in train_blocks))

TRAIN = np.empty((n_train, total_dim), dtype=np.float32)
TEST = np.empty((n_test, total_dim), dtype=np.float32)

c = 0
for tr_b, te_b in zip(train_blocks, test_blocks):
    d = tr_b.shape[1]
    TRAIN[:, c : c + d] = tr_b
    TEST[:, c : c + d] = te_b
    c += d

targets_train = train["Pawpularity"].values.astype(np.float32)

print("TRAIN:", TRAIN.shape, "TEST:", TEST.shape, "y:", targets_train.shape)




## === cell 10
def svr_predict_or_train(TRAIN, y, TEST, n_splits=5):
    """
    Bugfix retained: load saved SVR joblibs if present; else train with KFold CV and average.
    Core logic preserved: SVR over concatenated embeddings, fold averaging, clipping to [1,100].
    """
    ypredtest_ = np.zeros(TEST.shape[0], dtype=np.float32)

    loaded_models = 0
    for index in range(n_splits):
        model_path = f"{Config.svr_dir}/svr_model_fold_{index}.joblib"
        model = _try_joblib_load(model_path)
        if model is None:
            loaded_models = 0
            break
        ypredtest_ += np.clip(model.predict(TEST), 1, 100).astype(
            np.float32, copy=False
        )
        loaded_models += 1
        del model
        gc.collect()

    if loaded_models > 0:
        ypredtest_ /= loaded_models
        return ypredtest_

    if Config.require_pretrained_svr:
        raise FileNotFoundError(
            f"Required SVR fold models not found under: {Config.svr_dir}\n"
            f"To avoid 600s timeout, fallback local SVR training is disabled."
        )

    print(
        f"[FALLBACK] No saved SVR joblibs found in {Config.svr_dir}. Training SVR with {n_splits}-fold CV locally."
    )

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=Config.random_seed)
    fold = 0
    for tr_idx, va_idx in kf.split(TRAIN):
        Xtr = TRAIN[tr_idx]
        ytr = y[tr_idx]

        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(Xtr)

        svr = SVR(C=20.0, epsilon=2.0, gamma="scale", kernel="rbf")
        svr.fit(Xtr_s, ytr)

        Xte_s = scaler.transform(TEST)
        ypredtest_ += np.clip(svr.predict(Xte_s), 1, 100).astype(np.float32, copy=False)

        fold += 1
        del scaler, svr, Xtr, Xtr_s, Xte_s, ytr
        gc.collect()

    ypredtest_ /= max(1, fold)
    return ypredtest_


ypred_test = svr_predict_or_train(TRAIN, targets_train, TEST, n_splits=5)
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
/tmp/ipykernel_55/2220543069.py in <cell line: 0>()
     59 
     60 
---> 61 ypred_test = svr_predict_or_train(TRAIN, targets_train, TEST, n_splits=5)
     62 print(
     63     "Preds:",

/tmp/ipykernel_55/2220543069.py in svr_predict_or_train(TRAIN, y, TEST, n_splits)
     27     # This keeps intended behavior (use provided fold models) and avoids timeout.
     28     if Config.require_pretrained_svr:
---> 29         raise FileNotFoundError(
     30             f"Required SVR fold models not found under: {Config.svr_dir}\n"
     31             f"To avoid 600s timeout, fallback local SVR training is disabled."

FileNotFoundError: Required SVR fold models not found under: /kaggle/input/svr-4-models-5-folds/pytorch/default/1
To avoid 600s timeout, fallback local SVR training is disabled.

## === cell 11
test["Pawpularity"] = np.array(ypred_test, dtype=np.float32)
output = test[["Id", "Pawpularity"]].copy()
print(output.head())
print("Submission DF shape:", output.shape)



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/975786608.py in <cell line: 0>()
----> 1 test["Pawpularity"] = np.array(ypred_test, dtype=np.float32)
      2 output = test[["Id", "Pawpularity"]].copy()
      3 print(output.head())
      4 print("Submission DF shape:", output.shape)
      5 

NameError: name 'ypred_test' is not defined

## === cell 12
output.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", output.shape)
print(output.head())

## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1457901592.py in <cell line: 0>()
----> 1 output.to_csv("submission.csv", index=False)
      2 print("Wrote submission.csv with shape:", output.shape)
      3 print(output.head())

NameError: name 'output' is not defined
