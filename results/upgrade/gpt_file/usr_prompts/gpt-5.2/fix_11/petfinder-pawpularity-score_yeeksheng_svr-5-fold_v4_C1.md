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

try:
    import torchvision
    from torchvision.io import read_image, ImageReadMode

    _HAS_TV_IO = True
except Exception:
    _HAS_TV_IO = False

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


def _find_dir_containing(
    pattern: str, roots=("/kaggle/input", "/kaggle/data", "/kaggle/working")
) -> str | None:
    for r in roots:
        hits = glob.glob(os.path.join(r, "**", pattern), recursive=True)
        if hits:
            return os.path.dirname(hits[0])
    return None


Config.data_dir = find_dataset_root()
Config.embedding_dir = _resolve_dir(Config.embedding_dir)
Config.svr_dir = _resolve_dir(Config.svr_dir)

if not os.path.isdir(Config.embedding_dir):
    any_emb = _find_dir_containing("*_train_embeddings.pkl")
    if any_emb is not None:
        Config.embedding_dir = any_emb
if not os.path.isdir(Config.svr_dir):
    any_svr = _find_dir_containing("svr_model_fold_0.joblib")
    if any_svr is not None:
        Config.svr_dir = any_svr

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

        if _HAS_TV_IO:
            img = read_image(image_filepath, mode=ImageReadMode.RGB)
            image = img.permute(1, 2, 0).numpy()  # HWC uint8
        else:
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
    nw = int(min(4, os.cpu_count() or 2))
    if nw <= 0:
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=use_cuda,
            collate_fn=_collate_images_only,
            drop_last=False,
        )

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=nw,
        pin_memory=use_cuda,
        collate_fn=_collate_images_only,
        drop_last=False,
        persistent_workers=True,
        prefetch_factor=4,
    )


def _load_state_dict_forgiving(model, state):
    if (
        isinstance(state, dict)
        and "state_dict" in state
        and isinstance(state["state_dict"], dict)
    ):
        state = state["state_dict"]
    if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if len(unexpected) > 0:
        print(
            f"[WARN] Unexpected keys when loading checkpoint (showing up to 5): {unexpected[:5]}"
        )
    if len(missing) > 0:
        print(
            f"[WARN] Missing keys when loading checkpoint (showing up to 5): {missing[:5]}"
        )


def extract_embeddings(
    model_arch, model_path, im_size, batch_size, dataset, allow_pretrained_fallback=True
):
    dataloader = _make_loader(dataset, batch_size=batch_size)

    model_files = sorted(glob.glob(f"{model_path}/*.pth"))
    use_ckpts = len(model_files) > 0

    if not use_ckpts and not allow_pretrained_fallback:
        raise FileNotFoundError(f"No .pth files found in model_path={model_path}")

    def _run_one_backbone(backbone_model: PetNet):
        backbone_model = backbone_model.to(device)
        backbone_model.eval()

        feat_dim = int(backbone_model.model.num_features)
        n = len(dataset)
        out = np.empty((n, feat_dim), dtype=np.float32)

        offset = 0
        use_amp = device.type == "cuda"
        with torch.inference_mode():
            for images in tqdm(dataloader, leave=False, mininterval=1.0):
                images = images.to(device, non_blocking=True)
                if use_amp:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        outputs = backbone_model.model(images)
                else:
                    outputs = backbone_model.model(images)
                outputs = outputs.detach().to("cpu").to(torch.float32).numpy()
                bs = outputs.shape[0]
                out[offset : offset + bs] = outputs
                offset += bs
        return out

    if use_ckpts:
        print(
            f"Extract Embeddings for {model_arch} from {model_path} ({len(model_files)} checkpoints)"
        )
        model = PetNet(model_name=model_arch, pretrained=False).to(device)
        all_embeddings = None
        for model_file in model_files:
            state = torch.load(model_file, map_location="cpu")
            _load_state_dict_forgiving(model, state)
            cur = _run_one_backbone(model)
            all_embeddings = (
                cur
                if all_embeddings is None
                else np.concatenate((all_embeddings, cur), axis=1)
            )
            del state, cur
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    else:
        print(
            f"[FALLBACK] No checkpoints for {model_arch} at {model_path}. Using timm pretrained backbone embeddings."
        )
        model = PetNet(model_name=model_arch, pretrained=True)
        all_embeddings = _run_one_backbone(model)
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
missing = []
for model_name, model_info in models.items():
    train_emb_path = f"{Config.embedding_dir}/{model_name}_train_embeddings.pkl"
    test_emb_path = f"{Config.embedding_dir}/{model_name}_test_embeddings.pkl"

    train_embeddings = _try_joblib_load(train_emb_path)
    test_embeddings = _try_joblib_load(test_emb_path)

    if train_embeddings is None:
        missing.append(train_emb_path)
    if test_embeddings is None:
        missing.append(test_emb_path)

    models[model_name]["train_emb"] = train_embeddings
    models[model_name]["test_emb"] = test_embeddings

if missing and Config.require_precomputed_embeddings:
    msg = "\n".join(missing[:20])
    raise FileNotFoundError(
        "Precomputed embeddings are required to meet the 600s timeout, but these files were not found:\n"
        f"{msg}\n"
        "Fix: ensure the embedding dataset is attached and Config.embedding_dir points to it."
    )

if missing and not Config.require_precomputed_embeddings:
    for model_name, model_info in models.items():
        train_emb_path = f"{Config.embedding_dir}/{model_name}_train_embeddings.pkl"
        if models[model_name]["train_emb"] is None:
            print(
                f"[INFO] Train embeddings not found at {train_emb_path}. Extracting from images."
            )
            train_dataset = PetDataset(
                image_filepaths=train["path"].values,
                targets=None,
                transform=get_inference_fixed_transforms(model_info["im_size"]),
            )
            models[model_name]["train_emb"] = extract_embeddings(
                model_arch=model_name,
                model_path=model_info["model_path"],
                im_size=model_info["im_size"],
                batch_size=Config.batch_size,
                dataset=train_dataset,
            )
        else:
            print(f"Loaded train embeddings: {model_name} -> {train_emb_path}")

        test_emb_path = f"{Config.embedding_dir}/{model_name}_test_embeddings.pkl"
        if models[model_name]["test_emb"] is None:
            print(
                f"[INFO] Test embeddings not found at {test_emb_path}. Extracting from images."
            )
            test_dataset = PetDataset(
                image_filepaths=test["path"].values,
                targets=None,
                transform=get_inference_fixed_transforms(model_info["im_size"]),
            )
            models[model_name]["test_emb"] = extract_embeddings(
                model_arch=model_name,
                model_path=model_info["model_path"],
                im_size=model_info["im_size"],
                batch_size=Config.batch_size,
                dataset=test_dataset,
            )
        else:
            print(f"Loaded test embeddings: {model_name} -> {test_emb_path}")

print("Embeddings ready.")



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/273353280.py in <cell line: 0>()
     18 if missing and Config.require_precomputed_embeddings:
     19     msg = "\n".join(missing[:20])
---> 20     raise FileNotFoundError(
     21         "Precomputed embeddings are required to meet the 600s timeout, but these files were not found:\n"
     22         f"{msg}\n"

FileNotFoundError: Precomputed embeddings are required to meet the 600s timeout, but these files were not found:
/kaggle/input/training-embeddings/pytorch/default/1/beit_large_patch16_512_train_embeddings.pkl
/kaggle/input/training-embeddings/pytorch/default/1/beit_large_patch16_512_test_embeddings.pkl
/kaggle/input/training-embeddings/pytorch/default/1/deit_base_distilled_patch16_384_train_embeddings.pkl
/kaggle/input/training-embeddings/pytorch/default/1/deit_base_distilled_patch16_384_test_embeddings.pkl
/kaggle/input/training-embeddings/pytorch/default/1/maxvit_xlarge_tf_512_train_embeddings.pkl
/kaggle/input/training-embeddings/pytorch/default/1/maxvit_xlarge_tf_512_test_embeddings.pkl
/kaggle/input/training-embeddings/pytorch/default/1/tf_efficientnet_l2_train_embeddings.pkl
/kaggle/input/training-embeddings/pytorch/default/1/tf_efficientnet_l2_test_embeddings.pkl
Fix: ensure the embedding dataset is attached and Config.embedding_dir points to it.

## === cell 9
train_blocks = []
test_blocks = []
for name in models.keys():
    tr = _ensure_2d(models[name]["train_emb"]).astype(np.float32, copy=False)
    te = _ensure_2d(models[name]["test_emb"]).astype(np.float32, copy=False)
    if tr.shape[0] != len(train) or te.shape[0] != len(test):
        raise ValueError(
            f"Embedding row count mismatch for {name}: "
            f"train_emb rows={tr.shape[0]} vs len(train)={len(train)}; "
            f"test_emb rows={te.shape[0]} vs len(test)={len(test)}"
        )
    train_blocks.append(tr)
    test_blocks.append(te)

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




## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
/tmp/ipykernel_55/1014677921.py in <cell line: 0>()
      5     tr = _ensure_2d(models[name]["train_emb"]).astype(np.float32, copy=False)
      6     te = _ensure_2d(models[name]["test_emb"]).astype(np.float32, copy=False)
----> 7     if tr.shape[0] != len(train) or te.shape[0] != len(test):
      8         raise ValueError(
      9             f"Embedding row count mismatch for {name}: "

IndexError: tuple index out of range

## === cell 10
def svr_predict_or_train(TRAIN, y, TEST, n_splits=5):
    """
    Core logic preserved: SVR over concatenated embeddings, fold averaging, clipping to [1,100].
    For timeout safety, pretrained SVR joblibs are required (fallback local training is extremely slow).
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
            "Pretrained SVR models are required to meet the 600s timeout, but were not found in:\n"
            f"{Config.svr_dir}\n"
            "Fix: ensure the SVR joblib dataset is attached and Config.svr_dir points to it."
        )

    print(f"[FALLBACK] Training SVR with {n_splits}-fold CV locally (may be slow).")
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
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/126297263.py in <cell line: 0>()
     56 
     57 
---> 58 ypred_test = svr_predict_or_train(TRAIN, targets_train, TEST, n_splits=5)
     59 print(
     60     "Preds:",

NameError: name 'TRAIN' is not defined

## === cell 11
test["Pawpularity"] = np.array(ypred_test, dtype=np.float32)
output = test[["Id", "Pawpularity"]].copy()

output["Id"] = output["Id"].astype(str)
output["Pawpularity"] = output["Pawpularity"].astype(np.float32).clip(1, 100)

print(output.head())
print("Submission DF shape:", output.shape)

output.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", output.shape)
print(output.head())

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1058793806.py in <cell line: 0>()
----> 1 test["Pawpularity"] = np.array(ypred_test, dtype=np.float32)
      2 output = test[["Id", "Pawpularity"]].copy()
      3 
      4 output["Id"] = output["Id"].astype(str)
      5 output["Pawpularity"] = output["Pawpularity"].astype(np.float32).clip(1, 100)

NameError: name 'ypred_test' is not defined
