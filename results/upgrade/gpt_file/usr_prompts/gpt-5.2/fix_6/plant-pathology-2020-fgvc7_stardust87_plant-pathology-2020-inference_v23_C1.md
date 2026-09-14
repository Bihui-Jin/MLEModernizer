# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Detect apple diseases from images.

## Metric
Mean column-wise ROC AUC.

## Submission Format
For each image_id in the test set, you must predict a probability for each target variable. The file should contain a header and have the following format:

```
image_id,
test_0,0.25,0.25,0.25,0.25
test_1,0.25,0.25,0.25,0.25
test_2,0.25,0.25,0.25,0.25
etc.
```

## Dataset
Given a photo of an apple leaf, can you accurately assess its health? This competition will challenge you to distinguish between leaves which are healthy, those which are infected with apple rust, those that have apple scab, and those with more than one disease.

**train.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

**images**

A folder containing the train and test images, in jpg format.

**test.csv**

- `image_id`: the foreign key

**sample_submission.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

# 2. Python version

3.8

# 3. Installed packages

albumentations==2.0.8
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
plotly==5.24.1
plotly-express==0.4.1
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
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        input/
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        working/
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
```

-> data/plant-pathology-2020-fgvc7/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/plant-pathology-2020-fgvc7/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/plant-pathology-2020-fgvc7/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os, time, random

import numpy as np
import pandas as pd

import albumentations as A
import cv2

import torch
import torch.nn as nn
import torchvision
import torch.optim as optim

from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
from albumentations.pytorch import ToTensorV2

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import warnings

warnings.filterwarnings("ignore")

DIR_INPUT = "/kaggle/input/plant-pathology-2020-fgvc7"
IMAGE_INPUT = DIR_INPUT

SEED = 42
N_FOLDS = 5
N_EPOCHS = 15
BATCH_SIZE = 8
IMAGE_SIZE = (
    409,
    273,
)  # (W, H) from the original comment; we will use (H, W) in transforms
LR = 1e-4  # conservative default; only used if we need to train

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(SEED)
device




## === cell 1
def _scan_images_dir(base_dir: str):
    if not os.path.isdir(base_dir):
        return {}
    mp = {}
    for fn in os.listdir(base_dir):
        if fn.lower().endswith((".jpg", ".jpeg", ".png")):
            stem, _ = os.path.splitext(fn)
            mp[stem] = os.path.join(base_dir, fn)
            mp[fn] = os.path.join(base_dir, fn)
    return mp


_BASE_CANDIDATES = [
    os.path.join(IMAGE_INPUT, "images_409_273"),
    os.path.join(IMAGE_INPUT, "images"),
]
_DIR_FILE_MAPS = {base: _scan_images_dir(base) for base in _BASE_CANDIDATES}


def resolve_image_path(image_id: str) -> str:
    direct = [
        image_id,
        f"{image_id}.jpg",
        f"{image_id}.JPG",
        f"{image_id}.jpeg",
        f"{image_id}.png",
    ]
    for base in _BASE_CANDIDATES:
        fmap = _DIR_FILE_MAPS.get(base, {})
        for name in direct:
            p = fmap.get(name)
            if p is not None:
                return p
    raise FileNotFoundError(
        f"Image not found for image_id={image_id}. Tried under: {_BASE_CANDIDATES} with common extensions."
    )


def build_image_path_map(image_ids):
    mp = {}
    for iid in image_ids:
        mp[iid] = resolve_image_path(iid)
    return mp




## === cell 2
class PlantDataset(Dataset):
    def __init__(self, df, transforms=None, test_set=False, image_path_map=None):
        self.df = df.reset_index(drop=True)
        self.transforms = transforms
        self.test_set = test_set
        self.image_path_map = image_path_map
        if self.transforms is None:
            self.transforms = A.Compose([ToTensorV2(p=1.0)])

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        image_id = self.df.loc[idx, "image_id"]

        if self.image_path_map is None:
            image_src = resolve_image_path(image_id)
        else:
            image_src = self.image_path_map[image_id]

        image = cv2.imread(image_src, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"cv2.imread failed for: {image_src}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        transformed = self.transforms(image=image)
        image = transformed["image"]

        if not self.test_set:
            labels = self.df.loc[
                idx, ["healthy", "multiple_diseases", "rust", "scab"]
            ].values.astype(np.float32)
            labels = torch.from_numpy(labels)
            return image, labels
        else:
            return image




## === cell 3
def trim_network_at_index(network, index=-1):
    assert index < 0, f"Param index must be negative. Received {index}"
    return nn.Sequential(*list(network.children())[:index])


class PlantModel(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.backbone = torchvision.models.resnet50(
            weights=torchvision.models.ResNet50_Weights.DEFAULT
        )
        in_features = self.backbone.fc.in_features
        self.backbone = trim_network_at_index(self.backbone, -1)
        self.logit = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        x = self.logit(x)
        return x




## === cell 4
H, W = 273, 409

transforms_valid = A.Compose(
    [
        A.Resize(height=H, width=W, p=1.0),
        A.Normalize(p=1.0),
        ToTensorV2(p=1.0),
    ]
)

transforms_train = A.Compose(
    [
        A.RandomResizedCrop(size=(H, W), scale=(0.8, 1.0), ratio=(0.9, 1.1), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.Normalize(p=1.0),
        ToTensorV2(p=1.0),
    ]
)

train_df = pd.read_csv(os.path.join(DIR_INPUT, "train.csv"))
test_df = pd.read_csv(os.path.join(DIR_INPUT, "test.csv"))

TARGET_COLS = ["healthy", "multiple_diseases", "rust", "scab"]

test_path_map = build_image_path_map(test_df["image_id"].values)

dataset_test = PlantDataset(
    df=test_df, test_set=True, transforms=transforms_valid, image_path_map=test_path_map
)


def make_loader(dataset, batch_size, shuffle, drop_last):
    use_cuda = torch.cuda.is_available()
    nw = min(8, os.cpu_count() or 2)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=nw,
        pin_memory=use_cuda,
        drop_last=drop_last,
        persistent_workers=(nw > 0),
        prefetch_factor=4 if nw > 0 else None,
    )


testloader = make_loader(dataset_test, BATCH_SIZE, shuffle=False, drop_last=False)




## === cell 5
def _prepare_model_for_fast_infer(model: nn.Module) -> nn.Module:
    model.eval()
    if torch.cuda.is_available():
        model = model.to(memory_format=torch.channels_last)
    try:
        model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
    except Exception:
        pass
    return model


def predict_probs(model: nn.Module, loader: DataLoader) -> np.ndarray:
    model = _prepare_model_for_fast_infer(model)
    all_probs = []
    with torch.inference_mode():
        for batch in loader:
            images = batch.to(device, non_blocking=True)
            if torch.cuda.is_available():
                images = images.to(memory_format=torch.channels_last)
            logits = model(images)
            probs = torch.sigmoid(logits)
            all_probs.append(probs.detach().cpu().numpy())
    return np.concatenate(all_probs, axis=0)




## === cell 6
EXTERNAL_MODEL_DIR = "/kaggle/input/plant-pathology-2020-training"


def external_fold_checkpoint_path(i_fold: int) -> str:
    return os.path.join(EXTERNAL_MODEL_DIR, f"modelF{i_fold}.pth")


def can_use_external_models(n_folds: int) -> bool:
    return os.path.isdir(EXTERNAL_MODEL_DIR) and all(
        os.path.exists(external_fold_checkpoint_path(i)) for i in range(n_folds)
    )


def load_fold_model_from_path(model_path: str) -> nn.Module:
    model = PlantModel(num_classes=4).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    return model




## === cell 7
def train_one_fold(fold: int, trn_idx: np.ndarray, val_idx: np.ndarray) -> str:
    trn_df = train_df.iloc[trn_idx].reset_index(drop=True)
    val_df = train_df.iloc[val_idx].reset_index(drop=True)

    tr_path_map = build_image_path_map(trn_df["image_id"].values)
    va_path_map = build_image_path_map(val_df["image_id"].values)

    ds_tr = PlantDataset(
        trn_df, transforms=transforms_train, test_set=False, image_path_map=tr_path_map
    )
    ds_va = PlantDataset(
        val_df, transforms=transforms_valid, test_set=False, image_path_map=va_path_map
    )

    dl_tr = make_loader(ds_tr, BATCH_SIZE, shuffle=True, drop_last=False)
    dl_va = make_loader(ds_va, BATCH_SIZE, shuffle=False, drop_last=False)

    model = PlantModel(num_classes=4).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    best_auc = -1.0
    best_path = os.path.join("/kaggle/working", f"modelF{fold}.pth")

    for epoch in range(N_EPOCHS):
        model.train()
        for images, labels in dl_tr:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        val_logits = []
        val_labels = []
        with torch.inference_mode():
            for images, labels in dl_va:
                images = images.to(device, non_blocking=True)
                logits = model(images)
                val_logits.append(logits.detach().cpu().numpy())
                val_labels.append(labels.detach().cpu().numpy())
        val_logits = np.concatenate(val_logits, axis=0)
        val_labels = np.concatenate(val_labels, axis=0)
        val_probs = 1.0 / (1.0 + np.exp(-val_logits))

        try:
            fold_auc = roc_auc_score(val_labels, val_probs, average=None)
            fold_auc = float(np.mean(fold_auc))
        except Exception:
            fold_auc = -1.0

        if fold_auc > best_auc:
            best_auc = fold_auc
            torch.save(model.state_dict(), best_path)

    return best_path




## === cell 8
start = time.perf_counter()

torch.backends.cudnn.benchmark = True

test_probs_folds = []

if can_use_external_models(N_FOLDS):
    for i_fold in range(N_FOLDS):
        mp = external_fold_checkpoint_path(i_fold)
        model = load_fold_model_from_path(mp)
        test_probs_folds.append(predict_probs(model, testloader))
else:
    strat_labels = train_df[TARGET_COLS].values.argmax(axis=1)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    for i_fold, (trn_idx, val_idx) in enumerate(skf.split(train_df, strat_labels)):
        best_path = train_one_fold(i_fold, trn_idx, val_idx)
        model = load_fold_model_from_path(best_path)
        test_probs_folds.append(predict_probs(model, testloader))

test_probs_folds = np.stack(test_probs_folds, axis=0)  # (n_folds, n_test, 4)
test_probs_mean = test_probs_folds.mean(axis=0)  # (n_test, 4)

print(
    f"Finished inference (and training if needed) in {(time.perf_counter() - start):.2f} seconds"
)
test_probs_mean.shape



## === cell 9
submission_df = pd.read_csv(os.path.join(DIR_INPUT, "sample_submission.csv"))
submission_df[TARGET_COLS] = test_probs_mean.astype(np.float32)
submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)

submission_df.head()



## === cell 10
assert submission_path.endswith(".csv")
assert submission_df.shape[0] == test_df.shape[0]
assert list(submission_df.columns) == ["image_id"] + TARGET_COLS
assert np.isfinite(submission_df[TARGET_COLS].values).all()
submission_path
