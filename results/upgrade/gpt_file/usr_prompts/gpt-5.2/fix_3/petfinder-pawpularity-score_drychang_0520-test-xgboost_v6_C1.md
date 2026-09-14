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

19.83621172974976

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, cv2, torch, timm, random, warnings
import numpy as np, pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

warnings.filterwarnings("ignore")

DATA_ROOT = "/kaggle/input/petfinder-pawpularity-score"
TRAIN_CSV = f"{DATA_ROOT}/train.csv"
TEST_CSV = f"{DATA_ROOT}/test.csv"
TRAIN_DIR = f"{DATA_ROOT}/train"
TEST_DIR = f"{DATA_ROOT}/test"

DEVICE = torch.device("cpu")  # per prompt
META_COLS = [
    "Subject Focus",
    "Eyes",
    "Face",
    "Near",
    "Action",
    "Accessory",
    "Group",
    "Collage",
    "Human",
    "Occlusion",
    "Info",
    "Blur",
]
IMAGE_SIZE = 224
BATCH_SIZE = 32

SEED = 42
EPOCHS = 2
LR = 2e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 2  # safe on Kaggle CPU

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


seed_everything(SEED)




## === cell 1
class PetDataset(Dataset):
    """
    Minimal dataset used for both train and test.
    For test: label is None and returned as -1.

    Optimization (equivalent): precompute ids, image paths, metadata matrix, and labels once in __init__
    to avoid per-item pandas iloc + Python list comprehensions (major CPU overhead).
    """

    def __init__(self, df, img_dir, transform=None, is_train=True):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.is_train = is_train

        ids = self.df["Id"].astype(str).to_numpy()
        self.ids = ids
        self.image_paths = np.char.add(
            np.char.add(img_dir, os.sep), np.char.add(ids, ".jpg")
        )

        meta = self.df[META_COLS].fillna(0.0).to_numpy(dtype=np.float32, copy=True)
        self.meta = np.ascontiguousarray(meta)

        if self.is_train:
            y = self.df["Pawpularity"].to_numpy(dtype=np.float32, copy=True)
            self.y = np.ascontiguousarray(y)
        else:
            self.y = None

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            image = self.transform(image=image)["image"]
        else:
            image = ToTensorV2()(image=image)["image"]

        metadata = torch.from_numpy(self.meta[idx])

        if self.is_train:
            y = torch.tensor(self.y[idx], dtype=torch.float32)
        else:
            y = torch.tensor(-1.0, dtype=torch.float32)

        return image, metadata, y, self.ids[idx]




## === cell 2
class EffNetWithMeta(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = timm.create_model("efficientnet_b0", pretrained=True, num_classes=0)
        self.mlp = nn.Sequential(
            nn.Linear(self.cnn.num_features + len(META_COLS), 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, image, metadata):
        img_feat = self.cnn(image)
        x = torch.cat([img_feat, metadata], dim=1)
        return self.mlp(x)




## === cell 3
train_transform = A.Compose(
    [
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2(),
    ]
)

valid_transform = A.Compose(
    [
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2(),
    ]
)

train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

idx = np.arange(len(train_df))
rng = np.random.RandomState(SEED)
rng.shuffle(idx)
split = int(0.9 * len(train_df))
tr_idx, va_idx = idx[:split], idx[split:]

tr_df = train_df.iloc[tr_idx].reset_index(drop=True)
va_df = train_df.iloc[va_idx].reset_index(drop=True)

train_ds = PetDataset(tr_df, TRAIN_DIR, transform=train_transform, is_train=True)
valid_ds = PetDataset(va_df, TRAIN_DIR, transform=valid_transform, is_train=True)
test_ds = PetDataset(test_df, TEST_DIR, transform=valid_transform, is_train=False)

dl_kwargs = dict(
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    drop_last=False,
    pin_memory=False,  # CPU device; pinning can add overhead without benefit here
)
train_dl = DataLoader(
    train_ds,
    shuffle=True,
    persistent_workers=(NUM_WORKERS > 0),
    prefetch_factor=4 if NUM_WORKERS > 0 else None,
    **dl_kwargs,
)
valid_dl = DataLoader(
    valid_ds,
    shuffle=False,
    persistent_workers=(NUM_WORKERS > 0),
    prefetch_factor=4 if NUM_WORKERS > 0 else None,
    **dl_kwargs,
)
test_dl = DataLoader(
    test_ds,
    shuffle=False,
    persistent_workers=(NUM_WORKERS > 0),
    prefetch_factor=4 if NUM_WORKERS > 0 else None,
    **dl_kwargs,
)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2334325705.py in <cell line: 0>()
     29 va_df = train_df.iloc[va_idx].reset_index(drop=True)
     30 
---> 31 train_ds = PetDataset(tr_df, TRAIN_DIR, transform=train_transform, is_train=True)
     32 valid_ds = PetDataset(va_df, TRAIN_DIR, transform=valid_transform, is_train=True)
     33 test_ds = PetDataset(test_df, TEST_DIR, transform=valid_transform, is_train=False)

/tmp/ipykernel_55/2398748480.py in __init__(self, df, img_dir, transform, is_train)
     18         self.ids = ids
     19         self.image_paths = np.char.add(
---> 20             np.char.add(img_dir, os.sep), np.char.add(ids, ".jpg")
     21         )
     22 

/usr/local/lib/python3.11/dist-packages/numpy/core/defchararray.py in add(x1, x2)
    330         # object dtype itemsize as num chars (worked on short strings).
    331         # bytes + void worked but promoting void->bytes is dubious also.
--> 332         raise TypeError(
    333             "np.char.add() requires both arrays of the same dtype kind, but "
    334             f"got dtypes: '{arr1.dtype}' and '{arr2.dtype}' (the few cases "

TypeError: np.char.add() requires both arrays of the same dtype kind, but got dtypes: 'object' and '<U4' (the few cases where this used to work often lead to incorrect results).

## === cell 4
model = EffNetWithMeta().to(DEVICE)

criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)


def rmse_from_mse(mse):
    return float(np.sqrt(max(mse, 0.0)))


for epoch in range(1, EPOCHS + 1):
    model.train()
    train_losses = []

    for images, metas, y, _ in tqdm(
        train_dl, desc=f"Epoch {epoch}/{EPOCHS} - train", leave=False
    ):
        images = images.to(DEVICE)
        metas = metas.to(DEVICE)
        y = y.to(DEVICE).view(-1, 1)

        optimizer.zero_grad(set_to_none=True)
        pred = model(images, metas)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

    model.eval()
    valid_losses = []
    with torch.inference_mode():
        for images, metas, y, _ in tqdm(
            valid_dl, desc=f"Epoch {epoch}/{EPOCHS} - valid", leave=False
        ):
            images = images.to(DEVICE)
            metas = metas.to(DEVICE)
            y = y.to(DEVICE).view(-1, 1)
            pred = model(images, metas)
            loss = criterion(pred, y)
            valid_losses.append(loss.item())

    tr_mse = float(np.mean(train_losses)) if train_losses else float("nan")
    va_mse = float(np.mean(valid_losses)) if valid_losses else float("nan")
    print(
        f"Epoch {epoch}: train RMSE {rmse_from_mse(tr_mse):.4f} | valid RMSE {rmse_from_mse(va_mse):.4f}"
    )



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2309490657.py in <cell line: 0>()
     15 
     16     for images, metas, y, _ in tqdm(
---> 17         train_dl, desc=f"Epoch {epoch}/{EPOCHS} - train", leave=False
     18     ):
     19         images = images.to(DEVICE)

NameError: name 'train_dl' is not defined

## === cell 5
model.eval()
ids, preds = [], []

with torch.inference_mode():
    for images, metas, _, id_batch in tqdm(test_dl, desc="Predicting"):
        images = images.to(DEVICE)
        metas = metas.to(DEVICE)
        outputs = model(images, metas).view(-1).cpu().numpy()
        preds.extend(outputs.tolist())
        ids.extend(list(id_batch))

preds = np.clip(np.array(preds, dtype=np.float32), 1.0, 100.0)

submission = pd.DataFrame({"Id": ids, "Pawpularity": preds})
submission = test_df[["Id"]].merge(submission, on="Id", how="left")

submission.to_csv("submission.csv", index=False)
print("submission.csv saved:", submission.shape)
print(submission.head())

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4118671232.py in <cell line: 0>()
      4 # --- Speed: inference_mode is faster than no_grad and equivalent for inference outputs.
      5 with torch.inference_mode():
----> 6     for images, metas, _, id_batch in tqdm(test_dl, desc="Predicting"):
      7         images = images.to(DEVICE)
      8         metas = metas.to(DEVICE)

NameError: name 'test_dl' is not defined
