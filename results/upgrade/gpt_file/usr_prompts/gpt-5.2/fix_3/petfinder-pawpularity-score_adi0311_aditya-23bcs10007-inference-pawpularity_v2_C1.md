# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

3.14

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

20.08891479829771

# 6. Current score

34.56633

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

- What this solution (achieved 34.56633) has done: 'Your code already writes a valid `submission.csv`, but it likely scored poorly because the model is `pretrained=False` and no fold weights are actually available at `/kaggle/input/training-pawpularity`, so you’re effectively predicting from random weights. To move RMSE down toward the target, I (1) switch the backbone to `pretrained=True` (same architecture, same forward pass) and (2) add a minimal, deterministic fallback: if no checkpoints load, run a short training on `train.csv` and then predict test. This preserves your core model, loss (MSE), and training semantics, while ensuring predictions are learned rather than random. I also ensure submission row order matches `sample_submission.csv`’s `Id` order (safe alignment fix that can prevent accidental mismatches).'

# 9. Code solution

## === cell 0
import os
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
import albumentations as A
import timm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


seed_everything(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)




## === cell 1
class args:
    batch_size = 64
    image_size = 384
    epochs = 2
    lr = 3e-4




## === cell 2
def sigmoid(x):
    return 1 / (1 + math.exp(-x))




## === cell 3
class PawpularDataset(Dataset):
    def __init__(self, image_paths, dense_features, targets, augmentations):
        self.image_paths = image_paths
        self.dense_features = dense_features
        self.targets = targets
        self.augmentations = augmentations

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        image = cv2.imread(self.image_paths[item])
        if image is None:
            raise FileNotFoundError(f"Could not read image: {self.image_paths[item]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.augmentations is not None:
            augmented = self.augmentations(image=image)
            image = augmented["image"]

        image = np.transpose(image, (2, 0, 1)).astype(np.float32)

        features = self.dense_features[item, :]
        targets = self.targets[item]

        return {
            "image": torch.tensor(image, dtype=torch.float),
            "features": torch.tensor(features, dtype=torch.float),
            "targets": torch.tensor(targets, dtype=torch.float),
        }




## === cell 4
class PawpularModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model("resnet50", pretrained=True, in_chans=3)
        self.dropout = nn.Dropout(0.5)
        self.out = nn.Linear(1000 + 12, 1)

    def forward(self, image, features, targets=None):
        x = self.model(image)
        x = self.dropout(x)
        x = torch.cat([x, features], dim=1)
        x = self.dropout(x)
        x = self.out(x)

        if targets is not None:
            loss = nn.MSELoss()(x, targets.view(-1, 1))
            return x, loss, {"rmse": loss}
        return x


def load_weights_if_present(model: nn.Module, ckpt_path: str) -> bool:
    """
    Try to load if present and compatible; otherwise skip to keep the run end-to-end.
    """
    if not os.path.exists(ckpt_path):
        return False

    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and any(
        k in state for k in ["state_dict", "model_state_dict"]
    ):
        state = state.get("state_dict", state.get("model_state_dict"))

    if isinstance(state, dict):
        new_state = {}
        for k, v in state.items():
            nk = k
            for prefix in ["module.", "model."]:
                if nk.startswith(prefix):
                    nk = nk[len(prefix) :]
            new_state[nk] = v
        state = new_state

    try:
        model.load_state_dict(state, strict=True)
        return True
    except Exception:
        model.load_state_dict(state, strict=False)
        return True




## === cell 5
test_aug = A.Compose(
    [
        A.Resize(args.image_size, args.image_size, p=1),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0,
        ),
    ],
    p=1.0,
)



## === cell 6
BASE1 = Path("../input/petfinder-pawpularity-score")
BASE2 = Path("/kaggle/data/petfinder-pawpularity-score")
BASE = BASE1 if BASE1.exists() else BASE2

train_csv_path = BASE / "train.csv"
test_csv_path = BASE / "test.csv"
sample_sub_path = BASE / "sample_submission.csv"

train_img_dir = BASE / "train"
test_img_dir = BASE / "test"

df_train = pd.read_csv(train_csv_path)
df_test = pd.read_csv(test_csv_path)
df_sample = pd.read_csv(sample_sub_path)

dense_features = [
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

train_img_paths = [str(train_img_dir / f"{x}.jpg") for x in df_train["Id"].values]
test_img_paths = [str(test_img_dir / f"{x}.jpg") for x in df_test["Id"].values]

train_dataset = PawpularDataset(
    image_paths=train_img_paths,
    dense_features=df_train[dense_features].values.astype(np.float32),
    targets=df_train["Pawpularity"].values.astype(np.float32),
    augmentations=test_aug,  # keep same preprocessing semantics as original
)
test_dataset = PawpularDataset(
    image_paths=test_img_paths,
    dense_features=df_test[dense_features].values.astype(np.float32),
    targets=np.ones(len(test_img_paths), dtype=np.float32),
    augmentations=test_aug,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
    drop_last=False,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=2 * args.batch_size,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)




## === cell 7
def train_one_epoch(model, loader, optimizer):
    model.train()
    running = 0.0
    n = 0
    for batch in tqdm(loader, desc="Train", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        feats = batch["features"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        _, loss, _ = model(images, feats, targets=targets)
        loss.backward()
        optimizer.step()

        bs = images.size(0)
        running += loss.detach().item() * bs
        n += bs
    return running / max(n, 1)


def predict(model, loader):
    model.eval()
    fold_preds = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Predict", leave=False):
            images = batch["image"].to(device, non_blocking=True)
            feats = batch["features"].to(device, non_blocking=True)
            preds = model(images, feats)
            fold_preds.append(preds.detach().cpu().numpy().reshape(-1))
    return np.concatenate(fold_preds, axis=0)




## === cell 8
super_final_predictions = []
weights_root = Path("/kaggle/input/training-pawpularity")  # original expected dataset
n_folds = 5

any_weights_loaded = False

for i in range(n_folds):
    ckpt_path = str(weights_root / f"model_f{i}.bin")
    model = PawpularModel().to(device)

    loaded = load_weights_if_present(model, ckpt_path)
    any_weights_loaded = any_weights_loaded or loaded
    print(f"Fold {i}: weights_loaded={loaded} path={ckpt_path}")

    if not loaded and i == 0 and not any_weights_loaded:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        for ep in range(args.epochs):
            tr_loss = train_one_epoch(model, train_loader, optimizer)
            print(f"Fallback training epoch {ep+1}/{args.epochs} - MSE: {tr_loss:.4f}")

    fold_preds = predict(model, test_loader)
    super_final_predictions.append(fold_preds)

super_final_predictions = np.mean(np.column_stack(super_final_predictions), axis=1)
super_final_predictions = np.clip(super_final_predictions, 0.0, 100.0)

pred_map = dict(zip(df_test["Id"].values, super_final_predictions))
ordered_preds = np.array(
    [pred_map[i] for i in df_sample["Id"].values], dtype=np.float32
)

df_sub = pd.DataFrame({"Id": df_sample["Id"].values, "Pawpularity": ordered_preds})
df_sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", df_sub.shape)
print(df_sub.head())



## === cell 9
df = pd.read_csv("/kaggle/working/submission.csv")
print(df.shape)
print(df.head())
