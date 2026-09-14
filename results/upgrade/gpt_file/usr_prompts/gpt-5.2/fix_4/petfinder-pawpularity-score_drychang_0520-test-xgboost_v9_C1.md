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

20.46575363230089

# 6. Current score

23.03777

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 23.47778) has done: 'I fix the immediate runtime failure by making the weight loading robust: if the referenced checkpoint path doesn’t exist in this environment, the script fall back to using the model as-is (random weights) instead of crashing, ensuring a submission CSV is always produced. I also fix a subtle DataLoader/squeeze issue that can break when the last batch has size 1 by enforcing a consistent `(batch,)` prediction shape. Finally, I make the paths flexible so the code works with either `/kaggle/input/...` or the provided `/kaggle/data/...` layout without changing the core model or inference logic.'
- What this solution (achieved 23.4424) has done: 'Your current score (23.48 RMSE) is worse than the target (20.47), and the biggest issue is that the model is running with random weights when the checkpoint isn’t present, which caps performance. I keep your exact model and inference semantics, but add an optional “safe download” of the public checkpoint via `kagglehub` (works on Kaggle without internet) so the weights are actually available in this environment. I also fix metadata scaling to use train-fitted statistics (instead of fitting on test), which avoids distribution shift and typically reduces RMSE without changing the model architecture. Finally, I keep the same submission schema and paths, and still fall back to random weights if the checkpoint cannot be found.'
- What this solution (achieved 23.03777) has done: 'Your current RMSE (23.4424) is worse than the target (20.4658), so we should improve performance modestly without changing the model itself. The biggest safe gain here is fixing the input normalization to match what ViT models in timm expect: your current `(mean,std)=(0.5,0.5,0.5)` is a common mistake and can significantly hurt a pretrained checkpoint’s usefulness. I swap the normalization to ImageNet mean/std (keeping the same resize and tensor conversion), and I also make checkpoint loading tolerant to common state_dict wrapper keys (`state_dict`, `model`) without changing weights. Everything else (architecture, sigmoid*100, metadata scaling, submission format/paths) stays the same.'

# 9. Code solution

## === cell 0
import os
import torch
import timm
import pandas as pd
import numpy as np
import cv2
from torch import nn
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 224  # match vit_base_patch16_224 input
BATCH_SIZE = 32


class PretrainViT(nn.Module):
    def __init__(self, input_size=(3, 64, 64), logger=None):
        super(PretrainViT, self).__init__()
        self.vit = timm.create_model(
            "vit_base_patch16_224", pretrained=False, num_classes=1024
        )
        self.linear = nn.Sequential(
            nn.Linear(1024, 512), nn.LeakyReLU(), nn.Linear(512, 1)
        )
        self.logger = logger
        if self.logger:
            self.logger.info(f"ViT model initialized with input size: {input_size}")

    def forward(self, images, metadata=None):
        x = self.vit(images)
        x = self.linear(x)
        return torch.sigmoid(x)


class PetTestDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, scaler=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.meta_cols = [
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

        self.scaler = scaler
        if self.scaler is not None and all(
            col in self.df.columns for col in self.meta_cols
        ):
            self.df[self.meta_cols] = self.scaler.transform(self.df[self.meta_cols])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["Id"] + ".jpg")
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found or unreadable: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image=image)["image"]

        metadata = (
            torch.tensor(row[self.meta_cols].values.astype(np.float32))
            if all(col in row for col in self.meta_cols)
            else torch.zeros(12, dtype=torch.float32)
        )

        return image, metadata, row["Id"]


transform = A.Compose(
    [
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)




## === cell 1
def pick_first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


test_csv_path = pick_first_existing(
    [
        "/kaggle/input/petfinder-pawpularity-score/test.csv",
        "/kaggle/data/petfinder-pawpularity-score/test.csv",
        "/kaggle/input/test.csv",
        "/kaggle/data/test.csv",
    ]
)

test_img_dir = pick_first_existing(
    [
        "/kaggle/input/petfinder-pawpularity-score/test",
        "/kaggle/data/petfinder-pawpularity-score/test",
        "/kaggle/input/test",
        "/kaggle/data/test",
    ]
)

train_csv_path = pick_first_existing(
    [
        "/kaggle/input/petfinder-pawpularity-score/train.csv",
        "/kaggle/data/petfinder-pawpularity-score/train.csv",
        "/kaggle/input/train.csv",
        "/kaggle/data/train.csv",
    ]
)

if test_csv_path is None or test_img_dir is None:
    raise FileNotFoundError(
        f"Could not locate test.csv or test image directory. "
        f"test_csv_path={test_csv_path}, test_img_dir={test_img_dir}"
    )

ckpt_path = None
try:
    import kagglehub  # available on Kaggle; no external internet needed for Kaggle datasets

    pretrain_dir = kagglehub.dataset_download("pretrainvit")
    target_name = "pretrainvit_rmse_17.9851.pth"
    for root, _, files in os.walk(pretrain_dir):
        if target_name in files:
            ckpt_path = os.path.join(root, target_name)
            break
except Exception as e:
    print(
        f"NOTE: kagglehub download not available/failed ({type(e).__name__}: {e}). Proceeding without it."
    )

ckpt_candidates = [
    "/kaggle/input/pretrainvit/pytorch/default/1/pretrainvit_rmse_17.9851.pth",
    "/kaggle/data/pretrainvit/pytorch/default/1/pretrainvit_rmse_17.9851.pth",
]
ckpt_path = ckpt_path or pick_first_existing(ckpt_candidates)

model = PretrainViT()

if ckpt_path is not None and os.path.exists(ckpt_path):
    state = torch.load(ckpt_path, map_location=DEVICE)
    if (
        isinstance(state, dict)
        and "state_dict" in state
        and isinstance(state["state_dict"], dict)
    ):
        state = state["state_dict"]
    elif (
        isinstance(state, dict)
        and "model" in state
        and isinstance(state["model"], dict)
    ):
        state = state["model"]

    if isinstance(state, dict):
        new_state = {}
        for k, v in state.items():
            nk = k[7:] if k.startswith("module.") else k
            new_state[nk] = v
        state = new_state

    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Loaded checkpoint: {ckpt_path}")
    if len(missing) or len(unexpected):
        print(
            f"NOTE: non-strict load. missing={len(missing)}, unexpected={len(unexpected)}"
        )
else:
    print(
        "WARNING: Pretrained checkpoint not found; running with randomly initialized weights."
    )

model.to(DEVICE)
model.eval()

meta_cols = [
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
scaler = None
if train_csv_path is not None and os.path.exists(train_csv_path):
    train_df_for_scaler = pd.read_csv(
        train_csv_path, usecols=["Id"] + meta_cols + ["Pawpularity"]
    )
    scaler = StandardScaler()
    scaler.fit(train_df_for_scaler[meta_cols].values)
    print(f"Fitted StandardScaler on train metadata from: {train_csv_path}")
else:
    print(
        "NOTE: train.csv not found; metadata will not be scaled (keeps zeros/identity behavior)."
    )



## === cell 2
test_df = pd.read_csv(test_csv_path)
test_ds = PetTestDataset(test_df, test_img_dir, transform, scaler=scaler)
test_dl = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)

ids, preds = [], []
with torch.no_grad():
    for images, metas, id_batch in tqdm(test_dl, desc="Inference"):
        images = images.to(DEVICE, non_blocking=True)

        outputs = (
            model(images).detach().float().view(-1).cpu().numpy() * 100.0
        )  # scale sigmoid to [0,100]
        outputs = np.clip(outputs, 0.0, 100.0)

        preds.extend(outputs.tolist())
        ids.extend(list(id_batch))

submission = pd.DataFrame({"Id": ids, "Pawpularity": preds})

if len(submission) != len(test_df):
    raise RuntimeError(
        f"Submission length mismatch: {len(submission)} vs test {len(test_df)}"
    )
if submission["Id"].duplicated().any():
    raise RuntimeError("Duplicate Ids found in submission.")

submission.to_csv("submission.csv", index=False)
print("submission.csv saved:", os.path.abspath("submission.csv"))
print(submission.head())
