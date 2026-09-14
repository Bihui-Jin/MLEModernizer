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

21.59534263020257

# 6. Current score

24.48831

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 23.93041) has done: 'I fix the runtime crash by removing the hard dependency on a missing external checkpoint path and instead conditionally loading weights only if a checkpoint exists in the Kaggle environment. To keep the same core ViT inference pipeline, I leave the model architecture and prediction scaling unchanged, but make the test-time augmentation deterministic by removing random augmentations from the test transform (random flips/rotations/brightness at inference can destabilize outputs and worsen RMSE). I also ensure DataLoader settings are compatible with the Kaggle runtime and that the submission is written with the required columns and a `.csv` suffix. The result run end-to-end and always produce a valid `submission.csv`.'
- What this solution (achieved 23.93041) has done: 'Your current score is worse than the target (23.93 vs 21.60; lower is better), so we should make the smallest change that legitimately improves RMSE without changing the model/training core logic. The biggest avoidable error is that you standardize the test metadata using statistics fit on the test set itself (and you do it only for test, not train), which changes feature scaling unpredictably and can hurt generalization; we instead fit the scaler on train metadata and apply it to test (no leakage of labels, and consistent preprocessing). I also make checkpoint loading tolerant to common state_dict formats (`{"state_dict": ...}`) to increase the chance you actually load meaningful weights (random weights would be catastrophically bad if that happens). Finally, I keep the exact same inference pipeline (ViT forward + sigmoid*100 + clip) and still write a valid `submission.csv`.'
- What this solution (achieved 24.4779) has done: 'Your current score (23.93 RMSE) is worse than the target (21.60), so we make the smallest changes that plausibly improve generalization without changing the model/training core logic. The main fix is to align image normalization with what ViT models are typically trained with (ImageNet mean/std) instead of (0.5, 0.5, 0.5), which often causes a sizable inference mismatch and worse RMSE. I also actually pass the (already-scaled) metadata into the model forward in a no-op safe way (without changing architecture) by keeping the call signature consistent, and make DataLoader deterministic to avoid any subtle ordering/alignment issues. Submission writing stays the same format and still produces a valid `submission.csv`.'
- What this solution (achieved 24.47048) has done: 'Your RMSE (24.4779) is worse than the target (21.5953), so we should make a small, legitimate improvement without changing the model/training core logic. The biggest fix we can do at inference-time is to use test-time augmentation ensembling (deterministic flips) to reduce prediction noise and usually improve RMSE, while keeping the same ViT forward and sigmoid*100 scaling. We also ensure the submission order exactly matches `test.csv` by reindexing to `test_df["Id"]` (not via a merge that can introduce NaNs/duplicates). Finally, we keep checkpoint-loading behavior and metadata scaling the same, and still write a valid `submission.csv`.'
- What this solution (achieved 24.48831) has done: 'Your current RMSE (24.47048) is worse than the target (21.5953), so we make the smallest inference-only changes that usually improve RMSE without changing the model architecture or training. The main issue is that your current TTA uses the same normalization twice for the hflip branch (you flip, then apply `base_transform` which includes resize/normalize again), which can distort inputs and hurt predictions; we fix the TTA pipeline so resize/normalize happens exactly once. We also add the vertical flip as an additional deterministic TTA branch (still the same forward pass and averaging) to reduce prediction variance a bit. Finally, we keep the exact same sigmoid*100 scaling, clipping, ordering, and submission format.'

# 9. Code solution

## === cell 0
import os
import glob
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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 224  # match vit_base_patch16_224 input
BATCH_SIZE = 32
input_size = (3, 224, 224)

torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


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
            raise FileNotFoundError(f"Could not read image at path: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image=image)["image"]

        has_meta = all(col in self.df.columns for col in self.meta_cols)
        metadata = (
            torch.tensor(row[self.meta_cols].values.astype(np.float32))
            if has_meta
            else torch.zeros(12, dtype=torch.float32)
        )
        return image, metadata, row["Id"]


_post_transform = A.Compose(
    [
        A.Resize(input_size[1], input_size[2]),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ]
)


def find_checkpoint_path():
    candidates = [
        "/kaggle/input/pretrainvit/pytorch/default/1/pretrainvit_rmse_17.9851.pth",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p

    for ext in ("*.pth", "*.pt"):
        hits = glob.glob(f"/kaggle/input/**/{ext}", recursive=True)
        preferred = [h for h in hits if "pretrainvit" in os.path.basename(h).lower()]
        if preferred:
            return preferred[0]
        if hits:
            return hits[0]
    return None


def _unwrap_state_dict(state):
    if isinstance(state, dict):
        for k in ("state_dict", "model", "model_state_dict", "net", "weights"):
            if k in state and isinstance(state[k], dict):
                return state[k]
    return state




## === cell 1
model = PretrainViT()
ckpt_path = find_checkpoint_path()

if ckpt_path is not None:
    state = torch.load(ckpt_path, map_location=DEVICE)
    state = _unwrap_state_dict(state)

    if isinstance(state, dict):
        new_state = {}
        for k, v in state.items():
            nk = k
            if nk.startswith("module."):
                nk = nk[len("module.") :]
            if nk.startswith("model."):
                nk = nk[len("model.") :]
            new_state[nk] = v
        state = new_state

    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Loaded checkpoint: {ckpt_path}")
    if missing:
        print(
            f"WARNING: Missing keys when loading checkpoint (showing up to 10): {missing[:10]}"
        )
    if unexpected:
        print(
            f"WARNING: Unexpected keys when loading checkpoint (showing up to 10): {unexpected[:10]}"
        )
else:
    print(
        "WARNING: No checkpoint found under /kaggle/input. Proceeding with randomly initialized weights."
    )

model.to(DEVICE)
model.eval()

test_csv_path = "/kaggle/input/petfinder-pawpularity-score/test.csv"
test_img_dir = "/kaggle/input/petfinder-pawpularity-score/test"
train_csv_path = "/kaggle/input/petfinder-pawpularity-score/train.csv"

if not os.path.exists(test_csv_path):
    test_csv_path = "/kaggle/data/petfinder-pawpularity-score/test.csv"
    test_img_dir = "/kaggle/data/petfinder-pawpularity-score/test"
    train_csv_path = "/kaggle/data/petfinder-pawpularity-score/train.csv"

test_df = pd.read_csv(test_csv_path)

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
if os.path.exists(train_csv_path):
    train_df = pd.read_csv(train_csv_path, usecols=["Id"] + meta_cols + ["Pawpularity"])
    scaler = StandardScaler()
    scaler.fit(train_df[meta_cols])
else:
    print("WARNING: train.csv not found; proceeding without metadata scaling.")

tta_transforms = [
    ("orig", A.Compose([_post_transform])),
    ("hflip", A.Compose([A.HorizontalFlip(p=1.0), _post_transform])),
    ("vflip", A.Compose([A.VerticalFlip(p=1.0), _post_transform])),
]

all_ids = test_df["Id"].tolist()
tta_preds = []

for tta_name, tform in tta_transforms:
    test_ds = PetTestDataset(test_df, test_img_dir, tform, scaler=scaler)
    test_dl = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        persistent_workers=False,
    )

    ids, preds = [], []
    with torch.no_grad():
        for images, metas, id_batch in test_dl:
            images = images.to(DEVICE, non_blocking=True)
            metas = metas.to(DEVICE, non_blocking=True)

            outputs = model(images, metas).squeeze(1).detach().cpu().numpy() * 100.0
            outputs = np.clip(outputs, 0.0, 100.0)
            preds.extend(outputs.tolist())
            ids.extend(list(id_batch))

    pred_s = pd.Series(preds, index=ids, name=tta_name)
    pred_s = pred_s.reindex(all_ids)
    if pred_s.isna().any():
        raise RuntimeError(
            f"TTA '{tta_name}' produced missing predictions for some Ids. "
            "Check dataset loading / Id alignment."
        )
    tta_preds.append(pred_s.values.astype(np.float32))

final_preds = np.mean(np.stack(tta_preds, axis=0), axis=0)
final_preds = np.clip(final_preds, 0.0, 100.0)

submission = pd.DataFrame({"Id": all_ids, "Pawpularity": final_preds})
submission.to_csv("submission.csv", index=False)
print("submission.csv saved.")
print(submission.head())
print(submission.shape)
