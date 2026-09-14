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

18.742972495583928

# 6. Current score

23.32326

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

- What this solution (achieved 23.32326) has done: 'Your notebook currently won’t yield a meaningful Kaggle score because (a) it’s using test-time normalization values that don’t match the EfficientNet-B3 pretraining defaults and (b) it’s missing the standard “sigmoid + scale to [0,100]” post-processing that most Pawpularity B3 regressors use (your current clip likely collapses outputs and hurts RMSE). I keep your exact model architecture and inference loop, but change only the input normalization to EfficientNet’s expected mean/std and adjust prediction post-processing to `sigmoid * 100` (then clip), which is a minimal semantic fix for probability-like targets. I also make the test paths robust to the two common dataset locations in your environment so the script reliably produces `submission.csv`. These are small changes aimed at producing a valid submission and improving RMSE toward your target.'

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
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 256
BATCH_SIZE = 32


def find_checkpoint_path(preferred_path: str | None = None) -> str:
    """
    Try to find a .pth checkpoint if attached to the notebook.
    If none exists, caller should handle and proceed without a checkpoint.
    """
    candidates = []
    if preferred_path:
        candidates.append(preferred_path)

    search_roots = [
        "/kaggle/input",
        "/kaggle/data/input",
    ]
    for root in search_roots:
        if os.path.isdir(root):
            candidates.extend(
                glob.glob(os.path.join(root, "**", "*.pth"), recursive=True)
            )

    for p in candidates:
        if os.path.isfile(p) and (
            "effinet_best_2" in os.path.basename(p)
            or "efficientnet" in os.path.basename(p)
            or "effnet" in os.path.basename(p)
        ):
            return p

    for p in candidates:
        if os.path.isfile(p) and p.endswith(".pth"):
            return p

    raise FileNotFoundError(
        "Could not find any .pth checkpoint under /kaggle/input. "
        "Proceeding without a checkpoint is required in this environment."
    )


def resolve_petfinder_paths():
    """
    Minimal robustness fix: your environment shows both /kaggle/input/... and /kaggle/data/...
    We keep the same dataset and semantics, just pick an existing root so the run always yields a submission.csv.
    """
    candidates = [
        "/kaggle/input/petfinder-pawpularity-score",
        "/kaggle/data/input/petfinder-pawpularity-score",
    ]
    for base in candidates:
        if os.path.isfile(os.path.join(base, "test.csv")) and os.path.isdir(
            os.path.join(base, "test")
        ):
            return os.path.join(base, "test.csv"), os.path.join(base, "test")

    raise FileNotFoundError(
        "Could not locate petfinder-pawpularity-score test.csv/test folder under expected roots."
    )


class PetTestDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

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
        return image, row["Id"]


class EffNetWithMeta(nn.Module):
    def __init__(self, pretrained_backbone: bool = False):
        super().__init__()
        self.cnn = timm.create_model(
            "efficientnet_b3", pretrained=pretrained_backbone, num_classes=0
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(1536, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 1),
        )

    def forward(self, x, meta=None):
        x = self.cnn(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


transform = A.Compose(
    [
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)

torch.set_float32_matmul_precision("high")




## === cell 1
preferred_ckpt = "/kaggle/input/eff_l2/pytorch/default/1/effinet_best_2.pth"

try:
    ckpt_path = find_checkpoint_path(preferred_ckpt)
    has_ckpt = True
except FileNotFoundError:
    ckpt_path = None
    has_ckpt = False

model = EffNetWithMeta(pretrained_backbone=not has_ckpt)

if has_ckpt:
    state = torch.load(ckpt_path, map_location="cpu")

    if (
        isinstance(state, dict)
        and "state_dict" in state
        and isinstance(state["state_dict"], dict)
    ):
        state = state["state_dict"]

    if isinstance(state, dict):
        new_state = {}
        for k, v in state.items():
            nk = k[len("module.") :] if k.startswith("module.") else k
            new_state[nk] = v
        state = new_state

    model.load_state_dict(state, strict=True)
    print(f"Loaded checkpoint: {ckpt_path}")
else:
    print(
        "No .pth checkpoint found; using ImageNet-pretrained EfficientNet-B3 backbone."
    )

model.to(DEVICE)
model.eval()




## === cell 2
test_csv_path, test_img_dir = resolve_petfinder_paths()

test_df = pd.read_csv(test_csv_path)
test_ds = PetTestDataset(test_df, test_img_dir, transform)

test_dl = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
    drop_last=False,
)




## === cell 3
ids, preds = [], []

with torch.no_grad():
    for images, id_batch in tqdm(test_dl, desc="Infer", total=len(test_dl)):
        images = images.to(DEVICE, non_blocking=True)
        outputs = model(images).view(-1)

        outputs = torch.sigmoid(outputs) * 100.0

        outputs = outputs.detach().cpu().numpy()
        outputs = np.clip(outputs, 0.0, 100.0)

        preds.extend(outputs.astype(float).tolist())
        ids.extend(list(id_batch))

pred_map = dict(zip(ids, preds))
ordered_preds = [float(pred_map[i]) for i in test_df["Id"].tolist()]

submission = pd.DataFrame({"Id": test_df["Id"].values, "Pawpularity": ordered_preds})

out_path = "submission.csv"
submission.to_csv(out_path, index=False)
print(f"submission saved to: {out_path}  shape={submission.shape}")
print(submission.head())
