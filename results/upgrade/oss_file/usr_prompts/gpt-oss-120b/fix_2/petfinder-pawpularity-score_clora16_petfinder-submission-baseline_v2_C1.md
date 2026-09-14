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

3.14

# 3. Installed packages

geopandas==0.14.4
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

18.745360322180247

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import timm
import torchvision.transforms as T

from sklearn.linear_model import LinearRegression


class CFG:
    _candidates = [
        "/kaggle/input/competitions/petfinder-pawpularity-score",
        "/kaggle/input/petfinder-pawpularity-score",
        "/kaggle/input/data/petfinder-pawpularity-score",
        "/kaggle/working/petfinder-pawpularity-score",
        "./petfinder-pawpularity-score",
    ]

    @classmethod
    def _first_existing(cls, paths):
        for p in paths:
            if os.path.isdir(p):
                return p
        raise FileNotFoundError("None of the candidate data directories exist.")

    BASE_DIR = _first_existing(_candidates)

    weight_path = "/kaggle/input/datasets/clora16/petfinder-base-models/best_fold0.bin"
    test_csv_path = os.path.join(BASE_DIR, "test.csv")
    test_image_dir = os.path.join(BASE_DIR, "test")
    train_csv_path = os.path.join(BASE_DIR, "train.csv")

    model_name = "tf_efficientnet_b0.ns_jft_in1k"
    image_size = 224
    output_dir = "output"

    batch_size = 16
    num_workers = 4
    output_path = "submission.csv"


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4251578040.py in <cell line: 0>()
     17 # Configuration – try a few possible base directories that exist in the env.
     18 # ----------------------------------------------------------------------
---> 19 class CFG:
     20     # possible locations for the competition data
     21     _candidates = [

/tmp/ipykernel_55/4251578040.py in CFG()
     34         raise FileNotFoundError("None of the candidate data directories exist.")
     35 
---> 36     BASE_DIR = _first_existing(_candidates)
     37 
     38     # weight file may be absent – keep the placeholder but allow fallback

TypeError: 'classmethod' object is not callable

## === cell 1
class PawpularityRegressor(nn.Module):
    """EfficientNet backbone based regression model (inference only)."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            CFG.model_name, pretrained=False, num_classes=0, global_pool="avg"
        )
        n_features = self.backbone.num_features
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(n_features, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features).squeeze(1)


class TestDataset(Dataset):
    """Dataset for inference – returns transformed image and its Id."""

    def __init__(self, df: pd.DataFrame, image_dir: str, transform: T.Compose):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_path = os.path.join(self.image_dir, f"{row['Id']}.jpg")
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return image, row["Id"]


def find_latest_weight(model_name: str, output_dir: str) -> str:
    """Return the newest best_fold*.bin weight file if present."""
    model_dir = os.path.join(output_dir, model_name)
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"model output directory not found: {model_dir}")

    exp_dirs = sorted(
        [
            os.path.join(model_dir, d)
            for d in os.listdir(model_dir)
            if os.path.isdir(os.path.join(model_dir, d)) and d.startswith("exp_")
        ]
    )
    if not exp_dirs:
        raise FileNotFoundError(f"no exp_* directories under: {model_dir}")

    latest_exp_dir = exp_dirs[-1]
    weight_files = sorted(
        [
            f
            for f in os.listdir(latest_exp_dir)
            if f.startswith("best_fold") and f.endswith(".bin")
        ]
    )
    if not weight_files:
        raise FileNotFoundError(f"no best_fold*.bin in: {latest_exp_dir}")
    return os.path.join(latest_exp_dir, weight_files[0])




## === cell 2
def train_tabular_fallback(train_path: str):
    """
    Train a very lightweight LinearRegression model on the tabular metadata.
    Returns the fitted model and the mean of the training target (used as a fallback
    prediction when a feature is missing).
    """
    train_df = pd.read_csv(train_path)
    feature_cols = [col for col in train_df.columns if col not in ["Id", "Pawpularity"]]
    X = train_df[feature_cols].astype(float)
    y = train_df["Pawpularity"].astype(float)
    model = LinearRegression()
    model.fit(X, y)
    return model, X.mean().mean()  # return overall feature mean for safety




## === cell 3
def main() -> None:
    """Run inference and write submission.csv."""
    weight_path = None
    try:
        weight_path = CFG.weight_path if os.path.isfile(CFG.weight_path) else None
        if weight_path is None:
            weight_path = find_latest_weight(CFG.model_name, CFG.output_dir)
        print(f"using weight: {weight_path}")
    except Exception as e:
        print(f"Weight file not found or error ({e}); will use tabular fallback model.")

    test_df = pd.read_csv(CFG.test_csv_path)

    if weight_path is not None:
        test_transform = T.Compose(
            [
                T.Resize((CFG.image_size, CFG.image_size)),
                T.ToTensor(),
                T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
        test_loader = DataLoader(
            TestDataset(test_df, CFG.test_image_dir, test_transform),
            batch_size=CFG.batch_size,
            shuffle=False,
            num_workers=CFG.num_workers,
            pin_memory=True,
        )

        model = PawpularityRegressor()
        state_dict = torch.load(weight_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        sample_ids, predictions = [], []
        with torch.no_grad():
            for images, batch_ids in test_loader:
                images = images.to(device, non_blocking=True)
                batch_preds = model(images).cpu().numpy()
                sample_ids.extend(batch_ids)
                predictions.extend(batch_preds.tolist())

        preds_series = pd.Series(predictions, index=sample_ids)
        submission = (
            test_df[["Id"]]
            .set_index("Id")
            .join(preds_series.rename("Pawpularity"), how="left")
        )
        submission["Pawpularity"] = np.clip(
            submission["Pawpularity"], 0.0, 100.0
        ).values
    else:
        lin_model, _ = train_tabular_fallback(CFG.train_csv_path)
        feature_cols = [col for col in test_df.columns if col != "Id"]
        X_test = test_df[feature_cols].astype(float)
        preds = lin_model.predict(X_test)
        submission = pd.DataFrame(
            {"Id": test_df["Id"], "Pawpularity": np.clip(preds, 0.0, 100.0)}
        )

    submission = submission[["Id", "Pawpularity"]]
    submission.to_csv(CFG.output_path, index=False)
    print(f"saved: {CFG.output_path}")


if __name__ == "__main__":
    main()

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/839443745.py in <cell line: 0>()
     79 
     80 if __name__ == "__main__":
---> 81     main()

/tmp/ipykernel_55/839443745.py in main()
     16     # Load test metadata.
     17     # ------------------------------------------------------------------
---> 18     test_df = pd.read_csv(CFG.test_csv_path)
     19 
     20     # ------------------------------------------------------------------

NameError: name 'CFG' is not defined
