# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

3.10

# 3. Installed packages

albumentations==2.0.8
cuml-cu12==25.2.1
fastai==2.8.5
geopandas==0.14.4
libcuml-cu12==25.2.1
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
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

# 5. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

import timm
from sklearn.ensemble import GradientBoostingRegressor

torch.backends.cudnn.benchmark = True
torch.set_num_threads(1)

base_dir = "/kaggle/input"
test_folder = os.path.join(base_dir, "petfinder-pawpularity-score", "test")
train_folder = os.path.join(base_dir, "petfinder-pawpularity-score", "train")
test_file = os.path.join(base_dir, "petfinder-pawpularity-score", "test.csv")
train_file = os.path.join(base_dir, "petfinder-pawpularity-score", "train.csv")



## === cell 1
test_csv = pd.read_csv(test_file)
train_csv = pd.read_csv(train_file)

test_csv["path_img"] = test_csv["Id"].apply(
    lambda x: os.path.join(test_folder, f"{x}.jpg")
)
train_csv["path_img"] = train_csv["Id"].apply(
    lambda x: os.path.join(train_folder, f"{x}.jpg")
)

test_csv["Pawpularity"] = 1.0




## === cell 2
class PetsDataset(Dataset):
    def __init__(self, df, transform=None):
        self.transform = transform
        self.df = df
        self.cat = [
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

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df["path_img"].iloc[idx]
        label_1 = self.df["Pawpularity"].iloc[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        df_data = self.df[self.cat].iloc[idx].values.astype(np.float32)
        return (img, df_data, label_1)




## === cell 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mean = (0.485, 0.456, 0.406)  # ImageNet mean
std = (0.229, 0.224, 0.225)  # ImageNet std

embed_transform = transforms.Compose(
    [
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
)

embed_model = timm.create_model("resnet18", pretrained=True, num_classes=0).to(device)
embed_model.eval()




## === cell 4
def extract_embeddings(df):
    """Return a (N, D) NumPy array of image embeddings for rows in df."""

    class ImgOnlyDataset(Dataset):
        def __init__(self, paths):
            self.paths = paths

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, idx):
            img = Image.open(self.paths[idx]).convert("RGB")
            img = embed_transform(img)
            return img

    img_dataset = ImgOnlyDataset(df["path_img"].values)

    loader = DataLoader(
        img_dataset,
        batch_size=128,  # larger batch → fewer kernel launches
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,  # keep workers alive for the second call
    )

    N = len(df)
    with torch.no_grad():
        sample_batch = next(iter(loader))
        sample_batch = sample_batch.to(device, non_blocking=True)
        sample_feat = embed_model(sample_batch)
        D = sample_feat.shape[1]

    embeddings = np.empty((N, D), dtype=np.float32)
    offset = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device, non_blocking=True)
            feats = embed_model(batch)  # (B, D)
            bsz = feats.shape[0]
            embeddings[offset : offset + bsz] = feats.cpu().numpy()
            offset += bsz

    return embeddings


print("Extracting train embeddings ...")
train_emb = extract_embeddings(train_csv)
print("Extracting test embeddings ...")
test_emb = extract_embeddings(test_csv)

emb_dim = train_emb.shape[1]
emb_cols = [f"emb_{i}" for i in range(emb_dim)]

for idx, col in enumerate(emb_cols):
    train_csv[col] = train_emb[:, idx]
    test_csv[col] = test_emb[:, idx]



## === cell 5
cat_features = [
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
] + emb_cols

X_train = train_csv[cat_features].astype(np.float32).fillna(0).to_numpy()
y_train = train_csv["Pawpularity"].astype(np.float32).fillna(0).to_numpy()

gb_regressor = GradientBoostingRegressor(
    n_estimators=1000,
    learning_rate=0.03,
    max_depth=4,
    random_state=42,
)

gb_regressor.fit(X_train, y_train)

X_test = test_csv[cat_features].astype(np.float32).fillna(0).to_numpy()
pred_array = gb_regressor.predict(X_test)



## === cell 6
final_predictions = np.clip(pred_array, 1, 100)
final_predictions = np.nan_to_num(final_predictions, nan=50.0)
test_csv["Pawpularity"] = final_predictions



## === cell 7
submission = test_csv[["Id", "Pawpularity"]]
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")



## === cell 8
print(submission.head())
