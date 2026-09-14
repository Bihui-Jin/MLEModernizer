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

3.9

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

19.360481049105307

# 6. Current score

22.2368

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 42.24644) has done: 'I fix the PIL resize crash by replacing the removed `Image.ANTIALIAS` constant with a backward/forward-compatible resampling mode, so all dataset loaders work. I also remove the hard dependency on the missing external weight file (`../input/pet-eff-b4/efficientnet_b4.pt`) by loading ImageNet-pretrained EfficientNet weights via timm instead, keeping the same architecture and forward pass. Finally, I clamp model predictions into the required `[1, 100]` range before writing the submission to satisfy the competition’s submission validator and ensure a `.csv` file is produced end-to-end.'
- What this solution (achieved 22.2368) has done: 'Your current script never trains the model (it only runs inference with ImageNet-pretrained weights), which explains the very weak RMSE. To move the score toward the target with minimal core-logic changes, I’m adding a small, single-pass training phase on the provided `train.csv` (same model, same loss semantics as regression) and then running the same test inference and clipping to `[1, 100]`. I’m also fixing the submission `Id` alignment by using `TEST["Id"]` (instead of `sample_submission` order) and ensuring the metadata features are consistently converted to `float32` for both train and test. The rest of the architecture/feature extraction remains unchanged, and the output is still a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd



## === cell 1
import sys

sys.path.append("../input/timm-folder/")



## === cell 2
TRAIN = pd.read_csv("/kaggle/input/petfinder-pawpularity-score/train.csv")
TEST = pd.read_csv("/kaggle/input/petfinder-pawpularity-score/test.csv")
SUB = pd.read_csv("/kaggle/input/petfinder-pawpularity-score/sample_submission.csv")



## === cell 3
import os
import gc
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from tqdm.notebook import tqdm

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import albumentations
import albumentations.pytorch

from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import f1_score

import cv2
from PIL import Image

try:
    RESAMPLE_MODE = Image.Resampling.LANCZOS
except Exception:
    RESAMPLE_MODE = Image.LANCZOS



## === cell 4
img = Image.open(
    "../input/petfinder-pawpularity-score/train/4388dabf50790924baa7fec88b192b02.jpg"
)
img = img.resize((380, 380), RESAMPLE_MODE)



## === cell 5
import matplotlib.pyplot as plt
from matplotlib import colors, cm, pyplot as plt

plt.imshow(img)
plt.axis("off")
plt.show()



## === cell 6
image = Image.open(
    "../input/petfinder-pawpularity-score/train/4388dabf50790924baa7fec88b192b02.jpg"
)
image = image.resize((380, 380), RESAMPLE_MODE)

image = np.asarray(image)
preprocess = albumentations.Compose(
    [
        albumentations.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        albumentations.pytorch.transforms.ToTensorV2(),
    ]
)
input_tensor = preprocess(image=image)

x = input_tensor["image"].numpy()
x = np.transpose(x, (1, 2, 0))
plt.imshow(
    (x * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])).clip(0, 1)
)
plt.axis("off")
plt.show()



## === cell 7
TRAIN = pd.read_csv("../input/petfinder-pawpularity-score/train.csv")
TRAIN["path"] = TRAIN["Id"].apply(
    lambda x: str("../input/petfinder-pawpularity-score" + "/train/" + str(x) + ".jpg")
)

TEST = pd.read_csv("../input/petfinder-pawpularity-score/test.csv")
TEST["path"] = TEST["Id"].apply(
    lambda x: str("../input/petfinder-pawpularity-score" + "/test/" + str(x) + ".jpg")
)



## === cell 8
from sklearn.model_selection import StratifiedKFold



## === cell 9
TRAIN = pd.concat((TRAIN, TRAIN.sample(8, random_state=42)), axis=0).reset_index(
    drop=True
)




## === cell 10
class CustomImageDataset(Dataset):
    def __init__(
        self,
        annotations_file,
        img_path,
        add_data,
        transform=None,
        target_transform=None,
    ):
        self.img_labels = annotations_file
        self.img_path = img_path
        self.add_data = add_data
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image = Image.open(self.img_path[idx]).convert("RGB")

        (width, height) = image.size
        mult = 380 / min(width, height)

        new_width = int(round(mult * width, 0))
        new_height = int(round(mult * height, 0))

        image = image.resize((new_width, new_height), RESAMPLE_MODE)

        image_num = np.asarray(image)

        preprocess = albumentations.Compose(
            [
                albumentations.CenterCrop(380, 380),
                albumentations.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                albumentations.pytorch.transforms.ToTensorV2(),
            ]
        )

        image = preprocess(image=image_num)

        label = float(self.img_labels[idx])
        label = torch.tensor(label, dtype=torch.float32)

        add_data = self.add_data[idx]
        add_data = torch.tensor(add_data, dtype=torch.float32)

        return image["image"], add_data, label




## === cell 11
torch.cuda.empty_cache()



## === cell 12
gc.collect()



## === cell 13
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))



## === cell 14
import timm




## === cell 15
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.cnn = timm.create_model(
            model_name="tf_efficientnet_b4_ns", pretrained=True
        )
        self.cnn.drop_rate = 0.4
        self.cnn.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=1792, out_features=30),
        )

        self.fc1 = nn.Linear(30 + 12, 60)
        self.fc2 = nn.Linear(60, 1)

    def forward(self, image, data):
        x1 = self.cnn(image)
        x2 = data
        x = torch.cat((x1, x2), dim=1)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = MyModel()
model.to(device=device)



## === cell 16
torch.cuda.empty_cache()




## === cell 17
class CustomImageDataset_test(Dataset):
    def __init__(self, img_path, add_data, transform=None, target_transform=None):
        self.img_path = img_path
        self.add_data = add_data
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        image = Image.open(self.img_path[idx]).convert("RGB")

        (width, height) = image.size
        mult = 380 / min(width, height)

        new_width = int(round(mult * width, 0))
        new_height = int(round(mult * height, 0))

        image = image.resize((new_width, new_height), RESAMPLE_MODE)

        image_num = np.asarray(image)

        preprocess = albumentations.Compose(
            [
                albumentations.CenterCrop(380, 380),
                albumentations.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                albumentations.pytorch.transforms.ToTensorV2(),
            ]
        )

        image = preprocess(image=image_num)

        add_data = self.add_data[idx]
        add_data = torch.tensor(add_data, dtype=torch.float32)

        return image["image"], add_data




## === cell 18
torch.manual_seed(42)
np.random.seed(42)

train_df, valid_df = train_test_split(TRAIN, test_size=0.1, random_state=42)

feature_cols = [c for c in TRAIN.columns if c not in ["Id", "path", "Pawpularity"]]

train_dataset = CustomImageDataset(
    annotations_file=train_df["Pawpularity"].values,
    img_path=train_df["path"].values,
    add_data=train_df[feature_cols].values.astype(np.float32),
)
valid_dataset = CustomImageDataset(
    annotations_file=valid_df["Pawpularity"].values,
    img_path=valid_df["path"].values,
    add_data=valid_df[feature_cols].values.astype(np.float32),
)

train_dataloader = DataLoader(
    train_dataset, batch_size=8, shuffle=True, num_workers=2, pin_memory=True
)
valid_dataloader = DataLoader(
    valid_dataset, batch_size=8, shuffle=False, num_workers=2, pin_memory=True
)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=2e-4)

model.train()
epochs = 1  # minimal to move score toward target while staying within time constraints

for epoch in range(epochs):
    running_loss = 0.0
    for image, data, label in tqdm(
        train_dataloader, desc=f"Training epoch {epoch+1}/{epochs}"
    ):
        image = image.to(device=device, non_blocking=True)
        data = data.to(device=device, non_blocking=True)
        label = label.to(device=device, non_blocking=True).view(-1, 1)

        optimizer.zero_grad(set_to_none=True)
        output = model(image, data)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * image.size(0)

        del image, data, label, output, loss

    train_mse = running_loss / len(train_dataset)

    model.eval()
    v_losses = 0.0
    with torch.no_grad():
        for image, data, label in valid_dataloader:
            image = image.to(device=device, non_blocking=True)
            data = data.to(device=device, non_blocking=True)
            label = label.to(device=device, non_blocking=True).view(-1, 1)
            output = model(image, data)
            v_losses += criterion(output, label).item() * image.size(0)
            del image, data, label, output
    val_mse = v_losses / len(valid_dataset)
    print(
        f"epoch {epoch+1}: train_rmse={np.sqrt(train_mse):.4f}, val_rmse={np.sqrt(val_mse):.4f}"
    )
    model.train()

torch.cuda.empty_cache()
gc.collect()



## === cell 19
batch_size = 8

test_loader = CustomImageDataset_test(
    TEST["path"].values, TEST.drop(["path", "Id"], axis=1).values.astype(np.float32)
)
test_dataloader = DataLoader(
    test_loader, batch_size=batch_size, num_workers=2, pin_memory=True
)



## === cell 20
model.eval()

final_outputs = []

with torch.no_grad():
    for image, data in tqdm(test_dataloader, position=0, leave=True, desc="Evaluating"):
        image = image.to(device=device, non_blocking=True)
        data = data.to(device=device, non_blocking=True)

        output = model(image, data)
        final_outputs.extend(output.detach().cpu().numpy().reshape(-1).tolist())

        del image, data, output

torch.cuda.empty_cache()



## === cell 21
res = np.array(final_outputs, dtype=np.float32)

res = np.clip(res, 1.0, 100.0)



## === cell 22
submission = pd.DataFrame({"Id": TEST["Id"].values, "Pawpularity": res})
submission.head()



## === cell 23
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.describe())
