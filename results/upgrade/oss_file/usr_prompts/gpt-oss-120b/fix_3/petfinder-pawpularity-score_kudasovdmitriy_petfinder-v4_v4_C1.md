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

42.24644

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

- What this solution (achieved 42.24644) has done: 'I fix the inference loop so it correctly appends scalar predictions, and build the submission DataFrame by explicitly pairing the Ids with the predicted Pawpularity values. This removes the TypeError and ensures the CSV contains valid scores between 1 and 100.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)



## === cell 1
import sys

sys.path.append("../input/timm-folder/")



## === cell 2
TRAIN = pd.read_csv("/kaggle/input/petfinder-pawpularity-score/train.csv")
TEST = pd.read_csv("/kaggle/input/petfinder-pawpularity-score/test.csv")
SUB = pd.read_csv("/kaggle/input/petfinder-pawpularity-score/sample_submission.csv")



## === cell 3
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
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

from sklearn.model_selection import KFold

from sklearn.metrics import f1_score

import cv2
from PIL import Image



## === cell 4
img = Image.open(
    "../input/petfinder-pawpularity-score/train/4388dabf50790924baa7fec88b192b02.jpg"
)
img = img.resize((380, 380), Image.Resampling.LANCZOS)



## === cell 5
import matplotlib.pyplot as plt
from matplotlib import colors, cm, pyplot as plt

img = img
plt.imshow(img)
plt.show()



## === cell 6
image = Image.open(
    "../input/petfinder-pawpularity-score/train/4388dabf50790924baa7fec88b192b02.jpg"
)
image = image.resize((380, 380), Image.Resampling.LANCZOS)

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

plt.imshow(input_tensor["image"].numpy().transpose())
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
TRAIN = pd.concat((TRAIN, TRAIN.sample(8)), axis=0).reset_index()




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

        image = Image.open(self.img_path[idx])

        (width, height) = image.size

        mult = 380 / min(width, height)

        new_width = int(round(mult * width, 0))
        new_height = int(round(mult * height, 0))

        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

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

        label = self.img_labels[idx]
        label = torch.tensor(label, dtype=torch.float)

        add_data = self.add_data[idx]
        add_data = torch.tensor(add_data, dtype=torch.float)

        return image["image"], add_data, label




## === cell 11
torch.cuda.empty_cache()



## === cell 12
import gc

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

        image = Image.open(self.img_path[idx])

        (width, height) = image.size

        mult = 380 / min(width, height)

        new_width = int(round(mult * width, 0))
        new_height = int(round(mult * height, 0))

        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

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
        add_data = torch.tensor(add_data, dtype=torch.float)

        return image["image"], add_data




## === cell 18
batch_size = 1

test_loader = CustomImageDataset_test(
    TEST["path"].values, TEST.drop(["path", "Id"], axis=1).values
)
test_dataloader = DataLoader(
    test_loader, batch_size=batch_size, num_workers=4, pin_memory=True
)



## === cell 19
model.eval()

final_outputs = []

with torch.no_grad():
    for image, data in tqdm(test_dataloader, position=0, leave=True, desc="Evaluating"):
        image = image.to(device=device)
        data = data.to(device=device)
        output = model(image, data)
        final_outputs.append(output.squeeze().cpu().item())
        del image, data, output
        torch.cuda.empty_cache()

final_outputs = np.clip(final_outputs, 1, 100)
final_outputs = np.round(final_outputs, 2).tolist()



## === cell 20
res = final_outputs



## === cell 21
res



## === cell 22
submission = pd.DataFrame({"Id": SUB["Id"], "Pawpularity": res})
submission



## === cell 23
submission.to_csv("submission.csv", index=False)
