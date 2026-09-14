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

3.12

# 3. Installed packages

cuml-cu12==25.2.1
geopandas==0.14.4
libcuml-cu12==25.2.1
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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sentence-transformers==4.1.0
sklearn-pandas==2.2.0
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
transformers==4.53.3

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

18.18046707404222

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
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from sklearn.svm import SVR  # CPU fallback for cuML SVR
from sklearn.preprocessing import PowerTransformer

os.environ["TRANSFORMERS_NO_FAST_IMAGE_PROCESSOR"] = "1"



## === cell 1
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)



## === cell 2
directory = "/kaggle/input/petfinder-pawpularity-score"
train_df = pd.read_csv(os.path.join(directory, "train.csv"))
test_df = pd.read_csv(os.path.join(directory, "test.csv"))

print("Train samples:", len(train_df))
print("Test samples:", len(test_df))




## === cell 3
def ExtractModelFeature(dataloader, model, Train_PCA=False):
    model_name = model.__class__.__name__
    if hasattr(model, "eval"):
        model.eval()

    X = []
    for batch in tqdm(dataloader, desc=f"Extract features ({model_name})"):
        with torch.no_grad():
            if model_name == "CLIPModel":
                batch = {k: v.to(device) for k, v in batch.items()}
                x = model.get_image_features(**batch)
            elif model_name == "PCA":
                x = batch
            else:
                raise Exception("Check if model is implemented!")

        X.append(x.detach().cpu().numpy())

    X = np.concatenate(X, axis=0)

    if model_name == "PCA":
        if Train_PCA:
            model.fit(X)
            X = model.transform(X)
            return X, model
        return model.transform(X)

    return X




## === cell 4
from transformers import CLIPModel, CLIPProcessor

model_path_1 = "/kaggle/input/clip-vit/pytorch/b-32-laion2b-s34b-b79k/1"

model_clip_1 = CLIPModel.from_pretrained(model_path_1)
processor_clip_1 = CLIPProcessor.from_pretrained(model_path_1, use_fast=False)

model_clip_1 = model_clip_1.to(device)
model_clip_1.eval()

print("Loaded CLIP:", type(model_clip_1).__name__)




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 5
class dataset_Clip(Dataset):
    def __init__(self, df, directory, processor, test=False):
        self.df = df.reset_index(drop=True)
        self.directory = directory
        self.processor = processor
        self.test = test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        split = "test" if self.test else "train"
        filename = self.df.loc[idx, "Id"]
        address = os.path.join(self.directory, split, filename + ".jpg")

        img = Image.open(address).convert("RGB")
        enc = self.processor(images=img, return_tensors="pt")
        enc = {k: v.squeeze(0) for k, v in enc.items()}
        return enc


def clip_collate_fn(batch):
    keys = batch[0].keys()
    out = {}
    for k in keys:
        out[k] = torch.stack([b[k] for b in batch], dim=0)
    return out




## === cell 6
train_dataset = dataset_Clip(train_df, directory, processor_clip_1, test=False)
test_dataset = dataset_Clip(test_df, directory, processor_clip_1, test=True)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=2,
    pin_memory=(device == "cuda"),
    collate_fn=clip_collate_fn,
)
test_dataloader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=2,
    pin_memory=(device == "cuda"),
    collate_fn=clip_collate_fn,
)

X2 = ExtractModelFeature(train_dataloader, model_clip_1)
X2_test = ExtractModelFeature(test_dataloader, model_clip_1)

print("train:", X2.shape)
print("test:", X2_test.shape)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2335896654.py in <cell line: 0>()
      1 # Feature extraction
----> 2 train_dataset = dataset_Clip(train_df, directory, processor_clip_1, test=False)
      3 test_dataset = dataset_Clip(test_df, directory, processor_clip_1, test=True)
      4 
      5 train_dataloader = DataLoader(

NameError: name 'processor_clip_1' is not defined

## === cell 7
x_meta = train_df.iloc[:, 1:13].values.astype(np.float32)
x_test_meta = test_df.iloc[:, 1:13].values.astype(np.float32)



## === cell 8
X = np.hstack((X2, x_meta))
X_test = np.hstack((X2_test, x_test_meta))

print("train_stacked:", X.shape)
print("test_stacked:", X_test.shape)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/495480636.py in <cell line: 0>()
      1 # Stack features (core logic preserved: CLIP image embedding + meta)
----> 2 X = np.hstack((X2, x_meta))
      3 X_test = np.hstack((X2_test, x_test_meta))
      4 
      5 print("train_stacked:", X.shape)

NameError: name 'X2' is not defined

## === cell 9
y = train_df["Pawpularity"].values.astype(np.float32)
print("y shape:", y.shape, "y min/max:", y.min(), y.max())



## === cell 10
pt = PowerTransformer()
pt.fit(np.vstack((X, X_test)))
X = pt.transform(X)
X_test = pt.transform(X_test)
print("Transformed X:", X.shape, "X_test:", X_test.shape)



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4215858033.py in <cell line: 0>()
      1 # Apply PowerTransformer exactly as originally intended (fit on combined train+test)
      2 pt = PowerTransformer()
----> 3 pt.fit(np.vstack((X, X_test)))
      4 X = pt.transform(X)
      5 X_test = pt.transform(X_test)

NameError: name 'X' is not defined

## === cell 11
reg = SVR(C=100.0, kernel="rbf", degree=3, max_iter=400000)
reg.fit(X, y)

print("Fitted SVR")



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3018111298.py in <cell line: 0>()
      1 # Bugfix: use sklearn SVR since cuML cannot import in this environment; keep same hyperparameters.
      2 reg = SVR(C=100.0, kernel="rbf", degree=3, max_iter=400000)
----> 3 reg.fit(X, y)
      4 
      5 print("Fitted SVR")

NameError: name 'X' is not defined

## === cell 12
print("Predicted values (train head):", reg.predict(X[:10, :]))
print("True values (train head):", y[:10])




## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3268139361.py in <cell line: 0>()
----> 1 print("Predicted values (train head):", reg.predict(X[:10, :]))
      2 print("True values (train head):", y[:10])
      3 
      4 

NameError: name 'X' is not defined

## === cell 13
def RMSE(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


train_rmse = RMSE(y, reg.predict(X))
print("Train RMSE:", train_rmse)



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3258074540.py in <cell line: 0>()
      5 
      6 
----> 7 train_rmse = RMSE(y, reg.predict(X))
      8 print("Train RMSE:", train_rmse)
      9 

NameError: name 'X' is not defined

## === cell 14
y_pred = reg.predict(X_test)

y_pred = np.clip(y_pred, 0.0, 100.0)

print("Predicted values (test head):", y_pred[:10])
print("Pred shape:", y_pred.shape)



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2251479767.py in <cell line: 0>()
----> 1 y_pred = reg.predict(X_test)
      2 
      3 # Keep predictions in valid target range to avoid pathological outputs
      4 y_pred = np.clip(y_pred, 0.0, 100.0)
      5 

NameError: name 'X_test' is not defined

## === cell 15
submit = pd.DataFrame({"Id": test_df["Id"].values, "Pawpularity": y_pred})
submit.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submit.shape)



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2136462753.py in <cell line: 0>()
----> 1 submit = pd.DataFrame({"Id": test_df["Id"].values, "Pawpularity": y_pred})
      2 submit.to_csv("submission.csv", index=False)
      3 print("Wrote submission.csv with shape:", submit.shape)
      4 

NameError: name 'y_pred' is not defined

## === cell 16
submit.head()

## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2717648686.py in <cell line: 0>()
----> 1 submit.head()

NameError: name 'submit' is not defined
