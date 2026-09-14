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

20.73251

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 20.77238) has done: 'I fixed the import errors (removed the CUDA‑only cuML import and the failing CLIP import), switched to the CPU‑based `sklearn.svm.SVR`, and simplified the pipeline to use only the provided metadata features (which are sufficient to train a model and create a valid submission). The script now runs end‑to‑end, computes a reasonable RMSE, and writes `submission.csv` with the correct columns.'
- What this solution (achieved 20.64754) has done: 'I add a polynomial feature expansion for the metadata (degree 2) to give the SVR more expressive inputs and slightly adjust the SVR hyper‑parameters (lower C and a small epsilon). These changes keep the overall pipeline and model type intact while aiming to reduce the RMSE toward the target value.'
- What this solution (achieved 20.68508) has done: 'We slightly strengthen the SVR by raising the regularization C and lowering epsilon, which gives the model more flexibility and typically reduces RMSE while keeping the same overall pipeline. Only the SVR initialization in cell 19 is changed.'
- What this solution (achieved 20.73424) has done: 'I slightly adjust the SVR hyper‑parameters to give the model more flexibility (increase C and reduce epsilon). This keeps the overall pipeline unchanged while aiming to lower the RMSE toward the target value.'
- What this solution (achieved 20.75044) has done: 'I tighten the preprocessing and SVR settings while keeping the overall pipeline unchanged. First, the scaler be fit only on the training features (removing the subtle leakage from including test data). Then I use a more flexible SVR by increasing the regularization C and lowering epsilon, and set gamma to “auto” so it adapts to the feature dimensionality. These small adjustments should lower the RMSE, moving the score closer to the target without altering the core model logic.'
- What this solution (achieved 20.73251) has done: 'I increase the polynomial feature degree to 3 to capture higher‑order interactions among the metadata, and adjust the SVR hyper‑parameters to a slightly less aggressive regularisation (C=100) and a larger epsilon (0.01) while using the default “scale” gamma. These minimal changes keep the same model type and preprocessing pipeline but give the regressor a better bias‑variance balance, which should lower the validation RMSE and move the score closer to the target.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch

from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error



## === cell 1
device = "cuda" if torch.cuda.is_available() else "cpu"



## === cell 2
directory = "/kaggle/input/petfinder-pawpularity-score"
train_df = pd.read_csv(os.path.join(directory, "train.csv"))
test_df = pd.read_csv(os.path.join(directory, "test.csv"))

print("Train samples: ", len(train_df), "\nTest samples: ", len(test_df), "\n")




## === cell 3
def ExtractModelFeature(dataloader, model, Train_PCA=False):
    X = []
    for img in tqdm(dataloader):
        with torch.no_grad():
            if model.__class__.__name__ == "EfficientNet":
                x = model(img.to(device))
            elif model.__class__.__name__ == "CLIPModel":
                x = model.get_image_features(**img.to(device))
            elif model.__class__.__name__ == "PCA":
                x = img
            else:
                raise Exception("Check if model is implimented !")
        X.append(x.cpu().detach().numpy())
    X = np.concatenate(X, axis=0)
    if model.__class__.__name__ == "PCA":
        if Train_PCA:
            model.fit(X)
            X = model.transform(X)
            return X, model
        else:
            return model.transform(X)
    return X




## === cell 4
"""
class dataset_EfficientNet:
    def __init__(self, df, directory, transform, test=False):
        self.df = df
        self.directory = directory
        self.transform = transform
        self.test = test
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        split = 'test' if self.test else 'train'
        filename = self.df.Id[idx]
        address = os.path.join(self.directory, split, filename+'.jpg')
        img = Image.open(address).convert('RGB')
        image = self.transform(img) # transform and add batch dimension
        return image
"""



## === cell 5
"""
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

ckp_path = "/kaggle/input/tf-efficientnet/pytorch/tf-efficientnet-b6/1/tf_efficientnet_b6_aa-80ba17e4.pth"
model_eff = timm.create_model('tf_efficientnet_b6', checkpoint_path=ckp_path)

config = resolve_data_config({}, model=model_eff)
transform = create_transform(**config)

model_eff = model_eff.to(device)
"""



## === cell 6
"""
train_dataset = dataset_EfficientNet(train_df, directory, transform, test=False)
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=False)

test_dataset = dataset_EfficientNet(test_df, directory, transform, test=True)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)
"""



## === cell 7
"""
X1 = ExtractModelFeature(train_dataloader, model_eff)
X1_test = ExtractModelFeature(test_dataloader, model_eff)

print('train: ', X1.shape)
print('test: ', X1_test.shape)
"""



## === cell 8
"""
class dataset_Clip:
    def __init__(self, df, directory, processor, test=False):
        self.df = df
        self.directory = directory
        self.processor = processor
        self.test = test
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        split = 'test' if self.test else 'train'
        filename = self.df.Id[idx]
        address = os.path.join(self.directory, split, filename+'.jpg')
        image = self.processor(images=Image.open(address), return_tensors="pt", padding=True)
        for key, val in image.items():
            image[key] = val.squeeze()
        return image
"""



## === cell 9
"""
train_dataset = dataset_Clip(train_df, directory, processor_clip, test=False)
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=False)

test_dataset = dataset_Clip(test_df, directory, processor_clip, test=True)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

X2 = ExtractModelFeature(train_dataloader, model_clip)
X2_test = ExtractModelFeature(test_dataloader, model_clip)

print('train: ', X2.shape)
print('test: ', X2_test.shape)
"""



## === cell 10
"""
train_dataset = dataset_Clip(train_df, directory, processor_clip_2, test=False)
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=False)

test_dataset = dataset_Clip(test_df, directory, processor_clip_2, test=True)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

X3 = ExtractModelFeature(train_dataloader, model_clip_2)
X3_test = ExtractModelFeature(test_dataloader, model_clip_2)

print('train: ', X3.shape)
print('test: ', X3_test.shape)
"""



## === cell 11
x_meta = train_df.iloc[:, 1:13].values
x_test_meta = test_df.iloc[:, 1:13].values



## === cell 12
"""
class ImageExtract:
    def __init__(self, df, directory, test=False):
        self.df = df
        self.directory = directory
        self.test = test
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        split = 'test' if self.test else 'train'
        filename = self.df.Id[idx]
        address = os.path.join(self.directory, split, filename+'.jpg')
        image = np.array(Image.open(address).convert('L').resize((128,128))).flatten()
        return image
"""



## === cell 13
"""
train_dataset = ImageExtract(train_df, directory, test=False)
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=False)

test_dataset = ImageExtract(test_df, directory, test=True)
test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)

pca = PCA(n_components=512)
X4, pca = ExtractModelFeature(train_dataloader, pca, Train_PCA=True)
X4_test = ExtractModelFeature(test_dataloader, pca)

print('train: ', X4.shape)
print('test: ', X4_test.shape)
"""



## === cell 14
"""
train_dataset = ImageExtract(train_df, directory, test=False)
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=False)

test_dataset = ImageExtract(test_df, directory, test=True)
test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)

pca = PCA(n_components=512)
X4 = ExtractModelFeature(train_dataloader, 'ImageExtract')
X4_test = ExtractModelFeature(test_dataloader, 'ImageExtract')

print('train: ', X4.shape)
print('test: ', X4_test.shape)
"""



## === cell 15
poly = PolynomialFeatures(degree=3, include_bias=False)
X = poly.fit_transform(x_meta)
X_test = poly.transform(x_test_meta)

print("train_features shape after poly:", X.shape)
print("test_features shape after poly:", X_test.shape)



## === cell 16
y = train_df["Pawpularity"].values
print("Target shape:", y.shape)



## === cell 17
scaler = StandardScaler()
scaler.fit(X)  # fit on training features only
X_scaled = scaler.transform(X)
X_test_scaled = scaler.transform(X_test)
print("Scaled train shape:", X_scaled.shape)



## === cell 18
reg = SVR(C=100.0, epsilon=0.01, kernel="rbf", gamma="scale", max_iter=500000)
reg.fit(X_scaled, y)

train_pred = reg.predict(X_scaled)
rmse = np.sqrt(mean_squared_error(y, train_pred))
print("Training RMSE:", rmse)



## === cell 19
y_pred = reg.predict(X_test_scaled)



## === cell 20
submission = pd.DataFrame({"Id": test_df["Id"], "Pawpularity": y_pred})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
print(submission.head())



## === cell 21
submission
