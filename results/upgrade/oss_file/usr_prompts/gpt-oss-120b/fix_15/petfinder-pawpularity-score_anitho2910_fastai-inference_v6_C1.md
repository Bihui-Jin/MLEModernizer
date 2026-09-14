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

3.10

# 3. Installed packages

albumentations==2.0.8
fastai==2.8.5
geopandas==0.14.4
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

# 5. Target score

17.90779

# 6. Current score

20.05653

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 20.08411) has done: 'Implemented missing imports, corrected variable definitions, and ensured a valid submission file is generated. Added a simple baseline that predicts the mean Pawpularity from the training set for all test entries, guaranteeing the script runs end‑to‑end and produces a proper CSV. This also provides a reasonable score toward the target without altering core model logic.'
- What this solution (achieved 20.083) has done: 'We move the pretrained Swin model to the same device as the dummy tensor to fix the type mismatch error, and then replace the naive mean‑only prediction with a lightweight linear‑adjustment that adds per‑feature offsets derived from the training data. This keeps the core architecture unchanged, resolves the runtime crashes, and should modestly lower the RMSE toward the target.'
- What this solution (achieved 30.42039) has done: 'I add a proper image transform, run the pretrained Swin‑based network on the test set, and blend its predictions with the existing simple offset model (using a modest weight for the network). This introduces only the inference step needed to leverage the model, keeps the original architecture untouched, and clips the final scores to the required 0‑100 range, which should lower the RMSE toward the target.'
- What this solution (achieved 21.52053) has done: 'I reduce the influence of the Swin model predictions, which currently hurt performance, and give more weight to the simpler offset‑based baseline that already achieved a much lower RMSE. By changing the blending ratio to favour the offset predictions (e.g., 0.2 × model + 0.8 × offset) we move the score closer to the target while keeping all core logic unchanged.'
- What this solution (achieved 20.46188) has done: 'I add a simple linear calibration derived from the training set to the offset‑only predictions and lower the weight of the Swin model in the final blend (reducing it from 0.2 to 0.1). This keeps the core architecture untouched while nudging the RMSE closer to the target.'
- What this solution (achieved 20.10152) has done: 'The update replaces the handcrafted offset calibration with a simple linear regression on the binary metadata features, which gives a more accurate tabular baseline. The model’s contribution is removed (weight = 0) so only the regression predictions are used, lowering the RMSE toward the target while keeping the original network untouched. Additionally, the required sklearn import is added.'
- What this solution (achieved 20.05783) has done: 'We replace the simple LinearRegression with a GradientBoostingRegressor, which usually captures nonlinear relationships in the metadata and should lower the RMSE toward the target while keeping the overall pipeline unchanged. The model‑only predictions are still ignored (weight = 0) to avoid degrading performance.'
- What this solution (achieved 20.08319) has done: 'I add a lightweight RandomForestRegressor to complement the existing GradientBoostingRegressor and average their predictions. This keeps the overall pipeline unchanged while providing a stronger tabular baseline that should lower the RMSE toward the target. The image‑model contribution remains zero to avoid degrading performance.'
- What this solution (achieved 20.05802) has done: 'The changes add a quick hold‑out split to estimate the optimal blending weight between the GradientBoosting and RandomForest predictions, then use that weight (clamped to [0, 1]) when combining their test predictions. This keeps the original architecture untouched, still ignores the image model (whose contribution is zero), and modestly improves the tabular ensemble so the RMSE moves closer to the target.'
- What this solution (achieved 20.05653) has done: 'I keep the overall pipeline unchanged but replace the handcrafted blending of the GradientBoosting and RandomForest predictions with a tiny linear‑regression stack that learns the optimal combination on the validation split. This simple change uses the same models, adds only a few lines, and is expected to lower the RMSE toward the target without affecting any other part of the code.'

# 9. Code solution

## === cell 0
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import pandas as pd
import timm

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
)  # added RandomForestRegressor for a stronger tabular model
from sklearn.model_selection import train_test_split  # new import for validation split

base_dir = "/kaggle/input"
model_weights = os.path.join(base_dir, "saved-weights", "swin_large_fastai(final).pth")
test_folder = os.path.join(base_dir, "petfinder-pawpularity-score", "test")
test_file = os.path.join(base_dir, "petfinder-pawpularity-score", "test.csv")
train_file = os.path.join(base_dir, "petfinder-pawpularity-score", "train.csv")



## === cell 1
input_shape = (224, 224)
mean, std_dev = [0.5023, 0.4615, 0.4226], [0.2640, 0.2593, 0.2575]
model_name = "swin_large_patch4_window7_224_in22k"
batch_size = 64
device = "cuda" if torch.cuda.is_available() else "cpu"
output_categories = 1  # single regression output
n_epochs = 15
num_of_hidden = 2
hidden_dimension = [256, 64]
save_name = "/kaggle/working/swin_fastai(final).pth"



## === cell 2
test_csv = pd.read_csv(test_file)
test_csv["path_img"] = test_csv["Id"].apply(
    lambda x: os.path.join(test_folder, f"{x}.jpg")
)
test_csv["Pawpularity"] = 1




## === cell 3
class PetsDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
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
        img_path = self.df.iloc[idx]["path_img"]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        tab = torch.tensor(self.df.iloc[idx][self.cat].values.astype(np.float32))
        label = torch.tensor(self.df.iloc[idx]["Pawpularity"], dtype=torch.float32)
        return img, tab, label




## === cell 4
class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class Network(nn.Module):
    def __init__(
        self,
        base,
        number_of_hidden,
        hidden,
        regression_out,
        output_categories,
        freeze_layer,
        input_shape,
        device,
    ):
        super().__init__()
        if number_of_hidden != len(hidden):
            raise ValueError(
                "Number of Hidden layer and length of hidden dim must be same"
            )
        base.head = Identity()  # remove original classifier
        dummy = torch.randn(1, 3, input_shape[0], input_shape[1]).to(device)
        with torch.no_grad():
            dummy_feat = base(dummy)
            if dummy_feat.dim() > 2:
                dummy_feat = torch.flatten(dummy_feat, 1)
        visual_feat_dim = dummy_feat.shape[1]  # e.g. 75276
        target_vis_dim = (
            base.head.in_features if hasattr(base.head, "in_features") else hidden[0]
        )

        self.proj = nn.Linear(visual_feat_dim, target_vis_dim)

        hidden_dim = hidden[:]
        hidden_dim.insert(
            0, target_vis_dim + 12
        )  # after concatenating 12 tabular features
        self.p = 0.5
        self.regression = self._fully_connected(
            number_of_hidden, hidden_dim, output_categories
        )
        self.network = self._freeze_layer(base, freeze_layer)
        self._initialise_weights()

    def _initialise_weights(self):
        for m in self.regression:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _freeze_layer(self, base, freeze_layer):
        for i, child in enumerate(base.children()):
            if i < freeze_layer:
                for param in child.parameters():
                    param.requires_grad = False
        return base

    def _fully_connected(self, number_of_hidden, hidden_dim, output_categories):
        layers = []
        for i in range(number_of_hidden):
            layers.append(nn.Linear(hidden_dim[i], hidden_dim[i + 1]))
            layers.append(nn.GELU())
            layers.append(nn.BatchNorm1d(hidden_dim[i + 1]))
            if i != number_of_hidden - 1:
                layers.append(nn.Dropout(self.p))
        layers.append(nn.Linear(hidden_dim[-1], output_categories))
        return nn.Sequential(*layers)

    def forward(self, x, tab):
        x_feat = self.network(x)
        if x_feat.dim() > 2:
            x_feat = torch.flatten(x_feat, 1)
        x_feat = self.proj(x_feat)
        x_comb = torch.cat([x_feat, tab], dim=1)
        return self.regression(x_comb)


network = timm.create_model(model_name, pretrained=True)
network = network.to(device)  # ensure model is on the same device as dummy tensor
model = Network(
    network,
    num_of_hidden,
    hidden_dimension,
    regression_out=1,
    output_categories=output_categories,
    freeze_layer=0,
    input_shape=input_shape,
    device=device,
).to(device)



## === cell 5
try:
    state_dict = torch.load(model_weights, map_location=device)
    model.load_state_dict(state_dict, strict=False)  # allow missing fastai keys
except Exception as e:
    print(f"Weight file not loaded ({e}); proceeding with pretrained model.")
model.eval()



## === cell 6
train_df = pd.read_csv(train_file)

features = [
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

X = train_df[features].astype(float)
y = train_df["Pawpularity"].astype(float)

X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

gbr_tmp = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    random_state=42,
)
gbr_tmp.fit(X_tr, y_tr)
gbr_val_pred = gbr_tmp.predict(X_val)

rf_tmp = RandomForestRegressor(
    n_estimators=500,
    max_depth=None,
    random_state=42,
    n_jobs=5,
)
rf_tmp.fit(X_tr, y_tr)
rf_val_pred = rf_tmp.predict(X_val)

stack_X_val = np.column_stack([gbr_val_pred, rf_val_pred])
lin_blend = LinearRegression()
lin_blend.fit(stack_X_val, y_val)

gbr = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    random_state=42,
)
gbr.fit(X, y)

rf = RandomForestRegressor(
    n_estimators=500,
    max_depth=None,
    random_state=42,
    n_jobs=5,
)
rf.fit(X, y)

preds_gbr = gbr.predict(test_csv[features].astype(float))
preds_rf = rf.predict(test_csv[features].astype(float))

stack_X_test = np.column_stack([preds_gbr, preds_rf])
preds_lr = lin_blend.predict(stack_X_test)
preds_lr = np.clip(preds_lr, 0, 100)



## === cell 7
transform = transforms.Compose(
    [
        transforms.Resize(input_shape),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std_dev),
    ]
)

test_dataset = PetsDataset(test_csv, transform=transform)
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
)

model_preds = []
model.eval()
with torch.no_grad():
    for img, tab, _ in test_loader:
        img = img.to(device)
        tab = tab.to(device)
        out = model(img, tab)  # shape (batch, 1)
        model_preds.append(out.squeeze().cpu().numpy())
model_preds = np.concatenate(model_preds)

weight_model = 0.0  # keep model contribution at zero as it currently hurts performance
final_preds = weight_model * model_preds + (1 - weight_model) * preds_lr
final_preds = np.clip(final_preds, 0, 100)



## === cell 8
submission = pd.DataFrame({"Id": test_csv["Id"], "Pawpularity": final_preds})
submission.to_csv("submission.csv", index=False)
