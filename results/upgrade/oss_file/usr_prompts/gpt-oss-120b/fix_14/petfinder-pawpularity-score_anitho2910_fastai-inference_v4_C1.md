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

17.756818644265625

# 6. Current score

28.69372

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 23.31015) has done: 'I fix the model’s forward pass so that the image features are flattened before concatenating with the tabular data (the error was due to mismatched tensor dimensions). I also guard against NaN predictions by replacing them before clipping, ensuring all “Pawpularity” values lie strictly within the required 1‑100 range. These minimal changes resolve the runtime error and produce a valid submission file.'
- What this solution (achieved 23.31015) has done: 'Implemented fixes to resolve dimension mismatch and align model output with the regression target:
- Set `output_categories` to **1** so the network predicts a single Pawpularity value.
- Adjusted the `Network` initialization to compute the correct feature dimension using `base.num_features` and added the 12 tabular features, ensuring the first regression layer matches the concatenated input size.
- Updated comments to clarify these changes.

These minimal corrections eliminate the runtime error and produce a valid submission CSV, moving the RMSE closer to the target score.'
- What this solution (achieved 26.9788) has done: 'The fix addresses the dimension mismatch in the regression head by building the final linear layer dynamically based on the actual concatenated feature size (image + tabular). This eliminates the runtime error while keeping the overall model architecture unchanged, ensuring a valid submission CSV is produced and moving the RMSE toward the target score.'
- What this solution (achieved 42.23759) has done: 'I keep the overall architecture unchanged and only modify the inference post‑processing: remove the unnecessary sigmoid + 100 scaling (the model was trained to output the Pawpularity score directly) and aggregate the TTA predictions using the median instead of the mean, which is often more robust to outliers. These small tweaks should pull the RMSE down toward the target without altering training logic.'
- What this solution (achieved 42.24644) has done: 'I enable the ImageNet‑pretrained backbone (change `pretrained=False` to `pretrained=True`) so the model starts from sensible visual features instead of random weights, which should markedly lower the RMSE toward the target while keeping all other architecture and inference steps unchanged.'
- What this solution (achieved 21.45272) has done: 'I add a small random horizontal‑flip to the test transform so each TTA round sees slightly different images, and I apply the sigmoid + 100 scaling that matches the way the model was trained. These minimal changes keep the architecture untouched while converting the raw regression output into the expected 1‑100 range, which should lower the RMSE substantially and move the score toward the target.'
- What this solution (achieved 27.15242) has done: 'I adjust the test‑time prediction aggregation from a median to a mean, which is a minimal change that often yields a slightly lower RMSE for regression tasks. This keeps the model architecture, training, and other preprocessing untouched while moving the score closer to the target.'
- What this solution (achieved 21.80099) has done: 'I replace the mean‑based test‑time aggregation with a median aggregation, which prior experiments showed yields a lower RMSE for this regression task. The change is limited to cell 10: I compute the per‑sample median over the TTA predictions, handle possible NaNs, and clamp the final values to the required [1, 100] range. This minor adjustment keeps the model architecture, training, and other preprocessing unchanged while moving the score closer to the target.'
- What this solution (achieved 42.2141) has done: 'I make two minimal, targeted tweaks that are expected to lower the RMSE toward the target:
1. Remove the random horizontal flip from the test‑time transform so predictions are deterministic, avoiding unnecessary variance from TTA.
2. Drop the sigmoid + 100 scaling on the model’s raw regression output, assuming the network was trained to predict Pawpularity directly; the later clipping keep values in the required [1, 100] range.

These adjustments keep the core architecture and training unchanged while improving the inference pipeline.'
- What this solution (achieved 21.10755) has done: 'I add the missing sigmoid + 100 scaling that matches the model’s original training target range. The model’s raw regression output can be far outside [1, 100]; applying a sigmoid and scaling to 0‑100 before clamping aligns predictions with the Pawpularity scale, which should lower the RMSE toward the target. The change is limited to the post‑processing cell and retains the existing architecture, data handling, and TTA‑median aggregation.'
- What this solution (achieved 20.87559) has done: 'I add the missing `train_file` path, switch the test‑time aggregation from median to mean (which has shown slightly better RMSE), and apply a simple bias calibration using the average Pawpularity from the training set before clipping. These small adjustments keep the model architecture unchanged while nudging the predictions toward the target score.'
- What this solution (achieved 28.69372) has done: 'I replace the simple mean‑offset calibration with a full linear scaling that matches both the mean and standard deviation of the training Pawpularity distribution. This modest adjustment aligns the prediction distribution more closely with the target values while preserving the existing architecture and inference pipeline, and should reduce the RMSE toward the desired score.'

# 9. Code solution

## === cell 0
import sys, os, gc

sys.path.append(
    "../input/d/anitho2910/saved-weights/pytorch-image-models/pytorch-image-models"
)
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
import timm




## === cell 1
def petfinder_rmse(input, target):
    return 100 * torch.sqrt(F.mse_loss(F.sigmoid(input.flatten()), target))




## === cell 2
base_dir = "/kaggle/input"
model_weights = os.path.join(base_dir, "saved-weights", "swin_fastai(final).pth")
test_folder = os.path.join(base_dir, "petfinder-pawpularity-score", "test")
test_file = os.path.join(base_dir, "petfinder-pawpularity-score", "test.csv")
train_file = os.path.join(base_dir, "petfinder-pawpularity-score", "train.csv")
input_shape = (224, 224, 3)
mean, std_dev = [0.5023, 0.4615, 0.4226], [0.2640, 0.2593, 0.2575]
model_name = "swin_base_patch4_window7_224"
batch_size = 48
device = "cuda" if torch.cuda.is_available() else "cpu"
output_categories = 1  # regression produces a single value
n_epochs = 15
num_of_hidden = 2
hidden_dimension = [256, 64]
save_name = "/kaggle/working/swin_fastai(final).pth"




## === cell 3
test_csv = pd.read_csv(test_file)




## === cell 4
test_csv["path_img"] = test_csv["Id"].apply(
    lambda x: os.path.join(test_folder, f"{x}.jpg")
)
test_csv["Pawpularity"] = 1  # dummy target, never used for inference




## === cell 5
class PetsDataset(Dataset):
    def __init__(self, df, transform=None, other=False):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.other = other
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
        img_path = self.df.loc[idx, "path_img"]
        label = self.df.loc[idx, "Pawpularity"]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)  # torchvision transform returns a tensor
        tabular = self.df.loc[idx, self.cat].values.astype(np.float32)
        if self.other:
            return img, label, self.df.iloc[idx, 1:-3].to_dict()
        return img, torch.from_numpy(tabular), label




## === cell 6
test_transform = transforms.Compose(
    [
        transforms.Resize((input_shape[0], input_shape[1])),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std_dev),
    ]
)

test_dataset = PetsDataset(test_csv, transform=test_transform)
testloader = DataLoader(
    test_dataset, batch_size=batch_size, num_workers=1, shuffle=False, pin_memory=True
)




## === cell 7
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
    ):
        super().__init__()
        if number_of_hidden != len(hidden):
            raise ValueError("Number of hidden layers and hidden dim length must match")
        feature_dim = getattr(base, "num_features", None)
        if feature_dim is None:
            raise RuntimeError("Backbone model does not expose num_features")
        self.regression = None
        self.p = 0.5
        base.head = Identity()
        self.network = self._freeze_layers(base, freeze_layer)
        self._init_weights_placeholder()

    def _init_weights_placeholder(self):
        pass

    def _freeze_layers(self, base, freeze_layer):
        for i, child in enumerate(base.children()):
            if i < freeze_layer:
                for p in child.parameters():
                    p.requires_grad = False
        return base

    def forward(self, x, tab):
        x_feat = self.network(x)  # may be 4‑D
        if x_feat.dim() == 4:  # ensure (B, C, H, W)
            x_feat = torch.nn.functional.adaptive_avg_pool2d(x_feat, (1, 1))
            x_feat = x_feat.view(x_feat.size(0), -1)  # (B, C)
        x_comb = torch.cat([x_feat, tab], dim=1)

        if self.regression is None:
            in_dim = x_comb.size(1)
            self.regression = nn.Linear(in_dim, 1).to(x_comb.device)
            nn.init.kaiming_normal_(self.regression.weight)
            if self.regression.bias is not None:
                nn.init.constant_(self.regression.bias, 0)

        return self.regression(x_comb)


base_model = timm.create_model(model_name, pretrained=True)
model = Network(
    base_model,
    num_of_hidden,
    hidden_dimension,
    regression_out=1,
    output_categories=output_categories,
    freeze_layer=0,
).to(device)




## === cell 8
if os.path.isfile(model_weights):
    state = torch.load(model_weights, map_location=device)
    if isinstance(state, dict) and "model" in state:
        model.load_state_dict(state["model"])
    else:
        model.load_state_dict(state)
else:
    print("Warning: pretrained weight file not found; using random weights.")

model.eval()




## === cell 9
tta_steps = 4
all_outputs = []

for _ in range(tta_steps):
    step_outputs = []
    with torch.no_grad():
        for imgs, tabular, _ in testloader:
            imgs = imgs.to(device)
            tabular = tabular.to(device)
            preds = model(imgs, tabular)  # raw regression output
            preds = preds.squeeze().cpu().numpy()  # shape (batch,)
            step_outputs.extend(preds.tolist())
    all_outputs.append(step_outputs)




## === cell 10
tta_mean = np.nanmean(np.array(all_outputs), axis=0)

tta_scaled = 100 * torch.sigmoid(torch.from_numpy(tta_mean)).cpu().numpy()

train_targets = pd.read_csv(train_file)["Pawpularity"]
train_mean = train_targets.mean()
train_std = train_targets.std()

pred_mean = tta_scaled.mean()
pred_std = tta_scaled.std()
if pred_std == 0:
    pred_std = 1.0

tta_calibrated = (tta_scaled - pred_mean) * (train_std / pred_std) + train_mean

fallback = np.nanmedian(tta_calibrated) if np.isnan(tta_calibrated).any() else 50.0
tta_clamped = np.nan_to_num(tta_calibrated, nan=fallback)
tta_clamped = np.clip(tta_clamped, 1, 100)




## === cell 11
test_csv["Pawpularity"] = tta_clamped
submission = test_csv[["Id", "Pawpularity"]]
submission.to_csv("submission.csv", index=False)
