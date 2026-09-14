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
cuml-cu12==25.2.1
fastai==2.8.5
geopandas==0.14.4
libcuml-cu12==25.2.1
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
model-signing==1.1.1
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
sigstore-models==0.0.5
sklearn-pandas==2.2.0
statsmodels==0.14.5
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

17.30894069590541

# 6. Current score

20.08338

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 42.24644) has done: 'I fixed the CUDA‑related import error, renamed the Albumentations import to avoid a name clash, added the missing functional import, and clipped the final predictions to the required [1, 100] range so the submission is accepted.'
- What this solution (achieved 20.44993) has done: 'I fix the dtype conversion error in `MetaDataset.__getitem__`, explicitly casting the selected metadata columns to `float32` before creating a tensor. This prevents the `numpy.object_` issue that stops the DataLoader, allowing the training and inference loops to run and produce a valid submission CSV. No other logic is changed, preserving the original model and workflow.'
- What this solution (achieved 20.11685) has done: 'I add reproducible seeding, a validation‑based checkpoint, and a learning‑rate scheduler while extending the training to 30 epochs. The training loop now saves the model that achieved the lowest validation RMSE and restores it before inference, which should lower the final RMSE and move the score closer to the target without altering the core MLP architecture or data handling.'
- What this solution (achieved 20.11861) has done: 'I add proper feature standardisation using the training set statistics so the MLP receives zero‑mean, unit‑variance inputs; this small preprocessing change often lowers RMSE without altering the model architecture or training loop. I compute means and standard deviations for the categorical columns, pass them to the MetaDataset, and normalise both train/validation and test features accordingly. The rest of the pipeline (model, loss, scheduler, prediction clipping) remains unchanged, keeping the core logic intact while moving the score closer to the target.'
- What this solution (achieved 20.11734) has done: 'I add a small L2 regularisation (weight decay) to the Adam optimizer and extend the training to up to 60 epochs while stopping early if the validation RMSE does not improve for 5 consecutive epochs. These modest tweaks keep the original MLP and training logic intact but can help the model converge to a lower error, moving the score closer to the target.'
- What this solution (achieved 20.13275) has done: 'I lower the initial learning rate and use AdamW with a smaller weight decay, extend the maximum epochs while giving early‑stopping a bit more patience, and make the ReduceLROnPlateau scheduler respond faster. These minimal tweaks keep the same MLP architecture and data handling but should help the model converge to a lower validation RMSE, moving the score closer to the target.'
- What this solution (achieved 20.11898) has done: 'I lower the learning rate, increase hidden layer sizes slightly, and give the early‑stopping patience a bit more room so the model can converge a little further without altering the overall pipeline. These modest tweaks are expected to shave off a few RMSE points, moving the score closer to the target while keeping the original architecture and data handling intact.'
- What this solution (achieved 20.05882) has done: 'I slightly enlarge the MLP hidden layers to give the model a bit more capacity, lower the initial learning rate, and give early stopping a bit more patience so training can converge a little further. These modest tweaks keep the overall architecture and training loop intact while aiming to reduce the validation RMSE and move the score closer to the target.'
- What this solution (achieved 20.07541) has done: 'I slightly increase the MLP capacity, lower the learning rate and weight decay, make the scheduler a bit more aggressive, and give early‑stopping a little more patience. These modest changes keep the same overall architecture and training loop but should help the model converge to a lower validation RMSE, moving the score closer to the target.'
- What this solution (achieved 20.08338) has done: 'We eliminate the Python‑level DataLoader overhead by materialising the normalized meta‑features once as tensors, then manually batching them (shuffling the training indices each epoch). This keeps the exact MLP architecture, loss, optimizer and early‑stopping logic, but speeds up data access dramatically and stays well within the 600 s limit.'

# 9. Code solution

## === cell 0
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import random

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)




## === cell 1
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[512, 256, 128, 64, 32]):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)




## === cell 2
train_path = "/kaggle/input/petfinder-pawpularity-score/train.csv"
test_path = "/kaggle/input/petfinder-pawpularity-score/test.csv"
sample_sub_path = "/kaggle/input/petfinder-pawpularity-score/sample_submission.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

cat_cols = [
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

train_df[cat_cols] = train_df[cat_cols].fillna(0).astype(np.float32)
test_df[cat_cols] = test_df[cat_cols].fillna(0).astype(np.float32)

cat_means = train_df[cat_cols].mean()
cat_stds = train_df[cat_cols].std().replace(0, 1)

train_feats = torch.tensor(
    ((train_df[cat_cols] - cat_means) / cat_stds).values,
    dtype=torch.float32,
)
train_labels = torch.tensor(train_df["Pawpularity"].values, dtype=torch.float32)

test_feats = torch.tensor(
    ((test_df[cat_cols] - cat_means) / cat_stds).values,
    dtype=torch.float32,
)
test_ids = test_df["Id"].values

val_frac = 0.1
val_size = int(len(train_feats) * val_frac)
train_size = len(train_feats) - val_size
indices = torch.randperm(len(train_feats), generator=torch.Generator().manual_seed(42))
train_idx = indices[:train_size]
val_idx = indices[train_size:]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



## === cell 3
model = MLP(input_dim=len(cat_cols)).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=1, verbose=False
)

epochs = 500
patience_es = 40
best_val_rmse = float("inf")
best_model_path = "/kaggle/working/best_model.pt"
no_improve_cnt = 0

model.train()
for epoch in range(epochs):
    perm = torch.randperm(train_size, generator=torch.Generator().manual_seed(epoch))
    epoch_loss = 0.0

    for start in range(0, train_size, 256):
        batch_idx = train_idx[perm[start : start + 256]]
        feats = train_feats[batch_idx].to(device)
        labels = train_labels[batch_idx].to(device)

        optimizer.zero_grad()
        outputs = model(feats)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * feats.size(0)

    epoch_loss /= train_size

    model.eval()
    with torch.no_grad():
        val_losses = []
        for start in range(0, val_size, 256):
            batch_idx = val_idx[start : start + 256]
            feats = train_feats[batch_idx].to(device)
            labels = train_labels[batch_idx].to(device)
            outputs = model(feats)
            val_losses.append(F.mse_loss(outputs, labels, reduction="sum").item())
        val_mse = sum(val_losses) / val_size
        val_rmse = np.sqrt(val_mse)

    if val_rmse < best_val_rmse:
        best_val_rmse = val_rmse
        torch.save(model.state_dict(), best_model_path)
        no_improve_cnt = 0
    else:
        no_improve_cnt += 1

    scheduler.step(val_rmse)

    print(
        f"Epoch {epoch+1}/{epochs} - Train Loss: {epoch_loss:.4f} - Val RMSE: {val_rmse:.4f}"
    )

    if no_improve_cnt >= patience_es:
        print(f"Early stopping triggered after {epoch+1} epochs.")
        break

    model.train()

model.load_state_dict(torch.load(best_model_path))
model.eval()



## === cell 4
model.eval()
preds = []
ids = []
batch_size = 256
with torch.no_grad():
    for start in range(0, len(test_feats), batch_size):
        feats = test_feats[start : start + batch_size].to(device)
        out = model(feats)
        out = out.cpu().numpy()
        out = np.clip(out, 0, 100)
        preds.extend(out.tolist())
        ids.extend(test_ids[start : start + batch_size])

submission = pd.DataFrame({"Id": ids, "Pawpularity": preds})
sample_sub = pd.read_csv(sample_sub_path)
submission = submission.set_index("Id").reindex(sample_sub["Id"]).reset_index()

output_path = "/kaggle/working/submission.csv"
submission.to_csv(output_path, index=False)
print(f"Submission written to {output_path}")
