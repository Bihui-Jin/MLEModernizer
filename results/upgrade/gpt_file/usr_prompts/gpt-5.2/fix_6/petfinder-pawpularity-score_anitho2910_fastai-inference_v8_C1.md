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

17.177004453848117

# 6. Current score

20.08411

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 20.08411) has done: 'I remove the hard dependency on a non-existent `/kaggle/input/saved-weights` folder by auto-detecting the correct competition dataset root and using it to locate weights if present; if not, the code still run end-to-end and produce a valid CSV. I also fix the `timm` model creation error by mapping the legacy model names in your `models_list` to the closest available equivalents in `timm==1.0.19`, while preserving the same inference/ensemble logic. Finally, I guarantee the submission constraints by clipping predictions to `[1, 100]` (inclusive) using a numerically safe epsilon so Kaggle validation won’t reject values like `100.00001` or `0.99999`. These changes are execution-blocking fixes and should allow a valid submission; if weights exist, the intended inference path be used, otherwise a safe fallback prediction is generated.'
- What this solution (achieved 20.08411) has done: 'Your current score (20.08411 RMSE; lower is better) suggests you’re likely not actually using the intended saved weights path on Kaggle (so you’re falling back to a constant-mean baseline, which typically scores around ~20). The smallest change that should move you toward the target (17.177) is to fix weight discovery to search under the *actual dataset root(s)* rather than only `/kaggle/input/saved-weights`, so the ensemble inference path can activate when weights are present. I also fix a subtle but important model-key bug where you derive `m_name` by truncating the timm name, which can cause systematic “missing weights” even when they exist; instead we use the full model key consistently for filenames. These changes preserve your model/tta/SVR blending logic exactly and only improve the chance that the correct pre-trained fold weights are loaded, which should reduce RMSE toward the target.'
- What this solution (achieved 20.08411) has done: 'Your RMSE (20.08411; lower is better) is consistent with the baseline fallback path being used (mean prediction), so the smallest score-improving change is to make the “saved weights” discovery more robust so the intended ensemble inference activates when weights exist. I keep your model/tta/SVR blending logic identical, but expand the search to include any `.pth/.pkl` pairs anywhere under `/kaggle/input` and `/kaggle/working`, and then set `weights_dir` to the directory that actually contains the expected fold files for your `models_list`. Additionally, I ensure fastai loads weights directly from their original location by setting `model_dir` to `weights_dir` (avoids copy/mismatch issues), while preserving filenames and fold logic. If no weights are found anywhere, it still fall back and produce a valid submission exactly as before.'
- What this solution (achieved 20.08411) has done: 'Your current RMSE (20.08411; lower is better) looks like the weighted inference path is either not triggering or is being harmed by test-time augmentations that include random geometric/color transforms (which are not appropriate for deterministic test inference and can easily worsen RMSE). To move toward the target (17.177), I keep your ensemble/weights/SVR/tta logic intact but make the test-time transform deterministic by removing the random augmentations while preserving resize/pad/crop/normalize exactly. This is a minimal change that should improve stability and usually improves RMSE versus applying training-style augmentations at test time. Everything else (model creation, fold loading, SVR blending, clipping, and submission writing) remains unchanged.'

# 9. Code solution

## === cell 0
import sys
import os
import gc
import random
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from PIL import Image

import albumentations as albu
from albumentations.pytorch import ToTensorV2

import timm

from fastai.vision.all import *
from fastai.data.core import *


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)



## === cell 1
from sklearn.svm import SVR  # noqa: F401




## === cell 2
def petfinder_rmse(input, target):
    return 100 * torch.sqrt(F.mse_loss(torch.sigmoid(input.flatten()), target))




## === cell 3
base_dir = "/kaggle/input"


def find_dataset_root(base: str, name: str = "petfinder-pawpularity-score") -> str:
    base_p = Path(base)
    direct = base_p / name
    if direct.exists():
        return str(direct)
    for p in base_p.rglob(name):
        if p.is_dir():
            return str(p)
    raise FileNotFoundError(f"Could not locate dataset folder '{name}' under {base}")


dataset_root = find_dataset_root(base_dir, "petfinder-pawpularity-score")

model_weights = os.path.join(base_dir, "saved-weights")
test_folder = os.path.join(dataset_root, "test")
test_file = os.path.join(dataset_root, "test.csv")
train_file = os.path.join(dataset_root, "train.csv")
sample_sub_file = os.path.join(dataset_root, "sample_submission.csv")



## === cell 4
input_shape = (224, 224, 3)
mean, std_dev = [0.5023, 0.4615, 0.4226], [0.2640, 0.2593, 0.2575]
device = "cuda" if torch.cuda.is_available() else "cpu"
N_FOLDS = 5
embed_dim = 128
num_of_hidden = 2
batch_size = 64
hidden_dimension = [256, 64]

models_list = {
    "beit_base_patch16_224_in22k": 64,
    "swin_base_patch4_window7_224_in22k": 64,
    "swin_large_patch4_window7_224_in22k": 64,
}

save_name = "/kaggle/working/"



## === cell 5
assert os.path.exists(test_file), f"Missing test.csv at {test_file}"
assert os.path.isdir(test_folder), f"Missing test image folder at {test_folder}"
assert os.path.exists(
    sample_sub_file
), f"Missing sample_submission.csv at {sample_sub_file}"


def discover_weights_dir(
    model_keys,
    n_folds: int,
    roots,
):
    roots = [Path(r) for r in roots if Path(r).exists()]

    candidate_dirs = []
    for r in roots:
        for p in r.rglob("saved-weights"):
            if p.is_dir():
                candidate_dirs.append(p)

    candidate_dirs.extend(roots)

    seen = set()
    candidate_dirs = [
        p for p in candidate_dirs if not (str(p) in seen or seen.add(str(p)))
    ]

    def score_dir(d: Path) -> int:
        cnt = 0
        for m in model_keys:
            for fold in range(n_folds):
                pth = d / f"{m}_fold_{fold}.pth"
                pkl = d / f"{m}_svr_fold_{fold}.pkl"
                if pth.exists() and pkl.exists():
                    cnt += 1
        return cnt

    best_dir = None
    best_cnt = 0
    for d in candidate_dirs:
        c = score_dir(d)
        if c > best_cnt:
            best_cnt = c
            best_dir = d

    if best_cnt == 0:
        for r in roots:
            for pth in r.rglob("*_fold_0.pth"):
                d = pth.parent
                c = score_dir(d)
                if c > best_cnt:
                    best_cnt = c
                    best_dir = d

    return str(best_dir) if (best_dir is not None and best_cnt > 0) else None


weights_dir = discover_weights_dir(
    model_keys=list(models_list.keys()),
    n_folds=N_FOLDS,
    roots=[base_dir, "/kaggle/working", dataset_root],
)

print(f"Dataset root: {dataset_root}")
print(f"Test CSV: {test_file}")
print(f"Test images: {test_folder}")
print(f"Weights dir found: {weights_dir if weights_dir else 'None (will fallback)'}")



## === cell 6
test_csv = pd.read_csv(test_file)
test_csv.head()



## === cell 7
test_csv["path_img"] = [os.path.join(test_folder, f"{x}.jpg") for x in test_csv["Id"]]
test_csv["Pawpularity"] = 1.0




## === cell 8
class PetsDataset(Dataset):
    def __init__(self, df, transform=None):
        self.transform = transform
        self.df = df.reset_index(drop=True)
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
        label_1 = float(self.df.loc[idx, "Pawpularity"])

        img = Image.open(img_path).convert("RGB")
        if self.transform:
            album = self.transform(image=np.array(img))
            img = album["image"]

        df_data = self.df.loc[idx, self.cat].values.astype(np.float32)
        df_data = torch.from_numpy(df_data)

        return (img, df_data, torch.tensor(label_1, dtype=torch.float32))




## === cell 9
test_transform = albu.Compose(
    [
        albu.LongestMaxSize(max_size=448, interpolation=1),
        albu.PadIfNeeded(
            min_height=input_shape[0],
            min_width=input_shape[1],
            border_mode=0,
            value=(0, 0, 0),
        ),
        albu.CenterCrop(height=input_shape[0], width=input_shape[1]),
        albu.Normalize(mean, std_dev),
        ToTensorV2(),
    ]
)

test_dataset = PetsDataset(test_csv, test_transform)
testloader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    num_workers=1,
    shuffle=False,
    pin_memory=torch.cuda.is_available(),
)



## === cell 10
dls = DataLoaders.from_dsets(
    test_dataset, test_dataset, bs=batch_size, device=torch.device(device)
)



## === cell 11
xb, tb, yb = next(iter(testloader))
assert xb.ndim == 4, f"Expected image batch [B,C,H,W], got {xb.shape}"
assert (
    tb.ndim == 2 and tb.shape[1] == 12
), f"Expected tabular batch [B,12], got {tb.shape}"




## === cell 12
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

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
        super(Network, self).__init__()
        if number_of_hidden != len(hidden):
            raise ValueError(
                "Number of Hidden layer and length of hidden dim must be same"
            )

        hidden_dim = hidden[:]
        hidden_dim.insert(0, embed_dim + 12)

        base.head = nn.Linear(base.head.in_features, embed_dim)
        nn.init.kaiming_normal_(base.head.weight)
        nn.init.constant_(base.head.bias, 0)

        self.p = 0.5
        self.regression = self.__fully_connected(
            number_of_hidden, hidden_dim, regression_out
        )
        self.network = self.__freeze_layer(base, freeze_layer)
        self.__initialise_weights()

    def __initialise_weights(self):
        for m in self.regression:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def __freeze_layer(self, base, freeze_layer):
        cnt = 0
        for child in base.children():
            cnt += 1
            if cnt > freeze_layer:
                break
            for param in child.parameters():
                param.requires_grad = False
        return base

    def __fully_connected(self, number_of_hidden, hidden_dim, output_categories):
        layers = []
        layers.append(nn.Dropout(self.p))
        for i in range(number_of_hidden):
            layers.append(nn.Linear(hidden_dim[i], hidden_dim[i + 1]))
            layers.append(nn.GELU())
            layers.append(nn.BatchNorm1d(hidden_dim[i + 1]))
            if i != (number_of_hidden - 1):
                layers.append(nn.Dropout(self.p))
        layers.append(nn.Linear(hidden_dim[-1], output_categories))
        return nn.Sequential(*layers)

    def forward(self, x, tab):
        x1 = self.network(x)
        x = torch.cat([x1, tab], dim=1)
        reg = self.regression(x)
        return reg, x




## === cell 13
def resolve_timm_name(name: str) -> str:
    if timm.is_model(name):
        return name

    mapping = {
        "beit_base_patch16_224_in22k": "beit_base_patch16_224",
        "swin_base_patch4_window7_224_in22k": "swin_base_patch4_window7_224",
        "swin_large_patch4_window7_224_in22k": "swin_large_patch4_window7_224",
    }
    mapped = mapping.get(name, name)
    if timm.is_model(mapped):
        return mapped

    fallback = "resnet18"
    print(f"[WARN] timm model '{name}' not found; using fallback '{fallback}'.")
    return fallback


def get_learner(dls, model_name, loss, metric, save_path):
    model_name_resolved = resolve_timm_name(model_name)
    network = timm.create_model(model_name_resolved, pretrained=False)

    if not hasattr(network, "head"):
        if hasattr(network, "fc"):  # e.g., resnet
            network.head = network.fc
            network.fc = Identity()
        else:
            raise AttributeError(
                f"Model {model_name_resolved} has neither 'head' nor 'fc' attribute."
            )

    model = Network(network, num_of_hidden, hidden_dimension, 1, 11, 0).to(device)

    effective_model_dir = weights_dir if (weights_dir is not None) else save_path
    learn = Learner(
        dls, model, loss_func=loss, metrics=metric, model_dir=effective_model_dir
    )
    if device == "cuda":
        learn = learn.to_fp16()
    return learn




## === cell 14
def tta(testloader, learn, svr, tta_steps=4):
    tta_outputs = []
    learn.model.eval()
    for _ in range(tta_steps):
        final_outputs = []
        svr_data = []
        with torch.no_grad():
            for images, tabular, _ in testloader:
                images = images.to(device, non_blocking=True)
                tabular = tabular.to(device, non_blocking=True)

                reg_output, embed = learn.model(images, tabular)
                reg_output = 100 * torch.sigmoid(reg_output)
                output = reg_output.detach().cpu().numpy().reshape(-1).tolist()
                final_outputs.extend(output)
                svr_data.extend(embed.detach().cpu().numpy())

        final_outputs = np.array(final_outputs, dtype=np.float32)
        svr_data = np.array(svr_data, dtype=np.float32)

        svr_preds = np.clip(svr.predict(svr_data), 0, 100).astype(np.float32)
        blended = 0.5 * svr_preds + 0.5 * final_outputs
        tta_outputs.append(blended)

    tta_outputs_arr = np.mean(np.stack(tta_outputs, axis=0), axis=0)
    return tta_outputs_arr




## === cell 15
final_predictions = []
used_weighted_path = False

if weights_dir is not None:
    any_weight_found = False

    for model_name in models_list.keys():
        m_name = model_name  # use full key consistently
        for fold in range(N_FOLDS):
            saved_name = f"{m_name}_fold_{fold}.pth"
            svr_name = f"{m_name}_svr_fold_{fold}.pkl"
            if os.path.exists(os.path.join(weights_dir, saved_name)) and os.path.exists(
                os.path.join(weights_dir, svr_name)
            ):
                any_weight_found = True
                break
        if any_weight_found:
            break

    if any_weight_found:
        used_weighted_path = True
        for model_name, bs in models_list.items():
            m_name = model_name  # use full key consistently
            print("#################################")
            print(f"Testing Model: {m_name}")
            print("#################################")
            fold_tta = []
            for fold in range(N_FOLDS):
                print(f"Fold: {fold}")
                saved_name = f"{m_name}_fold_{fold}.pth"
                svr_name = f"{m_name}_svr_fold_{fold}.pkl"

                pth_path = os.path.join(weights_dir, saved_name)
                pkl_path = os.path.join(weights_dir, svr_name)
                if not (os.path.exists(pth_path) and os.path.exists(pkl_path)):
                    print(
                        f"[WARN] Missing weights for {m_name} fold {fold}; skipping this fold."
                    )
                    continue

                learn = get_learner(
                    dls, model_name, BCEWithLogitsLossFlat(), petfinder_rmse, save_name
                )

                tmp_name = Path(saved_name).with_suffix("")  # remove .pth
                learn.load(tmp_name.name)

                with open(pkl_path, "rb") as f:
                    clf = pickle.load(f)

                fold_tta.append(tta(testloader, learn, clf))

                del learn
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

            if len(fold_tta) == 0:
                print(f"[WARN] No folds available for {m_name}; skipping this model.")
                continue

            fold_tta = np.mean(np.stack(fold_tta, axis=0), axis=0)
            final_predictions.append(fold_tta)

if not used_weighted_path:
    print(
        "[WARN] No usable saved weights found. Falling back to a simple baseline prediction."
    )
    baseline = 50.0
    if os.path.exists(train_file):
        train_df = pd.read_csv(train_file, usecols=["Pawpularity"])
        baseline = float(train_df["Pawpularity"].mean())
    final_predictions = np.full((len(test_csv),), baseline, dtype=np.float32)
else:
    final_predictions = np.array(final_predictions, dtype=np.float32)
    final_predictions = np.mean(final_predictions, axis=0)



## === cell 16
eps = 1e-6
test_csv["Pawpularity"] = np.clip(final_predictions, 1.0 + eps, 100.0 - eps).astype(
    np.float32
)



## === cell 17
test_csv = test_csv[["Id", "Pawpularity"]]
test_csv.head()



## === cell 18
sub_path = "submission.csv"
test_csv.to_csv(sub_path, index=False)
print(
    f"Wrote {sub_path} with shape {test_csv.shape} and Pawpularity range "
    f"[{test_csv.Pawpularity.min():.6f}, {test_csv.Pawpularity.max():.6f}]"
)
print(test_csv.dtypes)
