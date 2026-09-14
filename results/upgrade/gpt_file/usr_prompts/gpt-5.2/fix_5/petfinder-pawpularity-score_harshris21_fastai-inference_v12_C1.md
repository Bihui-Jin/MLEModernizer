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

17.026066695100802

# 6. Current score

42.24644

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 42.24644) has done: 'I fix three execution blockers while preserving the modeling/inference logic: (1) make `cuml` optional and fall back to `sklearn`’s `SVR` when CUDA driver issues prevent `cuml` from importing, (2) resolve the Albumentations namespace collision that makes `A` a `functools.partial` by importing Albumentations under a clean alias, and (3) correct the `tta()` function to actually use its passed `svr` model (it currently references an out-of-scope `clf`). Finally, I clamp predictions to the valid [1, 100] range to satisfy submission rules without changing the core ensemble/tta logic. These changes should make the notebook run end-to-end and produce a valid `submission.csv`.'
- What this solution (achieved 42.24644) has done: 'I fix the runtime error caused by `timm` no longer registering legacy model names like `beit_base_patch16_224_in22k` by adding a small compatibility mapping to current `timm` names and using it inside `get_learner` without changing the model/weights logic. I also make the environment path usage robust by pointing `timm` to the Kaggle dataset folder that actually contains the saved weights. Finally, I make loading tolerant to either FastAI-exported learner weights or plain PyTorch state_dicts (common across Kaggle datasets) so the loop can complete and always write a valid `submission.csv`.'
- What this solution (achieved 42.24644) has done: 'I fix the `timm` model-name compatibility so `create_model` no longer crashes on invalid pretrained tags (the code uses `pretrained=False`, so we should pass a plain architecture name, not a weight-tagged alias). I implement a small resolver that tries a safe sequence of candidate names (including your mapping and a tag-stripped fallback) and falls back to listing close matches if everything fails, without changing the model architecture logic. I also make the dataset/dataloader deterministic-ish and robust by setting `num_workers=0` (avoids occasional multiprocessing issues in Kaggle notebooks) while keeping the same transforms/tta loop. Finally, the pipeline run end-to-end and always write a valid `submission.csv` with the correct columns.'
- What this solution (achieved 42.24644) has done: 'I fix the `timm` model creation crash by ensuring we always pass a pure architecture name (no pretrained tag like `.in22k` / `.ms_in22k`) to `timm.create_model(..., pretrained=False)`, since newer `timm` versions error when a tag is present but `pretrained=False`. This is a minimal, execution-blocking fix that preserves your model/ensemble logic and should also improve RMSE versus the current broken/incorrect name resolution. I also add a small safety fallback in `Network` to handle backbones that don’t expose `.head` (some `timm` models use `head` vs `classifier`), without changing the architecture intent (replace the final classification layer with an embedding layer). Finally, the pipeline continue to write a valid `submission.csv` with the required columns.'

# 9. Code solution

## === cell 0
import sys

sys.path.append("../input/saved-weights/pytorch-image-models/pytorch-image-models")



## === cell 1
import os
import gc
import random
import pickle
import warnings

import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from PIL import Image

import timm
import torchvision

from fastai.vision.all import *
from fastai.data.core import *

import albumentations as alb
from albumentations.pytorch import ToTensorV2

try:
    import cuml  # noqa: F401
    from cuml.svm import SVR  # noqa: F401

    _SVR_BACKEND = "cuml"
except Exception as e:
    from sklearn.svm import SVR  # type: ignore

    _SVR_BACKEND = "sklearn"
    print(
        f"[WARN] cuML unavailable ({type(e).__name__}: {e}). Falling back to sklearn SVR for loading/predicting."
    )

import matplotlib.pyplot as plt

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)




## === cell 2
def petfinder_rmse(input, target):
    return 100 * torch.sqrt(F.mse_loss(F.sigmoid(input.flatten()), target))




## === cell 3
base_dir = "/kaggle/input"
model_weights_dir = os.path.join(base_dir, "saved-weights")
if not os.path.isdir(model_weights_dir):
    alt = os.path.join(base_dir, "petfinder-pawpularity-score", "saved-weights")
    if os.path.isdir(alt):
        model_weights_dir = alt

test_folder = os.path.join(base_dir, "petfinder-pawpularity-score", "test")
test_file = os.path.join(base_dir, "petfinder-pawpularity-score", "test.csv")



## === cell 4
mean, std_dev = [0.5023, 0.4615, 0.4226], [0.2640, 0.2593, 0.2575]
device = "cuda" if torch.cuda.is_available() else "cpu"
N_FOLDS = 5
embed_dim = 128
num_of_hidden = 2
batch_size = 64
hidden_dimension = [256, 64]
input_shape_224 = (224, 224, 3)
input_shape_384 = (384, 384, 3)
max_size_384 = 480
max_size_224 = 384

models_list = {
    "beit_base_patch16_224_in22k": 128,
    "swin_large_patch4_window12_384_in22k": 64,
    "swin_base_patch4_window7_224_in22k": 128,
    "swin_large_patch4_window7_224_in22k": 64,
}
save_name = "/kaggle/working/"

weights_svr = {
    "beit_base_patch16_224_in22k": [0.5, 0.5],
    "swin_large_patch4_window12_384_in22k": [0.1, 0.9],
    "swin_base_patch4_window7_224_in22k": [0.5, 0.5],
    "swin_large_patch4_window7_224_in22k": [0.4, 0.6],
}

model_weights = {
    "beit_base_patch16_224_in22k": 0.15,
    "swin_large_patch4_window12_384_in22k": 0.5,
    "swin_base_patch4_window7_224_in22k": 0.15,
    "swin_large_patch4_window7_224_in22k": 0.2,
}

TIMM_NAME_COMPAT = {
    "beit_base_patch16_224_in22k": "beit_base_patch16_224.in22k",
    "swin_large_patch4_window12_384_in22k": "swin_large_patch4_window12_384.ms_in22k",
    "swin_base_patch4_window7_224_in22k": "swin_base_patch4_window7_224.ms_in22k",
    "swin_large_patch4_window7_224_in22k": "swin_large_patch4_window7_224.ms_in22k",
}



## === cell 5
test_csv = pd.read_csv(test_file)



## === cell 6
test_csv["path_img"] = list(
    map(lambda x: os.path.join(test_folder, x + ".jpg"), test_csv["Id"])
)
test_csv["Pawpularity"] = [1] * len(test_csv)




## === cell 7
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
            album_out = self.transform(image=np.array(img))
            img = album_out["image"]

        df_data = self.df[self.cat].iloc[idx].values.astype(np.float32)

        return (img, df_data, label_1)




## === cell 8
def get_data(batch_size, max_size, input_shape):
    test_transform = alb.Compose(
        [
            alb.LongestMaxSize(max_size=max_size, interpolation=1),
            alb.PadIfNeeded(
                min_height=input_shape[0],
                min_width=input_shape[1],
                border_mode=0,
                value=(0, 0, 0),
            ),
            alb.ShiftScaleRotate(
                shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5
            ),
            alb.ColorJitter(
                brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=0.6
            ),
            alb.CenterCrop(height=input_shape[0], width=input_shape[1]),
            alb.HorizontalFlip(p=0.6),
            alb.Normalize(mean, std_dev),
            ToTensorV2(),
        ]
    )

    test_dataset = PetsDataset(test_csv, test_transform)

    testloader = DataLoader(
        test_dataset, batch_size=batch_size, num_workers=0, shuffle=False
    )

    dls = DataLoaders.from_dsets(test_dataset, bs=batch_size)
    return dls, testloader




## === cell 9
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
            raise ValueError(
                "Number of Hidden layer and length of hidden dim must be same"
            )

        hidden_dim = hidden[:]
        hidden_dim.insert(0, embed_dim + 12)

        if hasattr(base, "head") and isinstance(getattr(base, "head"), nn.Module):
            in_features = base.head.in_features
            base.head = nn.Linear(in_features, embed_dim)
            nn.init.kaiming_normal_(base.head.weight)
            nn.init.constant_(base.head.bias, 0)
        elif hasattr(base, "classifier") and isinstance(
            getattr(base, "classifier"), nn.Module
        ):
            in_features = base.classifier.in_features
            base.classifier = nn.Linear(in_features, embed_dim)
            nn.init.kaiming_normal_(base.classifier.weight)
            nn.init.constant_(base.classifier.bias, 0)
        else:
            raise AttributeError(
                "Backbone has neither '.head' nor '.classifier'; cannot replace final layer for embeddings."
            )

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
        layers = [nn.Dropout(self.p)]
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




## === cell 10
def resolve_timm_arch_name(model_name: str) -> str:
    """
    Bugfix: timm>=1.0 treats 'arch.tag' as a (pretrained) *tagged* model name and validates the tag.
    Since we call create_model(..., pretrained=False), we must pass only the bare arch name (no tag).
    This resolver returns a valid bare architecture name present in timm's registry.
    """
    candidates = []

    mapped = TIMM_NAME_COMPAT.get(model_name, model_name)
    candidates.append(mapped.split(".", 1)[0])

    candidates.append(model_name.split(".", 1)[0])
    if model_name.endswith("_in22k"):
        candidates.append(model_name[: -len("_in22k")])

    seen = set()
    candidates = [c for c in candidates if not (c in seen or seen.add(c))]

    for cand in candidates:
        if timm.is_model(cand):
            return cand

    all_models = timm.list_models()
    prefix = model_name.split("_")[0]
    close = [m for m in all_models if prefix in m][:25]
    raise RuntimeError(
        f"Could not resolve a valid timm arch name from '{model_name}'. "
        f"Tried (bare arch only): {candidates}. Example close matches: {close}"
    )


def get_learner(model_name, batch_size, loss, metric, max_size, input_size, save_path):
    dls, testloader = get_data(batch_size, max_size, input_size)
    dls = dls.to(device)

    timm_arch = resolve_timm_arch_name(model_name)
    network = timm.create_model(timm_arch, pretrained=False)
    model = Network(network, num_of_hidden, hidden_dimension, 1, 11, 0).to(device)

    learn = Learner(dls, model, loss_func=loss, metrics=metric, model_dir=save_path)
    if device == "cuda":
        learn = learn.to_fp16()
    return learn, testloader




## === cell 11
def tta(testloader, learn, svr, svr_weight, tta_steps=4):
    tta_outputs = []
    learn.model.eval()

    for _ in range(tta_steps):
        final_outputs = []
        svr_data = []
        with torch.no_grad():
            for images, tabular, _ in testloader:
                images = images.to(device)
                tabular = tabular.to(device)

                reg_output, embed = learn.model(images, tabular)
                reg_output = 100 * torch.sigmoid(reg_output)

                output = (
                    reg_output.detach()
                    .to("cpu")
                    .numpy()
                    .reshape(
                        -1,
                    )
                    .tolist()
                )
                final_outputs.extend(output)
                svr_data.extend(embed.detach().cpu().numpy())

        final_outputs = np.array(final_outputs)
        svr_data = np.array(svr_data)

        svr_preds = svr.predict(svr_data)
        final_outputs = svr_weight[0] * svr_preds + svr_weight[1] * final_outputs

        tta_outputs.append(final_outputs)

    tta_outputs_arr = np.mean(np.array(tta_outputs), axis=0)
    return tta_outputs_arr




## === cell 12
def safe_fastai_load_or_state_dict(learn, path_no_ext):
    """
    fastai Learner.load expects a file name *without* suffix and looks in learner.path/learner.model_dir.
    Our weights may be stored as:
      - fastai export: {name}.pth in model_dir (load with learn.load(name))
      - torch state_dict: a direct .pth file to be loaded into learn.model
    This helper tries both while keeping the core model logic unchanged.
    """
    try:
        learn.load(path_no_ext)
        return
    except Exception:
        pass

    cand = path_no_ext
    if not cand.endswith(".pth"):
        cand_pth = cand + ".pth"
    else:
        cand_pth = cand
    if os.path.isfile(cand_pth):
        sd = torch.load(cand_pth, map_location=device)
        if (
            isinstance(sd, dict)
            and "state_dict" in sd
            and isinstance(sd["state_dict"], dict)
        ):
            sd = sd["state_dict"]
        if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
            sd = sd["model"]
        learn.model.load_state_dict(sd, strict=False)
        return

    raise FileNotFoundError(
        f"Could not load weights from fastai '{path_no_ext}' or file '{cand_pth}'"
    )




## === cell 13
final_predictions = []
for model_name, bs in models_list.items():
    m_name = model_name[: model_name.find("_", 5)]

    fold_tta = []
    svr_weights = weights_svr[model_name]
    w = model_weights[model_name]
    m_384 = False
    if model_name.find("384") != -1:
        m_384 = True
        m_name = m_name + "_384"

    print("#################################")
    print("Testing Model: {}".format(m_name))
    print("#################################")

    for fold in range(N_FOLDS):
        print("Fold: {}".format(fold))
        saved_name = m_name + "_fold_{}_full".format(fold)
        svr_name = m_name + "_svr_fold_{}_full.pkl".format(fold)

        if m_384:
            learn, testloader = get_learner(
                model_name,
                bs,
                BCEWithLogitsLossFlat(),
                petfinder_rmse,
                max_size_384,
                input_shape_384,
                save_name,
            )
        else:
            learn, testloader = get_learner(
                model_name,
                bs,
                BCEWithLogitsLossFlat(),
                petfinder_rmse,
                max_size_224,
                input_shape_224,
                save_name,
            )

        safe_fastai_load_or_state_dict(
            learn, os.path.join(model_weights_dir, saved_name)
        )

        svr_path = os.path.join(model_weights_dir, svr_name)
        svr_model = pickle.load(open(svr_path, "rb"))

        fold_tta.append(tta(testloader, learn, svr_model, svr_weights))

        del learn
        torch.cuda.empty_cache()
        gc.collect()

    fold_tta = np.array(fold_tta)
    fold_tta = np.mean(fold_tta, axis=0)
    final_predictions.append(w * fold_tta)



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/1270300552.py in <cell line: 0>()
     41             )
     42 
---> 43         safe_fastai_load_or_state_dict(
     44             learn, os.path.join(model_weights_dir, saved_name)
     45         )

/tmp/ipykernel_55/777339994.py in safe_fastai_load_or_state_dict(learn, path_no_ext)
     31         return
     32 
---> 33     raise FileNotFoundError(
     34         f"Could not load weights from fastai '{path_no_ext}' or file '{cand_pth}'"
     35     )

FileNotFoundError: Could not load weights from fastai '/kaggle/input/saved-weights/beit_base_fold_0_full' or file '/kaggle/input/saved-weights/beit_base_fold_0_full.pth'

## === cell 14
final_predictions = np.array(final_predictions)



## === cell 15
final_predictions = np.sum(final_predictions, axis=0)



## === cell 16
final_predictions = np.clip(final_predictions, 1.0, 100.0)
test_csv["Pawpularity"] = final_predictions



## === cell 17
submission = test_csv[["Id", "Pawpularity"]]



## === cell 18
submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print("SVR backend:", _SVR_BACKEND)
print("Weights dir:", model_weights_dir)
