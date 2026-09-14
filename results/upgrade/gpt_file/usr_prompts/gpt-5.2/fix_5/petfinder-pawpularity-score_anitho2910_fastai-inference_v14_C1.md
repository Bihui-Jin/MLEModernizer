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

17.052717217668818

# 6. Current score

42.24644

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 42.24644) has done: 'I fix the runtime-breaking imports by removing the GPU-only `cuml` dependency (it fails due to insufficient CUDA driver) and loading the existing SVR pickles with standard `pickle` instead. I also fix the Albumentations namespace collision caused by `from fastai.vision.all import *` overwriting `A`, by re-importing Albumentations after fastai and using it via an unambiguous alias. Finally, I correct a bug in `tta()` that ignores its `svr` argument and mistakenly uses a global `clf`, and I clamp predictions to `[1, 100]` to satisfy the competition’s submission constraints (score-neutral vs invalid submission).'
- What this solution (achieved 42.24644) has done: 'I fix the failing `models.convnext` import by removing the hard dependency on a missing local `models` package and instead using `timm`’s built-in ConvNeXt models when `conv_next=True`. This keeps the same overall ensemble/loop/architecture pattern (CNN backbone → embedding head → tabular concat → MLP regressor) but makes the code runnable in the Kaggle environment. I also make `Network` robust to backbones that don’t expose `.head` (e.g., timm ConvNeXt uses `head.fc`) by detecting and replacing the correct classifier layer. These changes are execution-critical and should improve score substantially versus the current broken ConvNeXt path (which previously prevented the intended ensemble from running correctly).'
- What this solution (achieved 42.24644) has done: 'I fix the immediate `FileNotFoundError` by stopping `fastai.Learner.load()` from appending a second “.pth” extension (your `saved_name` already includes `.pth`). I also make the weights path resolution robust by pointing to the actual dataset folder that exists in your environment and adding a clear error message if a required weight/pickle is missing. These are execution-critical and score-neutral changes (they ensure the intended ensemble actually runs end-to-end). Finally, I keep the rest of the modeling/TTAs unchanged and still write a valid `submission.csv` with the required columns.'
- What this solution (achieved 42.24644) has done: 'I fix the failing weights-path resolution by searching for the actual `saved-weights` directory inside the available Kaggle input tree (including nested competition folders) and falling back safely with a clear error if nothing is found. I also remove the now-incorrect `sys.path.append("../input/...")` entries that point to non-existent datasets in your environment, since `timm` is already installed and those paths can only break imports. These changes are execution-critical and score-neutral: they only ensure your intended pre-trained ensemble/SVR pipeline can actually load its fold weights and run inference end-to-end. The submission writing remains unchanged and produce a valid `submission.csv` with `Id,Pawpularity`.'

# 9. Code solution

## === cell 0
import sys
import os
import gc
import random
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.nn.functional as F

from PIL import Image

import torchvision
import timm

from fastai.vision.all import *
from fastai.data.core import *

import albumentations as A_albu
from albumentations.pytorch import ToTensorV2

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)




## === cell 1
def petfinder_rmse(input, target):
    return 100 * torch.sqrt(F.mse_loss(torch.sigmoid(input.flatten()), target))




## === cell 2
base_dir = "/kaggle/input"
model_weights = os.path.join(base_dir, "saved-weights")
test_folder = os.path.join(base_dir, "petfinder-pawpularity-score", "test")
test_file = os.path.join(base_dir, "petfinder-pawpularity-score", "test.csv")



## === cell 3
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
models_list = {  #'convnext_base': 128,
    "conv_next": 64,
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
    "conv_next": [0.2, 0.8],
    "convnext_base": [0.2, 0.8],
}

model_weights = {
    "beit_base_patch16_224_in22k": 0.0,
    "swin_large_patch4_window12_384_in22k": 0.3,
    "swin_base_patch4_window7_224_in22k": 0.1,
    "swin_large_patch4_window7_224_in22k": 0.2,
    "convnext_base": 0.0,
    "conv_next": 0.4,
}



## === cell 4
test_csv = pd.read_csv(test_file)
test_csv.head()



## === cell 5
test_csv["path_img"] = list(
    map(lambda x: os.path.join(test_folder, x + ".jpg"), test_csv["Id"])
)
test_csv["Pawpularity"] = [1] * len(test_csv)




## === cell 6
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
            album = self.transform(image=np.array(img))
            img = album["image"]

        df_data = self.df[self.cat].iloc[idx].values.astype(np.float32)

        return (img, df_data, label_1)




## === cell 7
def get_data(batch_size, max_size, input_shape):
    test_transform = A_albu.Compose(
        [
            A_albu.LongestMaxSize(max_size=max_size, interpolation=1),
            A_albu.PadIfNeeded(
                min_height=input_shape[0],
                min_width=input_shape[1],
                border_mode=0,
                value=(0, 0, 0),
            ),
            A_albu.ShiftScaleRotate(
                shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5
            ),
            A_albu.ColorJitter(
                brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=0.6
            ),
            A_albu.CenterCrop(height=input_shape[0], width=input_shape[1]),
            A_albu.HorizontalFlip(p=0.6),
            A_albu.Normalize(mean, std_dev),
            ToTensorV2(),
        ]
    )

    test_dataset = PetsDataset(test_csv, test_transform)
    testloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=2,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )

    dls = DataLoaders.from_dsets(test_dataset, bs=batch_size)

    return dls, testloader




## === cell 8
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

        def _replace_head_with_embedding(m):
            if hasattr(m, "head"):
                h = m.head
                if isinstance(h, nn.Linear):
                    in_f = h.in_features
                    m.head = nn.Linear(in_f, embed_dim)
                    nn.init.kaiming_normal_(m.head.weight)
                    nn.init.constant_(m.head.bias, 0)
                    return True
                if hasattr(h, "fc") and isinstance(h.fc, nn.Linear):
                    in_f = h.fc.in_features
                    h.fc = nn.Linear(in_f, embed_dim)
                    nn.init.kaiming_normal_(h.fc.weight)
                    nn.init.constant_(h.fc.bias, 0)
                    return True
            if hasattr(m, "get_classifier") and hasattr(m, "reset_classifier"):
                cls = m.get_classifier()
                if isinstance(cls, nn.Linear):
                    m.reset_classifier(num_classes=embed_dim)
                    new_cls = m.get_classifier()
                    if isinstance(new_cls, nn.Linear):
                        nn.init.kaiming_normal_(new_cls.weight)
                        nn.init.constant_(new_cls.bias, 0)
                    return True
            raise AttributeError(
                "Could not locate a replaceable classifier/head on backbone."
            )

        _replace_head_with_embedding(base)

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




## === cell 9
def get_learner(
    model_name, batch_size, loss, metric, max_size, input_size, conv_next, save_path
):
    dls, testloader = get_data(batch_size, max_size, input_size)
    dls = dls.to(device)

    if conv_next:
        if model_name == "convnext_base":
            network = timm.create_model("convnext_base", pretrained=False)
        else:
            network = timm.create_model("convnext_large", pretrained=False)
        model = Network(network, num_of_hidden, hidden_dimension, 1, 11, 0)
    else:
        network = timm.create_model(model_name, pretrained=False)
        model = Network(network, num_of_hidden, hidden_dimension, 1, 11, 0)

    model = model.to(device)
    learn = Learner(
        dls, model, loss_func=loss, metrics=metric, model_dir=save_path
    ).to_fp16()
    return learn, testloader




## === cell 10
def tta(testloader, learn, svr, svr_weight, tta_steps=4):
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

        final_outputs = np.array(final_outputs, dtype=np.float32)
        svr_data = np.array(svr_data, dtype=np.float32)

        svr_preds = svr.predict(svr_data).astype(np.float32)
        final_outputs = svr_weight[0] * svr_preds + svr_weight[1] * final_outputs

        tta_outputs.append(final_outputs)

    tta_outputs_arr = np.mean(np.array(tta_outputs), axis=0)
    return tta_outputs_arr




## === cell 11
def resolve_saved_weights_dir():
    root = "/kaggle/input"
    if not os.path.isdir(root):
        return None

    direct = os.path.join(root, "saved-weights")
    if os.path.isdir(direct):
        return direct

    best = None
    best_score = -1

    for dirpath, dirnames, filenames in os.walk(root):
        if os.path.basename(dirpath) != "saved-weights":
            continue

        has_pth = any(fn.endswith(".pth") for fn in filenames)
        has_pkl = any(fn.endswith(".pkl") for fn in filenames)
        score = int(has_pth) + int(has_pkl)

        if score > best_score:
            best_score = score
            best = dirpath

        if score == 2:
            return dirpath

    return best


SAVED_WEIGHTS_DIR = resolve_saved_weights_dir()
print("Using SAVED_WEIGHTS_DIR:", SAVED_WEIGHTS_DIR)

if SAVED_WEIGHTS_DIR is None or (not os.path.isdir(SAVED_WEIGHTS_DIR)):
    raise FileNotFoundError(
        "Could not locate a 'saved-weights' directory under /kaggle/input. "
        "Please add/mount the weights dataset (with .pth model weights and .pkl SVR files)."
    )

final_predictions = []
for model_name, bs in models_list.items():
    m_name = model_name[: model_name.find("_", 5)]

    fold_tta = []
    svr_weights = weights_svr[model_name]
    w = model_weights[model_name]
    m_384 = False
    conv_next = False
    if model_name.find("384") != -1:
        m_384 = True
        m_name = m_name + "_384"

    if model_name.find("conv") != -1:
        conv_next = True
        m_name = model_name
        m_384 = True

    print("#################################")
    print("Testing Model: {}".format(m_name))
    print("#################################")
    for fold in range(N_FOLDS):
        print("Fold: {}".format(fold))
        saved_name = m_name + "_fold_{}_full.pth".format(fold)
        svr_name = m_name + "_svr_fold_{}_full.pkl".format(fold)

        if m_384:
            learn, testloader = get_learner(
                model_name,
                bs,
                BCEWithLogitsLossFlat(),
                petfinder_rmse,
                max_size_384,
                input_shape_384,
                conv_next,
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
                conv_next,
                save_name,
            )

        weight_path = os.path.join(SAVED_WEIGHTS_DIR, saved_name)
        svr_path = os.path.join(SAVED_WEIGHTS_DIR, svr_name)

        if not os.path.exists(weight_path):
            raise FileNotFoundError(
                f"Missing model weight: {weight_path}. "
                f"Available files in {SAVED_WEIGHTS_DIR}: {sorted(os.listdir(SAVED_WEIGHTS_DIR))[:50]} (showing first 50)"
            )
        if not os.path.exists(svr_path):
            raise FileNotFoundError(
                f"Missing SVR pickle: {svr_path}. "
                f"Available files in {SAVED_WEIGHTS_DIR}: {sorted(os.listdir(SAVED_WEIGHTS_DIR))[:50]} (showing first 50)"
            )

        learn.load(os.path.splitext(weight_path)[0])

        with open(svr_path, "rb") as f:
            svr_model = pickle.load(f)

        fold_tta.append(tta(testloader, learn, svr_model, svr_weights))

        del learn, svr_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    fold_tta = np.array(fold_tta)
    fold_tta = np.mean(np.array(fold_tta), axis=0)
    final_predictions.append(w * fold_tta)



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/3800381752.py in <cell line: 0>()
     41 
     42 if SAVED_WEIGHTS_DIR is None or (not os.path.isdir(SAVED_WEIGHTS_DIR)):
---> 43     raise FileNotFoundError(
     44         "Could not locate a 'saved-weights' directory under /kaggle/input. "
     45         "Please add/mount the weights dataset (with .pth model weights and .pkl SVR files)."

FileNotFoundError: Could not locate a 'saved-weights' directory under /kaggle/input. Please add/mount the weights dataset (with .pth model weights and .pkl SVR files).

## === cell 12
final_predictions = np.array(final_predictions)



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2667995920.py in <cell line: 0>()
----> 1 final_predictions = np.array(final_predictions)
      2 

NameError: name 'final_predictions' is not defined

## === cell 13
final_predictions = np.sum(final_predictions, axis=0)



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1834424963.py in <cell line: 0>()
----> 1 final_predictions = np.sum(final_predictions, axis=0)
      2 

NameError: name 'final_predictions' is not defined

## === cell 14
final_predictions = np.clip(final_predictions, 1.0, 100.0)



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/221902986.py in <cell line: 0>()
----> 1 final_predictions = np.clip(final_predictions, 1.0, 100.0)
      2 

NameError: name 'final_predictions' is not defined

## === cell 15
test_csv["Pawpularity"] = final_predictions



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/529829.py in <cell line: 0>()
----> 1 test_csv["Pawpularity"] = final_predictions
      2 

NameError: name 'final_predictions' is not defined

## === cell 16
test_csv = test_csv[["Id", "Pawpularity"]]
test_csv.head()



## === cell 17
test_csv.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", test_csv.shape)
print(test_csv.describe(include="all"))
