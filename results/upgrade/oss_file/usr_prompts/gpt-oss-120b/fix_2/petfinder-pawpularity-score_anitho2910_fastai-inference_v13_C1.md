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

17.00575828788845

# 6. Current score

42.24644

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

- What this solution (achieved 42.24644) has done: 'The fix removes the CUDA‑only cuml import, avoids the name clash that made `A` a partial object, correctly uses the passed‑in SVR model inside TTA, and clamps the final predictions to the required [1, 100] range so the submission CSV is valid.'

# 9. Code solution

## === cell 0
import sys

sys.path.append("../input/saved-weights/pytorch-image-models/pytorch-image-models")



## === cell 1
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import albumentations as albu
from albumentations.pytorch import ToTensorV2
from torchvision import transforms
from torch import nn
from PIL import Image
import timm
import numpy as np
import pandas as pd
import os
import gc
import random
import pickle
from fastai.vision.all import *
from fastai.data.core import *
import matplotlib.pyplot as plt




## === cell 2
def petfinder_rmse(input, target):
    return 100 * torch.sqrt(F.mse_loss(F.sigmoid(input.flatten()), target))




## === cell 3
base_dir = "/kaggle/input"
model_weights_dir = os.path.join(base_dir, "saved-weights")
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
    "swin_large_patch4_window12_384_in22k": 64,
    "swin_base_patch4_window7_224_in22k": 128,
    "swin_large_patch4_window7_224_in22k": 64,
}
weights_svr = {
    "swin_large_patch4_window12_384_in22k": [0.1, 0.9],
    "swin_base_patch4_window7_224_in22k": [0.5, 0.5],
    "swin_large_patch4_window7_224_in22k": [0.4, 0.6],
}
model_weights = {
    "swin_large_patch4_window12_384_in22k": 0.5,
    "swin_base_patch4_window7_224_in22k": 0.2,
    "swin_large_patch4_window7_224_in22k": 0.3,
}
save_name = "/kaggle/working/"



## === cell 5
test_csv = pd.read_csv(test_file)
test_csv.head()



## === cell 6
test_csv["path_img"] = test_csv["Id"].apply(
    lambda x: os.path.join(test_folder, f"{x}.jpg")
)
test_csv["Pawpularity"] = 1  # placeholder label for dataset compatibility




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
            img = self.transform(image=np.array(img))["image"]
        tabular = self.df[self.cat].iloc[idx].values.astype(np.float32)
        return img, tabular, label_1




## === cell 8
def get_data(batch_size, max_size, input_shape):
    test_transform = albu.Compose(
        [
            albu.LongestMaxSize(max_size=max_size, interpolation=1),
            albu.PadIfNeeded(
                min_height=input_shape[0],
                min_width=input_shape[1],
                border_mode=0,
                value=(0, 0, 0),
            ),
            albu.ShiftScaleRotate(
                shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5
            ),
            albu.ColorJitter(
                brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=0.6
            ),
            albu.CenterCrop(height=input_shape[0], width=input_shape[1]),
            albu.HorizontalFlip(p=0.6),
            albu.Normalize(mean, std_dev),
            ToTensorV2(),
        ]
    )
    test_dataset = PetsDataset(test_csv, test_transform)
    testloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=4,
        shuffle=False,
        pin_memory=True,
    )
    dls = DataLoaders.from_dsets(test_dataset, bs=batch_size)
    return dls, testloader




## === cell 9
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
            raise ValueError("Number of Hidden layers and hidden dim length mismatch")
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
        for i, child in enumerate(base.children()):
            if i >= freeze_layer:
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
            if i != number_of_hidden - 1:
                layers.append(nn.Dropout(self.p))
        layers.append(nn.Linear(hidden_dim[-1], output_categories))
        return nn.Sequential(*layers)

    def forward(self, x, tab):
        x1 = self.network(x)
        x = torch.cat([x1, tab], dim=1)
        reg = self.regression(x)
        return reg, x




## === cell 10
def get_learner(
    model_name, batch_size, loss_func, metric, max_size, input_size, save_path
):
    dls, testloader = get_data(batch_size, max_size, input_size)
    dls = dls.to(device)
    network = timm.create_model(model_name, pretrained=False)
    model = Network(network, num_of_hidden, hidden_dimension, 1, 11, 0)
    model = model.to(device)
    learn = Learner(
        dls, model, loss_func=loss_func, metrics=metric, model_dir=save_path
    ).to_fp16()
    return learn, testloader




## === cell 11
def tta(testloader, learn, svr_model, svr_weight, tta_steps=4):
    learn.model.eval()
    tta_outputs = []
    for _ in range(tta_steps):
        final_outputs = []
        svr_data = []
        with torch.no_grad():
            for images, tabular, _ in testloader:
                images, tabular = images.to(device), tabular.to(device)
                reg_output, embed = learn.model(images, tabular)
                reg_output = 100 * torch.sigmoid(reg_output)
                final_outputs.extend(reg_output.squeeze().cpu().numpy())
                svr_data.extend(embed.cpu().numpy())
        final_outputs = np.array(final_outputs)
        svr_data = np.array(svr_data)
        svr_preds = svr_model.predict(svr_data)
        final_outputs = svr_weight[0] * svr_preds + svr_weight[1] * final_outputs
        tta_outputs.append(final_outputs)
    return np.mean(np.array(tta_outputs), axis=0)




## === cell 12
final_predictions = []
for model_name, bs in models_list.items():
    base_name = model_name.split("_", 5)[0]
    m_name = base_name
    svr_weights = weights_svr[model_name]
    weight = model_weights[model_name]
    is_384 = "384" in model_name
    if is_384:
        m_name += "_384"
    print("\n#################################")
    print(f"Testing Model: {m_name}")
    print("#################################")
    fold_preds = []
    for fold in range(N_FOLDS):
        print(f"Fold: {fold}")
        saved_name = f"{m_name}_fold_{fold}_full.pth"
        svr_name = f"{m_name}_svr_fold_{fold}.pkl"
        if is_384:
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
        learn.load(os.path.join("/kaggle/input/saved-weights", saved_name))
        svr_model = pickle.load(
            open(os.path.join("/kaggle/input/saved-weights", svr_name), "rb")
        )
        fold_preds.append(tta(testloader, learn, svr_model, svr_weights))
        del learn
        torch.cuda.empty_cache()
        gc.collect()
    fold_mean = np.mean(fold_preds, axis=0)
    final_predictions.append(weight * fold_mean)



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/1764646010.py in <cell line: 0>()
     36                 save_name,
     37             )
---> 38         learn.load(os.path.join("/kaggle/input/saved-weights", saved_name))
     39         svr_model = pickle.load(
     40             open(os.path.join("/kaggle/input/saved-weights", svr_name), "rb")

/usr/local/lib/python3.11/dist-packages/fastai/learner.py in load(self, file, device, **kwargs)
    426     file = join_path_file(file, self.path/self.model_dir, ext='.pth')
    427     distrib_barrier()
--> 428     load_model(file, self.model, self.opt, device=device, **kwargs)
    429     return self
    430 

/usr/local/lib/python3.11/dist-packages/fastai/learner.py in load_model(file, model, opt, with_opt, device, strict, **torch_load_kwargs)
     57     else: context = nullcontext()
     58     with context:
---> 59         state = torch.load(file, map_location=device, **torch_load_kwargs)
     60     hasopt = set(state)=={'model', 'opt'}
     61     model_state = state['model'] if hasopt else state

/usr/local/lib/python3.11/dist-packages/torch/serialization.py in load(f, map_location, pickle_module, weights_only, mmap, **pickle_load_args)
   1423         pickle_load_args["encoding"] = "utf-8"
   1424 
-> 1425     with _open_file_like(f, "rb") as opened_file:
   1426         if _is_zipfile(opened_file):
   1427             # The zipfile reader is going to advance the current file position.

/usr/local/lib/python3.11/dist-packages/torch/serialization.py in _open_file_like(name_or_buffer, mode)
    749 def _open_file_like(name_or_buffer, mode):
    750     if _is_path(name_or_buffer):
--> 751         return _open_file(name_or_buffer, mode)
    752     else:
    753         if "w" in mode:

/usr/local/lib/python3.11/dist-packages/torch/serialization.py in __init__(self, name, mode)
    730 class _open_file(_opener):
    731     def __init__(self, name, mode):
--> 732         super().__init__(open(name, mode))
    733 
    734     def __exit__(self, *args):

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/saved-weights/swin_384_fold_0_full.pth.pth'

## === cell 13
final_predictions = np.array(final_predictions)



## === cell 14
final_predictions = np.sum(final_predictions, axis=0)



## === cell 15
test_csv["Pawpularity"] = final_predictions



## === cell 16
test_csv["Pawpularity"] = np.clip(test_csv["Pawpularity"], 1, 100)



## === cell 17
test_csv = test_csv[["Id", "Pawpularity"]]



## === cell 18
test_csv.head()



## === cell 19
test_csv.to_csv("submission.csv", index=False)
