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

17.177004453848117

# 6. Current score

23.31015

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 23.31015) has done: 'I fixed the runtime errors by correcting the model names for timm.create_model (replacing “_in22k” with “.in22k”), adding the missing FastAI loss import, and handling possible NaN predictions before clipping. These changes ensure the model loads correctly, the learner is created without import errors, and the final submission values are safely bounded between 1 and 100, producing a valid submission.csv file.'
- What this solution (achieved 23.31015) has done: 'I modify the learner creation so that timm receives a valid model name: the “_in22k” suffix is removed entirely because these architectures don’t support the “in22k” pretrained tag in the installed timm version. This fixes the RuntimeError and allows the loop to load the saved weights and generate predictions, producing a correct submission.csv file. The rest of the pipeline remains unchanged, preserving the original logic and score‑related behavior.'
- What this solution (achieved 23.31015) has done: 'I correct the weight‑loading path that caused a “.pth.pth” double extension and adjust the base directory for the saved weights. This fixes the FileNotFoundError, allowing the model to load properly and generate a valid submission.csv while keeping the original logic intact.'

# 9. Code solution

## === cell 0
import sys

sys.path.append("../input/saved-weights/pytorch-image-models/pytorch-image-models")



## === cell 1
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from torchvision import transforms
from torch import nn
from PIL import Image
from albumentations.pytorch import ToTensorV2
import torchvision
import timm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from fastai.vision.all import *
from fastai.data.core import *
from fastai.losses import BCEWithLogitsLossFlat
import gc
import random
import pickle
import torch.nn.functional as F  # added for loss calculation




## === cell 2
def petfinder_rmse(input, target):
    return 100 * torch.sqrt(F.mse_loss(F.sigmoid(input.flatten()), target))




## === cell 3
base_dir = "/kaggle/input"
model_weights = os.path.join(
    base_dir, "saved-weights", "pytorch-image-models", "pytorch-image-models"
)
test_folder = os.path.join(base_dir, "petfinder-pawpularity-score", "test")
test_file = os.path.join(base_dir, "petfinder-pawpularity-score", "test.csv")



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
test_csv = pd.read_csv(test_file)
test_csv.head()



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
            img = self.transform(img)
        df_data = self.df[self.cat].iloc[idx].values.astype(np.float32)
        return (img, df_data, label_1)




## === cell 8
test_transform = transforms.Compose(
    [
        transforms.Resize(448, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop((input_shape[0], input_shape[1])),
        transforms.RandomHorizontalFlip(p=0.6),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std_dev),
    ]
)
test_dataset = PetsDataset(test_csv, test_transform)
testloader = DataLoader(
    test_dataset, batch_size=batch_size, num_workers=1, shuffle=False
)



## === cell 9
dls = DataLoaders.from_dsets(test_dataset, test_dataset, bs=batch_size, device=device)




## === cell 10
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
            raise "Number of Hidden layer and length of hidden dim must be same"

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




## === cell 11
def get_learner(dls, model_name, loss, metric, save_path):
    dls = dls.to(device)
    model_name_fixed = model_name.replace("_in22k", "")
    network = timm.create_model(model_name_fixed, pretrained=False)
    model = Network(network, num_of_hidden, hidden_dimension, 1, 11, 0)
    model = model.to(device)
    learn = Learner(
        dls, model, loss_func=loss, metrics=metric, model_dir=save_path
    ).to_fp16()
    return learn




## === cell 12
def tta(testloader, learn, tta_steps=4):
    tta_outputs = []
    learn.model.eval()
    for i in range(tta_steps):
        final_outputs = []
        with torch.no_grad():
            for images, tabular, _ in testloader:
                images, tabular = images.to(device), tabular.to(device)
                reg_output, _ = learn.model(images, tabular)
                reg_output = 100 * torch.sigmoid(reg_output)
                output = (
                    reg_output.detach()
                    .cpu()
                    .numpy()
                    .reshape(
                        -1,
                    )
                    .tolist()
                )
                final_outputs.extend(output)
        final_outputs = np.array(final_outputs)
        tta_outputs.append(final_outputs)
    tta_outputs_arr = np.mean(np.array(tta_outputs), axis=0)
    return tta_outputs_arr




## === cell 13
final_predictions = []
for model_name, bs in models_list.items():
    m_name = model_name[: model_name.find("_", 5)]
    print("#################################")
    print(f"Testing Model: {m_name}")
    print("#################################")
    fold_tta = []
    for fold in range(N_FOLDS):
        print(f"Fold: {fold}")
        saved_name = f"{m_name}_fold_{fold}"
        learn = get_learner(
            dls, model_name, BCEWithLogitsLossFlat(), petfinder_rmse, save_name
        )
        learn.load(os.path.join(model_weights, saved_name))
        fold_tta.append(tta(testloader, learn))
        del learn
        torch.cuda.empty_cache()
        gc.collect()
    fold_tta = np.mean(np.array(fold_tta), axis=0)
    final_predictions.append(fold_tta)



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/1518410171.py in <cell line: 0>()
     14             dls, model_name, BCEWithLogitsLossFlat(), petfinder_rmse, save_name
     15         )
---> 16         learn.load(os.path.join(model_weights, saved_name))
     17         fold_tta.append(tta(testloader, learn))
     18         del learn

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

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/saved-weights/pytorch-image-models/pytorch-image-models/beit_base_fold_0.pth'

## === cell 14
final_predictions = np.array(final_predictions)



## === cell 15
final_predictions = np.mean(final_predictions, axis=0)



## === cell 16
final_predictions = np.clip(final_predictions, 1, 100)
final_predictions = np.nan_to_num(final_predictions, nan=50.0)
test_csv["Pawpularity"] = final_predictions



## === cell 17
test_csv = test_csv[["Id", "Pawpularity"]]



## === cell 18
test_csv.head()



## === cell 19
test_csv.to_csv("submission.csv", index=False)
