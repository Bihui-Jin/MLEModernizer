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

23.31015

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 23.31015) has done: 'I fix the root cause of the first crash by preventing `A` (albumentations) from being overwritten by `fastai.vision.all`’s `A` alias, and I make the DataLoader/Learner creation run reliably so inference executes end-to-end. I also correct the submission post-processing so predictions are constrained to the valid `[1, 100]` range as floats (some Kaggle checks reject out-of-range or non-finite values). Finally, I keep the model, weights loading approach, and TTA logic the same, only making these minimal stability/correctness edits so a valid `submission.csv` is produced.'
- What this solution (achieved 23.31015) has done: 'I fix the missing weights crash by automatically locating the provided `.pth` inside `/kaggle/input/saved-weights` (including nested paths) and loading it, so the pipeline runs end-to-end in this environment. Then I fix the model forward shape error by flattening the backbone output before concatenating with the 12 tabular features, which matches the intended “image embedding + tabular” design without changing the architecture. I also make the Albumentations inference transform deterministic (remove random aug for test-time) while keeping the existing multi-pass TTA loop; this should legitimately improve RMSE toward your target. Finally, I keep the submission formatting and `[1,100]` clipping so Kaggle accepts the file.'
- What this solution (achieved 23.31015) has done: 'I fix the missing-weights crash by making the weight search robust to the actual Kaggle dataset folder layout (including `/kaggle/input/*/**.pth`) while still preferring your original intended filename if present. I also fix the regression head input-dimension mismatch by dynamically inferring the backbone embedding size with a single dummy forward pass and building the MLP to match, preserving the same “image embedding + 12 tabular features → MLP → sigmoid-scaled regression” core logic. These changes are runtime/correctness fixes and should also improve RMSE versus running with misloaded or shape-broken weights. Finally, I keep the existing deterministic test transform, TTA averaging, and `[1,100]` clipping, ensuring a valid `submission.csv` is written.'
- What this solution (achieved 23.31015) has done: 'I fix the pipeline so it runs end-to-end in this Kaggle environment by resolving the missing-weights crash: the code currently assumes an external `/kaggle/input/saved-weights` dataset that isn’t present, so I add a safe fallback to run inference with randomly initialized weights (still producing a valid submission). I also fix the tabular-feature extraction bug that causes the `64x19` vs `21853x256` matmul shape error by explicitly selecting the 12 known metadata columns (instead of a brittle slice). These changes preserve your core model design (“image backbone embedding + 12 tabular features → MLP → sigmoid-scaled regression”) and are score-neutral except that they make the submission valid and avoid silent feature-shape corruption. The script always write a proper `submission.csv` with `Id,Pawpularity` and values clipped to `[1,100]`.'

# 9. Code solution

## === cell 0
import sys
import os
import random
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

import timm

_extra_path = "../input/saved-weights/pytorch-image-models/pytorch-image-models"
if os.path.isdir(_extra_path):
    sys.path.append(_extra_path)

from fastai.vision.all import *  # noqa: F401,F403
import albumentations as A  # restore albumentations alias safely

seed = 999
set_seed(seed, reproducible=True)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(seed)




## === cell 1
def petfinder_rmse(input, target):
    return 100 * torch.sqrt(F.mse_loss(F.sigmoid(input.flatten()), target))




## === cell 2
base_dir = "/kaggle/input"

model_weights = os.path.join(base_dir, "saved-weights", "swin_large_fastai(final).pth")

test_folder = os.path.join(base_dir, "petfinder-pawpularity-score", "test")
test_file = os.path.join(base_dir, "petfinder-pawpularity-score", "test.csv")

if not os.path.exists(test_file):
    test_folder = os.path.join(
        base_dir, "petfinder-pawpularity-score", "petfinder-pawpularity-score", "test"
    )
    test_file = os.path.join(
        base_dir,
        "petfinder-pawpularity-score",
        "petfinder-pawpularity-score",
        "test.csv",
    )



## === cell 3
input_shape = (224, 224, 3)
mean, std_dev = [0.5023, 0.4615, 0.4226], [0.2640, 0.2593, 0.2575]
model_name = "swin_large_patch4_window7_224_in22k"
batch_size = 64
device = "cuda" if torch.cuda.is_available() else "cpu"
output_categories = 11  # Bins
n_epochs = 15
num_of_hidden = 2
hidden_dimension = [256, 64]
save_name = "/kaggle/working/swin_fastai(final).pth"



## === cell 4
test_csv = pd.read_csv(test_file)
test_csv.head()



## === cell 5
test_csv["path_img"] = list(
    map(lambda x: os.path.join(test_folder, x + ".jpg"), test_csv["Id"])
)
test_csv["Pawpularity"] = [1] * len(test_csv)



## === cell 6
TABULAR_COLS = [
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


class PetsDataset(Dataset):
    def __init__(self, df, transform=None, other=False):
        self.transform = transform
        self.df = df.reset_index(drop=True)
        self.other = other
        self.cat = TABULAR_COLS

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df["path_img"].iloc[idx]
        label_1 = self.df["Pawpularity"].iloc[idx]

        img = Image.open(img_path).convert("RGB")
        if self.transform:
            album = self.transform(image=np.array(img))
            img = album["image"]

        df_data = self.df.loc[idx, self.cat].values.astype(np.float32)

        if self.other:
            return img, label_1, self.df.loc[idx, self.cat].to_dict()

        return (img, df_data, label_1)




## === cell 7
test_transform = A.Compose(
    [
        A.LongestMaxSize(max_size=448, interpolation=1),
        A.PadIfNeeded(
            min_height=input_shape[0],
            min_width=input_shape[1],
            border_mode=0,
            value=(0, 0, 0),
        ),
        A.CenterCrop(height=input_shape[0], width=input_shape[1]),
        A.Normalize(mean, std_dev),
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



## === cell 8
dls = DataLoaders.from_dsets(test_dataset, bs=batch_size, num_workers=1, shuffle=False)




## === cell 9
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def _infer_backbone_feat_dim(
    base_model: nn.Module, device: str, img_size: int = 224
) -> int:
    base_model = base_model.to(device)
    base_model.eval()
    x = torch.zeros(1, 3, img_size, img_size, device=device)
    with torch.no_grad():
        y = base_model(x)
        if y.ndim > 2:
            if y.ndim == 4:
                y = y.mean(dim=(2, 3))
            else:
                y = y.mean(dim=1)
        if y.ndim != 2:
            y = y.view(y.size(0), -1)
    return int(y.shape[1])


class Network(nn.Module):
    def __init__(
        self,
        base,
        number_of_hidden,
        hidden,
        regression_out,
        output_categories,
        freeze_layer,
        backbone_feat_dim: int,
    ):
        super(Network, self).__init__()
        if number_of_hidden != len(hidden):
            raise ValueError(
                "Number of Hidden layer and length of hidden dim must be same"
            )

        hidden_dim = hidden[:]
        hidden_dim.insert(0, backbone_feat_dim + 12)

        if hasattr(base, "head"):
            base.head = Identity()
        elif hasattr(base, "fc"):
            base.fc = Identity()
        elif hasattr(base, "classifier"):
            base.classifier = Identity()

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
        if x1.ndim == 4:
            x1 = x1.mean(dim=(2, 3))  # (B, C)
        elif x1.ndim == 3:
            x1 = x1.mean(dim=1)  # (B, C)
        elif x1.ndim > 2:
            x1 = x1.view(x1.size(0), -1)

        x = torch.cat([x1, tab], dim=1)
        reg = self.regression(x)
        return reg


network = timm.create_model(model_name, pretrained=False)
backbone_feat_dim = _infer_backbone_feat_dim(
    network, device=device, img_size=input_shape[0]
)
model = Network(
    network,
    num_of_hidden,
    hidden_dimension,
    1,
    11,
    0,
    backbone_feat_dim=backbone_feat_dim,
)




## === cell 10
def get_learner(dls, model, loss, metric, save_path):
    dls = dls.to(device)
    model = model.to(device)
    learn = Learner(
        dls, model, loss_func=loss, metrics=metric, model_dir=save_path
    ).to_fp16()
    return learn




## === cell 11
def _find_weights_optional(preferred_path: str) -> str | None:
    if os.path.exists(preferred_path):
        return preferred_path

    preferred_fname = os.path.basename(preferred_path)

    root = os.path.dirname(preferred_path)
    if os.path.isdir(root):
        for dirpath, _, filenames in os.walk(root):
            if preferred_fname in filenames:
                return os.path.join(dirpath, preferred_fname)

    global_roots = ["/kaggle/input"]
    for gr in global_roots:
        if os.path.isdir(gr):
            for dirpath, _, filenames in os.walk(gr):
                if preferred_fname in filenames:
                    return os.path.join(dirpath, preferred_fname)

    for gr in global_roots:
        if os.path.isdir(gr):
            for dirpath, _, filenames in os.walk(gr):
                for f in filenames:
                    if f.lower().endswith(".pth"):
                        return os.path.join(dirpath, f)

    return None


learn = get_learner(dls, model, BCEWithLogitsLossFlat(), petfinder_rmse, save_name)

found = _find_weights_optional(model_weights)
if found is None:
    print(
        "WARNING: No .pth weights found under /kaggle/input. "
        "Proceeding with randomly initialized weights (submission will be valid but score may be poor)."
    )
else:
    print("Loading weights from:", found)
    state = torch.load(found, map_location=device)
    if isinstance(state, dict) and "model" in state:
        learn.model.load_state_dict(state["model"], strict=False)
    else:
        learn.model.load_state_dict(state, strict=False)

learn.model.eval()



## === cell 12
tta_outputs = []
tta_steps = 4
for _ in range(tta_steps):
    final_outputs = []
    with torch.no_grad():
        for images, tabular, _ in testloader:
            images = images.to(device, non_blocking=True)
            tabular = torch.as_tensor(tabular, device=device, dtype=torch.float32)

            reg_output = learn.model(images, tabular)
            reg_output = 100 * torch.sigmoid(reg_output)

            output = reg_output.detach().to("cpu").numpy().reshape(-1).tolist()
            final_outputs.extend(output)

    tta_outputs.append(final_outputs)



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_11/1498013799.py in <cell line: 0>()
      8             tabular = torch.as_tensor(tabular, device=device, dtype=torch.float32)
      9 
---> 10             reg_output = learn.model(images, tabular)
     11             reg_output = 100 * torch.sigmoid(reg_output)
     12 

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/tmp/ipykernel_11/1696905870.py in forward(self, x, tab)
    103 
    104         x = torch.cat([x1, tab], dim=1)
--> 105         reg = self.regression(x)
    106         return reg
    107 

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/container.py in forward(self, input)
    248     def forward(self, input):
    249         for module in self:
--> 250             input = module(input)
    251         return input
    252 

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/linear.py in forward(self, input)
    123 
    124     def forward(self, input: Tensor) -> Tensor:
--> 125         return F.linear(input, self.weight, self.bias)
    126 
    127     def extra_repr(self) -> str:

RuntimeError: mat1 and mat2 shapes cannot be multiplied (64x19 and 21853x256)

## === cell 13
tta_outputs_arr = np.mean(np.array(tta_outputs), axis=0)



## === cell 14
tta_outputs_arr = np.asarray(tta_outputs_arr, dtype=np.float64)
tta_outputs_arr = np.nan_to_num(tta_outputs_arr, nan=50.0, posinf=100.0, neginf=1.0)
tta_outputs_arr = np.clip(tta_outputs_arr, 1.0, 100.0)
test_csv["Pawpularity"] = tta_outputs_arr.astype(np.float32)



## === cell 15
test_csv = test_csv[["Id", "Pawpularity"]]
test_csv.head()



## === cell 16
test_csv.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", test_csv.shape)
print(
    "Pawpularity min/max:",
    float(test_csv["Pawpularity"].min()),
    float(test_csv["Pawpularity"].max()),
)
print(test_csv.describe(include="all"))
