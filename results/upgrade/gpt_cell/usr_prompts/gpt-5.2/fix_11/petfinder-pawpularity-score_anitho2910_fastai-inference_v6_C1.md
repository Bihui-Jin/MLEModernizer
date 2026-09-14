# Goal

You will receive environment details and a partial notebook export.

# Requirements

- Fix the bug that causes the error in cell k.
- Do NOT adjust any other non-buggy cells.
- You may reference cell k+1 only to preserve variable/interface compatibility.
- Do not complete or extend code logic in cell k, k+1, or later cells.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (bug fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Output must follow your strict format: Diagnosis / Patch summary / Updated cells / Compatibility notes for cell k+1 / Assumptions.


# 1. Python version

3.10

# 2. Installed packages

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

# 3. Data file paths

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

# 4. Code solution

## === cell 0
!pip install timm --no-index --find-links=file:///kaggle/input/libraries/ 


## === cell 1
import sys
sys.path.append("../input/saved-weights/pytorch-image-models/pytorch-image-models")


## === cell 2
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from torchvision import transforms
from torch import nn
from PIL import Image
from albumentations.pytorch import ToTensorV2
import albumentations.pytorch
import torchvision
import timm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from fastai.vision.all import *
from fastai.data.core import *
import gc
import random

seed = 999
set_seed(seed, reproducible=True)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(seed)
%matplotlib inline
%config InlineBackend.figure_format = 'retina'


## === cell 3
def petfinder_rmse(input,target):
    return 100*torch.sqrt(F.mse_loss(F.sigmoid(input.flatten()), target))


## === cell 4
base_dir = '/kaggle/input'
model_weights = os.path.join(base_dir, 'saved-weights', 'swin_large_fastai(final).pth')
test_folder = os.path.join(base_dir, 'petfinder-pawpularity-score', 'test')
test_file = os.path.join(base_dir, 'petfinder-pawpularity-score', 'test.csv')


## === cell 5
input_shape = (224, 224, 3)
mean, std_dev = [0.5023, 0.4615, 0.4226], [0.2640, 0.2593, 0.2575]
model_name = 'swin_large_patch4_window7_224_in22k'
batch_size = 64
device = 'cuda' if torch.cuda.is_available() else 'cpu'
output_categories = 11 # Bins
n_epochs = 15
num_of_hidden = 2
hidden_dimension = [256, 64]
save_name = '/kaggle/working/swin_fastai(final).pth'


## === cell 6
test_csv = pd.read_csv(test_file)
test_csv.head()


## === cell 7
test_csv['path_img'] = list(map(lambda x: os.path.join(test_folder, x+'.jpg'), test_csv['Id']))
test_csv['Pawpularity'] = [1]*len(test_csv)


## === cell 8
class PetsDataset(Dataset):
    
    def __init__(self, df, transform = None, other = False):
        self.transform = transform
        self.df = df
        self.other = other
        self.cat = ['Subject Focus', 'Eyes', 'Face', 'Near', 'Action',
       'Accessory', 'Group', 'Collage', 'Human', 'Occlusion', 'Info', 'Blur']
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_path = self.df['path_img'].iloc[idx]
        label_1 = self.df['Pawpularity'].iloc[idx]
        img = Image.open(img_path)
        if self.transform:
            album = self.transform(image = np.array(img))
            img = album['image']


        df_data = self.df[self.cat].iloc[idx].values
            
        if self.other:
            return img, label_1, self.df.iloc[idx, 1:-3].to_dict()

        
        return (img, df_data, label_1)


## === cell 9
import albumentations as albu

test_transform = albu.Compose(
    [
        albu.LongestMaxSize(max_size=448, interpolation=1),
        albu.PadIfNeeded(
            min_height=input_shape[0],
            min_width=input_shape[1],
            border_mode=0,
            value=(0, 0, 0),
        ),
        albu.ShiftScaleRotate(
            shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5
        ),
        albu.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=0.6),
        albu.RandomCrop(height=input_shape[0], width=input_shape[1]),
        albu.HorizontalFlip(p=0.6),
        albu.Normalize(mean, std_dev),
        ToTensorV2(),
    ]
)
test_dataset = PetsDataset(test_csv, test_transform)
testloader = DataLoader(
    test_dataset, batch_size=batch_size, num_workers=1, shuffle=False
)


## === cell 10
dls = DataLoaders.from_dsets(test_dataset)


## === cell 11
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

class Network(nn.Module):
    def __init__(self, base, number_of_hidden, hidden, regression_out, output_categories, freeze_layer):
        super(Network, self).__init__()
        if (number_of_hidden != len(hidden)):
            raise "Number of Hidden layer and length of hidden dim must be same"
            
        
        hidden_dim = hidden[:]
        hidden_dim.insert(0, base.head.in_features+12)
        
        base.head = Identity()
        self.p = 0.5

        self.regression = self.__fully_connected(number_of_hidden, hidden_dim, regression_out)
        
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
            cnt+=1
            if cnt > freeze_layer:
                break

            for param in child.parameters():
                param.requires_grad = False
        return base


    def __fully_connected(self, number_of_hidden, hidden_dim, output_categories):
        layers = []
        for i in range(number_of_hidden):
            layers.append(nn.Linear(hidden_dim[i], hidden_dim[i+1]))
            layers.append(nn.GELU())
            layers.append(nn.BatchNorm1d(hidden_dim[i+1]))
            if i != (number_of_hidden-1):
                layers.append(nn.Dropout(self.p))

        layers.append(nn.Linear(hidden_dim[-1], output_categories))
    
        return nn.Sequential(*layers)
        
    def forward(self,x, tab):
        x1 = self.network(x)
        x = torch.cat([x1, tab], dim=1)
        reg = self.regression(x)
        return reg

network = timm.create_model(model_name, pretrained = False)
model = Network(network, num_of_hidden, hidden_dimension, 1, 11, 0)


## === cell 12
def get_learner(dls, model, loss, metric, save_path):
    dls = dls.to(device)
    model = model.to(device)
    learn = Learner(dls, model, loss_func=loss, metrics=metric, 
                    model_dir = save_path).to_fp16()
    return learn


## === cell 13
learn = get_learner(dls, model, BCEWithLogitsLossFlat(), petfinder_rmse, save_name)


def _resolve_checkpoint_path(p: str) -> str:
    candidates = [p]

    base = os.path.basename(p)
    candidates.append(
        os.path.join("/kaggle/input/saved-weights/pytorch-image-models", base)
    )

    candidates.append(os.path.join("/kaggle/working", base))

    for c in candidates:
        if os.path.isfile(c):
            return c
    raise FileNotFoundError(f"Checkpoint not found. Tried: {candidates}")


try:
    _load_fullpath = _resolve_checkpoint_path(model_weights)

    learn.path = Path(os.path.dirname(_load_fullpath))
    learn.model_dir = ""  # so join_path_file uses learn.path directly

    _load_path = os.path.basename(_load_fullpath)
    if _load_path.endswith(".pth"):
        _load_path = _load_path[:-4]
    learn.load(_load_path)
except FileNotFoundError as e:
    print(str(e))
    print("Proceeding without loading pretrained weights (checkpoint not available).")


## === cell 14
tta_outputs = []
tta_steps = 4
for i in range(tta_steps):
    m = len(testloader)
    running_loss = 0.0
    model.eval()
    final_outputs = []
    with torch.no_grad():
        for images, tabular, _ in testloader:
            images, tabular = images.to(device), tabular.to(device)

            x1 = learn.model.network(images)
            if x1.ndim > 2:
                x1 = x1.mean(dim=tuple(range(2, x1.ndim)))
            if x1.ndim == 1:
                x1 = x1.unsqueeze(0)

            x = torch.cat([x1, tabular], dim=1)
            reg_output = learn.model.regression(x)

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

    tta_outputs.append(final_outputs)


## --- ERROR in cell 14, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mRuntimeError[0m                              Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/1791272370.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     21[0m [0;34m[0m[0m
[1;32m     22[0m             [0mx[0m [0;34m=[0m [0mtorch[0m[0;34m.[0m[0mcat[0m[0;34m([0m[0;34m[[0m[0mx1[0m[0;34m,[0m [0mtabular[0m[0;34m][0m[0;34m,[0m [0mdim[0m[0;34m=[0m[0;36m1[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 23[0;31m             [0mreg_output[0m [0;34m=[0m [0mlearn[0m[0;34m.[0m[0mmodel[0m[0;34m.[0m[0mregression[0m[0;34m([0m[0mx[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     24[0m [0;34m[0m[0m
[1;32m     25[0m             [0mreg_output[0m [0;34m=[0m [0;36m100[0m [0;34m*[0m [0mtorch[0m[0;34m.[0m[0msigmoid[0m[0;34m([0m[0mreg_output[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py[0m in [0;36m_wrapped_call_impl[0;34m(self, *args, **kwargs)[0m
[1;32m   1737[0m             [0;32mreturn[0m [0mself[0m[0;34m.[0m[0m_compiled_call_impl[0m[0;34m([0m[0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m  [0;31m# type: ignore[misc][0m[0;34m[0m[0;34m[0m[0m
[1;32m   1738[0m         [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1739[0;31m             [0;32mreturn[0m [0mself[0m[0;34m.[0m[0m_call_impl[0m[0;34m([0m[0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1740[0m [0;34m[0m[0m
[1;32m   1741[0m     [0;31m# torchrec tests the code consistency with the following code[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py[0m in [0;36m_call_impl[0;34m(self, *args, **kwargs)[0m
[1;32m   1748[0m                 [0;32mor[0m [0m_global_backward_pre_hooks[0m [0;32mor[0m [0m_global_backward_hooks[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1749[0m                 or _global_forward_hooks or _global_forward_pre_hooks):
[0;32m-> 1750[0;31m             [0;32mreturn[0m [0mforward_call[0m[0;34m([0m[0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1751[0m [0;34m[0m[0m
[1;32m   1752[0m         [0mresult[0m [0;34m=[0m [0;32mNone[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/nn/modules/container.py[0m in [0;36mforward[0;34m(self, input)[0m
[1;32m    248[0m     [0;32mdef[0m [0mforward[0m[0;34m([0m[0mself[0m[0;34m,[0m [0minput[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    249[0m         [0;32mfor[0m [0mmodule[0m [0;32min[0m [0mself[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 250[0;31m             [0minput[0m [0;34m=[0m [0mmodule[0m[0;34m([0m[0minput[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    251[0m         [0;32mreturn[0m [0minput[0m[0;34m[0m[0;34m[0m[0m
[1;32m    252[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py[0m in [0;36m_wrapped_call_impl[0;34m(self, *args, **kwargs)[0m
[1;32m   1737[0m             [0;32mreturn[0m [0mself[0m[0;34m.[0m[0m_compiled_call_impl[0m[0;34m([0m[0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m  [0;31m# type: ignore[misc][0m[0;34m[0m[0;34m[0m[0m
[1;32m   1738[0m         [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1739[0;31m             [0;32mreturn[0m [0mself[0m[0;34m.[0m[0m_call_impl[0m[0;34m([0m[0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1740[0m [0;34m[0m[0m
[1;32m   1741[0m     [0;31m# torchrec tests the code consistency with the following code[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py[0m in [0;36m_call_impl[0;34m(self, *args, **kwargs)[0m
[1;32m   1748[0m                 [0;32mor[0m [0m_global_backward_pre_hooks[0m [0;32mor[0m [0m_global_backward_hooks[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1749[0m                 or _global_forward_hooks or _global_forward_pre_hooks):
[0;32m-> 1750[0;31m             [0;32mreturn[0m [0mforward_call[0m[0;34m([0m[0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1751[0m [0;34m[0m[0m
[1;32m   1752[0m         [0mresult[0m [0;34m=[0m [0;32mNone[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/nn/modules/linear.py[0m in [0;36mforward[0;34m(self, input)[0m
[1;32m    123[0m [0;34m[0m[0m
[1;32m    124[0m     [0;32mdef[0m [0mforward[0m[0;34m([0m[0mself[0m[0;34m,[0m [0minput[0m[0;34m:[0m [0mTensor[0m[0;34m)[0m [0;34m->[0m [0mTensor[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 125[0;31m         [0;32mreturn[0m [0mF[0m[0;34m.[0m[0mlinear[0m[0;34m([0m[0minput[0m[0;34m,[0m [0mself[0m[0;34m.[0m[0mweight[0m[0;34m,[0m [0mself[0m[0;34m.[0m[0mbias[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    126[0m [0;34m[0m[0m
[1;32m    127[0m     [0;32mdef[0m [0mextra_repr[0m[0;34m([0m[0mself[0m[0;34m)[0m [0;34m->[0m [0mstr[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;31mRuntimeError[0m: mat1 and mat2 shapes cannot be multiplied (64x19 and 1548x256)

## === cell 15
tta_outputs_arr = np.mean(np.array(tta_outputs), axis = 0)
