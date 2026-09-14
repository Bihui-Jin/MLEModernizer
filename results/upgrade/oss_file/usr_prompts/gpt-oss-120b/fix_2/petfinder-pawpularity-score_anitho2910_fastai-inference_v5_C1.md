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

17.74031641015639

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, random, gc
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import timm

seed = 999
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False




## === cell 1
def petfinder_rmse(input, target):
    return 100 * torch.sqrt(F.mse_loss(torch.sigmoid(input.flatten()), target))




## === cell 2
base_dir = "/kaggle/input"
test_folder = os.path.join(base_dir, "petfinder-pawpularity-score", "test")
test_file = os.path.join(base_dir, "petfinder-pawpularity-score", "test.csv")
train_file = os.path.join(base_dir, "petfinder-pawpularity-score", "train.csv")
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "swin_base_patch4_window7_224"
batch_size = 64
input_shape = (224, 224)



## === cell 3
test_csv = pd.read_csv(test_file)
test_csv["path_img"] = test_csv["Id"].apply(
    lambda x: os.path.join(test_folder, f"{x}.jpg")
)




## === cell 4
class PetsDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
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
        row = self.df.iloc[idx]
        img_path = row["path_img"]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)  # returns a Tensor
        tab = torch.tensor(row[self.cat].values, dtype=torch.float32)
        label = torch.tensor(0.0, dtype=torch.float32)
        return img, tab, label




## === cell 5
test_transform = transforms.Compose(
    [
        transforms.Resize(448, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(input_shape),
        transforms.RandomHorizontalFlip(p=0.6),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5023, 0.4615, 0.4226], std=[0.2640, 0.2593, 0.2575]
        ),
    ]
)

test_dataset = PetsDataset(test_csv, transform=test_transform)
testloader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True
)




## === cell 6
class Identity(nn.Module):
    def forward(self, x):
        return x


class Network(nn.Module):
    def __init__(
        self,
        base,
        num_hidden,
        hidden_dims,
        out_features,
        output_categories,
        freeze_until,
    ):
        super().__init__()
        if num_hidden != len(hidden_dims):
            raise ValueError("num_hidden must match length of hidden_dims")
        hidden = [base.head.in_features + 12] + hidden_dims
        base.head = Identity()  # remove original head
        self.backbone = self._freeze_layers(base, freeze_until)
        self.regressor = self._make_regressor(num_hidden, hidden, output_categories)
        self.dropout_p = 0.5

    def _freeze_layers(self, model, freeze_until):
        for i, child in enumerate(model.children()):
            if i < freeze_until:
                for p in child.parameters():
                    p.requires_grad = False
        return model

    def _make_regressor(self, n_hidden, dims, out_features):
        layers = []
        for i in range(n_hidden):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.GELU())
            layers.append(nn.BatchNorm1d(dims[i + 1]))
            if i != n_hidden - 1:
                layers.append(nn.Dropout(self.dropout_p))
        layers.append(nn.Linear(dims[-1], out_features))
        return nn.Sequential(*layers)

    def forward(self, img, tab):
        feats = self.backbone(img)  # [B, feat_dim]
        x = torch.cat([feats, tab], dim=1)
        return self.regressor(x)


backbone = timm.create_model(model_name, pretrained=True)
model = Network(
    backbone,
    num_hidden=2,
    hidden_dims=[256, 64],
    out_features=1,
    output_categories=1,
    freeze_until=0,
).to(device)
model.eval()



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_55/2802802826.py in <cell line: 0>()
     50 # create backbone with pretrained weights
     51 backbone = timm.create_model(model_name, pretrained=True)
---> 52 model = Network(
     53     backbone,
     54     num_hidden=2,

/tmp/ipykernel_55/2802802826.py in __init__(self, base, num_hidden, hidden_dims, out_features, output_categories, freeze_until)
     21         base.head = Identity()  # remove original head
     22         self.backbone = self._freeze_layers(base, freeze_until)
---> 23         self.regressor = self._make_regressor(num_hidden, hidden, output_categories)
     24         self.dropout_p = 0.5
     25 

/tmp/ipykernel_55/2802802826.py in _make_regressor(self, n_hidden, dims, out_features)
     38             layers.append(nn.BatchNorm1d(dims[i + 1]))
     39             if i != n_hidden - 1:
---> 40                 layers.append(nn.Dropout(self.dropout_p))
     41         layers.append(nn.Linear(dims[-1], out_features))
     42         return nn.Sequential(*layers)

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in __getattr__(self, name)
   1926             if name in modules:
   1927                 return modules[name]
-> 1928         raise AttributeError(
   1929             f"'{type(self).__name__}' object has no attribute '{name}'"
   1930         )

AttributeError: 'Network' object has no attribute 'dropout_p'

## === cell 7
tta_steps = 4
all_outputs = []

with torch.no_grad():
    for _ in range(tta_steps):
        step_outputs = []
        for imgs, tabs, _ in testloader:
            imgs = imgs.to(device, non_blocking=True)
            tabs = tabs.to(device, non_blocking=True)
            preds = model(imgs, tabs)  # raw logits
            preds = 100 * torch.sigmoid(preds)  # match competition scaling
            step_outputs.extend(preds.squeeze(1).cpu().numpy().tolist())
        all_outputs.append(step_outputs)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1850226051.py in <cell line: 0>()
      5     for _ in range(tta_steps):
      6         step_outputs = []
----> 7         for imgs, tabs, _ in testloader:
      8             imgs = imgs.to(device, non_blocking=True)
      9             tabs = tabs.to(device, non_blocking=True)

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in __next__(self)
    706                 # TODO(https://github.com/pytorch/pytorch/issues/76750)
    707                 self._reset()  # type: ignore[call-arg]
--> 708             data = self._next_data()
    709             self._num_yielded += 1
    710             if (

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _next_data(self)
   1478                 del self._task_info[idx]
   1479                 self._rcvd_idx += 1
-> 1480                 return self._process_data(data)
   1481 
   1482     def _try_put_index(self):

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _process_data(self, data)
   1503         self._try_put_index()
   1504         if isinstance(data, ExceptionWrapper):
-> 1505             data.reraise()
   1506         return data
   1507 

/usr/local/lib/python3.11/dist-packages/torch/_utils.py in reraise(self)
    731             # instantiate since we don't know how to
    732             raise RuntimeError(msg) from None
--> 733         raise exception
    734 
    735 

TypeError: Caught TypeError in DataLoader worker process 0.
Original Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/worker.py", line 349, in _worker_loop
    data = fetcher.fetch(index)  # type: ignore[possibly-undefined]
           ^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py", line 52, in fetch
    data = [self.dataset[idx] for idx in possibly_batched_index]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py", line 52, in <listcomp>
    data = [self.dataset[idx] for idx in possibly_batched_index]
            ~~~~~~~~~~~~^^^^^
  File "/tmp/ipykernel_55/1187488496.py", line 29, in __getitem__
    tab = torch.tensor(row[self.cat].values, dtype=torch.float32)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: can't convert np.ndarray of type numpy.object_. The only supported types are: float64, float32, float16, complex64, complex128, int64, int32, int16, int8, uint64, uint32, uint16, uint8, and bool.


## === cell 8
tta_mean = np.mean(np.array(all_outputs), axis=0)



## === cell 9
tta_clipped = np.clip(tta_mean, 1, 100)

test_csv["Pawpularity"] = tta_clipped
submission = test_csv[["Id", "Pawpularity"]]
submission.to_csv("submission.csv", index=False)

## --- ERROR in outputing the csv:
Invalid submission: Pawpularity in submission should be between 1 and 100
