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

fastai==2.8.5

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

21.759740481525046

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
from fastai.vision.all import *
from fastai.tabular.all import *
import pandas as pd
import numpy as np
import os

set_seed(42, reproducible=True)



## === cell 1
path = Path("../input/petfinder-pawpularity-score")

train_dir = path / "train"
train_df = pd.read_csv(path / "train.csv")

train_df["path"] = train_df["Id"].apply(lambda x: str(train_dir / f"{x}.jpg"))
train_df.head()



## === cell 2
meta_cols = [
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

splits = RandomSplitter(valid_pct=0.2, seed=42)(range_of(train_df))

to = TabularPandas(
    train_df,
    procs=[FillMissing, Normalize],
    cont_names=meta_cols,
    cat_names=[],
    y_names="Pawpularity",
    splits=splits,
)

tab_dls = to.dataloaders(bs=64, num_workers=2)

img_dblock = DataBlock(
    blocks=(ImageBlock, RegressionBlock),
    get_x=ColReader("path"),
    get_y=ColReader("Pawpularity"),
    splitter=IndexSplitter(splits[1]),
    item_tfms=Resize(224),
    batch_tfms=Normalize.from_stats(*imagenet_stats),
)
img_dls = img_dblock.dataloaders(train_df, bs=64, num_workers=2)

img_dls.show_batch(max_n=8)



## === cell 3
import torch
import torch.nn as nn
import torchvision


class MultiModalModel(nn.Module):
    def __init__(
        self,
        arch=torchvision.models.resnet18,
        n_cont=12,
        layers=[200, 100],
        p=0.5,
        y_range=(1, 100),
    ):
        super().__init__()
        self.y_range = y_range

        self.body = create_body(arch, pretrained=True)
        nf = num_features_model(self.body)
        self.pool = nn.AdaptiveAvgPool2d(1)

        tab_szs = [n_cont] + layers
        tab_layers = []
        for i in range(len(tab_szs) - 1):
            tab_layers += [
                nn.Linear(tab_szs[i], tab_szs[i + 1]),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(tab_szs[i + 1]),
                nn.Dropout(p),
            ]
        self.tab_mlp = (
            nn.Sequential(*tab_layers) if len(tab_layers) > 0 else nn.Identity()
        )

        final_in = nf + (layers[-1] if len(layers) > 0 else n_cont)
        self.head = nn.Sequential(
            nn.Linear(final_in, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Dropout(p),
            nn.Linear(128, 1),
        )

    def forward(self, x_img, x_cont):
        x = self.body(x_img)
        x = self.pool(x)
        x = torch.flatten(x, 1)

        t = self.tab_mlp(x_cont)

        z = torch.cat([x, t], dim=1)
        out = self.head(z)

        if self.y_range is not None:
            out = sigmoid_range(out, self.y_range[0], self.y_range[1])
        return out


class MixedDL:
    """
    Zips an image dataloader and tabular dataloader and yields:
      - train/valid: ( (x_img, x_cont), y )
      - test: ( (x_img, x_cont), )
    """

    def __init__(self, img_dl, tab_dl):
        self.img_dl, self.tab_dl = img_dl, tab_dl
        self.device = img_dl.device
        self.bs = img_dl.bs

    def __len__(self):
        return len(self.img_dl)

    def _extract_cont(self, xb_tab):
        if isinstance(xb_tab, (tuple, list)) and len(xb_tab) == 2:
            return xb_tab[1]
        return xb_tab

    def __iter__(self):
        for img_batch, tab_batch in zip(self.img_dl, self.tab_dl):
            if isinstance(img_batch, (tuple, list)) and len(img_batch) == 2:
                xb_img, yb = img_batch
            elif isinstance(img_batch, (tuple, list)) and len(img_batch) == 1:
                xb_img, yb = img_batch[0], None
            else:
                xb_img, yb = img_batch, None

            if isinstance(tab_batch, (tuple, list)) and len(tab_batch) == 2:
                xb_tab, _yb_tab = tab_batch
            elif isinstance(tab_batch, (tuple, list)) and len(tab_batch) == 1:
                xb_tab = tab_batch[0]
            else:
                xb_tab = tab_batch

            x_cont = self._extract_cont(xb_tab)

            if yb is None:
                yield ((xb_img, x_cont),)
            else:
                yield ((xb_img, x_cont), yb)


class MixedDLS:
    "A minimal DataLoaders-like container for fastai Learner."

    def __init__(self, img_dls, tab_dls):
        self.img_dls, self.tab_dls = img_dls, tab_dls
        self.train = MixedDL(img_dls.train, tab_dls.train)
        self.valid = MixedDL(img_dls.valid, tab_dls.valid)
        self.device = img_dls.device

    def __getitem__(self, i):
        if i == 0:
            return self.train
        if i == 1:
            return self.valid
        raise IndexError(i)

    def to(self, device):
        self.img_dls.to(device)
        self.tab_dls.to(device)
        self.device = self.img_dls.device
        self.train.device = self.img_dls.device
        self.valid.device = self.img_dls.device
        return self


class FastaiTupleModel(nn.Module):
    """
    Adapter: fastai Learner calls model(*xb). We make xb be a single arg: (x_img,x_cont).
    """

    def __init__(self, core_model: nn.Module):
        super().__init__()
        self.core_model = core_model

    def forward(self, xb):
        x_img, x_cont = xb
        return self.core_model(x_img, x_cont)


dls = MixedDLS(img_dls, tab_dls).to(default_device())

core_model = MultiModalModel(
    arch=torchvision.models.resnet18,
    n_cont=len(meta_cols),
    layers=[200, 100],
    p=0.5,
    y_range=(1, 100),
)
model = FastaiTupleModel(core_model)

learn = Learner(dls, model, loss_func=MSELossFlat(), metrics=rmse)

learn.fit_one_cycle(3, 2e-3)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_55/2719131566.py in <cell line: 0>()
    148 dls = MixedDLS(img_dls, tab_dls).to(default_device())
    149 
--> 150 core_model = MultiModalModel(
    151     arch=torchvision.models.resnet18,
    152     n_cont=len(meta_cols),

/tmp/ipykernel_55/2719131566.py in __init__(self, arch, n_cont, layers, p, y_range)
     18         # Bugfix: don't pass a fully-instantiated model into create_body with pretrained=True again.
     19         # Keep semantics: imagenet-pretrained backbone.
---> 20         self.body = create_body(arch, pretrained=True)
     21         nf = num_features_model(self.body)
     22         self.pool = nn.AdaptiveAvgPool2d(1)

/usr/local/lib/python3.11/dist-packages/fastai/vision/learner.py in create_body(model, n_in, pretrained, cut)
     83     _update_first_layer(model, n_in, pretrained)
     84     if cut is None:
---> 85         ll = list(enumerate(model.children()))
     86         cut = next(i for i,o in reversed(ll) if has_pool_type(o))
     87     return cut_model(model, cut)

AttributeError: 'function' object has no attribute 'children'

## === cell 4
test_dir = path / "test"
test_df = pd.read_csv(path / "test.csv")
test_df["path"] = test_df["Id"].apply(lambda x: str(test_dir / f"{x}.jpg"))
test_df.head()



## === cell 5
img_test_dl = img_dls.test_dl(test_df)

to_test = to.new(test_df)  # applies FillMissing/Normalize consistently
tab_test_dl = tab_dls.test_dl(to_test.items)

test_dl = MixedDL(img_test_dl, tab_test_dl)

preds, _ = learn.get_preds(dl=test_dl)

preds_np = preds.squeeze(1).cpu().numpy()
preds_np = np.clip(preds_np, 1, 100)

submission = test_df[["Id"]].copy()
submission["Pawpularity"] = preds_np.astype(np.float32)

submission.to_csv("submission.csv", index=False)
submission.head()

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3066356176.py in <cell line: 0>()
      6 test_dl = MixedDL(img_test_dl, tab_test_dl)
      7 
----> 8 preds, _ = learn.get_preds(dl=test_dl)
      9 
     10 preds_np = preds.squeeze(1).cpu().numpy()

NameError: name 'learn' is not defined
