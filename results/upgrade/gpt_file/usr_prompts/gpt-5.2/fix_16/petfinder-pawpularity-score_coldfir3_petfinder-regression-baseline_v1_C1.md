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
train_df = train_df[train_df["path"].map(os.path.exists)].reset_index(drop=True)

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

        backbone = arch(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        self.body = create_body(backbone)
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

        x_cont = x_cont.float()
        t = self.tab_mlp(x_cont)

        z = torch.cat([x, t], dim=1)
        out = self.head(z)

        if self.y_range is not None:
            out = sigmoid_range(out, self.y_range[0], self.y_range[1])
        return out


class MixedDL:
    """
    Zips an image dataloader and tabular dataloader and yields:
      - labeled: ( (x_img, x_cont), y )
      - unlabeled: (x_img, x_cont)

    BUGFIXES (runtime correctness):
      - Provide `.device` as an attribute (not property-only) to satisfy fastai's expectations in some paths.
      - Set `n_inp=2` so fastai treats xb as 2 inputs (x_img, x_cont) rather than a single nested tuple.
      - Implement `__len__` based on the *zipped* length to keep get_preds progress/termination correct.
    """

    def __init__(self, img_dl, tab_dl):
        self.img_dl, self.tab_dl = img_dl, tab_dl
        self.device = getattr(img_dl, "device", default_device())

    @property
    def bs(self):
        return self.img_dl.bs

    @property
    def n_inp(self):
        return 2

    def __len__(self):
        return min(len(self.img_dl), len(self.tab_dl))

    def to(self, device):
        if hasattr(self.img_dl, "to"):
            self.img_dl.to(device)
        if hasattr(self.tab_dl, "to"):
            self.tab_dl.to(device)
        self.device = device
        return self

    def new(self, **kwargs):
        img_new = (
            self.img_dl.new(**kwargs) if hasattr(self.img_dl, "new") else self.img_dl
        )
        tab_new = (
            self.tab_dl.new(**kwargs) if hasattr(self.tab_dl, "new") else self.tab_dl
        )
        return MixedDL(img_new, tab_new)

    def _extract_cont(self, xb_tab):
        if isinstance(xb_tab, (tuple, list)) and len(xb_tab) == 1:
            xb_tab = xb_tab[0]

        if isinstance(xb_tab, (tuple, list)) and len(xb_tab) >= 2:
            x0, x1 = xb_tab[0], xb_tab[1]
            if isinstance(x1, torch.Tensor) and x1.ndim == 2:
                return x1
            if isinstance(x0, torch.Tensor) and x0.ndim == 2:
                return x0
            if isinstance(x1, torch.Tensor):
                return x1
            if isinstance(x0, torch.Tensor):
                return x0

        return xb_tab

    def __iter__(self):
        for img_batch, tab_batch in zip(self.img_dl, self.tab_dl):
            if isinstance(img_batch, (tuple, list)):
                if len(img_batch) == 2:
                    xb_img, yb = img_batch
                elif len(img_batch) == 1:
                    xb_img, yb = img_batch[0], None
                else:
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
            xb = (xb_img, x_cont)

            if yb is None:
                yield xb
            else:
                yield (xb, yb)


class MixedDLS:
    "A minimal DataLoaders-like container for fastai Learner."

    def __init__(self, img_dls, tab_dls):
        self.img_dls, self.tab_dls = img_dls, tab_dls
        self.train = MixedDL(img_dls.train, tab_dls.train)
        self.valid = MixedDL(img_dls.valid, tab_dls.valid)
        self.device = getattr(img_dls, "device", default_device())

    def __getitem__(self, i):
        if i == 0:
            return self.train
        if i == 1:
            return self.valid
        raise IndexError(i)

    def to(self, device):
        self.img_dls.to(device)
        self.tab_dls.to(device)
        self.train.to(device)
        self.valid.to(device)
        self.device = device
        return self


dls = MixedDLS(img_dls, tab_dls).to(default_device())

model = MultiModalModel(
    arch=torchvision.models.resnet18,
    n_cont=len(meta_cols),
    layers=[200, 100],
    p=0.5,
    y_range=(1, 100),
)

learn = Learner(dls, model, loss_func=MSELossFlat(), metrics=rmse)

learn.fit_one_cycle(3, 2e-3)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2350387710.py in <cell line: 0>()
    191 learn = Learner(dls, model, loss_func=MSELossFlat(), metrics=rmse)
    192 
--> 193 learn.fit_one_cycle(3, 2e-3)
    194 

/usr/local/lib/python3.11/dist-packages/fastai/callback/schedule.py in fit_one_cycle(self, n_epoch, lr_max, div, div_final, pct_start, wd, moms, cbs, reset_opt, start_epoch)
    119     scheds = {'lr': combined_cos(pct_start, lr_max/div, lr_max, lr_max/div_final),
    120               'mom': combined_cos(pct_start, *(self.moms if moms is None else moms))}
--> 121     self.fit(n_epoch, cbs=ParamScheduler(scheds)+L(cbs), reset_opt=reset_opt, wd=wd, start_epoch=start_epoch)
    122 
    123 # %% ../../nbs/14_callback.schedule.ipynb 50

/usr/local/lib/python3.11/dist-packages/fastai/learner.py in fit(self, n_epoch, lr, wd, cbs, reset_opt, start_epoch)
    270             self.opt.set_hypers(lr=self.lr if lr is None else lr)
    271             self.n_epoch = n_epoch
--> 272             self._with_events(self._do_fit, 'fit', CancelFitException, self._end_cleanup)
    273 
    274     def _end_cleanup(self): self.dl,self.xb,self.yb,self.pred,self.loss = None,(None,),(None,),None,None

/usr/local/lib/python3.11/dist-packages/fastai/learner.py in _with_events(self, f, event_type, ex, final)
    205 
    206     def _with_events(self, f, event_type, ex, final=noop):
--> 207         try: self(f'before_{event_type}');  f()
    208         except ex: self(f'after_cancel_{event_type}')
    209         self(f'after_{event_type}');  final()

/usr/local/lib/python3.11/dist-packages/fastai/learner.py in _do_fit(self)
    259         for epoch in range(self.n_epoch):
    260             self.epoch=epoch
--> 261             self._with_events(self._do_epoch, 'epoch', CancelEpochException)
    262 
    263     def fit(self, n_epoch, lr=None, wd=None, cbs=None, reset_opt=False, start_epoch=0):

/usr/local/lib/python3.11/dist-packages/fastai/learner.py in _with_events(self, f, event_type, ex, final)
    205 
    206     def _with_events(self, f, event_type, ex, final=noop):
--> 207         try: self(f'before_{event_type}');  f()
    208         except ex: self(f'after_cancel_{event_type}')
    209         self(f'after_{event_type}');  final()

/usr/local/lib/python3.11/dist-packages/fastai/learner.py in _do_epoch(self)
    253 
    254     def _do_epoch(self):
--> 255         self._do_epoch_train()
    256         self._do_epoch_validate()
    257 

/usr/local/lib/python3.11/dist-packages/fastai/learner.py in _do_epoch_train(self)
    245     def _do_epoch_train(self):
    246         self.dl = self.dls.train
--> 247         self._with_events(self.all_batches, 'train', CancelTrainException)
    248 
    249     def _do_epoch_validate(self, ds_idx=1, dl=None):

/usr/local/lib/python3.11/dist-packages/fastai/learner.py in _with_events(self, f, event_type, ex, final)
    205 
    206     def _with_events(self, f, event_type, ex, final=noop):
--> 207         try: self(f'before_{event_type}');  f()
    208         except ex: self(f'after_cancel_{event_type}')
    209         self(f'after_{event_type}');  final()

/usr/local/lib/python3.11/dist-packages/fastai/learner.py in all_batches(self)
    211     def all_batches(self):
    212         self.n_iter = len(self.dl)
--> 213         for o in enumerate(self.dl): self.one_batch(*o)
    214 
    215     def _backward(self): self.loss_grad.backward()

/usr/local/lib/python3.11/dist-packages/fastai/learner.py in one_batch(self, i, b)
    241         b = self._set_device(b)
    242         self._split(b)
--> 243         self._with_events(self._do_one_batch, 'batch', CancelBatchException)
    244 
    245     def _do_epoch_train(self):

/usr/local/lib/python3.11/dist-packages/fastai/learner.py in _with_events(self, f, event_type, ex, final)
    205 
    206     def _with_events(self, f, event_type, ex, final=noop):
--> 207         try: self(f'before_{event_type}');  f()
    208         except ex: self(f'after_cancel_{event_type}')
    209         self(f'after_{event_type}');  final()

/usr/local/lib/python3.11/dist-packages/fastai/learner.py in _do_one_batch(self)
    222 
    223     def _do_one_batch(self):
--> 224         self.pred = self.model(*self.xb)
    225         self('after_pred')
    226         if len(self.yb):

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

TypeError: MultiModalModel.forward() missing 1 required positional argument: 'x_cont'

## === cell 4
test_dir = path / "test"
test_df = pd.read_csv(path / "test.csv")
test_df["path"] = test_df["Id"].apply(lambda x: str(test_dir / f"{x}.jpg"))
test_df = test_df[test_df["path"].map(os.path.exists)].reset_index(drop=True)

test_df.head()



## === cell 5
img_test_dl = img_dls.test_dl(test_df, with_labels=False)

to_test = to.new(test_df)
tab_test_dl = to_test.dataloaders(bs=64, num_workers=2, shuffle=False).train

test_dl = MixedDL(img_test_dl, tab_test_dl).to(default_device())

preds, _ = learn.get_preds(dl=test_dl)

preds_np = preds.squeeze(1).cpu().numpy()
preds_np = np.clip(preds_np, 1, 100)

submission = test_df[["Id"]].copy()
submission["Pawpularity"] = preds_np.astype(np.float32)

submission.to_csv("submission.csv", index=False)
submission.head()

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4183428288.py in <cell line: 0>()
      8 test_dl = MixedDL(img_test_dl, tab_test_dl).to(default_device())
      9 
---> 10 preds, _ = learn.get_preds(dl=test_dl)
     11 
     12 preds_np = preds.squeeze(1).cpu().numpy()

/usr/local/lib/python3.11/dist-packages/fastai/learner.py in get_preds(self, ds_idx, dl, with_input, with_decoded, with_loss, act, inner, reorder, cbs, **kwargs)
    314         if with_loss: ctx_mgrs.append(self.loss_not_reduced())
    315         with ContextManagers(ctx_mgrs):
--> 316             self._do_epoch_validate(dl=dl)
    317             if act is None: act = getcallable(self.loss_func, 'activation')
    318             res = cb.all_tensors()

/usr/local/lib/python3.11/dist-packages/fastai/learner.py in _do_epoch_validate(self, ds_idx, dl)
    250         if dl is None: dl = self.dls[ds_idx]
    251         self.dl = dl
--> 252         with torch.no_grad(): self._with_events(self.all_batches, 'validate', CancelValidException)
    253 
    254     def _do_epoch(self):

/usr/local/lib/python3.11/dist-packages/fastai/learner.py in _with_events(self, f, event_type, ex, final)
    205 
    206     def _with_events(self, f, event_type, ex, final=noop):
--> 207         try: self(f'before_{event_type}');  f()
    208         except ex: self(f'after_cancel_{event_type}')
    209         self(f'after_{event_type}');  final()

/usr/local/lib/python3.11/dist-packages/fastai/learner.py in all_batches(self)
    211     def all_batches(self):
    212         self.n_iter = len(self.dl)
--> 213         for o in enumerate(self.dl): self.one_batch(*o)
    214 
    215     def _backward(self): self.loss_grad.backward()

/usr/local/lib/python3.11/dist-packages/fastai/learner.py in one_batch(self, i, b)
    241         b = self._set_device(b)
    242         self._split(b)
--> 243         self._with_events(self._do_one_batch, 'batch', CancelBatchException)
    244 
    245     def _do_epoch_train(self):

/usr/local/lib/python3.11/dist-packages/fastai/learner.py in _with_events(self, f, event_type, ex, final)
    205 
    206     def _with_events(self, f, event_type, ex, final=noop):
--> 207         try: self(f'before_{event_type}');  f()
    208         except ex: self(f'after_cancel_{event_type}')
    209         self(f'after_{event_type}');  final()

/usr/local/lib/python3.11/dist-packages/fastai/learner.py in _do_one_batch(self)
    222 
    223     def _do_one_batch(self):
--> 224         self.pred = self.model(*self.xb)
    225         self('after_pred')
    226         if len(self.yb):

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

TypeError: MultiModalModel.forward() missing 1 required positional argument: 'x_cont'
