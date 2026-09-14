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
Detect apple diseases from images.

## Metric
Mean F1-Score

## Submission Format
labels should be a space-delimited list.

The file should contain a header and have the following format:

```
image, labels
85f8cb619c66b863.jpg,healthy
ad8770db05586b59.jpg,healthy
c7b03e718489f3ca.jpg,healthy
```

## Dataset
**train.csv** - the training set metadata.

- `image` - the image ID.
- `labels` - the target classes, a space delimited list of all diseases found in the image. Unhealthy leaves with too many diseases to classify visually will have the `complex` class, and may also have a subset of the diseases identified.

**sample_submission.csv** - A sample submission file in the correct format.

- `image`
- `labels`

**train_images** - The training set images.

**test_images** - The test set images. This competition has a hidden test set: only three images are provided here as samples while the remaining 5,000 images will be available to your notebook once it is submitted.

# 2. Python version

3.9

# 3. Installed packages

fastai==2.8.5

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
        input/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
        working/
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
```

-> data/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> data/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> (stopped after 10 files for performance)

# 5. Target score

0.7698799630655587

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
from fastai.vision.all import *
import os, pandas as pd, numpy as np
from pathlib import Path
import torch

torch.backends.cudnn.benchmark = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

BASE_CANDIDATES = [
    "/kaggle/input/plant-pathology-2021-fgvc8",
    "/kaggle/data/plant-pathology-2021-fgvc8",
    "../input/plant-pathology-2021-fgvc8",
]
BASE = next((p for p in BASE_CANDIDATES if os.path.isdir(p)), None)
if BASE is None:
    raise FileNotFoundError(
        f"Could not find dataset directory in candidates: {BASE_CANDIDATES}"
    )

TRAIN_CSV = os.path.join(BASE, "train.csv")
SAMPLE_SUB = os.path.join(BASE, "sample_submission.csv")
TRAIN_IMGS = os.path.join(BASE, "train_images")
TEST_IMGS = os.path.join(BASE, "test_images")

for p in [TRAIN_CSV, SAMPLE_SUB, TRAIN_IMGS, TEST_IMGS]:
    if not os.path.exists(p):
        raise FileNotFoundError(f"Required path missing: {p}")

valid_exts = {".jpg", ".jpeg", ".png"}
test_filenames = sorted(
    e.name
    for e in os.scandir(TEST_IMGS)
    if e.is_file() and os.path.splitext(e.name)[1].lower() in valid_exts
)

train_df = pd.read_csv(TRAIN_CSV)
sample_sub = pd.read_csv(SAMPLE_SUB)
print("BASE:", BASE)
print(
    "train_df:",
    train_df.shape,
    "sample_sub:",
    sample_sub.shape,
    "n_test_files:",
    len(test_filenames),
)



## === cell 1


def split_labels(s):
    if pd.isna(s) or str(s).strip() == "":
        return []
    return str(s).strip().split()


train_df = train_df.copy()
train_df["labels_list"] = train_df["labels"].map(split_labels)

dblock = DataBlock(
    blocks=(ImageBlock, MultiCategoryBlock),
    get_x=ColReader("image", pref=lambda o: os.path.join(TRAIN_IMGS, o)),
    get_y=ColReader("labels_list"),
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    item_tfms=Resize(460),
    batch_tfms=aug_transforms(size=384, min_scale=0.75),
)

ncpu = os.cpu_count() or 2
nw = min(8, max(2, ncpu // 2))
dls = dblock.dataloaders(
    train_df, bs=16, num_workers=nw, pin_memory=torch.cuda.is_available()
)

learn = vision_learner(
    dls,
    resnet50,
    pretrained=True,
    loss_func=BCEWithLogitsLossFlat(),
    metrics=[F1ScoreMulti(thresh=0.5, average="macro")],
).to_fp32()

learn.fine_tune(3, base_lr=2e-3)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3110104778.py in <cell line: 0>()
     24 ncpu = os.cpu_count() or 2
     25 nw = min(8, max(2, ncpu // 2))
---> 26 dls = dblock.dataloaders(
     27     train_df, bs=16, num_workers=nw, pin_memory=torch.cuda.is_available()
     28 )

/usr/local/lib/python3.11/dist-packages/fastai/data/block.py in dataloaders(self, source, path, verbose, **kwargs)
    155         **kwargs
    156     ) -> DataLoaders:
--> 157         dsets = self.datasets(source, verbose=verbose)
    158         kwargs = {**self.dls_kwargs, **kwargs, 'verbose': verbose}
    159         return dsets.dataloaders(path=path, after_item=self.item_tfms, after_batch=self.batch_tfms, **kwargs)

/usr/local/lib/python3.11/dist-packages/fastai/data/block.py in datasets(self, source, verbose)
    147         splits = (self.splitter or RandomSplitter())(items)
    148         pv(f"{len(splits)} datasets of sizes {','.join([str(len(s)) for s in splits])}", verbose)
--> 149         return Datasets(items, tfms=self._combine_type_tfms(), splits=splits, dl_type=self.dl_type, n_inp=self.n_inp, verbose=verbose)
    150 
    151     def dataloaders(self, 

/usr/local/lib/python3.11/dist-packages/fastai/data/core.py in __init__(self, items, tfms, tls, n_inp, dl_type, **kwargs)
    448     ):
    449         super().__init__(dl_type=dl_type)
--> 450         self.tls = L(tls if tls else [TfmdLists(items, t, **kwargs) for t in L(ifnone(tfms,[None]))])
    451         self.n_inp = ifnone(n_inp, max(1, len(self.tls)-1))
    452 

/usr/local/lib/python3.11/dist-packages/fastai/data/core.py in <listcomp>(.0)
    448     ):
    449         super().__init__(dl_type=dl_type)
--> 450         self.tls = L(tls if tls else [TfmdLists(items, t, **kwargs) for t in L(ifnone(tfms,[None]))])
    451         self.n_inp = ifnone(n_inp, max(1, len(self.tls)-1))
    452 

/usr/local/lib/python3.11/dist-packages/fastcore/foundation.py in __call__(cls, x, *args, **kwargs)
    103     def __call__(cls, x=None, *args, **kwargs):
    104         if not args and not kwargs and x is not None and isinstance(x,cls): return x
--> 105         return super().__call__(x, *args, **kwargs)
    106 
    107 # %% ../nbs/02_foundation.ipynb

/usr/local/lib/python3.11/dist-packages/fastai/data/core.py in __init__(self, items, tfms, use_list, do_setup, split_idx, train_setup, splits, types, verbose, dl_type)
    362         if do_setup:
    363             pv(f"Setting up {self.tfms}", verbose)
--> 364             self.setup(train_setup=train_setup)
    365 
    366     def _new(self, items, split_idx=None, **kwargs):

/usr/local/lib/python3.11/dist-packages/fastai/data/core.py in setup(self, train_setup)
    389             for f in self.tfms.fs:
    390                 self.types.append(getattr(f, 'input_types', type(x)))
--> 391                 x = f(x)
    392             self.types.append(type(x))
    393         types = L(t if is_listy(t) else [t] for t in self.types).concat().unique()

/usr/local/lib/python3.11/dist-packages/fastai/data/transforms.py in __call__(self, o, **kwargs)
    218 
    219     def __call__(self, o, **kwargs):
--> 220         if len(self.cols) == 1: return self._do_one(o, self.cols[0])
    221         return L(self._do_one(o, c) for c in self.cols)
    222 

/usr/local/lib/python3.11/dist-packages/fastai/data/transforms.py in _do_one(self, r, c)
    213     def _do_one(self, r, c):
    214         o = r[c] if isinstance(c, int) or not c in getattr(r, '_fields', []) else getattr(r, c)
--> 215         if len(self.pref)==0 and len(self.suff)==0 and self.label_delim is None: return o
    216         if self.label_delim is None: return f'{self.pref}{o}{self.suff}'
    217         else: return o.split(self.label_delim) if len(o)>0 else []

TypeError: object of type 'function' has no len()

## === cell 2

val_probs, val_targs = learn.get_preds(dl=learn.dls.valid)
val_targs = val_targs.int()


def f1_macro_from_probs(probs, targs, thr: float):
    preds = (probs > thr).int()
    tp = (preds & targs).sum(dim=0).float()
    fp = (preds & (1 - targs)).sum(dim=0).float()
    fn = ((1 - preds) & targs).sum(dim=0).float()
    denom = (2 * tp + fp + fn).clamp_min(1e-9)
    f1 = (2 * tp) / denom
    return f1.mean().item()


thr_grid = np.linspace(0.05, 0.95, 19)
best_thr, best_f1 = 0.5, -1.0
for thr in thr_grid:
    f1 = f1_macro_from_probs(val_probs, val_targs, float(thr))
    if f1 > best_f1:
        best_f1, best_thr = f1, float(thr)

print("Selected threshold:", best_thr, "val macro F1:", best_f1)

test_files = [os.path.join(TEST_IMGS, fn) for fn in test_filenames]
test_dl = learn.dls.test_dl(test_files, with_labels=False)

with torch.inference_mode():
    test_probs, _ = learn.get_preds(dl=test_dl)

vocab = list(learn.dls.vocab)
test_pred_bin = (test_probs > best_thr).cpu().numpy().astype(bool)


def bin_to_labelstr(row_bool):
    labs = [vocab[i] for i, b in enumerate(row_bool) if b]
    return "healthy" if len(labs) == 0 else " ".join(labs)


pred_labels = [bin_to_labelstr(r) for r in test_pred_bin]

pred_map = pd.Series(
    pred_labels, index=pd.Index(test_filenames, name="image"), name="labels"
)
sub = sample_sub[["image"]].copy()
sub = sub.join(pred_map, on="image")
sub["labels"] = sub["labels"].fillna("healthy")

sub.to_csv("submission.csv", index=False)
print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
print("submission.csv exists:", os.path.exists("submission.csv"))

## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1305039864.py in <cell line: 0>()
      3 
      4 # Get validation predictions for threshold selection
----> 5 val_probs, val_targs = learn.get_preds(dl=learn.dls.valid)
      6 val_targs = val_targs.int()
      7 

NameError: name 'learn' is not defined
