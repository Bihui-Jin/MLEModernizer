# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Given a dataset of images of dogs, predict the breed of each image.

## Metric
Multi Class Log Loss.

## Submission Format
For each image in the test set, you must predict a probability for each of the different breeds. The file should contain a header and have the following format:
```
id,affenpinscher,afghan_hound,..,yorkshire_terrier
000621fb3cbb32d8935728e48679680e,0.0083,0.0,...,0.0083
etc.
```

## Dataset Description
- `train.zip` - the training set, you are provided the breed for these dogs
- `test.zip` - the test set, you must predict the probability of each breed for each image
- `sample_submission.csv` - a sample submission file in the correct format
- `labels.csv` - the breeds for the images in the train set

# 2. Python version

3.9

# 3. Installed packages

fastai==2.8.5
geopandas==0.14.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
sklearn-pandas==2.2.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (169 lines)
            labels.csv (9200 lines)
            labels.csv.zip (201.6 kB)
            sample_submission.csv (1024 lines)
            sample_submission.csv.zip (32.1 kB)
            test.zip (36.4 MB)
            train.zip (324.7 MB)
            dog-breed-identification/
                description.md (169 lines)
                labels.csv (9200 lines)
                ... and 5 other files
                dog-breed-identification/
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
            test/
                bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                ... and 1021 other files
                test/
            train/
                868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                ... and 9197 other files
                train/
        input/
            description.md (169 lines)
            labels.csv (9200 lines)
            labels.csv.zip (201.6 kB)
            sample_submission.csv (1024 lines)
            sample_submission.csv.zip (32.1 kB)
            test.zip (36.4 MB)
            train.zip (324.7 MB)
            dog-breed-identification/
                description.md (169 lines)
                labels.csv (9200 lines)
                ... and 5 other files
                dog-breed-identification/
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
            test/
                bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                ... and 1021 other files
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
            train/
                868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                ... and 9197 other files
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
        working/
            dog-breed-identification/
                description.md (169 lines)
                labels.csv (9200 lines)
                ... and 5 other files
                dog-breed-identification/
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
```

-> data/dog-breed-identification/labels.csv has 9199 rows and 2 columns.
The columns are: id, breed

-> data/dog-breed-identification/sample_submission.csv has 1023 rows and 121 columns.
The columns are: id, affenpinscher, afghan_hound, african_hunting_dog, airedale, american_staffordshire_terrier, appenzeller, australian_terrier, basenji, basset, beagle, bedlington_terrier, bernese_mountain_dog, black-and-tan_coonhound, blenheim_spaniel... and 106 more columns

-> data/labels.csv has 9199 rows and 2 columns.
The columns are: id, breed

-> data/sample_submission.csv has 1023 rows and 121 columns.
The columns are: id, affenpinscher, afghan_hound, african_hunting_dog, airedale, american_staffordshire_terrier, appenzeller, australian_terrier, basenji, basset, beagle, bedlington_terrier, bernese_mountain_dog, black-and-tan_coonhound, blenheim_spaniel... and 106 more columns

-> input/dog-breed-identification/labels.csv has 9199 rows and 2 columns.
The columns are: id, breed

-> input/dog-breed-identification/sample_submission.csv has 1023 rows and 121 columns.
The columns are: id, affenpinscher, afghan_hound, african_hunting_dog, airedale, american_staffordshire_terrier, appenzeller, australian_terrier, basenji, basset, beagle, bedlington_terrier, bernese_mountain_dog, black-and-tan_coonhound, blenheim_spaniel... and 106 more columns

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
from fastai.vision.all import *
import pandas as pd
import torch
import torch.nn.functional as F
import os
from pathlib import Path

set_seed(42, reproducible=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = torch.cuda.is_available()


def _recommended_num_workers():
    ncpu = os.cpu_count() or 2
    if torch.cuda.is_available():
        return min(4, max(2, ncpu // 2))
    else:
        return min(2, max(0, ncpu // 2))


DEFAULT_NUM_WORKERS = _recommended_num_workers()

USE_CUDA = torch.cuda.is_available()
PIN_MEMORY = USE_CUDA
PREFETCH_FACTOR = 2 if DEFAULT_NUM_WORKERS > 0 else None

if USE_CUDA:
    torch.cuda.set_device(0)
    default_device = torch.device("cuda")
else:
    default_device = torch.device("cpu")



## === cell 1
labels = pd.read_csv(
    "../input/dog-breed-identification/labels.csv", usecols=["id", "breed"]
)
labels



## === cell 2
labels["fname"] = labels["id"].astype(str) + ".jpg"
path = "../input/dog-breed-identification/train"

dblock = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_x=ColReader("fname", pref=path + os.sep),
    get_y=ColReader("breed"),
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    item_tfms=RandomResizedCrop(224),
    batch_tfms=[*aug_transforms(size=224), Normalize.from_stats(*imagenet_stats)],
)

use_persistent = DEFAULT_NUM_WORKERS >= 2

dls = dblock.dataloaders(
    labels,
    bs=(64 if USE_CUDA else 32),
    num_workers=DEFAULT_NUM_WORKERS,
    persistent_workers=use_persistent,
    pin_memory=PIN_MEMORY,
    prefetch_factor=PREFETCH_FACTOR,
    device=default_device,
)



## === cell 3
pass




## === cell 4
def log_loss(inputs, targ):
    return F.cross_entropy(inputs, targ)




## === cell 5
learn = cnn_learner(dls, densenet201, loss_func=log_loss, path=".", model_dir="models")
learn.to_fp32()  # preserve original numeric behavior (no mixed precision)
learn.model.to(default_device)

model_fname = "densenet201_dogs_ft15"



## === cell 6
pass



## === cell 7
model_path = Path(learn.path) / learn.model_dir / f"{model_fname}.pth"
model_path.parent.mkdir(parents=True, exist_ok=True)

if model_path.exists():
    learn.load(model_fname)
else:
    learn.fine_tune(
        15, 5e-3, cbs=SaveModelCallback(fname=model_fname, monitor="valid_loss")
    )



## === cell 8
test_dir = Path("../input/dog-breed-identification/test")
test_files = L(sorted(test_dir.glob("*.jpg"), key=lambda p: p.name))

test_dl = dls.test_dl(
    test_files,
    num_workers=DEFAULT_NUM_WORKERS,
    persistent_workers=use_persistent,
    pin_memory=PIN_MEMORY,
    prefetch_factor=PREFETCH_FACTOR,
)



## === cell 9
preds, _ = learn.get_preds(dl=test_dl)
preds = preds.float().cpu().numpy()

sample_sub = pd.read_csv("../input/dog-breed-identification/sample_submission.csv")
breed_cols = sample_sub.columns[1:].tolist()

vocab = list(dls.vocab)
col_index = {c: i for i, c in enumerate(vocab)}
take_idx = [col_index[c] for c in breed_cols]
preds_aligned = preds[:, take_idx]

ids = [p.stem for p in test_files]
sub = sample_sub.copy()
sub.set_index("id", inplace=True)

sub.loc[ids, breed_cols] = preds_aligned

sub.reset_index(inplace=True)

prob_cols = breed_cols
if sub[prob_cols].isna().any().any():
    sub[prob_cols] = sub[prob_cols].fillna(1.0 / len(prob_cols))

row_sums = sub[prob_cols].sum(axis=1).to_numpy()
row_sums[row_sums == 0] = 1.0
sub.loc[:, prob_cols] = sub[prob_cols].div(row_sums, axis=0)

sub.to_csv("submission.csv", index=False)
print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
print("Any NaNs in submission?", sub.isna().any().any())
print(
    "Min/Max row sum:",
    sub[prob_cols].sum(axis=1).min(),
    sub[prob_cols].sum(axis=1).max(),
)
