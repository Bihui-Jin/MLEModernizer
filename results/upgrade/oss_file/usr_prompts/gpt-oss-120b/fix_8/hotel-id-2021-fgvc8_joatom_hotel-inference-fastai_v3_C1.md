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
Identify hotels from images.

## Metric
Mean Average Precision @ 5 (MAP@5)

## Submission Format
For each image in the test set, you must predict a space-delimited list of hotel IDs that could match that image. The first ID should be the most relevant one and the last the least relevant one. The file should contain a header and have the following format:

```
image,hotel_id
99e91ad5f2870678.jpg,36363 53586 18807 64314 60181
b5cc62ab665591a9.jpg,36363 53586 18807 64314 60181
d5664a972d5a644b.jpg,36363 53586 18807 64314 60181
```

## Dataset
**train.csv** - The training set metadata.

- `image` - The image ID.

- `chain` - An ID code for the hotel chain. A `chain` of zero (0) indicates that the hotel is either not part of a chain or the chain is not known. This field is not available for the test set. The number of hotels per chain varies widely.

- `hotel_id` - The hotel ID. The target class.

- `timestamp` - When the image was taken. Provided for the training set only.

**sample_submission.csv** - A sample submission file in the correct format.

- `image` The image ID

- `hotel_id` The hotel ID. The target class.

**train_images** - The training set contains 97000+ images from around 7700 hotels from across the globe. All of the images for each hotel chain are in a dedicated subfolder for that chain.

**test_images** - The test set images. This competition has a hidden test set: only three images are provided here as samples while the remaining 13,000 images will be available to your notebook once it is submitted.

# 2. Python version

3.9

# 3. Installed packages

dill==0.4.0
fastai==2.8.5
geopandas==0.14.4
numpy==1.26.4
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
            description.md (120 lines)
            sample_submission.csv (9757 lines)
            sample_submission.csv.zip (106.8 kB)
            test.zip (160 Bytes)
            test_images.zip (2.6 GB)
            train.csv (87799 lines)
            train.csv.zip (1.9 MB)
            train.zip (162 Bytes)
            train_images.zip (23.5 GB)
            hotel-id-2021-fgvc8/
                description.md (120 lines)
                sample_submission.csv (9757 lines)
                ... and 7 other files
                hotel-id-2021-fgvc8/
                test/
                    test/
                test_images/
                    ccc436fc41bf402f.jpg (72.8 kB)
                    fb9d48b39c614c32.jpg (91.3 kB)
                    ... and 9754 other files
                    test_images/
                train/
                    train/
                train_images/
                    0/
                        b5bd0a0a2de05bb5.jpg (73.8 kB)
                        c242bcf0719f9d61.jpg (71.9 kB)
                        ... and 18211 other files
                    1/
                        a7ad6a44813b77c8.jpg (81.1 kB)
                        9b89db65b496490d.jpg (630.0 kB)
                        ... and 1116 other files
                    ... and 87 other folders
            test/
                test/
            test_images/
                ccc436fc41bf402f.jpg (72.8 kB)
                fb9d48b39c614c32.jpg (91.3 kB)
                ... and 9754 other files
                test_images/
            train/
                train/
            train_images/
                0/
                    b5bd0a0a2de05bb5.jpg (73.8 kB)
                    c242bcf0719f9d61.jpg (71.9 kB)
                    ... and 18211 other files
                1/
                    a7ad6a44813b77c8.jpg (81.1 kB)
                    9b89db65b496490d.jpg (630.0 kB)
                    ... and 1116 other files
                ... and 87 other folders
        input/
            description.md (120 lines)
            sample_submission.csv (9757 lines)
            sample_submission.csv.zip (106.8 kB)
            test.zip (160 Bytes)
            test_images.zip (2.6 GB)
            train.csv (87799 lines)
            train.csv.zip (1.9 MB)
            train.zip (162 Bytes)
            train_images.zip (23.5 GB)
            hotel-id-2021-fgvc8/
                description.md (120 lines)
                sample_submission.csv (9757 lines)
                ... and 7 other files
                hotel-id-2021-fgvc8/
                test/
                    test/
                test_images/
                    ccc436fc41bf402f.jpg (72.8 kB)
                    fb9d48b39c614c32.jpg (91.3 kB)
                    ... and 9754 other files
                    test_images/
                train/
                    train/
                train_images/
                    0/
                        b5bd0a0a2de05bb5.jpg (73.8 kB)
                        c242bcf0719f9d61.jpg (71.9 kB)
                        ... and 18211 other files
                    1/
                        a7ad6a44813b77c8.jpg (81.1 kB)
                        9b89db65b496490d.jpg (630.0 kB)
                        ... and 1116 other files
                    ... and 87 other folders
            test/
                test/
                    test/
            test_images/
                ccc436fc41bf402f.jpg (72.8 kB)
                fb9d48b39c614c32.jpg (91.3 kB)
                ... and 9754 other files
                test_images/
                    ccc436fc41bf402f.jpg (72.8 kB)
                    fb9d48b39c614c32.jpg (91.3 kB)
                    ... and 9754 other files
                    test_images/
            train/
                train/
                    train/
            train_images/
                0/
                    b5bd0a0a2de05bb5.jpg (73.8 kB)
                    c242bcf0719f9d61.jpg (71.9 kB)
                    ... and 18211 other files
                1/
                    a7ad6a44813b77c8.jpg (81.1 kB)
                    9b89db65b496490d.jpg (630.0 kB)
                    ... and 1116 other files
                ... and 87 other folders
        working/
            hotel-id-2021-fgvc8/
                description.md (120 lines)
                sample_submission.csv (9757 lines)
                ... and 7 other files
                hotel-id-2021-fgvc8/
                test/
                    test/
                test_images/
                    ccc436fc41bf402f.jpg (72.8 kB)
                    fb9d48b39c614c32.jpg (91.3 kB)
                    ... and 9754 other files
                    test_images/
                train/
                    train/
                train_images/
                    0/
                        b5bd0a0a2de05bb5.jpg (73.8 kB)
                        c242bcf0719f9d61.jpg (71.9 kB)
                        ... and 18211 other files
                    1/
                        a7ad6a44813b77c8.jpg (81.1 kB)
                        9b89db65b496490d.jpg (630.0 kB)
                        ... and 1116 other files
                    ... and 87 other folders
```

-> data/hotel-id-2021-fgvc8/sample_submission.csv has 9756 rows and 2 columns.
The columns are: image, hotel_id

-> data/hotel-id-2021-fgvc8/train.csv has 87798 rows and 4 columns.
The columns are: image, chain, hotel_id, timestamp

-> data/sample_submission.csv has 9756 rows and 2 columns.
The columns are: image, hotel_id

-> data/train.csv has 87798 rows and 4 columns.
The columns are: image, chain, hotel_id, timestamp

-> input/hotel-id-2021-fgvc8/sample_submission.csv has 9756 rows and 2 columns.
The columns are: image, hotel_id

-> input/hotel-id-2021-fgvc8/train.csv has 87798 rows and 4 columns.
The columns are: image, chain, hotel_id, timestamp

-> (stopped after 10 files for performance)

# 5. Target score

0.5748358171255361

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
from fastai.vision.all import *
import dill, torch, warnings, os

warnings.filterwarnings("ignore")
torch.backends.cudnn.benchmark = True



## === cell 1
DATA_ROOT = Path("../input/hotel-id-2021-fgvc8")
TRAIN_CSV = DATA_ROOT / "train.csv"
TEST_SUBMIT = DATA_ROOT / "sample_submission.csv"
TRAIN_IMG_ROOT = DATA_ROOT / "train_images"
TEST_IMG_ROOT = DATA_ROOT / "test_images"

train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_SUBMIT)

train_df["img_path"] = (
    TRAIN_IMG_ROOT / train_df["chain"].astype(str) / train_df["image"]
)
test_df["img_path"] = TEST_IMG_ROOT / test_df["image"]



## === cell 2
workers = min(16, os.cpu_count() or 1)  # use more workers when available
dls = ImageDataLoaders.from_df(
    train_df,
    path=".",  # base path not used because fn_col holds full paths
    folder=None,
    valid_pct=0.05,
    seed=42,
    fn_col="img_path",
    label_col="hotel_id",
    item_tfms=Resize(128),
    batch_tfms=Normalize.from_stats(*imagenet_stats),
    bs=512,
    num_workers=workers,
    pin_memory=True,
    dl_kwargs=dict(persistent_workers=True),  # avoid worker restart each epoch
)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_55/3795271170.py in <cell line: 0>()
      1 # Increase parallel data loading and keep workers alive across epochs
      2 workers = min(16, os.cpu_count() or 1)  # use more workers when available
----> 3 dls = ImageDataLoaders.from_df(
      4     train_df,
      5     path=".",  # base path not used because fn_col holds full paths

/usr/local/lib/python3.11/dist-packages/fastai/vision/data.py in from_df(cls, df, path, valid_pct, seed, fn_col, folder, suff, label_col, label_delim, y_block, valid_col, item_tfms, batch_tfms, img_cls, **kwargs)
    177                            item_tfms=item_tfms,
    178                            batch_tfms=batch_tfms)
--> 179         return cls.from_dblock(dblock, df, path=path, **kwargs)
    180 
    181     @classmethod

/usr/local/lib/python3.11/dist-packages/fastai/data/core.py in from_dblock(cls, dblock, source, path, bs, val_bs, shuffle, device, **kwargs)
    278         **kwargs
    279     ):
--> 280         return dblock.dataloaders(source, path=path, bs=bs, val_bs=val_bs, shuffle=shuffle, device=device, **kwargs)
    281 
    282     _docs=dict(__getitem__="Retrieve `DataLoader` at `i` (`0` is training, `1` is validation)",

/usr/local/lib/python3.11/dist-packages/fastai/data/block.py in dataloaders(self, source, path, verbose, **kwargs)
    157         dsets = self.datasets(source, verbose=verbose)
    158         kwargs = {**self.dls_kwargs, **kwargs, 'verbose': verbose}
--> 159         return dsets.dataloaders(path=path, after_item=self.item_tfms, after_batch=self.batch_tfms, **kwargs)
    160 
    161     _docs = dict(new="Create a new `DataBlock` with other `item_tfms` and `batch_tfms`",

/usr/local/lib/python3.11/dist-packages/fastai/data/core.py in dataloaders(self, bs, shuffle_train, shuffle, val_shuffle, n, path, dl_type, dl_kwargs, device, drop_last, val_bs, **kwargs)
    329         val_kwargs={k[4:]:v for k,v in kwargs.items() if k.startswith('val_')}
    330         def_kwargs = {'bs':bs,'shuffle':shuffle,'drop_last':drop_last,'n':n,'device':device}
--> 331         dl = dl_type(self.subset(0), **merge(kwargs,def_kwargs, dl_kwargs[0]))
    332         def_kwargs = {'bs':bs if val_bs is None else val_bs,'shuffle':val_shuffle,'n':None,'drop_last':False}
    333         dls = [dl] + [dl.new(self.subset(i), **merge(kwargs,def_kwargs,val_kwargs,dl_kwargs[i]))

KeyError: 0

## === cell 3
learn = cnn_learner(dls, resnet34, metrics=top_k_accuracy, pretrained=True)
learn.to_fp16()  # mixed‑precision remains unchanged
learn.fine_tune(1)  # same 2‑epoch schedule as original



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/362708637.py in <cell line: 0>()
----> 1 learn = cnn_learner(dls, resnet34, metrics=top_k_accuracy, pretrained=True)
      2 learn.to_fp16()  # mixed‑precision remains unchanged
      3 learn.fine_tune(1)  # same 2‑epoch schedule as original
      4 

NameError: name 'dls' is not defined

## === cell 4
test_dl = learn.dls.test_dl(
    test_df["img_path"],
    bs=512,
    num_workers=workers,
    dl_kwargs=dict(persistent_workers=True),
)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3947861191.py in <cell line: 0>()
      1 # Apply the same persistent‑worker settings to the test DataLoader
----> 2 test_dl = learn.dls.test_dl(
      3     test_df["img_path"],
      4     bs=512,
      5     num_workers=workers,

NameError: name 'learn' is not defined

## === cell 5
probs, _ = learn.tta(dl=test_dl, n=1)  # same TTA configuration



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3800349448.py in <cell line: 0>()
----> 1 probs, _ = learn.tta(dl=test_dl, n=1)  # same TTA configuration
      2 

NameError: name 'learn' is not defined

## === cell 6
preds_idx = probs.topk(5)[1]



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3593509576.py in <cell line: 0>()
----> 1 preds_idx = probs.topk(5)[1]
      2 

NameError: name 'probs' is not defined

## === cell 7
preds = [" ".join(map(str, [learn.dls.vocab[p] for p in row])) for row in preds_idx]



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3234044973.py in <cell line: 0>()
----> 1 preds = [" ".join(map(str, [learn.dls.vocab[p] for p in row])) for row in preds_idx]
      2 

NameError: name 'preds_idx' is not defined

## === cell 8
submission = pd.read_csv(TEST_SUBMIT)  # keep original ordering and headers
submission["hotel_id"] = preds
submission.to_csv("submission.csv", index=False)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2166833458.py in <cell line: 0>()
      1 submission = pd.read_csv(TEST_SUBMIT)  # keep original ordering and headers
----> 2 submission["hotel_id"] = preds
      3 submission.to_csv("submission.csv", index=False)
      4 

NameError: name 'preds' is not defined

## === cell 9
submission.head()
